// Matrix multiplies on the AMD XDNA2 NPU, from inside the CPU backend.
//
// The NPU is good at one thing here: C[M x N] = A[M x K] * W[N x K]^T where W is a constant weight matrix. The core
// program (tools/xdna/gemm_bfp.py, mm_bfp_f32.cc) uses all 32 cores and serves every shape with M a multiple of 32, K of
// 64 and N of 512; other shapes are padded with zeros. Weights and activations go to the NPU as bfp16ebs8: eight values
// sharing one exponent byte, each with a signed 8-bit mantissa. Sums stay in float32.
//
// What costs time is the transport, not the arithmetic: all of W crosses from memory for each call, and the NPU reads
// at most 57 GB/s in all, however many streams it uses. And a job costs about 75 us in the driver and firmware, whatever
// it does. So a graph's matmuls run as one job: the job's stream knows them all in advance and waits for a doorbell in
// shared memory before each, and the host sees a matmul has finished when a sentinel in its output is overwritten. No
// driver call per matmul, 20 us instead of 75. All weights live in a few large buffers, because a job can name only five
// buffers. And each matmul's weights start crossing as soon as the last one is done, into the memory tiles, so that the
// host's work between matmuls overlaps with the transfer.
//
// Settings, read once:
//   GGML_XDNA=1               turn it on (--backend npu does)
//   GGML_XDNA_XCLBIN=<file>   the compiled core program (default: xdna/gemm_bfp.xclbin next to the program)
//   GGML_XDNA_MIN_M=<n>       smallest M sent to the NPU (default 1: the cost estimate decides)
//   GGML_XDNA_FORCE=1         send every eligible matmul, whatever it costs (tests)
//   GGML_XDNA_TRACE=1         print where the time went at exit; a larger n: every n calls as well
//   GGML_XDNA_TEST_STALL=<n>  don't ring the n-th doorbell, to see the NPU fail and come back (tests)
#include "ggml-xdna.h"
#include "ggml-xdna-insts.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <immintrin.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

extern "C" void ggml_barrier(struct ggml_threadpool * tp);

namespace {

using namespace ggml_xdna;

// Kernel arguments of a job. The stream names them by these numbers.
constexpr uint32_t ARG_A = 0, ARG_C = 2;
constexpr uint32_t ARG_W[3] = {1, 3, 4};
constexpr uint64_t A_DATA = 4096;  // A's buffer starts with a page for the mailbox
constexpr size_t MAX_STEPS_PER_JOB = 64;
constexpr uint32_t SENTINEL = 0xffc0dead;  // a NaN no multiply here produces

// ---- the host's side of memory: the NPU does not see the CPU's caches, nor they it --------------------------------------

inline void flush_range(const void * p, size_t n) {
    const char * c = (const char *) ((uintptr_t) p & ~(uintptr_t) 63);
    const char * end = (const char *) p + n;
#ifdef __CLFLUSHOPT__
    for (; c < end; c += 64) _mm_clflushopt((void *) c);
#else
    for (; c < end; c += 64) _mm_clflush(c);
#endif
}

// ---- operands -----------------------------------------------------------------------------------------------------------

inline void encode_block(const float * x, uint8_t * o) {
    const __m256 v = _mm256_loadu_ps(x);
    const __m256 a = _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    const int e = (_mm_cvtsi128_si32(_mm_castps_si128(m)) >> 23) & 255;
    if (e < 8) { memset(o, 0, BLOCK); return; }  // zero, or too small to matter
    const __m256 scale = _mm256_castsi256_ps(_mm256_set1_epi32((133 - e + 127) << 23));
    const __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(v, scale));
    o[0] = (uint8_t) e;
#if defined(__AVX512VL__) && defined(__AVX512BW__)
    _mm_storel_epi64((__m128i *) (o + 1), _mm256_cvtsepi32_epi8(q));
#else
    const __m128i s16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
    _mm_storel_epi64((__m128i *) (o + 1), _mm_packs_epi16(s16, s16));
#endif
}

inline int64_t up_to(int64_t v, int64_t grain) { return (v + grain - 1) / grain * grain; }
// K as the design takes it: a multiple of its tile, and a column's weights a whole number of B_GROUP tiles.
inline int64_t pad_k(int64_t K, int64_t Np) {
    const int64_t per_col = Np / TN / COLS;
    int64_t Kp = std::max(up_to(K, TK), 2 * TK);
    while ((Kp / TK * per_col) % B_GROUP != 0) Kp += TK;
    return Kp;
}

// Block index of (tile, row inside the tile, column block) in a tiled operand. Inside a tile of 16 rows or more the row
// pairs (16 rows) come one after another, and in a pair the 8x8 subtiles alternate between its two rows of subtiles.
// A tile of 8 rows is just its subtiles in order.
inline int64_t tiled(int64_t first_tile, int64_t rr, int64_t cb, int64_t tile_rows, int64_t nk) {
    const int64_t tkb = TK / 8, tile = tile_rows * tkb, base = (first_tile * nk + cb / tkb) * tile;
    if (tile_rows == 8) return base + (cb % tkb) * 8 + rr;
    return base + (rr / 16) * tkb * 16 + (cb % tkb) * 16 + rr % 16;
}

// One row of K values into its blocks; past K the operand is zero. Kp is K padded for the design.
template <typename Where>
inline void encode_row(const float * x, int64_t K, int64_t Kp, uint8_t * o, Where && where) {
    const int64_t whole = K / 8;
    for (int64_t cb = 0; cb < whole; cb++) encode_block(x + cb * 8, o + BLOCK * where(cb));
    int64_t next = whole;
    if (K % 8) {
        float last[8] = {0};
        memcpy(last, x + whole * 8, (K % 8) * sizeof(float));
        encode_block(last, o + BLOCK * where(whole));
        next++;
    }
    for (int64_t cb = next; cb < Kp / 8; cb++) memset(o + BLOCK * where(cb), 0, BLOCK);
}

// Only the real rows: what stale data does to the padding rows' outputs, nobody reads.
void pack_rows(const float * x, int64_t from, int64_t to, int64_t K, int64_t Kp, uint8_t * o) {
    const int64_t nk = Kp / TK;
    for (int64_t r = from; r < to; r++)
        encode_row(x + r * K, K, Kp, o, [&](int64_t cb) { return tiled(r / TM, r % TM, cb, TM, nk); });
}

void pack_w(const ggml_tensor * w, int64_t N, int64_t K, int64_t Np, int64_t Kp, uint8_t * o) {
    const int64_t nk = Kp / TK, per_col = Np / TN / COLS;
    const auto * traits = ggml_get_type_traits(w->type);
    memset(o, 0, (size_t) (Np * Kp / 8 * BLOCK));
    const auto part = [&](int64_t from, int64_t to) {
        std::vector<float> row(K);
        for (int64_t j = from; j < to; j++) {
            const char * p = (const char *) w->data + j * w->nb[1];
            if (w->type == GGML_TYPE_F32) memcpy(row.data(), p, K * sizeof(float));
            else traits->to_float(p, row.data(), K);
            const int64_t g = j / TN, first = (g % COLS) * per_col + g / COLS;
            encode_row(row.data(), K, Kp, o, [&](int64_t cb) { return tiled(first, j % TN, cb, TN, nk); });
        }
    };
    constexpr int64_t THREADS = 8;
    std::vector<std::thread> team;
    for (int64_t t = 1; t < THREADS; t++) team.emplace_back(part, N * t / THREADS, N * (t + 1) / THREADS);
    part(0, N / THREADS);
    for (auto & t : team) t.join();
}

// The real rows and columns of the output, from one array column's tiles. Walks them in the order they lie in memory,
// which has just been invalidated, so that the prefetcher does the work.
void unpack_col(const float * c, int col, int64_t rows, int64_t M, int64_t N, int64_t Np, float * dst) {
    const int64_t per_col = Np / TN / COLS, rbs = M / TM / ROWS, c_rb = per_col * ROWS * TM * TN;
    const int64_t tile_floats = TM * TN, blocks_n = TN / 8;
    const int64_t used_tiles = (rows + TM - 1) / TM;  // core rows that hold real output, over all row blocks
    for (int64_t rb = 0; rb < rbs; rb++) {
        const int64_t first_row = rb * ROWS * TM;
        if (first_row >= rows) break;
        for (int64_t t = 0; t < per_col; t++) {
            const int64_t at = (col + COLS * t) * TN;
            if (at >= N) continue;
            const float * group = c + (col * rbs + rb) * c_rb + t * (ROWS * tile_floats);
            for (int64_t core = 0; core < ROWS && rb * ROWS + core < used_tiles; core++) {
                const float * tile = group + core * tile_floats;
                const int64_t tile_row = first_row + core * TM;
                for (int64_t z = 0; z < TM / 8; z++) {
                    const int64_t row0 = tile_row + z * 8;
                    if (row0 >= rows) break;
                    const int64_t nrows = std::min<int64_t>(8, rows - row0);
                    for (int64_t jb = 0; jb < blocks_n; jb++) {
                        const int64_t col0 = at + jb * 8;
                        if (col0 >= N) break;
                        const float * block = tile + (z * blocks_n + jb) * 64;
                        if (col0 + 8 <= N) {
                            for (int64_t i = 0; i < nrows; i++)
                                _mm256_storeu_ps(dst + (row0 + i) * N + col0, _mm256_loadu_ps(block + i * 8));
                        } else {
                            for (int64_t i = 0; i < nrows; i++)
                                memcpy(dst + (row0 + i) * N + col0, block + i * 8, (N - col0) * sizeof(float));
                        }
                    }
                }
            }
        }
    }
}

// ---- the device ---------------------------------------------------------------------------------------------------------

using clk = std::chrono::steady_clock;
inline double us_since(clk::time_point t) { return std::chrono::duration<double, std::micro>(clk::now() - t).count(); }

struct step {
    int64_t M = 0, Kp = 0, Np = 0;
    int chunk = 0;
    uint64_t off = 0;
};

// One job's stream, and two runs of it so that the next can be queued while the last is still finishing.
struct segment {
    size_t first = 0, count = 0;
    xrt::bo insts[2];  // a stream for each JOB_MARK
    unsigned words = 0;
    std::unique_ptr<xrt::run> runs[2];
    bool started[2] = {false, false};
    int next = 0;
};

struct program {
    std::vector<step> steps;
    std::vector<segment> segments;
};

struct plan {
    program * prog = nullptr;
    std::vector<ggml_tensor *> nodes;
    size_t cur = 0, seg = 0;
};

struct npu {
    xrt::device dev;
    xrt::xclbin xb;
    xrt::hw_context ctx;
    xrt::kernel kern;
    xrt::bo a, c;
    size_t a_cap = 0, c_cap = 0;
    struct chunk { xrt::bo bo; uint8_t * map = nullptr; size_t cap = 0, used = 0; };
    std::vector<chunk> chunks;
    struct slot { int chunk = 0; uint64_t off = 0; uint64_t sum = 0; };
    std::unordered_map<const void *, slot> weights;
    std::unordered_map<uint64_t, std::unique_ptr<program>> programs;
    volatile uint32_t * mailbox = nullptr;
    int mark = 0;  // the running job's JOB_MARK
    long long test_stall = 0;
    // accounting
    long long trace_every = 0, calls = 0, jobs = 0;
    double us_pack = 0, us_run = 0, us_unpack = 0, us_w = 0, macs = 0;
    size_t w_bytes = 0;

    static xrt::hw_context open(xrt::device & d, xrt::xclbin & x) {
        d.register_xclbin(x);
        return xrt::hw_context(d, x.get_uuid());
    }
    explicit npu(const std::string & path) : dev(0u), xb(path), ctx(open(dev, xb)), kern(ctx, "MLIR_AIE") {
        reserve(4096 * 64, 4096 * 64);
    }
    void print_trace() const {
        fprintf(stderr, "xdna: %lld calls in %lld jobs, %.1f GMAC; pack %.0f ms, run %.0f ms, unpack %.0f ms; weights %.1f MB packed in %.0f ms; %zu programs\n",
                calls, jobs, macs / 1e9, us_pack / 1e3, us_run / 1e3, us_unpack / 1e3, w_bytes / 1e6, us_w / 1e3, programs.size());
    }
    void reserve(size_t an, size_t cn) {
        if (an + A_DATA > a_cap) {
            a_cap = an + A_DATA;
            a = xrt::bo(dev, a_cap, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
            memset(a.map<void *>(), 0, a_cap);
            a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            mailbox = (volatile uint32_t *) a.map<uint8_t *>();
        }
        if (cn > c_cap) {
            c_cap = cn;
            c = xrt::bo(dev, c_cap, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
        }
    }
    uint8_t * a_data() { return a.map<uint8_t *>() + A_DATA; }

    // A weight's place in the big buffers, packing it if it has none. False if there is no room.
    bool place_weight(const ggml_tensor * w, int64_t N, int64_t K, int64_t Np, int64_t Kp, slot & out) {
        // the content tells a reloaded model apart from the one that used to sit at this address
        uint64_t sum = 14695981039346656037ULL;
        const size_t n = ggml_nbytes(w);
        const uint8_t * b = (const uint8_t *) w->data;
        for (size_t i = 0; i < 32; i++) {
            sum = (sum ^ b[i]) * 1099511628211ULL;
            sum = (sum ^ b[n / 2 + i]) * 1099511628211ULL;
            sum = (sum ^ b[n - 1 - i]) * 1099511628211ULL;
        }
        auto it = weights.find(w->data);
        if (it != weights.end() && it->second.sum == sum) { out = it->second; return true; }
        const size_t bytes = (size_t) (Np * Kp / 8 * BLOCK);
        const size_t aligned = (bytes + 4095) & ~(size_t) 4095;
        int ci = (int) chunks.size() - 1;
        if (ci < 0 || chunks[ci].used + aligned > chunks[ci].cap) {
            if (chunks.size() == 3) return false;
            // the whole of the weights' buffer as bfp16 is the most a chunk can be asked to hold
            size_t cap = 256u << 20;
            if (w->buffer != nullptr) cap = std::max(cap, (size_t) (ggml_backend_buffer_get_size(w->buffer) * 5 / 4));
            cap = std::max(cap, aligned);
            chunk nc;
            nc.bo = xrt::bo(dev, cap, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
            nc.map = nc.bo.map<uint8_t *>();
            nc.cap = cap;
            chunks.push_back(std::move(nc));
            ci = (int) chunks.size() - 1;
        }
        const auto t = clk::now();
        chunk & ch = chunks[ci];
        pack_w(w, N, K, Np, Kp, ch.map + ch.used);
        flush_range(ch.map + ch.used, bytes);
        _mm_sfence();
        out.chunk = ci;
        out.off = ch.used;
        out.sum = sum;
        ch.used += aligned;
        weights[w->data] = out;
        us_w += us_since(t);
        w_bytes += bytes;
        return true;
    }

    program * get_program(const std::vector<ggml_tensor *> & nodes) {
        std::vector<step> steps(nodes.size());
        uint64_t h = 14695981039346656037ULL;
        const auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
        size_t an = 0, cn = 0;
        for (size_t i = 0; i < nodes.size(); i++) {
            const ggml_tensor * w = nodes[i]->src[0];
            const ggml_tensor * x = nodes[i]->src[1];
            const int64_t K = w->ne[0], N = w->ne[1], rows = x->ne[1] * x->ne[2] * x->ne[3];
            step & s = steps[i];
            s.M = up_to(rows, GRAIN_M);
            s.Np = up_to(N, GRAIN_N);
            s.Kp = pad_k(K, s.Np);
            slot sl;
            if (!place_weight(w, N, K, s.Np, s.Kp, sl)) return nullptr;
            s.chunk = sl.chunk;
            s.off = sl.off;
            mix((uint64_t) s.M); mix((uint64_t) s.Kp); mix((uint64_t) s.Np); mix((uint64_t) s.chunk); mix(s.off);
            an = std::max(an, (size_t) (s.M * s.Kp / 8 * BLOCK));
            cn = std::max(cn, (size_t) (s.M * s.Np) * sizeof(float));
        }
        reserve(an, cn);
        auto it = programs.find(h);
        if (it != programs.end()) return it->second.get();
        auto p = std::make_unique<program>();
        p->steps = std::move(steps);
        for (size_t first = 0; first < p->steps.size(); first += MAX_STEPS_PER_JOB) {
            segment seg;
            seg.first = first;
            seg.count = std::min(MAX_STEPS_PER_JOB, p->steps.size() - first);
            for (int m = 0; m < 2; m++) {
                stream st;
                st.doorbell_start({ARG_A, 0});
                for (size_t i = 0; i < seg.count; i++) {
                    const step & s = p->steps[first + i];
                    st.matmul(s.M, s.Kp, s.Np, {ARG_A, A_DATA}, {ARG_W[s.chunk], s.off}, {ARG_C, 0}, JOB_MARK[m] | (uint32_t) (i + 1));
                }
                st.doorbell_stop();
                const auto words = st.finish();
                seg.words = (unsigned) words.size();
                seg.insts[m] = xrt::bo(dev, words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
                memcpy(seg.insts[m].map<void *>(), words.data(), words.size() * 4);
                seg.insts[m].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
            p->segments.push_back(std::move(seg));
        }
        return programs.emplace(h, std::move(p)).first->second.get();
    }

    void start(segment & seg) {
        xrt::run * r;
        {
            int i = seg.next;
            seg.next = 1 - i;
            if (!seg.runs[i]) seg.runs[i] = std::make_unique<xrt::run>(kern);
            r = seg.runs[i].get();
            if (seg.started[i]) {
                const auto state = r->wait();
                if (state != ERT_CMD_STATE_COMPLETED) throw std::runtime_error("an NPU job ended in state " + std::to_string((int) state));
            }
            seg.started[i] = true;
        }
        mark = 1 - mark;
        r->set_arg(0, 3u);
        r->set_arg(1, seg.insts[mark]);
        r->set_arg(2, seg.words);
        r->set_arg(3 + ARG_A, a);
        r->set_arg(3 + ARG_C, c);
        for (size_t i = 0; i < chunks.size(); i++) r->set_arg(3 + ARG_W[i], chunks[i].bo);
        r->start();
        jobs++;
    }

    // Does the NPU run a job? A job of a process that was killed in the middle of one waits for its doorbell until the driver's
    // own timeout for a job, 60 s, and nothing else runs until then.
    void check() {
        stream st;
        st.doorbell_start({ARG_A, 0});
        st.doorbell_stop();
        const auto words = st.finish();
        xrt::bo insts(dev, words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
        memcpy(insts.map<void *>(), words.data(), words.size() * 4);
        insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        xrt::run r(kern);
        r.set_arg(0, 3u);
        r.set_arg(1, insts);
        r.set_arg(2, (unsigned) words.size());
        r.set_arg(3 + ARG_A, a);
        r.set_arg(3 + ARG_C, c);
        r.start();
        const auto state = r.wait(std::chrono::milliseconds(1000));
        if (state != ERT_CMD_STATE_COMPLETED) throw std::runtime_error("the NPU did not run a test job: state " + std::to_string((int) state));
    }

    // A new context, for when the NPU has stopped answering. Programs and buffers stay.
    void reopen() {
        for (auto & kv : programs)
            for (segment & seg : kv.second->segments) {
                seg.runs[0].reset();
                seg.runs[1].reset();
                seg.started[0] = seg.started[1] = false;
                seg.next = 0;
            }
        kern = xrt::kernel();
        ctx = xrt::hw_context();
        ctx = xrt::hw_context(dev, xb.get_uuid());
        kern = xrt::kernel(ctx, "MLIR_AIE");
    }

    // Rings the doorbell for a step of the running job and waits for its output to arrive.
    void run_step(const step & s, uint32_t seq) {
        float * cm = c.map<float *>();
        for (int col = 0; col < COLS; col++) {
            float * e = cm + (col + 1) * s.M * s.Np / COLS - 1;
            memcpy(e, &SENTINEL, 4);
            _mm_clflush(e);
        }
        _mm_mfence();
        const auto t = clk::now();
        if (test_stall == 0 || --test_stall != 0) {
            _mm_store_si128((__m128i *) const_cast<uint32_t *>(mailbox), _mm_set1_epi32((int) (JOB_MARK[mark] | seq)));
            _mm_clflush((const void *) mailbox);
            _mm_mfence();
        }
        // Most of this wait is the weights crossing, part of which has already happened: no need to keep asking until the
        // rest should have. (Asking means reading through the cache, every 100 ns.)
        const double expected = 0.4 * (double) (s.M / GRAIN_M) * s.Kp * s.Np * BLOCK / 8 / 57e3;
        while (us_since(t) < expected) _mm_pause();
        for (int col = 0; col < COLS; col++) {
            const float * e = cm + (col + 1) * s.M * s.Np / COLS - 1;
            for (;;) {
                _mm_clflush(e);
                _mm_mfence();
                uint32_t v;
                memcpy(&v, e, 4);
                if (v != SENTINEL) break;
                if (us_since(t) > 2e6) throw std::runtime_error("no answer from the NPU in 2 s: doorbell " + std::to_string(seq) + ", " + std::to_string(s.M) + " x " + std::to_string(s.Kp) + " x " + std::to_string(s.Np) + ", weights in buffer " + std::to_string(s.chunk) + " at " + std::to_string(s.off) + ", column " + std::to_string(col));
                _mm_pause();
            }
        }
        us_run += us_since(t);
    }
};

std::once_flag g_once;
npu * g_npu = nullptr;
std::atomic<bool> g_on{false};
int64_t g_min_m = 1;
bool g_force = false;
std::mutex g_busy;
plan g_plan;
std::atomic<plan *> g_active{nullptr};
thread_local bool t_holds_plan = false;  // this thread started the job, and is the one to end it
std::atomic<bool> g_step_failed{false};
std::unordered_set<const void *> g_no_room;
int g_failures = 0;
clk::time_point g_last_failure;

void give_up(const std::exception & e);

void trace_at_exit() {
    if (g_npu) g_npu->print_trace();
}

void init() {
    const char * on = getenv("GGML_XDNA");
    if (!on || !*on || !strcmp(on, "0")) return;
    if (const char * v = getenv("GGML_XDNA_MIN_M")) g_min_m = atoll(v);
    g_force = getenv("GGML_XDNA_FORCE") != nullptr;
    std::string path;
    if (const char * v = getenv("GGML_XDNA_XCLBIN")) path = v;
    if (path.empty()) {
        // next to the program, in xdna/
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        if (n > 0) {
            path.assign(exe, (size_t) n);
            path = path.substr(0, path.rfind('/')) + "/xdna/gemm_bfp.xclbin";
        }
    }
#ifdef GGML_XDNA_XRT_ROOT_DIR
    setenv("XILINX_XRT", GGML_XDNA_XRT_ROOT_DIR, 0);  // XRT finds its plugins through this
#endif
    try {
        const auto t = clk::now();
        g_npu = new npu(path);  // never freed: the driver cleans up when the process ends
        fprintf(stderr, "xdna: NPU %s ready in %.0f ms\n", g_npu->dev.get_info<xrt::info::device::name>().c_str(), us_since(t) / 1e3);
        if (const char * v = getenv("GGML_XDNA_TRACE")) {
            atexit(trace_at_exit);
            if (atoll(v) > 1) g_npu->trace_every = atoll(v);  // also every so many calls, for a server
        }
        if (const char * v = getenv("GGML_XDNA_TEST_STALL")) g_npu->test_stall = atoll(v);
        try {
            g_npu->check();
            g_on = true;
        } catch (const std::exception & e) {
            give_up(e);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "xdna: no NPU, staying on the CPU: %s\n", e.what());
    }
}

// Is this matmul the NPU's? A pure function of the node and the team's size.
bool owns(const ggml_tensor * dst, int nth) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int64_t K = w->ne[0], N = w->ne[1];
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    // Weights only: a constant the NPU can keep a copy of.
    const bool is_weights = w->buffer != nullptr && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    if (!is_weights && !ggml_is_quantized(w->type)) return false;
    if (!ggml_is_quantized(w->type) && w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) return false;
    if (w->nb[0] != ggml_type_size(w->type) || !ggml_is_contiguous(x) || !ggml_is_contiguous(dst)) return false;
    const int64_t rows = x->ne[1] * x->ne[2] * x->ne[3];
    if (rows < g_min_m || rows > 4096) return false;
    if (!g_no_room.empty() && g_no_room.count(w->data)) return false;
    // Measured on Strix Halo. The NPU: a doorbell, then W at 52 GB/s for every row block, then A and C on the host, shared
    // among the team. The CPU on 8-bit weights: about 25 GMAC/s a thread, up to what memory gives, about 100 GMAC/s together.
    const int64_t Np = up_to(N, GRAIN_N), Kp = pad_k(K, Np), row_blocks = up_to(rows, GRAIN_M) / GRAIN_M;
    const double npu_us = 25.0 + (double) row_blocks * Kp * Np * BLOCK / 8 / 52e3 + 2e-3 * rows * (Kp + Np) / nth;
    const double cpu_us = (double) rows * K * N / std::min(25e3 * nth, 100e3);
    return g_force || npu_us < cpu_us;
}

// The plain multiply, for a call the NPU failed: only thread 0 is left to do it.
void mul_mat_fallback(const ggml_tensor * w, const float * x, float * out, int64_t rows, int64_t K, int64_t N) {
    std::vector<float> row(K);
    const auto * traits = ggml_get_type_traits(w->type);
    for (int64_t j = 0; j < N; j++) {
        const char * p = (const char *) w->data + j * w->nb[1];
        if (w->type == GGML_TYPE_F32) memcpy(row.data(), p, K * sizeof(float));
        else traits->to_float(p, row.data(), K);
        for (int64_t r = 0; r < rows; r++) {
            float s = 0;
            const float * a = x + r * K;
            for (int64_t q = 0; q < K; q++) s += a[q] * row[q];
            out[r * N + j] = s;
        }
    }
}

// Everything goes on on the CPU while a new context is tried in the background, until one runs a job. The more failures of late,
// the longer before it is used again: a failure costs the listener 2 s, the CPU only speed.
void give_up(const std::exception & e) {
    fprintf(stderr, "xdna: the NPU failed, back to the CPU: %s\n", e.what());
    g_on = false;
    const auto now = clk::now();
    if (now - g_last_failure > std::chrono::hours(1)) g_failures = 0;
    g_last_failure = now;
    const int seconds = g_failures == 0 ? 0 : 30 << std::min(g_failures - 1, 6);
    g_failures++;
    std::thread([seconds] {
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
        const auto t = clk::now();
        for (int attempt = 0; attempt < 30; attempt++) {
            {
                std::lock_guard<std::mutex> lock(g_busy);
                try {
                    g_npu->reopen();
                    g_npu->check();
                    g_on = true;
                    fprintf(stderr, "xdna: the NPU is back after %.1f s\n", us_since(t) / 1e6);
                    return;
                } catch (const std::exception &) {
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        fprintf(stderr, "xdna: the NPU stays off\n");
    }).detach();
}

}  // namespace

extern "C" void ggml_xdna_graph_begin(const struct ggml_cgraph * graph, int nth) {
    std::call_once(g_once, init);
    if (!g_on) return;
    std::vector<ggml_tensor *> nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        ggml_tensor * n = graph->nodes[i];
        if (n->op == GGML_OP_MUL_MAT && (n->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 && owns(n, nth)) nodes.push_back(n);
    }
    if (nodes.empty()) return;
    g_busy.lock();
    try {
        program * p = g_npu->get_program(nodes);
        if (p == nullptr) {
            // No room for these weights: they stay on the CPU, from now on.
            for (ggml_tensor * n : nodes) g_no_room.insert(n->src[0]->data);
            fprintf(stderr, "xdna: no room for this graph's weights in the NPU's buffers; it stays on the CPU\n");
            g_busy.unlock();
            return;
        }
        g_plan.prog = p;
        g_plan.nodes = std::move(nodes);
        g_plan.cur = 0;
        g_plan.seg = 0;
        g_step_failed = false;
        g_npu->start(p->segments[0]);
        g_active = &g_plan;
        t_holds_plan = true;
    } catch (const std::exception & e) {
        give_up(e);
        g_active = nullptr;
        g_busy.unlock();
    }
}

extern "C" void ggml_xdna_graph_end(void) {
    if (!t_holds_plan) return;
    t_holds_plan = false;
    plan * pl = g_active.load();
    g_active = nullptr;
    // A graph that stopped early, or whose nodes went another way: walk the job to its end, or it would wait for ever.
    try {
        if (g_on) {
            npu & n = *g_npu;
            program & p = *pl->prog;
            while (pl->cur < p.steps.size()) {
                const segment & seg = p.segments[pl->seg];
                n.run_step(p.steps[pl->cur], (uint32_t) (pl->cur - seg.first + 1));
                pl->cur++;
                if (pl->cur == seg.first + seg.count && pl->cur < p.steps.size()) n.start(p.segments[++pl->seg]);
            }
        }
    } catch (const std::exception & e) {
        give_up(e);
    }
    g_busy.unlock();
}

extern "C" bool ggml_xdna_planned(const struct ggml_tensor * dst) {
    const plan * pl = g_active.load(std::memory_order_relaxed);
    // after a failure the rest of the graph is the CPU's; every thread sees that from the same node on, past a barrier
    return pl != nullptr && !g_step_failed.load(std::memory_order_relaxed) && pl->cur < pl->nodes.size() && pl->nodes[pl->cur] == dst;
}

extern "C" void ggml_xdna_mul_mat(int ith, int nth, struct ggml_threadpool * threadpool, struct ggml_tensor * dst) {
    plan & pl = *g_active.load(std::memory_order_relaxed);
    npu & n = *g_npu;
    program & p = *pl.prog;
    const step & s = p.steps[pl.cur];
    const size_t seg_i = pl.seg;
    const uint32_t seq = (uint32_t) (pl.cur - p.segments[seg_i].first + 1);
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * xt = dst->src[1];
    const int64_t K = w->ne[0], N = w->ne[1], rows = xt->ne[1] * xt->ne[2] * xt->ne[3];
    const float * x = (const float *) xt->data;
    float * out = (float *) dst->data;

    // The host's work is shared: each thread packs its rows
    auto t = clk::now();
    pack_rows(x, rows * ith / nth, rows * (ith + 1) / nth, K, s.Kp, n.a_data());
    ggml_barrier(threadpool);

    // Thread 0 starts the NPU and waits
    if (ith == 0) {
        flush_range(n.a_data(), (size_t) (up_to(rows, TM) * s.Kp / 8 * BLOCK));
        _mm_sfence();
        n.us_pack += us_since(t);
        try {
            n.run_step(s, seq);
        } catch (const std::exception & e) {
            give_up(e);
            g_step_failed = true;
            mul_mat_fallback(w, x, out, rows, K, N);
        }
    }
    ggml_barrier(threadpool);

    // Each thread unpacks its columns
    if (!g_step_failed.load(std::memory_order_relaxed)) {
        t = clk::now();
        const float * cm = n.c.map<const float *>();
        const int64_t c_col = s.M * s.Np / COLS;
        for (int col = ith; col < COLS; col += nth) {
            flush_range(cm + col * c_col, (size_t) c_col * sizeof(float));
            _mm_mfence();
            unpack_col(cm, col, rows, s.M, N, s.Np, out);
        }
        if (ith == 0) n.us_unpack += us_since(t);
    }

    if (ith == 0) {
        n.calls++;
        n.macs += (double) rows * K * N;
        if (n.trace_every > 0 && n.calls % n.trace_every == 0) n.print_trace();
        // the next matmul may be in the next job
        ggml_barrier(threadpool);
        pl.cur++;
        if (!g_step_failed.load(std::memory_order_relaxed) && pl.cur == p.segments[seg_i].first + p.segments[seg_i].count && pl.cur < p.steps.size()) {
            pl.seg = seg_i + 1;
            try {
                n.start(p.segments[pl.seg]);
            } catch (const std::exception & e) {
                give_up(e);
                g_step_failed = true;
            }
        }
    } else {
        ggml_barrier(threadpool);
    }
}
