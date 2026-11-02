// Matrix multiplies on the AMD XDNA2 NPU, from inside the CPU backend.
//
// The NPU is good at one thing here: C[M x N] = A[M x K] * W[N x K]^T where W is a constant weight
// matrix. The core program (tools/xdna/gemm_bfp.py, mm_bfp_f32.cc) uses all 32 cores and serves
// every shape with M a multiple of 64, K of 64 and N of 512; other shapes are padded with zeros.
// Only the instruction stream changes with the shape, and that is generated here. Weights and
// activations go to the NPU as bfp16ebs8: eight values sharing one exponent byte, each with a signed
// 8-bit mantissa. Sums stay in float32.
//
// What costs time is about 70 us per dispatch in the driver and firmware, plus moving the whole of
// W from memory for each call at up to 50 GB/s; the arithmetic itself is negligible. So a call is
// worth sending only when K * N is large and M is not tiny.
//
// Settings, read once:
//   GGML_XDNA=1               turn it on (--backend npu does)
//   GGML_XDNA_XCLBIN=<file>   the compiled core program (default: xdna/gemm_bfp.xclbin next to the program)
//   GGML_XDNA_MIN_M=<n>       smallest M sent to the NPU (default 1: the cost estimate decides)
//   GGML_XDNA_FORCE=1         send every eligible matmul, whatever it costs (tests)
//   GGML_XDNA_TRACE=1         print where the time went at exit; a larger n: every n calls as well
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sys/resource.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace ggml_xdna;

// ---- operands -------------------------------------------------------------------------------

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

// Block index of (tile, row inside the tile, column block) in a tiled operand. Inside a tile the row
// pairs (16 rows) come one after another, and in a pair the 8x8 subtiles alternate between its two
// rows of subtiles.
inline int64_t tiled(int64_t first_tile, int64_t rr, int64_t cb, int64_t tile_rows, int64_t nk) {
    (void) nk;
    const int64_t tkb = TK / 8, tile = tile_rows * tkb;
    return (first_tile * nk + cb / tkb) * tile + (rr / 16) * tkb * 16 + (cb % tkb) * 16 + rr % 16;
}

inline int64_t up_to(int64_t v, int64_t grain) { return (v + grain - 1) / grain * grain; }
inline int64_t pad_k(int64_t K) { return std::max(up_to(K, TK), 2 * TK); }

// One row of K values into its blocks; past K the operand stays zero. Kp is K padded for the design.
template <typename Where>
inline void encode_row(const float * x, int64_t K, uint8_t * o, Where && where) {
    const int64_t whole = K / 8;
    for (int64_t cb = 0; cb < whole; cb++) encode_block(x + cb * 8, o + BLOCK * where(cb));
    if (K % 8) {
        float last[8] = {0};
        memcpy(last, x + whole * 8, (K % 8) * sizeof(float));
        encode_block(last, o + BLOCK * where(whole));
    }
}

void pack_a(const float * x, int64_t rows, int64_t M, int64_t K, int64_t Kp, uint8_t * o) {
    const int64_t nk = Kp / TK;
    memset(o, 0, (size_t) (M * Kp / 8 * BLOCK));
    for (int64_t r = 0; r < rows; r++)
        encode_row(x + r * K, K, o, [&](int64_t cb) { return tiled(r / TM, r % TM, cb, TM, nk); });
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
            encode_row(row.data(), K, o, [&](int64_t cb) { return tiled(first, j % TN, cb, TN, nk); });
        }
    };
    constexpr int64_t THREADS = 8;
    std::vector<std::thread> team;
    for (int64_t t = 1; t < THREADS; t++) team.emplace_back(part, N * t / THREADS, N * (t + 1) / THREADS);
    part(0, N / THREADS);
    for (auto & t : team) t.join();
}

// The real rows and columns of the output, out of the core tiles. Walks the tiles in the order they lie
// in memory, which has just been invalidated, so that the prefetcher does the work.
void unpack_c(const float * c, int64_t rows, int64_t M, int64_t N, int64_t Np, float * dst) {
    const int64_t per_col = Np / TN / COLS, rbs = M / TM / ROWS, c_rb = per_col * ROWS * TM * TN;
    const int64_t tile_floats = TM * TN, blocks_n = TN / 8;
    const int64_t used_cores_total = (rows + TM - 1) / TM;  // core rows that hold real output, over all row blocks
    for (int col = 0; col < COLS; col++) {
        for (int64_t rb = 0; rb < rbs; rb++) {
            const int64_t first_row = rb * ROWS * TM;
            if (first_row >= rows) break;
            for (int64_t t = 0; t < per_col; t++) {
                const int64_t at = (col + COLS * t) * TN;
                if (at >= N) continue;
                const float * group = c + (col * rbs + rb) * c_rb + t * (ROWS * tile_floats);
                for (int64_t core = 0; core < ROWS && rb * ROWS + core < used_cores_total; core++) {
                    const float * tile = group + core * tile_floats;
                    const int64_t tile_row = first_row + core * TM;
                    for (int64_t z = 0; z < TM / 8; z++) {
                        for (int64_t jb = 0; jb < blocks_n; jb++) {
                            const float * block = tile + (z * blocks_n + jb) * 64;
                            const int64_t col0 = at + jb * 8;
                            if (col0 >= N) break;
                            const int64_t row0 = tile_row + z * 8;
                            const int64_t nrows = std::min<int64_t>(8, rows - row0);
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
}

// ---- the device -----------------------------------------------------------------------------

using clk = std::chrono::steady_clock;
inline double us_since(clk::time_point t) { return std::chrono::duration<double, std::micro>(clk::now() - t).count(); }

struct npu {
    xrt::device dev;
    xrt::xclbin xb;
    xrt::hw_context ctx;
    xrt::kernel kern;
    xrt::run run;
    struct shape { xrt::bo bo; unsigned words = 0; };
    struct weight { xrt::bo bo; uint64_t sum = 0; };
    std::map<std::tuple<int64_t, int64_t, int64_t>, shape> shapes;
    std::unordered_map<const void *, weight> weights;
    xrt::bo a, c;
    size_t a_cap = 0, c_cap = 0;
    std::mutex mu;
    // accounting
    long long trace_every = 0;
    long long calls = 0;
    double us_pack = 0, us_run = 0, us_unpack = 0, us_w = 0, macs = 0;
    size_t w_bytes = 0;

    void print_trace() const {
        fprintf(stderr, "xdna: %lld calls, %.1f GMAC; pack %.0f ms, run %.0f ms, unpack %.0f ms; weights %.1f MB packed in %.0f ms; %zu shapes\n",
                calls, macs / 1e9, us_pack / 1e3, us_run / 1e3, us_unpack / 1e3, w_bytes / 1e6, us_w / 1e3, shapes.size());
    }
    static xrt::hw_context open(xrt::device & d, xrt::xclbin & x) {
        d.register_xclbin(x);
        return xrt::hw_context(d, x.get_uuid());
    }
    explicit npu(const std::string & path)
        : dev(0u), xb(path), ctx(open(dev, xb)), kern(ctx, "MLIR_AIE"), run(kern) {}

    shape & get_shape(int64_t M, int64_t K, int64_t N) {
        const auto key = std::make_tuple(M, K, N);
        auto it = shapes.find(key);
        if (it != shapes.end()) return it->second;
        const auto words = make_insts(M, K, N);
        shape s;
        s.words = (unsigned) words.size();
        s.bo = xrt::bo(dev, words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
        memcpy(s.bo.map<void *>(), words.data(), words.size() * 4);
        s.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return shapes.emplace(key, std::move(s)).first->second;
    }
    weight & get_weight(const ggml_tensor * w, int64_t N, int64_t K, int64_t Np, int64_t Kp) {
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
        if (it != weights.end() && it->second.sum == sum) return it->second;
        const auto t = clk::now();
        const size_t bytes = (size_t) (Np * Kp / 8 * BLOCK);
        weight x;
        x.sum = sum;
        x.bo = xrt::bo(dev, bytes, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
        pack_w(w, N, K, Np, Kp, x.bo.map<uint8_t *>());
        x.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        us_w += us_since(t);
        w_bytes += bytes;
        weight & slot = weights[w->data];
        slot = std::move(x);
        return slot;
    }
    void mul_mat(const ggml_tensor * w, const float * x, float * out, int64_t rows, int64_t K, int64_t N) {
        const int64_t M = up_to(rows, GRAIN_M), Kp = pad_k(K), Np = up_to(N, GRAIN_N);
        std::lock_guard<std::mutex> lock(mu);
        const size_t an = (size_t) (M * Kp / 8 * BLOCK), cn = (size_t) (M * Np) * sizeof(float);
        if (an > a_cap) { a = xrt::bo(dev, an, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3)); a_cap = an; }
        if (cn > c_cap) { c = xrt::bo(dev, cn, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5)); c_cap = cn; }
        shape & s = get_shape(M, Kp, Np);
        weight & wb = get_weight(w, N, K, Np, Kp);
        auto t = clk::now();
        pack_a(x, rows, M, K, Kp, a.map<uint8_t *>());
        a.sync(XCL_BO_SYNC_BO_TO_DEVICE, an, 0);
        us_pack += us_since(t);
        t = clk::now();
        run.set_arg(0, 3u);
        run.set_arg(1, s.bo);
        run.set_arg(2, s.words);
        run.set_arg(3, a);
        run.set_arg(4, wb.bo);
        run.set_arg(5, c);
        run.start();
        const auto state = run.wait();
        us_run += us_since(t);
        if (state != ERT_CMD_STATE_COMPLETED) throw std::runtime_error("the NPU run ended in state " + std::to_string((int) state));
        t = clk::now();
        c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, cn, 0);
        _mm_mfence();
        unpack_c(c.map<const float *>(), rows, M, N, Np, out);
        us_unpack += us_since(t);
        calls++;
        macs += (double) rows * K * N;
        if (trace_every > 0 && calls % trace_every == 0) print_trace();
    }
};

std::once_flag g_once;
npu * g_npu = nullptr;
bool g_on = false;
int64_t g_min_m = 1;
bool g_force = false;

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
        g_on = true;
        fprintf(stderr, "xdna: NPU %s ready in %.0f ms\n", g_npu->dev.get_info<xrt::info::device::name>().c_str(), us_since(t) / 1e3);
        if (const char * v = getenv("GGML_XDNA_TRACE")) {
            atexit(trace_at_exit);
            if (atoll(v) > 1) g_npu->trace_every = atoll(v);  // also every so many calls, for a server
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "xdna: no NPU, staying on the CPU: %s\n", e.what());
    }
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

}  // namespace

extern "C" bool ggml_xdna_owns(const struct ggml_tensor * dst, int nth) {
    std::call_once(g_once, init);
    if (!g_on) return false;
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
    // Measured on Strix Halo. The NPU: a dispatch, then W at 47 GB/s for every 64 rows, then A and C on
    // the host. The CPU on 8-bit weights: about 25 GMAC/s a thread, up to what memory gives, about
    // 100 GMAC/s together.
    const int64_t Kp = pad_k(K), Np = up_to(N, GRAIN_N), row_blocks = up_to(rows, GRAIN_M) / GRAIN_M;
    const double npu_us = 75.0 + (double) row_blocks * Kp * Np * BLOCK / 8 / 47e3 + 2e-3 * rows * (Kp + Np);
    const double cpu_us = (double) rows * K * N / std::min(25e3 * nth, 100e3);
    return g_force || npu_us < cpu_us;
}

extern "C" void ggml_xdna_mul_mat(struct ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int64_t K = w->ne[0], N = w->ne[1], rows = x->ne[1] * x->ne[2] * x->ne[3];
    try {
        g_npu->mul_mat(w, (const float *) x->data, (float *) dst->data, rows, K, N);
    } catch (const std::exception & e) {
        fprintf(stderr, "xdna: the NPU failed, back to the CPU: %s\n", e.what());
        g_on = false;
        mul_mat_fallback(w, (const float *) x->data, (float *) dst->data, rows, K, N);
    }
}
