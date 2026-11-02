// Matrix multiplies on the AMD XDNA2 NPU, from inside the CPU backend.
//
// The NPU is good at one thing here: C[M x N] = A[M x K] * W[N x K]^T where W is a constant weight
// matrix. The core program (tools/xdna/gemm_bfp.py, mm_bfp_f32.cc) uses all 32 cores and serves
// every shape with M a multiple of 64, K of 64 and N of 512; only the instruction stream changes,
// and that is generated here. Weights and activations go to the NPU as bfp16ebs8: eight values
// sharing one exponent byte, each with a signed 8-bit mantissa. Sums stay in float32.
//
// What costs time is about 70 us per dispatch in the driver and firmware, plus moving the whole of
// W from memory for each call at up to 50 GB/s; the arithmetic itself is negligible. So a call is
// worth sending only when K * N is large and M is not tiny.
//
// Settings, read once:
//   GGML_XDNA=1               turn it on
//   GGML_XDNA_XCLBIN=<file>   the compiled core program
//   GGML_XDNA_MIN_M=<n>       smallest M sent to the NPU (default 8)
//   GGML_XDNA_TRACE=1         print where the time went at exit
#include "ggml-xdna.h"
#include "ggml-impl.h"

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
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int64_t TM = 16, TK = 64, TN = 64, ROWS = 4, COLS = 8;  // tile sizes and core grid of the design
constexpr int64_t GRAIN_M = TM * ROWS, GRAIN_N = TN * COLS;
constexpr size_t BLOCK = 9;

// ---- instruction stream ---------------------------------------------------------------------
// Reproduces what IRON lowers gemm_bfp.py to, word for word (tools/xdna/check_insts.py compares).

// Where the placer put each stream. Fixed by the design, not by the shape.
constexpr int A_SHIM[ROWS][2] = {{3, 0}, {3, 1}, {2, 0}, {4, 0}};
constexpr int B_SHIM[COLS][2] = {{0, 0}, {1, 0}, {2, 1}, {4, 1}, {5, 0}, {5, 1}, {6, 0}, {7, 0}};
constexpr uint32_t ARG_A = 0, ARG_B = 1, ARG_C = 2;
constexpr uint32_t BD_BASE = 0x1d000, BD_STRIDE = 0x20, S2MM0_CTRL = 0x1d200, S2MM0_QUEUE = 0x1d204;
constexpr uint32_t MM2S0_QUEUE = 0x1d214, MM2S1_QUEUE = 0x1d21c;
constexpr uint32_t RTP_NMID = 0x09000, RTP_TILES = 0x09004, BARRIER_LOCK = 0x1f060;

uint32_t tile_addr(int col, int row, uint32_t off) { return ((uint32_t) col << 25) | ((uint32_t) row << 20) | off; }

struct stream {
    std::vector<uint32_t> w;
    uint32_t ops = 0;
    std::set<int> used[COLS];

    void put(std::initializer_list<uint32_t> words) { w.insert(w.end(), words); ops++; }
    void write32(uint32_t addr, uint32_t v) { put({0, 0, addr, 0, v, 24}); }
    int bd(int col) { int b = 0; while (used[col].count(b)) b++; used[col].insert(b); return b; }
    void task(std::vector<std::pair<int, int>> & group, int col, uint32_t arg, uint32_t off, uint32_t words, uint32_t queue, uint32_t repeat, bool drain) {
        const int b = bd(col);
        group.push_back({col, b});
        const uint32_t a = tile_addr(col, 0, BD_BASE + BD_STRIDE * (uint32_t) b);
        put({1, 0, a, 48, words, off, 0, 0, 0xc0000000, 0x02000000, 0, 0x02000000});
        put({0x81, 48, 0, 0, 0, 0, a + 4, 0, arg, 0, off, 0});
        if (drain) put({3, 0, tile_addr(col, 0, S2MM0_CTRL), 0, 0xf00, 0x1f00, 28});
        write32(tile_addr(col, 0, queue), (drain ? 0x80000000u : 0u) | ((repeat - 1) << 16) | (uint32_t) b);
    }
    void wait_all(std::vector<std::pair<int, int>> & group) {
        for (int c = 0; c < COLS; c++) put({0x80, 16, (uint32_t) c << 16, 0x00010100});
        for (auto [col, b] : group) used[col].erase(b);
        group.clear();
    }
};

std::vector<uint32_t> make_insts(int64_t M, int64_t K, int64_t N) {
    stream s;
    const int64_t nk = K / TK, per_col = N / TN / COLS, rbs = M / TM / ROWS;
    for (int row = 2; row < 2 + ROWS; row++)
        for (int col = 0; col < COLS; col++) {
            s.write32(tile_addr(col, row, RTP_NMID), (uint32_t) (nk - 1));
            s.write32(tile_addr(col, row, RTP_TILES), (uint32_t) ((M / TM) * (N / TN) / (ROWS * COLS)));
            s.write32(tile_addr(col, row, BARRIER_LOCK), 1);
        }
    const int64_t a_row = TM * K / 8, b_col = (N / COLS) * K / 8, c_rb = per_col * ROWS * TM * TN;
    std::vector<std::vector<std::pair<int, int>>> groups(rbs);
    for (int64_t rb = 0; rb < rbs; rb++) {
        if (rb >= 4) s.wait_all(groups[rb - 4]);
        for (int c = 0; c < COLS; c++) {
            s.task(groups[rb], c, ARG_C, (uint32_t) ((c * rbs + rb) * c_rb * 4), (uint32_t) c_rb, S2MM0_QUEUE, 1, true);
            if (c < ROWS)
                s.task(groups[rb], A_SHIM[c][0], ARG_A, (uint32_t) ((rb * ROWS + c) * a_row * BLOCK), (uint32_t) (a_row * BLOCK / 4),
                       A_SHIM[c][1] ? MM2S1_QUEUE : MM2S0_QUEUE, (uint32_t) per_col, false);
            s.task(groups[rb], B_SHIM[c][0], ARG_B, (uint32_t) (c * b_col * BLOCK), (uint32_t) (b_col * BLOCK / 4),
                   B_SHIM[c][1] ? MM2S1_QUEUE : MM2S0_QUEUE, 1, false);
        }
    }
    for (int64_t rb = std::max<int64_t>(0, rbs - 4); rb < rbs; rb++) s.wait_all(groups[rb]);
    std::vector<uint32_t> out = {0x06040100, 0x108, s.ops, (uint32_t) ((s.w.size() + 4) * 4)};
    out.insert(out.end(), s.w.begin(), s.w.end());
    return out;
}

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

void pack_a(const float * x, int64_t rows, int64_t M, int64_t K, uint8_t * o) {
    const int64_t nk = K / TK;
    memset(o, 0, (size_t) (M * K / 8 * BLOCK));
    for (int64_t r = 0; r < rows; r++)
        for (int64_t cb = 0; cb < K / 8; cb++)
            encode_block(x + r * K + cb * 8, o + BLOCK * tiled(r / TM, r % TM, cb, TM, nk));
}

void pack_w(const ggml_tensor * w, int64_t N, int64_t K, uint8_t * o) {
    const int64_t nk = K / TK, per_col = N / TN / COLS;
    std::vector<float> row(K);
    const auto * traits = ggml_get_type_traits(w->type);
    for (int64_t j = 0; j < N; j++) {
        const char * p = (const char *) w->data + j * w->nb[1];
        if (w->type == GGML_TYPE_F32) memcpy(row.data(), p, K * sizeof(float));
        else traits->to_float(p, row.data(), K);
        const int64_t g = j / TN, first = (g % COLS) * per_col + g / COLS;
        for (int64_t cb = 0; cb < K / 8; cb++)
            encode_block(row.data() + cb * 8, o + BLOCK * tiled(first, j % TN, cb, TN, nk));
    }
}

// The real rows of the output, out of the core tiles.
void unpack_c(const float * c, int64_t rows, int64_t M, int64_t N, float * dst) {
    const int64_t per_col = N / TN / COLS, rbs = M / TM / ROWS, c_rb = per_col * ROWS * TM * TN;
    for (int64_t r = 0; r < rows; r++) {
        const int64_t rb = r / (ROWS * TM), core = (r / TM) % ROWS, rr = r % TM;
        float * out = dst + r * N;
        for (int col = 0; col < COLS; col++)
            for (int64_t t = 0; t < per_col; t++) {
                const float * tile = c + (col * rbs + rb) * c_rb + t * (ROWS * TM * TN) + core * (TM * TN);
                const int64_t at = (col + COLS * t) * TN;
                for (int64_t jb = 0; jb < TN / 8; jb++)
                    memcpy(out + at + jb * 8, tile + ((rr / 8) * (TN / 8) + jb) * 64 + (rr % 8) * 8, 8 * sizeof(float));
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
    struct weight { xrt::bo bo; };
    std::map<std::tuple<int64_t, int64_t, int64_t>, shape> shapes;
    std::unordered_map<const void *, weight> weights;
    xrt::bo a, c;
    size_t a_cap = 0, c_cap = 0;
    std::mutex mu;
    // accounting
    long long calls = 0;
    double us_pack = 0, us_run = 0, us_unpack = 0, us_w = 0, macs = 0;
    size_t w_bytes = 0;

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
    weight & get_weight(const ggml_tensor * w, int64_t N, int64_t K) {
        auto it = weights.find(w->data);
        if (it != weights.end()) return it->second;
        const auto t = clk::now();
        const size_t bytes = (size_t) (N * K / 8 * BLOCK);
        weight x;
        x.bo = xrt::bo(dev, bytes, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
        pack_w(w, N, K, x.bo.map<uint8_t *>());
        x.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        us_w += us_since(t);
        w_bytes += bytes;
        return weights.emplace(w->data, std::move(x)).first->second;
    }
    void mul_mat(const ggml_tensor * w, const float * x, float * out, int64_t rows, int64_t K, int64_t N) {
        const int64_t M = (rows + GRAIN_M - 1) / GRAIN_M * GRAIN_M;
        std::lock_guard<std::mutex> lock(mu);
        const size_t an = (size_t) (M * K / 8 * BLOCK), cn = (size_t) (M * N) * sizeof(float);
        if (an > a_cap) { a = xrt::bo(dev, an, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3)); a_cap = an; }
        if (cn > c_cap) { c = xrt::bo(dev, cn, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5)); c_cap = cn; }
        shape & s = get_shape(M, K, N);
        weight & wb = get_weight(w, N, K);
        auto t = clk::now();
        pack_a(x, rows, M, K, a.map<uint8_t *>());
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
        unpack_c(c.map<const float *>(), rows, M, N, out);
        us_unpack += us_since(t);
        calls++;
        macs += (double) rows * K * N;
    }
};

std::once_flag g_once;
npu * g_npu = nullptr;
bool g_on = false;
int64_t g_min_m = 8;

void trace_at_exit() {
    if (!g_npu) return;
    npu & n = *g_npu;
    fprintf(stderr, "xdna: %lld calls, %.1f GMAC; pack %.0f ms, run %.0f ms, unpack %.0f ms; weights %.1f MB packed in %.0f ms; %zu shapes\n",
            n.calls, n.macs / 1e9, n.us_pack / 1e3, n.us_run / 1e3, n.us_unpack / 1e3, n.w_bytes / 1e6, n.us_w / 1e3, n.shapes.size());
}

void init() {
    const char * on = getenv("GGML_XDNA");
    if (!on || !*on || !strcmp(on, "0")) return;
    if (const char * v = getenv("GGML_XDNA_MIN_M")) g_min_m = atoll(v);
    const char * path = getenv("GGML_XDNA_XCLBIN");
    if (!path || !*path) {
        fprintf(stderr, "xdna: GGML_XDNA_XCLBIN is not set; staying on the CPU\n");
        return;
    }
    try {
        const auto t = clk::now();
        g_npu = new npu(path);  // never freed: the driver cleans up when the process ends
        g_on = true;
        fprintf(stderr, "xdna: NPU %s ready in %.0f ms\n", g_npu->dev.get_info<xrt::info::device::name>().c_str(), us_since(t) / 1e3);
        if (getenv("GGML_XDNA_TRACE")) atexit(trace_at_exit);
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

extern "C" bool ggml_xdna_owns(const struct ggml_tensor * dst) {
    std::call_once(g_once, init);
    if (!g_on) return false;
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int64_t K = w->ne[0], N = w->ne[1];
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (K < 2 * TK || K % TK != 0 || N < GRAIN_N || N % GRAIN_N != 0) return false;
    // Weights only: a constant the NPU can keep a copy of.
    if (!ggml_is_quantized(w->type)) return false;
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(dst)) return false;
    const int64_t rows = x->ne[1] * x->ne[2] * x->ne[3];
    return rows >= g_min_m && rows <= 1024;
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
