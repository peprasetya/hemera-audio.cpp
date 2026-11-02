// The NPU instruction stream for one shape of the matmul in tools/xdna/gemm_bfp.py, generated on the
// host. tools/xdna/check_insts.cpp compares it with what the IRON compiler produces, word for word.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <utility>
#include <vector>

namespace ggml_xdna {

constexpr int64_t TM = 16, TK = 64, TN = 64, ROWS = 4, COLS = 8;  // tile sizes and core grid of the design
constexpr int64_t GRAIN_M = TM * ROWS, GRAIN_N = TN * COLS;
constexpr size_t BLOCK = 9;

// Where the placer put each stream. Fixed by the design, not by the shape.
constexpr int A_SHIM[ROWS][2] = {{3, 0}, {3, 1}, {2, 0}, {4, 0}};
constexpr int B_SHIM[COLS][2] = {{0, 0}, {1, 0}, {2, 1}, {4, 1}, {5, 0}, {5, 1}, {6, 0}, {7, 0}};
constexpr uint32_t ARG_A = 0, ARG_B = 1, ARG_C = 2;
constexpr uint32_t BD_BASE = 0x1d000, BD_STRIDE = 0x20, S2MM0_CTRL = 0x1d200, S2MM0_QUEUE = 0x1d204;
constexpr uint32_t MM2S0_QUEUE = 0x1d214, MM2S1_QUEUE = 0x1d21c;
constexpr uint32_t MM2S0_STATUS = 0x1d228, MM2S1_STATUS = 0x1d22c, QUEUE_FULL = 0x400000;
constexpr uint32_t RTP_NMID = 0x09000, RTP_TILES = 0x09004, BARRIER_LOCK = 0x1f060;

inline uint32_t tile_addr(int col, int row, uint32_t off) { return ((uint32_t) col << 25) | ((uint32_t) row << 20) | off; }

struct stream {
    std::vector<uint32_t> w;
    uint32_t ops = 0;
    std::set<int> used[COLS];

    void put(std::initializer_list<uint32_t> words) { w.insert(w.end(), words); ops++; }
    void write32(uint32_t addr, uint32_t v) { put({0, 0, addr, 0, v, 24}); }
    int bd(int col) { int b = 0; while (used[col].count(b)) b++; used[col].insert(b); return b; }
    void task(std::vector<std::pair<int, int>> & group, int col, uint32_t arg, uint32_t off, uint32_t words, uint32_t queue, uint32_t repeat, bool drain, bool room = false) {
        const int b = bd(col);
        group.push_back({col, b});
        const uint32_t a = tile_addr(col, 0, BD_BASE + BD_STRIDE * (uint32_t) b);
        put({1, 0, a, 48, words, off, 0, 0, 0xc0000000, 0x02000000, 0, 0x02000000});
        put({0x81, 48, 0, 0, 0, 0, a + 4, 0, arg, 0, off, 0});
        if (drain) put({3, 0, tile_addr(col, 0, S2MM0_CTRL), 0, 0xf00, 0x1f00, 28});
        if (room) put({4, 0, tile_addr(col, 0, queue == MM2S0_QUEUE ? MM2S0_STATUS : MM2S1_STATUS), 0, 0, QUEUE_FULL, 28});
        write32(tile_addr(col, 0, queue), (drain ? 0x80000000u : 0u) | ((repeat - 1) << 16) | (uint32_t) b);
    }
    void wait_all(std::vector<std::pair<int, int>> & group) {
        for (int c = 0; c < COLS; c++) put({0x80, 16, (uint32_t) c << 16, 0x00010100});
        for (auto [col, b] : group) used[col].erase(b);
        group.clear();
    }
};

inline std::vector<uint32_t> make_insts(int64_t M, int64_t K, int64_t N) {
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
                       A_SHIM[c][1] ? MM2S1_QUEUE : MM2S0_QUEUE, (uint32_t) per_col, false, rb >= 4);
            s.task(groups[rb], B_SHIM[c][0], ARG_B, (uint32_t) (c * b_col * BLOCK), (uint32_t) (b_col * BLOCK / 4),
                   B_SHIM[c][1] ? MM2S1_QUEUE : MM2S0_QUEUE, 1, false, rb >= 4);
        }
    }
    for (int64_t rb = std::max<int64_t>(0, rbs - 4); rb < rbs; rb++) s.wait_all(groups[rb]);
    std::vector<uint32_t> out = {0x06040100, 0x108, s.ops, (uint32_t) ((s.w.size() + 4) * 4)};
    out.insert(out.end(), s.w.begin(), s.w.end());
    return out;
}

}  // namespace ggml_xdna
