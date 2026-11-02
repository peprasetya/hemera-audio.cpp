// The NPU instruction stream for the matmul in tools/xdna/gemm_bfp.py, generated on the host.
// tools/xdna/check_insts.cpp compares a single matmul with what the IRON compiler produces, word for word.
//
// A job is usually many matmuls: the stream waits for a doorbell in shared memory before each one, so starting a matmul
// needs no driver call. A doorbell costs about 20 us where a job costs about 75 us. The doorbell's path is one more
// stream in the core program: shim DMA, which keeps re-reading a word of host memory, to memory tile DMA, which keeps
// writing it to the same place; the stream polls that place.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <utility>
#include <vector>

namespace ggml_xdna {

constexpr int64_t TM = 8, TK = 64, TN = 64, ROWS = 4, COLS = 8;  // tile sizes and core grid of the design
constexpr int64_t GRAIN_M = TM * ROWS, GRAIN_N = TN * COLS;
constexpr size_t BLOCK = 9;
constexpr int64_t B_GROUP = 16;  // a column's weights transfer is a whole number of these many tiles

// Where the placer put each stream. Fixed by the design, not by the shape.
constexpr int A_SHIM[ROWS][2] = {{3, 0}, {3, 1}, {2, 0}, {4, 0}};
constexpr int B_SHIM[COLS][2] = {{0, 0}, {1, 0}, {2, 1}, {4, 1}, {5, 0}, {5, 1}, {6, 0}, {7, 0}};
constexpr uint32_t BD_BASE = 0x1d000, BD_STRIDE = 0x20, S2MM0_CTRL = 0x1d200, S2MM0_QUEUE = 0x1d204;
constexpr uint32_t MM2S0_QUEUE = 0x1d214, MM2S1_QUEUE = 0x1d21c;
constexpr uint32_t MM2S0_STATUS = 0x1d228, MM2S1_STATUS = 0x1d22c, QUEUE_FULL = 0x400000;
constexpr uint32_t RTP_NMID = 0x08800, RTP_TILES = 0x08804, BARRIER_LOCK = 0x1f060;

// The doorbell's path: shim column 7 channel 1 into memory tile column 7 channel 5, both free in the design.
constexpr int DB_COL = 7, DB_SHIM_BD = 15, DB_MEM_BD = 47;
constexpr uint32_t DB_MEM_ADDR = 0x70000;  // inside the memory tile, past what the design allocates
constexpr uint32_t MEM_BD_BASE = 0xA0000, MEM_S2MM5_QUEUE = 0xA062C;

inline uint32_t tile_addr(int col, int row, uint32_t off) { return ((uint32_t) col << 25) | ((uint32_t) row << 20) | off; }

// An operand: which kernel argument, how far into its buffer.
struct place {
    uint32_t arg;
    uint64_t off;
};

struct stream {
    std::vector<uint32_t> w;
    uint32_t ops = 0;
    std::set<int> used[COLS];

    void put(std::initializer_list<uint32_t> words) { w.insert(w.end(), words); ops++; }
    void write32(uint32_t addr, uint32_t v) { put({0, 0, addr, 0, v, 24}); }
    void poll(uint32_t addr, uint32_t value, uint32_t mask) { put({4, 0, addr, 0, value, mask, 28}); }
    int bd(int col) { int b = 0; while (used[col].count(b)) b++; used[col].insert(b); return b; }
    void task(std::vector<std::pair<int, int>> & group, int col, const place & p, uint64_t off, uint32_t words, uint32_t queue, uint32_t repeat, bool drain, bool room = false) {
        const int b = bd(col);
        group.push_back({col, b});
        const uint32_t a = tile_addr(col, 0, BD_BASE + BD_STRIDE * (uint32_t) b);
        const uint64_t at = p.off + off;
        put({1, 0, a, 48, words, (uint32_t) at, (uint32_t) (at >> 32), 0, 0xc0000000, 0x02000000, 0, 0x02000000});
        put({0x81, 48, 0, 0, 0, 0, a + 4, 0, p.arg, 0, (uint32_t) at, (uint32_t) (at >> 32)});
        if (drain) put({3, 0, tile_addr(col, 0, S2MM0_CTRL), 0, 0xf00, 0x1f00, 28});
        if (room) poll(tile_addr(col, 0, queue == MM2S0_QUEUE ? MM2S0_STATUS : MM2S1_STATUS), 0, QUEUE_FULL);
        write32(tile_addr(col, 0, queue), (drain ? 0x80000000u : 0u) | ((repeat - 1) << 16) | (uint32_t) b);
    }
    void wait_all(std::vector<std::pair<int, int>> & group) {
        for (int c = 0; c < COLS; c++) put({0x80, 16, (uint32_t) c << 16, 0x00010100});
        for (auto [col, b] : group) used[col].erase(b);
        group.clear();
    }
    // With a doorbell number, the weights start crossing right away and the rest waits for the doorbell: the memory tiles
    // fill while the host is still working, so the host's time and the transfer's overlap.
    void matmul(int64_t M, int64_t K, int64_t N, const place & a, const place & b, const place & c, uint32_t doorbell = 0) {
        const int64_t nk = K / TK, per_col = N / TN / COLS, rbs = M / TM / ROWS;
        const int64_t b_col0 = (N / COLS) * K / 8;
        std::vector<std::vector<std::pair<int, int>>> groups(rbs);
        if (doorbell != 0) {
            for (int col = 0; col < COLS; col++)
                task(groups[0], B_SHIM[col][0], b, (uint64_t) (col * b_col0 * BLOCK), (uint32_t) (b_col0 * BLOCK / 4),
                     B_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE, 1, false);
            doorbell_wait(doorbell);
        }
        for (int row = 2; row < 2 + ROWS; row++)
            for (int col = 0; col < COLS; col++) {
                write32(tile_addr(col, row, RTP_NMID), (uint32_t) (nk - 1));
                write32(tile_addr(col, row, RTP_TILES), (uint32_t) ((M / TM) * (N / TN) / (ROWS * COLS)));
                write32(tile_addr(col, row, BARRIER_LOCK), 1);
            }
        const int64_t a_row = TM * K / 8, b_col = (N / COLS) * K / 8, c_rb = per_col * ROWS * TM * TN;
        for (int64_t rb = 0; rb < rbs; rb++) {
            if (rb >= 4) wait_all(groups[rb - 4]);
            for (int col = 0; col < COLS; col++) {
                task(groups[rb], col, c, (uint64_t) ((col * rbs + rb) * c_rb * 4), (uint32_t) c_rb, S2MM0_QUEUE, 1, true);
                if (col < ROWS)
                    task(groups[rb], A_SHIM[col][0], a, (uint64_t) ((rb * ROWS + col) * a_row * BLOCK), (uint32_t) (a_row * BLOCK / 4),
                         A_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE, (uint32_t) per_col, false, rb >= 4);
                if (rb > 0 || doorbell == 0)
                    task(groups[rb], B_SHIM[col][0], b, (uint64_t) (col * b_col * BLOCK), (uint32_t) (b_col * BLOCK / 4),
                         B_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE, 1, false, rb >= 4);
            }
        }
        for (int64_t rb = std::max<int64_t>(0, rbs - 4); rb < rbs; rb++) wait_all(groups[rb]);
    }
    // The doorbell: buffer descriptors that repeat themselves for ever. `mailbox` is a word of host memory.
    void doorbell_start(const place & mailbox) {
        write32(tile_addr(DB_COL, 1, DB_MEM_ADDR), 0);
        const uint32_t m = tile_addr(DB_COL, 1, MEM_BD_BASE + 0x20 * DB_MEM_BD);
        put({1, 0, m, 48, 4, (1u << 19) | ((uint32_t) DB_MEM_BD << 20) | ((0x80000u + DB_MEM_ADDR) >> 2), 0, 0, 0, 0, 0, 0x80000000u});
        write32(tile_addr(DB_COL, 1, MEM_S2MM5_QUEUE), DB_MEM_BD);
        const uint32_t a = tile_addr(DB_COL, 0, BD_BASE + BD_STRIDE * DB_SHIM_BD);
        put({1, 0, a, 48, 4, (uint32_t) mailbox.off, 0, 0, 0xc0000000, 0x02000000, 0, 0x02000000u | (1u << 26) | ((uint32_t) DB_SHIM_BD << 27)});
        put({0x81, 48, 0, 0, 0, 0, a + 4, 0, mailbox.arg, 0, (uint32_t) mailbox.off, 0});
        write32(tile_addr(DB_COL, 0, MM2S1_QUEUE), DB_SHIM_BD);
    }
    void doorbell_wait(uint32_t value) { poll(tile_addr(DB_COL, 1, DB_MEM_ADDR), value, 0xffffffff); }
    void doorbell_stop() {
        // Words still in the stream when this job stops reach the next job's memory tile word. They must not look like a
        // doorbell, so stop only once the host has put the mailbox back to 0 and that has come through.
        doorbell_wait(0);
        // no next descriptor: both channels finish the transfer they are in and stop
        write32(tile_addr(DB_COL, 0, BD_BASE + BD_STRIDE * DB_SHIM_BD + 0x1c), 0x02000000u);
        write32(tile_addr(DB_COL, 1, MEM_BD_BASE + 0x20 * DB_MEM_BD + 4), (0x80000u + DB_MEM_ADDR) >> 2);
    }
    std::vector<uint32_t> finish() const {
        std::vector<uint32_t> out = {0x06040100, 0x108, ops, (uint32_t) ((w.size() + 4) * 4)};
        out.insert(out.end(), w.begin(), w.end());
        return out;
    }
};

// One matmul as a job of its own, with the operands at the start of arguments 0, 1 and 2: what IRON compiles.
inline std::vector<uint32_t> make_insts(int64_t M, int64_t K, int64_t N) {
    stream s;
    s.matmul(M, K, N, {0, 0}, {1, 0}, {2, 0});
    return s.finish();
}

}  // namespace ggml_xdna
