// The NPU instruction stream for the matmul in tools/xdna/gemm_bfp.py, generated on the host.
// tools/xdna/check_insts.cpp compares a single matmul with what the IRON compiler produces, word for word.
//
// A job is usually many matmuls: the stream waits for a doorbell in shared memory before each one, so starting a matmul
// needs no driver call. A doorbell costs about 20 us where a job costs about 75 us. The doorbell's path is one more
// stream in the core program: shim DMA, which keeps re-reading a word of host memory, to memory tile DMA, which keeps
// writing it to the same place; the stream polls that place. Doorbell values carry the job's turn, odd or even: what is
// left in the stream from the job before can never look like one of this job's. So a job needs nothing from the host to
// end, and the next can be queued and rung at any time. A job leaves both channels idle (doorbell_stop): a task started on
// top of one still finishing has been seen, next to a busy GPU, never to deliver.
//
// The doorbell has to get through while the job waits for it, so nothing may hold up its shim's reads: a stream on the same
// shim that is blocked for lack of room in the memory tile can. That is why the doorbell's column fetches ahead only what its
// memory tile holds.
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

// The doorbell's path: shim column 7 channel 1 into memory tile column 7 channel 5, both free in the design, to a word at
// the end of that tile's memory. That channel reaches no other tile's. tools/xdna/check_doorbell_word.py sees that nothing
// the compiler allocates gets there.
constexpr int DB_COL = 7, DB_SHIM_BD = 15, DB_MEM_BD = 47;
constexpr int64_t DB_COL_AHEAD = 4 * 8192;  // blocks of weights: the depth of that column's B3 in gemm_bfp.py, times its size
constexpr int DB_WORD_COL = 7;
constexpr uint32_t DB_WORD_ADDR = 524032;
constexpr uint32_t MEM_BD_BASE = 0xA0000, MEM_S2MM5_CTRL = 0xA0628, MEM_S2MM5_QUEUE = 0xA062C;
constexpr uint32_t CHANNEL_BUSY = 0x00080003;  // of a channel's status: running, and its state
constexpr uint32_t JOB_MARK[2] = {1u << 16, 2u << 16};  // or'd with the matmul's number in the job

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
            const int64_t db_ahead = std::min(b_col0, DB_COL_AHEAD);
            for (int col = 0; col < COLS; col++)
                task(groups[0], B_SHIM[col][0], b, (uint64_t) (col * b_col0 * BLOCK), (uint32_t) ((B_SHIM[col][0] == DB_COL ? db_ahead : b_col0) * BLOCK / 4),
                     B_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE, 1, false);
            doorbell_wait(doorbell);
            if (db_ahead < b_col0)
                task(groups[0], DB_COL, b, (uint64_t) ((COLS - 1) * b_col0 * BLOCK + db_ahead * BLOCK), (uint32_t) ((b_col0 - db_ahead) * BLOCK / 4), MM2S0_QUEUE, 1, false);
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
        used[DB_COL].insert(DB_SHIM_BD);
        const uint32_t sb = (uint32_t) DB_SHIM_BD;
        write32(tile_addr(DB_WORD_COL, 1, DB_WORD_ADDR), 0);
        const uint32_t m = tile_addr(DB_COL, 1, MEM_BD_BASE + 0x20 * DB_MEM_BD);
        put({1, 0, m, 48, 4, (1u << 19) | ((uint32_t) DB_MEM_BD << 20) | ((0x80000u + DB_WORD_ADDR) >> 2), 0, 0, 0, 0, 0, 0x80000000u});
        write32(tile_addr(DB_COL, 1, MEM_S2MM5_QUEUE), DB_MEM_BD);
        const uint32_t a = tile_addr(DB_COL, 0, BD_BASE + BD_STRIDE * sb);
        put({1, 0, a, 48, 4, (uint32_t) mailbox.off, 0, 0, 0xc0000000, 0x02000000, 0, 0x02000000u | (1u << 26) | (sb << 27)});
        put({0x81, 48, 0, 0, 0, 0, a + 4, 0, mailbox.arg, 0, (uint32_t) mailbox.off, 0});
        write32(tile_addr(DB_COL, 0, MM2S1_QUEUE), sb);
    }
    void doorbell_wait(uint32_t value) { poll(tile_addr(DB_WORD_COL, 1, DB_WORD_ADDR), value, 0xffffffff); }
    void doorbell_stop() {
        // The shim's channel finishes what it is reading and stops; the memory tile's takes that in and goes on waiting for
        // more, which only a reset ends. What is still in the stream by then is taken in by the next job, and all four words
        // of a transfer being the same, it does not matter how many words that is.
        write32(tile_addr(DB_COL, 0, BD_BASE + BD_STRIDE * (uint32_t) DB_SHIM_BD + 0x1c), 0x02000000u);
        poll(tile_addr(DB_COL, 0, MM2S1_STATUS), 0, CHANNEL_BUSY);
        write32(tile_addr(DB_COL, 1, MEM_S2MM5_CTRL), 2);
        write32(tile_addr(DB_COL, 1, MEM_S2MM5_CTRL), 0);
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
