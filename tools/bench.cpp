// Phase 3 benchmark: how deep the search reaches within a per-move time
// budget on a representative mid-game position, plus nodes/sec.
#include "bitboard.hpp"
#include "board.hpp"
#include "eval.hpp"
#include "search.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace oog;

namespace {

std::uint64_t g_seed = 0x123456789ABCDEF0ULL;
std::uint64_t splitmix64() {
    g_seed += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = g_seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

template <typename F>
void bench(const char* name, long iters, F f) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    long long ops = 0;
    for (long i = 0; i < iters; ++i) ops += f();
    const auto t1 = clock::now();
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::printf("%-32s %12.1f ns/op   %10.2f Mops/s\n", name, ns / ops,
                ops / ns * 1e3);
}

bool randomLegalMove(Board& b, int& r, int& c) {
    for (int a = 0; a < 300; ++a) {
        const int rr = static_cast<int>(splitmix64() % N);
        const int cc = static_cast<int>(splitmix64() % N);
        if (b.isLegal({rr, cc})) {
            r = rr;
            c = cc;
            return true;
        }
    }
    return false;
}

long boardMakeUndo() {
    Board b;
    int moves = 0;
    for (int m = 0; m < 60 && !b.isGameOver(); ++m) {
        int r = 0;
        int c = 0;
        if (!randomLegalMove(b, r, c)) break;
        b.makeMove({r, c});
        ++moves;
    }
    for (int m = 0; m < moves; ++m) b.undoMove();
    return 2LL * moves;
}

long bitboardMakeUndo() {
    BitBoard bb;
    std::vector<int> hist;
    int color = 0;
    for (int m = 0; m < 60; ++m) {
        int sq = -1;
        for (int a = 0; a < 300; ++a) {
            const int s = static_cast<int>(splitmix64() % BitBoard::SIZE);
            if (BitBoard::isPlayableSq(s) && bb.isEmpty(s)) {
                sq = s;
                break;
            }
        }
        if (sq < 0) break;
        bb.place(color, sq);
        (void)bb.winsAt(color, sq);
        hist.push_back(sq);
        color ^= 1;
    }
    for (int i = static_cast<int>(hist.size()) - 1; i >= 0; --i) {
        bb.remove(i & 1, hist[static_cast<std::size_t>(i)]);
    }
    return 2LL * static_cast<long long>(hist.size());
}

long bitboardWinCheck() {
    BitBoard bb;
    int color = 0;
    for (int m = 0; m < 60; ++m) {
        int sq = -1;
        for (int a = 0; a < 300; ++a) {
            const int s = static_cast<int>(splitmix64() % BitBoard::SIZE);
            if (BitBoard::isPlayableSq(s) && bb.isEmpty(s)) {
                sq = s;
                break;
            }
        }
        if (sq < 0) break;
        bb.place(color, sq);
        color ^= 1;
    }
    (void)bb.hasFive(0);
    (void)bb.hasFive(1);
    return 2;
}

long movegen() {
    BitBoard bb;
    int color = 0;
    for (int m = 0; m < 60; ++m) {
        int sq = -1;
        for (int a = 0; a < 300; ++a) {
            const int s = static_cast<int>(splitmix64() % BitBoard::SIZE);
            if (BitBoard::isPlayableSq(s) && bb.isEmpty(s)) {
                sq = s;
                break;
            }
        }
        if (sq < 0) break;
        bb.place(color, sq);
        color ^= 1;
    }
    std::vector<int> out;
    bb.genMovesNear(out);
    return static_cast<long long>(out.size());
}

long evaluatePos() {
    BitBoard bb;
    int color = 0;
    for (int m = 0; m < 60; ++m) {
        int sq = -1;
        for (int a = 0; a < 300; ++a) {
            const int s = static_cast<int>(splitmix64() % BitBoard::SIZE);
            if (BitBoard::isPlayableSq(s) && bb.isEmpty(s)) {
                sq = s;
                break;
            }
        }
        if (sq < 0) break;
        bb.place(color, sq);
        color ^= 1;
    }
    (void)evaluate(bb, 0);
    return 1;
}

long evaluateMoveAll() {
    BitBoard bb;
    int color = 0;
    for (int m = 0; m < 60; ++m) {
        int sq = -1;
        for (int a = 0; a < 300; ++a) {
            const int s = static_cast<int>(splitmix64() % BitBoard::SIZE);
            if (BitBoard::isPlayableSq(s) && bb.isEmpty(s)) {
                sq = s;
                break;
            }
        }
        if (sq < 0) break;
        bb.place(color, sq);
        color ^= 1;
    }
    std::vector<int> moves;
    bb.genMovesNear(moves);
    long n = 0;
    for (const int m : moves) {
        (void)evaluateMove(bb, 0, m);
        ++n;
    }
    return n;
}

// Deterministic quiet mid-game position (30 stones, no immediate forced win
// for either side, verified against vcfWin), used for search depth/nodes.
BitBoard midGamePosition() {
    BitBoard bb;
    const int black[][2] = {{0, 0},  {4, 2},  {4, 3},  {5, 14}, {6, 8},
                            {8, 9},  {9, 3},  {9, 9},  {10, 0}, {11, 13},
                            {12, 4}, {12, 6}, {12, 8}, {13, 4}, {13, 11}};
    const int white[][2] = {{1, 8},  {2, 6},  {2, 10}, {4, 5},  {6, 9},
                            {7, 1},  {7, 4},  {7, 11}, {8, 3},  {8, 4},
                            {10, 1}, {11, 7}, {12, 2}, {12, 13}, {13, 9}};
    for (const auto& s : black) bb.place(0, sqOf(s[0], s[1]));
    for (const auto& s : white) bb.place(1, sqOf(s[0], s[1]));
    return bb;
}

void benchSearchDepth(long long budgetMs) {
    BitBoard bb = midGamePosition();
    Search search;
    SearchLimits lim;
    lim.maxTimeMs = budgetMs;
    const auto t0 = std::chrono::steady_clock::now();
    const SearchResult r = search.search(bb, 0, lim);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec =
        std::chrono::duration<double>(t1 - t0).count();
    const double nps = static_cast<double>(r.nodes) / sec;
    std::printf("search depth=%d  nodes=%llu  %6.1f Mnps  %5.1f ms  "
                "best=(%d,%d) score=%d%s\n",
                r.depth, static_cast<unsigned long long>(r.nodes), nps / 1e6,
                sec * 1e3, BitBoard::boardRow(r.bestMove),
                BitBoard::boardCol(r.bestMove), r.score,
                r.forcedWin ? " (forced)" : "");
}

long benchNodesAtDepth(int depth) {
    BitBoard bb = midGamePosition();
    Search search;
    SearchLimits lim;
    lim.maxDepth = depth;
    const SearchResult r = search.search(bb, 0, lim);
    std::printf("depth=%d  nodes=%llu  ttHits=%llu/%llu\n", depth,
                static_cast<unsigned long long>(r.nodes),
                static_cast<unsigned long long>(search.ttHits()),
                static_cast<unsigned long long>(search.ttProbes()));
    return static_cast<long long>(r.nodes);
}

}  // namespace

int main() {
    std::printf("Outer-Open Gomoku benchmarks (arm64 host)\n");
    bench("board make+undo (win-detect)", 20000, boardMakeUndo);
    bench("bitboard place+remove (win-detect)", 20000, bitboardMakeUndo);
    bench("bitboard hasFive x2", 200000, bitboardWinCheck);
    bench("bitboard genMovesNear", 100000, movegen);
    bench("evaluate (mid-game position)", 100000, evaluatePos);
    bench("evaluateMove (all candidates)", 10000, evaluateMoveAll);

    std::printf("\n--- search (mid-game position, maxDepth) ---\n");
    benchNodesAtDepth(2);
    benchNodesAtDepth(3);
    benchNodesAtDepth(4);
    benchNodesAtDepth(5);

    std::printf("\n--- search (mid-game position, time budget) ---\n");
    benchSearchDepth(250);
    benchSearchDepth(1000);
    return 0;
}