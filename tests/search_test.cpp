#include "doctest.h"

#include "bitboard.hpp"
#include "search.hpp"

#include <vector>

using namespace oog;

namespace {
void putBlack(BitBoard& bb, int r, int c) { bb.place(0, sqOf(r, c)); }
void putWhite(BitBoard& bb, int r, int c) { bb.place(1, sqOf(r, c)); }

int sq(int r, int c) { return sqOf(r, c); }

bool onRing(int r, int c) {
    return r < 2 || r >= N - 2 || c < 2 || c >= N - 2;
}
}  // namespace

TEST_CASE("search: VCF finds the double-four kill") {
    // Black stones form two fours-with-gap around (7,7); playing it creates
    // two open fours the defender cannot block both of.
    BitBoard bb;
    putBlack(bb, 7, 5);
    putBlack(bb, 7, 6);
    putBlack(bb, 7, 8);
    putBlack(bb, 5, 7);
    putBlack(bb, 6, 7);
    putBlack(bb, 8, 7);
    putWhite(bb, 0, 0);
    putWhite(bb, 14, 14);

    SearchLimits lim;
    lim.maxDepth = 2;
    Search s;
    const SearchResult r = s.search(bb, 0, lim);
    CHECK(r.forcedWin);
    CHECK(r.bestMove == sq(7, 7));
    CHECK(r.score == MATE);
}

TEST_CASE("search: vcfWin returns the double-four square directly") {
    BitBoard bb;
    putBlack(bb, 7, 5);
    putBlack(bb, 7, 6);
    putBlack(bb, 7, 8);
    putBlack(bb, 5, 7);
    putBlack(bb, 6, 7);
    putBlack(bb, 8, 7);
    putWhite(bb, 0, 0);

    Search s;
    CHECK(s.vcfWin(bb, 0, 12) == sq(7, 7));
}

TEST_CASE("search: completes an open four (win in one)") {
    BitBoard bb;
    putBlack(bb, 7, 7);
    putBlack(bb, 7, 8);
    putBlack(bb, 7, 9);
    putBlack(bb, 7, 10);
    putWhite(bb, 3, 3);

    SearchLimits lim;
    lim.maxDepth = 1;
    Search s;
    const SearchResult r = s.search(bb, 0, lim);
    CHECK(r.forcedWin);
    const bool winsOnEitherEnd =
        r.bestMove == sq(7, 6) || r.bestMove == sq(7, 11);
    CHECK(winsOnEitherEnd);
    bb.place(0, r.bestMove);
    CHECK(bb.winsAt(0, r.bestMove));
}

TEST_CASE("search: blocks the opponent's closed four") {
    // White has four in a row with the right end blocked by Black; the only
    // saving move for Black is the single winning point (7,5).
    BitBoard bb;
    putBlack(bb, 7, 10);
    putWhite(bb, 7, 6);
    putWhite(bb, 7, 7);
    putWhite(bb, 7, 8);
    putWhite(bb, 7, 9);

    SearchLimits lim;
    lim.maxDepth = 3;
    Search s;
    const SearchResult r = s.search(bb, 0, lim);
    CHECK_FALSE(r.forcedWin);
    CHECK(r.bestMove == sq(7, 5));
    CHECK(r.score < MATE_THRESHOLD);  // not a Black mate

    // After blocking, White no longer has a winning move anywhere.
    bb.place(0, r.bestMove);
    bool whiteCanWin = false;
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb.isEmpty(s)) continue;
        bb.place(1, s);
        whiteCanWin = whiteCanWin || bb.winsAt(1, s);
        bb.remove(1, s);
    }
    CHECK_FALSE(whiteCanWin);
}

TEST_CASE("search: no forced win in a dead two-line position") {
    // Two one-end-blocked threes: each extends to a closed four, but after the
    // defender blocks both, Black has no threats left -> not a forced win.
    BitBoard bb;
    putBlack(bb, 6, 6);
    putBlack(bb, 6, 7);
    putBlack(bb, 6, 8);
    putWhite(bb, 6, 5);
    putBlack(bb, 5, 12);
    putBlack(bb, 6, 12);
    putBlack(bb, 7, 12);
    putWhite(bb, 8, 12);

    SearchLimits lim;
    lim.maxDepth = 4;
    Search s;
    const SearchResult r = s.search(bb, 0, lim);
    CHECK_FALSE(r.forcedWin);
    CHECK(r.score < MATE_THRESHOLD);
}

TEST_CASE("search: iterative deepening reaches requested depth") {
    // Stones scattered far apart so no forced win cuts the search short.
    BitBoard bb;
    putBlack(bb, 2, 2);
    putBlack(bb, 5, 8);
    putBlack(bb, 10, 12);
    putBlack(bb, 13, 3);
    putWhite(bb, 2, 12);
    putWhite(bb, 5, 3);
    putWhite(bb, 10, 4);
    putWhite(bb, 13, 12);

    SearchLimits lim;
    lim.maxDepth = 4;
    Search s;
    const SearchResult r = s.search(bb, 0, lim);
    CHECK(r.depth == 4);
    CHECK(r.bestMove >= 0);
}

TEST_CASE("search: empty board first move respects the outer-ring rule") {
    BitBoard bb;
    SearchLimits lim;
    lim.maxDepth = 2;
    Search s;
    const SearchResult r = s.search(bb, 0, lim);
    REQUIRE(r.bestMove >= 0);
    const int br = BitBoard::boardRow(r.bestMove);
    const int bc = BitBoard::boardCol(r.bestMove);
    CHECK(onRing(br, bc));
}

TEST_CASE("search: VCF agrees with deep search on random positions") {
    Search s;
    std::uint64_t seed = 0xABCD1234ULL;
    for (int trial = 0; trial < 3; ++trial) {
        BitBoard bb;
        int placed = 0;
        while (placed < 12) {
            seed = seed * 6364136223846793005ULL + 1;
            const int r = static_cast<int>((seed >> 33) % N);
            seed = seed * 6364136223846793005ULL + 1;
            const int c = static_cast<int>((seed >> 33) % N);
            if (bb.isEmpty(sqOf(r, c))) {
                bb.place(placed & 1, sqOf(r, c));
                ++placed;
            }
        }
        const int color = 0;
        const int vcf = s.vcfWin(bb, color, 4);
        SearchLimits lim;
        lim.maxDepth = 8;
        lim.maxNodes = 2'000'000;
        const SearchResult r = s.search(bb, color, lim);
        CAPTURE(trial);
        CAPTURE(vcf);
        CAPTURE(r.bestMove);
        CAPTURE(r.depth);
        if (r.aborted) continue;  // budget exhausted; nothing to compare
        if (vcf >= 0) {
            CHECK(r.score >= MATE_THRESHOLD);
        } else {
            CHECK(r.score < MATE_THRESHOLD);
        }
    }
}
