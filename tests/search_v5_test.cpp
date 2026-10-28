// Tests for SearchV5.
//
// The new capability is root safety screening: refusing moves that hand the
// opponent a forced VCF win. These check that it (a) actually detects one,
// (b) does not cry wolf on a safe move, and (c) stands aside rather than
// leaving the engine mute when every candidate loses.
#include "doctest.h"

#include "bitboard.hpp"
#include "search.hpp"
#include "search_v4.hpp"
#include "search_v5.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace oog;

namespace {
void putBlack(BitBoard& bb, int r, int c) { bb.place(0, sqOf(r, c)); }
void putWhite(BitBoard& bb, int r, int c) { bb.place(1, sqOf(r, c)); }

int sq(int r, int c) { return sqOf(r, c); }

// White has a four on row 9 with its left end blocked by Black, so (9,7) is
// White's only completion. Black to move: blocking is safe, anything else
// hands White an immediate forced win.
BitBoard whiteFourOneGap() {
    BitBoard bb;
    for (int c = 3; c <= 6; ++c) putWhite(bb, 9, c);
    putBlack(bb, 9, 2);
    putBlack(bb, 5, 5);
    putBlack(bb, 5, 6);
    return bb;
}
}  // namespace

TEST_CASE("SearchV5: screen flags a move that allows an opponent VCF") {
    BitBoard bb = whiteFourOneGap();
    SearchV5 e;
    // A quiet move far from the threat loses on the spot.
    CHECK(e.moveAllowsOpponentVcf(bb, 0, sq(1, 12)));
    // Blocking White's only completion does not.
    CHECK_FALSE(e.moveAllowsOpponentVcf(bb, 0, sq(9, 7)));
}

TEST_CASE("SearchV5: screen does not flag safe moves on a quiet board") {
    BitBoard bb;
    putBlack(bb, 7, 7);
    putWhite(bb, 8, 8);
    SearchV5 e;
    CHECK_FALSE(e.moveAllowsOpponentVcf(bb, 0, sq(7, 8)));
    CHECK_FALSE(e.moveAllowsOpponentVcf(bb, 0, sq(6, 6)));
}

TEST_CASE("SearchV5: still returns a legal move when every candidate loses") {
    // White has an open four: both (7,3) and (7,8) win for White, so no
    // Black move survives. The screen must stand aside, not return -1.
    BitBoard bb;
    for (int c = 4; c <= 7; ++c) putWhite(bb, 7, c);
    putBlack(bb, 2, 2);
    putBlack(bb, 2, 3);

    SearchLimits lim;
    lim.maxNodes = 20000;
    SearchV5 e;
    const SearchResult r = e.search(bb, 0, lim);
    REQUIRE(r.bestMove >= 0);
    CHECK(BitBoard::isPlayableSq(r.bestMove));
    CHECK(bb.isEmpty(r.bestMove));
}

TEST_CASE("SearchV5: tactical floor (take five / block five)") {
    SearchLimits lim;
    lim.maxNodes = 20000;

    SUBCASE("completes five when it can") {
        BitBoard bb;
        for (int c = 4; c <= 7; ++c) putBlack(bb, 7, c);
        putWhite(bb, 9, 4);
        putWhite(bb, 9, 5);
        putWhite(bb, 9, 6);
        SearchV5 e;
        const SearchResult r = e.search(bb, 0, lim);
        CHECK((r.bestMove == sq(7, 8) || r.bestMove == sq(7, 3)));
    }

    SUBCASE("blocks the opponent's only five point") {
        BitBoard bb = whiteFourOneGap();
        SearchV5 e;
        const SearchResult r = e.search(bb, 0, lim);
        CHECK(r.bestMove == sq(9, 7));
    }
}

TEST_CASE("SearchV5: VCT still finds the three-three fork") {
    BitBoard bb;
    putBlack(bb, 7, 5);
    putBlack(bb, 7, 6);
    putBlack(bb, 5, 7);
    putBlack(bb, 6, 7);
    putWhite(bb, 1, 1);
    putWhite(bb, 1, 2);
    putWhite(bb, 13, 13);
    putWhite(bb, 13, 12);

    SearchV5 e;
    CHECK(e.vcfWin(bb, 0) == -1);
    const int mv = e.vctWin(bb, 0, 4);
    REQUIRE(mv >= 0);
    CHECK(mv == sq(7, 7));
}

TEST_CASE("SearchV5: five-point primitive still matches SearchV4's") {
    // V5 inherits the primitive; this guards against the copy drifting.
    std::mt19937 rng(999);
    std::uniform_int_distribution<int> coord(0, N - 1);
    for (int trial = 0; trial < 150; ++trial) {
        BitBoard bb;
        const int cr = coord(rng);
        const int cc = coord(rng);
        for (int i = 0; i < 20; ++i) {
            std::uniform_int_distribution<int> off(-3, 3);
            const int r = cr + off(rng);
            const int c = cc + off(rng);
            if (r < 0 || r >= N || c < 0 || c >= N) continue;
            const int s = sqOf(r, c);
            if (bb.isEmpty(s)) bb.place(i % 2, s);
        }
        for (int color = 0; color < 2; ++color) {
            std::vector<int> a;
            std::vector<int> b;
            fivePointSquares(bb, color, a);    // V4
            fivePointSquaresV5(bb, color, b);  // V5
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            REQUIRE(a == b);
        }
    }
}

TEST_CASE("SearchV5: extensions terminate and respect the node budget") {
    // Forcing-move extensions must not run away: a tight node budget has to
    // still return promptly with a legal move.
    BitBoard bb;
    for (int c = 4; c <= 6; ++c) putBlack(bb, 7, c);
    for (int c = 4; c <= 6; ++c) putWhite(bb, 9, c);
    putBlack(bb, 6, 3);
    putWhite(bb, 10, 3);

    SearchLimits lim;
    lim.maxNodes = 8000;
    SearchV5 e;
    const SearchResult r = e.search(bb, 0, lim);
    REQUIRE(r.bestMove >= 0);
    CHECK(bb.isEmpty(r.bestMove));
    CHECK(e.nodes() < 200000);  // budget honoured within abort granularity
}
