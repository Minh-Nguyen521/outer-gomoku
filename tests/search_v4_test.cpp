// Tests for SearchV4.
//
// The load-bearing claim of SearchV4 is that its bitboard-parallel
// five-point detection computes *exactly* the same set as the
// place/winsAt/remove loop it replaces. Everything else in the engine --
// the exact forced-reply restriction, the null-move gate, VCF, razoring,
// futility pruning -- is built on that primitive, so it is verified against
// the brute-force original over randomised positions rather than on a
// handful of hand-picked ones.
#include "doctest.h"

#include "bitboard.hpp"
#include "search.hpp"
#include "search_v3.hpp"
#include "search_v4.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace oog;

namespace {
void putBlack(BitBoard& bb, int r, int c) { bb.place(0, sqOf(r, c)); }
void putWhite(BitBoard& bb, int r, int c) { bb.place(1, sqOf(r, c)); }

int sq(int r, int c) { return sqOf(r, c); }

// The original implementation, kept here verbatim as the reference oracle.
std::vector<int> bruteForceFivePoints(BitBoard& bb, int color) {
    std::vector<int> out;
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb.isEmpty(s)) continue;
        bb.place(color, s);
        const bool w = bb.winsAt(color, s);
        bb.remove(color, s);
        if (w) out.push_back(s);
    }
    return out;
}
}  // namespace

TEST_CASE("SearchV4: five-point mask matches brute force on random positions") {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> coord(0, N - 1);

    int positionsWithAThreat = 0;

    for (int trial = 0; trial < 400; ++trial) {
        BitBoard bb;
        // Bias towards clustered stones so that fours actually occur:
        // scatter a few seeds, then grow around them.
        const int seeds = 2 + (trial % 3);
        std::vector<std::pair<int, int>> centres;
        for (int i = 0; i < seeds; ++i) centres.emplace_back(coord(rng), coord(rng));

        const int stones = 6 + (trial % 25);
        for (int i = 0; i < stones; ++i) {
            const auto& ctr = centres[static_cast<std::size_t>(i) % centres.size()];
            std::uniform_int_distribution<int> off(-3, 3);
            const int r = ctr.first + off(rng);
            const int c = ctr.second + off(rng);
            if (r < 0 || r >= N || c < 0 || c >= N) continue;
            const int s = sqOf(r, c);
            if (!bb.isEmpty(s)) continue;
            bb.place(i % 2, s);
        }

        for (int color = 0; color < 2; ++color) {
            std::vector<int> fast;
            fivePointSquares(bb, color, fast);
            std::vector<int> slow = bruteForceFivePoints(bb, color);
            std::sort(fast.begin(), fast.end());
            std::sort(slow.begin(), slow.end());
            REQUIRE(fast == slow);
            if (!slow.empty()) ++positionsWithAThreat;
        }
    }

    // Guard against a vacuous pass: if the random positions never contained
    // a four, the comparison above would be "empty == empty" 800 times.
    CHECK(positionsWithAThreat > 20);
}

TEST_CASE("SearchV4: five-point mask handles board edges and sentinels") {
    SUBCASE("four along the top edge") {
        BitBoard bb;
        for (int c = 0; c < 4; ++c) putBlack(bb, 0, c);
        std::vector<int> fast;
        fivePointSquares(bb, 0, fast);
        std::vector<int> slow = bruteForceFivePoints(bb, 0);
        std::sort(fast.begin(), fast.end());
        std::sort(slow.begin(), slow.end());
        CHECK(fast == slow);
        CHECK(std::find(fast.begin(), fast.end(), sq(0, 4)) != fast.end());
    }
    SUBCASE("four ending at the right edge must not wrap to the next row") {
        // Stones at (5,11)..(5,14): the only completion is (5,10). A naive
        // shift would let the window run past column 14 into row 6.
        BitBoard bb;
        for (int c = 11; c <= 14; ++c) putBlack(bb, 5, c);
        std::vector<int> fast;
        fivePointSquares(bb, 0, fast);
        CHECK(fast.size() == 1);
        CHECK(fast[0] == sq(5, 10));
    }
    SUBCASE("four down the left column, diagonal cannot wrap") {
        BitBoard bb;
        for (int r = 0; r < 4; ++r) putBlack(bb, r, 0);
        std::vector<int> fast;
        fivePointSquares(bb, 0, fast);
        std::vector<int> slow = bruteForceFivePoints(bb, 0);
        std::sort(fast.begin(), fast.end());
        std::sort(slow.begin(), slow.end());
        CHECK(fast == slow);
    }
    SUBCASE("blocked four yields no completion") {
        BitBoard bb;
        for (int c = 5; c <= 8; ++c) putBlack(bb, 7, c);
        putWhite(bb, 7, 4);
        putWhite(bb, 7, 9);
        std::vector<int> fast;
        fivePointSquares(bb, 0, fast);
        CHECK(fast.empty());
    }
}

TEST_CASE("SearchV4: VCT finds a three-three fork that VCF cannot") {
    BitBoard bb;
    putBlack(bb, 7, 5);
    putBlack(bb, 7, 6);
    putBlack(bb, 5, 7);
    putBlack(bb, 6, 7);
    putWhite(bb, 1, 1);
    putWhite(bb, 1, 2);
    putWhite(bb, 13, 13);
    putWhite(bb, 13, 12);

    Search base;
    CHECK(base.vcfWin(bb, 0) == -1);  // fours-only search is blind here

    SearchV4 e;
    CHECK(e.vcfWin(bb, 0) == -1);
    const int mv = e.vctWin(bb, 0, 4);
    REQUIRE(mv >= 0);
    CHECK(mv == sq(7, 7));
}

TEST_CASE("SearchV4: tactical floor (take five / block five)") {
    SearchLimits lim;
    lim.maxNodes = 20000;

    SUBCASE("completes five when it can") {
        BitBoard bb;
        for (int c = 4; c <= 7; ++c) putBlack(bb, 7, c);
        putWhite(bb, 9, 4);
        putWhite(bb, 9, 5);
        putWhite(bb, 9, 6);
        SearchV4 e;
        const SearchResult r = e.search(bb, 0, lim);
        CHECK((r.bestMove == sq(7, 8) || r.bestMove == sq(7, 3)));
    }

    SUBCASE("blocks the opponent's only five point") {
        BitBoard bb;
        for (int c = 3; c <= 6; ++c) putWhite(bb, 9, c);
        putBlack(bb, 9, 2);
        putBlack(bb, 5, 5);
        putBlack(bb, 5, 6);
        SearchV4 e;
        const SearchResult r = e.search(bb, 0, lim);
        CHECK(r.bestMove == sq(9, 7));
    }
}

TEST_CASE("SearchV4: forced-reply restriction never discards a winning move") {
    // Both sides have a five available and it is Black's move: Black must
    // take its own five rather than block. A restriction that kept only the
    // opponent's points would get this wrong.
    BitBoard bb;
    for (int c = 4; c <= 7; ++c) putBlack(bb, 3, c);   // Black four, row 3
    for (int c = 4; c <= 7; ++c) putWhite(bb, 11, c);  // White four, row 11

    SearchLimits lim;
    lim.maxNodes = 20000;
    SearchV4 e;
    const SearchResult r = e.search(bb, 0, lim);
    CHECK((r.bestMove == sq(3, 8) || r.bestMove == sq(3, 3)));
}

TEST_CASE("SearchV4: returns a legal empty square on a fresh position") {
    SearchLimits lim;
    lim.maxNodes = 5000;
    BitBoard bb;
    putBlack(bb, 0, 7);
    putWhite(bb, 7, 7);
    SearchV4 e;
    const SearchResult r = e.search(bb, 0, lim);
    REQUIRE(r.bestMove >= 0);
    CHECK(BitBoard::isPlayableSq(r.bestMove));
    CHECK(bb.isEmpty(r.bestMove));
}
