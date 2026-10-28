// Correctness checks for the three strategies added in the heuristic-strategy
// round: SearchVct, SearchNullKiller, SearchThreatCand.
//
// The point of these is not to measure strength (that is the round-robin in
// tools/compare_search.cpp) but to prove each new heuristic actually does the
// thing its header claims, and that none of them has broken the tactical
// floor every engine in this project is expected to clear.
#include "doctest.h"

#include "bitboard.hpp"
#include "search.hpp"
#include "search_nullkiller.hpp"
#include "search_threatcand.hpp"
#include "search_v3.hpp"
#include "search_vct.hpp"

#include <vector>

using namespace oog;

namespace {
void putBlack(BitBoard& bb, int r, int c) { bb.place(0, sqOf(r, c)); }
void putWhite(BitBoard& bb, int r, int c) { bb.place(1, sqOf(r, c)); }

int sq(int r, int c) { return sqOf(r, c); }

// A three-three fork: Black at (7,7) simultaneously completes an open three
// on row 7 (with (7,5),(7,6)) and an open three on column 7 (with
// (5,7),(6,7)).  Neither three is a four, so *no* sequence of fours starts
// here and VCF is blind to it -- but White can only block one of the two, and
// Black converts the other into an open four next move.  This is precisely
// the class of win threat-space search is for (Allis et al. 1993).
BitBoard threeThreeFork() {
    BitBoard bb;
    putBlack(bb, 7, 5);
    putBlack(bb, 7, 6);
    putBlack(bb, 5, 7);
    putBlack(bb, 6, 7);
    // Distant White stones so the position is not absurdly one-sided and the
    // near-move generator has something on both sides to work with.
    putWhite(bb, 1, 1);
    putWhite(bb, 1, 2);
    putWhite(bb, 13, 13);
    putWhite(bb, 13, 12);
    return bb;
}
}  // namespace

TEST_CASE("SearchVct: VCT finds a three-three fork that VCF cannot") {
    BitBoard bb = threeThreeFork();

    // Baseline check: the fours-only threat search really is blind here, so
    // the test is measuring VCT's addition and not something VCF already did.
    Search base;
    CHECK(base.vcfWin(bb, 0) == -1);

    SearchVct e;
    CHECK(e.vcfWin(bb, 0) == -1);

    const int mv = e.vctWin(bb, 0, 4);
    REQUIRE(mv >= 0);
    CHECK(mv == sq(7, 7));
}

TEST_CASE("SearchVct: no VCT win is claimed on an empty-ish quiet position") {
    // Two lone stones: nothing forcing exists, so a threat search that
    // reports a win here would be reporting a false positive.
    BitBoard bb;
    putBlack(bb, 7, 7);
    putWhite(bb, 8, 8);
    SearchVct e;
    CHECK(e.vctWin(bb, 0, 4) == -1);
}

// Every engine must still take a five that is on the board, and must still
// block the opponent's five.  These are the tactical floor: a new pruning or
// candidate-restriction heuristic that breaks either one is unusable no
// matter what its match score looks like.
namespace {
template <typename Engine>
void checksTacticalFloor() {
    SearchLimits lim;
    lim.maxNodes = 20000;

    {  // Black to move has four in a row: must complete five at (7,8).
        BitBoard bb;
        putBlack(bb, 7, 4);
        putBlack(bb, 7, 5);
        putBlack(bb, 7, 6);
        putBlack(bb, 7, 7);
        putWhite(bb, 9, 4);
        putWhite(bb, 9, 5);
        putWhite(bb, 9, 6);
        Engine e;
        const SearchResult r = e.search(bb, 0, lim);
        const bool winning = r.bestMove == sq(7, 8) || r.bestMove == sq(7, 3);
        CHECK(winning);
    }

    {  // White has four in a row with a single winning point; Black to move
       // must block it at (9,7) (Black has no five of its own available).
        BitBoard bb;
        putWhite(bb, 9, 3);
        putWhite(bb, 9, 4);
        putWhite(bb, 9, 5);
        putWhite(bb, 9, 6);
        putBlack(bb, 9, 2);  // closes the left end, so (9,7) is the only point
        putBlack(bb, 5, 5);
        putBlack(bb, 5, 6);
        Engine e;
        const SearchResult r = e.search(bb, 0, lim);
        CHECK(r.bestMove == sq(9, 7));
    }
}
}  // namespace

TEST_CASE("new strategies: tactical floor (take five / block five)") {
    SUBCASE("Search (baseline reference)") { checksTacticalFloor<Search>(); }
    SUBCASE("SearchV3 (reference)") { checksTacticalFloor<SearchV3>(); }
    SUBCASE("SearchVct") { checksTacticalFloor<SearchVct>(); }
    SUBCASE("SearchNullKiller") { checksTacticalFloor<SearchNullKiller>(); }
    SUBCASE("SearchThreatCand") { checksTacticalFloor<SearchThreatCand>(); }
}

TEST_CASE("new strategies: return a legal, empty square on a fresh position") {
    SearchLimits lim;
    lim.maxNodes = 5000;

    BitBoard bb;
    putBlack(bb, 0, 7);  // ring-legal opening
    putWhite(bb, 7, 7);

    auto checkMove = [&](int mv) {
        REQUIRE(mv >= 0);
        CHECK(BitBoard::isPlayableSq(mv));
        CHECK(bb.isEmpty(mv));
    };

    {
        SearchVct e;
        BitBoard b = bb;
        checkMove(e.search(b, 0, lim).bestMove);
    }
    {
        SearchNullKiller e;
        BitBoard b = bb;
        checkMove(e.search(b, 0, lim).bestMove);
    }
    {
        SearchThreatCand e;
        BitBoard b = bb;
        checkMove(e.search(b, 0, lim).bestMove);
    }
}
