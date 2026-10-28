// Tests for EvalStateV2 -- the fork (double threat) term.
//
// Two things must hold: the term has to actually fire on real forks (and not
// on non-forks), and the incremental path has to agree with a full recompute,
// since the search calls update() thousands of times per node without ever
// resetting.
#include "doctest.h"

#include "bitboard.hpp"
#include "eval.hpp"
#include "eval_v2.hpp"

#include <random>
#include <vector>

using namespace oog;

namespace {
void putBlack(BitBoard& bb, int r, int c) { bb.place(0, sqOf(r, c)); }
void putWhite(BitBoard& bb, int r, int c) { bb.place(1, sqOf(r, c)); }

EvalStateV2 stateOf(const BitBoard& bb) {
    EvalStateV2 st;
    st.reset(bb);
    return st;
}
}  // namespace

TEST_CASE("EvalStateV2: counts four threats, deduped by span") {
    SUBCASE("contiguous four, one gain square") {
        BitBoard bb;
        for (int c = 4; c <= 7; ++c) putBlack(bb, 7, c);
        putWhite(bb, 7, 3);  // block the left end
        auto st = stateOf(bb);
        CHECK(st.fourThreats(0) == 1);  // only (7,8)
    }
    SUBCASE("an open four is ONE threat, not two") {
        // Regression. `_XXXX_` has two completion squares but is a single
        // threat, already scored S_OPEN_FOUR by the linear term. Counting it
        // as two made every open four look like a double four and stacked a
        // fork bonus on top -- measured 14 points worse than baseline against
        // rapfi before this was caught.
        BitBoard bb;
        for (int c = 4; c <= 7; ++c) putBlack(bb, 7, c);
        auto st = stateOf(bb);
        CHECK(st.fourThreats(0) == 1);
        CHECK(st.forkBonus(0) == 0);  // must NOT be treated as a fork
    }
    SUBCASE("two fours on different lines ARE a fork") {
        BitBoard bb;
        for (int c = 4; c <= 7; ++c) putBlack(bb, 3, c);
        putWhite(bb, 3, 3);   // one completion at (3,8)
        for (int c = 4; c <= 7; ++c) putBlack(bb, 11, c);
        putWhite(bb, 11, 3);  // one completion at (11,8)
        auto st = stateOf(bb);
        CHECK(st.fourThreats(0) == 2);
        CHECK(st.forkBonus(0) > 0);
    }
    SUBCASE("gapped four XX_XX is found") {
        BitBoard bb;
        putBlack(bb, 7, 4);
        putBlack(bb, 7, 5);
        putBlack(bb, 7, 7);
        putBlack(bb, 7, 8);
        auto st = stateOf(bb);
        CHECK(st.fourThreats(0) >= 1);  // (7,6) completes five
    }
    SUBCASE("three is not a four") {
        BitBoard bb;
        for (int c = 5; c <= 7; ++c) putBlack(bb, 7, c);
        auto st = stateOf(bb);
        CHECK(st.fourThreats(0) == 0);
    }
}

TEST_CASE("EvalStateV2: counts open threes") {
    SUBCASE("a single open three") {
        BitBoard bb;
        for (int c = 5; c <= 7; ++c) putBlack(bb, 7, c);
        auto st = stateOf(bb);
        CHECK(st.openThrees(0) == 1);
    }
    SUBCASE("a blocked three is not an open three") {
        BitBoard bb;
        for (int c = 5; c <= 7; ++c) putBlack(bb, 7, c);
        putWhite(bb, 7, 4);
        putWhite(bb, 7, 8);
        auto st = stateOf(bb);
        CHECK(st.openThrees(0) == 0);
    }
}

TEST_CASE("EvalStateV2: the fork term fires on real forks") {
    SUBCASE("double open three scores far above a lone open three") {
        BitBoard single;
        for (int c = 5; c <= 7; ++c) putBlack(single, 7, c);

        BitBoard fork;  // open three on row 7 AND on column 7, crossing
        for (int c = 5; c <= 7; ++c) putBlack(fork, 7, c);
        for (int r = 5; r <= 6; ++r) putBlack(fork, r, 7);

        auto a = stateOf(single);
        auto b = stateOf(fork);
        CHECK(a.openThrees(0) == 1);
        CHECK(b.openThrees(0) >= 2);
        CHECK(a.forkBonus(0) == 0);
        CHECK(b.forkBonus(0) > 0);
        // And the blended value must reflect it.
        CHECK(b.value(0) > a.value(0));
    }

    SUBCASE("double four outranks a single closed four") {
        BitBoard one;
        for (int c = 4; c <= 7; ++c) putBlack(one, 7, c);
        putWhite(one, 7, 3);
        putWhite(one, 7, 8);  // dead four: no gain square

        BitBoard two;  // two separate fours, each with one gain square
        for (int c = 4; c <= 7; ++c) putBlack(two, 3, c);
        putWhite(two, 3, 3);
        for (int c = 4; c <= 7; ++c) putBlack(two, 11, c);
        putWhite(two, 11, 3);

        auto a = stateOf(one);
        auto b = stateOf(two);
        CHECK(a.fourThreats(0) == 0);
        CHECK(b.fourThreats(0) == 2);
        CHECK(b.forkBonus(0) > a.forkBonus(0));
    }

    SUBCASE("no fork bonus on a quiet position") {
        BitBoard bb;
        putBlack(bb, 7, 7);
        putWhite(bb, 8, 8);
        auto st = stateOf(bb);
        CHECK(st.forkBonus(0) == 0);
        CHECK(st.forkBonus(1) == 0);
    }
}

TEST_CASE("EvalStateV2: this is a real gap in the old eval") {
    // The point of the whole exercise: the existing evaluate() rates a double
    // four *below* a single open four, though both are winning.
    BitBoard openFour;
    for (int c = 4; c <= 7; ++c) putBlack(openFour, 7, c);

    BitBoard doubleFour;
    for (int c = 4; c <= 7; ++c) putBlack(doubleFour, 3, c);
    putWhite(doubleFour, 3, 3);
    for (int c = 4; c <= 7; ++c) putBlack(doubleFour, 11, c);
    putWhite(doubleFour, 11, 3);

    const int oldOpen = evaluate(openFour, 0);
    const int oldDouble = evaluate(doubleFour, 0);
    CHECK(oldDouble < oldOpen);  // the bug, demonstrated

    // With the fork term the double four is no longer dismissed.
    const int newDouble = evaluateV2(doubleFour, 0);
    CHECK(newDouble > oldDouble);
}

TEST_CASE("EvalStateV2: incremental update matches a full reset") {
    // The search calls update() constantly and reset() almost never, so the
    // two paths must not drift.
    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> coord(0, N - 1);

    for (int trial = 0; trial < 60; ++trial) {
        BitBoard bb;
        EvalStateV2 inc;
        inc.reset(bb);

        std::vector<int> placed;
        const int cr = coord(rng);
        const int cc = coord(rng);
        for (int i = 0; i < 18; ++i) {
            std::uniform_int_distribution<int> off(-3, 3);
            const int r = cr + off(rng);
            const int c = cc + off(rng);
            if (r < 0 || r >= N || c < 0 || c >= N) continue;
            const int s = sqOf(r, c);
            if (!bb.isEmpty(s)) continue;
            bb.place(i % 2, s);
            inc.update(bb, s);
            placed.push_back(s);

            EvalStateV2 full;
            full.reset(bb);
            REQUIRE(inc.fourThreats(0) == full.fourThreats(0));
            REQUIRE(inc.fourThreats(1) == full.fourThreats(1));
            REQUIRE(inc.openThrees(0) == full.openThrees(0));
            REQUIRE(inc.openThrees(1) == full.openThrees(1));
            REQUIRE(inc.value(0) == full.value(0));
        }

        // Undo everything; counts must return to zero.
        for (auto it = placed.rbegin(); it != placed.rend(); ++it) {
            const int s = *it;
            const int color = bb.isBlack(s) ? 0 : 1;
            bb.remove(color, s);
            inc.update(bb, s);
        }
        REQUIRE(inc.fourThreats(0) == 0);
        REQUIRE(inc.fourThreats(1) == 0);
        REQUIRE(inc.openThrees(0) == 0);
        REQUIRE(inc.openThrees(1) == 0);
    }
}
