#include "doctest.h"

#include "bitboard.hpp"
#include "eval.hpp"

#include <vector>

using namespace oog;

namespace {

void putBlack(BitBoard& bb, int r, int c) { bb.place(0, sqOf(r, c)); }
void putWhite(BitBoard& bb, int r, int c) { bb.place(1, sqOf(r, c)); }

}  // namespace

TEST_CASE("classifyLine: contiguous patterns") {
    CHECK(classifyLine(0x00, 0x00) == ThreatLevel::None);
    // Own stones at +1..+4 -> placing center completes five.
    CHECK(classifyLine(0xF0, 0x00) == ThreatLevel::Five);
    CHECK(classifyLine(0x0F, 0x00) == ThreatLevel::Five);  // -4..-1
    // Own at +1..+3, both ends open -> open four.
    CHECK(classifyLine(0x70, 0x00) == ThreatLevel::OpenFour);
    // Same but blocked at +4 -> closed four.
    CHECK(classifyLine(0x70, 0x80) == ThreatLevel::ClosedFour);
    // Own at +1,+2 both ends open -> open three.
    CHECK(classifyLine(0x30, 0x00) == ThreatLevel::OpenThree);
    // Own at +1,+2, blocked at -1 and +3 -> closed three.
    CHECK(classifyLine(0x30, 0x48) == ThreatLevel::ClosedThree);
    // Own at +1 only -> no threat.
    CHECK(classifyLine(0x40, 0x00) == ThreatLevel::None);
}

TEST_CASE("classifyLine: jump four (gap) is a closed four") {
    // Own at +1,+2,+4, empty at +3: filling center still leaves a four-with-gap.
    CHECK(classifyLine(0xB0, 0x00) == ThreatLevel::ClosedFour);
    // Opponent stones unrelated elsewhere do not change it.
    CHECK(classifyLine(0xB0, 0x01) == ThreatLevel::ClosedFour);
}

TEST_CASE("classifyFeature: table agrees with classifyLine") {
    CHECK(classifyFeature(0xF000) == ThreatLevel::Five);
    CHECK(classifyFeature(0x7000) == ThreatLevel::OpenFour);
    CHECK(classifyFeature(0x0000) == ThreatLevel::None);
    // Exhaustive agreement for all 65536 features.
    bool mismatch = false;
    for (std::uint32_t f = 0; f < 65536; ++f) {
        const auto own8 = static_cast<std::uint8_t>(f >> 8);
        const auto opp8 = static_cast<std::uint8_t>(f & 0xFF);
        if (classifyFeature(static_cast<std::uint16_t>(f)) !=
            classifyLine(own8, opp8)) {
            mismatch = true;
            break;
        }
    }
    CHECK_FALSE(mismatch);
}

TEST_CASE("evaluateMove: winning move scores highest") {
    BitBoard bb;
    putBlack(bb, 7, 3);
    putBlack(bb, 7, 4);
    putBlack(bb, 7, 5);
    putBlack(bb, 7, 6);
    const int win = evaluateMove(bb, 0, sqOf(7, 7));   // completes five
    const int weak = evaluateMove(bb, 0, sqOf(0, 0));  // no threat
    CHECK(win >= levelScore(ThreatLevel::Five));
    CHECK(win > weak);
}

TEST_CASE("evaluateMove: open four > open three > open two > none") {
    const int four = evaluateMove(([] {
                                  BitBoard b;
                                  for (int c = 3; c <= 5; ++c) putBlack(b, 7, c);
                                  return b;
                              }()),
                                  0, sqOf(7, 6));
    const int three = evaluateMove(([] {
                                    BitBoard b;
                                    putBlack(b, 7, 3);
                                    putBlack(b, 7, 4);
                                    return b;
                                }()),
                                   0, sqOf(7, 5));
    const int two = evaluateMove(([] {
                                  BitBoard b;
                                  putBlack(b, 7, 3);
                                  return b;
                              }()),
                                 0, sqOf(7, 4));
    const int none = evaluateMove(BitBoard{}, 0, sqOf(0, 0));
    CHECK(four > three);
    CHECK(three > two);
    CHECK(two > none);
}

TEST_CASE("evaluateMove: blocking an opponent five is a top move") {
    BitBoard bb;
    putWhite(bb, 7, 3);
    putWhite(bb, 7, 4);
    putWhite(bb, 7, 5);
    putWhite(bb, 7, 6);
    putBlack(bb, 7, 2);  // white's run is blocked on the left only
    // Black to move: the block at (7,7) must outrank a quiet move.
    const int block = evaluateMove(bb, 0, sqOf(7, 7));
    const int quiet = evaluateMove(bb, 0, sqOf(0, 0));
    CHECK(block > quiet);
    CHECK(block >= levelScore(ThreatLevel::Five));
}

TEST_CASE("evaluate: empty board is neutral") {
    CHECK(evaluate(BitBoard{}, 0) == 0);
    CHECK(evaluate(BitBoard{}, 1) == 0);
}

TEST_CASE("evaluate: four favors its owner") {
    BitBoard bb;
    putBlack(bb, 7, 3);
    putBlack(bb, 7, 4);
    putBlack(bb, 7, 5);
    putBlack(bb, 7, 6);
    CHECK(evaluate(bb, 0) > 50'000);
    CHECK(evaluate(bb, 1) < -50'000);
}

TEST_CASE("evaluate: white four favors white") {
    BitBoard bb;
    putWhite(bb, 7, 3);
    putWhite(bb, 7, 4);
    putWhite(bb, 7, 5);
    putWhite(bb, 7, 6);
    CHECK(evaluate(bb, 1) > 50'000);
    CHECK(evaluate(bb, 0) < -50'000);
}

TEST_CASE("evaluate: open three outweighs a scattered opponent") {
    BitBoard bb;
    putBlack(bb, 7, 3);
    putBlack(bb, 7, 4);
    putBlack(bb, 7, 5);  // black open four threat (winning potential)
    putWhite(bb, 3, 3);
    putWhite(bb, 3, 4);  // white weak two
    CHECK(evaluate(bb, 0) > evaluate(bb, 1));
}

TEST_CASE("EvalState matches evaluate after random moves and undos") {
    std::uint64_t seed = 0xBEEFCAFEULL;
    for (int trial = 0; trial < 15; ++trial) {
        BitBoard bb;
        int placed = 0;
        while (placed < 60) {
            seed = seed * 6364136223846793005ULL + 1;
            const int r = static_cast<int>((seed >> 33) % N);
            seed = seed * 6364136223846793005ULL + 1;
            const int c = static_cast<int>((seed >> 33) % N);
            if (bb.isEmpty(sqOf(r, c))) {
                bb.place(placed & 1, sqOf(r, c));
                ++placed;
            }
        }
        EvalState es;
        es.reset(bb);
        CHECK(es.value(0) == evaluate(bb, 0));
        CHECK(es.value(1) == evaluate(bb, 1));
        std::vector<int> history;
        for (int m = 0; m < 6; ++m) {
            int s = -1;
            for (int a = 0; a < 200; ++a) {
                seed = seed * 6364136223846793005ULL + 1;
                const int cand = static_cast<int>((seed >> 33) % BitBoard::SIZE);
                if (BitBoard::isPlayableSq(cand) && bb.isEmpty(cand)) {
                    s = cand;
                    break;
                }
            }
            REQUIRE(s >= 0);
            bb.place(m & 1, s);
            es.update(bb, s);
            history.push_back(s);
            CHECK(es.value(0) == evaluate(bb, 0));
            CHECK(es.value(1) == evaluate(bb, 1));
        }
        for (int i = static_cast<int>(history.size()) - 1; i >= 0; --i) {
            bb.remove(i & 1, history[static_cast<std::size_t>(i)]);
            es.update(bb, history[static_cast<std::size_t>(i)]);
            CHECK(es.value(0) == evaluate(bb, 0));
            CHECK(es.value(1) == evaluate(bb, 1));
        }
    }
}