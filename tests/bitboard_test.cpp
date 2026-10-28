#include "doctest.h"

#include "bitboard.hpp"
#include "board.hpp"

#include <random>
#include <vector>

using oog::BitBoard;
using oog::Board;
using oog::Cell;
using oog::Coord;

namespace {

std::vector<int> playToBoard(Board& b, const std::vector<Coord>& moves) {
    std::vector<int> sqs;
    for (const auto& m : moves) {
        if (b.makeMove(m)) sqs.push_back(oog::sqOf(m.r, m.c));
    }
    return sqs;
}

}  // namespace

TEST_CASE("sqOf / boardRow / boardCol round-trip") {
    for (int r = 0; r < oog::N; ++r) {
        for (int c = 0; c < oog::N; ++c) {
            const int sq = oog::sqOf(r, c);
            CHECK(oog::BitBoard::isPlayableSq(sq));
            CHECK(oog::BitBoard::boardRow(sq) == r);
            CHECK(oog::BitBoard::boardCol(sq) == c);
        }
    }
    // Sentinel squares are not playable.
    CHECK_FALSE(oog::BitBoard::isPlayableSq(0));
    CHECK_FALSE(oog::BitBoard::isPlayableSq(15));        // col 15
    CHECK_FALSE(oog::BitBoard::isPlayableSq(16 * 16));   // bottom pad row
}

TEST_CASE("place / remove / at / count") {
    BitBoard bb;
    CHECK(bb.count() == 0);
    const int sq = oog::sqOf(0, 0);
    bb.place(0, sq);
    CHECK(bb.isBlack(sq));
    CHECK_FALSE(bb.isWhite(sq));
    CHECK_FALSE(bb.isEmpty(sq));
    CHECK(bb.at(sq) == Cell::Black);
    CHECK(bb.count() == 1);
    bb.place(1, oog::sqOf(14, 0));
    CHECK(bb.count() == 2);
    bb.remove(0, sq);
    CHECK(bb.isEmpty(sq));
    CHECK(bb.count() == 1);
}

TEST_CASE("hasFive detects horizontal / vertical / diagonal wins") {
    SUBCASE("horizontal") {
        BitBoard bb;
        for (int c = 0; c < 5; ++c) bb.place(0, oog::sqOf(0, c));
        CHECK(bb.hasFive(0));
        CHECK_FALSE(bb.hasFive(1));
    }
    SUBCASE("vertical") {
        BitBoard bb;
        for (int r = 0; r < 5; ++r) bb.place(1, oog::sqOf(r, 7));
        CHECK(bb.hasFive(1));
        CHECK_FALSE(bb.hasFive(0));
    }
    SUBCASE("main diagonal") {
        BitBoard bb;
        for (int i = 0; i < 5; ++i) bb.place(0, oog::sqOf(i, i));
        CHECK(bb.hasFive(0));
    }
    SUBCASE("anti diagonal") {
        BitBoard bb;
        for (int i = 0; i < 5; ++i) bb.place(0, oog::sqOf(i, 14 - i));
        CHECK(bb.hasFive(0));
    }
    SUBCASE("four in a row is not five") {
        BitBoard bb;
        for (int c = 0; c < 4; ++c) bb.place(0, oog::sqOf(0, c));
        CHECK_FALSE(bb.hasFive(0));
    }
}

TEST_CASE("hasFive does not wrap at board edges") {
    // Stones at the end of row 0 must not connect to row 1 via wrap.
    BitBoard bb;
    for (int c = 11; c <= 14; ++c) bb.place(0, oog::sqOf(0, c));
    CHECK_FALSE(bb.hasFive(0));
    // Bottom row vertical wrap must not connect to the sentinel row.
    BitBoard bb2;
    for (int r = 12; r <= 15; ++r) bb2.place(0, oog::sqOf(r, 0));
    CHECK_FALSE(bb2.hasFive(0));
}

TEST_CASE("winsAt matches Board::isWinAt across random games") {
    std::mt19937 rng(12345);
    for (int game = 0; game < 50; ++game) {
        Board b;
        BitBoard bb;
        int color = 0;
        while (!b.isGameOver()) {
            // Find a random legal move.
            bool placed = false;
            for (int attempt = 0; attempt < 200 && !placed; ++attempt) {
                const int r = static_cast<int>(rng() % oog::N);
                const int c = static_cast<int>(rng() % oog::N);
                const Coord m{r, c};
                if (b.isLegal(m)) {
                    const int sq = oog::sqOf(r, c);
                    REQUIRE(b.makeMove(m));
                    bb.place(color, sq);
                    const bool bw = b.isWinAt(oog::indexOf(m));
                    CHECK(bb.winsAt(color, sq) == bw);
                    CHECK(bb.hasFive(color) == bw);
                    CHECK(bb.at(sq) == (color == 0 ? Cell::Black : Cell::White));
                    color ^= 1;
                    placed = true;
                }
            }
            if (!placed) {
                // No random move found; force the board state and stop.
                break;
            }
        }
        // Final winner must agree.
        if (b.winner() == Cell::Black) {
            CHECK(bb.hasFive(0));
            CHECK_FALSE(bb.hasFive(1));
        } else if (b.winner() == Cell::White) {
            CHECK(bb.hasFive(1));
            CHECK_FALSE(bb.hasFive(0));
        } else {
            CHECK_FALSE(bb.hasFive(0));
            CHECK_FALSE(bb.hasFive(1));
        }
    }
}

TEST_CASE("genMovesNear: legal, empty, unique, non-empty") {
    Board b;
    playToBoard(b, {{0, 0}, {14, 7}, {0, 1}, {14, 8}, {0, 2}});
    BitBoard bb;
    bb.fillBoard(b);

    std::vector<int> moves;
    bb.genMovesNear(moves);
    CHECK_FALSE(moves.empty());
    for (const int m : moves) {
        CHECK(oog::BitBoard::isPlayableSq(m));
        CHECK(bb.isEmpty(m));
    }
    // No duplicates.
    std::vector<bool> seen(oog::BitBoard::SIZE, false);
    for (const int m : moves) {
        CHECK_FALSE(seen[static_cast<std::size_t>(m)]);
        seen[static_cast<std::size_t>(m)] = true;
    }
}

TEST_CASE("genMovesNear on empty board returns only the outer ring") {
    BitBoard bb;
    std::vector<int> moves;
    bb.genMovesNear(moves);
    REQUIRE_FALSE(moves.empty());
    for (const int m : moves) {
        const int r = oog::BitBoard::boardRow(m);
        const int c = oog::BitBoard::boardCol(m);
        const bool inRing = r < 2 || r >= oog::N - 2 || c < 2 || c >= oog::N - 2;
        CHECK(inRing);
    }
    const auto expected = static_cast<std::size_t>(oog::N * oog::N - 11 * 11);
    CHECK(moves.size() == expected);  // 225 - 121 = 104
}

TEST_CASE("hash is deterministic and place/remove restores it") {
    BitBoard a, b;
    const int sq0 = oog::sqOf(0, 0);
    const int sq1 = oog::sqOf(5, 5);
    a.place(0, sq0);
    b.place(0, sq0);
    CHECK(a.hash() == b.hash());
    a.place(1, sq1);
    CHECK(a.hash() != b.hash());
    a.remove(1, sq1);
    CHECK(a.hash() == b.hash());
    // Same position, different insertion order.
    BitBoard x, y;
    x.place(0, sq0);
    x.place(1, sq1);
    y.place(1, sq1);
    y.place(0, sq0);
    CHECK(x.hash() == y.hash());
}

TEST_CASE("fillBoard mirrors Board cell-by-cell") {
    Board b;
    playToBoard(b, {{0, 0}, {14, 1}, {0, 2}, {14, 3}, {7, 7}});
    BitBoard bb;
    bb.fillBoard(b);
    CHECK(bb.count() == 5);
    for (int r = 0; r < oog::N; ++r) {
        for (int c = 0; c < oog::N; ++c) {
            CHECK(bb.at(oog::sqOf(r, c)) == b.cell({r, c}));
        }
    }
}