#include "doctest.h"

#include "board.hpp"

#include <vector>

using oog::Board;
using oog::Cell;
using oog::Coord;

namespace {

// Play a list of (r,c) moves; stops if one is rejected.
void play(Board& b, const std::vector<Coord>& moves) {
    for (const auto& m : moves) b.makeMove(m);
}

}  // namespace

TEST_CASE("reset: board empty, black to move, not game over") {
    Board b;
    REQUIRE(b.ply() == 0);
    REQUIRE(b.isBlackToMove());
    REQUIRE_FALSE(b.isGameOver());
    REQUIRE_FALSE(b.isDraw());
    CHECK(b.winner() == Cell::Empty);
    CHECK(b.lastMoveIndex() == -1);
    for (int i = 0; i < oog::N; ++i)
        for (int j = 0; j < oog::N; ++j) CHECK(b.cell({i, j}) == Cell::Empty);
}

TEST_CASE("reset: border ring is Border in padding") {
    Board b;
    CHECK(b.at(0) == Cell::Border);                       // top-left padding
    CHECK(b.at(oog::indexOf({0, 0})) == Cell::Empty);      // first playable
    CHECK(b.at(oog::indexOf({oog::N - 1, oog::N - 1})) == Cell::Empty);
}

TEST_CASE("first move must be on the outer ring") {
    Board b;
    // Center (7,7) is illegal for Black's first move.
    CHECK_FALSE(b.isLegal({7, 7}));
    CHECK_FALSE(b.isLegal({3, 3}));
    // Outer ring is legal.
    CHECK(b.isLegal({0, 0}));
    CHECK(b.isLegal({0, 7}));
    CHECK(b.isLegal({14, 0}));
    CHECK(b.isLegal({1, 1}));
    CHECK(b.isLegal({13, 13}));

    // After the first move the restriction disappears.
    REQUIRE(b.makeMove({0, 0}));
    CHECK(b.isLegal({7, 7}));
}

TEST_CASE("cannot play on occupied cell") {
    Board b;
    play(b, {{0, 0}, {14, 0}, {0, 1}, {14, 1}, {0, 2}, {14, 2}, {0, 3},
             {14, 3}});
    CHECK_FALSE(b.isLegal({0, 3}));  // occupied
    CHECK(b.isLegal({3, 3}));        // empty, not game over
}

TEST_CASE("horizontal five wins") {
    Board b;
    play(b, {{0, 0}, {14, 7}, {0, 1}, {14, 8}, {0, 2}, {14, 9}, {0, 3},
             {14, 10}});
    REQUIRE_FALSE(b.isGameOver());
    REQUIRE(b.makeMove({0, 4}));  // Black completes a row of 5.
    CHECK(b.winner() == Cell::Black);
    CHECK(b.isGameOver());
    CHECK(b.isWinAt(oog::indexOf({0, 4})));
    CHECK(b.isWin(Cell::Black));
}

TEST_CASE("vertical five wins") {
    Board b;
    // Black builds column 5 (never completes), White builds column 0.
    play(b, {{0, 5}, {0, 0}, {1, 5}, {1, 0}, {2, 5}, {2, 0}, {3, 5},
             {3, 0}, {0, 2}});
    REQUIRE(b.winner() == Cell::Empty);
    REQUIRE(b.makeMove({4, 0}));  // White: column 0 -> 5 in a row.
    CHECK(b.winner() == Cell::White);
}

TEST_CASE("diagonal five wins") {
    Board b;
    play(b, {{0, 0}, {14, 0}, {1, 1}, {14, 1}, {2, 2}, {14, 2}, {3, 3},
             {14, 3}});
    REQUIRE(b.winner() == Cell::Empty);
    REQUIRE(b.makeMove({4, 4}));  // Black: main diagonal 5 in a row.
    CHECK(b.winner() == Cell::Black);
}

// NOTE: overlines (6+ in a row) count as a win in freestyle Outer-Open
// Gomoku.  They are unreachable through normal play because the game ends as
// soon as 5 in a row appears; the ">= 5" comparison in isWinAt covers them.

TEST_CASE("non-consecutive stones are not a win") {
    Board b;
    play(b, {{0, 0}, {14, 0}, {0, 2}, {14, 1}, {0, 4}, {14, 2}});
    CHECK(b.winner() == Cell::Empty);
    CHECK_FALSE(b.isWin(Cell::Black));
}

TEST_CASE("undo restores board and zobrist") {
    Board b;
    play(b, {{0, 0}, {14, 1}, {0, 2}});
    const auto key = b.zobrist();
    const auto n = b.ply();
    b.undoMove();
    CHECK(b.ply() == n - 1);
    CHECK(b.cell({0, 2}) == Cell::Empty);
    b.undoMove();
    b.undoMove();
    b.undoMove();
    CHECK(b.ply() == 0);
    CHECK(b.isBlackToMove());
    // Redo the same sequence; zobrist should match exactly.
    play(b, {{0, 0}, {14, 1}, {0, 2}});
    CHECK(b.zobrist() == key);
}

TEST_CASE("serialization round-trips") {
    Board b;
    play(b, {{0, 0}, {14, 7}, {0, 1}, {14, 8}, {0, 2}});
    const auto s = b.serialize();
    const auto turn = b.turn();
    Board c;
    REQUIRE(c.load(s));
    CHECK(c.turn() == turn);
    CHECK(c.serialize() == s);
    CHECK(c.zobrist() == b.zobrist());
    CHECK(c.toString() == b.toString());
}

TEST_CASE("load rejects mismatched side to move") {
    // 2 moves -> Black to move, but the string claims White to move.
    Board c;
    CHECK_FALSE(c.load("W 0,0 14,1"));
    CHECK(c.load("B 0,0 14,1"));
}

TEST_CASE("load rejects illegal opening") {
    Board c;
    CHECK_FALSE(c.load("B 7,7"));  // center first move is illegal
}

TEST_CASE("full board: win takes precedence over draw") {
    Board b;
    // Row-major fill produces a checkerboard (r+c even = Black).  The
    // anti-diagonal r+c == 4 is fully black, so Black wins when its last cell
    // (4,0) is placed (move 61).  This also verifies makeMove rejects moves
    // once the game is over.
    bool allPlaced = true;
    for (int r = 0; r < oog::N && allPlaced; ++r) {
        for (int c = 0; c < oog::N && allPlaced; ++c) {
            allPlaced = b.makeMove({r, c});
        }
    }
    CHECK_FALSE(allPlaced);
    CHECK(b.isGameOver());
    CHECK_FALSE(b.isDraw());
    CHECK(b.winner() == Cell::Black);
    // No further moves accepted after the win.
    CHECK_FALSE(b.makeMove({14, 14}));
}

TEST_CASE("zobrist distinguishes positions") {
    Board a, b;
    a.makeMove({0, 0});
    b.makeMove({14, 14});
    CHECK(a.zobrist() != b.zobrist());
    // Same position reached by different move order: still equal.
    Board x, y;
    x.makeMove({0, 0});
    x.makeMove({14, 1});
    y.makeMove({0, 0});
    y.makeMove({14, 1});
    CHECK(x.zobrist() == y.zobrist());
}