#include "doctest.h"

#include "bitboard.hpp"
#include "book.hpp"
#include "search.hpp"

#include <vector>

using namespace oog;

namespace {
void putBlack(BitBoard& bb, int r, int c) { bb.place(0, sqOf(r, c)); }
void putWhite(BitBoard& bb, int r, int c) { bb.place(1, sqOf(r, c)); }
}  // namespace

TEST_CASE("book: probe returns the weighted best move") {
    const std::string text = R"(
        # comment
         : 0,7:3 1,7:1
        0,7 : 7,7:2 7,8:1
    )";
    OpeningBook b;
    CHECK(b.loadText(text) > 0);
    CHECK(b.probe(BitBoard{}, 0) == sqOf(0, 7));
    BitBoard bb;
    putBlack(bb, 0, 7);
    CHECK(b.probe(bb, 1) == sqOf(7, 7));
    // Unknown position -> no move.
    BitBoard other;
    putBlack(other, 7, 7);
    CHECK(b.probe(other, 1) == -1);
}

TEST_CASE("book: symmetry expansion covers rotated lines") {
    // One orientation only; the loader adds the other 7 symmetries.
    const std::string text = "0,7 : 7,7:2\n";
    OpeningBook b;
    CHECK(b.loadText(text) > 0);
    // Rotate the position by 90 degrees: Black at (7,14), White to move.
    BitBoard bb;
    putBlack(bb, 7, 14);
    CHECK(b.probe(bb, 1) == sqOf(7, 7));  // center is fixed by the rotation
}

TEST_CASE("book: first move stays on the ring after expansion") {
    const std::string text = " : 0,7:3\n";
    OpeningBook b;
    CHECK(b.loadText(text) > 0);
    BitBoard empty;
    const int mv = b.probe(empty, 0);
    REQUIRE(mv >= 0);
    const int r = BitBoard::boardRow(mv);
    const int c = BitBoard::boardCol(mv);
    const bool onRing = r < 2 || r >= N - 2 || c < 2 || c >= N - 2;
    CHECK(onRing);
}

TEST_CASE("book: illegal lines are skipped") {
    // A sequence that repeats an occupied square must be dropped entirely.
    const std::string text = "0,7 0,7 : 0,8\n";
    OpeningBook b;
    b.loadText(text);
    BitBoard bb;
    putBlack(bb, 0, 7);
    CHECK(b.probe(bb, 1) == -1);
}

TEST_CASE("book: search plays the book move directly when requested") {
    const std::string text = " : 0,7:3\n";
    OpeningBook b;
    CHECK(b.loadText(text) > 0);

    SearchLimits lim;
    lim.maxDepth = 3;
    lim.playBook = true;
    Search s;
    BitBoard empty;
    const SearchResult r = s.search(empty, 0, lim, &b);
    CHECK(r.bestMove == sqOf(0, 7));
    CHECK(r.nodes == 0);
}

TEST_CASE("book: search prefers book moves in root ordering") {
    const std::string text = " : 0,7:3\n";
    OpeningBook b;
    CHECK(b.loadText(text) > 0);

    // Without playBook the search still runs, but the book move is ordered
    // first and the reported best move should be the book's ring move.
    SearchLimits lim;
    lim.maxDepth = 2;
    Search s;
    BitBoard empty;
    const SearchResult r = s.search(empty, 0, lim, &b);
    CHECK(r.bestMove == sqOf(0, 7));
}

TEST_CASE("book: loadFile on missing path returns 0") {
    OpeningBook b;
    CHECK(b.loadFile("/nonexistent/definitely_missing.txt") == 0);
}
