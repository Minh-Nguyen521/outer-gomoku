#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "board.hpp"

namespace oog {

// Compact 15x15 bitboard with sentinel rows/columns so that 4-direction line
// scans and shift-based pattern detection never wrap out of the board.
//
// Layout: each row is ROW_W = 16 bits wide (columns 0..14 playable, bit 15 is
// a sentinel that is always 0).  There is one all-zero pad row on top and one
// on the bottom (rows 0 and 16), so rows 1..15 are playable.
//
//   flat bit index sq = row * ROW_W + col,  row in [1,15], col in [0,14].
//
// Direction steps (positive = down/right):
//   horizontal  +1,  vertical  +ROW_W,  diag-down-right  +ROW_W+1,
//   diag-down-left  +ROW_W-1.
class BitBoard {
public:
    static constexpr int ROW_W = 16;
    static constexpr int ROWS = 17;               // pad + 15 + pad
    static constexpr int SIZE = ROW_W * ROWS;     // 272
    static constexpr int WORDS = (SIZE + 63) / 64;  // 5

    BitBoard() { clear(); }

    void clear();

    // color: 0 = Black, 1 = White.
    void place(int color, int sq);
    void remove(int color, int sq);

    bool isBlack(int sq) const;
    bool isWhite(int sq) const;
    bool isEmpty(int sq) const;
    Cell at(int sq) const;
    int count() const;

    // Any 5 (or more) in a row for the given color anywhere on the board.
    bool hasFive(int color) const;
    // True if the stone at sq completes (or is part of) a 5-in-a-row.
    bool winsAt(int color, int sq) const;

    // Moves (empty playable squares) within Chebyshev distance 2 of any stone;
    // on an empty board returns the outer-ring opening candidates.
    void genMovesNear(std::vector<int>& out) const;

    // Same, with an explicit radius. Radius 2 is the historical behaviour and
    // what genMovesNear() uses. It is also the *narrowest* of the six options
    // rapfi exposes as a tunable (square2 .. full_board): any strong move
    // further than 2 from every stone is unplayable for us, and no amount of
    // search or evaluation work can recover it. Radius is opt-in so the
    // baseline stays byte-identical and previously measured results hold.
    void genMovesNearRadius(std::vector<int>& out, int radius) const;

    // Stones-only Zobrist hash, updated incrementally.  Does NOT include the
    // side to move (callers fold that in when needed).
    std::uint64_t hash() const { return key_; }

    // Coordinate helpers (board coords are 0..14, matching Board's (r,c)).
    static int sqOf(int br, int bc) { return (br + 1) * ROW_W + bc; }
    static int boardRow(int sq) { return sq / ROW_W - 1; }
    static int boardCol(int sq) { return sq % ROW_W; }
    static bool isPlayableSq(int sq) {
        const int r = sq / ROW_W;
        const int c = sq % ROW_W;
        return r >= 1 && r <= 15 && c >= 0 && c <= 14;
    }

    // Copy the stone layout of a Board into this bitboard (for testing).
    void fillBoard(const Board& b);

    const std::array<std::uint64_t, WORDS>& bits(int color) const {
        return color == 0 ? black_ : white_;
    }

private:
    std::array<std::uint64_t, WORDS> black_{};
    std::array<std::uint64_t, WORDS> white_{};
    std::uint64_t key_ = 0;
};

// Free-function shorthand matching Board's indexOf-style helpers.
inline int sqOf(int br, int bc) { return BitBoard::sqOf(br, bc); }

// Fold the side to move into the stones-only Zobrist key.  Shared by the
// search's transposition table and the opening book.
inline std::uint64_t sideKey(int color) {
    return color == 0 ? 0x9E3779B97F4A7C15ULL : 0xBF58476D1CE4E5B9ULL;
}

inline std::uint64_t positionKey(const BitBoard& bb, int color) {
    return bb.hash() ^ sideKey(color);
}

}  // namespace oog
