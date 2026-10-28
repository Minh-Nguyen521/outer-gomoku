#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace oog {

// Playable board is N x N.  Internally the grid is padded on all four sides
// with a PAD-cell "Border" ring so that 5-in-a-row scans never need bounds
// checks (branchless boundary handling).
inline constexpr int N = 15;
inline constexpr int PAD = 4;
inline constexpr int S = N + 2 * PAD;  // 23
inline constexpr std::size_t N_CELLS = static_cast<std::size_t>(S * S);

enum class Cell : std::uint8_t { Empty = 0, Black = 1, White = 2, Border = 3 };

struct Coord {
    int r = 0;
    int c = 0;
    friend bool operator==(const Coord&, const Coord&) = default;
};

// Map a playable coordinate (r,c in [0,N)) to the padded internal index.
inline int indexOf(Coord cd) { return (cd.r + PAD) * S + (cd.c + PAD); }

// Inverse of indexOf.
inline Coord coordOf(int index) {
    return {index / S - PAD, index % S - PAD};
}

class Board {
public:
    Board() { reset(); }

    void reset();

    // --- state queries -----------------------------------------------------
    Cell at(int index) const { return cells_[static_cast<std::size_t>(index)]; }
    Cell cell(Coord cd) const { return at(indexOf(cd)); }
    int turn() const { return turn_; }  // 0 = Black to move, 1 = White
    bool isBlackToMove() const { return turn_ == 0; }
    int ply() const { return ply_; }  // number of moves made
    Cell winner() const { return winner_; }
    bool isGameOver() const { return winner_ != Cell::Empty || isDraw(); }
    bool isDraw() const { return winner_ == Cell::Empty && ply_ >= N * N; }
    int lastMoveIndex() const {
        return history_.empty() ? -1 : history_.back();
    }

    // --- move legality -----------------------------------------------------
    bool isLegal(Coord cd) const { return isLegalIndex(indexOf(cd)); }
    bool isLegalIndex(int index) const;
    bool isRingMove(int index) const;  // move in the outer 2 rows/columns

    // --- move application ---------------------------------------------------
    // Returns false if the move is illegal; otherwise applies it.
    bool makeMove(Coord cd) { return makeMoveIndex(indexOf(cd)); }
    bool makeMoveIndex(int index);
    void undoMove();

    // --- win detection ------------------------------------------------------
    bool isWinAt(int index) const;  // win completed by the stone at index
    bool isWin(Cell color) const;   // full-board scan

    // --- hashing ------------------------------------------------------------
    std::uint64_t zobrist() const { return zobristKey_; }  // includes side to move

    // --- serialization ------------------------------------------------------
    // Format: "<B|W> <r,c> <r,c> ..." (side to move, then move list).
    std::string serialize() const;
    bool load(const std::string& fen);

    std::string toString() const;  // ASCII board for debugging

private:
    std::array<Cell, N_CELLS> cells_{};
    int turn_ = 0;
    int ply_ = 0;
    std::vector<int> history_;
    std::uint64_t zobristKey_ = 0;
    Cell winner_ = Cell::Empty;
};

}  // namespace oog
