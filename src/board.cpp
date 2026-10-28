#include "board.hpp"

#include "rules.hpp"

#include <random>
#include <sstream>

namespace oog {

namespace detail {

// Deterministic Zobrist keys so hashes are reproducible across runs.
// z[color * N_CELLS + index], color in {0 = Black, 1 = White}.
inline const std::array<std::uint64_t, 2 * N_CELLS> ZOBRIST = [] {
    std::array<std::uint64_t, 2 * N_CELLS> z{};
    std::mt19937_64 rng(0xC0FFEE5EED);
    for (auto& v : z) v = rng();
    return z;
}();

// XORed into the key whenever the side to move flips.
inline constexpr std::uint64_t TURN_KEY = 0x9E3779B97F4A7C15ULL;

}  // namespace detail

void Board::reset() {
    cells_.fill(Cell::Border);
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cells_[static_cast<std::size_t>((r + PAD) * S + (c + PAD))] =
                Cell::Empty;
        }
    }
    turn_ = 0;
    ply_ = 0;
    history_.clear();
    zobristKey_ = 0;
    winner_ = Cell::Empty;
}

bool Board::isRingMove(int index) const {
    Coord cd = coordOf(index);
    return cd.r < 2 || cd.r >= N - 2 || cd.c < 2 || cd.c >= N - 2;
}

bool Board::isLegalIndex(int index) const {
    if (isGameOver()) return false;
    if (at(index) != Cell::Empty) return false;
    // Outer-Open Gomoku rule: Black's first move must be on the outer
    // 2 rows/columns of the board.
    // Outer-Open only: in freestyle any first move is legal, and
    // rejecting the opponent's centre opening would forfeit the game.
    if (ply_ == 0 && outerOpenRule() && !isRingMove(index)) return false;
    return true;
}

bool Board::makeMoveIndex(int index) {
    if (!isLegalIndex(index)) return false;
    history_.push_back(index);
    const int color = turn_;
    cells_[static_cast<std::size_t>(index)] =
        color == 0 ? Cell::Black : Cell::White;
    zobristKey_ ^= detail::ZOBRIST[static_cast<std::size_t>(color) * N_CELLS +
                                   static_cast<std::size_t>(index)];
    zobristKey_ ^= detail::TURN_KEY;
    ++ply_;
    turn_ ^= 1;
    if (isWinAt(index)) winner_ = cells_[static_cast<std::size_t>(index)];
    return true;
}

void Board::undoMove() {
    if (history_.empty()) return;
    const int index = history_.back();
    history_.pop_back();
    const Cell col = at(index);
    const int color = col == Cell::Black ? 0 : 1;
    cells_[static_cast<std::size_t>(index)] = Cell::Empty;
    zobristKey_ ^= detail::ZOBRIST[static_cast<std::size_t>(color) * N_CELLS +
                                   static_cast<std::size_t>(index)];
    zobristKey_ ^= detail::TURN_KEY;
    --ply_;
    turn_ ^= 1;
    winner_ = Cell::Empty;
}

bool Board::isWinAt(int index) const {
    const Cell col = at(index);
    if (col != Cell::Black && col != Cell::White) return false;
    // (dr, dc) for the four line directions.
    constexpr int DIRS[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (const auto& d : DIRS) {
        const int step = d[0] * S + d[1];
        int fwd = 0;
        int i = index;
        while (at(i) == col) {
            ++fwd;
            i += step;
        }
        int bwd = 0;
        i = index - step;
        while (at(i) == col) {
            ++bwd;
            i -= step;
        }
        if (fwd + bwd >= 5) return true;
    }
    return false;
}

bool Board::isWin(Cell color) const {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            const int idx = (r + PAD) * S + (c + PAD);
            if (at(idx) == color && isWinAt(idx)) return true;
        }
    }
    return false;
}

std::string Board::serialize() const {
    std::string s = turn_ == 0 ? "B" : "W";
    for (const int i : history_) {
        const Coord cd = coordOf(i);
        s += ' ';
        s += std::to_string(cd.r);
        s += ',';
        s += std::to_string(cd.c);
    }
    return s;
}

bool Board::load(const std::string& fen) {
    reset();
    std::istringstream iss(fen);
    std::string tok;
    if (!(iss >> tok) || (tok != "B" && tok != "W")) return false;
    const int expectedTurn = tok == "B" ? 0 : 1;
    while (iss >> tok) {
        const auto comma = tok.find(',');
        if (comma == std::string::npos) return false;
        int r = 0;
        int c = 0;
        try {
            r = std::stoi(tok.substr(0, comma));
            c = std::stoi(tok.substr(comma + 1));
        } catch (...) {
            return false;
        }
        if (r < 0 || r >= N || c < 0 || c >= N) return false;
        if (!makeMove({r, c})) return false;
    }
    return turn_ == expectedTurn;
}

std::string Board::toString() const {
    std::string s;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            const Cell v = at((r + PAD) * S + (c + PAD));
            s += v == Cell::Empty     ? '.'
                 : v == Cell::Black   ? 'B'
                 : v == Cell::White   ? 'W'
                                      : '?';
            if (c != N - 1) s += ' ';
        }
        if (r != N - 1) s += '\n';
    }
    return s;
}

}  // namespace oog
