#pragma once

#include <array>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchMinibal implements the Minibal_+ decision rule from Cohen-Solal &
// Cazenave, "Minibal: Balanced Game-Playing Without Opponent Modeling"
// (arXiv:2603.23059), on top of this project's ordinary alpha-beta search.
//
// The paper's own algorithm is built on Unbounded Minimax, a different
// search backbone from this project's iterative-deepening alpha-beta/PVS.
// What's ported here is the *decision rule*, not the search backbone:
// everywhere except the root, this class's negamax is byte-for-byte
// identical to the baseline Search's (same TT, same flat LMR, same move
// ordering) -- the "opponent plays rationally" side of the paper's design.
// Only the root's own move selection changes:
//
//   Minibal_+: if any legal move has a positive (winning) score, play the
//   one with the SMALLEST positive score (win, but by the narrowest
//   margin found); otherwise play the move with the LARGEST (least
//   negative) score, same as normal alpha-beta would.
//
// This never intentionally throws a game that could be won -- it only
// avoids running up the score once a win is already secured, unlike the
// paper's Minibal_n variant (which the paper's own results show losing far
// more often, and which is not implemented here).
//
// Getting each root move's *true* full-window value (needed to compare
// candidates against each other) means the root loop cannot use ordinary
// alpha-beta pruning across siblings the way Search's inner nodes do --
// every candidate is searched at a full window. This makes the root pass
// itself more expensive than Search's root pass at the same depth; only
// the root differs, so the cost is bounded by the (already-limited)
// candidate count, not by tree size.
class SearchMinibal {
public:
    SearchMinibal();

    SearchResult search(BitBoard& bb, int color, const SearchLimits& limits,
                        const OpeningBook* book = nullptr);

    int vcfWin(BitBoard& bb, int attacker, int maxAttacks = 12);

    std::uint64_t nodes() const { return nodes_; }
    void clearTT() { tt_.clear(); }

private:
    TT tt_;
    EvalState eval_;
    BitBoard* bb_ = nullptr;
    SearchLimits limits_;
    std::uint64_t nodes_ = 0;
    int rootColor_ = 0;
    int rootBestMove_ = -1;
    int lastSq_ = -1;
    int lastColor_ = 0;
    bool aborted_ = false;
    std::chrono::steady_clock::time_point t0_;
    std::vector<int> history_;

    int vcfFirstMove_ = -1;
    std::array<long long, BitBoard::SIZE> bookBonus_{};

    std::uint64_t keyWithSide(int color) const;
    bool outOfTime();
    void orderMoves(std::vector<int>& moves, int color, int ttMove, int depth,
                    int ply);
    int negamax(int color, int depth, int alpha, int beta, int ply);
    bool vcf(int attacker, int depth, int ply);
    void collectWinningPoints(int color, std::vector<int>& out);
    bool hasImmediateWin(int color);

    // Root pass for one iterative-deepening depth: searches every candidate
    // move at a full window, applies the Minibal_+ rule, and sets
    // rootBestMove_ / returns the chosen move's own score.
    int rootMinibalPass(int color, int depth);
};

}  // namespace oog
