#pragma once

#include <array>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchThreatCand = SearchV3's search core, plus **threat-driven candidate
// generation**: when the position is forcing, don't generate the usual
// neighbourhood of ~40 moves and rely on ordering to sort them -- generate
// only the moves that can possibly matter.
//
// Every other engine in this project generates the same move set at every
// node (`genMovesNear`, all empty squares within Chebyshev distance 2 of a
// stone) and leans entirely on move *ordering* plus a fixed candidate cap of
// 24 to keep the branching factor down.  Ordering is a soft tool: a
// hopeless move still costs a node before it is refuted.  Restricting
// *generation* is a hard tool, and in a game as threat-dominated as Go-Moku
// most nodes are forcing enough for it to apply.
//
// Sources: the "candidate move generation" step of the tournament-winning
// Go-Moku engine documented at sortingsearching.com, which considers "only
// moves that defend against the danger" from the opponent's threat
// sequences; and Cao & Lin (2019, arXiv:1912.05407), which filters MCTS
// candidates down to "effective attacks" by threat value.  Both are the
// applied form of Allis's threat-space idea (see `search_vct.hpp`) pushed
// from a separate proof search into the main search's move generator.
//
// Two levels, deliberately separated because only one of them is exact:
//
//   Level 1 -- forced-reply restriction (**exact, not a heuristic**).  If
//   the opponent has at least one square that completes five for them, then
//   any move that neither completes five for us nor occupies one of those
//   squares loses outright.  So the candidate list *is* exactly
//   `our five-points ∪ their five-points`, and dropping everything else
//   removes no viable move.  This applies at every depth and typically
//   collapses a forcing node from ~24 candidates to 1-2.
//
//   Level 2 -- open-three restriction (**heuristic**).  If the opponent has
//   no five available but does threaten to make an open four, restrict to:
//   the squares that kill that open four, our own four-making squares, and
//   our own open-four gain squares (i.e. defend, or out-force them).  This
//   is not exact -- a quiet move outside the set can occasionally be the
//   best answer -- so it is gated to nodes with enough remaining depth to be
//   worth the extra generation cost, and never applied at the root, where a
//   wrong restriction cannot be recovered from.
//
// Cost note: level 2 needs an O(candidates^2) scan (for each empty square,
// does placing a stone there create two winning points), which is far more
// expensive per node than level 1's linear scan.  It is gated on remaining
// depth so the cost is only paid where the node saving repays it.
class SearchThreatCand {
public:
    SearchThreatCand();

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
    std::array<int, MAX_PLY> evalAtPly_{};
    std::vector<int> moveThreat_;

    std::uint64_t keyWithSide(int color) const;
    bool outOfTime();
    void orderMoves(std::vector<int>& moves, int color, int ttMove, int depth,
                    int ply);
    int negamax(int color, int depth, int alpha, int beta, int ply);
    bool vcf(int attacker, int depth, int ply);
    void collectWinningPoints(int color, std::vector<int>& out);
    bool hasImmediateWin(int color);

    int computeLmrReduction(int color, int m, int depth, int ply,
                            int moveIndex, bool nodeIsPv) const;

    // --- threat-driven candidate generation ------------------------------
    // Squares in `cand` where `color` would complete five.
    void winPointsIn(int color, const std::vector<int>& cand,
                     std::vector<int>& out);
    // Squares in `cand` where `color` would create an open four (two winning
    // points) -- the gain squares of `color`'s open threes.
    void openFourPointsIn(int color, const std::vector<int>& cand,
                          std::vector<int>& out);
    // Squares in `cand` where `color` would create at least a four.
    void fourPointsIn(int color, const std::vector<int>& cand,
                      std::vector<int>& out);
    // Narrow `moves` to the threat-relevant subset for `color` to move.
    // Returns true if a restriction was applied.
    bool restrictToThreats(int color, int depth, bool rootNode,
                           std::vector<int>& moves);
};

}  // namespace oog
