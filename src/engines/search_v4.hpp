#pragma once

#include <array>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchV4 -- the Round 6 attempt to beat every strategy in this project.
//
// Round 5 (see docs/search-comparison.md) built three paper-derived variants
// on top of SearchV3 -- threat-space search extended to VCT, null-move
// pruning, and threat-driven candidate generation. Under a matched *node*
// budget all three finished within half a point of SearchV3; under a matched
// *wall-clock* budget all three lost, one of them 2-8. The diagnosis was the
// same in every case and it was not the algorithms: each one leans on the
// question "where can this colour complete five?", and this project answered
// that question by looping over all ~272 squares doing place / winsAt /
// remove. A node budget does not charge for that; a clock does.
//
// SearchV4's premise is therefore that Round 5 measured the *primitive*, not
// the ideas. It replaces the primitive and then re-adopts the ideas that the
// cheaper primitive can now afford.
//
// 1. **Bitboard-parallel five-point detection.** `fivePoints()` computes the
//    complete set of squares where a colour would make five, for the whole
//    board at once, with shift/AND over the 272-bit grid: for each of the 4
//    directions and each of the 5 positions the gap can occupy in a
//    five-window, require the other four cells to be own stones. It is
//    exactly equivalent to the old loop (a five-window containing an empty s
//    and four own stones is precisely a square where placing wins) and is
//    verified against it over random positions in
//    `tests/search_v4_test.cpp`. This makes `hasImmediateWin()` and
//    `collectWinningPoints()` -- both on the hot path of VCF, of razoring,
//    and of futility pruning, all inherited from SearchV3 -- dramatically
//    cheaper, which speeds up the *existing* search before anything new is
//    added.
//
// 2. **Exact forced-reply candidate restriction** (Round 5's
//    `SearchThreatCand` level 1, the one component that drew SearchV3 under
//    both budgets). If the opponent has any square that completes five, then
//    every move except "make five ourselves" or "occupy one of theirs" loses
//    outright, so the candidate list *is* `our five-points ∪ theirs` and
//    restricting to it discards nothing playable. Round 5 paid for this with
//    a per-node linear scan; here it is two mask computations.
//    Round 5's heuristic level 2 (the open-three restriction) is **not**
//    carried over -- it needs a quadratic scan that the mask trick does not
//    make cheap, and it was the risky half.
//
// 3. **Null-move pruning with a cheap gate** (Round 5's `SearchNullKiller`).
//    Go-Moku has no zugzwang -- placing a stone can never worsen your own
//    position -- so passing is a strict lower bound on moving and the
//    null-move assumption is sounder here than in chess where it was
//    invented. Round 5 gated it on two full-board `hasImmediateWin()` scans
//    per attempt, which is what made it the worst performer on the clock;
//    the gate is now two masks that the node has already computed.
//
// 4. **Killer moves and the counter-move heuristic** (Akl & Newborn 1977;
//    Uiterwijk 1992) -- pure move ordering, essentially free, and the one
//    part of Round 5 with no measured downside.
//
// 5. **Root VCT, gated** (Round 5's `SearchVct`; Allis, van den Herik &
//    Huntjens 1993). The threat search demonstrably finds wins VCF cannot,
//    but Round 5 ran it every single move, spending up to a third of the
//    budget whether or not any three-threat existed. Here it runs only when
//    a cheap precondition says the position is actually forcing, and its
//    internal scans use the fast primitive.
class SearchV4 {
public:
    SearchV4();

    SearchResult search(BitBoard& bb, int color, const SearchLimits& limits,
                        const OpeningBook* book = nullptr);

    int vcfWin(BitBoard& bb, int attacker, int maxAttacks = 12);
    int vctWin(BitBoard& bb, int attacker, int maxThreats = 4);

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
    int vctFirstMove_ = -1;
    std::uint64_t vctNodeCap_ = 0;

    std::array<long long, BitBoard::SIZE> bookBonus_{};
    std::array<int, MAX_PLY> evalAtPly_{};
    std::vector<int> moveThreat_;

    std::array<std::array<int, 2>, MAX_PLY> killers_{};
    std::vector<int> counterMove_;
    int nullPly_ = -1;

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

    // Narrow `moves` to `our five-points ∪ theirs` when the opponent
    // threatens five. Exact: discards only moves that lose on the spot.
    // Returns true if a restriction was applied.
    bool restrictToForcedReplies(int color, std::vector<int>& moves);

    // Threat-space (VCT) helpers, all using the fast primitive.
    void openFourPointsIn(int color, const std::vector<int>& cand,
                          std::vector<int>& out);
    void fourPointsIn(int color, const std::vector<int>& cand,
                      std::vector<int>& out);
    bool vct(int attacker, int depth, int ply);
    // Cheap check for "is a threat search worth starting here at all".
    bool worthVct(int color);

    // Bound a pre-search phase to ~1/denom of the move budget by temporarily
    // lowering the deadline; returns the previous value for endPhaseBudget().
    long long beginPhaseBudget(int denom);
    void endPhaseBudget(long long saved);
};

// Exposed for testing: the complete set of squares where `color` would
// complete five, as a list of squares. Equivalent to looping over every
// empty square and testing place/winsAt/remove, but computed with
// board-wide shift/AND. See `tests/search_v4_test.cpp`.
void fivePointSquares(const BitBoard& bb, int color, std::vector<int>& out);

}  // namespace oog
