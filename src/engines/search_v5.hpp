#pragma once

#include <array>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchV5 = SearchV4 plus a defensive capability no engine in this project
// has ever had, and three cheaper search refinements to pay for it.
//
// ## The gap V5 targets
//
// Every engine here runs `vcf()` at the root to find *its own* forced win.
// None of them ever asks the mirror question: **does my candidate move hand
// the opponent a forced win?** That matters because a VCF chain is searched
// to 12 forcing plies, while the main search reaches roughly depth 8-9 at a
// realistic time control. There is therefore a band of forced losses that is
// structurally invisible: the engine plays a natural-looking move, the
// opponent starts a chain of fours, and by the time the main search is deep
// enough to see the mate it has already been played into.
//
// **Root safety screening** closes it. Before the iterative deepening loop,
// each of the top root candidates is played and the *opponent* is given a
// VCF search from that position. Any move that hands over a forced win is
// marked unsafe and skipped at the root — unless every candidate is unsafe,
// in which case the position is lost anyway and the screen steps aside
// rather than leaving the engine with no move to play.
//
// This is affordable only because of SearchV4's bitboard-parallel five-point
// primitive: `vcf()` calls `collectWinningPoints()` and `hasImmediateWin()`
// at every node, and those are 9.5x cheaper than they were before Round 6.
// The screen is additionally bounded by its own node allowance so it can
// never consume the budget the real search needs.
//
// ## Paying for it
//
// 1. **Node-level mask caching.** SearchV4 recomputes the two five-point
//    masks up to six times per node — once each for the razoring gate and
//    the futility gate, twice for the null-move gate, twice for the
//    forced-reply restriction — even though the board does not change
//    between them. V5 computes them at most once per node and shares them.
//
// 2. **Forced-reply extension.** When the exact restriction leaves exactly
//    one legal non-losing move, the position is forced, and spending a whole
//    ply of the depth budget on a move with no alternative wastes the
//    forcing information. Such nodes are searched one ply deeper instead.
//    This terminates: every extended ply places a stone, and the board holds
//    225 of them.
//
// 3. **Four-threat extension.** A move that creates a four forces the reply,
//    so the same argument applies. The trigger is the threat score
//    `orderMoves()` has already computed, so it costs nothing extra.
//    Extensions share a per-path budget, and an extended move is never also
//    reduced by LMR.
//
// 4. **Internal iterative deepening.** At a high-depth node with no TT move,
//    a reduced-depth search is run first purely to produce one, because
//    ordering quality dominates alpha-beta's node count.
//
// 5. **Aspiration windows.** Iterative deepening re-searches around the
//    previous iteration's score rather than opening a full window every
//    time, widening on a fail high/low.
class SearchV5 {
public:
    SearchV5();

    SearchResult search(BitBoard& bb, int color, const SearchLimits& limits,
                        const OpeningBook* book = nullptr);

    int vcfWin(BitBoard& bb, int attacker, int maxAttacks = 12);
    int vctWin(BitBoard& bb, int attacker, int maxThreats = 4);

    // True if, after `color` plays `sq`, the opponent has a forced VCF win.
    // Exposed for testing the screen directly.
    bool moveAllowsOpponentVcf(BitBoard& bb, int color, int sq, int vcfDepth = 8);

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

    // Root safety screen state.
    std::array<bool, BitBoard::SIZE> rootUnsafe_{};
    bool rootHasSafe_ = false;

    // Extension budget along the current path.
    int extUsed_ = 0;

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

    void openFourPointsIn(int color, const std::vector<int>& cand,
                          std::vector<int>& out);
    void fourPointsIn(int color, const std::vector<int>& cand,
                      std::vector<int>& out);
    bool vct(int attacker, int depth, int ply);
    bool worthVct(int color);

    // Five-point bitmask over the padded 272-bit grid, one bit per square.
    using MaskArr = std::array<std::uint64_t, BitBoard::WORDS>;

    // Narrow `moves` to `our five-points ∪ theirs` when the opponent
    // threatens five. Exact: discards only moves that lose on the spot.
    // Masks are parameters, not members, because negamax recurses.
    // Returns true if a restriction was applied.
    bool restrictToForcedReplies(const MaskArr& ourFive, const MaskArr& oppFive,
                                 std::vector<int>& moves);

    // Mark root candidates that hand the opponent a forced VCF win.
    void screenRootMoves(BitBoard& bb, int color);

    // Bound a pre-search phase to ~1/denom of the move budget by temporarily
    // lowering the deadline; returns the previous value for endPhaseBudget().
    long long beginPhaseBudget(int denom);
    void endPhaseBudget(long long saved);
};

void fivePointSquaresV5(const BitBoard& bb, int color, std::vector<int>& out);

}  // namespace oog
