#pragma once

#include <array>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchVct = SearchV3's search core (LMR + futility/move-count pruning +
// razoring-VCF), plus a **threat-space search extended from VCF to VCT**.
//
// Source: Allis, van den Herik & Huntjens, "Go-Moku and Threat-Space Search"
// (1993) -- the technique that solved Go-Moku.  Its central claim is that a
// winning line in Go-Moku is almost always a *chain of threats*, so searching
// only the threat sub-space finds wins that a general-depth search misses at
// the same cost.
//
// This project already implements the weaker half of that idea: `vcf()`
// (Victory by Continuous Fours) chains moves that create a four, where the
// defender's reply is always forced to exactly one square.  Allis's full
// threat space also admits **three-threats** (a move creating an open three,
// i.e. threatening to make an *open four* next), where the defender has
// several legal defences instead of one.  Chains mixing threes and fours
// (VCT, Victory by Continuous Threats) are where most real Go-Moku wins live
// -- a three-four-five combination is invisible to a fours-only search.
//
// What is implemented here:
//   - `vct()` recurses over both threat kinds.  Four-threats behave exactly
//     as in `vcf()` (single forced reply).  Three-threats enumerate a
//     defence set and require the attacker to win against *every* member.
//   - The defence set for a three-threat is the attacker's own open-four gain
//     squares (blocking any one kills the open three) **union** the
//     defender's four-making squares (Allis's counter-threat refutation: a
//     defender who can start their own forcing sequence is not obliged to
//     defend passively).
//   - Every candidate attacking move is rejected outright if the defender
//     already has an immediate five available, same unsoundness guard
//     `vcf()` uses.
//
// Deliberate deviations from the paper, and why:
//   - The paper's dependency-based search / "all defensive moves at once"
//     simplification is *not* implemented.  Those exist to prove wins over
//     very long threat chains offline; here the chain length is bounded to a
//     few threats and the search runs inside a per-move node budget, so the
//     straightforward enumeration is the cheaper engineering choice.
//   - The defence set is a heuristic subset, not provably every legal
//     defence, so a claimed VCT win is *not* a proof.  It is therefore used
//     only as a **root move suggestion with a normal-search cross-check**,
//     never returned as a mate score on its own -- see `search()`.
//
// Budget: VCT shares the search's own node counter (so node-budget
// comparisons against the other engines stay honest) and is additionally
// capped at a fraction of the per-move budget, so a fruitless threat search
// can never starve the main alpha-beta search.
class SearchVct {
public:
    SearchVct();

    SearchResult search(BitBoard& bb, int color, const SearchLimits& limits,
                        const OpeningBook* book = nullptr);

    int vcfWin(BitBoard& bb, int attacker, int maxAttacks = 12);

    // Victory-by-continuous-threats probe: returns a first move of a threat
    // chain that appears to win for `attacker`, or -1.  Heuristic (see the
    // class comment) -- callers must verify before trusting it as a win.
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
    // Hard ceiling on nodes_ for the whole VCT phase (0 = no VCT running).
    std::uint64_t vctNodeCap_ = 0;

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

    // --- threat-space (VCT) helpers -------------------------------------
    // Winning points of `color`, searched only over `cand` (five-completing
    // squares are always adjacent to stones, so the near-move list is a
    // sufficient domain and far cheaper than a full 272-cell scan).
    void winPointsIn(int color, const std::vector<int>& cand,
                     std::vector<int>& out);
    // Squares where `color` would create an *open four* (two winning points)
    // -- i.e. the gain squares of `color`'s open threes.
    void openFourPointsIn(int color, const std::vector<int>& cand,
                          std::vector<int>& out);
    // Squares where `color` would create at least a four (one winning point).
    void fourPointsIn(int color, const std::vector<int>& cand,
                      std::vector<int>& out);
    bool vct(int attacker, int depth, int ply);
};

}  // namespace oog
