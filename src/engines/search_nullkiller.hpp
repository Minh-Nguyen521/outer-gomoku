#pragma once

#include <array>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchNullKiller = SearchV3's search core, plus the three classical
// alpha-beta enhancements this codebase had none of: **null-move forward
// pruning**, **killer moves**, and the **counter-move heuristic**.
//
// Null-move forward pruning (Beal 1989; Donninger 1993).  At a node where
// the static eval already looks good, give the *opponent* a free move and
// search the result shallowly: if the position is still at or above beta
// even after handing over a tempo, it is almost certainly a real cutoff and
// the node can be skipped without generating any moves.
//
// Why it is a better fit here than in chess: the standard objection to null
// move is **zugzwang** -- chess positions exist where being obliged to move
// is a disadvantage, so "passing" is not a lower bound on "moving" and the
// pruning is unsound.  Free-style Go-Moku has no zugzwang at all: placing a
// stone can never worsen your own position (stones are never removed, never
// blocked, and an extra stone only ever adds to your own lines).  Passing is
// therefore a strict lower bound on moving, which makes the null-move
// assumption *sounder in this game than in the one it was invented for*.
// This matches practice -- the tournament-winning Go-Moku engine documented
// at sortingsearching.com uses null-move forward pruning gated on "no
// immediate opponent threat", the same gate used below.
//
// Guards kept anyway (a null move is still a heuristic, not a proof):
//   - never at the root, on the PV, or when already in a mate-scored window;
//   - never when either side has an immediate five available, so a forced
//     tactical position is never pruned on a static eval;
//   - never twice in a row (`nullPly_`), which would just be a double pass.
//
// Killer moves (Akl & Newborn 1977).  Two "killer" squares are remembered
// per ply -- quiet moves that produced a beta cutoff at that ply elsewhere
// in the tree -- and are given a large ordering bonus. This project already
// had a history heuristic, which is the *global* version of the same idea;
// killers are the sibling-local version and are complementary to it.
//
// Counter-move heuristic (Uiterwijk 1992).  Indexed by the opponent's last
// move, remembers the reply that refuted it, on the observation that a good
// answer to a stone on a square tends to be the same answer wherever in the
// tree that stone appears.  In Go-Moku this is unusually apt: the reply to a
// threatening stone is genuinely a local, position-independent property most
// of the time (block the open end).
class SearchNullKiller {
public:
    SearchNullKiller();

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

    // Two killer squares per ply, most recent first.
    std::array<std::array<int, 2>, MAX_PLY> killers_{};
    // counterMove_[color * SIZE + sq] = reply that refuted `color` playing sq.
    std::vector<int> counterMove_;
    // Ply at which a null move is currently in effect (-1 = none); blocks a
    // second consecutive null move.
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
};

}  // namespace oog
