#pragma once

#include <array>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// Experimental variant of Search with a richer Late Move Reduction (LMR)
// strategy, modeled on techniques found in dhbloo/rapfi's alpha-beta search
// (Rapfi/search/ab/search.cpp, computeLmrReduction / parameter.h, reduction()):
//
//   - a continuous log(depth) * log(moveIndex) base reduction instead of a
//     flat -1 ply once past move 4;
//   - a threat-aware exemption: a move that scores at or above an open-three
//     via evaluateMove() is never reduced, mirroring rapfi's importance/
//     killer-style carve-outs;
//   - a history-based adjustment reusing the same history table move
//     ordering already maintains;
//   - a PV-node-aware cut, using the node's alpha/beta window width as the
//     PV proxy (this project's TT has no stored ttPv flag to reuse);
//   - an "improving" trend check comparing this node's static eval against
//     the same side's eval two plies back.
//
// Kept as a fully separate class (rather than a flag on Search) so the two
// LMR strategies can be benchmarked head-to-head without touching the
// baseline implementation at all. See tools/compare_search.cpp.
class SearchV2 {
public:
    SearchV2();

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

    // Static eval recorded per ply during the current search, indexed by
    // ply; valid because negamax is a plain depth-first recursion, so the
    // slot for ply-2 always still holds this node's actual ancestor's value
    // (no sibling can have overwritten it in between).  Used by the
    // "improving" check.
    std::array<int, MAX_PLY> evalAtPly_{};

    // Per-node scratch: the raw evaluateMove() score for each ordered move
    // at the current negamax call, aligned by index with the sorted move
    // list, so negamax doesn't have to recompute it for the LMR decision.
    std::vector<int> moveThreat_;

    std::uint64_t keyWithSide(int color) const;
    bool outOfTime();
    void orderMoves(std::vector<int>& moves, int color, int ttMove, int depth,
                    int ply);
    int negamax(int color, int depth, int alpha, int beta, int ply);
    bool vcf(int attacker, int depth, int ply);
    void collectWinningPoints(int color, std::vector<int>& out);
    bool hasImmediateWin(int color);

    // Reduction (in plies, >= 0, < depth) for the moveIndex-th ordered move
    // at this node.  nodeIsPv: whether this node's own window is non-null.
    int computeLmrReduction(int color, int m, int depth, int ply,
                            int moveIndex, bool nodeIsPv) const;
};

}  // namespace oog
