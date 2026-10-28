#pragma once

#include <array>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchV3 = SearchV2's LMR machinery, plus the three biggest single levers
// identified researching dhbloo/rapfi's Rapfi/search/ab/search.cpp pipeline
// (elo deltas are rapfi's own measured comments, for reference/motivation
// only -- the margins below are untuned first guesses in this project's own
// score units, not ported numbers):
//
//   - Mate distance pruning (Step 3 in rapfi): tightens alpha/beta to the
//     best/worst score reachable from this ply, essentially free.
//   - Futility pruning, child node (~121 elo in rapfi, their single biggest
//     lever): if the static eval already clears beta by a comfortable
//     margin at shallow remaining depth, trust it and skip the node
//     entirely -- no moves generated at all.
//   - Move-count pruning (~107 elo in rapfi): once enough moves have been
//     tried at a shallow node without a cutoff, skip the (already
//     threat-ordered) tail of the move list rather than searching it.
//   - Razoring with VCF (~68 elo in rapfi): when the static eval is well
//     below alpha, don't just trust it -- this project already has a
//     working forced-win search (vcf()/vcfWin()), so a cheap VCF probe
//     verifies there's no hidden forced win being pruned away before
//     returning the eval.
//
// Both new node-level cuts are skipped whenever the opponent's last move
// left an immediate winning threat on the board (hasImmediateWin), mirroring
// rapfi's "oppo4" gate: a position that's already forced shouldn't be
// pruned on eval, which is likely misleading there.
class SearchV3 {
public:
    SearchV3();

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
};

}  // namespace oog
