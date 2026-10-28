#pragma once

#include <array>

#include "eval_v2.hpp"
#include "search_v4.hpp"  // fivePointSquares: fast threat primitive
#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchQ = `SearchEvalV2` (baseline search + fork-aware evaluation) plus a
// **quiescence search at the horizon**.
//
// The defect it targets: at `depth <= 0` every engine here returns a *static*
// evaluation. If a forced win is one forcing sequence away, that number is
// simply wrong, and the search has no way to know -- the classic horizon
// effect. Rapfi dives into VCF when depth reaches zero and rates it ~17 elo.
//
// This matters more for us than the raw figure suggests. We measured that our
// engines collapse when search depth is reduced (games under CPU contention
// went from 22% to 1% against the same opponent) while rapfi barely noticed.
// That is a symptom of untrustworthy leaves: rapfi's evaluation is good
// enough to stand on its own, ours is not. Making leaves *tactically aware*
// attacks that directly, without needing more depth and without any training.
//
// The gate is free. `EvalStateV2` already maintains `fourGains` per colour
// incrementally, so "does the side to move have a four available?" costs a
// comparison rather than a board scan -- the probe only runs where a forcing
// sequence could actually exist. `collectWinningPoints`/`hasImmediateWin`
// also use the bitboard-parallel primitive from `SearchV4` (9.5x faster,
// verified equivalent to the loop it replaces), because a VCF probe at leaves
// is only affordable with it.
//
// No training, no network, no fitted parameters -- this is pure algorithm.
class SearchQ {
public:
    SearchQ();

    SearchResult search(BitBoard& bb, int color, const SearchLimits& limits,
                        const OpeningBook* book = nullptr);

    int vcfWin(BitBoard& bb, int attacker, int maxAttacks = 12);

    std::uint64_t nodes() const { return nodes_; }
    std::uint64_t ttHits() const { return tt_.hits(); }
    std::uint64_t ttProbes() const { return tt_.probes(); }
    void clearTT() { tt_.clear(); }

private:
    TT tt_;
    EvalStateV2 eval_;  // the only structural difference from `Search`
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
};

}  // namespace oog
