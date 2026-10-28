#pragma once

#include <array>

#include "eval_v2.hpp"
#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, INF, MAX_PLY

namespace oog {

class OpeningBook;

// SearchEvalV2 = the baseline `Search`, identical in every respect except
// that it evaluates with `EvalStateV2` (window score **plus a fork term**)
// instead of `EvalState`.
//
// Built on the baseline rather than on `SearchV4`/`SearchV5` deliberately.
// The baseline is the engine with a clean external anchor (22.0% vs NNUE
// rapfi, 3.0% vs classical rapfi over 100 games each), and it is the engine
// whose search has *not* been modified across six rounds. That makes the fork
// term the only variable, so any difference is attributable to it rather than
// confounded with search changes that already failed to transfer externally
// (see docs/search-comparison.md Round 7).
//
// The motivating defect: `evaluate()` sums window scores linearly, so a
// double four scores S_CLOSED_FOUR * 2 = 20,000 while a single open four
// scores S_OPEN_FOUR = 100,000 -- though both are winning. Forks decide
// Go-Moku games, and the old evaluation cannot see them; only the search
// ever finds them, which is why our engines collapse when depth is reduced
// and rapfi does not.
class SearchEvalV2 {
public:
    SearchEvalV2();

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
