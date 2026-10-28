#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "bitboard.hpp"
#include "eval.hpp"
#include "tt.hpp"

namespace oog {

class OpeningBook;

// Search scoring constants.
inline constexpr int MATE = 1'000'000'000;
inline constexpr int INF = 2'000'000'000;
inline constexpr int MAX_PLY = 256;
inline constexpr int MATE_THRESHOLD = MATE - 2 * MAX_PLY;

struct SearchLimits {
    std::uint64_t maxNodes = 0;  // 0 = unlimited
    int maxDepth = 0;            // 0 = unlimited (bounded by time/nodes)
    long long maxTimeMs = 0;     // 0 = unlimited
    bool playBook = false;       // if true, play the best book move directly
};

struct SearchResult {
    int bestMove = -1;          // square, or -1 if none / game already over
    int score = 0;              // from the mover's perspective
    std::uint64_t nodes = 0;
    int depth = 0;              // completed iterative-deepening depth
    bool forcedWin = false;     // bestMove came from the threat-space (VCF) precheck
    bool aborted = false;       // stopped early (time/node budget)
};

// Alpha-beta (PVS) search with iterative deepening, a bounded transposition
// table, threat-based move ordering, and a threat-space (VCF 算杀) precheck
// that finds forcing wins at the root.
//
// One Search instance owns one TT and per-search history; it is safe to have
// one instance per game table (no shared mutable state), so parallel games
// can run on independent Search objects.
class Search {
public:
    Search();

    SearchResult search(BitBoard& bb, int color, const SearchLimits& limits,
                        const OpeningBook* book = nullptr);

    // Threat-space search: is there a forced win for `attacker` (side to move)
    // within `maxAttacks` forcing four-moves?  Returns the winning first move,
    // or -1.  A move is "forcing" if it leaves the defender with exactly one
    // reply (blocking the four) until the attacker makes five.
    int vcfWin(BitBoard& bb, int attacker, int maxAttacks = 12);

    std::uint64_t nodes() const { return nodes_; }
    std::uint64_t ttHits() const { return tt_.hits(); }
    std::uint64_t ttProbes() const { return tt_.probes(); }
    void clearTT() { tt_.clear(); }

private:
    TT tt_;
    EvalState eval_;
    BitBoard* bb_ = nullptr;
    SearchLimits limits_;
    std::uint64_t nodes_ = 0;
    int rootColor_ = 0;
    int rootBestMove_ = -1;
    int lastSq_ = -1;    // last stone placed by the search (for win checks)
    int lastColor_ = 0;
    bool aborted_ = false;
    std::chrono::steady_clock::time_point t0_;
    std::vector<int> history_;  // [color * BitBoard::SIZE + sq]

    int vcfFirstMove_ = -1;

    // Root move-order bonus from the opening book (indexed by square).
    std::array<long long, BitBoard::SIZE> bookBonus_{};

    std::uint64_t keyWithSide(int color) const;
    bool outOfTime();
    void orderMoves(std::vector<int>& moves, int color, int ttMove, int depth,
                    int ply);
    int negamax(int color, int depth, int alpha, int beta, int ply);
    bool vcf(int attacker, int depth, int ply);
    // All empty squares where a stone of `color` would complete five.
    void collectWinningPoints(int color, std::vector<int>& out);
    // True if `color` can complete five with one move right now.
    bool hasImmediateWin(int color);
};

}  // namespace oog
