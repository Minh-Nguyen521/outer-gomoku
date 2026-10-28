#pragma once

#include <cstdint>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "bitboard.hpp"
#include "book.hpp"
#include "search.hpp"

namespace oog {

// Which search the protocol binary should play with. The Search* classes
// share an API by convention rather than by inheritance (each is deliberately
// standalone so the baseline stays untouched), so the brain holds a small
// type-erased handle instead of naming one of them directly.
enum class EngineKind {
    Baseline,   // Search           -- the original, unmodified
    V2,         // SearchV2         -- rapfi-style multi-signal LMR
    V3,         // SearchV3         -- V2 + futility/move-count/razoring-VCF
    Vct,        // SearchVct        -- V3 + Allis threat-space search (VCT)
    NullKiller, // SearchNullKiller -- V3 + null move + killer/counter-move
    ThreatCand, // SearchThreatCand -- V3 + threat-driven candidate generation
    V4,         // SearchV4         -- fast bitboard threat primitive
    V5,         // SearchV5         -- V4 + root safety screen + extensions
    EvalV2,     // SearchEvalV2     -- baseline search + fork-aware evaluation
    Q,          // SearchQ          -- EvalV2 + quiescence (VCF) at the horizon
};

// Parse an engine name ("search"/"baseline", "v4", "v5"), case-insensitive.
// Returns false if the name is unknown.
bool engineKindFromName(const std::string& name, EngineKind& out);
const char* engineKindName(EngineKind kind);

class BrainEngine {
public:
    virtual ~BrainEngine() = default;
    virtual SearchResult search(BitBoard& bb, int color,
                                const SearchLimits& limits,
                                const OpeningBook* book) = 0;
    virtual void clearTT() = 0;
};

std::unique_ptr<BrainEngine> makeBrainEngine(EngineKind kind);

// Gomocup / Piskvork protocol brain for Outer-Open Gomoku.
//
// Speaks the Gomocup protocol on stdin/stdout for the 15x15 board:
//   START, RECTSTART, RESTART, BACK, INFO, TURN, BEGIN, BOARD, ABOUT, END.
// Coordinates are the protocol's 0-based "x,y" (x = column, y = row).
//
// The manager/GUI is treated as the authority on the position: stones it sends
// are placed verbatim (occupied squares are still rejected).  The side to move
// is derived from the number of stones (parity), which stays correct even when
// BOARD lists stones in any order.  The brain answers TURN / BEGIN / BOARD with
// a search move and records it so later commands see a consistent game.
//
// One brain owns one Search (and its TT), so a separate process per game table
// gives the independent-instance guarantees the parallel runner relies on.
class PiskvorkBrain {
public:
    // Defaults to the baseline so existing callers and tests are unaffected.
    explicit PiskvorkBrain(EngineKind kind = EngineKind::Baseline);

    // Feed one protocol line.  `in` is the same input stream the caller reads
    // lines from, so multi-line commands (BOARD) can consume their payload.
    // Returns the reply the brain expects to print ("" = no reply needed).
    std::string handleLine(const std::string& line, std::istream& in);

    bool ended() const { return ended_; }
    bool started() const { return started_; }

    // Optional deterministic per-move node budget (0 = time only).
    void setNodeLimit(std::uint64_t n) { nodeLimit_ = n; }
    // Replace the opening book (empty result -> no book).  Called before the
    // first command; the protocol has no book command of its own.
    void loadBook(const std::string& path) { book_.loadFile(path); }

    // Testing hooks.
    int lastMove() const { return lastMove_; }
    int lastScore() const { return lastScore_; }
    int stones() const { return stones_; }
    EngineKind engineKind() const { return engineKind_; }

private:
    bool placeOpponent(int r, int c);
    bool placeRaw(int r, int c, int color);
    int computeMove();
    std::string replyMove(int mv);
    void reset();
    void undo(int n = 2);
    int sideToMove() const { return stones_ % 2; }
    long long moveBudgetMs() const;

    BitBoard bb_;
    std::vector<int> history_;  // protocol square, in move order
    int stones_ = 0;
    EngineKind engineKind_ = EngineKind::Baseline;
    std::unique_ptr<BrainEngine> engine_;
    OpeningBook book_;

    std::int64_t timeoutTurnMs_ = 0;   // per-turn limit (0 = not given)
    std::int64_t timeoutMatchMs_ = 0;  // whole-match limit (0 = not given)
    std::int64_t timeLeftMs_ = 0;      // last INFO time_left (ms)
    bool haveTimeoutTurn_ = false;
    bool haveTimeLeft_ = false;
    std::uint64_t nodeLimit_ = 0;  // optional --nodes override

    int lastMove_ = -1;
    int lastScore_ = 0;
    bool started_ = false;
    bool ended_ = false;
};

}  // namespace oog
