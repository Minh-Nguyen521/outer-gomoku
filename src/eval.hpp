#pragma once

#include <cstdint>

#include "bitboard.hpp"

namespace oog {

// Threat levels, ordered by strength (higher = more urgent).
enum class ThreatLevel : std::uint8_t {
    None = 0,
    ClosedTwo,
    OpenTwo,
    ClosedThree,
    OpenThree,
    ClosedFour,  // 冲四 (four with one winning point)
    OpenFour,    // 活四 (four with two winning points, unstoppable)
    Five,        // win
};

// Score contribution of a threat level.
int levelScore(ThreatLevel t);

// Classify a single line through a candidate move.
// own8/opp8: 8-bit masks of the cells at offsets -4,-3,-2,-1,+1,+2,+3,+4
// (bit k <-> offset k-4 for k in 0..3, offset k-3 for k in 4..7), where the
// center cell is treated as a stone of the "own" color just placed.
ThreatLevel classifyLine(std::uint8_t own8, std::uint8_t opp8);

// 16-bit line feature (own8<<8 | opp8) -> threat level.  OOGiveMeFive-style
// M-mask table, precomputed lazily over all 65536 states.
ThreatLevel classifyFeature(std::uint16_t feature);

// Static evaluation of the whole position from `mover`'s perspective
// (mover: 0 = Black, 1 = White).  Positive favors the mover; units are
// arbitrary centi-stone-like scores.
int evaluate(const BitBoard& bb, int mover);

// Threat score of placing `color` at `sq` (used for move ordering and threat
// search).  Combines the threats the move creates for `color` with the
// threats it pre-empts for the opponent.
int evaluateMove(const BitBoard& bb, int color, int sq);

// Incremental window-based evaluation.  Maintains exactly the same score as
// evaluate(), but recomputes only the 5-cell windows near a move, so a leaf
// evaluation is O(1) and move/undo maintenance is O(windows near the square).
//
// Usage: reset() on the root position, then call update(bb, sq) immediately
// after every BitBoard::place/remove on the path.
class EvalState {
public:
    void reset(const BitBoard& bb);
    void update(const BitBoard& bb, int sq);
    int value(int mover) const;  // mover: 0 = Black, 1 = White
    int sum(int color) const { return sum_[color]; }

private:
    int sum_[2] = {0, 0};
    std::vector<int> win_[2];  // per-window contribution, per color
};

}  // namespace oog
