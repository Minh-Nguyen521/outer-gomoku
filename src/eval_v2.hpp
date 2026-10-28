#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "bitboard.hpp"
#include "eval.hpp"

namespace oog {

// EvalStateV2 = the existing window-based evaluation, plus the term it is
// missing: **fork (double threat) detection**.
//
// ## Why
//
// Go-Moku games are decided by forks. Two open threes, or a four plus a
// three, cannot both be answered -- the opponent blocks one and loses to the
// other. The existing `evaluate()` sums window scores linearly, so it values
// a double four at S_CLOSED_FOUR + S_CLOSED_FOUR = 20,000 while a single open
// four scores S_OPEN_FOUR = 100,000. Both are winning; the eval rates the
// fork at a fifth of the single threat.
//
// The practical consequence was measured: strip search depth (by running
// games under CPU contention) and our engines collapse while rapfi barely
// notices. A good evaluation still finds reasonable moves shallow; ours has
// nothing to fall back on, because the thing that actually wins the game is
// invisible to it and only the search ever finds it.
//
// ## What this adds
//
// Threat *counts* per colour, maintained per line so they can be updated
// incrementally, and a superlinear bonus when a colour holds more than one
// simultaneous threat:
//
//   - `fourThreats` -- distinct four threats, deduped **by the span of the
//     four stones**, not by completion square. An open four `_XXXX_` has two
//     completions but is *one* threat and is already scored S_OPEN_FOUR by
//     the linear term; counting completions made every open four look like a
//     fork and measured 14 points worse than baseline against rapfi. Counted
//     from 5-windows, so gapped fours (`XX_XX`) are included.
//   - `openThrees` -- threats that would become an *open four* next move.
//     Two or more is a double three.
//
// Counts are kept per (direction, line) and only the four lines through a
// moved square are recomputed, so an update stays bounded the same way
// `EvalState`'s window update does.
//
// ## Two terms the old evaluation simply did not have
//
//   - **Tempo.** `evaluate()` has no side-to-move term at all: it scores the
//     same position identically whoever is to move. In Go-Moku the move is
//     worth a great deal -- a player to move with an open three is winning,
//     the same position with the opponent to move is not.
//   - **Position.** No centre bias, no edge awareness. That is a specific
//     gap for Outer-Open, where Black's first stone is *forced* onto the
//     outer ring. Every engine here scores far worse as Black (baseline
//     8/50 vs 14/50 as White) and nothing in the evaluation models why.
//
// Both are tunable (`OOG_TEMPO`, `OOG_POS_W`) alongside the weights.
//
// ## What this deliberately does not change
//
// The linear window term is reused verbatim from `EvalState` rather than
// reimplemented -- it is already tested and correct, and keeping it means any
// measured difference is attributable to the fork term alone. The seven score
// constants are untouched here too; fitting those is a separate job, and
// doing both at once would confound them.
class EvalStateV2 {
public:
    void reset(const BitBoard& bb);
    void update(const BitBoard& bb, int sq);
    int value(int mover) const;  // mover: 0 = Black, 1 = White

    // Threat counts for `color`, for testing and for callers that want the
    // raw signal rather than the blended score.
    int fourThreats(int color) const { return fourThreats_[color]; }
    int openThrees(int color) const { return openThrees_[color]; }

    // The fork bonus alone, from `color`'s point of view (no opponent term).
    int forkBonus(int color) const;

    // Positional (centre-proximity) sum for `color`, maintained incrementally.
    int positional(int color) const { return positional_[color]; }

private:
    EvalState base_;

    // Per (direction, line) counts, so an update only recomputes the four
    // lines through the moved square.
    static constexpr int NDIR = 4;
    std::vector<std::array<int, 2>> lineFour_[NDIR];
    std::vector<std::array<int, 2>> lineOpen3_[NDIR];
    int fourThreats_[2] = {0, 0};
    int openThrees_[2] = {0, 0};
    int positional_[2] = {0, 0};
    // What the positional sum currently credits each square to: -1 empty,
    // else the colour. update() runs after both place() and remove(), and a
    // removal does not say which colour left, so the previous state must be
    // recorded rather than inferred.
    std::array<signed char, BitBoard::SIZE> posOwner_{};

    void recomputeLine(const BitBoard& bb, int dir, int lineIdx);
};

// Non-incremental whole-board evaluation with the fork term, matching
// EvalStateV2::value() exactly. Mirrors `evaluate()`'s role.
int evaluateV2(const BitBoard& bb, int mover);

}  // namespace oog
