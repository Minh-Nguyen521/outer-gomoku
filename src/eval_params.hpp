#pragma once

namespace oog {

// Runtime-tunable evaluation parameters.
//
// Every number in this project's evaluation was hand-picked and has never
// been fitted to anything -- PLAN.md Phase 2's "tune weights later" box is
// still open. Five independent measurements now say the evaluation is where
// our strength gap is (we lose 97/100 to rapfi with its network switched
// off; rapfi restricted to a 2-ply search still beats us 90/10; 16x more
// thinking time buys us only ~70 Elo). Fitting these is the highest-value
// work available, and it needs no neural training -- just an opponent we can
// trade games with, which calibration finally produced.
//
// The blocker was mechanical: the constants were `constexpr`, so trying a
// value meant a recompile, which makes any search over them impractical.
// They are now read once from the environment at first use.
//
// **Defaults are exactly the previous constants.** With no environment
// variables set, behaviour is bit-identical to before this file existed, so
// every measurement already recorded remains valid.
//
// Overrides (all optional, integers):
//   OOG_S_FIVE, OOG_S_OPEN_FOUR, OOG_S_CLOSED_FOUR,
//   OOG_S_OPEN_THREE, OOG_S_CLOSED_THREE,
//   OOG_S_OPEN_TWO, OOG_S_CLOSED_TWO,
//   OOG_DEF_NUM, OOG_DEF_DEN          (defence weighting, default 3/2)
struct EvalParams {
    int five = 1'000'000;
    int openFour = 100'000;
    int closedFour = 10'000;
    int openThree = 5'000;
    int closedThree = 1'000;
    int openTwo = 500;
    int closedTwo = 100;
    int defNum = 3;
    int defDen = 2;
};

// Read once at first call; cheap thereafter (returns a reference to a static).
const EvalParams& evalParams();

}  // namespace oog
