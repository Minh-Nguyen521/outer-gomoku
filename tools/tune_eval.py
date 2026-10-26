#!/usr/bin/env python3
"""SPSA tuning of the evaluation weights against a calibrated opponent.

Why this exists
---------------
Every number in this project's evaluation was picked by hand and has never
been fitted to data (PLAN.md Phase 2, "tune weights later" -- still open).
Five independent measurements agree that the evaluation is the strength gap:

  * we lose 97/100 to rapfi with its neural network switched off entirely;
  * rapfi restricted to a *2-ply* search still beats us 90/10;
  * 16x more thinking time buys our engine only ~70 Elo (chess engines get
    ~250 for the same);
  * under CPU contention we collapse 22% -> 1% while rapfi barely moves;
  * six rounds of *search* work failed to beat the untouched baseline
    externally.

This needs no neural training -- it is ordinary numerical optimisation, the
same level of work Embryo (Stockfish-derived) and rapfi's classical evaluator
both already have and we do not.

Why it was impossible until now
-------------------------------
Two blockers, both removed:

1. No measurable opponent. Everything beat us 3-25%, where a 30-Elo change is
   invisible. Calibration found Embryo handicapped to 15 ms/move puts our
   baseline near 40% -- see `docs/search-comparison.md`.
2. The weights were `constexpr`, so trying a value meant recompiling. They
   now read from the environment (src/eval_params.hpp), defaulting to the
   old constants so nothing already measured is invalidated.

Method
------
SPSA (Simultaneous Perturbation Stochastic Approximation): perturb *all*
parameters at once with random signs, play a match at (theta + delta) and at
(theta - delta), and step towards whichever scored better. Two matches per
iteration regardless of dimensionality, which is what makes it practical when
each evaluation costs a match rather than a function call.

Parameters are tuned in log space. The weights span 100 to 1,000,000, so an
additive step meaningful for S_CLOSED_TWO is a rounding error for S_FIVE.

Noise warning
-------------
A match of N games has a standard error of ~sqrt(0.25/N). At 40 games that is
~8 percentage points -- larger than most real improvements. SPSA tolerates
noisy gradients by averaging over many iterations, but do not read any single
iteration as signal, and always re-validate a tuned result with a longer
match than the tuner used.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (env var, default, lower bound, upper bound)
PARAMS = {
    "open_four":   ("OOG_S_OPEN_FOUR",   100_000, 20_000, 500_000),
    "closed_four": ("OOG_S_CLOSED_FOUR",  10_000,  2_000, 100_000),
    "open_three":  ("OOG_S_OPEN_THREE",    5_000,  1_000,  60_000),
    "closed_three":("OOG_S_CLOSED_THREE",  1_000,    100,  20_000),
    "open_two":    ("OOG_S_OPEN_TWO",        500,     20,   8_000),
    "closed_two":  ("OOG_S_CLOSED_TWO",      100,      5,   3_000),
    # EvalStateV2's additions. These are the newest hand-picked numbers in
    # the project and therefore the least likely to be near optimal -- the
    # whole point of tuning is that guessed constants are the problem, and
    # adding three more guesses (tempo 250, pos 6, fork 90k/60k/30k) did not
    # change that. `evalv2` measured +85 Elo over baseline with them unfitted.
    "tempo":       ("OOG_TEMPO",             250,      0,   4_000),
    "pos_w":       ("OOG_POS_W",               6,      0,     120),
    "fork_44":     ("OOG_FORK_44",        90_000, 10_000, 400_000),
    "fork_43":     ("OOG_FORK_43",        60_000,  5_000, 300_000),
    "fork_33":     ("OOG_FORK_33",        30_000,  2_000, 200_000),
}
# S_FIVE is deliberately not tuned: it is the terminal score and its exact
# magnitude only has to dominate everything else, which it does.

DEF_NUM = ("OOG_DEF_NUM", 3, 1, 8)   # defence weighting numerator (den fixed at 2)


def winrate(theta, cfg, seed):
    """Play a match with these weights; return score in [0,1] for our side."""
    env = dict(os.environ)
    for name, (var, *_rest) in PARAMS.items():
        env[var] = str(int(round(theta[name])))
    env[DEF_NUM[0]] = str(int(round(theta["def_num"])))
    env["OOG_DEF_DEN"] = "2"

    cmd = [
        sys.executable, os.path.join(REPO, "tools", "match.py"),
        "--games", str(cfg.games), "--parallel", "1",
        "--timeout-turn", str(cfg.our_ms),
        "--timeout-turn-b", str(cfg.opp_ms),
        "--timeout-match", "120000",
        "--seed", str(seed),
        "--engine-a", f"./build/release/bin/oog_piskvork --engine {cfg.engine}",
        "--engine-b", cfg.opponent,
    ]
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                         text=True).stdout
    m = re.search(r"A winrate:\s*([0-9.]+)%", out)
    if not m:
        print("  !! could not parse match output:", out[-300:], flush=True)
        return 0.0
    return float(m.group(1)) / 100.0


def clamp(theta):
    for name, (_var, _d, lo, hi) in PARAMS.items():
        theta[name] = min(hi, max(lo, theta[name]))
    theta["def_num"] = min(DEF_NUM[3], max(DEF_NUM[2], theta["def_num"]))
    return theta


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--games", type=int, default=40,
                   help="games per match; noise ~ sqrt(0.25/games)")
    p.add_argument("--our-ms", type=int, default=200)
    p.add_argument("--opp-ms", type=int, default=15,
                   help="opponent handicap; 15ms puts baseline near 40%%")
    p.add_argument("--opponent", default="third_party/embryo/embryo-run")
    p.add_argument("--engine", default="evalv2",
                   help="engine to tune; evalv2 is the current best (+85 Elo)")
    p.add_argument("--out", default=os.path.join(REPO, "tuned_eval.json"))
    p.add_argument("--seed", type=int, default=1)
    cfg = p.parse_args(argv)

    rng = random.Random(cfg.seed)
    theta = {n: float(d) for n, (_v, d, _lo, _hi) in PARAMS.items()}
    theta["def_num"] = float(DEF_NUM[1])
    keys = list(theta.keys())

    base = winrate(theta, cfg, cfg.seed)
    print(f"baseline winrate with default weights: {base*100:.1f}%", flush=True)
    best = dict(theta)
    best_score = base

    for k in range(1, cfg.iterations + 1):
        # SPSA gains. Decayed so early steps explore and later ones settle.
        a = 0.20 / (k ** 0.602)
        c = 0.12 / (k ** 0.101)

        signs = {n: (1 if rng.random() < 0.5 else -1) for n in keys}
        # Multiplicative perturbation: parameters span 100..1e6, so a step
        # that matters for closed_two would be noise for open_four.
        # Log-space steps, with a floor so parameters whose lower bound is 0
        # (tempo, pos_w) can still move -- exp() can never leave zero.
        def perturb(sign):
            out = {}
            for n in keys:
                base_v = max(theta[n], 1.0)
                out[n] = base_v * math.exp(c * sign * signs[n])
            return clamp(out)
        plus = perturb(+1)
        minus = perturb(-1)

        sp = winrate(plus, cfg, cfg.seed + 1000 * k)
        sm = winrate(minus, cfg, cfg.seed + 1000 * k + 1)
        grad = (sp - sm) / (2 * c)

        theta = clamp({n: max(theta[n], 1.0) * math.exp(a * grad * signs[n])
                       for n in keys})

        cur = max(sp, sm)
        marker = ""
        if cur > best_score:
            best_score = cur
            best = dict(plus if sp >= sm else minus)
            marker = "  <-- best so far"
        print(f"iter {k:3d}  +{sp*100:5.1f}%  -{sm*100:5.1f}%  "
              f"grad={grad:+7.3f}{marker}", flush=True)

        with open(cfg.out, "w") as f:
            json.dump({"best_winrate": best_score,
                       "best": {n: int(round(v)) for n, v in best.items()},
                       "current": {n: int(round(v)) for n, v in theta.items()},
                       "baseline_winrate": base,
                       "iterations_done": k,
                       "games_per_match": cfg.games}, f, indent=2)

    print(f"\nbest {best_score*100:.1f}% (baseline {base*100:.1f}%)")
    print("best weights:", {n: int(round(v)) for n, v in best.items()})
    print(f"written to {cfg.out}")
    print("\nRe-validate with a longer match before trusting this: at "
          f"{cfg.games} games the standard error is "
          f"~{math.sqrt(0.25/cfg.games)*100:.1f} points.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
