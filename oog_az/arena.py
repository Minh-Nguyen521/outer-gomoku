"""Measure a trained net against the EXTERNAL engine ladder.

WHY THIS EXISTS, and why it is not optional. train.py already gates each
candidate against the previous network. That catches regressions but cannot
catch shared blind spots, and this project has the definitive measurement of
why that matters: SearchV4 won all EIGHT of its self-play pairings at 80.0%,
then scored BELOW the untouched baseline against an outside opponent
(docs/search-comparison.md Round 7). Six rounds of work, validated only
against engines sharing its own assumptions, produced no real improvement.

The rule that follows: anchor externally before trusting any self-play delta.

THE GRADED LADDER. third_party/ holds engines spanning roughly 1450-2700 on
Gomocup's Freestyle15 table, so a run can be tracked against real strength
rather than only against its own history. Start at the bottom -- a network
that cannot beat Chis will learn nothing from being swept by KataGomo, and
n=20 only resolves decisive differences anyway.

READING THE RESULT. The `win types` line matters more than the score. A 0-20
with no five-in-a-row is a BROKEN INSTRUMENT, not a weak player -- that has
held on seven separate occasions in this repo, including three caused by
giving an engine less wall-clock than it needs for one move. This module
refuses to report a score when no game was decided by a real five.

MEASUREMENT HYGIENE, inherited from docs/engine-ladder.md:
  * --parallel 1 always. Wall-clock under concurrency is invalid: the same
    matchup measured 22.0% serial and 1.0% at 4 workers, with no forfeits --
    both engines simply searched shallower.
  * --paired-openings, so both sides face each start from both colours.
  * --rule 0 (freestyle). rapfi maps this field onto its own enum, so sending
    2 makes it load renju weights and play a different game.
  * --timeout-match left unset; sending it makes rapfi ration itself to ~45%
    of its per-turn allowance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Graded opponents, weakest first, with Gomocup Freestyle15 ratings for
# context. Ratings are indicative: docs/engine-ladder.md measures a different
# order locally, and the disagreement is itself unresolved.
LADDER = [
    ("chis",      1453, "third_party/Chis/Chis2.0/pbrain-chis"),
    ("pela",      1549, "third_party/wine-engine pela pbrain-pela.exe"),
    ("carbon",    1624, "third_party/Carbon/pbrain-carbon"),
    ("wine",      1853, "third_party/wine-src/pbrain-wine"),
    ("slowrenju", 1816, "third_party/SlowRenju/pbrain-slowrenju"),
    ("rapfi8t",   2716, "third_party/rapfi-8t/pbrain-rapfi"),
]


def run_match(engine_a: str, engine_b: str, games: int, timeout_turn: int,
              live_file: str | None = None) -> dict:
    """Play a match through tools/match.py and parse its summary."""
    cmd = [
        sys.executable, os.path.join(REPO, "tools", "match.py"),
        "--games", str(games),
        "--parallel", "1",               # never raise this; see module docstring
        "--paired-openings",
        "--rule", "0",
        "--timeout-turn", str(timeout_turn),
        "--engine-a", engine_a,
        "--engine-b", engine_b,
    ]
    if live_file:
        cmd += ["--live-file", live_file]

    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    out = proc.stdout

    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, out)
        return cast(m.group(1)) if m else default

    win_types = re.search(r"win types:\s*(.+)", out)
    win_types = win_types.group(1).strip() if win_types else ""
    fives = [int(n) for n in re.findall(r"(\d+)x five", win_types)]

    return {
        "winrate": grab(r"A winrate:\s*([0-9.]+)%"),
        "elo": grab(r"Elo difference \(A - B\):\s*([+-]?[0-9.]+)"),
        "a_wins": grab(r"A wins:\s*(\d+)", int),
        "b_wins": grab(r"B wins:\s*(\d+)", int),
        "draws": grab(r"draws:\s*(\d+)", int),
        "win_types": win_types,
        "clean": bool(fives) and sum(fives) > 0,
        "overline_warning": "OVERLINE" in win_types and
                            bool(re.search(r"[1-9]\d*x OVERLINE", win_types)),
        "stderr_tail": proc.stderr[-400:] if proc.returncode else "",
        "returncode": proc.returncode,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="oog_az/runs/dev/best.pt")
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--opponent", default="chis",
                   help="ladder name, or a full engine command")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--probe-games", type=int, default=4,
                   help="short probe first; 0 disables")
    p.add_argument("--timeout-turn", type=int, default=10000)
    p.add_argument("--out", default=None, help="append the result as JSONL")
    cfg = p.parse_args(argv)

    named = {name: spec for name, _rating, spec in LADDER}
    opponent = named.get(cfg.opponent, cfg.opponent)
    ours = (f"{sys.executable} {os.path.join(REPO, 'oog_az', 'player.py')} "
            f"--model {cfg.model} --simulations {cfg.simulations}")

    print(f"ours     : {ours}")
    print(f"opponent : {opponent}", flush=True)

    # Probe first. A 20-game run against a misconfigured opponent costs hours
    # and yields a number that looks like a strength result and is not.
    if cfg.probe_games:
        probe = run_match(ours, opponent, cfg.probe_games, cfg.timeout_turn)
        print(f"probe ({cfg.probe_games} games): {probe['win_types'] or 'no summary'}",
              flush=True)
        if not probe["clean"]:
            print("PROBE FAILED: no game decided by a real five. This is a "
                  "setup fault, not a strength result -- do not record it.")
            if probe["stderr_tail"]:
                print(probe["stderr_tail"])
            return 1

    res = run_match(ours, opponent, cfg.games, cfg.timeout_turn)
    if not res["clean"]:
        print("RESULT REJECTED: no five-in-a-row in the whole match.")
        return 1

    print(f"\nscore   : {res['a_wins']}-{res['b_wins']}-{res['draws']} "
          f"({res['winrate']}%)")
    print(f"elo     : {res['elo']:+.1f}" if res["elo"] is not None else "elo : n/a")
    print(f"types   : {res['win_types']}")
    if res["overline_warning"]:
        print("WARNING: overlines present. If they are one-sided, the two "
              "engines are playing different rules -- check ABOUT before "
              "recording this row (that is how Embryo's rows were withdrawn).")

    if cfg.out:
        with open(cfg.out, "a") as fh:
            fh.write(json.dumps({"stage": "arena", "model": cfg.model,
                                 "opponent": cfg.opponent, **res}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
