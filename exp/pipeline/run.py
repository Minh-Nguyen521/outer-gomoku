#!/usr/bin/env python3
"""Chain the three training methods into one pipeline.

    stage 1: distil   -- KataGomo teacher labels over harvest positions
                         (tools/label_positions.py + tools/train_policy.py)
    stage 2: exit     -- Expert Iteration seeded from stage 1's net
                         (exp/02_exit_iter, --init)
    stage 3: gumbel   -- Gumbel AZ self-play warm-started from stage 2
                         (exp/04_gumbel, --size 15 --init)

Every stage is optional and resumable: if its output checkpoint already
exists, the stage is skipped (delete the .pt to redo it). Stage runtime
estimates live in exp/README.md.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def run(cmd, desc):
    print(f"\n===== {desc} =====", flush=True)
    print("$ " + " ".join(cmd[:2]) + " ...", flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(f"!! {desc} failed with exit {r.returncode}", file=sys.stderr)
        return False
    return True


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/harvest_merged.jsonl")
    ap.add_argument("--stage", default="123",
                    help="which stages to run, e.g. 13 skips ExIt")
    ap.add_argument("--label-limit", type=int, default=93000)
    ap.add_argument("--exit-iters", type=int, default=10)
    ap.add_argument("--exit-games", type=int, default=100)
    ap.add_argument("--gumbel-iters", type=int, default=20)
    ap.add_argument("--gumbel-games", type=int, default=100)
    ap.add_argument("--gumbel-k", type=int, default=16)
    cfg = ap.parse_args(argv)

    stages = set(cfg.stage)
    distill_pt = os.path.join(HERE, "distilled.pt")
    exit_pt = os.path.join(HERE, "exit_net.pt")
    gumbel_pt = os.path.join(HERE, "gumbel_net.pt")
    positions = os.path.join(HERE, "positions.jsonl")
    labelled = os.path.join(HERE, "labelled.jsonl")

    # ---- stage 1: distil -------------------------------------------------
    if "1" in stages and not os.path.exists(distill_pt):
        if os.path.exists(positions) and os.path.exists(labelled) \
                and _count_lines(labelled) >= _count_lines(positions):
            print(f"[skip] 1a/1b: {labelled} is complete (re-using labels)")
            ok = True
        else:
            ok = run(["python3", os.path.join(HERE, "..", "03_distill",
                                              "sample_positions.py"),
                      "--data", cfg.data, "--out", positions,
                      "--limit", str(cfg.label_limit)], "1a sample positions")
            ok = ok and run(["python3", "tools/label_positions.py",
                             "--katago", "third_party/KataGomo/cpp/build2/katago",
                             "--model", "third_party/kgweights/tworule_20x_b18.bin.gz",
                             "--config", "third_party/KataGomo/cpp/build2/oog.cfg",
                             "--in", positions, "--out", labelled],
                            "1b KataGomo labels")
        ok = ok and run(["python3", "tools/train_policy.py",
                         "--data", labelled, "--epochs", "20",
                         "--out", distill_pt], "1c distil")
        if not ok and not os.path.exists(distill_pt):
            return 1
    elif "1" in stages:
        print("[skip] stage 1: distilled.pt exists")

    # ---- stage 2: expert iteration ---------------------------------------
    if "2" in stages:
        seed = distill_pt if os.path.exists(distill_pt) else \
            "exp/01_supervised/supervised.pt"
        if not os.path.exists(seed):
            print(f"[warn] no seed net at {seed}; run method 1 first",
                  file=sys.stderr)
            return 1
        ok = run(["python3", os.path.join(HERE, "..", "02_exit_iter",
                                          "train.py"),
                  "--init", seed,
                  "--seed-data", cfg.data,
                  "--iterations", str(cfg.exit_iters),
                  "--games", str(cfg.exit_games)],
                 f"2 ExIt (seed {os.path.basename(seed)})")
        if ok:
            import shutil
            shutil.copy(os.path.join(HERE, "..", "02_exit_iter",
                                     "exit_net.pt"), exit_pt)
        else:
            return 1
    elif "2" in stages:
        print("[skip] stage 2")

    # ---- stage 3: gumbel az ----------------------------------------------
    if "3" in stages:
        seed = exit_pt if os.path.exists(exit_pt) else distill_pt
        if not os.path.exists(seed):
            print(f"[warn] no seed net for gumbel; stages 1/2 produced "
                  "nothing", file=sys.stderr)
            return 1
        ok = run(["python3", os.path.join(HERE, "..", "04_gumbel",
                                          "train.py"),
                  "--size", "15",
                  "--init", seed,
                  "--prefill", cfg.data,
                  "--iterations", str(cfg.gumbel_iters),
                  "--games", str(cfg.gumbel_games),
                  "--k", str(cfg.gumbel_k),
                  "--out", gumbel_pt],
                 f"3 Gumbel AZ (seed {os.path.basename(seed)})")
        if not ok:
            return 1

    print("\npipeline done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
