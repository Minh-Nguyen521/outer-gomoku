#!/usr/bin/env python3
"""Regenerate the Colab cells for MiniZero training from the current script.

WHY THIS EXISTS. `build/colab_bootstrap.txt` embeds `tools/colab_minizero.sh`
as a base64'd gzip payload, because Colab has no access to this repo. That
embedding is a COPY, and it went stale: the bootstrap carried a 10,353-byte
script while the real one had grown to 20,307 bytes. Every fix added after the
lost training run -- Drive mount enforcement, the 60s checkpoint mirror, the
`Training.log` dump on exit -- was missing from the payload. Running those
cells would have reproduced the original failure with no diagnostics, which is
precisely the failure those fixes exist to prevent.

A hand-maintained copy of a file that changes is a bug waiting to happen. This
regenerates all of it from the source of truth, and prints the hash so the
sync can be verified rather than assumed.

Usage:
    tools/make_colab_cells.py            # regenerate into build/
    tools/make_colab_cells.py --check    # verify sync, non-zero exit if stale
"""

import argparse
import base64
import gzip
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "tools", "colab_minizero.sh")
OUTDIR = os.path.join(REPO, "build")
REMOTE = "/content/colab_minizero.sh"

BOOTSTRAP = """# ---- Cell 1: bootstrap + setup (run once per VM) ----
# Writes tools/colab_minizero.sh into the VM, then builds MiniZero for Gomoku.
#
# The script is embedded rather than fetched because this repo is not public.
# Regenerate with tools/make_colab_cells.py after ANY edit to the script --
# a stale copy here silently ships an older engine setup.
#
# embedded colab_minizero.sh: {size} bytes, sha256 {sha}
#
# Mount Drive FIRST (cell 0) -- train refuses to start without it, on purpose:
# a Colab VM can be recycled at any time and an unmirrored run is lost.
%%bash
set -e
echo '{payload}' | base64 -d | gunzip > {remote}
chmod +x {remote}
REPO=/content/oog bash {remote} setup
"""

SMOKE = """# ---- Cell 2: smoke test, 30 iterations ----
# Run this BEFORE the full 300. It proves the pipeline end to end and yields a
# weak model. A pipeline bug found at iteration 280 is expensive.
#
# The previous attempt got through self-play (2000 games, 112.6 plies average,
# model dir named gomoku_oo_15x15_gmz_1bx256_n16 -- so the rule and board size
# took effect) and then reported only "Training failed to complete". That line
# is quick-run.sh's own grep check, NOT an error from the trainer: it greps
# Training.log for "[Optimization] Finished." and prints that if absent.
#
# The real cause is in $train_dir/Training.log. This script now tails it and
# greps the [Optimization] / Traceback / Error lines on exit, success or
# failure, so the next run diagnoses itself.
#
# The learner is the part that failed, and it has a hard dependency worth
# knowing: train.py does
#     __import__(f'build.{{game_type}}', ..., ['minizero_py'])
# so it needs build/gomoku/minizero_py*.so to exist and be importable, built
# WITH -DGOMOKU. Self-play is pure C++ and runs regardless -- which is exactly
# the shape of "self-play fine, optimization failed".
%%bash
REPO=/content/oog bash {remote} train 30 gmz
"""

FULL = """# ---- Cell 3: the real run (Olympiad recipe) ----
# 300 iterations of Gumbel MuZero, which is what MiniZero24 used to win gold.
# Colab WILL disconnect before this finishes; the script mirrors checkpoints
# to Drive every 60s and restores them on restart, so re-running this cell
# after a reconnect resumes rather than starting over.
#
# Only run this once the smoke test has completed cleanly.
%%bash
REPO=/content/oog bash {remote} train 300 gmz
"""

MOUNT = """# ---- Cell 0: mount Drive (required) ----
from google.colab import drive
drive.mount('/content/drive')
"""

MATCH = """# ---- Cell 4: play the trained model against a reference engine ----
# MiniZero's console mode speaks GTP, so the existing bridge drives it
# unchanged -- no MiniZero-specific adapter is needed.
%%bash
REPO=/content/oog bash {remote} match
"""

FILES = {
    "colab_cell0_mount.txt": MOUNT,
    "colab_bootstrap.txt": BOOTSTRAP,
    "colab_cell2_smoketest.txt": SMOKE,
    "colab_cell3_full.txt": FULL,
    "colab_cell4_match.txt": MATCH,
}


def build():
    raw = open(SCRIPT, "rb").read()
    payload = base64.b64encode(gzip.compress(raw, 9)).decode()
    sha = hashlib.sha256(raw).hexdigest()
    out = {}
    for name, tmpl in FILES.items():
        out[name] = tmpl.format(payload=payload, remote=REMOTE,
                                size=len(raw), sha=sha[:16])
    return out, len(raw), sha


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="verify the generated cells match the current script")
    cfg = p.parse_args(argv)

    out, size, sha = build()
    print(f"source: {SCRIPT}")
    print(f"        {size} bytes, sha256 {sha[:16]}")

    stale = False
    for name, text in out.items():
        path = os.path.join(OUTDIR, name)
        old = open(path).read() if os.path.exists(path) else None
        if old == text:
            print(f"  = {name}")
            continue
        stale = True
        if cfg.check:
            print(f"  ! {name}  STALE")
        else:
            os.makedirs(OUTDIR, exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
            print(f"  + {name}  ({'updated' if old else 'created'})")

    # The old cell1 name is superseded by colab_bootstrap.txt; flag rather
    # than delete, so nothing is removed behind the user's back.
    legacy = os.path.join(OUTDIR, "colab_cell1_setup.txt")
    if os.path.exists(legacy):
        print(f"\nnote: {legacy} is superseded by colab_bootstrap.txt "
              f"(both wrote the setup cell); delete it to avoid confusion.")

    if cfg.check and stale:
        print("\nSTALE -- run tools/make_colab_cells.py to regenerate")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
