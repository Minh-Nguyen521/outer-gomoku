#!/usr/bin/env python3
"""Regenerate the Kaggle cells for MiniZero training from the current script.

Mirror of tools/make_colab_cells.py for Kaggle. Kaggle notebooks cannot fetch
this repo either, so tools/kaggle_minizero.sh is embedded as a base64'd gzip
payload in the generated cells. Regenerate after ANY edit to the script --
a stale copy ships an older engine setup.

Usage:
    tools/make_kaggle_cells.py            # regenerate into build/
    tools/make_kaggle_cells.py --check    # verify sync, non-zero exit if stale
"""

import argparse
import base64
import gzip
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "tools", "kaggle_minizero.sh")
OUTDIR = os.path.join(REPO, "build")
REMOTE = "/kaggle/working/kaggle_minizero.sh"

SETUP = """# ---- Cell 1: bootstrap + setup (run once per session) ----
# Writes tools/kaggle_minizero.sh into the VM, then builds MiniZero for Gomoku.
# Requires Internet = ON (Settings > Internet) for apt/pip/git.
#
# embedded kaggle_minizero.sh: {size} bytes, sha256 {sha}
%%bash
set -e
echo '{payload}' | base64 -d | gunzip > {remote}
chmod +x {remote}
REPO=/kaggle/working/oog bash {remote} setup
"""

SMOKE = """# ---- Cell 2: smoke test, 30 iterations ----
# Run this BEFORE the full 300. It proves the pipeline end to end.
%%bash
REPO=/kaggle/working/oog bash {remote} train 30 gmz
"""

FULL = """# ---- Cell 3: the real run (Olympiad recipe) ----
# 300 iterations of Gumbel MuZero. Kaggle caps a session at 12h; when it ends,
# Save Version (keeps /kaggle/working as output), then attach that output as an
# Input and re-run this cell to resume.
#
# Optional: set KAGGLE_DATASET=user/slug to auto-push checkpoints periodically.
%%bash
REPO=/kaggle/working/oog bash {remote} train 300 gmz
"""

MATCH = """# ---- Cell 4: play the trained model against a reference engine ----
%%bash
MODEL=$(ls -t /kaggle/working/minizero/gomoku_oo_15x15_gmz_run/model/*.pt | head -1)
REPO=/kaggle/working/oog bash {remote} match "$MODEL"
"""

FILES = {
    "kaggle_cell1_setup.txt": SETUP,
    "kaggle_cell2_smoketest.txt": SMOKE,
    "kaggle_cell3_full.txt": FULL,
    "kaggle_cell4_match.txt": MATCH,
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

    if cfg.check and stale:
        print("\nSTALE -- run tools/make_kaggle_cells.py to regenerate")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
