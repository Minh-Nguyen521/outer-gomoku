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
REMOTE = "/kaggle/working/oog/tools/kaggle_minizero.sh"

# The setup cell clones this repo (PRIVATE) instead of embedding the script
# as base64, so the notebook always runs the current code from git.
# Fill GITHUB_TOKEN before generating cells; it ends up in the generated
# notebook -- keep both the repo and the notebook PRIVATE, and use a
# fine-grained token scoped to Contents: Read-only for this repo only.
# GITHUB_TOKEN is intentionally NOT committed: it is read from the
# environment (or tools/.github_token, which is gitignored).
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or (
    open(os.path.join(REPO, "tools", ".github_token")).read().strip()
    if os.path.exists(os.path.join(REPO, "tools", ".github_token")) else "")
if not GITHUB_TOKEN:
    sys.exit("set GITHUB_TOKEN or create tools/.github_token before generating")
REPO_URL = f"https://{GITHUB_TOKEN}@github.com/Minh-Nguyen521/outer-gomoku.git"

SETUP = """%%bash
# ---- Cell 1: clone repo + setup (run once per session) ----
# Clones the private repo with an embedded read-only PAT, then builds MiniZero.
# Requires Internet = ON (Settings > Internet) for apt/pip/git.
# NOTE: %%bash MUST stay on the very first line -- any line before it
# (even a comment) turns the whole cell into Python and fails with
# "SyntaxError: invalid syntax".
set -e
# Force re-clone to pick up the latest fixes; cached copies may be stale.
rm -rf /kaggle/working/oog
git clone --depth 1 '{repo_url}' /kaggle/working/oog
REPO=/kaggle/working/oog bash /kaggle/working/oog/tools/kaggle_minizero.sh setup
"""

SMOKE = """%%bash
# ---- Cell 2: smoke test, 30 iterations ----
# Run this BEFORE the full 300. It proves the pipeline end to end.
REPO=/kaggle/working/oog bash {remote} train 30 gmz
"""

FULL = """%%bash
# ---- Cell 3: the real run (Olympiad recipe) ----
# 300 iterations of Gumbel MuZero. Kaggle caps a session at 12h; when it ends,
# Save Version (keeps /kaggle/working as output), then attach that output as an
# Input and re-run this cell to resume.
#
# Optional: set KAGGLE_DATASET=user/slug to auto-push checkpoints periodically.
REPO=/kaggle/working/oog bash {remote} train 300 gmz
"""

MATCH = """%%bash
# ---- Cell 4: play the trained model against a reference engine ----

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
    sha = hashlib.sha256(raw).hexdigest()
    out = {}
    for name, tmpl in FILES.items():
        out[name] = tmpl.format(repo_url=REPO_URL, remote=REMOTE,
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
