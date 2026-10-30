#!/usr/bin/env python3
"""Generate a Kaggle notebook for harvesting Outer-Open Gomoku data.

Downloads the top-5 CPU engines from Gomocup (VIBEFIVE, AlphaGomoku, SlowRenju,
RAPFI25, Yixin2018 -- unified ranking positions 2-6 in docs/engine-ladder.md,
minus GPU-bound KataGomo), then runs tools/harvest_games.py round-robin with
all 104 ring openings.

Kaggle sessions cap at 12h and /kaggle/working persists as notebook output
when committed. The harvest writes incrementally and flushes every game, so
an expired session still keeps everything played so far.

Usage:
    python tools/make_kaggle_harvest_notebook.py
    # -> build/kaggle_harvest.ipynb ; upload, Internet = ON, Run All.
"""

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "build")

# Verified against https://gomocup.org/download-gomoku-ai/ (2026-10-30).
# Each zip is unpacked into third_party/<dir> and the pbrain executable is
# located with the same find logic as tools/colab_jax.sh: never hardcode the
# exe path, archives vary in internal layout.
ENGINE_ZIPS = [
    ("VIBEFIVE", "vibefive"),
    ("ALPHAGOMOKU.MK26", "agmk26"),
    ("SLOWRENJU20", "slowrenju20"),
    ("RAPFI25", "rapfi25"),
    ("YIXIN18", "yixin18"),
]

SETUP_CELL = """\
%%bash
# ---- Cell 1: setup: repo, wine, engines (run once) ----
# Requires Internet = ON (Settings > Internet). CPU-only is fine -- the T4
# accelerator can be turned OFF, saving GPU quota for MiniZero training.
set -e
REPO=/kaggle/working/oog
DL=http://download.gomocup.com/ai

rm -rf "$REPO"
# On Kaggle (Linux), use oauth2 as username for PAT auth
GIT_TERMINAL_PROMPT=0 git clone --depth 1 \\
    https://oauth2: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@github.com/Minh-Nguyen521/Outer-Gomoku.git \\
    "$REPO"
rm -rf "$REPO/.git"   # drop the token from the VM right after cloning

# Wine from Ubuntu's repo is enough for these engines (no onnxruntime.dll);
# colab_jax.sh installs WineHQ only because JAX needs wine >= 8.
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \\
    wine unzip curl > /dev/null 2>&1
echo "wine: $(wine --version 2>/dev/null || echo MISSING)"

cd "$REPO/third_party"
declare -A ZIPS=(
  [vibefive]=VIBEFIVE [agmk26]=ALPHAGOMOKU.MK26 [slowrenju20]=SLOWRENJU20
  [rapfi25]=RAPFI25 [yixin18]=YIXIN18
)
for dir in "${!ZIPS[@]}"; do
  z="${ZIPS[$dir]}"
  echo "== $z -> $dir =="
  curl -sSL --max-time 600 -o "/tmp/$z.zip" "$DL/$z.zip"
  mkdir -p "$dir" && unzip -oq "/tmp/$z.zip" -d "$dir"
  # Same rule as colab_jax.sh: locate the pbrain exe, never assume the path.
  exe="$(find "$dir" -maxdepth 3 -iname 'pbrain-*' |
         grep -viE 'unins|setup|vcredist|\\.so' | head -n1)"
  if [ -n "$exe" ]; then
    chmod +x "$exe" 2>/dev/null || true
    echo "   exe: $exe ($(du -sh "$dir" | cut -f1))"
  else
    echo "   !! no pbrain found -- check: ls -R $dir | head"
  fi
done"""

SMOKE_CELL = """\
%%bash
# ---- Cell 2: smoke test, 2 games (proves every engine starts) ----
set -e
REPO=/kaggle/working/oog
cd "$REPO"
find_eng() {  # pbrain path for an engine dir, Windows exes get a wine prefix
  local exe
  exe="$(find "third_party/$1" -maxdepth 3 -iname 'pbrain-*' |
         grep -viE 'unins|setup|vcredist|\\.so' | head -n1)"
  case "$exe" in
    *.exe) echo "wine $REPO/$exe" ;;
    *)     chmod +x "$exe" 2>/dev/null; echo "$REPO/$exe" ;;
  esac
}
V=$(find_eng vibefive)
A=$(find_eng agmk26)
echo "VIBEFIVE : $V"
echo "ALPHAGOM : $A"
python3 tools/harvest_games.py \\
  --engine-a "$V" --engine-b "$A" \\
  --games 2 --out /tmp/smoke.jsonl \\
  --openings full-ring --timeout-turn 1000 --every 2 \\
  --min-ply 6 --skip-last 4
head -c 300 /tmp/smoke.jsonl; echo"""

HARVEST_CELL = """\
%%bash
# ---- Cell 3: the harvest (12h budget; resumes by appending) ----
# 208 games/pair = two laps of the 104 ring openings. 20 ordered pairs.
# 100 ms/move keeps a ~40-move game under a minute. Expect several thousand
# games and 30-60k positions in a session. Killed at the 12h cap? Everything
# already flushed to the jsonl is safe; rerun this cell to continue.
set -uo pipefail
REPO=/kaggle/working/oog
OUT=/kaggle/working/data/harvest_top5.jsonl
GAMES_PER_PAIR=${GAMES_PER_PAIR:-700}
TURN_MS=${TURN_MS:-100}
cd "$REPO"

find_eng() {
  local exe
  # Prefer the least-demanding build when a zip ships several CPU variants
  # (rapfi25 has avx2/avx512/avx512vnni/avxvnni; Kaggle CPUs lack AVX-512,
  # the avx512 exe segfaults on startup and kills the whole pair).
  exe="$(find "third_party/$1" -maxdepth 3 -iname 'pbrain-*' |
         grep -viE 'unins|setup|vcredist|\.so' |
         grep -i avx2 | head -n1)"
  [ -n "$exe" ] || exe="$(find "third_party/$1" -maxdepth 3 -iname 'pbrain-*' |
         grep -viE 'unins|setup|vcredist|\.so|avx512|avxvnni' | head -n1)"
  case "$exe" in
    *.exe) echo "wine $REPO/$exe" ;;
    *)     chmod +x "$exe" 2>/dev/null; echo "$REPO/$exe" ;;
  esac
}

mkdir -p "$(dirname "$OUT")"; touch "$OUT"

# Offset the opening cycle each lap so reruns of this cell (the session may
# hit the 12h cap) keep playing fresh ring openings instead of repeating.
LAP=$( [ -f /kaggle/working/lap.txt ] && cat /kaggle/working/lap.txt || echo 0 )
START_IDX=$(( LAP * GAMES_PER_PAIR ))
echo "lap=$LAP start_idx=$START_IDX"
echo $(( LAP + 1 )) > /kaggle/working/lap.txt

i=0
for a in vibefive agmk26 slowrenju20 rapfi25 yixin18; do
  for b in vibefive agmk26 slowrenju20 rapfi25 yixin18; do
    [ "$a" = "$b" ] && continue
    EA=$(find_eng "$a"); EB=$(find_eng "$b")
    [ -n "$EA" ] && [ -n "$EB" ] || { echo "skip $a/$b: exe missing"; continue; }
    echo "=== [$((++i))/20] $a vs $b ($GAMES_PER_PAIR games) ==="
    python3 tools/harvest_games.py \\
      --engine-a "$EA" --engine-b "$EB" \\
      --games "$GAMES_PER_PAIR" --out "$OUT" \\
      --start-idx "$START_IDX" \\
      --openings full-ring \\
      --timeout-turn "$TURN_MS" --timeout-match 3600000 \\
      --every 2 --min-ply 6 --skip-last 4 || echo "pair failed, continuing"
    sleep 1
  done
done

echo "TOTAL: $(wc -l < "$OUT") positions in $OUT"
tar czf /kaggle/working/harvest_top5.jsonl.tgz -C "$(dirname "$OUT")" harvest_top5.jsonl"""

STATS_CELL = '''\
# ---- Cell 4: stats on the harvest ----
import json, collections, statistics
p = "/kaggle/working/data/harvest_top5.jsonl"
recs = [json.loads(l) for l in open(p)]
print(f"positions        : {len(recs)}")
games = {tuple(map(tuple, r["moves"][:8])) for r in recs}
print(f"~distinct games  : {len(games)}")
print(f"winners          : {dict(collections.Counter(r['winner'] for r in recs))}")
print(f"first moves seen : {len({tuple(r['moves'][0]) for r in recs})}/104")
print(f"mean ply sampled : {statistics.mean(r['ply'] for r in recs):.1f}")'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "kaggle_harvest.ipynb"))
    args = ap.parse_args()

    cells = [c.strip("\n") for c in
             (SETUP_CELL, SMOKE_CELL, HARVEST_CELL, STATS_CELL)]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "none", "dataSources": [],
                       "isInternetEnabled": True, "language": "python"},
        },
        "cells": [{"cell_type": "code", "metadata": {},
                   "execution_count": None, "outputs": [], "source": c}
                  for c in cells],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"generated {args.out}")


if __name__ == "__main__":
    main()
