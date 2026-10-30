#!/bin/bash
# Round-robin harvest among the top-5 CPU engines from docs/engine-ladder.md
# (unified ranking 2026-09-17, positions 2-6; KataGomo excluded because it
# needs a GPU and is reserved as the labelling teacher).
#
# Windows binaries go through third_party/wine-engine (Docker+Wine, see
# docs/external-engines.md); rapfi runs natively (arm64 build).
#
# Each ordered pair plays GAMES_PER_PAIR games; harvest_games.py cycles all
# 104 ring openings, so every game starts legal Outer-Open. Output
# accumulates into data/harvest_top5.jsonl.
#
# Usage: tools/harvest_top5.sh [GAMES_PER_PAIR=208] [TURN_MS=1000]
#   208 = two laps of the 104 openings per pair; 20 pairs x 208 = 4160 games.
#
# NOTE: Runs ONE pair at a time (serial). Parallel Wine containers compete
# for Docker resources and cause init failures on macOS. Serial is slower
# but stable; throughput is still ~100 games/min with 500ms/move.

set -uo pipefail
cd "$(dirname "$0")/.."

WINE=third_party/wine-engine
ENG[0]="$WINE vibefive pbrain-vibefive.exe"
ENG[1]="$WINE agmk26 pbrain-AlphaGomoku.exe"
ENG[2]="third_party/SlowRenju/pbrain-slowrenju"
ENG[3]="third_party/rapfi/Rapfi/build/arm64/pbrain-rapfi"
ENG[4]="$WINE yixin pbrain-Yixin2018.exe"

OUT=data/harvest_top5.jsonl
GAMES_PER_PAIR="${1:-208}"
TURNT="${2:-1000}"

mkdir -p "$(dirname "$OUT")"
touch "$OUT"

for i in 0 1 2 3 4; do
  for j in 0 1 2 3 4; do
    [ "$i" -eq "$j" ] && continue
    echo "=== ${ENG[$i]##* } vs ${ENG[$j]##* } ($GAMES_PER_PAIR games) ==="
    python3 tools/harvest_games.py \
      --engine-a "${ENG[$i]}" --engine-b "${ENG[$j]}" \
      --games "$GAMES_PER_PAIR" --out "$OUT" \
      --openings full-ring \
      --timeout-turn "$TURNT" --timeout-match 3600000 \
      --every 2 --min-ply 6 --skip-last 4 || echo "PAIR FAILED, continuing"
    sleep 3  # let Docker container fully teardown before next pair
  done
done

echo "done: $(wc -l <"$OUT" | tr -d ' ') positions in $OUT"
