#!/bin/bash
# Distill KataGomo's raw policy+value onto harvest positions.
set -e
cd "$(dirname "$0")/../.."
OUT=exp/03_distill
LIMIT=${1:-20000}   # positions to label (KataGomo eval is fast but not free)

# 1) positions from harvest
python3 exp/03_distill/sample_positions.py \
    --data data/harvest_top5.jsonl --out "$OUT/positions.jsonl" \
    --limit "$LIMIT"

# 2) teacher labels (paths from docs/engine-ladder.md reproduction line)
python3 tools/label_positions.py \
    --katago third_party/KataGomo/cpp/build2/katago \
    --model third_party/kgweights/tworule_20x_b18.bin.gz \
    --config third_party/KataGomo/cpp/build2/oog.cfg \
    --in "$OUT/positions.jsonl" \
    --out "$OUT/labelled.jsonl"

# 3) distil
python3 tools/train_policy.py --data "$OUT/labelled.jsonl" \
    --epochs 20 --out "$OUT/distilled.pt"
