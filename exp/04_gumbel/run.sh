#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
SIZE=${1:-9}
ITERS=${2:-8}
GAMES=${3:-30}
K=${4:-8}
python3 exp/04_gumbel/train.py --size "$SIZE" --iterations "$ITERS" \
    --games "$GAMES" --k "$K"
