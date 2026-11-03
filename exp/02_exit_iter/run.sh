#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
ITERS=${1:-5}
GAMES=${2:-50}
python3 exp/02_exit_iter/train.py --iterations "$ITERS" --games "$GAMES"
