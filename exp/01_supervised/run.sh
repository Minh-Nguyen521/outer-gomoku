#!/bin/bash
# Train pure-SL baseline on harvested games.
set -e
cd "$(dirname "$0")/../.."
DATA=${1:-data/harvest_top5.jsonl}
EPOCHS=${2:-15}
python3 exp/01_supervised/train.py --data "$DATA" --epochs "$EPOCHS"
