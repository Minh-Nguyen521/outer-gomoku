#!/bin/bash
# SL net vs SlowRenju (ladder engine) using the 6-plane player.
set -e
cd "$(dirname "$0")/../.."
python3 tools/match.py --games 20 --parallel 1 --rule 0 --timeout-turn 10000 \
    --engine-a "python3 exp/01_supervised/player.py --model exp/01_supervised/supervised.pt --simulations 0" \
    --engine-b third_party/SlowRenju/pbrain-slowrenju 2>&1 | tail -8
