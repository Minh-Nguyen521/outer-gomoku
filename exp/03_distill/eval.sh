#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
python3 tools/match.py --games 20 --parallel 1 --rule 0 --timeout-turn 10000 \
    --engine-a "python3 oog_az/player.py --model exp/03_distill/distilled.pt --simulations 200" \
    --engine-b third_party/SlowRenju/pbrain-slowrenju 2>&1 | tail -5
