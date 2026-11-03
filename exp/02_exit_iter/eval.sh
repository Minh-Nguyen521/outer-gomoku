#!/bin/bash
# ExIt net vs Method-1 SL net, head to head (both raw policy + small search).
set -e
cd "$(dirname "$0")/../.."
python3 tools/match.py --games 20 --parallel 1 --rule 0 --timeout-turn 10000 \
    --engine-a "python3 oog_az/player.py --model exp/02_exit_iter/exit_net.pt --simulations 100" \
    --engine-b "python3 oog_az/player.py --model exp/01_supervised/supervised.pt --simulations 100" \
    2>&1 | tail -8
