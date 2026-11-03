# Method 1: Pure Supervised Learning (fastest test)

Train policy + value net on harvested engine games. No self-play.
Use this as the BASELINE: cheap (minutes-hours on MPS/CUDA), tells you
how far pure imitation takes you before investing in RL loops.

## Data
Uses `data/harvest_top5.jsonl` (or any jsonl with `moves`, `stm`,
`result`). Policy target = one-hot next move (weak) OR the played move
distribution; value target = game result from stm's perspective.

## Run
    ./run.sh                       # default: harvest_top5.jsonl, 15 epochs
    ./run.sh data/outer_gomocup/games.jsonl 40   # custom data, 40 epochs

## Evaluate after training
    ./eval.sh   # plays 20 games vs a weak engine via tools/match.py

Expected: minutes on MPS. If val top1 > 40% the imitation signal is there;
if the eval match is competitive, self-play will have a good warm start.
