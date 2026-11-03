# Method 4: Gumbel AlphaZero on a SMALL board (9x9 prototype)

Policy-improvement WITHOUT a full MCTS: sequential halving over the top-k
policy moves (Danihelka et al. 2022, "Policy improvement by planning with
Gumbel"). At k=16 the search does 16 net evals per move instead of
hundreds, so CPU/MPS self-play becomes viable.

Two sizes from the same script:
  * `--size 9`   : proof-of-concept run, minutes on MPS, answers "does the
                  pipeline learn at all?"
  * `--size 15` : full Outer-Open (ring adjusted to board size), slower.

Ring on 9x9 = outer 2 rows/cols of the 9x9 grid.

    ./run.sh                    # 9x9, 8 iterations x 30 games, k=8
    ./run.sh 15 20 100 16       # 15x15, 20 iters x 100 games, k=16
