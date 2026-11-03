# Method 2: Expert Iteration (mini-ExIt)

SL bootstrap + self-play loop WITHOUT search: the net plays itself at
raw-policy speed, its own games join the training pool, retrain, repeat.

Each iteration: ~50 self-play games (seconds-minutes on MPS) + 3 epochs
fine-tune. 5 iterations = a clear A/B against method 1's static SL.

    ./run.sh                 # 5 iterations x 50 games
    ./run.sh 10 100          # 10 iterations x 100 games

Answer it gives: "does bootstrapping from my own play improve beyond
imitation?" -- measurable via eval.sh win-rate vs method-1 net.
