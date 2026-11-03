# Combined pipeline: 3 -> 2 -> 4

Chains the experiment folders with checkpoint warm-starts, so each stage
begins where the last ended:

    KataGomo distil (m3)  ->  ExIt self-play (m2)  ->  Gumbel AZ (m4)
    policy distribution      adds bootstrap games     full RL loop at
    from a real teacher      to the pool             15x15 with k=16

Resumable: each stage writes `distilled.pt` / `exit_net.pt` /
`gumbel_net.pt` here; existing outputs are skipped (delete to redo).

    ./run.sh                 # all stages
    ./run.sh 13              # skip ExIt
    ./run.sh 2 3             # start from the existing distilled net

Runtime (M2/MPS ballparks): stage 1 ~2-3h (mostly KataGomo labelling),
stage 2 ~2-4h (10 iters x 100 games), stage 3 ~6-10h (20 iters x 100
games, k=16). All three are overnight-run material.
