#!/usr/bin/env python3
"""Expert Iteration (mini-ExIt): SL bootstrapping with self-generated games.

Loop per iteration (Anthony et al. 2017, simplified):
  1. Self-play N games with the current net (+ tiny epsilon-greedy noise
     for diversity). No search -- raw policy, fast.
  2. Append (position, played_move, result) to a growing jsonl.
  3. Retrain from SCRATCH-ish (or warm) on harvest data + self games,
     weighting recent iterations higher by simple concatenation.

Why faster than AlphaZero for a test: no MCTS at all, ~50-100 games per
iteration on CPU/MPS in minutes. Quality per game is lower, but the loop
answers "does iterating on my own games beat static SL?" within hours.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

N = 15
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from oog_az.env import Board, transform_planes, transform_policy  # noqa
from tools.train_policy import Net, build_planes  # noqa: E402


def az_planes_to6(planes5):
    """oog_az Board.planes() is 5-channel; tools/train_policy Net wants 6
    (extra constant-one plane). Pad cheaply."""
    p = np.zeros((6,) + planes5.shape[1:], dtype=np.float32)
    p[:5] = planes5
    p[5] = 1.0
    return p


def selfplay_games(net, dev, n_games, epsilon, ring_cycle):
    """Play n games net-vs-net; return (X, P, V) samples list."""
    net.eval()
    samples = []
    rng = np.random.default_rng()
    for g in range(n_games):
        board = Board()
        first = ring_cycle[g % len(ring_cycle)]
        board.play(first * N + 0 if False else first)
        history = []
        while not board.is_terminal():
            planes = torch.from_numpy(
                az_planes_to6(board.planes())[None]).to(dev)
            with torch.no_grad():
                logits, _ = net(planes)
            legal = torch.from_numpy(
                board.legal_mask()[None].astype(np.float32)).to(dev)
            probs = F.softmax(logits, dim=1) * legal
            probs = probs / probs.sum()
            if rng.random() < epsilon:
                cand = torch.nonzero(legal[0]).flatten()
                a = int(cand[rng.integers(len(cand))])
            else:
                a = int(torch.multinomial(probs[0], 1))
            # augment with one random dihedral symmetry
            sym = int(rng.integers(8))
            history.append((transform_planes(az_planes_to6(board.planes()), sym),
                            transform_policy(probs[0].cpu().numpy(), sym),
                            board.turn))
            board.play(a)
        z = board.result_for(0)  # from Black's perspective
        for planes, pol, turn in history:
            v = z if turn == 0 else -z
            samples.append((planes, pol, v))
    return samples


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-data", default="data/harvest_merged.jsonl")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4,
                    help="subsample seed plies (memory: full pool ~74GB RAM;"
                         " stride 4 ~= 1M samples ~= 18GB)")
    ap.add_argument("--init", default=None,
                    help="warm-start from a checkpoint (tools/train_policy "
                         "Net format); skips fresh initialisation")
    ap.add_argument("--out", default="exp/02_exit_iter/exit_net.pt")
    cfg = ap.parse_args(argv)

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")

    # Seed data from method-1 loader (kept on CPU as numpy; batches are
    # moved to the device on the fly — the full pool is far too large for
    # MPS unified memory)
    sys.path.insert(0, os.path.join(REPO, "exp", "01_supervised"))
    from train import load_harvest  # noqa: E402  (method 1 loader)
    Xs, Ps, Vs = load_harvest(cfg.seed_data, cfg.limit, stride=cfg.stride)
    Xt = [Xs, Ps, Vs]  # numpy pool on CPU

    net = Net(cfg.blocks, cfg.channels).to(dev)
    if cfg.init:
        ck = torch.load(cfg.init, map_location=dev)
        net.load_state_dict(ck["state_dict"])  # arch must match (blocks/ch)
        print(f"warm-started from {cfg.init}", file=sys.stderr)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # ring openings: first move must lie on the outer ring
    env = sys.modules["oog_az.env"]
    ring_cycle = [r * N + c for r in range(N) for c in range(N)
                  if env.Board.is_ring(r, c)]

    for it in range(cfg.iterations):
        # 1) self-play with the CURRENT net
        t0 = time.time()
        fresh = selfplay_games(net, dev, cfg.games, cfg.epsilon, ring_cycle)
        print(f"iter {it+1}: self-played {cfg.games} games, "
              f"{len(fresh)} samples in {time.time()-t0:.0f}s",
              file=sys.stderr)
        Xn = np.stack([s[0] for s in fresh])
        Pn = np.stack([s[1] for s in fresh])
        Vn = np.asarray([s[2] for s in fresh], dtype=np.float32)

        # 2) grow the training pool (self data REPLACES nothing; it appends
        #    and is upweighted implicitly by its freshness)
        X, P, V = (np.concatenate([Xt[0], Xn]),
                   np.concatenate([Xt[1], Pn]),
                   np.concatenate([Xt[2], Vn]))

        # 3) train epochs on the pool
        net.train()
        n_pool = X.shape[0]
        for ep in range(cfg.epochs):
            perm = np.random.permutation(n_pool)
            for i in range(0, n_pool, cfg.batch):
                idx = perm[i:i + cfg.batch]
                xb = torch.from_numpy(X[idx]).to(dev)
                pb = torch.from_numpy(P[idx]).to(dev)
                vb = torch.from_numpy(V[idx]).to(dev)
                logits, val = net(xb)
                lp = F.log_softmax(logits, dim=1)
                loss = -(pb * lp).sum(dim=1).mean() + \
                    F.mse_loss(val, vb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        print(f"iter {it+1}: trained {cfg.epochs} epochs on {n_pool} pool",
              file=sys.stderr)
        # fold the self-play data in permanently
        Xt = [X, P, V]
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "blocks": cfg.blocks, "channels": cfg.channels,
                "planes": 6, "board": N}, cfg.out)
    print(f"saved {cfg.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
