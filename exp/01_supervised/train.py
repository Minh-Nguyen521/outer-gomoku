#!/usr/bin/env python3
"""Supervised training on harvest jsonl (moves/stm/result records).

Turns each harvest record's LAST move into a one-hot policy target and the
record's `result` (from stm's perspective, +1 win / -1 loss / 0 draw) into
the value target. Cheaper than label_positions.py (needs no teacher) and
runs directly on data/harvest_top5.jsonl.

Adapted from tools/train_policy.py (same net arch, different data loader).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 15
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from tools.train_policy import Net, build_planes  # noqa: E402


def load_harvest(path, limit=0, stride=1):
    """Group records into games, then sample (prefix, next_move, result).

    harvest_games.py writes one record per sampled ply, but the LAST record
    of each game holds the full move list (records embed their prefix). So:
    split by game (engines + ply resets), take the last record's moves as the
    ground truth, re-sample every ply the same way the harvest did.

    stride > 1 subsamples plies within each game (stride=4 keeps ~1M
    samples from the full harvest) for memory-constrained callers.
    """
    games, cur = [], None
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cur is not None and (r["ply"] <= cur[-1]["ply"]
                                    or r["engines"] != cur[-1]["engines"]):
                games.append(cur)
                cur = None
            cur = (cur or []) + [r]
    if cur:
        games.append(cur)

    X, P, V = [], [], []
    for g in games:
        moves = g[-1]["moves"]          # full game
        winner = g[-1]["winner"]
        for ply in range(2, len(moves), stride):  # subsample if stride>1
            stm = "B" if ply % 2 == 0 else "W"
            nxt = moves[ply]
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            onehot = np.zeros(N * N, dtype=np.float32)
            onehot[nxt[1] * N + nxt[0]] = 1.0
            X.append(build_planes([tuple(m) for m in moves[:ply]], stm))
            P.append(onehot)
            if winner == "draw":
                res = 0.0
            else:
                res = 1.0 if winner == stm else -1.0
            V.append(res)
    print(f"games: {len(games)}  samples: {len(X)}", file=sys.stderr)
    return np.stack(X), np.stack(P), np.asarray(V, dtype=np.float32)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/harvest_top5.jsonl")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="exp/01_supervised/supervised.pt")
    cfg = ap.parse_args(argv)

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", file=sys.stderr)

    X, P, V = load_harvest(cfg.data, cfg.limit)
    n = len(X)
    n_val = max(1024, int(n * 0.05))
    tr_end = n - n_val
    print(f"positions: {n}  train {tr_end}  val {n_val}", file=sys.stderr)

    # Too big for GPU RAM: keep planes on CPU (shared-memory tensors from
    # numpy need copies; np arrays already share nothing) and move only the
    # mini-batch across. torch.from_numpy per batch keeps zero-copy reads.
    Xva = torch.from_numpy(X[tr_end:])
    Pva = torch.from_numpy(P[tr_end:])
    Vva = torch.from_numpy(V[tr_end:])

    net = Net(cfg.blocks, cfg.channels).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    ntr = tr_end
    for ep in range(cfg.epochs):
        net.train()
        perm = np.random.permutation(ntr)
        tot_p = tot_v = nb = 0.0
        t0 = time.time()
        for i in range(0, ntr, cfg.batch):
            idx = perm[i:i + cfg.batch]
            xb = torch.from_numpy(X[idx]).to(dev)
            pb = torch.from_numpy(P[idx]).to(dev)
            vb = torch.from_numpy(V[idx]).to(dev)
            logits, val = net(xb)
            lp = F.log_softmax(logits, dim=1)
            loss_p = -(pb * lp).sum(dim=1).mean()
            loss_v = F.mse_loss(val, vb)
            opt.zero_grad(set_to_none=True)
            (loss_p + loss_v).backward()
            opt.step()
            tot_p += loss_p.item(); tot_v += loss_v.item(); nb += 1
        sched.step()

        net.eval()
        with torch.no_grad():
            top1s, vmaes = [], []
            for i in range(0, len(Xva), 2048):
                xb = Xva[i:i + 2048].to(dev)
                lo, vv = net(xb)
                pv = Pva[i:i + 2048].to(dev)
                tv = Vva[i:i + 2048].to(dev)
                top1s.append((lo.argmax(1) == pv.argmax(1))
                             .float().sum().item())
                vmaes.append(((vv - tv).abs().sum()).item())
            top1 = sum(top1s) / len(Xva)
            vmae = sum(vmaes) / len(Xva)
        print(f"ep {ep+1:3d}  policy {tot_p/nb:.4f}  value {tot_v/nb:.4f}  "
              f"| val top1 {100*top1:.1f}%  vMAE {vmae:.3f}  "
              f"({time.time()-t0:.1f}s)", file=sys.stderr)
        # Save after every epoch: a multi-hour run killed at ep 14 should
        # not lose everything (the harvest runs taught us this lesson).
        os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
        torch.save({"state_dict": net.state_dict(),
                    "blocks": cfg.blocks, "channels": cfg.channels,
                    "planes": 6, "board": N, "epoch": ep + 1}, cfg.out)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "blocks": cfg.blocks, "channels": cfg.channels,
                "planes": 6, "board": N}, cfg.out)
    print(f"saved {cfg.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
