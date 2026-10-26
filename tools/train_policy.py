#!/usr/bin/env python3
"""Distil KataGomo's policy (and the game outcome) into a small network.

WHY THESE TARGETS. Measured on the 20,835-position labelled set:

    policy: median 1.42 nats entropy -> ~4.1 effective moves out of 225,
            top-1 probability ~0.46. Sharp, dense, informative.
    value : only 7.3% of positions have a teacher value in 0.2-0.8, so 93%
            carry essentially no gradient. It also disagrees with the eventual
            game outcome 30% of the time.

So the policy head learns from the TEACHER, and the value head learns from the
GAME RESULT (+1/-1/0, balanced 10204/9986/645). That split is not a
compromise -- it is what AlphaZero does, and here each target is used where it
is strong.

WHY A CONVNET rather than the flat NNUE-style vector in encode_features.py:
the policy target is a 225-way distribution over board squares, and a conv
stack keeps that spatial structure instead of asking a dense layer to
rediscover it. (The NNUE encoding remains the right choice for an alpha-beta
evaluator, which is a different job.)

INPUT PLANES (6 x 15 x 15), all from the side-to-move's point of view so the
network never has to learn colour symmetry:
    0  own stones
    1  opponent stones
    2  empty squares
    3  own stones adjacent (Chebyshev<=1) to any stone -- cheap locality cue
    4  opponent stones likewise
    5  constant 1.0 (lets convs recover board edges from padding)

Usage:
    tools/train_policy.py --data data/labelled.jsonl --epochs 30
    tools/train_policy.py --data data/labelled.jsonl --epochs 30 --blocks 6 --channels 96
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 15


def build_planes(moves, stm):
    """(6, 15, 15) float32 from the side-to-move's perspective."""
    board = np.zeros((N, N), dtype=np.int8)          # 1 = black, 2 = white
    for i, (x, y) in enumerate(moves):
        if 0 <= x < N and 0 <= y < N:
            board[y][x] = 1 if i % 2 == 0 else 2
    me = 1 if stm == "B" else 2
    you = 2 if me == 1 else 1

    p = np.zeros((6, N, N), dtype=np.float32)
    p[0] = (board == me)
    p[1] = (board == you)
    p[2] = (board == 0)
    # Neighbour-occupancy planes: which empty squares touch a stone. The move
    # generator in every engine here only considers squares near existing
    # stones, so handing the network that prior costs two planes and saves it
    # learning the obvious.
    occ_me = p[0]
    occ_you = p[1]
    for src, dst in ((occ_me, 3), (occ_you, 4)):
        acc = np.zeros((N, N), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                acc[max(0, dy):N + min(0, dy), max(0, dx):N + min(0, dx)] += \
                    src[max(0, -dy):N + min(0, -dy), max(0, -dx):N + min(0, -dx)]
        p[dst] = np.minimum(acc, 1.0)
    p[5] = 1.0
    return p


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, blocks=4, ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        # Policy head: 1x1 conv to 2 planes, then a linear map to 225 logits.
        self.ph = nn.Sequential(nn.Conv2d(ch, 2, 1, bias=False),
                                nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
                                nn.Linear(2 * N * N, N * N))
        # Value head: 1 plane -> 64 -> scalar in [-1, 1].
        self.vh = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=False),
                                nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
                                nn.Linear(N * N, 64), nn.ReLU(),
                                nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.ph(h), self.vh(h).squeeze(-1)


def load(path, limit=0):
    X, P, V, legal = [], [], [], []
    for i, line in enumerate(open(path)):
        if limit and i >= limit:
            break
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "policy" not in r or "result" not in r:
            continue
        moves = [tuple(m) for m in r["moves"]]
        X.append(build_planes(moves, r["stm"]))
        P.append(np.asarray(r["policy"], dtype=np.float32))
        V.append(float(r["result"]))
        occ = np.zeros(N * N, dtype=np.float32)
        for (x, y) in moves:
            if 0 <= x < N and 0 <= y < N:
                occ[y * N + x] = 1.0
        legal.append(1.0 - occ)
    return (np.stack(X), np.stack(P), np.asarray(V, dtype=np.float32),
            np.stack(legal))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labelled.jsonl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/policy_net.pt")
    cfg = ap.parse_args(argv)

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", file=sys.stderr)

    X, P, V, L = load(cfg.data, cfg.limit)
    n = len(X)
    # Split by INDEX ORDER, not at random: consecutive positions come from the
    # same game, so a random split would put near-identical positions in both
    # train and validation and report a flattering score.
    n_val = int(n * cfg.val_frac)
    tr = slice(0, n - n_val)
    va = slice(n - n_val, n)
    print(f"positions: {n}  train {n - n_val}  val {n_val}", file=sys.stderr)

    t = lambda a: torch.from_numpy(a).to(dev)
    Xtr, Ptr, Vtr = t(X[tr]), t(P[tr]), t(V[tr])
    Xva, Pva, Vva, Lva = t(X[va]), t(P[va]), t(V[va]), t(L[va])

    net = Net(cfg.blocks, cfg.channels).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"model: {cfg.blocks} blocks x {cfg.channels} ch, {nparam/1000:.0f}k params",
          file=sys.stderr)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    ntr = Xtr.shape[0]
    for ep in range(cfg.epochs):
        net.train()
        perm = torch.randperm(ntr, device=dev)
        tot_p = tot_v = 0.0
        nb = 0
        t0 = time.time()
        for i in range(0, ntr, cfg.batch):
            idx = perm[i:i + cfg.batch]
            logits, val = net(Xtr[idx])
            # Cross-entropy against the teacher's full distribution (not just
            # its argmax): the whole point of distillation is the shape of the
            # distribution, which carries far more per position than one label.
            lp = F.log_softmax(logits, dim=1)
            loss_p = -(Ptr[idx] * lp).sum(dim=1).mean()
            loss_v = F.mse_loss(val, Vtr[idx])
            loss = loss_p + loss_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_p += loss_p.item(); tot_v += loss_v.item(); nb += 1
        sched.step()

        net.eval()
        with torch.no_grad():
            lo, vv = net(Xva)
            # Mask occupied squares before argmax: an illegal "best move" is
            # useless in play, and the teacher never puts mass there.
            masked = lo.masked_fill(Lva < 0.5, -1e9)
            top1 = (masked.argmax(1) == Pva.argmax(1)).float().mean().item()
            k = min(5, N * N)
            in_top5 = (masked.topk(k, dim=1).indices ==
                       Pva.argmax(1, keepdim=True)).any(dim=1).float().mean().item()
            vmae = (vv - Vva).abs().mean().item()
            vsign = ((vv > 0) == (Vva > 0)).float().mean().item()
        print(f"ep {ep+1:3d}  policy {tot_p/nb:.4f}  value {tot_v/nb:.4f}  "
              f"| val top1 {100*top1:.1f}%  top5 {100*in_top5:.1f}%  "
              f"vMAE {vmae:.3f}  vSign {100*vsign:.1f}%  ({time.time()-t0:.1f}s)",
              file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "blocks": cfg.blocks, "channels": cfg.channels,
                "planes": 6, "board": N}, cfg.out)
    print(f"saved {cfg.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
