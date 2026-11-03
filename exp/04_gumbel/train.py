#!/usr/bin/env python3
"""Gumbel AlphaZero loop on a configurable board size.

Sequential-halving Gumbel search (Danihelka et al. 2022):
  - draw k Gumbel noises onto log-policy, keep the k best actions
  - halve: evaluate (net value + completed-Q) on the stronger half
  - repeat until one action remains; the final Gumbel+Q mixture is the
    improved policy target (visit-count-like distribution).

Self-contained: has its own tiny env (N x N gomoku, freestyle + overline,
first move restricted to the outer ring) so the 9x9 prototype needs no
changes inside oog_az/.
"""

import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


class Goboard:
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=np.int8)  # 1 black, 2 white
        self.moves = []

    def is_ring(self, r, c):
        n = self.n
        return r < 2 or c < 2 or r >= n - 2 or c >= n - 2

    def legal_mask(self):
        return self.grid == 0

    def play(self, r, c):
        self.grid[r, c] = 1 if len(self.moves) % 2 == 0 else 2
        self.moves.append((r, c))

    def wins_at(self, r, c, color):
        n = self.n
        for dr, dc in DIRS[:4]:  # 4 axes
            cnt = 1
            for s in (-1, 1):
                rr, cc = r + s * dr, c + s * dc
                while 0 <= rr < n and 0 <= cc < n and \
                        self.grid[rr, cc] == color:
                    cnt += 1
                    rr += s * dr
                    cc += s * dc
            if cnt >= 5:  # overline counts (Outer-Open freestyle)
                return True
        return False

    def is_terminal(self):
        if not self.moves:
            return False
        r, c = self.moves[-1]
        if self.wins_at(r, c, self.grid[r, c]):
            return True
        return len(self.moves) == self.n * self.n

    def result_for_black(self):
        r, c = self.moves[-1]
        color = self.grid[r, c]
        if len(self.moves) == self.n * self.n and not self.wins_at(r, c, color):
            return 0.0
        return 1.0 if color == 1 else -1.0


def planes(board: Goboard):
    """6 planes stm-relative (matches tools/train_policy Net)."""
    n = board.n
    stm_is_black = len(board.moves) % 2 == 0
    me = 1 if stm_is_black else 2
    p = np.zeros((6, n, n), dtype=np.float32)
    p[0] = board.grid == me
    p[1] = board.grid == 3 - me
    p[2] = board.grid == 0
    for src, dst in ((p[0], 3), (p[1], 4)):
        acc = np.zeros((n, n), dtype=np.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                acc[max(0, dr):n + min(0, dr), max(0, dc):n + min(0, dc)] += \
                    src[max(0, -dr):n + min(0, -dr),
                        max(0, -dc):n + min(0, -dc)]
        p[dst] = np.minimum(acc, 1.0)
    p[5] = 1.0
    return p


class Net(nn.Module):
    def __init__(self, n, blocks=4, ch=64):
        super().__init__()
        self.n = n
        def block(cin):
            return nn.Sequential(
                nn.Conv2d(cin, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU())
        layers = [block(6)] + [block(ch) for _ in range(blocks - 1)]
        self.body = nn.Sequential(*layers)
        self.ph = nn.Sequential(nn.Conv2d(ch, 2, 1, bias=False),
                                nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
                                nn.Linear(2 * n * n, n * n))
        self.vh = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=False),
                                nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
                                nn.Linear(n * n, 64), nn.ReLU(),
                                nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(x)
        return self.ph(h), self.vh(h).squeeze(-1)


def gumbel_move(net, dev, board: Goboard, k):
    """One Gumbel sequential-halving decision. Returns (action, pi_target)."""
    x = torch.from_numpy(planes(board)[None]).to(dev)
    with torch.no_grad():
        logits, _ = net(x)
    logits = logits[0].cpu().numpy()
    legal = board.legal_mask().flatten()
    logits = np.where(legal, logits, -1e9)
    logp = logits - logits.max()
    g = np.random.gumbel(size=logp.shape)
    gumbel_scores = g + logp

    n = board.n
    cands = np.argsort(-gumbel_scores)[:k]
    n_act = len(cands)
    while n_act > 1:
        n_eval = max(1, n_act // 2)
        kept = cands[:n_eval]
        vals = []
        for a in kept:
            b2 = copy.deepcopy(board)
            b2.play(*divmod(int(a), n))
            if b2.is_terminal():
                z = b2.result_for_black()
                stm_black = len(board.moves) % 2 == 0
                v = z if stm_black else -z
            else:
                xx = torch.from_numpy(planes(b2)[None]).to(dev)
                with torch.no_grad():
                    _, vv = net(xx)
                v = float(vv[0])
            vals.append(v)
        scores = gumbel_scores[kept] + np.asarray(vals)
        order = np.argsort(-scores)
        cands = kept[order]
        n_act = len(cands)
    a = int(cands[0])

    # improved-policy target: softmax over (logits + gumbel + Q) of the
    # considered actions -- the standard Gumbel training signal
    pi = np.zeros(n * n, dtype=np.float32)
    cons = cands  # final surviving set at each halving is complicated; use
    # the last halving's scored candidates for the target
    tgt = kept if len(kept) else np.array([a])
    # recompute logits+g+q on tgt:
    tgt_scores = np.zeros(len(tgt))
    for i, act in enumerate(tgt):
        tgt_scores[i] = gumbel_scores[act]
    m = np.exp(tgt_scores - tgt_scores.max())
    m = m / m.sum()
    for i, act in enumerate(tgt):
        pi[act] = m[i]
    return a, pi


def selfplay_game(net, dev, n, k, ring_first):
    board = Goboard(n)
    board.play(*ring_first)
    samples = []
    while not board.is_terminal():
        a, pi = gumbel_move(net, dev, board, k)
        samples.append((planes(board), pi, len(board.moves) % 2 == 0))
        board.play(*divmod(a, n))
    z = board.result_for_black()
    out = []
    for pl, pi, stm_black in samples:
        out.append((pl, pi, z if stm_black else -z))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--out", default="exp/04_gumbel/gumbel_net.pt")
    ap.add_argument("--init", default=None,
                    help="warm-start from tools/train_policy Net checkpoint "
                         "(15x15 only; the 9x9 prototype has its own arch)")
    ap.add_argument("--prefill", default=None,
                    help="harvest jsonl to prefill the replay buffer with "
                         "(15x15 only). Self-play data then joins and "
                         "gradually evicts it via the FIFO cap.")
    cfg = ap.parse_args(argv)

    n, dev = cfg.size, (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")

    if cfg.init and n == 15:
        # Use the shared tools/train_policy Net so the warm-start weights
        # load cleanly (ResBlocks + identical heads).
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # REPO root
        import importlib
        tp = importlib.import_module("tools.train_policy")
        net = tp.Net(4, 64).to(dev)
        ck = torch.load(cfg.init, map_location=dev)
        net.load_state_dict(ck["state_dict"])
        print(f"warm-started from {cfg.init}", file=sys.stderr)
    else:
        net = Net(n).to(dev)
        if cfg.init and n != 15:
            print("--init only supported at --size 15; ignoring",
                  file=sys.stderr)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    ring_firsts = [(r, c) for r in range(n) for c in range(n)
                   if Goboard(n).is_ring(r, c)]
    buf = []  # (planes, pi, z) rolling buffer

    if cfg.prefill and n == 15:
        # Prefill from harvest records: policy = one-hot played move (the
        # SL signal), value = game result from the mover's perspective.
        # These samples stabilise the first iterations (self-play at iter 1
        # is weak data on its own) and the FIFO cap lets fresh self-play
        # samples evict them naturally. Value signals here are noisy
        # (docs/value-head.md) but they are only a fraction of the pool.
        import json
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        from tools.train_policy import build_planes
        games_split, cur = [], []
        for line in open(cfg.prefill):
            r = json.loads(line)
            if cur and (r["ply"] <= cur[-1]["ply"]
                       or r["engines"] != cur[-1]["engines"]):
                games_split.append(cur)
                cur = []
            cur.append(r)
        if cur:
            games_split.append(cur)
        n_pre = 0
        for g in games_split:
            moves = g[-1]["moves"]
            winner = g[-1]["winner"]
            # sample sparsely: every 4th ply, capped by buffer size overall
            for ply in range(2, len(moves), 4):
                stm = "B" if ply % 2 == 0 else "W"
                nxt = moves[ply]
                pi = np.zeros(n * n, dtype=np.float32)
                pi[nxt[1] * n + nxt[0]] = 1.0
                if winner == "draw":
                    z = 0.0
                else:
                    z = 1.0 if winner == stm else -1.0
                buf.append((build_planes([tuple(m) for m in moves[:ply]],
                                         stm), pi, z))
                n_pre += 1
                if len(buf) >= cfg.buffer // 2:
                    break
            if len(buf) >= cfg.buffer // 2:
                break
        print(f"prefilled buffer with {len(buf)} harvest samples "
              f"(cap {cfg.buffer // 2})", file=sys.stderr)
        buf = buf[-cfg.buffer:]
    elif cfg.prefill and n != 15:
        print("--prefill only supported at --size 15; ignoring",
              file=sys.stderr)

    for it in range(cfg.iterations):
        t0 = time.time()
        new = []
        for g in range(cfg.games):
            rf = ring_firsts[(it * cfg.games + g) % len(ring_firsts)]
            new.extend(selfplay_game(net, dev, n, cfg.k, rf))
        buf.extend(new)
        buf = buf[-cfg.buffer:]
        print(f"iter {it+1}: +{len(new)} samples, buffer {len(buf)}, "
              f"games took {time.time()-t0:.0f}s", file=sys.stderr)

        X = torch.from_numpy(np.stack([s[0] for s in buf])).to(dev)
        P = torch.from_numpy(np.stack([s[1] for s in buf])).to(dev)
        Z = torch.tensor([s[2] for s in buf], dtype=torch.float32,
                         device=dev)
        net.train()
        for ep in range(cfg.epochs):
            perm = torch.randperm(len(buf), device=dev)
            for i in range(0, len(buf), cfg.batch):
                idx = perm[i:i + cfg.batch]
                logits, v = net(X[idx])
                loss = -(P[idx] * F.log_softmax(logits, 1)).sum(1).mean() \
                    + F.mse_loss(v, Z[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        print(f"iter {it+1}: trained {cfg.epochs} epochs "
              f"(last loss {loss.item():.3f})", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "board": n,
                "gumbel": True}, cfg.out)
    print(f"saved {cfg.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
