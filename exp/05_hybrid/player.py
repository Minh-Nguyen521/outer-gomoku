#!/usr/bin/env python3
"""Hybrid player: alpha-beta + threat movegen + net leaf eval.

Protocol adapter identical to exp/01_supervised/player.py (x,y wire format,
BOARD rebuild with colour-interleave).

Search skeleton (the classic-engine strength):
  - forced win / forced block detection at EVERY node
  - heuristic move ordering (threat classes) with alpha-beta pruning
  - net value head evaluates leaves beyond `--depth`
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search import HB, N  # noqa: E402
from tools.train_policy import Net  # noqa: E402

DEV = "cpu"


def planes_from_board(hb, me):
    you = 3 - me
    g = hb.g
    p = np.zeros((6, N, N), dtype=np.float32)
    p[0] = g == me
    p[1] = g == you
    p[2] = g == 0
    for src, dst in ((p[0], 3), (p[1], 4)):
        acc = np.zeros((N, N), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                acc[max(0, dy):N + min(0, dy),
                    max(0, dx):N + min(0, dx)] += \
                    src[max(0, -dy):N + min(0, -dy),
                        max(0, -dx):N + min(0, -dx)]
        p[dst] = np.minimum(acc, 1.0)
    p[5] = 1.0
    return p


def leaf_value(net, hb, me):
    x = torch.from_numpy(planes_from_board(hb, me)[None]).to(DEV)
    with torch.no_grad():
        _, v = net(x)
    return float(v[0])


def alphabeta(hb, me, depth, alpha, beta, net, nodes, deadline):
    """Score from `me`'s perspective in [-1,1]."""
    if time.monotonic() > deadline:
        raise TimeoutError
    nodes[0] += 1
    opp = 3 - me
    win = hb.forced(me, opp)
    if win is not None:
        return 1.0, win
    block = hb.forced(opp, me)
    if block is not None:
        r, c = block
        hb.put(r, c, me)
        try:
            sc = -alphabeta(hb, opp, max(depth - 1, 1), -beta, -alpha,
                            net, nodes, deadline)[0]
        finally:
            hb.undo(r, c)
        return sc, (r, c)
    if depth <= 0:
        return leaf_value(net, hb, me), None
    cands = hb.candidates(me, opp, width=12)
    if not cands:
        return 0.0, None
    cands = sorted(cands, key=lambda rc: -hb.move_score(rc[0], rc[1], me, opp))
    best, best_mv = -2.0, None
    for (r, c) in cands[:8]:
        hb.put(r, c, me)
        try:
            sc = -alphabeta(hb, opp, depth - 1, -beta, -alpha,
                            net, nodes, deadline)[0]
        except TimeoutError:
            hb.undo(r, c)
            raise
        hb.undo(r, c)
        if sc > best:
            best, best_mv = sc, (r, c)
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best, best_mv


class Thinker:
    def __init__(self, model_path, depth, tms):
        global DEV
        ck = torch.load(model_path, map_location="cpu")
        self.net = Net(ck.get("blocks", 4), ck.get("channels", 64))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        if torch.backends.mps.is_available():
            DEV = "mps"
            self.net.to(DEV)
        self.depth = depth
        self.tms = tms

    def move(self, hb, me):
        """hb carries ABSOLUTE colours (1 black, 2 white); `me` to move."""
        nodes = [0]
        deadline = time.monotonic() + self.tms / 1000.0
        d = self.depth
        best_mv = None
        try:
            while True:
                sc, mv = alphabeta(hb.clone(), me, d, -2.0, 2.0,
                                   self.net, nodes, deadline)
                if mv is None and best_mv is None:
                    mv = hb.candidates(me, 3 - me, width=1)[0]
                best_mv = mv
                d += 1
        except TimeoutError:
            pass
        if best_mv is None:
            c = hb.candidates(me, 3 - me, width=1)
            best_mv = c[0] if c else (N // 2, N // 2)
        r, c = best_mv
        return c, r  # wire format x,y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--movetime", type=int, default=3000,
                    help="soft time budget per move, ms")
    args = ap.parse_args()

    tk = Thinker(args.model, args.depth, args.movetime)

    import threading, queue
    out_q = queue.Queue()

    def pump():
        for line in sys.stdin:
            out_q.put(line.strip())
        out_q.put(None)
    threading.Thread(target=pump, daemon=True).start()

    moves = []   # wire-format (x,y) absolute order

    def build_board():
        hb = HB()
        color = 1
        for (x, y) in moves:
            hb.put(y, x, color)
            color = 3 - color
        return hb, color

    def send(s):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    while True:
        line = out_q.get()
        if line is None:
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd == "START":
            send("OK")
        elif cmd == "TURN":
            x, y = map(int, rest.split(","))
            moves.append((x, y))
            hb, me = build_board()
            mx, my = tk.move(hb, me)
            send(f"{mx},{my}")
        elif cmd == "BOARD":
            lines = []
            while True:
                l2 = out_q.get()
                if l2 is None or l2 == "DONE":
                    break
                lines.append(l2)
            mine, theirs = [], []
            for e in lines:
                xs, ys, cs = e.split(",")
                (mine if cs == "1" else theirs).append((int(xs), int(ys)))
            if len(mine) >= len(theirs):
                seq = [m for p in zip(mine, theirs) for m in p]
                if len(mine) > len(theirs):
                    seq.append(mine[-1])
            else:
                seq = [m for p in zip(theirs, mine) for m in p]
                seq.append(theirs[-1])
            moves = seq
            hb, me = build_board()
            send("%d,%d" % tk.move(hb, me))
        elif cmd == "INFO":
            pass
        elif cmd == "END":
            return


if __name__ == "__main__":
    main()
