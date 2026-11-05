#!/usr/bin/env python3
"""Pure-heuristic player: alpha-beta + threat-pattern evaluation.

No neural net anywhere: leaf eval is a hand-coded pattern score that
counts fours/open-threes for both colours plus centrality. This measures
how far classic-engine tricks alone go against SlowRenju.

Wire protocol identical to exp/05_hybrid/player.py.
"""

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search import HB, N  # noqa: E402
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def evaluate(hb, me):
    """Static eval from `me`'s perspective: my threats minus theirs."""
    opp = 3 - me
    my = threat_counts(hb, me)
    their = threat_counts(hb, opp)
    s = (my["five"] * 100000 + my["open4"] * 5000 + my["four"] * 1000 +
         my["open3"] * 300 + my["three"] * 50 +
         -their["five"] * 100000 - their["open4"] * 4500 -
         their["four"] * 900 - their["open3"] * 250 - their["three"] * 40)
    return s / 100000.0  # squashed into roughly [-1.3, 1.3]


def threat_counts(hb, color):
    """Count threat patterns per colour using line windows.

    Slide a 6-cell window along every line; classify the window content:
      - five: 5 contiguous stones with room (a made or imminent five)
      - open4: _XXXX_ (unstoppable)
      - four: XXXX with one extension (blocked one side or broken)
      - open3: _XXX_ with both extension cells reachable toward a four
      - three: simple three
    Broken shapes (X_XX_X) are approximated by the window counts.
    """
    g, n = hb.g, hb.n
    counts = {"five": 0, "open4": 0, "four": 0, "open3": 0, "three": 0}
    # walk all cells; for each STONE of ours, look right/down/diag runs
    # simpler robust approach: examine every possible 5-window
    seen_pairs = set()
    for r in range(n):
        for c in range(n):
            for dr, dc in DIRS:
                # windows starting at (r,c) of length 5
                er, ec = r + 4 * dr, c + 4 * dc
                if not (0 <= er < n and 0 <= ec < n):
                    break
                cells = [(r + i * dr, c + i * dc) for i in range(5)]
                vals = [g[rr, cc] for rr, cc in cells]
                if any(v not in (0, color) for v in vals):
                    continue
                cnt = sum(1 for v in vals if v == color)
                key = (tuple(cells), cnt)
                if cnt >= 5:
                    counts["five"] += 1
                elif cnt == 4:
                    # open if both flanks (before first, after last) are empty
                    b0 = (r - dr, c - dc)
                    a4 = (r + 5 * dr, c + 5 * dc)
                    ob = (0 <= b0[0] < n and 0 <= b0[1] < n
                          and g[b0] == 0)
                    oa = (0 <= a4[0] < n and 0 <= a4[1] < n
                          and g[a4] == 0)
                    counts["open4" if (ob and oa) else "four"] += 1
                elif cnt == 3:
                    b0 = (r - dr, c - dc)
                    a4 = (r + 5 * dr, c + 5 * dc)
                    ob = (0 <= b0[0] < n and 0 <= b0[1] < n
                          and g[b0] == 0)
                    oa = (0 <= a4[0] < n and 0 <= a4[1] < n
                          and g[a4] == 0)
                    if ob and oa:
                        # open three: _XXX_; broken handled elsewhere
                        counts["open3"] += 1
                    else:
                        counts["three"] += 1
    # deduplicate is imperfect but monotic and fast enough
    return counts


def alphabeta(hb, me, depth, alpha, beta, deadline, ply=0):
    if time.monotonic() > deadline:
        raise TimeoutError
    opp = 3 - me
    win = hb.forced(me, opp)
    if win is not None:
        return 1.0, win
    block = hb.forced(opp, me)
    if block is not None:
        r, c = block
        hb.put(r, c, me)
        try:
            sc = -alphabeta(hb, opp, depth - 1, -beta, -alpha,
                            deadline, ply + 1)[0]
        finally:
            hb.undo(r, c)
        return sc, (r, c)
    if depth <= 0:
        return evaluate(hb, me), None
    cands = hb.candidates(me, opp, width=14)
    if not cands:
        return 0.0, None
    cands = sorted(cands, key=lambda rc: -hb.move_score(rc[0], rc[1], me, opp))
    best, best_mv = -2.0, None
    for (r, c) in cands[:10]:
        hb.put(r, c, me)
        try:
            sc = -alphabeta(hb, opp, depth - 1, -beta, -alpha,
                            deadline, ply + 1)[0]
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
    def __init__(self, depth, tms):
        self.depth = depth
        self.tms = tms

    def move(self, hb, me):
        deadline = time.monotonic() + self.tms / 1000.0
        best_mv = None
        d = 2
        try:
            while True:
                sc, mv = alphabeta(hb.clone(), me, d, -2.0, 2.0, deadline)
                best_mv = mv
                d += 1
        except TimeoutError:
            pass
        if best_mv is None:
            c = hb.candidates(me, 3 - me, width=1)
            best_mv = c[0] if c else (N // 2, N // 2)
        r, c = best_mv
        return c, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--movetime", type=int, default=8000)
    args = ap.parse_args()
    tk = Thinker(args.depth, args.movetime)

    import threading, queue
    out_q = queue.Queue()

    def pump():
        for line in sys.stdin:
            out_q.put(line.strip())
        out_q.put(None)
    threading.Thread(target=pump, daemon=True).start()

    moves = []

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
            mx, my = tk.move(hb, me)
            send(f"{mx},{my}")
        elif cmd == "INFO":
            pass
        elif cmd == "END":
            return


if __name__ == "__main__":
    main()
