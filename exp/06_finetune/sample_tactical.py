#!/usr/bin/env python3
"""Sample TACTICAL harvest positions for focused re-distillation.

A position qualifies when either side has a live threat on the board:
  - a 4 (immediate five possible next move), or
  - an open 3 (five possible in two moves)
or when the NEXT played move creates one (decision points).

Uses the 5-window scanner from exp/05_hybrid/heuristic_player.py.
Output: jsonl of {moves, stm} ready for tools/label_positions.py.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "exp", "05_hybrid"))
sys.path.insert(0, REPO)

from search import HB, N  # noqa: E402

DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def has_threat(hb, color):
    """True if `color` has a four (cnt==4 window) or open three."""
    g, n = hb.g, hb.n
    for r in range(n):
        for c in range(n):
            for dr, dc in DIRS:
                er, ec = r + 4 * dr, c + 4 * dc
                if not (0 <= er < n and 0 <= ec < n):
                    break
                vals = [g[r + i * dr, c + i * dc] for i in range(5)]
                if any(v not in (0, color) for v in vals):
                    continue
                cnt = sum(1 for v in vals if v == color)
                if cnt == 4:
                    return True
                if cnt == 3:
                    b0 = (r - dr, c - dc)
                    a4 = (r + 5 * dr, c + 5 * dc)
                    ob = 0 <= b0[0] < n and 0 <= b0[1] < n and g[b0] == 0
                    oa = 0 <= a4[0] < n and 0 <= a4[1] < n and g[a4] == 0
                    if ob and oa:
                        return True
    return False


def iter_games(path):
    cur = None
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cur is not None and (r["ply"] <= cur[-1]["ply"]
                                or r["engines"] != cur[-1]["engines"]):
            yield cur
            cur = None
        cur = (cur or []) + [r]
    if cur:
        yield cur


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/harvest_merged.jsonl")
    ap.add_argument("--out", default="exp/06_finetune/tactical_positions.jsonl")
    ap.add_argument("--limit", type=int, default=40000)
    cfg = ap.parse_args(argv)

    out = open(cfg.out, "w")
    n_pos = n_games = 0
    for g in iter_games(cfg.data):
        moves = g[-1]["moves"]
        n_games += 1
        hb = HB()
        col = 1
        for m in moves[:2]:
            hb.put(m[1], m[0], col)
            col = 3 - col
        for ply in range(2, len(moves) - 1):
            x, y = moves[ply][0], moves[ply][1]
            if not (0 <= x < N and 0 <= y < N):
                break
            threat = has_threat(hb, 1) or has_threat(hb, 2)
            if threat and n_pos < cfg.limit:
                stm = "B" if col == 1 else "W"
                rec = {"moves": [[int(a), int(b)] for a, b in moves[:ply]],
                       "stm": stm}
                out.write(json.dumps(rec) + "\n")
                n_pos += 1
            hb.put(y, x, col)
            col = 3 - col
    out.close()
    print(f"games scanned: {n_games}  tactical positions: {n_pos}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
