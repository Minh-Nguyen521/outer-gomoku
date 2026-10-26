#!/usr/bin/env python3
"""Generate diverse, realistic Outer-Open Gomoku positions for training data.

Phase 6 step 1. These get labelled by a strong teacher (tools/label_positions.py)
rather than by our own engine, because six independent measurements this
session showed our evaluation is the weak component -- training on our own
judgements would just relearn our blind spots. The teacher supplies the
targets; this script only has to supply a sensible *distribution* of positions.

What "sensible" means here:

  * Legal under Outer-Open: Black's first stone lies on the outer two rows or
    columns. Positions the engine will never see are wasted labels.
  * Locally plausible: moves are drawn near existing stones (Chebyshev <= 2),
    the same neighbourhood our engine's move generator uses. Uniformly random
    boards look nothing like real play and would teach the network to
    evaluate positions that never occur.
  * Varied: a tunable fraction of moves is chosen greedily (extend/block the
    longest line) and the rest randomly, so the set spans quiet openings
    through sharp tactical middlegames rather than clustering on one style.
  * Sampled through the game, not just at the end, so early and late
    positions are both represented.

Output is JSONL, one position per line:
    {"moves": [[x,y], ...], "stm": "B"|"W", "ply": N}
"""

import argparse
import json
import random
import sys

N = 15
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))


def on_ring(r, c):
    return r < 2 or r >= N - 2 or c < 2 or c >= N - 2


def wins_at(g, r, c):
    col = g[r][c]
    for dr, dc in DIRS:
        cnt = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < N and 0 <= cc < N and g[rr][cc] == col:
                cnt += 1
                rr += dr * sign
                cc += dc * sign
        if cnt >= 5:
            return True
    return False


def line_len(g, r, c, col):
    """Longest run through (r,c) if `col` played there -- a crude move score."""
    best = 0
    for dr, dc in DIRS:
        cnt = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < N and 0 <= cc < N and g[rr][cc] == col:
                cnt += 1
                rr += dr * sign
                cc += dc * sign
        best = max(best, cnt)
    return best


def candidates(g, radius=2):
    out = set()
    any_stone = False
    for r in range(N):
        for c in range(N):
            if g[r][c] == ".":
                continue
            any_stone = True
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < N and 0 <= cc < N and g[rr][cc] == ".":
                        out.add((rr, cc))
    if not any_stone:
        return [(r, c) for r in range(N) for c in range(N) if on_ring(r, c)]
    return sorted(out)


def play_game(rng, greedy_p, max_ply):
    """One randomised game; returns the move list (stops at a win)."""
    g = [["."] * N for _ in range(N)]
    moves = []
    for ply in range(max_ply):
        col = "B" if ply % 2 == 0 else "W"
        if ply == 0:
            opts = [(r, c) for r in range(N) for c in range(N) if on_ring(r, c)]
        else:
            opts = candidates(g)
        if not opts:
            break
        # The opening is always random: on an empty board every ring square
        # scores identically, so the greedy rule would deterministically pick
        # the same corner every game and collapse the opening diversity.
        if ply > 0 and rng.random() < greedy_p:
            # Greedy: extend our own longest line, or block theirs -- whichever
            # is longer. Keeps games tactically coherent without a real engine.
            other = "W" if col == "B" else "B"
            best, pick = -1, None
            for (r, c) in opts:
                s = max(line_len(g, r, c, col), line_len(g, r, c, other))
                if s > best:
                    best, pick = s, (r, c)
            r, c = pick
        else:
            r, c = rng.choice(opts)
        g[r][c] = col
        moves.append((r, c))
        if wins_at(g, r, c):
            break
    return moves


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000, help="positions to emit")
    ap.add_argument("--greedy", type=float, default=0.65,
                    help="probability a move is chosen greedily rather than at "
                         "random; higher = sharper, more realistic games")
    ap.add_argument("--min-ply", type=int, default=4,
                    help="skip the first few plies; near-empty boards carry "
                         "little signal and all look alike")
    ap.add_argument("--max-ply", type=int, default=90)
    ap.add_argument("--per-game", type=int, default=6,
                    help="positions sampled per game, spread over its length")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="-")
    cfg = ap.parse_args(argv)

    rng = random.Random(cfg.seed)
    fh = sys.stdout if cfg.out == "-" else open(cfg.out, "w")
    emitted = 0
    seen = set()          # dedupe: greedy play repeats openings often
    while emitted < cfg.count:
        moves = play_game(rng, cfg.greedy, cfg.max_ply)
        if len(moves) <= cfg.min_ply:
            continue
        # Sample plies spread across the game, not just the end.
        lo, hi = cfg.min_ply, len(moves) - 1
        if hi <= lo:
            continue
        picks = sorted(rng.sample(range(lo, hi), min(cfg.per_game, hi - lo)))
        for p in picks:
            prefix = moves[:p]
            key = tuple(prefix)
            if key in seen:
                continue
            seen.add(key)
            fh.write(json.dumps({
                "moves": [[x, y] for (x, y) in prefix],
                "stm": "B" if p % 2 == 0 else "W",
                "ply": p,
            }) + "\n")
            emitted += 1
            if emitted >= cfg.count:
                break
    if fh is not sys.stdout:
        fh.close()
    print(f"wrote {emitted} positions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
