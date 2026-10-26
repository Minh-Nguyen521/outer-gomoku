#!/usr/bin/env python3
"""Encode positions as network input features (Phase 6 step 3).

Design constraint that drives everything here: our engine evaluates hundreds
of thousands of leaves per second. A convolutional network per leaf would be
orders of magnitude too slow, so this is an NNUE-style *sparse, incremental*
feature set -- the same shape rapfi uses, and cheap enough to update when one
stone changes rather than recomputing.

Feature groups, all derived from the threat vocabulary Phases 1-2 already
built (`ThreatLevel` / the M-mask line classifier in `eval.cpp`). The 2025
ICGA report notes the Outer-Open champion "explicitly encodes threat-space
features as input", which is exactly this idea -- and it is the one part of
the existing C++ work that carries into Phase 6 unchanged.

  1. Stones            2 planes x 225   own / opponent occupancy.
  2. Line patterns     4 dirs x 225 x levels
                       For each square and each of the four directions, the
                       threat level of the 9-cell window centred there, for
                       the side to move. This is the M-mask classification
                       already implemented in C++; here it is recomputed in
                       Python so the training data can be built before the
                       C++ inference path exists.
  3. Global counts     small dense vector: how many open/closed threats of
                       each level each side holds, plus stone count and side
                       to move. Cheap, and it captures the *fork* information
                       the hand-written eval was blind to.

Output: .npz with X (float32, sparse-ish but stored dense for simplicity),
value targets, and policy targets.

Note on scale: the full one-hot line-pattern encoding is large. For a first
network we use the *compact* variant (--compact, default): per-square threat
level as a small integer plane per direction, plus the global counts. That is
~2.2k floats per position instead of ~30k, trains fast on CPU/MPS, and can be
widened later if the value head proves the approach.
"""

import argparse
import json
import sys

import numpy as np

N = 15
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))

# Threat levels, mirroring eval.hpp's ThreatLevel enum so the C++ side can
# reuse the same classification when inference is ported.
NONE, CLOSED_TWO, OPEN_TWO, CLOSED_THREE, OPEN_THREE, CLOSED_FOUR, OPEN_FOUR, FIVE = range(8)
NUM_LEVELS = 8


def board_from_moves(moves):
    """0 = empty, 1 = black, 2 = white."""
    g = np.zeros((N, N), dtype=np.int8)
    for i, (x, y) in enumerate(moves):
        g[y][x] = 1 if i % 2 == 0 else 2
    return g


def classify(line, idx, colour):
    """Threat level if `colour` played at `line[idx]`.

    `line` is a 1-D list of cell states along one direction. Mirrors the
    run-length + open-ends logic of eval.cpp's runScore/runLevel.
    """
    if line[idx] != 0:
        return NONE
    n = len(line)
    best = NONE
    # Longest run through idx after placing, plus whether the ends are open.
    for start in range(max(0, idx - 4), min(n - 5, idx) + 1):
        win = line[start:start + 5]
        if len(win) < 5:
            continue
        own = sum(1 for v in win if v == colour)
        opp = sum(1 for v in win if v not in (0, colour))
        if opp:
            continue                      # blocked window: worthless
        own += 1                          # the stone we are placing
        l_open = start - 1 >= 0 and line[start - 1] == 0
        r_open = start + 5 < n and line[start + 5] == 0
        if own >= 5:
            lvl = FIVE
        elif own == 4:
            lvl = OPEN_FOUR if (l_open and r_open) else CLOSED_FOUR
        elif own == 3:
            lvl = OPEN_THREE if (l_open and r_open) else CLOSED_THREE
        elif own == 2:
            lvl = OPEN_TWO if (l_open and r_open) else CLOSED_TWO
        else:
            lvl = NONE
        best = max(best, lvl)
    return best


def lines_through(g, dx, dy):
    """All maximal lines in one direction, as (cells, coords)."""
    out = []
    seen = set()
    for y in range(N):
        for x in range(N):
            px, py = x - dx, y - dy
            if 0 <= px < N and 0 <= py < N:
                continue                  # not a line start
            cells, coords = [], []
            cx, cy = x, y
            while 0 <= cx < N and 0 <= cy < N:
                cells.append(int(g[cy][cx]))
                coords.append((cx, cy))
                cx += dx
                cy += dy
            if len(cells) >= 5:
                out.append((cells, coords))
    return out


def encode(moves, stm, compact=True):
    """Return a flat float32 feature vector for one position."""
    g = board_from_moves(moves)
    me = 1 if stm == "B" else 2
    you = 2 if me == 1 else 1

    own = (g == me).astype(np.float32).ravel()
    opp = (g == you).astype(np.float32).ravel()

    # Per-direction threat level for each empty square, for both sides.
    lvl_me = np.zeros((4, N, N), dtype=np.float32)
    lvl_you = np.zeros((4, N, N), dtype=np.float32)
    counts_me = np.zeros(NUM_LEVELS, dtype=np.float32)
    counts_you = np.zeros(NUM_LEVELS, dtype=np.float32)

    for d, (dx, dy) in enumerate(DIRS):
        for cells, coords in lines_through(g, dx, dy):
            for i, (cx, cy) in enumerate(coords):
                if cells[i] != 0:
                    continue
                a = classify(cells, i, me)
                b = classify(cells, i, you)
                if a:
                    lvl_me[d][cy][cx] = a / FIVE      # scale to ~[0,1]
                    counts_me[a] += 1
                if b:
                    lvl_you[d][cy][cx] = b / FIVE
                    counts_you[b] += 1

    globals_ = np.concatenate([
        counts_me / 50.0,                 # rough normalisation
        counts_you / 50.0,
        np.array([len(moves) / 225.0, 1.0 if stm == "B" else 0.0],
                 dtype=np.float32),
    ]).astype(np.float32)

    if compact:
        # Collapse the four directions to (max, sum) per square: keeps the
        # strongest threat and a crude count of how many directions carry one,
        # at a quarter of the size.
        me_max, me_sum = lvl_me.max(axis=0).ravel(), lvl_me.sum(axis=0).ravel() / 4.0
        you_max, you_sum = lvl_you.max(axis=0).ravel(), lvl_you.sum(axis=0).ravel() / 4.0
        return np.concatenate([own, opp, me_max, me_sum, you_max, you_sum,
                               globals_]).astype(np.float32)
    return np.concatenate([own, opp, lvl_me.ravel(), lvl_you.ravel(),
                           globals_]).astype(np.float32)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="full per-direction encoding instead of compact")
    ap.add_argument("--progress", type=int, default=2000)
    cfg = ap.parse_args(argv)

    X, V, P = [], [], []
    with open(cfg.inp) as fh:
        for i, line in enumerate(fh):
            if cfg.limit and len(X) >= cfg.limit:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "value_stm" not in r or "policy" not in r:
                continue
            X.append(encode([tuple(m) for m in r["moves"]], r["stm"],
                            compact=not cfg.full))
            V.append(r["value_stm"])
            P.append(r["policy"])
            if cfg.progress and len(X) % cfg.progress == 0:
                print(f"  encoded {len(X)}", file=sys.stderr, flush=True)

    Xa = np.asarray(X, dtype=np.float32)
    Va = np.asarray(V, dtype=np.float32)
    Pa = np.asarray(P, dtype=np.float32)
    np.savez_compressed(cfg.out, X=Xa, value=Va, policy=Pa)
    print(f"wrote {cfg.out}: X{Xa.shape} value{Va.shape} policy{Pa.shape}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
