#!/usr/bin/env python3
"""Hybrid alpha-beta searcher: heuristic move generation + net evaluation.

Motivation (measured, 2026-11-04): pure neural play (Gumbel k up to 128 on
the pipeline nets) loses 0-20 to SlowRenju. The nets are small and the RL
budget tiny, so search must carry tactics the net cannot see. Classic
gomoku engines win via threat-pattern move generation -- this module steals
that skeleton and plugs the net in as the leaf evaluator, so:
  - move ordering = hand-coded threat patterns (four > three > adjacency)
  - terminal detection = 5-in-row / forced-move pruning at every node
  - leaf value = net value head (fast, positional)
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

N = 15
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


class HB:
    """Lightweight board with threat-aware candidate generation."""

    __slots__ = ("g", "n")

    def __init__(self, n=N):
        self.n = n
        self.g = np.zeros((n, n), dtype=np.int8)  # 1 black, 2 white

    def clone(self):
        b = HB.__new__(HB)
        b.n = self.n
        b.g = self.g.copy()
        return b

    def play(self, r, c):
        self.g[r, c] = 3 - int(self.g.sum() % 3)  # placeholder, unused
        raise NotImplementedError

    def put(self, r, c, color):
        self.g[r, c] = color

    def undo(self, r, c):
        self.g[r, c] = 0

    def wins(self, r, c):
        """Stone just placed at (r,c) makes five+ for its color."""
        n, g = self.n, self.g
        color = g[r, c]
        for dr, dc in DIRS:
            cnt = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < n and 0 <= cc < n and g[rr, cc] == color:
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < n and 0 <= cc < n and g[rr, cc] == color:
                cnt += 1
                rr -= dr
                cc -= dc
            if cnt >= 5:
                return True
        return False

    def line_pattern(self, r, c, dr, dc, color):
        """Count (own contiguous, open ends) along an axis through (r,c)."""
        n, g = self.n, self.g
        cnt, open_e = 1, 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < n and 0 <= cc < n and g[rr, cc] == color:
            cnt += 1
            rr += dr
            cc += dc
        if 0 <= rr < n and 0 <= cc < n and g[rr, cc] == 0:
            open_e += 1
        rr, cc = r - dr, c - dc
        while 0 <= rr < n and 0 <= cc < n and g[rr, cc] == color:
            cnt += 1
            rr -= dr
            cc -= dc
        if 0 <= rr < n and 0 <= cc < n and g[rr, cc] == 0:
            open_e += 1
        return cnt, open_e

    def move_score(self, r, c, me, opp):
        """Heuristic value of PLAYING (r,c) for color `me`, empty cell.

        Threat classes (classic): win5 > block5 > open4/block-open4 >
        four > open3/block-open3 > three > proximity.
        """
        s = 0.0
        win = False
        blocks_win = False
        open4 = four = open3 = 0
        blocks_o4 = blocks_4 = blocks_o3 = 0
        for dr, dc in DIRS:
            cnt, oe = self.line_pattern(r, c, dr, dc, me)
            if cnt >= 5:
                win = True
            elif cnt == 4 and oe == 2:
                open4 += 1
            elif cnt == 4:
                four += 1
            elif cnt == 3 and oe == 2:
                open3 += 1
            cnt2, oe2 = self.line_pattern(r, c, dr, dc, opp)
            if cnt2 >= 5:
                blocks_win = True
            elif cnt2 == 4 and oe2 == 2:
                blocks_o4 += 1
            elif cnt2 == 4:
                blocks_4 += 1
            elif cnt2 == 3 and oe2 == 2:
                blocks_o3 += 1
            # proximity so the net sees a locally connected position
        prox = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N and self.g[rr, cc]:
                    prox += 1
        if win:
            return 1e9
        if blocks_win:
            s = 5e8
        elif open4:
            s = 4e8
        elif blocks_o4:
            s = 3e8
        elif four:
            s = 2e8
        elif blocks_4:
            s = 1.5e8
        elif open3:
            s = 1e8
        elif blocks_o3:
            s = 5e7
        s += open3 * 1e7 + prox * 10
        return s

    def candidates(self, me, opp, width):
        """Top-`width` empty cells by heuristic score; forced moves first."""
        scored = []
        best_block = -1.0
        for r in range(self.n):
            row = self.g[r]
            for c in range(self.n):
                if row[c]:
                    continue
                s = self.move_score(r, c, me, opp)
                if s >= 5e8:  # any win/block-win outranks everything
                    scored.append(((r, c), s))
        if scored:
            scored.sort(key=lambda t: -t[1])
            return [rc for rc, _ in scored[:width]]
        # otherwise generate around stones only
        out = []
        for r in range(self.n):
            for c in range(self.n):
                if self.g[r, c]:
                    continue
                near = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.n and 0 <= cc < self.n \
                                and self.g[rr, cc]:
                            near = True
                            break
                    if near:
                        break
                if near:
                    out.append((r, c))
        if not out:
            out = [(self.n // 2, self.n // 2)]
        return out

    def forced(self, me, opp):
        """Return (r,c) if `me` has an immediate win, else None."""
        for r in range(self.n):
            for c in range(self.n):
                if self.g[r, c]:
                    continue
                if self.move_score(r, c, me, opp) >= 1e9:
                    return (r, c)
        return None
