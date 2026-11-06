#!/usr/bin/env python3
"""Piskvork-protocol adapter around online_bot_heuristic.choose_move.

Runs the pure-heuristic bot as a Gomocup engine for tools/match.py.
BOARD fields here are RELATIVE (1 = own stone, 2 = opponent), per the Gomocup
spec that match.py implements; the heuristic is colour-blind so we feed it
relative colours directly.

Usage:
    python3 match_heuristic.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import online_bot_heuristic2 as hb  # noqa: E402

N = hb.N


def main() -> int:
    own, opp = set(), set()   # absolute for board rebuild; relative colours come in
    board = [[0] * N for _ in range(N)]
    player = 1  # relative: our colour in the heuristic's eyes
    ply = 0

    def reply(s: str) -> None:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    def rebuild() -> list[list[int]]:
        b = [[0] * N for _ in range(N)]
        for (r, c) in own:
            b[r][c] = 1
        for (r, c) in opp:
            b[r][c] = 2
        return b

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("START"):
            own.clear(); opp.clear(); ply = 0
            reply("OK")
        elif up.startswith("INFO"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].upper() == "TIMEOUT_TURN":
                pass  # heuristic is ~instant; no time management needed
        elif up.startswith("ABOUT"):
            reply('name="Heuristic-Full", version="0.1", author="oop", country=""')
        elif up.startswith("RESTART"):
            own.clear(); opp.clear(); ply = 0
            reply("OK")
        elif up.startswith("BEGIN"):
            b = rebuild()
            info = {"game_no": "m", "ply": ply, "time_budget": None,
                    "total_left": None, "moves": []}
            r, c = hb.choose_move(b, player, info)
            own.add((r, c)); ply += 1
            reply(f"{c},{r}")          # protocol: x=col, y=row
        elif up.startswith("TURN"):
            try:
                xs, ys = line.split(None, 1)[1].split(",")
                opp.add((int(ys), int(xs)))  # store as (row, col)
            except (IndexError, ValueError):
                reply("ERROR bad TURN"); continue
            b = rebuild()
            info = {"game_no": "m", "ply": ply, "time_budget": None,
                    "total_left": None, "moves": []}
            r, c = hb.choose_move(b, player, info)
            own.add((r, c)); ply += 1
            reply(f"{c},{r}")
        elif up.startswith("BOARD"):
            own.clear(); opp.clear()
            for bl in sys.stdin:
                t = bl.strip()
                if not t or t.upper().startswith("DONE"):
                    break
                try:
                    xs, ys, fs = t.split(",")
                    x, y, f = int(xs), int(ys), int(fs)
                except ValueError:
                    continue
                (own if f == 1 else opp).add((y, x))  # relative: 1 = our stone
            ply = len(own) + len(opp)
            b = rebuild()
            info = {"game_no": "m", "ply": ply, "time_budget": None,
                    "total_left": None, "moves": []}
            r, c = hb.choose_move(b, player, info)
            own.add((r, c)); ply += 1
            reply(f"{c},{r}")
        elif up.startswith("END"):
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
