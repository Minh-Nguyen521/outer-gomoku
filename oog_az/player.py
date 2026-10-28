#!/usr/bin/env python3
"""Gomocup-protocol adapter, so a trained net plugs into tools/match.py.

Conventions follow tools/nn_player.py and tools/nn_mcts_player.py, which are
the working templates in this repo. Four protocol details here are load-bearing
and each has cost real games:

  * BOARD FIELDS ARE RELATIVE. 1 is the RECEIVING engine's own stone, 2 the
    opponent's -- not Black and White. tools/match.py sent absolute colours for
    a long time, which handed every White-playing opponent a colour-swapped
    position and invalidated Rounds 5-8 of docs/search-comparison.md.

  * REBUILD THE POSITION FROM SCRATCH ON EVERY BOARD. match.py re-sends the
    complete position every turn (protocol-legal; BOARD is documented as
    imposing an entirely new playing field). Appending instead is the single
    most repeated bug in this repo: SlowRenju lost 18 of 20 games to it, Chis
    deadlocked in 7 of 20, and Carbon crashed mid-game on it today.

  * FLUSH AFTER EVERY REPLY. A block-buffered stdout on a pipe looks exactly
    like a hung engine to the manager.

  * NEVER LET A STRAY LINE REACH STDOUT. Anything that is not MESSAGE/DEBUG/
    ERROR-prefixed is parsed as a move. Diagnostics go to stderr, which the
    manager discards.

Usage (as an engine for tools/match.py):
    oog_az/player.py --model oog_az/runs/dev/best.pt --simulations 400

    python3 tools/match.py --games 20 --parallel 1 --paired-openings --rule 0 \\
        --timeout-turn 10000 \\
        --engine-a "python3 oog_az/player.py --model oog_az/runs/dev/best.pt" \\
        --engine-b third_party/rapfi-8t/pbrain-rapfi
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oog_az.env import BLACK, N, Board, WHITE          # noqa: E402
from oog_az.mcts import MCTS                            # noqa: E402
from oog_az.net import load_checkpoint, pick_device     # noqa: E402


def build_board(own: set, opp: set) -> Board:
    """Reconstruct a Board from the two relative stone sets.

    Colour inference matches tools/katagomo_bridge.py: it is always OUR turn
    when the manager asks, so equal stone counts mean we are the first player
    (Black); one extra opponent stone means we are White.

    `history` is filled with the occupied squares in arbitrary order. Only its
    LENGTH matters -- it is what tells legal_mask whether the outer-ring
    restriction is still in force -- and the protocol does not give move order
    anyway.
    """
    board = Board()
    we_are = BLACK if len(own) == len(opp) else WHITE
    for (x, y) in own:
        board.grid[y, x] = we_are
    for (x, y) in opp:
        board.grid[y, x] = we_are ^ 1
    board.turn = we_are
    board.history = [y * N + x for (x, y) in (own | opp)]
    board._empty = N * N - len(own) - len(opp)
    return board


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="oog_az/runs/dev/best.pt")
    ap.add_argument("--simulations", type=int, default=400)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="oog-az")
    cfg = ap.parse_args(argv)

    device = pick_device(cfg.device)
    net, meta = load_checkpoint(cfg.model, device)
    mcts = MCTS(net, device, simulations=cfg.simulations)

    own: set[tuple[int, int]] = set()
    opp: set[tuple[int, int]] = set()

    def reply(s: str) -> None:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    def note(s: str) -> None:
        # stderr, never stdout -- see the module docstring.
        sys.stderr.write(s + "\n")
        sys.stderr.flush()

    def choose() -> tuple[int, int]:
        board = build_board(own, opp)
        # No root noise in competition: it exists to diversify training data.
        action, _ = mcts.choose(board, temperature=0.0, add_noise=False)
        return action % N, action // N

    def move_and_reply() -> None:
        try:
            x, y = choose()
        except Exception as exc:                 # never die mid-game
            note(f"search failed: {exc!r}")
            free = [(x, y) for y in range(N) for x in range(N)
                    if (x, y) not in own and (x, y) not in opp]
            if not free:
                reply("ERROR no legal move")
                return
            x, y = free[len(free) // 2]
        own.add((x, y))
        reply(f"{x},{y}")

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        up = line.upper()

        if up.startswith("START"):
            own.clear(); opp.clear()
            parts = line.split()
            size = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else N
            if size != N:
                reply(f"ERROR only {N}x{N} is supported")
            else:
                reply("OK")
        elif up.startswith("RESTART"):
            own.clear(); opp.clear()
            reply("OK")
        elif up.startswith("ABOUT"):
            it = meta.get("iteration", "?")
            reply(f'name="{cfg.name}", version="0.1", iteration="{it}", '
                  f'author="oog_az", country="unknown"')
        elif up.startswith("INFO"):
            pass                                  # fixed simulation budget
        elif up.startswith("BEGIN"):
            own.clear(); opp.clear()
            move_and_reply()
        elif up.startswith("TURN"):
            try:
                xs, ys = line.split(None, 1)[1].split(",")
                opp.add((int(xs), int(ys)))
            except (IndexError, ValueError):
                reply("ERROR bad TURN")
                continue
            move_and_reply()
        elif up.startswith("BOARD"):
            own.clear(); opp.clear()             # rebuild, never append
            for bl in sys.stdin:
                b = bl.strip()
                if not b or b.upper().startswith("DONE"):
                    break
                try:
                    xs, ys, fs = b.split(",")
                    x, y, f = int(xs), int(ys), int(fs)
                except ValueError:
                    continue
                if 0 <= x < N and 0 <= y < N:
                    (own if f == 1 else opp).add((x, y))
            move_and_reply()
        elif up.startswith("END"):
            return 0
        else:
            note(f"unknown command: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
