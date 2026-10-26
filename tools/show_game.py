#!/usr/bin/env python3
"""Play one game between two engines and draw the board in the terminal.

tools/match.py records only the *result* of each game ("A  5-in-a-row at
(9,4)"), which is all a strength measurement needs but leaves nothing to look
at afterwards. This plays a single game and renders it, so a matchup can be
inspected rather than only tallied -- useful for checking that an engine is
playing sensibly rather than, say, winning on forfeits.

It drives the game loop itself (mirroring match.play_one_game) instead of
calling it, because play_one_game returns (idx, result, reason) and discards
the board. Protocol details -- relative BOARD fields, MESSAGE/DEBUG
filtering, the 120s init allowance -- come from match.py's Brain/request_move
so there is one implementation, not two that can drift.

Usage:
    tools/show_game.py --engine-a <spec> --engine-b <spec> [--every 10]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match  # noqa: E402  (path set above)

N = 15
GLYPH = {"B": "X", "W": "O", None: "."}


def render(moves, highlight=(), title="", last=None):
    """Board as text. `moves` is [(r, c, colour)] in play order."""
    cells = {(r, c): col for r, c, col in moves}
    out = []
    if title:
        out.append(title)
    out.append("     " + " ".join(f"{c:X}" for c in range(N)))
    for r in range(N):
        row = [f"{r:3d}  "]
        for c in range(N):
            g = GLYPH[cells.get((r, c))]
            if (r, c) in highlight:
                g = f"\033[1;31m{g}\033[0m"      # winning five
            elif last == (r, c):
                g = f"\033[1;33m{g}\033[0m"      # most recent move
            row.append(g + " ")
        out.append("".join(row).rstrip())
    return "\n".join(out)


def winning_line(moves, r, c):
    """The five-or-more cells through (r,c), or empty if none."""
    colour = {(rr, cc): col for rr, cc, col in moves}
    me = colour.get((r, c))
    if me is None:
        return set()
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        line = [(r, c)]
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while colour.get((rr, cc)) == me:
                line.append((rr, cc))
                rr += dr * sign
                cc += dc * sign
        if len(line) >= 5:
            return set(line)
    return set()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--every", type=int, default=0,
                    help="also draw a snapshot every N plies")
    ap.add_argument("--timeout-turn", type=int, default=5000)
    ap.add_argument("--timeout-match", type=int, default=300000)
    ap.add_argument("--rule", type=int, default=0)
    ap.add_argument("--game", type=int, default=0,
                    help="game index; even = A plays Black")
    cfg = ap.parse_args(argv)

    mcfg = match.parse_args([
        "--engine-a", cfg.engine_a, "--engine-b", cfg.engine_b,
        "--games", "1", "--rule", str(cfg.rule),
        "--timeout-turn", str(cfg.timeout_turn),
        "--timeout-match", str(cfg.timeout_match),
    ])
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    idx = cfg.game
    color_a = "B" if idx % 2 == 0 else "W"
    # Brain wants an argv list, and oog_piskvork additionally needs the
    # opening book and any --nodes budget spliced in -- build_engine_cmd does
    # both. Passing the raw spec string makes Popen look for a file literally
    # named "<path> --engine search".
    cmds = {"B": match.build_engine_cmd(
                cfg.engine_a if color_a == "B" else cfg.engine_b, mcfg),
            "W": match.build_engine_cmd(
                cfg.engine_a if color_a == "W" else cfg.engine_b, mcfg)}
    specs = {"B": " ".join(cmds["B"]), "W": " ".join(cmds["W"])}
    brains = {s: match.Brain(cmds[s], cwd) for s in ("B", "W")}
    board = match.Board()
    played = []          # (r, c, colour) for rendering

    try:
        for side in ("B", "W"):
            b = brains[side]
            b.send("START 15")
            if (b.recv(120.0) or "").strip().upper() != "OK":
                print(f"{side} failed to init", file=sys.stderr)
                return 1
            b.send(f"INFO rule {cfg.rule}")
            if cfg.timeout_turn > 0:
                b.send(f"INFO timeout_turn {cfg.timeout_turn}")
            if cfg.timeout_match > 0:
                b.send(f"INFO timeout_match {cfg.timeout_match}")

        mover, moves, reason = "B", 0, ""
        while True:
            first = (moves == 0)
            mv, err = match.request_move(brains[mover], board, None, mcfg,
                                         first, cfg.timeout_turn, mover)
            if mv is None:
                reason = err or "no move"
                break
            r, c = mv
            if not board.play(r, c, mover):
                reason = f"illegal move ({r},{c})"
                break
            played.append((r, c, mover))
            moves += 1
            if cfg.every and moves % cfg.every == 0:
                print(render(played, title=f"--- after {moves} plies ---",
                             last=(r, c)))
                print()
            if board.win_at(r, c):
                reason = f"{mover} five-in-a-row at ({r},{c})"
                break
            if board.full():
                reason = "draw (board full)"
                break
            mover = "W" if mover == "B" else "B"
    finally:
        for b in brains.values():
            b.kill()

    hl = winning_line(played, *played[-1][:2]) if played else set()
    last = played[-1][:2] if played else None
    print(render(played, highlight=hl,
                 title=f"--- final ({len(played)} plies) ---", last=last))
    who = {"B": specs["B"], "W": specs["W"]}
    print(f"\nX = Black : {who['B']}")
    print(f"O = White : {who['W']}")
    print(f"result: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
