#!/usr/bin/env python3
"""Live board viewer for tools/match.py -- watch games as they are played.

`match.py` is silent until the final summary, which makes a multi-hour run
opaque: you cannot tell a healthy match from one that is quietly forfeiting
every game until it is over. Six engines in docs/engine-ladder.md produced a
convincing 0-20 that turned out to be a setup fault, and every one of them
would have been obvious within two games if anyone had been able to see the
board.

Usage
-----
Run the match with a live file:

    tools/match.py --games 40 --timeout-turn 10000 --live-file /tmp/live.jsonl \\
        --engine-a ... --engine-b ...

Then, in another terminal:

    tools/watch_match.py /tmp/live.jsonl

It tails the file, so it can be started before, during, or after the match,
and it survives the match being restarted (a new "start" record for a lower
game index simply begins a new board).

Keys: none -- it is a passive viewer. Ctrl-C to quit.

Options
-------
    --replay SECONDS   pause between moves when catching up on an existing
                       file, so a finished game can be watched back rather
                       than appearing instantly. Default 0 (jump to live).
    --game N           follow only game N instead of whichever is newest.
    --no-color         plain ASCII, for logs and dumb terminals.
"""

import argparse
import json
import os
import sys
import time

SIZE = 15

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
BLACK_C = "\033[94m"    # blue X
WHITE_C = "\033[93m"    # yellow O
LAST_C = "\033[91m"     # red highlight for the move just played
RING_C = "\033[90m"


def is_ring(r, c):
    """Outer-Open: Black's first stone must sit in the outer TWO rows/cols."""
    return r < 2 or r >= SIZE - 2 or c < 2 or c >= SIZE - 2


class Game:
    def __init__(self, rec):
        self.idx = rec.get("game")
        self.black = rec.get("black", "?")
        self.white = rec.get("white", "?")
        self.board = {}
        self.last = None
        self.plies = 0
        self.times = {"B": [], "W": []}
        self.over = None
        self.reason = ""

    def move(self, rec):
        r, c, side = rec["r"], rec["c"], rec["side"]
        self.board[(r, c)] = side
        self.last = (r, c)
        self.plies = rec.get("n", self.plies + 1)
        ms = rec.get("ms", 0)
        if ms:
            self.times.setdefault(side, []).append(ms)

    def finish(self, rec):
        self.over = rec.get("winner")
        self.reason = rec.get("reason", "")


def short(name, width=34):
    """Engine command lines are long; keep the informative tail."""
    name = str(name)
    if len(name) <= width:
        return name
    return "..." + name[-(width - 3):]


def render(g, colour=True):
    def C(s, c):
        return f"{c}{s}{RESET}" if colour else s

    out = []
    head = f" game {g.idx}   ply {g.plies}"
    if g.over:
        head += f"   FINISHED: {g.over} ({g.reason})"
    out.append(C(head, BOLD) if colour else head)
    out.append(f"   {C('X', BLACK_C)} black  {short(g.black)}")
    out.append(f"   {C('O', WHITE_C)} white  {short(g.white)}")

    out.append("     " + " ".join(f"{x:2d}" for x in range(SIZE)))
    for y in range(SIZE):
        cells = []
        for x in range(SIZE):
            # Board keys are (r, c) and match.py:261 makes r the ROW, so the
            # lookup is (y, x), not (x, y). The transposed form renders a
            # legal-looking but mirrored board.
            v = g.board.get((y, x))
            if v is None:
                # Dim the outer two rows/cols: that is the Outer-Open ring,
                # the only legal squares for Black's first stone.
                cells.append(C(" .", RING_C) if (colour and is_ring(x, y))
                             else " .")
            else:
                ch = "X" if v == "B" else "O"
                if g.last == (y, x):
                    cells.append(C(f" {ch}", LAST_C))
                else:
                    cells.append(C(f" {ch}", BLACK_C if v == "B" else WHITE_C))
        out.append(f"  {y:2d} " + " ".join(cells))

    for side, label in (("B", "black"), ("W", "white")):
        t = g.times.get(side, [])
        if t:
            out.append(f"   {label} think: last {t[-1]/1000:5.2f}s   "
                       f"mean {sum(t)/len(t)/1000:5.2f}s   n={len(t)}")
    return "\n".join(out)


def tail(path, from_start):
    """Yield parsed records, following the file as it grows."""
    while not os.path.exists(path):
        time.sleep(0.3)
    with open(path, "r") as fh:
        if not from_start:
            fh.seek(0, os.SEEK_END)
        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.15)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A torn line from a writer mid-append; skip it.
                continue


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("live_file")
    ap.add_argument("--replay", type=float, default=0.0,
                    help="seconds to pause between moves while catching up")
    ap.add_argument("--game", type=int, default=None,
                    help="follow only this game index")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()

    colour = not a.no_color and sys.stdout.isatty()
    games = {}
    current = None
    wins = {"A": 0, "B": 0, "draw": 0}

    def draw():
        if current is None or current not in games:
            return
        if colour:
            sys.stdout.write("\033[H\033[J")
        g = games[current]
        tally = f"  score  A {wins['A']}  B {wins['B']}  draws {wins['draw']}"
        print(render(g, colour))
        print(tally)
        sys.stdout.flush()

    try:
        for rec in tail(a.live_file, from_start=True):
            t = rec.get("t")
            gi = rec.get("game")
            if a.game is not None and gi != a.game:
                continue
            if t == "start":
                games[gi] = Game(rec)
                current = gi
                draw()
            elif t == "move":
                g = games.get(gi)
                if g is None:
                    g = games[gi] = Game({"game": gi})
                g.move(rec)
                current = gi
                draw()
                if a.replay:
                    time.sleep(a.replay)
            elif t == "end":
                g = games.get(gi)
                if g:
                    g.finish(rec)
                res = rec.get("result")
                if res in wins:
                    wins[res] += 1
                current = gi
                draw()
                time.sleep(max(a.replay, 0.8))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
