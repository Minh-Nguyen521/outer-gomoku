#!/usr/bin/env python3
"""Reconstruct finished games from KataGo's GTP log, to inspect a match that is
still running (or one that was stopped before match.py printed its summary).

`match.py` accumulates silently and reports only at the end, so a long run is
opaque while in flight. KataGomo's own log is not: `oog.cfg` sets
`logAllGTPCommunication = true`, which records every controller command and
every engine reply, and that is enough to rebuild each game.

How the rebuild works, and why the obvious version is wrong
-----------------------------------------------------------
`tools/katagomo_bridge.py` re-sends the entire position every turn
(clear_board + play*) rather than playing moves incrementally, because the
Gomocup BOARD message is a full-position message and replaying it cannot
desync. So the log contains one complete rebuild per turn, each one move longer
than the last.

The first version of this script kept the LONGEST rebuild and appended KataGo's
replies to it. That silently corrupts long games: the base list goes stale while
replies keep stacking onto it, so moves land on occupied points. It reported a
99-ply game that held only 83 distinct stones, with a five for BOTH colours --
impossible in a real game, and the tell that caught the bug.

The correct rule is to keep the LAST rebuild seen before the final genmove, then
append the reply to that genmove. Every game this produces satisfies
`len(moves) == len(distinct stones)`; the script asserts that and flags any game
that fails it rather than scoring it.

Determining the winner
----------------------
If the final board holds a five, whoever owns it won -- freestyle, so five OR
MORE counts. If it holds none, the game ended on the very next move, which the
opponent played and KataGo therefore never logged, so the opponent won.

That inference cannot tell a genuine opponent win from a game that is simply
still in progress, and it cannot see draws or forfeits. The last game in a live
log is normally the one being played; it will show no five and be reported as
IN PROGRESS on ply count alone. `match.py`'s own summary remains authoritative.

Usage:
    tools/replay_kata_log.py kata.log ["YYYY-MM-DD HH:MM:SS" start cutoff]
"""

import re
import sys

SIZE = 15
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # GTP skips 'I'


def to_xy(vertex):
    """GTP vertex -> (x, y), zero-indexed from the top-left."""
    v = vertex.strip().upper()
    if v in ("PASS", "RESIGN", ""):
        return None
    try:
        return LETTERS.index(v[0]), SIZE - int(v[1:])
    except (ValueError, IndexError):
        return None


def has_five(board, colour):
    """Freestyle: five or more in a row wins, so an overline counts."""
    for y in range(SIZE):
        for x in range(SIZE):
            if board.get((x, y)) != colour:
                continue
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                n, cx, cy = 1, x + dx, y + dy
                while board.get((cx, cy)) == colour:
                    n += 1
                    cx, cy = cx + dx, cy + dy
                if n >= 5:
                    return True
    return False


def parse(path, start=None):
    ts = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
    play = re.compile(r"Controller: play ([BW]) (\S+)")
    gen = re.compile(r"Controller: genmove ([BW])")
    resp = re.compile(r"^\S+ \S+ = (\S+)\s*$")

    # One katago PROCESS is one game. Do NOT split on "Controller: quit":
    # it is emitted once per TWO games (32 process starts against 16 quits in
    # a measured run), so splitting on it merges games in pairs and scores only
    # the second of each -- which silently halves the sample and biases it to
    # one colour assignment.
    launch = re.compile(r"Started, ready to begin|GTP ready")

    # Each entry is (moves, kata_colour). The colour is essential: a five on
    # the final board does NOT imply KataGomo won it. Scoring every five as a
    # KataGomo win reported a 31-0 sweep for a match whose real interim score
    # was 18-1, because a KataGomo LOSS also leaves a five on the board -- the
    # opponent's.
    games, cur, snap, pending, final = [], [], None, None, None
    kata_colour = None
    live = start is None

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = ts.match(line)
            if not m:
                continue
            if not live:
                if m.group(1) < start:
                    continue
                live = True

            if launch.search(line):
                if final:
                    games.append((final, kata_colour))
                cur, snap, pending, final = [], None, None, None
                kata_colour = None
                continue
            if "Controller: clear_board" in line:
                cur = []
                continue
            p = play.search(line)
            if p:
                xy = to_xy(p.group(2))
                if xy:
                    cur.append((p.group(1), xy))
                continue
            g = gen.search(line)
            if g:
                snap, pending = cur[:], g.group(1)
                continue
            r = resp.match(line)
            if r and pending is not None:
                xy = to_xy(r.group(1))
                if xy:
                    # The first genmove of a run is a probe on an EMPTY board
                    # and is always "genmove B" whatever colour KataGomo has.
                    # Only a genmove with stones already down reveals the real
                    # colour.
                    if kata_colour is None and snap:
                        kata_colour = pending
                    if kata_colour is not None:
                        final = snap + [(pending, xy)]
                pending = None
                continue
    if final:
        games.append((final, kata_colour))
    return games


def show(board):
    out = ["    " + " ".join(f"{x:2d}" for x in range(SIZE))]
    for y in range(SIZE):
        row = " ".join(
            " ." if board.get((x, y)) is None
            else (" X" if board[(x, y)] == "B" else " O")
            for x in range(SIZE)
        )
        out.append(f"{y:2d}  {row}")
    return "\n".join(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "kata.log"
    start = sys.argv[2] if len(sys.argv) > 2 else None
    games = parse(path, start)

    print(f"reconstructed games: {len(games)}\n")
    kata = opp = corrupt = 0
    for i, (moves, kcol) in enumerate(games, 1):
        board = {}
        for colour, xy in moves:
            board[xy] = colour
        dup = len(moves) - len(board)
        fb, fw = has_five(board, "B"), has_five(board, "W")
        winner = "B" if fb else ("W" if fw else None)

        if dup or (fb and fw) or kcol is None:
            tag, corrupt = "CORRUPT", corrupt + 1
        elif winner is not None:
            # Whose five is it? A five belonging to the OTHER colour is a
            # KataGomo loss, not a win.
            if winner == kcol:
                tag, kata = "KataGomo", kata + 1
            else:
                tag, opp = "opponent", opp + 1
        elif i == len(games):
            tag = "IN PROGRESS (not a result)"
        else:
            # No five in KataGomo's view: the opponent played the winning move
            # after KataGomo's last turn, so it never reached this log.
            tag, opp = "opponent (five not logged)", opp + 1

        print(f"  game {i:2d}: {len(moves):3d} plies, {len(board):3d} stones, "
              f"dup={dup:2d}, kata={kcol or '?'}, five B/W={int(fb)}/{int(fw)}"
              f"  -> {tag}")

    print(f"\nKataGomo {kata}  -  opponent {opp}   corrupt {corrupt}")
    if corrupt:
        print("CORRUPT games mean the parse is wrong, not that the game was.")


if __name__ == "__main__":
    main()
