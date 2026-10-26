#!/usr/bin/env python3
"""Convert KataGo's GTP log into the --live-file format, so a KataGomo match
started WITHOUT --live-file can still be watched in the viewers.

`tools/match.py --live-file` did not exist when the long KataGomo matches were
launched, and match.py prints nothing until its final summary. KataGomo,
however, logs every controller command and every reply (oog.cfg sets
logAllGTPCommunication = true), which is enough to reconstruct the games and
re-emit them in the viewer's own JSON-lines format.

    tools/kata_log_to_live.py kata.log /tmp/live.jsonl --since "2026-09-10 09:27:03" --follow
    tools/watch_match_web.py /tmp/live.jsonl

With --follow it keeps reading as the match plays, so the board updates live.

Two details this has to get right, both learned the hard way:

* One katago PROCESS is one game. Splitting on "Controller: quit" is wrong --
  quit is emitted once per TWO games (32 process starts against 16 quits in a
  measured run), which silently halves the sample.
* The bridge re-sends the whole position every turn, so each genmove is
  preceded by a complete rebuild. The move list is that rebuild plus the reply
  to the genmove; taking the LONGEST rebuild instead of the LAST one corrupts
  long games by stacking replies onto a stale base.

Colours come from the genmove colour itself. The first genmove of a run is a
probe on an empty board and is skipped.
"""

import argparse
import json
import os
import re
import time

SIZE = 15
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # GTP skips 'I'

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
LAUNCH = re.compile(r"Started, ready to begin|GTP ready")
PLAY = re.compile(r"Controller: play ([BW]) (\S+)")
GEN = re.compile(r"Controller: genmove ([BW])")
RESP = re.compile(r"^\S+ \S+ = (\S+)\s*$")


def to_rc(vertex):
    """GTP vertex -> (row, col), matching match.py's (r, c) convention."""
    v = vertex.strip().upper()
    if v in ("PASS", "RESIGN", ""):
        return None
    try:
        return SIZE - int(v[1:]), LETTERS.index(v[0])
    except (ValueError, IndexError):
        return None


def has_five(board, colour):
    for (r, c), v in board.items():
        if v != colour:
            continue
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n, rr, cc = 1, r + dr, c + dc
            while board.get((rr, cc)) == colour:
                n += 1
                rr, cc = rr + dr, cc + dc
            if n >= 5:
                return True
    return False


class Emitter:
    def __init__(self, out_path, kata_name, opp_name):
        self.out = out_path
        self.kata = kata_name
        self.opp = opp_name
        self.game = -1
        self.emitted_pos = set()   # squares already written for this game
        self.n = 0                 # ply counter for this game
        self.kata_colour = None

    def write(self, rec):
        with open(self.out, "a") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close_game(self, moves):
        if self.game < 0 or not moves:
            return
        board = {}
        for colour, rc in moves:
            board[rc] = colour
        last = moves[-1][0]
        if has_five(board, last):
            winner, reason = last, "5-in-a-row (from GTP log)"
        elif has_five(board, "B") or has_five(board, "W"):
            winner = "B" if has_five(board, "B") else "W"
            reason = "5-in-a-row (from GTP log)"
        else:
            # No five in KataGomo's view: the opponent played the winning move
            # after KataGomo's last turn, so it never reached this log.
            winner = "W" if last == "B" else "B"
            reason = "opponent five (not in KataGomo's log)"
        result = "B" if winner == self.kata_colour else "A"
        self.write({"t": "end", "game": self.game, "winner": winner,
                    "result": result, "reason": reason, "plies": len(moves)})

    def start_game(self, colour):
        self.game += 1
        self.kata_colour = colour
        self.emitted_pos = set()
        self.n = 0
        black = self.kata if colour == "B" else self.opp
        white = self.opp if colour == "B" else self.kata
        self.write({"t": "start", "game": self.game,
                    "black": black, "white": white,
                    "a_color": "B" if black == self.opp else "W",
                    "opening": None})

    def sync(self, moves):
        """Emit stones that are new since the last rebuild.

        This CANNOT be a tail diff. match.py sends BOARD by scanning the grid
        row by row, so each rebuild is the position ordered by square, not the
        order the moves were played. A stone landing in an early row shifts
        every later entry, so `moves[len(emitted):]` returns whatever happens
        to sit at the end of the scan -- which re-emitted one square 15 times
        in a 30-move game and left the board almost entirely one colour.

        Diff by SQUARE instead. At most two stones appear per turn (the
        opponent's move and KataGomo's reply), and the reply is always the last
        element, so emit it last to keep the last-move marker correct.
        """
        if self.kata_colour is None and moves:
            self.start_game(moves[-1][0])
        reply = moves[-1] if moves else None
        fresh = [m for m in moves if m[1] not in self.emitted_pos
                 and m is not reply]
        if reply is not None and reply[1] not in self.emitted_pos:
            fresh.append(reply)
        for colour, (r, c) in fresh:
            self.write({"t": "move", "game": self.game, "n": self.n,
                        "side": colour, "r": r, "c": c, "ms": 0})
            self.emitted_pos.add((r, c))
            self.n += 1


def run(log_path, out_path, since, follow, kata_name, opp_name):
    em = Emitter(out_path, kata_name, opp_name)
    cur, snap, pending, final = [], None, None, []
    live = since is None
    pos = 0
    started = False

    while True:
        with open(log_path, "r", errors="replace") as fh:
            fh.seek(pos)
            for line in fh:
                if not line.endswith("\n"):
                    break
                pos += len(line.encode())
                m = TS.match(line)
                if not m:
                    continue
                if not live:
                    if m.group(1) < since:
                        continue
                    live = True

                if LAUNCH.search(line):
                    if started:
                        em.close_game(final)
                    started = True
                    cur, snap, pending, final = [], None, None, []
                    em.kata_colour = None
                    continue
                if "Controller: clear_board" in line:
                    cur = []
                    continue
                p = PLAY.search(line)
                if p:
                    rc = to_rc(p.group(2))
                    if rc:
                        cur.append((p.group(1), rc))
                    continue
                g = GEN.search(line)
                if g:
                    snap, pending = cur[:], g.group(1)
                    continue
                r = RESP.match(line)
                if r and pending is not None:
                    rc = to_rc(r.group(1))
                    if rc:
                        # The FIRST genmove of a run is a probe on an empty
                        # board and is always "genmove B", whichever colour
                        # KataGomo actually has. Taking the colour from it
                        # marks every game as Black and makes the winner
                        # alternate mechanically -- a plausible-looking 9-9
                        # that is pure artefact. A real turn always has stones
                        # on the board already, so require a non-empty snap.
                        if em.kata_colour is None:
                            if not snap:
                                pending = None
                                continue
                            em.start_game(pending)
                        final = snap + [(pending, rc)]
                        em.sync(final)
                    pending = None
        if not follow:
            break
        time.sleep(1.0)
    em.close_game(final)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kata_log")
    ap.add_argument("out")
    ap.add_argument("--since", default=None,
                    help='only games after this timestamp, "YYYY-MM-DD HH:MM:SS"')
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--kata-name", default="KataGomo")
    ap.add_argument("--opp-name", default="SlowRenju")
    a = ap.parse_args()
    if os.path.exists(a.out):
        os.remove(a.out)
    run(a.kata_log, a.out, a.since, a.follow, a.kata_name, a.opp_name)


if __name__ == "__main__":
    main()
