#!/usr/bin/env python3
"""Regression check for Brain.recv() after the ERROR-tolerance change.

WHY. `match.py` used to treat any ERROR line as the engine's reply, which
forfeited the game. Yixin 2018 re-parses the stones in a repeated BOARD and
emits `ERROR opponents's move [7,0]` for one already on the board, then plays
a legal move on the very next line -- so a probe match came back 0-4 with
ZERO games decided by a five, the signature of a broken instrument rather than
a weak engine. recv() now records the ERROR and keeps reading.

That change sits on the path of every match this project runs, so it needs to
be shown to affect the ERROR case ONLY. Running 20 games would take ten
minutes and prove less: engine noise and the unrecorded time control of the
original 18-2 make it a weak comparison. Driving recv() directly with
synthetic line sequences is exact.

Usage: tools/test_recv_regression.py
"""

import os
import queue
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match  # noqa: E402


def make_brain(lines):
    """A Brain with a pre-filled queue and no subprocess."""
    b = match.Brain.__new__(match.Brain)   # bypass __init__/Popen
    b.q = queue.Queue()
    b.warnings = 0
    for ln in lines:
        b.q.put(ln)
    return b


CASES = [
    # (name, lines fed, expected recv() result, expected warnings, changed?)
    ("plain move",
     ["7,7\n"], "7,7", 0, False),
    ("MESSAGE then move (rapfi-style chatter)",
     ["MESSAGE Load config\n", "MESSAGE Speed=1000\n", "7,7\n"], "7,7", 0, False),
    ("DEBUG/SUGGEST skipped",
     ["DEBUG foo\n", "SUGGEST 1,1\n", "3,4\n"], "3,4", 0, False),
    ("blank lines skipped",
     ["\n", "  \n", "5,5\n"], "5,5", 0, False),
    ("ERROR then move (Yixin) -> takes the move",
     ["ERROR opponents's move [7,0]\n", "MESSAGE Speed=1046\n", "7,4\n"],
     "7,4", 1, True),
    ("ERROR with no move -> still forfeits, reports the ERROR",
     ["ERROR something fatal\n"], "ERROR something fatal", 1, False),
    ("UNKNOWN with no move -> still forfeits",
     ["UNKNOWN command\n"], "UNKNOWN command", 1, False),
    ("nothing at all -> None (timeout forfeit)",
     [], None, 0, False),
    ("EOF sentinel -> None (crash forfeit)",
     [None], None, 0, False),
]


def main():
    width = max(len(c[0]) for c in CASES)
    failures = 0
    print(f"{'case'.ljust(width)}  {'got':<26} {'warn':>4}  status")
    print("-" * (width + 44))
    for name, lines, expected, exp_warn, changed in CASES:
        b = make_brain(lines)
        got = b.recv(0.25)
        got_s = None if got is None else got.strip()
        ok = (got_s == expected) and (b.warnings == exp_warn)
        if not ok:
            failures += 1
        tag = "CHANGED (intended)" if changed else "unchanged"
        print(f"{name.ljust(width)}  {str(got_s):<26} {b.warnings:>4}  "
              f"{'ok' if ok else 'FAIL'}  {tag}")

    print()
    if failures:
        print(f"{failures} case(s) FAILED")
        return 1
    print("All cases pass. Only the 'ERROR then move' case changed behaviour;")
    print("every forfeit path (ERROR with no move, timeout, EOF) is preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
