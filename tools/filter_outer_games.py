#!/usr/bin/env python3
"""Filter Gomocup PSQ archives down to games legal under Outer-Open rules.

Outer-Open (this repo's game, oog_az/env.py):
  * 15x15 board, freestyle rules (five OR MORE in a row wins, no forbidden moves)
  * Black's first stone must lie in the outer two rows or columns ("ring")

Gomocup archives (gomocup.org/static/tournaments/<year>/results/) contain PSQ
files, one game each:

    Piskvorky 15x15, 11:11, 0      <- header: board size, origin, TZ flag
    12,14,0                        <- x,y,time_ms for every stone incl. opening
    ...
    STARPOINT.zip                  <- engine binary names + result code
    RAPFI25.zip

A game is KEPT only if every check passes:

  1. header declares 15x15
  2. dir/league is freestyle (default: Freestyle15_*) -- Standard is exactly-5
     (overlines do NOT win there), so it is a different game
  3. every move is in-bounds and on an empty square
  4. move 1 (Black) lies on the outer ring            <- the Outer-Open rule
  5. no five-in-a-row exists before the final move (game did not continue
     past a win)
  6. the final move completes five-or-more for its colour (a real win, not a
     timeout/forfeit ending)

Rejection reasons are counted and reported. Openings in Gomocup are mostly
central, so the ring test rejects most games -- that is expected and is
exactly the point: what survives is a free Outer-Open-legal dataset.

Output:
  --out DIR   copies matching .psq files into DIR preserving <year>/<league>/
  --json PATH optional JSONL: {"file", "year", "league", "winner", "moves",
               "result_code"}
  --keep-dir  additional league globs to scan (default Freestyle15_*)

Usage:
    python tools/filter_outer_games.py /tmp/gomocup2024 /tmp/gomocup2025 \
        /tmp/gomocup2026 --out data/outer_gomocup
"""

import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys

N = 15
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def is_ring(r, c):
    """Must match oog_az/env.py Board.is_ring exactly."""
    return r < 2 or r >= N - 2 or c < 2 or c >= N - 2


def has_five(board, r, c):
    """Five-or-more through (r,c). Freestyle: overlines win."""
    colour = board[r][c]
    for dr, dc in DIRS:
        count = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < N and 0 <= cc < N and board[rr][cc] == colour:
                count += 1
                rr += dr * sign
                cc += dc * sign
        if count >= 5:
            return True
    return False


def parse_psq(path):
    """Return (moves, result_code, ok, reason). moves = [(x, y), ...].

    Move lines come first, all '<int>,<int>,...' with the third field the
    thinking time. Everything after the first non-move line (engine zips,
    result codes like '-1' or '2,League') is metadata.
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    header = lines[0] if lines else ""
    m = re.match(r"Piskvorky (\d+)x(\d+)", header)
    if not m or (int(m.group(1)), int(m.group(2))) != (N, N):
        return None, None, False, "wrong-size"

    moves = []
    result_code = None
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        parts = s.split(",")
        if len(parts) >= 2 and all(p.lstrip("-").isdigit() for p in parts[:2]):
            if len(parts) >= 3 and not parts[2].lstrip("-").isdigit():
                return None, None, False, "malformed"
            moves.append((int(parts[0]), int(parts[1])))
        else:
            # first metadata line: result code is either '-1' or '2,<league>'
            if s == "-1" or re.match(r"^-?\d+(,.*)?$", s):
                result_code = s
            break
    return moves, result_code, True, None


def classify(moves):
    """Apply Outer-Open checks. Return (ok, reason, winner) or (False, why, None)."""
    board = [[None] * N for _ in range(N)]

    for i, (x, y) in enumerate(moves):
        if not (1 <= x <= N and 1 <= y <= N):
            return False, "out-of-bounds", None
        r, c = y - 1, x - 1          # PSQ is 1-based, origin top-left
        if board[r][c] is not None:
            return False, "occupied", None
        if i == 0 and not is_ring(r, c):
            return False, "not-ring-first", None

        player = i % 2               # 0 = Black
        board[r][c] = player

        won = has_five(board, r, c)
        last = i == len(moves) - 1
        if won and not last:
            return False, "continued-past-win", None
        if not won and last:
            return False, "ended-without-five", None   # timeout / forfeit / draw
        if won and last:
            return True, "ok", player

    return False, "empty", None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archives", nargs="+", help="extracted gomocupYYYY results dirs")
    ap.add_argument("--leagues", default="Freestyle15_*",
                    help="league dir glob(s), comma-separated (default Freestyle15_*)")
    ap.add_argument("--out", help="directory to copy matching .psq files into")
    ap.add_argument("--json", dest="jsonl", help="optional JSONL index of matches")
    args = ap.parse_args()

    stats = collections.Counter()
    kept = 0
    records = []

    for archive in args.archives:
        year = re.search(r"(20\d\d)", os.path.basename(os.path.normpath(archive)))
        year = year.group(1) if year else "????"
        for pattern in args.leagues.split(","):
            pattern = pattern.strip()
            for league_dir in sorted(glob.glob(os.path.join(archive, pattern))):
                league = os.path.basename(league_dir)
                for path in sorted(glob.glob(os.path.join(league_dir, "*.psq"))):
                    stats["scanned"] += 1
                    moves, code, ok, reason = parse_psq(path)
                    if not ok:
                        stats[reason] += 1
                        continue
                    good, why, winner = classify(moves)
                    if not good:
                        stats[why] += 1
                        continue
                    kept += 1
                    stats[f"kept:{league}"] += 1
                    rec = {
                        "file": path,
                        "year": year,
                        "league": league,
                        "moves": moves,
                        "num_moves": len(moves),
                        "winner": "B" if winner == 0 else "W",
                        "result_code": code,
                    }
                    records.append(rec)
                    if args.out:
                        dst = os.path.join(args.out, year, league,
                                           os.path.basename(path))
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(path, dst)

    if args.jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.jsonl)), exist_ok=True)
        with open(args.jsonl, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    total = stats["scanned"]
    print(f"scanned            : {total}")
    print(f"kept (Outer-Open)  : {kept} ({100.0 * kept / max(total, 1):.2f}%)")
    print("rejected:")
    for reason in ("wrong-size", "malformed", "out-of-bounds", "occupied",
                   "not-ring-first", "continued-past-win",
                   "ended-without-five", "empty"):
        if stats[reason]:
            print(f"  {reason:<20}: {stats[reason]:>6}"
                  f" ({100.0 * stats[reason] / max(total, 1):.1f}%)")
    for k in sorted(k for k in stats if k.startswith("kept:")):
        print(f"{k}: {stats[k]}")
    if args.out:
        print(f"copied to: {args.out}")
    if args.jsonl:
        print(f"index   at: {args.jsonl}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
