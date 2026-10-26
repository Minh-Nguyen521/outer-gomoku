#!/usr/bin/env python3
"""Engine-vs-engine match runner for Outer-Open Gomoku (Gomocup protocol).

Spawns two engine processes per game and plays them against each other on
15x15 with the Outer-Open first-move rule (Black's first stone is placed on
the outer ring).  Fixed ring openings are played by the manager by default;
`--no-openings` lets the Black engine open with BEGIN instead.

`--parallel N` runs N games concurrently as independent process pairs (one
engine instance + seed each), which is the model the tournament table runner
uses; no shared state between games.

Deterministic node-count mode: pass `--nodes N` and the default engine builds
with it (`oog_piskvork --nodes N`) so games are reproducible and comparable
under load.  The manager enforces per-turn and whole-match clocks.

Usage examples:
  tools/match.py --games 20 --timeout-turn 2000
  tools/match.py --games 100 --parallel 8 --timeout-match 60000
  tools/match.py --games 40 --nodes 200000
  tools/match.py --engine-a "./build/release/bin/oog_piskvork --nodes 100000" \\
                 --engine-b "other_engine args"

External opponent (rapfi), for absolute strength rather than self-play:
  tools/rapfi_setup.sh                      # one-time: build + stage rapfi
  tools/match.py --games 100 --parallel 4 --timeout-turn 1000 \\
      --engine-b third_party/rapfi-bin/pbrain-rapfi

Outer-Open and freestyle Gomoku differ *only* in Black's first move, and this
runner plays that move itself from a fixed ring opening, so any freestyle
engine is a valid opponent from move 2 onward.  Keep --rule 0 (the default):
see its help text for why sending anything else silently changes the game.
"""

import argparse
import json
import os
import queue
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

N = 15
RING = [
    (r, c)
    for r in range(N)
    for c in range(N)
    if r < 2 or r >= N - 2 or c < 2 or c >= N - 2
]
# Outer-Open legal first moves: the outer TWO rows/columns, i.e. rows and
# columns 0, 1, 13, 14 on a 15x15 board. This must agree with the engine --
# src/bitboard.cpp:206 uses the identical test:
#     br < 2 || br >= N - 2 || bc < 2 || bc >= N - 2
#
# It previously did not. (2,7) and (12,7) are the THIRD row from each edge and
# are NOT legal Outer-Open openings, so roughly one game in seven of every
# match recorded before 2026-09-08 began from a position the rule forbids.
# They are replaced here by (7,14) and (13,6), which mirror the existing
# (7,0) and (1,6) and keep the spread. `python3 -c` over is_ring() is a cheap
# way to re-check this set after any edit.
DEFAULT_OPENINGS = [
    (0, 7), (0, 6), (0, 8), (1, 7), (1, 6), (1, 8), (7, 14), (0, 5),
    (7, 0), (13, 7), (14, 7), (14, 6), (14, 8), (13, 6),
]
UNLIMITED = 2147483647
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# Board model (runner-side rules)
# --------------------------------------------------------------------------
class Board:
    def __init__(self):
        self.g = [["."] * N for _ in range(N)]
        self.moves = []
        self.win_len = 0

    def play(self, r, c, color):
        if not (0 <= r < N and 0 <= c < N) or self.g[r][c] != ".":
            return False
        self.g[r][c] = color
        self.moves.append((r, c))
        return True

    def is_ring(self, r, c):
        return r < 2 or r >= N - 2 or c < 2 or c >= N - 2

    def win_at(self, r, c):
        """True if the stone at (r,c) completes five *or more*.

        `self.win_len` records the length, so a caller can tell a plain five
        from an overline. That distinction matters when the opponent plays
        Standard rules (overlines do NOT win) while this manager adjudicates
        freestyle (they do): every overline win would be awarded to us on a
        technicality the opponent never saw coming. Embryo's only Linux build
        is the Standard variant, so this is a live concern, not a hypothetical.
        """
        color = self.g[r][c]
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            cnt = 1
            for sign in (1, -1):
                rr, cc = r + dr * sign, c + dc * sign
                while 0 <= rr < N and 0 <= cc < N and self.g[rr][cc] == color:
                    cnt += 1
                    rr += dr * sign
                    cc += dc * sign
            if cnt >= 5:
                self.win_len = cnt
                return True
        return False

    def full(self):
        return len(self.moves) >= N * N

    def dump(self):
        lines = []
        for r in range(N):
            row = []
            for c in range(N):
                g = self.g[r][c]
                row.append("1" if g == "B" else "2" if g == "W" else "0")
            lines.append("".join(row))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Engine process wrapper
# --------------------------------------------------------------------------
class Brain:
    def __init__(self, cmd, cwd):
        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        self.q = queue.Queue()
        self.time_used = 0.0  # seconds spent thinking
        # ERROR/UNKNOWN lines the engine emitted but recovered from. Surfaced
        # at the end of a match: an engine that plays on after complaining is
        # usable, but a high count means the harness and the engine disagree
        # about the protocol and the result deserves a second look.
        self.warnings = 0
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line.rstrip("\n"))
        except Exception:
            pass
        self.q.put(None)  # EOF sentinel

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    # Informational lines a brain may emit at any time. Per the Gomocup
    # protocol these are never responses and the manager must ignore them.
    INFO_PREFIXES = ("MESSAGE", "DEBUG", "SUGGEST")

    def recv(self, timeout_s):
        """Next protocol *response*, skipping informational chatter.

        Engines differ a lot here: this project's own brain stays quiet, but
        rapfi prints two MESSAGE lines at startup and several per move. A
        runner that takes the first line as the answer reads
        "MESSAGE Load config..." as the reply to START, fails the OK check,
        and forfeits every game on its opponent's behalf -- which looks
        exactly like winning 6-0.
        """
        deadline = time.monotonic() + timeout_s
        pending_error = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Nothing playable arrived. If the engine complained on the
                # way, report that rather than a bare timeout.
                return pending_error
            try:
                line = self.q.get(timeout=remaining)
            except queue.Empty:
                return pending_error
            if line is None:
                return pending_error  # EOF sentinel
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith(self.INFO_PREFIXES):
                continue
            # An ERROR is not always fatal. Yixin 2018 re-parses the stones in
            # a repeated BOARD as fresh moves and emits
            #     ERROR opponents's move [7,0]
            # for one already on the board -- then plays a perfectly legal
            # move on the next line. Treating the first line as the answer
            # forfeited all four games of a probe match with 0 wins decided by
            # a five, which is the signature of a broken instrument rather
            # than a weak engine.
            #
            # So: remember it and keep reading. If a move follows, take the
            # move and count a warning; if nothing playable arrives before the
            # deadline, the recorded ERROR is returned and the game is
            # forfeited exactly as before. A genuinely broken engine still
            # fails -- it just fails on "produced no move" rather than on
            # "said something alarming".
            if s.upper().startswith(("ERROR", "UNKNOWN")):
                pending_error = line
                self.warnings += 1
                continue
            return line

    def kill(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.kill()
        self.proc.wait(timeout=5)


# --------------------------------------------------------------------------
# Game driver
# --------------------------------------------------------------------------
def build_engine_cmd(spec, cfg):
    parts = shlex.split(spec)
    base = os.path.basename(parts[0])
    if "oog_piskvork" not in base:
        return parts
    args = []
    book = os.path.join(REPO, "build", "release", "bin", "openings.txt")
    if os.path.exists(book):
        args.append(book)
    if cfg.nodes:
        args += ["--nodes", str(cfg.nodes)]
    return parts[:1] + args + parts[1:]


def parse_reply(line):
    if line is None:
        return None, "no reply (timeout or crash)"
    s = line.strip()
    if not s or s.upper().startswith("ERROR") or s.upper().startswith("UNKNOWN"):
        return None, f"bad reply: {s!r}"
    comma = s.find(",")
    if comma < 0:
        return None, f"bad reply: {s!r}"
    try:
        x, y = int(s[:comma]), int(s[comma + 1 :])
    except ValueError:
        return None, f"bad reply: {s!r}"
    return (y, x), None  # protocol x,y -> (r,c)


def request_move(brain, board, opening, cfg, first, turn_budget_override=0,
                 mover="B"):
    # Whole-match clock: forfeit if already out of time.
    if cfg.timeout_match > 0:
        remaining = max(0, cfg.timeout_match - int(brain.time_used * 1000))
        if remaining <= 0:
            return None, "out of time (match clock)"
        brain.send(f"INFO time_left {remaining}")
        turn_budget = cfg.timeout_turn if cfg.timeout_turn > 0 else min(
            cfg.timeout_match, 5000
        )
    else:
        brain.send(f"INFO time_left {UNLIMITED}")
        turn_budget = cfg.timeout_turn if cfg.timeout_turn > 0 else 5000

    if first and opening is None:
        brain.send("BEGIN")
    else:
        brain.send("BOARD")
        # Gomocup BOARD fields are RELATIVE to the receiving engine:
        #   1 = the engine's own stone, 2 = the opponent's.
        # They are NOT absolute colours. Sending 1 for Black regardless of who
        # is asking hands every White-playing engine a colour-swapped board,
        # so it plays as though it were Black. This project's own brain reads
        # the field as an absolute colour, so self-play never exposed it --
        # only a strict external engine did (AlphaGomoku rejects the position
        # outright: "too many stones of either color"). Engines that do not
        # validate simply play the swapped position badly, which is almost
        # certainly why every opponent looked so weak as White.
        for r in range(N):
            for c in range(N):
                g = board.g[r][c]
                if g == ".":
                    continue
                field = 1 if g == mover else 2
                brain.send(f"{c},{r},{field}")
        brain.send("DONE")

    t0 = time.time()
    # The engine is entitled to use its full turn budget; grant a grace period
    # for pipe latency and clock skew before forfeiting.
    if turn_budget_override > 0:
        turn_budget = turn_budget_override
    grace = turn_budget * 0.25 + 500
    line = brain.recv((turn_budget + grace) / 1000.0)
    brain.time_used += time.time() - t0
    return parse_reply(line)


def play_one_game(idx, spec_a, spec_b, cfg, cwd):
    color_a = "B" if idx % 2 == 0 else "W"
    specs = {"B": spec_a if color_a == "B" else spec_b,
             "W": spec_a if color_a == "W" else spec_b}
    brains = {}
    for side in ("B", "W"):
        brains[side] = Brain(specs[side], cwd)
    board = Board()

    try:
        for side in ("B", "W"):
            b = brains[side]
            b.send("START 15")
            # Neural engines load a network and initialise a GPU backend here;
            # KataGomo needs ~1.2s to load and ~4s more to compile OpenCL
            # kernels. 5s was too tight and cost it whole games.
            #
            # 30s was still too tight once the bridge's warm-up genmove ran at
            # the match's visit count: at --visits 400 START takes ~26s idle,
            # and any CPU contention pushes it past 30. Windows engines under
            # Wine additionally pay container start plus Wine prefix creation.
            # This bound only has to exceed the slowest engine's init -- being
            # generous costs one wait per game, while being tight silently
            # converts init into a forfeited game.
            r = b.recv(120.0)
            if r is None or r.strip().upper() != "OK":
                loser_side = side
                winner_side = "B" if loser_side == "W" else "W"
                result = "draw" if winner_side == "draw" else (
                    "A" if winner_side == color_a else "B"
                )
                return (idx, result, "init failed")
            b.send(f"INFO rule {cfg.rule}")
            # Engine B may run on a shorter clock. Handicapping the opponent
            # is how we get a *measurable* match: at 3-25% winrate a 30-Elo
            # change is invisible, and capping the opponent's search depth
            # turned out not to weaken it monotonically (see --timeout-turn-b
            # help). Time is the lever that actually works.
            side_is_a = (side == color_a)
            turn_ms = cfg.timeout_turn
            if not side_is_a and cfg.timeout_turn_b > 0:
                turn_ms = cfg.timeout_turn_b
            if turn_ms > 0:
                b.send(f"INFO timeout_turn {turn_ms}")
            if cfg.timeout_match > 0:
                b.send(f"INFO timeout_match {cfg.timeout_match}")

        # Colour already alternates with idx (even = A plays Black), so seeding
        # the opening from idx//2 makes each consecutive PAIR of games share an
        # opening with the colours swapped: both engines meet the identical
        # position from each side. Seeding from idx instead (the default) gives
        # every game a different opening, so colour is balanced but position is
        # not controlled -- fine for a large sample, but it leaves a single
        # lopsided result open to "it drew easier positions".
        if cfg.paired_openings and cfg.openings:
            # Cycle rather than sample: random choice draws with replacement,
            # so some openings repeat while others never appear. Cycling gives
            # every opening the same number of games, which is the point of
            # controlling for position in the first place.
            opening = cfg.openings[(idx // 2) % len(cfg.openings)]
        else:
            rng = random.Random(cfg.seed * 10**6 + idx)
            opening = (
                rng.choice(cfg.openings)
                if cfg.openings
                else None
            )
        if opening is not None:
            board.play(opening[0], opening[1], "B")

        mover = "W" if opening is not None else "B"
        reason = ""
        winner_side = None
        moves = 0

        live_emit(cfg, {"t": "start", "game": idx,
                        "black": cfg.engine_a if color_a == "B" else cfg.engine_b,
                        "white": cfg.engine_a if color_a == "W" else cfg.engine_b,
                        "a_color": color_a,
                        "opening": list(opening) if opening else None})
        if opening is not None:
            live_emit(cfg, {"t": "move", "game": idx, "n": 0,
                            "side": "B", "r": opening[0], "c": opening[1],
                            "ms": 0})
        while True:
            first = opening is None and moves == 0
            mover_is_a = (mover == color_a)
            mover_budget = (cfg.timeout_turn_b
                            if (not mover_is_a and cfg.timeout_turn_b > 0)
                            else cfg.timeout_turn)
            _t0 = time.monotonic()
            move, err = request_move(brains[mover], board, opening, cfg, first,
                                     mover_budget, mover)
            _dt_ms = int((time.monotonic() - _t0) * 1000)
            if move is None:
                winner_side = "B" if mover == "W" else "W"
                reason = err
                break
            r, c = move
            if not board.play(r, c, mover):
                # First move must honor the ring rule (Outer-Open).
                if moves == 0 and opening is None and not board.is_ring(r, c):
                    winner_side = "B" if mover == "W" else "W"
                    reason = f"first move ({r},{c}) not on the outer ring"
                    break
                winner_side = "B" if mover == "W" else "W"
                reason = f"illegal move ({r},{c})"
                break
            moves += 1
            live_emit(cfg, {"t": "move", "game": idx, "n": moves,
                            "side": mover, "r": r, "c": c, "ms": _dt_ms})
            if board.win_at(r, c):
                winner_side = mover
                # Record the length so overline wins can be counted: against a
                # Standard-rules opponent those are wins the opponent never
                # recognised as threats, which would inflate our score.
                reason = f"{board.win_len}-in-a-row at ({r},{c})"
                break
            if board.full():
                winner_side = "draw"
                reason = "board full"
                break
            mover = "B" if mover == "W" else "W"

        result = "draw" if winner_side == "draw" else (
            "A" if winner_side == color_a else "B"
        )
        live_emit(cfg, {"t": "end", "game": idx, "winner": winner_side,
                        "result": result, "reason": reason, "plies": moves})
        return (idx, result, reason)
    finally:
        for b in brains.values():
            try:
                b.send("END")
            except Exception:
                pass
            b.kill()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def load_completed(path):
    """Games already finished in a live file: {idx: (result, reason)}.

    Only "end" records count. A game with moves but no end record was in
    flight when the run died and must be replayed -- its board is half-built
    and its result unknown.
    """
    done = {}
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a killed writer
                if rec.get("t") == "end" and "game" in rec:
                    done[rec["game"]] = (rec.get("result"), rec.get("reason", ""))
    except FileNotFoundError:
        pass
    return done


_LIVE_LOCK = threading.Lock()


def live_emit(cfg, record):
    """Append one JSON line to cfg.live_file, if a live file was requested.

    Deliberately fire-and-forget: a viewer is a convenience, and a match that
    has been running for hours must never die because the log directory went
    away or a disk filled. Every failure is swallowed.

    Opened and closed per record rather than held open, so a viewer can be
    started after the match, the file can be rotated underneath us, and a
    killed match leaves no partial buffered line. One line per move is cheap
    next to a 1-10s search.

    Under --parallel the games interleave, so each record carries its own
    "game" index and the lock keeps lines from being torn.
    """
    path = getattr(cfg, "live_file", "")
    if not path:
        return
    try:
        line = json.dumps(record, separators=(",", ":"))
        with _LIVE_LOCK:
            with open(path, "a") as fh:
                fh.write(line + "\n")
    except Exception:
        pass


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=20, help="number of games")
    p.add_argument("--parallel", type=int, default=1, help="concurrent tables")
    p.add_argument("--engine-a", default=os.path.join(REPO, "build", "release",
                                                      "bin", "oog_piskvork"))
    p.add_argument("--engine-b", default=os.path.join(REPO, "build", "release",
                                                      "bin", "oog_piskvork"))
    p.add_argument("--rule", type=int, default=0, metavar="N",
                   help="Gomocup INFO rule value (default 0 = freestyle). "
                        "This project's own engine ignores the field and "
                        "always plays freestyle, but external engines do not: "
                        "rapfi maps the value straight onto its own enum "
                        "(0=freestyle, 1=standard, 2=renju), so sending 2 "
                        "makes it load renju weights and play with forbidden "
                        "moves -- a different game from ours. Keep this at 0 "
                        "unless you know the opponent's mapping.")
    p.add_argument("--timeout-turn", type=int, default=2000, metavar="MS")
    p.add_argument("--timeout-turn-b", type=int, default=0, metavar="MS",
                   help="Separate per-turn budget for engine B (0 = same as "
                        "--timeout-turn). Use this to handicap a much stronger "
                        "opponent down into a measurable range. Capping an "
                        "opponent's search depth was tried first and did not "
                        "work: rapfi at max_search_depth 2/4/6/8 scored "
                        "90/75/97.5/75% against us, i.e. erratic and never "
                        "near even, because its strength is in its evaluation "
                        "rather than its search depth. Time is the lever that "
                        "scales smoothly.")
    p.add_argument("--timeout-match", type=int, default=0, metavar="MS")
    p.add_argument("--resume", action="store_true",
                   help="skip games already finished in --live-file and carry "
                        "their results into the summary. match.py otherwise "
                        "keeps every result in memory and prints it only at "
                        "exit, so an interrupted match loses everything. Game "
                        "index selects the opening, so skipping by index keeps "
                        "the paired-opening schedule intact. Requires "
                        "--live-file, and that file must NOT be truncated "
                        "before the rerun.")
    p.add_argument("--live-file", type=str, default="", metavar="PATH",
                   help="append every move to PATH as JSON lines so a viewer "
                        "can follow games as they happen. Use with "
                        "tools/watch_match.py. Costs one short appended line "
                        "per move; safe under --parallel (a lock serialises "
                        "writers, and each record carries its game index).")
    p.add_argument("--nodes", type=int, default=0,
                   help="per-move node budget (default engine only)")
    p.add_argument("--openings", type=str, default="default",
                   help="file with ring moves 'r,c' per line, 'default', or "
                        "'none' to use BEGIN openings")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--paired-openings", action="store_true",
                   help="play each opening TWICE with colours reversed "
                        "(games 2k and 2k+1 share an opening). Removes "
                        "position luck as an explanation for a lopsided "
                        "result: both engines face the identical start from "
                        "each side. Use an even --games; pass 2x the number "
                        "of openings for full coverage. Off by default so "
                        "earlier measurements stay reproducible.")
    p.add_argument("--cwd", default=None,
                   help="working directory for engine processes (default: repo)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def load_openings(spec):
    if spec == "none":
        return None
    if spec == "default":
        return list(DEFAULT_OPENINGS)
    moves = []
    with open(spec) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r, c = line.split(",")
            r, c = int(r), int(c)
            if (r, c) in RING:
                moves.append((r, c))
    return moves or list(DEFAULT_OPENINGS)


def main(argv):
    cfg = parse_args(argv)
    cfg.seed = cfg.seed
    cfg.nodes = cfg.nodes
    cfg.openings = load_openings(cfg.openings)
    cmd_a = build_engine_cmd(cfg.engine_a, cfg)
    cmd_b = build_engine_cmd(cfg.engine_b, cfg)
    cwd = cfg.cwd or REPO

    if cfg.verbose:
        print(f"engine A: {' '.join(cmd_a)}")
        print(f"engine B: {' '.join(cmd_b)}")
        print(f"openings: {len(cfg.openings) if cfg.openings else 'BEGIN'}"
              f"  parallel: {cfg.parallel}")

    overline = {"A": 0, "B": 0}
    fives = {"A": 0, "B": 0}
    results = []

    done = {}
    if cfg.resume:
        if not cfg.live_file:
            sys.exit("--resume needs --live-file")
        done = load_completed(cfg.live_file)
        for idx, (result, reason) in sorted(done.items()):
            if idx >= cfg.games:
                continue
            m = re.match(r"(\d+)-in-a-row", reason or "")
            if m and result in ("A", "B"):
                if int(m.group(1)) >= 6:
                    overline[result] += 1
                else:
                    fives[result] += 1
            results.append((idx, result, reason))
        if results:
            print(f"resuming: {len(results)} of {cfg.games} games already "
                  f"finished in {cfg.live_file}", flush=True)

    todo = [i for i in range(cfg.games) if i not in done]
    with ThreadPoolExecutor(max_workers=cfg.parallel) as ex:
        futures = [
            ex.submit(play_one_game, i, cmd_a, cmd_b, cfg, cwd)
            for i in todo
        ]
        for f in as_completed(futures):
            idx, result, reason = f.result()
            m = re.match(r"(\d+)-in-a-row", reason or "")
            if m and result in ("A", "B"):
                if int(m.group(1)) >= 6:
                    overline[result] += 1
                else:
                    fives[result] += 1
            results.append((idx, result, reason))
            if cfg.verbose:
                print(f"game {idx:3d}: {result:<5s} {reason}", flush=True)

    results.sort(key=lambda t: t[0])
    a = sum(1 for _, r, _ in results if r == "A")
    b = sum(1 for _, r, _ in results if r == "B")
    d = sum(1 for _, r, _ in results if r == "draw")
    total = len(results)

    def winrate(w, dr, lo):
        n = w + dr + lo
        return (100.0 * (w + 0.5 * dr) / n) if n else 0.0

    print(f"\nGames: {total}   A wins: {a}   B wins: {b}   draws: {d}")
    tot_ol = overline["A"] + overline["B"]
    print(f"win types: A {fives['A']}x five + {overline['A']}x OVERLINE, "
          f"B {fives['B']}x five + {overline['B']}x OVERLINE"
          + ("   <-- overlines present: a Standard-rules opponent does not "
             "treat these as wins" if tot_ol else ""))
    print(f"A winrate: {winrate(a, d, b):.1f}%")

    import math

    def elo_diff(score):
        if score <= 0.0:
            return -4000.0
        if score >= 1.0:
            return 4000.0
        return -400.0 * math.log10(1.0 / score - 1.0)

    print(f"Elo difference (A - B): {elo_diff((a + 0.5 * d) / total):+.1f}")

    for side in ("B", "W"):
        games_side = [r for (i, r, _) in results if i % 2 == (0 if side == "B" else 1)]
        aw = sum(1 for r in games_side if r == "A")
        bw = sum(1 for r in games_side if r == "B")
        dr = sum(1 for r in games_side if r == "draw")
        if games_side:
            print(f"A as {side}: {aw} wins / {dr} draws / {bw} losses")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
