#!/usr/bin/env python3
"""Piskvork/Gomocup <-> GTP bridge, so KataGomo can play in tools/match.py.

KataGomo is a KataGo fork that plays Gomoku (FREESTYLE / STANDARD / RENJU).
It is the only strong neural engine we can run *natively* on Apple Silicon --
JAX, KataGomo's own release binaries, EGOZERO and Yixin are Windows-only, and
MiniZero needs an NVIDIA GPU. But KataGo speaks GTP, while match.py speaks the
Gomocup protocol, so this translates between them.

Two details that are easy to get wrong, and both silently corrupt games:

1. Coordinates. Gomocup uses "x,y", zero-indexed, origin top-left. GTP uses a
   column *letter* that SKIPS 'I' (A..H, J..T) and a one-indexed row counted
   from the BOTTOM. So x=8 is 'J', not 'I', and y=0 is row 15 on a 15x15
   board. Getting either wrong yields legal-looking moves in the wrong place.

2. Colour. Gomocup BOARD fields are relative -- 1 = the receiving engine's own
   stone, 2 = the opponent's (this project's match.py got that wrong until it
   was fixed, which crippled every White-playing opponent). GTP needs absolute
   colours, so we infer: with equal stone counts it is the first player's turn,
   so we are Black; if the opponent has one more, we are White.

Usage (as an engine for match.py):
    tools/katagomo_bridge.py --katago <path> --model <net.bin.gz>
                             --config <cfg> [--maxtime SECONDS]
"""

import argparse
import os
import subprocess
import sys
import threading


def to_gtp(x, y, size):
    """Gomocup (x=col, y=row, 0-indexed, top-left) -> GTP vertex."""
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # note: no 'I'
    return f"{letters[x]}{size - y}"


def from_gtp(v, size):
    """GTP vertex -> Gomocup (x, y)."""
    v = v.strip().upper()
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    x = letters.index(v[0])
    y = size - int(v[1:])
    return x, y


class Gtp:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)

    def send(self, line):
        """Send one GTP command; return its response payload (or None)."""
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        while True:
            ln = self.p.stdout.readline()
            if ln == "":
                return None                 # engine died
            ln = ln.rstrip("\n")
            if ln.startswith("= ") or ln == "=":
                out.append(ln[2:] if len(ln) > 1 else "")
                break
            if ln.startswith("?"):
                return None                 # GTP error
            # CHAT/informational lines before the response: ignore
        while True:                          # responses end at a blank line
            ln = self.p.stdout.readline()
            if ln in ("\n", ""):
                break
            out.append(ln.rstrip("\n"))
        return "\n".join(out).strip()

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago")
    ap.add_argument("--model")
    ap.add_argument("--config")
    ap.add_argument("--cmd",
                    help="Run an arbitrary GTP engine instead of KataGomo, "
                         "given as a full shell-style command. Everything "
                         "below the protocol layer -- relative BOARD fields, "
                         "the I-skipping column letters, bottom-up rows, "
                         "colour inference, the PASS fallback -- is identical "
                         "for any GTP engine, so MiniZero's console mode "
                         "(-mode console, which implements boardsize / "
                         "clear_board / play / genmove) is driven by the same "
                         "code path. KataGo-specific -override-config flags "
                         "are skipped in this mode.")
    ap.add_argument("--maxtime", type=float, default=0.0,
                    help="seconds per move; overrides the config's maxTime")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="chosenMoveTemperature: sample the move from the "
                         "root visit distribution instead of taking the best. "
                         "0 = strongest. This is the only dial that weakens "
                         "the engine BELOW policy-optimal play -- even 1 visit "
                         "(a single network evaluation, no search) beat our "
                         "baseline 20-0, so visits alone cannot reach our "
                         "level. Needs enough visits for a distribution to "
                         "exist; use with --visits >= ~10.")
    ap.add_argument("--visits", type=int, default=0,
                    help="fixed visit count per move. Unlike a time budget "
                         "this is DETERMINISTIC and immune to CPU load, which "
                         "makes parallel matches valid again and gives a "
                         "smooth strength dial for calibration.")
    args = ap.parse_args()

    if args.cmd:
        import shlex
        cmd = shlex.split(args.cmd)
    else:
        missing = [n for n in ("katago", "model", "config")
                   if not getattr(args, n)]
        if missing:
            ap.error("need --cmd, or all of --katago/--model/--config "
                     f"(missing: {', '.join('--' + m for m in missing)})")
        cmd = [args.katago, "gtp", "-model", args.model, "-config", args.config]
    overrides = []
    if args.visits > 0:
        # Fixed visits: remove any time cap so the count is what binds.
        overrides += [f"maxVisits={args.visits}", "maxTime=1000000"]
    elif args.maxtime > 0:
        overrides += [f"maxTime={args.maxtime}"]
    # ALWAYS set the move-selection temperature, including 0. KataGo's own
    # default randomises early moves (chosenMoveTemperatureEarly = 0.5), so
    # without an explicit override even "temperature 0" plays differently run
    # to run -- which would quietly destroy the determinism that makes fixed
    # visits a usable instrument. Halflife is capped at 100000 by KataGo, and
    # a large value means the weakening does not fade out mid-game.
    overrides += [f"chosenMoveTemperature={args.temperature}",
                  f"chosenMoveTemperatureEarly={args.temperature}",
                  "chosenMoveTemperatureHalflife=100000"]
    # -override-config is KataGo's own flag; a generic GTP engine given via
    # --cmd carries its own settings on its command line instead.
    if overrides and not args.cmd:
        cmd += ["-override-config", ",".join(overrides)]
    gtp = Gtp(cmd)

    # Warm-up. The FIRST genmove costs ~4s on OpenCL (kernel compilation and
    # GPU init) even though later ones are fast. That one-off blew past the
    # per-move clock and silently forfeited the opening move of some games.
    # Pay it here, before the manager starts timing anything.
    gtp.send("boardsize 15")
    gtp.send("clear_board")
    gtp.send("genmove B")
    gtp.send("clear_board")

    size = 15
    stones = []          # (x, y, is_own) in play order, for BOARD replay
    started = False

    def reply(s):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    def our_colour(n_own, n_opp):
        # Equal counts -> the first player is to move -> that is us.
        return "B" if n_own == n_opp else "W"

    def genmove(colour):
        """A move, never None while a legal square remains.

        KataGo is a Go engine, where PASS is a legal and often correct move --
        it returns one in long games. Gomoku has no pass, so treating it as a
        failure forfeited the game: that cost KataGomo 3 of 30 games against
        AlphaGomoku, all recorded as losses it never actually played.

        RESIGN is likewise not a Gomoku concept here; the manager decides the
        result. In both cases fall back to the network's own top-ranked legal
        move (via the raw policy), which is the closest thing to "what the
        engine would have played".
        """
        v = gtp.send(f"genmove {colour}")
        if v is not None and v.upper() not in ("RESIGN", "PASS"):
            try:
                return from_gtp(v, size)
            except (ValueError, IndexError):
                pass
        return best_policy_move()

    def best_policy_move():
        """Highest-probability legal square from the raw policy head."""
        resp = gtp.send("kata-raw-nn 0")
        if not resp:
            return None
        pol, reading = [], False
        # send() returns ONE string, not a list of lines. Iterating it
        # directly walks single characters, so the "policy" header never
        # matches, `pol` stays empty and this returns None -- turning every
        # PASS into a forfeited game, which is exactly what this fallback
        # exists to prevent. It silently did nothing until a match showed
        # "bad reply: 'ERROR no move'".
        for ln in resp.splitlines():
            t = ln.strip()
            if t == "policy":
                reading = True
                continue
            if reading:
                parts = t.split()
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    reading = False
                    continue
                # NAN marks occupied squares -- never playable.
                pol.extend(-1.0 if x != x else x for x in vals)
        pol = pol[:size * size]
        if not pol:
            return None
        best = max(range(len(pol)), key=lambda i: pol[i])
        if pol[best] < 0:
            return None
        return best % size, best // size

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        up = line.upper()

        if up.startswith("START"):
            parts = line.split()
            size = int(parts[1]) if len(parts) > 1 else 15
            gtp.send(f"boardsize {size}")
            gtp.send("clear_board")
            stones = []
            started = True
            reply("OK")

        elif up.startswith("INFO"):
            pass  # timeouts handled via --maxtime; nothing to forward

        elif up.startswith("ABOUT"):
            reply('name="KataGomo", version="Gom2024", author="hzyhhzy '
                  '(KataGo fork)", country="unknown"')

        elif up.startswith("RESTART"):
            gtp.send("clear_board")
            stones = []
            reply("OK")

        elif up.startswith("BEGIN"):
            # Empty board, we move first -> we are Black.
            mv = genmove("B")
            if mv is None:
                reply("ERROR no move")
            else:
                gtp_played = to_gtp(mv[0], mv[1], size)
                stones.append((mv[0], mv[1], True))
                reply(f"{mv[0]},{mv[1]}")

        elif up.startswith("TURN"):
            try:
                xs, ys = line.split(None, 1)[1].split(",")
                ox, oy = int(xs), int(ys)
            except (IndexError, ValueError):
                reply("ERROR bad TURN")
                continue
            n_own = sum(1 for s in stones if s[2])
            n_opp = sum(1 for s in stones if not s[2])
            us = our_colour(n_own, n_opp + 1)  # after their move lands
            them = "W" if us == "B" else "B"
            gtp.send(f"play {them} {to_gtp(ox, oy, size)}")
            stones.append((ox, oy, False))
            mv = genmove(us)
            if mv is None:
                reply("ERROR no move")
            else:
                stones.append((mv[0], mv[1], True))
                reply(f"{mv[0]},{mv[1]}")

        elif up.startswith("BOARD"):
            # Replay the whole position, then move. Fields are RELATIVE:
            # 1 = our stone, 2 = opponent's.
            own, opp = [], []
            for bl in sys.stdin:
                b = bl.strip()
                if not b or b.upper().startswith("DONE"):
                    break
                try:
                    xs, ys, fs = b.split(",")
                    x, y, f = int(xs), int(ys), int(fs)
                except ValueError:
                    continue
                (own if f == 1 else opp).append((x, y))
            us = our_colour(len(own), len(opp))
            them = "W" if us == "B" else "B"
            gtp.send("clear_board")
            # Interleave so the engine sees a plausible move order; exact
            # order does not matter for Gomoku legality, only occupancy.
            for i in range(max(len(own), len(opp))):
                if us == "B":
                    if i < len(own): gtp.send(f"play {us} {to_gtp(*own[i], size)}")
                    if i < len(opp): gtp.send(f"play {them} {to_gtp(*opp[i], size)}")
                else:
                    if i < len(opp): gtp.send(f"play {them} {to_gtp(*opp[i], size)}")
                    if i < len(own): gtp.send(f"play {us} {to_gtp(*own[i], size)}")
            stones = [(x, y, True) for x, y in own] + \
                     [(x, y, False) for x, y in opp]
            mv = genmove(us)
            if mv is None:
                reply("ERROR no move")
            else:
                stones.append((mv[0], mv[1], True))
                reply(f"{mv[0]},{mv[1]}")

        elif up.startswith("END"):
            gtp.close()
            return 0

    gtp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
