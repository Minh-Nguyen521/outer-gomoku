"""
Online Gomoku wrapper: rapfi (strongest free Gomoku engine, this project's
named reference). Subprocess speaking the Piskvork protocol, adapted to the
online template's load_bot / choose_move interface.

Same structure as online_bot_slowrenju.py (stateless per turn, BOARD-rebuild,
relative colours, wrapper-supplied ring opening) with these rapfi specifics:

  * rapfi accepts `INFO rule 0` (freestyle). Outer-Open differs from freestyle
    ONLY in Black's first move, and the wrapper plays that move itself:
    a data-driven ring opening from data/opening_book.json (each first move's
    statistically best reply was ALSO mined from engine-vs-engine harvest,
    so when we are White the book supplies the reply too).
  * rapfi's NNUE weights load lazily on the first genmove; the probe match
    showed the first reply after START can take several seconds. The wrapper
    therefore sends a warm-up dummy position during load_bot so the clock
    never pays for weight decompression (server counts network latency AND
    engine time alike).
  * MESSAGE lines: rapfi prints OK/MESSAGE chatter; _recv skips everything
    except the coordinate reply.

Coordinate notes:
  * Server:  board[row][col], choose_move returns (row, col).
  * Piskvork: "x,y" = column,row — swapped on the way in and out.

Usage (headless):
    python online_bot_rapfi.py --url http://host:8765 --code ABC123
Via GUI: client.py -> Client -> select this file as the bot.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

try:
    from client import BOARD_SIZE, is_outer, run_bot
except ImportError:
    from gomoku_online import BOARD_SIZE, is_outer, run_bot

BOT_NAME = "OuterGomoku-Rapfi"

ENGINE_PATH = Path(__file__).resolve().parents[1] / "third_party/Rapfi-gomocup/pbrain-rapfi2018"
BOOK_PATH = Path(__file__).resolve().parents[1] / "data/opening_book.json"

_proc: subprocess.Popen | None = None
_book: dict = {}
_expected_turn: int | None = None  # engine knows stones sent via TURN so far


def _send(line: str) -> None:
    assert _proc and _proc.stdin
    _proc.stdin.write(line + "\n")
    _proc.stdin.flush()


def _recv() -> str:
    assert _proc and _proc.stdout
    while True:
        out = _proc.stdout.readline()
        if not out:
            raise RuntimeError("engine terminated")
        out = out.strip()
        if out.upper().startswith(("MESSAGE", "DEBUG", "SUGGEST", "ERROR", "OK")):
            continue
        if not out or "," not in out:
            continue
        return out


def load_bot(room) -> None:
    """Spawn the engine once per match session; warm up off the clock.

    Engine choice: rapfi2018 (Gomocup build). The newer arm64 build
    (third_party/rapfi) hangs on the BOARD command; this one implements
    full BOARD rebuild correctly, which the stateless wrapper requires.
    """
    global _proc, _book
    room.status(f"Tải engine: {ENGINE_PATH}")
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy engine: {ENGINE_PATH}")
    _proc = subprocess.Popen(
        [str(ENGINE_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    _send("INFO rule 0")
    _send(f"START {BOARD_SIZE}")
    # Warm-up genmove (legal quiet position, reply discarded): loads NNUE
    # weights + pages in the binary BEFORE the server clock starts ticking.
    _send("BOARD")
    _send(f"{BOARD_SIZE // 2},{BOARD_SIZE // 2},1")
    _send(f"{BOARD_SIZE // 2},{BOARD_SIZE // 2 + 1},2")
    _send("DONE")
    try:
        _recv()
    except RuntimeError:
        pass
    if BOOK_PATH.exists():
        try:
            _book = json.loads(BOOK_PATH.read_text())
        except ValueError:
            _book = {}
    room.ready()


def _ring_opening():
    return random.choice(
        [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
         if is_outer(r, c)]
    )


def choose_move(board: list[list[int]], player: int, info: dict) -> tuple[int, int]:
    """board: board[row][col] absolute colours. Returns (row, col)."""
    ply = info["ply"]

    if ply == 0:
        # Outer-Open ring rule; rapfi has no ring knowledge, wrapper opens.
        return _ring_opening()

    if ply == 1 and player == 2:
        lm = info.get("last_move")
        if lm and _book:
            key = f"{lm[0]},{lm[1]}"
            entry = _book.get(key)
            if entry:
                rx, ry = entry["reply"].split(",")
                r, c = int(rx), int(ry)
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE \
                        and board[r][c] == 0:
                    return r, c

    budget = info.get("time_budget")
    turn_ms = int(budget * 1000) if budget is not None else 10000
    _send(f"INFO timeout_turn {min(turn_ms, 60000)}")
    total = info.get("total_left")
    if total is not None:
        _send(f"INFO timeout_match {int(total * 1000)}")
        _send(f"INFO time_left {int(max(total * 1000 - 500, 1000))}")

    # Relative-colour BOARD: 1 = our stones, 2 = opponent's
    _send("BOARD")
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = board[r][c]
            if v:
                _send(f"{c},{r},{1 if v == player else 2}")
    _send("DONE")
    reply = _recv()
    col_s, _, row_s = reply.partition(",")
    r, c = int(row_s), int(col_s)
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE) or board[r][c] != 0:
        raise RuntimeError(f"rapfi returned illegal move ({r},{c})")
    return r, c


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy rapfi không giao diện")
    parser.add_argument("--url", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--name", default=BOT_NAME)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_bot(args.url, args.code, args.name, choose_move,
            margin=args.margin, once=args.once, load_bot=load_bot)


if __name__ == "__main__":
    main()
