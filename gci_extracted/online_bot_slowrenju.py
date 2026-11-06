"""
Online Gomoku wrapper: SlowRenju 5.1.3 (external engine, Elo -43 on the ladder).

Subprocess speaking the Piskvork protocol, adapted to the online template's
load_bot / choose_move interface. Stateless per turn: BOARD-rebuild each call
so retries, reconnects and colour swaps cannot desync the engine.

Coordinate notes:
  * Server:  board[row][col], choose_move returns (row, col).
  * Piskvork: "x,y" = column,row — swapped on the way in and out.
  * SlowRenju's BOARD fields are RELATIVE (1 = own stone) per Gomocup spec,
    unlike our own oog_piskvork brain which reads them as absolute colours.
  * Turn 0 must land on the outer 2 rows/cols; SlowRenju has no ring-rule
    knowledge (freestyle/renju engine), so the wrapper supplies the opening
    itself and lets SlowRenju play from ply 2 onward.

Usage (headless):
    python online_bot_slowrenju.py --url http://host:8765 --code ABC123
Via GUI: client.py -> Client -> select this file as the bot.
"""
from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

try:
    from client import BOARD_SIZE, is_outer, run_bot
except ImportError:
    from gomoku_online import BOARD_SIZE, is_outer, run_bot

BOT_NAME = "SlowRenju-5.1.3"

ENGINE_PATH = Path(__file__).resolve().parents[1] / "third_party/SlowRenju/pbrain-slowrenju"

_proc: subprocess.Popen | None = None


def _send(line: str) -> None:
    assert _proc and _proc.stdin
    _proc.stdin.write(line + "\n")
    _proc.stdin.flush()


def _recv() -> str:
    assert _proc and _proc.stdout
    # SlowRenju prints MESSAGE lines freely; skip to the actual reply
    while True:
        out = _proc.stdout.readline()
        if not out:
            raise RuntimeError("engine terminated")
        out = out.strip()
        if out.upper().startswith(("MESSAGE", "DEBUG", "SUGGEST", "ERROR")):
            continue
        return out


def load_bot(room) -> None:
    global _proc
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
    _send(f"START {BOARD_SIZE}")
    ack = _recv()
    if ack != "OK":
        _proc.kill()
        raise RuntimeError(f"engine rejected START: {ack!r}")
    room.ready()


def choose_move(board: list[list[int]], player: int, info: dict) -> tuple[int, int]:
    """board: board[row][col] absolute colours. Returns (row, col)."""
    if info["ply"] == 0:
        # Outer-Open ring rule: SlowRenju would open centre (illegal here)
        return random.choice(
            [(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if is_outer(x, y)]
        )

    budget = info.get("time_budget")
    turn_ms = int(budget * 1000) if budget is not None else 10000
    _send(f"INFO timeout_turn {min(turn_ms, 60000)}")
    total = info.get("total_left")
    if total is not None:
        _send(f"INFO timeout_match {int(total * 1000)}")
        _send(f"INFO time_left {int(total * 1000)}")

    # Relative-colour BOARD: 1 = our stones, 2 = opponent's
    lines = ["BOARD"]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = board[r][c]
            if v:
                field = 1 if v == player else 2
                lines.append(f"{c},{r},{field}")
    lines.append("DONE")
    _send("\n".join(lines))
    reply = _recv()
    col_s, _, row_s = reply.partition(",")
    col, row = int(col_s), int(row_s)
    return row, col


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy SlowRenju không giao diện")
    parser.add_argument("--url", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--name", default=BOT_NAME)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_bot(args.url, args.code, args.name, choose_move, margin=args.margin, once=args.once, load_bot=load_bot)


if __name__ == "__main__":
    main()
