"""
Online Gomoku wrapper around the C++ engine (oog_piskvork).

Adapts GCI_GMK_X's template interface (load_bot / choose_move) to the repo's
native engine by spawning tools/piskvork_main.cpp binary and speaking the
Piskvork protocol over stdin/stdout.

Usage:
    python online_bot.py --url http://host:8765 --code ABC123

Engine binary resolved in order:
    $OOG_PISKVORK env var -> ../../build/release/bin/oog_piskvork (repo layout)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from client import BOARD_SIZE, is_outer, run_bot
except ImportError:
    from gomoku_online import BOARD_SIZE, is_outer, run_bot

BOT_NAME = "OuterGomoku-Cpp"

ENGINE_PATH = Path(
    os.environ.get("OOG_PISKVORK", "")
    or Path(__file__).resolve().parents[1] / "build/release/bin/oog_piskvork"
)

_proc: subprocess.Popen | None = None
_moves: list[tuple[int, int]] = []  # (x, y) in move order, absolute colours tracked by parity
_game_no: str | None = None


def _start_engine() -> subprocess.Popen:
    proc = subprocess.Popen(
        [str(ENGINE_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    _send(proc, f"START {BOARD_SIZE}")
    ack = _recv(proc)
    if ack != "OK":
        proc.kill()
        raise RuntimeError(f"engine rejected START: {ack!r}")
    return proc


def _send(proc: subprocess.Popen, line: str) -> None:
    assert proc.stdin
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> str:
    assert proc.stdout
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("engine terminated")
    return line.strip()


def _restart_and_replay(reason: str) -> None:
    global _proc
    print(f"[bot] restarting engine: {reason}", file=sys.stderr)
    if _proc:
        try:
            _send(_proc, "END")
            _proc.wait(timeout=2)
        except Exception:
            _proc.kill()
    _proc = _start_engine()


def load_bot(room) -> None:
    global _proc
    room.status(f"Tải engine: {ENGINE_PATH}")
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy engine binary: {ENGINE_PATH}")
    _proc = _start_engine()
    room.ready()


def choose_move(board: list[list[int]], player: int, info: dict) -> tuple[int, int]:
    """board: server coords, board[row][col]. Engine protocol: "x,y" = column,row."""
    global _game_no
    if info["game_no"] != _game_no or _proc is None or _proc.poll() is not None:
        _game_no = info["game_no"]
        _restart_and_replay("new game" if _proc else "engine died")

    budget = info.get("time_budget")
    turn_ms = int(budget * 1000) if budget is not None else 10000
    _send(_proc, f"INFO timeout_turn {min(turn_ms, 60000)}")
    total = info.get("total_left")
    if total is not None:
        _send(_proc, f"INFO timeout_match {int(total * 1000)}")
        _send(_proc, f"INFO time_left {int(total * 1000)}")

    # BOARD rebuilds the whole position; side-to-move is inferred by stone
    # parity, and the engine answers with its move in the same reply.
    lines = ["BOARD"]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c]:
                lines.append(f"{c},{r},{board[r][c]}")  # x=col, y=row, 1=Black 2=White
    lines.append("DONE")
    _send(_proc, "\n".join(lines))
    reply = _recv(_proc)
    if reply.startswith("ERROR"):
        raise RuntimeError(f"engine error: {reply}")
    col_s, _, row_s = reply.partition(",")
    col, row = int(col_s), int(row_s)

    if info["ply"] == 0 and not is_outer(row, col):
        raise RuntimeError(f"engine broke ring rule: {reply}")

    return row, col


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy bot C++ không giao diện")
    parser.add_argument("--url", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--name", default=BOT_NAME)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_bot(args.url, args.code, args.name, choose_move, margin=args.margin, once=args.once, load_bot=load_bot)


if __name__ == "__main__":
    main()
