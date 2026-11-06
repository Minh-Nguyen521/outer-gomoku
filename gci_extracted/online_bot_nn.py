"""
Online Gomoku wrapper: big_net_best.pt (policy-value net) + PUCT MCTS.

In-process: imports the search engine class from tools/nn_mcts_player.py and
drives it directly, bypassing that file's protocol loop (its BOARD handler
reads relative colours, which the online wrapper does not send).

Rules handled here that nn_mcts_player.py does not implement:
  * first move of a game must be on the outer 2 rows/columns (ring rule);
    the net wants the centre, so on an empty board we pick a ring corner-ish
    cell instead.

Usage (headless):
    python online_bot_nn.py --url http://host:8765 --code ABC123
Via GUI: client.py -> Client -> select this file as the bot.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

try:
    from client import BOARD_SIZE, is_outer, run_bot
except ImportError:
    from gomoku_online import BOARD_SIZE, is_outer, run_bot

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import torch  # noqa: E402  (after path setup so error messages stay local)

import nn_mcts_player as nmp  # noqa: E402
from nn_mcts_player import Engine, Node  # noqa: E402

BOT_NAME = "OuterGomoku-NN"

MODEL_PATH = REPO / "exp/06_finetune/big_net_best.pt"
SIMS = 400

_engine: Engine | None = None
_own: set | None = None      # our stones, (col, row)
_opp: set | None = None      # opponent stones
_game_no: str | None = None


def load_bot(room) -> None:
    global _engine
    room.status(f"Tải model: {MODEL_PATH.name}")
    _engine = Engine(str(MODEL_PATH), SIMS)
    room.ready()


def choose_move(board: list[list[int]], player: int, info: dict) -> tuple[int, int]:
    """board[row][col]; returns (row, col). Engine coords are (col, row)."""
    global _own, _opp, _game_no
    if _game_no != info["game_no"]:
        _game_no = info["game_no"]
        _own, _opp = set(), set()

    # rebuild stones from the authoritative board each turn
    _own = {(c, r) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == player}
    _opp = {(c, r) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == 3 - player}

    if not _own and not _opp:
        # empty board: ring rule — the net's centre preference is illegal
        return random.choice([(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if is_outer(y, x)])

    col, row = _search_timed(_own, _opp, _time_cap(info))
    if info["ply"] == 0 and not is_outer(row, col):
        raise RuntimeError(f"net broke ring rule: {(row, col)}")
    return row, col


def _time_cap(info: dict) -> float | None:
    """Số giây tối đa cho nước này, từ đồng hồ server. None = không giới hạn."""
    budget = info.get("time_budget")
    if budget is None:
        return None
    # giữ 15% biên an toàn cho Python/rebuild +network jitter
    return max(0.5, budget * 0.85)


def _search_timed(own, opp, cap: float | None):
    """Engine.search nhưng dừng theo deadline thay vì số sims cố định."""
    eng = _engine
    mv = eng.forced(own, opp)
    if mv is not None:
        return mv

    root = Node(0.0)
    pri, _ = eng.evaluate(own, opp)
    for m, p in pri.items():
        root.children[m] = Node(p)
    root.expanded = True

    deadline = None if cap is None else time.perf_counter() + cap
    done = 0
    while done < SIMS and (deadline is None or time.perf_counter() < deadline):
        node = root
        path = [node]
        o, e = set(own), set(opp)
        mv_last = None
        # SELECT
        while node.expanded and node.children:
            best, best_u = None, -1e18
            sqrt_n = math.sqrt(max(1, node.visits))
            for m, ch in node.children.items():
                u = -ch.q() + eng.cpuct * ch.prior * sqrt_n / (1 + ch.visits)
                if u > best_u:
                    best_u, best = u, m
            mv_last = best
            node = node.children[best]
            path.append(node)
            o.add(best)
            o, e = e, o
            if nmp.makes_five(e, best):
                break
        # EVALUATE
        if mv_last is not None and nmp.makes_five(e, mv_last):
            value = -1.0
        else:
            pri, value = eng.evaluate(o, e)
            if not node.expanded:
                for m, p in pri.items():
                    node.children[m] = Node(p)
                node.expanded = True
        # BACKUP
        for nd in reversed(path):
            nd.visits += 1
            nd.value_sum += value
            value = -value
        done += 1

    return max(root.children.items(), key=lambda kv: kv[1].visits)[0]


def main() -> None:
    global SIMS
    parser = argparse.ArgumentParser(description="Chạy bot NN+MCTS không giao diện")
    parser.add_argument("--url", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--name", default=BOT_NAME)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--sims", type=int, default=SIMS)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    SIMS = args.sims
    run_bot(args.url, args.code, args.name, choose_move, margin=args.margin, once=args.once, load_bot=load_bot)


if __name__ == "__main__":
    main()
