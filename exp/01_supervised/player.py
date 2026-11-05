#!/usr/bin/env python3
"""Gomocup-protocol adapter for the 6-plane Net from tools/train_policy.py.

Same role as oog_az/player.py but for checkpoints saved by
tools/train_policy.py / exp/01_supervised / exp/02_exit_iter
(state_dict with 6 input planes: own, opp, empty, own-adj, opp-adj, const).

Usage:
    python3 exp/01_supervised/player.py --model exp/01_supervised/supervised.pt \
        [--simulations 0] [--temperature 0.5]
simulations=0 -> raw policy; >0 adds a shallow 1-ply winning/blunting check
(beyond that use oog_az AZNet players).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from tools.train_policy import Net, build_planes  # noqa: E402

N = 15


def gumbel_search(net, dev, moves, stm, k):
    """Gumbel sequential-halving move selection at inference time.

    Mirrors exp/04_gumbel/train.py gumbel_move() but builds the candidate
    boards from a flat move list (the protocol adapter doesn't own a
    Goboard). Terminal checks reuse the same 5-in-a-row logic.
    """
    import copy
    import sys as _sys
    _sys.path.insert(0, os.path.join(REPO, "exp", "04_gumbel"))
    gb = _sys.modules.get("train_gumbel")
    if gb is None:
        import importlib
        # exp/04_gumbel/train.py doubles as a module named 'train' inside
        # its own dir; import lazily without clashing with exp/01's train
        spec = importlib.util.spec_from_file_location(
            "train_gumbel",
            os.path.join(REPO, "exp", "04_gumbel", "train.py"))
        gb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gb)
    board = gb.Goboard(N)
    for m in moves:
        # protocol moves are (x, y); Goboard indexes grid[row=y][col=x]
        board.play(m[1], m[0])
    a, _pi = gb.gumbel_move(net, dev, board, k)
    return f"{a % N},{a // N}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--simulations", type=int, default=0,
                    help="0 = raw policy; >0 = Gumbel k=<simulations> "
                         "sequential-halving search at inference")
    ap.add_argument("--temperature", type=float, default=0.25)
    args = ap.parse_args()

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=dev)
    net = Net(ck.get("blocks", 4), ck.get("channels", 64)).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    def predict(moves, stm):
        p = build_planes([tuple(m) for m in moves], stm)
        with torch.no_grad():
            logits, val = net(torch.from_numpy(p[None]).to(dev))
        pol = F.softmax(logits, dim=1)[0].cpu().numpy()
        return pol, float(val[0])

    # --- Gomocup protocol loop (conventions from oog_az/player.py) ---
    import threading, queue
    out_q = queue.Queue()

    def pump():
        for line in sys.stdin:
            out_q.put(line.strip())
        out_q.put(None)
    threading.Thread(target=pump, daemon=True).start()

    moves = []          # (x, y) moves played so far, our coordinate frame

    def best_move():
        stm = "B" if len(moves) % 2 == 0 else "W"
        if args.simulations > 0:
            res = gumbel_search(net, dev, moves, stm, args.simulations)
            if os.environ.get("PLAYER_DEBUG"):
                with open("/tmp/player_dbg.log", "a") as f:
                    f.write(f"move: stm={stm} nmov={len(moves)} "
                            f"moves={moves} -> {res}\n")
            return res
        pol, val = predict(moves, stm)
        occ = np.zeros(N * N, dtype=bool)
        for (x, y) in moves:
            occ[y * N + x] = True
        pol = np.where(occ, 0.0, pol)
        if pol.sum() <= 0:
            pol = np.where(~occ, 1.0, 0.0)
        pol = pol / pol.sum()
        if args.temperature != 1.0:
            p = pol ** (1.0 / max(args.temperature, 1e-3))
            pol = p / p.sum()
        a = int(np.random.choice(len(pol), p=pol))
        return f"{a % N},{a // N}"

    def send(s):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    while True:
        line = out_q.get()
        if line is None:
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd == "START":
            send("OK")
        elif cmd == "TURN":
            x, y = map(int, rest.split(","))
            moves.append((x, y))
            send(best_move())
        elif cmd == "BOARD":
            # rebuild position from scratch every time (player.py lesson)
            rest_lines = []
            while True:
                l2 = out_q.get()
                if l2 is None or l2 == "DONE":
                    break
                rest_lines.append(l2)
            moves = []
            mine, theirs = [], []
            for entry in rest_lines:
                xs, ys, cs = entry.split(",")
                # Gomocup BOARD: 1 = OUR stones, 2 = OPPONENT's (relative).
                # Track both lists separately to reconstruct the interleaved
                # order: colour-1 moves and colour-2 moves alternate by move
                # number; with equal counts ours moved first as B, else W.
                (mine if cs == "1" else theirs).append((int(xs), int(ys)))
            # True move order: if we're Black (ours came first) the sequence
            # alternates m,t,m,...; if White it alternates t,m,t,...
            # Since BOARD arrives on OUR turn, Black has played one more
            # stone when we're Black, counts are equal when we're White.
            if len(mine) >= len(theirs):
                seq = [m for pair in zip(mine, theirs) for m in pair]
                if len(mine) > len(theirs):
                    seq.append(mine[-1])
            else:
                seq = [m for pair in zip(theirs, mine) for m in pair]
                seq.append(theirs[-1])
            moves = seq
            if os.environ.get("PLAYER_DEBUG"):
                with open("/tmp/player_dbg.log", "a") as f:
                    f.write(f"board: mine={mine} theirs={theirs}\n")
            send(best_move())
        elif cmd == "INFO":
            pass
        elif cmd == "END":
            return


if __name__ == "__main__":
    main()
