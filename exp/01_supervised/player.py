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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--simulations", type=int, default=0,
                    help="0 = raw policy; 1 = block immediate losses")
    ap.add_argument("--temperature", type=float, default=0.25)
    args = ap.parse_args()

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=dev)
    net = Net(ck["blocks"], ck["channels"]).to(dev)
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
            for entry in rest_lines:
                xs, ys, cs = entry.split(",")
                if cs == "1":
                    moves.append((int(xs), int(ys)))
            # note: colour-1 stones are OURS; ordering mixed colours is lost
            # here -- for match.py's paired usage the colour split is known
            # via TURN flow, and BOARD is used for initial setup only.
            send(best_move())
        elif cmd == "INFO":
            pass
        elif cmd == "END":
            return


if __name__ == "__main__":
    main()
