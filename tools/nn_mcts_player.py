#!/usr/bin/env python3
"""Play Gomocup with PUCT MCTS guided by the distilled network.

WHY: the policy-only player (tools/nn_player.py) lost 0-20 to our own baseline
with 20/20 clean fives -- a genuine loss, not a broken harness. The sample
game showed the failure precisely: the opponent built an unblocked vertical
five and the network never answered it. Agreeing with the teacher 98.9% of the
time is not enough when a single missed block in ~40 moves loses outright.

Search fixes exactly that class of error, and it is what the network was
trained for: a policy prior to order moves and a value to judge leaves.

Two things here are not optional in Gomoku and are handled before any search:

  * an immediate win is played at once;
  * an immediate loss (opponent has four with an open completion) is blocked
    at once.

MCTS *should* find both given enough simulations, but at a few hundred
simulations it may not, and each miss is an instant loss. Checking directly
costs microseconds and removes the failure mode that actually beat us.

Candidate moves are restricted to squares within Chebyshev distance 2 of an
existing stone -- the same neighbourhood every engine in this repo uses. That
takes branching from 225 to ~30 and is what makes a few hundred simulations
meaningful rather than a rounding error.

Usage:
    tools/nn_mcts_player.py --model data/policy_net.pt --sims 300
"""

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 15
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class Net(nn.Module):
    def __init__(self, blocks=4, ch=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(6, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.ph = nn.Sequential(nn.Conv2d(ch, 2, 1, bias=False),
                                nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
                                nn.Linear(2 * N * N, N * N))
        self.vh = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=False),
                                nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
                                nn.Linear(N * N, 64), nn.ReLU(),
                                nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.ph(h), self.vh(h).squeeze(-1)


def planes(own, opp):
    p = np.zeros((6, N, N), dtype=np.float32)
    for (x, y) in own:
        p[0][y][x] = 1.0
    for (x, y) in opp:
        p[1][y][x] = 1.0
    p[2] = 1.0 - p[0] - p[1]
    for src, dst in ((p[0], 3), (p[1], 4)):
        acc = np.zeros((N, N), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                acc[max(0, dy):N + min(0, dy), max(0, dx):N + min(0, dx)] += \
                    src[max(0, -dy):N + min(0, -dy), max(0, -dx):N + min(0, -dx)]
        p[dst] = np.minimum(acc, 1.0)
    p[5] = 1.0
    return p


def makes_five(stones, mv):
    """Would `mv` complete five-in-a-row for the owner of `stones`?"""
    x, y = mv
    for dx, dy in DIRS:
        c = 1
        for s in (1, -1):
            cx, cy = x + dx * s, y + dy * s
            while (cx, cy) in stones:
                c += 1
                cx += dx * s
                cy += dy * s
        if c >= 5:
            return True
    return False


def candidates(own, opp, radius=2):
    occ = own | opp
    if not occ:
        return [(N // 2, N // 2)]
    out = set()
    for (x, y) in occ:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                p = (x + dx, y + dy)
                if 0 <= p[0] < N and 0 <= p[1] < N and p not in occ:
                    out.add(p)
    return sorted(out)


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "expanded")

    def __init__(self, prior=0.0):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.expanded = False

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


class Engine:
    def __init__(self, model, sims, device=None, cpuct=1.5):
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(model, map_location=self.dev, weights_only=False)
        self.net = Net(ck.get("blocks", 4), ck.get("channels", 64)).to(self.dev)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.sims = sims
        self.cpuct = cpuct

    def evaluate(self, own, opp):
        """(policy dict over candidates, value in [-1,1]) for side-to-move."""
        with torch.no_grad():
            x = torch.from_numpy(planes(own, opp)[None]).to(self.dev)
            logits, val = self.net(x)
            pol = torch.softmax(logits[0].float(), dim=0).cpu().numpy()
            v = float(val[0])
        cands = candidates(own, opp)
        pri = {m: float(pol[m[1] * N + m[0]]) for m in cands}
        tot = sum(pri.values())
        if tot > 0:
            pri = {m: p / tot for m, p in pri.items()}
        else:
            pri = {m: 1.0 / len(cands) for m in cands}
        return pri, v

    def forced(self, own, opp):
        """Win now, or block an immediate loss. None if neither applies."""
        for m in candidates(own, opp):
            if makes_five(own | {m}, m):
                return m
        for m in candidates(own, opp):
            if makes_five(opp | {m}, m):
                return m
        return None

    def search(self, own, opp):
        mv = self.forced(own, opp)
        if mv is not None:
            return mv

        root = Node()
        pri, _ = self.evaluate(own, opp)
        for m, p in pri.items():
            root.children[m] = Node(p)
        root.expanded = True

        for _ in range(self.sims):
            node = root
            path = [node]
            o, e = set(own), set(opp)
            mv_last = None
            # SELECT
            while node.expanded and node.children:
                best, best_u = None, -1e18
                sqrt_n = math.sqrt(max(1, node.visits))
                for m, ch in node.children.items():
                    u = -ch.q() + self.cpuct * ch.prior * sqrt_n / (1 + ch.visits)
                    if u > best_u:
                        best_u, best = u, m
                mv_last = best
                node = node.children[best]
                path.append(node)
                o.add(best)
                o, e = e, o                      # side to move flips
                if makes_five(e, best):          # the mover just won
                    break
            # EVALUATE
            if mv_last is not None and makes_five(e, mv_last):
                value = -1.0                     # side to move has already lost
            else:
                pri, value = self.evaluate(o, e)
                if not node.expanded:
                    for m, p in pri.items():
                        node.children[m] = Node(p)
                    node.expanded = True
            # BACKUP, flipping sign at each ply
            for nd in reversed(path):
                nd.visits += 1
                nd.value_sum += value
                value = -value

        return max(root.children.items(), key=lambda kv: kv[1].visits)[0]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/policy_net.pt")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--device", default=None)
    cfg = ap.parse_args(argv)

    eng = Engine(cfg.model, cfg.sims, cfg.device)
    own, opp = set(), set()

    def reply(s):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("START"):
            own.clear(); opp.clear(); reply("OK")
        elif up.startswith("INFO"):
            pass
        elif up.startswith("ABOUT"):
            reply('name="oog-distill-mcts", version="0.1", '
                  'author="distilled from KataGomo", country="unknown"')
        elif up.startswith("RESTART"):
            own.clear(); opp.clear(); reply("OK")
        elif up.startswith("BEGIN"):
            m = eng.search(own, opp); own.add(m); reply(f"{m[0]},{m[1]}")
        elif up.startswith("TURN"):
            try:
                xs, ys = line.split(None, 1)[1].split(",")
                opp.add((int(xs), int(ys)))
            except (IndexError, ValueError):
                reply("ERROR bad TURN"); continue
            m = eng.search(own, opp); own.add(m); reply(f"{m[0]},{m[1]}")
        elif up.startswith("BOARD"):
            own.clear(); opp.clear()
            for bl in sys.stdin:
                b = bl.strip()
                if not b or b.upper().startswith("DONE"):
                    break
                try:
                    xs, ys, fs = b.split(",")
                    x, y, f = int(xs), int(ys), int(fs)
                except ValueError:
                    continue
                (own if f == 1 else opp).add((x, y))
            m = eng.search(own, opp); own.add(m); reply(f"{m[0]},{m[1]}")
        elif up.startswith("END"):
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
