#!/usr/bin/env python3
"""Play Gomocup using the distilled policy network. No search.

This is deliberately the simplest possible player: one forward pass, take the
highest-probability legal square. That makes it a clean test of what
distillation actually captured, with nothing else contributing.

There is a precedent for the result being interesting: KataGomo at ONE visit
-- a single network evaluation with no search at all -- beat this project's
hand-written alpha-beta engine 20-0. If a distilled policy can do anything
similar, the network is carrying real strength and adding search on top is
worth doing. If it cannot, more search would only be polishing a weak
evaluator.

The plane encoding MUST match tools/train_policy.py exactly, or the network
sees inputs unlike anything it trained on. It is duplicated here rather than
imported because the training script builds planes from a move list (it has
game records) while the protocol gives own/opponent sets with no move order --
and order is irrelevant to these planes, only occupancy matters.

Usage (as an engine for tools/match.py):
    tools/nn_player.py --model data/policy_net.pt
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 15


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
        self.stem = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1, bias=False),
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


def planes_from_sets(own, opp):
    """(1, 6, 15, 15) -- identical layout to train_policy.build_planes."""
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
    return p[None, ...]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/policy_net.pt")
    ap.add_argument("--device", default=None)
    cfg = ap.parse_args(argv)

    dev = cfg.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(cfg.model, map_location=dev, weights_only=False)
    net = Net(ck.get("blocks", 4), ck.get("channels", 64)).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    own, opp = set(), set()

    def reply(s):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    def choose():
        """Highest-probability EMPTY square."""
        with torch.no_grad():
            x = torch.from_numpy(planes_from_sets(own, opp)).to(dev)
            logits, _ = net(x)
            pol = logits[0].float().cpu().numpy()
        for (px, py) in own | opp:
            pol[py * N + px] = -1e9          # never return an occupied square
        i = int(pol.argmax())
        return i % N, i // N

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        up = line.upper()

        if up.startswith("START"):
            own.clear(); opp.clear()
            reply("OK")
        elif up.startswith("INFO"):
            pass                              # one forward pass; no clock use
        elif up.startswith("ABOUT"):
            reply('name="oog-distill", version="0.1", '
                  'author="distilled from KataGomo", country="unknown"')
        elif up.startswith("RESTART"):
            own.clear(); opp.clear()
            reply("OK")
        elif up.startswith("BEGIN"):
            x, y = choose()
            own.add((x, y))
            reply(f"{x},{y}")
        elif up.startswith("TURN"):
            try:
                xs, ys = line.split(None, 1)[1].split(",")
                opp.add((int(xs), int(ys)))
            except (IndexError, ValueError):
                reply("ERROR bad TURN")
                continue
            x, y = choose()
            own.add((x, y))
            reply(f"{x},{y}")
        elif up.startswith("BOARD"):
            # Fields are RELATIVE: 1 = our stone, 2 = the opponent's.
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
            x, y = choose()
            own.add((x, y))
            reply(f"{x},{y}")
        elif up.startswith("END"):
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
