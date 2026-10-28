#!/usr/bin/env python3
"""pbrain-OogZero: a Piskvork/Gomocup engine backed by an ONNX MiniZero model.

Wraps the network exported by tools/export_model.py (AlphaZero or MuZero
flavour) in the same PUCT search MiniZero itself uses (mcts.cpp: puct_bias =
puct_init + log((1+N+puct_base)/puct_base); Q from leaf values; no noise at
play time; action chosen by max visit count), driven over the Gomocup text
protocol on stdin/stdout:

    START 15 / RECTSTART      BEGIN   TURN x,y      BOARD ... DONE
    INFO ...                  ABOUT   END

Rules: Outer-Open Gomoku, freestyle (five OR MORE in a row wins; overlines
win). Black's FIRST move must lie in the outer two rows/columns, matching
minizero's gomoku.cpp:53. Subsequent moves are unconstrained.

Feature encoding is minizero's 4-plane layout (gomoku.cpp getFeatures):
    0: current player's stones, 1: opponent's, 2: side-to-move-is-black,
    3: side-to-move-is-white -- all from the perspective of the player to
    move.

Usage:
    pbrain_oogzero.py MODEL.onnx [NUM_SIMULATIONS]

The model path may also be given via env var OOGZERO_MODEL. Simulations
default to 400, matching colab_minizero.sh's actor_num_simulation.

Tested with tools/match.py:
    python3 tools/match.py --games 20 \
        --engine-a "python3 tools/pbrain_oogzero.py exports/model.onnx 400" \
        --engine-b third_party/rapfi-bin/pbrain-rapfi
"""

import math
import os
import sys

import numpy as np
import onnxruntime as ort

SIZE = 15
INF = float("inf")

# Search constants, mirroring minizero defaults (configuration.h):
#   actor_mcts_puct_init  = 1.1     actor_mcts_puct_base = 19652
# Dirichlet noise is a self-play aid and is OFF for competitive play.
PUCT_INIT = 1.1
PUCT_BASE = 19652.0


# ---------------------------------------------------------------- board ----

class Board:
    """Outer-Open freestyle Gomoku board."""

    def __init__(self, size=SIZE):
        self.size = size
        self.reset()

    def reset(self):
        self.grid = [0] * (self.size * self.size)   # 0 empty, 1 black, 2 white
        self.moves = []                              # [(pos, color), ...]

    def in_ring(self, pos):
        i, j = divmod(pos, self.size)
        return i < 2 or i >= self.size - 2 or j < 2 or j >= self.size - 2

    def legal(self, pos):
        if not (0 <= pos < self.size * self.size) or self.grid[pos]:
            return False
        if not self.moves and not self.in_ring(pos):
            return False                             # first move: outer ring
        return True

    def legal_moves(self):
        return [p for p in range(self.size * self.size) if self.legal(p)]

    def play(self, pos, color):
        self.grid[pos] = color
        self.moves.append((pos, color))

    def undo(self):
        pos, _ = self.moves.pop()
        self.grid[pos] = 0

    def won(self, pos):
        """Freestyle: five or more consecutive stones through pos wins."""
        i, j = divmod(pos, self.size)
        color = self.grid[pos]
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = i + sign * di, j + sign * dj
                while (0 <= r < self.size and 0 <= c < self.size
                       and self.grid[r * self.size + c] == color):
                    count += 1
                    r += sign * di
                    c += sign * dj
            if count >= 5:
                return True
        return False

    def full(self):
        return all(self.grid)

    def features(self, to_move):
        """minizero 4-plane features, CHW, perspective of `to_move`."""
        f = np.zeros((4, self.size, self.size), dtype=np.float32)
        for pos, color in self.moves:
            ch = 0 if color == to_move else 1
            f[ch][divmod(pos, self.size)] = 1.0
        f[2 if to_move == 1 else 3, :, :] = 1.0
        return f


# ------------------------------------------------------------- network ----

class Network:
    """ONNX session wrapper for the exported MiniZero network."""

    def __init__(self, model_path):
        self.sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])

    def evaluate(self, features, legal_mask):
        """Return (policy[N*N], value) with illegal moves masked out."""
        state = features[np.newaxis]
        outs = self.sess.run(None, {"state": state})
        by_name = {o.name: arr for o, arr
                   in zip(self.sess.get_outputs(), outs)}
        logit = by_name["policy_logit"][0]
        value = float(by_name["value"][0][0])
        policy = np.exp(logit - logit.max())
        policy *= legal_mask
        total = policy.sum()
        if total <= 0:                                # all mass on illegal
            policy = legal_mask.astype(np.float32)
            total = policy.sum()
        return policy / total, value


# ---------------------------------------------------------------- search ----

class Node:
    __slots__ = ("parent", "pos", "color", "children", "prior",
                 "visits", "value_sum", "expanded")

    def __init__(self, parent, pos, color):
        self.parent = parent
        self.pos = pos                 # -1 for the root
        self.color = color             # colour that PLAYED pos
        self.children = []
        self.prior = 0.0
        self.visits = 0
        self.value_sum = 0.0
        self.expanded = False



def think(board, to_move, net, simulations):
    """PUCT search from `board` with `to_move` to play; returns best pos.
    Values are kept from the root player's perspective throughout."""
    root = Node(None, -1, 3 - to_move)
    mask = np.zeros(board.size * board.size, dtype=np.float32)
    root_moves = board.legal_moves()
    for p in root_moves:
        mask[p] = 1.0
    policy, value = net.evaluate(board.features(to_move), mask)
    root.expanded = True
    for p in root_moves:
        ch = Node(root, p, to_move)
        ch.prior = float(policy[p])
        root.children.append(ch)
    root.visits = 1
    root.value_sum = value

    puct_bias = lambda n: PUCT_INIT + math.log((1 + n + PUCT_BASE) / PUCT_BASE)

    for _ in range(simulations):
        node = root
        path = [root]
        while node.expanded and node.children:
            # PUCT selection
            bias = puct_bias(root.visits)
            best, best_score = None, -INF
            for ch in node.children:
                q = (ch.value_sum / ch.visits) if ch.visits else 0.0
                u = bias * ch.prior * math.sqrt(root.visits) / (1 + ch.visits)
                score = u + q
                if score > best_score:
                    best, best_score = ch, score
            node = best
            board.play(node.pos, node.color)
            path.append(node)

        if node.pos >= 0 and board.won(node.pos):
            value = 1.0 if node.color == to_move else -1.0
        elif board.full():
            value = 0.0
        else:
            leaf_to_move = 3 - node.color
            mask = np.zeros(board.size * board.size, dtype=np.float32)
            moves = board.legal_moves()
            for p in moves:
                mask[p] = 1.0
            pol, v = net.evaluate(board.features(leaf_to_move), mask)
            node.expanded = True
            for p in moves:
                ch = Node(node, p, leaf_to_move)
                ch.prior = float(pol[p])
                node.children.append(ch)
            value = v if leaf_to_move == to_move else -v

        sign = 1.0 if node.color == to_move else -1.0
        for n in reversed(path):
            n.visits += 1
            n.value_sum += sign * value
            sign = -sign
        for _ in path[1:]:
            board.undo()

    best = max(root.children, key=lambda c: c.visits)
    return best.pos


# -------------------------------------------------------------- protocol ----

class Protocol:
    def __init__(self, model, sims):
        self.net = Network(model)
        self.sims = sims
        self.board = Board(SIZE)
        self.timeout_turn_ms = 10000.0
        self.time_left_ms = 100000.0

    def reply(self, line):
        print(line, flush=True)

    def run(self):
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].upper()
            if cmd in ("START", "RECTSTART"):
                self.board = Board(SIZE)
                self.reply("OK")
            elif cmd == "BEGIN":
                self.reply(self._coord(think(
                    self.board, 1, self.net, self.sims)))
            elif cmd == "TURN":
                x, y = map(int, parts[1].split(","))
                self.board.play(y * SIZE + x, self._side_to_move())
                self.reply(self._coord(think(
                    self.board, self._side_to_move(), self.net, self.sims)))
            elif cmd == "BOARD":
                self._board_mode()
            elif cmd == "INFO":
                self._info(parts)
            elif cmd == "ABOUT":
                self.reply('name="pbrain-oogzero", version="1.0", '
                           'author="minh", country="VN"')
            elif cmd == "END":
                break
            else:
                self.reply("UNKNOWN")

    # helpers -----------------------------------------------------------

    def _side_to_move(self):
        black = sum(1 for _, c in self.board.moves if c == 1)
        white = len(self.board.moves) - black
        return 1 if black == white else 2

    def _coord(self, pos):
        i, j = divmod(pos, SIZE)
        return f"{j},{i}"

    def _board_mode(self):
        # Fields are RELATIVE: 1 = our stone, 2 = opponent's. Reconstruct an
        # alternating sequence (order does not affect Gomoku legality, only
        # occupancy -- see katagomo_bridge.py's BOARD handler).
        own, opp = [], []
        while True:
            line = sys.stdin.readline().strip()
            if not line:
                continue
            if line.upper() == "DONE":
                break
            try:
                x, y, f = line.split(",")
                x, y, f = int(x), int(y), int(f)
            except ValueError:
                continue
            (own if f == 1 else opp).append((x, y))
        # our colour: if counts are equal, black is to move and we ARE
        # black (we move after BOARD); if opponent has one extra, they are
        # black and we are white.
        if len(own) == len(opp):
            us = 1
        else:
            us = 2
        seq = []
        if us == 1:
            for i in range(max(len(own), len(opp))):
                if i < len(opp):
                    seq.append((opp[i], 2))
                if i < len(own):
                    seq.append((own[i], 1))
        else:
            for i in range(max(len(own), len(opp))):
                if i < len(opp):
                    seq.append((opp[i], 1))
                if i < len(own):
                    seq.append((own[i], 2))
        for (x, y), color in seq:
            self.board.play(y * SIZE + x, color)
        self.reply(self._coord(think(
            self.board, self._side_to_move(), self.net, self.sims)))

    def _info(self, parts):
        i = 1
        while i < len(parts):
            key = parts[i].upper()
            val = parts[i + 1] if i + 1 < len(parts) else ""
            if key == "TIMEOUT_TURN":
                try:
                    self.timeout_turn_ms = float(val)
                except ValueError:
                    pass
            elif key == "TIME_LEFT":
                try:
                    self.time_left_ms = float(val)
                except ValueError:
                    pass
            i += 2


def main():
    args = sys.argv[1:]
    model = args[0] if args else os.environ.get("OOGZERO_MODEL")
    if not model or not os.path.exists(model):
        print("usage: pbrain_oogzero.py MODEL.onnx [SIMS]", file=sys.stderr)
        sys.exit(1)
    sims = int(args[1]) if len(args) > 1 else 400
    Protocol(model, sims).run()


if __name__ == "__main__":
    main()
