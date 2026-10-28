"""Self-play game generation and the replay buffer.

THE LEARNING SIGNAL IS SELF-PLAY, NOT OTHER ENGINES. That distinction is the
whole design, and this project has already paid for getting it wrong: a policy
distilled from KataGomo reached 98.9% top-1 agreement with its teacher and the
resulting players still lost 0-20, because the value head had no usable target
(docs/value-head.md). Imitating strong engines teaches you their moves; it does
not teach you to evaluate positions. External engines earn their place in
arena.py, as a yardstick, not here.

VALUE TARGETS COME FROM THE GAME'S OWN OUTCOME, which in self-play is sound in
a way it was NOT for the harvested data. docs/value-head.md found that the
20,835 labelled positions were contaminated: they came from mismatched engine
pairs where the stronger engine usually held White, so White won 72.4% of games
in a variant Black is proven to win, and the "value" the head learned was
really which binary was playing. Self-play removes that by construction -- one
network plays both sides -- and the ring opening removes the other half, since
the position is not the freestyle first-player win either.

AUGMENTATION uses the board's 8-fold dihedral symmetry, which is exact here:
the rules are symmetric, INCLUDING the outer-open ring (the ring maps onto
itself under every rotation and reflection). Each position therefore yields 8
training samples at no extra search cost.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .env import (Board, NUM_ACTIONS, NUM_PLANES, transform_planes,
                  transform_policy)
from .mcts import MCTS


class ReplayBuffer:
    """Bounded FIFO over (planes, policy, value) samples, saved as .npz.

    Kept deliberately simple and on disk: a Colab VM can be recycled at any
    time, and this project has already lost an entire training run to exactly
    that (docs/minizero.md). Anything worth keeping must survive the VM.
    """

    def __init__(self, capacity: int = 300_000) -> None:
        self.capacity = capacity
        self.planes: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.values: list[float] = []

    def __len__(self) -> int:
        return len(self.values)

    def add(self, planes, policy, value) -> None:
        self.planes.append(planes.astype(np.float32))
        self.policies.append(policy.astype(np.float32))
        self.values.append(float(value))
        excess = len(self.values) - self.capacity
        if excess > 0:
            del self.planes[:excess]
            del self.policies[:excess]
            del self.values[:excess]

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        np.savez_compressed(
            tmp,
            planes=np.asarray(self.planes, dtype=np.float32),
            policies=np.asarray(self.policies, dtype=np.float32),
            values=np.asarray(self.values, dtype=np.float32),
        )
        # Atomic replace: a half-written buffer that looks complete is worse
        # than no buffer, and Colab kills processes mid-write.
        os.replace(tmp + ".npz", path)

    @classmethod
    def load(cls, path: str, capacity: int = 300_000) -> "ReplayBuffer":
        buf = cls(capacity)
        if not os.path.exists(path):
            return buf
        data = np.load(path)
        buf.planes = list(data["planes"])
        buf.policies = list(data["policies"])
        buf.values = list(data["values"])
        return buf

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.values), size=batch_size)
        return (
            np.asarray([self.planes[i] for i in idx], dtype=np.float32),
            np.asarray([self.policies[i] for i in idx], dtype=np.float32),
            np.asarray([self.values[i] for i in idx], dtype=np.float32),
        )


def play_game(mcts: MCTS, temp_moves: int = 12, max_plies: int = 225,
              augment: bool = True):
    """Play one self-play game. Returns (samples, winner, plies).

    `temp_moves` is the number of opening plies sampled at temperature 1 for
    diversity; after that the search plays its best move. Without it every game
    from the same opening is near-identical and the buffer fills with
    duplicates -- docs/policy-ordering.md hit precisely that when sampling
    harvested games at a fixed ply and got "one node count printed 13 times".
    """
    board = Board()
    # (planes, policy, player_to_move) -- the value is unknown until the end.
    pending: list[tuple[np.ndarray, np.ndarray, int]] = []

    while not board.is_terminal() and len(board.history) < max_plies:
        temperature = 1.0 if len(board.history) < temp_moves else 0.0
        player = board.turn
        action, policy = mcts.choose(board, temperature=temperature,
                                     add_noise=True)
        pending.append((board.planes(), policy, player))
        board.play(action)

    if board.winner is None:          # hit max_plies without resolution
        board.winner = -1

    samples = []
    for planes, policy, player in pending:
        value = board.result_for(player)
        if augment:
            for sym in range(8):
                samples.append((transform_planes(planes, sym),
                                transform_policy(policy, sym),
                                value))
        else:
            samples.append((planes, policy, value))
    return samples, board.winner, len(board.history)


def generate_parallel(net, device, games: int, simulations: int,
                      buffer: ReplayBuffer, concurrency: int = 32,
                      log_path: str | None = None, temp_moves: int = 12,
                      augment: bool = True, max_plies: int = 225,
                      quiet: bool = False, progress_every: int = 5,
                      gumbel: bool = False, n_considered: int = 16):
    """Self-play with the network evaluated in batches across games.

    THIS IS THE ONE THAT SHOULD BE USED FOR TRAINING. `generate` below plays
    games one at a time, so every MCTS simulation is a batch-of-one forward
    pass and the accelerator idles between launches -- at the training
    defaults that is roughly 320,000 serial passes per iteration, and it was
    the dominant cost of this package.

    Here `concurrency` games advance in lockstep. Each contributes one leaf
    per simulation, so the network sees batches of up to `concurrency`
    positions. The searches remain completely independent: batching across
    games (rather than collecting several leaves from one tree) means no
    virtual loss is needed and no individual search behaves differently from
    the serial version.

    Games finish at different lengths, so the active set shrinks and is topped
    up from the remaining budget -- otherwise the batch would decay towards 1
    exactly when the endgames that matter most are being played.
    """
    mcts = MCTS(net, device, simulations=simulations, gumbel=gumbel,
                n_considered=n_considered)
    wins = {0: 0, 1: 0, -1: 0}
    plies_total = 0
    finished = 0
    started = 0
    t0 = time.time()

    # Each slot: (board, pending samples awaiting the game's outcome)
    slots: list[tuple[Board, list]] = []

    def start_game():
        nonlocal started
        started += 1
        return (Board(), [])

    while len(slots) < min(concurrency, games):
        slots.append(start_game())

    while slots:
        boards = [s[0] for s in slots]
        # Temperature is per-game: it depends on how far THAT game has got.
        # A single batch can mix opening and endgame positions, so the sample
        # step is done per slot after the shared search.
        results = mcts.choose_many(boards, temperature=1.0, add_noise=True)

        next_slots = []
        for slot, (action, target) in zip(slots, results):
            board, pending = slot
            # Re-derive the move at this game's own temperature. choose_many
            # sampled at T=1; past the opening we want the argmax instead.
            if (not gumbel) and len(board.history) >= temp_moves \
                    and target.sum() > 0:
                action = int(np.argmax(target))
            pending.append((board.planes(), target, board.turn))
            board.play(action)

            if board.is_terminal() or len(board.history) >= max_plies:
                if board.winner is None:
                    board.winner = -1
                for planes, policy, player in pending:
                    value = board.result_for(player)
                    if augment:
                        for sym in range(8):
                            buffer.add(transform_planes(planes, sym),
                                       transform_policy(policy, sym), value)
                    else:
                        buffer.add(planes, policy, value)
                wins[board.winner] += 1
                plies_total += len(board.history)
                finished += 1
                if not quiet and finished % progress_every == 0:
                    rate = finished / max(time.time() - t0, 1e-6)
                    print(f"  {finished}/{games} games  "
                          f"({rate * 60:.1f}/min)  buffer={len(buffer)}",
                          flush=True)
                if started < games:          # top the batch back up
                    next_slots.append(start_game())
            else:
                next_slots.append(slot)
        slots = next_slots

    stats = {
        "games": finished,
        "concurrency": concurrency,
        "black_wins": wins[0],
        "white_wins": wins[1],
        "draws": wins[-1],
        "mean_plies": plies_total / max(finished, 1),
        "seconds": round(time.time() - t0, 1),
        "buffer": len(buffer),
    }
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(json.dumps({"stage": "selfplay", **stats}) + "\n")
    return stats


def generate(net, device, games: int, simulations: int, buffer: ReplayBuffer,
             log_path: str | None = None, temp_moves: int = 12,
             augment: bool = True, quiet: bool = False):
    """Fill `buffer` with `games` self-play games. Returns a stats dict."""
    mcts = MCTS(net, device, simulations=simulations)
    wins = {0: 0, 1: 0, -1: 0}
    plies_total = 0
    t0 = time.time()

    for g in range(games):
        samples, winner, plies = play_game(mcts, temp_moves=temp_moves,
                                           augment=augment)
        for planes, policy, value in samples:
            buffer.add(planes, policy, value)
        wins[winner] += 1
        plies_total += plies
        if not quiet:
            # Flush every game: a run that prints nothing until the end is
            # indistinguishable from a hung one, and this project has chased
            # that phantom twice (docs/search-comparison.md Rounds 6b and 7).
            print(f"  game {g + 1}/{games}: winner={winner} plies={plies} "
                  f"buffer={len(buffer)}", flush=True)

    stats = {
        "games": games,
        "black_wins": wins[0],
        "white_wins": wins[1],
        "draws": wins[-1],
        "mean_plies": plies_total / max(games, 1),
        "seconds": round(time.time() - t0, 1),
        "buffer": len(buffer),
    }
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(json.dumps({"stage": "selfplay", **stats}) + "\n")
    return stats


__all__ = ["ReplayBuffer", "play_game", "generate", "generate_parallel"]
