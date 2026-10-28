"""Outer-Open Gomoku environment: rules, legality, and feature planes.

THE RULES, precisely, because two of them have cost this project real games:

  * 15x15, FREESTYLE: five OR MORE in a row wins. An overline (six+) is a WIN,
    not a null. Engines built for the "Standard" variant treat an overline as
    nothing, and adjudicating one of those as freestyle punishes it for playing
    correctly -- that is how Embryo's three ladder rows were withdrawn. If you
    ever change `wins_at` to require exactly five, you have changed the game.

  * OUTER-OPEN: Black's FIRST stone must lie in the outer two rows or columns.
    From move two on, the game is identical to freestyle. That is the entire
    difference.

THE RING RULE IS A LEGALITY MASK, NOT A FEATURE PLANE.

This mirrors MiniZero, the 2025 Olympiad gold medallist at this exact variant
(third_party/minizero/.../gomoku.cpp:53):

    if (actions_.empty() && config::env_gomoku_rule == "outer_open")
        return (i < 2 || i >= size-2) || (j < 2 || j >= size-2);

and it is the right call rather than an imitation: the constraint binds only on
move 1, and on move 1 the board is empty, so a feature plane would have nothing
position-dependent to say. Masking illegal actions out of the policy teaches
the restriction implicitly and exactly. MiniZero's own network input carries no
ring encoding at all, only four plain planes.
"""

from __future__ import annotations

import numpy as np

N = 15
NUM_ACTIONS = N * N

# Four line directions. The scan walks both ways from the placed stone, so
# these four cover all eight rays.
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))

BLACK, WHITE = 0, 1

# ---------------------------------------------------------------------------
# Feature planes
# ---------------------------------------------------------------------------
# FIVE planes, all from the side-to-move's point of view so the network never
# has to learn colour symmetry:
#
#   0  own stones
#   1  opponent stones
#   2  empty squares
#   3  all-ones iff the side to move is BLACK (the first player)
#   4  all-ones constant
#
# Why these and not others:
#
#   * Plane 3 is not redundant even in a side-to-move-relative encoding.
#     Freestyle Gomoku is a proven first-player win (Allis 1993), and KataGomo
#     evaluates opening positions at ~0.99 for Black, so "am I the first
#     player" carries real information the other planes cannot supply.
#     MiniZero spends two planes on this (black-turn and white-turn); one
#     suffices, since white-turn is its complement.
#
#   * Plane 4 lets the convolutions recover the board edge. With zero padding,
#     an off-board cell and an empty cell are both zeros everywhere else; a
#     constant-one plane makes the boundary visible. tools/train_policy.py uses
#     the same trick.
#
#   * There is deliberately NO "last move" plane, though recency is normally
#     valuable in Gomoku. The Gomocup BOARD command sends own/opponent stone
#     SETS WITH NO MOVE ORDER (see tools/nn_player.py), so a last-move plane
#     would be reconstructible during training and NOT during play -- the
#     network would meet inputs at match time unlike anything it trained on.
#     Any feature added here must survive that round trip.
NUM_PLANES = 5


class Board:
    """Mutable position with O(1) make/undo and incremental win detection.

    Self-play generates millions of positions, so this avoids reallocating and
    avoids full-board win scans: a win can only be completed by the stone just
    placed, so `wins_at` checks the four lines through that square alone.
    """

    __slots__ = ("grid", "turn", "history", "winner", "_empty")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # -1 empty, 0 black, 1 white. int8 keeps it cache-friendly.
        self.grid = np.full((N, N), -1, dtype=np.int8)
        self.turn = BLACK
        self.history: list[int] = []
        self.winner: int | None = None      # None = ongoing, -1 = draw
        self._empty = NUM_ACTIONS

    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.grid = self.grid.copy()
        b.turn = self.turn
        b.history = list(self.history)
        b.winner = self.winner
        b._empty = self._empty
        return b

    # -- legality ----------------------------------------------------------
    @staticmethod
    def is_ring(r: int, c: int) -> bool:
        """The outer two rows OR columns. Must agree with the manager.

        tools/match.py:90 and src/bitboard.cpp:206 use this identical test. It
        previously did not agree everywhere: match.py's default opening list
        contained (2,7) and (12,7), the THIRD row from each edge, so roughly
        one game in seven of every Outer-Open match before 2026-09-08 began
        from a position the rule forbids.
        """
        return r < 2 or r >= N - 2 or c < 2 or c >= N - 2

    def legal_mask(self) -> np.ndarray:
        """Boolean mask over all 225 actions.

        The ring restriction applies only to the very first stone of the game.
        `len(self.history) == 0` is the exact condition MiniZero uses
        (`actions_.empty()`), and it is stricter than "the board is empty" only
        in that it cannot be fooled by a position loaded without history.
        """
        mask = (self.grid == -1).reshape(-1).copy()
        if not self.history:
            ring = np.zeros((N, N), dtype=bool)
            ring[:2, :] = True
            ring[N - 2:, :] = True
            ring[:, :2] = True
            ring[:, N - 2:] = True
            mask &= ring.reshape(-1)
        return mask

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask())

    # -- play --------------------------------------------------------------
    def play(self, action: int) -> None:
        r, c = divmod(action, N)
        if self.grid[r, c] != -1:
            raise ValueError(f"square {(r, c)} is occupied")
        if not self.history and not self.is_ring(r, c):
            raise ValueError(f"first move {(r, c)} is not on the outer ring")
        self.grid[r, c] = self.turn
        self.history.append(action)
        self._empty -= 1
        if self.wins_at(r, c):
            self.winner = self.turn
        elif self._empty == 0:
            self.winner = -1                         # board full: draw
        self.turn ^= 1

    def undo(self) -> None:
        action = self.history.pop()
        r, c = divmod(action, N)
        self.grid[r, c] = -1
        self._empty += 1
        self.winner = None
        self.turn ^= 1

    # -- terminal ----------------------------------------------------------
    def wins_at(self, r: int, c: int) -> bool:
        """Does the stone at (r,c) complete five OR MORE in a row?

        `>= 5`, not `== 5`. Overlines win in freestyle. See the module
        docstring -- this is the line that defines which game we are playing.
        """
        colour = self.grid[r, c]
        if colour == -1:
            return False
        g = self.grid
        for dr, dc in DIRS:
            count = 1
            for sign in (1, -1):
                rr, cc = r + dr * sign, c + dc * sign
                while 0 <= rr < N and 0 <= cc < N and g[rr, cc] == colour:
                    count += 1
                    rr += dr * sign
                    cc += dc * sign
            if count >= 5:
                return True
        return False

    def is_terminal(self) -> bool:
        return self.winner is not None

    def result_for(self, player: int) -> float:
        """+1 win, -1 loss, 0 draw, from `player`'s point of view."""
        if self.winner is None:
            raise ValueError("position is not terminal")
        if self.winner == -1:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    # -- features ----------------------------------------------------------
    def planes(self) -> np.ndarray:
        """(NUM_PLANES, N, N) float32, from the side-to-move's perspective."""
        return planes_from_grid(self.grid, self.turn)


def planes_from_grid(grid: np.ndarray, turn: int) -> np.ndarray:
    """Shared by the env and the Gomocup player, so the two cannot drift.

    tools/nn_player.py duplicated its encoder from the trainer and warned that
    the two must match exactly; one implementation used by both is strictly
    safer.
    """
    p = np.zeros((NUM_PLANES, N, N), dtype=np.float32)
    p[0] = (grid == turn)
    p[1] = (grid == (turn ^ 1))
    p[2] = (grid == -1)
    if turn == BLACK:
        p[3] = 1.0
    p[4] = 1.0
    return p


# ---------------------------------------------------------------------------
# Candidate restriction
# ---------------------------------------------------------------------------
def neighbourhood_mask(grid: np.ndarray, radius: int = 2) -> np.ndarray:
    """Empty squares within Chebyshev `radius` of any stone, as a flat mask.

    Every engine in this repo restricts to distance 2; it takes branching from
    225 to roughly 30 and is what makes a few hundred simulations viable. Note
    it is a HEURISTIC, not a rule: a strong move further than 2 from every
    stone becomes unplayable, and no amount of search recovers it. rapfi treats
    the radius as a tunable (square2 .. full_board) rather than a constant.

    Returns an all-False mask on an empty board; callers must fall back to the
    legal mask there (which, on move 1, is the ring).
    """
    occupied = (grid != -1)
    if not occupied.any():
        return np.zeros(NUM_ACTIONS, dtype=bool)
    near = np.zeros((N, N), dtype=bool)
    idx = np.argwhere(occupied)
    for r, c in idx:
        r0, r1 = max(0, r - radius), min(N, r + radius + 1)
        c0, c1 = max(0, c - radius), min(N, c + radius + 1)
        near[r0:r1, c0:c1] = True
    near &= ~occupied
    return near.reshape(-1)


def candidate_mask(board: Board, radius: int = 2) -> np.ndarray:
    """Legal moves, narrowed to the neighbourhood when that leaves anything."""
    legal = board.legal_mask()
    near = neighbourhood_mask(board.grid, radius) & legal
    return near if near.any() else legal


# ---------------------------------------------------------------------------
# Immediate tactics
# ---------------------------------------------------------------------------
def winning_moves(board: Board, player: int, mask: np.ndarray | None = None):
    """Actions where `player` completes five immediately.

    Scans the candidate neighbourhood rather than all 225 squares. That is
    sound for a *completion*: a five must touch existing stones of that colour,
    so it is always within distance 2 of one.
    """
    if mask is None:
        mask = candidate_mask(board)
    out = []
    g = board.grid
    for action in np.flatnonzero(mask):
        r, c = divmod(int(action), N)
        g[r, c] = player
        if board.wins_at(r, c):
            out.append(int(action))
        g[r, c] = -1
    return out


def forced_move(board: Board) -> int | None:
    """Play a win now, or block the opponent's win now. None if neither.

    This runs BEFORE any search, deliberately. tools/nn_mcts_player.py records
    why: its predecessor lost 0-20 with 20/20 clean fives, and the sample game
    showed the opponent building an unblocked vertical five that the network
    simply never answered. MCTS *should* find both given enough simulations; at
    a few hundred it may not, and each miss loses outright. Checking directly
    costs microseconds.
    """
    me = board.turn
    mask = candidate_mask(board)
    mine = winning_moves(board, me, mask)
    if mine:
        return mine[0]
    theirs = winning_moves(board, me ^ 1, mask)
    if theirs:
        # Only one block is possible against a single threat; against a genuine
        # double threat the position is already lost and any block will do.
        return theirs[0]
    return None


# ---------------------------------------------------------------------------
# Dihedral symmetry (8-fold), for training-data augmentation
# ---------------------------------------------------------------------------
def transform_planes(planes: np.ndarray, sym: int) -> np.ndarray:
    """Apply one of the 8 square symmetries to a (C, N, N) stack."""
    out = np.rot90(planes, sym % 4, axes=(1, 2))
    if sym >= 4:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def transform_policy(policy: np.ndarray, sym: int) -> np.ndarray:
    """Apply the same symmetry to a flat 225-vector of action probabilities."""
    grid = policy.reshape(N, N)
    grid = np.rot90(grid, sym % 4)
    if sym >= 4:
        grid = grid[:, ::-1]
    return np.ascontiguousarray(grid).reshape(-1)


__all__ = [
    "N", "NUM_ACTIONS", "NUM_PLANES", "BLACK", "WHITE", "Board",
    "planes_from_grid", "neighbourhood_mask", "candidate_mask",
    "winning_moves", "forced_move", "transform_planes", "transform_policy",
]
