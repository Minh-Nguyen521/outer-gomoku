"""PUCT Monte-Carlo tree search over the Outer-Open Gomoku environment.

Standard AlphaZero search: select by PUCT, expand one leaf per simulation
using the network's policy as a prior and its value as the leaf estimate,
back up with the sign flipping at every ply.

Three things here are specific to this game and this project:

  * TERMINAL NODES ARE NEVER EXPANDED and never take a network value. A
    finished game has a known result; asking the network to estimate it adds
    noise and, worse, lets a losing line look survivable.

  * CANDIDATES ARE RESTRICTED to the distance-2 neighbourhood (see env), which
    is what makes a few hundred simulations meaningful at a branching factor
    that starts at 225.

  * FORCED MOVES SHORT-CIRCUIT THE SEARCH ENTIRELY. If we can make five now, or
    must block five now, the search is skipped. This is not an optimisation --
    tools/nn_mcts_player.py records a predecessor losing 0-20, with every
    failure traced to a missed immediate block rather than to evaluation.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .env import (Board, NUM_ACTIONS, candidate_mask, forced_move)


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "player")

    def __init__(self, prior: float, player: int) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}
        self.player = player          # side to move AT this node

    @property
    def q(self) -> float:
        """Mean value from the perspective of the side to move here.

        An unvisited node returns 0 (neutral) rather than -1 (pessimistic);
        with PUCT the prior term already handles exploration of fresh nodes,
        and a pessimistic default makes the search reluctant to look at them.
        """
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    def __init__(self, net, device, simulations: int = 400, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.15, dirichlet_frac: float = 0.25,
                 radius: int = 2, gumbel: bool = False,
                 n_considered: int = 16, c_visit: float = 50.0,
                 c_scale: float = 1.0) -> None:
        self.net = net
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        # Gumbel settings. `n_considered` is the paper's m: how many root
        # actions sequential halving starts from. 16 is their default and
        # suits a ~30-candidate neighbourhood. c_visit/c_scale are the sigma
        # transform's constants, also the paper's defaults.
        self.gumbel = gumbel
        self.n_considered = n_considered
        self.c_visit = c_visit
        self.c_scale = c_scale
        # 0.15 follows the AlphaZero convention of scaling alpha inversely with
        # the typical number of legal moves; Gomoku's ~30-square neighbourhood
        # sits between chess (0.3) and Go 19x19 (0.03).
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.radius = radius

    # -- network ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, board: Board):
        """(priors over legal candidates, value for side to move)."""
        x = torch.from_numpy(board.planes()[None, ...]).to(self.device)
        logits, value = self.net(x)
        logits = logits[0].float().cpu().numpy()

        mask = candidate_mask(board, self.radius)
        # Mask BEFORE the softmax. Zeroing after would leave the surviving
        # probabilities summing to less than one.
        logits = np.where(mask, logits, -np.inf)
        logits -= logits.max()
        exp = np.exp(logits)
        total = exp.sum()
        priors = exp / total if total > 0 else mask / max(mask.sum(), 1)
        return priors, float(value.item())

    # -- search -----------------------------------------------------------
    def run(self, board: Board, add_noise: bool = True) -> np.ndarray:
        """Return a visit-count distribution over all 225 actions.

        The board is restored to its entry state before returning.
        """
        root = Node(0.0, board.turn)
        priors, _ = self.evaluate(board)
        self._expand(root, board, priors)

        if add_noise and root.children:
            self._add_dirichlet(root)

        for _ in range(self.simulations):
            node, path, depth = root, [root], 0
            # -- select
            while node.expanded and not board.is_terminal():
                action, node = self._select(node)
                board.play(action)
                path.append(node)
                depth += 1

            # -- evaluate
            if board.is_terminal():
                # Known result, from the perspective of the side to move at
                # this node -- which has no move, so it is the loser unless
                # the game is drawn.
                value = board.result_for(node.player)
            else:
                priors, value = self.evaluate(board)
                self._expand(node, board, priors)

            # -- back up, flipping sign each ply
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value if n.player == node.player else -value

            for _ in range(depth):
                board.undo()

        counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.visits
        return counts

    # -- Gumbel root selection --------------------------------------------
    #
    # Danihelka, Guez, Schrittwieser & Silver, "Policy Improvement by Planning
    # with Gumbel" (ICLR 2022) -- the method MiniZero uses, and the reason it
    # trains at 50 simulations per move where vanilla AlphaZero needs ~800.
    #
    # THE PROBLEM IT SOLVES. Standard AlphaZero trains the policy toward the
    # root VISIT COUNTS. That is only a sound policy-improvement target when
    # the counts are informative, which needs a large budget: at 50
    # simulations over ~30 candidates most actions get one visit or none, and
    # the "distribution" is mostly the tie-breaking order. You can see that
    # directly in this package -- at 256 simulations on a quiet position the
    # top move held 2.0% of visits, i.e. no signal at all.
    #
    # WHAT REPLACES IT. Two changes, both at the root only; the rest of the
    # tree is unchanged PUCT.
    #
    #   1. ACTING. Sample Gumbel noise once per action, take the top `m` by
    #      (gumbel + logit) -- which is exactly sampling m actions without
    #      replacement from the prior -- then run SEQUENTIAL HALVING over
    #      those m, repeatedly simulating the survivors and discarding the
    #      worse half. The argmax of (gumbel + logit + sigma(q)) over the
    #      survivors is provably a policy improvement for ANY budget, however
    #      small. This also removes the need for Dirichlet noise: the Gumbel
    #      draw is the exploration.
    #
    #   2. LEARNING. The training target is not visit counts but
    #      softmax(logit + sigma(completedQ)), where completedQ uses the
    #      searched value for visited actions and a single mixed value
    #      estimate `v_mix` for the rest. Every action gets a sensible target
    #      even at tiny budgets, which is what makes low-simulation training
    #      work.

    def _sigma(self, root: Node, child: Node) -> float:
        """Monotone transform of Q, scaled by the root's visit count.

        `c_visit` keeps the transform from exploding early, when max_visits is
        small and Q estimates are least reliable. Q is negated because a
        child's value is stored from the CHILD's point of view.
        """
        max_visits = max((c.visits for c in root.children.values()), default=0)
        return (self.c_visit + max_visits) * self.c_scale * (-child.q)

    def _simulate_from_child(self, root: Node, board: Board, action: int) -> None:
        """One simulation whose first move at the root is fixed to `action`."""
        node = root.children[action]
        board.play(action)
        path, depth = [root, node], 1

        while node.expanded and not board.is_terminal():
            nxt_action, node = self._select(node)
            board.play(nxt_action)
            path.append(node)
            depth += 1

        if board.is_terminal():
            value = board.result_for(node.player)
        else:
            priors, value = self.evaluate(board)
            self._expand(node, board, priors)

        leaf_player = node.player
        for n in reversed(path):
            n.visits += 1
            n.value_sum += value if n.player == leaf_player else -value
        for _ in range(depth):
            board.undo()

    def _completed_policy(self, root: Node, root_value: float) -> np.ndarray:
        """softmax(logit + sigma(completedQ)) over all root children.

        Unvisited actions take `v_mix`, the paper's blend of the root's own
        value with the prior-weighted value of the actions that WERE searched.
        Using a flat 0, or the root value alone, would systematically mis-rank
        everything the budget did not reach.
        """
        actions = list(root.children)
        priors = np.array([root.children[a].prior for a in actions], dtype=np.float64)
        priors = priors / max(priors.sum(), 1e-12)
        visits = np.array([root.children[a].visits for a in actions], dtype=np.float64)
        qs = np.array([-root.children[a].q for a in actions], dtype=np.float64)

        total_visits = visits.sum()
        seen = visits > 0
        if seen.any() and total_visits > 0:
            weighted = float((priors[seen] * qs[seen]).sum())
            weight = float(priors[seen].sum())
            v_mix = (root_value + total_visits * (weighted / max(weight, 1e-12))) \
                / (1.0 + total_visits)
        else:
            v_mix = root_value

        completed = np.where(seen, qs, v_mix)
        max_visits = visits.max() if len(visits) else 0
        logits = np.log(np.maximum(priors, 1e-12))
        scores = logits + (self.c_visit + max_visits) * self.c_scale * completed
        scores -= scores.max()
        exp = np.exp(scores)
        probs = exp / exp.sum()

        target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a, p in zip(actions, probs):
            target[a] = p
        return target

    def run_gumbel(self, board: Board):
        """Gumbel root search. Returns (action, improved policy target).

        The board is restored to its entry state before returning.
        """
        root = Node(0.0, board.turn)
        priors, root_value = self.evaluate(board)
        self._expand(root, board, priors)

        actions = list(root.children)
        if not actions:
            return None, np.zeros(NUM_ACTIONS, dtype=np.float32)

        logits = np.log(np.maximum(
            np.array([root.children[a].prior for a in actions]), 1e-12))
        gumbel = np.random.gumbel(size=len(actions))
        m = int(min(self.n_considered, len(actions)))
        chosen = np.argsort(-(gumbel + logits))[:m]
        # g + logit is fixed for the whole search; only sigma(q) moves.
        gl = {actions[i]: float(gumbel[i] + logits[i]) for i in chosen}
        remaining = [actions[i] for i in chosen]

        phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        while True:
            per_action = max(1, self.simulations // (phases * max(len(remaining), 1)))
            for action in remaining:
                for _ in range(per_action):
                    self._simulate_from_child(root, board, action)
            if len(remaining) <= 1:
                break
            remaining.sort(key=lambda a: -(gl[a] + self._sigma(root, root.children[a])))
            remaining = remaining[:max(1, len(remaining) // 2)]

        best = max(remaining, key=lambda a: gl[a] + self._sigma(root, root.children[a]))
        return int(best), self._completed_policy(root, root_value)

    # -- batched search ---------------------------------------------------
    #
    # WHY THIS EXISTS. `run()` above evaluates ONE position per network call,
    # so a GPU sits idle between forward passes and self-play is bound by
    # kernel-launch latency rather than by arithmetic. At the training
    # defaults that is ~320,000 serial passes per iteration.
    #
    # The fix used here is GAME-parallel, not leaf-parallel: N independent
    # searches advance in lockstep, and the one leaf each produces per
    # simulation is gathered into a single batch of N. This is the same shape
    # MiniZero uses for training self-play, and it was chosen over
    # leaf-parallelism inside one tree deliberately -- collecting several
    # leaves from ONE tree requires virtual loss to stop every path selecting
    # the same child, which changes what the search explores. Batching across
    # independent games changes nothing about any individual search: each tree
    # still sees exactly one leaf per simulation, so `run_many` returns the
    # same distribution `run` would, only faster.

    @torch.no_grad()
    def evaluate_many(self, boards: list[Board]):
        """One forward pass for many positions -> (priors list, values list)."""
        if not boards:
            return [], []
        x = torch.from_numpy(np.stack([b.planes() for b in boards])).to(self.device)
        logits, values = self.net(x)
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy().reshape(-1)

        priors = []
        for i, board in enumerate(boards):
            mask = candidate_mask(board, self.radius)
            lg = np.where(mask, logits[i], -np.inf)
            lg = lg - lg.max()
            exp = np.exp(lg)
            total = exp.sum()
            priors.append(exp / total if total > 0 else mask / max(mask.sum(), 1))
        return priors, [float(v) for v in values]

    def run_many(self, boards: list[Board], add_noise: bool = True):
        """Visit-count distributions for several positions at once.

        Every board is restored to its entry state before returning, exactly
        as `run` does. A board that is already terminal yields all-zero counts
        rather than being searched.
        """
        n = len(boards)
        roots = [Node(0.0, b.turn) for b in boards]
        priors, _ = self.evaluate_many(boards)
        for root, board, prior in zip(roots, boards, priors):
            self._expand(root, board, prior)
            if add_noise and root.children:
                self._add_dirichlet(root)

        for _ in range(self.simulations):
            paths: list[list[Node]] = []
            depths: list[int] = []
            leaves: list[Node] = []
            pending: list[Board] = []       # leaves needing a network value
            pending_idx: list[int] = []

            # -- select, one leaf per board
            for i, (board, root) in enumerate(zip(boards, roots)):
                node, path, depth = root, [root], 0
                while node.expanded and not board.is_terminal():
                    action, node = self._select(node)
                    board.play(action)
                    path.append(node)
                    depth += 1
                paths.append(path)
                depths.append(depth)
                leaves.append(node)
                if not board.is_terminal():
                    pending.append(board)
                    pending_idx.append(i)

            # -- ONE network call for every non-terminal leaf
            batch_priors, batch_values = self.evaluate_many(pending)

            values: list[float | None] = [None] * n
            for k, i in enumerate(pending_idx):
                # Expand BEFORE undoing: `boards[i]` is still at the leaf here,
                # and _expand reads its side to move.
                self._expand(leaves[i], boards[i], batch_priors[k])
                values[i] = batch_values[k]
            for i, board in enumerate(boards):
                if values[i] is None:       # terminal: the result is known
                    values[i] = board.result_for(leaves[i].player)

            # -- back up and rewind
            for i, board in enumerate(boards):
                value, leaf = values[i], leaves[i]
                for node in reversed(paths[i]):
                    node.visits += 1
                    node.value_sum += value if node.player == leaf.player else -value
                for _ in range(depths[i]):
                    board.undo()

        out = []
        for root in roots:
            counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for action, child in root.children.items():
                counts[action] = child.visits
            out.append(counts)
        return out

    def choose_many(self, boards: list[Board], temperature: float = 1.0,
                    add_noise: bool = True):
        """`choose` for several boards at once -> list of (action, target).

        Forced positions are resolved without search and are kept OUT of the
        batch, so a batch of 64 games where 5 are forced costs a 59-wide
        forward pass rather than 64.
        """
        results: list[tuple[int, np.ndarray] | None] = [None] * len(boards)
        todo, todo_idx = [], []
        for i, board in enumerate(boards):
            forced = forced_move(board)
            if forced is not None:
                target = np.zeros(NUM_ACTIONS, dtype=np.float32)
                target[forced] = 1.0
                results[i] = (forced, target)
            else:
                todo.append(board)
                todo_idx.append(i)

        if todo and self.gumbel:
            # Gumbel's sequential halving keeps per-root state that does not
            # step in lockstep across boards, so these run one board at a
            # time. The network call inside is still batch-1 -- closing that
            # gap means interleaving each root's halving schedule, which is
            # the obvious next optimisation and is NOT done here.
            for k, board in enumerate(todo):
                action, target = self.run_gumbel(board)
                if action is None:
                    legal = np.flatnonzero(board.legal_mask())
                    action = int(np.random.choice(legal))
                    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    target[action] = 1.0
                results[todo_idx[k]] = (int(action), target)
            return results

        if todo:
            for k, counts in enumerate(self.run_many(todo, add_noise=add_noise)):
                i = todo_idx[k]
                board = boards[i]
                total = counts.sum()
                if total == 0:
                    legal = np.flatnonzero(board.legal_mask())
                    action = int(np.random.choice(legal))
                    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    target[action] = 1.0
                    results[i] = (action, target)
                    continue
                target = counts / total
                if temperature <= 1e-3:
                    action = int(counts.argmax())
                else:
                    scaled = counts ** (1.0 / temperature)
                    action = int(np.random.choice(NUM_ACTIONS, p=scaled / scaled.sum()))
                results[i] = (action, target)
        return results

    def _expand(self, node: Node, board: Board, priors: np.ndarray) -> None:
        child_player = board.turn ^ 1
        for action in np.flatnonzero(priors > 0):
            node.children[int(action)] = Node(float(priors[action]), child_player)

    def _add_dirichlet(self, root: Node) -> None:
        """Root exploration noise, so self-play does not collapse onto one line."""
        actions = list(root.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, eps in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - self.dirichlet_frac) * child.prior + \
                          self.dirichlet_frac * float(eps)

    def _select(self, node: Node):
        """PUCT: argmax over Q + c_puct * P * sqrt(sum N) / (1 + N)."""
        total = math.sqrt(max(node.visits, 1))
        best, best_score, best_child = None, -float("inf"), None
        for action, child in node.children.items():
            # Child Q is from the CHILD's perspective (the opponent), so negate
            # it to score the move from this node's point of view.
            score = -child.q + self.c_puct * child.prior * total / (1 + child.visits)
            if score > best_score:
                best, best_score, best_child = action, score, child
        return best, best_child

    # -- move choice ------------------------------------------------------
    def choose(self, board: Board, temperature: float = 1.0,
               add_noise: bool = True):
        """(action, policy_target). Policy target is the visit distribution.

        The returned target is always the *untempered* normalised visit count:
        temperature governs how we SAMPLE, not what we teach. Training on a
        temperature-sharpened target would make the network imitate the
        exploration schedule rather than the search.
        """
        forced = forced_move(board)
        if forced is not None:
            target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            target[forced] = 1.0
            return forced, target

        if self.gumbel:
            # Gumbel does its own acting AND its own target. Temperature and
            # Dirichlet do not apply: the Gumbel draw already provides the
            # exploration, and sampling on top of it would break the
            # policy-improvement guarantee that is the whole point.
            action, target = self.run_gumbel(board)
            if action is not None:
                return action, target

        counts = self.run(board, add_noise=add_noise)
        total = counts.sum()
        if total == 0:                        # no candidate expanded: fall back
            legal = np.flatnonzero(board.legal_mask())
            action = int(np.random.choice(legal))
            target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            target[action] = 1.0
            return action, target

        target = counts / total
        if temperature <= 1e-3:
            action = int(counts.argmax())
        else:
            scaled = counts ** (1.0 / temperature)
            action = int(np.random.choice(NUM_ACTIONS, p=scaled / scaled.sum()))
        return action, target


__all__ = ["MCTS", "Node"]
