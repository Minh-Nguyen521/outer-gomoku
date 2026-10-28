# Policy-ordered alpha-beta: measured, and abandoned

**Result: there is no move-ordering headroom in SearchV5. The idea is dead.**
Measured in about half an hour, before any neural-inference plumbing was
written. This document exists so nobody proposes it again without new
evidence.

---

## The idea

The distillation post-mortem (see [search-comparison.md](search-comparison.md))
left an asymmetric result:

- the teacher's **policy** is excellent — median 1.42 nats of entropy, i.e.
  **~4.1 effective moves out of 225**, top-1 probability ~0.46;
- the teacher's **value** is nearly useless — only 7.3% of positions carry any
  gradient, and it disagrees with the eventual game outcome 30% of the time.

Our distilled net reproduces the policy well: **98.9% top-1 agreement** with
the teacher's best move on held-out positions.

Both distilled players lost 0-20, and every failure traced to the value, not
the policy — MCTS cannot work without a value. So the proposal was to use the
half that works and discard the half that does not: feed the net's move
prediction into SearchV5's alpha-beta purely as an **ordering hint**, and keep
SearchV5's own proven evaluation.

Alpha-beta's node count is roughly exponential in ordering quality, so on
paper this is the cheapest possible win: no new training, both components
already exist.

## Why it fails

SearchV5 already orders moves with a TT move, killers, history,
counter-moves, threat scores, **and** internal iterative deepening — the last
of which exists, by its own comment, precisely because "ordering quality
dominates alpha-beta's node count". The open question was never whether a
neural hint is *accurate*; it is whether SearchV5's existing stack leaves
anything on the table.

### The measurement

`tools/ordering_headroom.cpp` (target `oog_headroom`) searches each position
twice **without clearing the TT** between runs. The second search finds a
stored best move at essentially every node it revisits, so its ordering is
close to perfect everywhere. The cold/warm node ratio is therefore an **upper
bound** on what any ordering scheme — including a perfect oracle — could buy.

It is a *loose* upper bound, and deliberately so: the warm search also reuses
TT scores and bounds, which cut nodes for reasons unrelated to ordering. If
even this inflated figure is small, the idea cannot work.

    depth 6, 20 positions:   mean 1.11x   median 0.84x
    depth 8, 24 positions:   mean 1.11x   median 0.83x

**There is no headroom.** The median is *below* 1.0 — the warm search
typically uses slightly *more* nodes, because `tt_.newSearch()` ages entries
and retained history can mis-order the second pass. Whatever the sign, the
effect is noise, not the multiple-x that would justify the work.

### Why this makes sense

Gomoku threats are unusually informative. A move that completes a four, or
blocks one, is almost always the move — and SearchV5's threat scoring finds it
directly and for free. A neural policy trained on the same positions largely
*rediscovers the same ordering*. High agreement with the teacher was never in
doubt; what the probe shows is that the engine already had that ordering.

This is the same lesson as the distillation failure, from the other side: the
policy is not where our engine is losing.

## Two traps in the probe itself

Both cost real time and would silently invalidate a rerun:

1. **`BitBoard::place(int color, int sq)`** takes colour *first*. Calling it
   `place(sq, color)` indexes an array by square number and crashes with
   SIGBUS — with no output at all, because stdout is lost on the signal.

2. **Harvested games share long deterministic opening prefixes.** Cutting
   every game at a fixed ply returns the *same position* over and over: the
   first run printed one node count 13 times and reported a confident,
   meaningless "median 2.65x". The probe now varies the cut ply per position
   and skips duplicate boards. Any future sampling from `harvest_all.jsonl`
   needs the same care.

## What this rules in

The negative result narrows the field usefully. The gap to the neural tier —
externally confirmed on Botzone, where SlowRenju is swept 0-4 by both Jax
(2072) and elu (1997) — is **not** a move-ordering gap, and is therefore not
reachable by bolting a policy onto a classical search.

That leaves the two options that address *evaluation* rather than ordering:

- **Retrain the value head on better targets.** The current head learns from
  game outcomes in a set that is 84.5% already-decided. Labelling from
  KataGomo's *search* value at high visit counts would give a value with real
  gradient — which is what MCTS needs to function at all.
- **MiniZero self-play training** (see [minizero.md](minizero.md)), which
  learns policy and value together and is the path the top of the Botzone
  ladder actually took.
