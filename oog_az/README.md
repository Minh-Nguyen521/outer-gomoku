# oog_az — AlphaZero for Outer-Open Gomoku

Self-play reinforcement learning for the 15×15 outer-open variant. The network
learns **only from games it plays against itself**; the engines in
`third_party/` are a measuring stick, never the training signal.

---

## Read this before running anything

**This implementation will not produce a strong engine on any compute you are
likely to give it.** That is not false modesty, it is arithmetic — see
[Compute](#compute-the-honest-number) below. It exists to be understood,
modified and measured. If the goal is a strong Outer-Open net, **`third_party/minizero`
is the better bet**: it won the 2025 ICGA Computer Olympiad gold at this exact
variant, implements the rule natively, and `tools/colab_minizero.sh` already
drives its training.

---

## Why the learning signal is self-play, not our engines

The request that produced this package was to "train with these engines". That
is imitation learning, not AlphaZero, and **this project already tried it and it
failed**. From `docs/value-head.md`: a policy distilled from KataGomo reached
**98.9% top-1 agreement** with its teacher, and both resulting players still
**lost 0-20**. The policy was never the problem; the value head had no usable
target.

Engines have three legitimate roles here, and all three are wired in:

| Role | Where | Why |
|---|---|---|
| Arena / gating | `arena.py` | An absolute yardstick. Self-play Elo can rise while real strength does not. |
| Evaluation ladder | `arena.py` `LADDER` | Graded opponents ~1450–2700 to track a run against. |
| Optional bootstrap | *not implemented* | See below — the data for it does not exist yet. |

The bootstrap is deliberately absent rather than half-built. `docs/value-head.md`
shows the existing 20,835 labelled positions are unusable for it: they were
generated under **freestyle, centre-opened**, with **0 of 22 games starting on
the ring**, and their outcomes encode which engine held which colour (White won
72.4% in a variant Black is proven to win). Regenerating that data in-variant is
a prerequisite, not a detail.

## Why gating exists

`train.py` gates every candidate against the previous network, and `arena.py`
gates against an outside engine. The second is not redundant. From
`docs/search-comparison.md` Round 7: **SearchV4 won all eight of its self-play
pairings at 80.0%, then scored *below* the untouched baseline against an
external opponent.** Six rounds of work, validated only against engines sharing
its own blind spots, produced no real improvement.

> Anchor externally before trusting any self-play delta.

---

## Layout

| File | What it is |
|---|---|
| `env.py` | Board, ring legality, overline win detection, 8-fold symmetry |
| `net.py` | Residual policy+value net, device selection, checkpoints |
| `mcts.py` | PUCT search, Dirichlet noise, forced-move short circuit |
| `selfplay.py` | Game generation, replay buffer |
| `train.py` | Loss, optimisation, the iteration loop, self-gating |
| `arena.py` | Measure a net against the external ladder |
| `player.py` | Gomocup-protocol adapter for `tools/match.py` |

## The rules, as implemented

* **Freestyle**: five **or more** wins. An overline is a win, not a null.
  `env.wins_at` uses `>= 5`; changing it to `== 5` changes the game and would
  make every measurement against a freestyle opponent invalid.
* **Outer-open**: Black's first stone must be in the outer two rows/columns.
  Implemented as a **legality mask on move 1** (`env.Board.legal_mask`), matching
  MiniZero (`gomoku.cpp:53`). Not a feature plane — on move 1 the board is
  empty, so a plane has nothing position-dependent to say.

## Input planes (5)

`own · opponent · empty · black-to-move · ones`, all side-to-move relative.

There is deliberately **no last-move plane**, though recency is normally
valuable in Gomoku. The Gomocup `BOARD` command gives own/opponent stone *sets
with no move order*, so such a plane would be computable during training and not
at play time — the net would meet unfamiliar inputs in a real match. Any feature
added here must survive that round trip.

---

## Running it

```bash
# One training iteration, small, to see the pipeline work end to end.
python3 -m oog_az.train --run-dir oog_az/runs/dev \
    --iterations 1 --games 4 --simulations 50 --steps 50 \
    --batch-size 64 --blocks 2 --channels 32

# A real (still small) run. Resumes automatically from best.pt.
python3 -m oog_az.train --run-dir oog_az/runs/dev \
    --iterations 50 --games 40 --simulations 200 --steps 400

# Measure against the external ladder (starts with a 4-game probe).
python3 oog_az/arena.py --model oog_az/runs/dev/best.pt \
    --opponent chis --games 20 --simulations 400

# Use it as an engine anywhere match.py is used.
python3 tools/match.py --games 20 --parallel 1 --paired-openings --rule 0 \
    --timeout-turn 10000 \
    --engine-a "python3 oog_az/player.py --model oog_az/runs/dev/best.pt" \
    --engine-b third_party/Chis/Chis2.0/pbrain-chis
```

Metrics land in `runs/<name>/metrics.jsonl`. Checkpoints are `best.pt` plus one
per promoted iteration.

**Never run training and a measurement match at the same time.** Wall-clock
results here are contention-sensitive: the same matchup measured **22.0% serial
and 1.0% at 4 workers**, with no forfeits — both engines simply searched
shallower.

---

## Compute: the honest number

The dominant cost is that **MCTS here runs one position at a time, batch size 1**.
Every simulation is its own forward pass. Real AlphaZero implementations batch
evaluations across many parallel games and gain one to two orders of magnitude.

Rough arithmetic at the defaults: 200 simulations/move × ~40 moves = 8,000
forward passes per game; 40 games = **~320,000 forward passes per iteration**,
serial. On CPU that is hours per iteration; on a single T4, still well short of
what a meaningful run needs, because the GPU sits idle between batch-1 calls.

So:

* **Days to weeks of GPU time** before this is competitive with even the bottom
  of the ladder (Chis, ~1453).
* The first improvement worth making is **batched/parallel self-play**, not
  architecture or hyperparameters.
* MiniZero already solves all of this and won the event. Use this package to
  understand and experiment; use MiniZero to win.

## What is verified, and what is not

**Verified** (cheap tests, run at build time):

* Ring mask: exactly 104 legal first moves; centre rejected; **(2,7) rejected** —
  that is the third row, and the square that made ~1 game in 7 of every
  pre-2026-09-08 match start from an illegal position
* Ring applies to move 1 only (224 legal on move 2)
* Overline (six) wins; exactly five wins; four does not; diagonals detected
* `play()` raises on a non-ring first move
* Forced-move detection finds both its own five and the block
* Planes and policy transform identically under all 8 symmetries
* **The ring is invariant under all 8 symmetries** — which is what makes the
  augmentation in `selfplay.py` exact rather than approximate
* Network forward pass shapes; priors sum to 1 with support exactly the legal set
* MCTS returns a legal, on-ring first move and **restores the board** afterwards
* Checkpoint save/load round-trip; replay buffer save/load round-trip
* Training step runs; initial policy loss ≈ ln(225) = 5.42 as expected for an
  untrained net
* Protocol: `START 15`→OK, `START 20`→ERROR, `BEGIN`→ring move, `END`→clean exit
* **Identical `BOARD` three times returns the identical move** — the regression
  test for the most repeated bug in this repo (SlowRenju 18/20 games lost, Chis
  7/20 deadlocks, Carbon crashed mid-game)

**Not verified — no training run has been performed:**

* That the loop actually *learns*. Nothing here demonstrates strength gain.
* Gate thresholds, learning rate, simulation counts, buffer size: all
  first guesses, none tuned. `docs/search-comparison.md` is emphatic about what
  untuned constants are worth.
* Any match against a real engine.
* Behaviour on CUDA (developed on CPU/MPS).
* Long-run stability: buffer growth, checkpoint churn, resume-after-crash.
