# The value head: both targets are broken, for opposite reasons

**Do not retrain the value head on the existing data. Neither available target
is sound, and swapping one for the other makes it worse, not better.**

The fix is upstream in data generation, not in the training script.

---

## What we believed

From the distillation post-mortem, recorded in `tools/train_policy.py`:

> policy: median 1.42 nats entropy -> ~4.1 effective moves out of 225. Sharp,
> dense, informative.
> value : only 7.3% of positions have a teacher value in 0.2-0.8, so 93%
> carry essentially no gradient. It also disagrees with the eventual game
> outcome 30% of the time.

The conclusion drawn was that the *teacher's value is unreliable*, so the
value head should learn from the **game result** instead. That is what
`train_policy.py` does today.

**That conclusion was wrong.** The teacher's value is fine. The game result is
the contaminated target.

## What the data actually says

Measured over all 20,835 labelled positions in `data/labelled.jsonl`:

    ply < 10, Black to move:  median value_stm = 0.9930   (65% above 0.99)
    ply < 10, White to move:  median value_stm = 0.0071   (66% below 0.01)

KataGomo is reporting that **Black is winning ~99% from the opening**. That is
not a malfunction — freestyle gomoku is a proven first-player win (Allis,
1993). The teacher is stating the game-theoretic truth.

Now the actual outcomes of the games those positions came from:

    winner = White : 15094  (72.4%)
    winner = Black :  5096  (24.5%)
    draw           :   645   (3.1%)

**White won 72.4% of the games in a variant Black is proven to win.** That
number is not a property of gomoku; it is a property of who was playing:

    slowrenju vs chis    n=7843
    chis      vs rapfi   n=7188
    slowrenju vs rapfi   n=5402

These are *mismatched* pairings, and the stronger engine happened to sit on
White in most of them. The game outcome therefore encodes **which engine was
playing**, not what the position was worth.

### This explains the 30% "disagreement" exactly

Agreement between the teacher's value and the game result, by ply:

    ply   0- 10:  28.9%      <- anti-correlated
    ply  10- 20:  56.4%
    ply  20- 30:  95.6%
    ply  30- 45:  94.8%
    ply  45+   :  77.7%

In the opening the teacher says "Black is winning" (true) while the record
says "White won" (also true, of that particular game) — so they disagree
almost every time. By ply 20-30 the game has actually gone wrong for Black in
a concrete way the teacher can see on the board, and agreement jumps to 95%.

The disagreement was never evidence against the teacher. It was evidence that
**game outcomes from mismatched engine pairs are not position values.**

## Why swapping targets does not fix it

- Train on the **game result**: the head learns "White usually wins", which is
  an artefact of our pairings. This is what we currently do.
- Train on the **teacher value**: the head learns "Black wins from the
  opening" — correct, but nearly constant, so it carries almost no positional
  information. Hence the 7.3% figure. That number is not a labelling defect;
  it is what a correct evaluator reports about a solved-for-first-player game.

Both targets are degenerate. An MCTS value needs to discriminate *between*
positions, and neither does.

## What would actually fix it

1. **Generate data in the variant we are targeting.** This project is
   Outer-Open Gomoku, whose whole purpose is to blunt Black's first-move
   advantage by confining Black's opening move to the outer two rows and
   columns. The labelled set was generated under **freestyle** (`--rule 0`,
   see [engine-ladder.md](engine-ladder.md)), where the value function is
   close to degenerate. Data in the actual variant should have a far less
   saturated value landscape.

2. **Remove the matchup asymmetry.** Self-play from a single engine, or
   paired openings with colours swapped, so an outcome reflects the position
   rather than which binary held which colour. `tools/match.py` already has
   `--paired-openings`; `tools/harvest_games.py` does not use it.

3. **Only then** consider the teacher's value as the target, labelled at high
   visit counts — and expect it to be informative in the contested middlegame
   band (ply 10-35), which is where it already shows gradient.

## Method note

This was found by inspecting the label distribution *before* retraining, which
took minutes. Retraining first would have produced a value head that looked
plausible on validation loss and was learning our engine pairings. Check what
the target distribution says about the world before fitting to it.
