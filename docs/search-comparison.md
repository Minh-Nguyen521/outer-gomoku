# Search strategy comparison (Rounds 1-7, ten engines)

Status: exploratory / not yet actionable for tournament play. Small sample
sizes throughout (see Caveats). Written up as a record of methodology and
results so a future, larger run can be compared against it.

> ## ⚠ CONTAMINATED: every external result below was measured with a broken harness
>
> Rounds 5-8 played our engines against rapfi while `tools/match.py` sent
> **absolute** BOARD fields (1 = Black) instead of the **relative** fields the
> Gomocup protocol requires (1 = the receiving engine's own stone). Whichever
> opponent played White therefore received a colour-swapped position and
> played the wrong side. Our own brain reads the field as an absolute colour,
> so self-play never exposed it.
>
> **Do not quote any percentage, Elo figure or per-colour split from those
> rounds.** That includes the "22.0% vs rapfi" and "-220 Elo" figures below,
> and the `SearchV4` 13.2%. The lopsided 8/50-as-Black vs 14/50-as-White split
> reported later in this file is the *fingerprint of this bug*; it was
> originally explained as inherent first-player advantage in Outer-Open, and
> that explanation was wrong.
>
> The fix and its rationale are documented in `tools/match.py`. Nothing here
> has been re-run. `docs/engine-ladder.md` contains post-fix measurements, but
> of third-party engines against each other, not of our engines.
>
> The *qualitative* conclusion of Round 7 does survive, because it was later
> reproduced with a working harness: self-play rankings do not predict
> external strength.

> **Read Round 7 first.** Rounds 1-6 are entirely self-play between engines
> that share one `eval.cpp`, one move generator and one search core. Round 7
> added an external opponent (rapfi) and **reversed the ranking**: the
> untouched baseline `Search` scores 22.0% against rapfi while `SearchV4` --
> which won all eight self-play pairings at 80.0% in Round 6 -- scores 13.2%
> pooled over 300 games. Treat every self-play number below as a measurement
> of relative behaviour inside one closed family, not as evidence of strength.

Also note Round 6's methodology finding before comparing any two numbers: the
wall-clock tables in Round 5 were produced with a parallel game runner and are
not valid. Wall-clock runs must be single-worker; parallel runs must be
node-budgeted.

Engines, in the order they were added: `Search` (baseline), `SearchV2`,
`SearchV3` (Rounds 1-4); `SearchVct`, `SearchNullKiller`, `SearchThreatCand`
(Round 5); `SearchV4` (Round 6); `SearchV5` (Round 6b); plus `SearchMinibal`
and `SearchUctPb`, which pursue goals other than raw strength.

**Round 8 conclusion:** the gap is in *evaluation*, not search. We lose 94/100
to rapfi with its neural network switched off entirely, and our own eval has
never been fitted to data.

## Background

`search.cpp`'s baseline `Search` implements a minimal Late Move Reduction
(LMR): a binary trigger (`i >= 4 && depth >= 3`) and a flat `-1` ply
reduction, plus TT-move ordering, a history heuristic, and a book bonus at
the root. Researching `dhbloo/rapfi` (this project's own named reference
engine, see `PLAN.md`) turned up a much larger set of search-control
techniques with measured Elo impact in its own source comments
(`Rapfi/search/ab/search.cpp`, `Rapfi/search/ab/parameter.h`), none of which
existed in this project. Two experimental variants were built to measure
whether adopting them helps here, entirely as separate classes so the
baseline is untouched:

- **`SearchV2`** (`src/search_v2.*`) — richer LMR only. Same trigger
  structure as baseline, but the reduction amount is a continuous
  `log(depth) * log(moveIndex)` formula (rapfi's own shape) instead of a
  flat `-1`, with four additive adjustments: a threat-aware exemption (a
  move scoring at or above an open three via `evaluateMove()` is never
  reduced), a history-based adjustment (reuses the existing history table),
  a PV-node-aware cut (window-width proxy, since this project's TT has no
  stored `ttPv` flag to reuse), and an "improving" trend check (this node's
  static eval vs. the same side's eval two plies back).

- **`SearchV3`** (`src/search_v3.*`) — `SearchV2`'s LMR plus the three
  largest single levers found in rapfi's own Elo comments that this project
  had none of:
  - **Futility pruning** (rapfi: ~121 elo, their single biggest lever) — if
    the static eval already clears `beta` by a comfortable margin at
    shallow remaining depth, skip the node outright, no moves generated.
  - **Move-count pruning** (rapfi: ~107 elo) — once enough moves have been
    tried at a shallow node without a cutoff, skip the (already
    threat-ordered) tail of the candidate list; never skips a threat move.
  - **Razoring with VCF** (rapfi: ~68 elo) — when eval is well below
    `alpha`, don't just trust a pessimistic eval: reuse this project's
    already-existing `vcf()` forced-win search as a cheap insurance check
    before pruning.
  - Mate-distance pruning thrown in too (free correctness/perf tightening,
    no measured Elo tag in rapfi).

  Both new node-level cuts are skipped whenever the opponent's last move
  left an immediate winning threat on the board (`hasImmediateWin`),
  mirroring rapfi's "oppo4" gate — a forced position shouldn't be pruned on
  an eval that's likely misleading there. This check is a ~272-cell board
  scan, so it is deliberately ordered *last*, only evaluated once a cheap
  eval-margin condition has already decided to prune (see Methodology
  Notes below — this ordering bug was caught and fixed during the run).

All margins/constants in `SearchV3` (`razorMargin`, `futilityMargin`,
move-count limit) are **untuned first guesses** in this project's own
`evaluate()` score units (`S_CLOSED_TWO=100` .. `S_FIVE=1,000,000`, see
`eval.cpp`), sized to be roughly the right order of magnitude, not derived
from rapfi's numbers (which are in a different, NNUE-scaled unit system) or
from any tuning process. Real values are a Phase 7 (self-play tuning) task.

## Methodology

- Tool: `tools/compare_search.cpp`, built as `oog_compare`
  (`OOG_BUILD_COMPARE` CMake option, on by default).
- **Node efficiency test**: all three engines run on the same 3 fixed
  midgame positions (reached via a few `Board::makeMove` calls each) at a
  matched node budget, reporting `depth`/`score`/`nodes` from
  `SearchResult`.
- **Round-robin self-play**: each pair of engines plays 8 games (4 fixed
  ring-legal openings × both color assignments) at a matched **node**
  budget per move (not wall-clock), so results are reproducible run to run.
  `clearTT()` is called between games. No draws occurred in any run so far.
- All numbers below are from real executions of the built binaries on this
  machine, not estimates.

## Results

### Node efficiency @ 300,000 nodes/side (first run, `SearchV2` only)

| Position | Search | SearchV2 |
|---|---|---|
| ring-open + center contact | depth 7, score 18800 | depth 7, score 11850 |
| corner open + contact | depth 7, score 300 | depth 7, score 1550 |
| edge open + spread | **depth 8**, score -1850 | depth 7, score 9750 |

At this budget, `SearchV2` did not reach greater depth than baseline in any
position — in one case it reached one ply *less*. Its extra bookkeeping
(recomputed threat scores, `evalAtPly_` tracking, the multi-term reduction
formula) costs something per node that wasn't recovered as extra depth here.

### Self-play @ 150,000 nodes/move (first run, `SearchV2` only, 8 games)

```
Search 3 - 5 SearchV2 (draws: 0)
```
Most games followed a strong-Black-advantage pattern (whoever played Black
won); the `(14,7)` opening was the exception, with `SearchV2` winning as
both colors.

### Node efficiency @ 5,000 nodes/side (second run, all three engines)

| Position | Search | SearchV2 | SearchV3 |
|---|---|---|---|
| ring-open + center contact | depth 4, score 9200 | depth 5, score 10600 | **depth 6**, score 1700 |
| corner open + contact | depth 4, score -2500 | depth 5, score 5450 | **depth 6**, score -400 |
| edge open + spread | depth 4, score -3350 | depth 4, score -3600 | **depth 7**, score 3450 |

At this tighter budget, `SearchV3` reached 2-3 plies deeper than baseline on
every position — its pruning is converting the same node budget into
materially more real search depth.

### Round-robin self-play @ 5,000 nodes/move (second run, 8 games per pairing)

```
Search   2 - 6  SearchV3
SearchV2 1 - 7  SearchV3
Search   6 - 2  SearchV2
```

`SearchV3` won decisively against both other engines. `SearchV2` alone
**lost** to the plain baseline at this budget — the opposite direction from
the 300k-node run above (`SearchV2` won 5-3 there). See Discussion.

## Discussion

1. **The pruning techniques (`SearchV3`), not the LMR refinement
   (`SearchV2`) alone, are where the measured value is** — this matches the
   research: rapfi's own Elo comments rank futility pruning (~121) and
   move-count pruning (~107) well above any single LMR refinement term
   (~2-59 elo each). The 5,000-node round-robin result (`SearchV3` beating
   both other engines by wide margins, and reaching 2-3 plies deeper on the
   fixed positions) is consistent with that ranking.

2. **`SearchV2` vs. `Search` flipped direction between the two budgets
   tested** (won 5-3 at 300k nodes/move, lost 2-6 at 5k nodes/move). This
   was not deliberately investigated further and should be treated as an
   open question, not noise to dismiss — it suggests the LMR refinement's
   value (if any) may be budget/depth-dependent, or that 8 games at either
   budget is simply too few to separate a real effect from variance. Either
   explanation implies more games at more budgets are needed before trusting
   a direction here.

3. **A real bug was caught and fixed mid-experiment**: the first `SearchV3`
   implementation called `hasImmediateWin()` (a ~272-cell board scan)
   unconditionally at every eligible node before checking the cheap
   eval-margin condition, making the comparison run take longer than the
   10-minute tool timeout even at a 5,000-node budget. Reordering the checks
   so the expensive scan only runs once a cheap condition has already
   decided to prune fixed it. Worth remembering as a general lesson for any
   future node-level pruning addition here: order guards from cheapest to
   most expensive.

## Caveats (read before acting on any of the above)

- **8 games per pairing is a small sample.** The 5,000-node round-robin
  margins are wide enough (6-2, 7-1, 2-6) to suggest a real trend across all
  three pairings agreeing with each other, but this is not a rigorous Elo
  measurement. `tools/match.py` (this project's existing parallel match
  runner) is built for running hundreds of games and should be the next
  step before trusting a specific number.
- **Only two node budgets tested** (300k nodes/side for `SearchV2` alone,
  5k nodes/move for the three-way round-robin). The budget-dependent
  direction flip for `SearchV2` (finding 2 above) means a conclusion at one
  budget should not be assumed to hold at another, including realistic
  tournament time controls.
- **Node budgets, not wall-clock time, were used throughout** for
  reproducibility. `SearchV3` is measurably heavier per node (the board
  scans), so its advantage under a *wall-clock* budget (what actually
  matters in a real game) has not been measured and could be smaller than
  under a matched node budget, or could still be larger if its deeper
  effective search dominates — untested either way.
- **All `SearchV3` margins are untuned first guesses**, explicitly
  documented as such in `search_v3.cpp`. The reported results are for this
  specific set of constants, not for "futility/move-count pruning in
  general" — different margins could change the outcome.
- **3 fixed positions** for the node-efficiency table; not a representative
  sample of opening/midgame/endgame positions.

## Round 3: five strategies, including two from `docs/related-work.md`

Two more variants were added, both built as fully separate classes on the
same pattern as `SearchV2`/`SearchV3`:

- **`SearchMinibal`** (`src/search_minibal.*`) — ports the *decision rule*
  from Cohen-Solal & Cazenave's Minibal_+ (not its Unbounded-Minimax search
  backbone) onto this project's ordinary alpha-beta: everywhere except the
  root, its `negamax` is identical to baseline `Search`. Only the root move
  choice differs — every candidate is searched at a full window (so each
  has a true, comparable value), then the smallest *positive* score is
  chosen if any move wins, otherwise the largest (least negative) score.
  Never intentionally throws a winnable game.
- **`SearchUctPb`** (`src/search_uct_pb.*`) — genuine MCTS/UCT with Cao &
  Lin's Progressive Bias UCB term (`Q + k1*sqrt(ln(N)/n) + k2*(H/maxH)`,
  `k1=sqrt(2)` per the paper, `k2=1.0` untuned guess). The paper's ADP
  component (a small neural network trained via self-play to replace
  rollouts) is **not implemented** — training infrastructure is out of
  scope for a single search-strategy file — and is substituted with this
  project's existing static eval, squashed to `[-1,1]` via `tanh(eval /
  4000)`. `Athénan` (Cazenave et al., JMLR) was researched but deliberately
  **not implemented at all**: it's a from-scratch self-play RL system with
  its own search backbone, not portable as a comparable file the way the
  other four variants are.

Both reuse the existing `vcf()`/`vcfWin()` forced-win precheck at the root,
same as every other variant, so no engine has an unfair tactical blind spot
relative to the others.

### Node efficiency @ 20,000 nodes/side

| Position | Search | SearchV2 | SearchV3 | SearchMinibal | SearchUctPb |
|---|---|---|---|---|---|
| ring-open + center | depth 5, 10600 | depth 5, 10600 | depth 9, 7550 | depth 3, 200 | 4196 visits, value≈-0.28 |
| corner open | depth 5, 2800 | depth 5, 5450 | depth 8, 950 | depth 4, -1650 | 4244 visits, value≈0.09 |
| edge open | depth 5, 22500 | depth 5, 24000 | depth 8, 4900 | depth 3, 1050 | 4564 visits, value≈-0.53 |

`SearchUctPb`'s columns aren't apples-to-apples with the alpha-beta
variants' `depth`/`score` (MCTS has no iterative-deepening depth guarantee,
and its "value" is a squashed win-rate estimate, not a minimax score) —
included for its own trend (visit count, rough value sign) only.

**`SearchMinibal` reaches noticeably shallower depth than every alpha-beta
variant at the same node budget** — a direct, expected cost of its design:
the root pass can't prune across sibling candidates (it needs every
candidate's true value for the balance comparison), so it spends far more
of the budget at the root itself, leaving less for depth.

### Round-robin @ 8,000 nodes/move (4 games per pairing: 2 openings × 2 colors)

```
Search 4 - 0 SearchV2
Search 2 - 2 SearchV3          (contrast: SearchV3 won 6-2 at 5,000 nodes/move in Round 2 -- budget-sensitive, as already flagged)
Search 3 - 1 SearchMinibal
Search 4 - 0 SearchUctPb
SearchV3 4 - 0 SearchMinibal
SearchV3 4 - 0 SearchUctPb
```

Rough ranking at this budget: `SearchV3` ≈ `Search` > `SearchMinibal` >
`SearchV2` ≈ `SearchUctPb`.

- `SearchUctPb` was swept both times it played. Unsurprising: MCTS
  typically needs far more simulations than 8,000 to compete with
  alpha-beta at this game's branching factor, and the ADP substitution
  (existing heuristic instead of a trained network) gives it a cruder
  prior than the real algorithm would have.
- `SearchMinibal` lost more than it won (1-3, 0-4), consistent with the
  node-efficiency finding above — the fairness guarantee has a measurable
  strength cost here, which is itself a legitimate (if unsurprising)
  result, not a bug.
- **`SearchV3` vs. baseline flipped from a clear win (Round 2, 5,000
  nodes/move) to a tie (this round, 8,000 nodes/move).** A second, sharper
  instance of the same budget-sensitivity already noted for `SearchV2` in
  Round 2 — reinforces that no ranking here should be treated as
  budget-independent without further testing.

### Caveats specific to this round

- **4 games per pairing is even smaller than Round 2's 8** — the rankings
  above are directional hints, not statistically meaningful results.
- `SearchUctPb`'s constants (`K1`, `K2`, the `tanh(eval/4000)` squash, the
  24-move expansion cap) are all first guesses, same status as `SearchV3`'s
  margins. Its loss here says at least as much about these untuned
  constants and the missing ADP network as about MCTS/Progressive-Bias as
  a technique.
- `SearchMinibal`'s full-window root pass makes its per-move cost scale
  with candidate count in a way the other variants don't; a like-for-like
  wall-clock (not node-budget) comparison could look different.

## Round 4: 100 games/pairing, parallelized

`tools/compare_search.cpp` was parallelized (a thread pool, one game per
task, fresh engine instances per game -- `Search`'s own header already
documents this as safe, no shared mutable state) and openings expanded from
4 hardcoded squares to 50 deterministically-spread ring-legal squares (of
104 total), giving 100 games/pairing instead of 4-8. 3,000 nodes/move
(down from Round 3's 8,000, to keep total wall time reasonable at 25x the
game count). All 600 games completed.

```
Search   62 - 36  SearchV2       (draws: 2)
Search   38 - 59  SearchV3       (draws: 3)
Search   87 - 13  SearchMinibal  (draws: 0)
Search   94 -  6  SearchUctPb    (draws: 0)
SearchV3 85 - 12  SearchMinibal  (draws: 3)
SearchV3 97 -  3  SearchUctPb    (draws: 0)
```

**Ranking at n=100: `SearchV3` > `Search` > `SearchV2` > `SearchMinibal` >
`SearchUctPb`.** This is now a large enough sample to treat as a real
signal, not noise:

- **`SearchV3` beats baseline decisively (59-38)** — the futility/move-count
  pruning + razoring-VCF additions hold up at scale, consistent with Round
  2's node-efficiency finding (deeper search per node) and resolving the
  budget-sensitive tie seen in Round 3.
- **`SearchV2` is genuinely weaker than baseline (36-62)**, not a small-sample
  artifact — every round from Round 2 onward showed this same direction.
  The LMR refinement alone, without the pruning additions, does not pay for
  its own overhead in this codebase.
- **`SearchMinibal` and `SearchUctPb` both lose heavily**, as their designs
  would predict: `SearchMinibal` trades search depth for a fairness
  guarantee (confirmed again: shallower depth in Round 3's efficiency
  table), and `SearchUctPb` is missing the paper's actual strength source
  (a trained ADP network, substituted here with the existing heuristic).
  Neither result should be read as "the paper's technique doesn't work" --
  only that the fairness goal and the untrained MCTS variant built here
  are, as expected, weaker against a pure-strength opponent.

## Round 5: three new paper-derived strategies, eight-engine round-robin

A fresh research pass (written up in `docs/related-work.md` §5-6) turned up
three techniques with a concrete claim on this engine, each built as a
separate class on the same pattern as the earlier variants, and each derived
from `SearchV3` (the Round 4 winner) so that the *only* difference from the
current best is the one new heuristic:

- **`SearchVct`** (`src/search_vct.*`) — Allis, van den Herik & Huntjens
  (1993), the threat-space search that solved Go-Moku. This project already
  had the fours-only half of it (`vcf()`, Victory by Continuous Fours, where
  the defender's reply is forced to exactly one square). `SearchVct` adds
  **three-threats**: a move creating an open three threatens an *open four*
  next, the defender has several defences rather than one, and each must be
  refuted. Chains mixing threes and fours (VCT) are where most real Go-Moku
  wins live — a three-four-five combination is structurally invisible to a
  fours-only search. Also implements the paper's counter-threat refutation
  (the defender may answer with a four of their own instead of defending).
  The paper's dependency-based search and "all defences at once"
  simplification are deliberately **not** implemented: they exist to prove
  wins over very long chains offline, and here the chain is bounded to a few
  threats inside a node budget.
- **`SearchNullKiller`** (`src/search_nullkiller.*`) — **null-move forward
  pruning**, plus the killer heuristic (Akl & Newborn 1977) and the
  counter-move heuristic (Uiterwijk 1992), none of which existed here. The
  standard objection to null move is zugzwang; **Go-Moku has none** —
  placing a stone can never worsen your own position, so passing is a strict
  lower bound on moving and the null-move assumption is sounder here than in
  chess, where it was invented.
- **`SearchThreatCand`** (`src/search_threatcand.*`) — **threat-driven
  candidate generation**. Every other engine here generates the same ~40
  square neighbourhood at every node and relies on *ordering* plus a
  candidate cap to control branching; ordering is a soft tool, since a
  hopeless move still costs a node before it is refuted. Two levels, kept
  separate because only one is exact: **level 1** — if the opponent has any
  square completing five, the candidate set provably *is*
  `our five-points ∪ their five-points` (any other move loses on the spot),
  so this discards nothing playable; **level 2** — the open-three
  restriction, which is heuristic and therefore gated to deeper nodes and
  never applied at the root.

The harness (`tools/compare_search.cpp`) was reworked for this round: engines
are now registered behind a type-erased handle instead of hardcoded template
pairings, the tournament is a **full round-robin with a standings table and
cross-table**, every game of every pairing goes into **one global thread
pool** (a per-pairing pool left most cores idle at small game counts — the
first attempt ran at 144% CPU on 8 cores), and a **wall-clock budget mode**
was added alongside the node budget.

### Correctness first: does VCT actually add anything?

Strength numbers are meaningless if a new heuristic is simply inert or, worse,
claims wins it does not have. `tests/search_new_strategies_test.cpp` (new,
4 cases) pins this down:

- On a **three-three fork** (Black at (7,7) simultaneously completes an open
  three on row 7 and one on column 7), baseline `Search::vcfWin` returns −1 —
  VCF is *provably* blind there, since no four exists to start a chain — and
  `SearchVct::vctWin` returns (7,7). This is the exact class of win the
  technique is for, and it confirms VCT is adding reach rather than
  re-deriving VCF.
- On a quiet two-stone position `vctWin` returns −1, i.e. no false positive.
- All three new engines still take an available five and block the
  opponent's five (the tactical floor).

A real bug was caught by writing these: the first `vct()` truncated the
candidate list to the top 12 attacking moves and then reused that *truncated*
list as the scan domain for five-completion points, silently missing winning
points that did not rank in the top 12. Attack breadth and scan domain are
now two separate lists. (Same lesson as Round 3's ordering bug: the cheap
mistakes in threat search are all about *which set* you are scanning.)

### Round-robin @ 20,000 nodes/move (10 games/pairing, 8 engines, 280 games)

```
rank engine               points    games   score%
1    SearchV3               49.5       70    70.7%
2    SearchVct              49.0       70    70.0%
3    SearchThreatCand       48.5       70    69.3%
4    SearchNullKiller       48.0       70    68.6%
5    SearchV2               38.0       70    54.3%
6    Search                 36.0       70    51.4%
7    SearchMinibal           6.0       70     8.6%
8    SearchUctPb             5.0       70     7.1%
```

Cross-table (row engine's W-L vs column engine):

```
                      Search  SearchV2  SearchV3 SearchMin SearchUct SearchVct SearchNul SearchThr
Search                     -       4-6       4-6      10-0      10-0       3-7       3-7       2-8
SearchV2                 6-4         -       1-8      10-0      10-0       3-7       4-6       3-6
SearchV3                 6-4       8-1         -      10-0      10-0       5-5       5-5       5-5
SearchMinibal           0-10      0-10      0-10         -       5-5      0-10       1-9      0-10
SearchUctPb             0-10      0-10      0-10       5-5         -      0-10      0-10      0-10
SearchVct                7-3       7-3       5-5      10-0      10-0         -       5-5       5-5
SearchNullKiller         7-3       6-4       5-5       9-1      10-0       5-5         -       6-4
SearchThreatCand         8-2       6-3       5-5      10-0      10-0       5-5       4-6         -
```

### What this does and does not show

1. **All three new strategies clearly beat both pre-existing baselines.**
   Against `Search` they went 7-3, 7-3, 8-2; against `SearchV2`, 7-3, 6-4,
   6-3. Three independent heuristics all moving the same direction against
   two different opponents is the one solid signal in this table.

2. **The top four are not separated.** 49.5 / 49.0 / 48.5 / 48.0 is a spread
   of 1.5 points across 70 games, and **every one of the three new engines
   drew its head-to-head with `SearchV3` 5-5**. The rank order among the top
   four is therefore decided entirely by results against `SearchMinibal` and
   `SearchUctPb` — engines all four beat 10-0 or 9-1 — which is to say it is
   decided by noise. `SearchNullKiller` finishing 4th rather than 3rd is a
   **half-point gap, i.e. one drawn game**, and must not be read as
   "NullKiller is worse than ThreatCand."

3. **`SearchV2` beat baseline `Search` 6-4 here, reversing Round 4**, where
   `Search` won 62-36 at 3,000 nodes/move over 100 games. Round 4 is the
   more trustworthy of the two (10x the games), so this is most likely small-
   sample noise rather than a genuine budget effect — but it is the third
   time `SearchV2`-vs-baseline has changed direction across rounds, and the
   open question flagged in Round 2 remains open.

4. **`SearchMinibal` and `SearchUctPb` remain far behind**, consistent with
   Rounds 3-4 and with their designs (a fairness goal and an untrained MCTS
   respectively). Their 5-5 against each other is the only game they got.

### The methodological problem with the table above

A **node budget does not charge an engine for work done inside a node.** All
three new strategies add per-node board scanning that a node-matched
comparison is blind to:

- `SearchThreatCand` runs its level-1 five-point scan at essentially every
  node, and a quadratic open-four scan at deeper ones;
- `SearchNullKiller` runs two `hasImmediateWin()` full-board scans per
  null-move attempt;
- `SearchVct`'s threat search does repeated O(candidates²) scans.

None of that increments `nodes_`, so the table above flatters all three
relative to `SearchV3` — possibly enough to explain the entire (tiny) gap.
This is the same caveat raised in Rounds 2 and 4 and listed there as an
unaddressed next step; it is addressed below rather than deferred again.

### Wall-clock tiebreaker @ 200 ms/move (10 games/pairing, top 4 only)

Same engines, same openings, same game count — the only change is that the
per-move budget is **wall-clock time**, which charges every engine for all
the work it does, in-node scans included. Run with 4 workers on 8 cores
rather than 8, to reduce the CPU contention that makes time-based budgets
noisy.

```
rank engine               points    games   score%
1    SearchV3               19.5       30    65.0%
2    SearchVct              14.5       30    48.3%
3    SearchNullKiller       13.0       30    43.3%
4    SearchThreatCand       13.0       30    43.3%
```

```
                    SearchV3 SearchVct SearchNul SearchThr
SearchV3                   -       6-3       8-2       5-5
SearchVct                3-6         -       5-5       6-4
SearchNullKiller         2-8       5-5         -       6-4
SearchThreatCand         5-5       4-6       4-6         -
```

**This reverses the node-matched picture, and it is the table to believe.**
The near-tie at the top of the node-matched round-robin was substantially an
artifact of not charging for per-node work: given an equal amount of *time*,
`SearchV3` beats `SearchVct` 6-3 and `SearchNullKiller` 8-2, and its 65.0%
is a real gap rather than a half-point one. A node budget is the right tool
for reproducibility; it is the wrong tool for ranking engines whose per-node
costs differ, which is exactly the situation here. Rounds 2 and 4 both
predicted this specific failure mode — it is now measured, not hypothesised.

Points of note:

- **`SearchThreatCand` is the one that holds up.** It drew `SearchV3` 5-5
  under *both* budgets, the only new engine to do so. Its level-1
  restriction is exact (it provably discards no playable move), so its cost
  is pure overhead with no accuracy risk; the fact that it breaks even
  against `SearchV3` on time despite that overhead suggests the branching
  saving is genuinely paying for the scan. Making the scan cheaper — it
  currently re-derives five-points from scratch per node rather than
  maintaining them incrementally the way `EvalState` does — is the obvious
  follow-up, and would move it strictly ahead if the saving holds.
- **`SearchNullKiller` is the clear loser on time (2-8 vs `SearchV3`)**,
  the worst single result in this table, despite being within half a point
  of the lead on nodes. Two full-board `hasImmediateWin()` scans per
  null-move attempt is a heavy gate, and the R = 2 + depth/4 reduction and
  the depth ≥ 3 trigger are untuned first guesses. The *idea* is not
  refuted by this — the zugzwang-freedom argument for null move in Go-Moku
  still stands, and the technique is used by at least one competition-winning
  engine — but this implementation of it is not currently an improvement.
- **`SearchVct` is the best of the three new strategies overall**, ranking
  2nd under both budgets, but it does not beat `SearchV3`. Its threat search
  demonstrably finds wins VCF cannot (see the test above), yet those
  positions are apparently rare enough in real games at this budget that the
  cost of looking outweighs the payoff. Its VCT phase is capped at a third
  of the per-move budget, spent every move whether or not a threat chain
  exists; spending it only when a cheap precondition says a three-threat is
  present is the obvious fix.

### Verdict against the "keep it if it is top 3" bar

> **Superseded by Round 6.** The wall-clock table this verdict rests on was
> run with 4 parallel workers, which Round 6 shows is not a valid way to
> measure a time budget. See Round 6 for the corrected single-threaded
> result and a revised verdict. The node-matched table above is unaffected.

- **`SearchVct` — keep.** Top 3 on both budgets (2nd, 2nd).
- **`SearchThreatCand` — keep.** Top 3 node-matched (3rd), and the only new
  engine to hold `SearchV3` to a draw under both budgets.
- **`SearchNullKiller` — does not clear the bar.** 4th node-matched (by half
  a point, i.e. one drawn game) and joint-3rd on time, but with the worst
  head-to-head in the field against the leader. It is *not* promoted.

The `search_nullkiller.*` files are nonetheless retained rather than deleted,
following this document's existing convention: `SearchMinibal` and
`SearchUctPb` both lose heavily and are kept precisely because a measured
negative result is worth more than a gap in the record. Deleting the source
would also silently drop it from the round-robin field.

**None of the three new strategies displaces `SearchV3` as the engine to
promote.** The genuine, unambiguous finding of this round is narrower and
still useful: all three beat the *previous* baselines `Search` and `SearchV2`
decisively (7-3, 7-3, 8-2 and 7-3, 6-4, 6-3), so `SearchV3`'s Round 4
promotion recommendation is reinforced rather than overturned.

## Round 6: `SearchV4`, and a methodology error that invalidates part of Round 5

### The premise

Round 5's conclusion was that three paper-derived heuristics all lost on the
clock despite being level on nodes. The diagnosis was that they shared a
dependency: each one repeatedly asks *"where can this colour complete
five?"*, and this project answered that by looping over all ~272 squares
doing `place` / `winsAt` / `remove`. A node budget does not charge for that
loop; a clock does.

If that diagnosis is right, Round 5 measured the **primitive**, not the
ideas, and fixing the primitive should let the ideas back in.

`SearchV4` (`src/search_v4.*`) tests exactly that:

1. **Bitboard-parallel five-point detection.** For each of the 4 directions
   and each of the 5 positions a gap can occupy in a five-window, require the
   other four cells to be own stones — shift/AND over the 272-bit grid,
   whole board at once. **Measured 9.49x faster** (2.942 µs → 0.310 µs per
   call, `clang++ -O2`, arm64).
2. Because `hasImmediateWin()` and `collectWinningPoints()` keep their exact
   signatures and semantics, this **speeds up inherited SearchV3 code first**
   — both sit on the hot path of `vcf()`, razoring, and futility pruning.
3. Then re-adopt what the cheaper primitive can now afford: the **exact**
   forced-reply restriction (Round 5's `SearchThreatCand` level 1 — level 2,
   the heuristic half, is deliberately *not* carried over), null-move pruning
   with a mask-based gate instead of two full-board scans, killers and
   counter-moves, and root VCT **gated** on a cheap precondition rather than
   run every move.

### Correctness of the primitive

Everything above rests on the fast path returning *exactly* the same set as
the loop it replaces, so `tests/search_v4_test.cpp` verifies it against the
brute-force original as an oracle over **400 randomised clustered positions
× 2 colours**, plus hand-built edge cases (four on the top edge, a four
ending at column 14 that must not wrap into the next row, a four down the
left column, a blocked four). The random test also asserts that at least 20
of the positions actually contained a threat, so it cannot pass vacuously by
comparing empty sets.

### A methodology error, and the correction

The first two Round 6 tournaments were run with **4 concurrent workers and a
wall-clock budget**, and they disagreed with each other. A 5-engine field put
`SearchV4` first at 78.8%; the 9-engine field put baseline `Search` first at
72.5%, with `Search` beating `SearchV3` **8-1** — reversing every
node-matched result ever recorded here — and `SearchMinibal` leaping from
8.6% (Round 4) to 51.2%.

There was a tempting story for this (cheap engines push more nodes per
second, so they win on a clock). **It was wrong.** Re-running single-threaded,
with contention eliminated:

```
@ 200 ms/move, 1 worker, 10 games/pairing
1  SearchV4   81.7%     vs Search 7-2   vs SearchV2 7-3   vs SearchV3 10-0
2  SearchV2   50.0%
3  SearchV3   36.7%
4  Search     31.7%
```

`Search` vs `SearchV3` flips from 8-1 to **2-7** purely by removing
parallelism. The 4-worker tables were measuring the scheduler.

**This project had already written down the rule and this round broke it.**
`PLAN.md` Phase 7 specifies a "*Node-count benchmark mode: deterministic
per-move node limit (not wall-clock) so cross-engine scores remain comparable
under parallel load; fixed host/socket.*" Wall-clock budgets and a parallel
game runner are incompatible, and that was known before Round 5 started.

Consequences, stated plainly:

- **Round 5's wall-clock tiebreaker used 4 workers and is therefore not
  trustworthy.** Its ranking of `SearchVct` / `SearchNullKiller` /
  `SearchThreatCand` against `SearchV3`, and the verdicts drawn from it,
  are superseded by the single-threaded full-field run below.
- Round 5's *node-matched* 280-game table is unaffected — node budgets are
  deterministic and immune to load, which is the whole reason the project
  specified them.
- **Rule going forward: wall-clock runs are single-worker; parallel runs are
  node-budgeted.** `tools/compare_search.cpp` takes a worker count as its
  6th argument for exactly this reason.

### Definitive result: full field, 200 ms/move, single worker, 360 games

This run supersedes Round 5's wall-clock tiebreaker entirely.

```
rank engine               points    games   score%
1    SearchV4               64.0       80    80.0%
2    Search                 49.0       80    61.2%
3    SearchV2               48.0       80    60.0%
4    SearchThreatCand       45.5       80    56.9%
5    SearchV3               43.5       80    54.4%
6    SearchNullKiller       42.0       80    52.5%
7    SearchVct              40.0       80    50.0%
8    SearchMinibal          28.0       80    35.0%
9    SearchUctPb             0.0       80     0.0%
```

`SearchV4` head-to-head against every other engine:

```
vs SearchV2  10-0     vs SearchUctPb    10-0     vs SearchThreatCand  7-2
vs SearchVct  9-1     vs SearchNullKiller 6-2    vs Search            6-4
vs SearchMinibal 9-1                             vs SearchV3          5-4
```

**`SearchV4` wins all eight pairings** and is the first engine in this
document to beat every other engine in the field on a wall-clock budget.

Read with the appropriate care:

- **The aggregate is the signal, not the individual margins.** 80.0% over 80
  games is a wide gap, and sweeping 8/8 pairings is hard to get by luck. But
  the two closest pairings — `Search` 6-4 and `SearchV3` 5-4 — are *within
  noise at 10 games each*. "V4 beats V3" is supported by the sweep and by the
  mechanism, not by that 5-4.
- **The middle of the table is not trustworthy and is visibly
  non-transitive**: `Search` beats `SearchVct` 7-1, `SearchVct` draws
  `SearchV3` 5-5, and `SearchV3` beats `Search` 5-3. `SearchV2` beats
  `SearchV3` 8-1 here while losing to it 1-8 in Round 4's node-matched
  100-game run. At 10 games/pairing, ranks 2-7 should be treated as a
  scrambled tie, not an ordering.
- **`SearchUctPb` scores 0.0% across 80 games.** A clean 0-for-80 is worth
  flagging as a likely *bug* rather than mere weakness: it probably does not
  honour `maxTimeMs` the way the alpha-beta engines do (it was only ever
  exercised under node budgets). Not investigated here — but no conclusion
  about MCTS should be drawn from this row.

### Revised verdict on the Round 5 strategies

With the corrected methodology, **none of `SearchVct`, `SearchNullKiller` or
`SearchThreatCand` reaches the top 3 of the full field** — they land 4th,
6th and 7th, packed inside the untrustworthy middle band. Round 5's
"`SearchVct` is the best of the three" is *not* reproduced; single-threaded
it is the worst of the three. That earlier ordering came from the 4-worker
table and should be discarded.

The ideas nonetheless survive, in the only form that measured well: **bundled
into `SearchV4` behind a primitive fast enough to afford them.** The
individually-packaged versions are mid-table; the bundle is 1st by 19 points.

### What actually made the difference

`SearchV4` differs from `SearchV3` in exactly two ways that matter:

1. the threat primitive is 9.49x cheaper, which speeds up code V3 already
   had (VCF, razoring, futility) before adding anything; and
2. the three Round 5 ideas are re-added in their cheap/exact forms only —
   exact level-1 restriction (not the heuristic level 2), mask-gated null
   move (not double full-board scans), precondition-gated VCT (not every
   move).

That is consistent with, but does not by itself prove, the Round 5 diagnosis
that those heuristics failed on cost rather than on merit. A per-component
ablation of V4 would be needed to attribute the gain, and has not been run.

## Round 6b: `SearchV5` -- the defensive gap, and two bugs in closing it

`SearchV5` (`src/search_v5.*`) targets a capability no engine here had. Every
engine runs `vcf()` at the root to find *its own* forced win; none ever asks
the mirror question -- **does my candidate move hand the opponent one?** A VCF
chain searches 12 forcing plies while the main search reaches depth 8-9 at a
realistic time control, so there is a band of forced losses that is
structurally invisible: the engine plays a natural-looking move, the opponent
starts a chain of fours, and the mate is only visible once it is unavoidable.

**Root safety screening** closes it: each top root candidate is played, the
opponent is given a VCF search from the resulting position, and any move that
hands over a forced win is skipped at the root -- unless *every* candidate is
unsafe, in which case the position is lost anyway and the screen stands aside
rather than leaving the engine with no move. This is affordable only because
of `SearchV4`'s bitboard primitive, which made `collectWinningPoints()` and
`hasImmediateWin()` (called at every VCF node) 9.5x cheaper.

Four supporting changes: **node-level mask caching** (V4 recomputed the two
five-point masks up to six times per node though the board never changes
between them), a **forced-reply extension** (a node with one legal non-losing
move is forced; terminates because every extended ply places a stone), a
**four-threat extension**, and **IID + aspiration windows**.

### Two bugs, one of which was the dominant cost

- **Unbounded pre-search phases.** `screenRootMoves` and the VCT probe were
  bounded by a *node* allowance, meaningless under a wall-clock budget where
  `maxNodes` is 0. They ran against the whole move deadline, then cleared
  `aborted_`, leaving the main search with an expired clock. Fixed in both V4
  and V5 by bounding the *deadline* instead (`beginPhaseBudget`).
- **The four-threat extension trigger** (dominant). It used
  `evaluateMove()`'s score as a proxy for "this move creates a four", but that
  score combines threats a move *creates* with threats it *pre-empts*, so
  ordinary blocking moves cleared the threshold. The extension fired almost
  everywhere and inflated the tree ~4x per depth level -- depth 5/6/4 where
  V4 reached 8/9/7, at near-identical node counts. Disabled, with the
  measurement recorded in `search_v5.cpp`; re-enabling needs an *exact*
  trigger (test `fivePointsMask()` after placing the move), not a score proxy.

Fixing both took V5 from 3.0% to 13.0% against rapfi. Fixing V4 changed
nothing (14.0% -> 12.5%), which eliminated "it was just a bug" as the
explanation for V4's deficit -- see Round 7.

**A methodological note worth repeating:** the investigation began chasing a
phantom. A run appearing to hang (one game in 7.5 minutes) was line-buffered
progress output -- `oog_compare` printed dots with no newline until game 50,
hiding up to 49 completed games. The run was healthy. Every dot is now
flushed. That is the second time a harness artifact produced a false finding
here (see Round 6's parallel wall-clock tables); both times the tell was a
result that did not fit the mechanism.

## Round 7: the external anchor, and what it did to Rounds 1-6

Every result before this section came from self-play between engines that
share one `eval.cpp`, one move generator, one threat table and one negamax
core. That can rank our engines against each other. It cannot say whether any
of them is good, and it is structurally blind to any error they all share.

Round 7 wires in **rapfi** (`tools/rapfi_setup.sh`, `tools/match.py
--engine-b`) as an external opponent. Outer-Open differs from freestyle Gomoku
*only* in Black's first move, and the match runner plays that move itself from
a fixed ring opening, so any freestyle engine is a valid opponent from move 2
onward.

### Results (100 games each, 200 ms/turn, single worker)

```
engine                     score    Elo vs rapfi   as Black   as White
Search (baseline)          22.0%      -220          8/50       14/50
SearchV4  (pre-fix)        14.0%      -315          6/50        8/50
SearchV4  (post-fix)       12.5%      -338          2/50       10/50
SearchV5  (pre-fix)         3.0%      -604          2/50        1/50
SearchV5  (post-fix)       13.0%      -330          5/50        8/50
```

### The finding

**The untouched baseline `Search` is the strongest engine we have against an
external opponent.** Pooling the three fixed/valid advanced runs gives 13.2%
over 300 games against the baseline's 22.0% over 100 -- a 8.8 point gap,
SE ~4.6, z ~1.93 (p ~= 0.05). Borderline on its own, but the direction is
consistent across three independent runs.

Set against Round 6, where `SearchV4` won **all eight** self-play pairings at
80.0%. Six rounds of search work, validated only against engines sharing its
own blind spots, produced an engine that does not beat the code it started
from once the opponent is genuinely different.

This is not a subtle effect and it was not visible from inside the self-play
loop. It is the single most important measurement in this document.

### Two bugs found on the way, and one non-bug

- **Not a bug:** a run that appeared to hang (one game in 7.5 minutes) was
  line-buffered progress output. `oog_compare` prints dots with no newline
  until game 50, so up to 49 completed games were invisible. Every dot is now
  flushed. Second time in this project a harness artifact sent us chasing a
  phantom (see Round 6's parallel wall-clock tables) -- both times the tell
  was a result that did not fit the mechanism.

- **Real bug 1 -- unbounded pre-search phases.** `screenRootMoves` and the VCT
  probe were bounded by a *node* allowance, which is meaningless under a
  wall-clock budget (`maxNodes` is 0 there). They ran against the whole move
  deadline, then cleared `aborted_`, leaving the main search with an expired
  clock. Fixed in both V4 and V5 by bounding the *deadline* instead.

- **Real bug 2 -- the four-threat extension trigger.** V5 used
  `evaluateMove()`'s score as a proxy for "this move creates a four", but that
  score combines threats a move *creates* with threats it *pre-empts*, so
  ordinary blocking moves cleared the threshold. The extension fired almost
  everywhere and inflated the tree ~4x per depth level: depth 5/6/4 where V4
  reached 8/9/7, at near-identical node counts. Disabled (with the
  measurement recorded in `search_v5.cpp`); re-enabling needs an exact
  trigger, not a score proxy.

Bug 2 was the dominant cause of V5's 3% collapse; fixing both took V5 from
3.0% to 13.0%. Notably, **fixing V4 changed nothing** (14.0% -> 12.5%), which
eliminates "it was just a bug" as the explanation for V4's deficit.

### What this implies

1. **Anchor externally before trusting any self-play delta.** Every future
   change should be measured against rapfi, not against our own engines.
2. **Black play is where we bleed.** Every engine scores far worse as Black
   (baseline 8/50 vs 14/50; V4 2/50 vs 10/50). The forced ring opening is a
   large systematic factor and we have no Outer-Open opening theory. rapfi
   has none either -- this is the one uncontested edge available.
3. **The gap is evaluation-shaped, not search-shaped.** rapfi searches much
   as we do (hand-written alpha-beta, ~20 tuned steps) but evaluates with a
   trained NNUE carrying both a value head and a *policy* head driving move
   ordering (their own comments: policy-based reduction ~59 elo, policy-based
   pruning ~10 elo). A -220 Elo gap is not closed by more search heuristics.
   That is Phase 6, and it is now a measured priority rather than a guess.
4. **Do not assume borrowed Elo transfers.** rapfi rates null-move pruning at
   ~3 elo -- the technique Round 5 built an entire strategy around.

## Round 8: is our *evaluation* the weak part, and does parallelism lie?

Two questions Round 7 left open.

### Is -220 Elo the ceiling for heuristic engines, or is our engine bad?

rapfi ships a **classical evaluator** (`model210901.bin`) alongside its NNUE
weights, so it can be run with the neural network removed entirely. That
isolates the question exactly: our heuristics vs *their* heuristics, no
network on either side. `third_party/rapfi-classical/` is a staged copy with
the `[model.evaluator]` section stripped from `config.toml` and the `.lz4`
weights absent; a smoke test confirms it loads no NNUE.

Under matched (parallel, see below) conditions:

```
search vs NNUE      rapfi:  1.0%
search vs CLASSICAL rapfi:  6.0%
v4     vs CLASSICAL rapfi:  8.0%
```

The NNUE is worth real strength (1.0% -> 6.0% when it is removed). But we
still lose **94 of 100** to a hand-written evaluator. **The ceiling is not the
problem; our evaluation is.** Supporting evidence: OOGiveMeFive -- this
project's own reference engine -- won 6 golds across 8 competitions with
purely classical methods, and Allis solved Go-Moku in 1993 with threat-space
and proof-number search alone.

Our eval is seven hand-picked constants (`S_FIVE=1'000'000` ..
`S_CLOSED_TWO=100`) that have **never been fitted to data** -- `PLAN.md`
Phase 2's "tune weights later" box is still open, and this document already
describes the `SearchV3` margins as "untuned first guesses". Our candidate
generation is hardcoded to distance 2, the narrowest of the six options rapfi
exposes as a *tunable* (`square2` .. `full_board`).

(Serial re-measurement of the classical rows is pending; the numbers above
are valid as a *relative* comparison because all three ran under identical
contention, but their absolute values are not -- see below.)

### Parallelism control: wall-clock + parallel is invalid, measured

Round 6 asserted that wall-clock budgets and parallel game running do not mix.
Round 8 tested it directly by re-running one matchup whose serial answer was
already known:

```
search vs NNUE rapfi -- serial (1 worker):   22.0%
search vs NNUE rapfi -- parallel (4 workers):  1.0%
```

Same binaries, same matchup, same 100-game count: a **21-point distortion**.
Parallel wall-clock measurement is not usable here, now demonstrated rather
than argued. Since rapfi has no node-count mode, external matches are forced
to wall-clock, and therefore forced to a single worker.

**The failure mode is itself diagnostic.** There were no forfeits or timeouts
-- those games were lost on the board. Under contention both engines search
shallower; rapfi barely notices and we collapse. **Our strength rides on
search depth; rapfi's rides on evaluation quality.** A good evaluation still
picks reasonable moves at shallow depth; ours has nothing to fall back on.
That is independent evidence pointing at the same conclusion as the classical
comparison above.

## Suggested next steps

1. Re-run the round-robin with many more games (`tools/match.py`-style
   parallel runner) across a range of node budgets, including at least one
   that maps to a realistic per-move time control.
2. Re-test with a wall-clock time budget instead of node budget, to see
   whether `SearchV3`'s per-node overhead changes the picture.
3. Investigate the `SearchV2`-vs-baseline direction flip specifically,
   rather than folding it into `SearchV3`'s result.
4. If `SearchV3`'s advantage holds up under more games, consider promoting
   it (or the specific techniques that prove out) into the baseline
   `Search`, and begin tuning the margin constants per Phase 7 of `PLAN.md`.
5. If `SearchUctPb` is worth pursuing further, the highest-leverage next
   step is the one piece it's missing: an actual trained ADP-style network
   (even a small MLP over existing threat features, per `docs/related-work.md`'s
   note on Cao & Lin) rather than the eval-substitute used here.
6. ~~Re-run all five with matched wall-clock budgets, not just node budgets.~~
   **Done in Round 5** — and it mattered: the node-matched and time-matched
   rankings disagreed sharply. Treat wall-clock as the primary metric from
   here on, with node budgets used for reproducibility checks only.

### After Round 5

7. **Make `SearchThreatCand`'s scan incremental.** It is the most promising
   of the three new strategies (drew `SearchV3` under both budgets) and its
   entire handicap is the cost of re-deriving five-points per node. This
   project already has the pattern to copy: `EvalState` maintains window
   scores incrementally instead of rescanning. A threat board updated only
   within distance 4 of each move (as the sortingsearching engine does) is
   the same idea.
8. **Gate `SearchVct`'s threat search on a cheap precondition.** It
   currently spends up to a third of the per-move budget on VCT every move,
   whether or not any three-threat exists. Running it only when a cheap
   check says the position is forcing should recover most of the loss.
9. **Tune `SearchNullKiller` before writing it off.** Its R, its depth
   trigger, and its double `hasImmediateWin()` gate are all untuned first
   guesses, and the gate is the expensive part; a cheaper gate is the first
   thing to try. The zugzwang-freedom argument for null move in Go-Moku is
   sound independently of this implementation's result.
10. **Re-run Round 5 with far more than 10 games per pairing.** Every number
    in Round 5 rests on 10-game pairings; the 8-engine node-matched table in
    particular has a top four separated by less than one game.

## Source code

- `src/search.hpp` / `src/search.cpp` — baseline `Search`, unmodified.
- `src/search_v2.hpp` / `src/search_v2.cpp` — `SearchV2`.
- `src/search_v3.hpp` / `src/search_v3.cpp` — `SearchV3`.
- `src/search_minibal.hpp` / `src/search_minibal.cpp` — `SearchMinibal`.
- `src/search_uct_pb.hpp` / `src/search_uct_pb.cpp` — `SearchUctPb`.
- `src/search_vct.hpp` / `src/search_vct.cpp` — `SearchVct` (Round 5).
- `src/search_nullkiller.hpp` / `src/search_nullkiller.cpp` —
  `SearchNullKiller` (Round 5).
- `src/search_threatcand.hpp` / `src/search_threatcand.cpp` —
  `SearchThreatCand` (Round 5).
- `src/search_v4.hpp` / `src/search_v4.cpp` — `SearchV4` (Round 6):
  bitboard-parallel five-point primitive (9.49x) + Round 5 ideas in cheap
  form.
- `src/search_v5.hpp` / `src/search_v5.cpp` — `SearchV5` (Round 6b): root
  safety screening + mask caching + extensions + IID/aspiration.
- `tests/search_v4_test.cpp` — verifies the fast five-point primitive against
  the brute-force loop it replaces, over 400 random positions x 2 colours
  plus edge/wrap cases (with a guard against a vacuous pass).
- `tests/search_v5_test.cpp` — root-screen correctness: flags a move allowing
  an opponent VCF, clears a safe one, and still returns a legal move when
  every candidate loses.
- `tools/rapfi_setup.sh` — clones, builds and stages rapfi as an external
  opponent (arch-detecting: NEON on arm64, AVX2 on x86_64), with a smoke
  test. rapfi is GPL-3 but is run as a separate opponent *process*, which
  imposes nothing on this project's code; its weights are CC0.
- `oog_piskvork --engine search|v4|v5` — engine selection for the protocol
  binary, so any of our engines can be anchored externally.
- `tests/search_new_strategies_test.cpp` — correctness tests for the Round 5
  strategies (VCT finds a fork VCF cannot; no false positives; tactical
  floor).
- `tools/compare_search.cpp` — the comparison harness (`oog_compare`).
  Usage: `oog_compare [effNodes] [matchNodes] [openings] [msPerMove] [engineList] [threads]`.
  A non-zero `msPerMove` switches the tournament to a wall-clock budget;
  `engineList` is a comma-separated subset of engine names. The Round 5 runs
  were `oog_compare 20000 20000 5` and
  `oog_compare 1000 0 5 200 "SearchV3,SearchVct,SearchNullKiller,SearchThreatCand" 4`.
- Research sources: [dhbloo/rapfi](https://github.com/dhbloo/rapfi)
  (`Rapfi/search/ab/search.cpp`, `Rapfi/search/ab/parameter.h`, read
  directly from a clone of the repository); the papers in
  `docs/related-work.md` — §2/§4 for `SearchMinibal` and `SearchUctPb`, §5
  (Allis et al. 1993) for `SearchVct`, §6 for `SearchNullKiller` and
  `SearchThreatCand`.
