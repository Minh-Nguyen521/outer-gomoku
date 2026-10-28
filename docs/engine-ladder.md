# External engine ladder

Measured strength of the third-party engines in `third_party/`, and what each
number is and is not evidence for.

**Read the caveats before quoting any figure.** Roughly half the effort behind
this table went into discovering that an engine's first result was an artefact
of the harness rather than its strength. Six separate engines produced a
convincing 0-20 that turned out to be a bug in the setup.

---

## Results

All matches: freestyle rules (`--rule 0`), 15x15, colours alternating each
game, adjudicated by `tools/match.py`. "clean" counts games decided by an
actual five-in-a-row, as opposed to a forfeit, timeout or protocol failure.

> **Record the settings with every number.** The rows below were measured
> without noting the per-move budget, parallelism, or whether paired openings
> were used, and that turned out to matter — see
> [Reproducing a row](#reproducing-a-row-2026-09-08) below. Since three of the
> false results in this file trace directly to time controls, a score without
> its `--timeout-turn` is not reproducible. New rows should carry them.

> **This table is superseded.** Every row is measured against rapfi, the one
> engine we later found behaves anomalously here (1st on Gomocup, 5th in our
> fit), and the settings were never recorded. It is kept only as a record of
> what was believed before. Use the
> [unified ranking](#unified-ranking-514-games-12-engines-2026-09-17) instead.
> In particular it calls SlowRenju the strongest classical engine; SlowRenju
> is 7th of 12, losing 0-20 to KataGomo and 1-19 to VIBEFIVE.

| Engine | vs rapfi-nnue | clean | Notes |
|---|---|---|---|
| KataGomo (400 visits) | **17-4** (81%) | 20/20 | GPU-accelerated; see caveat 2 |
| SlowRenju 5.1.3 | **18-2** (90%) | 20/20 | *superseded* — see unified ranking |
| AlphaGomoku 5.9.3 | **18-2** (90%) | — | |
| VIBEFIVE 2026 | **17-3** (85%) | 20/20 | 264KB alpha-beta, no neural net |
| SKYZERO 2026 | **12-8** (60%) | 20/20 | Gumbel AlphaZero + trained weights |
| rapfi-nnue | — | — | ladder anchor; see caveat 1 |
| PentaZen 0.4.17 | 0-20 | 19/20 | |
| Rapfi-gomocup 2018 | 0-20 | 19/20 | Gomocup 2018, 4th place |

Head-to-head between the stronger engines:

### Unified ranking, 514 games, 12 engines (2026-09-17)

Every pairwise result in this file, fitted jointly with Bradley-Terry instead
of being read as a chain of separate matches. A single fit uses the whole graph
at once, so an engine's rating is informed by its opponents' opponents -- which
matters here because the match graph is sparse and several engines have only
one or two links to the rest of it.

Elo is relative, centred on the field mean. Ratings carry a light L2 prior, so
saturated pairs (20-0) give a large finite number rather than infinity.

| # | Engine | Elo | Games | Opp | W | L | D | Score |
|---|---|---|---|---|---|---|---|---|
| 1 | **KataGomo** | +416 | 118 | 6 | 102 | 14 | 2 | 87% |
| 2 | VIBEFIVE | +273 | 157 | 8 | 119 | 38 | 0 | 76% |
| 3 | AlphaGomoku | +251 | 99 | 5 | 61 | 35 | 3 | 63% |
| 4 | SKYZERO | +86 | 98 | 5 | 37 | 60 | 1 | 38% |
| 5 | rapfi-8t | +68 | 119 | 6 | 53 | 65 | 1 | 45% |
| 6 | Yixin2018 | −25 | 120 | 6 | 32 | 87 | 1 | 27% |
| 7 | SlowRenju | −43 | 160 | 7 | 84 | 75 | 1 | 53% |
| 8 | rapfi-1t | −110 | 20 | 1 | 7 | 13 | 0 | 35% |
| 9 | Barbakan | −142 | 17 | 1 | 0 | 17 | 0 | 0% |
| 10 | rapfi-4t | −183 | 40 | 1 | 10 | 29 | 1 | 26% |
| 11 | Chis | −193 | 20 | 1 | 4 | 16 | 0 | 20% |
| 12 | PentaZen | −398 | 60 | 3 | 0 | 60 | 0 | 0% |

Full head-to-head grid (row's W-L, draws in parens, `.` = never played):

| | Kata | VIBE | AlphaG | SKY | rapfi8t | Yixin | SlowR | rapfi1t | Barb | rapfi4t | Chis | PentaZ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **KataGomo** | – | 17-3 | 15-3(2) | 15-3 | 15-5 | 20-0 | 20-0 | . | . | . | . | . |
| **VIBEFIVE** | 3-17 | – | 6-14 | 15-5 | 19-1 | 20-0 | 19-1 | . | 17-0 | . | . | 20-0 |
| **AlphaGomoku** | 3-15(2) | 14-6 | – | 16-4 | 14-4(1) | 14-6 | . | . | . | . | . | . |
| **SKYZERO** | 3-15 | 5-15 | 4-16 | – | 10-10 | 15-4(1) | . | . | . | . | . | . |
| **rapfi-8t** | 5-15 | 1-19 | 4-14(1) | 10-10 | – | 13-7 | . | . | . | . | . | 20-0 |
| **Yixin2018** | 0-20 | 0-20 | 6-14 | 4-15(1) | 7-13 | – | 15-5 | . | . | . | . | . |
| **SlowRenju** | 0-20 | 1-19 | . | . | . | 5-15 | – | 13-7 | . | 29-10(1) | 16-4 | 20-0 |
| **rapfi-1t** | . | . | . | . | . | . | 7-13 | – | . | . | . | . |
| **Barbakan** | . | 0-17 | . | . | . | . | . | . | – | . | . | . |
| **rapfi-4t** | . | . | . | . | . | . | 10-29(1) | . | . | – | . | . |
| **Chis** | . | . | . | . | . | . | 4-16 | . | . | . | – | . |
| **PentaZen** | . | 0-20 | . | . | 0-20 | . | 0-20 | . | . | . | . | – |

Games decided by `no reply (timeout or crash)` are **excluded** from this fit.
Six such games exist across three matches (`barb_vs_vibe` 3, `sky_vs_kata` 2,
`ag_vs_rapfi8t` 1); Barbakan's headline `0-20` is really 0-17. They are
startup and crash failures, not play, and scoring them as losses biases every
rating that touches them.

The **Opponents** column is the number of distinct engines a row has actually
played. Rows with 1 are held in place by a single link and should be read as
provisional regardless of how many games back them.

**KataGomo is undefeated as a matchup**, not merely first: it wins every
pairing it has played -- 20-0 over both SlowRenju and Yixin, 16-4 over
AlphaGomoku, 17-3 over VIBEFIVE, 15-5 over rapfi-8t.

**Rows 2 and 3 are a genuine tension, not a ranking.** Their direct
head-to-head says AlphaGomoku (14-6 over VIBEFIVE), but the joint fit puts
VIBEFIVE 25 Elo higher, because VIBEFIVE sweeps Yixin 20-0 where AlphaGomoku
manages only 14-6. Twenty-five Elo over 80-160 games is inside the noise
either way. Treat them as tied and do not cite an ordering between them.

**The rapfi thread count separates cleanly** -- 8t (+68) > 1t (−110, n=20) >
4t (−183). The 8-thread build is the one to use for any rapfi comparison; the
1t/4t ordering is inside the noise and rests on one link each.

**rapfi-8t is now the best-measured engine in the table** -- 119 games across
6 distinct opponents, more than anything except SlowRenju and VIBEFIVE. Its
placement moved a long way to get here. It sat at +139 on a 4-game link to
VIBEFIVE (3-1); finishing that match returned **VIBEFIVE 19-1**, dropping it
to +55, and three further matches against AlphaGomoku, SKYZERO and Yixin
settled it at **+68**. An earlier, separately-run 18-game sample of the
VIBEFIVE pairing gave 14-4, so three independent samples agree on direction.

This makes the Gomocup disagreement *starker*, not weaker: rapfi is rated 1st
there and is **5th of 12** here, below SKYZERO and 93 Elo above Yixin -- an
engine Gomocup rates it ~250 points above. It is the single largest
unexplained gap in this file, and caveat 1 lists seven hypotheses that were
measured and eliminated.

**The emulation explanation does not survive the SKYZERO result.** The leading
hypothesis for rapfi's low placement has been the non-uniform emulation
penalty: Wine/Rosetta engines were assumed to be handicapped, inflating
rapfi's opponents by comparison. But rapfi-8t is **native arm64** and SKYZERO
is a Windows binary under Wine -> Docker -> Rosetta, and they finished exactly
level at **10-10**. If emulation were meaningfully handicapping SKYZERO, the
native engine should have won comfortably. It did not. Whatever depresses
rapfi here, it is not simply that its opponents are being slowed down.

**The fit was tested against three blind predictions and passed all three
(2026-09-17).** Each pairing had never been played, so the predicted score was
pure extrapolation through third parties:

| Match | Gap | Predicted | Actual | Error |
|---|---|---|---|---|
| AlphaGomoku vs rapfi-8t | 190 Elo | 75% | **14-4-1, 76.3%** | +1.3 |
| rapfi-8t vs SKYZERO | 38 Elo | 45% | **10-10, 50.0%** | +5.0 |
| rapfi-8t vs Yixin | 74 Elo | 60% | **13-7, 65.0%** | +5.0 |

All 60 games clean; only one crash across the three matches (excluded).

The last two matter more than the first. A lopsided prediction is easy to hit
-- any sane model says the much stronger engine wins most games. A *near-even*
prediction is falsifiable: noise cannot rescue it, and a 15-point miss would
be obvious. Both landed 5 points out, and both erred in the same direction
(rapfi-8t slightly stronger than modelled), which is a small consistent bias
rather than scatter.

These three matches took rapfi-8t from 3 distinct opponents to 6, so the most
contested engine in the table is now among the best connected.

Note this row is *not* emulation-free, despite an earlier claim in the run
script that it was. `third_party/ag584/ag584-run` is a wrapper around
`docker run --platform linux/amd64`, so AlphaGomoku is x86-64 under Rosetta.
The only fully-native engines here are rapfi (all thread counts), SlowRenju,
PentaZen and KataGomo.

Four edges come from rows recorded in this file whose live logs no longer
exist (`AlphaGomoku vs Yixin`, `Yixin vs SlowRenju`, `SlowRenju vs Chis`,
`SlowRenju vs rapfi-4t`). They are the only source for Chis and rapfi-4t, so
those two rows rest on a single link each and should be read as provisional.

**Scope:** external engines only. Our own engine is deliberately excluded --
every match involving it so far ran `EngineKind::Baseline`, because
`tools/piskvork_main.cpp` defaults to it and no match script ever passed
`--engine`. The `0-20` recorded for "ours" measured the simple anchor, not the
strongest of our ten searches, so it is not comparable to anything above.

### Measured 2026-09-10/11: 13 rows at recorded settings

Everything below was run at **20 games (40 where noted), 10s/move for BOTH
sides, paired openings, `--parallel 1`**, and each carries a `match.py`
summary. `--timeout-turn-b` was left at 0, so the clock is symmetric. This
supersedes most of the older table, which was measured without recording
settings and, in several cases, was badly wrong.

| Match | Score | % | Elo | sigma |
|---|---|---|---|---|
| KataGomo vs AlphaGomoku | 15-3-2 | 80.0 | +241 | 3.35 |
| KataGomo vs VIBEFIVE | 17-3 | 85.0 | +301 | 4.38 |
| KataGomo vs Yixin | 20-0 | 100.0 | inf | — |
| KataGomo vs SlowRenju | 20-0 | 100.0 | inf | — |
| KataGomo vs rapfi-8t | 15-5 | 75.0 | +191 | 2.58 |
| AlphaGomoku vs VIBEFIVE | 14-6 | 70.0 | +147 | 1.95 |
| AlphaGomoku vs Yixin | 14-6 | 70.0 | +147 | 1.95 |
| VIBEFIVE vs SKYZERO | 15-5 | 75.0 | +191 | 2.58 |
| VIBEFIVE vs Yixin | 20-0 | 100.0 | inf | — |
| VIBEFIVE vs SlowRenju | 19-1 | 95.0 | +512 | 9.23 |
| Yixin vs SlowRenju | 15-5 | 75.0 | +191 | 2.58 |
| SlowRenju vs Chis | 16-4 | 80.0 | +241 | 3.35 |
| SlowRenju vs rapfi-4t (n=40) | 29-10-1 | 73.8 | +179 | 3.41 |

**Ordering, every step measured directly:**

    KataGomo > AlphaGomoku > VIBEFIVE > SKYZERO
                                |
                              Yixin > SlowRenju > Chis

Two skip-links confirm rather than contradict it: `KataGomo vs VIBEFIVE 17-3`
and `KataGomo vs Yixin 20-0` both land where the chain predicts. That is the
opposite of what happened with rapfi (see caveat 4).

**Three findings that overturn the old table.**

1. **SlowRenju is weak, not strong.** It is swept 20-0 by KataGomo and 19-1 by
   VIBEFIVE, and loses to Yixin 5-15. Its one win is over rapfi. The old
   reading -- that SlowRenju was the strongest classical engine -- came from
   measuring it almost exclusively against rapfi. This matters: SlowRenju is
   what we currently run on Botzone (rated 1008, rank 360).

2. **VIBEFIVE is far above its Gomocup rank.** A **268 KB binary with no
   neural network** beats Yixin 20-0 (Gomocup rates Yixin *higher*, 2217 vs
   2172), SKYZERO 15-5, and SlowRenju 19-1. Only KataGomo and AlphaGomoku beat
   it.

3. **rapfi is far below its Gomocup rank**, and it is not misconfiguration --
   see the clock sweep and the ruled-out table in caveat 1.

Our ordering now **agrees** with Gomocup for KataGomo > AlphaGomoku. The
disagreement is confined to two engines, both alpha-beta: rapfi much too low,
VIBEFIVE much too high. The plausible common factor is the compute regime --
Gomocup runs 300s/move on 32 vCPU, we run 10s on 8 cores -- where a cheap
table evaluation saturates early and a deep NNUE search never gets its depth.

**Gaps closed since (2026-09-14).** Most of what this section listed as
missing has now been played, at 20 games / 10s per move / paired openings:

| Match | Result |
|---|---|
| VIBEFIVE vs rapfi-8t | 19-1 |
| SKYZERO vs Yixin | 15-4-1 |
| AlphaGomoku vs rapfi-8t | 14-4-1 |
| rapfi-8t vs SKYZERO | 10-10 |
| rapfi-8t vs Yixin | 13-7 |
| SKYZERO vs AlphaGomoku | 4-16 |
| SKYZERO vs KataGomo | 3-17 |
| VIBEFIVE vs SKYZERO | 15-5 |
| PentaZen vs VIBEFIVE | 0-20 |
| PentaZen vs SlowRenju | 0-20 |
| PentaZen vs rapfi-8t | 0-20 |
| Barbakan vs VIBEFIVE | 0-20 |
| VIBEFIVE vs Yixin | 20-0 |
| Yixin vs KataGomo | 0-20 |
| SlowRenju vs KataGomo | 0-20 |

**PentaZen and Barbakan resolved, both far below their Gomocup ratings.**
PentaZen is rated 2171 (tied with VIBEFIVE) and lost *every one of 60 games*
against three different opponents -- it is last in the unified fit at −388.
Barbakan is rated 2442 (rank 5) and was swept 0-20 by VIBEFIVE. Neither
result depends on rapfi, so the earlier caveat -- that PentaZen had only been
measured against the one engine known to behave anomalously -- no longer
applies. Both now look like further cases of the same compute-regime effect,
measured on the opposite side from VIBEFIVE.

**SKYZERO vs Yixin, the last unplayed pairing: SKYZERO 15-4-1.** All 20 games
clean (18 five-in-a-row, 1 overline, 1 board-full draw), average 46 plies.
This link moved the middle of the table more than any other single match:
SKYZERO rose +50 -> +86 and passed rapfi-8t, while Yixin fell +12 -> −25. The
match graph now has no missing edge worth chasing.

**A crashed run nearly entered the table as a real result.** The first attempt
at this match returned 10 consecutive `no reply (timeout or crash)` games,
scored as 10 straight Yixin wins -- which would have put Yixin *above*
SKYZERO, the exact opposite of the truth. It was caught only because the
average game length was 7 plies (one game ended at ply 1). SKYZERO
smoke-tested clean standalone, and a 2-game rerun of the identical pairing
went 1-1 with both games decided by a real five-in-a-row, so the cause was
environmental -- most likely Wine/Docker contention, since the run launched
the instant the previous match's containers began tearing down. The 10 games
are archived in `discarded_sky_vs_yixin_crashrun.jsonl` and excluded.

**This is a harness gap, not just bad luck.** `match.py` scores a
non-responding engine as a loss, which is right for a tournament and wrong
for a measurement: a crash loop silently becomes a 20-0 sweep. The `clean`
column exists to catch this but nothing *warns* when a match is mostly
unclean. Two cheap guards would have caught it immediately: abort a match
when unclean games exceed some fraction, and flag any game under ~15 plies.
Note `sky_vs_kata` also carries one `no reply` game among 19 clean ones.

---

| Match | Result | clean | settings |
|---|---|---|---|
| KataGomo vs AlphaGomoku | 24-6 | | not recorded |
| AlphaGomoku vs SlowRenju | 17-3 | 20/20 | not recorded |
| **AlphaGomoku vs Yixin 2018** | **14-6** (70%) | 20/20 | 3s/move, paired openings |
| **Yixin 2018 vs SlowRenju** | **15-5** (75%) | 20/20 | 3s/move, paired openings |
| ~~embryo vs SlowRenju~~ | ~~17-3~~ | — | **withdrawn — wrong variant, see below** |
| **Chis 2.0 vs SlowRenju** | **4-16** (20%) | 20/20 | 3s/move, paired openings |
| KataGomo vs SlowRenju | **20-0** | 19/20 | not recorded |
| SlowRenju vs Rapfi-2018 | 20-0 | 20/20 | not recorded |
| **SlowRenju vs rapfi-4t** | **29-10-1** (73.8%) | 40/40 | **10s/move, paired, parallel 2, 4 threads** |

**Yixin 2018 is the second-strongest engine measured here**, between
AlphaGomoku and SlowRenju. The two new rows are mutually consistent with the
existing one: chaining them predicts AlphaGomoku − SlowRenju ≈ +338 Elo
against +301 measured, which agrees within noise — a welcome contrast to the
transitivity failure in caveat 4.

Yixin is `third_party/yixin/pbrain-Yixin2018.exe`, a Windows binary run via
`third_party/wine-engine`, and it self-identifies as a **Demo** build, so it
may be strength-limited relative to the full engine. Like every Gomocup
binary it ships no licence: test-only, not redistributable, and not
submittable to Botzone (which requires source).

**embryo and Chis had never been played at all** before 2026-09-08 — embryo
appeared only in the inventory, and Chis was built and patched (five separate
source fixes, below) but used purely as a harvest data source.

Chis has a clean number: **4-16 to SlowRenju**, well below it.

### embryo plays a different game: results withdrawn

embryo first scored **17-3 over SlowRenju**, matching AlphaGomoku against the
same opponent, and was briefly recorded here as top-tier. **That was wrong.**
Its `ABOUT` line says so directly:

    name="Embryo", version="1.1.1-new_devel-04d52e8e", build="Speedy",
    variant="Standard", author="Mira Fontan, ..."

`variant="Standard"`. In Standard gomoku an **overline** (six or more in a
row) does *not* win; in freestyle it does. The three matches show the
signature unmistakably:

    embryo vs        score    embryo overline wins   opponent overline wins
    SlowRenju        17-3              0                      2
    AlphaGomoku       5-15             0                     10
    Yixin            11-9              0                      7

embryo **never once** won by overline and was repeatedly beaten by them: it
neither plays for overlines nor defends against them, because under its own
rules they are not wins. Adjudicated as freestyle, it is systematically
punished for playing correctly.

So embryo is **not a valid freestyle opponent** and all three rows are
withdrawn. Note the bias runs *against* embryo — counting only fives, it would
be 17-1, 5-5 and 11-2, i.e. plausibly strong. Its true strength is unmeasured,
not low.

Two lessons:

* **Check `ABOUT` before believing a number.** The engine announced its
  variant on the first line of every game.
* `match.py`'s **"overlines present"** warning caught this, and is worth
  treating as a hard stop rather than a note. A large overline count for one
  side only means the two engines are playing different games.

The tier boundary is sharp and falls between *classical* and *neural*
evaluation, not anywhere inside the classical group: SlowRenju beats every
classical engine and loses to every neural one.

---

## Caveats that change how these read

### 1. The anchor is single-threaded

**What the anchor is** (checked 2026-09-08, it was previously unrecorded):

    name="Rapfi", version="0.43.02 (clang++ 17.0.0 on Apple NEON)"
    MESSAGE Evaluator set to mix9svq.

A native build from our pinned source `third_party/rapfi @3c94c2a`, which is
**upstream master HEAD** — ahead of the newest tagged release (250615, June
2025), because upstream has committed through July 2026 without cutting one.
Its `config.toml` is byte-identical to upstream's own
`rapfi-networks/config-example/config.toml`.

Two things this rules out as explanations for anything below:

* **Not an outdated engine.** There is no newer rapfi to fetch. Gomocup's
  `RAPFI25.zip` is *older* — a 2025 tournament Windows binary — and would be a
  downgrade, not an upgrade.
* **Not an outdated model.** `rapfi-networks` also ships
  `classical/model220723.bin` (2022) while we and upstream's example both use
  `model210901.bin` (2021). Tested head-to-head, same binary and NNUE weights,
  only the classical model differing: **22-18 (55%), 40 games, 0.63 sigma —
  no measurable difference.** Most likely the classical model is largely inert
  while the mix9svq NNUE evaluator is active, which would also explain why
  upstream never switched their example config.

`third_party/rapfi-bin/config.toml` sets `default_thread_num = 1`, while
Gomocup gives engines full tournament hardware. Measured directly:

    rapfi 4-thread vs rapfi 1-thread:  13-7  (65%, ~+107 Elo, 20/20 clean)

At n=20 that is not significant (p ~= 0.13), but the direction is clear. Every
figure above expressed "vs rapfi" is therefore somewhat inflated.

**Measured directly, 2026-09-09.** SlowRenju replayed against a 4-thread rapfi
at a recorded clock, to find out how much of its 18-2 was the handicap:

    SlowRenju 5.1.3 vs rapfi-4t, 40 games, 10s/move, paired, parallel 2
    29-10-1  (73.8%, +179.5 Elo)   40/40 clean, no overlines either side

Two conclusions, and they point in opposite directions:

* **The 18-2 was inflated.** 90% -> 73.8%. But at these sample sizes the drop
  is only 1.68 sigma (16.2pp, se 9.7pp) -- suggestive, not established.
* **The result survives anyway.** 73.8% is 3.41 sigma from parity
  (se 6.96pp, p ~= 0.0007). A 2019 pattern engine really does beat a
  4-thread, NNUE-active rapfi at 10s/move in this variant. Threading is not
  the explanation.

**The colour split independently replicates the ring handicap.** Black scored
just **38.75%** across all 40 games -- and both engines were hurt by roughly
the same amount when handed the forced edge opening:

    rapfi     as Black 15.0%   as White 35.0%   (-20.0pp)
    SlowRenju as Black 62.5%   as White 85.0%   (-22.5pp)

That agrees with the self-play measurement (ring openings -> Black 45.8%) from
a completely different engine pair, which is worth more than either number
alone.

It also **kills the off-distribution hypothesis** -- the guess that rapfi's
learned evaluation is miscalibrated on edge openings its training data never
contained. If that were true, rapfi's Black penalty would exceed SlowRenju's.
The two penalties are within 2.5pp of each other. Whatever rapfi is losing to
here, it is not the ring opening specifically.

**Partly explained, and the clock half is REFUTED. See the clock sweep below
before reading the rest of this section -- it was written before the 1s data
came in and its central prediction failed.**

**We test rapfi at ~20-30x less compute than its rating.**
Gomocup 2026 ran on **300s/move, 1000s/match, 32 vCPU (AMD EPYC 9K65)**. With
~30 moves a side that is ~33s/move sustained, on eight times our cores. We test
at **10s/move on 4 threads of an M2**. rapfi is the engine type most sensitive
to both dials -- deep alpha-beta with NNUE gains from depth and from threads --
while SlowRenju's threat search and VIBEFIVE's pattern table saturate early.

The inversion is therefore a **regime difference, not necessarily a fault**.
Testable prediction: rapfi's score should climb with the clock.

#### Clock sweep, 2026-09-10: the prediction FAILED

Same pairing, same config, only `--timeout-turn` changed. 40 games each,
paired openings, parallel 1, both engines on the same clock:

    clock     SlowRenju      sigma vs parity
     1s/move  26-14   65.0%      1.99
    10s/move  29-10-1 73.8%      3.41

Predicted: less time -> rapfi worse. Observed: less time -> rapfi slightly
BETTER. The difference between the two rows is **0.85 sigma** (8.75pp,
se 10.26pp), so the honest reading is that **a 10x change in clock moved
nothing measurably**, and what movement there is runs against the hypothesis.

So the *clock* half of the compute-regime story is refuted over 1-10s. The
*hardware* half (4 threads vs 32 vCPU) is untouched by this experiment and
remains live -- as does the possibility that the effect only appears at
Gomocup-scale budgets far above 10s. A 30s run would test that; 1s and 10s
agreeing tells us the curve is flat down here, not what it does up there.

**For our own purposes this is the more useful finding.** Botzone gives 1s,
and SlowRenju beats a 4-thread NNUE rapfi at 1s just as it does at 10s. The
choice of engine for our target does not hinge on resolving why.

Third independent replication of the ring handicap, from this run: Black
scored **40.0%** (16/40), against 38.75% in the 10s run and 45.8% in
SlowRenju self-play.

This matters more than a ladder correction. **Botzone gives 1s/turn, CPU-only**
-- further into the low-compute regime than our 10s tests, not less. Whatever
wins at 10s is a better guide to our target than the Gomocup table is.

Ruled out as explanations (each measured, not argued):

| Hypothesis | Verdict |
|---|---|
| Outdated engine | ours is 0.43.02, newer than the rated 0.43.01 |
| NNUE not loading | `MESSAGE Evaluator set to mix9svq`, 4 weight files present |
| Single-threaded anchor | 4 threads only moved 90% -> 73.8% |
| Stale classical model | 210901 vs 220723: 22-18, 0.63 sigma |
| Starved per-move clock | uses 10.03/9.97/9.98/9.98/9.97s of a 10s budget |
| Match-level time budgeting | see below -- sending `timeout_match` makes it WORSE |
| Board size (20x20 vs 15x15) | Gomocup rates 15x15 separately; rapfi is 1st there too |

**`timeout_match` must stay unsent.** `match.py` omits it whenever it is 0,
which is the default. That is correct and must not be "fixed":

    timeout_match NOT sent:   10.03  9.97  9.98  9.98  9.97   (full budget)
    timeout_match = 600000:    5.13  3.56  4.82  4.78  3.94   (~45%)

With a match clock rapfi starts rationing against `match_space = 21.0` and
spends under half its per-turn allowance. Both conditions then show 0.00s on
the same trailing plies -- those are positions where White already has the
forced column-6 win, so rapfi returns instantly. A resolved position reads
exactly like a starved clock in a timing trace; check the position before
concluding the latter.

It does **not** explain the whole discrepancy, though. Gomocup 2026 ranked
freestyle 15x15 as RAPFI 1st, KATAGOMO 2nd (+9 Elo), VIBEFIVE 6th, SKYZERO
9th -- yet here VIBEFIVE beats rapfi 85% and KataGomo 81%. Threading accounts
for perhaps a third of that gap. The leading remaining hypothesis is the time
control: Gomocup uses far longer budgets than the 5s/move used here, and rapfi
is deep alpha-beta with NNUE, the engine type that gains most from depth.
Untested.

### 2. Our KataGomo has hardware the tournament version does not

We built KataGomo with the OpenCL backend, so it uses the Apple GPU. The
Gomocup entry (`KATAGOMO26`, "2026_nano") is CPU-only by design. Comparing our
GPU build against CPU-only opponents overstates it.

#### Partial rerun, 2026-09-10: score WITHDRAWN, settings findings kept

The original `KataGomo vs SlowRenju 20-0` carries no settings, so it was
replayed at a recorded clock. That run was stopped after 10 games and its
score reported as 10-0.

**That 10-0 is withdrawn, and so is every interim score from that session.**
They came from `tools/replay_kata_log.py`, which reconstructs games from
KataGomo's GTP log, and it counted **any** five on the final board as a
KataGomo win without checking whose five it was. A KataGomo *loss* also leaves
a five on the board -- the opponent's. So the tool could only ever report a
sweep. Four different figures were quoted from that session (31-0, 9-9, 18-1,
19-0), each from a distinct bug in the reconstruction; none is trustworthy.

The tool now tracks KataGomo's colour per game (taken from a genmove with
stones already on the board, never the empty-board probe, which is always
`genmove B` whatever colour it holds) and attributes wins accordingly.

**Only `match.py`'s own summary belongs in this file.** A reconstruction from
one engine's log is for watching a match in flight, never for a row: when the
opponent wins, its final move is never sent to the engine and so never appears
in that engine's log at all.

The settings findings below stand -- they were read from config files, not
inferred from games.

    KataGomo   v1.12.4, OpenCL/GPU, tworule_20x_b18.bin.gz
    config     cpp/build2/oog.cfg -- FREESTYLE, NOVC, numSearchThreads 1
    clock      SlowRenju 10s/move; KataGomo VISIT-capped, not time-capped
    parallel   1 (caveat 3: concurrent GPU matches forfeit)
    openings   paired, default ring set
    plies      mean 53.6, min 22, max 85

**The clock is not symmetric.** `oog.cfg` sets `maxVisits = 200`, which binds
before the bridge's `--maxtime 10`, so KataGomo spends roughly 4s of its 10s
budget (~48 visits/sec measured) while SlowRenju spends all of its. Two further
settings pull the same way: `searchFactorWhenWinning = 0.40` makes KataGomo
think less once it is winning, and the bridge re-sends the whole position each
turn so there is no tree reuse between moves.

All three handicap KataGomo, so any sweep against it is a **conservative**
figure -- but that run was not the clock-matched comparison with the rapfi-4t
row it was set up to be, and must not be read as one. The rerun now in progress
removes the visit cap and sets `searchFactorWhenWinning = 1.0`, and KataGomo was
verified to use 10.00s per move under those settings before it started.

Reproduce with:

    python3 tools/match.py --games 40 --parallel 1 --paired-openings \
      --timeout-turn 10000 \
      --engine-a third_party/SlowRenju/pbrain-slowrenju \
      --engine-b "python3 tools/katagomo_bridge.py \
        --katago third_party/KataGomo/cpp/build2/katago \
        --model third_party/kgweights/tworule_20x_b18.bin.gz \
        --config third_party/KataGomo/cpp/build2/oog.cfg --maxtime 10"

**Do not quote any figure from that stopped run.** Beyond the tool bug above:

1. **n=10, and it was stopped early**, so it is a trajectory, not a
   measurement. It is consistent with the old 20-0, nothing more.
2. **At least one loss looks wrong.** Game 4 (22 plies, the shortest) ended
   with White walking down column 6 -- G7, G6, G5, G4, G3 -- while SlowRenju
   blocked the top end once at move 11 and then never touched the bottom.
   At move 21 White held a live four at `(6,8)-(6,11)`, dead at `(6,7)` and
   open at `(6,12)`; SlowRenju played `(8,9)`, blocking a *three* on row 9
   instead. Its own four `(10,5)-(7,8)` was dead at both ends, so there was no
   winning race to justify ignoring the four.

An engine with a dedicated `VCF/` solver missing a forced block at 10s/move is
the shape of a harness fault, not of weakness -- and six engines in this file
have already produced a convincing score that turned out to be the setup.
**Before finishing this match, check whether the short losses share that
shape.** `tools/replay_kata_log.py` rebuilds the games from `kata.log` for
exactly this purpose.

Also worth fixing before the rerun: `oog.cfg` sets `logSearchInfo = true` and
every run appends to one `kata.log`, which has reached **339 MB** across 328
runs. And the bridge re-sends the whole position each turn, so KataGomo plays
without tree reuse -- correct and desync-proof, but it costs a real MCTS engine
something, which makes this a conservative measurement of KataGomo.

### 3. Contaminated measurements elsewhere

`docs/search-comparison.md` rounds 5-8 measured OUR engines against rapfi while
`tools/match.py` was sending absolute rather than relative BOARD fields, which
colour-swapped the position for whichever opponent played White. Those
percentages, and the per-colour splits derived from them, are not reliable. The
fix is described in `match.py`; the affected rounds are flagged in that file.

---

## Engines that could not be measured

| Engine | Reason |
|---|---|
| **EGOZERO25** | 20x20 only. `START 15` is rejected, and the ONNX model has fixed `[1,4,20,20]` input with a 400-way policy head, so it cannot evaluate a 15x15 board even if the size check were bypassed. |
| **JAX25 (320-net)** | *On this Mac:* the configuration that won Gomocup silver never returned a move in 600s under Wine; only the 128-channel net is usable (~174 visits/sec), a *different, weaker* engine. **Solved on Colab — see below.** |
| **KATAGOMO26** | Runs (~56 visits/sec) but too slow at short time controls to measure fairly. |
| **STARPOINT 2026** | Runs, but overshoots short clocks by ~48% (22.2s on a 15s budget) and needs ~36s/move with a core to itself, making a 20-game match a ~4 hour run. |
| **MiniZero** | Builds and runs, but ships no trained weights -- see `docs/minizero.md`. |
| **Clap_OOG**, **Marvin/Athénan** | 2025 Olympiad silver and bronze. No public source or weights found. |

### JAX25: it was the Wine version, not the engine (2026-09-08)

JAX sat on this list for months on the assumption its 320-channel net was too
heavy for emulated CPU. That was wrong. Run on Colab (`tools/colab_jax.sh`)
the full tournament net loads and moves:

    DEBUG load model: models/dbt-320-...onnx
    7,7

**The blocker was Ubuntu's wine 6.0.3**, which does not implement the
`api-ms-win-core-processtopology-l1-1-0` API set that `onnxruntime.dll`
imports. The engine died before loading any net:

    err:module:import_dll Library api-ms-win-core-processtopology-l1-1-0.dll
        (which is needed by onnxruntime.dll) not found
    err:module:LdrInitializeThunk ... failed, status c0000135

Our working local Docker image runs **wine-10.0**; that alone is the
difference. `colab_jax.sh setup` now installs WineHQ >= 8 and refuses to run
on anything older.

**It does not need a GPU.** Four of JAX's five shipped configs set
`device = "cpu"`; only `config_unlimited.toml` uses `tensorrt`, and the README
scopes that to the separate *unlimited* tournament (CUDA 11.8 + TensorRT
8.6.1.6). The 320-net on CPU **is** the configuration that took the placing.
Chasing CUDA-under-Wine would also be futile: Colab ships CUDA 12.8.

**Not yet a result.** A first 20-game match at 5s/move returned JAX 0-20 with
2 of SlowRenju's wins by forfeit -- a starved clock, not a strength result
(lesson 2 again). Colab has 2 vCPUs and JAX's config sets
`fast_search_visits_threshold = 400`, so hundreds of visits is its normal
regime. `colab_jax.sh bench` now measures steady-state move time so the clock
is chosen from data before any match is trusted.

---

## How each engine was made to build

Every engine below is MSVC-only upstream. None built without changes, and in
four cases the binary compiled and ran while playing incorrectly -- which is
far more dangerous than a compile error, because it looks like a strength
result. Compat headers live in each engine's `compat/` directory so upstream
sources stay unmodified where possible.

### SlowRenju (GPL-3)

* `compat/Windows.h`, `compat/process.h` -- `_beginthread`, `Sleep`,
  `_endthread`, `HANDLE`, `PVOID`, `LOWORD`/`HIWORD`, `GetModuleFileName`,
  and the MSVC "safe CRT" functions (`printf_s`, `sprintf_s`, `sscanf_s`).
* `-D__int64='long long'` (MSVC type).
* `VCF.cpp`: `allocator<pair<wstring,int>>` -> `pair<const wstring,int>`;
  libc++ enforces what MSVC did not.
* **`clock()` semantics.** MSVC returns wall-clock milliseconds
  (`CLOCKS_PER_SEC` == 1000); POSIX returns CPU microseconds (1000000). Every
  time budget expired ~1000x early, so the search aborted after a few
  milliseconds. Shimmed to Windows semantics.
* **`main.cpp` BOARD handler: `moven` was never reset.** BOARD carries the
  complete position, but each command appended to the previous turn's list,
  so the board was rebuilt from a mixture of current and stale stones, and
  `moveflist[225]` overflowed. **This cost 18 of 20 games to "illegal move".**
  Piskvork sends BOARD once then TURN, which hides it; our manager sends a
  full BOARD every move, which is protocol-legal.

Before the last two fixes: 0-20 vs rapfi. After: **18-2**.

### Chis 2.0 (Mozilla Public License)

* `compat/sys/sysinfo.h` (Linux-only header, included but unused).
* `-D__int64/-D__int32/-D__int16`.
* `chis_config.cpp`: `size_of(` -> `sizeof(` (upstream typo, in code the MSVC
  project evidently never compiled).
* `chis_config.cpp`: the `#else` branch of an `#ifdef WIN32` never closed
  `namespace chis`; only the WIN32 path was ever built.
* `chis_move.cpp`: `std::swap(moves, <temporary>)` x6. MSVC binds temporaries
  to non-const references; rewritten as move-assignment, which is what it
  meant.
* `simple.cpp`: `GetProcessMemoryInfo` (psapi) in a debug print, guarded.
* **Wrong entry point.** Upstream's only `main()` drives an interactive
  console UI, so the binary answered `START 15` with a picture of a board. The
  protocol loop is `chis::gomocup()`, which has no main -- added
  `gomocup_main.cpp`.
* **`gomocup.cpp`: two BOARD bugs.** It only moved after `DONE` if the *last*
  BOARD line happened to be an opponent stone, and otherwise blocked forever
  on `cin` (7 of 20 games deadlocked). And the replay ignored the recorded
  colours entirely, colouring stones by alternating turn -- so any order other
  than strict interleaving produced a **silently wrong position**.

### Rapfi-gomocup 2018 (MIT)

* `compat/Windows.h` -- console-colour API (`SetConsoleTextAttribute`,
  `GetStdHandle`, the `FOREGROUND_*` constants) used only by a debug board
  printer; `MAX_PATH`; `sscanf_s`; `GetModuleFileName`.
* `-Wno-c++11-narrowing` (MSVC-tolerated narrowing in initialiser lists).
* `random` collides with POSIX `random(3)`; renamed the engine's RNG to
  `rapfi_rng`.
* Missing `<algorithm>`; `reverse` -> `std::reverse`.

### PentaZen 0.4.17

* `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (upstream `cmake_minimum_required(2.6)`).
* The pattern-table generator's build rule hardcodes a Windows path
  (`.../Debug/PatternGen.exe`) and never runs on Unix -- run `PatternGen`
  manually, then copy `pattern_15s.h` into `src/`.
* `protocol.cpp`: `INFO RULE` emitted an error **unconditionally**, without
  reading which rule. That stray line desynchronised the pipe, so the engine
  answered a *move request* with `ERROR unsupported rule`.
* **`search.cpp`: `Thread::print_message()` disabled.** It writes one locked
  chunk (`sync_cout << "MESSAGE"`) followed by several *unlocked*
  `std::cout` continuations, so fragments reach the manager without the
  `MESSAGE` prefix -- e.g. `" ev +v17 nd 59k tm 21 sp 2853"`. A Gomocup
  manager reads any unprefixed line as the move. **14 of 14 games were lost
  to "bad reply" or to coordinates parsed out of that text.**

Unresolved: PentaZen's table is named `pattern_15s.h` / `Pattern_s`. If "s"
means *standard*, it plays a different rule from the freestyle adjudicated
here. No overline wins have appeared, which is the direct tell, so this is
open rather than confirmed.

### Windows engines under Wine

`third_party/wine-engine <dir> <exe>` runs a Windows binary under Wine in a
`linux/amd64` container (Rosetta-backed). Two details that cost real games:

* `wineboot -i` warms the prefix **before** the manager starts timing. An
  early probe of KATAGOMO26 suggested 2 visits/sec; that was measuring prefix
  creation and model load. Steady state is ~56/sec.
* stderr is discarded. These engines print DirectX/Wine noise such as
  `Failed to create DXGI factory.`, and any stray line can be read as a move.

---

## Reproducing a row (2026-09-08)

`tools/match.py` was changed so that a recoverable `ERROR` line no longer
forfeits the game (see *Yixin* below). That sits on the path of every match
here, so the SlowRenju vs rapfi row was re-run to check nothing broke:

    tools/match.py --games 20 --parallel 4 --paired-openings --rule 0 \
        --timeout-turn 3000 --timeout-match 300000 \
        --engine-a third_party/SlowRenju/pbrain-slowrenju \
        --engine-b third_party/rapfi-bin/pbrain-rapfi

    Games: 20   A wins: 14   B wins: 5   draws: 1     (72.5%)
    win types: A 14x five, B 5x five                  <- 19/19 decisive clean

**Instrument: sound.** Every decisive game ended in a real five, no forfeits.

**Score: 72.5% here vs the 90% recorded above, and the gap cannot be
attributed.** At n=20 the difference is ~1.75 standard errors, so sampling
noise alone explains it; but the original's time control was never written
down, and time controls have produced three fake results in this file. Both
explanations fit and they cannot be separated after the fact. Hence the
warning at the top: **the 18-2 is not reproducible as recorded.**

The direction is unaffected — SlowRenju still beats rapfi decisively.

### The colour split is evidence the variant works

    SlowRenju as White:  9 wins / 0 draws / 1 loss
    SlowRenju as Black:  5 wins / 1 draw  / 4 losses

The manager forces Black's first stone onto the outer ring, a deliberately
poor square, so the expectation is that **Black is handicapped**. This row is
consistent with that.

**It does not replicate cleanly per match.** Counting by which *side* won,
across three 20-game matches at identical settings:

    match                   as Black   as White      White / Black wins
    SlowRenju vs rapfi        5-4-1      9-1-0            13 /  6
    Yixin vs SlowRenju        8-2-0      7-3-0             9 / 11
    Yixin vs AlphaGomoku      1-9-0      5-5-0            14 /  6
                                                    ------------------
    aggregate                                          36 / 23   (n=59)

Two matches show a clear White edge, one is slightly reversed. Pooled, White
wins **61.0%** of decisive games — but that is only **1.69 standard errors**
above even, so it does not reach significance either.

Treat the ring handicap as **plausible, directionally supported, and not yet
established**. The pooled number is also confounded: these are three
*different* engine pairs, so strength differences and colour are entangled.

(An earlier revision of this file asserted the Yixin match confirmed the
asymmetry. That was written from a 4-game probe, before the 20-game result
existed, and was wrong.)

### Measured properly: the ring handicap is real

The colour splits above cannot settle it — three *different* engine pairs, so
strength and colour are entangled. The clean design is **one engine against
itself**, where strength cancels exactly and the opening rule is the only
variable. SlowRenju self-play, 48 games per condition, 1.5s/move:

    opening set        Black wins  White wins   Black %   vs 50%
    ring (Outer-Open)      22          26        45.8%    -0.58 sigma
    central (free)         31          14        68.9%    +2.53 sigma

    Black scores +23.1 pp better from a central opening than from the ring
    difference: 2.31 standard errors  -> SIGNIFICANT at 95%

All 96 games decided by a real five, no forfeits. Opening sets are in
`data/openings/{ring,central}.txt` (24 each, matched in size).

Two conclusions:

1. **From the centre, Black is winning** (68.9%), consistent with freestyle
   gomoku being a proven first-player win (Allis, 1993) and with KataGomo
   evaluating opening positions at ~0.99 for Black — the saturation that makes
   our labelled value data degenerate (`docs/value-head.md`).
2. **The ring rule removes that advantage.** Confined to the outer two rows,
   Black drops to 45.8%, statistically indistinguishable from even. The
   variant converts a first-player win into a balanced game, which is exactly
   what it exists to do.

This also explains why the Botzone games behave differently: those are
**freestyle**, Black unconstrained, and there Black won both rapfi games.

**Caveat:** one engine, 1.5s/move. A stronger engine might exploit the ring
better and recover some of the loss. The direction is solid; the magnitude is
a single measurement.

### The default opening set was illegal for ~1 game in 7

Found while building the experiment above. `DEFAULT_OPENINGS` in
`tools/match.py` contained `(2,7)` and `(12,7)` — the **third** row from each
edge. Outer-Open allows only the outer two, and the engine agrees:
`src/bitboard.cpp:206` uses the identical test to the manager's `is_ring()`.

With 14 openings cycled, roughly **one game in seven of every Outer-Open match
recorded before 2026-09-08 began from a position the rule forbids.** Those
games were still legal *freestyle*, so results are contaminated rather than
void, but every row above is affected. Fixed by replacing the two with
`(7,14)` and `(13,6)`.

---

## What n=20 can and cannot resolve

Several rows here are n=20, and that is weaker than it looks. Two worked
examples from 2026-09-08:

* **rapfi model220723 vs model210901**: 22-18 (55%), 95% CI **39.6%–70.4%**.
  Consistent with better, identical, or worse.
* **SlowRenju vs rapfi re-run**: 72.5% against the 90% recorded in this file.
  A ~1.75 sigma gap — noise alone explains it.

Detecting a true **55% edge at 80% power needs ~784 games**. At roughly 40
seconds a game that is about nine hours.

So n=20 can resolve *decisive* differences (an engine that wins 75%+) and
essentially nothing else. Read every close row here as "no measurable
difference", not as a ranking.

---

## Lessons that generalise

1. **A 0-20 with no fives is a broken instrument, not a weak engine.** This
   held every single time — seven occasions now. The `win types` line in
   `match.py` output is the check that matters, not the score.

   The seventh was Yixin 2018, and the fault was in the *manager*, not the
   engine. Yixin re-parses the stones in a repeated `BOARD` as fresh moves and
   emits `ERROR opponents's move [7,0]` for one already placed — then plays a
   perfectly legal move on the very next line. `match.py` treated any `ERROR`
   as the reply and forfeited, producing 0-4 with zero fives. It now records
   the error, keeps reading, and takes the move if one arrives; a game with no
   move at all still forfeits exactly as before. Re-run with the fix: **3-1 to
   Yixin, 4/4 clean.** `tools/test_recv_regression.py` pins the behaviour.
2. **Measure throughput before interpreting any result.** Three fake 0-20s
   came from giving an engine less wall-clock than it needs for one move
   (KATAGOMO26 at 3s, KataGomo at 5s, STARPOINT at 15s).
3. **One GPU match at a time.** Concurrent KataGomo matches split a single GPU
   and produce clock forfeits that look like losses.
4. **Transitivity fails.** rapfi won Gomocup against 54 entrants and finished
   last here. SlowRenju and AlphaGomoku both scored 18-2 against rapfi, then
   AlphaGomoku won their head-to-head 17-3.
