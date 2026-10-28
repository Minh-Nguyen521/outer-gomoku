# Related work

Four papers researched, each summarized with a note on relevance to this
project's actual goal (`PLAN.md`: a tournament-competitive engine for
Outer-Open Gomoku, targeting ICGA/TCGA/TAAI).

## 1. Chen, Lin & Chen (2017) — "An Algorithmic Design and Implementation of Outer-Open Gomoku"

- ICCCS 2017 (Krakow, Poland), pp. 26-30.
- Already this project's primary reference — `PLAN.md` cites it directly,
  and `zxkyjimmy/Outer-Open-Gomoku` (OOGiveMeFive, the paper's own
  implementation) is named as the source for this project's Phase 1-2
  techniques.
- Defines Outer-Open Gomoku itself around three goals: "maintaining connect
  five," "simple rule," and "fairness" — i.e. the outer-ring first-move
  restriction exists specifically to fix Gomoku's well-known first-move
  advantage without adding renju-style forbidden-move complexity.
- Technique: bitboard + "inverse bitboard" representation exploiting modern
  ISA instructions, reported at >56M nodes/sec on a typical PC. This is the
  same bitboard-with-sentinel-border idea already implemented in this
  project's `BitBoard` (`src/bitboard.hpp`), and the M-mask pattern table
  already implemented in `eval.cpp`'s `classifyFeature`.
- OOGiveMeFive (the paper's engine) won 6 gold medals across 8 computer-game
  competitions — the direct precedent for this project's own tournament
  ambitions.
- **Status: already fully absorbed.** No new action here; this is
  foundational, not new research.

## 2. Cohen-Solal & Cazenave (2026) — "Minibal: Balanced Game-Playing Without Opponent Modeling" (arXiv:2603.23059)

- Submitted March 2026. Authors: Quentin Cohen-Solal, Tristan Cazenave.
- **Directly relevant in an unexpected way: Outer-Open-Gomoku is one of the
  paper's 7 tested games** (alongside International Draughts, Chess, Lines
  of Action, Connect6, Xiangqi, Havannah) — an independent academic result
  exists for exactly this variant.
- **The paper's goal is the opposite of this project's.** `PLAN.md` wants
  maximum tournament strength; Minibal explicitly builds an agent that
  *deliberately plays worse* to stay balanced with a weaker opponent
  ("highly powerful game-playing agents... consistently overwhelm human
  players, offering little enjoyment"). Not applicable to the stated goal
  as-is.
- Algorithm family is also different: built on **Unbounded Minimax** (a
  best-first search over a resolution tree), not the alpha-beta/PVS +
  iterative deepening this project uses. Two variants tested:
  - `Minibal_n` (Minimum-Balance-Near): minimizes `|score|`.
  - `Minibal_+` (Minimum-Balance-Positive): wins if possible, but by the
    smallest margin.
- Results at 20s/move (their Table 2): `Minibal_+` reached a Binary Gain of
  −2.96% (near-perfect balance) vs. Unbounded Minimax's 92.68% (dominant),
  confirming the balancing mechanism works; `Minibal_n` overshot into
  losing 69% of games, flagged by the authors as unsuitable.
- **Possible relevance despite the goal mismatch**: if this project ever
  wants an adjustable-difficulty / practice-mode opponent (distinct from
  the tournament-strength engine), `Minibal_+`'s decision rule is a
  concrete, already-validated-on-this-exact-game starting point. Not
  something to build now, but worth remembering exists.

## 3. Cazenave et al. — "Learning to Play Two-Player Perfect-Information Games without Domain Knowledge" (JMLR v27)

- Presents **Athénan**, a unified self-play RL algorithm, compared against
  ExIt (a leading knowledge-free self-play baseline) on **Hex, Othello,
  Arimaa, and Morpion Solitaire** — not Gomoku directly, though Hex is the
  same "connection game" family.
- Core mechanisms: extends Unbounded Best-First Minimax's best-action
  sequences to terminal states; generalizes tree bootstrapping to
  non-linear function approximation; replaces the traditional binary
  win/loss signal with richer reinforcement signals (quick wins, delayed
  losses, scoring); a "completion mechanism" exploiting state resolution;
  a novel "ordinal distribution" for action selection. Reported to beat
  ExIt and reach state-of-the-art on all three tested games.
- **Relevance to this project**: `PLAN.md`'s Phase 6/7 plan is the
  rapfi-style path — hand-crafted classical eval (already built, see
  `eval.cpp`'s `ThreatLevel` table) later augmented/replaced by a trained
  NNUE, with weight tuning via self-play. Athénan represents a genuinely
  different paradigm: no hand-crafted domain knowledge at all, a different
  search backbone (best-first, not alpha-beta), and richer training
  signals than plain self-play win/loss. Adopting it wholesale would be a
  from-scratch redesign, not a drop-in addition — worth being aware of as
  an alternative direction for Phase 6/7, not a near-term action item.

## 4. Cao & Lin (2019) — "UCT-ADP Progressive Bias Algorithm for Solving Gomoku" (arXiv:1912.05407, IEEE SSCI 2019)

- **Gomoku-specific** (freestyle, not confirmed whether outer-open rules
  apply — likely standard Gomoku).
- Algorithm family: **MCTS/UCT**, not alpha-beta — a different search
  paradigm from this project's `negamax`.
- Three components, each with a direct analog or contrast in this project:
  1. **Progressive Bias heuristic**: `UCB = v_i + k1*sqrt(ln(N)/n_i) +
     k2*(H_i/max(H))`, where `H_i` is an *exponential* threat-value sum
     (decay factor 0.90) replacing an earlier quadratic formula, because
     exponential weighting separates e.g. "live three" from "sleep four"
     better. This project's own `ThreatLevel` scoring (`eval.cpp`:
     `S_CLOSED_TWO=100` .. `S_FIVE=1,000,000`) is already roughly
     exponential in shape (each tier ~5-10x the one below) — a useful
     independent confirmation that this project's existing hand-picked
     scale is in a reasonable range, not an arbitrary choice.
  2. **VCF/VCT pruning as candidate-move filtering**: the paper flags any
     move whose heuristic value is "on the order of 10³ or more" as an
     "effective attack" and prioritizes it in the candidate list. This
     project already has a dedicated, exact VCF search (`vcf()`/`vcfWin()`
     in `search.cpp`) rather than a heuristic threshold — a stronger,
     already-implemented version of the same idea (exact forced-win
     detection vs. a score-threshold heuristic).
  3. **ADP replacing MCTS rollouts**: a 3-layer MLP (32 hand-picked Gomoku
     pattern features + side-to-move, sigmoid output → win probability)
     trained via self-play, replacing full random-playout simulation.
     Reported 40x throughput improvement (500 → 20,000+ iterations per 20s)
     and faster convergence to the correct value than plain UCT.
     **This is a lightweight, low-effort stepping stone worth flagging for
     Phase 6**: rather than jumping straight to a full NNUE (rapfi's much
     larger network), a small MLP over hand-picked features is a
     dramatically cheaper first step toward "learned eval," reusing this
     project's existing feature/pattern vocabulary (`ThreatLevel`/M-mask)
     as the input representation.
  - Measured results: 30-0 vs. "Mushroom," 28-2 vs. "Valkyrie," 27-3 vs.
    "Pisq7" (baseline bots named in the paper, not otherwise identified
    here).

## 5. Allis, van den Herik & Huntjens (1993) — "Go-Moku and Threat-Space Search"

- The paper that solved Go-Moku (program *Victoria*); with the companion
  proof-number search work (Allis et al. 1994) it established free-style
  Go-Moku as a first-player win.
- **The single most relevant unimplemented idea found in this round of
  research.** Its claim is that a winning Go-Moku line is nearly always a
  *chain of threats*, so searching only the threat sub-space finds wins that
  a general-depth search at the same cost misses entirely.
- This project already implements the *weaker half* of it. `vcf()` in
  `search.cpp` is Victory-by-Continuous-Fours: it chains moves that create a
  four, where the defender's reply is forced to exactly one square. Allis's
  threat space also admits **three-threats** (a move creating an open three,
  threatening an *open four* next), where the defender has several defences
  rather than one. Chains mixing threes and fours — VCT, Victory by
  Continuous Threats — are where most real Go-Moku wins live, and a
  fours-only search cannot see a three-four-five combination at all.
- Other machinery in the paper: dependency-based search (ordering threats by
  which squares they depend on, via topological sort over a DAG of threat
  nodes), the "all defensive moves at once" simplification (let the defender
  answer a threat with *every* defence simultaneously — if the attacker still
  wins, the win is real), and counter-threat refutation (reject a threat
  sequence if the defender can start their own forcing sequence of equal or
  greater severity).
- **Status: acted on.** Implemented as `SearchVct` (`src/search_vct.*`) —
  see `docs/search-comparison.md` Round 5. The three-threat extension and
  counter-threat refutation are implemented; the dependency/all-defences
  machinery is not, because it exists to prove wins over very long chains
  offline and this engine's chains are node-budget-bounded to a few threats.

## 6. Practitioner reference — tournament-winning Go-Moku engine writeup

- <https://sortingsearching.com/2020/05/18/gomoku.html> — an engineering
  writeup rather than a paper, but it is a first-hand account of a
  competition-winning free-style Go-Moku engine and it names its techniques
  concretely, which is exactly what was missing from the academic sources.
- Techniques named that this project had **none** of at the time of reading:
  - **Null-move forward pruning**, explicitly gated on "no threat sequence
    for the opponent". Notable because the standard objection to null move —
    zugzwang — *does not exist in Go-Moku*: placing a stone can never worsen
    your own position, so passing is a strict lower bound on moving and the
    null-move assumption is sounder here than in chess, where it was
    invented. (Lineage: Beal 1989; Donninger 1993.)
  - **Threat-driven candidate generation** — considering "only moves that
    defend against the danger" rather than generating a fixed neighbourhood
    and relying on ordering. Restricting *generation* is a hard cut; this
    project only had the soft tool (ordering plus a candidate cap of 24).
  - Rotated bitboards (0°/90°/45°/135°) and per-player threat boards updated
    only within distance 4 of a move — an alternative to this project's
    `EvalState` incremental window scheme, not obviously better.
  - An Allis-style exponential static eval: for each empty intersection take
    the two best threats over the four directions and score
    `1.5·1.8^a + 1.8^b`. Independent confirmation (alongside Cao & Lin) that
    this project's roughly-exponential `ThreatLevel` scale is the right
    shape.
- **Status: acted on.** Null-move pruning became `SearchNullKiller`
  (`src/search_nullkiller.*`, bundled with the killer heuristic of Akl &
  Newborn 1977 and the counter-move heuristic of Uiterwijk 1992, both also
  absent here); threat-driven candidate generation became
  `SearchThreatCand` (`src/search_threatcand.*`). See
  `docs/search-comparison.md` Round 5.

## Cross-paper observations

- **Two different search paradigms** show up across these papers: this
  project (and Chen & Lin) use alpha-beta/PVS; Minibal uses Unbounded
  Minimax (best-first); Cao & Lin and implicitly Athénan use MCTS/UCT
  variants. None of the newly researched papers validate alpha-beta as the
  best choice for this game — worth noting as a gap, not a recommendation
  to switch (a paradigm change is a large undertaking, and `PLAN.md`'s
  alpha-beta-based plan already has working Phases 1-5).
- **Two independent, lightweight-first suggestions for Phase 6** (NNUE):
  Cao & Lin's small MLP-over-hand-picked-features as a cheaper first step
  than a full NNUE, and Athénan's richer-than-binary training signal as a
  refinement to whatever self-play tuning Phase 7 eventually does.
- **Minibal's inclusion of Outer-Open-Gomoku** is worth citing in `PLAN.md`
  regardless of relevance to the strength goal, simply because it's rare
  independent academic validation that this specific ruleset is used in
  real research beyond this project's own named references.
