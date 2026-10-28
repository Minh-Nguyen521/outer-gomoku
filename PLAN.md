# Outer-Open Gomoku — Tournament-Competitive Engine Plan

**Goal:** A tournament-competitive desktop engine for Outer-Open Gomoku
(外圍開局五子棋) targeting ICGA Computer Olympiad / TCGA / TAAI.

## Rules being implemented (locked in Phase 0)
- 15×15 board.
- Free-style Gomoku: win = 5 (or more) in a row, horizontally, vertically, or diagonally; overlines win; no forbidden moves.
- **Black's first move must be placed on the outer 2 rows or columns of the board** (the board's outer ring).
- Draw if no one reaches 5 in a row.

## Language / tooling
- C++20, compiled with **clang**, CMake.
- SIMD variants: portable baseline first, then AVX2 / AVX-512 (and VNNI for NNUE) via CMake presets.
- Test framework: doctest.

---

## Phase 0 — Project setup & rules correctness
- [ ] Scaffold CMake + clang presets (portable, AVX2, AVX-512).
- [ ] Board = 17×17 padded representation (15×15 playable + sentinel border ring).
- [ ] Implement and unit-test the rules above.
- [ ] Core API: `makeMove`, `undoMove`, `isLegal` (incl. Black outer-ring first-move rule), `isWin`, `isDraw`, `Zobrist hash`, `position serialization` (FEN-like), `turn`.
- [x] Test harness wired into CI.

## Phase 1 — Bitboard representation (speed foundation)
- [x] Per-color bitboards with border sentinels; branchless boundary checks (32-bit-padded approach).
- [x] Incremental move/undo updates; O(1) Zobrist (stones-only key; side-to-move folded in by search).
- [x] Benchmarks: make/undo+win-detect beats plain-array baseline (65 vs 42 Mops/s on arm64, ~1.56x); `hasFive` mask scan 141 ns, `genMovesNear` 37 ns/move.

## Phase 2 — Pattern evaluation (classical)
- [x] Threat taxonomy: L5, L4/D4, L3, etc. (`ThreatLevel` enum, ordered by strength).
- [x] OOGiveMeFive-style **M-mask** extraction: 16-bit per-line feature (`own8<<8|opp8`), 65536-entry precomputed table; off-board cells encoded as blocked-for-both (both-set bits) so edge cells classify correctly.
- [x] `evaluate()` = weighted own threats + weighted opponent threats (defense 3/2), window-method scan of all lines.
- [x] `evaluateMove()` = per-move threat (own threats + preempted opponent threats) for ordering.
- [x] Verify vs known positions (open/closed four, open three, blocking, winning moves; 33 test cases pass).
- [ ] Tune weights later via self-play (Phase 7).

## Phase 3 — Search
- [x] Alpha-beta → PVS with iterative deepening.
- [x] Transposition table (bounded, generation-based replacement; 2^20 entries).
- [x] Move ordering: TT best-move, threat-score (`evaluateMove`), history; candidate cap (24) + late-move reduction + history-only ordering near leaves.
- [x] **Threat-space search (算杀)** — VCF: forced-win precheck at the root (attacker four-threats, forced blocks, open-four kill). Decisive in Outer-Open.
- [x] Time management: fixed-per-move + node budgets with safe mid-iteration abort; tournament per-side clock wired in Phase 5.
- [x] Acceptance: 8 ply in ~1 s, 9+ ply at 5 s on a quiet 30-stone mid-game position (arm64); VCF-vs-deep-search agreement property test passes.
- [x] Incremental `EvalState`: leaf eval O(1), ~O(windows near move) update — 70x faster than full `evaluate()` per leaf, identical scores (equivalence-tested).

## Phase 4 — Opening book
- [x] Small curated book for the outer-ring opening phase (text format, 8-fold dihedral symmetry expansion, weighted responses).
- [ ] Source: published games / theses, then self-play.
- [x] Hooked into search as root move ordering (book bonus + `playBook` direct-move mode).
- [x] Tests: probe weighted best, symmetry coverage, ring-constrained first move, illegal-line skip, search book integration. (49 cases / 25,036 assertions pass release + dev.)

## Phase 5 — Protocol & external integration
- [x] Piskvork/Gomocup protocol binary (`oog_piskvork`): START/RECTSTART/RESTART/BACK/INFO/TURN/BEGIN/BOARD/ABOUT/END; 15×15 only; freestyle win rules; manager-authoritative positions; testable brain class.
- [x] Time management wired to protocol: `INFO timeout_turn`/`timeout_match`/`time_left` → `SearchLimits` (turn cap, whole-match pacing via ~1/30 remaining per move, small safety margin for strict managers).
- [x] Engine-vs-engine match runner over Piskvork (`tools/match.py`): fixed ring openings (or BEGIN mode), per-turn + whole-match clocks with grace period, move/ring legality checks, winrate/Elo aggregation, `--nodes` deterministic node-count mode.
- [x] **Parallel table runner**: `--parallel N` runs N games concurrently, each as an independent engine-process pair (one instance + seed); no shared state. Verified 8 games / 4 tables in ~11.6 s; node-count mode fully reproducible.
- [x] Tests: 11 protocol brain tests (START/ERROR, BEGIN ring first move, TURN, BOARD, INFO, UNKNOWN, ABOUT, END, RESTART/BACK). 60 cases / 25,068 assertions pass release + dev.

## Phase 6 — NNUE evaluation (strength multiplier, optional)
- [ ] NNUE eval to replace/augment classical eval (reference: rapfi `pytorch-nnue-trainer`).
- [ ] AVX-512 / VNNI eval kernels.
- [ ] Only started after Phases 1–5 are solid.

## Phase 7 — Tournament readiness
- [x] **Bounded matches vs reference engines → Elo gap measured.** rapfi wired
      into `tools/match.py` (`tools/rapfi_setup.sh`; `oog_piskvork
      --engine search|v4|v5`). Baseline `Search` scores **22% / ≈-220 Elo**
      vs NNUE rapfi over 100 games at 200 ms/turn, single worker.
      See `docs/search-comparison.md` Rounds 7-8.
- [ ] Self-play tuning (eval weights, time management).
- [ ] Bounded matches vs reference engines (OOGiveMeFive, rapfi) → measure Elo gap.
- [ ] **Node-count benchmark mode**: deterministic per-move node limit (not wall-clock) so cross-engine scores remain comparable under parallel load; fixed host/socket.
- [ ] Harden protocol edge cases, draw handling.
- [ ] Packaging for TCGA / TAAI / ICGA Computer Olympiad.

---

## Milestones
1. **Phases 0–4:** a decent, playable engine (human-beatable strong).
2. **Phase 5:** entrant-ready (plugs into tournament venues, testable vs others).
3. **Phase 6:** the long pole separating "good" from "medal contender."

## Key reference code
- `zxkyjimmy/Outer-Open-Gomoku` — OOGiveMeFive, bitboard + AVX-512 M-mask (Phases 1–2).
- `dhbloo/rapfi` — GPL-3; strongest free engine, NNUE + alpha-beta, Piskvork (Phases 3, 5, 6; GPL if shipped).
- `Azen-Huang/KataGo-Outer-Open-Gomoku` — neural/MCTS path (if pursued).
- Paper: "An algorithmic design and implementation of outer-open gomoku" (Chen, Lin, Chen, 2017).

## Findings that change the plan (Rounds 7-8, see docs/search-comparison.md)
- **Self-play rankings did not survive contact with an external opponent.**
  `SearchV4` won all eight self-play pairings at 80%, then scored *below* the
  untouched baseline against rapfi. Anchor externally before trusting any
  self-play delta.
- **The gap is evaluation-shaped, not search-shaped.** We lose 94/100 to rapfi
  with its NNUE removed entirely — i.e. to a hand-written evaluator. Adding
  more search heuristics measurably went backwards.
- **Phase 2's open "tune weights" box is now the highest-value item**, ahead
  of Phase 6: our seven eval constants have never been fitted to data, and
  `genMovesNear`'s distance-2 candidate radius is hardcoded where rapfi
  treats it as a tunable (`square2` .. `full_board`).
- **Black play is where we bleed** (baseline 8/50 as Black vs 14/50 as White).
  The forced ring opening is a large systematic factor and no engine —
  including rapfi — has Outer-Open opening theory. Uncontested edge.
- **Measurement rule:** wall-clock runs must be single-worker. A control
  re-running one matchup at 4 workers moved it from 22.0% to 1.0%.

## Notes / risks
- SIMD (AVX-512) is ~10-20% perf edge; portable AVX2 baseline keeps x86 compatibility.
- Integrating rapfi directly imposes GPL-3; building own engine avoids that.
- Benchmark on the target tournament socket before registering (SIMD capabilities vary).