# MiniZero: the Outer-Open Gomoku champion

MiniZero won the **gold medal in Outer-Open Gomoku at the 2025 ICGA Computer
Olympiad** -- our exact variant, not freestyle and not renju.

Source: Cohen-Solal & Cazenave, *The 2025 Computer Olympiad: Athénan Wins 9
Gold Medals*, ICGA Journal 48 (2026).

| Outer-Open Gomoku 2025 | Program | Team | Country |
|---|---|---|---|
| Gold | **MiniZero24** | Yu-Hung Chang, Ti-Rong Wu et al. | Taiwan |
| Silver | Clap_OOG | I-Chen Wu's group (NYCU) | Taiwan |
| Bronze | Marvin (Athénan) | Cohen-Solal & Cazenave | France |

The report also notes that MiniZero reimplements **Gumbel AlphaZero**, and
that *"in Connect6 and Gomoku, threat-space features are explicitly encoded as
input"* -- which is what `tools/encode_features.py` already does.

## It implements our variant natively

    gomoku.cpp:53   if (actions_.empty() && config::env_gomoku_rule == "outer_open")
    gomoku.h:38     name() -> kGomokuName + (rule == "outer_open" ? "_oo_" : "_") + size

So `env_gomoku_rule=outer_open` gives the first-move ring constraint directly,
and models are named `gomoku_oo_15x15`. No approximation needed.

## The weights are NOT published

This is the blocker. RLG Lab hosts trained models for 9x9 Go, 8x8 Othello and
Atari -- the games in their IEEE ToG paper -- but not the Olympiad entries.
Verified by probing their own hosting pattern:

    go_9x9.tar.gz            -> 200
    gomoku_oo_15x15.tar.gz   -> 404
    gomoku.tar.gz            -> 404

GitHub Releases is empty. So the architecture and rules are public; the
parameters are not. Options are to train our own (see below), or to ask RLG
Lab (Ti-Rong Wu, Academia Sinica) directly -- they publish models for their
papers and the Olympiad weights may simply never have been packaged.

## Build blockers

**None of these is a genuine CUDA requirement.** The README says NVIDIA and
`torch::cuda::device_count()` gates startup, but inference is LibTorch with an
existing CPU branch (`network.h`: `gpu_id_ == -1` -> `torch::Device("cpu")`).
It builds and runs on an Apple Silicon Mac with no GPU at all.

| # | Blocker | Fix | Where |
|---|---|---|---|
| 1 | `min(cuda::device_count(), N)` is 0 without CUDA, tripping `assert(num_networks > 0)` | fall back to one CPU network, pass `gpu_id = -1` | `actor_group.cpp` |
| 2 | `find_package(ale REQUIRED)` | make QUIET -- Atari-only | `CMakeLists.txt` |
| 3 | `GLOB_RECURSE` compiles the Atari sources regardless of game | exclude `/atari/` when ALE absent | `environment/CMakeLists.txt` |
| 4 | `-mpopcnt` is x86-only; clang rejects it on arm64 | gate on `CMAKE_SYSTEM_PROCESSOR` | `CMakeLists.txt` |
| 5 | `boost::asio::io_service` **removed in Boost 1.87** (19 errors on Boost 1.92) | port to `io_context`; `io_service::work` -> `executor_work_guard` | `base_server.h`, `zero_server.h` |
| 6 | `deadline_timer` now needs an opt-in Boost.DateTime dependency | switch to chrono `steady_timer`, `expires_from_now` -> `expires_after` | `zero_server.{h,cpp}` |
| 7 | `mode_handler.cpp` includes Atari-only headers unconditionally | guard under `#ifdef ATARI`, stub the two modes | `console/mode_handler.cpp` |
| 8 | pybind11 from apt is too old for **Python 3.13** (`PyFrameObject` became opaque) | `pip install "pybind11>=2.13"`, pass `-Dpybind11_DIR` | Colab only |

Plus one that produces no error at all: **`-DGOMOKU` must be passed
explicitly.** `CMakeLists.txt` uses `GAME_TYPE` only to name the executable,
never as a compile definition, so `environment.h` silently falls through to
the Atari branch without it.

Blockers 5, 6 and 8 are environment-dependent: Boost >= 1.87 and Python 3.13
are new enough that upstream has not met them. Ubuntu 22.04 (Colab's image)
ships Boost 1.74, which still has `io_service` -- **do not upgrade Boost there.**

## Running it

MiniZero's console mode speaks **GTP** (`boardsize`, `clear_board`, `play`,
`genmove`), so the existing Piskvork<->GTP bridge drives it unchanged:

    tools/katagomo_bridge.py --cmd '<path>/minizero_gomoku -mode console -conf_file <cfg>'

No MiniZero-specific bridge is needed. `--cmd` was added for this.

## Training

`tools/colab_minizero.sh` does setup / train / match on a Colab GPU. The
Olympiad recipe is Gumbel MuZero for 300 iterations:

    tools/quick-run.sh train gomoku gmz 300 -g 0 -conf_str "env_board_size=15:env_gomoku_rule=outer_open"

`env_gomoku_rule=outer_open` is the setting that matters. Without it this
trains plain freestyle, which is a different game.

**Status:** a 30-iteration smoke test on a Colab T4 completed self-play
successfully -- 2000 games, average length 112.6 plies, model directory named
`gomoku_oo_15x15_gmz_1bx256_n16` (confirming the rule and board size took
effect) -- then reported `Training failed to complete` during the optimization
step. That run was then lost entirely when the VM was recycled, because
nothing outside the VM held the checkpoints.

### Why `Training failed to complete` told us nothing

It is not an error from the trainer. `quick-run.sh` waits on the workers with
`wait -n` (which returns when *any* one exits), then greps the training log:

    last_iter=$(grep -Eio "\[Iteration\] =+[0-9]+=+" $train_dir/Training.log | tail -n1)
    if [[ ... == $zero_end_iteration ]] && grep -F "$last_iter" -A 100 \
        $train_dir/Training.log | grep -Eq "\[Optimization\] Finished."

If either test fails it prints that one line and exits 1. **The actual cause
is in `$train_dir/Training.log`**, which nothing preserved.

### Both problems are now fixed in `tools/colab_minizero.sh`

- `train` **refuses to start unless Drive is mounted**. The old version only
  advised it in a here-doc, which is the kind of instruction that gets
  skipped -- and was.
- Training writes to local disk (Drive's FUSE layer is far too slow for
  self-play's many small writes) and a background loop mirrors `*.pt`,
  `*.cfg` and `*.log` to Drive every 60s.
- `-n <run>` pins the training directory, so `Training.log` has a known path
  instead of a timestamped name we would have to guess.
- On exit -- success or failure -- it syncs, **verifies** the checkpoints
  actually arrived, and dumps the tail of `Training.log` plus every
  `[Optimization]` / `Traceback` / `Error` line. The next failure diagnoses
  itself.
- If Drive holds a previous run, it is restored first, so a recycled VM costs
  minutes rather than the run.

One trap found while testing this: **`cp -u` is GNU-only**. BSD/macOS `cp`
rejects it, and paired with `2>/dev/null || true` the sync failed silently --
the same shape of bug as the loss it was meant to prevent. It now uses rsync
where available, plain `cp` otherwise, and checks the result.

### Next step

Re-run the 30-iteration smoke test. It will now either complete or print the
real optimization error.
