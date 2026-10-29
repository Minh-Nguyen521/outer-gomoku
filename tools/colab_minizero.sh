#!/usr/bin/env bash
# Train MiniZero on Outer-Open Gomoku 15x15, on a Colab GPU.
#
# WHY THIS ENGINE: MiniZero won the GOLD MEDAL in Outer-Open Gomoku at the
# 2025 ICGA Computer Olympiad (Cohen-Solal & Cazenave, ICGA Journal 48, 2026:
# gold MiniZero24, silver Clap_OOG, bronze Marvin). It is the reigning
# champion at *our exact variant* -- not freestyle, not renju -- and its
# gomoku environment implements the rule natively:
#     gomoku.cpp:53  if (actions_.empty() && env_gomoku_rule == "outer_open")
#     model naming   gomoku_oo_15x15
#
# WHY TRAIN RATHER THAN DOWNLOAD: the competition weights are NOT published.
# RLG Lab hosts trained models for 9x9 Go, 8x8 Othello and Atari only --
# verified by probing their own hosting pattern:
#     go_9x9.tar.gz            -> 200
#     gomoku_oo_15x15.tar.gz   -> 404
#     gomoku.tar.gz            -> 404
# GitHub Releases is empty too. So the architecture and rules are public, the
# parameters are not. Consider also emailing RLG Lab (Ti-Rong Wu, Academia
# Sinica) to ask for the Olympiad weights -- it costs nothing and could save
# days of GPU time.
#
# WHY COLAB: every blocker we hit locally is absent there.
#   * CUDA present  -> torch::cuda::device_count() > 0, so the assert in
#     createNeuralNetworks() never fires and no CPU patch is needed.
#   * Ubuntu 22.04 ships Boost 1.74, which still has boost::asio::io_service.
#     Boost >= 1.87 REMOVED it; on Boost 1.92 the networking layer needs a
#     port to io_context (we did this locally -- 19 errors).
#     ***DO NOT UPGRADE BOOST.***
#   * x86_64 accepts -mpopcnt, which clang rejects on arm64.
#
# Usage (each step is a separate Colab cell):
#   from google.colab import drive; drive.mount('/content/drive')   <-- FIRST
#   !bash tools/colab_minizero.sh setup
#   !bash tools/colab_minizero.sh train 30 gmz      # smoke test first
#   !bash tools/colab_minizero.sh train 300 gmz     # then the real run
#   !bash tools/colab_minizero.sh match /path/to/model.pt
#
# PERSISTENCE. `train` refuses to start unless Drive is mounted, mirrors
# checkpoints and logs to it every 60s, verifies at exit that they actually
# arrived, and resumes automatically from Drive if the VM was recycled. A
# previous run reached 30 iterations of self-play and was lost entirely
# because none of that existed. Do not remove the guard.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MZ="${MZ:-$REPO/third_party/minizero}"
GAME=gomoku
BOARD=15
RULE=outer_open
# Where checkpoints survive a VM recycle. Overridable, but training refuses to
# start if this does not exist -- see the note in train().
DRIVE="${DRIVE:-/content/drive/MyDrive/oog-minizero}"
JOBS="$(nproc 2>/dev/null || echo 4)"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# MZ must be on LOCAL disk. If the script is uploaded to Drive and run from
# there, REPO (= script dir/..) lands inside Drive and so does MZ -- which
# means cloning, compiling and training all happen over Drive's FUSE layer.
# Compiling thousands of objects that way is glacial, and it defeats the
# local-disk-plus-mirror design entirely. Catch it rather than let someone
# wait hours for a build that should take minutes.
check_workdir() {
    case "$MZ" in
        /content/drive/*|*/MyDrive/*)
            die "MZ is inside Google Drive: $MZ

  Cloning, building and training over Drive's FUSE mount is far too slow --
  the build alone can take hours instead of minutes, and self-play writes
  many small files per iteration.

  Keeping the script on Drive is fine; just point MZ at local disk:

      !MZ=/content/minizero bash $0 ${1:-setup}

  Checkpoints still reach Drive: training mirrors them to \$DRIVE ($DRIVE)
  every 60s regardless of where MZ lives."
            ;;
    esac
}

# --------------------------------------------------------------------------
setup() {
    check_workdir setup
    log "GPU check"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader ||
        die "no GPU. Colab: Runtime > Change runtime type > GPU (T4 or better)."

    # Cache check: if a pre-built MiniZero exists on Drive, restore it instead of rebuilding.
    # This saves ~5 minutes on subsequent runs. The cache is stored at:
    #   /content/drive/MyDrive/oog-minizero/cache/minizero_build.tar.gz
    CACHE_TAR="$DRIVE/cache/minizero_build.tar.gz"
    if [ -f "$CACHE_TAR" ] && [ -x "$MZ/build/$GAME/minizero_$GAME" ]; then
        # Already restored (e.g. setup ran twice this session) -- nothing to do.
        log "Build already present: $MZ/build/$GAME/minizero_$GAME"
    elif [ -f "$CACHE_TAR" ]; then
        log "Restoring cached MiniZero build from Drive..."
        mkdir -p "$(dirname "$MZ")"
        tar -xzf "$CACHE_TAR" -C "$(dirname "$MZ")"
        if [ -x "$MZ/build/$GAME/minizero_$GAME" ]; then
            log "Cache restored: $MZ/build/$GAME/minizero_$GAME"
        else
            log "Cache corrupted; will rebuild from scratch."
            rm -rf "$MZ"
        fi
    fi

    log "System dependencies"
    # Boost 1.74 on 22.04 still has io_service. Do not upgrade it.
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        build-essential cmake git wget unzip \
        libboost-all-dev libopencv-dev python3-pybind11

    log "LibTorch (reuse Colab's CUDA-enabled torch)"
    # Avoids a separate 2GB C++ download and guarantees the C++ and Python
    # sides agree on the CUDA/ABI version.
    TORCH_CMAKE="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    python3 -c 'import torch; assert torch.cuda.is_available(), "torch sees no CUDA"'
    echo "  $TORCH_CMAKE"

    log "pybind11 (apt's is too old for Python 3.13)"
    # Colab now runs Python 3.13, where PyFrameObject became an opaque struct.
    # apt's python3-pybind11 is 2.9.1 and dereferences `frame->f_back`, so
    # learner/pybind.cpp fails with:
    #     error: invalid use of incomplete type 'PyFrameObject'
    # 2.13+ handles the opaque frame API. cmake finds /usr/include first, so
    # -Dpybind11_DIR must be passed explicitly -- installing alone is not
    # enough.
    # --ignore-installed is required, not cosmetic. The apt install above pulls
    # in python3-pybind11 2.11.1 as a dependency, and Debian packages ship no
    # RECORD file, so pip refuses to replace it:
    #     error: uninstall-no-record-file
    #     Cannot uninstall pybind11 2.11.1 -- contents are unknown
    # --ignore-installed skips the uninstall step and installs alongside it;
    # the pip copy wins because -Dpybind11_DIR below points cmake straight at
    # it. (The apt package targets python3.12 anyway, while Colab runs 3.13.)
    python3 -m pip install -q --ignore-installed "pybind11>=2.13"
    PYBIND_CMAKE="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
    echo "  $PYBIND_CMAKE  (python $(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])'), pybind11 $(python3 -c 'import pybind11; print(pybind11.__version__)'))"

    log "Learner Python dependencies"
    # THIS IS WHERE THE LOST RUN DIED. Self-play is pure C++ and completes
    # happily; the trainer is Python, and minizero/learner/train.py imports
    #     tools.tensorboard_logger -> torch.utils.tensorboard -> tensorboard
    # at module load, BEFORE any optimization runs. With tensorboard absent
    # the learner never starts, "[Optimization] Finished." never reaches
    # Training.log, and quick-run.sh reports only "Training failed to
    # complete" -- which is its own grep check, not an error from the trainer.
    #
    # Reproduced locally: a machine with torch but no tensorboard fails at
    # exactly that import. Before this step, setup installed pybind11 and
    # nothing else, so the learner had no dependencies installed at all.
    #
    # torch.utils.tensorboard is a shim: importing torch succeeds without the
    # standalone `tensorboard` package, and the failure only surfaces at the
    # SummaryWriter import. So verify the real import rather than trusting
    # that pip exited 0.
    python3 -m pip install -q tensorboard
    python3 - <<'PY' || die "learner dependencies are not importable -- training would fail at startup, before any optimization"
import sys
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    import matplotlib.pyplot, numpy  # noqa: F401
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
print("  SummaryWriter, matplotlib, numpy all import")
PY

    # Early exit: a full build was restored from Drive -- nothing left to do.
    if [ -x "$MZ/build/$GAME/minizero_$GAME" ]; then
        log "Setup complete (restored from Drive cache: $MZ)"
        return 0
    fi

    if [ ! -d "$MZ/minizero" ]; then
        log "Cloning MiniZero"
        git clone --depth 1 https://github.com/rlglab/minizero.git "$MZ"
    fi

    log "Making ALE/Atari optional (Gomoku does not need it)"
    python3 - "$MZ" <<'PY'
import os, shutil, sys
mz = sys.argv[1]

def patch(path, edits, label):
    p = os.path.join(mz, path)
    s = open(p, encoding="utf-8", errors="replace").read()
    before = s
    for old, new in edits:
        s = s.replace(old, new)
    if s != before:
        if not os.path.exists(p + ".orig"):
            shutil.copy(p, p + ".orig")
        open(p, "w").write(s)
        print("  patched", label)

# ALE and OpenCV are Atari-only but declared REQUIRED.
patch("CMakeLists.txt",
      [("find_package(ale REQUIRED)", "find_package(ale QUIET)"),
       ("find_package(OpenCV REQUIRED)", "find_package(OpenCV QUIET)")],
      "CMakeLists.txt (ale/OpenCV optional)")

# GLOB_RECURSE compiles the atari sources regardless of game.
patch("minizero/environment/CMakeLists.txt",
      [("file(GLOB_RECURSE SRCS *.cpp)",
        "file(GLOB_RECURSE SRCS *.cpp)\n"
        "if(NOT ale_FOUND)\n"
        "    list(FILTER SRCS EXCLUDE REGEX \"/atari/\")\n"
        "endif()"),
       ("ale::ale-lib", "$<$<BOOL:${ale_FOUND}>:ale::ale-lib>")],
      "environment/CMakeLists.txt (exclude atari)")

# mode_handler.cpp includes atari-only headers unconditionally, which pull in
# <ale_interface.hpp>. The two modes they serve are Atari-only.
p = os.path.join(mz, "minizero/console/mode_handler.cpp")
s = open(p, encoding="utf-8", errors="replace").read()
if '#ifdef ATARI' not in s:
    if not os.path.exists(p + ".orig"):
        shutil.copy(p, p + ".orig")
    s = s.replace('#include "obs_recover.h"\n#include "obs_remover.h"\n',
                  '#ifdef ATARI\n#include "obs_recover.h"\n#include "obs_remover.h"\n#endif\n')
    s = s.replace('    RegisterFunction("remove_obs", this, &ModeHandler::runRemoveObs);\n'
                  '    RegisterFunction("recover_obs", this, &ModeHandler::runRecoverObs);\n',
                  '#ifdef ATARI\n'
                  '    RegisterFunction("remove_obs", this, &ModeHandler::runRemoveObs);\n'
                  '    RegisterFunction("recover_obs", this, &ModeHandler::runRecoverObs);\n'
                  '#endif\n')

    def stub(text, fn):
        key = "void ModeHandler::" + fn + "()"
        i = text.find(key)
        if i < 0:
            return text
        b = text.find("{", i)
        depth, j = 0, b
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[b:j + 1]
        new = ("{\n#ifdef ATARI\n" + body[1:-1] +
               '#else\n    std::cerr << "' + fn + ': Atari-only mode, not built" << std::endl;\n'
               "#endif\n}")
        return text[:b] + new + text[j + 1:]

    for fn in ("runRemoveObs", "runRecoverObs"):
        s = stub(s, fn)
    open(p, "w").write(s)
    print("  patched console/mode_handler.cpp (guard atari modes)")
PY

    log "git_info.h (normally build-generated; only GIT_SHORT_HASH is used)"
    mkdir -p "$MZ/build/$GAME/git_info"
    printf '#pragma once\n#define GIT_SHORT_HASH "colab"\n' > "$MZ/build/$GAME/git_info/git_info.h"

    log "Configuring ($GAME, Release, -j$JOBS)"
    # -DGOMOKU is REQUIRED and not automatic: CMakeLists uses GAME_TYPE only
    # to name the executable, never as a compile definition, so environment.h
    # silently falls through to the Atari branch without it.
    cmake -S "$MZ" -B "$MZ/build/$GAME" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGAME_TYPE=GOMOKU \
        -DCMAKE_CXX_FLAGS="-DGOMOKU" \
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
        -Dpybind11_DIR="$PYBIND_CMAKE"

    # Fail loudly if cmake still picked apt's 2.9.1 -- the build would get all
    # the way to learner/pybind.cpp before erroring on PyFrameObject.
    #
    # Compare the cached path EXACTLY against the one we asked for. A previous
    # version tested `grep '=/usr'`, which is a prefix trap: the correct pip
    # path is /usr/local/lib/.../pybind11 and also begins with /usr, so a
    # perfectly good build was rejected.
    local cached_pybind
    cached_pybind="$(sed -n 's/^pybind11_DIR:[^=]*=//p' \
        "$MZ/build/$GAME/CMakeCache.txt" 2>/dev/null | head -n1)"
    if [ -n "$cached_pybind" ] && [ "$cached_pybind" != "$PYBIND_CMAKE" ]; then
        die "cmake resolved pybind11 to
      $cached_pybind
  but we asked for
      $PYBIND_CMAKE
  apt's 2.9.1 cannot build against Python 3.13. Remove the build dir and
  re-run setup:
      rm -rf $MZ/build/$GAME"
    fi

    cmake --build "$MZ/build/$GAME" -j"$JOBS"

    BIN="$MZ/build/$GAME/minizero_$GAME"
    [ -x "$BIN" ] || die "no binary at $BIN"
    log "Built $BIN"
    # Do NOT smoke-test `-mode console` here. console.cpp:44 does
    #     if (!network_) { network_ = createNetwork(config::nn_file_name, 0); }
    # so console mode loads a model at startup; with no model yet it calls
    # torch::jit::load("") and aborts. An earlier version of this script ran
    # that test and reported "WARNING: console mode did not respond" on a
    # perfectly good build -- console mode is simply not testable before
    # training has produced a .pt.
    #
    # zero_training_name needs no network and exercises config parsing, so it
    # is a real check that the binary works and the gomoku rule took effect.
    local probe
    if probe="$("$BIN" -mode zero_training_name \
            -conf_str "env_board_size=$BOARD:env_gomoku_rule=$RULE" 2>/dev/null)"; then
        echo "  binary OK; training name: $probe"
        case "$probe" in
            *_oo_*) echo "  outer_open rule confirmed in the model name" ;;
            *) echo "  WARNING: '_oo_' not in the name -- rule may not have applied" ;;
        esac
    else
        echo "  WARNING: 'zero_training_name' probe failed -- inspect $BIN manually"
    fi

    # ---- Save build cache to Drive so future VMs skip clone+patch+build.
    # Guarded by DRIVE mount: cache only when Drive is there, never fail setup
    # because of it (e.g. someone running setup before mounting).
    if [ -d "$(dirname "$DRIVE")" ]; then
        log "Caching build to Drive: $CACHE_TAR"
        mkdir -p "$(dirname "$CACHE_TAR")"
        # Only source + binary, not training runs or weights.
        tar -czf "$CACHE_TAR.tmp" \
            --exclude="$MZ/*/build/$GAME/CMakeFiles" \
            --exclude="*.o" --exclude="*.pt" --exclude="*.pkl" \
            -C "$(dirname "$MZ")" "$(basename "$MZ")" \
            && mv -f "$CACHE_TAR.tmp" "$CACHE_TAR" \
            && echo "  cached $(du -h "$CACHE_TAR" | cut -f1) to Drive" \
            || { echo "  WARNING: failed to cache build to Drive (non-fatal)"; rm -f "$CACHE_TAR.tmp"; }
    else
        echo "  (Drive not mounted -- skipping build cache)"
    fi
}

# --------------------------------------------------------------------------
train() {
    check_workdir train
    local iters="${1:-300}" alg="${2:-gmz}"
    # NOT `local`: finish() runs as an EXIT trap, which fires *after* this
    # function returns, so anything it touches must outlive the call. As
    # locals plus `set -u` these produced "mirror_pid: unbound variable" on
    # the success path only -- the failure path exits from inside train() and
    # so never noticed.
    run="${RUN_NAME:-${GAME}_oo_${BOARD}x${BOARD}_${alg}_run}"
    local_dir="$MZ/$run"
    drive_dir="$DRIVE/$run"
    mirror_pid=""

    # ---- Persistence. A previous run reached 30 iterations of self-play and
    # was then lost when the Colab VM was recycled, because nothing outside
    # the VM held the models. The old version of this function only *advised*
    # mounting Drive in a here-doc, which is exactly the kind of instruction
    # that gets skipped. It is now enforced.
    # Test the MOUNT POINT, not $DRIVE itself: $DRIVE is a directory this
    # script creates, so checking it would report "not mounted" on the very
    # first run with Drive correctly mounted.
    local mount_point
    mount_point="$(dirname "$DRIVE")"
    if [ ! -d "$mount_point" ]; then
        die "Drive not mounted: $mount_point does not exist.

  Run this in a Colab cell FIRST:

      from google.colab import drive
      drive.mount('/content/drive')

  Then re-run. Training without Drive is refused on purpose: a Colab VM can
  be recycled at any time and an unmirrored run is unrecoverable. Set
  DRIVE=/some/path to override if you really mean to."
    fi
    mkdir -p "$drive_dir" ||
        die "cannot write to $drive_dir -- is $mount_point read-only?"

    cd "$MZ"

    # ---- Resume. Copy any previous state back before starting so a recycled
    # VM costs minutes, not the whole run.
    if [ -n "$(ls -A "$drive_dir" 2>/dev/null)" ]; then
        log "Resuming: restoring $run from Drive"
        mkdir -p "$local_dir"
        cp -r "$drive_dir/." "$local_dir/" || true
        ls "$local_dir/model"/*.pt >/dev/null 2>&1 &&
            echo "  restored $(ls "$local_dir/model"/*.pt | wc -l) checkpoints" ||
            echo "  no checkpoints yet"
    fi

    # ---- Mirror loop. Training writes to LOCAL disk (Drive's FUSE layer is
    # far too slow for the many small per-iteration writes self-play makes);
    # a background loop copies checkpoints and logs out every 60s.
    #
    # `cp -u` is deliberately NOT used: it is GNU-only, and BSD/macOS cp
    # rejects it. Combined with `2>/dev/null || true` that made the sync fail
    # silently -- the same shape of bug as the run we lost, where the
    # persistence step quietly did nothing. rsync when available, plain cp
    # otherwise, and errors are visible.
    sync_out() {
        [ -d "$local_dir" ] || return 0
        mkdir -p "$drive_dir/model"
        if command -v rsync >/dev/null 2>&1; then
            rsync -a --include='*/' --include='*.pt' --include='*.cfg' \
                  --include='*.log' --exclude='*' \
                  "$local_dir/" "$drive_dir/"
        else
            for f in "$local_dir"/*.cfg "$local_dir"/*.log; do
                [ -f "$f" ] && cp -f "$f" "$drive_dir/"
            done
            for f in "$local_dir"/model/*.pt; do
                [ -f "$f" ] && cp -f "$f" "$drive_dir/model/"
            done
        fi
        return 0
    }

    # Detach stdout/stderr. Otherwise this subshell inherits the cell's pipe
    # and keeps it open, so a Colab cell (or any `| head`) never finishes even
    # after training returns -- found by a test that hung for two minutes.
    ( while true; do sleep 60; sync_out || true; done ) >/dev/null 2>&1 &
    mirror_pid=$!

    # Always run the final sync and surface the log, whatever happens. The
    # previous smoke test reported only "Training failed to complete" with no
    # detail -- the actual cause is in Training.log, which nothing preserved.
    finish() {
        local rc=$?
        kill "$mirror_pid" 2>/dev/null || true
        log "Final sync to Drive"
        sync_out
        local nloc ndrv
        nloc=$(ls "$local_dir"/model/*.pt 2>/dev/null | wc -l | tr -d ' ')
        ndrv=$(ls "$drive_dir"/model/*.pt 2>/dev/null | wc -l | tr -d ' ')
        echo "  checkpoints: $nloc local -> $ndrv on Drive"
        # Verify rather than assume. A sync that silently copies nothing is
        # how the previous run was lost.
        if [ "$nloc" -gt 0 ] && [ "$ndrv" -eq 0 ]; then
            printf '\033[1;31m  WARNING: %s\033[0m\n' \
                "local checkpoints exist but NOTHING reached Drive -- do not"
            printf '\033[1;31m  %s\033[0m\n' \
                "let this VM be recycled until you have copied them out."
        fi
        if [ -f "$local_dir/Training.log" ]; then
            log "Training.log (last 60 lines) -- the failure reason lives here"
            tail -n 60 "$local_dir/Training.log"
            log "Optimization lines"
            grep -nE "\[Optimization\]|Optimization_Done|Error|Traceback" \
                "$local_dir/Training.log" | tail -n 25 || echo "  none found"
        else
            echo "  no Training.log at $local_dir"
        fi
        return $rc
    }
    trap finish EXIT

    log "Training $GAME ($RULE, ${BOARD}x${BOARD}) with $alg for $iters iterations"
    echo "  local:  $local_dir"
    echo "  drive:  $drive_dir  (mirrored every 60s, resumable)"
    echo
    echo "  A short run (say 30 iterations) proves the pipeline end to end and"
    echo "  gives a weak model. Do that FIRST and play it against rapfi before"
    echo "  committing to 300 -- a pipeline bug found at iteration 280 is"
    echo "  expensive."
    echo

    # quick-run.sh starts the zero-server and self-play workers itself.
    # env_gomoku_rule=outer_open is the whole point: it enables the first-move
    # ring constraint. Without it you train plain freestyle and the model is
    # for a different game.
    #
    # -n pins the training directory so we know where Training.log is; without
    # it quick-run.sh invents a timestamped name we would have to guess at.
    #
    # INPUT IS A PIPE, NOT A TTY. quick-run.sh -- like Colab's `!bash` cells --
    # runs without a terminal, so zero-server.sh's interactive prompts
    #     "${train_dir} has existed. (R)estart / (C)ontinue / (Q)uit?"
    #     "Continue training from iteration: N ... Sure? (y/n)"
    # see EOF on `read`, leave their variables at the DEFAULTS -- and the
    # restart-prompt default is "R", which rm -rf's the exact directory the
    # resume step above just restored from Drive. A VM-recycled run was
    # silently restarted from iteration 0 this way. Pipe the intended answers
    # in explicitly. NOTE: both prompts use `read -n1`, which consumes exactly
    # one character and returns -- no newline. 'C\ny\n' would feed '\n' to the
    # second prompt (rejected -> server exits -> "Failed to start zero
    # training server"), so the answers must be sent WITHOUT separators.
    # The second prompt only appears on continue, and extra input after
    # training starts is ignored.
    #
    # program_use_color_message=false disables ANSI escapes that Colab shows
    # as literal "(B" sequences.
    printf 'Cy' | tools/quick-run.sh train "$GAME" "$alg" "$iters" \
        -n "$run" -g 0 --sp_progress \
        -conf_str "env_board_size=$BOARD:env_gomoku_rule=$RULE:program_use_color_message=false"
}

# --------------------------------------------------------------------------
match() {
    local model="${1:-}" games="${2:-20}"
    [ -n "$model" ] || die "usage: $0 match /path/to/model.pt [GAMES]"
    [ -f "$model" ] || die "model not found: $model"

    local conf="$MZ/build/$GAME/oog.cfg"
    cat > "$conf" <<EOF
env_board_size=$BOARD
env_gomoku_rule=$RULE
nn_file_name=$model
actor_num_simulation=400
actor_select_action_by_count=true
actor_use_dirichlet_noise=false
actor_use_gumbel_noise=false
EOF

    # MiniZero's console mode speaks GTP (boardsize/clear_board/play/genmove),
    # so our existing Piskvork<->GTP bridge drives it via --cmd; no
    # MiniZero-specific bridge is needed.
    local ENGINE="python3 $REPO/tools/katagomo_bridge.py --cmd '$MZ/build/$GAME/minizero_$GAME -mode console -conf_file $conf'"

    cd "$REPO"
    for name in rapfi slowrenju pentazen; do
        case "$name" in
            rapfi)     opp="third_party/rapfi-bin/pbrain-rapfi" ;;
            slowrenju) opp="third_party/SlowRenju/pbrain-slowrenju" ;;
            pentazen)  opp="third_party/PentaZen/build/src/PentaZen" ;;
        esac
        [ -x "$opp" ] || { echo "  skip $name (not built)"; continue; }
        log "MiniZero vs $name ($games games)"
        # Check the "win types" line: wins that are not 5-in-a-row are
        # forfeits. A 0% score with no fives means a broken instrument, not a
        # weak engine -- that mistake cost us several false results.
        python3 tools/match.py --games "$games" --parallel 2 \
            --timeout-turn 10000 --timeout-match 600000 \
            --rule 0 --verbose --engine-a "$ENGINE" --engine-b "$opp" || true
    done
}

case "${1:-}" in
    setup) setup ;;
    train) shift; train "$@" ;;
    match) shift; match "$@" ;;
    *)     sed -n '1,40p' "$0"; exit 1 ;;
esac
