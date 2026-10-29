#!/usr/bin/env bash
# Train MiniZero on Outer-Open Gomoku 15x15, on a Kaggle GPU notebook.
#
# This is the Kaggle twin of tools/colab_minizero.sh. MiniZero won GOLD in
# Outer-Open Gomoku at ICGA 2025; see that script for the full rationale.
#
# KAGGLE VS COLAB -- what changes and why:
#   * Paths. Everything lives under /kaggle/working (the only writable place
#     that also persists: Kaggle saves it as the notebook's output version
#     when the session is committed or expires).
#   * No Google Drive. Persistence = /kaggle/working output versions, plus an
#     OPTIONAL push to a Kaggle Dataset every PUSH_EVERY seconds (see
#     KAGGLE_DATASET below) if the kaggle CLI is configured.
#   * Resume. Attach a previous output/dataset as a notebook Input; anything
#     found under $KAGGLE_INPUT/oog-minizero is copied back before training.
#   * Internet is OFF by default on Kaggle. Setup needs it (apt, pip, git
#     clone). Settings > Internet > ON (requires phone verification).
#   * GPU quota is ~30h/week with 12h max per session; T4 x2 or P100.
#     The script uses GPU 0 only (same as Colab T4).
#   * Python is 3.11 with torch preinstalled -- the pybind11 >= 2.13 dance
#     from Colab (Python 3.13) is not needed, but pinning pybind11 >= 2.11
#     from pip is still done to override the apt version reliably.
#
# Usage (each step is a separate Kaggle notebook cell):
#   !bash kaggle_minizero.sh setup
#   !bash kaggle_minizero.sh train 30 gmz       # smoke test first
#   !bash kaggle_minizero.sh train 300 gmz     # the real run
#   !bash kaggle_minizero.sh match /path/to/model.pt
#
# PERSISTENCE ENV VARS
#   KAGGLE_DATASET  e.g. "username/oog-minizero". If set AND `kaggle` CLI
#                   works (KAGGLE_CONFIG_DIR/kaggle.json present), a Kaggle
#                   Dataset holding the run directory is created/updated
#                   every PUSH_EVERY seconds. Create it once with:
#                       kaggle datasets init -p RUNDIR
#                       # edit RUNDIR/dataset-metadata.json id/title
#                   (metadata is rewritten by this script automatically).
#   PUSH_EVERY      seconds between dataset pushes, default 1800.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MZ="${MZ:-/kaggle/working/minizero}"
GAME=gomoku
BOARD=15
RULE=outer_open
WORK=/kaggle/working
# Notebook inputs are mounted under /kaggle/input/<name>.
KAGGLE_INPUT="${KAGGLE_INPUT:-/kaggle/input}"
PERSIST_SRC="$KAGGLE_INPUT/oog-minizero"
KAGGLE_DATASET="${KAGGLE_DATASET:-}"
PUSH_EVERY="${PUSH_EVERY:-1800}"
JOBS="$(nproc 2>/dev/null || echo 4)"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

kaggle_ok() {
    command -v kaggle >/dev/null 2>&1 || return 1
    [ -n "$KAGGLE_DATASET" ] || return 1
    kaggle datasets status "$KAGGLE_DATASET" >/dev/null 2>&1 || return 1
}

# Push the run directory to the Kaggle Dataset. Non-fatal: /kaggle/working
# still persists as notebook output when the session is committed.
dataset_push() {
    local dir="$1" meta
    [ -d "$dir" ] || return 0
    meta="$dir/dataset-metadata.json"
    cat > "$meta" <<EOF
{
  "id": "$KAGGLE_DATASET",
  "licenses": [{"name": "CC0-1.0"}],
  "title": "$(basename "$KAGGLE_DATASET")"
}
EOF
    # pt + pkl are the irreplaceable artifacts; sgf can be re-derived but is
    # cheap, keep it. vrt/vdat are huge scratch buffers -- exclude.
    mkdir -p "$dir/.kaggle_exclude" 2>/dev/null || true
    kaggle datasets version -p "$dir" -m "auto-push $(date -u +%FT%TZ)" \
        >/dev/null 2>&1 \
        && echo "  dataset pushed: $KAGGLE_DATASET" \
        || echo "  (dataset push failed -- /kaggle/working still holds everything)"
    return 0
}

# --------------------------------------------------------------------------
setup() {
    log "GPU check"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader ||
        die "no GPU. Kaggle: Settings > Accelerator > GPU T4 x2 (or P100)."

    log "System dependencies"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        build-essential cmake git wget unzip rsync \
        libboost-all-dev libopencv-dev python3-pybind11

    log "LibTorch (reuse the notebook's CUDA-enabled torch)"
    TORCH_CMAKE="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    python3 -c 'import torch; assert torch.cuda.is_available(), "torch sees no CUDA"'
    echo "  $TORCH_CMAKE"

    log "pybind11 + learner dependencies"
    # Kaggle's Python is 3.11, so apt's pybind11 would mostly work, but pin the
    # pip copy anyway so cmake has one unambiguous recent version.
    python3 -m pip install -q --ignore-installed "pybind11>=2.11" tensorboard
    python3 - <<'PY' || die "learner dependencies are not importable"
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    import matplotlib.pyplot, numpy  # noqa: F401
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    raise SystemExit(1)
print("  SummaryWriter, matplotlib, numpy all import")
PY

    if [ ! -d "$MZ/minizero" ]; then
        log "Cloning MiniZero"
        mkdir -p "$MZ"
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

patch("CMakeLists.txt",
      [("find_package(ale REQUIRED)", "find_package(ale QUIET)"),
       ("find_package(OpenCV REQUIRED)", "find_package(OpenCV QUIET)")],
      "CMakeLists.txt (ale/OpenCV optional)")

patch("minizero/environment/CMakeLists.txt",
      [("file(GLOB_RECURSE SRCS *.cpp)",
        "file(GLOB_RECURSE SRCS *.cpp)\n"
        "if(NOT ale_FOUND)\n"
        "    list(FILTER SRCS EXCLUDE REGEX \"/atari/\")\n"
        "endif()"),
       ("ale::ale-lib", "$<$<BOOL:${ale_FOUND}>:ale::ale-lib>")],
      "environment/CMakeLists.txt (exclude atari)")

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
    printf '#pragma once\n#define GIT_SHORT_HASH "kaggle"\n' > "$MZ/build/$GAME/git_info/git_info.h"

    log "Configuring ($GAME, Release, -j$JOBS)"
    cmake -S "$MZ" -B "$MZ/build/$GAME" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGAME_TYPE=GOMOKU \
        -DCMAKE_CXX_FLAGS="-DGOMOKU" \
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
        -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"

    cmake --build "$MZ/build/$GAME" -j"$JOBS"

    BIN="$MZ/build/$GAME/minizero_$GAME"
    [ -x "$BIN" ] || die "no binary at $BIN"
    log "Built $BIN"
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
}

# --------------------------------------------------------------------------
train() {
    local iters="${1:-300}" alg="${2:-gmz}"
    shift 2 2>/dev/null || shift $#
    local extra_args=("$@")
    run="${RUN_NAME:-${GAME}_oo_${BOARD}x${BOARD}_${alg}_run}"
    local_dir="$MZ/$run"
    push_pid=""
    mkdir -p "$local_dir"

    # ---- Resume. Unlike Colab, /kaggle/working does NOT survive across
    # sessions; a previous run must be attached as notebook Input (either
    # the earlier notebook output, or the auto-pushed Kaggle Dataset).
    if [ -d "$PERSIST_SRC" ] && [ -n "$(ls -A "$PERSIST_SRC" 2>/dev/null)" ]; then
        log "Resuming: restoring $run from notebook input"
        cp -r "$PERSIST_SRC/." "$local_dir/" || true
        ls "$local_dir"/model/*.pt >/dev/null 2>&1 &&
            echo "  restored $(ls "$local_dir"/model/*.pt | wc -l) checkpoints" ||
            echo "  no checkpoints in the input"
    else
        echo "  (no notebook input at $PERSIST_SRC -- starting fresh)"
    fi

    # ---- Optional periodic dataset push (extra safety on top of
    # /kaggle/working, which only persists when the session is COMMITTED).
    if kaggle_ok; then
        log "Dataset push every ${PUSH_EVERY}s -> $KAGGLE_DATASET"
        ( while true; do sleep "$PUSH_EVERY"; dataset_push "$local_dir" || true; done ) \
            >/dev/null 2>&1 &
        push_pid=$!
    else
        echo "  (Kaggle Dataset push disabled -- set KAGGLE_DATASET=user/slug and"
        echo "   configure kaggle CLI to enable; /kaggle/working still persists"
        echo "   as notebook output when the session is saved)"
    fi

    finish() {
        local rc=$?
        kill "$push_pid" 2>/dev/null || true
        if kaggle_ok; then
            log "Final dataset push"
            dataset_push "$local_dir"
        fi
        log "Artifacts under $local_dir (saved with the notebook output)"
        ls -lh "$local_dir"/model/*.pt 2>/dev/null | tail -n 5 || echo "  no checkpoints"
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
    echo "  local:  $local_dir  (/kaggle/working -- saved with notebook output)"
    echo
    echo "  A short run (30 iterations) proves the pipeline end to end. Do that"
    echo "  FIRST -- a pipeline bug found at iteration 280 is expensive."
    echo
    echo "  12h session limit: when Kaggle recycles the VM, SAVE VERSION the"
    echo "  notebook (keeps /kaggle/working as output), then re-add this"
    echo "  notebook's output as an Input and re-run this cell to resume."
    echo

    # All visible GPUs, one self-play worker per GPU (plus -w to multiply; see
    # tools/quick-run.sh usage). Default quick-run behavior enumerates GPUs
    # only when CUDA_VISIBLE_DEVICES is unset, so derive it from nvidia-smi
    # rather than hardcoding "-g 0" (which pinned worker count to 1 GPU even
    # on T4 x2).
    local gpus
    gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
    [ -n "$gpus" ] || gpus=0

    cd "$MZ"
    ansi_filter() { sed -e 's/\x1b\[[0-9;?]*[ -\/]*[@-~]//g' -e 's/\x1b[()][0-9A-B]//g' -e 's/\x1b[@-~]//g'; }
    printf 'Cy' | tools/quick-run.sh train "$GAME" "$alg" "$iters" \
        -n "$run" -g "$gpus" --sp_progress \
        -conf_str "env_board_size=$BOARD:env_gomoku_rule=$RULE:program_use_color_message=false" \
        "${extra_args[@]}" \
        2>&1 | ansi_filter
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
