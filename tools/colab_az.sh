#!/usr/bin/env bash
# Train oog_az (our AlphaZero) on a Colab GPU, resumably.
#
# Companion to tools/colab_minizero.sh, and it makes the same three promises,
# because this project has already lost a complete training run to a recycled
# Colab VM (docs/minizero.md):
#
#   1. It REFUSES to start unless Drive is mounted. The old MiniZero script
#      only advised mounting it in a here-doc, which is exactly the kind of
#      instruction that gets skipped -- and was.
#   2. Training writes to LOCAL disk and mirrors out to Drive after every
#      iteration. Drive's FUSE layer is far too slow for self-play's many
#      small writes.
#   3. Resuming is the default. Re-running the same command after a
#      disconnect picks up from the last completed iteration rather than
#      starting over.
#
# WHAT MAKES RESUME CORRECT HERE: oog_az/train.py writes state.json after
# self-play, and trusts IT rather than the checkpoint's iteration number. The
# checkpoint only advances when a candidate is PROMOTED, so a run with several
# rejected gates would otherwise resume too early and replay games it already
# had.
#
# USAGE (in Colab, after mounting Drive):
#   bash colab_az.sh setup
#   bash colab_az.sh train 300
#   bash colab_az.sh status
#
# NOTE: written with the Write tool, not a `cat` heredoc -- `cat` is aliased to
# `bat` in this shell and injects ANSI escapes into heredoc bodies.

set -euo pipefail

REPO="${REPO:-/content/oog}"
DRIVE="${DRIVE:-/content/drive/MyDrive/oog_az}"
RUN="${RUN:-$REPO/oog_az/runs/colab}"

# Self-play settings. CONCURRENCY IS THE NETWORK BATCH SIZE -- it is the one
# knob that decides whether the GPU is busy or idle. 1 is the old serial
# behaviour (one forward pass per simulation); 64 keeps a T4 fed.
GAMES="${GAMES:-200}"
SIMS="${SIMS:-50}"        # Gumbel makes 50 workable; vanilla would need ~800
# GUMBEL=0 disables it. Tested with an explicit comparison, NOT with
# ${GUMBEL:+...}: that expands whenever the variable is non-EMPTY, so
# GUMBEL=0 would have switched Gumbel ON -- the exact opposite of what it
# reads as.
GUMBEL="${GUMBEL:-1}"
NCONSIDERED="${NCONSIDERED:-16}"
CONCURRENCY="${CONCURRENCY:-64}"
STEPS="${STEPS:-400}"
BATCH="${BATCH:-256}"
BLOCKS="${BLOCKS:-6}"
CHANNELS="${CHANNELS:-96}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

require_drive() {
    local mount_point
    mount_point="$(dirname "$(dirname "$DRIVE")")"   # /content/drive
    [ -d "$mount_point" ] || die "Drive not mounted: $mount_point missing.

  Run this in a Colab cell FIRST:

      from google.colab import drive
      drive.mount('/content/drive')

  Training without Drive is refused on purpose: a Colab VM can be recycled at
  any time and an unmirrored run is unrecoverable. Set DRIVE=/some/path to
  override if you really mean to."
    mkdir -p "$DRIVE" || die "cannot write to $DRIVE"
}

setup() {
    log "GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader ||
        echo "  no GPU found -- Runtime > Change runtime type > GPU. Training \
on CPU is possible but pointless at any useful scale."
    log "Python dependencies"
    # torch is preinstalled on Colab. numpy likewise. Nothing else is needed:
    # oog_az deliberately has no tensorboard dependency -- that exact missing
    # import silently killed a MiniZero run (self-play fine, learner dead at
    # import, and the only symptom was "Training failed to complete").
    python3 - <<'PY'
import torch, numpy
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  numpy {numpy.__version__}")
PY
    log "Sanity check (rules + a short search, no training)"
    cd "$REPO"
    python3 - <<'PY'
import sys; sys.path.insert(0, ".")
from oog_az.env import Board
b = Board()
legal = b.legal_actions()
assert len(legal) == 104, f"expected 104 ring openings, got {len(legal)}"
assert 7 * 15 + 7 not in legal, "centre must be illegal for Black's first move"
b.play(legal[0])
assert len(b.legal_actions()) == 224, "ring must bind on move 1 only"
print("  rules OK: 104 ring openings, centre rejected, ring binds once")
PY
    log "Ready. Next:  bash colab_az.sh train 300"
}

train() {
    local until="${1:-300}"
    require_drive
    cd "$REPO"

    # Restore first, so a recycled VM costs minutes rather than the run.
    if [ -n "$(ls -A "$DRIVE" 2>/dev/null)" ]; then
        log "Restoring from Drive"
        mkdir -p "$RUN"
        cp -f "$DRIVE"/best.pt "$DRIVE"/state.json "$DRIVE"/buffer.npz \
              "$DRIVE"/metrics.jsonl "$RUN"/ 2>/dev/null || true
        python3 -c "
import json, os
p = '$RUN/state.json'
print('  resuming from iteration', json.load(open(p))['iteration']) if os.path.exists(p) else print('  no previous state -- starting fresh')
"
    fi

    GUMBEL_FLAG=""
    [ "$GUMBEL" != "0" ] && GUMBEL_FLAG="--gumbel"

    log "Training to iteration $until"
    echo "  games/iter $GAMES   sims $SIMS   concurrency $CONCURRENCY (= NN batch)"
    echo "  gumbel $([ -n "$GUMBEL_FLAG" ] && echo on || echo off)   n_considered $NCONSIDERED"
    echo "  net ${BLOCKS}x${CHANNELS}   steps $STEPS   batch $BATCH"
    echo "  local $RUN"
    echo "  drive $DRIVE  (mirrored every iteration)"
    echo

    python3 -m oog_az.train \
        --run-dir "$RUN" \
        --until "$until" \
        --games "$GAMES" --simulations "$SIMS" --concurrency "$CONCURRENCY" \
        --steps "$STEPS" --batch-size "$BATCH" \
        --blocks "$BLOCKS" --channels "$CHANNELS" \
        --n-considered "$NCONSIDERED" \
        $GUMBEL_FLAG \
        --mirror-dir "$DRIVE"
}

status() {
    cd "$REPO"
    python3 - "$RUN" "$DRIVE" <<'PY'
import glob, json, os, sys
run, drive = sys.argv[1], sys.argv[2]
for label, d in (("local", run), ("drive", drive)):
    st = os.path.join(d, "state.json")
    it = json.load(open(st))["iteration"] if os.path.exists(st) else None
    ck = len(glob.glob(os.path.join(d, "*.pt")))
    print(f"  {label:6} iteration={it}  checkpoints={ck}  {d}")
m = os.path.join(run, "metrics.jsonl")
if os.path.exists(m):
    rows = [json.loads(l) for l in open(m) if l.strip()]
    gates = [r for r in rows if r.get("stage") == "gate"]
    print(f"  gates: {sum(g['promoted'] for g in gates)} promoted of {len(gates)}")
    tr = [r for r in rows if r.get("stage") == "train"]
    if tr:
        print(f"  last loss: policy={tr[-1]['policy_loss']} value={tr[-1]['value_loss']}")
PY
}

case "${1:-}" in
    setup)  setup ;;
    train)  shift; train "$@" ;;
    status) status ;;
    *) echo "usage: $0 {setup|train [until_iteration]|status}"; exit 1 ;;
esac
