#!/usr/bin/env bash
# Measure JAX25 on a Colab GPU box, because we cannot measure it here.
#
# WHY. JAX25 took Gomocup silver and is the one strong engine this project has
# never put a number on. `docs/engine-ladder.md` lists it under "engines that
# could not be measured":
#
#     The configuration that won Gomocup silver never returned a move in 600s
#     under Wine. Only the 128-channel net is usable here (~174 visits/sec),
#     which is a *different, weaker* engine.
#
# CORRECTION 2026-09-11: that entry is stale. All three nets DO answer locally
# now (128/256/320 all ~9s of a 10s budget) -- the container was upgraded to
# wine-10.0. Two other things were wrong instead, and both are why a local
# result is still not a measure of JAX:
#
#   1. configs/config.toml mapped Freestyle to the 128-net while every other
#      rule used the 320-net, so every freestyle measurement here was of the
#      small net. The README confirms the 320-net is the correct 15x15 model.
#   2. The shipped default is num_search_threads = 1. JAX's Gomocup rating was
#      earned with config_unlimited.toml, which sets 32.
#
# Even with both fixed (320-net, 8 threads) JAX lost its first games badly to
# VIBEFIVE, losing to plain open fours -- the signature of far too few visits
# rather than weak play. The remaining suspect is the emulation penalty, which
# is NOT uniform across engines: an MCTS engine doing ONNX float inference
# under x86->ARM translation loses far more than a 268KB integer alpha-beta
# engine. AlphaGomoku measured 65 nodes/sec emulated against 136 native.
#
# That is the hypothesis this script exists to test. Colab is native x86-64
# with no Rosetta, so if JAX plays well there and badly here, emulation is the
# answer -- and every Wine-engine row in the ladder needs an asterisk.
#
# That is an artefact of THIS machine, not of JAX. Locally the stack is
# x86 Windows binary -> Wine -> Rosetta -> Apple M2, with no GPU at all. Colab
# removes two layers: native x86-64 (no Rosetta) and a real NVIDIA GPU.
#
# It also matters beyond the ladder: a Botzone bot named "Jax" tops the Gomoku
# ranklist at 2072 and swept our SlowRenju entry 0-4 across both colours, and
# JAX placed top-ten at Gomocup 2026. It is plausibly the same author. The
# engine we most want a number for is the one we have never been able to run.
#
# CPU, NOT GPU -- and that is the competition configuration. Four of the five
# shipped configs set `device = "cpu"`; only config_unlimited.toml uses
# `device = "tensorrt"`, and the README scopes that to the separate *unlimited*
# tournament (CUDA 11.8 + TensorRT 8.6.1.6). So the 320-net on CPU IS what
# earned the Gomocup placing. No CUDA-under-Wine work is needed, and it would
# not help: Colab ships CUDA 12.8 against JAX's required 11.8.
#
# CONFIRMED WORKING on Colab 2026-09-08: the 320-net loads and moves.
#     DEBUG load model: models/dbt-320-...onnx
#     7,7
# The only thing that ever blocked it was the wine version (see below).
#
# `probe` measures which nets are usable before any match is run. That
# ordering is deliberate: three of the false results in engine-ladder.md came
# from giving an engine less wall-clock than it needs for one move.
#
# OPPONENT. SLOWRENJU20, also fetched from Gomocup, so nothing large is
# uploaded. SlowRenju is well-anchored in our ladder (Yixin beat it 15-5,
# AlphaGomoku 17-3), so a JAX result against it slots into the existing table.
# Caveat: that is Gomocup's Windows build, while our ladder rows used a native
# build from source. Same engine and version family, not bit-identical.
#
# Usage (each step a separate Colab cell). Upload tools/match.py first --
# it is ~20KB and this script cannot fetch it, being ours and not public.
#
#   !bash colab_jax.sh setup
#   !bash colab_jax.sh probe                 # which nets are usable?
#   !bash colab_jax.sh match 320 20          # net, games
#
# Set MATCH_PY=/path/to/match.py if it is not beside this script.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/content/jax-test}"
MATCH_PY="${MATCH_PY:-$HERE/match.py}"
DL=http://download.gomocup.com/ai

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
note() { printf '    %s\n' "$*"; }

# Locate an engine executable inside an extracted package.
#
# Do NOT hardcode these. An earlier version assumed the JAX archive contained
# `jax-run`, which is actually a wrapper WE wrote in third_party/jax25 -- the
# Gomocup zip ships `pbrain-Jax.exe`. `wine jax-run` died instantly and the
# manager reported only "BrokenPipeError" on the first START, which says
# nothing about the real cause. Archive layouts also vary: some zips put the
# exe at the top level, others inside a folder.
find_exe() {
    local dir="$1" exe
    exe="$(find "$dir" -maxdepth 3 -iname 'pbrain-*.exe' -o -maxdepth 3 -iname '*.exe' 2>/dev/null |
           grep -viE 'unins|setup|vcredist' | head -n1)"
    [ -n "$exe" ] || die "no .exe found under $dir -- inspect it:  ls -R $dir"
    printf '%s\n' "$exe"
}

# --------------------------------------------------------------------------
setup() {
    log "GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader ||
        note "no GPU visible -- CPU-only, still native x86 which is the main win"

    log "Wine (no Docker here: Colab is already linux/amd64)"
    # MUST be a modern Wine. Ubuntu's repo ships wine 6.0.3 (2021), which does
    # not implement the api-ms-win-core-processtopology-l1-1-0 API set that
    # onnxruntime.dll imports. On that build JAX fails before loading any net:
    #     err:module:import_dll Library api-ms-win-core-processtopology-l1-1-0.dll
    #         (which is needed by onnxruntime.dll) not found
    #     err:module:LdrInitializeThunk ... failed, status c0000135
    # and every net looks "unusable" for a reason that has nothing to do with
    # compute. Our working local image runs wine-10.0; that is the difference.
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unzip curl wget \
        >/dev/null 2>&1 || true

    local wv
    wv="$(wine --version 2>/dev/null | sed -E 's/^wine-([0-9]+).*/\1/')" || wv=0
    if [ -z "$wv" ] || [ "${wv:-0}" -lt 8 ] 2>/dev/null; then
        note "wine is ${wv:-absent} -- installing WineHQ stable (need >= 8)"
        local codename
        codename="$( (lsb_release -cs 2>/dev/null) || echo jammy )"
        dpkg --add-architecture i386
        mkdir -pm755 /etc/apt/keyrings
        wget -qO /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
        wget -qNP /etc/apt/sources.list.d/ \
            "https://dl.winehq.org/wine-builds/ubuntu/dists/$codename/winehq-$codename.sources" ||
            note "no WineHQ sources for '$codename' -- check https://dl.winehq.org/wine-builds/ubuntu/dists/"
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            --install-recommends winehq-stable ||
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq winehq-devel ||
            die "WineHQ install failed. Ubuntu's wine 6.x cannot run JAX -- see the
  note above. Try manually:
      !dpkg --add-architecture i386
      !wget -qO /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
      !wget -qNP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/$codename/winehq-$codename.sources
      !apt-get update && apt-get install -y --install-recommends winehq-stable"
    fi

    wine --version || die "wine did not install"
    wv="$(wine --version | sed -E 's/^wine-([0-9]+).*/\1/')"
    if [ "${wv:-0}" -lt 8 ] 2>/dev/null; then
        die "wine $(wine --version) is too old for onnxruntime.dll (need >= 8).
  JAX will fail with status c0000135 before loading any net."
    fi
    note "wine $(wine --version) OK"

    mkdir -p "$WORK"
    cd "$WORK"

    log "Fetching engines from Gomocup"
    for z in JAX25 SLOWRENJU20; do
        if [ ! -f "$z.zip" ]; then
            curl -sSL --max-time 600 -o "$z.zip" "$DL/$z.zip" ||
                die "download failed: $z"
        fi
        mkdir -p "${z,,}"
        unzip -oq "$z.zip" -d "${z,,}"
        note "$z -> $(du -sh "${z,,}" | cut -f1)"
    done

    # Warm the Wine prefix BEFORE anything is timed. Locally, skipping this
    # made KATAGOMO26 look like 2 visits/sec when the true figure was ~28x
    # higher -- it was measuring prefix creation and model load.
    log "Warming the Wine prefix (so it is not counted as think time)"
    mkdir -p /tmp/xdg
    WINEDEBUG=-all XDG_RUNTIME_DIR=/tmp/xdg wineboot -i >/dev/null 2>&1 || true

    log "Done. Next: bash $0 probe"
}

# --------------------------------------------------------------------------
# Which of the three nets can actually answer a move here, and how fast?
probe() {
    cd "$WORK/jax25" 2>/dev/null || die "run setup first"
    local exe; exe="$(find_exe "$WORK/jax25")"
    note "engine: $(basename "$exe")"
    # Wine version gates everything: 6.x cannot load onnxruntime.dll at all.
    # Print it here so a failed probe is never ambiguous about which problem
    # it is.
    note "wine:   $(wine --version 2>/dev/null || echo MISSING)"
    local wv; wv="$(wine --version 2>/dev/null | sed -E 's/^wine-([0-9]+).*/\1/')"
    if [ -n "$wv" ] && [ "$wv" -lt 8 ] 2>/dev/null; then
        die "wine $wv is too old -- it cannot load onnxruntime.dll, so every net
  will report 'unusable' for a reason unrelated to the nets. Re-run setup
  with the current script to install WineHQ >= 8."
    fi
    local cfg=configs/config.toml
    [ -f "$cfg" ] || die "no $cfg -- package layout differs, inspect $WORK/jax25"
    [ -f "$cfg.orig" ] || cp "$cfg" "$cfg.orig"

    log "Probing nets (one move each, 120s ceiling)"
    printf '    %-8s %-10s %s\n' NET TIME RESULT
    for net in 128 256 320; do
        local model
        model="$(ls models/ | grep -E "^dbt-$net-" | head -n1 || true)"
        if [ -z "$model" ]; then
            printf '    %-8s %-10s %s\n' "$net" "-" "no model file"
            continue
        fi
        # Repoint the config at this net.
        sed -E "s|dbt-[0-9]+-[^\"']*\.onnx|$model|g" "$cfg.orig" > "$cfg"

        # Keep stderr. Discarding it here is what made three earlier runs
        # unreadable: every net reported "unusable" while the real cause was a
        # missing DLL that wine had been printing all along.
        local t0 t1 out err
        err="$(mktemp)"
        t0=$(date +%s)
        out=$(printf 'START 15\nINFO timeout_turn 100000\nBEGIN\n' |
              timeout 120 env WINEDEBUG=-all XDG_RUNTIME_DIR=/tmp/xdg wine "$(basename "$exe")" 2>"$err" |
              grep -aE '^[0-9]+,[0-9]+$' | head -n1 || true)
        t1=$(date +%s)
        if [ -n "$out" ]; then
            printf '    %-8s %-10s %s\n' "$net" "$((t1-t0))s" "move $out  OK"
        else
            local why
            why="$(grep -aoE 'err:module:[^ ]+ [^\n]{0,90}' "$err" | head -n1)"
            [ -n "$why" ] || why="$(grep -aiE 'err:|error|exception|fatal' "$err" | head -n1)"
            [ -n "$why" ] || why="no output on stderr (engine ran but never moved)"
            printf '    %-8s %-10s %s\n' "$net" ">120s" "NO MOVE"
            printf '    %-8s %-10s   why: %s\n' "" "" "${why:0:110}"
        fi
        rm -f "$err"
    done
    cp "$cfg.orig" "$cfg"
    log "Pick the largest net that answered, then: bash $0 match <net> <games>"
}

# --------------------------------------------------------------------------
match() {
    local net="${1:-320}" games="${2:-20}"
    [ -f "$MATCH_PY" ] || die "match.py not found at $MATCH_PY (upload it, or set MATCH_PY=)"
    cd "$WORK/jax25" || die "run setup first"

    local cfg=configs/config.toml model
    model="$(ls models/ | grep -E "^dbt-$net-" | head -n1)" ||
        die "no model for net=$net"
    [ -f "$cfg.orig" ] || cp "$cfg" "$cfg.orig"
    sed -E "s|dbt-[0-9]+-[^\"']*\.onnx|$model|g" "$cfg.orig" > "$cfg"
    log "JAX net=$net  ($model)"

    # Thin wrappers so match.py can spawn each engine. Plain `wine`, not the
    # Docker+Rosetta wrapper used on the Mac -- Colab is already linux/amd64.
    local jexe sexe
    jexe="$(find_exe "$WORK/jax25")"
    sexe="$(find_exe "$WORK/slowrenju20")"
    note "jax:       $jexe"
    note "slowrenju: $sexe"
    # Each wrapper cds to the engine's OWN directory first: these engines
    # resolve configs/, models/ and DLLs relative to the working directory.
    cat > "$WORK/run-jax" <<EOF
#!/usr/bin/env bash
mkdir -p /tmp/xdg
cd "$(dirname "$jexe")" && exec env WINEDEBUG=-all XDG_RUNTIME_DIR=/tmp/xdg wine "$(basename "$jexe")" 2>/dev/null
EOF
    cat > "$WORK/run-slowrenju" <<EOF
#!/usr/bin/env bash
mkdir -p /tmp/xdg
cd "$(dirname "$sexe")" && exec env WINEDEBUG=-all XDG_RUNTIME_DIR=/tmp/xdg wine "$(basename "$sexe")" 2>/dev/null
EOF
    chmod +x "$WORK/run-jax" "$WORK/run-slowrenju"

    log "JAX25(net=$net) vs SLOWRENJU20 -- $games games, paired openings"
    note "Check the 'win types' line FIRST. A lopsided score with 0x five is a"
    note "broken instrument, not a strength result -- that has happened seven"
    note "times in this project. Only then read the score."
    note ""
    note "If net != 320 this is a WEAKENED JAX and must be labelled as such:"
    note "the 320-channel net is what earned the tournament placing."

    # PARALLEL 1 and a short clock, both deliberate.
    #
    # Colab gives 2 vCPUs (the MiniZero build runs at -j2). Two concurrent
    # games means FOUR engine processes on two cores, and under a time-based
    # control that measures CPU contention rather than the engines -- the same
    # trap as engine-ladder.md's "one GPU match at a time", which applies just
    # as much to CPU.
    #
    # 30s/move also made a 20-game match take hours: JAX has
    # visits_limit=100000 and will use whatever budget it is given. TURN_MS
    # defaults to 5000, close to the 3s used for the Yixin ladder rows, so the
    # result is roughly comparable to them.
    # 15s, not 5s. The 320-net was measured at 8.8-10.5s per move; a 5s clock
    # cuts it off mid-search and it forfeits. That produced a fake "JAX 0-20"
    # here once already -- 20 losses of which only 18 were decided by a five,
    # the other two being timeouts. Check the `win types` line before believing
    # any score: wins that are not fives are the instrument, not the engine.
    local turn="${TURN_MS:-15000}"
    note "clock: ${turn}ms/move, parallel 1 (2 vCPUs)"
    python3 "$MATCH_PY" --games "$games" --parallel 1 --paired-openings \
        --rule 0 --timeout-turn "$turn" \
        --engine-a "$WORK/run-jax" --engine-b "$WORK/run-slowrenju"
}

# --------------------------------------------------------------------------
# Why did the engine not answer? `probe` discards stderr so its output stays
# readable, which is wrong when NOTHING works: all three nets failing
# identically means the process is not starting, not that it is slow. The
# 128-net runs locally at ~174 visits/sec on a slower stack, so "too heavy"
# cannot explain a 128 failure on native x86.
diag() {
    cd "$WORK/jax25" 2>/dev/null || die "run setup first"
    local exe; exe="$(find_exe "$WORK/jax25")"

    log "1. wine present?"
    wine --version || note "wine MISSING -- rerun setup"
    command -v wine64 >/dev/null && note "wine64: yes" || note "wine64: no"

    log "2. what kind of binary is it?"
    file "$exe" 2>/dev/null || note "(file(1) not installed)"

    log "3. is the 64-bit prefix initialised?"
    note "WINEPREFIX=${WINEPREFIX:-$HOME/.wine}  WINEARCH=${WINEARCH:-default}"
    ls -d "${WINEPREFIX:-$HOME/.wine}/drive_c" 2>/dev/null ||
        note "no prefix yet -- run: WINEDEBUG=-all wineboot -i"

    log "4. run it with stderr VISIBLE (30s, first 40 lines)"
    note "this is the step that actually tells us the cause"
    printf 'START 15\nINFO timeout_turn 20000\nBEGIN\n' |
        timeout 30 wine "$(basename "$exe")" 2>&1 | head -40 || true

    log "5. do the CUDA/ONNX DLLs exist?"
    ls -la onnxruntime*.dll 2>/dev/null | awk '{print "    "$NF, $5}' ||
        note "no onnxruntime DLLs -- extraction may be incomplete"

    log "Common causes"
    note "* wine32/wine64 missing for this binary's architecture"
    note "* prefix not initialised (wineboot never ran)"
    note "* a DLL the engine loads is absent or refuses under wine"
    note "* the engine needs more than 120s for FIRST init (model load), even"
    note "  though later moves are fast -- rerun probe with a longer ceiling"
}

# --------------------------------------------------------------------------
# How long does JAX actually need per move, and how many visits does it get?
#
# WHY THIS EXISTS. A 20-game match at 5s/move returned JAX 0-20 against
# SlowRenju, with 18 of SlowRenju's wins by a real five and 2 by something
# else (i.e. JAX forfeiting on time). That is not a strength result: it is
# engine-ladder.md lesson 2, "measure throughput before interpreting any
# result -- three fake 0-20s came from giving an engine less wall-clock than
# it needs for one move".
#
# JAX's config sets fast_search_visits_threshold = 400, so hundreds of visits
# is its normal regime. On 2 vCPUs a 320-channel net may get nowhere near that
# in 5s. This measures steady-state move time (NOT the first move, which
# includes model load) so the clock can be chosen from data.
bench() {
    local net="${1:-320}" moves="${2:-6}" turn="${3:-30000}"
    cd "$WORK/jax25" 2>/dev/null || die "run setup first"
    local exe; exe="$(find_exe "$WORK/jax25")"
    local cfg=configs/config.toml model
    model="$(ls models/ | grep -E "^dbt-$net-" | head -n1)" || die "no model for net=$net"
    [ -f "$cfg.orig" ] || cp "$cfg" "$cfg.orig"
    sed -E "s|dbt-[0-9]+-[^\"\']*\.onnx|$model|g" "$cfg.orig" > "$cfg"

    log "Timing net=$net over $moves moves at ${turn}ms budget"
    note "the FIRST move includes model load -- ignore it"
    mkdir -p /tmp/xdg
    {
        echo "START 15"
        echo "INFO timeout_turn $turn"
        echo "BEGIN"
        # Feed plausible replies so the engine keeps moving.
        for i in $(seq 1 $((moves-1))); do echo "TURN $((i)),$((i+2))"; done
        echo "END"
    } | timeout $(( (moves+2) * (turn/1000) + 90 )) \
        env WINEDEBUG=-all XDG_RUNTIME_DIR=/tmp/xdg wine "$(basename "$exe")" 2>/dev/null |
      python3 -c '
import sys, time
t0=time.time(); last=t0; n=0
for line in sys.stdin:
    line=line.strip()
    if line and line[0].isdigit() and "," in line:
        now=time.time(); n+=1
        print(f"    move {n:2d}  {now-last:6.2f}s   {line}")
        last=now
    elif line.upper().startswith("DEBUG") and any(k in line.lower() for k in ("visit","nps","node")):
        print(f"      {line[:100]}")
'
    cp "$cfg.orig" "$cfg"
    log "Pick a clock that gives JAX a few hundred visits, then run match"
}

case "${1:-}" in
    setup) setup ;;
    probe) probe ;;
    bench) shift; bench "$@" ;;
    diag)  diag ;;
    match) shift; match "$@" ;;
    *)     sed -n '1,60p' "$0"; exit 1 ;;
esac
