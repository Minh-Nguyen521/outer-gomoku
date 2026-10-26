#!/usr/bin/env bash
# Stage Embryo as an external opponent for tools/match.py on Apple Silicon.
#
# Embryo (Mira Fontan, CZ) is a Stockfish-derived Gomoku engine -- its own
# ABOUT string credits "generic search: T. Romstad, M. Costalba, J. Kiiski,
# G. Linscott". It is a *pure heuristic* engine: no neural network. That makes
# it the comparison we actually want, since rapfi's NNUE makes it both too
# strong and a different class of opponent.
#
# Only Windows and Linux x86-64 binaries are published, and this is arm64
# macOS, so it runs in a linux/amd64 container. OrbStack backs that with
# Rosetta rather than QEMU, which measured ~2.19M nodes/sec -- about 3x our
# own engine's *native* rate -- so the emulation is not crippling it and a
# wall-clock match remains meaningful. Verify with --bench if in doubt.
#
# CAVEAT (read before trusting results): the only Linux build is the
# **Standard**-rule variant; the Freestyle build is Windows-only. Standard
# does not count an overline (6+) as a win, freestyle does, and match.py
# adjudicates freestyle. Overlines are rare, but on the rare board where one
# occurs the adjudication favours us. Treat close results with that in mind.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$REPO/third_party/embryo"
IMAGE="debian:stable-slim"
URL="https://github.com/Hexik/Embryo_engine/raw/HEAD/Gomoku/Linux/pbrain-embryo-1.1.1-04d52e8e-s.bz2"

mkdir -p "$DIR"

if [ ! -x "$DIR/pbrain-embryo" ]; then
  echo "==> downloading Embryo (Linux x86-64)"
  curl -sL -o "$DIR/embryo.bz2" "$URL"
  bunzip2 -kf "$DIR/embryo.bz2"
  mv -f "$DIR/embryo" "$DIR/pbrain-embryo"
  chmod +x "$DIR/pbrain-embryo"
fi

if ! docker ps >/dev/null 2>&1; then
  echo "docker daemon not reachable; start OrbStack (orb start) first" >&2
  exit 1
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull --platform linux/amd64 "$IMAGE"

# Wrapper that match.py can spawn like any local engine. -i keeps stdin open
# so the Gomocup dialogue works; the container is created once per game, not
# once per move, so startup cost is amortised.
cat > "$DIR/embryo-run" <<'WRAP'
#!/usr/bin/env bash
exec docker run --rm -i --platform linux/amd64 \
    -v "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)":/w -w /w \
    debian:stable-slim ./pbrain-embryo "$@"
WRAP
chmod +x "$DIR/embryo-run"

if [ "${1:-}" = "--bench" ]; then
  echo "==> throughput check (balanced midgame, 5s turn)"
  python3 - "$DIR/probe_bench.txt" <<'PY'
import sys
lines=['START 15','INFO max_memory 83886080','INFO game_type 1','INFO rule 1',
       'INFO timeout_match 120000','INFO timeout_turn 5000','INFO time_left 120000','BOARD']
for r,c in [(0,7),(7,8),(8,8)]: lines.append(f'{c},{r},1')
for r,c in [(7,7),(6,7),(5,6)]: lines.append(f'{c},{r},2')
lines.append('DONE')
open(sys.argv[1],'w').write('\n'.join(lines)+'\n')
PY
  docker run --rm --platform linux/amd64 -v "$DIR:/w" -w /w "$IMAGE" \
      sh -c 'timeout 40 ./pbrain-embryo < probe_bench.txt 2>&1' | grep -E 'n/ms' | tail -1
fi

echo
echo "Embryo ready: $DIR/embryo-run"
echo
echo "Run a match (Embryo needs a match clock -- match.py sends one):"
echo "  tools/match.py --games 30 --parallel 1 --timeout-turn 1000 \\"
echo "      --engine-b third_party/embryo/embryo-run"
