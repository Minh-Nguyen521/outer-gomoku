#!/usr/bin/env bash
# Build and stage rapfi as an external opponent for tools/match.py.
#
# Why: every strength number in docs/search-comparison.md comes from self-play
# between engines that share one eval.cpp, one move generator and one search
# core. That can rank our engines against each other; it cannot say whether
# any of them is good. rapfi is the strongest free Gomoku engine and is
# already this project's named reference (PLAN.md), so it is the natural
# absolute anchor.
#
# Licensing: rapfi is GPL-3. This script builds it and runs it as a separate
# opponent *process*; that imposes nothing on this project's own code. Only
# linking or shipping rapfi's code would. Its network weights are CC0.
#
# Rules note: rapfi plays freestyle/standard/renju, not Outer-Open. That is
# fine -- Outer-Open differs from freestyle *only* in Black's first move, and
# match.py plays that move itself from a fixed ring opening. From move 2 on,
# the two rulesets are identical. match.py therefore sends `INFO rule 0`
# (freestyle); do not change that without reading its --rule help.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$REPO/third_party"
BIN="$TP/rapfi-bin"

mkdir -p "$TP"

if [ ! -d "$TP/rapfi" ]; then
  echo "==> cloning rapfi"
  git clone --depth 1 https://github.com/dhbloo/rapfi.git "$TP/rapfi"
fi

if [ ! -d "$TP/rapfi-networks" ]; then
  echo "==> cloning rapfi-networks (~76 MB of weights)"
  git clone --depth 1 https://github.com/dhbloo/rapfi-networks.git "$TP/rapfi-networks"
fi

# Pick SIMD flags for the host architecture.
ARCH="$(uname -m)"
case "$ARCH" in
  arm64|aarch64) SIMD=(-DUSE_NEON=ON -DUSE_NEON_DOTPROD=OFF) ;;
  x86_64)        SIMD=(-DUSE_SSE=ON -DUSE_AVX2=ON -DUSE_AVX512=OFF -DUSE_BMI2=OFF -DUSE_VNNI=OFF) ;;
  *) echo "unknown arch $ARCH; building without SIMD"; SIMD=() ;;
esac

BUILD="$TP/rapfi/Rapfi/build/$ARCH"
echo "==> configuring rapfi for $ARCH"
mkdir -p "$BUILD"
cmake -S "$TP/rapfi/Rapfi" -B "$BUILD" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release "${SIMD[@]}" >/dev/null

echo "==> building rapfi"
cmake --build "$BUILD" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" >/dev/null

# rapfi resolves config.toml and its weights relative to the executable, so
# everything has to sit in one directory next to the binary.
echo "==> staging into $BIN"
mkdir -p "$BIN"
cp "$BUILD/pbrain-rapfi" "$BIN/"
cp "$TP/rapfi-networks/config-example/config.toml" "$BIN/"
cp "$TP/rapfi-networks/classical/model210901.bin" "$BIN/"
cp "$TP/rapfi-networks"/mix9svq/*.lz4 "$BIN/"

echo "==> smoke test"
printf 'ABOUT\nSTART 15\nINFO rule 0\nINFO timeout_turn 1000\nBOARD\n7,7,1\n7,8,2\nDONE\nEND\n' \
  | "$BIN/pbrain-rapfi" 2>&1 | grep -E '^(name=|[0-9]+,[0-9]+$)' || {
    echo "smoke test FAILED" >&2; exit 1; }

cat <<EOF

rapfi is ready at: $BIN/pbrain-rapfi

Run a match (freestyle rule 0 is the default and is what you want):
  tools/match.py --games 100 --parallel 4 --timeout-turn 1000 \\
      --engine-b third_party/rapfi-bin/pbrain-rapfi

Note: rapfi is far stronger than this engine is likely to be. Expect a heavy
loss on the first run -- that is the point of an absolute anchor.
EOF
