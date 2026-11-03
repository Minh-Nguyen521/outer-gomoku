#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

make -s
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Endgame scenario: two adjacent 8-tiles (exp 3) inside free zone (cells 14,15),
# target tile = 16 (exp 4).
SEED_HEX=0x0000000000000033
FREE_MASK=0xC000

echo "== gen =="
./etb gen "$SEED_HEX" "$FREE_MASK" "$TMP" 0.1 24

echo "== verify layer files exist and have rates =="
python3 - "$TMP" <<'PYEOF'
import struct, sys, os
outdir=sys.argv[1]
layers=0
while os.path.exists(f"{outdir}/layer_{layers:04d}.bin"):
    data=open(f"{outdir}/layer_{layers:04d}.bin",'rb').read()
    assert data[:4]==b'ETB1', "bad magic"
    n=struct.unpack('<Q',data[4:12])[0]
    assert len(data)==12+8*n, "unexpected pre-solve payload"
    layers+=1
print(f"total layers={layers}")
assert layers>1, "too few layers"
PYEOF

echo "== solve =="
./etb solve "$TMP" 4

echo "== query seed rate =="
./etb query "$TMP" "$SEED_HEX" 4

echo "== verify rates after solve =="
python3 - "$TMP" <<'PYEOF'
import struct, sys, os
outdir=sys.argv[1]
seed=0x0000000000000033
found=False
li=0
while os.path.exists(f"{outdir}/layer_{li:04d}.bin"):
    data=open(f"{outdir}/layer_{li:04d}.bin",'rb').read()
    n=struct.unpack('<Q',data[4:12])[0]
    assert len(data)==12+10*n, "rates missing after solve"
    keys=list(struct.unpack(f'<{n}Q',data[12:12+8*n]))
    if seed in keys:
        r=struct.unpack(f'<{n}H',data[12+8*n:12+10*n])[keys.index(seed)]
        print(f"seed rate = {r}/65535 = {r*100/65535:.2f}%")
        assert r>0, "seed should have nonzero success rate"
        found=True
    li+=1
assert found, "seed not in table"
PYEOF

echo "ALL TESTS PASSED"
