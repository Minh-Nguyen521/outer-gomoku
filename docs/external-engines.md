# External engines: where each one came from

Provenance for everything under `third_party/`, plus the Botzone opponents.
This is the "where did this binary come from and what may we do with it" file;
for **how strong** each engine is, see [engine-ladder.md](engine-ladder.md).

Written because half of these are opaque Windows binaries with no version
string, and because the licensing differs enough to matter: some of these we
may redistribute, some we may not, and one of them we already submitted to a
public server.

---

## 1. Built from source (git checkouts)

Each directory is a git clone; the commit is what the measurements in
`engine-ladder.md` were taken against.

| Directory | Upstream | Commit | Licence |
|---|---|---|---|
| `SlowRenju` | github.com/wind23/SlowRenju | `41007cf` | **GPL-3** |
| `rapfi` | github.com/dhbloo/rapfi | `3c94c2a` | **GPL-3** |
| `Rapfi-gomocup` | github.com/dhbloo/Rapfi-gomocup | `e2379d9` | no licence file |
| `rapfi-networks` | github.com/dhbloo/rapfi-networks | `e32ad77` | weights repo |
| `AlphaGomoku` | github.com/MaciejKozarzewski/AlphaGomoku | `1f81840` | **GPL-3** |
| `KataGomo` | github.com/hzyhhzy/KataGomo | `07bff40` | mixed — see its `LICENSE` |
| `minizero` | github.com/rlglab/minizero | `394b2e4` | Apache-2.0 |
| `Chis` | github.com/ChisBread/Chis | `ea34c4c` | MPL-2.0 |
| `PentaZen` | github.com/schibir/PentaZen | `5eedcde` | no licence file |
| `PentaZen-Qt` | github.com/alien-wxr/PentaZen-Qt | `7d0959c` | no licence file |
| `Piskvork` | github.com/plastovicka/Piskvork | `2c32965` | no licence file |

`KataGomo` is a KataGo fork; its `LICENSE` opens with *"The code in this
repository currently relies on several other libraries or parts of
libraries"*, so it is not a single blanket licence — read it before reusing
anything from it.

**Local modifications.** Three of these do not build or play correctly as
shipped on POSIX, and carry fixes in-tree. Do not update these checkouts
without re-applying them:

- `SlowRenju` — `compat/Windows.h` (POSIX shims; MSVC `clock()` returns
  wall-clock ms, POSIX returns CPU µs, so every time budget expired ~1000x
  early without the override) and a missing `moven` reset in the `BOARD`
  handler in `Common/main.cpp`.
- `Chis` — two `BOARD` bugs in `Chis2.0/gomocup.cpp`.
- `PentaZen` — `src/search.cpp`, telemetry printed outside the output lock and
  was parsed as moves.

---

## 2. Binary-only, from Gomocup

The original archives are kept in **`third_party/gomocup_dl/`** so the
provenance is not lost when a directory is re-extracted:

```
EGOZERO25.zip  JAX25.zip  KATAGOMO26.zip
SKYZERO.zip    STARPOINT.zip  VIBEFIVE.zip
BARBAKAN21.zip GORO05.zip  STARDUST.zip  WULIN.zip  WINE.zip  XOXO16.zip
```

Source: the Gomocup AI download page, <https://gomocup.org/download-gomoku-ai/>.
These are tournament entrants — Win32/64 `pbrain-*.exe` with their weights and
DLLs, exactly as the organisers publish them.

| Directory | Executable | Size | Notes |
|---|---|---|---|
| `jax25` | `jax-run`, ONNX + CUDA/TensorRT DLLs | 439M | neural |
| `starpoint` | `pbrain-starpoint_freestyle-15.exe` | 101M | `latest.onnx` + `pattern_table.bin` |
| `katagomo26` | ONNX nets + `.cfg` | 53M | `freestyle15x`, `caro15x`, `freestyle20x` |
| `kg26_fast` | same, tuned cfg | 53M | faster settings variant |
| `egozero25` | `pbrain-ego-zero-v1.exe` | 29M | ONNX runtime |
| `skyzero` | `pbrain-SkyZero.exe` | 16M | Gumbel AlphaZero; ships `LICENSE_KataGo.txt` |
| `vibefive` | `pbrain-vibefive.exe` | 268K | **no neural net** — pure alpha-beta |
| `barbakan` | `pbrain-barbakan15x15_avx2.exe` + 3.7M model | 12M | Gomocup rank 5 (2442); added 2026-09-11, untested |
| `stardust` | `pbrain-stardust-*.exe` + `models/` | 3.2M | added 2026-09-11, runs, untested |
| `wulin` | `pbrain-wulin.exe` | 200K | added 2026-09-11; starts but answers nothing |
| ~~`xoxo`~~, ~~`wine`~~ | 32-bit PE32 | — | **unusable**: our Wine container is 64-bit only |
| ~~`goro`~~ | `Goro2006.exe` | — | **unusable**: no `pbrain-` binary, not Gomocup protocol |

`vibefive` is worth noting: 268 KB, no weights, and it still beat our
rapfi anchor 17-3.

**Licensing: unknown.** None of these ship a licence. Gomocup requires
entrants to grant the organisers redistribution rights, which is why the
download page exists — it says nothing about what *we* may do. Treat them as
**test-only**: fine to run locally as opponents, not to redistribute, not to
vendor into anything we publish, and not submittable to Botzone (which needs
source anyway).

### Running them on macOS

They are Windows x86 binaries. `third_party/wine-engine` runs one as a
Gomocup-protocol opponent for `tools/match.py`:

```sh
third_party/wine-engine starpoint pbrain-starpoint_freestyle-15.exe
```

It shells out to `docker run --platform linux/amd64` + Wine (OrbStack backs
that with Rosetta). Two things in it are load-bearing and were each worth real
games:

- `wineboot -i` warms the prefix **before** the manager starts timing.
  Otherwise the first call pays prefix creation plus model load and the engine
  reads as catastrophically slow — an early probe suggested 2 visits/sec for
  KATAGOMO26 when the true figure was ~28x higher.
- stderr is discarded. These engines emit Wine/DirectX noise like
  `Failed to create DXGI factory.`, and a stray line gets parsed as a move and
  desynchronises the protocol.

Measured overhead of the two translation layers is small: KATAGOMO26 ~56
visits/sec under Wine vs our natively built KataGomo's ~48; JAX25 ~174/sec.

---

## 3. Prebuilt binaries and weights (not from Gomocup)

| Directory | Contents | Size |
|---|---|---|
| `kgweights` | `katagomo25.7z`, `tworule_20x_b18.bin.gz`, `zhizi.zip` | **2.2G** |
| `ag584` | AlphaGomoku 5.8.4 build: `pbrain-AlphaGomoku` + `_cuda`/`_opencl`, `networks/`, `ag.zip` | 85M |
| `embryo` | `pbrain-embryo`, `embryo-s.bz2` | 80M |
| `rapfi-bin` | `pbrain-rapfi` + `mix9svq*.bin.lz4`, `model210901.bin` | 40M |
| `rapfi-4t` | same binary, `config.toml` set to 4 threads | 40M |
| `rapfi-classical` | `pbrain-rapfi` + `model210901.bin` only (no NNUE) | 2.1M |
| `rapfi-d2` … `rapfi-d10` | fixed-depth configs, share the classical weights | 2.1M each |

`rapfi-bin` is the **ladder anchor**. Its `config.toml` pins
`default_thread_num = 1`, which is why every "vs rapfi" figure in
`engine-ladder.md` is somewhat inflated — see caveat 1 there.

`rapfi-d2`…`d10` exist to give a *graded* opponent ladder rather than one
binary pass/fail anchor.

---

## 4. Botzone opponents

Not files — bots hosted on <https://botzone.org.cn>, playable via a game
table. Recorded here because the IDs are tedious to recover (ranklist →
per-row **Copy ID** button) and are the only way to seat a specific opponent
through the "Enter BotID" control.

| Bot | Owner | Rating | Bot ID |
|---|---|---|---|
| Jax | Jax | 2072.16 | `654a0176ec1ab1389703ab4b` |
| Jax | osayes | 2054.39 | `64a107cfa24a147da2acc6e5` |
| Kissin_Flash | loker404 | 2033.35 | `68ece6ebdcee294b521db7eb` |
| rapfi | rapfi | 2018.28 | *(seated via Find Matching Bots)* |
| Sincerely | Sincerely | 2002.58 | `65bb33c313e80a09c4e338a3` |
| **SlowRenju_GPL3 v1** | khongcoten002 | 1000.00 (unrated) | `6a9e2b882418e3081fb3559a` |

Ratings are as of 2026-09-07 and are Botzone Elo from their automatic daily
ranking matches — not comparable to Gomocup Elo.

The top two are described by their authors as AlphaZero-like / "MCTS, NN,
VCF"; rapfi's own entry is `ab+nnue`. That the ladder's top tier is neural,
with rapfi 4th, is consistent with what we measured locally.

---

## 5. What we submitted

**`SlowRenju_GPL3`** on Botzone is our own build of `third_party/SlowRenju`,
amalgamated into one `.cpp` by `tools/make_botzone_slowrenju.py`.

Chosen because it is the only engine here that is simultaneously (a) strong —
the best classical engine we measured, (b) **source**, which Botzone requires,
and (c) weight-free, so it fits in a single uploaded file.

GPL-3 compliance: we distributed source, upstream attribution is preserved in
the generated file's banner, and the Botzone entry is flagged 开源 (open
source) so the code is publicly readable. That satisfies the licence.

Every stronger engine we have is disqualified from submission for one of two
reasons: binary-only (VIBEFIVE, STARPOINT, SKYZERO, JAX25, EGOZERO25,
AlphaGomoku's release build), or needs a weight file too large to inline
(KataGomo, rapfi). Note that the weight-size objection is **not** absolute —
Botzone gives each user 256 MB of storage at `./data` with ZIP upload, which
fits rapfi's 38 MB and KataGomo's 93 MB. The real blockers there are the
single-file compile and the absence of a GPU.
