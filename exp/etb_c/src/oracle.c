#include <stdint.h>
#include "oracle.h"

extern const uint32_t ORA256_B[ORA_256_MAX_LAYER];
extern const uint32_t ORA256_L[ORA_256_MAX_LAYER];
extern const uint16_t *const ORA256_SEEDS[ORA_256_MAX_LAYER];
extern const uint8_t *const ORA256_SIGS[ORA_256_MAX_LAYER];
extern const uint16_t *const ORA256_RATES[ORA_256_MAX_LAYER];

extern const uint32_t ORA512_B[ORA_512_MAX_LAYER];
extern const uint32_t ORA512_L[ORA_512_MAX_LAYER];
extern const uint16_t *const ORA512_SEEDS[ORA_512_MAX_LAYER];
extern const uint8_t *const ORA512_SIGS[ORA_512_MAX_LAYER];
extern const uint16_t *const ORA512_RATES[ORA_512_MAX_LAYER];

extern const uint32_t ORA1256_B[ORA_1256_MAX_LAYER];
extern const uint32_t ORA1256_L[ORA_1256_MAX_LAYER];
extern const uint16_t *const ORA1256_SEEDS[ORA_1256_MAX_LAYER];
extern const uint8_t *const ORA1256_SIGS[ORA_1256_MAX_LAYER];
extern const uint16_t *const ORA1256_RATES[ORA_1256_MAX_LAYER];

static const uint32_t ORA_POW9[10] = {1, 9, 81, 729, 6561, 59049, 531441,
                                      4782969, 43046721, 387420489};

static inline uint32_t ora_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x85ebca77U;
  x ^= x >> 13;
  x *= 0xc2b2ae3dU;
  x ^= x >> 16;
  return x;
}

static uint32_t ora_compress(uint64_t tb) {
  static const int offs[10] = {12, 28, 32, 36, 40, 44, 48, 52, 56, 60};
  uint64_t cv = 0;
  for (int i = 0; i < 10; i++)
    cv += ((tb >> offs[i]) & 0xFULL) * ORA_POW9[i];
  return (uint32_t)cv;
}

static uint32_t ora_board_sum(uint64_t tb) {
  uint32_t s = 0;
  for (int i = 0; i < 16; i++) {
    uint64_t v = (tb >> (4 * i)) & 0xF;
    if (v > 0) s += 1u << v;
  }
  return s;
}

static int32_t ora_hash_lookup(uint32_t cb, int layer, int tt) {
  uint32_t B, L;
  const uint16_t *seeds, *rates;
  const uint8_t *sigs;

  if (tt == 512) {
    if (layer < 0 || layer >= ORA_512_MAX_LAYER) return -1;
    B = ORA512_B[layer];
    L = ORA512_L[layer];
    if (!B || !L) return -1;
    seeds = ORA512_SEEDS[layer];
    sigs = ORA512_SIGS[layer];
    rates = ORA512_RATES[layer];
  } else if (tt == 256) {
    if (layer < 0 || layer >= ORA_256_MAX_LAYER) return -1;
    B = ORA256_B[layer];
    L = ORA256_L[layer];
    if (!B || !L) return -1;
    seeds = ORA256_SEEDS[layer];
    sigs = ORA256_SIGS[layer];
    rates = ORA256_RATES[layer];
  } else if (tt == 1256) {
    if (layer < 0 || layer >= ORA_1256_MAX_LAYER) return -1;
    B = ORA1256_B[layer];
    L = ORA1256_L[layer];
    if (!B || !L) return -1;
    seeds = ORA1256_SEEDS[layer];
    sigs = ORA1256_SIGS[layer];
    rates = ORA1256_RATES[layer];
  } else {
    return -1;
  }

  uint32_t bi = ora_mix32(cb) % B;
  uint32_t seed = seeds[bi];
  uint32_t hi = ora_mix32(cb ^ seed) % L;
  uint8_t sig = (uint8_t)((cb + (cb >> 8) + (cb >> 16) + (cb >> 24)) & 0xFFU);
  if (sigs[hi] == sig) return rates[hi];
  return -1;
}

void oracle_query(etb_board_t b, int table_type, double out[4]) {
  for (int d = 0; d < 4; d++) out[d] = -1.0;

  if (table_type != 1256 && (uint64_t)b == 0x012412567fff8fffULL) {
    out[0] = 0.9111; /* their hardcoded special case, dir 1 (left) */
    return;
  }

  for (int d = 0; d < 4; d++) {
    etb_board_t post = etb_move_dir(b, d);
    if (post == b) continue;
    uint64_t tp = (uint64_t)post;

    if (table_type == 1256) {
      uint64_t tr = (uint64_t)etb_transpose(post);
      tp = (tp < tr) ? tp : tr;
    }

    uint32_t sum = ora_board_sum(tp) % 32768u;
    uint32_t init = (table_type == 1256) ? 18 : 8;
    int layer = (int)((sum - init) / 2);

    uint32_t cb = ora_compress(tp);
    if (!cb) continue;
    int32_t r = ora_hash_lookup(cb, layer, table_type);
    if (r < 0) continue;
    double lo, hi;
    if (table_type == 512) { lo = 0.40; hi = 0.96; }
    else if (table_type == 256) { lo = 0.75; hi = 0.99999; }
    else { lo = 0.05; hi = 0.28; }
    out[d] = ((double)r / 65535.0) * (hi - lo) + lo;
  }
}
