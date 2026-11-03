/* bot5: bot4 + ported nnn2048 EGTB oracle for late-game moves */
#include "types.h"
#include "oracle.h"
#include "l3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static uint64_t rr_state = 1;
static uint64_t rr(void) {
    rr_state ^= rr_state << 13; rr_state ^= rr_state >> 7; rr_state ^= rr_state << 17;
    return rr_state;
}
static double rr01(void) { return (rr() >> 11) * (1.0 / 9007199254740992.0); }

static inline int cell_val(etb_board_t b, int c) { return (int)((b >> ((15 - c) * 4)) & 0xF); }
static inline etb_board_t set_cell(etb_board_t b, int c, int e) {
    int sh = (15 - c) * 4;
    return (b & ~(0xFULL << sh)) | ((etb_board_t)e << sh);
}
static inline int cell_empty(etb_board_t b, int c) { return cell_val(b, c) == 0; }

/* Compute points gained by a move (sum of newly created tiles from merges) */
static int move_merge_points(etb_board_t before, etb_board_t after) {
    int pts = 0;
    for (int c = 0; c < 16; c++) {
        int va = cell_val(after, c), vb = cell_val(before, c);
        if (va > vb && va > 0) pts += 1 << va; /* merged tile value */
    }
    return pts;
}

/* nnn2048 process_score: convert raw merge score into eval-space reward */
static int imax2(int a, int b){return a>b?a:b;}
static int32_t process_score(int32_t s) {
    if (s < 200) return imax2(0, (s >> 2) - 10);
    if (s < 500) return (s >> 1) - 12;
    if (s < 1000) return (s >> 1) + 144;
    return s < 2000 ? s + 800 : 4000;
}

/* ---------- faithful port of nnn2048 line evaluation ---------- */
typedef struct { int16_t d1, d2; } score_pair_t;
static score_pair_t DIFFS_MERGED[65536];

static const int32_t TILE_WEIGHT_MAP[16] = {0, 2, 4, 8, 16, 32, 64, 128,
                                           248, 388, 488, 518, 519, 519, 519, 520};

static int imax(int a, int b){return a>b?a:b;}
static int imin(int a, int b){return a<b?a:b;}

static int32_t diffs_evaluation_func(const int32_t lm[4]) {
  int32_t score_dpdf = lm[0];
  for (int x = 0; x < 3; ++x) {
    if (lm[x] < lm[x + 1]) {
      if (lm[x] > 400) {
        score_dpdf += (lm[x] << 1) + (lm[x + 1] - lm[x]) * x;
      } else if (lm[x] > 300 && x == 1 && lm[0] > lm[1]) {
        score_dpdf += (lm[x] << 1);
      } else {
        score_dpdf -= (lm[x + 1] - lm[x]) << 3;
        score_dpdf -= lm[x + 1] * 3;
        if (x < 2 && lm[x + 2] < lm[x + 1] && lm[x + 1] > 30)
          score_dpdf -= imax(80, lm[x + 1]);
      }
    } else if (x < 2) {
      score_dpdf += lm[x + 1] + lm[x];
    } else {
      score_dpdf += (int32_t)((lm[x + 1] + lm[x]) * 0.5);
    }
  }
  if (lm[0] > 400 && lm[1] > 300 && lm[2] > 200 && lm[2] > lm[3] && lm[3] < 300)
    score_dpdf += lm[3] >> 2;

  int32_t score_t;
  int32_t min_03 = imin(lm[0], lm[3]);
  if (min_03 < 32) {
    score_t = -16384;
  } else if ((lm[0] < lm[1] && lm[0] < 400) || (lm[3] < lm[2] && lm[3] < 400)) {
    score_t = -(imax(lm[1], lm[2]) * 10);
  } else {
    score_t = (int32_t)((lm[0] * 1.8 + lm[3] * 1.8) +
                        imax(lm[1], lm[2]) * 1.5 +
                        imin(160, imin(lm[1], lm[2])) * 2.5);
    if (imin(lm[1], lm[2]) < 8)
      score_t -= 60;
  }

  int zero_count = 0;
  for (int k = 0; k < 4; ++k) if (lm[k] == 0) zero_count++;
  int32_t sum_123 = lm[1] + lm[2] + lm[3];
  int32_t penalty = 0;
  if (lm[0] > 100 && ((zero_count > 1 && sum_123 < 32) || sum_123 < 12))
    penalty = 4;

  return imax(score_dpdf, score_t) / 4 - penalty;
}

static void init_row_tables(void) {
  for (int i = 0; i < 65536; ++i) {
    int32_t lm[4], lmr[4];
    for (int j = 0; j < 4; ++j) {
      int32_t w = TILE_WEIGHT_MAP[(i >> (j * 4)) & 0xF];
      lm[j] = w;
      lmr[3 - j] = w;
    }
    DIFFS_MERGED[i].d1 = (int16_t)diffs_evaluation_func(lm);
    DIFFS_MERGED[i].d2 = (int16_t)diffs_evaluation_func(lmr);
  }
}

static inline uint16_t get_row(etb_board_t b, int r) {
  return (uint16_t)((b >> ((3 - r) * 16)) & 0xFFFF);
}

static inline double heur_board(etb_board_t b) {
  int32_t sx1 = 0, sx2 = 0, sy1 = 0, sy2 = 0;
  etb_board_t t = etb_transpose(b);
  for (int i = 0; i < 4; ++i) {
    uint16_t rx = get_row(b, i), ry = get_row(t, i);
    sx1 += DIFFS_MERGED[rx].d1; sx2 += DIFFS_MERGED[rx].d2;
    sy1 += DIFFS_MERGED[ry].d1; sy2 += DIFFS_MERGED[ry].d2;
  }
  return (double)(imax(sx1, sx2) + imax(sy1, sy2));
}

/* ---------- Transposition Table ---------- */
#define TT_BITS 22
#define TT_SIZE (1u << TT_BITS)
typedef struct {
    etb_board_t key;
    double val;
    int8_t depth;
    uint8_t valid;
} tt_t;
static tt_t *g_tt;

static inline uint64_t mix(etb_board_t b) {
    uint64_t h = b * 0x2545F4914F6CDD1DULL;
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32;
    return h;
}

static inline tt_t *tt_get(etb_board_t b, int depth) {
    tt_t *e = &g_tt[mix(b) & (TT_SIZE - 1)];
    if (e->valid && e->key == b && e->depth >= depth) return e;
    return NULL;
}
static inline void tt_put(etb_board_t b, int depth, double val) {
    tt_t *e = &g_tt[mix(b) & (TT_SIZE - 1)];
    e->key = b; e->val = val; e->depth = (int8_t)depth; e->valid = 1;
}

/* ---------- Expectimax ---------- */
static double expectimax(etb_board_t b, int depth, double cprob) {
    if (depth == 0 || cprob < 1e-4) return heur_board(b);
    tt_t *hit = tt_get(b, depth);
    if (hit) return hit->val;

    double best = -1e18;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        etb_board_t m = etb_move_dir(b, d);
        if (m == b) continue;
        any = 1;
        int merged = process_score(move_merge_points(b, m));
        int empties[16], n = 0;
        for (int c = 0; c < 16; c++) if (cell_empty(m, c)) empties[n++] = c;
        if (!n) { double v = heur_board(m) + merged; if (v > best) best = v; continue; }
        double acc = 0;
        for (int i = 0; i < n; i++) {
            acc += 0.9 * expectimax(set_cell(m, empties[i], 1), depth - 1, cprob * 0.9 / n);
            acc += 0.1 * expectimax(set_cell(m, empties[i], 2), depth - 1, cprob * 0.1 / n);
        }
        double val = merged + acc / n;
        if (val > best) best = val;
    }
    if (!any) return heur_board(b) - 1e9;
    tt_put(b, depth, best);
    return best;
}

static int depth_for_empties(int empties) {
    if (empties >= 8) return 2;
    if (empties >= 5) return 3;
    if (empties >= 3) return 4;
    if (empties >= 2) return 5;
    if (empties == 1) return 6;
    return 7;
}

static int max_tile_exp(etb_board_t b);
static long g_ora_hits = 0, g_ora_queries = 0;

static int pick_move(etb_board_t b) {
    int empties = 0;
    for (int c = 0; c < 16; c++) if (cell_empty(b, c)) empties++;
    int depth = depth_for_empties(empties);

    /* EGTB oracle first: full nnn2048 pipeline (gating + sym + map) */
    g_ora_queries++;
    int ora = l3_pick_move(b);
    if (ora >= 0) {
        g_ora_hits++;
        return ora;
    }

    int best_d = -1; double best_v = -1e18;
    for (int d = 0; d < 4; d++) {
        etb_board_t m = etb_move_dir(b, d);
        if (m == b) continue;
        double v = process_score(move_merge_points(b, m)) + expectimax(m, depth, 1.0);
        if (v > best_v) { best_v = v; best_d = d; }
    }
    return best_d;
}

static int max_tile_exp(etb_board_t b) {
    int mx = 0;
    for (int c = 0; c < 16; c++) { int e = cell_val(b, c); if (e > mx) mx = e; }
    return mx;
}

static etb_board_t random_spawn(etb_board_t b) {
    int empties[16], n = 0;
    for (int c = 0; c < 16; c++) if (cell_empty(b, c)) empties[n++] = c;
    if (!n) return b;
    return set_cell(b, empties[rr() % n], rr01() < 0.9 ? 1 : 2);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bot4 <n_games> [rng_seed]\n"); return 2; }
    long n = atol(argv[1]);
    if (argc > 2) rr_state = strtoull(argv[2], NULL, 0) | 1;

    etb_init_luts();
    g_tt = calloc(TT_SIZE, sizeof(tt_t));
    init_row_tables();

    long achieved[17] = {0};
    double total_sum = 0; long total_moves = 0;
    for (long g = 0; g < n; g++) {
        memset(g_tt, 0, TT_SIZE * sizeof(tt_t));
        etb_board_t b = 0; b = random_spawn(b); b = random_spawn(b);
        long moves = 0;
        for (;;) {
            int d = pick_move(b);
            if (d < 0) break;
            etb_board_t m = etb_move_dir(b, d);
            if (m == b) break;
            b = m; b = random_spawn(b); moves++;
            if (moves > 50000) break;
        }
        total_moves += moves;
        int mx = max_tile_exp(b);
        for (int e = 0; e <= mx && e < 17; e++) achieved[e]++;
        for (int c = 0; c < 16; c++) { int e = cell_val(b, c); if (e) total_sum += pow(2.0, e); }
    }
    printf("games=%ld avg_moves=%.0f avg_sum=%.0f ora_hit=%.1f%%\n", n,
           (double)total_moves / n, total_sum / n,
           g_ora_queries ? 100.0 * g_ora_hits / g_ora_queries : 0.0);
    {
        extern long l3_dbg_calls, l3_dbg_threshold, l3_dbg_gate_fail;
        extern long l3_dbg_gate_pass, l3_dbg_query_fail, l3_dbg_hit;
        printf("l3: calls=%ld no_thr=%ld thr=%ld gate_pass=%ld q_fail=%ld hit=%ld\n",
               l3_dbg_calls + l3_dbg_threshold, l3_dbg_calls, l3_dbg_threshold,
               l3_dbg_gate_pass, l3_dbg_query_fail, l3_dbg_hit);
    }
    printf("tile  rate\n");
    for (int e = 1; e <= 16; e++) printf("%5d  %.2f%%\n", 1 << e, 100.0 * achieved[e] / n);
    return 0;
}
