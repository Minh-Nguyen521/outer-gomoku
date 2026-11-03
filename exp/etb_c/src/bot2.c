/* bot2: high-performance expectimax aimed at 32768+.
 * - Precomputed per-row snake scores (fast eval via 8 table lookups).
 * - Prob-aware transposition table keyed by (board, depth).
 * - Adaptive depth by empty count.
 */
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

/* ---------- snake weights (P4, proven best in earlier tests) ---------- */
#define P4(n) ((double)(1u << (2*(n))))
static const double SNAKE_W[16] = {
    P4(15), P4(14), P4(13), P4(12),
    P4(8),  P4(9),  P4(10), P4(11),
    P4(7),  P4(6),  P4(5),  P4(4),
    P4(3),  P4(2),  P4(1),  P4(0)
};

static double ROW_TABS[8][65536]; /* 0..3 board rows, 4..7 board cols */

static const double EMPTY_BONUS = 270.0;
static const double MONO_SCALE = 0.001;
static const double MERGE_BONUS = 10.0;

static void init_row_tables(void) {
    for (int i = 0; i < 8; i++) {
        double ws[4];
        if (i < 4) { /* board row i: cells i*4+0..3 left->right */
            ws[0] = SNAKE_W[i*4+0]; ws[1] = SNAKE_W[i*4+1];
            ws[2] = SNAKE_W[i*4+2]; ws[3] = SNAKE_W[i*4+3];
        } else { /* board col i-4: cells rows 0..3 top->bottom */
            ws[0] = SNAKE_W[i-4]; ws[1] = SNAKE_W[(i-4)+4];
            ws[2] = SNAKE_W[(i-4)+8]; ws[3] = SNAKE_W[(i-4)+12];
        }
        for (int r = 0; r < 65536; r++) {
            int e[4] = {(r >> 12) & 0xF, (r >> 8) & 0xF, (r >> 4) & 0xF, r & 0xF};
            double v = 0; int empty = 0;
            for (int j = 0; j < 4; j++) {
                if (!e[j]) { empty++; continue; }
                v += ws[j] * (double)(1u << e[j]);
            }
            for (int j = 0; j < 3; j++)
                if (e[j] && e[j+1])
                    v -= fabs((double)(1u << e[j]) - (double)(1u << e[j+1])) * MONO_SCALE;
            /* merge potential: adjacent equal tiles can combine */
            for (int j = 0; j < 3; j++)
                if (e[j] && e[j] == e[j+1])
                    v += MERGE_BONUS * (double)(1u << e[j]);
            v += empty * EMPTY_BONUS;
            ROW_TABS[i][r] = v;
        }
    }
}

static inline uint16_t get_row(etb_board_t b, int r) {
    return (uint16_t)((b >> ((3 - r) * 16)) & 0xFFFF);
}

static inline double heur_board(etb_board_t b) {
    double s = 0;
    etb_board_t t = etb_transpose(b);
    for (int i = 0; i < 4; i++) {
        s += ROW_TABS[i][get_row(b, i)];
        s += ROW_TABS[4 + i][get_row(t, i)]; /* cols of b = rows of transpose */
    }
    return s;
}

/* ---------- transposition table ---------- */
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

/* ---------- expectimax ---------- */
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
        int empties[16], n = 0;
        for (int c = 0; c < 16; c++) if (cell_empty(m, c)) empties[n++] = c;
        if (!n) { double v = heur_board(m); if (v > best) best = v; continue; }
        double acc = 0;
        for (int i = 0; i < n; i++) {
            acc += 0.9 * expectimax(set_cell(m, empties[i], 1), depth - 1, cprob * 0.9 / n);
            acc += 0.1 * expectimax(set_cell(m, empties[i], 2), depth - 1, cprob * 0.1 / n);
        }
        double val = acc / n;
        if (val > best) best = val;
    }
    if (!any) return heur_board(b) - 1e9;
    tt_put(b, depth, best);
    return best;
}

static int depth_for_empties(int empties) {
    if (empties >= 10) return 2;
    if (empties >= 7) return 3;
    if (empties >= 4) return 4;
    if (empties >= 3) return 5;
    if (empties == 2) return 6;
    if (empties == 1) return 7;
    return 8;
}

static int pick_move(etb_board_t b) {
    int empties = 0;
    for (int c = 0; c < 16; c++) if (cell_empty(b, c)) empties++;
    int depth = depth_for_empties(empties);
    int best_d = -1; double best_v = -1e18;
    for (int d = 0; d < 4; d++) {
        etb_board_t m = etb_move_dir(b, d);
        if (m == b) continue;
        double v = expectimax(m, depth, 1.0);
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
    if (argc < 2) { fprintf(stderr, "usage: bot2 <n_games> [rng_seed]\n"); return 2; }
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
    printf("games=%ld avg_moves=%.0f avg_sum=%.0f\n", n, (double)total_moves / n, total_sum / n);
    printf("tile  rate\n");
    for (int e = 1; e <= 16; e++) printf("%5d  %.2f%%\n", 1 << e, 100.0 * achieved[e] / n);
    return 0;
}
