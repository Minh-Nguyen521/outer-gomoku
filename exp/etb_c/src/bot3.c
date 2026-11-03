/* bot3: nnn2048-style tuple network heuristic + expectimax
 * Uses 8 tuple patterns (4 horizontal + 4 vertical), each evaluating adjacent pairs.
 * Weights are hand-tuned based on nnn2048 paper/approach.
 */
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* ---------- Tuple Network Weights (nnn2048 style) ----------
 * Each tuple evaluates adjacent cell pairs. Weight[tile_a][tile_b] gives score.
 * Empty tile (0) gets bonus when paired with non-empty.
 * Patterns: 4 horizontal rows (each row has 3 adjacent pairs) + 4 vertical cols
 */

/* Base weight matrix for a pair of tiles (a,b). 
 * Higher weights encourage placing high-value tiles together in preferred orientation.
 * Tuned to favor snake-like monotonic ordering. */
static double TUPLE_W[16][16];

static const int H_PAIRS[4][3][2] = { /* 4 rows, 3 pairs per row */
    {{0,1}, {1,2}, {2,3}},
    {{4,5}, {5,6}, {6,7}},
    {{8,9}, {9,10}, {10,11}},
    {{12,13}, {13,14}, {14,15}}
};

static const int V_PAIRS[4][3][2] = { /* 4 cols, 3 pairs per col */
    {{0,4}, {4,8}, {8,12}},
    {{1,5}, {5,9}, {9,13}},
    {{2,6}, {6,10}, {10,14}},
    {{3,7}, {7,11}, {11,15}}
};

static void init_tuple_weights(void) {
    /* Initialize with snake-favoring weights:
     * - High values for (high, high) pairs in decreasing order
     * - Medium for (high, empty) encouraging filling
     * - Low/negative for disorderly pairs */
    for (int a = 0; a < 16; a++) {
        for (int b = 0; b < 16; b++) {
            double wa = a ? pow(2.0, a) : 0;
            double wb = b ? pow(2.0, b) : 0;
            double empty_bonus = (a == 0 || b == 0) ? 100.0 : 0;
            
            /* Prefer decreasing order (a > b) for monotonicity */
            double order_bonus = 0;
            if (a > b && a > 0 && b > 0) order_bonus = 50.0;
            else if (a < b && a > 0 && b > 0) order_bonus = -30.0;
            
            /* Encourage merging (equal tiles) */
            double merge_bonus = (a == b && a > 0) ? 200.0 : 0;
            
            TUPLE_W[a][b] = 0.5 * (wa + wb) + empty_bonus + order_bonus + merge_bonus;
        }
    }
}

static inline double eval_tuple_pattern(etb_board_t b, const int pairs[][2], int npairs) {
    double score = 0;
    for (int i = 0; i < npairs; i++) {
        int c1 = pairs[i][0], c2 = pairs[i][1];
        int v1 = cell_val(b, c1);
        int v2 = cell_val(b, c2);
        score += TUPLE_W[v1][v2];
    }
    return score;
}

static inline double heur_board(etb_board_t b) {
    double s = 0;
    /* All 4 horizontal rows */
    for (int r = 0; r < 4; r++) {
        s += eval_tuple_pattern(b, H_PAIRS[r], 3);
    }
    /* All 4 vertical columns */
    for (int c = 0; c < 4; c++) {
        s += eval_tuple_pattern(b, V_PAIRS[c], 3);
    }
    return s;
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
    if (empties >= 8) return 2;
    if (empties >= 5) return 3;
    if (empties >= 3) return 4;
    if (empties >= 2) return 5;
    if (empties == 1) return 6;
    return 7;
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
    if (argc < 2) { fprintf(stderr, "usage: bot3 <n_games> [rng_seed]\n"); return 2; }
    long n = atol(argv[1]);
    if (argc > 2) rr_state = strtoull(argv[2], NULL, 0) | 1;

    etb_init_luts();
    g_tt = calloc(TT_SIZE, sizeof(tt_t));
    init_tuple_weights();

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
