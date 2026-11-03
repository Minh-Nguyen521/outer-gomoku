/* bot with iterative deepening + transposition table (simple open-addressing hash) */
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_max_depth = 3;
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
static etb_board_t do_move(etb_board_t b, int d) { return etb_move_dir(b, d); }

/* snake weight P4 */
#define P4(n) ((double)(1u << (2*(n))))
static const double W[16] = {
    P4(15), P4(14), P4(13), P4(12),
    P4(8),  P4(9),  P4(10), P4(11),
    P4(7),  P4(6),  P4(5),  P4(4),
    P4(3),  P4(2),  P4(1),  P4(0)
};

static double evaluate(etb_board_t b) {
    double pos = 0; int empt = 0;
    for (int c = 0; c < 16; c++) {
        int e = cell_val(b, c);
        if (!e) { empt++; continue; }
        pos += W[c] * (double)(1u << e);
    }
    double mono = 0;
    for (int r = 0; r < 4; r++)
        for (int cc = 0; cc < 3; cc++) {
            int a = cell_val(b, r*4+cc), d2 = cell_val(b, r*4+cc+1);
            if (a && d2) mono -= abs((1<<a)-(1<<d2))*0.001;
            int u = cell_val(b, cc*4+r), v2 = cell_val(b, (cc+1)*4+r);
            if (u && v2) mono -= abs((1<<u)-(1<<v2))*0.001;
        }
    return pos + empt * 270.0 + mono;
}

/* Transposition table: simple open addressing with linear probing */
#define TT_SIZE 1048576 /* 1M entries */
typedef struct { etb_board_t key; int depth; double val; int move; int valid; } tt_entry_t;
static tt_entry_t g_tt[TT_SIZE];

static uint64_t hash_board(etb_board_t b) {
    uint64_t h = b ^ 0x9E3779B97F4A7C15ULL;
    h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    return h ^ (h >> 31);
}

static uint64_t g_tt_probe = 0, g_tt_hit = 0;

static tt_entry_t *tt_lookup(etb_board_t b, int depth) {
    uint64_t h = hash_board(b);
    tt_entry_t *slot = NULL;
    for (int i = 0; i < 8; i++) {
        tt_entry_t *e = &g_tt[(h + i) & (TT_SIZE - 1)];
        if (e->valid && e->key == b) {
            g_tt_probe++;
            if (e->depth >= depth) { g_tt_hit++; return e; }
            slot = e; /* keep shallow entry for replacement */
        } else if (!e->valid) {
            if (!slot) slot = e;
            break;
        }
    }
    if (slot) { slot->key = b; slot->depth = -depth; slot->val = 0; slot->move = -1; slot->valid = 1; }
    return NULL;
}

static void tt_store(etb_board_t b, int depth, double val) {
    uint64_t h = hash_board(b);
    for (int i = 0; i < 8; i++) {
        tt_entry_t *e = &g_tt[(h + i) & (TT_SIZE - 1)];
        if (!e->valid || e->key == b) {
            e->key = b; e->depth = depth; e->val = val; e->move = -1; e->valid = 1;
            return;
        }
    }
    /* replace the shallowest entry */
    tt_entry_t *victim = &g_tt[h & (TT_SIZE - 1)];
    victim->key = b; victim->depth = depth; victim->val = val; victim->move = -1; victim->valid = 1;
}

static int pick_best_move_id(etb_board_t b, int depth, double *out_val);

static double expectimax_tt(etb_board_t b, int depth, double cprob) {
    if (depth == 0 || cprob < 0.0001) return evaluate(b);
    tt_entry_t *cached = tt_lookup(b, depth);
    if (cached) return cached->val;
    double best = -1e18;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        etb_board_t m = do_move(b, d);
        if (m == b) continue;
        any = 1;
        int empties[16], n = 0;
        for (int c = 0; c < 16; c++) if (cell_empty(m, c)) empties[n++] = c;
        double acc = 0;
        for (int i = 0; i < n; i++) {
            etb_board_t s2 = set_cell(m, empties[i], 1);
            acc += 0.9 * expectimax_tt(s2, depth - 1, cprob * 0.9 / n);
            if (depth >= 3) {
                etb_board_t s4 = set_cell(m, empties[i], 2);
                acc += 0.1 * expectimax_tt(s4, depth - 1, cprob * 0.1 / n);
            }
        }
        double val = n ? acc / n : evaluate(m);
        if (val > best) best = val;
    }
    if (!any) return evaluate(b) - 1e6;
    tt_store(b, depth, best);
    return best;
}

static int pick_best_move_id(etb_board_t b, int depth, double *out_val) {
    int best_d = -1; double best_v = -1e18;
    for (int d = 0; d < 4; d++) {
        etb_board_t m = do_move(b, d);
        if (m == b) continue;
        int empties[16], n = 0;
        for (int c = 0; c < 16; c++) if (cell_empty(m, c)) empties[n++] = c;
        double v = n ? expectimax_tt(m, depth - 1, 1.0) : evaluate(m);
        if (v > best_v) { best_v = v; best_d = d; }
    }
    if (out_val) *out_val = best_v;
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
    int c = empties[rr() % n];
    return set_cell(b, c, rr01() < 0.9 ? 1 : 2);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bot <n_games> [rng_seed] [max_depth]\n"); return 2; }
    long n = atol(argv[1]);
    if (argc > 2) rr_state = strtoull(argv[2], NULL, 0) | 1;
    if (argc > 3) g_max_depth = atoi(argv[3]);
    if (g_max_depth < 1) g_max_depth = 1;
    memset(g_tt, 0, sizeof(g_tt));
    etb_init_luts();
    long achieved[17] = {0};
    double total_sum = 0, total_moves = 0;
    for (long g = 0; g < n; g++) {
        etb_board_t b = 0; b = random_spawn(b); b = random_spawn(b);
        long moves = 0;
        for (;;) {
            double dummy; int d = pick_best_move_id(b, g_max_depth, &dummy);
            if (d < 0) break;
            etb_board_t m = do_move(b, d);
            if (m == b) break;
            b = m; b = random_spawn(b); moves++;
            if (moves > 20000) break;
        }
        total_moves += moves;
        int mx = max_tile_exp(b);
        for (int e = 0; e <= mx && e < 17; e++) achieved[e]++;
        for (int c = 0; c < 16; c++) { int e = cell_val(b, c); if (e) total_sum += pow(2.0, e); }
    }
    printf("games=%ld depth=%d avg_moves=%.0f avg_final_board_sum=%.0f\n",
           n, g_max_depth, total_moves/n, total_sum/n);
    printf("tile  rate\n");
    for (int e = 1; e <= 16; e++) printf("%5d  %.2f%%\n", 1 << e, 100.0 * achieved[e] / n);
    return 0;
}
