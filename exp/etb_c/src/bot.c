/* bot: full 2048 player.
 * Expectimax depth-limited with a snake-position + empties + monotonicity
 * heuristic. etb_best_move hook allows an EGTB oracle to take over endgames.
 *
 * usage: bot <n_games> [rng_seed] [max_depth]
 */
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

/* cell nibble index c (0..15): c = row*4+col, row 0 = top, stored at shift
   (15-c)*4 in the u64 (MSB-first). Helpers below use that convention. */
static inline int cell_val(etb_board_t b, int c) { return (int)((b >> ((15 - c) * 4)) & 0xF); }
static inline etb_board_t set_cell(etb_board_t b, int c, int e) {
    int sh = (15 - c) * 4;
    return (b & ~(0xFULL << sh)) | ((etb_board_t)e << sh);
}
static inline int cell_empty(etb_board_t b, int c) { return cell_val(b, c) == 0; }

static etb_board_t do_move(etb_board_t b, int d) { return etb_move_dir(b, d); }

/* snake weight: high weights top-left descending then snake down */
#define P4(n) ((double)(1u << (2*(n))))
static const double W[16] = {
    P4(15), P4(14), P4(13), P4(12),
    P4(8),  P4(9),  P4(10), P4(11),
    P4(7),  P4(6),  P4(5),  P4(4),
    P4(3),  P4(2),  P4(1),  P4(0)
};

static double evaluate(etb_board_t b) {
    double pos = 0;
    int empt = 0;
    for (int c = 0; c < 16; c++) {
        int e = cell_val(b, c);
        if (!e) { empt++; continue; }
        pos += W[c] * (double)(1u << e);
    }
    /* monotonicity penalty along rows and columns */
    double mono = 0;
    for (int r = 0; r < 4; r++)
        for (int cc = 0; cc < 3; cc++) {
            int a = cell_val(b, r * 4 + cc), d2 = cell_val(b, r * 4 + cc + 1);
            if (a && d2) mono -= abs((1 << a) - (1 << d2)) * 0.001;
            int u = cell_val(b, cc * 4 + r), v2 = cell_val(b, (cc + 1) * 4 + r);
            if (u && v2) mono -= abs((1 << u) - (1 << v2)) * 0.001;
        }
    return pos + empt * 270.0 + mono;
}

/* hook: overridden by EGTB-aware builds; default = disabled */
int etb_best_move(etb_board_t b, int *best) { (void)b; (void)best; return 0; }

static double expectimax(etb_board_t b, int depth, double cprob) {
    if (depth == 0 || cprob < 0.0001)
        return evaluate(b);
    double best = -1e18;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        etb_board_t m = do_move(b, d);
        if (m == b) continue;
        any = 1;
        int empties[16], n = 0;
        for (int c = 0; c < 16; c++)
            if (cell_empty(m, c)) empties[n++] = c;
        double acc = 0;
        for (int i = 0; i < n; i++) {
            etb_board_t s2 = set_cell(m, empties[i], 1);
            acc += 0.9 * expectimax(s2, depth - 1, cprob * 0.9 / n);
            if (depth >= 3) { /* only model 4-spawns near the root */
                etb_board_t s4 = set_cell(m, empties[i], 2);
                acc += 0.1 * expectimax(s4, depth - 1, cprob * 0.1 / n);
            }
        }
        double val = n ? acc / n : evaluate(m);
        if (val > best) best = val;
    }
    if (!any) return evaluate(b) - 1e6;
    return best;
}

static int pick_best_move(etb_board_t b) {
    int tb = -1;
    if (etb_best_move(b, &tb)) return tb;
    const char *mode = getenv("BOT_MODE"); /* ""|expectimax|greedy|random */
    if (mode && !strcmp(mode, "random")) {
        int dirs[4], nd = 0;
        for (int d = 0; d < 4; d++) if (do_move(b, d) != b) dirs[nd++] = d;
        return nd ? dirs[rr() % nd] : -1;
    }
    int greedy = mode && !strcmp(mode, "greedy");
    int best_d = -1; double best_v = -1e18;
    for (int d = 0; d < 4; d++) {
        etb_board_t m = do_move(b, d);
        if (m == b) continue;
        int empties[16], n = 0;
        for (int c = 0; c < 16; c++)
            if (cell_empty(m, c)) empties[n++] = c;
        double v = greedy ? evaluate(m) :
                   (n ? expectimax(m, g_max_depth - 1, 1.0) : evaluate(m));
        if (v > best_v) { best_v = v; best_d = d; }
    }
    return best_d;
}

static int max_tile_exp(etb_board_t b) {
    int mx = 0;
    for (int c = 0; c < 16; c++) {
        int e = cell_val(b, c);
        if (e > mx) mx = e;
    }
    return mx;
}

static etb_board_t random_spawn(etb_board_t b) {
    int empties[16], n = 0;
    for (int c = 0; c < 16; c++)
        if (cell_empty(b, c)) empties[n++] = c;
    if (!n) return b;
    int c = empties[rr() % n];
    return set_cell(b, c, rr01() < 0.9 ? 1 : 2);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bot <n_games> [rng_seed] [max_depth]\n");
        return 2;
    }
    long n = atol(argv[1]);
    if (argc > 2) rr_state = strtoull(argv[2], NULL, 0) | 1;
    if (argc > 3) g_max_depth = atoi(argv[3]);
    if (g_max_depth < 1) g_max_depth = 1;

    etb_init_luts();

    long achieved[17] = {0};
    double total_sum = 0;
    long total_moves = 0;
    for (long g = 0; g < n; g++) {
        etb_board_t b = 0;
        b = random_spawn(b);
        b = random_spawn(b);
        long moves = 0;
        for (;;) {
            int d = pick_best_move(b);
            if (d < 0) break;
            etb_board_t m = do_move(b, d);
            if (m == b) break;            b = m;
            b = random_spawn(b);
            moves++;
            if (moves > 20000) break;
        }
        total_moves += moves;
        int mx = max_tile_exp(b);
        for (int e = 0; e <= mx && e < 17; e++) achieved[e]++;
        for (int c = 0; c < 16; c++) {
            int e = cell_val(b, c);
            if (e) total_sum += pow(2.0, e);
        }
    }
    printf("games=%ld depth=%d avg_moves=%.0f avg_final_board_sum=%.0f\n",
           n, g_max_depth, (double)total_moves / n, total_sum / n);
    printf("tile  rate\n");
    for (int e = 1; e <= 16; e++)
        printf("%5d  %.2f%%\n", 1 << e, 100.0 * achieved[e] / n);
    return 0;
}
