/* mc_eval: Monte-Carlo evaluation of a solved tablebase.
 * Simulates games where the player picks the move maximizing the table's
 * expected value (same formula as the solver), and the environment spawns
 * 2 (90%) / 4 (10%) uniformly among free-zone cells.
 * Reports empirical win rate for the seed board and compares with the
 * table's stored rate.
 */
#include "gen.h"
#include "pack.h"
#include "symm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static char g_dir[512];
static Layer *g_layers[512];
static int g_nlayers = 0;
static uint64_t g_free_mask = ~0ULL;
static SymmGroup g_grp;

static uint64_t rr_state = 0x12345678;
static uint64_t rr(void) {
    rr_state ^= rr_state << 13; rr_state ^= rr_state >> 7; rr_state ^= rr_state << 17;
    return rr_state;
}
static double rr01(void) { return (rr() >> 11) * (1.0 / 9007199254740992.0); }

static int find_rate(etb_board_t b, etb_rate_t *out) {
    etb_board_t cb = etb_canonical_in(b, &g_grp);
    for (int li = 0; li < g_nlayers; li++) {
        uint64_t idx;
        if (layer_lookup(g_layers[li], cb, &idx)) { *out = g_layers[li]->rates[idx]; return 1; }
    }
    /* fall back to raw key in case symmetry was off at gen time */
    for (int li = 0; li < g_nlayers; li++) {
        uint64_t idx;
        if (layer_lookup(g_layers[li], b, &idx)) { *out = g_layers[li]->rates[idx]; return 1; }
    }
    return 0;
}

static int has_target(etb_board_t b, int te) {
    for (int c = 0; c < 16; c++)
        if (((b >> ((15 - c) * 4)) & 0xF) == te) return 1;
    return 0;
}

/* one simulated game: returns 1 win, 0 loss/dead */
static int play_game(etb_board_t start, int target_exp) {
    etb_board_t b = start;
    for (int step = 0; step < 1000; step++) {
        if (has_target(b, target_exp)) return 1;
        /* pick argmax-EV move using table values; ties -> first */
        int best_d = -1; double best_ev = -1;
        for (int d = 0; d < 4; d++) {
            etb_board_t m = etb_move_dir(b, d);
            if (m == b) continue;
            int empt = 0;
            for (int c = 0; c < 16; c++)
                if (!((m >> ((15 - c) * 4)) & 0xF) && ((g_free_mask >> c) & 1)) empt++;
            if (empt == 0) continue;
            double acc = 0;
            for (int c = 0; c < 16; c++) {
                if (((m >> ((15 - c) * 4)) & 0xF) != 0) continue;
                if (!((g_free_mask >> c) & 1)) continue;
                for (int sp = 1; sp <= 2; sp++) {
                    etb_board_t s = m | ((etb_board_t)sp << ((15 - c) * 4));
                    etb_rate_t r = 0;
                    if (find_rate(s, &r))
                        acc += (sp == 1 ? 0.9 : 0.1) * (r / 65535.0);
                }
            }
            double ev = acc / empt;
            if (ev > best_ev) { best_ev = ev; best_d = d; }
        }
        if (best_d < 0) return 0; /* stuck: no legal move */
        etb_board_t m = etb_move_dir(b, best_d);
        /* sample spawn */
        int cells[16], nc = 0;
        for (int c = 0; c < 16; c++)
            if (!((m >> ((15 - c) * 4)) & 0xF) && ((g_free_mask >> c) & 1)) cells[nc++] = c;
        if (nc == 0) return 0;
        int cell = cells[rr() % nc];
        int sp = (rr01() < 0.9) ? 1 : 2;
        b = m | ((etb_board_t)sp << ((15 - cell) * 4));
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: mc_eval <outdir> <seed_hex> <target_exp> <n_games> [seed]\n");
        return 2;
    }
    etb_init_luts();
    etb_symm_init();
    snprintf(g_dir, sizeof g_dir, "%s", argv[1]);
    etb_board_t start = strtoull(argv[2], NULL, 16);
    int target_exp = atoi(argv[3]);
    long n = atol(argv[4]);
    if (argc > 5) rr_state = strtoull(argv[5], NULL, 0) | 1;

    while (g_nlayers < 512) {
        Layer *l = load_layer(g_dir, g_nlayers);
        if (!l) break;
        g_layers[g_nlayers++] = l;
    }
    if (!g_nlayers) { fprintf(stderr, "no layers\n"); return 1; }
    printf("[mc] loaded %d layers\n", g_nlayers);

    /* meta: free mask + symmetry */
    char mp[600];
    snprintf(mp, sizeof mp, "%s/meta.bin", g_dir);
    FILE *mf = fopen(mp, "rb");
    if (mf) {
        uint32_t magic = 0; uint64_t occ = 0; int nemp = 0; uint64_t msum = 0;
        uint8_t use_symm = 0; uint64_t stsl = 0; double pb = 0;
        if (fread(&magic, 4, 1, mf) == 1 && magic == 0x31425445u &&
            fread(&occ, 8, 1, mf) == 1 && fread(&nemp, 4, 1, mf) == 1 &&
            fread(&msum, 8, 1, mf) == 1 && fread(&use_symm, 1, 1, mf) == 1 &&
            fread(&stsl, 8, 1, mf) == 1 && fread(&pb, 8, 1, mf) == 1) {
            g_free_mask = ~occ;
            if (use_symm) etb_build_group((uint16_t)(g_free_mask & 0xFFFF), &g_grp);
        }
        fclose(mf);
    } else {
        g_grp.count = 1; g_grp.idx[0] = 0;
    }

    etb_rate_t tr;
    double table_rate = find_rate(start, &tr) ? tr / 65535.0 : -1;
    long wins = 0;
    for (long i = 0; i < n; i++) wins += play_game(start, target_exp);
    double emp = (double)wins / n;
    double se = sqrt(emp * (1 - emp) / n);
    printf("table rate = %.4f%%\nempirical  = %ld/%ld = %.4f%% (+-%.4f%%)\n",
           table_rate * 100, wins, n, emp * 100, 1.96 * se * 100);
    if (table_rate >= 0) {
        double z = (emp - table_rate) / (se > 0 ? se : 1e-9);
        printf("z = %+.2f -> %s\n", z,
               (z > -3 && z < 3) ? "AGREE within noise" : "MISMATCH");
    }
    return 0;
}
