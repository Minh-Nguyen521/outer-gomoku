/* bot6: bot4 heuristic + parallel root search (pthread) + deeper depth */
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>

#define NTHREADS 8

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

static int move_merge_points(etb_board_t before, etb_board_t after) {
    int pts = 0;
    for (int c = 0; c < 16; c++) {
        int va = cell_val(after, c), vb = cell_val(before, c);
        if (va > vb && va > 0) pts += 1 << va;
    }
    return pts;
}

static int imax2(int a, int b){return a>b?a:b;}
static int32_t process_score(int32_t s) {
    if (s < 200) return imax2(0, (s >> 2) - 10);
    if (s < 500) return (s >> 1) - 12;
    if (s < 1000) return (s >> 1) + 144;
    return s < 2000 ? s + 800 : 4000;
}

typedef struct { int16_t d1, d2; } score_pair_t;
static score_pair_t DIFFS_MERGED[65536];

static const int32_t TILE_WEIGHT_MAP[16] = {0, 2, 4, 8, 16, 32, 64, 128,
                                           248, 388, 488, 518, 519, 519, 519, 520};

static int imax(int a, int b){return a>b?a:b;}

static int32_t diffs_evaluation_func(const int32_t lm[4]) {
    int32_t score_dpdf = lm[0];
    for (int x = 0; x < 3; ++x) {
        if (lm[x] < lm[x + 1]) {
            if (lm[x] > 400) {
                if (x == 0) score_dpdf += lm[x + 1] - lm[x];
                else if (lm[x] < 600 && x == 1)
                    score_dpdf += (lm[x] << 1) + (lm[x + 1] - lm[x]) * x;
            } else if (lm[x] > 300 && x == 1 && lm[0] > lm[1]) {
                score_dpdf += (lm[x] << 1);
            } else if (lm[x] < 300) {
                score_dpdf -= (lm[x + 1] - lm[x]) << 3;
                score_dpdf -= lm[x + 1] * 3;
                if (x < 2 && lm[x + 2] < lm[x + 1] && lm[x + 1] > 30)
                    score_dpdf -= imax(80, lm[x + 1]);
            } else {
                score_dpdf += (lm[x] << 1) + lm[x + 1];
            }
        } else {
            if (lm[x] < 300 && lm[x + 1] < 300)
                score_dpdf += (lm[x] + lm[x + 1]) * 0.5;
            else
                score_dpdf += lm[x + 1] + lm[x];
        }
    }
    if (lm[0] > 400 && lm[1] > 300 && lm[2] > 200 && lm[3] > 100)
        score_dpdf += 400;
    return score_dpdf;
}

static void init_row_tables(void) {
    for (int32_t r = 0; r < 65536; r++) {
        int32_t lm[4];
        int32_t tw = 0;
        for (int i = 0; i < 4; i++)
            lm[i] = TILE_WEIGHT_MAP[(r >> ((3 - i) * 4)) & 0xF];
        for (int i = 0; i < 4; i++)
            tw += TILE_WEIGHT_MAP[(r >> ((3 - i) * 4)) & 0xF];
        (void)tw;
        score_pair_t pair;
        pair.d1 = (int16_t)diffs_evaluation_func(lm);
        /* d2: mirrored line */
        int32_t lmr[4] = {lm[3], lm[2], lm[1], lm[0]};
        pair.d2 = (int16_t)diffs_evaluation_func(lmr);
        DIFFS_MERGED[r] = pair;
    }
}

static inline uint16_t get_row(etb_board_t b, int r) {
    return (uint16_t)((b >> ((3 - r) * 16)) & 0xFFFF);
}

/* transpose for column access */
static inline etb_board_t btrans(etb_board_t b) {
    etb_board_t t = 0;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            int nib = (int)((b >> ((15 - (r * 4 + c)) * 4)) & 0xF);
            t |= (etb_board_t)nib << ((15 - (c * 4 + r)) * 4);
        }
    return t;
}

static double heur_board(etb_board_t b) {
    etb_board_t s = b;
    etb_board_t sr = btrans(s);
    int32_t sum_x1 = 0, sum_x2 = 0, sum_y1 = 0, sum_y2 = 0;
    for (int i = 0; i < 4; ++i) {
        uint16_t l1 = get_row(s, i);
        uint16_t l2 = get_row(sr, i);
        score_pair_t px = DIFFS_MERGED[l1];
        sum_x1 += px.d1; sum_x2 += px.d2;
        score_pair_t py = DIFFS_MERGED[l2];
        sum_y1 += py.d1; sum_y2 += py.d2;
    }
    double v = (double)(imax(sum_x1, sum_x2) + imax(sum_y1, sum_y2)) / 1.0;
    return v;
}

/* ---------- per-thread TT ---------- */
#define TT_SIZE (1 << 20)
typedef struct { etb_board_t key; double val; int8_t depth; uint8_t valid; } tt_t;

static tt_t *g_tts[NTHREADS];

static inline uint64_t mix(etb_board_t b) {
    uint64_t h = b * 0x2545F4914F6CDD1DULL;
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32;
    return h;
}

/* ---------- Expectimax (thread-local TT) ---------- */
static double expectimax(tt_t *tt, etb_board_t b, int depth, double cprob) {
    if (depth == 0 || cprob < 1e-4) return heur_board(b);
    tt_t *e = &tt[mix(b) & (TT_SIZE - 1)];
    if (e->valid && e->key == b && e->depth >= depth) return e->val;

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
            acc += 0.9 * expectimax(tt, set_cell(m, empties[i], 1), depth - 1, cprob * 0.9 / n);
            acc += 0.1 * expectimax(tt, set_cell(m, empties[i], 2), depth - 1, cprob * 0.1 / n);
        }
        double val = merged + acc / n;
        if (val > best) best = val;
    }
    if (!any) return heur_board(b) - 1e9;
    e->key = b; e->val = best; e->depth = (int8_t)depth; e->valid = 1;
    return best;
}

/* ---------- parallel root ---------- */
typedef struct {
    etb_board_t moved;   /* board after move dir */
    etb_board_t before;  /* board before */
    int depth;
    double result;
    tt_t *tt;
    int valid;
} root_job_t;

typedef struct {
    root_job_t jobs[4];
    int n_jobs;
    int next_job;
    pthread_mutex_t mu;
} jobq_t;

static void *worker(void *arg) {
    jobq_t *q = (jobq_t *)arg;
    for (;;) {
        int idx;
        pthread_mutex_lock(&q->mu);
        if (q->next_job >= q->n_jobs) { pthread_mutex_unlock(&q->mu); return NULL; }
        idx = q->next_job++;
        pthread_mutex_unlock(&q->mu);
        root_job_t *j = &q->jobs[idx];
        /* simulate one chance ply inside job: average over spawns, with
           recursive expectimax below (same structure as serial version) */
        etb_board_t m = j->moved;
        int merged = process_score(move_merge_points(j->before, m));
        int empties[16], n = 0;
        for (int c = 0; c < 16; c++) if (cell_empty(m, c)) empties[n++] = c;
        if (!n) { j->result = heur_board(m) + merged; continue; }
        double acc = 0;
        for (int i = 0; i < n; i++) {
            acc += 0.9 * expectimax(j->tt, set_cell(m, empties[i], 1), j->depth - 1, 0.9 / n);
            acc += 0.1 * expectimax(j->tt, set_cell(m, empties[i], 2), j->depth - 1, 0.1 / n);
        }
        j->result = merged + acc / n;
    }
}

static int depth_for_empties(int empties) {
    /* deeper than bot4: was 2..7, now 3..9 */
    if (empties >= 8) return 3;
    if (empties >= 5) return 4;
    if (empties >= 3) return 5;
    if (empties >= 2) return 6;
    if (empties == 1) return 7;
    return 9;
}

static int pick_move(etb_board_t b, jobq_t *q, tt_t **tts) {
    int empties = 0;
    for (int c = 0; c < 16; c++) if (cell_empty(b, c)) empties++;
    int depth = depth_for_empties(empties);

    q->n_jobs = 0;
    for (int d = 0; d < 4; d++) {
        etb_board_t m = etb_move_dir(b, d);
        if (m == b) continue;
        q->jobs[q->n_jobs].moved = m;
        q->jobs[q->n_jobs].before = b;
        q->jobs[q->n_jobs].depth = depth;
        q->jobs[q->n_jobs].valid = d;
        q->jobs[q->n_jobs].tt = tts[q->n_jobs % NTHREADS];
        q->n_jobs++;
    }
    if (!q->n_jobs) return -1;
    q->next_job = 0;

    static pthread_t th[NTHREADS - 1];
    for (int i = 0; i < NTHREADS - 1; i++)
        pthread_create(&th[i], NULL, worker, q);
    worker(q);
    for (int i = 0; i < NTHREADS - 1; i++)
        pthread_join(th[i], NULL);

    int best_i = 0;
    for (int i = 1; i < q->n_jobs; i++)
        if (q->jobs[i].result > q->jobs[best_i].result) best_i = i;
    return q->jobs[best_i].valid;
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
    if (argc < 2) { fprintf(stderr, "usage: bot6 <n_games> [rng_seed]\n"); return 2; }
    long n = atol(argv[1]);
    if (argc > 2) rr_state = strtoull(argv[2], NULL, 0) | 1;

    etb_init_luts();
    init_row_tables();
    for (int i = 0; i < NTHREADS; i++) g_tts[i] = calloc(TT_SIZE, sizeof(tt_t));

    jobq_t *q = malloc(sizeof(jobq_t));
    pthread_mutex_init(&q->mu, NULL);

    long achieved[17] = {0};
    double total_sum = 0; long total_moves = 0;
    for (long g = 0; g < n; g++) {
        for (int i = 0; i < NTHREADS; i++) memset(g_tts[i], 0, TT_SIZE * sizeof(tt_t));
        etb_board_t b = 0; b = random_spawn(b); b = random_spawn(b);
        long moves = 0;
        for (;;) {
            int d = pick_move(b, q, g_tts);
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
