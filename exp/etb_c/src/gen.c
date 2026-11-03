#include "gen.h"
#include "pack.h"
#include "symm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <lzma.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Semantics (aligned with real 2048 alternation):                     */
/*   State = board after opponent's spawn.                             */
/*   Transition: B --player move--> M --spawn(2|4 @ empty)--> S.        */
/*   Generation: BFS over that graph from the seed board(s).           */
/*   Solve: value iteration V(B)=max_d E_spawn[V], terminal V=1 if    */
/*   the target exponent is present on the board.                       */
/* ------------------------------------------------------------------ */

static uint64_t board_sum(etb_board_t b) {
    uint64_t s = 0;
    for (int i = 0; i < 16; i++) {
        int nib = (b >> ((15 - i) * 4)) & 0xF;
        if (nib) s += 1ULL << nib;
    }
    return s;
}

#define MAX_LAYERS 1024

typedef struct {
    etb_board_t *keys;
    uint64_t n, cap;
} KeyBuf;

static KeyBuf *kb_create(void) {
    KeyBuf *k = malloc(sizeof(KeyBuf));
    k->n = 0; k->cap = 1 << 16;
    k->keys = malloc(k->cap * sizeof(etb_board_t));
    return k;
}
static void kb_push(KeyBuf *k, etb_board_t v) {
    if (k->n == k->cap) {
        k->cap *= 2;
        k->keys = realloc(k->keys, k->cap * sizeof(etb_board_t));
    }
    k->keys[k->n++] = v;
}
static void kb_finalize(KeyBuf *k) {
    if (k->n > 1) {
        Layer l = {k->keys, NULL, k->n, k->cap};
        layer_finalize(&l, NULL);
        k->keys = l.keys; k->n = l.n; k->cap = l.cap;
    }
}

/* global visited set (open addressing, power-of-two capacity) */
typedef struct {
    etb_board_t *keys;
    uint8_t *used;
    uint64_t cap, count;
} SeenSet;

static uint64_t ss_hash(etb_board_t k) {
    uint64_t h = k * 0x9E3779B97F4A7C15ULL;
    return h ^ (h >> 32);
}
static void ss_init(SeenSet *ss, uint64_t cap) {
    ss->cap = cap; ss->count = 0;
    ss->keys = calloc(cap, sizeof(etb_board_t));
    ss->used = calloc(cap, 1);
}
static int ss_contains(SeenSet *ss, etb_board_t k) {
    uint64_t slot = ss_hash(k) & (ss->cap - 1);
    while (ss->used[slot]) {
        if (ss->keys[slot] == k) return 1;
        slot = (slot + 1) & (ss->cap - 1);
    }
    return 0;
}
static void ss_grow(SeenSet *ss) {
    SeenSet ns;
    ss_init(&ns, ss->cap * 2);
    for (uint64_t i = 0; i < ss->cap; i++) {
        if (!ss->used[i]) continue;
        uint64_t slot = ss_hash(ss->keys[i]) & (ns.cap - 1);
        while (ns.used[slot]) slot = (slot + 1) & (ns.cap - 1);
        ns.used[slot] = 1; ns.keys[slot] = ss->keys[i];
        ns.count++;
    }
    free(ss->keys); free(ss->used);
    *ss = ns;
}
static void ss_add(SeenSet *ss, etb_board_t k) {
    if ((ss->count + 1) * 10 > ss->cap * 7) ss_grow(ss);
    uint64_t slot = ss_hash(k) & (ss->cap - 1);
    while (ss->used[slot]) {
        if (ss->keys[slot] == k) return;
        slot = (slot + 1) & (ss->cap - 1);
    }
    ss->used[slot] = 1; ss->keys[slot] = k; ss->count++;
}

int etb_generate(const etb_board_t *seeds, int n_seeds,
                 const PatternSpec *spec, double spawn_rate4,
                 const char *outdir) {
    (void)spawn_rate4;
    KeyBuf *layers[MAX_LAYERS] = {0};
    int n_layers = 0;
    uint64_t free_mask = ~spec->fixed_occupied; /* cells allowed to receive spawns */
    uint16_t free16 = (uint16_t)(free_mask & 0xFFFF);

    /* pattern-preserving symmetry subgroup (original's SymmMode selection) */
    SymmGroup grp;
    etb_symm_init();
    etb_build_group(free16, &grp);
    if (spec->use_symmetry && grp.count > 1)
        printf("[gen] symmetry group: %d transforms\n", grp.count);
    else
        grp.count = 1, grp.idx[0] = 0; /* identity only */

    /* STSL: budget on the sum of small tiles (<=4) before spawn-2 merge chains
       become hopeless; computed from spec->stsl_limit (0 = disabled). */

    KeyBuf *cur = kb_create();
    for (int i = 0; i < n_seeds; i++)
        kb_push(cur, etb_canonical_in(seeds[i], &grp));
    kb_finalize(cur);

    SeenSet ss;
    ss_init(&ss, 1 << 20);
    for (int i = 0; i < n_seeds; i++) ss_add(&ss, etb_canonical_in(seeds[i], &grp));

    while (cur->n > 0 && n_layers < MAX_LAYERS) {
        layers[n_layers++] = cur;
        KeyBuf *next = kb_create();

        for (uint64_t i = 0; i < cur->n; i++) {
            etb_board_t b = cur->keys[i];
            /* player moves */
            for (int d = 0; d < 4; d++) {
                etb_board_t m = etb_move_dir(b, d);
                if (m == b) continue;
                /* opponent spawns 2/4 on empty cells INSIDE the free zone */
                for (int cell = 0; cell < 16; cell++) {
                    if (((m >> ((15 - cell) * 4)) & 0xF) != 0) continue;
                    if (!((free_mask >> cell) & 1)) continue;
                    for (int sp = 1; sp <= 2; sp++) {
                        etb_board_t s = m | ((etb_board_t)sp << ((15 - cell) * 4));
                        /* STSL: prune spawn-2 states drowning in small tiles */
                        if (spec->stsl_limit && sp == 1) {
                            uint64_t smalls = 0;
                            for (int c2 = 0; c2 < 16; c2++) {
                                int nib = (s >> ((15 - c2) * 4)) & 0xF;
                                if (nib && nib <= 2) smalls += 1ULL << nib;
                            }
                            if (smalls > spec->stsl_limit) continue;
                        }
                        if (spec->max_sum) {
                            uint64_t bs = board_sum(s);
                            if (bs > spec->max_sum) continue;
                        }
                        etb_board_t cs = etb_canonical_in(s, &grp);
                        if (ss_contains(&ss, cs)) continue;
                        ss_add(&ss, cs);
                        kb_push(next, cs);
                    }
                }
            }
        }
        kb_finalize(next);
        cur = next;
        printf("[gen] layer %d: %llu states\n", n_layers - 1,
               (unsigned long long)cur->n);
        fflush(stdout);
    }

    mkdir(outdir, 0755);
    { /* meta: pattern info (kept for compatibility) */
        char path[512];
        snprintf(path, sizeof path, "%s/meta.bin", outdir);
        FILE *mf = fopen(path, "wb");
        if (mf) {
            uint32_t magic = 0x31425445u;
            fwrite(&magic, 4, 1, mf);
            fwrite(&spec->fixed_occupied, 8, 1, mf);
            fwrite(&spec->num_empty, 4, 1, mf);
            fwrite(&spec->max_sum, 8, 1, mf);
            uint8_t use_symm = (uint8_t)(grp.count > 1 ? 1 : 0);
            fwrite(&use_symm, 1, 1, mf);
            fwrite(&spec->stsl_limit, 8, 1, mf);
            double pb = spec->prune_below;
            fwrite(&pb, 8, 1, mf);
            fclose(mf);
        }
    }
    for (int i = 0; i < n_layers; i++) {
        char path[512];
        snprintf(path, sizeof path, "%s/layer_%04d.bin", outdir, i);
        FILE *f = fopen(path, "wb");
        if (!f) return -1;
        uint32_t magic = 0x31425445u;
        fwrite(&magic, 4, 1, f);
        fwrite(&layers[i]->n, 8, 1, f);
        fwrite(layers[i]->keys, sizeof(etb_board_t), layers[i]->n, f);
        fclose(f);
        free(layers[i]->keys); free(layers[i]);
    }
    free(ss.keys); free(ss.used);
    return n_layers;
}

/* ---------------- solve ---------------- */

Layer *load_layer(const char *outdir, int i) {
    char path[512], path2[512];
    snprintf(path, sizeof path, "%s/layer_%04d.bin", outdir, i);
    snprintf(path2, sizeof path2, "%s/layer_%04d.bin.lzma", outdir, i);
    size_t sz = 0;
    uint8_t *buf = etb_slurp_file(path, &sz);
    if (!buf) {
        buf = etb_slurp_file(path2, &sz);
        if (buf) {
            size_t raw_sz = 0;
            uint8_t *raw = etb_lzma_decompress(buf, sz, &raw_sz);
            free(buf);
            buf = raw; sz = raw_sz;
        }
    }
    if (!buf) return NULL;
    FILE *f = fmemopen(buf, sz, "rb");
    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x31425445u) { fclose(f); free(buf); return NULL; }
    Layer *l = calloc(1, sizeof(Layer));
    if (fread(&l->n, 8, 1, f) != 1) { free(l); fclose(f); free(buf); return NULL; }
    l->cap = l->n ? l->n : 1;
    l->keys = malloc(l->cap * sizeof(etb_board_t));
    if (fread(l->keys, sizeof(etb_board_t), l->n, f) != l->n) {
        free(l->keys); free(l); fclose(f); free(buf); return NULL;
    }
    l->rates = calloc(l->cap, sizeof(etb_rate_t));
    size_t want = 12 + (size_t)l->n * 8;
    if (sz >= want + (size_t)l->n * 2)
        if (fread(l->rates, sizeof(etb_rate_t), l->n, f) != l->n) { /* no rates */ }
    fclose(f); free(buf);
    return l;
}

/* ---- multi-threaded value iteration -------------------------------- */

typedef struct VIJob {
    int tid, nthr;
    Layer **layers; int n;
    etb_rate_t *rate, *newrate;
    const uint8_t *terminal;
} VIJob;

typedef struct VIState {
    SeenSet *ss;
    uint64_t free_mask;
    SymmGroup grp;
} VIState;

static VIState g_vi;
static VIJob g_jobs[64];
static pthread_t g_th[64];
static volatile uint64_t g_changed;

static void vi_worker(VIJob *job);
static void *vi_thread(void *arg) { vi_worker((VIJob *)arg); return NULL; }

static void vi_worker(VIJob *job) {
    SeenSet *ss = g_vi.ss;
    uint64_t free_mask = g_vi.free_mask;
    SymmGroup grp = g_vi.grp;
    for (int li = 0; li < job->n; li++) {
        Layer *l = job->layers[li];
        for (uint64_t i = (uint64_t)job->tid; i < l->n; i += (uint64_t)job->nthr) {
            etb_board_t b = l->keys[i];
            uint64_t slot = 0, probe = ss_hash(b) & (ss->cap - 1);
            while (!(ss->used[probe] && ss->keys[probe] == b)) probe = (probe + 1) & (ss->cap - 1);
            slot = probe;
            if (job->terminal[slot]) continue;
            double best = 0.0;
            for (int d = 0; d < 4; d++) {
                etb_board_t m = etb_move_dir(b, d);
                if (m == b) continue;
                int empt = 0;
                for (int c2 = 0; c2 < 16; c2++)
                    if (!((m >> ((15 - c2) * 4)) & 0xF) && ((free_mask >> c2) & 1)) empt++;
                if (empt == 0) continue;
                double acc = 0.0;
                for (int cell = 0; cell < 16; cell++) {
                    if (((m >> ((15 - cell) * 4)) & 0xF) != 0) continue;
                    if (!((free_mask >> cell) & 1)) continue;
                    for (int sp = 1; sp <= 2; sp++) {
                        etb_board_t sb = m | ((etb_board_t)sp << ((15 - cell) * 4));
                        etb_board_t cs = etb_canonical_in(sb, &grp);
                        if (!ss_contains(ss, cs)) continue; /* off-graph: rate 0, already excluded from acc */
                        uint64_t sslot = ss_hash(cs) & (ss->cap - 1);
                        while (!(ss->used[sslot] && ss->keys[sslot] == cs))
                            sslot = (sslot + 1) & (ss->cap - 1);
                        acc += (sp == 1 ? 0.9 : 0.1) * (job->rate[sslot] / 65535.0);
                    }
                }
                double ev = acc / empt;
                if (ev > best) best = ev;
            }
            etb_rate_t nr = (etb_rate_t)(best * 65535.0 + 0.5);
            if (nr != job->rate[slot]) { job->newrate[slot] = nr; g_changed++; }
            else if (job->newrate[slot] != nr) job->newrate[slot] = nr;
        }
    }
}

int etb_solve(const char *outdir, int target_exp) {
    Layer *layers[MAX_LAYERS];
    int n = 0;
    while (n < MAX_LAYERS) {
        Layer *l = load_layer(outdir, n);
        if (!l) break;
        layers[n++] = l;
    }
    if (n == 0) { fprintf(stderr, "no layers in %s\n", outdir); return -1; }
    printf("[solve] loaded %d layers\n", n);

    uint64_t free_mask = ~0ULL; /* default: unrestricted spawns */
    double prune_below = 0.0;
    SymmGroup grp; grp.count = 1; grp.idx[0] = 0;
    etb_symm_init();
    {
        char path[512];
        snprintf(path, sizeof path, "%s/meta.bin", outdir);
        FILE *mf = fopen(path, "rb");
        if (mf) {
            uint32_t magic = 0;
            uint64_t occ = 0; int nemp = 0; uint64_t msum = 0;
            uint8_t use_symm = 0; uint64_t stsl = 0; double pb = 0.0;
            if (fread(&magic, 4, 1, mf) == 1 && magic == 0x31425445u &&
                fread(&occ, 8, 1, mf) == 1 &&
                fread(&nemp, 4, 1, mf) == 1 &&
                fread(&msum, 8, 1, mf) == 1 &&
                fread(&use_symm, 1, 1, mf) == 1 &&
                fread(&stsl, 8, 1, mf) == 1 &&
                fread(&pb, 8, 1, mf) == 1) {
                free_mask = ~occ;
                if (use_symm)
                    etb_build_group((uint16_t)(free_mask & 0xFFFF), &grp);
                prune_below = pb;
            }
            fclose(mf);
        }
    }

    /* global index (open addressing) over all states */
    uint64_t total = 0;
    for (int i = 0; i < n; i++) total += layers[i]->n;
    SeenSet ss;
    ss_init(&ss, 1);
    while (ss.cap < total * 2) ss_grow(&ss);
    for (int li = 0; li < n; li++)
        for (uint64_t i = 0; i < layers[li]->n; i++)
            ss_add(&ss, layers[li]->keys[i]);

    /* rates live in a parallel table sized like the set */
    etb_rate_t *rate = calloc(ss.cap, sizeof(etb_rate_t));
    uint8_t *terminal = calloc(ss.cap, 1);

#define SLOT(k) ({ uint64_t _s = ss_hash(k) & (ss.cap-1); \
                   while (!(ss.used[_s] && ss.keys[_s]==(k))) _s=(_s+1)&(ss.cap-1); _s; })

    /* boundary: terminal win iff target present */
    for (int li = 0; li < n; li++) {
        for (uint64_t i = 0; i < layers[li]->n; i++) {
            etb_board_t b = layers[li]->keys[i];
            int hit = 0;
            for (int c = 0; c < 16 && !hit; c++)
                if (((b >> ((15 - c) * 4)) & 0xF) == target_exp) hit = 1;
            if (hit) {
                uint64_t slot = SLOT(b);
                rate[slot] = 65535;
                terminal[slot] = 1;
            }
        }
    }

    /* value iteration: V(b) = max_d E_spawn[V].
       Sweeps run multi-threaded: each thread owns a disjoint stripe of the
       state space, reading `rate` and writing `newrate` (Jacobi), so no locks. */
    etb_rate_t *newrate = calloc(ss.cap, sizeof(etb_rate_t));
    memcpy(newrate, rate, ss.cap * sizeof(etb_rate_t));

    g_vi.ss = &ss; g_vi.free_mask = free_mask; g_vi.grp = grp;

    int nthr = 1;
    { const char *t = getenv("ETB_THREADS");
      if (t) { nthr = atoi(t); if (nthr < 1) nthr = 1; } }
    if (nthr > 64) nthr = 64;

    for (int iter = 0; iter < 100000; iter++) {
        g_changed = 0;
        for (int t = 0; t < nthr; t++) {
            g_jobs[t].tid = t; g_jobs[t].nthr = nthr;
            g_jobs[t].layers = layers; g_jobs[t].n = n;
            g_jobs[t].rate = rate; g_jobs[t].newrate = newrate;
            g_jobs[t].terminal = terminal;
            if (nthr == 1) vi_worker(&g_jobs[t]);
            else if (pthread_create(&g_th[t], NULL, vi_thread, &g_jobs[t])) {
                fprintf(stderr, "[solve] pthread_create failed, falling back\n");
                for (int u = 0; u < t; u++) pthread_join(g_th[u], NULL);
                nthr = 1; vi_worker(&g_jobs[0]); break;
            }
        }
        if (nthr > 1)
            for (int t = 0; t < nthr; t++) pthread_join(g_th[t], NULL);

        uint64_t changed = g_changed;
        if (!changed) { printf("[solve] converged after %d iterations\n", iter + 1); break; }
        /* commit sweep */
        etb_rate_t *tmp = rate; rate = newrate; newrate = tmp;
    }
    free(newrate);

    /* prune-low-success: zero out (drop) states whose converged rate sits
       below the threshold -- the table then behaves as if those states
       never existed (rate 0), matching the original's deletion_threshold. */
    uint64_t pruned = 0;
    if (prune_below > 0.0) {
        for (int li = 0; li < n; li++) {
            for (uint64_t i = 0; i < layers[li]->n; i++) {
                uint64_t slot = SLOT(layers[li]->keys[i]);
                if (terminal[slot]) continue;
                if (rate[slot] / 65535.0 < prune_below && rate[slot] != 0) {
                    rate[slot] = 0;
                    pruned++;
                }
            }
        }
        printf("[solve] pruned %llu low-success states (< %.1f%%)\n",
               (unsigned long long)pruned, prune_below * 100.0);
    }

    /* write rates into layer files */
    for (int li = 0; li < n; li++) {
        Layer *l = layers[li];
        for (uint64_t i = 0; i < l->n; i++)
            l->rates[i] = rate[SLOT(l->keys[i])];
        char path[512];
        snprintf(path, sizeof path, "%s/layer_%04d.bin", outdir, li);
        FILE *f = fopen(path, "wb");
        if (!f) return -1;
        uint32_t magic = 0x31425445u;
        fwrite(&magic, 4, 1, f);
        fwrite(&l->n, 8, 1, f);
        fwrite(l->keys, sizeof(etb_board_t), l->n, f);
        fwrite(l->rates, sizeof(etb_rate_t), l->n, f);
        fclose(f);
    }

    free(rate); free(terminal);
    free(ss.keys); free(ss.used);
    for (int i = 0; i < n; i++) layer_destroy(layers[i]);
    return n;
#undef SLOT
}

etb_rate_t etb_query(const char *outdir, etb_board_t board, int target_exp) {
    (void)target_exp;
    for (int li = 0; ; li++) {
        Layer *l = load_layer(outdir, li);
        if (!l) break;
        uint64_t idx;
        if (layer_lookup(l, board, &idx)) {
            etb_rate_t r = l->rates ? l->rates[idx] : 0;
            layer_destroy(l);
            return r;
        }
        layer_destroy(l);
    }
    return 0;
}
