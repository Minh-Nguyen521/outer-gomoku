#include "layer.h"
#include <stdlib.h>
#include <string.h>

/* ---------- radix sort (LSD, 4 passes x 16 bits) ----------
   Uniform uint64 keys -> counting sort per pass, stable, O(n). */
static void radix_sort_u64(etb_board_t *a, etb_board_t *tmp, uint64_t n) {
    if (n < 64) { /* insertion sort for tiny arrays */
        for (uint64_t i = 1; i < n; i++) {
            etb_board_t v = a[i];
            int64_t j = (int64_t)i - 1;
            while (j >= 0 && a[j] > v) { a[j+1] = a[j]; j--; }
            a[j+1] = v;
        }
        return;
    }
    static const int PASSES = 4;
    for (int p = 0; p < PASSES; p++) {
        uint32_t counts[65536] = {0};
        int shift = p * 16;
        for (uint64_t i = 0; i < n; i++)
            counts[(a[i] >> shift) & 0xFFFF]++;
        uint32_t offsets[65536];
        uint32_t sum = 0;
        for (int i = 0; i < 65536; i++) { offsets[i] = sum; sum += counts[i]; }
        for (uint64_t i = 0; i < n; i++)
            tmp[offsets[(a[i] >> shift) & 0xFFFF]++] = a[i];
        etb_board_t *sw = a; a = tmp; tmp = sw;
    }
    /* after 4 (even) passes result is back in original buffer */
}

Layer *layer_create(uint64_t cap_hint) {
    Layer *l = calloc(1, sizeof(Layer));
    l->cap = cap_hint ? cap_hint : 1024;
    l->keys = malloc(l->cap * sizeof(etb_board_t));
    return l;
}

void layer_destroy(Layer *l) {
    if (!l) return;
    free(l->keys);
    free(l->rates);
    free(l);
}

static void layer_grow(Layer *l, uint64_t need) {
    if (need <= l->cap) return;
    uint64_t cap = l->cap * 2;
    while (cap < need) cap *= 2;
    l->keys = realloc(l->keys, cap * sizeof(etb_board_t));
    l->cap = cap;
}

void layer_append(Layer *l, etb_board_t key) {
    if (l->n == l->cap) layer_grow(l, l->n + 1);
    l->keys[l->n++] = key;
}

void layer_shrink(Layer *l) {
    if (l->n == l->cap) return;
    l->keys = realloc(l->keys, (l->n ? l->n : 1) * sizeof(etb_board_t));
    l->cap = l->n ? l->n : 1;
}

void layer_finalize(Layer *l, uint64_t *removed) {
    uint64_t dupes = 0;
    if (l->n > 1) {
        etb_board_t *tmp = malloc(l->n * sizeof(etb_board_t));
        radix_sort_u64(l->keys, tmp, l->n);
        free(tmp);
        uint64_t w = 1;
        for (uint64_t i = 1; i < l->n; i++) {
            if (l->keys[i] != l->keys[w-1]) l->keys[w++] = l->keys[i];
        }
        dupes = l->n - w;
        l->n = w;
    }
    layer_shrink(l);
    if (removed) *removed = dupes;
}

/* binary search: returns 1 + idx via out_idx, 0 if missing */
int layer_lookup(const Layer *l, etb_board_t key, uint64_t *out_idx) {
    uint64_t lo = 0, hi = l->n;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (l->keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo < l->n && l->keys[lo] == key) {
        if (out_idx) *out_idx = lo;
        return 1;
    }
    return 0;
}

#define LAYER_MAGIC 0x31425445u /* "ETB1" */

int layer_save(const Layer *l, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = LAYER_MAGIC;
    fwrite(&magic, 4, 1, f);
    fwrite(&l->n, 8, 1, f);
    fwrite(l->keys, sizeof(etb_board_t), l->n, f);
    if (l->rates)
        fwrite(l->rates, sizeof(etb_rate_t), l->n, f);
    fclose(f);
    return 0;
}

Layer *layer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != LAYER_MAGIC) { fclose(f); return NULL; }
    Layer *l = calloc(1, sizeof(Layer));
    if (fread(&l->n, 8, 1, f) != 1) { free(l); fclose(f); return NULL; }
    l->cap = l->n;
    l->keys = malloc((l->n ? l->n : 1) * sizeof(etb_board_t));
    fread(l->keys, sizeof(etb_board_t), l->n, f);
    fclose(f);
    return l;
}
