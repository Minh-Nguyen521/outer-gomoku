#ifndef ETB_LAYER_H
#define ETB_LAYER_H

#include "types.h"
#include <stdio.h>

/* One BFS layer: sorted keys + parallel success-rate array (SoA). */
typedef struct {
    etb_board_t *keys;     /* sorted ascending */
    etb_rate_t  *rates;    /* parallel; filled during solve phase */
    uint64_t     n;        /* count */
    uint64_t     cap;      /* capacity */
} Layer;

Layer *layer_create(uint64_t cap_hint);
void   layer_destroy(Layer *l);
void   layer_shrink(Layer *l);
int    layer_lookup(const Layer *l, etb_board_t key, uint64_t *out_idx);

/* append raw then finalize (sort + dedupe); returns dedupe stats */
void   layer_append(Layer *l, etb_board_t key);
void   layer_finalize(Layer *l, uint64_t *removed);

/* persist: simple binary format [magic][n][keys][rates] */
int    layer_save(const Layer *l, const char *path);
Layer *layer_load(const char *path);

#endif
