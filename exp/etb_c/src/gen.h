#ifndef ETB_GEN_H
#define ETB_GEN_H

#include "types.h"
#include "layer.h"
#include "symm.h"
#include "layer.h"

/* load layer file (with .lzma auto-decompression) */
Layer *load_layer(const char *outdir, int i);

/* Pattern = allowed empty-cell shape expressed as a mask over occupied cells.
   mask & board: large tiles stay (masked to 0xF by caller convention),
   pattern_masks[i] & b must be nonzero iff cells required occupied are occupied.
   Simplified: pattern constrains which cells may be EMPTY. */

typedef struct {
    /* occupancy template: bit i set = cell i must be OCCUPIED by a large tile (value >= 4, i.e. >=16) */
    uint64_t fixed_occupied;
    int      num_empty;      /* number of free cells (for naming/reporting) */
    uint64_t max_sum;        /* 0 = unlimited; else stop when board sum exceeds */
    uint64_t stsl_limit;    /* small-tile sum limit: prune spawn-2 branches when
                               sum of tiles <= 4 exceeds this (original's STSL) */
    double   prune_below;   /* prune states whose converged rate falls below
                               this fraction (0 = keep all; original ~0.05) */
    int      use_symmetry;  /* canonicalize under pattern-preserving subgroup */
} PatternSpec;

/* Generate all reachable layers by BFS forward (per help.md 6.2.1).
   Writes layer files layer_%04d.bin in outdir.
   Returns number of layers, or -1 on error. */
int etb_generate(const etb_board_t *seeds, int n_seeds,
                 const PatternSpec *spec, double spawn_rate4,
                 const char *outdir);

/* Backward DP solve: reads layers from outdir, computes success rates for
   target tile appearing at any position (target exponent), saves back. */
int etb_solve(const char *outdir, int target_exp);

/* Query rate for a board: canonicalize, search the right layer by sum. */
etb_rate_t etb_query(const char *outdir, etb_board_t board, int target_exp);

#endif
