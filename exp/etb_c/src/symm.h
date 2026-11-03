#ifndef ETB_SYMM_H
#define ETB_SYMM_H

#include "types.h"

/* 8 board transforms (the D4 group over the 4x4 grid):
   0: identity      1: flip left-right      2: flip up-down    3: LR+UD
   4: transpose     5: transpose o LR       6: transpose o UD  7: transpose o LRUD */

void etb_symm_init(void);
etb_board_t etb_apply_symm(etb_board_t b, int i);
uint16_t etb_apply_symm16(uint16_t mask, int i); /* same perm applied to a 16-bit cell mask */

etb_board_t etb_flip_lr(etb_board_t b);
etb_board_t etb_flip_ud(etb_board_t b);
etb_board_t etb_trans(etb_board_t b);
etb_board_t etb_rotate90(etb_board_t b);
etb_board_t etb_flip_diag(etb_board_t b);
etb_board_t etb_canonical(etb_board_t b);

/* Symmetry subgroup that preserves a cell mask (e.g. the pattern free zone).
   Used to canonicalize boards only under symmetries that keep the pattern
   frame intact -- the same trick as the original's SymmMode selection. */
typedef struct {
    int idx[8];
    int count;
} SymmGroup;

void etb_build_group(uint16_t mask, SymmGroup *g);
etb_board_t etb_canonical_in(etb_board_t b, const SymmGroup *g);

#endif
