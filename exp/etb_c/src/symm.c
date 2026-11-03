#include "symm.h"

/* A transform T is a map cell->cell: the nibble at cell c moves to T(c).
   Boards and 16-bit masks are transformed with the same T, guaranteeing
   consistency between board symmetry and pattern-frame symmetry. */

typedef struct { int t[16]; } Xform;

static etb_board_t xform_board(etb_board_t b, const Xform *x) {
    etb_board_t out = 0;
    for (int c = 0; c < 16; c++) {
        etb_board_t nib = (b >> ((15 - c) * 4)) & 0xFULL;
        out |= nib << ((15 - x->t[c]) * 4);
    }
    return out;
}

static uint16_t xform_mask(uint16_t m, const Xform *x) {
    uint16_t out = 0;
    for (int c = 0; c < 16; c++)
        if ((m >> c) & 1) out |= (uint16_t)(1u << x->t[c]);
    return out;
}

static Xform compose(const Xform *a, const Xform *b) {
    Xform r; /* a o b : apply b first */
    for (int c = 0; c < 16; c++) r.t[c] = a->t[b->t[c]];
    return r;
}

static const Xform IDENT = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}};
static const Xform LR    = {{3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12}};
static const Xform UD    = {{12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3}};
static const Xform TRANS = {{0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15}};

static Xform GROUP[8];
static int group_ready = 0;

static void ensure_group(void) {
    if (group_ready) return;
    GROUP[0] = IDENT;
    GROUP[1] = LR;
    GROUP[2] = UD;
    GROUP[3] = compose(&LR, &UD);
    GROUP[4] = TRANS;
    GROUP[5] = compose(&TRANS, &LR);
    GROUP[6] = compose(&TRANS, &UD);
    GROUP[7] = compose(&TRANS, GROUP + 3);
    group_ready = 1;
}

etb_board_t etb_flip_lr(etb_board_t b)  { return xform_board(b, &LR); }
etb_board_t etb_flip_ud(etb_board_t b)  { return xform_board(b, &UD); }
etb_board_t etb_trans(etb_board_t b)   { return xform_board(b, &TRANS); }
etb_board_t etb_rotate90(etb_board_t b) { return xform_board(b, GROUP + 7); }
etb_board_t etb_flip_diag(etb_board_t b){ return xform_board(b, &TRANS); }

etb_board_t etb_canonical(etb_board_t b) {
    ensure_group();
    etb_board_t m = b;
    for (int i = 1; i < 8; i++) {
        etb_board_t t = xform_board(b, GROUP + i);
        if (t < m) m = t;
    }
    return m;
}

void etb_symm_init(void) { ensure_group(); }

etb_board_t etb_apply_symm(etb_board_t b, int i) {
    ensure_group();
    return xform_board(b, GROUP + (i & 7));
}

uint16_t etb_apply_symm16(uint16_t mask, int i) {
    ensure_group();
    return xform_mask(mask, GROUP + (i & 7));
}

/* ---------- pattern-preserving subgroup ------------------------------ */

void etb_build_group(uint16_t mask, SymmGroup *g) {
    ensure_group();
    int cnt = 0;
    for (int i = 0; i < 8; i++) {
        if (xform_mask(mask, GROUP + i) == mask)
            g->idx[cnt++] = i;
    }
    g->count = cnt;
}

etb_board_t etb_canonical_in(etb_board_t b, const SymmGroup *g) {
    etb_board_t m = b;
    for (int i = 0; i < g->count; i++) {
        etb_board_t t = etb_apply_symm(b, g->idx[i]);
        if (t < m) m = t;
    }
    return m;
}
