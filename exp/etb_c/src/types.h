#ifndef ETB_TYPES_H
#define ETB_TYPES_H

#include <stdint.h>

typedef uint64_t etb_board_t;   /* 4x4 board, 4 bits per cell (exponent 0..15) */
typedef uint16_t etb_row_t;     /* one row, 16 bits (4 nibbles) */
typedef uint16_t etb_rate_t;    /* success rate quantized to 0..65535 (0 == pruned/unknown) */

/* Layer record: board key + success rate.
   Sorted ascending by key; rates parallel array. Kept separate (SoA) for
   cache-friendly binary search (keys only touched during lookup). */

/* Global move LUTs */
extern etb_row_t g_move_left[65536];
extern etb_row_t g_move_right[65536];

void etb_init_luts(void);

/* board <-> row helpers */
static inline etb_row_t etb_row_get(etb_board_t b, int r) {
    return (etb_row_t)((b >> ((3 - r) * 16)) & 0xFFFF);
}

static inline etb_board_t etb_row_set(etb_board_t b, int r, etb_row_t row) {
    int shift = (3 - r) * 16;
    return (b & ~((etb_board_t)0xFFFF << shift)) | ((etb_board_t)row << shift);
}

static inline etb_board_t etb_transpose(etb_board_t b) {
    /* cell index r*4+c (MSB-first nibble order); transpose maps to c*4+r */
    etb_board_t t = 0;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            int nib = (int)((b >> ((15 - (r * 4 + c)) * 4)) & 0xF);
            t |= (etb_board_t)nib << ((15 - (c * 4 + r)) * 4);
        }
    return t;
}

etb_board_t etb_move_dir(etb_board_t b, int dir);
static inline etb_board_t etb_rev16(etb_row_t r) {
    return (etb_row_t)(((r & 0x000F) << 12) | ((r & 0x00F0) << 4) |
                       ((r & 0x0F00) >> 4) | ((r & 0xF000) >> 12));
}

#endif /* ETB_TYPES_H */
