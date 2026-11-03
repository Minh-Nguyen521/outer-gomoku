#include "types.h"

etb_row_t g_move_left[65536];
etb_row_t g_move_right[65536];

static etb_row_t slide_left(etb_row_t row) {
    unsigned char v[4], out[4] = {0,0,0,0};
    int n = 0;
    for (int i = 0; i < 4; i++) {
        unsigned char nib = (row >> ((3 - i) * 4)) & 0xF;
        if (nib) v[n++] = nib;
    }
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (i + 1 < n && v[i] == v[i+1] && v[i] < 15) {
            out[m++] = (unsigned char)(v[i] + 1);
            i++;
        } else {
            out[m++] = v[i];
        }
    }
    etb_row_t res = 0;
    for (int i = 0; i < 4; i++)
        res |= (etb_row_t)out[i] << ((3 - i) * 4);
    return res;
}

static etb_row_t rev16(etb_row_t r) {
    return (etb_row_t)(((r & 0x000F) << 12) | ((r & 0x00F0) << 4) |
                       ((r & 0x0F00) >> 4) | ((r & 0xF000) >> 12));
}

void etb_init_luts(void) {
    for (uint32_t r = 0; r < 65536; r++) {
        etb_row_t row = (etb_row_t)r;
        g_move_left[r] = slide_left(row);
        /* slide right = reverse, slide left, reverse back */
        g_move_right[r] = rev16(slide_left(rev16(row)));
    }
}

etb_board_t etb_move_dir(etb_board_t b, int dir) {
    /* dir: 0=left 1=right 2=up 3=down (row-major cells, MSB=row0cell0) */
    if (dir == 0 || dir == 1) {
        etb_board_t out = 0;
        for (int r = 0; r < 4; r++) {
            etb_row_t row = etb_row_get(b, r);
            etb_row_t nr = (dir == 0) ? g_move_left[row] : g_move_right[row];
            out = etb_row_set(out, r, nr);
        }
        return out;
    }
    /* up/down = transpose, slide left/right, transpose back */
    etb_board_t t = etb_transpose(b);
    etb_board_t out = 0;
    for (int r = 0; r < 4; r++) {
        etb_row_t row = etb_row_get(t, r);
        etb_row_t nr = (dir == 2) ? g_move_left[row] : g_move_right[row];
        out = etb_row_set(out, r, nr);
    }
    return etb_transpose(out);
}
