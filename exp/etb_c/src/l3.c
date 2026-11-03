/* l3.c: faithful port of nnn2048's L3Manager endgame EGTB probing.
 *
 * Their board convention is identical to ours (row chunk 16-bit, nibble at
 * bit 4*i = cell 15-i of our row-major MSB-first indexing, col 0 at MSB).
 * Directions: their 1,2,3,4 = our 0,1,2,3 (L,R,U,D).
 */
#include "l3.h"
#include "oracle.h"
#include "types.h"
#include <stdint.h>
#include <string.h>

long l3_dbg_calls, l3_dbg_threshold, l3_dbg_probe_calls, l3_dbg_hit;
long l3_dbg_gate_fail;
long l3_dbg_gate_pass, l3_dbg_query_fail;

/* ---------- symmetry transforms (Calculator.py ports, u64 bit tricks) ---- */
static uint64_t reverse_lr(uint64_t b) {
    b = (b & 0xFF00FF00FF00FF00ULL) >> 8 | (b & 0x00FF00FF00FF00FFULL) << 8;
    b = (b & 0xF0F0F0F0F0F0F0F0ULL) >> 4 | (b & 0x0F0F0F0F0F0F0F0FULL) << 4;
    return b;
}
static uint64_t reverse_ud(uint64_t b) {
    b = (b & 0xFFFFFFFF00000000ULL) >> 32 | (b & 0x00000000FFFFFFFFULL) << 32;
    b = (b & 0xFFFF0000FFFF0000ULL) >> 16 | (b & 0x0000FFFF0000FFFFULL) << 16;
    return b;
}
static uint64_t reverse_ul(uint64_t b) { /* main-diagonal mirror = transpose */
    b = (b & 0xFF00FF0000FF00FFULL) | ((b & 0x00FF00FF00000000ULL) >> 24) |
        ((b & 0x00000000FF00FF00ULL) << 24);
    b = (b & 0xF0F00F0FF0F00F0FULL) | ((b & 0x0F0F00000F0F0000ULL) >> 12) |
        ((b & 0x0000F0F00000F0F0ULL) << 12);
    return b;
}
static uint64_t reverse_ur(uint64_t b) { /* anti-diagonal mirror */
    b = (b & 0x0F0FF0F00F0FF0F0ULL) | ((b & 0xF0F00000F0F00000ULL) >> 20) |
        ((b & 0x00000F0F00000F0FULL) << 20);
    b = (b & 0x00FF00FFFF00FF00ULL) | ((b & 0xFF00FF0000000000ULL) >> 40) |
        ((b & 0x0000000000FF00FFULL) << 40);
    return b;
}
static uint64_t rotate180(uint64_t b) {
    b = (b & 0xFFFFFFFF00000000ULL) >> 32 | (b & 0x00000000FFFFFFFFULL) << 32;
    b = (b & 0xFFFF0000FFFF0000ULL) >> 16 | (b & 0x0000FFFF0000FFFFULL) << 16;
    b = (b & 0xFF00FF00FF00FF00ULL) >> 8 | (b & 0x00FF00FF00FF00FFULL) << 8;
    b = (b & 0xF0F0F0F0F0F0F0F0ULL) >> 4 | (b & 0x0F0F0F0F0F0F0F0FULL) << 4;
    return b;
}
static uint64_t rotate_l(uint64_t b) {
    b = (b & 0xFF00FF0000000000ULL) >> 32 | (b & 0x00FF00FF00000000ULL) << 8 |
        ((b & 0x00000000FF00FF00ULL) >> 8) | ((b & 0x0000000000FF00FFULL) << 32);
    b = (b & 0xF0F00000F0F00000ULL) >> 16 | (b & 0x0F0F00000F0F0000ULL) << 4 |
        ((b & 0x0000F0F00000F0F0ULL) >> 4) | ((b & 0x00000F0F00000F0FULL) << 16);
    return b;
}
static uint64_t rotate_r(uint64_t b) {
    b = (b & 0xFF00FF0000000000ULL) >> 8 | (b & 0x00FF00FF00000000ULL) >> 32 |
        ((b & 0x00000000FF00FF00ULL) << 32) | ((b & 0x0000000000FF00FFULL) << 8);
    b = (b & 0xF0F00000F0F00000ULL) >> 4 | (b & 0x0F0F00000F0F0000ULL) >> 16 |
        ((b & 0x0000F0F00000F0F0ULL) << 16) | ((b & 0x00000F0F00000F0FULL) << 4);
    return b;
}

/* move mapping: map_move[sym][their_dir 1-4] = original their_dir 1-4 */
static const int MAP_MOVE[8][5] = {
    {0, 1, 2, 3, 4}, /* identity */
    {0, 3, 4, 2, 1}, /* rotateL */
    {0, 2, 1, 4, 3}, /* rotate180 */
    {0, 4, 3, 1, 2}, /* rotateR */
    {0, 2, 1, 3, 4}, /* reverseLR */
    {0, 1, 2, 4, 3}, /* reverseUD */
    {0, 3, 4, 1, 2}, /* reverseUL */
    {0, 4, 3, 2, 1}, /* reverseUR */
};
static uint64_t (*const SYMS[8])(uint64_t) = {
    NULL, rotate_l, rotate180, rotate_r, reverse_lr, reverse_ud, reverse_ul,
    reverse_ur,
};

static uint64_t mask_large_tiles(uint64_t b, int threshold, int mask) {
    uint64_t res = 0;
    for (int i = 0; i < 16; i++) {
        uint64_t v = (b >> (4 * i)) & 0xF;
        if ((int)v >= threshold) v = (uint64_t)mask;
        res |= v << (4 * i);
    }
    return res;
}

/* move mapping handled via MAP_MOVE; etb_move_dir from types.h/board.c */

/* query one masked+symmed board; returns mapped-original-dir rates
 * (index 0..3 = L,R,U,D in original frame) or 0 on miss */
static int probe_one(uint64_t mb, int sym, int table_type, double rates[4]) {
    l3_dbg_probe_calls++;
    rates[0] = rates[1] = rates[2] = rates[3] = -1.0;
    double wr[4];
    oracle_query((etb_board_t)mb, table_type, wr);
    int hit = 0;
    for (int d = 1; d <= 4; d++) {
        double r = wr[d - 1];
        if (r > 0.0) {
            rates[MAP_MOVE[sym][d] - 1] = r;
            hit = 1;
        }
    }
    return hit;
}

/* probe_44_128 sub-case: 4x4-with-128 top-half trick */
static int probe_44_128(uint64_t masked, int sym, int board_sum,
                        double rates[4]) {
    int sum512 = board_sum % 512;
    if (!(390 < sum512 && sum512 < 480 && board_sum > 63000)) return 0;
    uint64_t remask = mask_large_tiles(masked, 7, 0xF);
    if ((remask & 0xFFFFFFFFULL) != 0xFFFFFFFFULL) return 0;
    uint64_t rb = (reverse_lr(remask) & 0xFFFFFFFF00000000ULL) + 0x7FFF8FFFULL;
    double wr[4];
    oracle_query((etb_board_t)rb, 512, wr);
    if (wr[0] > 0 || wr[1] > 0 || wr[2] > 0 || wr[3] > 0) {
        double t = wr[0]; wr[0] = wr[1]; wr[1] = t; /* their LR swap */
        rates[0] = rates[1] = rates[2] = rates[3] = -1.0;
        for (int d = 1; d <= 4; d++)
            if (wr[d - 1] > 0.0) rates[MAP_MOVE[sym][d] - 1] = wr[d - 1];
        return 1;
    }
    return 0;
}

/* probe_L3 */
static int probe_l3(uint64_t masked, const int *ttypes, int ntt,
                    int board_sum, double rates[4]) {
    if (65280 < board_sum && board_sum < 65436) {
        masked = mask_large_tiles(masked, 8, 0xF);
        int t1256[] = {1256};
        return probe_l3(masked, t1256, 1, board_sum, rates);
    }
    if (65000 < board_sum && board_sum < 65280) {
        masked = mask_large_tiles(masked, 9, 0x8);
        int t512[] = {512};
        return probe_l3(masked, t512, 1, board_sum, rates);
    }

    for (int s = 0; s < 8; s++) {
        uint64_t b = s ? SYMS[s](masked) : masked;
        if ((b & 0x000000000FFF0FFFULL) != 0x000000000FFF0FFFULL) continue;
        l3_dbg_gate_pass++;
        for (int t = 0; t < ntt; t++) {
            int tt = ttypes[t];
            if (tt == 512) {
                double r44[4];
                if (probe_44_128(b, s, board_sum, r44)) {
                    memcpy(rates, r44, sizeof(double) * 4);
                    return 1;
                }
            }
            double r[4];
            if (probe_one(b, s, tt, r)) {
                l3_dbg_hit++;
                memcpy(rates, r, sizeof(double) * 4);
                return 1;
            }
            l3_dbg_query_fail++;
        }
    }
    return 0;
}

/* probe_441: 7-large-tile layered probe */
static int probe_441(uint64_t board, int threshold, int board_sum,
                     double rates[4]) {
    uint64_t masked = mask_large_tiles(board, threshold, threshold);
    for (int s = 0; s < 8; s++) {
        uint64_t b = s ? SYMS[s](masked) : masked;
        uint64_t want = 0x1110111ULL * (uint64_t)threshold;
        if ((b & 0x000000000FFF0FFFULL) == want)
            b |= 0x000000000FFF0FFFULL;
        else
            continue;
        double r44[4];
        if (probe_44_128(b, s, board_sum, r44)) {
            memcpy(rates, r44, sizeof(double) * 4);
            return 1;
        }
        double r[4];
        if (probe_one(b, s, 512, r)) {
            memcpy(rates, r, sizeof(double) * 4);
            return 1;
        }
    }
    return 0;
}





int l3_pick_move(etb_board_t b) {
    uint64_t ub = (uint64_t)b;

    /* board_sum = raw sum of 2^exp over nonzero cells (matching their impl) */
    int board_sum = 0;
    int counts[16] = {0};
    for (int i = 0; i < 16; i++) {
        int v = (int)((ub >> (4 * i)) & 0xF);
        counts[v]++;
        if (v > 0) board_sum += 1 << v;
    }

    /* ---- probe(): endgame recognition ---- */
    int threshold = 0;
    int c10 = 0, mx_high = 0, c9 = 0, c8 = 0, c7 = 0;
    for (int v = 10; v < 16; v++) c10 += counts[v];
    for (int v = 10; v < 15; v++) if (counts[v] > mx_high) mx_high = counts[v];
    for (int v = 9; v < 16; v++) c9 += counts[v];
    for (int v = 8; v < 16; v++) c8 += counts[v];
    for (int v = 7; v < 16; v++) c7 += counts[v];
    int mx9 = 0, mx8 = 0, mx7 = 0;
    for (int v = 9; v < 15; v++) if (counts[v] > mx9) mx9 = counts[v];
    for (int v = 8; v < 15; v++) if (counts[v] > mx8) mx8 = counts[v];
    for (int v = 7; v < 15; v++) if (counts[v] > mx7) mx7 = counts[v];

    int bs1024 = board_sum % 1024, bs512 = board_sum % 512;
    int bs256 = board_sum % 256, bs128 = board_sum % 128;

    if (c10 == 6 && mx_high == 1) {
        if (!((bs1024 < 480) && counts[9] == 1)) {
            if (bs1024 > 96) threshold = 10;
            else if (bs1024 > 60) threshold = 8;
        }
    } else if (c9 == 6 && mx9 == 1) {
        if (!((bs512 < 240) && counts[8] == 1)) threshold = 9;
    } else if (c8 == 6 && mx8 == 1 && bs256 < 240) {
        int skip = (bs256 < 120 && counts[7] == 1) ||
                   (c10 == 5 && counts[9] == 0 && bs256 < 64);
        if (!skip) threshold = 8;
    } else if (c7 == 6 && mx7 == 1 && bs128 > 20 && bs128 < 120 &&
               counts[7] == 1) {
        if (!((bs128 < 60) && counts[6] == 1)) threshold = 7;
    }

    if (threshold == 0) { l3_dbg_calls++; l3_dbg_gate_fail++; goto cascade; }
    l3_dbg_threshold++;

    {
        uint64_t masked = mask_large_tiles(ub, threshold, 0xF);
        int ttypes[2];
        int ntt;
        if (threshold == 10 && bs1024 < 128) { ttypes[0] = 256; ttypes[1] = 512; ntt = 2; }
        else if (threshold > 8 && (bs256 > 128 || bs512 < 72)) { ttypes[0] = 512; ntt = 1; }
        else if (threshold > 8) { ttypes[0] = 512; ttypes[1] = 256; ntt = 2; }
        else if ((threshold == 8 && bs256 > 60) || (threshold == 7 && bs128 > 60)) {
            ttypes[0] = 256; ttypes[1] = 512; ntt = 2;
        } else { ttypes[0] = 256; ntt = 1; }

        double rates[4];
        if (probe_l3(masked, ttypes, ntt, board_sum, rates)) {
            double best = 0; int bd = -1;
            for (int d = 0; d < 4; d++)
                if (rates[d] > best) { best = rates[d]; bd = d; }
            if (best > 0) return bd;
        }
    }

cascade:;
    /* generalized layered cascade for 7 large tiles */
    if (c8 == 7 && mx8 == 1 && bs256 < 240) {
        double rates[4];
        if (probe_441(ub, 8, board_sum, rates)) {
            double best = 0; int bd = -1;
            for (int d = 0; d < 4; d++)
                if (rates[d] > best) { best = rates[d]; bd = d; }
            if (best > 0) return bd;
        }
    }
    if (c7 == 7 && mx7 == 1 && bs128 < 120) {
        double rates[4];
        if (probe_441(ub, 7, board_sum, rates)) {
            double best = 0; int bd = -1;
            for (int d = 0; d < 4; d++)
                if (rates[d] > best) { best = rates[d]; bd = d; }
            if (best > 0) return bd;
        }
    }
    return -1;
}
