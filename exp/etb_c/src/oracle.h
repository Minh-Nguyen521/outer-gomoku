#ifndef ORACLE_H
#define ORACLE_H

#include <stdint.h>
#include "types.h"

/* Number of layers per table */
#define ORA_256_MAX_LAYER 72
#define ORA_512_MAX_LAYER 248
#define ORA_1256_MAX_LAYER 72

/*
 * Query the ported EGTB tables for the win rate of each move direction.
 * b: our board representation (cell c=r*4+col at nibble shift (15-c)*4).
 * dir 0=L,1=R,2=U,3=D. Returns rate in [lo,hi] of the table, or -1 on miss.
 * out[i] set to -1.0 when move invalid or table miss.
 * table_type: 256 or 512 (1256 available but rarely applicable).
 */
void oracle_query(etb_board_t b, int table_type, double out[4]);

#endif
