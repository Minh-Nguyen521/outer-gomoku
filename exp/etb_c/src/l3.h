#ifndef L3_H
#define L3_H

#include "types.h"

/* Full port of nnn2048 L3Manager endgame oracle pipeline.
 * Returns best move dir (0=L,1=R,2=U,3=D) or -1 if no confident hit. */
int l3_pick_move(etb_board_t b);

#endif
