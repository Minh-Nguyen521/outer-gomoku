#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oracle.h"

int main(void) {
  /* their hardcoded special case board — validates our direction mapping */
  etb_board_t sp = (etb_board_t)0x012412567fff8fffULL;
  double out[4];
  oracle_query(sp, 256, out);
  printf("special: L=%.4f R=%.4f U=%.4f D=%.4f (expect L~0.9111)\n",
         out[0], out[1], out[2], out[3]);
  return 0;
}
