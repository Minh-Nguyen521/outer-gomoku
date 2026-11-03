#ifndef ETB_PACK_H
#define ETB_PACK_H

#include <stdint.h>
#include <stddef.h>

/* pack: compress every layer_XXXX.bin in-place (renames to .lzma).
   unpack: restores them. Sizes printed by caller. Returns 0 on success. */
int etb_pack(const char *dir);
int etb_unpack(const char *dir);

/* shared helpers (also used by gen.c to lazily load .lzma layers) */
uint8_t *etb_slurp_file(const char *path, size_t *out_sz);
uint8_t *etb_lzma_decompress(const uint8_t *in, size_t in_sz, size_t *out_sz);

#endif
