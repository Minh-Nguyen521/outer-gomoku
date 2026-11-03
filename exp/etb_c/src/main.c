/* etb: tablebase generator CLI (reimplementation of 2048EndgameTablebase
 * classic pipeline in portable C). Usage:
 *   etb gen    <seedspec> <empty_mask_hex> <outdir>
 *   etb solve  <outdir> <target_exp>
 *   etb query  <outdir> <board_hex>
 * seedspec: "free9" style not parsed yet; pass explicit hex seed boards
 * comma-separated. empty-mask hex = bitmask of FREE cells (cell 0 = top-left).
 */
#include "gen.h"
#include "pack.h"
#include "symm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  etb gen <seed_hex,...> <free_mask_hex> <outdir> [spawn4_rate] [max_sum] [stsl] [prune_below] [use_symm]\n"
        "  etb solve <outdir> <target_exp>\n"
        "  etb query <outdir> <board_hex> <target_exp>\n"
        "  etb pack <outdir> | etb unpack <outdir>\n");
    exit(2);
}

int main(int argc, char **argv) {
    etb_init_luts();
    if (argc < 2) usage();

    if (!strcmp(argv[1], "gen")) {
        if (argc < 5) usage();
        /* parse seeds */
        etb_board_t seeds[256];
        int n = 0;
        char *s = strdup(argv[2]);
        for (char *tok = strtok(s, ","); tok && n < 256; tok = strtok(NULL, ","))
            seeds[n++] = strtoull(tok, NULL, 16);
        PatternSpec spec;
        memset(&spec, 0, sizeof spec);
        spec.fixed_occupied = ~strtoull(argv[3], NULL, 16) & 0xFFFFULL;
        spec.num_empty = __builtin_popcountll(~spec.fixed_occupied & 0xFFFFULL);
        double sr = argc > 5 ? atof(argv[5]) : 0.1;
        uint64_t max_sum = argc > 6 ? strtoull(argv[6], NULL, 0) : 0;
        spec.max_sum = max_sum;
        spec.stsl_limit = argc > 7 ? strtoull(argv[7], NULL, 0) : 0;
        spec.prune_below = argc > 8 ? atof(argv[8]) : 0.0;
        spec.use_symmetry = argc > 9 ? atoi(argv[9]) : 1;
        int rc = etb_generate(seeds, n, &spec, sr, argv[4]);
        printf("generated %d layers into %s\n", rc, argv[4]);
        return rc < 0;
    }

    if (!strcmp(argv[1], "solve")) {
        if (argc < 4) usage();
        int rc = etb_solve(argv[2], atoi(argv[3]));
        printf("solved (rc=%d)\n", rc);
        return rc < 0;
    }

    if (!strcmp(argv[1], "pack") || !strcmp(argv[1], "unpack")) {
        if (argc < 3) usage();
        int rc = !strcmp(argv[1], "pack") ? etb_pack(argv[2]) : etb_unpack(argv[2]);
        if (rc) { fprintf(stderr, "%s failed\n", argv[1]); return 1; }
        printf("%s ok\n", argv[1]);
        return 0;
    }

    if (!strcmp(argv[1], "query")) {
        if (argc < 4) usage();
        etb_rate_t r = etb_query(argv[2], strtoull(argv[3], NULL, 16), atoi(argv[4]));
        printf("rate = %u/65535 = %.4f%%\n", r, r * 100.0 / 65535.0);
        return 0;
    }

    usage();
    return 0;
}
