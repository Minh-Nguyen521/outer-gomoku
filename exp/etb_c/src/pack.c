#include "pack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <lzma.h>

uint8_t *etb_slurp_file(const char *path, size_t *out_sz) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    if (sz < 0) { fclose(f); return NULL; }
    uint8_t *buf = malloc(sz ? (size_t)sz : 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_sz = (size_t)sz;
    return buf;
}

static int spit(const char *path, const uint8_t *buf, size_t sz) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (sz && fwrite(buf, 1, sz, f) != sz) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

static uint8_t *lzma_compress_buf(const uint8_t *in, size_t in_sz, size_t *out_sz) {
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_easy_encoder(&strm, LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64) != LZMA_OK)
        return NULL;
    size_t cap = in_sz + in_sz / 2 + 4096;
    uint8_t *out = malloc(cap);
    strm.next_in = in; strm.avail_in = in_sz;
    strm.next_out = out; strm.avail_out = cap;
    for (;;) {
        lzma_ret r = lzma_code(&strm, LZMA_FINISH);
        if (r == LZMA_STREAM_END) break;
        if (r != LZMA_OK) { free(out); lzma_end(&strm); return NULL; }
        if (r == LZMA_OK && strm.avail_out == 0) { free(out); lzma_end(&strm); return NULL; }
    }
    lzma_end(&strm);
    *out_sz = strm.total_out;
    return out;
}

uint8_t *etb_lzma_decompress(const uint8_t *in, size_t in_sz, size_t *out_sz) {
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_auto_decoder(&strm, UINT64_MAX, 0) != LZMA_OK) return NULL;
    size_t cap = in_sz * 4 + 4096;
    uint8_t *out = malloc(cap);
    strm.next_in = in; strm.avail_in = in_sz;
    for (;;) {
        strm.next_out = out + strm.total_out;
        strm.avail_out = cap - strm.total_out;
        lzma_ret r = lzma_code(&strm, LZMA_RUN);
        if (r == LZMA_STREAM_END) break;
        if (r == LZMA_OK && strm.avail_out == 0) { /* grow */
            cap *= 2;
            uint8_t *nout = realloc(out, cap);
            if (!nout) { free(out); lzma_end(&strm); return NULL; }
            out = nout;
            continue;
        }
        if (r != LZMA_OK) { free(out); lzma_end(&strm); return NULL; }
    }
    lzma_end(&strm);
    *out_sz = strm.total_out;
    return out;
}

static int has_ext(const char *name, const char *ext) {
    size_t nl = strlen(name), el = strlen(ext);
    return nl > el && !strcmp(name + nl - el, ext);
}

static uint64_t pack_dir(const char *dir, int compress) {
    DIR *d = opendir(dir);
    if (!d) return (uint64_t)-1;
    struct dirent *ent;
    uint64_t files = 0, raw_bytes = 0, comp_bytes = 0;
    while ((ent = readdir(d)) != NULL) {
        char in[512], out[512];
        if (compress) {
            if (strncmp(ent->d_name, "layer_", 6) || has_ext(ent->d_name, ".lzma")) continue;
            snprintf(in, sizeof in, "%s/%s", dir, ent->d_name);
            snprintf(out, sizeof out, "%s.lzma", in);
        } else {
            if (!has_ext(ent->d_name, ".lzma")) continue;
            snprintf(in, sizeof in, "%s/%s", dir, ent->d_name);
            snprintf(out, sizeof out, "%.*s", (int)(strlen(in) - 5), in);
        }
        size_t in_sz;
        uint8_t *inbuf = etb_slurp_file(in, &in_sz);
        if (!inbuf) continue;
        size_t out_sz;
        uint8_t *outbuf = compress ? lzma_compress_buf(inbuf, in_sz, &out_sz)
                                   : etb_lzma_decompress(inbuf, in_sz, &out_sz);
        free(inbuf);
        if (!outbuf) { closedir(d); return (uint64_t)-1; }
        if (spit(out, outbuf, out_sz) != 0) { free(outbuf); closedir(d); return (uint64_t)-1; }
        raw_bytes += in_sz; comp_bytes += out_sz;
        free(outbuf);
        remove(in);
        files++;
    }
    closedir(d);
    if (files)
        printf("[%s] %llu files, %llu -> %llu bytes (%.1f%%)\n",
               compress ? "pack" : "unpack", (unsigned long long)files,
               (unsigned long long)(compress ? raw_bytes : comp_bytes),
               (unsigned long long)(compress ? comp_bytes : raw_bytes),
               raw_bytes ? 100.0 * comp_bytes / raw_bytes : 0.0);
    return files;
}

int etb_pack(const char *dir)   { return pack_dir(dir, 1) == (uint64_t)-1 ? -1 : 0; }
int etb_unpack(const char *dir) { return pack_dir(dir, 0) == (uint64_t)-1 ? -1 : 0; }
