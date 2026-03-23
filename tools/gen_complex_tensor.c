/*
 * gen_complex_tensor.c
 *
 * Generic COMPLEX128 rank-4 tensor generator for contraction testing.
 *
 * Usage:  gen_complex_tensor [DIM [CHUNK_SIDE [PREFIX]]]
 *
 *   DIM        global size per index (default: 80)
 *   CHUNK_SIDE chunk size per index  (default: 16)
 *   PREFIX     filename prefix       (default: "small")
 *
 * Creates {PREFIX}_A.h5 and {PREFIX}_B.h5 with:
 *   A shape: (DIM, DIM, DIM, DIM) — dim order (i, j, a, b)
 *   B shape: (DIM, DIM, DIM, DIM) — dim order (a, k, b, l)
 *   Fill: 1.0 + 0.5i everywhere
 *
 * Default small case (DIM=80, CHUNK=16):
 *   Chunk: 16^4 × 16 B = 1 MiB  |  Tiles: 5^4 = 625  |  Size: 655 MB each
 *
 * Expected contraction result (ijab,akbl->klji):
 *   C(k,l,j,i) = DIM^2 × (1+0.5i)^2 = DIM^2 × (0.75 + 1.0i)
 *   For DIM=80: 6400 × 0.75 = 4800.0  +  6400 × 1.0 = 6400.0i
 */

#include "tensor_store.h"
#include "odometer.h"
#include <hdf5.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GEN_RANK    4
#define FILL_REAL   1.0
#define FILL_IMAG   0.5

static int generate_file(const char *fname, const char *dset_name,
                         int dim, int chunk_side,
                         double _Complex fill)
{
    hsize_t shape[GEN_RANK], chunk_dims[GEN_RANK];
    for (int d = 0; d < GEN_RANK; d++) {
        shape[d]      = (hsize_t)dim;
        chunk_dims[d] = (hsize_t)chunk_side;
    }

    size_t elems_per_chunk = 1;
    for (int d = 0; d < GEN_RANK; d++) elems_per_chunk *= (size_t)chunk_side;
    size_t bytes_per_chunk = elems_per_chunk * sizeof(double _Complex);

    size_t n_tiles[GEN_RANK], total_tiles = 1;
    for (int d = 0; d < GEN_RANK; d++) {
        n_tiles[d]   = (size_t)((dim + chunk_side - 1) / chunk_side);
        total_tiles *= n_tiles[d];
    }

    double total_mib = (double)(total_tiles * bytes_per_chunk)
                       / (1024.0 * 1024.0);
    double total_gib = total_mib / 1024.0;

    printf("Generating %s\n"
           "  shape : (%d^%d)   dtype : COMPLEX128\n"
           "  chunk : (%d^%d)   %zu MiB/chunk   %zu tiles\n"
           "  total : %.1f MiB (%.3f GiB)\n",
           fname, dim, GEN_RANK, chunk_side, GEN_RANK,
           bytes_per_chunk / (1024 * 1024), total_tiles,
           total_mib, total_gib);

    if (create_chunked_dataset_einsum(fname, dset_name, GEN_RANK,
                                      shape, chunk_dims,
                                      DTYPE_COMPLEX128) < 0) {
        fprintf(stderr, "gen: create_chunked_dataset_einsum failed '%s'\n",
                fname);
        return -1;
    }

    hid_t fid = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (fid < 0) { fprintf(stderr, "gen: H5Fopen failed\n"); return -1; }

    hid_t dset = dset_open_no_cache(fid, dset_name);
    if (dset < 0) {
        fprintf(stderr, "gen: dset_open_no_cache failed\n");
        H5Fclose(fid);
        return -1;
    }

    hid_t h5ctype = create_h5_complex_type();
    if (h5ctype < 0) {
        fprintf(stderr, "gen: create_h5_complex_type failed\n");
        H5Dclose(dset); H5Fclose(fid);
        return -1;
    }

    double _Complex *buf = (double _Complex *)malloc(bytes_per_chunk);
    if (!buf) {
        fprintf(stderr, "gen: malloc failed\n");
        H5Tclose(h5ctype); H5Dclose(dset); H5Fclose(fid);
        return -1;
    }
    for (size_t i = 0; i < elems_per_chunk; i++) buf[i] = fill;

    size_t tile_idx[GEN_RANK], grid_sz[GEN_RANK];
    memset(tile_idx, 0, sizeof(tile_idx));
    for (int d = 0; d < GEN_RANK; d++) grid_sz[d] = n_tiles[d];

    int    ret     = 0;
    size_t written = 0;

    struct timespec t0, tnow;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    do {
        hsize_t offset[GEN_RANK];
        for (int d = 0; d < GEN_RANK; d++)
            offset[d] = (hsize_t)tile_idx[d] * chunk_dims[d];

        if (write_chunk_typed(dset, offset, buf,
                              sizeof(double _Complex), GEN_RANK,
                              chunk_dims, h5ctype) < 0) {
            fprintf(stderr,
                    "gen: write_chunk_typed failed at tile "
                    "[%zu,%zu,%zu,%zu]\n",
                    tile_idx[0], tile_idx[1], tile_idx[2], tile_idx[3]);
            ret = -1;
            break;
        }

        written++;
        if (written % 50 == 0 || written == total_tiles) {
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            double elapsed = (double)(tnow.tv_sec - t0.tv_sec)
                           + (double)(tnow.tv_nsec - t0.tv_nsec) * 1e-9;
            double pct = 100.0 * (double)written / (double)total_tiles;
            double mib = (double)(written * bytes_per_chunk)
                         / (1024.0 * 1024.0);
            double bw  = (elapsed > 0.0) ? mib / elapsed : 0.0;
            printf("\r  %5.1f%%  %4zu/%zu tiles  %7.1f MiB  "
                   "%5.1f s  %6.1f MiB/s",
                   pct, written, total_tiles, mib, elapsed, bw);
            fflush(stdout);
        }

    } while (odometer_step(GEN_RANK, tile_idx, grid_sz));

    printf("\n");
    free(buf);
    H5Tclose(h5ctype);
    H5Dclose(dset);
    H5Fclose(fid);

    if (ret == 0) {
        struct timespec tf;
        clock_gettime(CLOCK_MONOTONIC, &tf);
        double elapsed = (double)(tf.tv_sec - t0.tv_sec)
                       + (double)(tf.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("  done  %.1f MiB  %.2f s  %.1f MiB/s\n",
               total_mib, elapsed,
               (elapsed > 0.0) ? total_mib / elapsed : 0.0);
    }
    return ret;
}

int main(int argc, char **argv)
{
    int         dim        = 80;
    int         chunk_side = 16;
    const char *prefix     = "small";

    if (argc > 1) dim        = atoi(argv[1]);
    if (argc > 2) chunk_side = atoi(argv[2]);
    if (argc > 3) prefix     = argv[3];

    if (dim <= 0 || chunk_side <= 0) {
        fprintf(stderr,
                "Usage: gen_complex_tensor [DIM [CHUNK_SIDE [PREFIX]]]\n");
        return 1;
    }

    printf("=== Complex Tensor Generator ===\n");
    printf("DIM=%d  CHUNK=%d  PREFIX=%s\n", dim, chunk_side, prefix);
    printf("Fill: %.1f + %.1fi\n\n", FILL_REAL, FILL_IMAG);

    double expected_real = (double)(dim * dim)
                           * (FILL_REAL * FILL_REAL - FILL_IMAG * FILL_IMAG);
    double expected_imag = (double)(dim * dim)
                           * 2.0 * FILL_REAL * FILL_IMAG;
    printf("Expected contraction result (ijab,akbl->klji):\n"
           "  C = %.1f + %.1fi  (every element)\n\n",
           expected_real, expected_imag);

    /* Build file names: {prefix}_A.h5, {prefix}_B.h5 */
    char fname_A[256], fname_B[256];
    snprintf(fname_A, sizeof(fname_A), "%s_A.h5", prefix);
    snprintf(fname_B, sizeof(fname_B), "%s_B.h5", prefix);

    if (generate_file(fname_A, "tensor",
                      dim, chunk_side,
                      CMPLX(FILL_REAL, FILL_IMAG)) < 0)
        return 1;
    printf("\n");
    if (generate_file(fname_B, "tensor",
                      dim, chunk_side,
                      CMPLX(FILL_REAL, FILL_IMAG)) < 0)
        return 1;

    printf("\nDone. %s and %s are ready.\n", fname_A, fname_B);
    return 0;
}
