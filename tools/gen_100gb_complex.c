/*
 * gen_100gb_complex.c
 *
 * Generates A_100gb.h5 and B_100gb.h5 — rank-4 COMPLEX128 HDF5 tensors for
 * the 100 GB per-tensor stress test of run_contraction_einsum.
 *
 * Layout
 * ------
 *   A  shape: (i=280, j=280, a=280, b=280)
 *   B  shape: (a=280, k=280, b=280, l=280)
 *
 * Both tensors have the same global shape (280^4) and use 32^4 chunks
 * (32^4 × 16 bytes = 16 MiB per chunk, ceil(280/32)^4 = 9^4 = 6561 tiles).
 *
 * Fill value: 1.0 + 0.5i everywhere.
 * Expected contraction result: 78400 × (0.75 + 1.0i) = 58800 + 78400i.
 *
 * Only one chunk buffer (16 MiB) is held in RAM at any time.
 */

#include "tensor_store.h"
#include "odometer.h"
#include <hdf5.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GEN_RANK       4
#define GLOBAL_DIM     280
#define CHUNK_SIDE     32     /* 32^4 × 16 B = 16 MiB per chunk */
#define FILL_REAL      1.0
#define FILL_IMAG      0.5

/* ----------------------------------------------------------------------- */
/* generate_file                                                             */
/* ----------------------------------------------------------------------- */

static int generate_file(const char *fname, const char *dset_name,
                          double _Complex fill)
{
    hsize_t shape[GEN_RANK], chunk_dims[GEN_RANK];
    for (int d = 0; d < GEN_RANK; d++) {
        shape[d]      = GLOBAL_DIM;
        chunk_dims[d] = CHUNK_SIDE;
    }

    size_t elems_per_chunk = 1;
    for (int d = 0; d < GEN_RANK; d++) elems_per_chunk *= CHUNK_SIDE;
    size_t bytes_per_chunk = elems_per_chunk * sizeof(double _Complex);

    size_t n_tiles[GEN_RANK], total_tiles = 1;
    for (int d = 0; d < GEN_RANK; d++) {
        n_tiles[d]   = (GLOBAL_DIM + CHUNK_SIDE - 1) / CHUNK_SIDE;
        total_tiles *= n_tiles[d];
    }

    double total_gb  = (double)(total_tiles * bytes_per_chunk) / 1e9;
    double total_gib = (double)(total_tiles * bytes_per_chunk)
                       / (1024.0 * 1024.0 * 1024.0);

    printf("Generating %s\n"
           "  shape : (%d^%d)   dtype : COMPLEX128\n"
           "  chunk : (%d^%d) = %zu MiB  tiles : %zu\n"
           "  total : %.2f GB  (%.2f GiB)\n",
           fname, GLOBAL_DIM, GEN_RANK, CHUNK_SIDE, GEN_RANK,
           bytes_per_chunk / (1024 * 1024), total_tiles,
           total_gb, total_gib);

    /* Create the dataset. */
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

    /* One chunk buffer, pre-filled. */
    double _Complex *buf = (double _Complex *)malloc(bytes_per_chunk);
    if (!buf) {
        fprintf(stderr, "gen: malloc failed (%zu MiB)\n",
                bytes_per_chunk / (1024 * 1024));
        H5Tclose(h5ctype); H5Dclose(dset); H5Fclose(fid);
        return -1;
    }
    for (size_t i = 0; i < elems_per_chunk; i++) buf[i] = fill;

    /* Odometer over the tile grid. */
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
        if (written % 250 == 0 || written == total_tiles) {
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            double elapsed = (double)(tnow.tv_sec - t0.tv_sec)
                           + (double)(tnow.tv_nsec - t0.tv_nsec) * 1e-9;
            double pct = 100.0 * (double)written / (double)total_tiles;
            double gib = (double)(written * bytes_per_chunk)
                         / (1024.0 * 1024.0 * 1024.0);
            double bw  = (elapsed > 0.0)
                         ? gib / elapsed : 0.0;
            printf("\r  %5.1f%%  %5zu/%zu tiles  %6.1f GiB  "
                   "%5.1f s  %5.1f GiB/s",
                   pct, written, total_tiles, gib, elapsed, bw);
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
        printf("  done  %.2f GiB  %.1f s  %.2f GiB/s\n",
               total_gib, elapsed,
               (elapsed > 0.0) ? total_gib / elapsed : 0.0);
    }
    return ret;
}

/* ----------------------------------------------------------------------- */
/* main                                                                      */
/* ----------------------------------------------------------------------- */

int main(void)
{
    printf("=== 100 GB Complex Tensor Generator ===\n");
    printf("Fill value : %.1f + %.1fi\n\n", FILL_REAL, FILL_IMAG);

    /* A layout: (i, j, a, b) */
    if (generate_file("A_100gb.h5", "tensor", CMPLX(FILL_REAL, FILL_IMAG)) < 0)
        return 1;

    printf("\n");

    /* B layout: (a, k, b, l) — same shape, same fill */
    if (generate_file("B_100gb.h5", "tensor", CMPLX(FILL_REAL, FILL_IMAG)) < 0)
        return 1;

    printf("\nDone.  A_100gb.h5 and B_100gb.h5 are ready.\n");
    printf("Expected contraction result: %.1f + %.1fi\n",
           (double)(GLOBAL_DIM * GLOBAL_DIM) * (FILL_REAL*FILL_REAL
                                                 - FILL_IMAG*FILL_IMAG),
           (double)(GLOBAL_DIM * GLOBAL_DIM) * 2.0 * FILL_REAL * FILL_IMAG);
    return 0;
}
