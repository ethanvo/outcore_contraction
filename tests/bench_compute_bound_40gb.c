/*
 * tests/bench_compute_bound_40gb.c
 *
 * Compute-bound benchmark for the out-of-core COMPLEX128 tensor engine.
 *
 * Contraction: "ijab,akbl->klji"
 *   C(k,l,j,i) = sum_{a,b} A(i,j,a,b) * B(a,k,b,l)
 *
 * Parameters:
 *   GLOBAL_DIM = 224   (override with -DGLOBAL_DIM=N)
 *   CHUNK_DIM  = 32    (override with -DCHUNK_DIM=N)
 *
 * Tensor sizes:
 *   Each tensor: 224^4 × 16 B ≈ 40.13 GiB
 *   Chunk tile:  32^4  × 16 B = 16 MiB  (arithmetic intensity >> roofline)
 *   Grid:        7^4 = 2401 tiles per tensor
 *
 * Inner BLAS call: cblas_zgemm M=N=K=1024 → 8×1024^3 ≈ 8.59 GFLOPs
 *   At 600 GFLOPS: ~14.3 ms BLAS per contracted pair
 *   B I/O for 49 tiles × 16 MiB = 784 MiB: ~4 ms at 200 GB/s NVMe
 *   Double-buffering hides I/O behind BLAS → sustained ~600 GFLOPS
 *
 * Expected output (fill = 1+0.5i):
 *   C(k,l,j,i) = DIM^2 × (1+0.5i)^2 = DIM^2 × (0.75+i)
 *   For DIM=224: 224^2 = 50176  →  37632.0 + 50176.0i  (every element)
 *
 * Total FLOPs: 8 × 224^6 ≈ 1.01 PFLOPs
 *
 * File generation:
 *   Skipped if A_compute_40gb.h5 and B_compute_40gb.h5 already exist.
 *   Otherwise generated inline with fill = 1+0.5i.
 *
 * Reported metrics:
 *   Total Elapsed Time (s)
 *   Sustained GFLOPS
 *   Effective NVMe Read Bandwidth (GB/s)
 *   Effective NVMe Write Bandwidth (GB/s)
 */

#include "engine.h"
#include "registry.h"
#include "tensor_store.h"
#include "odometer.h"
#include <hdf5.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef GLOBAL_DIM
#  define GLOBAL_DIM  224
#endif
#ifndef CHUNK_DIM
#  define CHUNK_DIM   32
#endif
#ifndef POOL_CAP_MB
#  define POOL_CAP_MB 512
#endif

#define RANK    4
#define FILE_A  "A_compute_40gb.h5"
#define FILE_B  "B_compute_40gb.h5"
#define FILE_C  "C_compute_40gb.h5"
#define DSET    "tensor"
#define EXPR    "ijab,akbl->klji"

/* Fill value: 1 + 0.5i */
#define FILL_REAL  1.0
#define FILL_IMAG  0.5

/*
 * Expected output:
 *   C = DIM^2 * (FILL_REAL + FILL_IMAG*i)^2
 *     = DIM^2 * (FILL_REAL^2 - FILL_IMAG^2) + DIM^2 * 2*FILL_REAL*FILL_IMAG * i
 */
#define EXPECTED_REAL  ((double)(GLOBAL_DIM) * (double)(GLOBAL_DIM) \
                        * (FILL_REAL * FILL_REAL - FILL_IMAG * FILL_IMAG))
#define EXPECTED_IMAG  ((double)(GLOBAL_DIM) * (double)(GLOBAL_DIM) \
                        * (2.0 * FILL_REAL * FILL_IMAG))

static double elapsed_s(const struct timespec *a, const struct timespec *b)
{
    return (double)(b->tv_sec  - a->tv_sec)
         + (double)(b->tv_nsec - a->tv_nsec) * 1e-9;
}

/* ----------------------------------------------------------------------- */
/* Inline file generation                                                    */
/* ----------------------------------------------------------------------- */

static int generate_file(const char *fname, const char *dset_name)
{
    const int rank = RANK;
    hsize_t shape[RANK], chunk_dims[RANK];
    for (int d = 0; d < rank; d++) {
        shape[d]      = (hsize_t)GLOBAL_DIM;
        chunk_dims[d] = (hsize_t)CHUNK_DIM;
    }

    size_t elems_per_chunk = 1;
    for (int d = 0; d < rank; d++) elems_per_chunk *= (size_t)CHUNK_DIM;
    size_t bytes_per_chunk = elems_per_chunk * sizeof(double _Complex);

    size_t n_tiles[RANK], total_tiles = 1;
    for (int d = 0; d < rank; d++) {
        n_tiles[d]   = (size_t)((GLOBAL_DIM + CHUNK_DIM - 1) / CHUNK_DIM);
        total_tiles *= n_tiles[d];
    }

    double total_gib = (double)(total_tiles * bytes_per_chunk)
                       / (1024.0 * 1024.0 * 1024.0);

    printf("Generating %s\n"
           "  shape : (%d^%d)   dtype : COMPLEX128\n"
           "  chunk : (%d^%d)   %zu MiB/chunk   %zu tiles\n"
           "  total : %.2f GiB\n",
           fname, GLOBAL_DIM, rank, CHUNK_DIM, rank,
           bytes_per_chunk / (1024 * 1024), total_tiles,
           total_gib);

    if (create_chunked_dataset_einsum(fname, dset_name, rank,
                                      shape, chunk_dims,
                                      DTYPE_COMPLEX128) < 0) {
        fprintf(stderr, "gen: create_chunked_dataset_einsum failed '%s'\n", fname);
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

    /* Align chunk buffer to 16 KiB (NVMe page boundary). */
    void *raw_buf = NULL;
    if (posix_memalign(&raw_buf, 16384, bytes_per_chunk) != 0) {
        fprintf(stderr, "gen: posix_memalign failed\n");
        H5Tclose(h5ctype); H5Dclose(dset); H5Fclose(fid);
        return -1;
    }
    double _Complex *buf = (double _Complex *)raw_buf;
    for (size_t i = 0; i < elems_per_chunk; i++)
        buf[i] = CMPLX(FILL_REAL, FILL_IMAG);

    size_t tile_idx[RANK], grid_sz[RANK];
    memset(tile_idx, 0, sizeof(tile_idx));
    for (int d = 0; d < rank; d++) grid_sz[d] = n_tiles[d];

    int    ret     = 0;
    size_t written = 0;

    struct timespec t0, tnow;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    do {
        hsize_t offset[RANK];
        for (int d = 0; d < rank; d++)
            offset[d] = (hsize_t)tile_idx[d] * chunk_dims[d];

        if (write_chunk_typed(dset, offset, buf,
                              sizeof(double _Complex), rank,
                              chunk_dims, h5ctype) < 0) {
            fprintf(stderr, "gen: write_chunk_typed failed\n");
            ret = -1;
            break;
        }

        written++;
        if (written % 100 == 0 || written == total_tiles) {
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            double elapsed = elapsed_s(&t0, &tnow);
            double gib_done = (double)(written * bytes_per_chunk)
                              / (1024.0 * 1024.0 * 1024.0);
            double bw = (elapsed > 0.0) ? gib_done / elapsed : 0.0;
            printf("\r  %5.1f%%  %4zu/%zu tiles  %6.2f GiB  "
                   "%5.1f s  %5.2f GiB/s",
                   100.0 * (double)written / (double)total_tiles,
                   written, total_tiles, gib_done, elapsed, bw);
            fflush(stdout);
        }

    } while (odometer_step(rank, tile_idx, grid_sz));

    printf("\n");
    free(raw_buf);
    H5Tclose(h5ctype);
    H5Dclose(dset);
    H5Fclose(fid);

    if (ret == 0) {
        struct timespec tf;
        clock_gettime(CLOCK_MONOTONIC, &tf);
        double elapsed = elapsed_s(&t0, &tf);
        double total_gib_written = (double)(total_tiles * bytes_per_chunk)
                                   / (1024.0 * 1024.0 * 1024.0);
        printf("  done  %.2f GiB  %.2f s  %.2f GiB/s\n",
               total_gib_written, elapsed,
               (elapsed > 0.0) ? total_gib_written / elapsed : 0.0);
    }
    return ret;
}

/* ----------------------------------------------------------------------- */
/* Output verification                                                       */
/* ----------------------------------------------------------------------- */

static int verify_output(void)
{
    const double eps_rel = 1e-9;
    const double eps = eps_rel * sqrt(EXPECTED_REAL * EXPECTED_REAL
                                       + EXPECTED_IMAG * EXPECTED_IMAG);

    printf("\n--- Verifying output ---\n");
    printf("  Expected: %.1f + %.1fi  (all elements)\n",
           EXPECTED_REAL, EXPECTED_IMAG);

    hid_t fc = H5Fopen(FILE_C, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fc < 0) {
        fprintf(stderr, "verify: cannot open %s\n", FILE_C);
        return -1;
    }
    hid_t dset_C = dset_open_no_cache(fc, DSET);
    if (dset_C < 0) {
        fprintf(stderr, "verify: cannot open dataset %s\n", DSET);
        H5Fclose(fc);
        return -1;
    }

    TensorRegistry *reg_C = registry_create_from_dset(dset_C);
    if (!reg_C) {
        fprintf(stderr, "verify: registry_create_from_dset failed\n");
        H5Dclose(dset_C); H5Fclose(fc);
        return -1;
    }
    if (registry_scan_file(dset_C, reg_C) < 0) {
        fprintf(stderr, "verify: registry_scan_file failed\n");
        registry_destroy(reg_C); H5Dclose(dset_C); H5Fclose(fc);
        return -1;
    }

    int rank = reg_C->rank;
    printf("  C grid: ");
    size_t total_tiles = 1;
    for (int d = 0; d < rank; d++) {
        printf("%s%llu", (d ? "x" : ""), (unsigned long long)reg_C->grid_dims[d]);
        total_tiles *= (size_t)reg_C->grid_dims[d];
    }
    printf("  (%zu tiles total)\n", total_tiles);

    hid_t h5ctype = create_h5_complex_type();
    if (h5ctype < 0) {
        fprintf(stderr, "verify: create_h5_complex_type failed\n");
        registry_destroy(reg_C); H5Dclose(dset_C); H5Fclose(fc);
        return -1;
    }

    size_t elems = 1;
    for (int d = 0; d < rank; d++)
        elems *= (size_t)reg_C->chunk_dims[d];

    void *raw_buf = NULL;
    if (posix_memalign(&raw_buf, 16384, elems * sizeof(double _Complex)) != 0) {
        fprintf(stderr, "verify: posix_memalign failed\n");
        H5Tclose(h5ctype);
        registry_destroy(reg_C); H5Dclose(dset_C); H5Fclose(fc);
        return -1;
    }
    double _Complex *buf = (double _Complex *)raw_buf;

    /* Sample tiles: corner (0,...,0), all-last, centre. */
    hsize_t samples[3][8];
    memset(samples[0], 0, sizeof(samples[0]));
    for (int d = 0; d < rank; d++)
        samples[1][d] = reg_C->grid_dims[d] - 1;
    for (int d = 0; d < rank; d++)
        samples[2][d] = reg_C->grid_dims[d] / 2;

    int total_fail = 0;

    for (int si = 0; si < 3; si++) {
        hsize_t *coord = samples[si];

        TileMetadata *m = registry_get_tile(reg_C, coord);
        if (!m || m->status != TILE_STATUS_ON_DISK) {
            fprintf(stderr, "verify: tile not on disk (sample %d)\n", si);
            total_fail++;
            continue;
        }

        /* Pre-poison buffer. */
        for (size_t i = 0; i < elems; i++)
            buf[i] = CMPLX(-1e30, -1e30);

        if (read_chunk_typed(dset_C, m->phys_offset, buf,
                             sizeof(double _Complex), rank,
                             reg_C->chunk_dims, h5ctype) < 0) {
            fprintf(stderr, "verify: read_chunk_typed failed (sample %d)\n", si);
            total_fail++;
            continue;
        }

        /* Physical extents for this tile. */
        hsize_t act[8];
        for (int d = 0; d < rank; d++) {
            hsize_t end = m->phys_offset[d] + reg_C->chunk_dims[d];
            act[d] = (end > reg_C->global_dims[d])
                     ? reg_C->global_dims[d] - m->phys_offset[d]
                     : reg_C->chunk_dims[d];
        }

        /* Row-major strides over nominal chunk_dims. */
        size_t strides[8];
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--)
            strides[d] = strides[d + 1] * (size_t)reg_C->chunk_dims[d + 1];

        size_t extents[8];
        for (int d = 0; d < rank; d++)
            extents[d] = (size_t)act[d];

        double max_re_err = 0.0, max_im_err = 0.0;
        size_t mismatch = 0;

        size_t total_elems = 1;
        for (int d = 0; d < rank; d++) total_elems *= extents[d];

        for (size_t flat = 0; flat < total_elems; flat++) {
            /* Map flat → buffer index via nominal strides. */
            size_t idx = 0, rem = flat;
            size_t ext_strides[8];
            ext_strides[rank - 1] = 1;
            for (int d = rank - 2; d >= 0; d--)
                ext_strides[d] = ext_strides[d + 1] * extents[d + 1];
            for (int d = 0; d < rank; d++) {
                size_t coord_d = rem / ext_strides[d];
                rem %= ext_strides[d];
                idx += coord_d * strides[d];
            }

            double re = creal(buf[idx]);
            double im = cimag(buf[idx]);
            double re_err = fabs(re - EXPECTED_REAL);
            double im_err = fabs(im - EXPECTED_IMAG);
            if (re_err > max_re_err) max_re_err = re_err;
            if (im_err > max_im_err) max_im_err = im_err;
            if ((re_err > eps || im_err > eps) && mismatch == 0) {
                fprintf(stderr,
                        "  MISMATCH sample %d element %zu: "
                        "got %.6f+%.6fi  expected %.1f+%.1fi\n",
                        si, flat, re, im, EXPECTED_REAL, EXPECTED_IMAG);
            }
            if (re_err > eps || im_err > eps) mismatch++;
        }

        int tile_fail = (mismatch > 0);
        total_fail += tile_fail;

        printf("  tile(");
        for (int d = 0; d < rank; d++)
            printf("%s%llu", (d ? "," : ""), (unsigned long long)coord[d]);
        printf(")  act(");
        for (int d = 0; d < rank; d++)
            printf("%s%llu", (d ? "x" : ""), (unsigned long long)act[d]);
        printf(")  max_err=(%.2e,%.2e)  %s\n",
               max_re_err, max_im_err,
               tile_fail ? "FAIL" : "PASS");
    }

    free(raw_buf);
    H5Tclose(h5ctype);
    registry_destroy(reg_C);
    H5Dclose(dset_C);
    H5Fclose(fc);

    return (total_fail > 0) ? -1 : 0;
}

/* ----------------------------------------------------------------------- */
/* main                                                                      */
/* ----------------------------------------------------------------------- */

int main(void)
{
    /* Tensor geometry. */
    const double elems_per_tensor = /* GLOBAL_DIM^4 */
        (double)GLOBAL_DIM * (double)GLOBAL_DIM *
        (double)GLOBAL_DIM * (double)GLOBAL_DIM;
    const double bytes_per_tensor = elems_per_tensor * 16.0;
    const double gib_per_tensor   = bytes_per_tensor / (1024.0 * 1024.0 * 1024.0);

    /* FLOPs: ijab,akbl->klji contracts a,b (2 dims) over i,j,k,l (4 free).
     * Total multiplies = 8 × DIM^6  (complex multiply = 6 FLOPs,
     *                                complex add     = 2 FLOPs → 8 per pair). */
    double total_flops = 8.0;
    for (int i = 0; i < 6; i++) total_flops *= (double)GLOBAL_DIM;

    /* Chunk tile size. */
    double chunk_elems = 1.0;
    for (int i = 0; i < RANK; i++) chunk_elems *= (double)CHUNK_DIM;
    double chunk_mib = chunk_elems * 16.0 / (1024.0 * 1024.0);

    printf("=================================================================\n");
    printf("  Compute-Bound 40 GiB Contraction Benchmark\n");
    printf("  Expression : %s\n", EXPR);
    printf("  Dtype      : COMPLEX128\n");
    printf("  Global dim : %d per index\n", GLOBAL_DIM);
    printf("  Chunk dim  : %d per index  (%.0f MiB/tile)\n",
           CHUNK_DIM, chunk_mib);
    printf("  Tensor     : %.2f GiB each\n", gib_per_tensor);
    printf("  FLOPs      : %.3e\n", total_flops);
    printf("  Expected C : %.1f + %.1fi  (every element)\n",
           EXPECTED_REAL, EXPECTED_IMAG);
    printf("=================================================================\n\n");

    /* ------------------------------------------------------------------ */
    /* Phase 1 — generate input files if they do not already exist         */
    /* ------------------------------------------------------------------ */
    {
        int need_gen = 0;
        FILE *fa = fopen(FILE_A, "rb");
        FILE *fb = fopen(FILE_B, "rb");
        if (!fa || !fb) need_gen = 1;
        if (fa) fclose(fa);
        if (fb) fclose(fb);

        if (need_gen) {
            printf("--- Generating input files ---\n");
            if (generate_file(FILE_A, DSET) < 0) return 1;
            printf("\n");
            if (generate_file(FILE_B, DSET) < 0) return 1;
            printf("\n");
        } else {
            printf("Input files found — skipping generation.\n\n");
        }
    }

    /* ------------------------------------------------------------------ */
    /* Phase 2 — run contraction                                           */
    /* ------------------------------------------------------------------ */

    /* Cap the memory pool so the engine operates out-of-core. */
    char pool_str[32];
    snprintf(pool_str, sizeof(pool_str), "%d", POOL_CAP_MB);
    setenv("TENSOR_POOL_MB", pool_str, 1);
    printf("TENSOR_POOL_MB=%s MiB\n\n", pool_str);

    printf("--- Running contraction ---\n");
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int ret = run_contraction_einsum(EXPR,
                                     FILE_A, DSET,
                                     FILE_B, DSET,
                                     FILE_C, DSET);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = elapsed_s(&t_start, &t_end);

    if (ret != 0) {
        fprintf(stderr, "Contraction failed (rc=%d).\n", ret);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Phase 3 — performance metrics                                       */
    /* ------------------------------------------------------------------ */

    double gflops     = (total_flops / elapsed) / 1.0e9;
    double read_gib   = 2.0 * gib_per_tensor;   /* A + B */
    double write_gib  = gib_per_tensor;          /* C */
    double read_bw    = read_gib  / elapsed;
    double write_bw   = write_gib / elapsed;

    printf("\n=================================================================\n");
    printf("  Performance\n");
    printf("  Total Elapsed Time         : %.3f s\n",    elapsed);
    printf("  Sustained GFLOPS           : %.2f\n",      gflops);
    printf("  Effective NVMe Read BW     : %.2f GiB/s  (%.2f GB/s)\n",
           read_bw,  read_bw  * 1.073741824);
    printf("  Effective NVMe Write BW    : %.2f GiB/s  (%.2f GB/s)\n",
           write_bw, write_bw * 1.073741824);
    printf("=================================================================\n");

    /* ------------------------------------------------------------------ */
    /* Phase 4 — verify output                                             */
    /* ------------------------------------------------------------------ */

    ret = verify_output();
    printf("\n=== Result: %s ===\n", ret ? "FAILED" : "ALL PASSED");
    return ret ? 1 : 0;
}
