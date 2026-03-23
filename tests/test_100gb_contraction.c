/*
 * test_100gb_contraction.c
 *
 * Production-scale validation of the rank-4 out-of-core contraction engine.
 *
 *   C(k,l,j,i) = sum_{a,b} A(i,j,a,b) * B(a,k,b,l)
 *
 * Tensor sizes
 * ------------
 *   All six free indices (i,j,a,b,k,l) have global dimension GLOBAL_DIM=340.
 *   Each tensor therefore has 340^4 = 13,363,360,000 elements × 8 bytes
 *   ≈ 106 GB on disk.
 *
 *   To run a quick smoke test without 300+ GB of disk and hours of compute,
 *   compile with -DSMALL_TEST=1 (sets GLOBAL_DIM=40, ~820 MB per tensor).
 *
 * Data generation
 * ---------------
 *   A(i,j,a,b) = 1.0 everywhere.
 *   B(a,k,b,l) = 1.0 everywhere.
 *
 * Expected output
 * ---------------
 *   C(k,l,j,i) = sum_{a=0}^{DIM-1} sum_{b=0}^{DIM-1} 1.0 * 1.0
 *              = GLOBAL_DIM^2  for every valid (k,l,j,i) position.
 *
 * Verification
 * ------------
 *   A sample of C tiles is read back and every element checked against
 *   GLOBAL_DIM^2.  Boundary tiles are included (the permutation and
 *   zero-padding logic must produce correct partial sums).
 */

#include "engine.h"
#include "registry.h"
#include "tensor_store.h"
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Tuneable constants                                                   */
/* ------------------------------------------------------------------ */

#ifndef SMALL_TEST
#  define GLOBAL_DIM        340ULL   /* ≈ 106 GB per tensor */
#  define CHUNK_BYTES_FIXED 0        /* derive from query_physical_ram() */
#else
/*
 * SMALL_TEST: 40^4 tensors with 1 MB chunks.
 * calculate_chunk_dims(1 MB, 4, {40,...}) → chunk_dim ≈ 19.
 * ceil(40/19) = 3 tiles per dim → 3^4 = 81 C tiles including boundary
 * tiles (last tile dim = 40 - 2×19 = 2 elements per axis).
 * Expected C[k,l,j,i] = 40^2 = 1600 everywhere.
 */
#  define GLOBAL_DIM        40ULL
#  define CHUNK_BYTES_FIXED (1UL * 1024 * 1024)  /* 1 MB — forces multi-tile */
#endif

/* File / dataset names */
#define FILE_A  "A4d.h5"
#define FILE_B  "B4d.h5"
#define FILE_C  "C4d.h5"
#define DSET_A  "TensorA"
#define DSET_B  "TensorB"
#define DSET_C  "TensorC"

/* ------------------------------------------------------------------ */
/* generate_rank4_ones                                                  */
/*                                                                      */
/* Creates an HDF5 file containing one rank-4 dataset filled with 1.0. */
/* Uses write_chunk_fast with nominal chunk_dims (boundary clamping is  */
/* handled internally).                                                 */
/* ------------------------------------------------------------------ */
static int generate_rank4_ones(const char *fname, const char *dset_name,
                                const hsize_t shape[4], size_t chunk_bytes)
{
    printf("Generating %s  shape=(%llu\xc3\x97%llu\xc3\x97%llu\xc3\x97%llu)"
           "  chunk_bytes=%.0f MB\n",
           fname,
           (unsigned long long)shape[0], (unsigned long long)shape[1],
           (unsigned long long)shape[2], (unsigned long long)shape[3],
           (double)chunk_bytes / (1024.0 * 1024.0));

    if (create_chunked_dataset(fname, dset_name, 4, shape, chunk_bytes) < 0) {
        fprintf(stderr, "generate_rank4_ones: create_chunked_dataset failed "
                        "for '%s'\n", fname);
        return -1;
    }

    hid_t fid = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "generate_rank4_ones: H5Fopen failed\n");
        return -1;
    }

    hid_t dset = dset_open_no_cache(fid, dset_name);
    if (dset < 0) {
        fprintf(stderr, "generate_rank4_ones: dset_open_no_cache failed\n");
        H5Fclose(fid);
        return -1;
    }

    hsize_t chunk_dims[4];
    calculate_chunk_dims(chunk_bytes, 4, shape, chunk_dims);

    hsize_t n_tiles[4];
    for (int d = 0; d < 4; d++)
        n_tiles[d] = (shape[d] + chunk_dims[d] - 1) / chunk_dims[d];

    size_t elems = (size_t)chunk_dims[0] * (size_t)chunk_dims[1]
                 * (size_t)chunk_dims[2] * (size_t)chunk_dims[3];
    double *buf = (double *)malloc(elems * sizeof(double));
    if (!buf) {
        fprintf(stderr, "generate_rank4_ones: malloc failed (%zu bytes)\n",
                elems * sizeof(double));
        H5Dclose(dset); H5Fclose(fid);
        return -1;
    }
    for (size_t i = 0; i < elems; i++) buf[i] = 1.0;

    int ret = 0;
    hsize_t ti0, ti1, ti2, ti3;
    for (ti0 = 0; ti0 < n_tiles[0] && ret == 0; ti0++) {
      for (ti1 = 0; ti1 < n_tiles[1] && ret == 0; ti1++) {
        for (ti2 = 0; ti2 < n_tiles[2] && ret == 0; ti2++) {
          for (ti3 = 0; ti3 < n_tiles[3] && ret == 0; ti3++) {
            hsize_t offset[4] = {
                ti0 * chunk_dims[0],
                ti1 * chunk_dims[1],
                ti2 * chunk_dims[2],
                ti3 * chunk_dims[3]
            };
            if (write_chunk_fast(dset, offset, buf, 4, chunk_dims) < 0) {
                fprintf(stderr,
                        "generate_rank4_ones: write_chunk_fast failed at "
                        "[%llu,%llu,%llu,%llu]\n",
                        (unsigned long long)ti0, (unsigned long long)ti1,
                        (unsigned long long)ti2, (unsigned long long)ti3);
                ret = -1;
            }
          }
        }
      }
    }

    free(buf);
    H5Dclose(dset);
    H5Fclose(fid);

    if (ret == 0) {
        hsize_t total = n_tiles[0] * n_tiles[1] * n_tiles[2] * n_tiles[3];
        printf("  done  chunk_dims=(%llu\xc3\x97%llu\xc3\x97%llu\xc3\x97%llu)"
               "  %llu tiles\n",
               (unsigned long long)chunk_dims[0],
               (unsigned long long)chunk_dims[1],
               (unsigned long long)chunk_dims[2],
               (unsigned long long)chunk_dims[3],
               (unsigned long long)total);
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/* verify_c_tile                                                         */
/*                                                                      */
/* Read the C tile at grid position (ki,li,ji,ii) and verify that every */
/* element equals expected_val.  Returns 0 on pass, -1 on failure.      */
/* ------------------------------------------------------------------ */
static int verify_c_tile(hid_t dset_C, TensorRegistry *reg_C,
                         hsize_t ki, hsize_t li, hsize_t ji, hsize_t ii,
                         double expected_val)
{
    hsize_t coords[4] = {ki, li, ji, ii};
    TileMetadata *m = registry_get_tile(reg_C, coords);
    if (!m || m->status != TILE_STATUS_ON_DISK) {
        fprintf(stderr,
                "verify_c_tile: tile (%llu,%llu,%llu,%llu) not on disk\n",
                (unsigned long long)ki, (unsigned long long)li,
                (unsigned long long)ji, (unsigned long long)ii);
        return -1;
    }

    size_t elems = 1;
    for (int d = 0; d < 4; d++) elems *= (size_t)reg_C->chunk_dims[d];

    double *buf = (double *)malloc(elems * sizeof(double));
    if (!buf) { fprintf(stderr, "verify_c_tile: malloc failed\n"); return -1; }

    /* Poison with sentinel so partial reads are visible. */
    for (size_t i = 0; i < elems; i++) buf[i] = -999.0;

    int ret = 0;
    if (read_chunk_fast(dset_C, m->phys_offset, buf, 4,
                        reg_C->chunk_dims) < 0) {
        fprintf(stderr, "verify_c_tile: read_chunk_fast failed\n");
        free(buf); return -1;
    }

    /* Determine actual extents for this tile. */
    hsize_t act[4];
    for (int d = 0; d < 4; d++) {
        hsize_t end = m->phys_offset[d] + reg_C->chunk_dims[d];
        act[d] = (end > reg_C->global_dims[d])
                 ? reg_C->global_dims[d] - m->phys_offset[d]
                 : reg_C->chunk_dims[d];
    }
    int l_n = (int)reg_C->chunk_dims[1];
    int j_n = (int)reg_C->chunk_dims[2];

    /* Check only actual (non-padded) positions. */
    for (hsize_t k = 0; k < act[0] && ret == 0; k++) {
      for (hsize_t l = 0; l < act[1] && ret == 0; l++) {
        for (hsize_t j = 0; j < act[2] && ret == 0; j++) {
          for (hsize_t i = 0; i < act[3]; i++) {
            size_t idx = (size_t)k * (l_n * j_n * (int)reg_C->chunk_dims[3])
                       + (size_t)l * (j_n * (int)reg_C->chunk_dims[3])
                       + (size_t)j * (int)reg_C->chunk_dims[3]
                       + (size_t)i;
            if (fabs(buf[idx] - expected_val) > 1e-6 * expected_val) {
                fprintf(stderr,
                        "MISMATCH at tile(%llu,%llu,%llu,%llu) "
                        "local(%llu,%llu,%llu,%llu): "
                        "got %.6f expected %.6f\n",
                        (unsigned long long)ki, (unsigned long long)li,
                        (unsigned long long)ji, (unsigned long long)ii,
                        (unsigned long long)k, (unsigned long long)l,
                        (unsigned long long)j, (unsigned long long)i,
                        buf[idx], expected_val);
                ret = -1;
                break;
            }
          }
        }
      }
    }

    free(buf);
    if (ret == 0)
        printf("  tile(%llu,%llu,%llu,%llu)  actual(%llux%llux%llux%llu)"
               "  PASS\n",
               (unsigned long long)ki, (unsigned long long)li,
               (unsigned long long)ji, (unsigned long long)ii,
               (unsigned long long)act[0], (unsigned long long)act[1],
               (unsigned long long)act[2], (unsigned long long)act[3]);
    return ret;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== 100 GB Rank-4 Contraction Test ===\n");
    printf("Global dim per index: %llu\n", (unsigned long long)GLOBAL_DIM);
    printf("Bytes per tensor: %.1f GB\n",
           (double)(GLOBAL_DIM * GLOBAL_DIM * GLOBAL_DIM * GLOBAL_DIM)
               * 8.0 / (1024.0 * 1024.0 * 1024.0));

    /*
     * Chunk size: pool / 20 to keep ≥20 pages available.  Minimum 64 MB.
     * The same chunk_bytes is used for A and B; C gets chunk_bytes from
     * run_contraction_4d's internal query_physical_ram() / 1000.
     */
    size_t ram = query_physical_ram();
    size_t chunk_bytes;
#if CHUNK_BYTES_FIXED
    chunk_bytes = CHUNK_BYTES_FIXED;
#else
    {
        size_t pool_bytes = (size_t)((double)ram * 0.8);
        chunk_bytes = pool_bytes / 20;
        if (chunk_bytes < 64UL * 1024 * 1024)
            chunk_bytes = 64UL * 1024 * 1024;
    }
#endif

    printf("Physical RAM: %.1f GB  chunk target: %.3f MB\n",
           (double)ram / (1024.0 * 1024.0 * 1024.0),
           (double)chunk_bytes / (1024.0 * 1024.0));

    /* ---------------------------------------------------------------- */
    /* Step 1: Generate A(i,j,a,b) = 1.0 and B(a,k,b,l) = 1.0          */
    /* ---------------------------------------------------------------- */
    hsize_t shape_A[4] = {GLOBAL_DIM, GLOBAL_DIM, GLOBAL_DIM, GLOBAL_DIM};
    hsize_t shape_B[4] = {GLOBAL_DIM, GLOBAL_DIM, GLOBAL_DIM, GLOBAL_DIM};

    printf("\n--- Generating input tensors ---\n");
    if (generate_rank4_ones(FILE_A, DSET_A, shape_A, chunk_bytes) < 0)
        return 1;
    if (generate_rank4_ones(FILE_B, DSET_B, shape_B, chunk_bytes) < 0)
        return 1;

    /* ---------------------------------------------------------------- */
    /* Step 2: Contract C(k,l,j,i) = sum_{a,b} A(i,j,a,b) * B(a,k,b,l) */
    /* ---------------------------------------------------------------- */
    printf("\n--- Running contraction ---\n");
    if (run_contraction_4d(FILE_A, DSET_A,
                           FILE_B, DSET_B,
                           FILE_C, DSET_C) < 0) {
        fprintf(stderr, "Contraction failed.\n");
        return 1;
    }

    /* ---------------------------------------------------------------- */
    /* Step 3: Verify — expected C[k,l,j,i] = GLOBAL_DIM^2 everywhere   */
    /* ---------------------------------------------------------------- */
    double expected = (double)GLOBAL_DIM * (double)GLOBAL_DIM;
    printf("\n--- Verifying output (expected C[*,*,*,*] = %.0f) ---\n",
           expected);

    hid_t fc     = H5Fopen(FILE_C, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset_C = (fc >= 0) ? dset_open_no_cache(fc, DSET_C) : -1;
    if (fc < 0 || dset_C < 0) {
        fprintf(stderr, "Cannot open %s for verification\n", FILE_C);
        if (fc >= 0) H5Fclose(fc);
        return 1;
    }

    TensorRegistry *reg_C = registry_create_from_dset(dset_C);
    if (!reg_C) {
        fprintf(stderr, "registry_create_from_dset(C) failed\n");
        H5Dclose(dset_C); H5Fclose(fc);
        return 1;
    }

    if (registry_scan_file(dset_C, reg_C) < 0) {
        fprintf(stderr, "registry_scan_file(C) failed\n");
        registry_destroy(reg_C);
        H5Dclose(dset_C); H5Fclose(fc);
        return 1;
    }

    int fail = 0;

    /*
     * Verify a representative sample:
     *   - First interior tile (0,0,0,0)
     *   - Middle tile (if grid has ≥3 tiles in each dim)
     *   - Last tile (boundary) in each dimension
     */
    hsize_t kg = reg_C->grid_dims[0];
    hsize_t lg = reg_C->grid_dims[1];
    hsize_t jg = reg_C->grid_dims[2];
    hsize_t ig = reg_C->grid_dims[3];

    printf("C grid: k=%llu l=%llu j=%llu i=%llu  total=%llu tiles\n",
           (unsigned long long)kg, (unsigned long long)lg,
           (unsigned long long)jg, (unsigned long long)ig,
           (unsigned long long)(kg * lg * jg * ig));

    /* Interior tile. */
    fail |= verify_c_tile(dset_C, reg_C, 0, 0, 0, 0, expected);

    /* Boundary tiles (last index in each dimension). */
    fail |= verify_c_tile(dset_C, reg_C, kg-1, 0,    0,    0,    expected);
    fail |= verify_c_tile(dset_C, reg_C, 0,    lg-1, 0,    0,    expected);
    fail |= verify_c_tile(dset_C, reg_C, 0,    0,    jg-1, 0,    expected);
    fail |= verify_c_tile(dset_C, reg_C, 0,    0,    0,    ig-1, expected);

    /* All-boundary corner tile. */
    fail |= verify_c_tile(dset_C, reg_C, kg-1, lg-1, jg-1, ig-1, expected);

    /* Middle tile (if grid is large enough). */
    if (kg > 2 && lg > 2 && jg > 2 && ig > 2)
        fail |= verify_c_tile(dset_C, reg_C,
                              kg/2, lg/2, jg/2, ig/2, expected);

    registry_destroy(reg_C);
    H5Dclose(dset_C);
    H5Fclose(fc);

    printf("\n=== Result: %s ===\n", fail ? "FAILED" : "ALL PASSED");
    return fail ? 1 : 0;
}
