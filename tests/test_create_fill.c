/*
 * tests/test_create_fill.c
 *
 * Correctness tests for tensor_engine_create() and tensor_engine_fill().
 *
 * Seven test cases:
 *   T1 – create FP64 rank-2 with default tile_bytes, check dims
 *   T2 – create COMPLEX128 rank-3, check dims and dtype
 *   T3 – tile_bytes config controls chunk dimensions
 *   T4 – fill FP64 tensor, read every element back and verify
 *   T5 – fill COMPLEX128 tensor, read back and verify
 *   T6 – boundary tiles: non-divisible shapes still fill correctly
 *   T7 – end-to-end: create A + B, fill, contract, verify result
 *
 * All files use the prefix "cf_t{N}_" in the current working directory.
 *
 * Build: added to CMakeLists.txt as test_create_fill.
 * Run:   ./build/test_create_fill
 * Exit:  0 on success, 1 on any failure.
 */

#include "tensor_engine.h"
#include "tensor_store.h"
#include "registry.h"
#include "odometer.h"
#include <hdf5.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------------- */
/* Test infrastructure                                                       */
/* ----------------------------------------------------------------------- */

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            printf("  PASS: %s\n", msg); \
            g_pass++; \
        } else { \
            printf("  FAIL: %s  (line %d)\n", msg, __LINE__); \
            g_fail++; \
        } \
    } while (0)

/* ----------------------------------------------------------------------- */
/* Helper: open dataset and read its rank + chunk dims                       */
/* ----------------------------------------------------------------------- */
static int read_dset_info(const char *file, int *rank_out,
                          hsize_t *global_out, hsize_t *chunk_out,
                          tensor_dtype_t *dtype_out)
{
    hid_t fid = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) return -1;

    hid_t dset = dset_open_no_cache(fid, "tensor");
    if (dset < 0) { H5Fclose(fid); return -1; }

    TensorRegistry *reg = registry_create_from_dset(dset);
    if (!reg) { H5Dclose(dset); H5Fclose(fid); return -1; }

    *rank_out  = reg->rank;
    *dtype_out = reg->dtype;
    for (int d = 0; d < reg->rank; d++) {
        global_out[d] = reg->global_dims[d];
        chunk_out[d]  = reg->chunk_dims[d];
    }

    registry_destroy(reg);
    H5Dclose(dset);
    H5Fclose(fid);
    return 0;
}

/* ----------------------------------------------------------------------- */
/* Helper: read every element of an FP64 tensor; call cb for each value.    */
/* Returns -1 on I/O error.  max_abs_err is the supremum |read - expected|. */
/* ----------------------------------------------------------------------- */
static int verify_fp64(const char *file, double expected, double *max_err_out)
{
    hid_t fid = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) return -1;
    hid_t dset = dset_open_no_cache(fid, "tensor");
    if (dset < 0) { H5Fclose(fid); return -1; }

    TensorRegistry *reg = registry_create_from_dset(dset);
    if (!reg) { H5Dclose(dset); H5Fclose(fid); return -1; }

    int    rank = reg->rank;
    size_t tile_elems = 1;
    size_t ntiles[MAX_RANK];
    for (int d = 0; d < rank; d++) {
        tile_elems *= (size_t)reg->chunk_dims[d];
        ntiles[d]   = ((size_t)reg->global_dims[d]
                       + (size_t)reg->chunk_dims[d] - 1)
                      / (size_t)reg->chunk_dims[d];
    }

    double *buf = calloc(tile_elems, sizeof(double));
    if (!buf) {
        registry_destroy(reg); H5Dclose(dset); H5Fclose(fid);
        return -1;
    }

    /* Scan so we know which tiles are on disk. */
    registry_scan_file(dset, reg);

    double max_err = 0.0;
    size_t tile[MAX_RANK] = {0};
    do {
        hsize_t phys_off[MAX_RANK];
        for (int d = 0; d < rank; d++)
            phys_off[d] = (hsize_t)tile[d] * reg->chunk_dims[d];

        /* Zero the buffer; read_chunk_fast pre-zeros unwritten regions. */
        memset(buf, 0, tile_elems * sizeof(double));
        if (read_chunk_fast(dset, phys_off, buf, rank,
                            reg->chunk_dims) < 0) {
            free(buf); registry_destroy(reg);
            H5Dclose(dset); H5Fclose(fid);
            return -1;
        }

        /* Only check physical (non-padding) elements. */
        for (int d = 0; d < rank; d++) {
            hsize_t lo = phys_off[d];
            hsize_t hi = lo + reg->chunk_dims[d];
            if (hi > reg->global_dims[d]) hi = reg->global_dims[d];
            (void)lo; (void)hi;
        }

        /* Compute physical extents for this tile. */
        size_t phys[MAX_RANK];
        for (int d = 0; d < rank; d++) {
            hsize_t rem = reg->global_dims[d] - phys_off[d];
            phys[d] = (size_t)((rem < reg->chunk_dims[d])
                               ? rem : reg->chunk_dims[d]);
        }

        /* Build strides for the nominal chunk layout. */
        size_t strides[MAX_RANK];
        {
            size_t dims_sz[MAX_RANK];
            for (int d = 0; d < rank; d++)
                dims_sz[d] = (size_t)reg->chunk_dims[d];
            compute_strides((size_t)rank, dims_sz, strides);
        }

        /* Iterate over physical elements only. */
        size_t elem_coord[MAX_RANK] = {0};
        do {
            size_t idx = compute_flat_index((size_t)rank, elem_coord, strides);
            double err = fabs(buf[idx] - expected);
            if (err > max_err) max_err = err;
        } while (odometer_step((size_t)rank, elem_coord, phys));
    } while (odometer_step((size_t)rank, tile, ntiles));

    *max_err_out = max_err;
    free(buf);
    registry_destroy(reg);
    H5Dclose(dset);
    H5Fclose(fid);
    return 0;
}

/* Same as verify_fp64 but for COMPLEX128. */
static int verify_complex(const char *file,
                           double _Complex expected,
                           double *max_err_out)
{
    hid_t fid = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) return -1;
    hid_t dset = dset_open_no_cache(fid, "tensor");
    if (dset < 0) { H5Fclose(fid); return -1; }

    TensorRegistry *reg = registry_create_from_dset(dset);
    if (!reg) { H5Dclose(dset); H5Fclose(fid); return -1; }

    int    rank      = reg->rank;
    size_t elem_size = sizeof(double _Complex);
    size_t tile_elems = 1;
    size_t ntiles[MAX_RANK];
    for (int d = 0; d < rank; d++) {
        tile_elems *= (size_t)reg->chunk_dims[d];
        ntiles[d]   = ((size_t)reg->global_dims[d]
                       + (size_t)reg->chunk_dims[d] - 1)
                      / (size_t)reg->chunk_dims[d];
    }

    hid_t h5ctype = create_h5_complex_type();
    if (h5ctype < 0) {
        registry_destroy(reg); H5Dclose(dset); H5Fclose(fid);
        return -1;
    }

    double _Complex *buf = calloc(tile_elems, elem_size);
    if (!buf) {
        H5Tclose(h5ctype);
        registry_destroy(reg); H5Dclose(dset); H5Fclose(fid);
        return -1;
    }

    registry_scan_file(dset, reg);

    double max_err = 0.0;
    size_t tile[MAX_RANK] = {0};
    do {
        hsize_t phys_off[MAX_RANK];
        for (int d = 0; d < rank; d++)
            phys_off[d] = (hsize_t)tile[d] * reg->chunk_dims[d];

        memset(buf, 0, tile_elems * elem_size);
        if (read_chunk_typed(dset, phys_off, buf, elem_size,
                             rank, reg->chunk_dims, h5ctype) < 0) {
            free(buf); H5Tclose(h5ctype);
            registry_destroy(reg); H5Dclose(dset); H5Fclose(fid);
            return -1;
        }

        size_t phys[MAX_RANK];
        for (int d = 0; d < rank; d++) {
            hsize_t rem = reg->global_dims[d] - phys_off[d];
            phys[d] = (size_t)((rem < reg->chunk_dims[d])
                               ? rem : reg->chunk_dims[d]);
        }

        size_t strides[MAX_RANK];
        {
            size_t dims_sz[MAX_RANK];
            for (int d = 0; d < rank; d++)
                dims_sz[d] = (size_t)reg->chunk_dims[d];
            compute_strides((size_t)rank, dims_sz, strides);
        }

        size_t ec[MAX_RANK] = {0};
        do {
            size_t idx = compute_flat_index((size_t)rank, ec, strides);
            double err = cabs(buf[idx] - expected);
            if (err > max_err) max_err = err;
        } while (odometer_step((size_t)rank, ec, phys));
    } while (odometer_step((size_t)rank, tile, ntiles));

    *max_err_out = max_err;
    free(buf);
    H5Tclose(h5ctype);
    registry_destroy(reg);
    H5Dclose(dset);
    H5Fclose(fid);
    return 0;
}

/* ----------------------------------------------------------------------- */
/* T1 — create FP64 rank-2, verify shape and chunk dims are stored          */
/* ----------------------------------------------------------------------- */
static int t1_create_fp64(void)
{
    printf("\n=== T1: create FP64 rank-2, verify shape / chunk dims ===\n");
    tensor_engine_config_t cfg = {0};
    tensor_engine_t *eng = tensor_engine_init(&cfg);
    CHECK(eng != NULL, "engine init");
    if (!eng) return 1;

    const size_t shape[2] = {64, 48};
    int rc = tensor_engine_create(eng, "cf_t1_A.h5", 2, shape,
                                  TENSOR_DTYPE_FP64);
    CHECK(rc == TENSOR_ENGINE_OK, "create returns OK");

    int rank; hsize_t global[MAX_RANK], chunk[MAX_RANK];
    tensor_dtype_t dtype;
    CHECK(read_dset_info("cf_t1_A.h5", &rank, global, chunk, &dtype) == 0,
          "read_dset_info succeeds");
    CHECK(rank == 2, "rank == 2");
    CHECK(global[0] == 64, "global[0] == 64");
    CHECK(global[1] == 48, "global[1] == 48");
    CHECK(chunk[0] >= 1 && chunk[0] <= 64, "chunk[0] in valid range");
    CHECK(chunk[1] >= 1 && chunk[1] <= 48, "chunk[1] in valid range");
    CHECK(dtype == DTYPE_FP64, "dtype == FP64");
    printf("  chunk_dims = (%llu, %llu)\n",
           (unsigned long long)chunk[0], (unsigned long long)chunk[1]);

    tensor_engine_free(eng);
    return (g_fail == 0) ? 0 : 1;
}

/* ----------------------------------------------------------------------- */
/* T2 — create COMPLEX128 rank-3, verify dtype                              */
/* ----------------------------------------------------------------------- */
static int t2_create_complex(void)
{
    printf("\n=== T2: create COMPLEX128 rank-3, verify dtype ===\n");
    tensor_engine_config_t cfg = {0};
    tensor_engine_t *eng = tensor_engine_init(&cfg);
    CHECK(eng != NULL, "engine init");
    if (!eng) return 1;

    const size_t shape[3] = {12, 10, 8};
    int rc = tensor_engine_create(eng, "cf_t2_A.h5", 3, shape,
                                  TENSOR_DTYPE_COMPLEX128);
    CHECK(rc == TENSOR_ENGINE_OK, "create returns OK");

    int rank; hsize_t global[MAX_RANK], chunk[MAX_RANK];
    tensor_dtype_t dtype;
    CHECK(read_dset_info("cf_t2_A.h5", &rank, global, chunk, &dtype) == 0,
          "read_dset_info succeeds");
    CHECK(rank == 3, "rank == 3");
    CHECK(global[0] == 12, "global[0] == 12");
    CHECK(global[1] == 10, "global[1] == 10");
    CHECK(global[2] == 8,  "global[2] == 8");
    CHECK(dtype == DTYPE_COMPLEX128, "dtype == COMPLEX128");
    printf("  chunk_dims = (%llu, %llu, %llu)\n",
           (unsigned long long)chunk[0],
           (unsigned long long)chunk[1],
           (unsigned long long)chunk[2]);

    tensor_engine_free(eng);
    return (g_fail == 0) ? 0 : 1;
}

/* ----------------------------------------------------------------------- */
/* T3 — tile_bytes config changes chunk dims                                */
/* ----------------------------------------------------------------------- */
static int t3_tile_bytes_config(void)
{
    printf("\n=== T3: tile_bytes config controls chunk dimensions ===\n");

    /* Small tile: 64 KiB → for FP64 rank-2 that is 64*1024/8 = 8192 elems,
     * side ≈ 91.  For a 200×200 tensor the chunk should be much smaller
     * than the full dimension. */
    tensor_engine_config_t cfg_small = {.tile_bytes = 64UL * 1024};
    tensor_engine_t *eng_small = tensor_engine_init(&cfg_small);
    CHECK(eng_small != NULL, "engine_small init");

    /* Large tile: 4 MiB → side ≈ 724 for rank-2, so clamped to global dims. */
    tensor_engine_config_t cfg_large = {.tile_bytes = 4UL * 1024 * 1024};
    tensor_engine_t *eng_large = tensor_engine_init(&cfg_large);
    CHECK(eng_large != NULL, "engine_large init");

    const size_t shape[2] = {200, 200};
    CHECK(tensor_engine_create(eng_small, "cf_t3_small.h5", 2, shape,
                               TENSOR_DTYPE_FP64) == TENSOR_ENGINE_OK,
          "create small tile");
    CHECK(tensor_engine_create(eng_large, "cf_t3_large.h5", 2, shape,
                               TENSOR_DTYPE_FP64) == TENSOR_ENGINE_OK,
          "create large tile");

    int rank_s, rank_l;
    hsize_t gs[MAX_RANK], cs[MAX_RANK], gl[MAX_RANK], cl[MAX_RANK];
    tensor_dtype_t ds, dl;
    read_dset_info("cf_t3_small.h5", &rank_s, gs, cs, &ds);
    read_dset_info("cf_t3_large.h5", &rank_l, gl, cl, &dl);

    printf("  small tile chunk = (%llu, %llu)\n",
           (unsigned long long)cs[0], (unsigned long long)cs[1]);
    printf("  large tile chunk = (%llu, %llu)\n",
           (unsigned long long)cl[0], (unsigned long long)cl[1]);

    /* Small config should produce strictly smaller chunks than large. */
    CHECK(cs[0] < cl[0] || cs[1] < cl[1],
          "small tile_bytes → smaller chunks than large tile_bytes");

    /* Large config: chunk clamped ≤ global dim (200). */
    CHECK(cl[0] <= 200 && cl[1] <= 200, "large chunks clamped to global dims");

    tensor_engine_free(eng_small);
    tensor_engine_free(eng_large);
    return (g_fail == 0) ? 0 : 1;
}

/* ----------------------------------------------------------------------- */
/* T4 — fill FP64, read every element back and verify                       */
/* ----------------------------------------------------------------------- */
static int t4_fill_fp64(void)
{
    printf("\n=== T4: fill FP64 tensor, read back every element ===\n");
    tensor_engine_config_t cfg = {.tile_bytes = 64UL * 1024};
    tensor_engine_t *eng = tensor_engine_init(&cfg);
    CHECK(eng != NULL, "engine init");
    if (!eng) return 1;

    /* Non-divisible shape so boundary tiles are exercised. */
    const size_t shape[2] = {17, 11};
    CHECK(tensor_engine_create(eng, "cf_t4_A.h5", 2, shape,
                               TENSOR_DTYPE_FP64) == TENSOR_ENGINE_OK,
          "create");

    double fill_val = 7.5;
    CHECK(tensor_engine_fill(eng, "cf_t4_A.h5", &fill_val) == TENSOR_ENGINE_OK,
          "fill returns OK");

    double max_err = 0.0;
    int rc = verify_fp64("cf_t4_A.h5", fill_val, &max_err);
    CHECK(rc == 0, "verify_fp64 succeeds");
    CHECK(max_err == 0.0, "all 187 elements == 7.5 (max_err=0)");
    printf("  max_err = %.2e\n", max_err);

    tensor_engine_free(eng);
    return (g_fail == 0) ? 0 : 1;
}

/* ----------------------------------------------------------------------- */
/* T5 — fill COMPLEX128, read every element back and verify                 */
/* ----------------------------------------------------------------------- */
static int t5_fill_complex(void)
{
    printf("\n=== T5: fill COMPLEX128 tensor, read back every element ===\n");
    tensor_engine_config_t cfg = {.tile_bytes = 64UL * 1024};
    tensor_engine_t *eng = tensor_engine_init(&cfg);
    CHECK(eng != NULL, "engine init");
    if (!eng) return 1;

    const size_t shape[2] = {9, 7};
    CHECK(tensor_engine_create(eng, "cf_t5_A.h5", 2, shape,
                               TENSOR_DTYPE_COMPLEX128) == TENSOR_ENGINE_OK,
          "create");

    double _Complex fill_val = 3.0 + 4.0 * _Complex_I;
    CHECK(tensor_engine_fill(eng, "cf_t5_A.h5", &fill_val) == TENSOR_ENGINE_OK,
          "fill returns OK");

    double max_err = 0.0;
    int rc = verify_complex("cf_t5_A.h5", fill_val, &max_err);
    CHECK(rc == 0, "verify_complex succeeds");
    CHECK(max_err == 0.0, "all 63 elements == 3+4i (max_err=0)");
    printf("  max_err = %.2e\n", max_err);

    tensor_engine_free(eng);
    return (g_fail == 0) ? 0 : 1;
}

/* ----------------------------------------------------------------------- */
/* T6 — boundary tiles: odd shapes fill completely                          */
/* ----------------------------------------------------------------------- */
static int t6_boundary_fill(void)
{
    printf("\n=== T6: boundary tiles filled completely (rank-3) ===\n");
    /* Use a very small tile so lots of boundary tiles are exercised. */
    tensor_engine_config_t cfg = {.tile_bytes = 16 * 1024};
    tensor_engine_t *eng = tensor_engine_init(&cfg);
    CHECK(eng != NULL, "engine init");
    if (!eng) return 1;

    /* Dimensions chosen so no dim is divisible by the chunk side.
     * With tile_bytes=16 KiB: target_elems = 16384/8 = 2048,
     * side = cbrt(2048) ≈ 12.7 → chunk_side = 13.
     * Extents 10, 7, 5 are all smaller than 13, so each dim gets
     * chunk = min(13, dim) = that dim → one tile per dim, no boundary issue.
     * Use larger dims to force boundaries. */
    const size_t shape[3] = {25, 19, 13};
    CHECK(tensor_engine_create(eng, "cf_t6_A.h5", 3, shape,
                               TENSOR_DTYPE_FP64) == TENSOR_ENGINE_OK,
          "create rank-3");

    double fill_val = -2.0;
    CHECK(tensor_engine_fill(eng, "cf_t6_A.h5", &fill_val) == TENSOR_ENGINE_OK,
          "fill returns OK");

    double max_err = 0.0;
    int rc = verify_fp64("cf_t6_A.h5", fill_val, &max_err);
    CHECK(rc == 0, "verify_fp64 succeeds");
    /* 25*19*13 = 6175 elements, all must be -2. */
    CHECK(max_err == 0.0, "all 6175 elements == -2.0 (max_err=0)");
    printf("  max_err = %.2e  (shape 25×19×13)\n", max_err);

    tensor_engine_free(eng);
    return (g_fail == 0) ? 0 : 1;
}

/* ----------------------------------------------------------------------- */
/* T7 — end-to-end: create A+B via engine, fill, contract, verify C        */
/* ----------------------------------------------------------------------- */
static int t7_create_fill_contract(void)
{
    printf("\n=== T7: create + fill + contract, verify C ===\n");
    /*
     * A(6×8) filled with a_val, B(8×5) filled with b_val.
     * C(6×5) = A @ B via "ij,jk->ik".
     * Expected C[i,j] = sum_{k=0}^{7} a_val * b_val = 8 * a_val * b_val.
     */
    const double a_val    = 2.0;
    const double b_val    = 3.0;
    const double expected = 8.0 * a_val * b_val;   /* = 48.0 */

    tensor_engine_config_t cfg = {.tile_bytes = 16UL * 1024};
    tensor_engine_t *eng = tensor_engine_init(&cfg);
    CHECK(eng != NULL, "engine init");
    if (!eng) return 1;

    const size_t shA[2] = {6, 8};
    const size_t shB[2] = {8, 5};
    CHECK(tensor_engine_create(eng, "cf_t7_A.h5", 2, shA,
                               TENSOR_DTYPE_FP64) == TENSOR_ENGINE_OK,
          "create A");
    CHECK(tensor_engine_create(eng, "cf_t7_B.h5", 2, shB,
                               TENSOR_DTYPE_FP64) == TENSOR_ENGINE_OK,
          "create B");
    CHECK(tensor_engine_fill(eng, "cf_t7_A.h5", &a_val) == TENSOR_ENGINE_OK,
          "fill A with 2.0");
    CHECK(tensor_engine_fill(eng, "cf_t7_B.h5", &b_val) == TENSOR_ENGINE_OK,
          "fill B with 3.0");

    int rc = tensor_engine_contract(eng, "ij,jk->ik",
                                    "cf_t7_A.h5", "cf_t7_B.h5",
                                    "cf_t7_C.h5");
    CHECK(rc == TENSOR_ENGINE_OK, "contract returns OK");

    double max_err = 0.0;
    CHECK(verify_fp64("cf_t7_C.h5", expected, &max_err) == 0,
          "verify_fp64 succeeds");
    CHECK(max_err < 1e-9,
          "all 30 C elements == 48.0 (max_err < 1e-9)");
    printf("  expected=%.1f  max_err=%.2e\n", expected, max_err);

    tensor_engine_free(eng);
    return (g_fail == 0) ? 0 : 1;
}

/* ----------------------------------------------------------------------- */
/* main                                                                      */
/* ----------------------------------------------------------------------- */
int main(void)
{
    printf("=== test_create_fill: tensor_engine_create/fill() correctness ===\n");

    t1_create_fp64();
    t2_create_complex();
    t3_tile_bytes_config();
    t4_fill_fp64();
    t5_fill_complex();
    t6_boundary_fill();
    t7_create_fill_contract();

    printf("\n--- Results: %d passed, %d failed ---\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
