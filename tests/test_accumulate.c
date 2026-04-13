/*
 * tests/test_accumulate.c
 *
 * Correctness tests for tensor_engine_accumulate().
 *
 * Five test cases:
 *   T1 – rank-2 FP64        "ij,jk->ik"   two terms
 *   T2 – rank-2 FP64        "ij,jk->ki"   two terms (transposed output)
 *   T3 – rank-2 FP64        "ij,jk->ik"   three terms
 *   T4 – rank-3 FP64        "abc,cbd->ad" two terms
 *   T5 – rank-2 COMPLEX128  "ij,jk->ik"   two terms
 *
 * All test data is written as small chunked HDF5 files in the current working
 * directory (prefix "acc_t{N}_").  Dimensions are chosen so that chunk
 * boundaries do not divide evenly into the global shape, exercising boundary
 * tile handling.
 *
 * A companion Python script (tests/verify_accumulate.py) re-reads these files
 * and cross-checks the outputs independently against numpy.einsum.
 *
 * Build: added to CMakeLists.txt as test_accumulate.
 * Run:   ./build/test_accumulate
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
/* Tensor generation helpers                                                 */
/* ----------------------------------------------------------------------- */

/*
 * Create a rank-N FP64 HDF5 file and fill every tile with a constant value.
 * Boundary tiles are handled automatically by write_chunk_fast.
 */
static int gen_fp64(const char *fname, int rank,
                    const hsize_t *shape, const hsize_t *chunk,
                    double fill)
{
    if (create_chunked_dataset_einsum(fname, "tensor", rank,
                                      shape, chunk, DTYPE_FP64) < 0) {
        fprintf(stderr, "  gen_fp64: create failed '%s'\n", fname);
        return -1;
    }

    hid_t fid = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "  gen_fp64: H5Fopen failed '%s'\n", fname);
        return -1;
    }
    hid_t dset = dset_open_no_cache(fid, "tensor");
    if (dset < 0) {
        fprintf(stderr, "  gen_fp64: dset_open_no_cache failed\n");
        H5Fclose(fid);
        return -1;
    }

    size_t elems = 1;
    for (int d = 0; d < rank; d++) elems *= (size_t)chunk[d];
    double *buf = malloc(elems * sizeof(double));
    if (!buf) { H5Dclose(dset); H5Fclose(fid); return -1; }
    for (size_t i = 0; i < elems; i++) buf[i] = fill;

    size_t n_tiles[MAX_RANK], tile[MAX_RANK];
    memset(tile, 0, sizeof(tile));
    for (int d = 0; d < rank; d++)
        n_tiles[d] = ((size_t)shape[d] + (size_t)chunk[d] - 1) / (size_t)chunk[d];

    int ret = 0;
    do {
        hsize_t offset[MAX_RANK];
        for (int d = 0; d < rank; d++)
            offset[d] = (hsize_t)tile[d] * chunk[d];
        if (write_chunk_fast(dset, offset, buf, rank, chunk) < 0) {
            fprintf(stderr, "  gen_fp64: write_chunk_fast failed\n");
            ret = -1;
            break;
        }
    } while (odometer_step((size_t)rank, tile, n_tiles));

    free(buf);
    H5Dclose(dset);
    H5Fclose(fid);
    return ret;
}

/*
 * Create a rank-N COMPLEX128 HDF5 file and fill every tile with a constant
 * complex value (re + im*i).
 */
static int gen_complex128(const char *fname, int rank,
                          const hsize_t *shape, const hsize_t *chunk,
                          double re, double im)
{
    if (create_chunked_dataset_einsum(fname, "tensor", rank,
                                      shape, chunk, DTYPE_COMPLEX128) < 0) {
        fprintf(stderr, "  gen_complex128: create failed '%s'\n", fname);
        return -1;
    }

    hid_t fid = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "  gen_complex128: H5Fopen failed\n");
        return -1;
    }
    hid_t dset = dset_open_no_cache(fid, "tensor");
    if (dset < 0) {
        fprintf(stderr, "  gen_complex128: dset_open_no_cache failed\n");
        H5Fclose(fid);
        return -1;
    }

    hid_t h5ctype = create_h5_complex_type();
    if (h5ctype < 0) {
        fprintf(stderr, "  gen_complex128: create_h5_complex_type failed\n");
        H5Dclose(dset);
        H5Fclose(fid);
        return -1;
    }

    size_t elems = 1;
    for (int d = 0; d < rank; d++) elems *= (size_t)chunk[d];
    double _Complex *buf = malloc(elems * sizeof(double _Complex));
    if (!buf) {
        H5Tclose(h5ctype);
        H5Dclose(dset);
        H5Fclose(fid);
        return -1;
    }
    const double _Complex fill = CMPLX(re, im);
    for (size_t i = 0; i < elems; i++) buf[i] = fill;

    size_t n_tiles[MAX_RANK], tile[MAX_RANK];
    memset(tile, 0, sizeof(tile));
    for (int d = 0; d < rank; d++)
        n_tiles[d] = ((size_t)shape[d] + (size_t)chunk[d] - 1) / (size_t)chunk[d];

    int ret = 0;
    do {
        hsize_t offset[MAX_RANK];
        for (int d = 0; d < rank; d++)
            offset[d] = (hsize_t)tile[d] * chunk[d];
        if (write_chunk_typed(dset, offset, buf,
                              sizeof(double _Complex), rank,
                              chunk, h5ctype) < 0) {
            fprintf(stderr, "  gen_complex128: write_chunk_typed failed\n");
            ret = -1;
            break;
        }
    } while (odometer_step((size_t)rank, tile, n_tiles));

    free(buf);
    H5Tclose(h5ctype);
    H5Dclose(dset);
    H5Fclose(fid);
    return ret;
}

/* ----------------------------------------------------------------------- */
/* Verification helpers                                                      */
/* ----------------------------------------------------------------------- */

/*
 * Read the entire FP64 "tensor" dataset from fname and compare every element
 * against expected within abs tolerance tol.  Returns 0 on success, -1 on
 * any mismatch or I/O error.
 */
static int check_fp64_all(const char *fname, double expected, double tol)
{
    hid_t fid = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "  check_fp64_all: cannot open '%s'\n", fname);
        return -1;
    }
    hid_t dset = H5Dopen2(fid, "tensor", H5P_DEFAULT);
    if (dset < 0) { H5Fclose(fid); return -1; }

    hid_t space = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[MAX_RANK];
    H5Sget_simple_extent_dims(space, dims, NULL);
    H5Sclose(space);

    size_t total = 1;
    for (int d = 0; d < ndims; d++) total *= (size_t)dims[d];

    double *buf = malloc(total * sizeof(double));
    if (!buf) { H5Dclose(dset); H5Fclose(fid); return -1; }

    herr_t err = H5Dread(dset, H5T_NATIVE_DOUBLE,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset);
    H5Fclose(fid);

    if (err < 0) {
        fprintf(stderr, "  check_fp64_all: H5Dread failed\n");
        free(buf);
        return -1;
    }

    size_t mismatch = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < total; i++) {
        double e = fabs(buf[i] - expected);
        if (e > max_err) max_err = e;
        if (e > tol) mismatch++;
    }
    free(buf);

    if (mismatch > 0) {
        fprintf(stderr, "  FAIL: %zu/%zu elements wrong, max_err=%.2e, expected=%.6f\n",
                mismatch, total, max_err, expected);
        return -1;
    }
    printf("  PASS: %zu elements correct (max_err=%.2e)\n", total, max_err);
    return 0;
}

/*
 * Read the entire COMPLEX128 "tensor" dataset and compare every element
 * against (exp_re + exp_im*i) within abs tolerance tol.
 */
static int check_complex128_all(const char *fname,
                                double exp_re, double exp_im, double tol)
{
    hid_t fid = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "  check_complex128_all: cannot open '%s'\n", fname);
        return -1;
    }
    hid_t dset = H5Dopen2(fid, "tensor", H5P_DEFAULT);
    if (dset < 0) { H5Fclose(fid); return -1; }

    hid_t space = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[MAX_RANK];
    H5Sget_simple_extent_dims(space, dims, NULL);
    H5Sclose(space);

    size_t total = 1;
    for (int d = 0; d < ndims; d++) total *= (size_t)dims[d];

    double _Complex *buf = malloc(total * sizeof(double _Complex));
    if (!buf) { H5Dclose(dset); H5Fclose(fid); return -1; }

    hid_t h5ctype = create_h5_complex_type();
    herr_t err = H5Dread(dset, h5ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Tclose(h5ctype);
    H5Dclose(dset);
    H5Fclose(fid);

    if (err < 0) {
        fprintf(stderr, "  check_complex128_all: H5Dread failed\n");
        free(buf);
        return -1;
    }

    size_t mismatch = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < total; i++) {
        double re_err = fabs(creal(buf[i]) - exp_re);
        double im_err = fabs(cimag(buf[i]) - exp_im);
        double e = sqrt(re_err * re_err + im_err * im_err);
        if (e > max_err) max_err = e;
        if (e > tol) mismatch++;
    }
    free(buf);

    if (mismatch > 0) {
        fprintf(stderr,
                "  FAIL: %zu/%zu elements wrong, max_err=%.2e, "
                "expected=(%.6f+%.6fi)\n",
                mismatch, total, max_err, exp_re, exp_im);
        return -1;
    }
    printf("  PASS: %zu elements correct (max_err=%.2e)\n", total, max_err);
    return 0;
}

/* ----------------------------------------------------------------------- */
/* T1: rank-2 FP64  "ij,jk->ik"  two terms                                 */
/*                                                                           */
/*  A shape (6×8), B shape (8×5), chunk=4 — boundary tiles in M and N dims  */
/*  Term1: fill_A=2.0, fill_B=3.0  →  K * 2*3 = 8*6  = 48.0               */
/*  Term2: fill_A=1.0, fill_B=4.0  →  K * 1*4 = 8*4  = 32.0               */
/*  Expected: 80.0 (every output element)                                   */
/* ----------------------------------------------------------------------- */

static int test_rank2_two_terms(tensor_engine_t *eng)
{
    printf("\n=== T1: ij,jk->ik  FP64  two terms ===\n");
    printf("  A:(6×8)  B:(8×5)  chunk=4   expected=80.0\n");

    hsize_t shA[2] = {6, 8}, ckA[2] = {4, 4};
    hsize_t shB[2] = {8, 5}, ckB[2] = {4, 4};

    if (gen_fp64("acc_t1_A1.h5", 2, shA, ckA, 2.0) < 0) return -1;
    if (gen_fp64("acc_t1_B1.h5", 2, shB, ckB, 3.0) < 0) return -1;
    if (gen_fp64("acc_t1_A2.h5", 2, shA, ckA, 1.0) < 0) return -1;
    if (gen_fp64("acc_t1_B2.h5", 2, shB, ckB, 4.0) < 0) return -1;

    if (tensor_engine_contract(eng, "ij,jk->ik",
                               "acc_t1_A1.h5", "acc_t1_B1.h5",
                               "acc_t1_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T1 contract failed\n");
        return -1;
    }
    if (tensor_engine_accumulate(eng, "ij,jk->ik",
                                 "acc_t1_A2.h5", "acc_t1_B2.h5",
                                 "acc_t1_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T1 accumulate failed\n");
        return -1;
    }

    /* C[i,k] = K*(a1*b1 + a2*b2) = 8*(6+4) = 80 */
    return check_fp64_all("acc_t1_C.h5", 8.0 * (2.0*3.0 + 1.0*4.0), 1e-10);
}

/* ----------------------------------------------------------------------- */
/* T2: rank-2 FP64  "ij,jk->ki"  two terms (transposed output)             */
/*                                                                           */
/*  Same shapes and fills as T1; C shape becomes (5×6) instead of (6×5).   */
/*  Expected element value is still 80.0 (constant fill, just permuted).    */
/* ----------------------------------------------------------------------- */

static int test_rank2_transposed(tensor_engine_t *eng)
{
    printf("\n=== T2: ij,jk->ki  FP64  two terms (transposed output) ===\n");
    printf("  A:(6×8)  B:(8×5)  chunk=4   C:(5×6)  expected=80.0\n");

    hsize_t shA[2] = {6, 8}, ckA[2] = {4, 4};
    hsize_t shB[2] = {8, 5}, ckB[2] = {4, 4};

    if (gen_fp64("acc_t2_A1.h5", 2, shA, ckA, 2.0) < 0) return -1;
    if (gen_fp64("acc_t2_B1.h5", 2, shB, ckB, 3.0) < 0) return -1;
    if (gen_fp64("acc_t2_A2.h5", 2, shA, ckA, 1.0) < 0) return -1;
    if (gen_fp64("acc_t2_B2.h5", 2, shB, ckB, 4.0) < 0) return -1;

    if (tensor_engine_contract(eng, "ij,jk->ki",
                               "acc_t2_A1.h5", "acc_t2_B1.h5",
                               "acc_t2_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T2 contract failed\n");
        return -1;
    }
    if (tensor_engine_accumulate(eng, "ij,jk->ki",
                                 "acc_t2_A2.h5", "acc_t2_B2.h5",
                                 "acc_t2_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T2 accumulate failed\n");
        return -1;
    }

    return check_fp64_all("acc_t2_C.h5", 8.0 * (2.0*3.0 + 1.0*4.0), 1e-10);
}

/* ----------------------------------------------------------------------- */
/* T3: rank-2 FP64  "ij,jk->ik"  three terms                               */
/*                                                                           */
/*  A shape (4×5), B shape (5×3), chunk=3 — boundary tiles in all dims      */
/*  Term1: A=1.0, B=2.0  →  K*1*2 = 5*2  = 10                              */
/*  Term2: A=3.0, B=1.0  →  K*3*1 = 5*3  = 15                              */
/*  Term3: A=2.0, B=2.0  →  K*2*2 = 5*4  = 20                              */
/*  Expected: 45.0                                                           */
/* ----------------------------------------------------------------------- */

static int test_rank2_three_terms(tensor_engine_t *eng)
{
    printf("\n=== T3: ij,jk->ik  FP64  three terms ===\n");
    printf("  A:(4×5)  B:(5×3)  chunk=3   expected=45.0\n");

    hsize_t shA[2] = {4, 5}, ckA[2] = {3, 3};
    hsize_t shB[2] = {5, 3}, ckB[2] = {3, 3};

    if (gen_fp64("acc_t3_A1.h5", 2, shA, ckA, 1.0) < 0) return -1;
    if (gen_fp64("acc_t3_B1.h5", 2, shB, ckB, 2.0) < 0) return -1;
    if (gen_fp64("acc_t3_A2.h5", 2, shA, ckA, 3.0) < 0) return -1;
    if (gen_fp64("acc_t3_B2.h5", 2, shB, ckB, 1.0) < 0) return -1;
    if (gen_fp64("acc_t3_A3.h5", 2, shA, ckA, 2.0) < 0) return -1;
    if (gen_fp64("acc_t3_B3.h5", 2, shB, ckB, 2.0) < 0) return -1;

    if (tensor_engine_contract(eng, "ij,jk->ik",
                               "acc_t3_A1.h5", "acc_t3_B1.h5",
                               "acc_t3_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T3 first contract failed\n");
        return -1;
    }
    if (tensor_engine_accumulate(eng, "ij,jk->ik",
                                 "acc_t3_A2.h5", "acc_t3_B2.h5",
                                 "acc_t3_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T3 first accumulate failed\n");
        return -1;
    }
    if (tensor_engine_accumulate(eng, "ij,jk->ik",
                                 "acc_t3_A3.h5", "acc_t3_B3.h5",
                                 "acc_t3_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T3 second accumulate failed\n");
        return -1;
    }

    /* C[i,k] = K*(1*2 + 3*1 + 2*2) = 5*(2+3+4) = 45 */
    return check_fp64_all("acc_t3_C.h5", 5.0 * (1.0*2.0 + 3.0*1.0 + 2.0*2.0), 1e-10);
}

/* ----------------------------------------------------------------------- */
/* T4: rank-3 FP64  "abc,cbd->ad"  two terms                                */
/*                                                                           */
/*  A shape (4,5,3) — indices (a,b,c)                                       */
/*  B shape (3,5,6) — indices (c,b,d)                                       */
/*  chunk=3 in every dimension                                               */
/*  contracted indices: b (dim=5), c (dim=3)                                */
/*  C[a,d] = sum_{b,c} A[a,b,c]*B[c,b,d]                                   */
/*  Term1: A=1.0, B=2.0  →  b*c * 1*2 = 5*3*2 = 30                         */
/*  Term2: A=2.0, B=1.5  →  b*c * 2*1.5 = 5*3*3 = 45                       */
/*  Expected: 75.0                                                           */
/* ----------------------------------------------------------------------- */

static int test_rank3_two_terms(tensor_engine_t *eng)
{
    printf("\n=== T4: abc,cbd->ad  FP64  two terms ===\n");
    printf("  A:(4×5×3)  B:(3×5×6)  chunk=3   expected=75.0\n");

    hsize_t shA[3] = {4, 5, 3}, ckA[3] = {3, 3, 3};
    hsize_t shB[3] = {3, 5, 6}, ckB[3] = {3, 3, 3};

    if (gen_fp64("acc_t4_A1.h5", 3, shA, ckA, 1.0) < 0) return -1;
    if (gen_fp64("acc_t4_B1.h5", 3, shB, ckB, 2.0) < 0) return -1;
    if (gen_fp64("acc_t4_A2.h5", 3, shA, ckA, 2.0) < 0) return -1;
    if (gen_fp64("acc_t4_B2.h5", 3, shB, ckB, 1.5) < 0) return -1;

    if (tensor_engine_contract(eng, "abc,cbd->ad",
                               "acc_t4_A1.h5", "acc_t4_B1.h5",
                               "acc_t4_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T4 contract failed\n");
        return -1;
    }
    if (tensor_engine_accumulate(eng, "abc,cbd->ad",
                                 "acc_t4_A2.h5", "acc_t4_B2.h5",
                                 "acc_t4_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T4 accumulate failed\n");
        return -1;
    }

    /* C[a,d] = b*c*(a1*b1 + a2*b2) = 5*3*(1*2 + 2*1.5) = 15*(2+3) = 75 */
    return check_fp64_all("acc_t4_C.h5", 5.0 * 3.0 * (1.0*2.0 + 2.0*1.5), 1e-10);
}

/* ----------------------------------------------------------------------- */
/* T5: rank-2 COMPLEX128  "ij,jk->ik"  two terms                            */
/*                                                                           */
/*  A shape (4×6), B shape (6×4), chunk=3 — boundary tiles                  */
/*  Term1: A=(2+1i), B=(1+2i)                                               */
/*    (2+1i)*(1+2i) = 2+4i+i+2i² = 2-2+5i = 0+5i  per j                   */
/*    C[i,k] = K*(0+5i) = 0+30i                                             */
/*  Term2: A=(1+0i), B=(0+3i)                                               */
/*    (1+0i)*(0+3i) = 0+3i  per j                                           */
/*    C[i,k] += K*(0+3i) = 0+18i                                            */
/*  Expected: 0+48i                                                          */
/* ----------------------------------------------------------------------- */

static int test_complex128_two_terms(tensor_engine_t *eng)
{
    printf("\n=== T5: ij,jk->ik  COMPLEX128  two terms ===\n");
    printf("  A:(4×6)  B:(6×4)  chunk=3   expected=(0+48i)\n");

    hsize_t shA[2] = {4, 6}, ckA[2] = {3, 3};
    hsize_t shB[2] = {6, 4}, ckB[2] = {3, 3};

    if (gen_complex128("acc_t5_A1.h5", 2, shA, ckA, 2.0, 1.0) < 0) return -1;
    if (gen_complex128("acc_t5_B1.h5", 2, shB, ckB, 1.0, 2.0) < 0) return -1;
    if (gen_complex128("acc_t5_A2.h5", 2, shA, ckA, 1.0, 0.0) < 0) return -1;
    if (gen_complex128("acc_t5_B2.h5", 2, shB, ckB, 0.0, 3.0) < 0) return -1;

    if (tensor_engine_contract(eng, "ij,jk->ik",
                               "acc_t5_A1.h5", "acc_t5_B1.h5",
                               "acc_t5_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T5 contract failed\n");
        return -1;
    }
    if (tensor_engine_accumulate(eng, "ij,jk->ik",
                                 "acc_t5_A2.h5", "acc_t5_B2.h5",
                                 "acc_t5_C.h5") != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  T5 accumulate failed\n");
        return -1;
    }

    /* K=6 contractions per (i,k):
       term1: (2+i)(1+2i) = 0+5i  → 6*(0+5i) = 0+30i
       term2: (1+0i)(0+3i) = 0+3i → 6*(0+3i) = 0+18i
       total: 0+48i  */
    return check_complex128_all("acc_t5_C.h5", 0.0, 48.0, 1e-10);
}

/* ----------------------------------------------------------------------- */
/* main                                                                      */
/* ----------------------------------------------------------------------- */

int main(void)
{
    printf("=== test_accumulate: tensor_engine_accumulate() correctness ===\n");

    tensor_engine_config_t cfg = {0};   /* auto-tune pool and tile size */
    tensor_engine_t *eng = tensor_engine_init(&cfg);
    if (!eng) {
        fprintf(stderr, "tensor_engine_init failed\n");
        return 1;
    }

    int pass = 0, fail = 0;

#define RUN(fn) do { \
    if ((fn) == 0) { pass++; } \
    else           { fail++; } \
} while (0)

    RUN(test_rank2_two_terms(eng));
    RUN(test_rank2_transposed(eng));
    RUN(test_rank2_three_terms(eng));
    RUN(test_rank3_two_terms(eng));
    RUN(test_complex128_two_terms(eng));

#undef RUN

    tensor_engine_free(eng);

    printf("\n--- Results: %d passed, %d failed ---\n", pass, fail);
    if (pass + fail > 0)
        printf("(Run 'python3 tests/verify_accumulate.py' to cross-check against numpy)\n");
    return (fail == 0) ? 0 : 1;
}
