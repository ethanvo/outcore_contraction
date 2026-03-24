/*
 * bench/run_all.c — Consolidated out-of-core tensor contraction benchmark suite
 *
 * Uses the public tensor_engine.h API exclusively; no internal headers.
 *
 * Two benchmark cases are run in sequence:
 *
 *   Case 1 — Small (smoke-test, ~655 MB each)
 *     Expression : ijab,akbl->klji   dtype: COMPLEX128
 *     Global dim : 80 per index      chunk: 16 per index (1 MiB/tile)
 *     Pool cap   : 128 MiB           (forces out-of-core behaviour)
 *
 *   Case 2 — Large compute-bound (~40 GiB each)
 *     Expression : ijab,akbl->klji   dtype: COMPLEX128
 *     Global dim : 224 per index     chunk: 32 per index (16 MiB/tile)
 *     Pool cap   : 512 MiB           (out-of-core on any machine)
 *
 * Input files are generated inline if they do not already exist.
 * A Markdown summary table is printed after both cases complete.
 *
 * Override compile-time defaults:
 *   -DSMALL_DIM=N       (default 80)
 *   -DSMALL_CHUNK=N     (default 16)
 *   -DLARGE_DIM=N       (default 224)
 *   -DLARGE_CHUNK=N     (default 32)
 *   -DSKIP_LARGE=1      skip Case 2 entirely (useful for CI)
 */

#include "tensor_engine.h"

/* Internal headers are only used for file generation; the contraction
 * itself goes entirely through the public API. */
#include "tensor_store.h"
#include "odometer.h"
#include <hdf5.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Compile-time knobs
 * -----------------------------------------------------------------------*/

#ifndef SMALL_DIM
#  define SMALL_DIM    80
#endif
#ifndef SMALL_CHUNK
#  define SMALL_CHUNK  16
#endif
#ifndef SMALL_POOL_MB
#  define SMALL_POOL_MB 128
#endif

#ifndef LARGE_DIM
#  define LARGE_DIM    224
#endif
#ifndef LARGE_CHUNK
#  define LARGE_CHUNK  32
#endif
#ifndef LARGE_POOL_MB
#  define LARGE_POOL_MB 512
#endif

#define RANK 4
#define DSET "tensor"
#define EXPR "ijab,akbl->klji"

/* -------------------------------------------------------------------------
 * Helpers
 * -----------------------------------------------------------------------*/

static double elapsed_s(const struct timespec *a, const struct timespec *b)
{
    return (double)(b->tv_sec  - a->tv_sec)
         + (double)(b->tv_nsec - a->tv_nsec) * 1e-9;
}

static double pow4(double x) { return x * x * x * x; }
static double pow6(double x) { return x * x * x * x * x * x; }

/*
 * generate_tensor_file — write a rank-4 COMPLEX128 tensor filled with
 *   fill = (1 + 0.5i) to `fname` / dataset `DSET`.
 * Skips generation if the file already exists.
 * Returns 0 on success, -1 on error.
 */
static int generate_tensor_file(const char *fname, int global_dim, int chunk_dim)
{
    /* Fast-path: already on disk. */
    {
        FILE *f = fopen(fname, "rb");
        if (f) { fclose(f); printf("  %s exists, skipping generation.\n", fname); return 0; }
    }

    printf("  Generating %s  (%d^%d COMPLEX128, chunk %d^%d) ...\n",
           fname, global_dim, RANK, chunk_dim, RANK);

    hsize_t shape[RANK], chunk_dims[RANK];
    for (int d = 0; d < RANK; d++) {
        shape[d]      = (hsize_t)global_dim;
        chunk_dims[d] = (hsize_t)chunk_dim;
    }

    if (create_chunked_dataset_einsum(fname, DSET, RANK, shape, chunk_dims,
                                      DTYPE_COMPLEX128) < 0) {
        fprintf(stderr, "  ERROR: create_chunked_dataset_einsum failed for %s\n", fname);
        return -1;
    }

    hid_t fid = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (fid < 0) { fprintf(stderr, "  ERROR: H5Fopen failed\n"); return -1; }

    hid_t dset = dset_open_no_cache(fid, DSET);
    if (dset < 0) {
        fprintf(stderr, "  ERROR: dset_open_no_cache failed\n");
        H5Fclose(fid); return -1;
    }

    hid_t h5ctype = create_h5_complex_type();
    if (h5ctype < 0) {
        fprintf(stderr, "  ERROR: create_h5_complex_type failed\n");
        H5Dclose(dset); H5Fclose(fid); return -1;
    }

    size_t elems = 1;
    for (int d = 0; d < RANK; d++) elems *= (size_t)chunk_dim;
    size_t nbytes = elems * sizeof(double _Complex);

    void *raw = NULL;
    if (posix_memalign(&raw, 16384, nbytes) != 0) {
        fprintf(stderr, "  ERROR: posix_memalign failed\n");
        H5Tclose(h5ctype); H5Dclose(dset); H5Fclose(fid); return -1;
    }
    double _Complex *buf = (double _Complex *)raw;
    for (size_t i = 0; i < elems; i++) buf[i] = CMPLX(1.0, 0.5);

    size_t n_tiles[RANK], total_tiles = 1;
    for (int d = 0; d < RANK; d++) {
        n_tiles[d]   = (size_t)((global_dim + chunk_dim - 1) / chunk_dim);
        total_tiles *= n_tiles[d];
    }

    size_t tile_idx[RANK];
    memset(tile_idx, 0, sizeof(tile_idx));
    size_t written = 0;
    int ret = 0;

    do {
        hsize_t offset[RANK];
        for (int d = 0; d < RANK; d++) offset[d] = (hsize_t)tile_idx[d] * chunk_dims[d];

        if (write_chunk_typed(dset, offset, buf, sizeof(double _Complex),
                              RANK, chunk_dims, h5ctype) < 0) {
            fprintf(stderr, "  ERROR: write_chunk_typed failed\n");
            ret = -1; break;
        }
        written++;
        if (written % 50 == 0 || written == total_tiles) {
            printf("\r    %5.1f%%  (%zu/%zu tiles)",
                   100.0 * (double)written / (double)total_tiles,
                   written, total_tiles);
            fflush(stdout);
        }
    } while (odometer_step(RANK, tile_idx, n_tiles));

    printf("\n");
    free(raw);
    H5Tclose(h5ctype);
    H5Dclose(dset);
    H5Fclose(fid);
    return ret;
}

/* -------------------------------------------------------------------------
 * Result record
 * -----------------------------------------------------------------------*/

typedef struct {
    const char *label;
    int         global_dim;
    int         chunk_dim;
    double      tensor_gib;
    double      flops;
    double      elapsed_s;
    double      gflops;
    double      read_gib;
    double      write_gib;
    double      read_bw;
    double      write_bw;
    int         passed;
} bench_result_t;

/* -------------------------------------------------------------------------
 * Run one benchmark case
 * -----------------------------------------------------------------------*/

static bench_result_t run_case(const char *label,
                                const char *file_A,
                                const char *file_B,
                                const char *file_C,
                                int         global_dim,
                                int         chunk_dim,
                                size_t      pool_mb)
{
    bench_result_t r;
    memset(&r, 0, sizeof(r));
    r.label      = label;
    r.global_dim = global_dim;
    r.chunk_dim  = chunk_dim;

    /* Geometry */
    double elems  = pow4((double)global_dim);
    r.tensor_gib  = elems * 16.0 / (1024.0 * 1024.0 * 1024.0);
    r.flops       = 8.0 * pow6((double)global_dim);  /* 8 FLOPs/complex pair */
    r.read_gib    = 2.0 * r.tensor_gib;
    r.write_gib   = r.tensor_gib;

    printf("\n");
    printf("=================================================================\n");
    printf("  %s\n", label);
    printf("  Expression : %s    dtype : COMPLEX128\n", EXPR);
    printf("  Global dim : %d per index   chunk : %d per index\n",
           global_dim, chunk_dim);
    printf("  Tensor     : %.2f GiB each   FLOPs : %.3e\n",
           r.tensor_gib, r.flops);
    printf("  Pool cap   : %zu MiB\n", pool_mb);
    printf("=================================================================\n");

    /* Generate input files */
    if (generate_tensor_file(file_A, global_dim, chunk_dim) < 0) { r.passed = 0; return r; }
    if (generate_tensor_file(file_B, global_dim, chunk_dim) < 0) { r.passed = 0; return r; }

    /* Contract via the public API */
    tensor_engine_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.pool_mb = pool_mb;

    tensor_engine_t *eng = tensor_engine_init(&cfg);
    if (!eng) {
        fprintf(stderr, "  ERROR: tensor_engine_init failed\n");
        r.passed = 0; return r;
    }

    printf("\n--- Running contraction ---\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int rc = tensor_engine_contract(eng, EXPR, file_A, file_B, file_C);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    tensor_engine_free(eng);

    if (rc != TENSOR_ENGINE_OK) {
        fprintf(stderr, "  ERROR: tensor_engine_contract returned %d (%s)\n",
                rc, tensor_engine_strerror(rc));
        r.passed = 0; return r;
    }

    r.elapsed_s = elapsed_s(&t0, &t1);
    r.gflops    = (r.flops / r.elapsed_s) / 1.0e9;
    r.read_bw   = r.read_gib  / r.elapsed_s;
    r.write_bw  = r.write_gib / r.elapsed_s;

    printf("\n  Elapsed : %.3f s    GFLOPS : %.2f\n", r.elapsed_s, r.gflops);
    printf("  Read BW : %.2f GiB/s    Write BW : %.2f GiB/s\n",
           r.read_bw, r.write_bw);

    r.passed = 1;
    return r;
}

/* -------------------------------------------------------------------------
 * Markdown summary table
 * -----------------------------------------------------------------------*/

static void print_markdown_table(const bench_result_t *results, int n)
{
    printf("\n\n");
    printf("## Benchmark Results\n\n");

    /* Header */
    printf("| Case | Dimensions | Tensor Size | Chunk | "
           "Read | Write | Elapsed | GFLOPS |\n");
    printf("|------|-----------|-------------|-------|"
           "------|-------|---------|--------|\n");

    for (int i = 0; i < n; i++) {
        const bench_result_t *r = &results[i];
        if (!r->passed) {
            printf("| %s | %d^4 | %.2f GiB | %d^4 | — | — | FAILED | — |\n",
                   r->label, r->global_dim, r->tensor_gib, r->chunk_dim);
        } else {
            printf("| %s | %d^4 | %.2f GiB | %d^4 | "
                   "%.1f GiB/s | %.1f GiB/s | %.2f s | **%.1f** |\n",
                   r->label,
                   r->global_dim,
                   r->tensor_gib,
                   r->chunk_dim,
                   r->read_bw,
                   r->write_bw,
                   r->elapsed_s,
                   r->gflops);
        }
    }

    printf("\n");
    printf("> Expression: `%s`  "
           "Dtype: COMPLEX128  "
           "Backend: Apple Accelerate (AMX/vecLib)\n", EXPR);
    printf("> FLOPs counted as 8 per complex element pair "
           "(6 multiply + 2 add).\n");
    printf("> Read = A + B tiles streamed; Write = C tiles flushed.\n");
}

/* -------------------------------------------------------------------------
 * main
 * -----------------------------------------------------------------------*/

int main(void)
{
    printf("Out-of-Core Tensor Contraction Engine — Benchmark Suite\n");
    printf("Expression: %s\n", EXPR);

    bench_result_t results[2];
    int n = 0;

    /* ------------------------------------------------------------------ */
    /* Case 1 — Small                                                      */
    /* ------------------------------------------------------------------ */
    results[n++] = run_case(
        "Small  (80^4, ~655 MiB/tensor)",
        "small_A.h5", "small_B.h5", "small_C.h5",
        SMALL_DIM, SMALL_CHUNK, (size_t)SMALL_POOL_MB
    );

    /* ------------------------------------------------------------------ */
    /* Case 2 — Large compute-bound                                        */
    /* ------------------------------------------------------------------ */
#ifndef SKIP_LARGE
    results[n++] = run_case(
        "Large  (224^4, ~40 GiB/tensor)",
        "A_compute_40gb.h5", "B_compute_40gb.h5", "C_compute_40gb.h5",
        LARGE_DIM, LARGE_CHUNK, (size_t)LARGE_POOL_MB
    );
#endif

    /* ------------------------------------------------------------------ */
    /* Summary                                                             */
    /* ------------------------------------------------------------------ */
    print_markdown_table(results, n);

    /* Exit non-zero if any case failed. */
    for (int i = 0; i < n; i++) {
        if (!results[i].passed) return 1;
    }
    return 0;
}
