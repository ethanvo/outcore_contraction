/*
 * bench_odd_strides_40gb.c
 *
 * Odd-stride boundary stress test and benchmark for run_contraction_einsum.
 *
 * Contraction : ijab,akbl->klji    dtype: COMPLEX128
 *   C(k,l,j,i) = sum_{a,b} A(i,j,a,b) * B(a,k,b,l)
 *
 * Dimensions
 * ----------
 *   GLOBAL_DIM = 227   CHUNK_SIDE = 31
 *
 *   Fringe analysis:  227 mod 31 = 10
 *     Full tiles per dim : 7   (extent 31)
 *     Fringe tile per dim: 1   (extent 10)
 *     Tile grid per dim  : 8   (= ceil(227/31))
 *     Total tiles        : 8^4 = 4096 per tensor
 *
 * BLAS leading-dimension stress
 * -----------------------------
 *   Inner tile  31×31  lda=31, ldb=31, ldc=31   (standard)
 *   i-fringe    10×31  lda=31 but M=10           (M < lda)
 *   j-fringe    31×10  lda=10, N=10              (non-square row stride)
 *   Corner      10×10  lda=10, M=10, N=10        (double-fringe)
 *
 *   The engine must pass actual tile extents — not CHUNK_SIDE — to
 *   cblas_zgemm for every contracted (a,b) inner pair.
 *
 * 16 KB NVMe alignment
 * --------------------
 *   31^4 × 16 B = 14,776,336 B.  The engine rounds pool pages up to the
 *   next 16 KB boundary (14,794,752 B) via NVME_PAGE_BYTES alignment in
 *   pool_create and bytes_per_page rounding in run_contraction_einsum.
 *   The HDF5 chunk dimensions stay at 31 per axis; alignment is in RAM only.
 *
 * Files (relative to working directory / repo root)
 * --------------------------------------------------
 *   A_odd_42gb.h5   input  (generated here if absent)
 *   B_odd_42gb.h5   input  (generated here if absent)
 *   C_odd_42gb.h5   output (created / overwritten by the engine)
 *
 * Performance output
 * ------------------
 *   Total elapsed time (s), GFLOPS, NVMe read BW (GB/s), NVMe write BW (GB/s)
 *
 * No Python, no correctness loops.  Pure C11 hardware + boundary-logic stress.
 */

#include "engine.h"
#include "registry.h"
#include "tensor_store.h"
#include "odometer.h"
#include <hdf5.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ----------------------------------------------------------------------- */
/* Dimensions                                                                */
/* ----------------------------------------------------------------------- */

#define GLOBAL_DIM   227
#define CHUNK_SIDE    31
#define GEN_RANK       4

/* 227 mod 31 = 10 — the fringe tile extent in every dimension. */
#define FRINGE_EXTENT  (GLOBAL_DIM % CHUNK_SIDE)   /* = 10 */
#define TILES_PER_DIM  ((GLOBAL_DIM + CHUNK_SIDE - 1) / CHUNK_SIDE)  /* = 8 */

/* Fill value: avoids exact-integer cancellation while keeping the
 * contraction result predictable for manual spot-checks if desired.     */
#define FILL_REAL  1.0
#define FILL_IMAG  0.1

#define FILE_A  "A_odd_42gb.h5"
#define FILE_B  "B_odd_42gb.h5"
#define FILE_C  "C_odd_42gb.h5"
#define DSET    "tensor"

/* ----------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ----------------------------------------------------------------------- */

static double elapsed_s(const struct timespec *a, const struct timespec *b)
{
    return (double)(b->tv_sec  - a->tv_sec)
         + (double)(b->tv_nsec - a->tv_nsec) * 1e-9;
}

/* ----------------------------------------------------------------------- */
/* generate_file                                                              */
/*                                                                           */
/* Creates fname with a single rank-4 COMPLEX128 dataset of shape 227^4,   */
/* chunked at 31^4.  Every element is set to (FILL_REAL + FILL_IMAG*i).    */
/*                                                                           */
/* Only one chunk buffer (~14 MiB) is resident in RAM at a time.           */
/* Hyperslab selection is handled inside write_chunk_typed; boundary        */
/* tiles are automatically clamped to the actual tensor extent.             */
/* ----------------------------------------------------------------------- */

static int generate_file(const char *fname, const char *dset_name,
                         double _Complex fill)
{
    hsize_t shape[GEN_RANK], chunk_dims[GEN_RANK];
    for (int d = 0; d < GEN_RANK; d++) {
        shape[d]      = GLOBAL_DIM;
        chunk_dims[d] = CHUNK_SIDE;
    }

    /* Nominal chunk buffer: CHUNK_SIDE^4 × 16 B. */
    size_t elems_per_chunk = 1;
    for (int d = 0; d < GEN_RANK; d++) elems_per_chunk *= (size_t)CHUNK_SIDE;
    size_t bytes_per_chunk = elems_per_chunk * sizeof(double _Complex);

    size_t n_tiles[GEN_RANK], total_tiles = 1;
    for (int d = 0; d < GEN_RANK; d++) {
        n_tiles[d]   = TILES_PER_DIM;
        total_tiles *= n_tiles[d];
    }

    double gb_data  = (double)total_tiles * (double)bytes_per_chunk / 1e9;
    double gib_data = (double)total_tiles * (double)bytes_per_chunk
                      / (1024.0 * 1024.0 * 1024.0);

    printf("Generating %s\n", fname);
    printf("  shape  : (%d^%d)     dtype : COMPLEX128\n",
           GLOBAL_DIM, GEN_RANK);
    printf("  chunk  : (%d^%d) = %zu MiB\n",
           CHUNK_SIDE, GEN_RANK, bytes_per_chunk / (1024 * 1024));
    printf("  tiles  : %zu^%d = %zu total\n",
           (size_t)TILES_PER_DIM, GEN_RANK, total_tiles);
    printf("  fringe : %d elements in last tile of each dim\n", FRINGE_EXTENT);
    printf("  total  : %.2f GB  (%.2f GiB)\n", gb_data, gib_data);

    /* ------------------------------------------------------------------
     * Create the dataset.
     * create_chunked_dataset_einsum uses H5D_ALLOC_TIME_INCR so only
     * written chunks consume disk space (block-sparse layout).
     * ------------------------------------------------------------------ */
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

    /* ------------------------------------------------------------------
     * Single chunk buffer, pre-filled with the constant fill value.
     * write_chunk_typed passes the full nominal chunk_dims to HDF5's
     * hyperslab machinery; for boundary tiles it automatically trims the
     * selection to the dataset extent, so the same buffer is reused for
     * every tile — full and fringe alike.
     * ------------------------------------------------------------------ */
    double _Complex *buf = (double _Complex *)malloc(bytes_per_chunk);
    if (!buf) {
        fprintf(stderr, "gen: malloc failed (%zu MiB)\n",
                bytes_per_chunk / (1024 * 1024));
        H5Tclose(h5ctype); H5Dclose(dset); H5Fclose(fid);
        return -1;
    }
    for (size_t i = 0; i < elems_per_chunk; i++) buf[i] = fill;

    /* Odometer over the 4-D tile grid. */
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

        /* Progress bar: every 128 tiles or on the final tile. */
        if (written % 128 == 0 || written == total_tiles) {
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            double elapsed = elapsed_s(&t0, &tnow);
            double pct = 100.0 * (double)written / (double)total_tiles;
            double gib = (double)(written * bytes_per_chunk)
                         / (1024.0 * 1024.0 * 1024.0);
            double bw  = (elapsed > 0.0) ? gib / elapsed : 0.0;
            printf("\r  %5.1f%%  %4zu/%zu tiles  %6.1f GiB  "
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
        double elapsed = elapsed_s(&t0, &tf);
        printf("  done  %.2f GiB  %.1f s  %.2f GiB/s\n",
               gib_data, elapsed,
               (elapsed > 0.0) ? gib_data / elapsed : 0.0);
    }
    return ret;
}

/* ----------------------------------------------------------------------- */
/* main                                                                      */
/* ----------------------------------------------------------------------- */

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* Header                                                              */
    /* ------------------------------------------------------------------ */
    printf("=================================================================\n");
    printf("  Odd-Stride 40 GB Boundary Stress Test\n");
    printf("=================================================================\n");
    printf("  Expression  : ijab,akbl->klji\n");
    printf("  Dtype       : COMPLEX128\n");
    printf("  Global dim  : %d per index\n", GLOBAL_DIM);
    printf("  Chunk side  : %d per index\n", CHUNK_SIDE);
    printf("  Fringe      : %d mod %d = %d  (boundary tile extent)\n",
           GLOBAL_DIM, CHUNK_SIDE, FRINGE_EXTENT);
    printf("  Tile grid   : %d per dim  (%d full + 1 fringe)\n",
           TILES_PER_DIM, TILES_PER_DIM - 1);
    printf("\n");
    printf("  BLAS lda/ldb/ldc stress cases:\n");
    printf("    Full  tile  %d×%d  lda=%d  ldb=%d  ldc=%d\n",
           CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE);
    printf("    i-fringe    %d×%d  lda=%d  M=%d\n",
           FRINGE_EXTENT, CHUNK_SIDE, CHUNK_SIDE, FRINGE_EXTENT);
    printf("    j-fringe    %d×%d  lda=%d  N=%d\n",
           CHUNK_SIDE, FRINGE_EXTENT, FRINGE_EXTENT, FRINGE_EXTENT);
    printf("    Corner      %d×%d  lda=%d  M=%d  N=%d\n",
           FRINGE_EXTENT, FRINGE_EXTENT,
           FRINGE_EXTENT, FRINGE_EXTENT, FRINGE_EXTENT);
    printf("\n");

    /* ------------------------------------------------------------------ */
    /* Size arithmetic                                                     */
    /* ------------------------------------------------------------------ */
    double elems_per_tensor = 1.0;
    for (int i = 0; i < 4; i++) elems_per_tensor *= (double)GLOBAL_DIM;
    double gb_per_tensor  = elems_per_tensor * 16.0 / 1e9;
    double gib_per_tensor = elems_per_tensor * 16.0 / (1024.0*1024.0*1024.0);

    /*
     * FLOPs: 8 per complex multiply-accumulate.
     * Free indices i,j,k,l each range over GLOBAL_DIM.
     * Contracted indices a,b each range over GLOBAL_DIM.
     * Total MACs = GLOBAL_DIM^6; total FLOPs = 8 × GLOBAL_DIM^6.
     */
    double total_flops = 8.0;
    for (int i = 0; i < 6; i++) total_flops *= (double)GLOBAL_DIM;

    printf("  Tensor size : %.2f GB  (%.2f GiB) each\n",
           gb_per_tensor, gib_per_tensor);
    printf("  Total FLOPs : %.4e\n\n", total_flops);

    /* ------------------------------------------------------------------ */
    /* RAM and pool feasibility check                                      */
    /* ------------------------------------------------------------------ */
    size_t ram = query_physical_ram();

    size_t chunk_elems = 1;
    for (int d = 0; d < 4; d++) chunk_elems *= (size_t)CHUNK_SIDE;
    size_t bytes_page_nom = chunk_elems * sizeof(double _Complex);
    /* Pool pages are aligned to 16 KB by the engine; show the aligned size. */
    const size_t nvme_page = 16384UL;
    size_t bytes_page_aln  = (bytes_page_nom + nvme_page - 1) & ~(nvme_page - 1);
    size_t pool_bytes = (size_t)((double)ram * 0.8);
    size_t n_pages    = pool_bytes / bytes_page_aln;

    printf("Memory:\n");
    printf("  Physical RAM      : %.2f GB\n", (double)ram / 1e9);
    printf("  Pool budget (80%%) : %.2f GB\n", (double)pool_bytes / 1e9);
    printf("  Chunk (nominal)   : %zu B  (%d^4 × 16 B)\n",
           bytes_page_nom, CHUNK_SIDE);
    printf("  Page (NVMe-aln)   : %zu B  (rounded to %zu KB)\n",
           bytes_page_aln, nvme_page / 1024);
    printf("  Pages available   : %zu  (engine needs >= 8)\n\n", n_pages);

    if (n_pages < 8) {
        fprintf(stderr,
                "ERROR: insufficient RAM for 8 pages "
                "(%zu B needed, %zu B available)\n",
                8 * bytes_page_aln, pool_bytes);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Generate input files if absent                                      */
    /* ------------------------------------------------------------------ */
    {
        int need_a = 1, need_b = 1;
        FILE *fa = fopen(FILE_A, "rb");
        FILE *fb = fopen(FILE_B, "rb");
        if (fa) { fclose(fa); need_a = 0; }
        if (fb) { fclose(fb); need_b = 0; }

        if (need_a || need_b) {
            printf("--- Generating input tensors ---\n");
            double _Complex fill = CMPLX(FILL_REAL, FILL_IMAG);

            if (need_a) {
                if (generate_file(FILE_A, DSET, fill) < 0) return 1;
                printf("\n");
            } else {
                printf("  %s  already exists, skipping.\n\n", FILE_A);
            }

            if (need_b) {
                if (generate_file(FILE_B, DSET, fill) < 0) return 1;
                printf("\n");
            } else {
                printf("  %s  already exists, skipping.\n\n", FILE_B);
            }
        } else {
            printf("Input files found — skipping generation.\n\n");
        }
    }

    /* ------------------------------------------------------------------ */
    /* Contraction                                                          */
    /* ------------------------------------------------------------------ */
    printf("--- Running contraction ---\n");
    printf("  A: %s\n  B: %s\n  C: %s\n\n", FILE_A, FILE_B, FILE_C);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int ret = run_contraction_einsum(
        "ijab,akbl->klji",
        FILE_A, DSET,
        FILE_B, DSET,
        FILE_C, DSET);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = elapsed_s(&t0, &t1);

    if (ret != 0) {
        fprintf(stderr, "Contraction failed.\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Performance report                                                   */
    /* ------------------------------------------------------------------ */
    double gflops   = (total_flops / elapsed) / 1e9;
    double read_gb  = 2.0 * gb_per_tensor;   /* A + B */
    double write_gb = gb_per_tensor;           /* C     */
    double rd_bw    = read_gb  / elapsed;
    double wr_bw    = write_gb / elapsed;

    printf("\n=================================================================\n");
    printf("  Performance\n");
    printf("=================================================================\n");
    printf("  Elapsed time         : %.2f s\n",        elapsed);
    printf("  Sustained GFLOPS     : %.2f\n",          gflops);
    printf("  NVMe read  bandwidth : %.2f GB/s  "
           "(%.2f GB / %.2f s)\n",
           rd_bw, read_gb, elapsed);
    printf("  NVMe write bandwidth : %.2f GB/s  "
           "(%.2f GB / %.2f s)\n",
           wr_bw, write_gb, elapsed);
    printf("=================================================================\n");

    return 0;
}
