#include "engine.h"
#include "memory.h"
#include "registry.h"
#include "tensor_store.h"
#include <hdf5.h>
#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <dispatch/dispatch.h>
#define HAS_GCD 1
#endif

#ifdef USE_ACCELERATE
#  include <Accelerate/Accelerate.h>
#elif defined(USE_MKL)
#  include <mkl_cblas.h>
#elif defined(HAVE_CBLAS)
#  include <cblas.h>
#endif

/* Generic BLAS dispatch macros.
 * TENSOR_DGEMM / TENSOR_ZGEMM expand to the correct function regardless of
 * which backend (Accelerate, OpenBLAS, future MKL) was compiled in.
 * When no BLAS is present they are left undefined and the fallback scalar
 * kernels in the #ifndef HAVE_CBLAS blocks are used instead.
 */
#if defined(USE_ACCELERATE) || defined(HAVE_CBLAS)
#  define TENSOR_DGEMM(order,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc) \
          cblas_dgemm(order,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
#  define TENSOR_ZGEMM(order,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc) \
          cblas_zgemm(order,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
#endif

#include "einsum.h"
#include "odometer.h"
#include "write_queue.h"
#include "metal_backend.h"
#include <complex.h>

/* ----------------------------------------------------------------------- */
/* Feature A — Dynamic RAM query                                            */
/* ----------------------------------------------------------------------- */

size_t query_physical_ram(void)
{
    /* macOS: hw.memsize gives the exact installed RAM in one syscall. */
#if defined(__APPLE__)
    {
        uint64_t mem = 0;
        size_t   len = sizeof(mem);
        if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0 && mem > 0)
            return (size_t)mem;
    }
#endif

    /* POSIX (Linux, BSD, ...): product of page count × page size. */
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    {
        long pages = sysconf(_SC_PHYS_PAGES);
        long psz   = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && psz > 0)
            return (size_t)pages * (size_t)psz;
    }
#endif

    /* Safe fallback – keeps the engine functional on unknown platforms. */
    fprintf(stderr, "query_physical_ram: sysconf unavailable, using 512 MB\n");
    return 512UL * 1024UL * 1024UL;
}

/* NVMe hardware page size on Apple Silicon (16 KB).
 * Pool pages aligned to this boundary avoid read-amplification. */
#define NVME_PAGE_BYTES 16384UL

/* HDF5 raw-data chunk cache per file handle.
 * 1 GB keeps the HDF5 B-tree metadata and hot chunks resident in RAM
 * when the pool has space to spare. */
#define HDF5_CHUNK_CACHE_BYTES (1UL << 30)

/* ----------------------------------------------------------------------- */
/* engine_fopen_cached — open an HDF5 file with a large chunk cache        */
/* ----------------------------------------------------------------------- */
static hid_t engine_fopen_cached(const char *path, unsigned flags,
                                  size_t rdcc_nbytes)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0) return H5Fopen(path, flags, H5P_DEFAULT);

    /* Large prime slot count: keeps the hash table collision rate low.
     * rdcc_w0 = 0.75 — prefer evicting chunks that are not repeated. */
    H5Pset_cache(fapl, 0, 100003, rdcc_nbytes, 0.75);

    hid_t fid = H5Fopen(path, flags, fapl);
    H5Pclose(fapl);
    return fid;
}

/* ----------------------------------------------------------------------- */
/* Feature B — Tile multiply kernel (fallback when BLAS is absent)         */
/*                                                                           */
/* Computes C[0:M, 0:N] += A[0:M, 0:K] × B[0:K, 0:N].                    */
/* Buffers are row-major with nominal strides lda / ldb / ldc.             */
/* Passing the actual (clamped) dims avoids iterating over zero padding,   */
/* which is especially important for boundary tiles.                        */
/* ----------------------------------------------------------------------- */
#ifndef HAVE_CBLAS
static void compute_tile(const double * restrict A, int lda,
                         const double * restrict B, int ldb,
                         double       * restrict C, int ldc,
                         int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            double a_val = A[m * lda + k];
            if (a_val == 0.0) continue;          /* skip sparse columns    */
            for (int n = 0; n < N; n++)
                C[m * ldc + n] += a_val * B[k * ldb + n];
        }
    }
}
#endif /* !HAVE_CBLAS */

#ifndef HAVE_CBLAS
static void compute_tile_z(const double _Complex * restrict A, int lda,
                            const double _Complex * restrict B, int ldb,
                            double _Complex       * restrict C, int ldc,
                            int M, int N, int K)
{
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            double _Complex a = A[m * lda + k];
            for (int n = 0; n < N; n++)
                C[m * ldc + n] += a * B[k * ldb + n];
        }
}
#endif /* !HAVE_CBLAS */

/* ----------------------------------------------------------------------- */
/* Feature C — Async I/O double-buffer                                      */
/*                                                                           */
/* Two ping-pong slots allow the I/O thread to prefetch the next tile pair  */
/* while the compute thread (main) multiplies the current one.              */
/*                                                                           */
/* Thread safety contract:                                                   */
/*   • All HDF5 calls are confined to io_thread_func.                       */
/*   • All pool_acquire / pool_release calls are made under IOShared.mu     */
/*     so pool_acquire/pool_release do not need their own lock.             */
/*   • The compute thread reads slot fields under the lock and only calls   */
/*     cblas_dgemm / compute_tile after releasing it, using locally copied  */
/*     pointers that remain valid until pool_release (also under the lock). */
/* ----------------------------------------------------------------------- */

#define SLOT_FREE  0   /* I/O thread may fill this slot                     */
#define SLOT_READY 1   /* Data loaded; compute thread may consume           */
#define SLOT_EOF   2   /* No more tiles; compute thread should exit loop    */

typedef struct {
    /* Ping-pong buffer slots (indices 0 and 1) */
    double  *buf_A[2], *buf_B[2];
    size_t   id_A[2],   id_B[2];
    int      actual_M[2];    /* clamped row count of A tile (cblas_dgemm M) */
    int      actual_K[2];    /* clamped col count of A tile (cblas_dgemm K) */
    int      actual_N[2];    /* clamped col count of B tile (cblas_dgemm N) */
    int      state[2];       /* SLOT_FREE / SLOT_READY / SLOT_EOF           */

    pthread_mutex_t mu;
    pthread_cond_t  cond;

    /* Parameters owned by the I/O thread. */
    hid_t           dset_A, dset_B, dset_C;
    TensorRegistry *reg_A, *reg_B, *reg_C;
    BufferPool     *pool;
    hsize_t         i, j, K_tiles;   /* current output tile coords          */

    /* C-tile flush handshake (compute → I/O → compute). */
    double *buf_C;
    size_t  id_C;
    int     flush_req;
    int     flush_done;

    int     io_err;           /* set to 1 by I/O thread on any failure       */
} IOShared;

static void *io_thread_func(void *arg)
{
    IOShared *s = (IOShared *)arg;
    int write_idx = 0;

    for (hsize_t k = 0; k < s->K_tiles; k++) {
        hsize_t ca[2] = {s->i, k};
        hsize_t cb[2] = {k, s->j};
        TileMetadata *mA = registry_get_tile(s->reg_A, ca);
        TileMetadata *mB = registry_get_tile(s->reg_B, cb);

        /* Block-sparse: skip tile pairs where either operand is absent. */
        if (!mA || mA->status != TILE_STATUS_ON_DISK) continue;
        if (!mB || mB->status != TILE_STATUS_ON_DISK) continue;

        /*
         * Clamp nominal chunk_dims to the dataset boundary so cblas_dgemm
         * receives the true element count for boundary tiles.
         */
        hsize_t nomM = s->reg_A->chunk_dims[0];
        hsize_t nomK = s->reg_A->chunk_dims[1];
        hsize_t nomN = s->reg_B->chunk_dims[1];
        int aM = (int)((mA->phys_offset[0] + nomM > s->reg_A->global_dims[0])
                       ? s->reg_A->global_dims[0] - mA->phys_offset[0] : nomM);
        int aK = (int)((mA->phys_offset[1] + nomK > s->reg_A->global_dims[1])
                       ? s->reg_A->global_dims[1] - mA->phys_offset[1] : nomK);
        int aN = (int)((mB->phys_offset[1] + nomN > s->reg_B->global_dims[1])
                       ? s->reg_B->global_dims[1] - mB->phys_offset[1] : nomN);

        /*
         * Wait for the write slot to become FREE, then acquire pool pages
         * atomically under the same lock.  The compute thread also calls
         * pool_release under this mutex, so pool ops are always serialised.
         */
        pthread_mutex_lock(&s->mu);
        while (s->state[write_idx] != SLOT_FREE)
            pthread_cond_wait(&s->cond, &s->mu);

        size_t  id_A = SIZE_MAX, id_B = SIZE_MAX;
        double *bA   = pool_acquire(s->pool, &id_A);
        double *bB   = pool_acquire(s->pool, &id_B);
        pthread_mutex_unlock(&s->mu);

        if (!bA || !bB) {
            /* Return whatever was acquired before flagging the error. */
            pthread_mutex_lock(&s->mu);
            if (bA) pool_release(s->pool, id_A);
            if (bB) pool_release(s->pool, id_B);
            pthread_mutex_unlock(&s->mu);
            fprintf(stderr, "io_thread: pool exhausted at k=%llu\n",
                    (unsigned long long)k);
            s->io_err = 1;
            break;
        }

        /* Heavy disk I/O outside the mutex — this is the overlapped region. */
        if (read_chunk_fast(s->dset_A, mA->phys_offset, bA,
                            2, s->reg_A->chunk_dims) < 0 ||
            read_chunk_fast(s->dset_B, mB->phys_offset, bB,
                            2, s->reg_B->chunk_dims) < 0) {
            pthread_mutex_lock(&s->mu);
            pool_release(s->pool, id_B);
            pool_release(s->pool, id_A);
            pthread_mutex_unlock(&s->mu);
            fprintf(stderr, "io_thread: read_chunk_fast failed at k=%llu\n",
                    (unsigned long long)k);
            s->io_err = 1;
            break;
        }

        /* Publish the loaded slot to the compute thread. */
        pthread_mutex_lock(&s->mu);
        s->buf_A[write_idx]    = bA;   s->id_A[write_idx]     = id_A;
        s->buf_B[write_idx]    = bB;   s->id_B[write_idx]     = id_B;
        s->actual_M[write_idx] = aM;
        s->actual_K[write_idx] = aK;
        s->actual_N[write_idx] = aN;
        s->state[write_idx]    = SLOT_READY;
        pthread_cond_broadcast(&s->cond);
        pthread_mutex_unlock(&s->mu);

        write_idx ^= 1;
    }

    /*
     * Signal EOF.  write_idx == read_idx at this point (both advance in
     * lock-step), so the compute thread will see SLOT_EOF when it next
     * waits on this slot.
     */
    pthread_mutex_lock(&s->mu);
    while (s->state[write_idx] != SLOT_FREE)
        pthread_cond_wait(&s->cond, &s->mu);
    s->state[write_idx] = SLOT_EOF;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mu);

    /* Wait for the compute thread to finish accumulating C, then flush. */
    pthread_mutex_lock(&s->mu);
    while (!s->flush_req)
        pthread_cond_wait(&s->cond, &s->mu);
    pthread_mutex_unlock(&s->mu);

    hsize_t cc[2] = {s->i, s->j};
    TileMetadata *mC = registry_get_tile(s->reg_C, cc);
    if (!mC ||
        write_chunk_fast(s->dset_C, mC->phys_offset, s->buf_C,
                         2, s->reg_C->chunk_dims) < 0) {
        fprintf(stderr, "io_thread: write_chunk_fast failed for C(%llu,%llu)\n",
                (unsigned long long)s->i, (unsigned long long)s->j);
        s->io_err = 1;
    }

    pthread_mutex_lock(&s->mu);
    s->flush_done = 1;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mu);

    return NULL;
}

/* ----------------------------------------------------------------------- */
/* Cleanup helper — called from every early-exit path                       */
/* ----------------------------------------------------------------------- */
static void engine_cleanup(BufferPool    *pool,
                           TensorRegistry *reg_A,
                           TensorRegistry *reg_B,
                           TensorRegistry *reg_C,
                           hid_t dset_A, hid_t dset_B, hid_t dset_C,
                           hid_t fa,     hid_t fb,     hid_t fc)
{
    if (pool)  pool_destroy(pool);
    if (reg_A) registry_destroy(reg_A);
    if (reg_B) registry_destroy(reg_B);
    if (reg_C) registry_destroy(reg_C);
    if (dset_A >= 0) H5Dclose(dset_A);
    if (dset_B >= 0) H5Dclose(dset_B);
    if (dset_C >= 0) H5Dclose(dset_C);
    if (fa >= 0) H5Fclose(fa);
    if (fb >= 0) H5Fclose(fb);
    if (fc >= 0) H5Fclose(fc);
}

/* ----------------------------------------------------------------------- */
/* run_contraction                                                           */
/* ----------------------------------------------------------------------- */

int run_contraction(const char *file_A, const char *name_A,
                    const char *file_B, const char *name_B,
                    const char *file_C, const char *name_C)
{
    printf("\n=== Tensor Contraction Engine ===\n");

    /* ------------------------------------------------------------------ */
    /* 1. Open input files and datasets (chunk cache disabled)            */
    /* ------------------------------------------------------------------ */
    hid_t fa = H5Fopen(file_A, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t fb = H5Fopen(file_B, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fa < 0 || fb < 0) {
        fprintf(stderr, "run_contraction: cannot open '%s' or '%s'\n",
                file_A, file_B);
        if (fa >= 0) H5Fclose(fa);
        if (fb >= 0) H5Fclose(fb);
        return -1;
    }

    hid_t dset_A = dset_open_no_cache(fa, name_A);
    hid_t dset_B = dset_open_no_cache(fb, name_B);
    if (dset_A < 0 || dset_B < 0) {
        fprintf(stderr, "run_contraction: cannot open dataset '%s' or '%s'\n",
                name_A, name_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    /* ------------------------------------------------------------------ */
    /* 2. Read rank and shape from the actual file dataspaces             */
    /* ------------------------------------------------------------------ */
    hid_t fsp_A = H5Dget_space(dset_A);
    hid_t fsp_B = H5Dget_space(dset_B);
    if (fsp_A < 0 || fsp_B < 0) {
        if (fsp_A >= 0) H5Sclose(fsp_A);
        if (fsp_B >= 0) H5Sclose(fsp_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    int rank_A = H5Sget_simple_extent_ndims(fsp_A);
    int rank_B = H5Sget_simple_extent_ndims(fsp_B);
    if (rank_A != 2 || rank_B != 2) {
        fprintf(stderr,
                "run_contraction: requires rank-2 inputs (got %d and %d)\n",
                rank_A, rank_B);
        H5Sclose(fsp_A); H5Sclose(fsp_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    hsize_t global_A[2], global_B[2];
    H5Sget_simple_extent_dims(fsp_A, global_A, NULL);
    H5Sget_simple_extent_dims(fsp_B, global_B, NULL);
    H5Sclose(fsp_A);
    H5Sclose(fsp_B);

    if (global_A[1] != global_B[0]) {
        fprintf(stderr,
                "run_contraction: dimension mismatch – "
                "A col=%llu but B row=%llu\n",
                (unsigned long long)global_A[1],
                (unsigned long long)global_B[0]);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    hsize_t dim_M = global_A[0];
    hsize_t dim_K = global_A[1];
    hsize_t dim_N = global_B[1];
    printf("A: (%llu\xc3\x97%llu)  B: (%llu\xc3\x97%llu)  "
           "C: (%llu\xc3\x97%llu)\n",
           (unsigned long long)dim_M, (unsigned long long)dim_K,
           (unsigned long long)dim_K, (unsigned long long)dim_N,
           (unsigned long long)dim_M, (unsigned long long)dim_N);

    /* ------------------------------------------------------------------ */
    /* 3. Build registries from the files' own chunk metadata             */
    /* ------------------------------------------------------------------ */
    TensorRegistry *reg_A = registry_create_from_dset(dset_A);
    TensorRegistry *reg_B = registry_create_from_dset(dset_B);
    if (!reg_A || !reg_B) {
        fprintf(stderr, "run_contraction: registry_create_from_dset failed\n");
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    printf("Scanning input tiles...\n");
    long tiles_A = registry_scan_file(dset_A, reg_A);
    long tiles_B = registry_scan_file(dset_B, reg_B);
    printf("  A: %ld on-disk tiles   B: %ld on-disk tiles\n",
           tiles_A, tiles_B);

    /* ------------------------------------------------------------------ */
    /* 4. Create output file C                                            */
    /* ------------------------------------------------------------------ */
    /* Use the same RAM-scaled chunk size so all three tensors have        */
    /* consistent tile granularity.                                        */
    size_t ram          = query_physical_ram();
    size_t chunk_bytes  = ram / 1000;                    /* ~1 MB per GB  */
    if (chunk_bytes < 2UL * 1024 * 1024)
        chunk_bytes = 2UL * 1024 * 1024;

    hsize_t global_C[2] = {dim_M, dim_N};
    if (create_chunked_dataset(file_C, name_C, 2, global_C,
                               chunk_bytes) < 0) {
        fprintf(stderr, "run_contraction: create_chunked_dataset failed "
                        "for '%s'\n", file_C);
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    hid_t fc     = H5Fopen(file_C, H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t dset_C = (fc >= 0) ? dset_open_no_cache(fc, name_C) : -1;
    if (fc < 0 || dset_C < 0) {
        fprintf(stderr, "run_contraction: cannot open output '%s'\n", file_C);
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    TensorRegistry *reg_C = registry_create_from_dset(dset_C);
    if (!reg_C) {
        fprintf(stderr, "run_contraction: registry_create_from_dset(C) "
                        "failed\n");
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    /* ------------------------------------------------------------------ */
    /* 5. Initialise memory pool at 80% of physical RAM                   */
    /*                                                                     */
    /* Minimum 5 pages: 2 × (A + B) double-buffer slots + 1 C tile.      */
    /* ------------------------------------------------------------------ */
    size_t elems_A = (size_t)reg_A->chunk_dims[0] * (size_t)reg_A->chunk_dims[1];
    size_t elems_B = (size_t)reg_B->chunk_dims[0] * (size_t)reg_B->chunk_dims[1];
    size_t elems_C = (size_t)reg_C->chunk_dims[0] * (size_t)reg_C->chunk_dims[1];
    size_t elems_per_page = elems_A;
    if (elems_B > elems_per_page) elems_per_page = elems_B;
    if (elems_C > elems_per_page) elems_per_page = elems_C;

    size_t pool_bytes = (size_t)((double)ram * 0.8);
    size_t num_pages  = pool_bytes / (elems_per_page * sizeof(double));
    if (num_pages < 5) {
        fprintf(stderr,
                "run_contraction: RAM too small for 5 pages "
                "(need %zu bytes)\n",
                5 * elems_per_page * sizeof(double));
        engine_cleanup(NULL, reg_A, reg_B, reg_C,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    BufferPool *pool = pool_create(num_pages, elems_per_page * sizeof(double));
    if (!pool) {
        fprintf(stderr, "run_contraction: pool_create failed\n");
        engine_cleanup(NULL, reg_A, reg_B, reg_C,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    printf("RAM: %.1f GB physical  "
           "Pool: %zu pages \xc3\x97 %zu elems = %.1f GB\n",
           (double)ram / (1024.0 * 1024.0 * 1024.0),
           num_pages, elems_per_page,
           (double)(num_pages * elems_per_page * sizeof(double))
               / (1024.0 * 1024.0 * 1024.0));

#if defined(USE_MKL)
    printf("Kernel: cblas_dgemm (Intel MKL)\n");
#elif defined(HAVE_CBLAS)
    printf("Kernel: cblas_dgemm (OpenBLAS)\n");
#else
    printf("Kernel: fallback (m\xe2\x86\x92k\xe2\x86\x92n loop)\n");
#endif

    /* ------------------------------------------------------------------ */
    /* 6. SUMMA execution loop with async I/O double-buffer               */
    /* ------------------------------------------------------------------ */
    hsize_t I_tiles = reg_C->grid_dims[0];
    hsize_t J_tiles = reg_C->grid_dims[1];
    hsize_t K_tiles = reg_A->grid_dims[1];

    printf("Grid: [%llu \xc3\x97 %llu]  K-tiles: %llu\n",
           (unsigned long long)I_tiles, (unsigned long long)J_tiles,
           (unsigned long long)K_tiles);

    int ret = 0;

    for (hsize_t i = 0; i < I_tiles && ret == 0; i++) {
        for (hsize_t j = 0; j < J_tiles && ret == 0; j++) {

            /* -- Acquire the C accumulator (zeroed) -- */
            size_t  id_C  = SIZE_MAX;
            double *buf_C = pool_acquire(pool, &id_C);
            if (!buf_C) {
                fprintf(stderr,
                        "run_contraction: pool exhausted acquiring "
                        "C(%llu,%llu)\n",
                        (unsigned long long)i, (unsigned long long)j);
                ret = -1;
                break;
            }
            memset(buf_C, 0, elems_per_page * sizeof(double));

            /* -- Initialise shared I/O state -- */
            IOShared s;
            memset(&s, 0, sizeof(s));   /* state[0,1]=SLOT_FREE; flags=0 */
            pthread_mutex_init(&s.mu, NULL);
            pthread_cond_init(&s.cond, NULL);

            s.dset_A = dset_A;  s.dset_B = dset_B;  s.dset_C = dset_C;
            s.reg_A  = reg_A;   s.reg_B  = reg_B;   s.reg_C  = reg_C;
            s.pool   = pool;
            s.i      = i;       s.j      = j;       s.K_tiles = K_tiles;
            s.buf_C  = buf_C;   s.id_C   = id_C;

            /* -- Spawn the I/O thread -- */
            pthread_t io_tid;
            if (pthread_create(&io_tid, NULL, io_thread_func, &s) != 0) {
                fprintf(stderr,
                        "run_contraction: pthread_create failed at "
                        "(%llu,%llu)\n",
                        (unsigned long long)i, (unsigned long long)j);
                pool_release(pool, id_C);
                pthread_mutex_destroy(&s.mu);
                pthread_cond_destroy(&s.cond);
                ret = -1;
                break;
            }

            /* -- Compute loop: drain READY slots, never touch HDF5 -- */
            int read_idx = 0;
            for (;;) {
                pthread_mutex_lock(&s.mu);
                /* Spin until the slot is no longer FREE. */
                while (s.state[read_idx] == SLOT_FREE)
                    pthread_cond_wait(&s.cond, &s.mu);

                int     cur_state = s.state[read_idx];
                double *bA        = s.buf_A[read_idx];
                double *bB        = s.buf_B[read_idx];
                int     aM        = s.actual_M[read_idx];
                int     aK        = s.actual_K[read_idx];
                int     aN        = s.actual_N[read_idx];
                pthread_mutex_unlock(&s.mu);

                if (cur_state == SLOT_EOF) break;

                /*
                 * C(i,j) += A(i,k) * B(k,j)
                 *
                 * aM / aK / aN are the true (clamped) extents for this
                 * tile pair.  The leading dimensions are the nominal
                 * chunk_dims, which are the in-memory row strides laid
                 * down by read_chunk_fast.
                 */
#ifdef HAVE_CBLAS
                cblas_dgemm(CblasRowMajor,
                            CblasNoTrans, CblasNoTrans,
                            aM, aN, aK,
                            1.0,
                            bA, (int)reg_A->chunk_dims[1],
                            bB, (int)reg_B->chunk_dims[1],
                            1.0,
                            buf_C, (int)reg_C->chunk_dims[1]);
#else
                compute_tile(bA, (int)reg_A->chunk_dims[1],
                             bB, (int)reg_B->chunk_dims[1],
                             buf_C, (int)reg_C->chunk_dims[1],
                             aM, aN, aK);
#endif

                /* Return A/B pages under mutex (pool is not thread-safe). */
                pthread_mutex_lock(&s.mu);
                pool_release(pool, s.id_A[read_idx]);
                pool_release(pool, s.id_B[read_idx]);
                s.state[read_idx] = SLOT_FREE;
                pthread_cond_broadcast(&s.cond);
                pthread_mutex_unlock(&s.mu);

                read_idx ^= 1;
            }

            /* -- Signal I/O thread to flush C, then wait -- */
            pthread_mutex_lock(&s.mu);
            s.flush_req = 1;
            pthread_cond_broadcast(&s.cond);
            while (!s.flush_done)
                pthread_cond_wait(&s.cond, &s.mu);
            pthread_mutex_unlock(&s.mu);

            pthread_join(io_tid, NULL);
            pthread_mutex_destroy(&s.mu);
            pthread_cond_destroy(&s.cond);

            if (s.io_err) {
                fprintf(stderr,
                        "run_contraction: I/O error at C(%llu,%llu)\n",
                        (unsigned long long)i, (unsigned long long)j);
                ret = -1;
            }

            pool_release(pool, id_C);

            if (ret == 0) { printf("."); fflush(stdout); }
        }
    }

    if (ret == 0) printf("\nContraction complete.\n");

    /* ------------------------------------------------------------------ */
    /* 7. Cleanup (always executed)                                        */
    /* ------------------------------------------------------------------ */
    engine_cleanup(pool, reg_A, reg_B, reg_C,
                   dset_A, dset_B, dset_C, fa, fb, fc);
    return ret;
}

/* ======================================================================= */
/* Rank-4 contraction: C(k,l,j,i) = sum_{a,b} A(i,j,a,b) * B(a,k,b,l)   */
/* ======================================================================= */

/* ----------------------------------------------------------------------- */
/* IOShared4D — shared state between compute thread and I/O thread for the  */
/* rank-4 double-buffer pipeline.                                            */
/*                                                                           */
/* The I/O thread loops over contracted (a_tile, b_tile) pairs and fills    */
/* ping-pong slots.  The compute thread permutes B, calls BLAS, and         */
/* scatter-accumulates into buf_C.  All pool operations are serialised under */
/* IOShared4D.mu to avoid concurrent pool_acquire / pool_release races.     */
/* ----------------------------------------------------------------------- */
typedef struct {
    double  *buf_A[2], *buf_B[2];
    size_t   id_A[2],   id_B[2];
    int      state[2];   /* SLOT_FREE / SLOT_READY / SLOT_EOF */

    pthread_mutex_t mu;
    pthread_cond_t  cond;

    hid_t           dset_A, dset_B, dset_C;
    TensorRegistry *reg_A, *reg_B, *reg_C;
    BufferPool     *pool;

    /* Output-tile indices in the C grid (k_tile, l_tile, j_tile, i_tile). */
    hsize_t  ki, li, ji, ii;
    hsize_t  A_tiles;   /* grid_dims[2] of A — contracted a-dim tile count */
    hsize_t  B_tiles;   /* grid_dims[3] of A — contracted b-dim tile count */

    double  *buf_C;     /* C accumulator (pre-zeroed, held by compute thread) */
    size_t   id_C;
    int      flush_req;
    int      flush_done;
    int      io_err;
} IOShared4D;

/* ----------------------------------------------------------------------- */
/* I/O thread for rank-4 contraction.  Prefetches A(ii,ji,a,b) and         */
/* B(a,ki,b,li) tile pairs into alternating double-buffer slots, then on    */
/* flush_req writes the finished C tile.  No HDF5 calls may occur anywhere  */
/* else.                                                                     */
/* ----------------------------------------------------------------------- */
static void *io_thread_func_4d(void *arg)
{
    IOShared4D *s = (IOShared4D *)arg;
    int write_idx = 0;

    for (hsize_t a = 0; a < s->A_tiles; a++) {
        for (hsize_t b = 0; b < s->B_tiles; b++) {

            /* Tile exists in A[ii, ji, a, b] and B[a, ki, b, li]? */
            hsize_t ca[4] = {s->ii, s->ji, a, b};
            hsize_t cb[4] = {a, s->ki, b, s->li};
            TileMetadata *mA = registry_get_tile(s->reg_A, ca);
            TileMetadata *mB = registry_get_tile(s->reg_B, cb);
            if (!mA || mA->status != TILE_STATUS_ON_DISK) continue;
            if (!mB || mB->status != TILE_STATUS_ON_DISK) continue;

            /* Wait for slot FREE, then acquire pool pages atomically. */
            pthread_mutex_lock(&s->mu);
            while (s->state[write_idx] != SLOT_FREE)
                pthread_cond_wait(&s->cond, &s->mu);

            size_t  id_A = SIZE_MAX, id_B = SIZE_MAX;
            double *bA   = pool_acquire(s->pool, &id_A);
            double *bB   = pool_acquire(s->pool, &id_B);
            pthread_mutex_unlock(&s->mu);

            if (!bA || !bB) {
                pthread_mutex_lock(&s->mu);
                if (bA) pool_release(s->pool, id_A);
                if (bB) pool_release(s->pool, id_B);
                pthread_mutex_unlock(&s->mu);
                fprintf(stderr,
                        "io_thread_4d: pool exhausted at (a=%llu,b=%llu)\n",
                        (unsigned long long)a, (unsigned long long)b);
                s->io_err = 1;
                goto eof;
            }

            /* Disk reads outside the mutex — the overlap window. */
            if (read_chunk_fast(s->dset_A, mA->phys_offset, bA,
                                4, s->reg_A->chunk_dims) < 0 ||
                read_chunk_fast(s->dset_B, mB->phys_offset, bB,
                                4, s->reg_B->chunk_dims) < 0) {
                pthread_mutex_lock(&s->mu);
                pool_release(s->pool, id_B);
                pool_release(s->pool, id_A);
                pthread_mutex_unlock(&s->mu);
                fprintf(stderr,
                        "io_thread_4d: read failed at (a=%llu,b=%llu)\n",
                        (unsigned long long)a, (unsigned long long)b);
                s->io_err = 1;
                goto eof;
            }

            /* Publish slot. */
            pthread_mutex_lock(&s->mu);
            s->buf_A[write_idx] = bA;  s->id_A[write_idx] = id_A;
            s->buf_B[write_idx] = bB;  s->id_B[write_idx] = id_B;
            s->state[write_idx] = SLOT_READY;
            pthread_cond_broadcast(&s->cond);
            pthread_mutex_unlock(&s->mu);

            write_idx ^= 1;
        }
    }

eof:
    /* Signal EOF on the next slot (write_idx == read_idx at this point). */
    pthread_mutex_lock(&s->mu);
    while (s->state[write_idx] != SLOT_FREE)
        pthread_cond_wait(&s->cond, &s->mu);
    s->state[write_idx] = SLOT_EOF;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mu);

    /* Wait for C flush request, then write. */
    pthread_mutex_lock(&s->mu);
    while (!s->flush_req)
        pthread_cond_wait(&s->cond, &s->mu);
    pthread_mutex_unlock(&s->mu);

    hsize_t cc[4] = {s->ki, s->li, s->ji, s->ii};
    TileMetadata *mC = registry_get_tile(s->reg_C, cc);
    if (!mC ||
        write_chunk_fast(s->dset_C, mC->phys_offset, s->buf_C,
                         4, s->reg_C->chunk_dims) < 0) {
        fprintf(stderr,
                "io_thread_4d: write failed for C(%llu,%llu,%llu,%llu)\n",
                (unsigned long long)s->ki, (unsigned long long)s->li,
                (unsigned long long)s->ji, (unsigned long long)s->ii);
        s->io_err = 1;
    }

    pthread_mutex_lock(&s->mu);
    s->flush_done = 1;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mu);

    return NULL;
}

/* ----------------------------------------------------------------------- */
/* run_contraction_4d                                                        */
/* ----------------------------------------------------------------------- */

int run_contraction_4d(const char *file_A, const char *name_A,
                       const char *file_B, const char *name_B,
                       const char *file_C, const char *name_C)
{
    printf("\n=== Rank-4 Tensor Contraction Engine ===\n");
    printf("C(k,l,j,i) = sum_{a,b} A(i,j,a,b) * B(a,k,b,l)\n");

    /* ------------------------------------------------------------------ */
    /* 1. Open A and B                                                     */
    /* ------------------------------------------------------------------ */
    hid_t fa = H5Fopen(file_A, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t fb = H5Fopen(file_B, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fa < 0 || fb < 0) {
        fprintf(stderr, "run_contraction_4d: cannot open '%s' or '%s'\n",
                file_A, file_B);
        if (fa >= 0) H5Fclose(fa);
        if (fb >= 0) H5Fclose(fb);
        return -1;
    }

    hid_t dset_A = dset_open_no_cache(fa, name_A);
    hid_t dset_B = dset_open_no_cache(fb, name_B);
    if (dset_A < 0 || dset_B < 0) {
        fprintf(stderr,
                "run_contraction_4d: cannot open dataset '%s' or '%s'\n",
                name_A, name_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    /* ------------------------------------------------------------------ */
    /* 2. Read rank and shape                                              */
    /* ------------------------------------------------------------------ */
    hid_t fsp_A = H5Dget_space(dset_A);
    hid_t fsp_B = H5Dget_space(dset_B);
    if (fsp_A < 0 || fsp_B < 0) {
        if (fsp_A >= 0) H5Sclose(fsp_A);
        if (fsp_B >= 0) H5Sclose(fsp_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    int rank_A = H5Sget_simple_extent_ndims(fsp_A);
    int rank_B = H5Sget_simple_extent_ndims(fsp_B);
    if (rank_A != 4 || rank_B != 4) {
        fprintf(stderr,
                "run_contraction_4d: requires rank-4 inputs "
                "(got %d and %d)\n", rank_A, rank_B);
        H5Sclose(fsp_A); H5Sclose(fsp_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    hsize_t global_A[4], global_B[4];
    H5Sget_simple_extent_dims(fsp_A, global_A, NULL);
    H5Sget_simple_extent_dims(fsp_B, global_B, NULL);
    H5Sclose(fsp_A);
    H5Sclose(fsp_B);

    /* A is (i,j,a,b)  →  global_A = {i, j, a, b}
       B is (a,k,b,l)  →  global_B = {a, k, b, l}
       Check contracted dimensions match.                                  */
    if (global_A[2] != global_B[0] || global_A[3] != global_B[2]) {
        fprintf(stderr,
                "run_contraction_4d: contracted dim mismatch — "
                "A.a=%llu B.a=%llu  A.b=%llu B.b=%llu\n",
                (unsigned long long)global_A[2],
                (unsigned long long)global_B[0],
                (unsigned long long)global_A[3],
                (unsigned long long)global_B[2]);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    hsize_t dim_i = global_A[0], dim_j = global_A[1];
    hsize_t dim_a = global_A[2], dim_b = global_A[3];
    hsize_t dim_k = global_B[1], dim_l = global_B[3];
    printf("i=%llu  j=%llu  a=%llu  b=%llu  k=%llu  l=%llu\n",
           (unsigned long long)dim_i, (unsigned long long)dim_j,
           (unsigned long long)dim_a, (unsigned long long)dim_b,
           (unsigned long long)dim_k, (unsigned long long)dim_l);

    /* ------------------------------------------------------------------ */
    /* 3. Build registries and scan tiles                                  */
    /* ------------------------------------------------------------------ */
    TensorRegistry *reg_A = registry_create_from_dset(dset_A);
    TensorRegistry *reg_B = registry_create_from_dset(dset_B);
    if (!reg_A || !reg_B) {
        fprintf(stderr,
                "run_contraction_4d: registry_create_from_dset failed\n");
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    printf("Scanning tiles...\n");
    long tiles_A = registry_scan_file(dset_A, reg_A);
    long tiles_B = registry_scan_file(dset_B, reg_B);
    printf("  A: %ld tiles   B: %ld tiles\n", tiles_A, tiles_B);

    /* ------------------------------------------------------------------ */
    /* 4. Create C with shape (k, l, j, i)                                */
    /*                                                                     */
    /* CRITICAL: C's chunk dims must align exactly with A's i/j chunks    */
    /* and B's k/l chunks so that SUMMA tile boundaries are consistent.   */
    /* Do NOT derive from a fresh target_chunk_bytes — read directly from  */
    /* the registries that were built from the actual HDF5 creation plists. */
    /* ------------------------------------------------------------------ */
    hsize_t chunk_dims_C[4] = {
        reg_B->chunk_dims[1],   /* k chunk — from B's dim 1 */
        reg_B->chunk_dims[3],   /* l chunk — from B's dim 3 */
        reg_A->chunk_dims[1],   /* j chunk — from A's dim 1 */
        reg_A->chunk_dims[0],   /* i chunk — from A's dim 0 */
    };

    hsize_t global_C[4] = {dim_k, dim_l, dim_j, dim_i};
    if (create_chunked_dataset_explicit(file_C, name_C, 4, global_C,
                                        chunk_dims_C) < 0) {
        fprintf(stderr,
                "run_contraction_4d: create_chunked_dataset failed "
                "for '%s'\n", file_C);
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    hid_t fc     = H5Fopen(file_C, H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t dset_C = (fc >= 0) ? dset_open_no_cache(fc, name_C) : -1;
    if (fc < 0 || dset_C < 0) {
        fprintf(stderr,
                "run_contraction_4d: cannot open output '%s'\n", file_C);
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    TensorRegistry *reg_C = registry_create_from_dset(dset_C);
    if (!reg_C) {
        fprintf(stderr,
                "run_contraction_4d: registry_create_from_dset(C) failed\n");
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    /* ------------------------------------------------------------------ */
    /* 5. Initialise pool (80% RAM, minimum 7 pages)                      */
    /*                                                                     */
    /* Rank-4 double-buffer needs 7 pages per output tile:                */
    /*   2×A + 2×B  (double-buffer slots)                                 */
    /*   1×C        (accumulator)                                          */
    /*   1×B_perm   (permuted B scratchpad)                                */
    /*   1×C_blas   (BLAS output scratchpad)                               */
    /* ------------------------------------------------------------------ */
    size_t ram = query_physical_ram();

    size_t elems_A = 1, elems_B = 1, elems_C = 1;
    for (int d = 0; d < 4; d++) {
        elems_A *= (size_t)reg_A->chunk_dims[d];
        elems_B *= (size_t)reg_B->chunk_dims[d];
        elems_C *= (size_t)reg_C->chunk_dims[d];
    }
    size_t elems_per_page = elems_A;
    if (elems_B > elems_per_page) elems_per_page = elems_B;
    if (elems_C > elems_per_page) elems_per_page = elems_C;

    size_t pool_bytes = (size_t)((double)ram * 0.8);
    size_t num_pages  = pool_bytes / (elems_per_page * sizeof(double));
    if (num_pages < 7) {
        fprintf(stderr,
                "run_contraction_4d: RAM too small for 7 pages "
                "(need %zu bytes)\n",
                7 * elems_per_page * sizeof(double));
        engine_cleanup(NULL, reg_A, reg_B, reg_C,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    BufferPool *pool = pool_create(num_pages, elems_per_page * sizeof(double));
    if (!pool) {
        fprintf(stderr, "run_contraction_4d: pool_create failed\n");
        engine_cleanup(NULL, reg_A, reg_B, reg_C,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    printf("RAM: %.1f GB  Pool: %zu pages \xc3\x97 %zu elems = %.1f GB\n",
           (double)ram / (1024.0 * 1024.0 * 1024.0),
           num_pages, elems_per_page,
           (double)(num_pages * elems_per_page * sizeof(double))
               / (1024.0 * 1024.0 * 1024.0));

    /* Nominal chunk dims (used as strides for all flat indexing). */
    int i_nom = (int)reg_A->chunk_dims[0];  /* A dim 0 */
    int j_nom = (int)reg_A->chunk_dims[1];  /* A dim 1 */
    int a_nom = (int)reg_A->chunk_dims[2];  /* A dim 2 */
    int b_nom = (int)reg_A->chunk_dims[3];  /* A dim 3 */
    int k_nom = (int)reg_B->chunk_dims[1];  /* B dim 1 */
    int l_nom = (int)reg_B->chunk_dims[3];  /* B dim 3 */

    /* Derived BLAS dimensions (nominal). */
    int M_blas = i_nom * j_nom;   /* rows of A and C_blas */
    int K_blas = a_nom * b_nom;   /* cols of A = rows of B_perm */
    int N_blas = k_nom * l_nom;   /* cols of B_perm and C_blas */

    printf("Nominal chunk: (i=%d,j=%d,a=%d,b=%d)  BLAS M=%d K=%d N=%d\n",
           i_nom, j_nom, a_nom, b_nom, M_blas, K_blas, N_blas);
#if defined(USE_MKL)
    printf("Kernel: cblas_dgemm (Intel MKL)\n");
#elif defined(HAVE_CBLAS)
    printf("Kernel: cblas_dgemm (OpenBLAS)\n");
#else
    printf("Kernel: fallback (m\xe2\x86\x92k\xe2\x86\x92n loop)\n");
#endif

    /* Grid over the output tile space. */
    hsize_t K_tiles = reg_C->grid_dims[0];   /* k-dim of C */
    hsize_t L_tiles = reg_C->grid_dims[1];   /* l-dim of C */
    hsize_t J_tiles = reg_C->grid_dims[2];   /* j-dim of C */
    hsize_t I_tiles = reg_C->grid_dims[3];   /* i-dim of C */
    hsize_t A_tiles = reg_A->grid_dims[2];   /* contracted a */
    hsize_t B_tiles = reg_A->grid_dims[3];   /* contracted b */

    printf("C grid: [k=%llu, l=%llu, j=%llu, i=%llu]  "
           "contracted: [a=%llu, b=%llu]\n",
           (unsigned long long)K_tiles, (unsigned long long)L_tiles,
           (unsigned long long)J_tiles, (unsigned long long)I_tiles,
           (unsigned long long)A_tiles, (unsigned long long)B_tiles);

    /* ------------------------------------------------------------------ */
    /* 6. 4-D SUMMA execution loop                                         */
    /* ------------------------------------------------------------------ */
    int ret = 0;

    for (hsize_t ki = 0; ki < K_tiles && ret == 0; ki++) {
      for (hsize_t li = 0; li < L_tiles && ret == 0; li++) {
        for (hsize_t ji = 0; ji < J_tiles && ret == 0; ji++) {
          for (hsize_t ii = 0; ii < I_tiles && ret == 0; ii++) {

            /*
             * Acquire three dedicated pages for the duration of this output
             * tile: C accumulator, B_perm scratchpad, C_blas scratchpad.
             * These are held across the entire (a,b) loop so the compute
             * thread never calls pool_acquire under the mutex.
             */
            size_t  id_C = SIZE_MAX, id_Bp = SIZE_MAX, id_Cb = SIZE_MAX;
            double *buf_C      = pool_acquire(pool, &id_C);
            double *buf_B_perm = (buf_C)  ? pool_acquire(pool, &id_Bp) : NULL;
            double *buf_C_blas = (buf_B_perm) ? pool_acquire(pool, &id_Cb) : NULL;

            if (!buf_C || !buf_B_perm || !buf_C_blas) {
                fprintf(stderr,
                        "run_contraction_4d: pool exhausted acquiring "
                        "scratchpads at (%llu,%llu,%llu,%llu)\n",
                        (unsigned long long)ki, (unsigned long long)li,
                        (unsigned long long)ji, (unsigned long long)ii);
                if (buf_C)      pool_release(pool, id_C);
                if (buf_B_perm) pool_release(pool, id_Bp);
                ret = -1;
                break;
            }

            /* Zero the C accumulator; scratchpads are overwritten per slot. */
            memset(buf_C, 0, elems_per_page * sizeof(double));

            /* Set up shared state for the I/O thread. */
            IOShared4D s;
            memset(&s, 0, sizeof(s));
            pthread_mutex_init(&s.mu, NULL);
            pthread_cond_init(&s.cond, NULL);

            s.dset_A = dset_A;  s.dset_B = dset_B;  s.dset_C = dset_C;
            s.reg_A  = reg_A;   s.reg_B  = reg_B;   s.reg_C  = reg_C;
            s.pool   = pool;
            s.ki     = ki;      s.li     = li;
            s.ji     = ji;      s.ii     = ii;
            s.A_tiles = A_tiles;
            s.B_tiles = B_tiles;
            s.buf_C  = buf_C;   s.id_C   = id_C;

            pthread_t io_tid;
            if (pthread_create(&io_tid, NULL, io_thread_func_4d, &s) != 0) {
                fprintf(stderr,
                        "run_contraction_4d: pthread_create failed\n");
                pool_release(pool, id_Cb);
                pool_release(pool, id_Bp);
                pool_release(pool, id_C);
                pthread_mutex_destroy(&s.mu);
                pthread_cond_destroy(&s.cond);
                ret = -1;
                break;
            }

            /* ----------------------------------------------------------
             * Compute loop: drain READY slots.
             *
             * For each slot:
             *  (a) Permute B tile from (a,k,b,l) storage to
             *      (a*b, k*l) contiguous layout in buf_B_perm.
             *  (b) cblas_dgemm: C_blas = A × B_perm  (beta = 0.0).
             *  (c) Scatter-accumulate C_blas into buf_C, converting
             *      the (i*j, k*l) row-major result to (k,l,j,i) layout.
             *
             * All three steps happen inside the compute thread without
             * holding the mutex (buf_B_perm and buf_C_blas are private).
             * ---------------------------------------------------------- */
            int read_idx = 0;
            for (;;) {
                pthread_mutex_lock(&s.mu);
                while (s.state[read_idx] == SLOT_FREE)
                    pthread_cond_wait(&s.cond, &s.mu);
                int     cur_state = s.state[read_idx];
                double *bA        = s.buf_A[read_idx];
                double *bB        = s.buf_B[read_idx];
                pthread_mutex_unlock(&s.mu);

                if (cur_state == SLOT_EOF) break;

                /* (a) Permute B: (a,k,b,l) raw → (a*b, k*l) contiguous.
                 *
                 * Raw layout (row-major, nominal strides):
                 *   bB[a*(k_nom*b_nom*l_nom) + k*(b_nom*l_nom) + b*l_nom + l]
                 *
                 * Target layout (K_blas × N_blas, row-major):
                 *   buf_B_perm[(a*b_nom + b) * N_blas + k*l_nom + l]
                 *
                 * Every position is written exactly once; no pre-zero needed.
                 * Elements beyond actual tile extent are zero in bB (from
                 * read_chunk_fast pre-zeroing), so they propagate correctly.
                 */
                for (int a = 0; a < a_nom; a++) {
                    for (int b = 0; b < b_nom; b++) {
                        int kb_row = a * b_nom + b;   /* target row index */
                        for (int k = 0; k < k_nom; k++) {
                            int src_kbase = a*(k_nom*b_nom*l_nom)
                                          + k*(b_nom*l_nom)
                                          + b*l_nom;
                            int dst_kbase = kb_row * N_blas + k * l_nom;
                            for (int l = 0; l < l_nom; l++)
                                buf_B_perm[dst_kbase + l] =
                                    bB[src_kbase + l];
                        }
                    }
                }

                /* (b) BLAS: C_blas(M×N) = A(M×K) × B_perm(K×N).
                 *
                 * A flat layout: A[i_local, j_local, a_local, b_local]
                 *   = bA[(i*j_nom + j) * K_blas + a*b_nom + b]
                 *   → naturally M_blas × K_blas with lda = K_blas. ✓
                 *
                 * beta = 0.0: C_blas is fully overwritten each call.
                 */
#ifdef HAVE_CBLAS
                cblas_dgemm(CblasRowMajor,
                            CblasNoTrans, CblasNoTrans,
                            M_blas, N_blas, K_blas,
                            1.0,
                            bA,        K_blas,
                            buf_B_perm, N_blas,
                            0.0,
                            buf_C_blas, N_blas);
#else
                memset(buf_C_blas, 0,
                       (size_t)M_blas * (size_t)N_blas * sizeof(double));
                compute_tile(bA,        K_blas,
                             buf_B_perm, N_blas,
                             buf_C_blas, N_blas,
                             M_blas, N_blas, K_blas);
#endif

                /* (c) Scatter-accumulate: C_blas(i*j, k*l) → buf_C(k,l,j,i).
                 *
                 * C_blas row m = i*j_nom + j, column n = k*l_nom + l.
                 * buf_C target index for (k,l,j,i):
                 *   c_idx = k*(l_nom*j_nom*i_nom)
                 *         + l*(j_nom*i_nom)
                 *         + j*i_nom
                 *         + i
                 *
                 * Loop order (i,j,k,l): keeps C_blas reads sequential
                 * (row m = i*j_nom+j, column stride 1 in l).
                 */
                for (int i = 0; i < i_nom; i++) {
                    for (int j = 0; j < j_nom; j++) {
                        int m = i * j_nom + j;
                        for (int k = 0; k < k_nom; k++) {
                            int n_base = k * l_nom;
                            int c_base = k * (l_nom * j_nom * i_nom)
                                       + j * i_nom
                                       + i;
                            for (int l = 0; l < l_nom; l++)
                                buf_C[c_base + l * (j_nom * i_nom)] +=
                                    buf_C_blas[m * N_blas + n_base + l];
                        }
                    }
                }

                /* Return slot pages under mutex. */
                pthread_mutex_lock(&s.mu);
                pool_release(pool, s.id_A[read_idx]);
                pool_release(pool, s.id_B[read_idx]);
                s.state[read_idx] = SLOT_FREE;
                pthread_cond_broadcast(&s.cond);
                pthread_mutex_unlock(&s.mu);

                read_idx ^= 1;
            }

            /* Signal I/O thread to flush C, wait for completion. */
            pthread_mutex_lock(&s.mu);
            s.flush_req = 1;
            pthread_cond_broadcast(&s.cond);
            while (!s.flush_done)
                pthread_cond_wait(&s.cond, &s.mu);
            pthread_mutex_unlock(&s.mu);

            pthread_join(io_tid, NULL);
            pthread_mutex_destroy(&s.mu);
            pthread_cond_destroy(&s.cond);

            if (s.io_err) {
                fprintf(stderr,
                        "run_contraction_4d: I/O error at "
                        "C(%llu,%llu,%llu,%llu)\n",
                        (unsigned long long)ki, (unsigned long long)li,
                        (unsigned long long)ji, (unsigned long long)ii);
                ret = -1;
            }

            pool_release(pool, id_Cb);
            pool_release(pool, id_Bp);
            pool_release(pool, id_C);

            if (ret == 0) { printf("."); fflush(stdout); }
          }
        }
      }
    }

    if (ret == 0) printf("\nRank-4 contraction complete.\n");

    engine_cleanup(pool, reg_A, reg_B, reg_C,
                   dset_A, dset_B, dset_C, fa, fb, fc);
    return ret;
}

/* ======================================================================= */
/* Generic N-D einsum contraction                                           */
/* ======================================================================= */


/* ----------------------------------------------------------------------- */
/* perm_is_identity — returns 1 if perm[d]==d for all d in [0,rank).       */
/* ----------------------------------------------------------------------- */
static int perm_is_identity(const int *perm, int rank)
{
    for (int d = 0; d < rank; d++)
        if (perm[d] != d) return 0;
    return 1;
}

/* ======================================================================= */
/* Work-Stealing Heterogeneous Compute                                      */
/* ======================================================================= */

/* ----------------------------------------------------------------------- */
/* ContractionShared — read-only config shared by all worker threads.       */
/* ----------------------------------------------------------------------- */
typedef struct {
    const char               *file_A, *name_A;
    const char               *file_B, *name_B;
    hid_t                     dset_C;
    TensorRegistry           *reg_A, *reg_B, *reg_C;
    contraction_plan_t        plan;
    int                       rank_A, rank_B, rank_C;
    tensor_dtype_t            dtype;
    size_t                    element_size;
    hid_t                     h5type_mem;
    size_t                    c_grid_sz[MAX_RANK];
    hsize_t                   contracted_grid[MAX_RANK];
    size_t                    bytes_per_page;
    int                       M_nom, N_nom, K_nom;
    size_t                    blas_dims[MAX_RANK];
    size_t                    blas_strides[MAX_RANK];
    size_t                    chunk_dims_A_sz[MAX_RANK];
    size_t                    chunk_dims_B_sz[MAX_RANK];
    size_t                    total_blas;
    size_t                   *scatter_idx;
    size_t                    pool_capacity_bytes;
    size_t                    pool_num_pages;
    int                       accumulate;   /* 1 = C += A*B; 0 = C = A*B */
} ContractionShared;

/* Per-GCD-task metadata for exec_macroblock_gcd. */
typedef struct {
    int    fb_exists;            /* 1 if the B tile was on disk              */
    int    is_boundary;          /* 1 if any blas_phys dim < nominal         */
    size_t blas_phys[MAX_RANK];  /* actual [free_A dims | free_B dims] sizes */
} MBTask;

/* ----------------------------------------------------------------------- */
/* IOProfiler — deterministic I/O accounting for exec_macroblock_gcd       */
/*                                                                           */
/* All counters are plain size_t — no atomics needed because:               */
/*   • A reads happen only on the main thread (Step 1 or A pre-cache).      */
/*   • B reads happen only on the serial b_io_q (load_b block or pre-cache).*/
/*   • C writes happen only on the main thread (Step 4).                    */
/*   • Main thread reads the struct only after dispatch_sync drains b_io_q. */
/* ----------------------------------------------------------------------- */
typedef struct {
    /* --- Actual I/O (bytes via read_chunk_typed / write_chunk_typed) --- */
    size_t bytes_read_A;      /* cumulative bytes read from Tensor A        */
    size_t bytes_read_B;      /* cumulative bytes read from Tensor B        */
    size_t bytes_read_C;      /* cumulative bytes read from Tensor C (=0)   */
    size_t bytes_written_C;   /* cumulative bytes written to Tensor C       */

    /* --- Tile counts -------------------------------------------------- */
    size_t tiles_read_A;
    size_t tiles_read_B;
    size_t tiles_written_C;

    /* --- Per-macro-block B redundancy tracking ------------------------ */
    size_t b_bytes_cur_mb;    /* B bytes read this macro-block (reset/loop) */
    size_t b_redundant_bytes; /* total bytes read beyond theoretical floor  */

    /* --- Theoretical minimum I/O (set once before the outer loop) ----- */
    size_t theo_read_A;       /* Size(A): each A tile read exactly once      */
    size_t theo_read_B;       /* Size(B) if B-cached, K×Size(B) otherwise   */
    size_t theo_read_C;       /* 0: C must never be read back from disk      */
    size_t theo_write_C;      /* Size(C): each C tile written exactly once   */
    size_t theo_b_bytes_mb;   /* expected B bytes per macro-block in loop    */

    /* --- Pool context for the report ---------------------------------- */
    size_t pool_capacity_bytes;
    size_t pool_num_pages;
    size_t bytes_per_page;    /* bpp                                        */
    size_t n_macroblocks;     /* K = total_fA                               */
} IOProfiler;

/* ----------------------------------------------------------------------- */
/* exec_macroblock_gcd — forward declaration (defined below)               */
/* ----------------------------------------------------------------------- */
static int exec_macroblock_gcd(const ContractionShared *sh,
                                hid_t dset_A, hid_t dset_B, hid_t dset_C);




/* ----------------------------------------------------------------------- */
/* exec_macroblock_gcd                                                       */
/*                                                                           */
/* Out-of-core block-caching contraction with GCD parallel BLAS.            */
/*                                                                           */
/* Loop structure (A-pinning macro-block strategy):                          */
/*                                                                           */
/*   for each free_A macro-block (e.g. fixed (i,j) tile coords):            */
/*     pin  A[i,j,a,b]   for ALL contracted (a,b) in RAM (A_cache)          */
/*     zero C_accum[k,l] for ALL free_B   (k,l) tile accumulators           */
/*                                                                           */
/*     for each contracted pair (a,b):                                       */
/*       permute A_cache[a,b] → A_perm  (once per pair, not per C tile)     */
/*       for each free_B tile (k,l) [SERIAL I/O]:                           */
/*         read B[a,k,b,l] → permute → B_perm[k,l]                          */
/*       dispatch_apply(total_free_B) [GCD PARALLEL]:                        */
/*         cblas_zgemm(A_perm, B_perm[k,l]) → C_blas[k,l]  (beta=0)        */
/*         scatter-accumulate C_blas[k,l] → C_accum[k,l]                    */
/*                                                                           */
/*     for each free_B tile (k,l) [SERIAL]:                                 */
/*       write_chunk_typed(C_accum[k,l])                                     */
/*                                                                           */
/* I/O reduction:                                                            */
/*   A reads  OLD: N_C_tiles × N_contracted = N^4 × N^2 = N^6               */
/*   A reads  NEW: N_free_A  × N_contracted = N^2 × N^2 = N^4  (N^2 fewer) */
/*   B reads unchanged (always O(N^6) unique element accesses).              */
/*                                                                           */
/* GCD parallelism: the free_B BLAS loop is embarrassingly parallel —       */
/* each (k,l) task writes to a unique C_accum[k,l] buffer, requiring        */
/* zero mutexes in the hot math path.                                        */
/*                                                                           */
/* Memory layout (all posix_memalign'd, 16 KB NVMe-aligned):                */
/*   A_cache     total_contracted × bytes_per_page   (pinned across pairs)  */
/*   A_perm      1 × bytes_per_page                  (permuted A, per pair) */
/*   B_raw       1 × bytes_per_page                  (read scratch)         */
/*   B_perm_base total_free_B × bytes_per_page       (permuted B, per pair) */
/*   C_blas_base total_free_B × bytes_per_page       (BLAS scratch/task)   */
/*   C_accum_base total_free_B × bytes_per_page      (running accumulators) */
/* ----------------------------------------------------------------------- */

static int exec_macroblock_gcd(const ContractionShared *sh,
                                hid_t dset_A, hid_t dset_B, hid_t dset_C)
{
    const contraction_plan_t *plan = &sh->plan;
    const int rank_A  = sh->rank_A;
    const int rank_B  = sh->rank_B;
    const int rank_C  = sh->rank_C;
    const int n_fA    = plan->n_free_A;
    const int n_fB    = plan->n_free_B;
    const int n_con   = plan->n_contracted;
    const int is_cplx = (sh->dtype != DTYPE_FP64);
    const size_t bpp  = sh->bytes_per_page;
    const size_t esz  = sh->element_size;

    /* ------------------------------------------------------------------ */
    /* Grid sizes along each axis                                          */
    /* ------------------------------------------------------------------ */
    size_t fa_grid[MAX_RANK], total_fA = 1;
    for (int p = 0; p < n_fA; p++) {
        fa_grid[(size_t)p] =
            (size_t)sh->reg_A->grid_dims[(size_t)plan->perm_A[p]];
        total_fA *= fa_grid[(size_t)p];
    }

    size_t con_grid[MAX_RANK], total_con = 1;
    for (int d = 0; d < n_con; d++) {
        con_grid[(size_t)d] = (size_t)sh->contracted_grid[(size_t)d];
        total_con *= con_grid[(size_t)d];
    }

    size_t fb_grid[MAX_RANK], total_fB = 1;
    for (int q = 0; q < n_fB; q++) {
        fb_grid[(size_t)q] =
            (size_t)sh->reg_B->grid_dims[
                (size_t)plan->perm_B[n_con + q]];
        total_fB *= fb_grid[(size_t)q];
    }

    /* 2D SUMMA block sizes: ceil(sqrt(K)) rounded up to nearest integer.   */
    /* P_A × P_B outer pairs, each pinning block_fA × total_con A tiles.  */
    size_t block_fA = (size_t)ceil(sqrt((double)total_fA));
    if (block_fA < 1) block_fA = 1;
    if (block_fA > total_fA) block_fA = total_fA;
    size_t block_fB = (size_t)ceil(sqrt((double)total_fB));
    if (block_fB < 1) block_fB = 1;
    if (block_fB > total_fB) block_fB = total_fB;
    size_t P_A = (total_fA + block_fA - 1) / block_fA;  /* A-group count  */
    size_t P_B = (total_fB + block_fB - 1) / block_fB;  /* B-group count  */

    printf("Macroblock-GCD 2D-SUMMA execution:\n");
    printf("  free_A : %zu tiles  ->  %zu groups of <=%zu  (P_A=%zu)\n",
           total_fA, P_A, block_fA, P_A);
    printf("  contr. : %zu tiles\n", total_con);
    printf("  free_B : %zu tiles  ->  %zu groups of <=%zu  (P_B=%zu)\n",
           total_fB, P_B, block_fB, P_B);
    printf("  A-cache/gA    : %.3f GiB  (%zu x %zu tiles, loaded once per gA)\n",
           (double)(block_fA * total_con * bpp) / (1024.0*1024*1024),
           block_fA, total_con);
    printf("  B-buf (2 slots): %.3f GiB  (%zu tiles x 2)\n",
           (double)(2 * block_fB * bpp) / (1024.0*1024*1024), block_fB);
    printf("  C-accum/pair  : %.3f GiB  (%zu x %zu tiles)\n",
           (double)(block_fA * block_fB * bpp) / (1024.0*1024*1024),
           block_fA, block_fB);

    /* Initialise profiler (theoretical minimums set after cache decisions). */
    IOProfiler prof;
    memset(&prof, 0, sizeof(prof));
    prof.bytes_per_page      = bpp;
    prof.pool_capacity_bytes = sh->pool_capacity_bytes;
    prof.pool_num_pages      = sh->pool_num_pages;
    prof.n_macroblocks       = P_A * P_B;

    /* ------------------------------------------------------------------ */
    /* Allocate buffers (16 KB NVMe-aligned, not from pool)               */
    /* ------------------------------------------------------------------ */
#define MB_ALLOC(ptr, n_pages) \
    do { \
        if (posix_memalign((void **)&(ptr), 16384, (n_pages) * bpp) != 0) { \
            fprintf(stderr, "exec_macroblock_gcd: alloc failed (%s)\n", #ptr); \
            goto mb_cleanup; \
        } \
    } while (0)

    int ret = 0;

    char   *A_cache_base  = NULL;
    char   *A_perm_buf    = NULL;
    char   *B_raw_buf     = NULL;
    char   *B_perm_buf[2] = {NULL, NULL};  /* ping-pong double buffers */
    char   *C_blas_base   = NULL;
    char   *C_accum_base  = NULL;
    MBTask *tasks_buf[2]  = {NULL, NULL};
    int    *A_exist       = NULL;
    hsize_t *con_all      = NULL;          /* contracted-pair coord array */

    /* B pre-cache: all contracted-pair B tiles, loaded once before the loop. */
    char   *B_full_cache  = NULL;
    MBTask *tasks_full    = NULL;
    int     use_b_cache   = 0;

    /* 2D SUMMA coordinate tables and per-gA A-cache physical sizes.         */
    hsize_t *fa_all       = NULL;   /* [total_fA × MAX_RANK]                 */
    hsize_t *fb_all       = NULL;   /* [total_fB × MAX_RANK]                 */
    size_t  *A_phys_cache = NULL;   /* [block_fA × total_con × MAX_RANK]     */

    /* GCD objects — declared here (before any goto) and initialised below. */
#ifdef HAS_GCD
    dispatch_queue_t     b_io_q   = NULL;
    dispatch_semaphore_t b_sem0   = NULL;
    dispatch_semaphore_t b_sem1   = NULL;
    int                 *b_io_err = NULL;  /* heap[2]: [0]=slot0, [1]=slot1 */
#endif

    /* A_cache holds block_fA × total_con permuted tiles; reused for all gB. */
    MB_ALLOC(A_cache_base,  block_fA * total_con);
    MB_ALLOC(A_perm_buf,    1);           /* scratch for A load+permute step */
    MB_ALLOC(B_raw_buf,     1);
    MB_ALLOC(B_perm_buf[0], block_fB);   /* double-buffer: slot 0           */
    MB_ALLOC(B_perm_buf[1], block_fB);   /* double-buffer: slot 1           */
    MB_ALLOC(C_blas_base,   block_fA * block_fB);
    MB_ALLOC(C_accum_base,  block_fA * block_fB);

    tasks_buf[0]  = (MBTask  *)malloc(block_fB * sizeof(MBTask));
    tasks_buf[1]  = (MBTask  *)malloc(block_fB * sizeof(MBTask));
    A_exist       = (int     *)malloc(block_fA * total_con * sizeof(int));
    con_all       = (hsize_t *)malloc(total_con * MAX_RANK * sizeof(hsize_t));
    fa_all        = (hsize_t *)malloc(total_fA  * MAX_RANK * sizeof(hsize_t));
    fb_all        = (hsize_t *)malloc(total_fB  * MAX_RANK * sizeof(hsize_t));
    A_phys_cache  = (size_t  *)malloc(
                        block_fA * total_con * MAX_RANK * sizeof(size_t));
    if (!tasks_buf[0] || !tasks_buf[1] || !A_exist || !con_all ||
        !fa_all || !fb_all || !A_phys_cache) {
        fprintf(stderr, "exec_macroblock_gcd: malloc failed (bufs/coords)\n");
        goto mb_cleanup;
    }

    /* Pre-enumerate all coord arrays (used by the 2D outer loop).           */
    {
        size_t con[MAX_RANK]; memset(con, 0, sizeof(con));
        size_t cf = 0;
        do {
            for (int d = 0; d < n_con; d++)
                con_all[cf * MAX_RANK + (size_t)d] = (hsize_t)con[(size_t)d];
            cf++;
        } while (odometer_step((size_t)n_con, con, con_grid));
    }
    {
        size_t fa[MAX_RANK]; memset(fa, 0, sizeof(fa));
        size_t fi = 0;
        do {
            for (int p = 0; p < n_fA; p++)
                fa_all[fi * MAX_RANK + (size_t)p] = (hsize_t)fa[(size_t)p];
            fi++;
        } while (odometer_step((size_t)n_fA, fa, fa_grid));
    }
    {
        size_t fb[MAX_RANK]; memset(fb, 0, sizeof(fb));
        size_t fi = 0;
        do {
            for (int q = 0; q < n_fB; q++)
                fb_all[fi * MAX_RANK + (size_t)q] = (hsize_t)fb[(size_t)q];
            fi++;
        } while (odometer_step((size_t)n_fB, fb, fb_grid));
    }

#ifdef HAS_GCD
    b_io_q  = dispatch_queue_create("mb.b_io", DISPATCH_QUEUE_SERIAL);
    b_sem0  = dispatch_semaphore_create(0);
    b_sem1  = dispatch_semaphore_create(0);
    b_io_err = (int *)calloc(2, sizeof(int));
    if (!b_io_q || !b_sem0 || !b_sem1 || !b_io_err) {
        fprintf(stderr, "exec_macroblock_gcd: GCD init failed\n");
        goto mb_cleanup;
    }
#endif

#undef MB_ALLOC

    /* ------------------------------------------------------------------ */
    /* B tile pre-cache (optional): if total_con × total_fB tiles fit in  */
    /* RAM, read every permuted B tile once and skip HDF5 in the loop.    */
    /* The limit is query_physical_ram() / 8 capped at 4 GiB.            */
    /* ------------------------------------------------------------------ */
    {
        size_t ram_limit = query_physical_ram() / 8;
        if (ram_limit > 4UL * 1024UL * 1024UL * 1024UL)
            ram_limit = 4UL * 1024UL * 1024UL * 1024UL;
        size_t b_cache_bytes = total_con * total_fB * bpp;

        if (b_cache_bytes <= ram_limit &&
            posix_memalign((void **)&B_full_cache, 16384,
                           b_cache_bytes) == 0) {
            tasks_full = (MBTask *)calloc(total_con * total_fB, sizeof(MBTask));
            if (tasks_full)
                use_b_cache = 1;
            else {
                free(B_full_cache);
                B_full_cache = NULL;
            }
        }

        printf("  B pre-cache : ");
        if (use_b_cache)
            printf("%.3f GiB  (loading all B tiles once)\n",
                   (double)b_cache_bytes / (1024.0 * 1024 * 1024));
        else
            printf("skipped  (%.3f GiB > limit or alloc failed)\n",
                   (double)b_cache_bytes / (1024.0 * 1024 * 1024));
    }

    if (use_b_cache) {
        /* Pre-load + permute every B tile into B_full_cache[cf][ff].    */
        size_t cf = 0;
        while (cf < total_con) {
            const hsize_t *con_row = con_all + cf * MAX_RANK;
            size_t fb[MAX_RANK];
            memset(fb, 0, sizeof(fb));
            size_t ff = 0;
            do {
                hsize_t b_tile[MAX_RANK];
                memset(b_tile, 0, sizeof(b_tile));
                for (int d = 0; d < n_con; d++)
                    b_tile[(size_t)plan->perm_B[d]] = con_row[(size_t)d];
                for (int q = 0; q < n_fB; q++)
                    b_tile[(size_t)plan->perm_B[n_con + q]] =
                        (hsize_t)fb[(size_t)q];

                MBTask *t = &tasks_full[cf * total_fB + ff];
                TileMetadata *mB = registry_get_tile(sh->reg_B, b_tile);
                t->fb_exists =
                    (mB && mB->status == TILE_STATUS_ON_DISK) ? 1 : 0;

                if (t->fb_exists) {
                    memset(B_raw_buf, 0, bpp);
                    if (read_chunk_typed(dset_B, mB->phys_offset, B_raw_buf,
                                         esz, rank_B, sh->reg_B->chunk_dims,
                                         sh->h5type_mem) < 0) {
                        fprintf(stderr,
                                "exec_macroblock_gcd: B pre-cache read error\n");
                        ret = -1;
                        goto mb_cleanup;
                    }
                    prof.bytes_read_B += bpp;
                    prof.tiles_read_B++;
                    size_t phys_B[MAX_RANK];
                    for (int d = 0; d < rank_B; d++) {
                        hsize_t end = mB->phys_offset[(size_t)d]
                                    + sh->reg_B->chunk_dims[(size_t)d];
                        phys_B[(size_t)d] = (size_t)(
                            (end > sh->reg_B->global_dims[(size_t)d])
                            ? sh->reg_B->global_dims[(size_t)d]
                              - mB->phys_offset[(size_t)d]
                            : sh->reg_B->chunk_dims[(size_t)d]);
                    }
                    char *dst = B_full_cache + (cf * total_fB + ff) * bpp;
                    if (perm_is_identity(plan->perm_B, rank_B)) {
                        memcpy(dst, B_raw_buf, bpp);
                    } else {
                        memset(dst, 0, bpp);
                        tensor_permute(B_raw_buf, dst, (size_t)rank_B, phys_B,
                                       sh->chunk_dims_B_sz, plan->perm_B, esz);
                    }
                    for (int q = 0; q < n_fB; q++)
                        t->blas_phys[(size_t)(n_fA + q)] =
                            phys_B[(size_t)plan->perm_B[n_con + q]];
                }
                ff++;
            } while (odometer_step((size_t)n_fB, fb, fb_grid));
            cf++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* Theoretical minimum I/O for 2D SUMMA.                             */
    /*                                                                    */
    /* A reads: Size(A) — A_cache loaded once per gA, reused for P_B gBs.*/
    /* B reads: P_A × Size(B) — B re-read for each of the P_A A-groups.  */
    /* C reads: 0 — C accumulators stay in RAM for the whole (gA,gB).    */
    /* C writes: Size(C) — each C tile written once at end of its pair.  */
    /* ------------------------------------------------------------------ */
    {
        size_t size_A = total_fA * total_con * bpp;
        size_t size_B = total_con * total_fB * bpp;
        size_t size_C = total_fA * total_fB * bpp;

        prof.theo_read_A    = size_A;
        prof.theo_read_B    = use_b_cache ? size_B : P_A * size_B;
        prof.theo_read_C    = sh->accumulate ? size_C : 0;
        prof.theo_write_C   = size_C;
        /* Per-(gA,gB) B budget: total_con contracted pairs × block_fB tiles.*/
        prof.theo_b_bytes_mb = use_b_cache ? 0 : total_con * block_fB * bpp;
    }

    /* ------------------------------------------------------------------ */
    /* 2D SUMMA outer loop                                                */
    /*                                                                    */
    /* Outer gA: load block_fA × total_con A tiles (once per gA).        */
    /* Inner gB: zero C, stream B (block_fB tiles/contracted pair,       */
    /*           double-buffered), dispatch_apply BLAS, write C.         */
    /* ------------------------------------------------------------------ */
    size_t pair_done = 0;

    for (size_t gA = 0; gA < P_A && ret == 0; gA++) {
        size_t fa_lo    = gA * block_fA;
        size_t fa_hi    = fa_lo + block_fA;
        if (fa_hi > total_fA) fa_hi = total_fA;
        size_t n_fA_cur = fa_hi - fa_lo;

        /* ---------------------------------------------------------------- */
        /* Step 1: Load and permute A cache for this gA group.             */
        /* A_cache_base[fai_local * total_con + cf] = permuted A tile.     */
        /* A_phys_cache[same index, MAX_RANK] = actual dims for boundary.  */
        /* ---------------------------------------------------------------- */
        for (size_t fai = fa_lo; fai < fa_hi && ret == 0; fai++) {
            size_t fai_local      = fai - fa_lo;
            const hsize_t *fa_row = fa_all + fai * MAX_RANK;

            for (size_t cf = 0; cf < total_con; cf++) {
                hsize_t a_tile[MAX_RANK];
                memset(a_tile, 0, sizeof(a_tile));
                for (int p = 0; p < n_fA; p++)
                    a_tile[(size_t)plan->perm_A[p]] = fa_row[(size_t)p];
                const hsize_t *con_row = con_all + cf * MAX_RANK;
                for (int d = 0; d < n_con; d++)
                    a_tile[(size_t)plan->perm_A[n_fA + d]] = con_row[(size_t)d];

                char *dst_A = A_cache_base + (fai_local * total_con + cf) * bpp;
                size_t *pa  = A_phys_cache +
                              (fai_local * total_con + cf) * MAX_RANK;

                TileMetadata *mA = registry_get_tile(sh->reg_A, a_tile);
                if (mA && mA->status == TILE_STATUS_ON_DISK) {
                    memset(A_perm_buf, 0, bpp);
                    if (read_chunk_typed(dset_A, mA->phys_offset, A_perm_buf,
                                         esz, rank_A, sh->reg_A->chunk_dims,
                                         sh->h5type_mem) < 0) {
                        fprintf(stderr, "exec_macroblock_gcd: A read error\n");
                        ret = -1; break;
                    }
                    prof.bytes_read_A += bpp;
                    prof.tiles_read_A++;
                    /* Compute physical dims (for boundary detection). */
                    for (int d = 0; d < rank_A; d++) {
                        hsize_t end = mA->phys_offset[(size_t)d]
                                    + sh->reg_A->chunk_dims[(size_t)d];
                        pa[(size_t)d] = (size_t)(
                            (end > sh->reg_A->global_dims[(size_t)d])
                            ? sh->reg_A->global_dims[(size_t)d]
                              - mA->phys_offset[(size_t)d]
                            : sh->reg_A->chunk_dims[(size_t)d]);
                    }
                    /* Permute into A_cache_base. */
                    if (perm_is_identity(plan->perm_A, rank_A)) {
                        memcpy(dst_A, A_perm_buf, bpp);
                    } else {
                        memset(dst_A, 0, bpp);
                        tensor_permute(A_perm_buf, dst_A, (size_t)rank_A, pa,
                                       sh->chunk_dims_A_sz, plan->perm_A, esz);
                    }
                    A_exist[fai_local * total_con + cf] = 1;
                } else {
                    memset(dst_A, 0, bpp);
                    memset(pa, 0, MAX_RANK * sizeof(size_t));
                    A_exist[fai_local * total_con + cf] = 0;
                }
            }
        }
        if (ret != 0) goto mb_cleanup;

        /* ---------------------------------------------------------------- */
        /* Inner gB loop                                                    */
        /* ---------------------------------------------------------------- */
        for (size_t gB = 0; gB < P_B && ret == 0; gB++) {
            size_t fb_lo    = gB * block_fB;
            size_t fb_hi    = fb_lo + block_fB;
            if (fb_hi > total_fB) fb_hi = total_fB;
            size_t n_fB_cur = fb_hi - fb_lo;

            /* Step 2: Initialise C accumulators for this (gA, gB) pair.
             *
             * Normal mode:     zero-fill (C = A*B from scratch).
             * Accumulate mode: read existing C tiles; zero if absent.
             *                  This implements C += A*B out-of-core.
             */
            if (!sh->accumulate) {
                memset(C_accum_base, 0, n_fA_cur * n_fB_cur * bpp);
            } else {
                for (size_t fai_l = 0; fai_l < n_fA_cur && ret == 0; fai_l++) {
                    size_t fai = fa_lo + fai_l;
                    const hsize_t *fa_row_c = fa_all + fai * MAX_RANK;
                    for (size_t fbi_l = 0; fbi_l < n_fB_cur; fbi_l++) {
                        size_t fbi = fb_lo + fbi_l;
                        const hsize_t *fb_row_c = fb_all + fbi * MAX_RANK;
                        /* Compute C tile coords from free-A/free-B coords. */
                        hsize_t c_tile[MAX_RANK];
                        memset(c_tile, 0, sizeof(c_tile));
                        for (int d = 0; d < rank_C; d++) {
                            int bc = plan->perm_C[(size_t)d];
                            c_tile[(size_t)d] = (bc < n_fA)
                                ? fa_row_c[(size_t)bc]
                                : fb_row_c[(size_t)(bc - n_fA)];
                        }
                        char *C_data = C_accum_base +
                                       (fai_l * n_fB_cur + fbi_l) * bpp;
                        /* Pre-zero handles missing tiles and boundary padding. */
                        memset(C_data, 0, bpp);
                        TileMetadata *mC = registry_get_tile(sh->reg_C, c_tile);
                        if (mC && mC->status == TILE_STATUS_ON_DISK) {
                            if (read_chunk_typed(dset_C, mC->phys_offset,
                                                 C_data, esz, rank_C,
                                                 sh->reg_C->chunk_dims,
                                                 sh->h5type_mem) < 0) {
                                fprintf(stderr,
                                        "exec_macroblock_gcd: C read error "
                                        "(accumulate mode)\n");
                                ret = -1;
                                goto mb_cleanup;
                            }
                            prof.bytes_read_C += bpp;
                        }
                    }
                    if (ret != 0) goto mb_cleanup;
                }
            }
            prof.b_bytes_cur_mb = 0;

            /* Step 3: Contracted loop. */
            if (use_b_cache) {
                /* B globally pre-cached: iterate pairs, BLAS only. */
                for (size_t cf = 0; cf < total_con && ret == 0; cf++) {
                    /* Check any A exists for this cf across fA group. */
                    int any_a = 0;
                    for (size_t fai_l = 0; fai_l < n_fA_cur; fai_l++)
                        if (A_exist[fai_l * total_con + cf]) { any_a = 1; break; }
                    if (!any_a) continue;

                    /* Pointers captured by the block. */
                    const char   *cap_Ap     = A_cache_base;  /* base; index = (fai*tcon+cf)*bpp */
                    /* B slice for this gB group within the pre-cache. */
                    const char   *cap_Bp     = B_full_cache
                                               + cf * total_fB * bpp
                                               + fb_lo * bpp;
                    char         *cap_Cb     = C_blas_base;
                    char         *cap_Ca     = C_accum_base;
                    const MBTask *cap_Brow   = tasks_full + cf * total_fB
                                               + fb_lo;
                    const size_t *cap_sidx   = sh->scatter_idx;
                    const size_t *cap_bstr   = sh->blas_strides;
                    size_t cap_tblas         = sh->total_blas;
                    int cap_Mn = sh->M_nom, cap_Kn = sh->K_nom,
                        cap_Nn = sh->N_nom;
                    int    cap_rC    = rank_C;
                    int    cap_cx    = is_cplx;
                    size_t cap_bpp   = bpp;
                    size_t cap_nfAc  = n_fA_cur;
                    size_t cap_nfBc  = n_fB_cur;
                    size_t cap_tcon  = total_con;
                    int   *cap_Aexist = A_exist;
                    const size_t *cap_Aphys = A_phys_cache;  /* base; index = (fai*tcon+cf)*MAX_RANK */
                    const size_t *cap_bd    = sh->blas_dims;
                    int cap_nfA_bc = n_fA;

#ifdef HAS_GCD
                    dispatch_apply(cap_nfAc * cap_nfBc,
                                   DISPATCH_APPLY_AUTO,
                                   ^(size_t task_idx) {
                        size_t fai_l = task_idx / cap_nfBc;
                        size_t fbi_l = task_idx % cap_nfBc;
                        if (!cap_Aexist[fai_l * cap_tcon + cf]) return;
                        if (!cap_Brow[fbi_l].fb_exists) return;
                        const void *bA  = cap_Ap + (fai_l * cap_tcon + cf) * cap_bpp;
                        const void *bB  = cap_Bp + fbi_l * cap_bpp;
                        void       *bCb = cap_Cb + task_idx * cap_bpp;
                        void       *bCa = cap_Ca + task_idx * cap_bpp;
#ifdef TENSOR_ZGEMM
                        if (!cap_cx) {
                            double alpha = 1.0, beta = 0.0;
                            TENSOR_DGEMM(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                                cap_Mn,cap_Nn,cap_Kn, alpha,
                                (const double *)bA, cap_Kn,
                                (const double *)bB, cap_Nn,
                                beta, (double *)bCb, cap_Nn);
                        } else {
                            double _Complex alpha=CMPLX(1.0,0.0),beta=CMPLX(0.0,0.0);
                            TENSOR_ZGEMM(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                                cap_Mn,cap_Nn,cap_Kn, &alpha,
                                (const double _Complex *)bA, cap_Kn,
                                (const double _Complex *)bB, cap_Nn,
                                &beta,(double _Complex *)bCb, cap_Nn);
                        }
#else
                        memset(bCb, 0, cap_bpp);
#endif
                        /* Combined blas_phys: free-A from A_phys, free-B from task. */
                        size_t bphys[MAX_RANK];
                        const size_t *pa2 = cap_Aphys +
                            (fai_l * cap_tcon + cf) * MAX_RANK;
                        for (int d=0; d<cap_rC; d++) bphys[(size_t)d] =
                            (d < cap_nfA_bc)
                                ? pa2[(size_t)plan->perm_A[d]]
                                : cap_Brow[fbi_l].blas_phys[(size_t)d];
                        int is_bnd = 0;
                        for (int d=0; d<cap_rC; d++)
                            if (bphys[(size_t)d] < cap_bd[(size_t)d]) { is_bnd=1; break; }
                        if (!is_bnd) {
                            if (!cap_cx) {
                                const double *src=(const double *)bCb;
                                double *dst=(double *)bCa;
                                for (size_t f=0;f<cap_tblas;f++) dst[cap_sidx[f]]+=src[f];
                            } else {
                                const double _Complex *src=(const double _Complex *)bCb;
                                double _Complex *dst=(double _Complex *)bCa;
                                for (size_t f=0;f<cap_tblas;f++) dst[cap_sidx[f]]+=src[f];
                            }
                        } else {
                            size_t bc[MAX_RANK];
                            memset(bc,0,(size_t)cap_rC*sizeof(size_t));
                            do {
                                size_t bf=compute_flat_index((size_t)cap_rC,bc,cap_bstr);
                                if (!cap_cx)
                                    ((double*)bCa)[cap_sidx[bf]]+=((const double*)bCb)[bf];
                                else
                                    ((double _Complex*)bCa)[cap_sidx[bf]]+=
                                        ((const double _Complex*)bCb)[bf];
                            } while (odometer_step((size_t)cap_rC, bc, bphys));
                        }
                    });
#else
                    for (size_t task_idx = 0; task_idx < cap_nfAc * cap_nfBc; task_idx++) {
                        size_t fai_l = task_idx / cap_nfBc;
                        size_t fbi_l = task_idx % cap_nfBc;
                        if (!cap_Aexist[fai_l * cap_tcon + cf]) continue;
                        if (!cap_Brow[fbi_l].fb_exists) continue;
                        const void *bA  = cap_Ap + (fai_l * cap_tcon + cf) * cap_bpp;
                        const void *bB  = cap_Bp + fbi_l * cap_bpp;
                        void       *bCb = cap_Cb + task_idx * cap_bpp;
                        void       *bCa = cap_Ca + task_idx * cap_bpp;
#ifdef TENSOR_ZGEMM
                        if (!cap_cx) {
                            double alpha=1.0,beta=0.0;
                            TENSOR_DGEMM(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                                cap_Mn,cap_Nn,cap_Kn, alpha,
                                (const double *)bA,cap_Kn,
                                (const double *)bB,cap_Nn,
                                beta,(double *)bCb,cap_Nn);
                        } else {
                            double _Complex alpha=CMPLX(1.0,0.0),beta=CMPLX(0.0,0.0);
                            TENSOR_ZGEMM(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                                cap_Mn,cap_Nn,cap_Kn, &alpha,
                                (const double _Complex *)bA,cap_Kn,
                                (const double _Complex *)bB,cap_Nn,
                                &beta,(double _Complex *)bCb,cap_Nn);
                        }
#endif
                        size_t bphys[MAX_RANK];
                        {
                            const size_t *pa2s = cap_Aphys +
                                (fai_l * cap_tcon + cf) * MAX_RANK;
                            for (int d=0;d<cap_rC;d++) bphys[(size_t)d] =
                                (d < cap_nfA_bc)
                                    ? pa2s[(size_t)plan->perm_A[d]]
                                    : cap_Brow[fbi_l].blas_phys[(size_t)d];
                        }
                        int is_bnd=0;
                        for (int d=0;d<cap_rC;d++)
                            if (bphys[(size_t)d]<cap_bd[(size_t)d]){is_bnd=1;break;}
                        if (!is_bnd) {
                            if (!cap_cx){
                                const double *src=(const double *)bCb;
                                double *dst=(double *)bCa;
                                for(size_t f=0;f<cap_tblas;f++) dst[cap_sidx[f]]+=src[f];
                            } else {
                                const double _Complex *src=(const double _Complex*)bCb;
                                double _Complex *dst=(double _Complex*)bCa;
                                for(size_t f=0;f<cap_tblas;f++) dst[cap_sidx[f]]+=src[f];
                            }
                        } else {
                            size_t bc[MAX_RANK];
                            memset(bc,0,(size_t)cap_rC*sizeof(size_t));
                            do {
                                size_t bf=compute_flat_index((size_t)cap_rC,bc,cap_bstr);
                                if(!cap_cx)
                                    ((double*)bCa)[cap_sidx[bf]]+=((const double*)bCb)[bf];
                                else
                                    ((double _Complex*)bCa)[cap_sidx[bf]]+=
                                        ((const double _Complex*)bCb)[bf];
                            } while(odometer_step((size_t)cap_rC,bc,bphys));
                        }
                    }
#endif /* HAS_GCD */
                } /* for cf (B-cached) */

            } else {
                /* -------------------------------------------------------- */
                /* Double-buffered B I/O + GCD BLAS.                       */
                /* load_b fills block_fB B tiles for (cf_idx, gB) into     */
                /* slot bslot; main thread runs BLAS on the ready slot.    */
                /* -------------------------------------------------------- */
#ifdef HAS_GCD
                b_io_err[0] = 0;
                b_io_err[1] = 0;

                char   *b_pb0 = B_perm_buf[0];
                char   *b_pb1 = B_perm_buf[1];
                MBTask *tb0   = tasks_buf[0];
                MBTask *tb1   = tasks_buf[1];

                IOProfiler *prof_ptr    = &prof;
                size_t      fb_lo_cap   = fb_lo;
                size_t      n_fB_cur_cap = n_fB_cur;

                void (^load_b)(size_t, int) = ^(size_t cf_idx, int bslot) {
                    const hsize_t *con_row = con_all + cf_idx * MAX_RANK;
                    char   *Bpb   = (bslot == 0) ? b_pb0 : b_pb1;
                    MBTask *btask = (bslot == 0) ? tb0   : tb1;
                    int    err    = 0;

                    for (size_t fbi_l = 0; fbi_l < n_fB_cur_cap; fbi_l++) {
                        size_t fbi = fb_lo_cap + fbi_l;
                        const hsize_t *fb_row = fb_all + fbi * MAX_RANK;

                        hsize_t b_tile[MAX_RANK];
                        memset(b_tile, 0, sizeof(b_tile));
                        for (int d = 0; d < n_con; d++)
                            b_tile[(size_t)plan->perm_B[d]] = con_row[(size_t)d];
                        for (int q = 0; q < n_fB; q++)
                            b_tile[(size_t)plan->perm_B[n_con + q]] =
                                fb_row[(size_t)q];

                        TileMetadata *mB = registry_get_tile(sh->reg_B, b_tile);
                        btask[fbi_l].fb_exists =
                            (mB && mB->status == TILE_STATUS_ON_DISK) ? 1 : 0;

                        if (btask[fbi_l].fb_exists) {
                            memset(B_raw_buf, 0, bpp);
                            if (read_chunk_typed(dset_B, mB->phys_offset,
                                                 B_raw_buf, esz, rank_B,
                                                 sh->reg_B->chunk_dims,
                                                 sh->h5type_mem) < 0) {
                                err = 1;
                                btask[fbi_l].fb_exists = 0;
                            } else {
                                prof_ptr->bytes_read_B   += bpp;
                                prof_ptr->tiles_read_B++;
                                prof_ptr->b_bytes_cur_mb += bpp;
                                size_t phys_B[MAX_RANK];
                                for (int d = 0; d < rank_B; d++) {
                                    hsize_t end = mB->phys_offset[(size_t)d]
                                                + sh->reg_B->chunk_dims[(size_t)d];
                                    phys_B[(size_t)d] = (size_t)(
                                        (end > sh->reg_B->global_dims[(size_t)d])
                                        ? sh->reg_B->global_dims[(size_t)d]
                                          - mB->phys_offset[(size_t)d]
                                        : sh->reg_B->chunk_dims[(size_t)d]);
                                }
                                char *bperm = Bpb + fbi_l * bpp;
                                if (perm_is_identity(plan->perm_B, rank_B)) {
                                    memcpy(bperm, B_raw_buf, bpp);
                                } else {
                                    memset(bperm, 0, bpp);
                                    tensor_permute(B_raw_buf, bperm,
                                                   (size_t)rank_B, phys_B,
                                                   sh->chunk_dims_B_sz,
                                                   plan->perm_B, esz);
                                }
                                /* Store free-B phys dims at blas_phys[n_fA+q]. */
                                for (int q = 0; q < n_fB; q++)
                                    btask[fbi_l].blas_phys[(size_t)(n_fA + q)] =
                                        phys_B[(size_t)plan->perm_B[n_con + q]];
                            }
                        }
                    }
                    b_io_err[bslot] = err;
                    dispatch_semaphore_signal(bslot == 0 ? b_sem0 : b_sem1);
                };

                /* Kick pipeline: cf=0→slot0, cf=1→slot1. */
                dispatch_async(b_io_q, ^{ load_b(0, 0); });
                if (total_con > 1)
                    dispatch_async(b_io_q, ^{ load_b(1, 1); });

                size_t cf = 0;
                while (cf < total_con && ret == 0) {
                    int bslot = (int)(cf % 2);
                    dispatch_semaphore_wait(bslot == 0 ? b_sem0 : b_sem1,
                                           DISPATCH_TIME_FOREVER);
                    if (b_io_err[bslot]) {
                        fprintf(stderr,
                                "exec_macroblock_gcd: async B read error\n");
                        ret = -1; break;
                    }

                    /* Check if any A tile exists for this cf in the group. */
                    int any_a = 0;
                    for (size_t fai_l = 0; fai_l < n_fA_cur; fai_l++)
                        if (A_exist[fai_l * total_con + cf]) { any_a = 1; break; }

                    if (any_a) {
                        MBTask *btask_cur = (bslot == 0) ? tb0 : tb1;
                        const char   *cap_Ap     = A_cache_base;
                        const char   *cap_Bp     = (bslot==0) ? b_pb0 : b_pb1;
                        char         *cap_Cb     = C_blas_base;
                        char         *cap_Ca     = C_accum_base;
                        const MBTask *cap_tasks  = btask_cur;
                        const size_t *cap_sidx   = sh->scatter_idx;
                        const size_t *cap_bstr   = sh->blas_strides;
                        size_t cap_tblas         = sh->total_blas;
                        int cap_Mn = sh->M_nom, cap_Kn = sh->K_nom,
                            cap_Nn = sh->N_nom;
                        int    cap_rC    = rank_C;
                        int    cap_cx    = is_cplx;
                        size_t cap_bpp   = bpp;
                        size_t cap_nfAc  = n_fA_cur;
                        size_t cap_nfBc  = n_fB_cur;
                        size_t cap_tcon  = total_con;
                        size_t cap_cf    = cf;
                        int   *cap_Aexist = A_exist;
                        const size_t *cap_Aphys = A_phys_cache;
                        const size_t *cap_bd    = sh->blas_dims;
                        int cap_nfA = n_fA;

                        dispatch_apply(cap_nfAc * cap_nfBc,
                                       DISPATCH_APPLY_AUTO,
                                       ^(size_t task_idx) {
                            size_t fai_l = task_idx / cap_nfBc;
                            size_t fbi_l = task_idx % cap_nfBc;
                            if (!cap_Aexist[fai_l * cap_tcon + cap_cf]) return;
                            if (!cap_tasks[fbi_l].fb_exists) return;
                            const void *bA  = cap_Ap +
                                (fai_l * cap_tcon + cap_cf) * cap_bpp;
                            const void *bB  = cap_Bp + fbi_l * cap_bpp;
                            void       *bCb = cap_Cb + task_idx * cap_bpp;
                            void       *bCa = cap_Ca + task_idx * cap_bpp;
#ifdef TENSOR_ZGEMM
                            if (!cap_cx) {
                                double alpha=1.0,beta=0.0;
                                TENSOR_DGEMM(CblasRowMajor,CblasNoTrans,
                                    CblasNoTrans, cap_Mn,cap_Nn,cap_Kn, alpha,
                                    (const double *)bA,cap_Kn,
                                    (const double *)bB,cap_Nn,
                                    beta,(double *)bCb,cap_Nn);
                            } else {
                                double _Complex alpha=CMPLX(1.0,0.0),
                                               beta=CMPLX(0.0,0.0);
                                TENSOR_ZGEMM(CblasRowMajor,CblasNoTrans,
                                    CblasNoTrans, cap_Mn,cap_Nn,cap_Kn,
                                    &alpha,(const double _Complex *)bA,cap_Kn,
                                    (const double _Complex *)bB,cap_Nn,
                                    &beta,(double _Complex *)bCb,cap_Nn);
                            }
#else
                            memset(bCb, 0, cap_bpp);
#endif
                            /* Combine A and B physical dims for boundary check. */
                            size_t bphys[MAX_RANK];
                            const size_t *pa3 = cap_Aphys +
                                (fai_l * cap_tcon + cap_cf) * MAX_RANK;
                            for (int d=0; d<cap_rC; d++)
                                bphys[(size_t)d] = (d < cap_nfA)
                                    ? pa3[(size_t)plan->perm_A[d]]
                                    : cap_tasks[fbi_l].blas_phys[(size_t)d];
                            int is_bnd=0;
                            for (int d=0;d<cap_rC;d++)
                                if (bphys[(size_t)d]<cap_bd[(size_t)d]){is_bnd=1;break;}
                            if (!is_bnd) {
                                if (!cap_cx) {
                                    const double *src=(const double *)bCb;
                                    double *dst=(double *)bCa;
                                    for (size_t f=0;f<cap_tblas;f++)
                                        dst[cap_sidx[f]]+=src[f];
                                } else {
                                    const double _Complex *src=
                                        (const double _Complex *)bCb;
                                    double _Complex *dst=(double _Complex *)bCa;
                                    for (size_t f=0;f<cap_tblas;f++)
                                        dst[cap_sidx[f]]+=src[f];
                                }
                            } else {
                                size_t bc[MAX_RANK];
                                memset(bc,0,(size_t)cap_rC*sizeof(size_t));
                                do {
                                    size_t bf=compute_flat_index(
                                        (size_t)cap_rC,bc,cap_bstr);
                                    if (!cap_cx)
                                        ((double*)bCa)[cap_sidx[bf]]+=
                                            ((const double*)bCb)[bf];
                                    else
                                        ((double _Complex*)bCa)[cap_sidx[bf]]+=
                                            ((const double _Complex*)bCb)[bf];
                                } while (odometer_step((size_t)cap_rC,bc,bphys));
                            }
                        }); /* dispatch_apply */
                    } /* if any_a */

                    /* Recycle slot for cf+2. */
                    if (cf + 2 < total_con)
                        dispatch_async(b_io_q, ^{ load_b(cf + 2, bslot); });
                    cf++;
                } /* while contracted loop (GCD) */

                /* Drain before leaving the gB block. */
                dispatch_sync(b_io_q, ^{});

#else /* !HAS_GCD — serial path */
                {
                    for (size_t cf = 0; cf < total_con && ret == 0; cf++) {
                        int any_a = 0;
                        for (size_t fai_l = 0; fai_l < n_fA_cur; fai_l++)
                            if (A_exist[fai_l * total_con + cf]) { any_a=1; break; }
                        if (!any_a) continue;

                        /* Serial B load for this cf and gB slice. */
                        MBTask *btask = tasks_buf[0];
                        const hsize_t *con_row = con_all + cf * MAX_RANK;
                        for (size_t fbi_l = 0; fbi_l < n_fB_cur; fbi_l++) {
                            size_t fbi = fb_lo + fbi_l;
                            const hsize_t *fb_row = fb_all + fbi * MAX_RANK;
                            hsize_t b_tile[MAX_RANK];
                            memset(b_tile, 0, sizeof(b_tile));
                            for (int d = 0; d < n_con; d++)
                                b_tile[(size_t)plan->perm_B[d]] = con_row[(size_t)d];
                            for (int q = 0; q < n_fB; q++)
                                b_tile[(size_t)plan->perm_B[n_con + q]] =
                                    fb_row[(size_t)q];
                            TileMetadata *mB = registry_get_tile(sh->reg_B, b_tile);
                            btask[fbi_l].fb_exists =
                                (mB && mB->status == TILE_STATUS_ON_DISK) ? 1 : 0;
                            if (btask[fbi_l].fb_exists) {
                                memset(B_raw_buf, 0, bpp);
                                if (read_chunk_typed(dset_B, mB->phys_offset,
                                    B_raw_buf, esz, rank_B,
                                    sh->reg_B->chunk_dims, sh->h5type_mem) < 0) {
                                    fprintf(stderr,
                                        "exec_macroblock_gcd: B read error\n");
                                    ret = -1; break;
                                }
                                prof.bytes_read_B   += bpp;
                                prof.tiles_read_B++;
                                prof.b_bytes_cur_mb += bpp;
                                size_t phys_B[MAX_RANK];
                                for (int d = 0; d < rank_B; d++) {
                                    hsize_t end = mB->phys_offset[(size_t)d]
                                                + sh->reg_B->chunk_dims[(size_t)d];
                                    phys_B[(size_t)d] = (size_t)(
                                        (end > sh->reg_B->global_dims[(size_t)d])
                                        ? sh->reg_B->global_dims[(size_t)d]
                                          - mB->phys_offset[(size_t)d]
                                        : sh->reg_B->chunk_dims[(size_t)d]);
                                }
                                char *bperm = B_perm_buf[0] + fbi_l * bpp;
                                if (perm_is_identity(plan->perm_B, rank_B)) {
                                    memcpy(bperm, B_raw_buf, bpp);
                                } else {
                                    memset(bperm, 0, bpp);
                                    tensor_permute(B_raw_buf, bperm,
                                        (size_t)rank_B, phys_B,
                                        sh->chunk_dims_B_sz, plan->perm_B, esz);
                                }
                                for (int q = 0; q < n_fB; q++)
                                    btask[fbi_l].blas_phys[(size_t)(n_fA + q)] =
                                        phys_B[(size_t)plan->perm_B[n_con + q]];
                            }
                        }
                        if (ret != 0) break;

                        /* Serial BLAS over (fai_l, fbi_l). */
                        for (size_t fai_l = 0; fai_l < n_fA_cur; fai_l++) {
                            if (!A_exist[fai_l * total_con + cf]) continue;
                            const size_t *pa3 = A_phys_cache +
                                (fai_l * total_con + cf) * MAX_RANK;
                            const void *bA = A_cache_base +
                                (fai_l * total_con + cf) * bpp;
                            for (size_t fbi_l = 0; fbi_l < n_fB_cur; fbi_l++) {
                                if (!btask[fbi_l].fb_exists) continue;
                                size_t task_idx = fai_l * n_fB_cur + fbi_l;
                                const void *bB  = B_perm_buf[0] + fbi_l * bpp;
                                void       *bCb = C_blas_base   + task_idx * bpp;
                                void       *bCa = C_accum_base  + task_idx * bpp;
#ifdef TENSOR_ZGEMM
                                if (!is_cplx) {
                                    double alpha=1.0,beta=0.0;
                                    TENSOR_DGEMM(CblasRowMajor,CblasNoTrans,
                                        CblasNoTrans, sh->M_nom,sh->N_nom,
                                        sh->K_nom, alpha,(const double *)bA,
                                        sh->K_nom,(const double *)bB,sh->N_nom,
                                        beta,(double *)bCb,sh->N_nom);
                                } else {
                                    double _Complex alpha=CMPLX(1.0,0.0),
                                                   beta=CMPLX(0.0,0.0);
                                    TENSOR_ZGEMM(CblasRowMajor,CblasNoTrans,
                                        CblasNoTrans, sh->M_nom,sh->N_nom,
                                        sh->K_nom, &alpha,
                                        (const double _Complex *)bA,sh->K_nom,
                                        (const double _Complex *)bB,sh->N_nom,
                                        &beta,(double _Complex *)bCb,sh->N_nom);
                                }
#endif
                                size_t bphys[MAX_RANK];
                                for (int d=0;d<rank_C;d++)
                                    bphys[(size_t)d] = (d < n_fA)
                                        ? pa3[(size_t)plan->perm_A[d]]
                                        : btask[fbi_l].blas_phys[(size_t)d];
                                int is_bnd=0;
                                for (int d=0;d<rank_C;d++)
                                    if (bphys[(size_t)d]<sh->blas_dims[(size_t)d])
                                        { is_bnd=1; break; }
                                if (!is_bnd) {
                                    if (!is_cplx) {
                                        const double *src=(const double *)bCb;
                                        double *dst=(double *)bCa;
                                        for(size_t f=0;f<sh->total_blas;f++)
                                            dst[sh->scatter_idx[f]]+=src[f];
                                    } else {
                                        const double _Complex *src=
                                            (const double _Complex *)bCb;
                                        double _Complex *dst=(double _Complex *)bCa;
                                        for(size_t f=0;f<sh->total_blas;f++)
                                            dst[sh->scatter_idx[f]]+=src[f];
                                    }
                                } else {
                                    size_t bc[MAX_RANK];
                                    memset(bc,0,(size_t)rank_C*sizeof(size_t));
                                    do {
                                        size_t bf=compute_flat_index(
                                            (size_t)rank_C,bc,sh->blas_strides);
                                        if (!is_cplx)
                                            ((double*)bCa)[sh->scatter_idx[bf]]+=
                                                ((const double*)bCb)[bf];
                                        else
                                            ((double _Complex*)bCa)
                                                [sh->scatter_idx[bf]]+=
                                                ((const double _Complex*)bCb)[bf];
                                    } while(odometer_step((size_t)rank_C,bc,bphys));
                                }
                            }
                        }
                    } /* for cf serial */
                }
#endif /* HAS_GCD */

            } /* !use_b_cache contracted loop */

            /* ------------------------------------------------------------ */
            /* Redundancy assertion: B reads this (gA,gB) pair must not     */
            /* exceed total_con × block_fB tiles (the irreducible minimum). */
            /* ------------------------------------------------------------ */
            if (!use_b_cache &&
                prof.b_bytes_cur_mb > prof.theo_b_bytes_mb) {
                size_t excess = prof.b_bytes_cur_mb - prof.theo_b_bytes_mb;
                prof.b_redundant_bytes += excess;
                fprintf(stderr,
                        "  [IOProfiler] REDUNDANT B READ pair (%zu,%zu): "
                        "read %zu tiles, expected %zu (%zu excess)\n",
                        gA, gB,
                        prof.b_bytes_cur_mb / bpp,
                        prof.theo_b_bytes_mb / bpp,
                        excess / bpp);
            }

            /* ------------------------------------------------------------ */
            /* Step 4: Write completed C tiles for this (gA, gB) pair.     */
            /* ------------------------------------------------------------ */
            for (size_t fai_l = 0; fai_l < n_fA_cur && ret == 0; fai_l++) {
                size_t fai = fa_lo + fai_l;
                const hsize_t *fa_row = fa_all + fai * MAX_RANK;
                for (size_t fbi_l = 0; fbi_l < n_fB_cur; fbi_l++) {
                    size_t fbi = fb_lo + fbi_l;
                    const hsize_t *fb_row = fb_all + fbi * MAX_RANK;

                    hsize_t c_tile[MAX_RANK];
                    memset(c_tile, 0, sizeof(c_tile));
                    for (int d = 0; d < rank_C; d++) {
                        int blas = plan->perm_C[(size_t)d];
                        c_tile[(size_t)d] = (blas < n_fA)
                            ? fa_row[(size_t)blas]
                            : fb_row[(size_t)(blas - n_fA)];
                    }

                    TileMetadata *mC = registry_get_tile(sh->reg_C, c_tile);
                    if (mC) {
                        char *C_data = C_accum_base +
                            (fai_l * n_fB_cur + fbi_l) * bpp;
                        if (write_chunk_typed(dset_C, mC->phys_offset,
                                              C_data, esz, rank_C,
                                              sh->reg_C->chunk_dims,
                                              sh->h5type_mem) < 0) {
                            fprintf(stderr,
                                    "exec_macroblock_gcd: write_chunk_typed "
                                    "failed at pair (%zu,%zu)\n", gA, gB);
                            ret = -1; break;
                        }
                        prof.bytes_written_C += bpp;
                        prof.tiles_written_C++;
                    }
                }
            }

            pair_done++;
            printf("\r  Block-pair %zu / %zu  (%.1f%%)",
                   pair_done, P_A * P_B,
                   100.0 * (double)pair_done / (double)(P_A * P_B));
            fflush(stdout);

        } /* for gB */
    } /* for gA */

    printf("\n");

mb_cleanup:
#ifdef HAS_GCD
    /* Drain any in-flight B-load before reading profiler (memory barrier). */
    if (b_io_q) {
        dispatch_sync(b_io_q, ^{});  /* ensures all prof_ptr writes are visible */
    }
#endif

    /* ------------------------------------------------------------------ */
    /* I/O Profiling Report                                                */
    /* ------------------------------------------------------------------ */
    {
        const double GiB = 1024.0 * 1024.0 * 1024.0;

        /* In normal mode, C must never be read back from disk.
         * In accumulate mode, C reads are expected (loading initial values). */
        if (!sh->accumulate && prof.bytes_read_C > 0) {
            fprintf(stderr,
                    "\n[IOProfiler] *** ASSERTION FAILED: "
                    "C read back from disk (%zu bytes = %zu tiles) ***\n"
                    "  Partial accumulators are leaking to NVMe!\n",
                    prof.bytes_read_C,
                    prof.bytes_read_C / (bpp > 0 ? bpp : 1));
        }

        printf("\n");
        printf("=================================================================\n");
        printf("  I/O Profiling Report\n");
        printf("=================================================================\n");
        printf("  Pool capacity         : %zu pages \xc3\x97 %.1f MiB = %.3f GiB\n",
               prof.pool_num_pages,
               (double)prof.bytes_per_page / (1024.0 * 1024.0),
               (double)prof.pool_capacity_bytes / GiB);
        printf("  Block-pairs (P_A*P_B) : %zu  (P_A=%zu, P_B=%zu)\n",
               prof.n_macroblocks, P_A, P_B);
        printf("  block_fA / block_fB   : %zu / %zu tiles\n", block_fA, block_fB);
        printf("  B pre-cache active    : %s\n", use_b_cache ? "YES" : "NO");
        printf("\n");

        /* Table header */
        printf("  %-24s  %12s  %12s  %9s  %s\n",
               "Metric", "Actual", "Theoretical", "Tiles", "Status");
        printf("  %.75s\n",
               "----------------------------------------------------------------------"
               "----------------------------------------------------------------------");

#define PROF_ROW(label, actual_b, theo_b, tiles) \
        do { \
            int _ok = ((actual_b) <= (theo_b)); \
            printf("  %-24s  %8.3f GiB  %8.3f GiB  %9zu  %s\n", \
                   (label), \
                   (double)(actual_b) / GiB, \
                   (double)(theo_b)   / GiB, \
                   (size_t)(tiles), \
                   _ok ? "OK" : "*** EXCESS ***"); \
            if (!_ok) \
                fprintf(stderr, \
                        "[IOProfiler] *** EXCESS: %s actual=%.3f GiB " \
                        "theo=%.3f GiB (+%.3f GiB redundant) ***\n", \
                        (label), \
                        (double)(actual_b) / GiB, \
                        (double)(theo_b)   / GiB, \
                        (double)((actual_b) - (theo_b)) / GiB); \
        } while (0)

        PROF_ROW("Reads  Tensor A",
                 prof.bytes_read_A,   prof.theo_read_A,  prof.tiles_read_A);
        PROF_ROW("Reads  Tensor B",
                 prof.bytes_read_B,   prof.theo_read_B,  prof.tiles_read_B);
        PROF_ROW(sh->accumulate ? "Reads  Tensor C (init)" : "Reads  Tensor C (must=0)",
                 prof.bytes_read_C,   prof.theo_read_C,  0);
        PROF_ROW("Writes Tensor C",
                 prof.bytes_written_C, prof.theo_write_C, prof.tiles_written_C);
#undef PROF_ROW

        printf("\n");
        if (prof.b_redundant_bytes > 0) {
            printf("  *** WARNING: %.3f GiB of redundant B reads detected! ***\n",
                   (double)prof.b_redundant_bytes / GiB);
            printf("  *** NVMe is taking unnecessary wear — check macro-block sizing. ***\n");
        } else {
            printf("  Zero redundant B reads — SSD I/O is optimal.\n");
        }

        if (sh->accumulate) {
            printf("  Accumulate mode: C tiles loaded from disk as initial values.\n");
        } else if (prof.bytes_read_C == 0) {
            printf("  C accumulators stayed in RAM — zero disk reads of C.\n");
        }
        printf("=================================================================\n");

        /* ---------------------------------------------------------------- */
        /* 2D SUMMA vs 1D Baseline comparison table                        */
        /* ---------------------------------------------------------------- */
        {
            size_t size_A = total_fA * total_con * bpp;
            size_t size_B = total_con * total_fB * bpp;
            size_t size_C = total_fA * total_fB * bpp;
            /* 1D baseline: A once, B re-read total_fA times. */
            size_t base_A_ram  = total_con * bpp;
            size_t base_B_ram  = total_fB  * bpp;
            size_t base_C_ram  = total_fB  * bpp;
            size_t base_rd_A   = size_A;
            size_t base_rd_B   = total_fA * size_B;
            /* 2D SUMMA actuals (from profiler). */
            size_t summa_A_ram = block_fA * total_con * bpp;
            size_t summa_B_ram = 2 * block_fB * bpp;
            size_t summa_C_ram = block_fA * block_fB * bpp;
            double total_base  = (double)(base_rd_A + base_rd_B);
            double total_summa = (double)(prof.bytes_read_A + prof.bytes_read_B);

            printf("\n");
            printf("=================================================================\n");
            printf("  2D SUMMA vs 1D Baseline Comparison\n");
            printf("=================================================================\n");
            printf("  %-28s  %12s  %12s\n",
                   "Metric", "1D Baseline", "2D SUMMA");
            printf("  %.65s\n",
                   "--------------------------------------------------------------"
                   "--------------------------------------------------------------");
            printf("  %-28s  %8.3f GiB  %8.3f GiB\n",
                   "A-cache RAM",
                   (double)base_A_ram  / GiB, (double)summa_A_ram / GiB);
            printf("  %-28s  %8.3f GiB  %8.3f GiB\n",
                   "B-buffer RAM (2 slots)",
                   (double)base_B_ram  / GiB, (double)summa_B_ram / GiB);
            printf("  %-28s  %8.3f GiB  %8.3f GiB\n",
                   "C-accum RAM",
                   (double)base_C_ram  / GiB, (double)summa_C_ram / GiB);
            printf("  %.65s\n",
                   "--------------------------------------------------------------"
                   "--------------------------------------------------------------");
            printf("  %-28s  %8.3f GiB  %8.3f GiB  (x%.1f)\n",
                   "Tensor A reads",
                   (double)base_rd_A            / GiB,
                   (double)prof.bytes_read_A     / GiB,
                   (prof.bytes_read_A > 0)
                       ? (double)base_rd_A / (double)prof.bytes_read_A : 0.0);
            printf("  %-28s  %8.3f GiB  %8.3f GiB  (x%.1f)\n",
                   "Tensor B reads",
                   (double)base_rd_B            / GiB,
                   (double)prof.bytes_read_B     / GiB,
                   (prof.bytes_read_B > 0)
                       ? (double)base_rd_B / (double)prof.bytes_read_B : 0.0);
            printf("  %-28s  %8.3f GiB  %8.3f GiB\n",
                   "Tensor C writes",
                   (double)size_C / GiB, (double)prof.bytes_written_C / GiB);
            printf("  %.65s\n",
                   "--------------------------------------------------------------"
                   "--------------------------------------------------------------");
            printf("  %-28s  %8.3f GiB  %8.3f GiB  (%.1fx less I/O)\n",
                   "Total A+B reads",
                   total_base / GiB,
                   total_summa / GiB,
                   total_summa > 0.0 ? total_base / total_summa : 0.0);
            printf("=================================================================\n");
        }
    }

#ifdef HAS_GCD
    /* Release GCD objects (already drained by dispatch_sync above). */
    if (b_io_q) {
        dispatch_release(b_io_q);
    }
    if (b_sem0) dispatch_release(b_sem0);
    if (b_sem1) dispatch_release(b_sem1);
    free(b_io_err);
#endif
    free(A_phys_cache);
    free(fb_all);
    free(fa_all);
    free(tasks_full);
    free(B_full_cache);
    free(con_all);
    free(A_exist);
    free(tasks_buf[1]);
    free(tasks_buf[0]);
    free(C_accum_base);
    free(C_blas_base);
    free(B_perm_buf[1]);
    free(B_perm_buf[0]);
    free(B_raw_buf);
    free(A_perm_buf);
    free(A_cache_base);
    return ret;
}

/* ----------------------------------------------------------------------- */
/* run_contraction_einsum                                                    */
/* ----------------------------------------------------------------------- */

static int run_einsum_impl(const char *expr,
                            const char *file_A, const char *name_A,
                            const char *file_B, const char *name_B,
                            const char *file_C, const char *name_C,
                            int accumulate)
{
    printf("\n=== N-D Einsum Contraction Engine%s ===\n",
           accumulate ? " (accumulate)" : "");
    printf("Expression: %s\n", expr);

    /* ------------------------------------------------------------------ */
    /* 1. Parse the einsum expression.                                     */
    /* ------------------------------------------------------------------ */
    contraction_plan_t plan;
    if (einsum_parse(expr, &plan) < 0) {
        fprintf(stderr,
                "run_contraction_einsum: einsum_parse failed for '%s'\n", expr);
        return -1;
    }
    {
        char buf[512];
        printf("%s\n", einsum_sprint_plan(&plan, buf, sizeof(buf)));
    }

    /* ------------------------------------------------------------------ */
    /* 2. Open A and B.                                                    */
    /* ------------------------------------------------------------------ */
    hid_t fa = engine_fopen_cached(file_A, H5F_ACC_RDONLY, HDF5_CHUNK_CACHE_BYTES);
    hid_t fb = engine_fopen_cached(file_B, H5F_ACC_RDONLY, HDF5_CHUNK_CACHE_BYTES);
    if (fa < 0 || fb < 0) {
        fprintf(stderr,
                "run_contraction_einsum: cannot open '%s' or '%s'\n",
                file_A, file_B);
        if (fa >= 0) H5Fclose(fa);
        if (fb >= 0) H5Fclose(fb);
        return -1;
    }

    hid_t dset_A = dset_open_no_cache(fa, name_A);
    hid_t dset_B = dset_open_no_cache(fb, name_B);
    if (dset_A < 0 || dset_B < 0) {
        fprintf(stderr,
                "run_contraction_einsum: cannot open dataset '%s' or '%s'\n",
                name_A, name_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    /* ------------------------------------------------------------------ */
    /* 3. Read rank and global dims; verify they match the parse result.  */
    /* ------------------------------------------------------------------ */
    hid_t fsp_A = H5Dget_space(dset_A);
    hid_t fsp_B = H5Dget_space(dset_B);
    if (fsp_A < 0 || fsp_B < 0) {
        if (fsp_A >= 0) H5Sclose(fsp_A);
        if (fsp_B >= 0) H5Sclose(fsp_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    int rank_A = H5Sget_simple_extent_ndims(fsp_A);
    int rank_B = H5Sget_simple_extent_ndims(fsp_B);
    hsize_t global_A[MAX_RANK], global_B[MAX_RANK];
    H5Sget_simple_extent_dims(fsp_A, global_A, NULL);
    H5Sget_simple_extent_dims(fsp_B, global_B, NULL);
    H5Sclose(fsp_A);
    H5Sclose(fsp_B);

    if (rank_A != plan.rank_A || rank_B != plan.rank_B) {
        fprintf(stderr,
                "run_contraction_einsum: rank mismatch — "
                "A has rank %d (plan %d), B has rank %d (plan %d)\n",
                rank_A, plan.rank_A, rank_B, plan.rank_B);
        engine_cleanup(NULL, NULL, NULL, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    /* ------------------------------------------------------------------ */
    /* 4. Build registries and scan tiles.                                 */
    /* ------------------------------------------------------------------ */
    TensorRegistry *reg_A = registry_create_from_dset(dset_A);
    TensorRegistry *reg_B = registry_create_from_dset(dset_B);
    if (!reg_A || !reg_B) {
        fprintf(stderr,
                "run_contraction_einsum: registry_create_from_dset failed\n");
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    /* Assert same dtype — mixed-type contraction is unsupported. */
    if (reg_A->dtype != reg_B->dtype) {
        fprintf(stderr,
                "run_contraction_einsum: dtype mismatch — "
                "A is %s, B is %s; mixed-type contraction not supported\n",
                (reg_A->dtype == DTYPE_FP64) ? "FP64" : "COMPLEX128",
                (reg_B->dtype == DTYPE_FP64) ? "FP64" : "COMPLEX128");
        engine_cleanup(NULL, reg_A, reg_B, NULL,
                       dset_A, dset_B, -1, fa, fb, -1);
        return -1;
    }

    tensor_dtype_t dtype        = reg_A->dtype;
    size_t         element_size = (dtype == DTYPE_FP64)
                                  ? sizeof(double)
                                  : sizeof(double _Complex);

    printf("Scanning input tiles...\n");
    long tiles_A = registry_scan_file(dset_A, reg_A);
    long tiles_B = registry_scan_file(dset_B, reg_B);
    printf("  A: %ld tiles   B: %ld tiles\n", tiles_A, tiles_B);

    /* ------------------------------------------------------------------ */
    /* 5. Validate contracted dimension compatibility.                     */
    /* ------------------------------------------------------------------ */
    for (int d = 0; d < plan.n_contracted; d++) {
        int a_dim = plan.perm_A[plan.n_free_A + d];
        int b_dim = plan.perm_B[d];
        if (global_A[(size_t)a_dim] != global_B[(size_t)b_dim]) {
            fprintf(stderr,
                    "run_contraction_einsum: contracted dim mismatch — "
                    "A dim %d = %llu, B dim %d = %llu\n",
                    a_dim, (unsigned long long)global_A[(size_t)a_dim],
                    b_dim, (unsigned long long)global_B[(size_t)b_dim]);
            engine_cleanup(NULL, reg_A, reg_B, NULL,
                           dset_A, dset_B, -1, fa, fb, -1);
            return -1;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6. Derive C global dims and chunk dims from A and B registries.    */
    /*                                                                     */
    /* C dim d maps to blas position perm_C[d]:                           */
    /*   blas < n_free_A  → A dim perm_A[blas]                            */
    /*   blas >= n_free_A → B dim perm_B[n_contracted + (blas-n_free_A)] */
    /* ------------------------------------------------------------------ */
    int     rank_C = plan.rank_C;
    hsize_t global_C[MAX_RANK], chunk_dims_C[MAX_RANK];
    for (int d = 0; d < rank_C; d++) {
        int blas = plan.perm_C[d];
        if (blas < plan.n_free_A) {
            int a_dim = plan.perm_A[blas];
            global_C[(size_t)d]     = global_A[(size_t)a_dim];
            chunk_dims_C[(size_t)d] = reg_A->chunk_dims[(size_t)a_dim];
        } else {
            int b_dim = plan.perm_B[plan.n_contracted + (blas - plan.n_free_A)];
            global_C[(size_t)d]     = global_B[(size_t)b_dim];
            chunk_dims_C[(size_t)d] = reg_B->chunk_dims[(size_t)b_dim];
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7. Open or create output dataset C.                                 */
    /* ------------------------------------------------------------------ */
    hid_t fc = -1, dset_C = -1;
    TensorRegistry *reg_C = NULL;

    if (!accumulate) {
        /* Normal mode: create a fresh C file. */
        if (create_chunked_dataset_einsum(file_C, name_C, rank_C,
                                          global_C, chunk_dims_C, dtype) < 0) {
            fprintf(stderr,
                    "run_contraction_einsum: create_chunked_dataset_einsum "
                    "failed for '%s'\n", file_C);
            engine_cleanup(NULL, reg_A, reg_B, NULL,
                           dset_A, dset_B, -1, fa, fb, -1);
            return -1;
        }
        fc     = engine_fopen_cached(file_C, H5F_ACC_RDWR, HDF5_CHUNK_CACHE_BYTES);
        dset_C = (fc >= 0) ? dset_open_no_cache(fc, name_C) : -1;
        if (fc < 0 || dset_C < 0) {
            fprintf(stderr,
                    "run_contraction_einsum: cannot open output '%s'\n", file_C);
            engine_cleanup(NULL, reg_A, reg_B, NULL,
                           dset_A, dset_B, dset_C, fa, fb, fc);
            return -1;
        }
        reg_C = registry_create_from_dset(dset_C);
        if (!reg_C) {
            fprintf(stderr,
                    "run_contraction_einsum: registry_create_from_dset(C) "
                    "failed\n");
            engine_cleanup(NULL, reg_A, reg_B, NULL,
                           dset_A, dset_B, dset_C, fa, fb, fc);
            return -1;
        }
    } else {
        /* Accumulate mode: open an existing C file and validate it. */
        fc     = engine_fopen_cached(file_C, H5F_ACC_RDWR, HDF5_CHUNK_CACHE_BYTES);
        dset_C = (fc >= 0) ? dset_open_no_cache(fc, name_C) : -1;
        if (fc < 0 || dset_C < 0) {
            fprintf(stderr,
                    "run_contraction_einsum_acc: cannot open existing C '%s'.\n"
                    "  C must exist before calling run_contraction_einsum_acc.\n",
                    file_C);
            engine_cleanup(NULL, reg_A, reg_B, NULL,
                           dset_A, dset_B, dset_C, fa, fb, fc);
            return -1;
        }
        reg_C = registry_create_from_dset(dset_C);
        if (!reg_C) {
            fprintf(stderr,
                    "run_contraction_einsum_acc: registry_create_from_dset(C) "
                    "failed\n");
            engine_cleanup(NULL, reg_A, reg_B, NULL,
                           dset_A, dset_B, dset_C, fa, fb, fc);
            return -1;
        }
        /* Validate shape compatibility. */
        if (reg_C->rank != rank_C) {
            fprintf(stderr,
                    "run_contraction_einsum_acc: C rank mismatch — "
                    "file has rank %d, contraction expects %d\n",
                    reg_C->rank, rank_C);
            engine_cleanup(NULL, reg_A, reg_B, reg_C,
                           dset_A, dset_B, dset_C, fa, fb, fc);
            return -1;
        }
        for (int d = 0; d < rank_C; d++) {
            if (reg_C->global_dims[(size_t)d] != global_C[(size_t)d]) {
                fprintf(stderr,
                        "run_contraction_einsum_acc: C dim %d mismatch — "
                        "file=%llu expected=%llu\n",
                        d,
                        (unsigned long long)reg_C->global_dims[(size_t)d],
                        (unsigned long long)global_C[(size_t)d]);
                engine_cleanup(NULL, reg_A, reg_B, reg_C,
                               dset_A, dset_B, dset_C, fa, fb, fc);
                return -1;
            }
        }
        if (reg_C->dtype != dtype) {
            fprintf(stderr,
                    "run_contraction_einsum_acc: C dtype mismatch — "
                    "file=%s, contraction expects %s\n",
                    (reg_C->dtype == DTYPE_FP64) ? "FP64" : "COMPLEX128",
                    (dtype          == DTYPE_FP64) ? "FP64" : "COMPLEX128");
            engine_cleanup(NULL, reg_A, reg_B, reg_C,
                           dset_A, dset_B, dset_C, fa, fb, fc);
            return -1;
        }
        /* Scan existing tiles so exec_macroblock_gcd knows which are on disk. */
        long tiles_C = registry_scan_file(dset_C, reg_C);
        printf("  C: %ld existing tiles (accumulate mode)\n", tiles_C);
    }

    /* ------------------------------------------------------------------ */
    /* 8. HDF5 memory type for typed I/O.                                 */
    /* ------------------------------------------------------------------ */
    hid_t h5type_mem;
    if (dtype == DTYPE_FP64) {
        h5type_mem = H5T_NATIVE_DOUBLE;
    } else {
        h5type_mem = create_h5_complex_type();
        if (h5type_mem < 0) {
            fprintf(stderr,
                    "run_contraction_einsum: create_h5_complex_type "
                    "failed\n");
            engine_cleanup(NULL, reg_A, reg_B, reg_C,
                           dset_A, dset_B, dset_C, fa, fb, fc);
            return -1;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 9. Initialise memory pool (80% RAM, min 8 pages).                  */
    /*                                                                     */
    /* Per output tile we hold:                                            */
    /*   2×A + 2×B  (double-buffer slots)                                 */
    /*   1×buf_C      (accumulator)                                        */
    /*   1×buf_A_perm (permuted A scratchpad)                              */
    /*   1×buf_B_perm (permuted B scratchpad)                              */
    /*   1×buf_C_blas (BLAS output scratchpad)                             */
    /* ------------------------------------------------------------------ */

    /* Number of heterogeneous workers — used here for pool headroom sizing
     * and again in section 13 when spawning threads. */
#ifdef __APPLE__
#  define N_WORKERS 2
#else
#  define N_WORKERS 1
#endif

    size_t ram = query_physical_ram();

    size_t elems_A = 1, elems_B = 1, elems_C = 1;
    for (int d = 0; d < rank_A; d++) elems_A *= (size_t)reg_A->chunk_dims[(size_t)d];
    for (int d = 0; d < rank_B; d++) elems_B *= (size_t)reg_B->chunk_dims[(size_t)d];
    for (int d = 0; d < rank_C; d++) elems_C *= (size_t)reg_C->chunk_dims[(size_t)d];

    size_t elems_per_page = elems_A;
    if (elems_B > elems_per_page) elems_per_page = elems_B;
    if (elems_C > elems_per_page) elems_per_page = elems_C;

    size_t bytes_per_page = elems_per_page * element_size;

    /* Align page size to NVMe hardware page boundary.
     * Each pool page will be 16 KB-aligned so OS DMA transfers land
     * directly into our buffers without read-amplification. */
    bytes_per_page = (bytes_per_page + NVME_PAGE_BYTES - 1)
                     & ~(NVME_PAGE_BYTES - 1);

    /* Primary pool budget: 80% of physical RAM. */
    size_t pool_bytes = (size_t)((double)ram * 0.8);

    /*
     * Cap against the actual data that needs to be in flight.
     * Allocating 51 GB is wasteful if A+B together fit in 2 GB.
     * Each worker needs overhead for scratch pages, buf_C, and the
     * write-queue in-flight tile; add N_WORKERS×12 pages of headroom.
     */
    if (tiles_A > 0 && tiles_B > 0) {
        size_t overhead   = (size_t)N_WORKERS * 12;
        size_t max_useful = ((size_t)tiles_A + (size_t)tiles_B + overhead)
                            * bytes_per_page;
        if (max_useful < pool_bytes) {
            pool_bytes = max_useful;
            printf("Pool capped to tensor data size: %.2f GB\n",
                   (double)pool_bytes / (1024.0 * 1024.0 * 1024.0));
        }
    }

    /* Allow TENSOR_POOL_MB env var to further cap the pool (for testing). */
    const char *env_mb = getenv("TENSOR_POOL_MB");
    if (env_mb) {
        size_t cap = (size_t)strtoul(env_mb, NULL, 10) * 1024UL * 1024UL;
        if (cap > 0 && cap < pool_bytes) pool_bytes = cap;
    }

    size_t num_pages      = pool_bytes / bytes_per_page;

    /* Capacity of the async write queue (tiles in-flight to HDF5 writer).
     * Each in-flight tile holds one pool page, so this bounds extra overhead. */
#define WQ_CAP 4

    /* Minimum pages: 3 scratch + 1 current buf_C + WQ_CAP write queue + 4 ring */
    if (num_pages < (size_t)(3 + 1 + WQ_CAP + 4)) {
        fprintf(stderr,
                "run_contraction_einsum: RAM too small for %d pages "
                "(need %zu bytes)\n", 3 + 1 + WQ_CAP + 4,
                (size_t)(3 + 1 + WQ_CAP + 4) * bytes_per_page);
        if (dtype != DTYPE_FP64) H5Tclose(h5type_mem);
        engine_cleanup(NULL, reg_A, reg_B, reg_C,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    BufferPool *pool = pool_create(num_pages, bytes_per_page);
    if (!pool) {
        fprintf(stderr, "run_contraction_einsum: pool_create failed\n");
        if (dtype != DTYPE_FP64) H5Tclose(h5type_mem);
        engine_cleanup(NULL, reg_A, reg_B, reg_C,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }

    printf("dtype: %s  element_size: %zu  RAM: %.1f GB\n"
           "Pool: %zu pages \xc3\x97 %zu elems = %.1f GB\n",
           (dtype == DTYPE_FP64) ? "FP64" : "COMPLEX128",
           element_size,
           (double)ram / (1024.0 * 1024.0 * 1024.0),
           num_pages, elems_per_page,
           (double)(num_pages * bytes_per_page) / (1024.0 * 1024.0 * 1024.0));

    /* ------------------------------------------------------------------ */
    /* 10. Precompute nominal BLAS dimensions.                             */
    /* ------------------------------------------------------------------ */
    int M_nom = 1, K_nom = 1, N_nom = 1;
    for (int p = 0; p < plan.n_free_A; p++)
        M_nom *= (int)reg_A->chunk_dims[(size_t)plan.perm_A[p]];
    for (int d = 0; d < plan.n_contracted; d++)
        K_nom *= (int)reg_A->chunk_dims[(size_t)plan.perm_A[plan.n_free_A + d]];
    for (int q = 0; q < plan.n_free_B; q++)
        N_nom *= (int)reg_B->chunk_dims[(size_t)plan.perm_B[plan.n_contracted + q]];

    printf("BLAS: M=%d  K=%d  N=%d\n", M_nom, K_nom, N_nom);
#ifdef USE_ACCELERATE
    printf("Kernel: cblas_dgemm/zgemm (Apple Accelerate/AMX)\n");
#elif defined(USE_MKL)
    printf("Kernel: cblas_dgemm/zgemm (Intel MKL)\n");
#elif defined(HAVE_CBLAS)
    printf("Kernel: cblas_dgemm/zgemm (OpenBLAS)\n");
#else
    printf("Kernel: fallback dense loop\n");
#endif

    /* Nominal dims for the blas output buffer (rank_C-dimensional). */
    size_t blas_dims[MAX_RANK];
    for (int p = 0; p < plan.n_free_A; p++)
        blas_dims[(size_t)p] = (size_t)reg_A->chunk_dims[(size_t)plan.perm_A[p]];
    for (int q = 0; q < plan.n_free_B; q++)
        blas_dims[(size_t)(plan.n_free_A + q)] =
            (size_t)reg_B->chunk_dims[(size_t)plan.perm_B[plan.n_contracted + q]];

    /* Row-major strides for blas output and for C tile. */
    size_t blas_strides[MAX_RANK], c_strides[MAX_RANK];
    compute_strides((size_t)rank_C, blas_dims, blas_strides);
    {
        size_t cdims[MAX_RANK];
        for (int d = 0; d < rank_C; d++)
            cdims[(size_t)d] = (size_t)reg_C->chunk_dims[(size_t)d];
        compute_strides((size_t)rank_C, cdims, c_strides);
    }

    /* Tile counts along contracted dimensions. */
    hsize_t contracted_grid[MAX_RANK];
    for (int d = 0; d < plan.n_contracted; d++)
        contracted_grid[(size_t)d] =
            reg_A->grid_dims[(size_t)plan.perm_A[plan.n_free_A + d]];

    /* C grid size for the outer odometer. */
    size_t c_grid_sz[MAX_RANK];
    for (int d = 0; d < rank_C; d++)
        c_grid_sz[(size_t)d] = (size_t)reg_C->grid_dims[(size_t)d];

    printf("C grid: ");
    for (int d = 0; d < rank_C; d++)
        printf("%s%zu", (d ? "\xc3\x97" : ""), c_grid_sz[(size_t)d]);
    printf("\n");

    /* ------------------------------------------------------------------ */
    /* 11a. Precompute scatter index table.                                */
    /*                                                                     */
    /* scatter_idx[blas_flat] = c_flat  maps every position in the BLAS   */
    /* output buffer (indexed in [free_A | free_B] order) to the          */
    /* corresponding position in the C tile buffer (indexed in C's native  */
    /* storage order).  Built once for the nominal chunk dims; used for    */
    /* every tile in the outer loop, giving O(1) per-element scatter.      */
    /* ------------------------------------------------------------------ */
    size_t total_blas = (size_t)M_nom * (size_t)N_nom;
    size_t *scatter_idx = (size_t *)malloc(total_blas * sizeof(size_t));
    if (!scatter_idx) {
        fprintf(stderr, "run_contraction_einsum: scatter_idx malloc failed\n");
        if (dtype != DTYPE_FP64) H5Tclose(h5type_mem);
        engine_cleanup(pool, reg_A, reg_B, reg_C,
                       dset_A, dset_B, dset_C, fa, fb, fc);
        return -1;
    }
    {
        /* Nominal blas extents (same as blas_dims). */
        size_t blas_nom[MAX_RANK];
        for (int p = 0; p < plan.n_free_A; p++)
            blas_nom[(size_t)p] =
                (size_t)reg_A->chunk_dims[(size_t)plan.perm_A[p]];
        for (int q = 0; q < plan.n_free_B; q++)
            blas_nom[(size_t)(plan.n_free_A + q)] =
                (size_t)reg_B->chunk_dims[(size_t)plan.perm_B[plan.n_contracted + q]];

        size_t bc[MAX_RANK];
        memset(bc, 0, (size_t)rank_C * sizeof(size_t));
        do {
            size_t bf = compute_flat_index((size_t)rank_C, bc, blas_strides);
            size_t cc[MAX_RANK];
            for (int d = 0; d < rank_C; d++)
                cc[(size_t)d] = bc[(size_t)plan.perm_C[d]];
            scatter_idx[bf] = compute_flat_index((size_t)rank_C, cc, c_strides);
        } while (odometer_step((size_t)rank_C, bc, blas_nom));
    }

    /* ------------------------------------------------------------------ */
    /* 11. Fill ContractionShared (read-only config for all workers).     */
    /* ------------------------------------------------------------------ */
    ContractionShared sh;
    memset(&sh, 0, sizeof(sh));
    sh.file_A = file_A; sh.name_A = name_A;
    sh.file_B = file_B; sh.name_B = name_B;
    sh.dset_C = dset_C;
    sh.reg_A  = reg_A;  sh.reg_B  = reg_B;  sh.reg_C  = reg_C;
    sh.plan   = plan;
    sh.rank_A = rank_A; sh.rank_B = rank_B; sh.rank_C = rank_C;
    sh.dtype  = dtype;
    sh.element_size = element_size;
    sh.h5type_mem   = h5type_mem;
    sh.bytes_per_page = bytes_per_page;
    sh.M_nom = M_nom; sh.N_nom = N_nom; sh.K_nom = K_nom;
    sh.total_blas = total_blas;
    sh.scatter_idx = scatter_idx;
    for (int d = 0; d < rank_C; d++)  sh.c_grid_sz[d]       = c_grid_sz[d];
    for (int d = 0; d < plan.n_contracted; d++) sh.contracted_grid[d] = contracted_grid[d];
    for (int d = 0; d < rank_C; d++)  sh.blas_dims[d]       = blas_dims[d];
    for (int d = 0; d < rank_C; d++)  sh.blas_strides[d]    = blas_strides[d];
    for (int d = 0; d < rank_A; d++)  sh.chunk_dims_A_sz[d] = (size_t)reg_A->chunk_dims[d];
    for (int d = 0; d < rank_B; d++)  sh.chunk_dims_B_sz[d] = (size_t)reg_B->chunk_dims[d];
    sh.pool_capacity_bytes = num_pages * bytes_per_page;
    sh.pool_num_pages      = num_pages;
    sh.accumulate          = accumulate;

    /* ------------------------------------------------------------------ */
    /* 12-13. Execute: A-pinning macro-block loop + GCD parallel BLAS.   */
    /* ------------------------------------------------------------------ */
    int ret = exec_macroblock_gcd(&sh, dset_A, dset_B, dset_C);
    printf("\nN-D contraction complete.\n");

    pool_destroy(pool);

    free(scatter_idx);
    if (dtype != DTYPE_FP64) H5Tclose(h5type_mem);
    /* Pass NULL for pool since we destroyed it above. */
    engine_cleanup(NULL, reg_A, reg_B, reg_C,
                   dset_A, dset_B, dset_C, fa, fb, fc);
    return ret;
}

/* ----------------------------------------------------------------------- */
/* Public API wrappers                                                       */
/* ----------------------------------------------------------------------- */

int run_contraction_einsum(const char *expr,
                            const char *file_A, const char *name_A,
                            const char *file_B, const char *name_B,
                            const char *file_C, const char *name_C)
{
    return run_einsum_impl(expr, file_A, name_A, file_B, name_B,
                           file_C, name_C, /*accumulate=*/0);
}

int run_contraction_einsum_acc(const char *expr,
                               const char *file_A, const char *name_A,
                               const char *file_B, const char *name_B,
                               const char *file_C, const char *name_C)
{
    return run_einsum_impl(expr, file_A, name_A, file_B, name_B,
                           file_C, name_C, /*accumulate=*/1);
}
