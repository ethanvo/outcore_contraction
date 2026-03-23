#ifndef ENGINE_H
#define ENGINE_H

#include <stddef.h>

/*
 * Return the total physical RAM installed in this machine (bytes).
 *
 * Tries sysconf(_SC_PHYS_PAGES * _SC_PAGE_SIZE) on POSIX and
 * sysctlbyname("hw.memsize") on macOS.  Falls back to 512 MB if both
 * syscalls fail or are unavailable.
 */
size_t query_physical_ram(void);

/*
 * Contract A(i,k) * B(k,j) -> C(i,j) for rank-2 chunked HDF5 tensors.
 *
 * Pool size is determined automatically from physical RAM (80%).
 * Block-sparsity is exploited: tile pairs where either operand has
 * TILE_STATUS_NULL are skipped without any I/O.
 *
 * I/O (HDF5 reads/writes) and compute (cblas_dgemm or fallback kernel)
 * run concurrently via a double-buffer pipeline: the I/O thread prefetches
 * the next tile pair while the compute thread multiplies the current one.
 *
 * Returns 0 on success, -1 on any error.  On failure the output file may
 * be in a partial state.
 */
int run_contraction(const char *file_A, const char *name_A,
                    const char *file_B, const char *name_B,
                    const char *file_C, const char *name_C);

/*
 * Rank-4 Einstein contraction: C(k,l,j,i) = sum_{a,b} A(i,j,a,b) * B(a,k,b,l)
 *
 * HDF5 layout expectations:
 *   file_A / name_A  — rank-4 dataset with dimension order (i, j, a, b)
 *   file_B / name_B  — rank-4 dataset with dimension order (a, k, b, l)
 *   file_C / name_C  — output file; created with shape (k, l, j, i)
 *
 * Compatibility check: global_A[2] must equal global_B[0]  (a-dim)
 *                      global_A[3] must equal global_B[2]  (b-dim)
 *
 * Execution model:
 *   Outer 4-D grid (k,l,j,i) drives the output tile loop.
 *   For each output tile the I/O thread double-buffers (a,b) contracted tile
 *   pairs; the compute thread permutes B → (a*b)×(k*l), calls cblas_dgemm,
 *   then scatter-accumulates the result into the (k,l,j,i)-layout C buffer.
 *   All tensors use 1-D flat indexing with manual stride arithmetic.
 *
 * Pool requires at least 7 pages:
 *   2×A + 2×B (double-buffer slots) + 1×C + 1×B_perm scratchpad
 *   + 1×C_blas scratchpad.
 *
 * Returns 0 on success, -1 on error.
 */
int run_contraction_4d(const char *file_A, const char *name_A,
                       const char *file_B, const char *name_B,
                       const char *file_C, const char *name_C);

/*
 * Generic N-D tensor contraction driven by an einsum expression string.
 *
 * expr     : null-terminated einsum string, e.g. "ijab,akbl->klji"
 * file_A/B : paths to existing HDF5 input files
 * name_A/B : dataset names inside those files
 * file_C   : output file path (created / overwritten)
 * name_C   : dataset name to create in file_C
 *
 * Supports DTYPE_FP64 and DTYPE_COMPLEX128; dtype is read from the A
 * dataset and asserted equal for B.  C is created with the same dtype.
 *
 * Uses the same double-buffer pthreads pipeline as run_contraction /
 * run_contraction_4d, generalised to arbitrary rank via odometer loops.
 *
 * Returns 0 on success, -1 on error.
 */
int run_contraction_einsum(const char *expr,
                            const char *file_A, const char *name_A,
                            const char *file_B, const char *name_B,
                            const char *file_C, const char *name_C);

#endif /* ENGINE_H */
