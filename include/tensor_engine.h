/*
 * tensor_engine.h — Public C11 API for the Out-of-Core Tensor Contraction Engine
 *
 * This header is the sole interface a caller needs. All internal details
 * (BufferPool, HDF5 I/O, GCD dispatch, double-buffering) are hidden behind
 * the opaque tensor_engine_t pointer.
 *
 * Quick start:
 *
 *   tensor_engine_config_t cfg = {0};   // all-zero → auto-tune
 *   tensor_engine_t *eng = tensor_engine_init(&cfg);
 *
 *   int rc = tensor_engine_contract(eng, "ijab,akbl->klji",
 *                                   "A.h5", "B.h5", "C.h5");
 *   if (rc != TENSOR_ENGINE_OK) { ... }
 *
 *   tensor_engine_free(eng);
 *
 * HDF5 dataset convention:
 *   Input and output files are expected to contain a single dataset named
 *   "tensor".  Use the lower-level run_contraction_einsum() from engine.h
 *   if you need a different dataset name.
 *
 * Thread safety:
 *   A single tensor_engine_t must not be shared across threads.  Each thread
 *   should create its own instance via tensor_engine_init().
 */

#ifndef TENSOR_ENGINE_H
#define TENSOR_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Error codes
 * All functions that can fail return one of these values.
 * TENSOR_ENGINE_OK is guaranteed to be 0; all error codes are negative.
 * -----------------------------------------------------------------------*/

/** Success. */
#define TENSOR_ENGINE_OK          0

/** Input file not found, unreadable, or output file cannot be created. */
#define TENSOR_ENGINE_ERR_FILE   -1

/** Tensor dimensions are incompatible with the requested contraction. */
#define TENSOR_ENGINE_ERR_DIMS   -2

/** Malformed or unsupported einsum expression string. */
#define TENSOR_ENGINE_ERR_EXPR   -3

/** Memory allocation failed (pool too small or system out of memory). */
#define TENSOR_ENGINE_ERR_MEM    -4

/** Unspecified internal error (I/O, HDF5, BLAS). */
#define TENSOR_ENGINE_ERR        -5

/* -------------------------------------------------------------------------
 * Configuration
 * -----------------------------------------------------------------------*/

/**
 * tensor_engine_config_t — initialisation parameters.
 *
 * Set any field to 0 to accept the built-in default.
 */
typedef struct {
    /**
     * Maximum buffer-pool size in MiB.
     *
     * The pool is a fixed slab of RAM used to stage tensor tiles during the
     * contraction.  Larger values reduce NVMe reads at the cost of RAM.
     *
     * Default (0): 80 % of physical RAM, capped so the OS is not starved.
     */
    size_t pool_mb;

    /**
     * Target tile size in bytes.
     *
     * Each tile is an isotropic chunk whose byte count is as close to this
     * value as possible (n-th root of byte budget divided by element size).
     *
     * Default (0): 16 MiB — matches the AMX register file and NVMe page
     *              granularity on Apple Silicon for optimal BLAS batching.
     */
    size_t tile_bytes;
} tensor_engine_config_t;

/* -------------------------------------------------------------------------
 * Opaque engine handle
 * -----------------------------------------------------------------------*/

/** Opaque engine handle.  Allocate with tensor_engine_init(). */
typedef struct tensor_engine tensor_engine_t;

/* -------------------------------------------------------------------------
 * Lifecycle
 * -----------------------------------------------------------------------*/

/**
 * tensor_engine_init — create and configure an engine instance.
 *
 * @param cfg  Pointer to a filled tensor_engine_config_t, or NULL to use
 *             all defaults (equivalent to passing an all-zero struct).
 *
 * @return  A newly allocated engine handle, or NULL if memory allocation
 *          failed.  The caller owns the handle and must release it with
 *          tensor_engine_free().
 */
tensor_engine_t *tensor_engine_init(const tensor_engine_config_t *cfg);

/**
 * tensor_engine_free — destroy an engine instance and release all resources.
 *
 * Safe to call with NULL (no-op).
 */
void tensor_engine_free(tensor_engine_t *engine);

/* -------------------------------------------------------------------------
 * Contraction
 * -----------------------------------------------------------------------*/

/**
 * tensor_engine_contract — perform an out-of-core N-D tensor contraction.
 *
 * Reads tensors A and B from @p file_A and @p file_B respectively (each
 * must contain a dataset named "tensor"), computes their contraction as
 * specified by @p einsum_expr, and writes the result to @p file_C.
 *
 * Supported dtypes: FP64 (double) and COMPLEX128 (double _Complex).
 * The dtype is inferred from the A dataset; B and C must match.
 *
 * The contraction is fully out-of-core: tile pairs are streamed from NVMe
 * through a double-buffer pipeline that overlaps I/O with BLAS execution.
 * Block-sparse tiles (not allocated on disk) are skipped without any I/O.
 *
 * @param engine       Engine handle from tensor_engine_init().
 * @param einsum_expr  Null-terminated Einstein summation string,
 *                     e.g. "ij,jk->ik" or "ijab,akbl->klji".
 * @param file_A       Path to the HDF5 file containing operand A.
 * @param file_B       Path to the HDF5 file containing operand B.
 * @param file_C       Path for the output HDF5 file (created or overwritten).
 *
 * @return  TENSOR_ENGINE_OK (0) on success, or a negative error code.
 *          On failure @p file_C may be in a partial state and should be
 *          discarded.
 */
int tensor_engine_contract(tensor_engine_t *engine,
                           const char      *einsum_expr,
                           const char      *file_A,
                           const char      *file_B,
                           const char      *file_C);

/**
 * tensor_engine_accumulate — accumulating out-of-core N-D tensor contraction.
 *
 * Computes C += A*B using the einsum notation, accumulating into an
 * existing @p file_C rather than overwriting it.  All parameters have the
 * same meaning as tensor_engine_contract().
 *
 * @p file_C must already exist and contain a dataset named "tensor" with
 * shape, rank, and dtype compatible with the contraction result.  Each
 * output tile is read from disk before accumulation, so the final C holds
 * the element-wise sum of its previous value and the new contraction result.
 *
 * Typical use — multi-term contraction:
 * @code
 *   // First term: create C.h5 from scratch.
 *   tensor_engine_contract(eng, "ij,jk->ik", "A1.h5", "B1.h5", "C.h5");
 *   // Subsequent terms: accumulate into the existing C.h5.
 *   tensor_engine_accumulate(eng, "ij,jk->ik", "A2.h5", "B2.h5", "C.h5");
 *   tensor_engine_accumulate(eng, "ij,jk->ik", "A3.h5", "B3.h5", "C.h5");
 * @endcode
 *
 * @return  TENSOR_ENGINE_OK (0) on success, or a negative error code.
 */
int tensor_engine_accumulate(tensor_engine_t *engine,
                             const char      *einsum_expr,
                             const char      *file_A,
                             const char      *file_B,
                             const char      *file_C);

/**
 * tensor_engine_strerror — human-readable description of an error code.
 *
 * The returned string is a string literal; do not free it.
 * Returns "unknown error" for unrecognised codes.
 */
const char *tensor_engine_strerror(int err);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_ENGINE_H */
