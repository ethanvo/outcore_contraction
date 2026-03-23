/*
 * metal_backend.h — Pure-C API for the Metal/MPS tile compute backend.
 *
 * Implemented in src/metal_backend.m (Objective-C).  On non-Apple platforms
 * the stub functions in that file always return NULL / -1.
 *
 * metal_ctx_init() compiles the embedded Metal shader at start-up and
 * pre-allocates three MTLBuffers of max_tile_bytes each for A, B, and C
 * scratch — no per-call allocations in the hot path.
 */

#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MetalCtx MetalCtx;

/*
 * Allocate Metal device + command queue, compile the GEMM shader, and
 * pre-allocate scratch MTLBuffers of max_tile_bytes bytes each.
 * Returns NULL if Metal is unavailable or initialisation fails.
 */
MetalCtx *metal_ctx_init(size_t max_tile_bytes);

/* Free all Metal resources (device, queue, buffers, pipeline states). */
void metal_ctx_destroy(MetalCtx *ctx);

/*
 * Compute C = A * B on the GPU.
 *
 *   A : M × K  (row-major, leading dimension lda, element_size bytes each)
 *   B : K × N  (row-major, leading dimension ldb)
 *   C : M × N  (row-major, leading dimension ldc) — OVERWRITTEN (beta = 0)
 *
 * is_complex = 0 → real double GEMM (dgemm)
 * is_complex = 1 → complex-128 GEMM (zgemm, elements are double[2])
 *
 * Returns 0 on success, -1 on error.
 */
int metal_compute_tile_task(MetalCtx   *ctx,
                             const void *A,
                             const void *B,
                             void       *C,
                             int         M,
                             int         K,
                             int         N,
                             int         lda,
                             int         ldb,
                             int         ldc,
                             int         is_complex);

#ifdef __cplusplus
}
#endif

#endif /* METAL_BACKEND_H */
