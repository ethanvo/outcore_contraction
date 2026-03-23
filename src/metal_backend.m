/*
 * metal_backend.m — Objective-C implementation of the Metal tile GEMM backend.
 *
 * The Metal shader is embedded as a string and compiled at runtime so no
 * separate .metallib is needed.  Three scratch MTLBuffers (A, B, C) are
 * pre-allocated in metal_ctx_init; metal_compute_tile_task does only memcpy
 * + GPU dispatch in the hot path.
 *
 * Double-precision support: the shader uses the Metal `double` type.
 * Apple Silicon GPU cores support float64 arithmetic natively.
 */

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Embedded Metal shader source.
 *
 * dgemm_kernel : C[m][n]  = sum_k  A[m*lda+k] * B[k*ldb+n]          (real)
 * zgemm_kernel : C[m][n] += sum_k  A[m*lda+k] * B[k*ldb+n]         (cplx128)
 *
 * Elements for zgemm are stored as interleaved (re, im) pairs, matching
 * the in-memory layout of C99 double _Complex / C11 _Complex double.
 * ------------------------------------------------------------------------- */
/*
 * Shaders use float / float2 (universally supported on Apple Silicon).
 * The host converts double ↔ float around each dispatch.
 */
static const char *k_shader_src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "kernel void sgemm_kernel(\n"
    "    device const float *A  [[buffer(0)]],\n"
    "    device const float *B  [[buffer(1)]],\n"
    "    device       float *C  [[buffer(2)]],\n"
    "    constant int       &M  [[buffer(3)]],\n"
    "    constant int       &K  [[buffer(4)]],\n"
    "    constant int       &N  [[buffer(5)]],\n"
    "    constant int       &lda [[buffer(6)]],\n"
    "    constant int       &ldb [[buffer(7)]],\n"
    "    constant int       &ldc [[buffer(8)]],\n"
    "    uint2 pos [[thread_position_in_grid]])\n"
    "{\n"
    "    int m = (int)pos.y, n = (int)pos.x;\n"
    "    if (m >= M || n >= N) return;\n"
    "    float acc = 0.0f;\n"
    "    for (int k = 0; k < K; k++)\n"
    "        acc += A[m * lda + k] * B[k * ldb + n];\n"
    "    C[m * ldc + n] = acc;\n"
    "}\n"
    "\n"
    "kernel void cgemm_kernel(\n"
    "    device const float2 *A  [[buffer(0)]],\n"
    "    device const float2 *B  [[buffer(1)]],\n"
    "    device       float2 *C  [[buffer(2)]],\n"
    "    constant int        &M  [[buffer(3)]],\n"
    "    constant int        &K  [[buffer(4)]],\n"
    "    constant int        &N  [[buffer(5)]],\n"
    "    constant int        &lda [[buffer(6)]],\n"
    "    constant int        &ldb [[buffer(7)]],\n"
    "    constant int        &ldc [[buffer(8)]],\n"
    "    uint2 pos [[thread_position_in_grid]])\n"
    "{\n"
    "    int m = (int)pos.y, n = (int)pos.x;\n"
    "    if (m >= M || n >= N) return;\n"
    "    float2 acc = {0.0f, 0.0f};\n"
    "    for (int k = 0; k < K; k++) {\n"
    "        float2 a = A[m * lda + k];\n"
    "        float2 b = B[k * ldb + n];\n"
    "        acc.x += a.x * b.x - a.y * b.y;\n"
    "        acc.y += a.x * b.y + a.y * b.x;\n"
    "    }\n"
    "    C[m * ldc + n] = acc;\n"
    "}\n";

/* -------------------------------------------------------------------------
 * MetalCtx
 * ------------------------------------------------------------------------- */
struct MetalCtx {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pso_sgemm;  /* real float32 */
    id<MTLComputePipelineState> pso_cgemm;  /* complex float32 */
    id<MTLBuffer>               buf_A;
    id<MTLBuffer>               buf_B;
    id<MTLBuffer>               buf_C;
    size_t                      scratch_cap; /* bytes per scratch buffer (double-sized) */
    /* CPU-side float conversion buffers; each holds scratch_cap/2 bytes. */
    float                      *conv_A;
    float                      *conv_B;
    float                      *conv_C;
};

MetalCtx *metal_ctx_init(size_t max_tile_bytes)
{
    MetalCtx *ctx = (MetalCtx *)calloc(1, sizeof(MetalCtx));
    if (!ctx) return NULL;

    @autoreleasepool {
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "metal_ctx_init: MTLCreateSystemDefaultDevice failed\n");
            free(ctx);
            return NULL;
        }

        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            fprintf(stderr, "metal_ctx_init: newCommandQueue failed\n");
            free(ctx);
            return NULL;
        }

        /* Compile embedded shader. */
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:k_shader_src];
        id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src
                                                       options:nil
                                                         error:&err];
        if (!lib) {
            fprintf(stderr, "metal_ctx_init: shader compile error: %s\n",
                    [[err localizedDescription] UTF8String]);
            free(ctx);
            return NULL;
        }

        id<MTLFunction> fn_s = [lib newFunctionWithName:@"sgemm_kernel"];
        id<MTLFunction> fn_c = [lib newFunctionWithName:@"cgemm_kernel"];
        if (!fn_s || !fn_c) {
            fprintf(stderr, "metal_ctx_init: kernel function not found\n");
            free(ctx);
            return NULL;
        }

        ctx->pso_sgemm = [ctx->device newComputePipelineStateWithFunction:fn_s
                                                                    error:&err];
        ctx->pso_cgemm = [ctx->device newComputePipelineStateWithFunction:fn_c
                                                                    error:&err];
        if (!ctx->pso_sgemm || !ctx->pso_cgemm) {
            fprintf(stderr, "metal_ctx_init: pipeline state creation failed\n");
            free(ctx);
            return NULL;
        }

        /*
         * Pre-allocate shared-memory scratch buffers.
         * Float tiles need max_tile_bytes/2; allocate full size for headroom.
         */
        MTLResourceOptions opts = MTLResourceStorageModeShared;
        ctx->buf_A = [ctx->device newBufferWithLength:max_tile_bytes options:opts];
        ctx->buf_B = [ctx->device newBufferWithLength:max_tile_bytes options:opts];
        ctx->buf_C = [ctx->device newBufferWithLength:max_tile_bytes options:opts];
        if (!ctx->buf_A || !ctx->buf_B || !ctx->buf_C) {
            fprintf(stderr, "metal_ctx_init: scratch buffer allocation failed\n");
            free(ctx);
            return NULL;
        }
        ctx->scratch_cap = max_tile_bytes;

        /* CPU conversion buffers: double → float before dispatch, float → double after. */
        ctx->conv_A = (float *)malloc(max_tile_bytes / 2);
        ctx->conv_B = (float *)malloc(max_tile_bytes / 2);
        ctx->conv_C = (float *)malloc(max_tile_bytes / 2);
        if (!ctx->conv_A || !ctx->conv_B || !ctx->conv_C) {
            fprintf(stderr, "metal_ctx_init: conv buffer allocation failed\n");
            free(ctx->conv_A); free(ctx->conv_B); free(ctx->conv_C);
            free(ctx);
            return NULL;
        }
    }

    return ctx;
}

void metal_ctx_destroy(MetalCtx *ctx)
{
    if (!ctx) return;
    free(ctx->conv_A);
    free(ctx->conv_B);
    free(ctx->conv_C);
    /* ARC releases the Objective-C objects. */
    free(ctx);
}

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
                             int         is_complex)
{
    /*
     * Element counts (not bytes).  lda/ldb/ldc are in elements.
     * Pool pages use double (8 B) or double _Complex (16 B) per element.
     * GPU buffers hold float (4 B) or float2 (8 B) per element.
     */
    size_t n_A = (size_t)M * (size_t)lda;   /* elements in A row-major block */
    size_t n_B = (size_t)K * (size_t)ldb;
    size_t n_C = (size_t)M * (size_t)ldc;
    size_t float_sz = is_complex ? 8 : 4;   /* bytes per element in float buffers */

    if (n_A * float_sz > ctx->scratch_cap ||
        n_B * float_sz > ctx->scratch_cap ||
        n_C * float_sz > ctx->scratch_cap) {
        fprintf(stderr,
                "metal_compute_tile_task: float tile %zu / %zu / %zu bytes "
                "exceeds scratch cap %zu\n",
                n_A * float_sz, n_B * float_sz, n_C * float_sz,
                ctx->scratch_cap);
        return -1;
    }

    /* --- Convert double inputs → float scratch --- */
    if (is_complex) {
        /* double _Complex: interleaved (re, im) pairs → float2 pairs */
        const double *dA = (const double *)A;
        const double *dB = (const double *)B;
        float *fA = ctx->conv_A;
        float *fB = ctx->conv_B;
        for (size_t i = 0; i < n_A * 2; i++) fA[i] = (float)dA[i];
        for (size_t i = 0; i < n_B * 2; i++) fB[i] = (float)dB[i];
    } else {
        const double *dA = (const double *)A;
        const double *dB = (const double *)B;
        float *fA = ctx->conv_A;
        float *fB = ctx->conv_B;
        for (size_t i = 0; i < n_A; i++) fA[i] = (float)dA[i];
        for (size_t i = 0; i < n_B; i++) fB[i] = (float)dB[i];
    }

    @autoreleasepool {
        /* Copy float input tiles into shared MTLBuffers. */
        size_t bytes_fA = n_A * float_sz;
        size_t bytes_fB = n_B * float_sz;
        size_t bytes_fC = n_C * float_sz;
        memcpy([ctx->buf_A contents], ctx->conv_A, bytes_fA);
        memcpy([ctx->buf_B contents], ctx->conv_B, bytes_fB);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        id<MTLComputePipelineState> pso =
            is_complex ? ctx->pso_cgemm : ctx->pso_sgemm;
        [enc setComputePipelineState:pso];
        [enc setBuffer:ctx->buf_A offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_B offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_C offset:0 atIndex:2];
        [enc setBytes:&M   length:sizeof(int) atIndex:3];
        [enc setBytes:&K   length:sizeof(int) atIndex:4];
        [enc setBytes:&N   length:sizeof(int) atIndex:5];
        [enc setBytes:&lda length:sizeof(int) atIndex:6];
        [enc setBytes:&ldb length:sizeof(int) atIndex:7];
        [enc setBytes:&ldc length:sizeof(int) atIndex:8];

        /* One thread per output element; threadgroup size capped to device max. */
        NSUInteger tg_w = pso.maxTotalThreadsPerThreadgroup;
        if (tg_w > 256) tg_w = 256;
        NSUInteger tg_x = (tg_w > 16) ? 16 : tg_w;
        NSUInteger tg_y = tg_w / tg_x;

        MTLSize grid   = MTLSizeMake((NSUInteger)N, (NSUInteger)M, 1);
        MTLSize tgSize = MTLSizeMake(tg_x, tg_y, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgSize];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        /* Copy float result out of the MTLBuffer into conv_C. */
        memcpy(ctx->conv_C, [ctx->buf_C contents], bytes_fC);
    }

    /* --- Convert float result → double output --- */
    if (is_complex) {
        const float *fC = ctx->conv_C;
        double *dC = (double *)C;
        for (size_t i = 0; i < n_C * 2; i++) dC[i] = (double)fC[i];
    } else {
        const float *fC = ctx->conv_C;
        double *dC = (double *)C;
        for (size_t i = 0; i < n_C; i++) dC[i] = (double)fC[i];
    }

    return 0;
}

#else  /* !__APPLE__ */

#include "metal_backend.h"
#include <stddef.h>

MetalCtx *metal_ctx_init(size_t max_tile_bytes)
{
    (void)max_tile_bytes;
    return NULL;
}

void metal_ctx_destroy(MetalCtx *ctx) { (void)ctx; }

int metal_compute_tile_task(MetalCtx   *ctx,
                             const void *A,
                             const void *B,
                             void       *C,
                             int M, int K, int N,
                             int lda, int ldb, int ldc,
                             int is_complex)
{
    (void)ctx; (void)A; (void)B; (void)C;
    (void)M; (void)K; (void)N;
    (void)lda; (void)ldb; (void)ldc; (void)is_complex;
    return -1;
}

#endif /* __APPLE__ */
