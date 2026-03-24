/*
 * src/tensor_engine.c — thin wrapper implementing the public tensor_engine.h API.
 *
 * This file contains no compute logic.  All heavy lifting is done by
 * run_contraction_einsum() in engine.c; this wrapper only manages the opaque
 * context struct and translates configuration into the env-var protocol that
 * the engine reads at startup.
 */

#include "tensor_engine.h"
#include "engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Default dataset name expected in every HDF5 file handled by the public API. */
#define DEFAULT_DSET "tensor"

/* -------------------------------------------------------------------------
 * Opaque handle definition (internal only)
 * -----------------------------------------------------------------------*/

struct tensor_engine {
    size_t pool_mb;
    size_t tile_bytes;
};

/* -------------------------------------------------------------------------
 * Lifecycle
 * -----------------------------------------------------------------------*/

tensor_engine_t *tensor_engine_init(const tensor_engine_config_t *cfg)
{
    tensor_engine_t *eng = (tensor_engine_t *)malloc(sizeof(*eng));
    if (!eng)
        return NULL;

    if (cfg) {
        eng->pool_mb    = cfg->pool_mb;
        eng->tile_bytes = cfg->tile_bytes;
    } else {
        eng->pool_mb    = 0;
        eng->tile_bytes = 0;
    }

    return eng;
}

void tensor_engine_free(tensor_engine_t *engine)
{
    free(engine);
}

/* -------------------------------------------------------------------------
 * Contraction
 * -----------------------------------------------------------------------*/

int tensor_engine_contract(tensor_engine_t *engine,
                           const char      *einsum_expr,
                           const char      *file_A,
                           const char      *file_B,
                           const char      *file_C)
{
    if (!engine || !einsum_expr || !file_A || !file_B || !file_C)
        return TENSOR_ENGINE_ERR;

    /* Publish pool cap via the environment variable that engine.c reads.
     * We only set it when the caller explicitly requested a cap (pool_mb > 0).
     * Otherwise we leave the variable alone so the engine auto-tunes to 80 %
     * of physical RAM. */
    char pool_buf[32];
    if (engine->pool_mb > 0) {
        snprintf(pool_buf, sizeof(pool_buf), "%zu", engine->pool_mb);
        setenv("TENSOR_POOL_MB", pool_buf, /*overwrite=*/1);
    }

    int rc = run_contraction_einsum(einsum_expr,
                                    file_A, DEFAULT_DSET,
                                    file_B, DEFAULT_DSET,
                                    file_C, DEFAULT_DSET);

    /* Clear the env-var after the call so it does not bleed into a subsequent
     * invocation that omits pool_mb. */
    if (engine->pool_mb > 0)
        unsetenv("TENSOR_POOL_MB");

    return (rc == 0) ? TENSOR_ENGINE_OK : TENSOR_ENGINE_ERR;
}

/* -------------------------------------------------------------------------
 * Error descriptions
 * -----------------------------------------------------------------------*/

const char *tensor_engine_strerror(int err)
{
    switch (err) {
    case TENSOR_ENGINE_OK:        return "success";
    case TENSOR_ENGINE_ERR_FILE:  return "file not found or I/O error";
    case TENSOR_ENGINE_ERR_DIMS:  return "incompatible tensor dimensions";
    case TENSOR_ENGINE_ERR_EXPR:  return "malformed einsum expression";
    case TENSOR_ENGINE_ERR_MEM:   return "memory allocation failed";
    case TENSOR_ENGINE_ERR:       return "internal engine error";
    default:                      return "unknown error";
    }
}
