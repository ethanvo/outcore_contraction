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
#include "tensor_store.h"
#include "registry.h"
#include "odometer.h"

#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

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

int tensor_engine_accumulate(tensor_engine_t *engine,
                             const char      *einsum_expr,
                             const char      *file_A,
                             const char      *file_B,
                             const char      *file_C)
{
    if (!engine || !einsum_expr || !file_A || !file_B || !file_C)
        return TENSOR_ENGINE_ERR;

    char pool_buf[32];
    if (engine->pool_mb > 0) {
        snprintf(pool_buf, sizeof(pool_buf), "%zu", engine->pool_mb);
        setenv("TENSOR_POOL_MB", pool_buf, /*overwrite=*/1);
    }

    int rc = run_contraction_einsum_acc(einsum_expr,
                                        file_A, DEFAULT_DSET,
                                        file_B, DEFAULT_DSET,
                                        file_C, DEFAULT_DSET);

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

/* -------------------------------------------------------------------------
 * Tensor creation
 * -----------------------------------------------------------------------*/

/* NVMe page size used to align chunk byte budgets — matches engine.c. */
#define CREATE_NVME_PAGE  16384UL
/* Default tile size in bytes (16 MiB) — matches engine.c default. */
#define CREATE_TILE_BYTES (16UL << 20)

int tensor_engine_create(tensor_engine_t *engine,
                         const char      *file_path,
                         int              rank,
                         const size_t    *shape,
                         int              dtype)
{
    if (!engine || !file_path || rank < 1 || rank > MAX_RANK || !shape)
        return TENSOR_ENGINE_ERR;

    for (int d = 0; d < rank; d++) {
        if (shape[d] == 0) return TENSOR_ENGINE_ERR_DIMS;
    }

    tensor_dtype_t tdtype;
    size_t elem_size;
    if (dtype == TENSOR_DTYPE_COMPLEX128) {
        tdtype    = DTYPE_COMPLEX128;
        elem_size = sizeof(double _Complex);
    } else {
        tdtype    = DTYPE_FP64;
        elem_size = sizeof(double);
    }

    /* Convert shape to HDF5 types. */
    hsize_t hshape[MAX_RANK];
    for (int d = 0; d < rank; d++)
        hshape[d] = (hsize_t)shape[d];

    /* Compute isotropic chunk dims from tile_bytes, generalised for any dtype.
     *
     * calculate_chunk_dims() is hardwired to sizeof(double)=8.  We replicate
     * the same algorithm here so COMPLEX128 tiles also hit the byte target:
     *   target_elems = tile_bytes / elem_size
     *   side         = round( nthroot(target_elems, rank) )
     * clamped per dimension to global_dims[d].
     */
    size_t tile_bytes = (engine->tile_bytes > 0)
                        ? engine->tile_bytes : CREATE_TILE_BYTES;
    /* Round up to NVMe page boundary so chunks stay aligned. */
    tile_bytes = (tile_bytes + CREATE_NVME_PAGE - 1) & ~(CREATE_NVME_PAGE - 1);

    size_t  target_elems = tile_bytes / elem_size;
    double  side_d       = pow((double)target_elems, 1.0 / (double)rank);
    hsize_t chunk_side   = (hsize_t)round(side_d);
    if (chunk_side < 1) chunk_side = 1;

    hsize_t hchunk[MAX_RANK];
    for (int d = 0; d < rank; d++)
        hchunk[d] = (chunk_side > hshape[d]) ? hshape[d] : chunk_side;

    if (create_chunked_dataset_einsum(file_path, DEFAULT_DSET,
                                      rank, hshape, hchunk, tdtype) < 0)
        return TENSOR_ENGINE_ERR_FILE;

    return TENSOR_ENGINE_OK;
}

/* -------------------------------------------------------------------------
 * Tensor fill
 * -----------------------------------------------------------------------*/

int tensor_engine_fill(tensor_engine_t *engine,
                       const char      *file_path,
                       const void      *value)
{
    if (!engine || !file_path || !value)
        return TENSOR_ENGINE_ERR;

    hid_t fid = H5Fopen(file_path, H5F_ACC_RDWR, H5P_DEFAULT);
    if (fid < 0)
        return TENSOR_ENGINE_ERR_FILE;

    hid_t dset = dset_open_no_cache(fid, DEFAULT_DSET);
    if (dset < 0) {
        H5Fclose(fid);
        return TENSOR_ENGINE_ERR_FILE;
    }

    TensorRegistry *reg = registry_create_from_dset(dset);
    if (!reg) {
        H5Dclose(dset);
        H5Fclose(fid);
        return TENSOR_ENGINE_ERR_MEM;
    }

    int    rank      = reg->rank;
    size_t elem_size = (reg->dtype == DTYPE_COMPLEX128)
                       ? sizeof(double _Complex) : sizeof(double);

    /* Build the HDF5 memory type.  For FP64 use the predefined constant
     * (no lifetime management needed).  For COMPLEX128 allocate a compound. */
    hid_t mem_type;
    int   own_mem_type = 0;
    if (reg->dtype == DTYPE_COMPLEX128) {
        mem_type     = create_h5_complex_type();
        own_mem_type = 1;
        if (mem_type < 0) {
            registry_destroy(reg);
            H5Dclose(dset);
            H5Fclose(fid);
            return TENSOR_ENGINE_ERR_MEM;
        }
    } else {
        mem_type = H5T_NATIVE_DOUBLE;
    }

    /* Allocate one nominal tile buffer and broadcast the scalar into it. */
    size_t tile_elems = 1;
    size_t ntiles[MAX_RANK];
    for (int d = 0; d < rank; d++) {
        tile_elems *= (size_t)reg->chunk_dims[d];
        ntiles[d]   = ((size_t)reg->global_dims[d]
                       + (size_t)reg->chunk_dims[d] - 1)
                      / (size_t)reg->chunk_dims[d];
    }

    void *buf = malloc(tile_elems * elem_size);
    if (!buf) {
        if (own_mem_type) H5Tclose(mem_type);
        registry_destroy(reg);
        H5Dclose(dset);
        H5Fclose(fid);
        return TENSOR_ENGINE_ERR_MEM;
    }

    /* Broadcast: copy the scalar into every element slot of the tile buffer. */
    for (size_t i = 0; i < tile_elems; i++)
        memcpy((char *)buf + i * elem_size, value, elem_size);

    /* Iterate over every tile in row-major order and write it. */
    int    ret  = TENSOR_ENGINE_OK;
    size_t tile[MAX_RANK];
    memset(tile, 0, sizeof(tile));
    do {
        hsize_t phys_off[MAX_RANK];
        for (int d = 0; d < rank; d++)
            phys_off[d] = (hsize_t)tile[d] * reg->chunk_dims[d];

        if (write_chunk_typed(dset, phys_off, buf, elem_size,
                              rank, reg->chunk_dims, mem_type) < 0) {
            ret = TENSOR_ENGINE_ERR_FILE;
            break;
        }
    } while (odometer_step((size_t)rank, tile, ntiles));

    free(buf);
    if (own_mem_type) H5Tclose(mem_type);
    registry_destroy(reg);
    H5Dclose(dset);
    H5Fclose(fid);
    return ret;
}
