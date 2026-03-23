#include "registry.h"
#include "tensor_store.h"  /* calculate_chunk_dims */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>

/* ----------------------------------------------------------------------- */
/* Internal: allocate and populate tile metadata from known rank/dims      */
/* ----------------------------------------------------------------------- */

static TensorRegistry *registry_alloc_and_init(int rank,
                                               const hsize_t *global_dims,
                                               const hsize_t *chunk_dims)
{
    TensorRegistry *reg = (TensorRegistry *)malloc(sizeof(TensorRegistry));
    if (!reg) return NULL;

    reg->rank        = rank;
    reg->total_tiles = 1;

    for (int d = 0; d < rank; d++) {
        reg->global_dims[d] = global_dims[d];
        reg->chunk_dims[d]  = chunk_dims[d];
        reg->grid_dims[d]   = (global_dims[d] + chunk_dims[d] - 1)
                              / chunk_dims[d];
        reg->total_tiles   *= reg->grid_dims[d];
    }

    reg->tiles = (TileMetadata *)calloc(reg->total_tiles, sizeof(TileMetadata));
    if (!reg->tiles) { free(reg); return NULL; }

    /* Pre-calculate coords and physical offsets for every tile slot. */
    for (size_t idx = 0; idx < reg->total_tiles; idx++) {
        size_t  temp = idx;
        hsize_t coords[MAX_RANK];

        /* Reverse-linearise: row-major ordering */
        for (int d = rank - 1; d >= 0; d--) {
            coords[d]  = (hsize_t)(temp % reg->grid_dims[d]);
            temp      /= reg->grid_dims[d];
        }

        TileMetadata *t = &reg->tiles[idx];
        t->status    = TILE_STATUS_NULL;
        t->buffer_id = SIZE_MAX;

        for (int d = 0; d < rank; d++) {
            t->global_coords[d] = coords[d];
            t->phys_offset[d]   = coords[d] * reg->chunk_dims[d];
        }
    }

    return reg;
}

/* ----------------------------------------------------------------------- */
/* Public constructors                                                      */
/* ----------------------------------------------------------------------- */

TensorRegistry *registry_create(int rank, const hsize_t *global_dims,
                                size_t target_chunk_bytes)
{
    if (rank <= 0 || rank > MAX_RANK) return NULL;

    hsize_t chunk_dims[MAX_RANK];
    calculate_chunk_dims(target_chunk_bytes, rank, global_dims, chunk_dims);
    TensorRegistry *reg = registry_alloc_and_init(rank, global_dims, chunk_dims);
    if (reg) reg->dtype = DTYPE_FP64;
    return reg;
}

TensorRegistry *registry_create_from_dset(hid_t dset_id)
{
    /* Read rank and global shape from the dataspace. */
    hid_t fspace_id = H5Dget_space(dset_id);
    if (fspace_id < 0) return NULL;

    int rank = H5Sget_simple_extent_ndims(fspace_id);
    if (rank <= 0 || rank > MAX_RANK) {
        H5Sclose(fspace_id);
        return NULL;
    }

    hsize_t global_dims[MAX_RANK];
    if (H5Sget_simple_extent_dims(fspace_id, global_dims, NULL) < 0) {
        H5Sclose(fspace_id);
        return NULL;
    }
    H5Sclose(fspace_id);

    /* Read chunk dims from the dataset creation property list. */
    hid_t dcpl_id = H5Dget_create_plist(dset_id);
    if (dcpl_id < 0) return NULL;

    hsize_t chunk_dims[MAX_RANK];
    if (H5Pget_chunk(dcpl_id, rank, chunk_dims) < 0) {
        H5Pclose(dcpl_id);
        return NULL;
    }
    H5Pclose(dcpl_id);

    TensorRegistry *reg = registry_alloc_and_init(rank, global_dims, chunk_dims);
    if (!reg) return NULL;

    /*
     * Detect element type: compound (two-field real/imag) → COMPLEX128,
     * everything else → FP64.
     */
    hid_t h5type = H5Dget_type(dset_id);
    if (h5type >= 0) {
        reg->dtype = (H5Tget_class(h5type) == H5T_COMPOUND)
                     ? DTYPE_COMPLEX128
                     : DTYPE_FP64;
        H5Tclose(h5type);
    } else {
        reg->dtype = DTYPE_FP64;
    }

    return reg;
}

/* ----------------------------------------------------------------------- */
/* Destructor                                                               */
/* ----------------------------------------------------------------------- */

void registry_destroy(TensorRegistry *reg)
{
    if (reg) {
        free(reg->tiles);
        free(reg);
    }
}

/* ----------------------------------------------------------------------- */
/* registry_get_tile                                                        */
/* ----------------------------------------------------------------------- */

TileMetadata *registry_get_tile(TensorRegistry *reg,
                                const hsize_t *tile_coords)
{
    /* Bounds check all dimensions. */
    for (int d = 0; d < reg->rank; d++) {
        if (tile_coords[d] >= reg->grid_dims[d])
            return NULL;
    }

    /* Row-major linearisation: index = sum_d( coord[d] * prod_{e>d} grid[e] ) */
    size_t index = 0;
    for (int d = 0; d < reg->rank; d++) {
        index = index * (size_t)reg->grid_dims[d] + (size_t)tile_coords[d];
    }

    return &reg->tiles[index];
}

/* ----------------------------------------------------------------------- */
/* registry_scan_file                                                       */
/* ----------------------------------------------------------------------- */

long registry_scan_file(hid_t dset_id, TensorRegistry *reg)
{
    hid_t fspace_id = H5Dget_space(dset_id);
    if (fspace_id < 0) {
        fprintf(stderr, "registry_scan_file: H5Dget_space failed\n");
        return -1;
    }

    hsize_t num_chunks = 0;
    if (H5Dget_num_chunks(dset_id, fspace_id, &num_chunks) < 0) {
        fprintf(stderr, "registry_scan_file: H5Dget_num_chunks failed\n");
        H5Sclose(fspace_id);
        return -1;
    }

    long found_count = 0;

    for (hsize_t ci = 0; ci < num_chunks; ci++) {
        hsize_t chunk_offset[MAX_RANK];
        memset(chunk_offset, 0, sizeof(chunk_offset));

        if (H5Dget_chunk_info(dset_id, fspace_id, ci,
                              chunk_offset, NULL, NULL, NULL) < 0) {
            fprintf(stderr,
                    "registry_scan_file: H5Dget_chunk_info failed at idx %llu"
                    " (skipping)\n", (unsigned long long)ci);
            continue;
        }

        /* Convert physical offset -> logical tile coordinates. */
        hsize_t tile_coords[MAX_RANK];
        for (int d = 0; d < reg->rank; d++)
            tile_coords[d] = chunk_offset[d] / reg->chunk_dims[d];

        TileMetadata *tile = registry_get_tile(reg, tile_coords);
        if (!tile) {
            fprintf(stderr,
                    "registry_scan_file: chunk at offset out of registry "
                    "bounds (skipping)\n");
            continue;
        }

        tile->status = TILE_STATUS_ON_DISK;
        found_count++;
    }

    H5Sclose(fspace_id);
    return found_count;
}
