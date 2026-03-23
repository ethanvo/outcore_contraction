#ifndef REGISTRY_H
#define REGISTRY_H

#include <hdf5.h>
#include <stddef.h>

/*
 * Maximum tensor rank supported by the registry.  All fixed-size arrays are
 * sized to MAX_RANK so that rank-4 and rank-5 tensors work without any
 * recompilation.
 */
#define MAX_RANK 8

/*
 * Element type carried by a tensor dataset.
 *   DTYPE_FP64       — native double (8 bytes), stored as H5T_NATIVE_DOUBLE.
 *   DTYPE_COMPLEX128 — double _Complex (16 bytes), stored as an HDF5 compound
 *                      type with fields "r" and "i" (each H5T_NATIVE_DOUBLE).
 */
typedef enum {
    DTYPE_FP64       = 0,
    DTYPE_COMPLEX128 = 1
} tensor_dtype_t;

typedef enum {
    TILE_STATUS_NULL   = 0,  /* Virtual / all-zero – never written to disk   */
    TILE_STATUS_ON_DISK,     /* Data exists as an allocated HDF5 chunk        */
    TILE_STATUS_IN_RAM       /* Currently held in a BufferPool page           */
} TileStatus;

typedef struct {
    hsize_t    global_coords[MAX_RANK]; /* Logical tile index per dimension   */
    hsize_t    phys_offset[MAX_RANK];   /* Element offset into the HDF5 file  */
    TileStatus status;
    size_t     buffer_id;               /* Pool page ID; SIZE_MAX if not loaded */
} TileMetadata;

typedef struct {
    int            rank;
    tensor_dtype_t dtype;                  /* Element type (FP64 or COMPLEX128) */
    hsize_t global_dims[MAX_RANK]; /* Full tensor shape                        */
    hsize_t chunk_dims[MAX_RANK];  /* Nominal chunk shape (from HDF5 creation) */
    hsize_t grid_dims[MAX_RANK];   /* Number of tiles per dimension            */
    size_t  total_tiles;           /* Product of grid_dims                     */
    TileMetadata *tiles;           /* Flat row-major array of all tile metadata */
} TensorRegistry;

/*
 * Create a registry by computing chunk dims from target_chunk_bytes.
 * Use registry_create_from_dset when an HDF5 dataset already exists, so the
 * chunk dims are read from the file rather than recomputed.
 */
TensorRegistry *registry_create(int rank, const hsize_t *global_dims,
                                size_t target_chunk_bytes);

/*
 * Create a registry by reading rank, global_dims, and chunk_dims directly from
 * an open HDF5 dataset.  This is the authoritative factory for existing files;
 * it eliminates any divergence between what was baked into the file and what
 * the registry believes the chunk boundaries to be.
 * The dataset must remain open for the lifetime of any subsequent
 * registry_scan_file call.
 */
TensorRegistry *registry_create_from_dset(hid_t dset_id);

void registry_destroy(TensorRegistry *reg);

/*
 * n-dimensional tile lookup.
 * tile_coords must contain reg->rank elements in row-major order.
 * Returns NULL if any coordinate is out of bounds.
 */
TileMetadata *registry_get_tile(TensorRegistry *reg,
                                const hsize_t *tile_coords);

/*
 * Scan an open HDF5 dataset and mark all allocated chunks TILE_STATUS_ON_DISK.
 * Returns the number of active chunks found, or -1 on error.
 */
long registry_scan_file(hid_t dset_id, TensorRegistry *reg);

#endif /* REGISTRY_H */
