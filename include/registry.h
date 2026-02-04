#ifndef REGISTRY_H
#define REGISTRY_H

#include <hdf5.h>
#include <stdbool.h>

// Status flags for a tile
typedef enum {
  TILE_STATUS_NULL = 0, // Does not exist / All-Zero (Virtual)
  TILE_STATUS_ON_DISK,  // Data exists in HDF5
  TILE_STATUS_IN_RAM    // Currently loaded in buffer (Future use)
} TileStatus;

// Metadata for a single tile
typedef struct {
  hsize_t global_coords[3]; // The logical coordinates (i, j, k)
  hsize_t phys_offset[3];   // The exact HDF5 offset (in elements)
  TileStatus status;
  int buffer_id; // ID of RAM buffer if loaded (-1 if not)
} TileMetadata;

// The Global Registry
typedef struct {
  int rank;
  hsize_t global_dims[3]; // Size of the full tensor
  hsize_t chunk_dims[3];  // Size of one chunk
  hsize_t grid_dims[3];   // Number of tiles along each axis
  size_t total_tiles;     // Total number of tiles
  TileMetadata *tiles;    // Flat array of all potential tiles
} TensorRegistry;

// Functions
TensorRegistry *registry_create(int rank, const hsize_t *global_dims,
                                size_t target_chunk_bytes);
void registry_destroy(TensorRegistry *reg);
TileMetadata *registry_get_tile(TensorRegistry *reg, hsize_t i, hsize_t j,
                                hsize_t k);

/* * Scans an open HDF5 dataset and marks existing chunks in the registry.
 * Returns the number of active chunks found.
 */
long registry_scan_file(hid_t dset_id, TensorRegistry *reg);

#endif
