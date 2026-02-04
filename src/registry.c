#include "registry.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper: Calculate chunk size (reusing logic from Phase 1) */
static void internal_calc_chunk(size_t target, int rank, const hsize_t *global,
                                hsize_t *out_chunk) {
  size_t elem_size = sizeof(double);
  size_t total_elems = target / elem_size;
  double side = pow((double)total_elems, 1.0 / rank);

  for (int i = 0; i < rank; i++) {
    hsize_t s = (hsize_t)floor(side);
    if (s < 1)
      s = 1;
    if (s > global[i])
      s = global[i];
    out_chunk[i] = s;
  }
}

TensorRegistry *registry_create(int rank, const hsize_t *global_dims,
                                size_t target_chunk_bytes) {
  TensorRegistry *reg = (TensorRegistry *)malloc(sizeof(TensorRegistry));
  reg->rank = rank;

  // 1. Copy Globals and Calculate Chunk Size
  for (int i = 0; i < rank; i++)
    reg->global_dims[i] = global_dims[i];
  internal_calc_chunk(target_chunk_bytes, rank, global_dims, reg->chunk_dims);

  // 2. Calculate Grid Dimensions (How many tiles per axis?)
  reg->total_tiles = 1;
  for (int i = 0; i < rank; i++) {
    // Ceiling division: (global + chunk - 1) / chunk
    reg->grid_dims[i] =
        (global_dims[i] + reg->chunk_dims[i] - 1) / reg->chunk_dims[i];
    reg->total_tiles *= reg->grid_dims[i];
  }

  // 3. Allocate the Flat Array
  // This is a "Dense" registry. Even empty tiles have a slot here.
  reg->tiles = (TileMetadata *)calloc(reg->total_tiles, sizeof(TileMetadata));
  if (!reg->tiles) {
    free(reg);
    return NULL;
  }

  // 4. Populate Metadata (Pre-calculate offsets)
  // We iterate through the flat array and compute i,j,k for each slot
  for (size_t idx = 0; idx < reg->total_tiles; idx++) {
    size_t temp = idx;
    hsize_t coords[3]; // i, j, k

    // Reverse linearization to find coords from index
    // This assumes Row-Major ordering
    for (int d = rank - 1; d >= 0; d--) {
      coords[d] = temp % reg->grid_dims[d];
      temp /= reg->grid_dims[d];
    }

    TileMetadata *t = &reg->tiles[idx];
    t->status = TILE_STATUS_NULL; // Default: Block Sparse (doesn't exist yet)
    t->buffer_id = -1;

    for (int d = 0; d < rank; d++) {
      t->global_coords[d] = coords[d];
      t->phys_offset[d] = coords[d] * reg->chunk_dims[d];
    }
  }

  printf("Registry Created: %zu tiles tracked.\n", reg->total_tiles);
  return reg;
}

TileMetadata *registry_get_tile(TensorRegistry *reg, hsize_t i, hsize_t j,
                                hsize_t k) {
  // 1. Bounds Check
  if (i >= reg->grid_dims[0] || j >= reg->grid_dims[1] ||
      k >= reg->grid_dims[2]) {
    return NULL;
  }

  // 2. Linearize: Index = i*(Ny*Nz) + j*(Nz) + k
  size_t ny = reg->grid_dims[1];
  size_t nz = reg->grid_dims[2];
  size_t index = i * (ny * nz) + j * nz + k;

  return &reg->tiles[index];
}

void registry_destroy(TensorRegistry *reg) {
  if (reg) {
    if (reg->tiles)
      free(reg->tiles);
    free(reg);
  }
}

long registry_scan_file(hid_t dset_id, TensorRegistry *reg) {
  hid_t fspace_id;
  hsize_t num_chunks;
  hsize_t chunk_offset[3]; // Buffer to hold the physical coordinate of a chunk
  hsize_t tile_coords[3];  // Buffer to hold the logical tile index
  int rank = reg->rank;
  long found_count = 0;

  // 1. Get the dataspace to query chunk info
  fspace_id = H5Dget_space(dset_id);
  if (fspace_id < 0) {
    fprintf(stderr, "Error: Could not get dataspace for scanning.\n");
    return -1;
  }

  // 2. Ask HDF5: How many chunks are actually allocated?
  // This is the "Sparsity" check. If the tensor is huge but empty, this returns
  // 0.
  if (H5Dget_num_chunks(dset_id, fspace_id, &num_chunks) < 0) {
    fprintf(stderr, "Error: Could not get number of chunks.\n");
    H5Sclose(fspace_id);
    return -1;
  }

  printf("Scanning file... Found %llu allocated chunks.\n", num_chunks);

  // 3. Iterate ONLY over allocated chunks
  for (hsize_t i = 0; i < num_chunks; i++) {
    // Retrieve the physical offset (e.g., [0, 64, 0]) for the i-th existing
    // chunk We don't care about filter_mask, addr, or size right now, so we
    // pass NULL.
    if (H5Dget_chunk_info(dset_id, fspace_id, i, chunk_offset, NULL, NULL,
                          NULL) < 0) {
      fprintf(stderr, "Warning: Failed to get info for chunk index %llu\n", i);
      continue;
    }

    // 4. Convert Physical Offset -> Logical Tile Index
    // Tile Index = Offset / Chunk_Dim
    for (int d = 0; d < rank; d++) {
      tile_coords[d] = chunk_offset[d] / reg->chunk_dims[d];
    }

    // 5. Update the Registry
    // We look up the tile in our flat array and flip the switch.
    TileMetadata *tile =
        registry_get_tile(reg, tile_coords[0], tile_coords[1], tile_coords[2]);

    if (tile) {
      tile->status = TILE_STATUS_ON_DISK;
      found_count++;

      // Debug print for first few found
      if (found_count <= 5) {
        printf(
            "  -> Found Tile [%llu, %llu, %llu] at Offset [%llu, %llu, %llu]\n",
            tile_coords[0], tile_coords[1], tile_coords[2], chunk_offset[0],
            chunk_offset[1], chunk_offset[2]);
      }
    } else {
      fprintf(stderr,
              "Error: Found a chunk in file that is out of registry bounds!\n");
    }
  }

  H5Sclose(fspace_id);
  return found_count;
}
