#include "registry.h"
#include "tensor_store.h" // Ensure this includes create_chunked_dataset
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  printf("Testing tensor storage I/O functionality...\n");

  const char *filename = "test_tensor_io.h5";
  const char *dataset_name = "test_tensor_io";

  // FIX 1: Make dimensions large enough to support multiple tiles.
  // 300^3 elements is roughly 27 million doubles (~216 MB).
  const int rank = 3;
  hsize_t global_dims[] = {300, 300, 300};

  // Create the dataset
  create_chunked_dataset(filename, dataset_name, rank, global_dims);

  hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
  if (file_id < 0) {
    fprintf(stderr, "ERROR: Could not open file\n");
    return 1;
  }

  hid_t dset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
  if (dset_id < 0) {
    fprintf(stderr, "ERROR: Could not open dataset\n");
    H5Fclose(file_id);
    return 1;
  }

  // Determine the chunk size that was actually created
  // We calculate it again to match what 'create_chunked_tensor' did
  hsize_t chunk_dims[3];
  calculate_chunk_dims(2 * 1024 * 1024, rank, global_dims, chunk_dims);

  printf("DEBUG: Global Dims: [%llu, %llu, %llu]\n", global_dims[0],
         global_dims[1], global_dims[2]);
  printf("DEBUG: Chunk Dims:  [%llu, %llu, %llu]\n", chunk_dims[0],
         chunk_dims[1], chunk_dims[2]);

  // Calculate buffer size based on actual chunk dimensions
  size_t elements_per_chunk = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];
  double *write_data = (double *)malloc(elements_per_chunk * sizeof(double));
  double *read_data = (double *)malloc(elements_per_chunk * sizeof(double));

  if (!write_data || !read_data) {
    fprintf(stderr, "ERROR: Memory allocation failed\n");
    return 1;
  }

  srand(time(NULL));

  // Loop 3 times (Tile 0,0,0 -> 1,1,1 -> 2,2,2)
  // We limit to 3 because 3 * 64 (approx chunk size) < 300.
  // If we went to 5, 5 * 64 = 320, which would crash again (Out of Bounds).
  for (int i = 0; i < 3; i++) {
    printf("\n--- Operation %d (Tile %d,%d,%d) ---\n", i, i, i, i);

    // 1. Generate Data
    for (size_t j = 0; j < elements_per_chunk; j++) {
      write_data[j] = (double)i + (double)j / 1000.0; // Predictable pattern
    }

    // 2. Calculate Physical Offsets
    hsize_t tile_coords[] = {(hsize_t)i, (hsize_t)i, (hsize_t)i};
    hsize_t phys_offset[3];
    get_physical_offset(rank, tile_coords, chunk_dims, phys_offset);

    printf("    Writing to Offset: [%llu, %llu, %llu]\n", phys_offset[0],
           phys_offset[1], phys_offset[2]);

    // 3. Write
    if (write_chunk_fast(dset_id, phys_offset, write_data, rank, chunk_dims) <
        0) {
      fprintf(stderr, "ERROR: Write failed at step %d\n", i);
      break;
    }

    // 4. Read Back Immediately to Verify
    // (In a real engine, we wouldn't read immediately, but this is a unit test)
    if (read_chunk_fast(dset_id, phys_offset, read_data, rank, chunk_dims) <
        0) {
      fprintf(stderr, "ERROR: Read failed at step %d\n", i);
      break;
    }

    // 5. Verify Content
    if (read_data[0] != write_data[0] ||
        read_data[elements_per_chunk - 1] !=
            write_data[elements_per_chunk - 1]) {
      fprintf(stderr, "DATA MISMATCH! Expected %f, got %f\n", write_data[0],
              read_data[0]);
    } else {
      printf("    Verification Successful.\n");
    }
  }

  // ... previous code that wrote chunks ...

  printf("\n--- Testing Registry Scanning ---\n");

  // 1. Initialize Registry (In memory)
  // Use the same dimensions you used to create the file (300^3)
  // Use the same target size (2MB)
  TensorRegistry *reg = registry_create(rank, global_dims, 2 * 1024 * 1024);

  // 2. Verify Initial State (Should be empty)
  TileMetadata *t_check = registry_get_tile(reg, 0, 0, 0);
  if (t_check->status == TILE_STATUS_NULL) {
    printf("Registry initially empty (Correct).\n");
  }

  // 3. Scan the File
  long found = registry_scan_file(dset_id, reg);

  // 4. Verification
  if (found == 3) {
    printf("SUCCESS: Registry correctly identified 3 chunks on disk.\n");
  } else {
    printf("FAILURE: Registry found %ld chunks, expected 3.\n", found);
  }

  // 5. Check specific tile status
  t_check = registry_get_tile(reg, 1, 1, 1); // We wrote this one earlier
  if (t_check && t_check->status == TILE_STATUS_ON_DISK) {
    printf("Tile (1,1,1) is marked ON_DISK (Correct).\n");
  }

  t_check = registry_get_tile(reg, 0, 1, 0); // We NEVER wrote this one
  if (t_check && t_check->status == TILE_STATUS_NULL) {
    printf("Tile (0,1,0) is marked NULL (Correct).\n");
  }

  registry_destroy(reg);

  free(write_data);
  free(read_data);
  H5Dclose(dset_id);
  H5Fclose(file_id);
  remove(filename); // Cleanup

  printf("\nTest Complete.\n");
  return 0;
}
