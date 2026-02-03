#include "tensor_store.h"
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Helper: Converts Logical Tile Coordinates (e.g., 0, 1, 5)
 * into Physical Element Offsets (e.g., 0, 64, 320)
 */
void get_physical_offset(int rank, const hsize_t *tile_coords,
                          const hsize_t *chunk_dims, hsize_t *phys_offset_out) {
  for (int i = 0; i < rank; i++) {
    phys_offset_out[i] = tile_coords[i] * chunk_dims[i];
  }
}

/* * Helper: Calculate chunk dimensions based on target cache size.
 * target_bytes: Desired size of one chunk in bytes (e.g., 2MB).
 * rank: Number of dimensions (e.g., 3 for a tensor)
 * global_dims: The full shape of the tensor.
 * chunk_dims_out: Array to store the calculated chunk sizes.
 */

void calculate_chunk_dims(size_t target_bytes, int rank,
                          const hsize_t *global_dims, hsize_t *chunk_dims_out) {
  size_t element_size = sizeof(double);
  size_t total_elements = target_bytes / element_size;

  // Calculate the size length of a cube that fits the target size
  // side_length = cube_root(total_elements)
  double side_length = pow((double)total_elements, 1.0 / rank);
  hsize_t side_int = (hsize_t)round(side_length);

  // Safety floor: At least 1x1... block
  if (side_int < 1)
    side_int = 1;

  for (int i = 0; i < rank; i++) {
    // The chunk dimension cannot exceed the global dimension
    if (side_int > global_dims[i]) {
      chunk_dims_out[i] = global_dims[i];
    } else {
      chunk_dims_out[i] = side_int;
    }
  }
}

/**
 * Optimized Read: Accepts an already OPEN dataset handle (dset_id).
 * This avoids the overhead of H5Dopen/close on every call.
 * dset_id: HDF5 dataset identifier (already opened)
 * phys_offset: Array of coordinates specifying the chunk offset
 * data_ptr: Buffer to read data into
 * rank: Number of dimensions
 * chunk_dims: Size of each chunk dimension
 */
herr_t read_chunk_fast(hid_t dset_id, const hsize_t *phys_offset,
                        double *data_ptr, int rank, const hsize_t *chunk_dims) {
  hid_t filespace_id, memspace_id;
  herr_t status;

  // 1. Get the file dataspace (cheap operation on open dataset)
  filespace_id = H5Dget_space(dset_id);

  // 2. Select the Hyperslab (The specific chunk on disk)
  status = H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, phys_offset, NULL,
                                chunk_dims, NULL);
  if (status < 0) {
    H5Sclose(filespace_id);
    return -1;
  }

  // 3. Create the Memory Dataspace (The buffer in RAM)
  // We want a flat, contiguous block in memory that matches the chunk size
  memspace_id = H5Screate_simple(rank, chunk_dims, NULL);

  // 4. Execute Read
  status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace_id, filespace_id,
                    H5P_DEFAULT, data_ptr);

  // 5. Cleanup (Only close the temporary spaces)
  H5Sclose(memspace_id);
  H5Sclose(filespace_id);

  return status;
}

/**
 * Optimized Write: Accepts an already OPEN dataset handle.
 * dset_id: HDF5 dataset identifier (already opened)
 * phys_offset: Array of coordinates specifying the chunk offset
 * data_ptr: Buffer containing data to write
 * rank: Number of dimensions
 * chunk_dims: Size of each chunk dimension
 */
herr_t write_chunk_fast(hid_t dset_id, const hsize_t *phys_offset,
                         const double *data_ptr, int rank,
                         const hsize_t *chunk_dims) {
  hid_t filespace_id, memspace_id;
  herr_t status;

  filespace_id = H5Dget_space(dset_id);

  status = H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, phys_offset, NULL,
                                chunk_dims, NULL);
  if (status < 0) {
    H5Sclose(filespace_id);
    return -1;
  }

  memspace_id = H5Screate_simple(rank, chunk_dims, NULL);

  status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace_id, filespace_id,
                    H5P_DEFAULT, data_ptr);

  H5Sclose(memspace_id);
  H5Sclose(filespace_id);

  return status;
}

/*
 * Create an HDF5 file and a chunked dataset optimized for TBLIS.
 */
void create_chunked_dataset(const char *filename, const char *dataset_name,
                            int rank, const hsize_t *global_dims) {
  // This is a simplified version that doesn't use MPI for now
  hid_t file_id, space_id, dset_id, dcpl_id;
  herr_t status;
  hsize_t chunk_dims[rank];

  // 1. Create a new HDF5 file
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {
    fprintf(stderr, "Error creating file %s\n", filename);
    return;
  }

  // 2. Create the Dataspace (Logical Shape)
  space_id = H5Screate_simple(rank, global_dims, NULL);

  // 3. Setup Property List for Chunking
  dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

  // Calculate optimal chunk size (Target: 2MB = 2 * 1024 * 1024 bytes)
  calculate_chunk_dims(2 * 1024 * 1024, rank, global_dims, chunk_dims);

  // Apply chunking parameters
  status = H5Pset_chunk(dcpl_id, rank, chunk_dims);

  // 4. Set Fill Value (Optimization for Sparsity)
  double fill_value = 0.0;
  status = H5Pset_fill_value(dcpl_id, H5T_NATIVE_DOUBLE, &fill_value);

  // Optional: Set allocation time to "Incremental"
  status = H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_INCR);

  // 5. Create the Dataset
  dset_id = H5Dcreate2(file_id, dataset_name, H5T_NATIVE_DOUBLE, space_id,
                        H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

  if (dset_id < 0) {
    fprintf(stderr, "Error creating dataset %s\n", dataset_name);
  } else {
    printf("Successfully created tensor '%s' .\n", dataset_name);
    printf("Logical shape: ");
    for (int i = 0; i < rank; i++)
      printf("%llu ", global_dims[i]);
    printf("\nChunk Shape: ");
    for (int i = 0; i < rank; i++)
      printf("%llu ", chunk_dims[i]);
    printf("\n");
  }

  // 6. Cleanup Resources
  H5Pclose(dcpl_id);
  H5Dclose(dset_id);
  H5Sclose(space_id);
  H5Fclose(file_id);
}
