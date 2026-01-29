#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* * Helper: Calculate chunk dimensions based on target cache size.
 * target_bytes: Desired size of one chunk in bytes (e.g., 2MB).
 * rank: Number of dimensions (e.g., 3 for a tensor)
 * global_dims: The full shape of the tensor.
 * chunk_dims_out: Array to store the calculated chunk sizes.
 */

void calculate_chunk_dims(size_t target_bytes, int rank, const hsize_t *global_dims, hsize_t *chunk_dims_out) {
  size_t element_size = sizeof(double);
  size_t total_elements = target_bytes / element_size;

  // Calculate the size length of a cube that fits the target size
  // side_length = cube_root(total_elements)
  double side_length = pow((double) total_elements, 1.0 / rank);
  hsize_t side_int = (hsize_t) round(side_length);

  // Safety floor: At least 1x1... block
  if (side_int < 1) side_int = 1;

  for (int i = 0; i < rank; i++) {
    // The chunk dimension cannot exceed the global dimension
    if (side_int > global_dims[i]) {
      chunk_dims_out[i] = global_dims[i];
    } else {
      chunk_dims_out[i] = side_int;
    }
  }
}

/*
 * Create an HDF5 file and a chunked dataset optimized for TBLIS.
 */
void create_chunked_dataset(const char *filename, const char *dataset_name, int rank, const hsize_t *global_dims) {
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

  // Caculate optimal chunk size (Target: 2MB = 2 * 1024 * 1024 bytes)
  calculate_chunk_dims(2 * 1024 * 1024, rank, global_dims, chunk_dims);

  // Apply chunking parameters
  status = H5Pset_chunk(dcpl_id, rank, chunk_dims);

  // 4. Set Fill Value (Optimization for Sparsity)
  // If we never write to a chunk, it takes 0 disk space.
  double fill_value = 0.0;
  status = H5Pset_fill_value(dcpl_id, H5T_NATIVE_DOUBLE, &fill_value);

  // Optional: Set allocation time to "Incremental" (allocate only when written)
  // This is often default for chunked datasets, but explicit is safer.
  status = H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_INCR);

  // 5. Create the Dataset
  dset_id = H5Dcreate2(file_id, dataset_name, H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

  if (dset_id < 0) {
    fprintf(stderr, "Error creating dataset %s\n", dataset_name);
  } else {
    printf("Successfully created tensor '%s' .\n", dataset_name);
    printf("Logical shape: ");
    for (int i = 0; i < rank; i++) printf("%llu ", global_dims[i]);
    printf("\nChunk Shape: ");
    for (int i = 0; i < rank; i++) printf("%llu ", chunk_dims[i]);
    printf("\n");
  }

  // 6. Cleanup Resources
  H5Pclose(dcpl_id);
  H5Dclose(dset_id);
  H5Sclose(space_id);
  H5Fclose(file_id);
}

int main() {
  // Example: Create a 3D Tensor of size 1000 x 1000 x 1000
  hsize_t dims[3] = {1000, 1000, 1000};
  create_chunked_dataset("tensor_store.h5", "TensorA", 3, dims);

  // Example: Create a "Flat" Tensor where global dim is smaller than ideal chunk
  hsize_t dims_flat[3] = {10, 5000, 5000};
  create_chunked_dataset("tensor_store.h5", "TensorB", 3, dims_flat);

  return 0;
}
