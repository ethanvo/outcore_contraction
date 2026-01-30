#ifndef TENSOR_STORE_H
#define TENSOR_STORE_H

#include <hdf5.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Helper: Calculate chunk dimensions based on target cache size.
 * target_bytes: Desired size of one chunk in bytes (e.g., 2MB).
 * rank: Number of dimensions (e.g., 3 for a tensor)
 * global_dims: The full shape of the tensor.
 * chunk_dims_out: Array to store the calculated chunk sizes.
 */
void calculate_chunk_dims(size_t target_bytes, int rank, const hsize_t *global_dims, hsize_t *chunk_dims_out);

/**
 * Create an HDF5 file and a chunked dataset optimized for TBLIS.
 * filename: Path to the HDF5 file to create.
 * dataset_name: Name of the dataset within the file.
 * rank: Number of dimensions.
 * global_dims: The full shape of the tensor.
 */
void create_chunked_dataset(const char *filename, const char *dataset_name, int rank, const hsize_t *global_dims);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_STORE_H