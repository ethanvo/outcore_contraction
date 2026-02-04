#ifndef TENSOR_STORE_H
#define TENSOR_STORE_H

#include <hdf5.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Helper: Converts Logical Tile Coordinates (e.g., 0, 1, 5)
 * into Physical Element Offsets (e.g., 0, 64, 320)
 */
void get_physical_offset(int rank, const hsize_t *tile_coords,
                         const hsize_t *chunk_dims, hsize_t *phys_offset_out);

/**
 * Helper: Calculate chunk dimensions based on target cache size.
 * target_bytes: Desired size of one chunk in bytes (e.g., 2MB).
 * rank: Number of dimensions (e.g., 3 for a tensor)
 * global_dims: The full shape of the tensor.
 * chunk_dims_out: Array to store the calculated chunk sizes.
 */
void calculate_chunk_dims(size_t target_bytes, int rank,
                          const hsize_t *global_dims, hsize_t *chunk_dims_out);

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
                       double *data_ptr, int rank, const hsize_t *chunk_dims);

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
                        const hsize_t *chunk_dims);

/**
 * Create an HDF5 file and a chunked dataset optimized for TBLIS.
 * filename: Path to the HDF5 file to create.
 * dataset_name: Name of the dataset within the file.
 * rank: Number of dimensions.
 * global_dims: The full shape of the tensor.
 */
void create_chunked_dataset(const char *filename, const char *dataset_name,
                            int rank, const hsize_t *global_dims);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_STORE_H
