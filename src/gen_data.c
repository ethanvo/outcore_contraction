#include "tensor_store.h"
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>

void generate_matrix(const char *fname, const char *dset, int rows, int cols,
                     double value_factor) {
  hsize_t global_dims[] = {rows, cols};
  create_chunked_dataset(fname, dset, 2, global_dims);

  hid_t file = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dset_id = H5Dopen2(file, dset, H5P_DEFAULT);

  hsize_t chunk_dims[2];
  calculate_chunk_dims(2 * 1024 * 1024, 2, global_dims, chunk_dims);

  size_t elems = chunk_dims[0] * chunk_dims[1];
  double *data = malloc(elems * sizeof(double));

  // Fill logic: Iterate over tiles and write them
  int grid_r = (rows + chunk_dims[0] - 1) / chunk_dims[0];
  int grid_c = (cols + chunk_dims[1] - 1) / chunk_dims[1];

  for (int i = 0; i < grid_r; i++) {
    for (int j = 0; j < grid_c; j++) {
      // Simple fill: 1.0 everywhere (easy to verify math)
      // C = A(1.0) * B(1.0) -> C element should be N (inner dim size)
      for (size_t k = 0; k < elems; k++)
        data[k] = value_factor;

      hsize_t offset[2] = {i * chunk_dims[0], j * chunk_dims[1]};
      write_chunk_fast(dset_id, offset, data, 2, chunk_dims);
    }
  }

  free(data);
  H5Dclose(dset_id);
  H5Fclose(file);
  printf("Generated %s\n", fname);
}

int main() {
  // Generate A and B as 300x300 matrices
  generate_matrix("A.h5", "MatrixA", 300, 300, 1.0);
  generate_matrix("B.h5", "MatrixB", 300, 300, 1.0);
  return 0;
}
