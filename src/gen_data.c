#include "engine.h"
#include "tensor_store.h"
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Generate a rank-2 (rows×cols) HDF5 tensor whose every element is set to
 * fill_value.  Tiles are written through the tensor_store layer using the
 * nominal chunk_dims strides, letting write_chunk_fast handle boundary
 * clamping internally.
 */
static int generate_matrix(const char *fname, const char *dset_name,
                            hsize_t rows, hsize_t cols, double fill_value,
                            size_t chunk_bytes)
{
    hsize_t global_dims[2] = {rows, cols};

    if (create_chunked_dataset(fname, dset_name, 2, global_dims,
                               chunk_bytes) < 0) {
        fprintf(stderr, "generate_matrix: create_chunked_dataset failed "
                        "for '%s'\n", fname);
        return -1;
    }

    hid_t file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "generate_matrix: H5Fopen failed for '%s'\n", fname);
        return -1;
    }

    hid_t dset_id = dset_open_no_cache(file_id, dset_name);
    if (dset_id < 0) {
        fprintf(stderr, "generate_matrix: dset_open_no_cache failed\n");
        H5Fclose(file_id);
        return -1;
    }

    /* Derive chunk dims the same way create_chunked_dataset did. */
    hsize_t chunk_dims[2];
    calculate_chunk_dims(chunk_bytes, 2, global_dims, chunk_dims);

    hsize_t grid_rows = (rows + chunk_dims[0] - 1) / chunk_dims[0];
    hsize_t grid_cols = (cols + chunk_dims[1] - 1) / chunk_dims[1];

    /* Allocate one page sized for the nominal (full) chunk. */
    size_t max_elems = (size_t)chunk_dims[0] * (size_t)chunk_dims[1];
    double *buf = (double *)malloc(max_elems * sizeof(double));
    if (!buf) {
        fprintf(stderr, "generate_matrix: malloc failed\n");
        H5Dclose(dset_id);
        H5Fclose(file_id);
        return -1;
    }
    for (size_t idx = 0; idx < max_elems; idx++)
        buf[idx] = fill_value;

    int ret = 0;
    for (hsize_t ti = 0; ti < grid_rows && ret == 0; ti++) {
        for (hsize_t tj = 0; tj < grid_cols && ret == 0; tj++) {
            hsize_t offset[2] = {ti * chunk_dims[0], tj * chunk_dims[1]};
            /*
             * Pass the nominal chunk_dims: write_chunk_fast clamps to the
             * dataset boundary internally.  The buffer contains fill_value
             * everywhere, so boundary padding in the buffer is harmless.
             */
            if (write_chunk_fast(dset_id, offset, buf, 2, chunk_dims) < 0) {
                fprintf(stderr,
                        "generate_matrix: write_chunk_fast failed at "
                        "tile [%llu, %llu]\n",
                        (unsigned long long)ti, (unsigned long long)tj);
                ret = -1;
            }
        }
    }

    free(buf);
    H5Dclose(dset_id);
    H5Fclose(file_id);

    if (ret == 0)
        printf("Generated %s  shape=(%llu\xc3\x97%llu)  "
               "chunks=(%llu\xc3\x97%llu)\n",
               fname,
               (unsigned long long)rows,  (unsigned long long)cols,
               (unsigned long long)chunk_dims[0],
               (unsigned long long)chunk_dims[1]);
    return ret;
}

int main(void)
{
    /*
     * Scale chunk size with available RAM: ~1 MB per GB, minimum 2 MB.
     * On a 64 GB machine this targets ~64 MB chunks so that large matrices
     * (e.g. 10000×10000) produce a manageable tile grid.
     */
    size_t ram         = query_physical_ram();
    size_t chunk_bytes = ram / 1000;
    if (chunk_bytes < 2UL * 1024 * 1024)
        chunk_bytes = 2UL * 1024 * 1024;

    int ret = 0;
    ret |= generate_matrix("A.h5", "MatrixA", 300, 300, 1.0, chunk_bytes);
    ret |= generate_matrix("B.h5", "MatrixB", 300, 300, 1.0, chunk_bytes);
    return ret;
}
