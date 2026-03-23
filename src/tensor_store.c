#include "tensor_store.h"
/* registry.h is transitively included via tensor_store.h */
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

/* ----------------------------------------------------------------------- */
/* Public helpers                                                           */
/* ----------------------------------------------------------------------- */

void get_physical_offset(int rank, const hsize_t *tile_coords,
                         const hsize_t *chunk_dims,
                         hsize_t *phys_offset_out)
{
    for (int d = 0; d < rank; d++)
        phys_offset_out[d] = tile_coords[d] * chunk_dims[d];
}

void calculate_chunk_dims(size_t target_bytes, int rank,
                          const hsize_t *global_dims,
                          hsize_t *chunk_dims_out)
{
    /* Round target up to 16 KB so chunk byte sizes are NVMe-page aligned. */
    const size_t nvme_page = 16384;
    target_bytes = (target_bytes + nvme_page - 1) & ~(nvme_page - 1);

    size_t total_elems = target_bytes / sizeof(double);
    double side = pow((double)total_elems, 1.0 / (double)rank);
    hsize_t side_int = (hsize_t)round(side);
    if (side_int < 1) side_int = 1;

    for (int d = 0; d < rank; d++)
        chunk_dims_out[d] = (side_int > global_dims[d]) ? global_dims[d]
                                                        : side_int;
}

/* ----------------------------------------------------------------------- */
/* dset_open_no_cache                                                       */
/* ----------------------------------------------------------------------- */

hid_t dset_open_no_cache(hid_t file_id, const char *dset_name)
{
    hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
    if (dapl_id < 0) return -1;

    /*
     * Disable the HDF5-internal chunk cache (nslots=0, nbytes=0, w0 unused).
     * The engine manages its own BufferPool; a second cache layer wastes RAM
     * and competes for the same pages.
     */
    if (H5Pset_chunk_cache(dapl_id, 0, 0, 0.0) < 0) {
        H5Pclose(dapl_id);
        return -1;
    }

    hid_t dset_id = H5Dopen2(file_id, dset_name, dapl_id);
    H5Pclose(dapl_id);
    return dset_id;
}

/* ----------------------------------------------------------------------- */
/* create_chunked_dataset                                                   */
/* ----------------------------------------------------------------------- */

/* Shared body: create an HDF5 file with one chunked dataset. */
static herr_t create_chunked_impl(const char *filename,
                                   const char *dataset_name,
                                   int rank,
                                   const hsize_t *global_dims,
                                   const hsize_t *chunk_dims)
{
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "create_chunked_dataset: H5Fcreate failed for '%s'\n",
                filename);
        return -1;
    }

    hid_t space_id = H5Screate_simple(rank, global_dims, NULL);
    if (space_id < 0) { H5Fclose(file_id); return -1; }

    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    if (dcpl_id < 0) { H5Sclose(space_id); H5Fclose(file_id); return -1; }

    if (H5Pset_chunk(dcpl_id, rank, chunk_dims) < 0) {
        H5Pclose(dcpl_id); H5Sclose(space_id); H5Fclose(file_id);
        return -1;
    }

    double fill_value = 0.0;
    H5Pset_fill_value(dcpl_id, H5T_NATIVE_DOUBLE, &fill_value);
    H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_INCR);

    hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
    if (dapl_id < 0) {
        H5Pclose(dcpl_id); H5Sclose(space_id); H5Fclose(file_id);
        return -1;
    }
    H5Pset_chunk_cache(dapl_id, 0, 0, 0.0);

    hid_t dset_id = H5Dcreate2(file_id, dataset_name, H5T_NATIVE_DOUBLE,
                                space_id, H5P_DEFAULT, dcpl_id, dapl_id);
    H5Pclose(dapl_id);
    H5Pclose(dcpl_id);
    H5Sclose(space_id);

    if (dset_id < 0) {
        fprintf(stderr,
                "create_chunked_dataset: H5Dcreate2 failed for '%s'\n",
                dataset_name);
        H5Fclose(file_id);
        return -1;
    }

    H5Dclose(dset_id);
    H5Fclose(file_id);
    return 0;
}

herr_t create_chunked_dataset(const char *filename, const char *dataset_name,
                              int rank, const hsize_t *global_dims,
                              size_t target_chunk_bytes)
{
    hsize_t chunk_dims[MAX_RANK];
    calculate_chunk_dims(target_chunk_bytes, rank, global_dims, chunk_dims);
    return create_chunked_impl(filename, dataset_name, rank,
                               global_dims, chunk_dims);
}

herr_t create_chunked_dataset_explicit(const char *filename,
                                       const char *dataset_name,
                                       int rank,
                                       const hsize_t *global_dims,
                                       const hsize_t *chunk_dims)
{
    return create_chunked_impl(filename, dataset_name, rank,
                               global_dims, chunk_dims);
}

/* ----------------------------------------------------------------------- */
/* Internal: compute actual (boundary-clamped) dims and is_partial flag    */
/* ----------------------------------------------------------------------- */

static void compute_actual_dims(int rank,
                                const hsize_t *file_dims,
                                const hsize_t *phys_offset,
                                const hsize_t *chunk_dims,
                                hsize_t *actual_dims_out,
                                int *is_partial_out)
{
    *is_partial_out = 0;
    for (int d = 0; d < rank; d++) {
        actual_dims_out[d] = chunk_dims[d];
        hsize_t end = phys_offset[d] + chunk_dims[d];
        if (end > file_dims[d]) {
            actual_dims_out[d] = file_dims[d] - phys_offset[d];
            *is_partial_out = 1;
        }
    }
}

/* ----------------------------------------------------------------------- */
/* read_chunk_fast                                                          */
/* ----------------------------------------------------------------------- */

herr_t read_chunk_fast(hid_t dset_id, const hsize_t *phys_offset,
                       double *data_ptr, int rank,
                       const hsize_t *chunk_dims)
{
    hsize_t file_dims[MAX_RANK];
    hsize_t actual_dims[MAX_RANK];
    hsize_t zero_offset[MAX_RANK];
    int     is_partial;

    memset(zero_offset, 0, sizeof(zero_offset));

    hid_t filespace_id = H5Dget_space(dset_id);
    if (filespace_id < 0) return -1;

    if (H5Sget_simple_extent_dims(filespace_id, file_dims, NULL) < 0) {
        H5Sclose(filespace_id);
        return -1;
    }

    compute_actual_dims(rank, file_dims, phys_offset, chunk_dims,
                        actual_dims, &is_partial);

    /*
     * For boundary tiles: pre-zero the entire nominal page so that the
     * unread padding region never carries stale data into the compute kernel.
     */
    if (is_partial) {
        size_t nominal_count = 1;
        for (int d = 0; d < rank; d++)
            nominal_count *= (size_t)chunk_dims[d];
        memset(data_ptr, 0, nominal_count * sizeof(double));
    }

    herr_t status = H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
                                        phys_offset, NULL,
                                        actual_dims, NULL);
    if (status < 0) { H5Sclose(filespace_id); return -1; }

    /*
     * Memory dataspace = full nominal chunk.
     * For partial tiles, select only the actual subregion at the origin so
     * file data lands at the correct row-major positions within the larger
     * buffer.  The compute kernel can then use chunk_dims strides uniformly
     * for both full and boundary tiles.
     */
    hid_t memspace_id = H5Screate_simple(rank, chunk_dims, NULL);
    if (memspace_id < 0) { H5Sclose(filespace_id); return -1; }

    if (is_partial) {
        status = H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET,
                                     zero_offset, NULL,
                                     actual_dims, NULL);
        if (status < 0) {
            H5Sclose(memspace_id);
            H5Sclose(filespace_id);
            return -1;
        }
    }

    status = H5Dread(dset_id, H5T_NATIVE_DOUBLE,
                     memspace_id, filespace_id,
                     H5P_DEFAULT, data_ptr);

    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    return status;
}

/* ----------------------------------------------------------------------- */
/* write_chunk_fast                                                         */
/* ----------------------------------------------------------------------- */

herr_t write_chunk_fast(hid_t dset_id, const hsize_t *phys_offset,
                        const double *data_ptr, int rank,
                        const hsize_t *chunk_dims)
{
    hsize_t file_dims[MAX_RANK];
    hsize_t actual_dims[MAX_RANK];
    hsize_t zero_offset[MAX_RANK];
    int     is_partial;

    memset(zero_offset, 0, sizeof(zero_offset));

    hid_t filespace_id = H5Dget_space(dset_id);
    if (filespace_id < 0) return -1;

    if (H5Sget_simple_extent_dims(filespace_id, file_dims, NULL) < 0) {
        H5Sclose(filespace_id);
        return -1;
    }

    compute_actual_dims(rank, file_dims, phys_offset, chunk_dims,
                        actual_dims, &is_partial);

    herr_t status = H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
                                        phys_offset, NULL,
                                        actual_dims, NULL);
    if (status < 0) { H5Sclose(filespace_id); return -1; }

    /*
     * Memory dataspace = full nominal chunk.
     * For partial tiles, select the valid subregion at the origin so that
     * data_ptr[row * chunk_dims[1] + col] (the canonical buffer layout) is
     * correctly extracted and written.
     */
    hid_t memspace_id = H5Screate_simple(rank, chunk_dims, NULL);
    if (memspace_id < 0) { H5Sclose(filespace_id); return -1; }

    if (is_partial) {
        status = H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET,
                                     zero_offset, NULL,
                                     actual_dims, NULL);
        if (status < 0) {
            H5Sclose(memspace_id);
            H5Sclose(filespace_id);
            return -1;
        }
    }

    status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE,
                      memspace_id, filespace_id,
                      H5P_DEFAULT, data_ptr);

    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    return status;
}

/* ----------------------------------------------------------------------- */
/* create_h5_complex_type                                                  */
/* ----------------------------------------------------------------------- */

hid_t create_h5_complex_type(void)
{
    /*
     * C99 guarantees double _Complex is laid out as two consecutive doubles:
     *   real part  at byte offset 0
     *   imag part  at byte offset sizeof(double)
     * This compound type mirrors that layout exactly.
     */
    hid_t type = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
    if (type < 0) return -1;

    if (H5Tinsert(type, "r", 0,              H5T_NATIVE_DOUBLE) < 0 ||
        H5Tinsert(type, "i", sizeof(double), H5T_NATIVE_DOUBLE) < 0) {
        H5Tclose(type);
        return -1;
    }
    return type;
}

/* ----------------------------------------------------------------------- */
/* read_chunk_typed                                                         */
/* ----------------------------------------------------------------------- */

herr_t read_chunk_typed(hid_t dset_id, const hsize_t *phys_offset,
                        void *data_ptr, size_t element_size,
                        int rank, const hsize_t *chunk_dims,
                        hid_t mem_type)
{
    hsize_t file_dims[MAX_RANK];
    hsize_t actual_dims[MAX_RANK];
    hsize_t zero_offset[MAX_RANK];
    int     is_partial;

    memset(zero_offset, 0, sizeof(zero_offset));

    hid_t filespace_id = H5Dget_space(dset_id);
    if (filespace_id < 0) return -1;

    if (H5Sget_simple_extent_dims(filespace_id, file_dims, NULL) < 0) {
        H5Sclose(filespace_id);
        return -1;
    }

    compute_actual_dims(rank, file_dims, phys_offset, chunk_dims,
                        actual_dims, &is_partial);

    /* Pre-zero the full nominal buffer for boundary tiles. */
    if (is_partial) {
        size_t nominal_count = 1;
        for (int d = 0; d < rank; d++)
            nominal_count *= (size_t)chunk_dims[d];
        memset(data_ptr, 0, nominal_count * element_size);
    }

    herr_t status = H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
                                        phys_offset, NULL,
                                        actual_dims, NULL);
    if (status < 0) { H5Sclose(filespace_id); return -1; }

    hid_t memspace_id = H5Screate_simple(rank, chunk_dims, NULL);
    if (memspace_id < 0) { H5Sclose(filespace_id); return -1; }

    if (is_partial) {
        status = H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET,
                                     zero_offset, NULL,
                                     actual_dims, NULL);
        if (status < 0) {
            H5Sclose(memspace_id); H5Sclose(filespace_id);
            return -1;
        }
    }

    status = H5Dread(dset_id, mem_type, memspace_id, filespace_id,
                     H5P_DEFAULT, data_ptr);

    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    return status;
}

/* ----------------------------------------------------------------------- */
/* write_chunk_typed                                                        */
/* ----------------------------------------------------------------------- */

herr_t write_chunk_typed(hid_t dset_id, const hsize_t *phys_offset,
                         const void *data_ptr, size_t element_size,
                         int rank, const hsize_t *chunk_dims,
                         hid_t mem_type)
{
    hsize_t file_dims[MAX_RANK];
    hsize_t actual_dims[MAX_RANK];
    hsize_t zero_offset[MAX_RANK];
    int     is_partial;

    (void)element_size;   /* not needed for writes; kept for API symmetry */
    memset(zero_offset, 0, sizeof(zero_offset));

    hid_t filespace_id = H5Dget_space(dset_id);
    if (filespace_id < 0) return -1;

    if (H5Sget_simple_extent_dims(filespace_id, file_dims, NULL) < 0) {
        H5Sclose(filespace_id);
        return -1;
    }

    compute_actual_dims(rank, file_dims, phys_offset, chunk_dims,
                        actual_dims, &is_partial);

    herr_t status = H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
                                        phys_offset, NULL,
                                        actual_dims, NULL);
    if (status < 0) { H5Sclose(filespace_id); return -1; }

    hid_t memspace_id = H5Screate_simple(rank, chunk_dims, NULL);
    if (memspace_id < 0) { H5Sclose(filespace_id); return -1; }

    if (is_partial) {
        status = H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET,
                                     zero_offset, NULL,
                                     actual_dims, NULL);
        if (status < 0) {
            H5Sclose(memspace_id); H5Sclose(filespace_id);
            return -1;
        }
    }

    status = H5Dwrite(dset_id, mem_type, memspace_id, filespace_id,
                      H5P_DEFAULT, data_ptr);

    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    return status;
}

/* ----------------------------------------------------------------------- */
/* create_chunked_dataset_einsum                                            */
/* ----------------------------------------------------------------------- */

herr_t create_chunked_dataset_einsum(const char *filename,
                                     const char *dataset_name,
                                     int rank,
                                     const hsize_t *global_dims,
                                     const hsize_t *chunk_dims,
                                     tensor_dtype_t dtype)
{
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr,
                "create_chunked_dataset_einsum: H5Fcreate failed for '%s'\n",
                filename);
        return -1;
    }

    hid_t space_id = H5Screate_simple(rank, global_dims, NULL);
    if (space_id < 0) { H5Fclose(file_id); return -1; }

    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    if (dcpl_id < 0) { H5Sclose(space_id); H5Fclose(file_id); return -1; }

    if (H5Pset_chunk(dcpl_id, rank, chunk_dims) < 0) {
        H5Pclose(dcpl_id); H5Sclose(space_id); H5Fclose(file_id);
        return -1;
    }
    if (dtype == DTYPE_FP64) {
        double fill = 0.0;
        H5Pset_fill_value(dcpl_id, H5T_NATIVE_DOUBLE, &fill);
    }
    H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_INCR);

    hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
    if (dapl_id < 0) {
        H5Pclose(dcpl_id); H5Sclose(space_id); H5Fclose(file_id);
        return -1;
    }
    H5Pset_chunk_cache(dapl_id, 0, 0, 0.0);

    /* Select HDF5 data type. */
    hid_t h5type;
    int   close_type = 0;
    if (dtype == DTYPE_COMPLEX128) {
        h5type = create_h5_complex_type();
        if (h5type < 0) {
            H5Pclose(dapl_id); H5Pclose(dcpl_id);
            H5Sclose(space_id); H5Fclose(file_id);
            return -1;
        }
        close_type = 1;
    } else {
        h5type = H5T_NATIVE_DOUBLE;
    }

    hid_t dset_id = H5Dcreate2(file_id, dataset_name, h5type,
                                space_id, H5P_DEFAULT, dcpl_id, dapl_id);
    if (close_type) H5Tclose(h5type);
    H5Pclose(dapl_id);
    H5Pclose(dcpl_id);
    H5Sclose(space_id);

    if (dset_id < 0) {
        fprintf(stderr,
                "create_chunked_dataset_einsum: H5Dcreate2 failed for '%s'\n",
                dataset_name);
        H5Fclose(file_id);
        return -1;
    }

    H5Dclose(dset_id);
    H5Fclose(file_id);
    return 0;
}
