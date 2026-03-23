#ifndef TENSOR_STORE_H
#define TENSOR_STORE_H

#include "registry.h"   /* tensor_dtype_t, MAX_RANK */
#include <hdf5.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Convert logical tile coordinates into physical element offsets.
 * phys_offset_out[d] = tile_coords[d] * chunk_dims[d]
 */
void get_physical_offset(int rank, const hsize_t *tile_coords,
                         const hsize_t *chunk_dims,
                         hsize_t *phys_offset_out);

/*
 * Calculate isotropic chunk dimensions that target target_bytes per chunk.
 * Each dimension gets the same side length: round(nthroot(target_bytes/8, rank)),
 * clamped to the corresponding global_dims entry.
 */
void calculate_chunk_dims(size_t target_bytes, int rank,
                          const hsize_t *global_dims,
                          hsize_t *chunk_dims_out);

/*
 * Read one chunk from an already-open dataset into data_ptr.
 *
 * The destination buffer must be sized for the full nominal chunk
 * (product of chunk_dims elements).  For boundary tiles where the actual
 * extent is smaller than chunk_dims, the buffer is pre-zeroed and data is
 * placed at the correct row-major positions so that the compute kernel can
 * use chunk_dims strides uniformly.
 *
 * Returns 0 on success, -1 on error.
 */
herr_t read_chunk_fast(hid_t dset_id, const hsize_t *phys_offset,
                       double *data_ptr, int rank,
                       const hsize_t *chunk_dims);

/*
 * Write one chunk from data_ptr to an already-open dataset.
 *
 * data_ptr is treated as a nominal chunk_dims-strided buffer.  For boundary
 * tiles, only the valid subregion (clamped to the dataset extent) is written.
 *
 * Returns 0 on success, -1 on error.
 */
herr_t write_chunk_fast(hid_t dset_id, const hsize_t *phys_offset,
                        const double *data_ptr, int rank,
                        const hsize_t *chunk_dims);

/*
 * Create a new HDF5 file containing one chunked, double-precision dataset.
 * Chunk size is derived from target_chunk_bytes.
 * Incremental allocation is used so unwritten chunks consume no disk space
 * (block-sparse layout).
 *
 * Returns 0 on success, -1 on failure.
 */
herr_t create_chunked_dataset(const char *filename, const char *dataset_name,
                              int rank, const hsize_t *global_dims,
                              size_t target_chunk_bytes);

/*
 * Like create_chunked_dataset but takes explicit chunk_dims instead of a
 * target byte budget.  Use this when chunk boundaries must align exactly
 * with those of another tensor (e.g. the C output of a SUMMA contraction
 * must share i/j chunk boundaries with A and k/l boundaries with B).
 */
herr_t create_chunked_dataset_explicit(const char *filename,
                                       const char *dataset_name,
                                       int rank,
                                       const hsize_t *global_dims,
                                       const hsize_t *chunk_dims);

/*
 * Open a dataset with the HDF5-internal chunk cache fully disabled.
 * Call this instead of H5Dopen2(..., H5P_DEFAULT) whenever the engine manages
 * its own BufferPool, to avoid double-caching and wasted memory.
 *
 * The returned hid_t must be closed with H5Dclose().
 * Returns a negative value on failure.
 */
hid_t dset_open_no_cache(hid_t file_id, const char *dset_name);

/* ----------------------------------------------------------------------- */
/* Complex type support                                                     */
/* ----------------------------------------------------------------------- */

/*
 * Create an HDF5 compound type representing double _Complex (16 bytes):
 *   field "r" at offset 0            — H5T_NATIVE_DOUBLE (real part)
 *   field "i" at offset sizeof(double) — H5T_NATIVE_DOUBLE (imaginary part)
 *
 * The returned hid_t must be closed by the caller with H5Tclose().
 * Returns a negative value on failure.
 */
hid_t create_h5_complex_type(void);

/* ----------------------------------------------------------------------- */
/* Typed I/O (generic element type)                                        */
/* ----------------------------------------------------------------------- */

/*
 * read_chunk_typed / write_chunk_typed — like read_chunk_fast /
 * write_chunk_fast but accept arbitrary element types.
 *
 *   data_ptr     : raw buffer pointer (cast to void*).
 *   element_size : sizeof per element (e.g. sizeof(double) or
 *                  sizeof(double _Complex)).
 *   mem_type     : HDF5 memory type to use (H5T_NATIVE_DOUBLE for FP64,
 *                  or the result of create_h5_complex_type() for COMPLEX128).
 *                  The caller owns the type lifecycle.
 *
 * All other semantics identical to read_chunk_fast / write_chunk_fast.
 */
herr_t read_chunk_typed(hid_t dset_id, const hsize_t *phys_offset,
                        void *data_ptr, size_t element_size,
                        int rank, const hsize_t *chunk_dims,
                        hid_t mem_type);

herr_t write_chunk_typed(hid_t dset_id, const hsize_t *phys_offset,
                         const void *data_ptr, size_t element_size,
                         int rank, const hsize_t *chunk_dims,
                         hid_t mem_type);

/* ----------------------------------------------------------------------- */
/* Typed dataset creation (for einsum engine)                              */
/* ----------------------------------------------------------------------- */

/*
 * Like create_chunked_dataset_explicit but creates the HDF5 dataset with the
 * native type for dtype (H5T_NATIVE_DOUBLE for DTYPE_FP64, or a compound
 * double-pair type for DTYPE_COMPLEX128).
 *
 * Use this function when the engine needs to create an output tensor whose
 * element type is determined at runtime.  The chunk dims must be supplied
 * explicitly so they can be aligned with the input registries.
 */
herr_t create_chunked_dataset_einsum(const char *filename,
                                     const char *dataset_name,
                                     int rank,
                                     const hsize_t *global_dims,
                                     const hsize_t *chunk_dims,
                                     tensor_dtype_t dtype);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_STORE_H */
