#ifndef ODOMETER_H
#define ODOMETER_H

#include <stddef.h>
#include <stdbool.h>

/*
 * Ceiling on tensor rank; must be >= the MAX_RANK in registry.h.
 * Defined here independently so odometer.h has no HDF5 dependency.
 */
#ifndef MAX_RANK
#  define MAX_RANK 8
#endif

/* ----------------------------------------------------------------------- */
/* Odometer iterator                                                        */
/* ----------------------------------------------------------------------- */

/*
 * Advance a multi-dimensional coordinate array by one step in row-major
 * (C) order — the rightmost index increments first.
 *
 *   coords   : current coordinates, modified in-place.
 *   extents  : upper bound (exclusive) for each dimension.
 *   rank     : number of dimensions.
 *
 * Returns true  if there are more elements to visit (coords now holds the
 *               next valid coordinate tuple).
 * Returns false when the entire space is exhausted (all coords rolled back
 *               to zero — do NOT process coords again on false).
 *
 * Canonical usage:
 *
 *   size_t coords[MAX_RANK] = {0};
 *   do {
 *       process(coords);
 *   } while (odometer_step(rank, coords, extents));
 */
bool odometer_step(size_t rank, size_t *coords, const size_t *extents);

/* ----------------------------------------------------------------------- */
/* Flat-index helpers                                                       */
/* ----------------------------------------------------------------------- */

/*
 * Compute row-major strides from an array of dimension sizes.
 *
 *   strides_out[rank-1] = 1
 *   strides_out[d]      = dims[d+1] * strides_out[d+1]
 *
 * strides_out must point to at least rank elements.
 */
void compute_strides(size_t rank, const size_t *dims, size_t *strides_out);

/*
 * Compute a flat row-major index:
 *   index = sum_d( coords[d] * strides[d] )
 *
 * strides should be pre-computed with compute_strides() for memory safety.
 */
size_t compute_flat_index(size_t rank,
                          const size_t *coords,
                          const size_t *strides);

/* ----------------------------------------------------------------------- */
/* Tensor permutation                                                       */
/* ----------------------------------------------------------------------- */

/*
 * Rearrange the axes of a flat row-major tensor buffer.
 *
 *   src                : source buffer; must be sized for the full nominal
 *                        chunk (product of nominal_chunk_dims elements).
 *   dst                : destination buffer; must be sized for the full
 *                        permuted nominal chunk
 *                        (product of nominal_chunk_dims[perm[d]] elements).
 *   rank               : number of tensor dimensions.
 *   physical_extents   : actual element counts per source dimension, <= the
 *                        corresponding nominal_chunk_dims entry.  Only this
 *                        sub-volume is read from src / written to dst.
 *                        (Boundary tiles have physical_extents < nominal_chunk_dims;
 *                        the surrounding zero-padding in src is never touched.)
 *   nominal_chunk_dims : full nominal chunk size per source dimension.
 *                        Used to derive row-major strides for both src
 *                        (directly) and dst (after permuting dims by perm).
 *   perm               : perm[i] gives the source dimension that maps to
 *                        destination dimension i.
 *                        Example: perm = {2, 0, 1} means
 *                          dst_dim0 <- src_dim2
 *                          dst_dim1 <- src_dim0
 *                          dst_dim2 <- src_dim1
 *                        so dst[c_src2, c_src0, c_src1] = src[c_src0, c_src1, c_src2].
 *
 * element_size : sizeof per element — sizeof(double) for FP64,
 *                sizeof(double _Complex) for COMPLEX128.
 *                The odometer iterates over logical element indices; the
 *                actual byte copy is memcpy(dst_byte + dst_idx*element_size,
 *                                          src_byte + src_idx*element_size,
 *                                          element_size).
 *
 * No dynamic allocation; all temporary arrays are fixed-size [MAX_RANK].
 */
void tensor_permute(const void   *src,
                    void         *dst,
                    size_t        rank,
                    const size_t *physical_extents,
                    const size_t *nominal_chunk_dims,
                    const int    *perm,
                    size_t        element_size);

#endif /* ODOMETER_H */
