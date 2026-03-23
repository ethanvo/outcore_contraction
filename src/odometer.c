#include "odometer.h"
#include <string.h>   /* memset */

/* ----------------------------------------------------------------------- */
/* odometer_step                                                            */
/* ----------------------------------------------------------------------- */

bool odometer_step(size_t rank, size_t *coords, const size_t *extents)
{
    if (rank == 0) return false;

    /*
     * Walk dimensions from rightmost to leftmost.
     * Increment, and if within bounds we are done.
     * If we overflow, zero this dimension and carry to the next.
     * If the carry propagates past dimension 0 the space is exhausted.
     */
    size_t d = rank;
    while (d-- > 0) {
        coords[d] += 1;
        if (coords[d] < extents[d])
            return true;   /* no carry — more elements remain */
        coords[d] = 0;     /* overflow — propagate carry */
    }
    /* All dimensions rolled over: back to all-zeros, space exhausted. */
    return false;
}

/* ----------------------------------------------------------------------- */
/* compute_strides                                                          */
/* ----------------------------------------------------------------------- */

void compute_strides(size_t rank, const size_t *dims, size_t *strides_out)
{
    if (rank == 0) return;
    strides_out[rank - 1] = 1;
    /* Iterate right-to-left without unsigned underflow */
    size_t d = rank - 1;
    while (d-- > 0)
        strides_out[d] = dims[d + 1] * strides_out[d + 1];
}

/* ----------------------------------------------------------------------- */
/* compute_flat_index                                                       */
/* ----------------------------------------------------------------------- */

size_t compute_flat_index(size_t rank,
                          const size_t *coords,
                          const size_t *strides)
{
    size_t idx = 0;
    for (size_t d = 0; d < rank; d++)
        idx += coords[d] * strides[d];
    return idx;
}

/* ----------------------------------------------------------------------- */
/* tensor_permute                                                           */
/* ----------------------------------------------------------------------- */

void tensor_permute(const void   *src,
                    void         *dst,
                    size_t        rank,
                    const size_t *physical_extents,
                    const size_t *nominal_chunk_dims,
                    const int    *perm,
                    size_t        element_size)
{
    const char *src_bytes = (const char *)src;
    char       *dst_bytes = (char       *)dst;

    /* Scalar (rank-0): copy one element. */
    if (rank == 0) {
        memcpy(dst_bytes, src_bytes, element_size);
        return;
    }

    /* --- Strides for the source buffer (nominal, row-major) ------------ */
    size_t src_strides[MAX_RANK];
    compute_strides(rank, nominal_chunk_dims, src_strides);

    /* --- Permuted nominal dims → strides for the destination buffer ---- */
    size_t perm_nominal_dims[MAX_RANK];
    for (size_t i = 0; i < rank; i++)
        perm_nominal_dims[i] = nominal_chunk_dims[(size_t)perm[i]];

    size_t dst_strides[MAX_RANK];
    compute_strides(rank, perm_nominal_dims, dst_strides);

    /* --- Traverse the physical (clamped) sub-volume -------------------- */
    size_t coords[MAX_RANK];
    memset(coords, 0, rank * sizeof(size_t));

    do {
        /* 1. Source flat index using nominal strides. */
        size_t src_idx = compute_flat_index(rank, coords, src_strides);

        /* 2. Permuted coordinate tuple: dst_coord[i] = src_coord[perm[i]] */
        size_t perm_coords[MAX_RANK];
        for (size_t i = 0; i < rank; i++)
            perm_coords[i] = coords[(size_t)perm[i]];

        /* 3. Destination flat index using permuted nominal strides. */
        size_t dst_idx = compute_flat_index(rank, perm_coords, dst_strides);

        /* 4. Copy one element (works for any element_size). */
        memcpy(dst_bytes + dst_idx * element_size,
               src_bytes + src_idx * element_size,
               element_size);

    } while (odometer_step(rank, coords, physical_extents));
}
