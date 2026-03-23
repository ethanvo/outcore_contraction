/*
 * test_high_rank.c
 *
 * Mathematical verification suite for rank-4 and rank-5 tensor operations.
 *
 * Covers:
 *   - MAX_RANK-sized TileMetadata / TensorRegistry arrays
 *   - registry_create_from_dset   (reads rank and chunk dims from HDF5 file)
 *   - registry_get_tile           (n-dimensional coords array)
 *   - registry_scan_file          (tile count and status verification)
 *   - read_chunk_fast / write_chunk_fast  (full tiles and boundary tiles)
 *   - Boundary clamping: data must appear at correct row-major positions in
 *     the nominal chunk_dims-strided buffer
 *
 * Each rank-N test follows:
 *   1. Create a small HDF5 dataset whose dimensions are NOT divisible by the
 *      chunk size, so boundary tiles appear in every dimension.
 *   2. Fill every element with its global flat index (unique, deterministic).
 *   3. Write all tiles via write_chunk_fast.
 *   4. Build a registry via registry_create_from_dset and scan the file.
 *   5. Verify the tile count and the status of specific tiles.
 *   6. Read every tile back; for each valid element verify its value equals
 *      the expected global flat index using the correct nominal-stride layout.
 */

#include "registry.h"
#include "tensor_store.h"
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Small chunk target so multiple tiles appear even for small tensors. */
#define TEST_CHUNK_BYTES (256 * 1024)   /* 256 KB */

/* ----------------------------------------------------------------------- */
/* Value pattern: global flat index as a double.                           */
/* Uniquely identifies every element in the tensor.                        */
/* ----------------------------------------------------------------------- */
static double elem_expected(int rank, const hsize_t *global_coords,
                            const hsize_t *global_dims)
{
    hsize_t flat = 0;
    for (int d = 0; d < rank; d++)
        flat = flat * global_dims[d] + global_coords[d];
    return (double)flat;
}

/* ----------------------------------------------------------------------- */
/* Fill a tile buffer (laid out with nominal chunk_dims strides) with the  */
/* expected global-flat-index values.  Out-of-bounds positions (padding    */
/* beyond the tensor boundary) are set to 0.                               */
/* ----------------------------------------------------------------------- */
static void fill_tile(double *buf, int rank,
                      const hsize_t *tile_coords,
                      const hsize_t *chunk_dims,
                      const hsize_t *global_dims)
{
    hsize_t local[MAX_RANK];
    memset(local, 0, sizeof(local));

    size_t total = 1;
    for (int d = 0; d < rank; d++)
        total *= (size_t)chunk_dims[d];

    for (size_t flat = 0; flat < total; flat++) {
        hsize_t global[MAX_RANK];
        int oob = 0;
        for (int d = 0; d < rank; d++) {
            global[d] = tile_coords[d] * chunk_dims[d] + local[d];
            if (global[d] >= global_dims[d]) oob = 1;
        }
        buf[flat] = oob ? 0.0 : elem_expected(rank, global, global_dims);

        /* Increment local in row-major order. */
        for (int d = rank - 1; d >= 0; d--) {
            if (++local[d] < chunk_dims[d]) break;
            local[d] = 0;
        }
    }
}

/* ----------------------------------------------------------------------- */
/* Verify only the valid (non-padding) region of a tile buffer.            */
/* Returns 1 if all elements match, 0 on the first mismatch.              */
/* ----------------------------------------------------------------------- */
static int verify_tile(const double *buf, int rank,
                       const hsize_t *tile_coords,
                       const hsize_t *chunk_dims,
                       const hsize_t *actual_dims,
                       const hsize_t *global_dims)
{
    hsize_t local[MAX_RANK];
    memset(local, 0, sizeof(local));

    size_t actual_total = 1;
    for (int d = 0; d < rank; d++)
        actual_total *= (size_t)actual_dims[d];

    for (size_t ai = 0; ai < actual_total; ai++) {
        /* Compute the nominal flat index (chunk_dims strides). */
        size_t nom_flat = 0;
        size_t stride   = 1;
        for (int d = rank - 1; d >= 0; d--) {
            nom_flat += (size_t)local[d] * stride;
            stride   *= (size_t)chunk_dims[d];
        }

        /* Compute expected value from global coordinates. */
        hsize_t global[MAX_RANK];
        for (int d = 0; d < rank; d++)
            global[d] = tile_coords[d] * chunk_dims[d] + local[d];

        double expected = elem_expected(rank, global, global_dims);
        if (buf[nom_flat] != expected) {
            fprintf(stderr,
                    "    MISMATCH: nom_flat=%zu  expected=%.0f  got=%.0f\n",
                    nom_flat, expected, buf[nom_flat]);
            return 0;
        }

        /* Advance local in row-major order within actual_dims. */
        for (int d = rank - 1; d >= 0; d--) {
            if (++local[d] < actual_dims[d]) break;
            local[d] = 0;
        }
    }
    return 1;
}

/* ----------------------------------------------------------------------- */
/* Core test: full write → registry scan → element-wise read/verify        */
/* ----------------------------------------------------------------------- */
static int run_roundtrip(int rank, const hsize_t *global_dims,
                         const char *fname)
{
    const char *dset_name = "tensor";

    printf("  shape=(");
    for (int d = 0; d < rank; d++)
        printf(d ? "×%llu" : "%llu", (unsigned long long)global_dims[d]);
    printf(")\n");

    /* Create the HDF5 dataset. */
    if (create_chunked_dataset(fname, dset_name, rank, global_dims,
                               TEST_CHUNK_BYTES) < 0) {
        fprintf(stderr, "  FAIL: create_chunked_dataset\n");
        return 1;
    }

    hid_t file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "  FAIL: H5Fopen\n"); return 1; }

    hid_t dset_id = dset_open_no_cache(file_id, dset_name);
    if (dset_id < 0) {
        H5Fclose(file_id);
        fprintf(stderr, "  FAIL: dset_open_no_cache\n");
        return 1;
    }

    /* Read chunk dims from the file — this is the source of truth. */
    hid_t dcpl_id = H5Dget_create_plist(dset_id);
    hsize_t chunk_dims[MAX_RANK];
    H5Pget_chunk(dcpl_id, rank, chunk_dims);
    H5Pclose(dcpl_id);

    printf("  chunk_dims=(");
    for (int d = 0; d < rank; d++)
        printf(d ? "×%llu" : "%llu", (unsigned long long)chunk_dims[d]);
    printf(")\n");

    /* Compute grid and allocate tile I/O buffer. */
    hsize_t grid_dims[MAX_RANK];
    size_t  total_tiles = 1;
    size_t  max_elems   = 1;
    for (int d = 0; d < rank; d++) {
        grid_dims[d]  = (global_dims[d] + chunk_dims[d] - 1) / chunk_dims[d];
        total_tiles  *= (size_t)grid_dims[d];
        max_elems    *= (size_t)chunk_dims[d];
    }
    printf("  grid=(");
    for (int d = 0; d < rank; d++)
        printf(d ? "×%llu" : "%llu", (unsigned long long)grid_dims[d]);
    printf(")  total_tiles=%zu\n", total_tiles);

    double *buf = (double *)malloc(max_elems * sizeof(double));
    if (!buf) {
        fprintf(stderr, "  FAIL: malloc(%zu)\n",
                max_elems * sizeof(double));
        H5Dclose(dset_id); H5Fclose(file_id);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* WRITE PASS: iterate all tiles in row-major order                    */
    /* ------------------------------------------------------------------ */
    hsize_t tile_coords[MAX_RANK];
    memset(tile_coords, 0, sizeof(tile_coords));
    int failures = 0;

    for (size_t t = 0; t < total_tiles; t++) {
        hsize_t phys_offset[MAX_RANK];
        for (int d = 0; d < rank; d++)
            phys_offset[d] = tile_coords[d] * chunk_dims[d];

        fill_tile(buf, rank, tile_coords, chunk_dims, global_dims);

        if (write_chunk_fast(dset_id, phys_offset, buf,
                             rank, chunk_dims) < 0) {
            fprintf(stderr, "  FAIL: write_chunk_fast at tile %zu\n", t);
            failures++;
            break;
        }

        /* Advance tile_coords in row-major order. */
        for (int d = rank - 1; d >= 0; d--) {
            if (++tile_coords[d] < grid_dims[d]) break;
            tile_coords[d] = 0;
        }
    }

    if (failures) {
        free(buf); H5Dclose(dset_id); H5Fclose(file_id);
        remove(fname);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* REGISTRY PASS: verify tile count and n-dimensional lookup           */
    /* ------------------------------------------------------------------ */
    TensorRegistry *reg = registry_create_from_dset(dset_id);
    if (!reg) {
        fprintf(stderr, "  FAIL: registry_create_from_dset\n");
        free(buf); H5Dclose(dset_id); H5Fclose(file_id); remove(fname);
        return 1;
    }

    /* Verify registry read the correct rank and dims. */
    if (reg->rank != rank) {
        fprintf(stderr, "  FAIL: registry rank=%d expected %d\n",
                reg->rank, rank);
        registry_destroy(reg);
        free(buf); H5Dclose(dset_id); H5Fclose(file_id); remove(fname);
        return 1;
    }

    long found = registry_scan_file(dset_id, reg);
    if ((size_t)found != total_tiles) {
        fprintf(stderr,
                "  FAIL: registry_scan_file found %ld tiles, expected %zu\n",
                found, total_tiles);
        registry_destroy(reg);
        free(buf); H5Dclose(dset_id); H5Fclose(file_id); remove(fname);
        return 1;
    }
    printf("  Registry scan: %ld/%zu tiles on disk  (OK)\n",
           found, total_tiles);

    /* Spot-check: first tile should be ON_DISK. */
    {
        hsize_t first[MAX_RANK];
        memset(first, 0, sizeof(first));
        TileMetadata *m = registry_get_tile(reg, first);
        if (!m || m->status != TILE_STATUS_ON_DISK) {
            fprintf(stderr,
                    "  FAIL: first tile not marked ON_DISK\n");
            registry_destroy(reg);
            free(buf); H5Dclose(dset_id); H5Fclose(file_id); remove(fname);
            return 1;
        }
    }

    /* Spot-check: out-of-bounds coords should return NULL. */
    {
        hsize_t oob[MAX_RANK];
        for (int d = 0; d < rank; d++) oob[d] = grid_dims[d]; /* one past end */
        if (registry_get_tile(reg, oob) != NULL) {
            fprintf(stderr,
                    "  FAIL: out-of-bounds registry_get_tile returned non-NULL\n");
            registry_destroy(reg);
            free(buf); H5Dclose(dset_id); H5Fclose(file_id); remove(fname);
            return 1;
        }
    }

    /* ------------------------------------------------------------------ */
    /* READ & VERIFY PASS                                                  */
    /* ------------------------------------------------------------------ */
    memset(tile_coords, 0, sizeof(tile_coords));

    for (size_t t = 0; t < total_tiles && failures == 0; t++) {
        hsize_t phys_offset[MAX_RANK];
        hsize_t actual_dims[MAX_RANK];
        for (int d = 0; d < rank; d++) {
            phys_offset[d] = tile_coords[d] * chunk_dims[d];
            hsize_t end    = phys_offset[d] + chunk_dims[d];
            actual_dims[d] = (end > global_dims[d])
                             ? global_dims[d] - phys_offset[d]
                             : chunk_dims[d];
        }

        /* Verify registry_get_tile returns the correct tile. */
        TileMetadata *meta = registry_get_tile(reg, tile_coords);
        if (!meta || meta->status != TILE_STATUS_ON_DISK) {
            fprintf(stderr,
                    "  FAIL: registry_get_tile returned bad status at "
                    "tile %zu\n", t);
            failures++;
            break;
        }

        /* Poison the buffer so stale data is detectable. */
        for (size_t e = 0; e < max_elems; e++) buf[e] = -1.0;

        if (read_chunk_fast(dset_id, phys_offset, buf,
                            rank, chunk_dims) < 0) {
            fprintf(stderr, "  FAIL: read_chunk_fast at tile %zu\n", t);
            failures++;
            break;
        }

        if (!verify_tile(buf, rank, tile_coords,
                         chunk_dims, actual_dims, global_dims)) {
            fprintf(stderr, "  FAIL: element mismatch in tile %zu\n", t);
            failures++;
        }

        /* Advance tile_coords. */
        for (int d = rank - 1; d >= 0; d--) {
            if (++tile_coords[d] < grid_dims[d]) break;
            tile_coords[d] = 0;
        }
    }

    registry_destroy(reg);
    free(buf);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    remove(fname);

    if (failures == 0)
        printf("  All %zu tiles verified element-wise  (PASS)\n", total_tiles);

    return failures ? 1 : 0;
}

/* ----------------------------------------------------------------------- */
/* main                                                                     */
/* ----------------------------------------------------------------------- */

int main(void)
{
    int result = 0;

    printf("=== High-Rank Tensor Test Suite ===\n\n");

    /*
     * Rank 4: shape (33×27×21×15)
     * With 256 KB target: 4th root(32768) ≈ 13.4 → round to 13.
     * chunk_dims = (13,13,13,13).
     * grid_dims  = (3,3,2,2) = 36 tiles.
     * Boundary tile in every dimension (dim0: 33-26=7, dim1: 27-26=1,
     * dim2: 21-13=8, dim3: 15-13=2).
     */
    printf("--- Test 1: Rank-4 (33×27×21×15) ---\n");
    {
        hsize_t shape4[] = {33, 27, 21, 15};
        result |= run_roundtrip(4, shape4, "test_rank4.h5");
    }

    printf("\n");

    /*
     * Rank 5: shape (15×12×18×10×14)
     * With 256 KB target: 5th root(32768) = exactly 8.
     * chunk_dims = (8,8,8,8,8).
     * grid_dims  = (2,2,3,2,2) = 48 tiles.
     * Boundary tiles: dim0: 15-8=7, dim1: 12-8=4, dim2: 18-16=2,
     *                 dim3: 10-8=2, dim4: 14-8=6.
     */
    printf("--- Test 2: Rank-5 (15×12×18×10×14) ---\n");
    {
        hsize_t shape5[] = {15, 12, 18, 10, 14};
        result |= run_roundtrip(5, shape5, "test_rank5.h5");
    }

    printf("\n=== Result: %s ===\n",
           result == 0 ? "ALL PASSED" : "FAILURES DETECTED");
    return result;
}
