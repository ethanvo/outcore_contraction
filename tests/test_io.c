#include "registry.h"
#include "tensor_store.h"
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    printf("=== Integration test: tensor_store + registry ===\n\n");

    const char *fname = "test_tensor_io.h5";
    const char *dname = "test_tensor_io";

    /* Rank-3 tensor: large enough to produce several tiles. */
    const int   rank        = 3;
    hsize_t     global_dims[] = {300, 300, 300};
    const size_t chunk_bytes  = 2 * 1024 * 1024;

    /* Create dataset. */
    if (create_chunked_dataset(fname, dname, rank, global_dims,
                               chunk_bytes) < 0) {
        fprintf(stderr, "ERROR: create_chunked_dataset failed\n");
        return 1;
    }

    hid_t file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "ERROR: H5Fopen\n"); return 1; }

    hid_t dset_id = dset_open_no_cache(file_id, dname);
    if (dset_id < 0) {
        fprintf(stderr, "ERROR: dset_open_no_cache\n");
        H5Fclose(file_id);
        return 1;
    }

    /* Determine actual chunk dims from the file (source of truth). */
    hid_t  dcpl_id = H5Dget_create_plist(dset_id);
    hsize_t chunk_dims[3];
    H5Pget_chunk(dcpl_id, rank, chunk_dims);
    H5Pclose(dcpl_id);

    printf("Global dims : [%llu, %llu, %llu]\n",
           (unsigned long long)global_dims[0],
           (unsigned long long)global_dims[1],
           (unsigned long long)global_dims[2]);
    printf("Chunk dims  : [%llu, %llu, %llu]\n\n",
           (unsigned long long)chunk_dims[0],
           (unsigned long long)chunk_dims[1],
           (unsigned long long)chunk_dims[2]);

    size_t elems = (size_t)chunk_dims[0]
                 * (size_t)chunk_dims[1]
                 * (size_t)chunk_dims[2];
    double *wbuf = (double *)malloc(elems * sizeof(double));
    double *rbuf = (double *)malloc(elems * sizeof(double));
    if (!wbuf || !rbuf) {
        fprintf(stderr, "ERROR: malloc\n");
        free(wbuf); free(rbuf);
        H5Dclose(dset_id); H5Fclose(file_id);
        return 1;
    }

    srand((unsigned)time(NULL));

    /* Write 3 tiles at diagonal positions (0,0,0), (1,1,1), (2,2,2). */
    int write_ok = 1;
    for (int step = 0; step < 3 && write_ok; step++) {
        printf("--- Tile (%d,%d,%d) ---\n", step, step, step);
        for (size_t i = 0; i < elems; i++)
            wbuf[i] = (double)step + (double)i / 1000.0;

        hsize_t tile_coords[] = {(hsize_t)step, (hsize_t)step, (hsize_t)step};
        hsize_t phys_offset[3];
        get_physical_offset(rank, tile_coords, chunk_dims, phys_offset);

        printf("  Writing to offset [%llu, %llu, %llu]\n",
               (unsigned long long)phys_offset[0],
               (unsigned long long)phys_offset[1],
               (unsigned long long)phys_offset[2]);

        if (write_chunk_fast(dset_id, phys_offset, wbuf, rank,
                             chunk_dims) < 0) {
            fprintf(stderr, "  ERROR: write failed\n");
            write_ok = 0;
            break;
        }

        if (read_chunk_fast(dset_id, phys_offset, rbuf, rank,
                            chunk_dims) < 0) {
            fprintf(stderr, "  ERROR: read failed\n");
            write_ok = 0;
            break;
        }

        if (rbuf[0] != wbuf[0] || rbuf[elems-1] != wbuf[elems-1])
            fprintf(stderr, "  DATA MISMATCH at step %d\n", step);
        else
            printf("  Verification: OK\n");
    }

    /* --- Registry scan --- */
    printf("\n--- Registry scan ---\n");

    TensorRegistry *reg = registry_create_from_dset(dset_id);
    if (!reg) {
        fprintf(stderr, "ERROR: registry_create_from_dset\n");
        free(wbuf); free(rbuf);
        H5Dclose(dset_id); H5Fclose(file_id);
        return 1;
    }

    /* Verify initial state: spot-check tile (0,0,0) is NULL. */
    {
        hsize_t c0[] = {0, 0, 0};
        TileMetadata *t = registry_get_tile(reg, c0);
        printf("Before scan – tile (0,0,0) status: %s\n",
               (t && t->status == TILE_STATUS_NULL) ? "NULL (correct)" : "UNEXPECTED");
    }

    long found = registry_scan_file(dset_id, reg);

    if (found == 3)
        printf("Registry correctly identified 3 chunks on disk  (PASS)\n");
    else
        printf("FAIL: expected 3 chunks, got %ld\n", found);

    /* Tile (1,1,1) should now be ON_DISK. */
    {
        hsize_t c1[] = {1, 1, 1};
        TileMetadata *t = registry_get_tile(reg, c1);
        printf("Tile (1,1,1): %s\n",
               (t && t->status == TILE_STATUS_ON_DISK)
               ? "ON_DISK (correct)"
               : "UNEXPECTED status");
    }

    /* Tile (0,1,0) was never written – should still be NULL. */
    {
        hsize_t c010[] = {0, 1, 0};
        TileMetadata *t = registry_get_tile(reg, c010);
        printf("Tile (0,1,0): %s\n",
               (t && t->status == TILE_STATUS_NULL)
               ? "NULL (correct)"
               : "UNEXPECTED status");
    }

    registry_destroy(reg);
    free(wbuf);
    free(rbuf);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    remove(fname);

    printf("\nTest complete.\n");
    return 0;
}
