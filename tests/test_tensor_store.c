#include "tensor_store.h"
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_get_physical_offset(void)
{
    printf("Testing get_physical_offset...\n");

    hsize_t tile_coords[] = {0, 1, 2};
    hsize_t chunk_dims[]  = {10, 20, 30};
    hsize_t phys_offset[3];

    get_physical_offset(3, tile_coords, chunk_dims, phys_offset);

    if (phys_offset[0] != 0 || phys_offset[1] != 20 || phys_offset[2] != 60) {
        printf("  FAIL: expected (0,20,60) got (%llu,%llu,%llu)\n",
               (unsigned long long)phys_offset[0],
               (unsigned long long)phys_offset[1],
               (unsigned long long)phys_offset[2]);
        return 1;
    }
    printf("  PASS\n");
    return 0;
}

static int test_calculate_chunk_dims(void)
{
    printf("Testing calculate_chunk_dims...\n");

    /* Large tensor: chunk side should be ~64 for rank-3, 2 MB target. */
    hsize_t global_large[] = {1000, 1000, 1000};
    hsize_t chunk_large[3];
    calculate_chunk_dims(2 * 1024 * 1024, 3, global_large, chunk_large);
    for (int d = 0; d < 3; d++) {
        if (chunk_large[d] < 1 || chunk_large[d] > 1000) {
            printf("  FAIL: chunk_large[%d]=%llu out of range\n", d,
                   (unsigned long long)chunk_large[d]);
            return 1;
        }
    }

    /* Small tensor: chunk must be clamped to global size. */
    hsize_t global_small[] = {4, 4, 4};
    hsize_t chunk_small[3];
    calculate_chunk_dims(2 * 1024 * 1024, 3, global_small, chunk_small);
    for (int d = 0; d < 3; d++) {
        if (chunk_small[d] != 4) {
            printf("  FAIL: chunk_small[%d]=%llu expected 4\n", d,
                   (unsigned long long)chunk_small[d]);
            return 1;
        }
    }

    printf("  PASS\n");
    return 0;
}

static int test_create_chunked_dataset(void)
{
    printf("Testing create_chunked_dataset...\n");

    const char *fname = "test_create.h5";
    const char *dname = "tensor";
    hsize_t dims[] = {100, 100};

    if (create_chunked_dataset(fname, dname, 2, dims,
                               2 * 1024 * 1024) < 0) {
        printf("  FAIL: create_chunked_dataset returned error\n");
        return 1;
    }

    hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) { printf("  FAIL: H5Fopen\n"); return 1; }

    hid_t dset_id = H5Dopen2(file_id, dname, H5P_DEFAULT);
    if (dset_id < 0) {
        printf("  FAIL: H5Dopen2\n");
        H5Fclose(file_id);
        return 1;
    }

    H5Dclose(dset_id);
    H5Fclose(file_id);
    remove(fname);

    printf("  PASS\n");
    return 0;
}

static int test_read_write_chunk_fast(void)
{
    printf("Testing read_chunk_fast / write_chunk_fast (full tile)...\n");

    const char *fname = "test_rw.h5";
    const char *dname = "rw_tensor";
    hsize_t dims[]       = {10, 10};
    hsize_t chunk_dims[] = {10, 10};
    hsize_t offset[]     = {0, 0};
    const int rank = 2;

    if (create_chunked_dataset(fname, dname, rank, dims,
                               2 * 1024 * 1024) < 0) {
        printf("  FAIL: create_chunked_dataset\n");
        return 1;
    }

    hid_t file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) { printf("  FAIL: H5Fopen\n"); return 1; }

    hid_t dset_id = dset_open_no_cache(file_id, dname);
    if (dset_id < 0) {
        printf("  FAIL: dset_open_no_cache\n");
        H5Fclose(file_id);
        return 1;
    }

    double write_buf[100], read_buf[100];
    for (int i = 0; i < 100; i++) write_buf[i] = (double)(i + 1);

    if (write_chunk_fast(dset_id, offset, write_buf, rank, chunk_dims) < 0) {
        printf("  FAIL: write_chunk_fast\n");
        H5Dclose(dset_id); H5Fclose(file_id);
        return 1;
    }
    if (read_chunk_fast(dset_id, offset, read_buf, rank, chunk_dims) < 0) {
        printf("  FAIL: read_chunk_fast\n");
        H5Dclose(dset_id); H5Fclose(file_id);
        return 1;
    }

    for (int i = 0; i < 100; i++) {
        if (write_buf[i] != read_buf[i]) {
            printf("  FAIL: mismatch at %d (expected %g got %g)\n",
                   i, write_buf[i], read_buf[i]);
            H5Dclose(dset_id); H5Fclose(file_id);
            return 1;
        }
    }

    H5Dclose(dset_id);
    H5Fclose(file_id);
    remove(fname);
    printf("  PASS\n");
    return 0;
}

static int test_high_rank_tensor(void)
{
    printf("Testing 4-D tensor I/O...\n");

    const char *fname = "test_4d.h5";
    const char *dname = "tensor_4d";
    const int   rank  = 4;

    hsize_t global_dims[] = {10, 10, 10, 10};
    hsize_t chunk_dims[]  = {5, 5, 5, 5};
    hsize_t offset[]      = {5, 0, 5, 0};   /* fits exactly: 5+5=10 ≤ 10 */

    if (create_chunked_dataset(fname, dname, rank, global_dims,
                               2 * 1024 * 1024) < 0) {
        printf("  FAIL: create_chunked_dataset\n");
        return 1;
    }

    hid_t file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) { printf("  FAIL: H5Fopen\n"); return 1; }

    hid_t dset_id = dset_open_no_cache(file_id, dname);
    if (dset_id < 0) {
        H5Fclose(file_id);
        printf("  FAIL: dset_open_no_cache\n");
        return 1;
    }

    size_t nelems = 5*5*5*5;  /* 625 */
    double *wbuf  = (double *)malloc(nelems * sizeof(double));
    double *rbuf  = (double *)malloc(nelems * sizeof(double));
    if (!wbuf || !rbuf) {
        free(wbuf); free(rbuf);
        H5Dclose(dset_id); H5Fclose(file_id);
        printf("  FAIL: malloc\n");
        return 1;
    }

    for (size_t i = 0; i < nelems; i++) wbuf[i] = (double)(i + 1) * 1.5;

    int ret = 0;
    if (write_chunk_fast(dset_id, offset, wbuf, rank, chunk_dims) < 0) {
        printf("  FAIL: write_chunk_fast\n"); ret = 1;
    }
    if (!ret &&
        read_chunk_fast(dset_id, offset, rbuf, rank, chunk_dims) < 0) {
        printf("  FAIL: read_chunk_fast\n"); ret = 1;
    }
    if (!ret) {
        for (size_t i = 0; i < nelems; i++) {
            if (wbuf[i] != rbuf[i]) {
                printf("  FAIL: mismatch at %zu (expected %g got %g)\n",
                       i, wbuf[i], rbuf[i]);
                ret = 1;
                break;
            }
        }
    }

    free(wbuf); free(rbuf);
    H5Dclose(dset_id); H5Fclose(file_id);
    remove(fname);
    if (!ret) printf("  PASS\n");
    return ret;
}

static int test_small_chunks_and_boundaries(void)
{
    printf("Testing boundary clamping with nominal-stride buffer layout...\n");

    const char *fname = "test_boundary.h5";
    const char *dname = "tensor_2d";
    const int   rank  = 2;

    hsize_t global_dims[] = {25, 25};
    hsize_t chunk_dims[]  = {10, 10};

    if (create_chunked_dataset(fname, dname, rank, global_dims,
                               2 * 1024 * 1024) < 0) {
        printf("  FAIL: create_chunked_dataset\n");
        return 1;
    }

    hid_t file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) { printf("  FAIL: H5Fopen\n"); return 1; }

    hid_t dset_id = dset_open_no_cache(file_id, dname);
    if (dset_id < 0) {
        H5Fclose(file_id);
        printf("  FAIL: dset_open_no_cache\n");
        return 1;
    }

    double wbuf[100], rbuf[100];
    /* Sequential fill: wbuf[i] = i+1. */
    for (int i = 0; i < 100; i++) wbuf[i] = (double)(i + 1);

    /* --- Full tile at origin --- */
    hsize_t off_origin[] = {0, 0};
    if (write_chunk_fast(dset_id, off_origin, wbuf, rank, chunk_dims) < 0 ||
        read_chunk_fast (dset_id, off_origin, rbuf, rank, chunk_dims) < 0) {
        printf("  FAIL: origin tile I/O\n");
        H5Dclose(dset_id); H5Fclose(file_id); return 1;
    }
    for (int i = 0; i < 100; i++) {
        if (wbuf[i] != rbuf[i]) {
            printf("  FAIL: origin mismatch at %d\n", i);
            H5Dclose(dset_id); H5Fclose(file_id); return 1;
        }
    }

    /*
     * --- Boundary tile at [20, 20] ---
     *
     * Dataset is 25×25; nominal chunk is 10×10.
     * Actual extent: 5×5 (25-20=5 in each dim).
     *
     * write_chunk_fast: memspace=10×10 with selection [0..4,0..4] at origin.
     *   wbuf[row*10+col] for row<5, col<5 → file[20+row, 20+col]
     *
     * read_chunk_fast:  rbuf is pre-zeroed; memspace=10×10 with selection
     *   [0..4,0..4].  rbuf[row*10+col] = wbuf[row*10+col] for row<5,col<5.
     *   rbuf[row*10+col] = 0 for col>=5 (padding, never written).
     */
    hsize_t off_edge[] = {20, 20};
    memset(rbuf, 0, sizeof(rbuf));

    if (write_chunk_fast(dset_id, off_edge, wbuf, rank, chunk_dims) < 0 ||
        read_chunk_fast (dset_id, off_edge, rbuf, rank, chunk_dims) < 0) {
        printf("  FAIL: boundary tile I/O\n");
        H5Dclose(dset_id); H5Fclose(file_id); return 1;
    }

    /* Verify valid region: strided positions [row*10+col], row<5, col<5. */
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            int idx = row * 10 + col;
            if (rbuf[idx] != wbuf[idx]) {
                printf("  FAIL: boundary valid region mismatch at "
                       "[%d,%d] (idx=%d)  expected=%g got=%g\n",
                       row, col, idx, wbuf[idx], rbuf[idx]);
                H5Dclose(dset_id); H5Fclose(file_id); return 1;
            }
        }
    }

    /* Verify padding region is zero. */
    for (int row = 0; row < 5; row++) {
        for (int col = 5; col < 10; col++) {
            if (rbuf[row * 10 + col] != 0.0) {
                printf("  FAIL: padding at [%d,%d] not zero (got %g)\n",
                       row, col, rbuf[row * 10 + col]);
                H5Dclose(dset_id); H5Fclose(file_id); return 1;
            }
        }
    }

    H5Dclose(dset_id);
    H5Fclose(file_id);
    remove(fname);
    printf("  PASS\n");
    return 0;
}

int main(void)
{
    printf("=== tensor_store tests ===\n");
    int result = 0;
    result |= test_get_physical_offset();
    result |= test_calculate_chunk_dims();
    result |= test_create_chunked_dataset();
    result |= test_read_write_chunk_fast();
    result |= test_high_rank_tensor();
    result |= test_small_chunks_and_boundaries();
    printf(result == 0 ? "\nAll tests PASSED\n" : "\nSome tests FAILED\n");
    return result;
}
