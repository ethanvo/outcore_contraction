#include "memory.h"
#include "registry.h"
#include "tensor_store.h" // For create_chunked_dataset if needed
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- MOCK COMPUTE KERNEL ---
// In the final version, replace this with: tblis_tensor_mult(...)
void compute_kernel(const double *A, const double *B, double *C, hsize_t M,
                    hsize_t N, hsize_t K) {
  // Simple naive matrix multiplication: C += A * B
  // A is (M x K), B is (K x N), C is (M x N)
  for (hsize_t m = 0; m < M; m++) {
    for (hsize_t n = 0; n < N; n++) {
      double sum = 0.0;
      for (hsize_t k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] += sum;
    }
  }
}

// --- MAIN ENGINE LOGIC ---
void run_contraction(const char *file_A, const char *name_A, const char *file_B,
                     const char *name_B, const char *file_C, const char *name_C,
                     size_t ram_pool_size) {

  printf("\n=== Starting Engine ===\n");

  // 1. Open Input Files
  hid_t fa = H5Fopen(file_A, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t fb = H5Fopen(file_B, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fa < 0 || fb < 0) {
    fprintf(stderr, "Error opening input files.\n");
    exit(1);
  }

  hid_t dset_A = H5Dopen2(fa, name_A, H5P_DEFAULT);
  hid_t dset_B = H5Dopen2(fb, name_B, H5P_DEFAULT);

  // 2. Prepare Registry
  // We assume all tensors have the same dimensions for this demo (Square Matrix
  // Mult) In a real engine, you'd read dims from the file dataspace.
  hsize_t global_dims[3] = {
      300, 300, 1}; // 300x300 matrices (Rank 2 treated as Rank 3 flat)
  size_t chunk_bytes = 2 * 1024 * 1024; // 2MB chunks

  TensorRegistry *reg_A = registry_create(2, global_dims, chunk_bytes);
  TensorRegistry *reg_B = registry_create(2, global_dims, chunk_bytes);
  TensorRegistry *reg_C = registry_create(2, global_dims, chunk_bytes);

  printf("Scanning Inputs...\n");
  registry_scan_file(dset_A, reg_A);
  registry_scan_file(dset_B, reg_B);

  // 3. Prepare Output File
  // We create C fresh.
  create_chunked_dataset(file_C, name_C, 2, global_dims);
  hid_t fc = H5Fopen(file_C, H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dset_C = H5Dopen2(fc, name_C, H5P_DEFAULT);

  // 4. Initialize Memory Pool
  // Calculate how many elements fit in a chunk
  size_t elems_per_chunk = reg_A->chunk_dims[0] * reg_A->chunk_dims[1];
  size_t num_pages = ram_pool_size / (elems_per_chunk * sizeof(double));
  BufferPool *pool = pool_create(num_pages, elems_per_chunk);

  printf("Memory Pool: %zu pages available.\n", num_pages);

  // 5. Execution Loop (SUMMA)
  hsize_t I_tiles = reg_C->grid_dims[0];
  hsize_t J_tiles = reg_C->grid_dims[1];
  hsize_t K_tiles = reg_A->grid_dims[1]; // The contraction dimension

  printf("Processing Grid [%llu x %llu]...\n", I_tiles, J_tiles);

  for (hsize_t i = 0; i < I_tiles; i++) {
    for (hsize_t j = 0; j < J_tiles; j++) {

      // A. Acquire Accumulator for C(i,j)
      int id_C;
      double *buf_C = pool_acquire(pool, &id_C);
      if (!buf_C) {
        fprintf(stderr, "Pool Exhausted!\n");
        exit(1);
      }
      memset(buf_C, 0, elems_per_chunk * sizeof(double)); // Zero init

      // B. Contract Loop
      for (hsize_t k = 0; k < K_tiles; k++) {

        TileMetadata *meta_A = registry_get_tile(reg_A, i, k, 0);
        TileMetadata *meta_B = registry_get_tile(reg_B, k, j, 0);

        // SPARSITY CHECK: Only compute if both input tiles exist
        if (meta_A->status == TILE_STATUS_ON_DISK &&
            meta_B->status == TILE_STATUS_ON_DISK) {

          int id_A, id_B;
          double *buf_A = pool_acquire(pool, &id_A);
          double *buf_B = pool_acquire(pool, &id_B);

          // Load Data
          read_chunk_fast(dset_A, meta_A->phys_offset, buf_A, 2,
                          reg_A->chunk_dims);
          read_chunk_fast(dset_B, meta_B->phys_offset, buf_B, 2,
                          reg_B->chunk_dims);

          // Compute
          compute_kernel(buf_A, buf_B, buf_C, reg_A->chunk_dims[0],
                         reg_B->chunk_dims[1], reg_A->chunk_dims[1]);

          // Release Inputs
          pool_release(pool, id_A);
          pool_release(pool, id_B);
        }
      }

      // C. Write Result
      TileMetadata *meta_C = registry_get_tile(reg_C, i, j, 0);
      write_chunk_fast(dset_C, meta_C->phys_offset, buf_C, 2,
                       reg_C->chunk_dims);

      // Release Output
      pool_release(pool, id_C);
      printf(".");
      fflush(stdout);
    }
  }

  printf("\nDone.\n");

  // Cleanup
  pool_destroy(pool);
  registry_destroy(reg_A);
  registry_destroy(reg_B);
  registry_destroy(reg_C);
  H5Dclose(dset_A);
  H5Dclose(dset_B);
  H5Dclose(dset_C);
  H5Fclose(fa);
  H5Fclose(fb);
  H5Fclose(fc);
}

int main() {
  // Run with 100MB RAM limit
  run_contraction("A.h5", "MatrixA", "B.h5", "MatrixB", "C.h5", "MatrixC",
                  100 * 1024 * 1024);
  return 0;
}
