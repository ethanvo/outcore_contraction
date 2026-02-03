#include "tensor_store.h"
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Test function for get_physical_offset
int test_get_physical_offset() {
  printf("Testing get_physical_offset function...\n");
  
  hsize_t tile_coords[] = {0, 1, 2};
  hsize_t chunk_dims[] = {10, 20, 30};
  hsize_t phys_offset[3];
  
  get_physical_offset(3, tile_coords, chunk_dims, phys_offset);
  
  // Verify results
  if (phys_offset[0] != 0 || phys_offset[1] != 20 || phys_offset[2] != 60) {
    printf("ERROR: get_physical_offset produced incorrect results\n");
    return 1;
  }
  
  printf("get_physical_offset test PASSED\n");
  return 0;
}

// Test function for calculate_chunk_dims
int test_calculate_chunk_dims() {
  printf("Testing calculate_chunk_dims function...\n");
  
  // Test with large dimensions
  hsize_t global_dims[] = {1000, 1000, 1000};
  hsize_t chunk_dims_test[3];
  
  calculate_chunk_dims(2 * 1024 * 1024, 3, global_dims, chunk_dims_test);
  
  // Verify that results are reasonable (should be around 64x64x64 for 2MB chunks)
  if (chunk_dims_test[0] < 1 || chunk_dims_test[0] > 1000 ||
      chunk_dims_test[1] < 1 || chunk_dims_test[1] > 1000 ||
      chunk_dims_test[2] < 1 || chunk_dims_test[2] > 1000) {
    printf("ERROR: calculate_chunk_dims produced invalid results\n");
    return 1;
  }
  
  // Test with smaller dimensions
  hsize_t small_dims[] = {100, 100, 100};
  hsize_t small_chunk_dims[3];
  
  calculate_chunk_dims(2 * 1024 * 1024, 3, small_dims, small_chunk_dims);
  
  if (small_chunk_dims[0] < 1 || small_chunk_dims[0] > 100 ||
      small_chunk_dims[1] < 1 || small_chunk_dims[1] > 100 ||
      small_chunk_dims[2] < 1 || small_chunk_dims[2] > 100) {
    printf("ERROR: calculate_chunk_dims with small dims produced invalid results\n");
    return 1;
  }
  
  printf("calculate_chunk_dims test PASSED\n");
  return 0;
}

// Test function for create_chunked_dataset
int test_create_chunked_dataset() {
  printf("Testing create_chunked_dataset function...\n");
  
  // Create a temporary file name for testing
  const char *filename = "test_tensor.h5";
  const char *dataset_name = "test_tensor";
  
  hsize_t test_dims[] = {100, 100};
  
  // This function should execute without crashing
  create_chunked_dataset(filename, dataset_name, 2, test_dims);
  
  // Try to open the file to verify it was created successfully
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    printf("ERROR: Could not open created dataset file\n");
    return 1;
  }
  
  // Try to open the dataset
  hid_t dset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
  if (dset_id < 0) {
    printf("ERROR: Could not open created dataset\n");
    H5Fclose(file_id);
    return 1;
  }
  
  // Clean up
  H5Dclose(dset_id);
  H5Fclose(file_id);
  
  // Remove the test file
  remove(filename);
  
  printf("create_chunked_dataset test PASSED\n");
  return 0;
}

// Test function for read_chunk_fast and write_chunk_fast
int test_read_write_chunk_fast() {
  printf("Testing read_chunk_fast and write_chunk_fast functions...\n");
  
  // Create a temporary file name for testing
  const char *filename = "test_rw.h5";
  const char *dataset_name = "test_rw_tensor";
  
  hsize_t test_dims[] = {10, 10};
  const int rank = 2;
  
  // Create the dataset
  create_chunked_dataset(filename, dataset_name, rank, test_dims);
  
  // Open the file and dataset
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
  if (file_id < 0) {
    printf("ERROR: Could not open dataset file\n");
    return 1;
  }
  
  hid_t dset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
  if (dset_id < 0) {
    printf("ERROR: Could not open dataset\n");
    H5Fclose(file_id);
    return 1;
  }
  
  // Test data
  double write_data[100];
  for (int i = 0; i < 100; i++) {
    write_data[i] = (double)(i + 1);
  }
  
  hsize_t phys_offset[] = {0, 0};
  hsize_t chunk_dims[] = {10, 10};
  
  // Write data using write_chunk_fast
  herr_t write_status = write_chunk_fast(dset_id, phys_offset, write_data, rank, chunk_dims);
  if (write_status < 0) {
    printf("ERROR: write_chunk_fast failed\n");
    H5Dclose(dset_id);
    H5Fclose(file_id);
    return 1;
  }
  
  // Read data back using read_chunk_fast
  double read_data[100];
  herr_t read_status = read_chunk_fast(dset_id, phys_offset, read_data, rank, chunk_dims);
  if (read_status < 0) {
    printf("ERROR: read_chunk_fast failed\n");
    H5Dclose(dset_id);
    H5Fclose(file_id);
    return 1;
  }
  
  // Verify data integrity
  for (int i = 0; i < 100; i++) {
    if (write_data[i] != read_data[i]) {
      printf("ERROR: Data mismatch at index %d\n", i);
      H5Dclose(dset_id);
      H5Fclose(file_id);
      return 1;
    }
  }
  
  // Clean up
  H5Dclose(dset_id);
  H5Fclose(file_id);
  
  // Remove the test file
  remove(filename);
  
  printf("read_chunk_fast and write_chunk_fast test PASSED\n");
  return 0;
}

// Main test function
int main() {
  printf("Testing tensor_store.c functions...\n");
  
  int result = 0;
  
  // Run all tests
  result |= test_get_physical_offset();
  result |= test_calculate_chunk_dims();
  result |= test_create_chunked_dataset();
  result |= test_read_write_chunk_fast();
  
  if (result == 0) {
    printf("All tensor_store.c tests PASSED!\n");
  } else {
    printf("Some tensor_store.c tests FAILED!\n");
  }
  
  return result;
}
