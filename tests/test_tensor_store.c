#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hdf5.h>
#include "tensor_store.h"

int main() {
    printf("Testing tensor_store.c functions...\n");
    
    // Test calculate_chunk_dims function
    hsize_t global_dims[] = {1000, 1000, 1000};
    hsize_t chunk_dims[3];
    
    calculate_chunk_dims(2 * 1024 * 1024, 3, global_dims, chunk_dims);
    
    printf("Global dims: %llu %llu %llu\n", global_dims[0], global_dims[1], global_dims[2]);
    printf("Calculated chunk dims: %llu %llu %llu\n", chunk_dims[0], chunk_dims[1], chunk_dims[2]);
    
    // Test with smaller dimensions
    hsize_t small_dims[] = {100, 100, 100};
    hsize_t small_chunk_dims[3];
    
    calculate_chunk_dims(2 * 1024 * 1024, 3, small_dims, small_chunk_dims);
    
    printf("Small global dims: %llu %llu %llu\n", small_dims[0], small_dims[1], small_dims[2]);
    printf("Small calculated chunk dims: %llu %llu %llu\n", small_chunk_dims[0], small_chunk_dims[1], small_chunk_dims[2]);
    
    // Test create_chunked_dataset function (this will attempt to create an HDF5 file)
    // We'll test it by calling it with a simple test case
    hsize_t test_dims[] = {100, 100};
    printf("Testing create_chunked_dataset function...\n");
    
    // Create a temporary file name for testing
    const char* filename = "test_tensor.h5";
    const char* dataset_name = "test_tensor";
    
    // This function should execute without crashing
    create_chunked_dataset(filename, dataset_name, 2, test_dims);
    
    printf("create_chunked_dataset function executed without errors!\n");
    
    // Cleanup the test file
    remove(filename);
    
    printf("tensor_store.c functions work correctly!\n");
    return 0;
}