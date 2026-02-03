# outcore_contraction
High-performance C++ engine for out-of-core tensor contractions. Bridges TBLIS &amp; HDF5 to process tensors exceeding RAM. Features a threaded Producer-Consumer model, async prefetching, double-buffering, and LRU caching to minimize I/O latency. Includes automatic HDF5 chunk alignment and block-sparsity support.

## Storage Layer Implementation

This document describes the implementation of the storage layer components for the Out-of-Core Tensor Contraction Engine.

### Implemented Functions

#### 1. `read_chunk` function
- **Purpose**: Reads a specific hyperslab (chunk) from an HDF5 dataset
- **Parameters**: 
  - `file_id`: HDF5 file identifier
  - `dataset_name`: Name of the dataset within the file
  - `offset_coords`: Array of coordinates specifying the chunk offset
  - `data_ptr`: Buffer to read data into
  - `rank`: Number of dimensions
  - `chunk_dims`: Size of each chunk dimension
- **Implementation**: 
  - Opens the dataset
  - Gets the dataset's dataspace
  - Creates memory space for the chunk
  - Selects hyperslab in the file dataspace using offset coordinates
  - Reads data into the provided buffer
  - Handles proper error checking and resource cleanup

#### 2. `write_chunk` function
- **Purpose**: Writes data to a specific hyperslab (chunk) in an HDF5 dataset
- **Parameters**:
  - `file_id`: HDF5 file identifier
  - `dataset_name`: Name of the dataset within the file
  - `offset_coords`: Array of coordinates specifying the chunk offset
  - `data_ptr`: Buffer containing data to write
  - `rank`: Number of dimensions
  - `chunk_dims`: Size of each chunk dimension
- **Implementation**:
  - Opens the dataset
  - Gets the dataset's dataspace
  - Creates memory space for the chunk
  - Selects hyperslab in the file dataspace using offset coordinates
  - Writes data from the provided buffer
  - Handles proper error checking and resource cleanup

### Testing Approach

Unit tests for these functions require HDF5 compilation and linking, which isn't available in the current environment. The functions have been implemented and tested conceptually according to the Software Development Plan.

To test these functions properly:
1. Ensure HDF5 development libraries are installed
2. Compile with: `gcc -I/usr/local/include -L/usr/local/lib test_tensor_store.c tensor_store.c -lhdf5 -o test_tensor_store`
3. Run the resulting executable

### Integration with SDP

These functions implement Phase 1, Step 1.2 of the Software Development Plan:
- Abstract away the complexity of HDF5 hyperslabs and types
- Provide the "shovels" needed for the rest of the engine to move data between HDF5 storage and memory
- Enable proper chunking and I/O operations for large tensors
