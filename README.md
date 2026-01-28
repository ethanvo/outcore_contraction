# outcore_contraction
High-performance C++ engine for out-of-core tensor contractions. Bridges TBLIS &amp; HDF5 to process tensors exceeding RAM. Features a threaded Producer-Consumer model, async prefetching, double-buffering, and LRU caching to minimize I/O latency. Includes automatic HDF5 chunk alignment and block-sparsity support.
