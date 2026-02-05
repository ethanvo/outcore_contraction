#ifndef ENGINE_H
#define ENGINE_H

#include <hdf5.h>

/*
 * Main Entry Point.
 * Contracts A(i,k) * B(k,j) -> C(i,j)
 * All tensors must be chunked and compatible.
 */
void run_contraction(const char *file_A, const char *name_A, const char *file_B,
                     const char *name_B, const char *file_C, const char *name_C,
                     size_t ram_pool_size);

#endif
