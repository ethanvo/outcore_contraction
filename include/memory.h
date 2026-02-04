#ifndef MEMORY_H
#define MEMORY_H

#include <stdlib.h>

typedef struct BufferPool BufferPool;

/*
 * Creates a memory pool.
 * num_pages: Total number of slots available (e.g., 100).
 * elements_per_page: Size of one slot in doubles (e.g., 64*64*64 = 262144).
 */
BufferPool *pool_create(size_t num_pages, size_t elements_per_page);

/*
 * Destroys the pool and frees all memory.
 */
void pool_destroy(BufferPool *pool);

/*
 * Acquires a pointer to a free memory page.
 * out_id: (Output) Stores the unique ID of the page (0 to N-1).
 * You need this ID to release the page later.
 * Returns: Pointer to the memory, or NULL if pool is empty.
 */
double *pool_acquire(BufferPool *pool, int *out_id);

/*
 * Returns a page to the pool, making it available for reuse.
 * page_id: The ID obtained during pool_acquire.
 */
void pool_release(BufferPool *pool, int page_id);

/*
 * Helper: Get the pointer for a specific ID without "acquiring" it.
 * Useful for debugging or lookups if you already hold the lock.
 */
double *pool_get_ptr(BufferPool *pool, int page_id);

/*
 * Diagnostic: Returns number of currently free pages.
 */
size_t pool_free_count(BufferPool *pool);

#endif
