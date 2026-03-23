#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

typedef struct BufferPool BufferPool;

/*
 * Allocate a pool of num_pages pages, each bytes_per_page bytes in size.
 * All pages reside in a single contiguous allocation.
 *
 * For FP64 tensors: bytes_per_page = elements_per_page * sizeof(double).
 * For COMPLEX128:   bytes_per_page = elements_per_page * sizeof(double _Complex).
 */
BufferPool *pool_create(size_t num_pages, size_t bytes_per_page);

/* Release all pool memory. */
void pool_destroy(BufferPool *pool);

/*
 * Acquire a free page.  *out_id receives the page's ID (0 … num_pages-1).
 * Use SIZE_MAX as the "not acquired" sentinel when initialising an ID before
 * a matching pool_acquire call.
 * Returns a pointer to the page's raw memory, or NULL if the pool is exhausted.
 * Callers cast the returned pointer to the appropriate element type.
 */
void *pool_acquire(BufferPool *pool, size_t *out_id);

/*
 * Return page page_id to the free stack.
 * Passing SIZE_MAX or any out-of-range value is a no-op (logged to stderr).
 */
void pool_release(BufferPool *pool, size_t page_id);

/* Return a pointer to page page_id without acquiring it.  Returns NULL for
 * out-of-range IDs (including SIZE_MAX). */
void *pool_get_ptr(BufferPool *pool, size_t page_id);

/* Number of currently free pages. */
size_t pool_free_count(BufferPool *pool);

#endif /* MEMORY_H */
