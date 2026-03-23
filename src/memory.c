#include "memory.h"
#include <stdio.h>
#include <stdlib.h>

struct BufferPool {
    char   *data;       /* Single contiguous byte allocation for all pages   */
    size_t *free_stack; /* Stack of available page IDs (0 … num_pages-1)    */
    size_t  top;        /* Stack pointer; equals free count                  */
    size_t  num_pages;
    size_t  page_bytes; /* Bytes per page                                    */
};

BufferPool *pool_create(size_t num_pages, size_t bytes_per_page)
{
    BufferPool *pool = (BufferPool *)malloc(sizeof(BufferPool));
    if (!pool) return NULL;

    pool->num_pages = num_pages;
    pool->page_bytes = bytes_per_page;
    pool->top        = num_pages;  /* all pages free initially */

    /* 16 KB alignment matches Apple NVMe hardware page size, eliminating
     * read-amplification when the OS DMA-transfers chunks directly into
     * pool pages.  posix_memalign guarantees alignment and is POSIX. */
    if (posix_memalign((void **)&pool->data, 16384,
                       num_pages * bytes_per_page) != 0) {
        pool->data = NULL;
    }
    if (!pool->data) { free(pool); return NULL; }

    pool->free_stack = (size_t *)malloc(num_pages * sizeof(size_t));
    if (!pool->free_stack) { free(pool->data); free(pool); return NULL; }

    for (size_t i = 0; i < num_pages; i++)
        pool->free_stack[i] = i;

    return pool;
}

void pool_destroy(BufferPool *pool)
{
    if (pool) {
        free(pool->data);
        free(pool->free_stack);
        free(pool);
    }
}

void *pool_acquire(BufferPool *pool, size_t *out_id)
{
    if (pool->top == 0) {
        fprintf(stderr, "pool_acquire: BufferPool exhausted\n");
        return NULL;
    }

    pool->top--;
    size_t page_id = pool->free_stack[pool->top];

    if (out_id) *out_id = page_id;
    return (void *)(pool->data + page_id * pool->page_bytes);
}

void pool_release(BufferPool *pool, size_t page_id)
{
    if (page_id >= pool->num_pages) {
        fprintf(stderr,
                "pool_release: invalid page_id %zu (pool has %zu pages)\n",
                page_id, pool->num_pages);
        return;
    }
    if (pool->top >= pool->num_pages) {
        fprintf(stderr, "pool_release: overflow – possible double-free of "
                        "page_id %zu\n", page_id);
        return;
    }

    pool->free_stack[pool->top] = page_id;
    pool->top++;
}

void *pool_get_ptr(BufferPool *pool, size_t page_id)
{
    if (page_id >= pool->num_pages) return NULL;
    return (void *)(pool->data + page_id * pool->page_bytes);
}

size_t pool_free_count(BufferPool *pool)
{
    return pool->top;
}
