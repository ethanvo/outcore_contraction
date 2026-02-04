#include "memory.h"
#include <stdio.h>
#include <string.h> // for memset

struct BufferPool {
  double *data;     // The massive contiguous memory block
  int *free_stack;  // Stack of available page IDs
  int top;          // Stack pointer (index of the next free slot)
  size_t num_pages; // Total capacity
  size_t page_size; // Elements per page
};

BufferPool *pool_create(size_t num_pages, size_t elements_per_page) {
  BufferPool *pool = (BufferPool *)malloc(sizeof(BufferPool));
  if (!pool)
    return NULL;

  pool->num_pages = num_pages;
  pool->page_size = elements_per_page;
  pool->top = num_pages; // Stack is full (all pages are free)

  // 1. Allocate the Big Block (The Pool)
  size_t total_doubles = num_pages * elements_per_page;
  pool->data = (double *)malloc(total_doubles * sizeof(double));
  if (!pool->data) {
    free(pool);
    return NULL;
  }

  // 2. Allocate the Free Stack
  pool->free_stack = (int *)malloc(num_pages * sizeof(int));
  if (!pool->free_stack) {
    free(pool->data);
    free(pool);
    return NULL;
  }

  // 3. Initialize Stack: [0, 1, 2, ... N-1]
  // Initially, all pages are free.
  for (int i = 0; i < (int)num_pages; i++) {
    pool->free_stack[i] = i;
  }

  // Optional: Zero out memory (safety vs speed tradeoff)
  // memset(pool->data, 0, total_doubles * sizeof(double));

  printf("BufferPool Initialized: %zu pages x %zu elements (%zu MB total)\n",
         num_pages, elements_per_page,
         (total_doubles * sizeof(double)) / (1024 * 1024));

  return pool;
}

double *pool_acquire(BufferPool *pool, int *out_id) {
  if (pool->top <= 0) {
    // Pool is empty!
    // In a real engine, you would wait on a condition variable here.
    fprintf(stderr, "WARNING: BufferPool Exhausted!\n");
    return NULL;
  }

  // Pop from stack
  pool->top--;
  int page_id = pool->free_stack[pool->top];

  if (out_id)
    *out_id = page_id;

  // --- MANUAL POINTER ARITHMETIC ---
  // Base Address + (Page ID * Stride)
  return pool->data + (page_id * pool->page_size);
}

void pool_release(BufferPool *pool, int page_id) {
  if (page_id < 0 || page_id >= pool->num_pages) {
    fprintf(stderr, "ERROR: Attempted to release invalid Page ID %d\n",
            page_id);
    return;
  }

  if (pool->top >= pool->num_pages) {
    fprintf(stderr, "ERROR: BufferPool Overflow (Double Free?)\n");
    return;
  }

  // Push back onto stack
  pool->free_stack[pool->top] = page_id;
  pool->top++;
}

double *pool_get_ptr(BufferPool *pool, int page_id) {
  if (page_id < 0 || page_id >= pool->num_pages)
    return NULL;
  return pool->data + (page_id * pool->page_size);
}

size_t pool_free_count(BufferPool *pool) { return (size_t)pool->top; }

void pool_destroy(BufferPool *pool) {
  if (pool) {
    if (pool->data)
      free(pool->data);
    if (pool->free_stack)
      free(pool->free_stack);
    free(pool);
    printf("BufferPool Destroyed.\n");
  }
}
