/*
 * write_queue.h
 *
 * Asynchronous write queue: the compute thread pushes completed C-tile pages
 * here instead of calling write_chunk_typed synchronously.  A dedicated writer
 * thread drains the queue, writes each tile, and returns the pool page so the
 * prefetch thread can reuse it.
 *
 * Sentinel: push a task with id_C == SIZE_MAX to tell the writer thread to exit.
 */

#ifndef WRITE_QUEUE_H
#define WRITE_QUEUE_H

#include <pthread.h>
#include <stddef.h>
#include "registry.h"   /* MAX_RANK, hsize_t */
#include "memory.h"     /* BufferPool        */

typedef struct {
    void           *buf_C;              /* pool page holding the finished C tile   */
    size_t          id_C;               /* pool page ID; SIZE_MAX → sentinel/exit  */
    hsize_t         phys_off[MAX_RANK]; /* physical HDF5 chunk offset for C tile   */
    /* Ownership: writer thread releases id_C back to owning_pool and
     * broadcasts pool_cond under pool_mu so the worker can reuse the page. */
    BufferPool     *owning_pool;
    pthread_mutex_t *pool_mu;
    pthread_cond_t  *pool_cond;
} write_task_t;

typedef struct {
    write_task_t   *tasks;  /* ring buffer storage                         */
    int             cap;    /* ring capacity (number of slots)             */
    int             head;   /* producer cursor (mod cap)                   */
    int             tail;   /* consumer cursor (mod cap)                   */
    int             count;  /* number of tasks currently in the ring       */
    pthread_mutex_t mu;
    pthread_cond_t  cond;   /* broadcast on every push or pop              */
    volatile int    err;    /* set to 1 by writer thread on I/O failure    */
} WriteQueue;

/* Allocate and initialise a WriteQueue with room for cap tasks. */
WriteQueue  *wq_create(int cap);

/* Destroy and free a WriteQueue (must be idle). */
void         wq_destroy(WriteQueue *wq);

/*
 * Push one task onto the queue (blocks if the ring is full).
 * Called by the compute thread.
 */
void         wq_push(WriteQueue *wq, const write_task_t *task);

/*
 * Pop one task from the queue (blocks if the ring is empty).
 * Called by the writer thread.
 */
write_task_t wq_pop(WriteQueue *wq);

/*
 * Push the sentinel task (id_C == SIZE_MAX) to signal the writer thread
 * that no more tiles will arrive.
 */
void         wq_push_sentinel(WriteQueue *wq);

#endif /* WRITE_QUEUE_H */
