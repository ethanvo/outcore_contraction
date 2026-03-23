/*
 * write_queue.c — Async write queue implementation.
 */

#include "write_queue.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

WriteQueue *wq_create(int cap)
{
    if (cap <= 0) return NULL;
    WriteQueue *wq = (WriteQueue *)calloc(1, sizeof(WriteQueue));
    if (!wq) return NULL;
    wq->tasks = (write_task_t *)calloc((size_t)cap, sizeof(write_task_t));
    if (!wq->tasks) { free(wq); return NULL; }
    wq->cap = cap;
    pthread_mutex_init(&wq->mu,   NULL);
    pthread_cond_init(&wq->cond,  NULL);
    return wq;
}

void wq_destroy(WriteQueue *wq)
{
    if (!wq) return;
    pthread_mutex_destroy(&wq->mu);
    pthread_cond_destroy(&wq->cond);
    free(wq->tasks);
    free(wq);
}

void wq_push(WriteQueue *wq, const write_task_t *task)
{
    pthread_mutex_lock(&wq->mu);
    while (wq->count == wq->cap)
        pthread_cond_wait(&wq->cond, &wq->mu);
    wq->tasks[wq->head % wq->cap] = *task;
    wq->head++;
    wq->count++;
    pthread_cond_broadcast(&wq->cond);
    pthread_mutex_unlock(&wq->mu);
}

write_task_t wq_pop(WriteQueue *wq)
{
    pthread_mutex_lock(&wq->mu);
    while (wq->count == 0)
        pthread_cond_wait(&wq->cond, &wq->mu);
    write_task_t t = wq->tasks[wq->tail % wq->cap];
    wq->tail++;
    wq->count--;
    pthread_cond_broadcast(&wq->cond);
    pthread_mutex_unlock(&wq->mu);
    return t;
}

void wq_push_sentinel(WriteQueue *wq)
{
    write_task_t t;
    memset(&t, 0, sizeof(t));
    t.id_C = SIZE_MAX;  /* sentinel value */
    wq_push(wq, &t);
}
