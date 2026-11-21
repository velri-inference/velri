#ifndef VELRI_LIB_POOL_H
#define VELRI_LIB_POOL_H

#include "velri_lib/types.h"

/*
 * Opaque persistent worker pool owned by the CPU backend.
 * - Uses C23 atomics + OS native threads (pthreads on POSIX, Win32 threads on Windows).
 * - Caller thread participates as worker 0; background workers spin briefly for
 *   back-to-back dispatches and park on condition variables when idle.
 * - One submitting thread at a time per pool; calls are synchronous.
 */
typedef struct velri_pool velri_pool;

typedef void (*velri_parallel_fn)(void *ctx, size_t worker_id, size_t num_workers);

velri_status velri_pool_create(size_t num_workers, velri_pool **out_pool);
void velri_pool_destroy(velri_pool *pool);

/* Dispatch `fn` synchronously across `active_workers` (clamped to [1, pool->num_workers]). */
void velri_pool_parallel(velri_pool *pool, size_t active_workers, velri_parallel_fn fn, void *ctx);

#endif /* VELRI_LIB_POOL_H */
