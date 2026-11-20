#ifndef VELRI_LIB_POOL_H
#define VELRI_LIB_POOL_H

#include "velri_lib/types.h"

/*
 * Opaque persistent worker pool owned by the caller / Velri-lib context.
 * - Uses C23 atomics + pthreads (zero OpenMP dependency, zero OS affinity mutation).
 * - Caller thread participates as worker 0; `num_workers - 1` background pthreads use
 *   a user-space spin period (currently 250 ms) for back-to-back dispatches and park on condition
 * variables when idle.
 * - Each worker owns a pre-allocated 64-byte-aligned bounded scratch buffer (`scratch_per_worker`)
 *   so hot-path GEMM, GEMV, FWHT, and vector/norm kernels perform no heap allocations.
 */
/* One submitting thread at a time per pool. Calls are synchronous; nested
 * dispatch on the same pool and concurrent destruction are not supported. */
typedef struct velri_pool velri_pool;

typedef void (*velri_parallel_fn)(void *ctx, size_t worker_id, size_t num_workers,
                                  void *worker_scratch, size_t scratch_bytes);

/*
 * Create a persistent worker pool:
 * - `num_workers`: total workers including the calling thread (>= 1).
 * - `scratch_per_worker`: bounded scratch bytes allocated once per worker (64-byte aligned).
 */
velri_status velri_pool_create(size_t num_workers, size_t scratch_per_worker,
                               velri_pool **out_pool);

void velri_pool_destroy(velri_pool *pool);

size_t velri_pool_num_workers(const velri_pool *pool);
size_t velri_pool_scratch_bytes(const velri_pool *pool);
void *velri_pool_worker0_scratch(velri_pool *pool);

/* Dispatch `fn` synchronously across `active_workers` (clamped to [1, pool->num_workers]). */
void velri_pool_parallel(velri_pool *pool, size_t active_workers, velri_parallel_fn fn, void *ctx);

#endif /* VELRI_LIB_POOL_H */
