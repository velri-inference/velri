#define _POSIX_C_SOURCE 200809L

#include "velri_lib/pool.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#else
    /* no-op */
#endif
}

static inline uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

typedef struct {
    alignas(64) _Atomic uint64_t done_gen;
    size_t worker_id;
    void *scratch;
    velri_pool *pool;
    pthread_t thread;
    pthread_mutex_t mtx;
    pthread_cond_t cv;
    bool sync_ready;
} velri_worker_slot;

struct velri_pool {
    size_t num_workers;
    size_t scratch_bytes;
    atomic_bool stopping;
    velri_parallel_fn fn;
    void *fn_ctx;
    size_t last_nw;
    uint64_t host_gen;
    alignas(64) _Atomic uint64_t dispatch_word; /* (gen << 16) | (nw & 0xFFFFU) */
    alignas(64) _Atomic uint64_t parked_mask[4];
    velri_worker_slot *slots;
};

static void *worker_entry(void *arg) {
    velri_worker_slot *slot = (velri_worker_slot *)arg;
    velri_pool *pool = slot->pool;
    const size_t wid = slot->worker_id;
    const size_t mask_idx = wid >> 6;
    const uint64_t mask_bit = UINT64_C(1) << (wid & 63U);

    if (slot->scratch && pool->scratch_bytes > 0) {
        memset(slot->scratch, 0, pool->scratch_bytes);
    }
    atomic_store_explicit(&slot->done_gen, UINT64_C(1), memory_order_release);

    uint64_t g = UINT64_C(1);
    for (;;) {
        uint64_t word = atomic_load_explicit(&pool->dispatch_word, memory_order_acquire);
        uint64_t cg = word >> 16;
        size_t cnw = (size_t)(word & UINT64_C(0xFFFF));

        if (cg == g && wid < cnw) {
            /*
             * Keep recently active workers spinning on the shared broadcast line `dispatch_word`
             * (read-only S state in L1d) for up to 250 ms between back-to-back dispatches so OS
             * `futex_wake` (`wake_affine`) never pulls unpinned workers off distinct physical
             * cores. Inactive workers (`wid >= cnw`) skip spinning and park immediately.
             */
            const uint64_t deadline = monotonic_ns() + UINT64_C(250000000);
            do {
                for (int spin = 0; spin < 2048; ++spin) {
                    cpu_relax();
                    word = atomic_load_explicit(&pool->dispatch_word, memory_order_acquire);
                    cg = word >> 16;
                    if (cg != g) {
                        break;
                    }
                }
                cnw = (size_t)(word & UINT64_C(0xFFFF));
                if (cg != g || wid >= cnw ||
                    atomic_load_explicit(&pool->stopping, memory_order_relaxed)) {
                    break;
                }
            } while (monotonic_ns() < deadline);
        }

        if (cg == g || wid >= cnw) {
            g = cg;
            pthread_mutex_lock(&slot->mtx);
            atomic_fetch_or_explicit(&pool->parked_mask[mask_idx], mask_bit, memory_order_seq_cst);
            while (!atomic_load_explicit(&pool->stopping, memory_order_acquire)) {
                word = atomic_load_explicit(&pool->dispatch_word, memory_order_seq_cst);
                cg = word >> 16;
                cnw = (size_t)(word & UINT64_C(0xFFFF));
                if (cg != g && wid < cnw) {
                    break;
                }
                g = cg;
                pthread_cond_wait(&slot->cv, &slot->mtx);
            }
            atomic_fetch_and_explicit(&pool->parked_mask[mask_idx], ~mask_bit,
                                      memory_order_relaxed);
            pthread_mutex_unlock(&slot->mtx);
        }

        if (atomic_load_explicit(&pool->stopping, memory_order_acquire)) {
            return nullptr;
        }

        pool->fn(pool->fn_ctx, wid, cnw, slot->scratch, pool->scratch_bytes);

        atomic_store_explicit(&slot->done_gen, cg, memory_order_release);
        g = cg;
    }
}

static void signal_worker_slot(velri_worker_slot *slot) {
    if (slot->sync_ready) {
        pthread_mutex_lock(&slot->mtx);
        pthread_cond_signal(&slot->cv);
        pthread_mutex_unlock(&slot->mtx);
    }
}

velri_status velri_pool_create(size_t num_workers, size_t scratch_per_worker,
                               velri_pool **out_pool) {
    if (!out_pool || num_workers == 0 || num_workers > 256) {
        return VELRI_ERR_INVALID;
    }
    *out_pool = nullptr;
    if (scratch_per_worker > SIZE_MAX - 63)
        return VELRI_ERR_OVERFLOW;

    velri_pool *pool = nullptr;
    if (posix_memalign((void **)&pool, 64, sizeof(velri_pool)) != 0 || !pool) {
        return VELRI_ERR_NOMEM;
    }
    memset(pool, 0, sizeof(*pool));
    pool->num_workers = num_workers;
    pool->scratch_bytes = (scratch_per_worker + 63ULL) & ~63ULL;
    pool->last_nw = 1ULL;
    pool->host_gen = UINT64_C(1);
    atomic_init(&pool->stopping, false);
    atomic_init(&pool->dispatch_word, (UINT64_C(1) << 16) | UINT64_C(1));
    for (size_t m = 0; m < 4; ++m) {
        atomic_init(&pool->parked_mask[m], UINT64_C(0));
    }

    if (posix_memalign((void **)&pool->slots, 64, num_workers * sizeof(velri_worker_slot)) != 0 ||
        !pool->slots) {
        free(pool);
        return VELRI_ERR_NOMEM;
    }
    memset(pool->slots, 0, num_workers * sizeof(velri_worker_slot));

    for (size_t i = 0; i < num_workers; ++i) {
        pool->slots[i].worker_id = i;
        pool->slots[i].pool = pool;
        atomic_init(&pool->slots[i].done_gen, UINT64_C(0));
        if (i > 0) {
            if (pthread_mutex_init(&pool->slots[i].mtx, nullptr) != 0) {
                for (size_t j = 0; j < i; ++j) {
                    if (pool->slots[j].sync_ready) {
                        pthread_cond_destroy(&pool->slots[j].cv);
                        pthread_mutex_destroy(&pool->slots[j].mtx);
                    }
                    free(pool->slots[j].scratch);
                }
                free(pool->slots);
                free(pool);
                return VELRI_ERR_NOMEM;
            }
            if (pthread_cond_init(&pool->slots[i].cv, nullptr) != 0) {
                pthread_mutex_destroy(&pool->slots[i].mtx);
                for (size_t j = 0; j < i; ++j) {
                    if (pool->slots[j].sync_ready) {
                        pthread_cond_destroy(&pool->slots[j].cv);
                        pthread_mutex_destroy(&pool->slots[j].mtx);
                    }
                    free(pool->slots[j].scratch);
                }
                free(pool->slots);
                free(pool);
                return VELRI_ERR_NOMEM;
            }
            pool->slots[i].sync_ready = true;
        }
        if (pool->scratch_bytes > 0) {
            if (posix_memalign(&pool->slots[i].scratch, 64, pool->scratch_bytes) != 0 ||
                !pool->slots[i].scratch) {
                for (size_t j = 0; j <= i; ++j) {
                    if (pool->slots[j].sync_ready) {
                        pthread_cond_destroy(&pool->slots[j].cv);
                        pthread_mutex_destroy(&pool->slots[j].mtx);
                    }
                    free(pool->slots[j].scratch);
                }
                free(pool->slots);
                free(pool);
                return VELRI_ERR_NOMEM;
            }
        }
    }

    if (pool->slots[0].scratch && pool->scratch_bytes > 0) {
        memset(pool->slots[0].scratch, 0, pool->scratch_bytes);
    }
    atomic_store_explicit(&pool->slots[0].done_gen, UINT64_C(1), memory_order_release);

    for (size_t i = 1; i < num_workers; ++i) {
        if (pthread_create(&pool->slots[i].thread, nullptr, worker_entry, &pool->slots[i]) != 0) {
            atomic_store_explicit(&pool->stopping, true, memory_order_seq_cst);
            uint64_t next_gen = ++pool->host_gen;
            atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | (uint64_t)num_workers,
                                  memory_order_seq_cst);
            for (size_t j = 1; j < i; ++j) {
                signal_worker_slot(&pool->slots[j]);
            }
            for (size_t j = 1; j < i; ++j) {
                (void)pthread_join(pool->slots[j].thread, nullptr);
            }
            for (size_t j = 0; j < num_workers; ++j) {
                if (pool->slots[j].sync_ready) {
                    pthread_cond_destroy(&pool->slots[j].cv);
                    pthread_mutex_destroy(&pool->slots[j].mtx);
                }
                free(pool->slots[j].scratch);
            }
            free(pool->slots);
            free(pool);
            return VELRI_ERR_NOMEM;
        }
    }

    for (size_t i = 1; i < num_workers; ++i) {
        while (atomic_load_explicit(&pool->slots[i].done_gen, memory_order_acquire) !=
               UINT64_C(1)) {
            cpu_relax();
        }
    }

    *out_pool = pool;
    return VELRI_OK;
}

void velri_pool_destroy(velri_pool *pool) {
    if (!pool) {
        return;
    }
    if (pool->num_workers > 1) {
        atomic_store_explicit(&pool->stopping, true, memory_order_seq_cst);
        uint64_t next_gen = ++pool->host_gen;
        atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | (uint64_t)pool->num_workers,
                              memory_order_seq_cst);
        for (size_t i = 1; i < pool->num_workers; ++i) {
            signal_worker_slot(&pool->slots[i]);
        }
        for (size_t i = 1; i < pool->num_workers; ++i) {
            (void)pthread_join(pool->slots[i].thread, nullptr);
        }
    }
    for (size_t i = 0; i < pool->num_workers; ++i) {
        if (pool->slots[i].sync_ready) {
            pthread_cond_destroy(&pool->slots[i].cv);
            pthread_mutex_destroy(&pool->slots[i].mtx);
        }
        free(pool->slots[i].scratch);
    }
    free(pool->slots);
    free(pool);
}

size_t velri_pool_num_workers(const velri_pool *pool) { return pool ? pool->num_workers : 0; }

size_t velri_pool_scratch_bytes(const velri_pool *pool) { return pool ? pool->scratch_bytes : 0; }

void *velri_pool_worker0_scratch(velri_pool *pool) {
    return (pool && pool->num_workers > 0) ? pool->slots[0].scratch : nullptr;
}

void velri_pool_parallel(velri_pool *pool, size_t active_workers, velri_parallel_fn fn, void *ctx) {
    if (!fn) {
        return;
    }
    if (!pool || active_workers <= 1 || pool->num_workers == 1) {
        if (pool && pool->last_nw > 1) {
            pool->last_nw = 1ULL;
            uint64_t next_gen = ++pool->host_gen;
            atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | UINT64_C(1),
                                  memory_order_release);
        }
        void *s0 = pool ? pool->slots[0].scratch : nullptr;
        size_t sb = pool ? pool->scratch_bytes : 0;
        fn(ctx, 0, 1, s0, sb);
        return;
    }

    size_t nw = active_workers < pool->num_workers ? active_workers : pool->num_workers;
    pool->last_nw = nw;
    pool->fn = fn;
    pool->fn_ctx = ctx;

    uint64_t next_gen = ++pool->host_gen;
    atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | (uint64_t)nw,
                          memory_order_release);
    atomic_thread_fence(memory_order_seq_cst);

    size_t n_words = (nw + 63ULL) >> 6;
    for (size_t m = 0; m < n_words; ++m) {
        uint64_t valid = ~UINT64_C(0);
        if (m == 0) {
            valid &= ~UINT64_C(1); /* slot 0 is caller */
        }
        if ((m + 1ULL) * 64ULL > nw) {
            size_t rem = nw - m * 64ULL;
            valid &= (UINT64_C(1) << rem) - UINT64_C(1);
        }
        uint64_t pm = atomic_load_explicit(&pool->parked_mask[m], memory_order_relaxed) & valid;
        while (pm != 0) {
            size_t bit = (size_t)__builtin_ctzll(pm);
            pm &= pm - UINT64_C(1);
            signal_worker_slot(&pool->slots[m * 64ULL + bit]);
        }
    }

    fn(ctx, 0, nw, pool->slots[0].scratch, pool->scratch_bytes);

    for (size_t i = 1; i < nw; ++i) {
        while (atomic_load_explicit(&pool->slots[i].done_gen, memory_order_acquire) != next_gen) {
            cpu_relax();
        }
    }
}
