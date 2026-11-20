#if defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#define WIN32_LEAN_AND_MEAN
#include <malloc.h>
#include <windows.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <time.h>
#endif

#include "velri_lib/pool.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline void cpu_relax(void) {
#if defined(_WIN32)
    YieldProcessor();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#else
    /* no-op */
#endif
}

static inline size_t ctz64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward64(&idx, x);
    return (size_t)idx;
#else
    return (size_t)__builtin_ctzll(x);
#endif
}

static inline void *aligned_alloc64(size_t bytes) {
    size_t sz = bytes ? ((bytes + 63ULL) & ~63ULL) : 64ULL;
#if defined(_WIN32)
    return _aligned_malloc(sz, 64);
#else
    void *p = nullptr;
    if (posix_memalign(&p, 64, sz) != 0) {
        return nullptr;
    }
    return p;
#endif
}

static inline void aligned_free64(void *p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

static inline uint64_t monotonic_ns(void) {
#if defined(_WIN32)
    static _Atomic uint64_t g_qpf = 0;
    uint64_t freq = atomic_load_explicit(&g_qpf, memory_order_relaxed);
    if (freq == 0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        freq = (f.QuadPart > 0) ? (uint64_t)f.QuadPart : UINT64_C(1);
        atomic_store_explicit(&g_qpf, freq, memory_order_relaxed);
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    uint64_t q = (uint64_t)c.QuadPart;
    return (q / freq) * UINT64_C(1000000000) + ((q % freq) * UINT64_C(1000000000)) / freq;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
#endif
}

typedef struct {
    alignas(64) _Atomic uint64_t done_gen;
    size_t worker_id;
    void *scratch;
    velri_pool *pool;
#if defined(_WIN32)
    HANDLE thread;
    CRITICAL_SECTION mtx;
    CONDITION_VARIABLE cv;
#else
    pthread_t thread;
    pthread_mutex_t mtx;
    pthread_cond_t cv;
#endif
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

static inline bool slot_sync_init(velri_worker_slot *slot) {
#if defined(_WIN32)
    InitializeCriticalSection(&slot->mtx);
    InitializeConditionVariable(&slot->cv);
    slot->sync_ready = true;
    return true;
#else
    if (pthread_mutex_init(&slot->mtx, nullptr) != 0) {
        return false;
    }
    if (pthread_cond_init(&slot->cv, nullptr) != 0) {
        pthread_mutex_destroy(&slot->mtx);
        return false;
    }
    slot->sync_ready = true;
    return true;
#endif
}

static inline void slot_sync_destroy(velri_worker_slot *slot) {
    if (!slot->sync_ready) {
        return;
    }
#if defined(_WIN32)
    DeleteCriticalSection(&slot->mtx);
#else
    pthread_cond_destroy(&slot->cv);
    pthread_mutex_destroy(&slot->mtx);
#endif
    slot->sync_ready = false;
}

static inline void slot_lock(velri_worker_slot *slot) {
#if defined(_WIN32)
    EnterCriticalSection(&slot->mtx);
#else
    pthread_mutex_lock(&slot->mtx);
#endif
}

static inline void slot_unlock(velri_worker_slot *slot) {
#if defined(_WIN32)
    LeaveCriticalSection(&slot->mtx);
#else
    pthread_mutex_unlock(&slot->mtx);
#endif
}

static inline void slot_wait(velri_worker_slot *slot) {
#if defined(_WIN32)
    SleepConditionVariableCS(&slot->cv, &slot->mtx, INFINITE);
#else
    pthread_cond_wait(&slot->cv, &slot->mtx);
#endif
}

static void signal_worker_slot(velri_worker_slot *slot) {
    if (slot->sync_ready) {
        slot_lock(slot);
#if defined(_WIN32)
        WakeConditionVariable(&slot->cv);
#else
        pthread_cond_signal(&slot->cv);
#endif
        slot_unlock(slot);
    }
}

static void worker_loop(velri_worker_slot *slot) {
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
             * wakeups never pull unpinned workers off distinct physical cores. Inactive workers
             * (`wid >= cnw`) skip spinning and park immediately.
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
            slot_lock(slot);
            atomic_fetch_or_explicit(&pool->parked_mask[mask_idx], mask_bit, memory_order_seq_cst);
            while (!atomic_load_explicit(&pool->stopping, memory_order_acquire)) {
                word = atomic_load_explicit(&pool->dispatch_word, memory_order_seq_cst);
                cg = word >> 16;
                cnw = (size_t)(word & UINT64_C(0xFFFF));
                if (cg != g && wid < cnw) {
                    break;
                }
                g = cg;
                slot_wait(slot);
            }
            atomic_fetch_and_explicit(&pool->parked_mask[mask_idx], ~mask_bit,
                                      memory_order_relaxed);
            slot_unlock(slot);
        }

        if (atomic_load_explicit(&pool->stopping, memory_order_acquire)) {
            return;
        }

        pool->fn(pool->fn_ctx, wid, cnw, slot->scratch, pool->scratch_bytes);

        atomic_store_explicit(&slot->done_gen, cg, memory_order_release);
        g = cg;
    }
}

#if defined(_WIN32)
static DWORD WINAPI worker_entry(LPVOID arg) {
    worker_loop((velri_worker_slot *)arg);
    return 0;
}
#else
static void *worker_entry(void *arg) {
    worker_loop((velri_worker_slot *)arg);
    return nullptr;
}
#endif

static bool worker_spawn(velri_worker_slot *slot) {
#if defined(_WIN32)
    slot->thread = CreateThread(nullptr, 0, worker_entry, slot, 0, nullptr);
    return slot->thread != nullptr;
#else
    return pthread_create(&slot->thread, nullptr, worker_entry, slot) == 0;
#endif
}

static void worker_join(velri_worker_slot *slot) {
#if defined(_WIN32)
    if (slot->thread) {
        WaitForSingleObject(slot->thread, INFINITE);
        CloseHandle(slot->thread);
        slot->thread = nullptr;
    }
#else
    (void)pthread_join(slot->thread, nullptr);
#endif
}

static void free_pool_slots(velri_pool *pool, size_t count) {
    for (size_t j = 0; j < count; ++j) {
        slot_sync_destroy(&pool->slots[j]);
        aligned_free64(pool->slots[j].scratch);
    }
    aligned_free64(pool->slots);
    aligned_free64(pool);
}

velri_status velri_pool_create(size_t num_workers, size_t scratch_per_worker,
                               velri_pool **out_pool) {
    if (!out_pool || num_workers == 0 || num_workers > 256) {
        return VELRI_ERR_INVALID;
    }
    *out_pool = nullptr;
    if (scratch_per_worker > SIZE_MAX - 63) {
        return VELRI_ERR_OVERFLOW;
    }

    velri_pool *pool = (velri_pool *)aligned_alloc64(sizeof(velri_pool));
    if (!pool) {
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

    pool->slots = (velri_worker_slot *)aligned_alloc64(num_workers * sizeof(velri_worker_slot));
    if (!pool->slots) {
        aligned_free64(pool);
        return VELRI_ERR_NOMEM;
    }
    memset(pool->slots, 0, num_workers * sizeof(velri_worker_slot));

    for (size_t i = 0; i < num_workers; ++i) {
        pool->slots[i].worker_id = i;
        pool->slots[i].pool = pool;
        atomic_init(&pool->slots[i].done_gen, UINT64_C(0));
        if (i > 0 && !slot_sync_init(&pool->slots[i])) {
            free_pool_slots(pool, i);
            return VELRI_ERR_NOMEM;
        }
        if (pool->scratch_bytes > 0) {
            pool->slots[i].scratch = aligned_alloc64(pool->scratch_bytes);
            if (!pool->slots[i].scratch) {
                free_pool_slots(pool, i + 1);
                return VELRI_ERR_NOMEM;
            }
        }
    }

    if (pool->slots[0].scratch && pool->scratch_bytes > 0) {
        memset(pool->slots[0].scratch, 0, pool->scratch_bytes);
    }
    atomic_store_explicit(&pool->slots[0].done_gen, UINT64_C(1), memory_order_release);

    for (size_t i = 1; i < num_workers; ++i) {
        if (!worker_spawn(&pool->slots[i])) {
            atomic_store_explicit(&pool->stopping, true, memory_order_seq_cst);
            uint64_t next_gen = ++pool->host_gen;
            atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | (uint64_t)num_workers,
                                  memory_order_seq_cst);
            for (size_t j = 1; j < i; ++j) {
                signal_worker_slot(&pool->slots[j]);
            }
            for (size_t j = 1; j < i; ++j) {
                worker_join(&pool->slots[j]);
            }
            free_pool_slots(pool, num_workers);
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
            worker_join(&pool->slots[i]);
        }
    }
    free_pool_slots(pool, pool->num_workers);
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
            size_t bit = ctz64(pm);
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
