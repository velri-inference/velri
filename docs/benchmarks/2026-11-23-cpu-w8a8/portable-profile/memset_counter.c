/* Linux diagnostic wrapper for the unmodified Velri W8A8 graph driver. */
#include "velri_lib/velri_lib.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uintptr_t caller;
    size_t calls;
    size_t bytes;
} callsite;

enum { max_workers = 64, max_sites = 32 };
static struct {
    _Alignas(64) callsite sites[max_sites];
} counts[max_workers];
static _Atomic unsigned next_worker;
static _Thread_local unsigned worker_id;
static _Atomic bool counting;

void *__real_memset(void *, int, size_t);
void *__wrap_memset(void *ptr, int fill, size_t size) {
    if (fill == 0 && atomic_load_explicit(&counting, memory_order_relaxed)) {
        if (worker_id == 0) {
            worker_id = 1 + atomic_fetch_add(&next_worker, 1);
            if (worker_id > max_workers) abort();
        }
        uintptr_t caller = (uintptr_t)__builtin_return_address(0);
        callsite *entry = counts[worker_id - 1].sites;
        unsigned i = 0;
        while (i < max_sites && entry[i].caller && entry[i].caller != caller) ++i;
        if (i == max_sites) abort();
        entry[i].caller = caller;
        ++entry[i].calls;
        entry[i].bytes += size;
    }
    return __real_memset(ptr, fill, size);
}

velri_status __real_velri_slot_run(velri_slot *, const velri_step_params *, velri_cancel_token *);
velri_status __wrap_velri_slot_run(velri_slot *slot, const velri_step_params *params,
                                  velri_cancel_token *token) {
    /* The unmodified driver runs twice before the timed samples. */
    static unsigned invocation;
    if (++invocation != 3) return __real_velri_slot_run(slot, params, token);
    atomic_store(&counting, true);
    velri_status status = __real_velri_slot_run(slot, params, token);
    atomic_store(&counting, false);
    for (unsigned t = 0; t < atomic_load(&next_worker); ++t) {
        for (unsigned j = 0; j < max_sites && counts[t].sites[j].caller; ++j) {
            callsite s = counts[t].sites[j];
            fprintf(stderr, "site %zx calls %zu bytes %zu\n", (size_t)s.caller, s.calls, s.bytes);
        }
    }
    return status;
}
