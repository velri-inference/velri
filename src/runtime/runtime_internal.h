#ifndef VELRI_RUNTIME_INTERNAL_H
#define VELRI_RUNTIME_INTERNAL_H

#include "config/placement.h"
#include "model/arch_registry.h"
#include "runtime/sampler.h"
#include "velri/velri.h"

#include <velri_lib/velri_lib.h>

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

/* An exclusive owner (destruction or weight binding) excludes all child readers. */
enum { VELRI_EXCLUSIVE_BIT = 0x80000000U };

struct velri_cancel {
    velri_cancel_token tok;
};

struct velri_ctx {
    velri_context *vl_ctx;
    bool vulkan_requested;
    velri_cpu_backend_cfg cpu_policy;
    _Atomic uint32_t live_children; /* Models + Programs; high bit = VELRI_EXCLUSIVE_BIT */
    atomic_flag err_lock;
    char last_error[256];
};

typedef struct {
    char name[VELRI_MANIFEST_NAME_MAX];
    const void *host_ptr;
    size_t byte_size;
} velri_rt_weight_entry;

struct velri_model {
    velri_ctx *ctx;
    velri_model_desc desc;
    const velri_arch_ops *arch_ops;
    velri_dtype vl_dtype;
    velri_act_kind vl_act;
    velri_rope_mode vl_rope;

    velri_rt_weight_entry *weights;
    _Atomic uint32_t num_weights;
    uint32_t cap_weights;
    _Atomic uint32_t live_programs; /* Programs; high bit = VELRI_EXCLUSIVE_BIT */
};

struct velri_program {
    velri_ctx *ctx;
    velri_model *model;
    velri_engine_config cfg;
    velri_placement_plan placement;

    velri_manifest decode_manifest;
    velri_plan *decode_plan;

    bool has_distinct_prefill;
    velri_manifest prefill_manifest;
    velri_plan *prefill_plan;

    float *auto_rope_cos;
    float *auto_rope_sin;
    size_t auto_rope_bytes;

    uint32_t max_ctx;
    uint32_t prefill_chunk_size;
    uint32_t vocab_size;
    velri_dtype logits_dtype;
    uint64_t default_seed;
    bool has_default_seed;

    velri_program_mem_report mem_report;
    _Atomic uint32_t live_sessions; /* Sessions; high bit = VELRI_EXCLUSIVE_BIT */
};

struct velri_session {
    velri_program *prog;
    velri_slot *decode_slot;
    velri_slot *prefill_slot; /* Non-null when prog->has_distinct_prefill; shares STATE */

    int32_t *token_ids_buf;
    size_t token_ids_bytes;
    float *logits_buf;
    size_t logits_bytes;

    velri_sampler sampler;

    _Atomic uint32_t kv_pos;
    uint32_t max_ctx;
    uint32_t prefill_chunk_size;
    uint32_t vocab_size;
    _Atomic bool needs_reset;
    _Atomic bool in_step;
};

static inline bool velri_rt_try_acquire_child(_Atomic uint32_t *counter) {
    uint32_t cur = atomic_load_explicit(counter, memory_order_acquire);
    for (;;) {
        if ((cur & VELRI_EXCLUSIVE_BIT) != 0U || cur >= (VELRI_EXCLUSIVE_BIT - 1U)) {
            return false;
        }
        if (atomic_compare_exchange_weak_explicit(counter, &cur, cur + 1U, memory_order_acq_rel,
                                                  memory_order_acquire)) {
            return true;
        }
    }
}

static inline void velri_rt_release_child(_Atomic uint32_t *counter) {
    atomic_fetch_sub_explicit(counter, 1U, memory_order_acq_rel);
}

static inline bool velri_rt_try_acquire_exclusive(_Atomic uint32_t *counter) {
    uint32_t expected = 0U;
    return atomic_compare_exchange_strong_explicit(counter, &expected, VELRI_EXCLUSIVE_BIT,
                                                   memory_order_acq_rel, memory_order_acquire);
}

static inline void *velri_rt_aligned_alloc(size_t align, size_t bytes) {
    if (align < sizeof(void *)) {
        align = sizeof(void *);
    }
    if (bytes == 0U) {
        bytes = align;
    }
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#else
    size_t rem = bytes % align;
    if (rem != 0U) {
        size_t add = align - rem;
        if (bytes > SIZE_MAX - add) {
            return nullptr;
        }
        bytes += add;
    }
    return aligned_alloc(align, bytes);
#endif
}

static inline void velri_rt_aligned_free(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline size_t velri_rt_sat_add_size(size_t a, size_t b) {
    return (a > SIZE_MAX - b) ? SIZE_MAX : (a + b);
}

/* Cross-TU runtime helpers (defined in context.c and model.c). */
velri_rc velri_rt_rc_from_vl(velri_status st);
velri_rc velri_rt_check_cpu_policy(const velri_cpu_backend_cfg *request,
                                   const velri_cpu_backend_cfg *existing,
                                   char *err_msg, size_t err_cap);
void velri_rt_ctx_set_error(velri_ctx *ctx, const char *fmt, ...);
void velri_rt_ctx_clear_tls_error(const velri_ctx *ctx);
const velri_rt_weight_entry *velri_rt_model_find_weight(const velri_model *model, const char *name);

#endif /* VELRI_RUNTIME_INTERNAL_H */
