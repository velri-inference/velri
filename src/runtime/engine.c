#include "velri/velri.h"

#include "config/placement.h"
#include "model/arch/gpt2.h"
#include "model/arch/llama.h"
#include "model/layers.h"
#include "runtime/sampler.h"

#include <velri_lib/velri_lib.h>

#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

static_assert((uint32_t)VELRI_PROGRAM_MAX_DEVICES == (uint32_t)VELRI_MAX_REPORT_DEVICES,
              "VELRI_PROGRAM_MAX_DEVICES must match VELRI_MAX_REPORT_DEVICES");

/* Sentinel bit on atomic child counters: prevents TOCTOU races between parent destroy and child create. */
enum { VELRI_DESTROYING_BIT = 0x80000000U };

static bool try_acquire_child(_Atomic uint32_t *counter) {
    uint32_t cur = atomic_load_explicit(counter, memory_order_acquire);
    for (;;) {
        if ((cur & VELRI_DESTROYING_BIT) != 0U || cur >= (VELRI_DESTROYING_BIT - 1U)) {
            return false;
        }
        if (atomic_compare_exchange_weak_explicit(counter, &cur, cur + 1U, memory_order_acq_rel,
                                                  memory_order_acquire)) {
            return true;
        }
    }
}

static void release_child(_Atomic uint32_t *counter) {
    atomic_fetch_sub_explicit(counter, 1U, memory_order_acq_rel);
}

static bool try_mark_destroying(_Atomic uint32_t *counter) {
    uint32_t expected = 0U;
    return atomic_compare_exchange_strong_explicit(counter, &expected, VELRI_DESTROYING_BIT,
                                                   memory_order_acq_rel, memory_order_acquire);
}

/* ---- Portable Aligned Allocation & Checked Arithmetic Helpers ---- */

static void *engine_aligned_alloc(size_t align, size_t bytes) {
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

static void engine_aligned_free(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline size_t sat_add_size(size_t a, size_t b) {
    return (a > SIZE_MAX - b) ? SIZE_MAX : (a + b);
}

static velri_rc rc_from_vl(velri_status st) {
    switch (st) {
    case VELRI_OK:
        return VELRI_RC_OK;
    case VELRI_ERR_INVALID:
        return VELRI_RC_ERR_INVALID;
    case VELRI_ERR_OVERFLOW:
        return VELRI_RC_ERR_OVERFLOW;
    case VELRI_ERR_UNSUPPORTED:
        return VELRI_RC_ERR_UNSUPPORTED;
    case VELRI_ERR_NOMEM:
        return VELRI_RC_ERR_NOMEM;
    case VELRI_ERR_CANCELLED:
        return VELRI_RC_ERR_CANCELLED;
    case VELRI_ERR_EXEC:
        return VELRI_RC_ERR_EXEC;
    case VELRI_ERR_BUSY:
        return VELRI_RC_ERR_BUSY;
    case VELRI_ERR_NOT_FOUND:
        return VELRI_RC_ERR_NOT_FOUND;
    default:
        return VELRI_RC_ERR_EXEC;
    }
}

const char *velri_rc_str(velri_rc rc) {
    switch (rc) {
    case VELRI_RC_OK:
        return "ok";
    case VELRI_RC_ERR_INVALID:
        return "invalid argument";
    case VELRI_RC_ERR_OVERFLOW:
        return "numeric or capacity overflow";
    case VELRI_RC_ERR_UNSUPPORTED:
        return "unsupported operation or configuration";
    case VELRI_RC_ERR_NOMEM:
        return "out of memory";
    case VELRI_RC_ERR_CANCELLED:
        return "cancelled";
    case VELRI_RC_ERR_EXEC:
        return "execution error";
    case VELRI_RC_ERR_BUSY:
        return "resource busy or requires reset";
    case VELRI_RC_ERR_NOT_FOUND:
        return "not found";
    case VELRI_RC_ERR_IO:
        return "I/O error";
    default:
        return "unknown error";
    }
}

/* ---- Opaque Handle Definitions ---- */

struct velri_cancel {
    velri_cancel_token tok;
};

struct velri_ctx {
    velri_context *vl_ctx;
    bool vulkan_requested;
    _Atomic uint32_t live_children; /* Models + Programs; high bit = VELRI_DESTROYING_BIT */
    atomic_flag err_lock;
    char last_error[256];
};

typedef struct {
    char name[VELRI_MANIFEST_NAME_MAX];
    const void *host_ptr;
    size_t byte_size;
} model_weight_entry;

struct velri_model {
    velri_ctx *ctx;
    velri_model_desc desc;
    model_weight_entry *weights;
    uint32_t num_weights;
    uint32_t cap_weights;
    _Atomic uint32_t live_programs; /* Programs; high bit = VELRI_DESTROYING_BIT */
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

    velri_program_mem_report mem_report;
    _Atomic uint32_t live_sessions; /* Sessions; high bit = VELRI_DESTROYING_BIT */
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

/* ---- Thread-Local + Context-Backed Diagnostics ---- */

typedef struct {
    const velri_ctx *owner;
    char msg[256];
} tls_err_slot;

static _Thread_local tls_err_slot g_tls_err = {.owner = nullptr, .msg = {0}};

static inline void err_lock_acquire(velri_ctx *ctx) {
    while (atomic_flag_test_and_set_explicit(&ctx->err_lock, memory_order_acquire)) {
        /* Held only for <= 256-byte memcpy. */
    }
}

static inline void err_lock_release(velri_ctx *ctx) {
    atomic_flag_clear_explicit(&ctx->err_lock, memory_order_release);
}

static void ctx_set_error(velri_ctx *ctx, const char *fmt, ...) {
    if (!ctx || !fmt) {
        return;
    }
    char tmp[sizeof(ctx->last_error)];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    tmp[sizeof(tmp) - 1U] = '\0';

    g_tls_err.owner = ctx;
    memcpy(g_tls_err.msg, tmp, sizeof(tmp));

    err_lock_acquire(ctx);
    memcpy(ctx->last_error, tmp, sizeof(tmp));
    err_lock_release(ctx);
}

static void ctx_clear_tls_error(const velri_ctx *ctx) {
    if (g_tls_err.owner == ctx) {
        g_tls_err.msg[0] = '\0';
    }
}

/* ---- Cancel Handle API ---- */

velri_rc velri_cancel_create(velri_cancel **out_cancel) {
    if (!out_cancel) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_cancel = nullptr;
    velri_cancel *c = (velri_cancel *)calloc(1U, sizeof(*c));
    if (!c) {
        return VELRI_RC_ERR_NOMEM;
    }
    velri_cancel_init(&c->tok);
    *out_cancel = c;
    return VELRI_RC_OK;
}

void velri_cancel_destroy(velri_cancel *cancel) { free(cancel); }

void velri_cancel_trigger(velri_cancel *cancel) {
    if (cancel) {
        velri_cancel_request(&cancel->tok);
    }
}

void velri_cancel_reset(velri_cancel *cancel) {
    if (cancel) {
        velri_cancel_clear(&cancel->tok);
    }
}

bool velri_cancel_triggered(const velri_cancel *cancel) {
    return cancel ? velri_cancel_is_requested(&cancel->tok) : false;
}

/* ---- Context API ---- */

velri_rc velri_ctx_create(const velri_ctx_options *opts, velri_ctx **out_ctx) {
    if (!out_ctx) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_ctx = nullptr;

    velri_ctx *ctx = (velri_ctx *)calloc(1U, sizeof(*ctx));
    if (!ctx) {
        return VELRI_RC_ERR_NOMEM;
    }
    atomic_init(&ctx->live_children, 0U);
    atomic_flag_clear(&ctx->err_lock);
    ctx->vulkan_requested = opts ? opts->enable_vulkan : false;

    velri_context_config vcfg = {
        .cpu_workers = opts ? (size_t)opts->cpu_threads : 0U,
        .cpu_variant = VELRI_CPU_AUTO,
        .enable_vulkan = ctx->vulkan_requested,
    };
    velri_status st = velri_context_create(&vcfg, &ctx->vl_ctx);
    if (st != VELRI_OK) {
        free(ctx);
        return rc_from_vl(st);
    }

    *out_ctx = ctx;
    return VELRI_RC_OK;
}

velri_rc velri_ctx_destroy(velri_ctx *ctx) {
    if (!ctx) {
        return VELRI_RC_OK;
    }
    if (!try_mark_destroying(&ctx->live_children)) {
        uint32_t lc =
            atomic_load_explicit(&ctx->live_children, memory_order_acquire) & ~VELRI_DESTROYING_BIT;
        ctx_set_error(ctx, "ctx_destroy: cannot destroy context with %u live dependent object(s)",
                      lc);
        return VELRI_RC_ERR_BUSY;
    }
    if (g_tls_err.owner == ctx) {
        g_tls_err.owner = nullptr;
        g_tls_err.msg[0] = '\0';
    }
    velri_context_destroy(ctx->vl_ctx);
    free(ctx);
    return VELRI_RC_OK;
}

uint32_t velri_ctx_device_count(const velri_ctx *ctx) {
    return (ctx && ctx->vl_ctx) ? velri_context_device_count(ctx->vl_ctx) : 0U;
}

velri_rc velri_ctx_device_info(const velri_ctx *ctx, uint32_t device_id,
                               velri_device_info *out_info) {
    if (!ctx || !ctx->vl_ctx || !out_info) {
        return VELRI_RC_ERR_INVALID;
    }
    if (device_id >= velri_context_device_count(ctx->vl_ctx)) {
        ctx_set_error((velri_ctx *)ctx, "ctx_device_info: device %u not found", device_id);
        return VELRI_RC_ERR_NOT_FOUND;
    }
    velri_device_caps caps;
    velri_status st = velri_context_device_caps(ctx->vl_ctx, device_id, &caps);
    if (st != VELRI_OK) {
        ctx_set_error((velri_ctx *)ctx, "ctx_device_info: failed querying device %u", device_id);
        return rc_from_vl(st);
    }
    memset(out_info, 0, sizeof(*out_info));
    out_info->device_id = device_id;
    out_info->local_index = caps.local_index;
    snprintf(out_info->backend_name, sizeof(out_info->backend_name), "%s", caps.backend_name);
    snprintf(out_info->device_name, sizeof(out_info->device_name), "%s", caps.device_name);
    out_info->is_host_memory = caps.is_host_memory;
    out_info->is_uma = caps.is_uma;
    out_info->total_memory_bytes = caps.mem_total_known ? caps.mem_total_bytes : 0ULL;
    return VELRI_RC_OK;
}

const char *velri_ctx_last_error(const velri_ctx *ctx) {
    static _Thread_local char fallback_buf[256];
    if (!ctx) {
        fallback_buf[0] = '\0';
        return fallback_buf;
    }
    if (g_tls_err.owner == ctx) {
        return g_tls_err.msg;
    }
    velri_ctx *mctx = (velri_ctx *)ctx;
    err_lock_acquire(mctx);
    memcpy(fallback_buf, mctx->last_error, sizeof(fallback_buf));
    err_lock_release(mctx);
    fallback_buf[sizeof(fallback_buf) - 1U] = '\0';
    return fallback_buf;
}

/* ---- Enum Translation Helpers ---- */

static bool map_type_to_vl(velri_type t, velri_dtype *out) {
    switch (t) {
    case VELRI_TYPE_F32:
        *out = VELRI_DTYPE_F32;
        return true;
    case VELRI_TYPE_F16:
        *out = VELRI_DTYPE_F16;
        return true;
    case VELRI_TYPE_BF16:
        *out = VELRI_DTYPE_BF16;
        return true;
    default:
        return false;
    }
}

static bool map_act_to_vl(velri_activation a, velri_act_kind *out) {
    switch (a) {
    case VELRI_ACTIVATION_SILU:
        *out = VELRI_ACT_SILU;
        return true;
    case VELRI_ACTIVATION_GELU_TANH:
        *out = VELRI_ACT_GELU_TANH;
        return true;
    case VELRI_ACTIVATION_RELU2:
        *out = VELRI_ACT_RELU2;
        return true;
    default:
        return false;
    }
}

static bool map_rope_to_vl(velri_rotary_mode m, velri_rope_mode *out) {
    switch (m) {
    case VELRI_ROTARY_HALF:
        *out = VELRI_ROPE_HALF;
        return true;
    case VELRI_ROTARY_INTERLEAVED:
        *out = VELRI_ROPE_INTERLEAVED;
        return true;
    default:
        return false;
    }
}

/* ---- Model API ---- */

velri_rc velri_model_create_synthetic(velri_ctx *ctx, const velri_model_desc *desc,
                                      velri_model **out_model) {
    if (!out_model) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_model = nullptr;
    if (!ctx || !desc) {
        if (ctx) {
            ctx_set_error(ctx, "model_create_synthetic: null descriptor");
        }
        return VELRI_RC_ERR_INVALID;
    }
    ctx_clear_tls_error(ctx);

    if (!try_acquire_child(&ctx->live_children)) {
        ctx_set_error(ctx, "model_create_synthetic: context is being destroyed");
        return VELRI_RC_ERR_BUSY;
    }

    if (desc->arch != VELRI_ARCH_GPT2 && desc->arch != VELRI_ARCH_LLAMA) {
        ctx_set_error(ctx, "model_create_synthetic: unsupported architecture %d", (int)desc->arch);
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_UNSUPPORTED;
    }
    velri_dtype vl_dtype;
    velri_act_kind vl_act;
    velri_rope_mode vl_rope;
    if (!map_type_to_vl(desc->weight_type, &vl_dtype) || !map_act_to_vl(desc->hidden_act, &vl_act) ||
        !map_rope_to_vl(desc->rope_mode, &vl_rope)) {
        ctx_set_error(ctx, "model_create_synthetic: invalid dtype, activation, or rotary mode");
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_INVALID;
    }
    if (desc->vocab_size == 0U || desc->vocab_size > (uint32_t)INT32_MAX ||
        desc->max_position_embeddings == 0U || desc->hidden_size == 0U ||
        desc->num_hidden_layers == 0U || desc->num_attention_heads == 0U ||
        desc->intermediate_size == 0U) {
        ctx_set_error(ctx, "model_create_synthetic: model dimensions must be non-zero and valid");
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_INVALID;
    }

    velri_model_desc norm = *desc;
    if (norm.num_key_value_heads == 0U) {
        norm.num_key_value_heads = norm.num_attention_heads;
    }
    if (norm.num_key_value_heads > norm.num_attention_heads ||
        (norm.num_attention_heads % norm.num_key_value_heads) != 0U) {
        ctx_set_error(
            ctx,
            "model_create_synthetic: num_attention_heads (%u) must be a multiple of num_key_value_heads (%u)",
            norm.num_attention_heads, norm.num_key_value_heads);
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_INVALID;
    }
    if (norm.head_dim == 0U) {
        if ((norm.hidden_size % norm.num_attention_heads) != 0U) {
            ctx_set_error(
                ctx,
                "model_create_synthetic: hidden_size (%u) not divisible by num_attention_heads (%u)",
                norm.hidden_size, norm.num_attention_heads);
            release_child(&ctx->live_children);
            return VELRI_RC_ERR_INVALID;
        }
        norm.head_dim = norm.hidden_size / norm.num_attention_heads;
    }
    if (norm.head_dim == 0U || norm.head_dim > VELRI_ATTN_MAX_HEAD_DIM) {
        ctx_set_error(ctx, "model_create_synthetic: invalid head_dim %u", norm.head_dim);
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_INVALID;
    }
    if (norm.norm_eps == 0.0f) {
        norm.norm_eps = 1e-5f;
    }
    if (!isfinite(norm.norm_eps) || norm.norm_eps <= 0.0f) {
        ctx_set_error(ctx, "model_create_synthetic: norm_eps must be finite and > 0");
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_INVALID;
    }
    if (norm.arch == VELRI_ARCH_LLAMA) {
        if (norm.rope_theta == 0.0f) {
            norm.rope_theta = 10000.0f;
        }
        if (!isfinite(norm.rope_theta) || norm.rope_theta <= 0.0f) {
            ctx_set_error(ctx, "model_create_synthetic: rope_theta must be finite and > 0");
            release_child(&ctx->live_children);
            return VELRI_RC_ERR_INVALID;
        }
        if (norm.rotary_dim == 0U) {
            norm.rotary_dim = norm.head_dim;
        }
        if (norm.rotary_dim > norm.head_dim || (norm.rotary_dim & 1U) != 0U) {
            ctx_set_error(ctx,
                          "model_create_synthetic: rotary_dim (%u) must be even and <= head_dim (%u)",
                          norm.rotary_dim, norm.head_dim);
            release_child(&ctx->live_children);
            return VELRI_RC_ERR_INVALID;
        }
    }

    velri_model *m = (velri_model *)calloc(1U, sizeof(*m));
    if (!m) {
        ctx_set_error(ctx, "model_create_synthetic: out of memory");
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_NOMEM;
    }
    m->ctx = ctx;
    m->desc = norm;
    atomic_init(&m->live_programs, 0U);

    *out_model = m;
    return VELRI_RC_OK;
}

velri_rc velri_model_destroy(velri_model *model) {
    if (!model) {
        return VELRI_RC_OK;
    }
    if (!try_mark_destroying(&model->live_programs)) {
        uint32_t lp = atomic_load_explicit(&model->live_programs, memory_order_acquire) &
                      ~VELRI_DESTROYING_BIT;
        ctx_set_error(model->ctx, "model_destroy: cannot destroy model with %u live program(s)", lp);
        return VELRI_RC_ERR_BUSY;
    }
    velri_ctx *ctx = model->ctx;
    free(model->weights);
    free(model);
    release_child(&ctx->live_children);
    return VELRI_RC_OK;
}

static model_weight_entry *model_find_weight(velri_model *model, const char *name) {
    for (uint32_t i = 0U; i < model->num_weights; ++i) {
        if (strcmp(model->weights[i].name, name) == 0) {
            return &model->weights[i];
        }
    }
    return nullptr;
}

velri_rc velri_model_bind_weight(velri_model *model, const char *name, const void *host_data,
                                 size_t byte_size) {
    if (!model) {
        return VELRI_RC_ERR_INVALID;
    }
    if (atomic_load_explicit(&model->live_programs, memory_order_acquire) != 0U) {
        ctx_set_error(model->ctx, "model_bind_weight: model is immutable while programs are alive");
        return VELRI_RC_ERR_BUSY;
    }
    if (!name || name[0] == '\0' || !host_data || byte_size == 0U) {
        ctx_set_error(model->ctx, "model_bind_weight: invalid weight name, pointer, or size");
        return VELRI_RC_ERR_INVALID;
    }
    size_t nlen = strlen(name);
    if (nlen >= VELRI_MANIFEST_NAME_MAX) {
        ctx_set_error(model->ctx, "model_bind_weight: weight name exceeds %u bytes",
                      (unsigned)(VELRI_MANIFEST_NAME_MAX - 1U));
        return VELRI_RC_ERR_OVERFLOW;
    }

    model_weight_entry *existing = model_find_weight(model, name);
    if (existing) {
        existing->host_ptr = host_data;
        existing->byte_size = byte_size;
        return VELRI_RC_OK;
    }

    if (model->num_weights == model->cap_weights) {
        uint32_t new_cap = model->cap_weights ? (model->cap_weights * 2U) : 64U;
        size_t new_cap_sz = new_cap;
        if (new_cap <= model->cap_weights || new_cap_sz > SIZE_MAX / sizeof(model_weight_entry)) {
            ctx_set_error(model->ctx, "model_bind_weight: weight table overflow");
            return VELRI_RC_ERR_OVERFLOW;
        }
        model_weight_entry *nb =
            (model_weight_entry *)realloc(model->weights, new_cap_sz * sizeof(model_weight_entry));
        if (!nb) {
            ctx_set_error(model->ctx, "model_bind_weight: out of memory");
            return VELRI_RC_ERR_NOMEM;
        }
        model->weights = nb;
        model->cap_weights = new_cap;
    }

    model_weight_entry *e = &model->weights[model->num_weights++];
    memcpy(e->name, name, nlen + 1U);
    e->host_ptr = host_data;
    e->byte_size = byte_size;
    return VELRI_RC_OK;
}

uint32_t velri_model_weight_count(const velri_model *model) {
    return model ? model->num_weights : 0U;
}

velri_rc velri_model_get_desc(const velri_model *model, velri_model_desc *out_desc) {
    if (!model || !out_desc) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_desc = model->desc;
    return VELRI_RC_OK;
}

/* ---- Program Construction Helpers ---- */

static velri_device_id select_default_device(const velri_ctx *ctx) {
    velri_device_id dev = 0U;
    if (ctx->vulkan_requested &&
        velri_context_find_device(ctx->vl_ctx, "vulkan", 0U, &dev) == VELRI_OK) {
        return dev;
    }
    if (velri_context_find_device(ctx->vl_ctx, "cpu", 0U, &dev) == VELRI_OK) {
        return dev;
    }
    return 0U;
}

/* Check ONLY scopes actually exercised by dense architectures (`GPT-2`, `Llama`) so unused
 * MoE/GDN/PLE rules in a shared config never allocate a redundant second prefill plan. */
static bool arch_has_distinct_prefill(const velri_placement_plan *plan, uint32_t num_hidden_layers) {
    static const char *const k_dense_root_scopes[] = {"", "input-layer", "lm-head"};
    for (size_t i = 0U; i < sizeof(k_dense_root_scopes) / sizeof(k_dense_root_scopes[0]); ++i) {
        velri_resolved_placement res;
        if (velri_placement_resolve(plan, k_dense_root_scopes[i], -1, &res, nullptr, 0U) ==
                VELRI_OK &&
            res.prefill_dev != res.compute_dev) {
            return true;
        }
    }
    static const char *const k_dense_layer_scopes[] = {
        "hidden-layers",
        "hidden-layers/attention",
        "hidden-layers/attention/kv-cache",
        "hidden-layers/dense-mlp",
    };
    for (uint32_t l = 0U; l < num_hidden_layers; ++l) {
        for (size_t i = 0U; i < sizeof(k_dense_layer_scopes) / sizeof(k_dense_layer_scopes[0]);
             ++i) {
            velri_resolved_placement res;
            if (velri_placement_resolve(plan, k_dense_layer_scopes[i], (int32_t)l, &res, nullptr,
                                        0U) == VELRI_OK &&
                res.prefill_dev != res.compute_dev) {
                return true;
            }
        }
    }
    return false;
}

static velri_rc build_arch_graph(velri_ctx *ctx, const velri_model *model,
                                 const velri_placement_plan *placement, bool is_prefill,
                                 uint32_t max_seq_len, uint32_t max_ctx, bool latent_kv,
                                 velri_graph **out_graph, velri_manifest *out_manifest) {
    velri_graph *g = nullptr;
    velri_status st = velri_graph_create(&g);
    if (st != VELRI_OK) {
        ctx_set_error(ctx, "program_prepare: graph creation failed (%s)",
                      velri_rc_str(rc_from_vl(st)));
        return rc_from_vl(st);
    }

    velri_manifest_init(out_manifest);

    velri_placement_builder_adapter ad;
    memset(&ad, 0, sizeof(ad));
    ad.plan = placement;
    ad.is_prefill = is_prefill;
    ad.err = VELRI_OK;

    velri_builder b;
    velri_b_init(&b, g, out_manifest, placement->default_store_dev, placement->default_compute_dev,
                 velri_placement_builder_cb, &ad);
    ad.builder = &b;

    velri_dtype vl_dtype = VELRI_DTYPE_F32;
    velri_act_kind vl_act = VELRI_ACT_SILU;
    velri_rope_mode vl_rope = VELRI_ROPE_HALF;
    (void)map_type_to_vl(model->desc.weight_type, &vl_dtype);
    (void)map_act_to_vl(model->desc.hidden_act, &vl_act);
    (void)map_rope_to_vl(model->desc.rope_mode, &vl_rope);

    velri_model_params mp = {
        .vocab_size = model->desc.vocab_size,
        .max_position_embeddings = model->desc.max_position_embeddings,
        .hidden_size = model->desc.hidden_size,
        .num_hidden_layers = model->desc.num_hidden_layers,
        .num_attention_heads = model->desc.num_attention_heads,
        .num_key_value_heads = model->desc.num_key_value_heads,
        .head_dim = model->desc.head_dim,
        .intermediate_size = model->desc.intermediate_size,
        .sliding_window = model->desc.sliding_window,
        .norm_eps = model->desc.norm_eps,
        .hidden_act = vl_act,
        .tie_word_embeddings = model->desc.tie_word_embeddings,
        .dtype = vl_dtype,
        .rope_theta = model->desc.rope_theta,
        .rotary_dim = model->desc.rotary_dim,
        .rope_mode = vl_rope,
        .qkv_bias = model->desc.qkv_bias,
        .max_seq_len = max_seq_len,
        .max_ctx = max_ctx,
        .latent_kv = latent_kv,
    };

    if (model->desc.arch == VELRI_ARCH_GPT2) {
        st = velri_arch_gpt2_build(&b, &mp);
    } else if (model->desc.arch == VELRI_ARCH_LLAMA) {
        st = velri_arch_llama_build(&b, &mp);
    } else {
        st = VELRI_ERR_UNSUPPORTED;
    }

    if (st != VELRI_OK) {
        if (ad.err != VELRI_OK && ad.err_msg[0] != '\0') {
            ctx_set_error(ctx, "program_prepare: %s", ad.err_msg);
            st = ad.err;
        } else {
            ctx_set_error(ctx, "program_prepare: failed building %s graph (%s)",
                          is_prefill ? "prefill" : "decode", velri_rc_str(rc_from_vl(st)));
        }
        velri_manifest_destroy(out_manifest);
        velri_graph_destroy(g);
        return rc_from_vl(st);
    }

    if (out_manifest->token_ids_in == VELRI_INVALID_ID ||
        out_manifest->logits_out == VELRI_INVALID_ID) {
        ctx_set_error(ctx, "program_prepare: graph missing token_ids input or logits output");
        velri_manifest_destroy(out_manifest);
        velri_graph_destroy(g);
        return VELRI_RC_ERR_INVALID;
    }

    *out_graph = g;
    return VELRI_RC_OK;
}

static velri_rc ensure_auto_rope(velri_program *prog) {
    if (prog->auto_rope_cos && prog->auto_rope_sin) {
        return VELRI_RC_OK;
    }
    uint32_t half = prog->model->desc.rotary_dim / 2U;
    if (half == 0U || (size_t)prog->max_ctx > SIZE_MAX / (size_t)half) {
        ctx_set_error(prog->ctx, "program_prepare: RoPE table size overflow");
        return VELRI_RC_ERR_OVERFLOW;
    }
    size_t elems = (size_t)prog->max_ctx * (size_t)half;
    if (elems > SIZE_MAX / sizeof(float)) {
        ctx_set_error(prog->ctx, "program_prepare: RoPE table byte size overflow");
        return VELRI_RC_ERR_OVERFLOW;
    }
    size_t bytes = elems * sizeof(float);
    float *c = (float *)engine_aligned_alloc(64U, bytes);
    float *s = (float *)engine_aligned_alloc(64U, bytes);
    if (!c || !s) {
        engine_aligned_free(c);
        engine_aligned_free(s);
        ctx_set_error(prog->ctx, "program_prepare: out of memory allocating auto RoPE tables");
        return VELRI_RC_ERR_NOMEM;
    }
    velri_status st = velri_fill_rope_tables(c, s, prog->max_ctx, prog->model->desc.rotary_dim,
                                             prog->model->desc.rope_theta);
    if (st != VELRI_OK) {
        engine_aligned_free(c);
        engine_aligned_free(s);
        ctx_set_error(prog->ctx, "program_prepare: failed generating RoPE tables");
        return rc_from_vl(st);
    }
    prog->auto_rope_cos = c;
    prog->auto_rope_sin = s;
    prog->auto_rope_bytes = bytes;
    return VELRI_RC_OK;
}

static velri_rc bind_program_weights(velri_program *prog) {
    /* On Llama, custom RoPE tables must be supplied as a complete pair (both cos & sin or neither). */
    if (prog->model->desc.arch == VELRI_ARCH_LLAMA) {
        bool has_cos = (model_find_weight(prog->model, "model.rope.cos") != nullptr);
        bool has_sin = (model_find_weight(prog->model, "model.rope.sin") != nullptr);
        if (has_cos != has_sin) {
            ctx_set_error(prog->ctx,
                          "program_prepare: must bind both 'model.rope.cos' and 'model.rope.sin' "
                          "(or omit both for auto-generation)");
            return VELRI_RC_ERR_INVALID;
        }
    }

    bool auto_rope_on_host = false;
    const velri_manifest *m = &prog->decode_manifest;
    for (uint32_t i = 0U; i < m->num_weights; ++i) {
        const velri_manifest_entry *e = &m->weights[i];
        const char *lookup = (e->source_name[0] != '\0') ? e->source_name : e->name;
        const void *ptr = nullptr;
        size_t bytes = 0U;
        bool is_auto_rope = false;

        model_weight_entry *mw = model_find_weight(prog->model, lookup);
        if (mw) {
            ptr = mw->host_ptr;
            bytes = mw->byte_size;
        } else if (prog->model->desc.arch == VELRI_ARCH_LLAMA &&
                   (strcmp(lookup, "model.rope.cos") == 0 ||
                    strcmp(lookup, "model.rope.sin") == 0)) {
            velri_rc rrc = ensure_auto_rope(prog);
            if (rrc != VELRI_RC_OK) {
                return rrc;
            }
            ptr = (strcmp(lookup, "model.rope.cos") == 0) ? (const void *)prog->auto_rope_cos
                                                          : (const void *)prog->auto_rope_sin;
            bytes = prog->auto_rope_bytes;
            is_auto_rope = true;
        } else {
            ctx_set_error(prog->ctx, "program_prepare: missing required weight '%s'", lookup);
            return VELRI_RC_ERR_NOT_FOUND;
        }

        velri_status st = velri_plan_bind_weight_host(prog->decode_plan, e->storage_id, ptr, bytes);
        if (st != VELRI_OK) {
            ctx_set_error(prog->ctx, "program_prepare: failed binding weight '%s' (%zu B): %s",
                          lookup, bytes, velri_rc_str(rc_from_vl(st)));
            return rc_from_vl(st);
        }

        if (is_auto_rope) {
            velri_device_caps caps;
            if (velri_context_device_caps(prog->ctx->vl_ctx, e->device_id, &caps) != VELRI_OK ||
                caps.is_host_memory) {
                auto_rope_on_host = true;
            }
        }
    }

    /* Release temporary host RoPE staging buffers immediately when uploaded solely to non-host VRAM. */
    if (!auto_rope_on_host && (prog->auto_rope_cos || prog->auto_rope_sin)) {
        engine_aligned_free(prog->auto_rope_cos);
        engine_aligned_free(prog->auto_rope_sin);
        prog->auto_rope_cos = nullptr;
        prog->auto_rope_sin = nullptr;
        prog->auto_rope_bytes = 0U;
    }
    return VELRI_RC_OK;
}

static velri_rc compute_program_memory_report(velri_program *prog) {
    velri_memory_report dec_rep;
    velri_status st = velri_plan_memory_report(prog->decode_plan, &dec_rep);
    if (st != VELRI_OK) {
        return rc_from_vl(st);
    }

    velri_program_mem_report *out = &prog->mem_report;
    memset(out, 0, sizeof(*out));
    out->num_devices =
        (dec_rep.num_devices <= VELRI_PROGRAM_MAX_DEVICES) ? dec_rep.num_devices : VELRI_PROGRAM_MAX_DEVICES;
    out->has_distinct_prefill_plan = prog->has_distinct_prefill;
    out->max_ctx = prog->max_ctx;
    out->prefill_chunk_size = prog->prefill_chunk_size;
    out->num_ram_transfers_decode = dec_rep.num_ram_transfers;
    out->ram_staging_bytes = dec_rep.ram_staging_bytes;

    for (uint32_t d = 0U; d < out->num_devices; ++d) {
        out->devices[d] = (velri_program_device_mem){
            .device_id = dec_rep.devices[d].device_id,
            .is_uma = dec_rep.devices[d].is_uma,
            .weight_bytes = dec_rep.devices[d].weight_bytes,
            .state_bytes = dec_rep.devices[d].state_bytes,
            .input_output_bytes = dec_rep.devices[d].input_output_bytes,
            .scratch_packed_bytes = dec_rep.devices[d].scratch_packed_bytes,
            .scratch_unpacked_bytes = dec_rep.devices[d].scratch_unpacked_bytes,
            .allocator_overhead_bytes = dec_rep.devices[d].allocator_overhead_bytes,
        };
    }

    if (prog->has_distinct_prefill && prog->prefill_plan) {
        velri_memory_report pre_rep;
        st = velri_plan_memory_report(prog->prefill_plan, &pre_rep);
        if (st != VELRI_OK) {
            return rc_from_vl(st);
        }
        out->num_ram_transfers_prefill = pre_rep.num_ram_transfers;
        out->ram_staging_bytes = sat_add_size(out->ram_staging_bytes, pre_rep.ram_staging_bytes);

        uint32_t pnd = (pre_rep.num_devices <= VELRI_PROGRAM_MAX_DEVICES)
                           ? pre_rep.num_devices
                           : VELRI_PROGRAM_MAX_DEVICES;
        for (uint32_t d = 0U; d < pnd; ++d) {
            velri_program_device_mem *dm = &out->devices[d];
            velri_device_caps caps;
            bool is_host = (velri_context_device_caps(prog->ctx->vl_ctx, d, &caps) == VELRI_OK &&
                            caps.is_host_memory);

            /* On host-memory devices, `decode_slot` and `prefill_slot` bind the SAME session
             * `token_ids_buf` & `logits_buf` (max, not sum); non-host devices allocate separate IO arenas. */
            if (is_host) {
                if (pre_rep.devices[d].input_output_bytes > dm->input_output_bytes) {
                    dm->input_output_bytes = pre_rep.devices[d].input_output_bytes;
                }
            } else {
                dm->input_output_bytes =
                    sat_add_size(dm->input_output_bytes, pre_rep.devices[d].input_output_bytes);
            }

            dm->scratch_packed_bytes =
                sat_add_size(dm->scratch_packed_bytes, pre_rep.devices[d].scratch_packed_bytes);
            dm->scratch_unpacked_bytes =
                sat_add_size(dm->scratch_unpacked_bytes, pre_rep.devices[d].scratch_unpacked_bytes);

            /* Exclude duplicated STATE alignment padding from prefill overhead (STATE is shared 1x). */
            size_t pre_ovh = pre_rep.devices[d].allocator_overhead_bytes;
            size_t pre_st_ovh = pre_rep.devices[d].state_overhead_bytes;
            size_t pre_extra_ovh = (pre_ovh > pre_st_ovh) ? (pre_ovh - pre_st_ovh) : 0U;
            dm->allocator_overhead_bytes = sat_add_size(dm->allocator_overhead_bytes, pre_extra_ovh);
        }
    }

    size_t total = out->ram_staging_bytes;
    for (uint32_t d = 0U; d < out->num_devices; ++d) {
        const velri_program_device_mem *dm = &out->devices[d];
        total = sat_add_size(total, dm->weight_bytes);
        total = sat_add_size(total, dm->state_bytes);
        total = sat_add_size(total, dm->input_output_bytes);
        total = sat_add_size(total, dm->scratch_packed_bytes);
        total = sat_add_size(total, dm->allocator_overhead_bytes);
    }
    out->total_physical_bytes = total;
    return VELRI_RC_OK;
}

static void program_cleanup_partial(velri_program *prog) {
    if (!prog) {
        return;
    }
    if (prog->prefill_plan) {
        velri_plan_destroy(prog->prefill_plan);
    }
    if (prog->decode_plan) {
        velri_plan_destroy(prog->decode_plan);
    }
    velri_manifest_destroy(&prog->prefill_manifest);
    velri_manifest_destroy(&prog->decode_manifest);
    engine_aligned_free(prog->auto_rope_cos);
    engine_aligned_free(prog->auto_rope_sin);
    velri_engine_config_destroy(&prog->cfg);
    free(prog);
}

/* ---- Program API ---- */

velri_rc velri_program_prepare(velri_ctx *ctx, velri_model *model,
                               const velri_program_options *opts, velri_program **out_prog) {
    if (!out_prog) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_prog = nullptr;
    if (!ctx || !model) {
        if (ctx) {
            ctx_set_error(ctx, "program_prepare: null model");
        }
        return VELRI_RC_ERR_INVALID;
    }
    ctx_clear_tls_error(ctx);

    if (model->ctx != ctx) {
        ctx_set_error(ctx, "program_prepare: model belongs to a different context");
        return VELRI_RC_ERR_INVALID;
    }
    if (opts && ((opts->placement_json == nullptr && opts->placement_json_len > 0U) ||
                 (opts->placement_json != nullptr && opts->placement_json_len == 0U))) {
        ctx_set_error(ctx, "program_prepare: placement_json and placement_json_len must both be set or both empty");
        return VELRI_RC_ERR_INVALID;
    }

    /* Acquire parent child-references UP FRONT so concurrent model_bind_weight / destroy fail BUSY. */
    if (!try_acquire_child(&ctx->live_children)) {
        ctx_set_error(ctx, "program_prepare: context is being destroyed");
        return VELRI_RC_ERR_BUSY;
    }
    if (!try_acquire_child(&model->live_programs)) {
        ctx_set_error(ctx, "program_prepare: model is being destroyed");
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_BUSY;
    }

    velri_program *prog = (velri_program *)calloc(1U, sizeof(*prog));
    if (!prog) {
        ctx_set_error(ctx, "program_prepare: out of memory");
        release_child(&model->live_programs);
        release_child(&ctx->live_children);
        return VELRI_RC_ERR_NOMEM;
    }
    prog->ctx = ctx;
    prog->model = model;
    prog->vocab_size = model->desc.vocab_size;
    prog->logits_dtype = VELRI_DTYPE_F32; /* `velri_block_lm_head` always emits F32 logits */
    velri_engine_config_init(&prog->cfg);
    velri_manifest_init(&prog->decode_manifest);
    velri_manifest_init(&prog->prefill_manifest);
    atomic_init(&prog->live_sessions, 0U);

    velri_rc fail_rc = VELRI_RC_OK;
    char err_buf[VELRI_CFG_ERR_MAX] = {0};

    if (opts && opts->placement_json && opts->placement_json_len > 0U) {
        velri_status st = velri_engine_config_parse(opts->placement_json, opts->placement_json_len,
                                                    &prog->cfg, err_buf, sizeof(err_buf));
        if (st != VELRI_OK) {
            ctx_set_error(ctx, "program_prepare: %s",
                          err_buf[0] ? err_buf : "invalid placement JSON");
            fail_rc = rc_from_vl(st);
            goto fail;
        }
    }

    /* Reject parsed config options that are not yet implemented instead of silently ignoring them. */
    if (prog->cfg.kv_k_bits != 0U || prog->cfg.kv_v_bits != 0U) {
        ctx_set_error(ctx, "program_prepare: kv-quantization (k-bits=%u, v-bits=%u) is not yet supported",
                      prog->cfg.kv_k_bits, prog->cfg.kv_v_bits);
        fail_rc = VELRI_RC_ERR_UNSUPPORTED;
        goto fail;
    }
    if (prog->cfg.cpu_backend.affinity == VELRI_AFFINITY_REQUIRED) {
        ctx_set_error(ctx, "program_prepare: backends.cpu.affinity='required' is not yet supported");
        fail_rc = VELRI_RC_ERR_UNSUPPORTED;
        goto fail;
    }

    bool strict_layers = opts ? opts->strict_layer_selectors : false;
    velri_status st = velri_engine_config_validate_layers(
        &prog->cfg, model->desc.num_hidden_layers, strict_layers, err_buf, sizeof(err_buf));
    if (st != VELRI_OK) {
        ctx_set_error(ctx, "program_prepare: %s",
                      err_buf[0] ? err_buf : "layer selector validation failed");
        fail_rc = rc_from_vl(st);
        goto fail;
    }

    velri_device_id fallback_dev = select_default_device(ctx);
    st = velri_placement_plan_bind(&prog->cfg, ctx->vl_ctx, fallback_dev, &prog->placement, err_buf,
                                   sizeof(err_buf));
    if (st != VELRI_OK) {
        ctx_set_error(ctx, "program_prepare: %s", err_buf[0] ? err_buf : "placement bind failed");
        fail_rc = rc_from_vl(st);
        goto fail;
    }

    uint32_t max_ctx = (opts && opts->max_ctx > 0U)
                           ? opts->max_ctx
                           : (prog->cfg.context_size > 0U ? prog->cfg.context_size
                                                          : model->desc.max_position_embeddings);
    if (max_ctx == 0U) {
        ctx_set_error(ctx, "program_prepare: max_ctx must be > 0");
        fail_rc = VELRI_RC_ERR_INVALID;
        goto fail;
    }
    if (model->desc.arch == VELRI_ARCH_GPT2 && max_ctx > model->desc.max_position_embeddings) {
        ctx_set_error(ctx,
                      "program_prepare: max_ctx (%u) exceeds GPT-2 max_position_embeddings (%u)",
                      max_ctx, model->desc.max_position_embeddings);
        fail_rc = VELRI_RC_ERR_INVALID;
        goto fail;
    }

    uint32_t chunk = 64U;
    if (opts && opts->prefill_chunk_size > 0U) {
        chunk = opts->prefill_chunk_size;
    } else if (prog->cfg.vulkan_backend.prefill_chunk_size > 0U && ctx->vulkan_requested) {
        chunk = prog->cfg.vulkan_backend.prefill_chunk_size;
        if (prog->cfg.max_seq_len > 0U && chunk > prog->cfg.max_seq_len) {
            chunk = prog->cfg.max_seq_len;
        }
    } else if (prog->cfg.max_seq_len > 0U) {
        chunk = prog->cfg.max_seq_len;
    }
    if (chunk > max_ctx) {
        chunk = max_ctx;
    }
    if (chunk == 0U) {
        chunk = 1U;
    }

    bool latent_kv =
        (opts && opts->latent_kv) || (prog->cfg.attn_mechanism == VELRI_ATTN_MECH_LATENT_KV);
    bool allow_cpu_fb = opts ? opts->allow_cpu_fallback : false;

    prog->max_ctx = max_ctx;
    prog->prefill_chunk_size = chunk;
    prog->has_distinct_prefill =
        (max_ctx > 1U) && arch_has_distinct_prefill(&prog->placement, model->desc.num_hidden_layers);

    uint32_t dec_max_seq = prog->has_distinct_prefill ? 1U : chunk;
    velri_graph *dec_graph = nullptr;
    fail_rc = build_arch_graph(ctx, model, &prog->placement, false, dec_max_seq, max_ctx, latent_kv,
                               &dec_graph, &prog->decode_manifest);
    if (fail_rc != VELRI_RC_OK) {
        goto fail;
    }

    velri_plan_options dec_opts = {
        .allow_cpu_fallback = allow_cpu_fb,
        .share_weights_from = nullptr,
    };
    st = velri_plan_prepare(ctx->vl_ctx, dec_graph, &dec_opts, &prog->decode_plan);
    velri_graph_destroy(dec_graph);
    if (st != VELRI_OK) {
        ctx_set_error(ctx, "program_prepare: decode plan preparation failed (%s)",
                      velri_rc_str(rc_from_vl(st)));
        fail_rc = rc_from_vl(st);
        goto fail;
    }

    if (prog->has_distinct_prefill) {
        velri_graph *pre_graph = nullptr;
        fail_rc = build_arch_graph(ctx, model, &prog->placement, true, chunk, max_ctx, latent_kv,
                                   &pre_graph, &prog->prefill_manifest);
        if (fail_rc != VELRI_RC_OK) {
            goto fail;
        }
        velri_plan_options pre_opts = {
            .allow_cpu_fallback = allow_cpu_fb,
            .share_weights_from = prog->decode_plan,
        };
        st = velri_plan_prepare(ctx->vl_ctx, pre_graph, &pre_opts, &prog->prefill_plan);
        velri_graph_destroy(pre_graph);
        if (st != VELRI_OK) {
            ctx_set_error(ctx, "program_prepare: prefill plan preparation failed (%s)",
                          velri_rc_str(rc_from_vl(st)));
            fail_rc = rc_from_vl(st);
            goto fail;
        }
    }

    /* Bind immutable weights ONCE on `decode_plan`; `prefill_plan` shares slabs & host bindings. */
    fail_rc = bind_program_weights(prog);
    if (fail_rc != VELRI_RC_OK) {
        goto fail;
    }

    fail_rc = compute_program_memory_report(prog);
    if (fail_rc != VELRI_RC_OK) {
        ctx_set_error(ctx, "program_prepare: failed computing memory report");
        goto fail;
    }

    *out_prog = prog;
    return VELRI_RC_OK;

fail:
    program_cleanup_partial(prog);
    release_child(&model->live_programs);
    release_child(&ctx->live_children);
    return fail_rc;
}

velri_rc velri_program_destroy(velri_program *prog) {
    if (!prog) {
        return VELRI_RC_OK;
    }
    if (!try_mark_destroying(&prog->live_sessions)) {
        uint32_t ls = atomic_load_explicit(&prog->live_sessions, memory_order_acquire) &
                      ~VELRI_DESTROYING_BIT;
        ctx_set_error(prog->ctx, "program_destroy: cannot destroy program with %u live session(s)",
                      ls);
        return VELRI_RC_ERR_BUSY;
    }
    velri_model *model = prog->model;
    velri_ctx *ctx = prog->ctx;
    /* Release GPU slabs and plans BEFORE releasing parent references! */
    program_cleanup_partial(prog);
    release_child(&model->live_programs);
    release_child(&ctx->live_children);
    return VELRI_RC_OK;
}

velri_rc velri_program_memory_report(const velri_program *prog,
                                     velri_program_mem_report *out_report) {
    if (!prog || !out_report) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_report = prog->mem_report;
    return VELRI_RC_OK;
}

size_t velri_program_inspect(const velri_program *prog, char *out_buf, size_t out_cap) {
    if (!prog || !out_buf || out_cap == 0U) {
        return 0U;
    }
    velri_memory_report vl_rep;
    memset(&vl_rep, 0, sizeof(vl_rep));
    vl_rep.num_devices = prog->mem_report.num_devices;
    vl_rep.ram_staging_bytes = prog->mem_report.ram_staging_bytes;
    vl_rep.num_ram_transfers =
        prog->mem_report.num_ram_transfers_decode + prog->mem_report.num_ram_transfers_prefill;
    vl_rep.total_physical_bytes = prog->mem_report.total_physical_bytes;
    for (uint32_t d = 0U; d < vl_rep.num_devices && d < VELRI_MAX_REPORT_DEVICES; ++d) {
        const velri_program_device_mem *dm = &prog->mem_report.devices[d];
        vl_rep.devices[d] = (velri_device_memory_report){
            .device_id = dm->device_id,
            .is_uma = dm->is_uma,
            .weight_bytes = dm->weight_bytes,
            .input_output_bytes = dm->input_output_bytes,
            .state_bytes = dm->state_bytes,
            .scratch_unpacked_bytes = dm->scratch_unpacked_bytes,
            .scratch_packed_bytes = dm->scratch_packed_bytes,
            .allocator_overhead_bytes = dm->allocator_overhead_bytes,
        };
    }

    size_t pos = velri_placement_format_inspect(&prog->placement, prog->model->desc.num_hidden_layers,
                                                &vl_rep, out_buf, out_cap);
    if (pos + 1U < out_cap) {
        uint32_t dec_ops = 0U, dec_elided = 0U;
        (void)velri_plan_barrier_stats(prog->decode_plan, &dec_ops, &dec_elided);
        const char *arch_name = (prog->model->desc.arch == VELRI_ARCH_LLAMA) ? "llama" : "gpt2";
        int n = 0;
        if (prog->has_distinct_prefill && prog->prefill_plan) {
            uint32_t pre_ops = 0U, pre_elided = 0U;
            (void)velri_plan_barrier_stats(prog->prefill_plan, &pre_ops, &pre_elided);
            n = snprintf(out_buf + pos, out_cap - pos,
                         "Program: arch=%s | max_ctx=%u | prefill_chunk=%u | mode=dual-plan "
                         "(shared weights+state) | decode_ops=%u (elided_barriers=%u) | "
                         "prefill_ops=%u (elided_barriers=%u)\n",
                         arch_name, prog->max_ctx, prog->prefill_chunk_size, dec_ops, dec_elided,
                         pre_ops, pre_elided);
        } else {
            n = snprintf(out_buf + pos, out_cap - pos,
                         "Program: arch=%s | max_ctx=%u | prefill_chunk=%u | mode=unified | "
                         "ops=%u (elided_barriers=%u)\n",
                         arch_name, prog->max_ctx, prog->prefill_chunk_size, dec_ops, dec_elided);
        }
        if (n > 0) {
            size_t avail = out_cap - 1U - pos;
            pos += ((size_t)n < avail) ? (size_t)n : avail;
        }
    }
    return pos;
}

bool velri_program_has_distinct_prefill(const velri_program *prog) {
    return prog ? prog->has_distinct_prefill : false;
}

velri_rc velri_program_default_sampling(const velri_program *prog,
                                        velri_sampling_params *out_params) {
    if (!prog || !out_params) {
        return VELRI_RC_ERR_INVALID;
    }
    out_params->temperature = prog->cfg.sampling.temperature;
    out_params->top_p = prog->cfg.sampling.top_p;
    out_params->top_k = prog->cfg.sampling.top_k;
    out_params->min_p = prog->cfg.sampling.min_p;
    out_params->seed = prog->cfg.sampling.seed;
    out_params->reseed = false;
    return VELRI_RC_OK;
}

/* ---- Session API ---- */

static void session_cleanup_partial(velri_session *sess) {
    if (!sess) {
        return;
    }
    if (sess->prefill_slot) {
        velri_slot_destroy(sess->prefill_slot);
    }
    if (sess->decode_slot) {
        velri_slot_destroy(sess->decode_slot);
    }
    velri_sampler_destroy(&sess->sampler);
    engine_aligned_free(sess->token_ids_buf);
    engine_aligned_free(sess->logits_buf);
    free(sess);
}

velri_rc velri_session_create(velri_program *prog, const velri_session_options *opts,
                              velri_session **out_sess) {
    if (!out_sess) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_sess = nullptr;
    if (!prog) {
        return VELRI_RC_ERR_INVALID;
    }
    ctx_clear_tls_error(prog->ctx);

    /* Acquire parent reference UP FRONT to prevent races with `velri_program_destroy`. */
    if (!try_acquire_child(&prog->live_sessions)) {
        ctx_set_error(prog->ctx, "session_create: program is being destroyed");
        return VELRI_RC_ERR_BUSY;
    }

    if (prog->logits_dtype != VELRI_DTYPE_F32) {
        ctx_set_error(prog->ctx, "session_create: only F32 logits output is supported currently");
        release_child(&prog->live_sessions);
        return VELRI_RC_ERR_UNSUPPORTED;
    }

    size_t chunk_sz = prog->prefill_chunk_size;
    size_t vocab_sz = prog->vocab_size;
    if (chunk_sz > SIZE_MAX / sizeof(int32_t) || vocab_sz > SIZE_MAX / sizeof(float)) {
        ctx_set_error(prog->ctx, "session_create: IO buffer size overflow");
        release_child(&prog->live_sessions);
        return VELRI_RC_ERR_OVERFLOW;
    }

    velri_session *sess = (velri_session *)calloc(1U, sizeof(*sess));
    if (!sess) {
        ctx_set_error(prog->ctx, "session_create: out of memory");
        release_child(&prog->live_sessions);
        return VELRI_RC_ERR_NOMEM;
    }
    sess->prog = prog;
    sess->max_ctx = prog->max_ctx;
    sess->prefill_chunk_size = prog->prefill_chunk_size;
    sess->vocab_size = prog->vocab_size;
    sess->token_ids_bytes = chunk_sz * sizeof(int32_t);
    sess->logits_bytes = vocab_sz * sizeof(float);
    atomic_init(&sess->kv_pos, 0U);
    atomic_init(&sess->needs_reset, false);
    atomic_init(&sess->in_step, false);

    velri_rc fail_rc = VELRI_RC_OK;

    sess->token_ids_buf = (int32_t *)engine_aligned_alloc(64U, sess->token_ids_bytes);
    sess->logits_buf = (float *)engine_aligned_alloc(64U, sess->logits_bytes);
    if (!sess->token_ids_buf || !sess->logits_buf) {
        ctx_set_error(prog->ctx, "session_create: out of memory allocating IO buffers");
        fail_rc = VELRI_RC_ERR_NOMEM;
        goto fail;
    }
    memset(sess->token_ids_buf, 0, sess->token_ids_bytes);
    memset(sess->logits_buf, 0, sess->logits_bytes);

    uint64_t seed = 0x56454C52495F3031ULL; /* Deterministic default ("VELRI_01") */
    if (opts && opts->has_prng_seed) {
        seed = opts->prng_seed;
    } else if (prog->cfg.sampling.has_seed) {
        seed = prog->cfg.sampling.seed;
    }
    fail_rc = velri_sampler_init(&sess->sampler, prog->vocab_size, seed);
    if (fail_rc != VELRI_RC_OK) {
        ctx_set_error(prog->ctx, "session_create: sampler init failed");
        goto fail;
    }

    velri_status st = velri_slot_create(prog->decode_plan, &sess->decode_slot);
    if (st != VELRI_OK) {
        ctx_set_error(prog->ctx, "session_create: decode slot creation failed (%s)",
                      velri_rc_str(rc_from_vl(st)));
        fail_rc = rc_from_vl(st);
        goto fail;
    }
    st = velri_slot_bind_host(sess->decode_slot, prog->decode_manifest.token_ids_in,
                              sess->token_ids_buf, sess->token_ids_bytes);
    if (st == VELRI_OK) {
        st = velri_slot_bind_host(sess->decode_slot, prog->decode_manifest.logits_out,
                                  sess->logits_buf, sess->logits_bytes);
    }
    if (st != VELRI_OK) {
        ctx_set_error(prog->ctx, "session_create: decode slot IO bind failed (%s)",
                      velri_rc_str(rc_from_vl(st)));
        fail_rc = rc_from_vl(st);
        goto fail;
    }

    if (prog->has_distinct_prefill && prog->prefill_plan) {
        st = velri_slot_create_shared_state(prog->prefill_plan, sess->decode_slot,
                                            &sess->prefill_slot);
        if (st != VELRI_OK) {
            ctx_set_error(prog->ctx,
                          "session_create: shared-state prefill slot creation failed (%s)",
                          velri_rc_str(rc_from_vl(st)));
            fail_rc = rc_from_vl(st);
            goto fail;
        }
        st = velri_slot_bind_host(sess->prefill_slot, prog->prefill_manifest.token_ids_in,
                                  sess->token_ids_buf, sess->token_ids_bytes);
        if (st == VELRI_OK) {
            st = velri_slot_bind_host(sess->prefill_slot, prog->prefill_manifest.logits_out,
                                      sess->logits_buf, sess->logits_bytes);
        }
        if (st != VELRI_OK) {
            ctx_set_error(prog->ctx, "session_create: prefill slot IO bind failed (%s)",
                          velri_rc_str(rc_from_vl(st)));
            fail_rc = rc_from_vl(st);
            goto fail;
        }
    }

    *out_sess = sess;
    return VELRI_RC_OK;

fail:
    session_cleanup_partial(sess);
    release_child(&prog->live_sessions);
    return fail_rc;
}

velri_rc velri_session_destroy(velri_session *sess) {
    if (!sess) {
        return VELRI_RC_OK;
    }
    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&sess->in_step, &expected, true,
                                                 memory_order_acq_rel, memory_order_acquire)) {
        ctx_set_error(sess->prog->ctx, "session_destroy: cannot destroy session during active step");
        return VELRI_RC_ERR_BUSY;
    }
    velri_program *prog = sess->prog;
    session_cleanup_partial(sess);
    release_child(&prog->live_sessions);
    return VELRI_RC_OK;
}

velri_rc velri_session_step(velri_session *sess, const int32_t *token_ids, uint32_t num_tokens,
                            const velri_sampling_params *sampling, velri_cancel *cancel,
                            velri_step_result *out_result) {
    if (!out_result) {
        return VELRI_RC_ERR_INVALID;
    }
    memset(out_result, 0, sizeof(*out_result));
    out_result->stop_reason = VELRI_STEP_STOP_ERROR;
    out_result->sampled_token = -1;

    if (!sess || !token_ids || num_tokens == 0U) {
        if (sess) {
            ctx_set_error(sess->prog->ctx, "session_step: invalid token_ids or num_tokens == 0");
        }
        return VELRI_RC_ERR_INVALID;
    }

    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&sess->in_step, &expected, true,
                                                 memory_order_acq_rel, memory_order_acquire)) {
        ctx_set_error(sess->prog->ctx, "session_step: concurrent call on the same session");
        return VELRI_RC_ERR_BUSY;
    }

    velri_ctx *ctx = sess->prog->ctx;
    velri_rc rc = VELRI_RC_OK;
    uint32_t cur_kv = atomic_load_explicit(&sess->kv_pos, memory_order_relaxed);
    out_result->vocab_size = sess->vocab_size;
    out_result->kv_len = cur_kv;

    /* Strict pre-accept validation: failures here never mutate session state. */
    if (atomic_load_explicit(&sess->needs_reset, memory_order_acquire)) {
        ctx_set_error(ctx, "session_step: session requires velri_session_reset() before continuing");
        rc = VELRI_RC_ERR_BUSY;
        goto done;
    }
    if (sampling) {
        rc = velri_sampler_validate_params(sampling);
        if (rc != VELRI_RC_OK) {
            ctx_set_error(ctx, "session_step: invalid sampling parameters");
            goto done;
        }
    }
    if (num_tokens > sess->max_ctx || cur_kv > sess->max_ctx - num_tokens) {
        ctx_set_error(ctx, "session_step: sequence length (%u + %u) exceeds max_ctx (%u)", cur_kv,
                      num_tokens, sess->max_ctx);
        rc = VELRI_RC_ERR_OVERFLOW;
        goto done;
    }
    for (uint32_t i = 0U; i < num_tokens; ++i) {
        if (token_ids[i] < 0 || (uint32_t)token_ids[i] >= sess->vocab_size) {
            ctx_set_error(ctx, "session_step: token_ids[%u]=%d out of range [0, %u)", i,
                          token_ids[i], sess->vocab_size);
            rc = VELRI_RC_ERR_INVALID;
            goto done;
        }
    }
    if (cancel && velri_cancel_is_requested(&cancel->tok)) {
        out_result->stop_reason = VELRI_STEP_STOP_CANCELLED;
        rc = VELRI_RC_ERR_CANCELLED;
        goto done;
    }

    /* Multi-token steps execute on `prefill_slot` when distinct prefill placement is active;
     * single-token decode steps execute on `decode_slot` (hitting captured decode graphs). */
    velri_slot *active_slot =
        (sess->prog->has_distinct_prefill && num_tokens > 1U && sess->prefill_slot)
            ? sess->prefill_slot
            : sess->decode_slot;
    velri_cancel_token *vl_cancel = cancel ? &cancel->tok : nullptr;

    uint32_t offset = 0U;
    uint64_t total_allocs = 0U;
    uint64_t total_replays = 0U;
    bool any_chunk_committed = false;

    while (offset < num_tokens) {
        if (vl_cancel && velri_cancel_is_requested(vl_cancel)) {
            if (any_chunk_committed) {
                atomic_store_explicit(&sess->needs_reset, true, memory_order_release);
            }
            out_result->stop_reason = VELRI_STEP_STOP_CANCELLED;
            out_result->tokens_processed = offset;
            out_result->kv_len = cur_kv;
            out_result->last_step_allocs = total_allocs;
            out_result->last_step_graph_replays = total_replays;
            rc = VELRI_RC_ERR_CANCELLED;
            goto done;
        }

        uint32_t rem = num_tokens - offset;
        uint32_t chunk = (rem > sess->prefill_chunk_size) ? sess->prefill_chunk_size : rem;
        memcpy(sess->token_ids_buf, token_ids + offset, (size_t)chunk * sizeof(int32_t));

        velri_step_params sp = {
            .pos_offset_override = (size_t)cur_kv,
            .has_pos_offset = true,
            .seq_len_override = (size_t)chunk,
            .has_seq_len = true,
            .kv_len_override = (size_t)cur_kv,
            .has_kv_len = true,
        };

        velri_status st = velri_slot_run(active_slot, &sp, vl_cancel);
        total_allocs += velri_slot_last_run_allocs(active_slot);
        total_replays += velri_slot_last_run_graph_replays(active_slot);

        if (st != VELRI_OK) {
            if (velri_slot_needs_reset(active_slot) || any_chunk_committed) {
                atomic_store_explicit(&sess->needs_reset, true, memory_order_release);
            }
            rc = rc_from_vl(st);
            out_result->stop_reason =
                (st == VELRI_ERR_CANCELLED) ? VELRI_STEP_STOP_CANCELLED : VELRI_STEP_STOP_ERROR;
            out_result->tokens_processed = offset;
            out_result->kv_len = cur_kv;
            out_result->last_step_allocs = total_allocs;
            out_result->last_step_graph_replays = total_replays;
            if (st != VELRI_ERR_CANCELLED) {
                ctx_set_error(ctx, "session_step: slot execution failed at offset %u (%s)", offset,
                              velri_rc_str(rc));
            }
            goto done;
        }

        cur_kv += chunk;
        atomic_store_explicit(&sess->kv_pos, cur_kv, memory_order_release);
        offset += chunk;
        any_chunk_committed = true;
    }

    int32_t sampled = -1;
    if (sampling) {
        rc = velri_sampler_sample(&sess->sampler, sess->logits_buf, sampling, &sampled);
        if (rc != VELRI_RC_OK) {
            atomic_store_explicit(&sess->needs_reset, true, memory_order_release);
            ctx_set_error(ctx, "session_step: sampling failed (%s)", velri_rc_str(rc));
            out_result->stop_reason = VELRI_STEP_STOP_ERROR;
            out_result->tokens_processed = offset;
            out_result->kv_len = cur_kv;
            out_result->last_step_allocs = total_allocs;
            out_result->last_step_graph_replays = total_replays;
            goto done;
        }
    }

    out_result->stop_reason = VELRI_STEP_STOP_OK;
    out_result->logits = sess->logits_buf;
    out_result->vocab_size = sess->vocab_size;
    out_result->sampled_token = sampled;
    out_result->kv_len = cur_kv;
    out_result->tokens_processed = num_tokens;
    out_result->last_step_allocs = total_allocs;
    out_result->last_step_graph_replays = total_replays;
    rc = VELRI_RC_OK;

done:
    atomic_store_explicit(&sess->in_step, false, memory_order_release);
    return rc;
}

velri_rc velri_session_reset(velri_session *sess) {
    if (!sess) {
        return VELRI_RC_ERR_INVALID;
    }
    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&sess->in_step, &expected, true,
                                                 memory_order_acq_rel, memory_order_acquire)) {
        ctx_set_error(sess->prog->ctx, "session_reset: concurrent call during active step");
        return VELRI_RC_ERR_BUSY;
    }

    velri_status st = velri_slot_reset(sess->decode_slot);
    bool decode_reset_ok = (st == VELRI_OK);
    if (decode_reset_ok && sess->prefill_slot) {
        st = velri_slot_reset(sess->prefill_slot);
    }
    velri_rc rc = rc_from_vl(st);
    if (st == VELRI_OK) {
        atomic_store_explicit(&sess->kv_pos, 0U, memory_order_release);
        atomic_store_explicit(&sess->needs_reset, false, memory_order_release);
        velri_sampler_reset(&sess->sampler);
    } else {
        /* If decode_slot already wiped shared STATE before prefill_slot failed, keep dirty! */
        if (decode_reset_ok) {
            atomic_store_explicit(&sess->kv_pos, 0U, memory_order_release);
        }
        atomic_store_explicit(&sess->needs_reset, true, memory_order_release);
        ctx_set_error(sess->prog->ctx, "session_reset: slot reset failed (%s)", velri_rc_str(rc));
    }

    atomic_store_explicit(&sess->in_step, false, memory_order_release);
    return rc;
}

bool velri_session_needs_reset(const velri_session *sess) {
    return sess ? atomic_load_explicit(&sess->needs_reset, memory_order_acquire) : false;
}

uint32_t velri_session_kv_len(const velri_session *sess) {
    return sess ? atomic_load_explicit(&sess->kv_pos, memory_order_acquire) : 0U;
}

uint32_t velri_session_max_ctx(const velri_session *sess) { return sess ? sess->max_ctx : 0U; }
