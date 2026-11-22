#include "runtime/runtime_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ---- Status Translation ---- */

velri_rc velri_rt_rc_from_vl(velri_status st) {
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

/* ---- Thread-Local + Context-Backed Diagnostics ---- */

typedef struct {
    const velri_ctx *owner;
    char msg[256];
} tls_err_slot;

static thread_local tls_err_slot g_tls_err = {.owner = nullptr, .msg = {0}};

static inline void err_lock_acquire(velri_ctx *ctx) {
    while (atomic_flag_test_and_set_explicit(&ctx->err_lock, memory_order_acquire)) {
        /* Held only for <= 256-byte memcpy. */
    }
}

static inline void err_lock_release(velri_ctx *ctx) {
    atomic_flag_clear_explicit(&ctx->err_lock, memory_order_release);
}

void velri_rt_ctx_set_error(velri_ctx *ctx, const char *fmt, ...) {
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

void velri_rt_ctx_clear_tls_error(const velri_ctx *ctx) {
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

void velri_cancel_destroy(velri_cancel *cancel) {
    free(cancel);
}

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
        return velri_rt_rc_from_vl(st);
    }
    *out_ctx = ctx;
    return VELRI_RC_OK;
}

velri_rc velri_ctx_destroy(velri_ctx *ctx) {
    if (!ctx) {
        return VELRI_RC_OK;
    }
    if (!velri_rt_try_mark_destroying(&ctx->live_children)) {
        uint32_t lc = atomic_load_explicit(&ctx->live_children, memory_order_acquire) &
                      ~VELRI_DESTROYING_BIT;
        velri_rt_ctx_set_error(ctx,
                               "ctx_destroy: cannot destroy context with %u live dependent object(s)",
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
        /* Diagnostic cache is logically mutable on query errors. */
        velri_rt_ctx_set_error((velri_ctx *)ctx, "ctx_device_info: device %u not found", device_id);
        return VELRI_RC_ERR_NOT_FOUND;
    }
    velri_device_caps caps;
    velri_status st = velri_context_device_caps(ctx->vl_ctx, device_id, &caps);
    if (st != VELRI_OK) {
        velri_rt_ctx_set_error((velri_ctx *)ctx, "ctx_device_info: failed querying device %u",
                               device_id);
        return velri_rt_rc_from_vl(st);
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
    static thread_local char fallback_buf[256];
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
