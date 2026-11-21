#include "core_internal.h"
#include "velri_lib/slang_cpu.h"
#include <stdio.h>
#include <string.h>

/*
 * Thin 6-group CPU backend adapter wrapping Slang AOT `VelriLib::cpu` and `velri_pool`.
 *
 * Concurrency note:
 *   Multiple slots may invoke `submit` concurrently on different threads.
 *   We guard shared `velri_pool` execution with a lightweight atomic lock ONLY when
 *   dispatching multi-threaded pool work; if another slot already holds the pool,
 *   the second caller executes directly on its own thread (active_workers=1) so two
 *   concurrent slots never block or corrupt each other's worker pool state.
 */

struct velri_backend_device {
    velri_pool *pool;
    size_t workers;
    velri_cpu_variant variant;
    atomic_flag pool_busy;
};

struct velri_prepared_op {
    velri_resolved_op op;
};

struct velri_completion {
    uint8_t dummy;
};

/* Static singleton completion sentinel => 0 B heap allocations inside `submit`. */
static struct velri_completion g_cpu_sync_done = {.dummy = 1U};

static size_t g_default_cpu_workers = 0U;
static velri_cpu_variant g_default_cpu_variant = VELRI_CPU_AUTO;

void velri_cpu_backend_set_defaults(size_t workers, velri_cpu_variant variant) {
    g_default_cpu_workers = workers;
    g_default_cpu_variant = variant;
}

static const void *binding_cptr(const velri_buffer_binding *b) {
    if (!b || !b->mem_handle) {
        return nullptr;
    }
    return (const uint8_t *)b->mem_handle + b->byte_offset;
}

static void *binding_ptr(const velri_buffer_binding *b) {
    if (!b || !b->mem_handle) {
        return nullptr;
    }
    return (uint8_t *)b->mem_handle + b->byte_offset;
}

/* ---- 1. Registration ---- */

static velri_status cpu_validate_option(const char *key, const char *value) {
    if (!key || !value) {
        return VELRI_ERR_INVALID;
    }
    if (strcmp(key, "workers") == 0 || strcmp(key, "variant") == 0) {
        return VELRI_OK;
    }
    return VELRI_ERR_UNSUPPORTED;
}

static velri_status cpu_enumerate_devices(uint32_t *out_count) {
    if (!out_count) {
        return VELRI_ERR_INVALID;
    }
    *out_count = 1U;
    return VELRI_OK;
}

/* ---- 2. Device ---- */

static velri_status cpu_open_device(uint32_t local_index, velri_backend_device **out_dev) {
    if (local_index != 0U || !out_dev) {
        return VELRI_ERR_NOT_FOUND;
    }
    velri_backend_device *dev = (velri_backend_device *)calloc(1U, sizeof(*dev));
    if (!dev) {
        return VELRI_ERR_NOMEM;
    }

    const velri_cpu_info *ci = velri_get_cpu_info();
    size_t nw = g_default_cpu_workers;
    if (nw == 0U) {
        int hw = ci->hardware_threads > 0 ? ci->hardware_threads : 1;
        nw = (size_t)(hw > 16 ? hw / 2 : hw);
        if (nw == 0U) {
            nw = 1U;
        }
    }
    dev->workers = nw;
    dev->variant = velri_cpu_resolve_variant(g_default_cpu_variant);
    atomic_flag_clear(&dev->pool_busy);

    if (nw > 1U) {
        velri_status st = velri_pool_create(nw, 0U, &dev->pool);
        if (st != VELRI_OK) {
            free(dev);
            return st;
        }
    }
    *out_dev = dev;
    return VELRI_OK;
}

static velri_status cpu_query_caps(const velri_backend_device *dev, velri_device_caps *out_caps) {
    if (!dev || !out_caps) {
        return VELRI_ERR_INVALID;
    }
    memset(out_caps, 0, sizeof(*out_caps));
    snprintf(out_caps->backend_name, sizeof(out_caps->backend_name), "%s", "cpu");
    snprintf(out_caps->device_name, sizeof(out_caps->device_name),
             "Slang AOT CPU (workers=%zu)", dev->workers);
    out_caps->local_index = 0U;
    out_caps->is_host_memory = true;
    out_caps->is_uma = true;
    out_caps->mem_total_known = false;
    out_caps->mem_total_bytes = 0U;
    out_caps->min_align_bytes = 64U;
    return VELRI_OK;
}

static void cpu_close_device(velri_backend_device *dev) {
    if (!dev) {
        return;
    }
    velri_pool_destroy(dev->pool);
    free(dev);
}

/* ---- 3. Memory ---- */

static velri_status cpu_query_mem_req(
    const velri_backend_device *dev,
    size_t logical_bytes,
    size_t align,
    velri_mem_req *out_req) {
    if (!dev || !out_req) {
        return VELRI_ERR_INVALID;
    }
    size_t eff_align = align < 64U ? 64U : align;
    size_t padded = velri_align_up(logical_bytes, eff_align);
    if (padded == SIZE_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    out_req->byte_size = padded;
    out_req->align_bytes = eff_align;
    out_req->alloc_overhead_bytes = padded - logical_bytes;
    return VELRI_OK;
}

static velri_status cpu_alloc(
    velri_backend_device *dev,
    size_t bytes,
    size_t align,
    void **out_mem_handle) {
    (void)dev;
    (void)align;
    if (!out_mem_handle) {
        return VELRI_ERR_INVALID;
    }
    void *p = velri_core_aligned_alloc64(bytes);
    if (!p) {
        return VELRI_ERR_NOMEM;
    }
    memset(p, 0, bytes ? ((bytes + 63ULL) & ~63ULL) : 64ULL);
    *out_mem_handle = p;
    return VELRI_OK;
}

static void cpu_free(velri_backend_device *dev, void *mem_handle) {
    (void)dev;
    velri_core_aligned_free64(mem_handle);
}

static velri_status cpu_copy_to_ram(
    velri_backend_device *dev,
    void *dst_ram,
    const velri_buffer_binding *src,
    size_t bytes) {
    (void)dev;
    if (!dst_ram || !src || !src->mem_handle) {
        return VELRI_ERR_INVALID;
    }
    if (bytes > 0U) {
        memcpy(dst_ram, binding_cptr(src), bytes);
    }
    return VELRI_OK;
}

static velri_status cpu_copy_from_ram(
    velri_backend_device *dev,
    const velri_buffer_binding *dst,
    const void *src_ram,
    size_t bytes) {
    (void)dev;
    if (!dst || !dst->mem_handle || !src_ram) {
        return VELRI_ERR_INVALID;
    }
    if (bytes > 0U) {
        memcpy(binding_ptr(dst), src_ram, bytes);
    }
    return VELRI_OK;
}

/* ---- 4. Preparation ---- */

static velri_status cpu_query_operation(
    const velri_backend_device *dev,
    const velri_resolved_op *op) {
    if (!dev || !op) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = velri_op_validate(op);
    if (st != VELRI_OK) {
        return st;
    }
    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        const velri_linear_desc *d = &op->raw.attr.linear.desc;
        bool ok =
            (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_F32 && d->y_dtype == VELRI_DTYPE_F32) ||
            (d->x_dtype == VELRI_DTYPE_F16 && d->w_dtype == VELRI_DTYPE_F16 && d->y_dtype == VELRI_DTYPE_F32) ||
            (d->x_dtype == VELRI_DTYPE_BF16 && d->w_dtype == VELRI_DTYPE_BF16 && d->y_dtype == VELRI_DTYPE_F32) ||
            (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_BF16 && d->y_dtype == VELRI_DTYPE_F32) ||
            (d->x_dtype == VELRI_DTYPE_F64 && d->w_dtype == VELRI_DTYPE_F64 && d->y_dtype == VELRI_DTYPE_F64) ||
            (d->x_dtype == VELRI_DTYPE_I8 && d->w_dtype == VELRI_DTYPE_I8 && d->y_dtype == VELRI_DTYPE_F32);
        return ok ? VELRI_OK : VELRI_ERR_UNSUPPORTED;
    }
    case VELRI_OP_RMSNORM: {
        velri_dtype dt = op->inputs[0].dtype;
        return (dt == VELRI_DTYPE_F32 || dt == VELRI_DTYPE_BF16 || dt == VELRI_DTYPE_F16)
                   ? VELRI_OK
                   : VELRI_ERR_UNSUPPORTED;
    }
    case VELRI_OP_ROPE: {
        velri_dtype dt = op->inputs[0].dtype;
        return (dt == VELRI_DTYPE_F32 || dt == VELRI_DTYPE_BF16) ? VELRI_OK : VELRI_ERR_UNSUPPORTED;
    }
    case VELRI_OP_FUSED_ADD_RMSNORM:
    case VELRI_OP_LAYERNORM:
    case VELRI_OP_ACTIVATION:
    case VELRI_OP_GATED_MLP:
    case VELRI_OP_SOFTMAX:
    case VELRI_OP_VEC_AXPBY:
    case VELRI_OP_VEC_MUL:
    case VELRI_OP_FWHT:
    case VELRI_OP_QUANTIZE_ROW:
        return VELRI_OK;
    default:
        return VELRI_ERR_UNSUPPORTED;
    }
}

static velri_status cpu_prepare_op(
    velri_backend_device *dev,
    const velri_resolved_op *op,
    velri_prepared_op **out_prepared) {
    if (!dev || !op || !out_prepared) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = cpu_query_operation(dev, op);
    if (st != VELRI_OK) {
        return st;
    }
    velri_prepared_op *p = (velri_prepared_op *)calloc(1U, sizeof(*p));
    if (!p) {
        return VELRI_ERR_NOMEM;
    }
    p->op = *op;
    *out_prepared = p;
    return VELRI_OK;
}

static void cpu_destroy_prepared_op(velri_backend_device *dev, velri_prepared_op *prepared) {
    (void)dev;
    free(prepared);
}

/* ---- 5. Execution ---- */

static velri_status cpu_dispatch_op(
    const velri_resolved_op *op,
    const velri_buffer_binding *in_b,
    const velri_buffer_binding *out_b,
    const velri_step_params *params,
    velri_cpu_variant var,
    velri_pool *pool,
    size_t nw) {
    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        const void *bias = op->input_present[2] ? binding_cptr(&in_b[2]) : nullptr;
        const float *xs = op->input_present[3] ? (const float *)binding_cptr(&in_b[3]) : nullptr;
        const float *ws = op->input_present[4] ? (const float *)binding_cptr(&in_b[4]) : nullptr;
        return velri_slang_cpu_linear(
            &op->raw.attr.linear.desc,
            op->inputs[0].dims[0], op->inputs[1].dims[0], op->inputs[0].dims[1],
            binding_cptr(&in_b[0]), op->inputs[0].strides[0],
            binding_cptr(&in_b[1]), op->inputs[1].strides[0],
            bias, xs, ws, binding_ptr(&out_b[0]), op->outputs[0].strides[0],
            var, pool, nw);
    }
    case VELRI_OP_RMSNORM: {
        const void *w = op->input_present[1] ? binding_cptr(&in_b[1]) : nullptr;
        return velri_slang_cpu_rmsnorm(
            op->inputs[0].dtype, op->inputs[0].dims[0], op->inputs[0].dims[1],
            op->raw.attr.rmsnorm.eps, op->raw.attr.rmsnorm.unit_offset,
            binding_cptr(&in_b[0]), op->inputs[0].strides[0], w,
            binding_ptr(&out_b[0]), op->outputs[0].strides[0], var, pool, nw);
    }
    case VELRI_OP_FUSED_ADD_RMSNORM: {
        const float *w = op->input_present[1] ? (const float *)binding_cptr(&in_b[1]) : nullptr;
        return velri_slang_cpu_fused_add_rmsnorm_f32(
            op->inputs[0].dims[0], op->inputs[0].dims[1],
            op->raw.attr.rmsnorm.eps, op->raw.attr.rmsnorm.unit_offset,
            (float *)binding_ptr(&out_b[0]), op->outputs[0].strides[0],
            (const float *)binding_cptr(&in_b[0]), op->inputs[0].strides[0],
            w, (float *)binding_ptr(&out_b[1]), op->outputs[1].strides[0], var, pool, nw);
    }
    case VELRI_OP_LAYERNORM: {
        const float *w = op->input_present[1] ? (const float *)binding_cptr(&in_b[1]) : nullptr;
        const float *b = op->input_present[2] ? (const float *)binding_cptr(&in_b[2]) : nullptr;
        return velri_slang_cpu_layernorm_f32(
            op->inputs[0].dims[0], op->inputs[0].dims[1], op->raw.attr.layernorm.eps,
            (const float *)binding_cptr(&in_b[0]), op->inputs[0].strides[0], w, b,
            (float *)binding_ptr(&out_b[0]), op->outputs[0].strides[0], var, pool, nw);
    }
    case VELRI_OP_ROPE: {
        size_t pos_off = (params && params->has_pos_offset) ? params->pos_offset_override
                                                            : op->raw.attr.rope.pos_offset;
        return velri_slang_cpu_rope(
            op->inputs[0].dtype, op->inputs[0].dims[0], op->inputs[0].dims[1],
            op->inputs[0].dims[2], op->raw.attr.rope.rotary_dim, pos_off,
            binding_cptr(&in_b[0]), op->inputs[0].strides[0], op->inputs[0].strides[1],
            (const float *)binding_cptr(&in_b[1]), (const float *)binding_cptr(&in_b[2]),
            binding_ptr(&out_b[0]), op->outputs[0].strides[0], op->outputs[0].strides[1],
            op->raw.attr.rope.mode, var, pool, nw);
    }
    case VELRI_OP_ACTIVATION: {
        size_t n = 0U;
        (void)velri_tensor_view_num_elements(&op->inputs[0], &n);
        return velri_slang_cpu_activation_f32(
            op->raw.attr.activation.kind, n,
            (const float *)binding_cptr(&in_b[0]), (float *)binding_ptr(&out_b[0]),
            var, pool, nw);
    }
    case VELRI_OP_GATED_MLP: {
        size_t n = 0U;
        (void)velri_tensor_view_num_elements(&op->inputs[0], &n);
        return velri_slang_cpu_gated_mlp_f32(
            op->raw.attr.gated_mlp.act, n, op->raw.attr.gated_mlp.clamp_limit,
            (const float *)binding_cptr(&in_b[0]), (const float *)binding_cptr(&in_b[1]),
            (float *)binding_ptr(&out_b[0]), var, pool, nw);
    }
    case VELRI_OP_SOFTMAX:
        return velri_slang_cpu_softmax_f32(
            op->inputs[0].dims[0], op->inputs[0].dims[1],
            (const float *)binding_cptr(&in_b[0]), op->inputs[0].strides[0],
            (float *)binding_ptr(&out_b[0]), op->outputs[0].strides[0], var, pool, nw);
    case VELRI_OP_VEC_AXPBY: {
        size_t n = 0U;
        (void)velri_tensor_view_num_elements(&op->inputs[0], &n);
        const float *b = op->input_present[1] ? (const float *)binding_cptr(&in_b[1]) : nullptr;
        return velri_slang_cpu_vec_axpby_f32(
            n, op->raw.attr.axpby.alpha, (const float *)binding_cptr(&in_b[0]),
            op->raw.attr.axpby.beta, b, (float *)binding_ptr(&out_b[0]), var, pool, nw);
    }
    case VELRI_OP_VEC_MUL: {
        size_t n = 0U;
        (void)velri_tensor_view_num_elements(&op->inputs[0], &n);
        return velri_slang_cpu_vec_mul_f32(
            n, (const float *)binding_cptr(&in_b[0]), (const float *)binding_cptr(&in_b[1]),
            (float *)binding_ptr(&out_b[0]), var, pool, nw);
    }
    case VELRI_OP_FWHT: {
        const void *suh = op->input_present[1] ? binding_cptr(&in_b[1]) : nullptr;
        return velri_slang_cpu_fwht_f32(
            op->inputs[0].dims[0], op->inputs[0].dims[1], op->raw.attr.fwht.block_size,
            (const float *)binding_cptr(&in_b[0]), op->inputs[0].strides[0],
            op->raw.attr.fwht.suh_dtype, suh,
            (float *)binding_ptr(&out_b[0]), op->outputs[0].strides[0], var, pool, nw);
    }
    case VELRI_OP_QUANTIZE_ROW:
        return velri_slang_cpu_quantize_row_f32_i8(
            op->inputs[0].dims[0], op->inputs[0].dims[1],
            (const float *)binding_cptr(&in_b[0]), op->inputs[0].strides[0],
            (int8_t *)binding_ptr(&out_b[0]), op->outputs[0].strides[0],
            (float *)binding_ptr(&out_b[1]), var, pool, nw);
    default:
        return VELRI_ERR_UNSUPPORTED;
    }
}

static velri_status cpu_submit(
    velri_backend_device *dev,
    const velri_prepared_op *prepared,
    const velri_buffer_binding *inputs,
    uint32_t num_inputs,
    const velri_buffer_binding *outputs,
    uint32_t num_outputs,
    const velri_step_params *params,
    velri_completion **out_completion) {
    if (!out_completion) {
        return VELRI_ERR_INVALID;
    }
    *out_completion = nullptr;
    if (!dev || !prepared || num_inputs != prepared->op.num_inputs ||
        num_outputs != prepared->op.num_outputs) {
        return VELRI_ERR_INVALID;
    }

    /* Acquire shared pool non-blockingly; concurrent slots execute cleanly on their own thread. */
    bool owns_pool = false;
    velri_pool *use_pool = nullptr;
    size_t use_nw = 1U;
    if (dev->pool && !atomic_flag_test_and_set_explicit(&dev->pool_busy, memory_order_acquire)) {
        owns_pool = true;
        use_pool = dev->pool;
        use_nw = dev->workers;
    }

    velri_status st = cpu_dispatch_op(&prepared->op, inputs, outputs, params, dev->variant,
                                      use_pool, use_nw);

    if (owns_pool) {
        atomic_flag_clear_explicit(&dev->pool_busy, memory_order_release);
    }
    if (st != VELRI_OK) {
        return st;
    }
    *out_completion = &g_cpu_sync_done;
    return VELRI_OK;
}

/* ---- 6. Completion ---- */

static velri_poll_state cpu_poll(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    return (comp == &g_cpu_sync_done) ? VELRI_POLL_DONE : VELRI_POLL_FAILED;
}

static velri_status cpu_wait(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    return (comp == &g_cpu_sync_done) ? VELRI_OK : VELRI_ERR_INVALID;
}

static void cpu_destroy_completion(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    (void)comp;
}

static const velri_backend_vtable g_cpu_vtable = {
    .name = "cpu",
    .contract_version = VELRI_BACKEND_CONTRACT_VERSION,
    .validate_option = cpu_validate_option,
    .enumerate_devices = cpu_enumerate_devices,
    .open_device = cpu_open_device,
    .query_caps = cpu_query_caps,
    .close_device = cpu_close_device,
    .query_mem_req = cpu_query_mem_req,
    .alloc = cpu_alloc,
    .free = cpu_free,
    .copy_to_ram = cpu_copy_to_ram,
    .copy_from_ram = cpu_copy_from_ram,
    .query_operation = cpu_query_operation,
    .prepare_op = cpu_prepare_op,
    .destroy_prepared_op = cpu_destroy_prepared_op,
    .submit = cpu_submit,
    .poll = cpu_poll,
    .wait = cpu_wait,
    .destroy_completion = cpu_destroy_completion,
};

const velri_backend_vtable *velri_cpu_backend_vtable(void) { return &g_cpu_vtable; }
