#include "core_internal.h"
#include "velri_lib/vulkan.h"
#include <stdio.h>
#include <string.h>

/*
 * Thin optional 6-group backend adapter wrapping `VelriLib::vulkan` (SPIR-V 1.6).
 * Compiled only when `-DVELRI_ENABLE_VULKAN=ON`.
 */

struct velri_backend_device {
    velri_vk_ctx *vk;
    atomic_flag queue_lock;
};

struct velri_prepared_op {
    velri_resolved_op op;
};

struct velri_completion {
    uint8_t dummy;
};

static struct velri_completion g_vk_sync_done = {.dummy = 1U};

static void vk_lock(velri_backend_device *dev) {
    while (atomic_flag_test_and_set_explicit(&dev->queue_lock, memory_order_acquire)) {
        /* spin */
    }
}

static void vk_unlock(velri_backend_device *dev) {
    atomic_flag_clear_explicit(&dev->queue_lock, memory_order_release);
}

static velri_vk_buf *binding_vk_buf(const velri_buffer_binding *b, velri_vk_buf_view *storage) {
    if (!b || !b->mem_handle) {
        return nullptr;
    }
    return velri_vk_buf_subview((const velri_vk_buf *)b->mem_handle, b->byte_offset, storage);
}

/* ---- 1. Registration ---- */

static velri_status vk_validate_option(const char *key, const char *value) {
    if (!key || !value) {
        return VELRI_ERR_INVALID;
    }
    return VELRI_ERR_UNSUPPORTED;
}

static velri_status vk_enumerate_devices(uint32_t *out_count) {
    if (!out_count) {
        return VELRI_ERR_INVALID;
    }
    velri_vk_ctx *probe = nullptr;
    if (velri_vk_create(&probe) == VELRI_OK && probe) {
        velri_vk_destroy(probe);
        *out_count = 1U;
    } else {
        *out_count = 0U;
    }
    return VELRI_OK;
}

/* ---- 2. Device ---- */

static velri_status vk_open_device(uint32_t local_index, velri_backend_device **out_dev) {
    if (local_index != 0U || !out_dev) {
        return VELRI_ERR_NOT_FOUND;
    }
    velri_backend_device *dev = (velri_backend_device *)calloc(1U, sizeof(*dev));
    if (!dev) {
        return VELRI_ERR_NOMEM;
    }
    velri_status st = velri_vk_create(&dev->vk);
    if (st != VELRI_OK) {
        free(dev);
        return st;
    }
    atomic_flag_clear(&dev->queue_lock);
    *out_dev = dev;
    return VELRI_OK;
}

static velri_status vk_query_caps(const velri_backend_device *dev, velri_device_caps *out_caps) {
    if (!dev || !out_caps) {
        return VELRI_ERR_INVALID;
    }
    const velri_vk_info *vi = velri_vk_get_info(dev->vk);
    memset(out_caps, 0, sizeof(*out_caps));
    snprintf(out_caps->backend_name, sizeof(out_caps->backend_name), "%s", "vulkan");
    snprintf(out_caps->device_name, sizeof(out_caps->device_name), "%s", vi->device_name);
    out_caps->local_index = 0U;
    out_caps->is_host_memory = false;
    out_caps->is_uma = false;
    out_caps->mem_total_known = false;
    out_caps->min_align_bytes = 256U;
    return VELRI_OK;
}

static void vk_close_device(velri_backend_device *dev) {
    if (!dev) {
        return;
    }
    velri_vk_destroy(dev->vk);
    free(dev);
}

/* ---- 3. Memory ---- */

static velri_status vk_query_mem_req(
    const velri_backend_device *dev,
    size_t logical_bytes,
    size_t align,
    velri_mem_req *out_req) {
    if (!dev || !out_req) {
        return VELRI_ERR_INVALID;
    }
    size_t eff_align = align < 256U ? 256U : align;
    size_t padded = velri_align_up(logical_bytes, eff_align);
    if (padded == SIZE_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    out_req->byte_size = padded;
    out_req->align_bytes = eff_align;
    out_req->alloc_overhead_bytes = padded - logical_bytes;
    return VELRI_OK;
}

static velri_status vk_alloc(
    velri_backend_device *dev,
    size_t bytes,
    size_t align,
    void **out_mem_handle) {
    (void)align;
    if (!dev || !out_mem_handle) {
        return VELRI_ERR_INVALID;
    }
    velri_vk_buf *buf = nullptr;
    vk_lock(dev);
    velri_status st = velri_vk_buf_alloc(dev->vk, bytes, &buf);
    vk_unlock(dev);
    if (st != VELRI_OK) {
        return st;
    }
    *out_mem_handle = buf;
    return VELRI_OK;
}

static void vk_free(velri_backend_device *dev, void *mem_handle) {
    if (!dev || !mem_handle) {
        return;
    }
    vk_lock(dev);
    velri_vk_buf_free(dev->vk, (velri_vk_buf *)mem_handle);
    vk_unlock(dev);
}

static velri_status vk_copy_to_ram(
    velri_backend_device *dev,
    void *dst_ram,
    const velri_buffer_binding *src,
    size_t bytes) {
    if (!dev || !dst_ram || !src || !src->mem_handle) {
        return VELRI_ERR_INVALID;
    }
    vk_lock(dev);
    velri_status st = velri_vk_buf_download_at(dev->vk, dst_ram, (const velri_vk_buf *)src->mem_handle,
                                               src->byte_offset, bytes);
    vk_unlock(dev);
    return st;
}

static velri_status vk_copy_from_ram(
    velri_backend_device *dev,
    const velri_buffer_binding *dst,
    const void *src_ram,
    size_t bytes) {
    if (!dev || !dst || !dst->mem_handle || !src_ram) {
        return VELRI_ERR_INVALID;
    }
    vk_lock(dev);
    velri_status st = velri_vk_buf_upload_at(dev->vk, (velri_vk_buf *)dst->mem_handle,
                                             dst->byte_offset, src_ram, bytes);
    vk_unlock(dev);
    return st;
}

/* ---- 4. Preparation ---- */

static velri_status vk_query_operation(
    const velri_backend_device *dev,
    const velri_resolved_op *op) {
    if (!dev || !op) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = velri_op_validate(op);
    if (st != VELRI_OK) {
        return st;
    }
    const velri_vk_info *vi = velri_vk_get_info(dev->vk);
    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        const velri_linear_desc *d = &op->raw.attr.linear.desc;
        if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_F32 &&
            d->y_dtype == VELRI_DTYPE_F32) {
            return VELRI_OK;
        }
        if (d->x_dtype == VELRI_DTYPE_F16 && d->w_dtype == VELRI_DTYPE_F16 &&
            d->y_dtype == VELRI_DTYPE_F32) {
            return vi->has_f16_i8 ? VELRI_OK : VELRI_ERR_UNSUPPORTED;
        }
        if (d->x_dtype == VELRI_DTYPE_BF16 && d->w_dtype == VELRI_DTYPE_BF16 &&
            d->y_dtype == VELRI_DTYPE_F32) {
            return vi->has_f16_i8 ? VELRI_OK : VELRI_ERR_UNSUPPORTED;
        }
        if (d->x_dtype == VELRI_DTYPE_I8 && d->w_dtype == VELRI_DTYPE_I8 &&
            d->y_dtype == VELRI_DTYPE_F32) {
            return (vi->has_f16_i8 && vi->has_int_dot) ? VELRI_OK : VELRI_ERR_UNSUPPORTED;
        }
        return VELRI_ERR_UNSUPPORTED;
    }
    case VELRI_OP_RMSNORM:
        if (op->inputs[0].dtype == VELRI_DTYPE_F32) return VELRI_OK;
        if (op->inputs[0].dtype == VELRI_DTYPE_BF16 || op->inputs[0].dtype == VELRI_DTYPE_F16) {
            return vi->has_f16_i8 ? VELRI_OK : VELRI_ERR_UNSUPPORTED;
        }
        return VELRI_ERR_UNSUPPORTED;
    case VELRI_OP_ROPE:
        if (op->inputs[0].dtype == VELRI_DTYPE_F32) return VELRI_OK;
        if (op->inputs[0].dtype == VELRI_DTYPE_BF16) {
            return vi->has_f16_i8 ? VELRI_OK : VELRI_ERR_UNSUPPORTED;
        }
        return VELRI_ERR_UNSUPPORTED;
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

static velri_status vk_prepare_op(
    velri_backend_device *dev,
    const velri_resolved_op *op,
    velri_prepared_op **out_prepared) {
    if (!dev || !op || !out_prepared) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = vk_query_operation(dev, op);
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

static void vk_destroy_prepared_op(velri_backend_device *dev, velri_prepared_op *prepared) {
    (void)dev;
    free(prepared);
}

/* ---- 5. Execution ---- */

static velri_status vk_dispatch_op(
    velri_vk_ctx *vk,
    const velri_resolved_op *op,
    const velri_buffer_binding *in_b,
    const velri_buffer_binding *out_b,
    const velri_step_params *params) {
    velri_vk_buf_view iv[VELRI_OP_MAX_INPUTS];
    velri_vk_buf_view ov[VELRI_OP_MAX_OUTPUTS];
    velri_vk_buf *ib[VELRI_OP_MAX_INPUTS] = {0};
    velri_vk_buf *ob[VELRI_OP_MAX_OUTPUTS] = {0};

    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        if (op->input_present[i]) {
            ib[i] = binding_vk_buf(&in_b[i], &iv[i]);
            if (!ib[i]) return VELRI_ERR_INVALID;
        }
    }
    for (uint32_t o = 0U; o < op->num_outputs; ++o) {
        ob[o] = binding_vk_buf(&out_b[o], &ov[o]);
        if (!ob[o]) return VELRI_ERR_INVALID;
    }

    switch (op->kind) {
    case VELRI_OP_LINEAR:
        return velri_vk_linear(
            vk, &op->raw.attr.linear.desc,
            (uint32_t)op->inputs[0].dims[0], (uint32_t)op->inputs[1].dims[0],
            (uint32_t)op->inputs[0].dims[1],
            ib[0], (uint32_t)op->inputs[0].strides[0],
            ib[1], (uint32_t)op->inputs[1].strides[0],
            ib[2], ib[3], ib[4],
            ob[0], (uint32_t)op->outputs[0].strides[0], 1U, nullptr);
    case VELRI_OP_RMSNORM:
        return velri_vk_rmsnorm(
            vk, op->inputs[0].dtype,
            (uint32_t)op->inputs[0].dims[0], (uint32_t)op->inputs[0].dims[1],
            op->raw.attr.rmsnorm.eps, op->raw.attr.rmsnorm.unit_offset,
            ib[0], (uint32_t)op->inputs[0].strides[0], ib[1],
            ob[0], (uint32_t)op->outputs[0].strides[0], 1U, nullptr);
    case VELRI_OP_FUSED_ADD_RMSNORM:
        return velri_vk_fused_add_rmsnorm_f32(
            vk, (uint32_t)op->inputs[0].dims[0], (uint32_t)op->inputs[0].dims[1],
            op->raw.attr.rmsnorm.eps, op->raw.attr.rmsnorm.unit_offset,
            ob[0], (uint32_t)op->outputs[0].strides[0],
            ib[0], (uint32_t)op->inputs[0].strides[0], ib[1],
            ob[1], (uint32_t)op->outputs[1].strides[0], 1U, nullptr);
    case VELRI_OP_LAYERNORM:
        return velri_vk_layernorm_f32(
            vk, (uint32_t)op->inputs[0].dims[0], (uint32_t)op->inputs[0].dims[1],
            op->raw.attr.layernorm.eps,
            ib[0], (uint32_t)op->inputs[0].strides[0], ib[1], ib[2],
            ob[0], (uint32_t)op->outputs[0].strides[0], 1U, nullptr);
    case VELRI_OP_ROPE: {
        uint32_t pos_off = (uint32_t)((params && params->has_pos_offset)
                                          ? params->pos_offset_override
                                          : op->raw.attr.rope.pos_offset);
        return velri_vk_rope(
            vk, op->inputs[0].dtype,
            (uint32_t)op->inputs[0].dims[0], (uint32_t)op->inputs[0].dims[1],
            (uint32_t)op->inputs[0].dims[2], (uint32_t)op->raw.attr.rope.rotary_dim, pos_off,
            ib[0], (uint32_t)op->inputs[0].strides[0], (uint32_t)op->inputs[0].strides[1],
            ib[1], ib[2],
            ob[0], (uint32_t)op->outputs[0].strides[0], (uint32_t)op->outputs[0].strides[1],
            op->raw.attr.rope.mode, 1U, nullptr);
    }
    case VELRI_OP_ACTIVATION: {
        size_t n = 0U;
        (void)velri_tensor_view_num_elements(&op->inputs[0], &n);
        return velri_vk_activation_f32(vk, op->raw.attr.activation.kind, (uint32_t)n,
                                       ib[0], ob[0], 1U, nullptr);
    }
    case VELRI_OP_GATED_MLP: {
        size_t n = 0U;
        (void)velri_tensor_view_num_elements(&op->inputs[0], &n);
        return velri_vk_gated_mlp_f32(vk, op->raw.attr.gated_mlp.act, (uint32_t)n,
                                      op->raw.attr.gated_mlp.clamp_limit,
                                      ib[0], ib[1], ob[0], 1U, nullptr);
    }
    case VELRI_OP_SOFTMAX:
        return velri_vk_softmax_f32(
            vk, (uint32_t)op->inputs[0].dims[0], (uint32_t)op->inputs[0].dims[1],
            ib[0], (uint32_t)op->inputs[0].strides[0],
            ob[0], (uint32_t)op->outputs[0].strides[0], 1U, nullptr);
    case VELRI_OP_VEC_AXPBY: {
        size_t n = 0U;
        (void)velri_tensor_view_num_elements(&op->inputs[0], &n);
        return velri_vk_vec_axpby_f32(vk, (uint32_t)n, op->raw.attr.axpby.alpha, ib[0],
                                      op->raw.attr.axpby.beta, ib[1], ob[0], 1U, nullptr);
    }
    case VELRI_OP_VEC_MUL: {
        size_t n = 0U;
        (void)velri_tensor_view_num_elements(&op->inputs[0], &n);
        return velri_vk_vec_mul_f32(vk, (uint32_t)n, ib[0], ib[1], ob[0], 1U, nullptr);
    }
    case VELRI_OP_FWHT:
        return velri_vk_fwht_f32(
            vk, (uint32_t)op->inputs[0].dims[0], (uint32_t)op->inputs[0].dims[1],
            (uint32_t)op->raw.attr.fwht.block_size,
            ib[0], (uint32_t)op->inputs[0].strides[0],
            op->raw.attr.fwht.suh_dtype, ib[1],
            ob[0], (uint32_t)op->outputs[0].strides[0], 1U, nullptr);
    case VELRI_OP_QUANTIZE_ROW:
        return velri_vk_quantize_row_f32_i8(
            vk, (uint32_t)op->inputs[0].dims[0], (uint32_t)op->inputs[0].dims[1],
            ib[0], (uint32_t)op->inputs[0].strides[0],
            ob[0], (uint32_t)op->outputs[0].strides[0], ob[1], 1U, nullptr);
    default:
        return VELRI_ERR_UNSUPPORTED;
    }
}

static velri_status vk_submit(
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
    vk_lock(dev);
    velri_status st = vk_dispatch_op(dev->vk, &prepared->op, inputs, outputs, params);
    vk_unlock(dev);
    if (st != VELRI_OK) {
        return st;
    }
    *out_completion = &g_vk_sync_done;
    return VELRI_OK;
}

/* ---- 6. Completion ---- */

static velri_poll_state vk_poll(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    return (comp == &g_vk_sync_done) ? VELRI_POLL_DONE : VELRI_POLL_FAILED;
}

static velri_status vk_wait(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    return (comp == &g_vk_sync_done) ? VELRI_OK : VELRI_ERR_INVALID;
}

static void vk_destroy_completion(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    (void)comp;
}

static const velri_backend_vtable g_vk_vtable = {
    .name = "vulkan",
    .contract_version = VELRI_BACKEND_CONTRACT_VERSION,
    .validate_option = vk_validate_option,
    .enumerate_devices = vk_enumerate_devices,
    .open_device = vk_open_device,
    .query_caps = vk_query_caps,
    .close_device = vk_close_device,
    .query_mem_req = vk_query_mem_req,
    .alloc = vk_alloc,
    .free = vk_free,
    .copy_to_ram = vk_copy_to_ram,
    .copy_from_ram = vk_copy_from_ram,
    .query_operation = vk_query_operation,
    .prepare_op = vk_prepare_op,
    .destroy_prepared_op = vk_destroy_prepared_op,
    .submit = vk_submit,
    .poll = vk_poll,
    .wait = vk_wait,
    .destroy_completion = vk_destroy_completion,
};

const velri_backend_vtable *velri_vulkan_backend_vtable(void) { return &g_vk_vtable; }
