#include "mock_async_backend.h"
#include "os.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static velri_mock_async_ctrl g_ctrl;

velri_mock_async_ctrl *velri_mock_async_control(void) { return &g_ctrl; }

void velri_mock_async_reset_ctrl(void) {
    atomic_store(&g_ctrl.delay_ms, 0U);
    atomic_store(&g_ctrl.fail_pre_accept, false);
    atomic_store(&g_ctrl.fail_post_accept, false);
    atomic_store(&g_ctrl.fail_partial_submit, false);
    atomic_store(&g_ctrl.in_flight, 0);
    atomic_store(&g_ctrl.total_submitted, 0U);
    atomic_store(&g_ctrl.total_completed, 0U);
    atomic_store(&g_ctrl.total_uploads, 0U);
    atomic_store(&g_ctrl.total_downloads, 0U);
}

typedef struct {
    uint8_t *bytes;
    size_t size;
} mock_buf;

struct velri_backend_device {
    uint32_t index;
};

struct velri_prepared_op {
    velri_resolved_op op;
};

struct velri_completion {
    velri_os_thread *thread;
    atomic_int state; /* VELRI_POLL_PENDING / DONE / FAILED */
    velri_status final_status;
    size_t count;
    float alpha;
    float beta;
    const float *a;
    const float *b;
    float *y;
    uint32_t delay_ms;
    bool inject_fail;
};

static void run_mock_job(void *raw) {
    struct velri_completion *c = (struct velri_completion *)raw;
    if (c->delay_ms > 0U) {
        velri_os_sleep_ms(c->delay_ms);
    }
    if (c->inject_fail) {
        /* Touch output memory while in flight to prove runtime awaits completion before free. */
        if (c->y && c->count > 0U) {
            c->y[0] = -999.0f;
        }
        c->final_status = VELRI_ERR_EXEC;
        atomic_fetch_add(&g_ctrl.total_completed, 1U);
        atomic_fetch_sub(&g_ctrl.in_flight, 1);
        atomic_store_explicit(&c->state, (int)VELRI_POLL_FAILED, memory_order_release);
        return;
    }

    for (size_t i = 0U; i < c->count; ++i) {
        float vb = c->b ? c->b[i] : 0.0f;
        c->y[i] = c->alpha * c->a[i] + c->beta * vb;
    }
    c->final_status = VELRI_OK;
    atomic_fetch_add(&g_ctrl.total_completed, 1U);
    atomic_fetch_sub(&g_ctrl.in_flight, 1);
    atomic_store_explicit(&c->state, (int)VELRI_POLL_DONE, memory_order_release);
}

/* ---- 1. Registration ---- */

static velri_status mock_validate_option(const char *key, const char *value) {
    (void)key;
    (void)value;
    return VELRI_ERR_UNSUPPORTED;
}

static velri_status mock_enumerate_devices(uint32_t *out_count) {
    if (!out_count) return VELRI_ERR_INVALID;
    *out_count = 1U;
    return VELRI_OK;
}

/* ---- 2. Device ---- */

static velri_status mock_open_device(uint32_t local_index, velri_backend_device **out_dev) {
    if (local_index != 0U || !out_dev) return VELRI_ERR_NOT_FOUND;
    velri_backend_device *dev = (velri_backend_device *)calloc(1U, sizeof(*dev));
    if (!dev) return VELRI_ERR_NOMEM;
    dev->index = local_index;
    *out_dev = dev;
    return VELRI_OK;
}

static velri_status mock_query_caps(const velri_backend_device *dev, velri_device_caps *out_caps) {
    if (!dev || !out_caps) return VELRI_ERR_INVALID;
    memset(out_caps, 0, sizeof(*out_caps));
    snprintf(out_caps->backend_name, sizeof(out_caps->backend_name), "%s", "mock_async");
    snprintf(out_caps->device_name, sizeof(out_caps->device_name), "%s", "Mock Async Device");
    out_caps->local_index = dev->index;
    out_caps->is_host_memory = false; /* Forces explicit RAM staging transfers! */
    out_caps->is_uma = false;
    out_caps->mem_total_known = true;
    out_caps->mem_total_bytes = 64ULL * 1024ULL * 1024ULL;
    out_caps->min_align_bytes = 64U;
    return VELRI_OK;
}

static void mock_close_device(velri_backend_device *dev) { free(dev); }

/* ---- 3. Memory ---- */

static velri_status mock_query_mem_req(
    const velri_backend_device *dev,
    size_t logical_bytes,
    size_t align,
    velri_mem_req *out_req) {
    if (!dev || !out_req) return VELRI_ERR_INVALID;
    size_t eff = align < 64U ? 64U : align;
    size_t padded = (logical_bytes + eff - 1U) & ~(eff - 1U);
    out_req->byte_size = padded;
    out_req->align_bytes = eff;
    out_req->alloc_overhead_bytes = padded - logical_bytes;
    return VELRI_OK;
}

static velri_status mock_alloc(
    velri_backend_device *dev,
    size_t bytes,
    size_t align,
    void **out_mem_handle) {
    (void)dev;
    (void)align;
    if (!out_mem_handle) return VELRI_ERR_INVALID;
    mock_buf *mb = (mock_buf *)calloc(1U, sizeof(*mb));
    if (!mb) return VELRI_ERR_NOMEM;
    size_t sz = bytes ? ((bytes + 63ULL) & ~63ULL) : 64ULL;
    mb->bytes = (uint8_t *)velri_os_aligned_alloc(64U, sz);
    if (!mb->bytes) {
        free(mb);
        return VELRI_ERR_NOMEM;
    }
    memset(mb->bytes, 0, sz);
    mb->size = sz;
    *out_mem_handle = mb;
    return VELRI_OK;
}

static void mock_free(velri_backend_device *dev, void *mem_handle) {
    (void)dev;
    mock_buf *mb = (mock_buf *)mem_handle;
    if (!mb) return;
    velri_os_aligned_free(mb->bytes);
    free(mb);
}

static velri_status mock_copy_to_ram(
    velri_backend_device *dev,
    void *dst_ram,
    const velri_buffer_binding *src,
    size_t bytes) {
    (void)dev;
    if (!dst_ram || !src || !src->mem_handle) return VELRI_ERR_INVALID;
    const mock_buf *mb = (const mock_buf *)src->mem_handle;
    if (src->byte_offset > mb->size || bytes > mb->size - src->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes > 0U) {
        memcpy(dst_ram, mb->bytes + src->byte_offset, bytes);
    }
    atomic_fetch_add(&g_ctrl.total_downloads, 1U);
    return VELRI_OK;
}

static velri_status mock_copy_from_ram(
    velri_backend_device *dev,
    const velri_buffer_binding *dst,
    const void *src_ram,
    size_t bytes) {
    (void)dev;
    if (!dst || !dst->mem_handle || !src_ram) return VELRI_ERR_INVALID;
    mock_buf *mb = (mock_buf *)dst->mem_handle;
    if (dst->byte_offset > mb->size || bytes > mb->size - dst->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes > 0U) {
        memcpy(mb->bytes + dst->byte_offset, src_ram, bytes);
    }
    atomic_fetch_add(&g_ctrl.total_uploads, 1U);
    return VELRI_OK;
}

/* ---- 4. Preparation ---- */

static velri_status mock_query_operation(
    const velri_backend_device *dev,
    const velri_resolved_op *op) {
    if (!dev || !op) return VELRI_ERR_INVALID;
    /* Intentionally supports ONLY VEC_AXPBY to test capability rejection & hybrid routing. */
    if (op->kind != VELRI_OP_VEC_AXPBY) {
        return VELRI_ERR_UNSUPPORTED;
    }
    return VELRI_OK;
}

static velri_status mock_prepare_op(
    velri_backend_device *dev,
    const velri_resolved_op *op,
    velri_prepared_op **out_prepared) {
    velri_status st = mock_query_operation(dev, op);
    if (st != VELRI_OK) return st;
    velri_prepared_op *p = (velri_prepared_op *)calloc(1U, sizeof(*p));
    if (!p) return VELRI_ERR_NOMEM;
    p->op = *op;
    *out_prepared = p;
    return VELRI_OK;
}

static void mock_destroy_prepared_op(velri_backend_device *dev, velri_prepared_op *prepared) {
    (void)dev;
    free(prepared);
}

/* ---- 5. Execution ---- */

static velri_status mock_submit(
    velri_backend_device *dev,
    const velri_prepared_op *prepared,
    const velri_buffer_binding *inputs,
    uint32_t num_inputs,
    const velri_buffer_binding *outputs,
    uint32_t num_outputs,
    const velri_step_params *params,
    velri_completion **out_completion) {
    (void)params;
    if (!out_completion) return VELRI_ERR_INVALID;
    *out_completion = nullptr;
    if (!dev || !prepared || num_inputs < 1U || num_outputs != 1U) {
        return VELRI_ERR_INVALID;
    }

    /* Pre-accept rejection: zero work started, completion remains nullptr. */
    if (atomic_load(&g_ctrl.fail_pre_accept)) {
        return VELRI_ERR_EXEC;
    }

    struct velri_completion *c = (struct velri_completion *)calloc(1U, sizeof(*c));
    if (!c) return VELRI_ERR_NOMEM;

    const mock_buf *ba = (const mock_buf *)inputs[0].mem_handle;
    const mock_buf *bb = (num_inputs > 1U && prepared->op.input_present[1])
                             ? (const mock_buf *)inputs[1].mem_handle
                             : nullptr;
    mock_buf *by = (mock_buf *)outputs[0].mem_handle;

    size_t count = 0U;
    (void)velri_tensor_view_num_elements(&prepared->op.inputs[0], &count);
    c->count = count;
    c->alpha = prepared->op.raw.attr.axpby.alpha;
    c->beta = prepared->op.raw.attr.axpby.beta;
    c->a = (const float *)(ba->bytes + inputs[0].byte_offset);
    c->b = bb ? (const float *)(bb->bytes + inputs[1].byte_offset) : nullptr;
    c->y = (float *)(by->bytes + outputs[0].byte_offset);
    c->delay_ms = atomic_load(&g_ctrl.delay_ms);
    c->inject_fail = atomic_load(&g_ctrl.fail_post_accept);
    atomic_init(&c->state, (int)VELRI_POLL_PENDING);

    atomic_fetch_add(&g_ctrl.total_submitted, 1U);
    atomic_fetch_add(&g_ctrl.in_flight, 1);

    if (!velri_os_thread_spawn(&c->thread, run_mock_job, c)) {
        atomic_fetch_sub(&g_ctrl.in_flight, 1);
        free(c);
        return VELRI_ERR_NOMEM;
    }
    *out_completion = c;

    /* Partial-accept error injection: work IS in flight AND submit returns error! */
    if (atomic_load(&g_ctrl.fail_partial_submit)) {
        return VELRI_ERR_EXEC;
    }
    return VELRI_OK;
}

/* ---- 6. Completion ---- */

static velri_poll_state mock_poll(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    if (!comp) return VELRI_POLL_FAILED;
    return (velri_poll_state)atomic_load_explicit(&comp->state, memory_order_acquire);
}

static velri_status mock_wait(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    if (!comp) return VELRI_ERR_INVALID;
    if (comp->thread) {
        velri_os_thread_join(comp->thread);
        comp->thread = nullptr;
    }
    return comp->final_status;
}

static void mock_destroy_completion(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    free(comp);
}

static const velri_backend_vtable g_mock_vtable = {
    .name = "mock_async",
    .contract_version = VELRI_BACKEND_CONTRACT_VERSION,
    .validate_option = mock_validate_option,
    .enumerate_devices = mock_enumerate_devices,
    .open_device = mock_open_device,
    .query_caps = mock_query_caps,
    .close_device = mock_close_device,
    .query_mem_req = mock_query_mem_req,
    .alloc = mock_alloc,
    .free = mock_free,
    .copy_to_ram = mock_copy_to_ram,
    .copy_from_ram = mock_copy_from_ram,
    .query_operation = mock_query_operation,
    .prepare_op = mock_prepare_op,
    .destroy_prepared_op = mock_destroy_prepared_op,
    .submit = mock_submit,
    .poll = mock_poll,
    .wait = mock_wait,
    .destroy_completion = mock_destroy_completion,
};

const velri_backend_vtable *velri_mock_async_backend_vtable(void) { return &g_mock_vtable; }
