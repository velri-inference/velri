/* Benchmark-only observers. The timed executable links the ordinary backend directly. */
#include "session_trace.h"
#include "velri/velri.h"
#include "os.h"
#include "runtime/sampler.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool trace_active;
static unsigned run, position;
static FILE *trace_file;

void trace_fail(const char *message) {
    fprintf(stderr, "session profiler: %s\n", message);
    exit(EXIT_FAILURE);
}

static int shape(char *dst, size_t size, const velri_resolved_op *op) {
    const velri_tensor_view *x = &op->inputs[0];
    size_t d = x->dims[x->rank - 1U];
    switch (op->kind) {
    case VELRI_OP_LINEAR:
        return snprintf(dst, size, "%s:%zux%zu", op->inputs[1].dtype == VELRI_DTYPE_I8
                        ? "W8A8" : "linear", op->inputs[1].dims[0], op->inputs[1].dims[1]);
    case VELRI_OP_ROPE:
        return snprintf(dst, size, "rope:%zux%zux%zu", x->dims[1], d, op->raw.attr.rope.rotary_dim);
    case VELRI_OP_ATTENTION: {
        const uint32_t dk = op->raw.attr.attention.head_dim, dv = op->raw.attr.attention.v_head_dim;
        return snprintf(dst, size, "attention:%ux%ux%ux%ux%zu", op->raw.attr.attention.n_heads,
                        op->raw.attr.attention.n_kv_heads, dk, dv ? dv : dk,
                        velri_dtype_bytes(op->outputs[1].dtype));
    }
    case VELRI_OP_QUANTIZE_ROW: return snprintf(dst, size, "quant:%zu", d);
    case VELRI_OP_RMSNORM: return snprintf(dst, size, "rms:%zu", d);
    case VELRI_OP_GATED_MLP: return snprintf(dst, size, "gated:%zu", d);
    case VELRI_OP_EMBEDDING: return snprintf(dst, size, "embedding:%zu", d);
    case VELRI_OP_COPY_LAST_ROW: return snprintf(dst, size, "last:%zu", d);
    case VELRI_OP_VEC_AXPBY: return snprintf(dst, size, "residual:%zu", d);
    case VELRI_OP_FUSED_ADD_RMSNORM: return snprintf(dst, size, "residual-rms:%zu", d);
    default: return snprintf(dst, size, "op-%d:%zu", (int)op->kind, d);
    }
}

void trace_shape(char *out, size_t size, const velri_submit_item *items, uint32_t count) {
    size_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (i) out[offset++] = '+';
        int n = shape(out + offset, size - offset, items[i].op);
        if (n < 0 || (size_t)n + 1 >= size - offset) trace_fail("shape descriptor too large");
        offset += (size_t)n;
    }
}

void trace_sample(int pipe, const char *description, double ns, uint32_t gx, uint32_t gy) {
    fprintf(trace_file, "%u,%u,%d,%s,%u,%u,%.1f\n", run, position, pipe, description, gx, gy, ns);
}

static const velri_backend_vtable *cpu;
typedef struct { const velri_resolved_op *op; uint64_t ns; } cpu_sample;
static cpu_sample *samples;
static size_t length, capacity;

static velri_status cpu_submit(velri_backend_device *dev, const velri_submit_item *item,
                               velri_completion **completion) {
    if (!trace_active) return cpu->submit(dev, item, completion);
    uint64_t start = velri_os_monotonic_ns();
    velri_status rc = cpu->submit(dev, item, completion);
    uint64_t elapsed = velri_os_monotonic_ns() - start;
    if (length == capacity) {
        capacity = capacity ? capacity * 2U : 1024U;
        samples = realloc(samples, capacity * sizeof(*samples));
        if (!samples) trace_fail("allocating CPU samples");
    }
    samples[length++] = (cpu_sample){item->op, elapsed};
    return rc;
}

const velri_backend_vtable *__real_velri_cpu_backend_vtable(void);
const velri_backend_vtable *__wrap_velri_cpu_backend_vtable(void) {
    static velri_backend_vtable copy;
    cpu = __real_velri_cpu_backend_vtable();
    copy = *cpu;
    copy.submit = cpu_submit;
    return &copy;
}

static uint64_t sampler_ns;
static uint32_t sampler_vocab;
velri_rc __real_velri_sampler_sample(velri_sampler *, const float *, const velri_sampling_params *, int32_t *);
velri_rc __wrap_velri_sampler_sample(velri_sampler *sampler, const float *logits,
                                     const velri_sampling_params *params, int32_t *token) {
    if (!trace_active) return __real_velri_sampler_sample(sampler, logits, params, token);
    uint64_t start = velri_os_monotonic_ns();
    velri_rc rc = __real_velri_sampler_sample(sampler, logits, params, token);
    sampler_ns = velri_os_monotonic_ns() - start;
    sampler_vocab = sampler->vocab_size;
    return rc;
}

velri_rc __real_velri_session_step(velri_session *, const int32_t *, uint32_t,
                                   const velri_sampling_params *, velri_cancel *, velri_step_result *);
velri_rc __wrap_velri_session_step(velri_session *session, const int32_t *ids, uint32_t count,
                                   const velri_sampling_params *sampling, velri_cancel *cancel,
                                   velri_step_result *result) {
    if (!trace_file) {
        const char *path = getenv("VELRI_TRACE_FILE");
        if (!path || !(trace_file = fopen(path, "w"))) trace_fail("set VELRI_TRACE_FILE to a writable path");
        fputs("sequence,position,pipeline,ops,gx,gy,ns\n", trace_file);
    }
    if (count > 1) ++run;
    position = velri_session_kv_len(session);
    const char *pick = getenv("VELRI_TRACE_RUN");
    trace_active = count == 1 && (!pick || strtoul(pick, NULL, 10) == run);
    length = 0;
    sampler_ns = 0;
    velri_rc rc = __real_velri_session_step(session, ids, count, sampling, cancel, result);
    if (trace_active && rc == VELRI_RC_OK) {
        char description[256];
        if (sampler_ns) {
            snprintf(description, sizeof(description), "sampler:%u", sampler_vocab);
            trace_sample(-4, description, sampler_ns, 0, 0);
        }
        for (size_t i = 0; i < length; ++i) {
            velri_submit_item item = {.op = samples[i].op};
            trace_shape(description, sizeof(description), &item, 1);
            trace_sample(-1, description, samples[i].ns, 0, 0);
        }
#ifdef TRACE_VULKAN
        trace_gpu_collect();
#endif
    }
    trace_active = false;
    return rc;
}
