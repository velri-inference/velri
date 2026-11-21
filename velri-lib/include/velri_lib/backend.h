#ifndef VELRI_LIB_BACKEND_H
#define VELRI_LIB_BACKEND_H

#include "velri_lib/linear_typed.h"
#include "velri_lib/ops.h"
#include "velri_lib/tensor.h"
#include "velri_lib/types.h"
#include <stdatomic.h>

/*
 * Internal & public C23 Backend Contract and Static Backend Registry.
 *
 * Implements the 6-group backend contract from `velri-lib/docs/EN/architecture.md §5`:
 *   1. Registration (name, contract version, option validation, device enumeration)
 *   2. Device       (open, capabilities/limits, close)
 *   3. Memory       (requirements, alloc/free, copy_to_ram / copy_from_ram)
 *   4. Preparation  (query_operation, prepare_op, destroy_prepared_op)
 *   5. Execution    (submit -> status + completion handle)
 *   6. Completion   (poll, wait, destroy_completion)
 *
 * Notice: No per-operation function pointers exist on the backend vtable.
 * All operations pass through `velri_op_desc` and the shared operation registry.
 */

#define VELRI_BACKEND_CONTRACT_VERSION 1U

enum {
    VELRI_OP_MAX_INPUTS = 6,
    VELRI_OP_MAX_OUTPUTS = 3
};

typedef enum {
    VELRI_OP_LINEAR = 0,
    VELRI_OP_RMSNORM = 1,
    VELRI_OP_FUSED_ADD_RMSNORM = 2,
    VELRI_OP_LAYERNORM = 3,
    VELRI_OP_ROPE = 4,
    VELRI_OP_ACTIVATION = 5,
    VELRI_OP_GATED_MLP = 6,
    VELRI_OP_SOFTMAX = 7,
    VELRI_OP_VEC_AXPBY = 8,
    VELRI_OP_VEC_MUL = 9,
    VELRI_OP_FWHT = 10,
    VELRI_OP_QUANTIZE_ROW = 11
} velri_op_kind;

typedef struct {
    velri_op_kind kind;
    uint32_t num_inputs;
    uint32_t num_outputs;
    velri_view_id inputs[VELRI_OP_MAX_INPUTS];   /* VELRI_INVALID_ID for optional absent inputs */
    velri_view_id outputs[VELRI_OP_MAX_OUTPUTS];
    union {
        struct {
            velri_linear_desc desc;
        } linear;
        struct {
            float eps;
            bool unit_offset;
        } rmsnorm;
        struct {
            float eps;
        } layernorm;
        struct {
            size_t rotary_dim;
            size_t pos_offset;
            velri_rope_mode mode;
        } rope;
        struct {
            velri_act_kind kind;
        } activation;
        struct {
            velri_act_kind act;
            float clamp_limit;
        } gated_mlp;
        struct {
            float alpha;
            float beta;
        } axpby;
        struct {
            size_t block_size;
            velri_dtype suh_dtype;
        } fwht;
    } attr;
} velri_op_desc;

/* Fully resolved view description passed to backend query_operation / prepare_op. */
typedef struct {
    velri_op_kind kind;
    uint32_t num_inputs;
    uint32_t num_outputs;
    bool input_present[VELRI_OP_MAX_INPUTS];
    velri_tensor_view inputs[VELRI_OP_MAX_INPUTS];
    velri_tensor_view outputs[VELRI_OP_MAX_OUTPUTS];
    velri_op_desc raw;
} velri_resolved_op;

/* Resolved device buffer binding passed to backend `submit` (zero heap allocations). */
typedef struct {
    void *mem_handle;    /* Backend allocation handle (or host ptr for host-backed bindings) */
    size_t byte_offset;  /* Arena offset + tensor view byte_offset */
    size_t byte_size;    /* Valid bound byte size from offset 0 of allocation */
    bool is_host_ptr;    /* True if `mem_handle` is a direct host pointer bound on a host device */
} velri_buffer_binding;

/* Bounded dynamic execution parameters supplied per step (0 B heap allocation). */
typedef struct {
    size_t pos_offset_override;
    bool has_pos_offset;
} velri_step_params;

/* Cooperative cancellation token shared between caller and runtime executor. */
typedef struct {
    atomic_bool cancelled;
} velri_cancel_token;

static inline void velri_cancel_init(velri_cancel_token *tok) {
    if (tok) {
        atomic_init(&tok->cancelled, false);
    }
}

static inline void velri_cancel_request(velri_cancel_token *tok) {
    if (tok) {
        atomic_store_explicit(&tok->cancelled, true, memory_order_release);
    }
}

static inline bool velri_cancel_is_requested(const velri_cancel_token *tok) {
    return tok ? atomic_load_explicit(&tok->cancelled, memory_order_acquire) : false;
}

typedef enum {
    VELRI_POLL_PENDING = 0,
    VELRI_POLL_DONE = 1,
    VELRI_POLL_FAILED = 2
} velri_poll_state;

typedef struct {
    char backend_name[32];
    char device_name[128];
    uint32_t local_index;
    bool is_host_memory;     /* Device directly executes on host RAM pointers */
    bool is_uma;             /* Shares physical RAM pool with CPU (never double-count budget) */
    bool mem_total_known;    /* False if driver does not report reliable memory size */
    size_t mem_total_bytes;  /* 0 if !mem_total_known */
    size_t min_align_bytes;  /* Required base allocation alignment */
} velri_device_caps;

typedef struct {
    size_t byte_size;
    size_t align_bytes;
    size_t alloc_overhead_bytes;
} velri_mem_req;

typedef struct velri_backend_device velri_backend_device;
typedef struct velri_prepared_op velri_prepared_op;
typedef struct velri_completion velri_completion;

/*
 * The 6-group backend contract vtable.
 *
 * Resource-safety contract for `submit`:
 *   - Pre-accept rejection: returns error AND sets `*out_completion = nullptr`.
 *     The device started nothing and holds no references to submitted buffers.
 *   - Acceptance (including partial acceptance followed by error): sets
 *     `*out_completion != nullptr`. The runtime MUST call `wait` (and then
 *     `destroy_completion`) before releasing or reusing any bound buffer.
 */
typedef struct velri_backend_vtable {
    /* 1. Registration */
    const char *name;
    uint32_t contract_version;
    velri_status (*validate_option)(const char *key, const char *value);
    velri_status (*enumerate_devices)(uint32_t *out_count);

    /* 2. Device */
    velri_status (*open_device)(uint32_t local_index, velri_backend_device **out_dev);
    velri_status (*query_caps)(const velri_backend_device *dev, velri_device_caps *out_caps);
    void (*close_device)(velri_backend_device *dev);

    /* 3. Memory */
    velri_status (*query_mem_req)(const velri_backend_device *dev, size_t logical_bytes,
                                  size_t align, velri_mem_req *out_req);
    velri_status (*alloc)(velri_backend_device *dev, size_t bytes, size_t align,
                          void **out_mem_handle);
    void (*free)(velri_backend_device *dev, void *mem_handle);
    velri_status (*copy_to_ram)(velri_backend_device *dev, void *dst_ram,
                                const velri_buffer_binding *src, size_t bytes);
    velri_status (*copy_from_ram)(velri_backend_device *dev, const velri_buffer_binding *dst,
                                  const void *src_ram, size_t bytes);

    /* 4. Preparation */
    velri_status (*query_operation)(const velri_backend_device *dev, const velri_resolved_op *op);
    velri_status (*prepare_op)(velri_backend_device *dev, const velri_resolved_op *op,
                               velri_prepared_op **out_prepared);
    void (*destroy_prepared_op)(velri_backend_device *dev, velri_prepared_op *prepared);

    /* 5. Execution */
    velri_status (*submit)(velri_backend_device *dev,
                           const velri_prepared_op *prepared,
                           const velri_buffer_binding *inputs, uint32_t num_inputs,
                           const velri_buffer_binding *outputs, uint32_t num_outputs,
                           const velri_step_params *params,
                           velri_completion **out_completion);

    /* 6. Completion */
    velri_poll_state (*poll)(velri_backend_device *dev, velri_completion *comp);
    velri_status (*wait)(velri_backend_device *dev, velri_completion *comp);
    void (*destroy_completion)(velri_backend_device *dev, velri_completion *comp);
} velri_backend_vtable;

/* Runtime context owning statically registered backends and open devices. */
typedef struct velri_context velri_context;

typedef struct {
    size_t cpu_workers;            /* 0 => auto (physical/logical threads capped reasonably) */
    velri_cpu_variant cpu_variant; /* VELRI_CPU_AUTO by default */
    bool enable_vulkan;            /* Try opening Vulkan device if compiled with Vulkan */
} velri_context_config;

/* Create runtime context and register compiled-in backends (CPU, optional Vulkan). */
velri_status velri_context_create(const velri_context_config *cfg, velri_context **out_ctx);
void velri_context_destroy(velri_context *ctx);

/* Register an additional static backend (e.g., mock async backend in contract tests). */
velri_status velri_context_register_backend(velri_context *ctx, const velri_backend_vtable *vtable);

/* Query total open devices and inspect capabilities by device_id or "<backend>:<index>". */
uint32_t velri_context_device_count(const velri_context *ctx);
velri_status velri_context_device_caps(const velri_context *ctx, velri_device_id device_id,
                                       velri_device_caps *out_caps);
velri_status velri_context_find_device(const velri_context *ctx, const char *backend_name,
                                       uint32_t local_index, velri_device_id *out_device_id);

/* Built-in CPU backend vtable (Slang AOT over persistent velri_pool). */
const velri_backend_vtable *velri_cpu_backend_vtable(void);

#endif /* VELRI_LIB_BACKEND_H */
