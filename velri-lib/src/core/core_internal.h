#ifndef VELRI_CORE_INTERNAL_H
#define VELRI_CORE_INTERNAL_H

#include "os.h"
#include "velri_lib/backend.h"
#include "velri_lib/graph.h"
#include "velri_lib/tensor.h"
#include <stdlib.h>

enum {
    VELRI_MAX_DEVICES = VELRI_MAX_REPORT_DEVICES,
    VELRI_MAX_BACKENDS = 16
};

static inline void *velri_core_aligned_alloc64(size_t bytes) {
    return velri_os_aligned_alloc(64U, bytes);
}

static inline void velri_core_aligned_free64(void *p) { velri_os_aligned_free(p); }

static inline size_t velri_align_up(size_t n, size_t align) {
    if (align <= 1U) {
        return n;
    }
    size_t mask = align - 1U;
    if (n > SIZE_MAX - mask) {
        return SIZE_MAX;
    }
    return (n + mask) & ~mask;
}

/* ---- Context & Registered Backends ---- */

typedef struct {
    const velri_backend_vtable *vt;
    velri_backend_device *dev;
    velri_device_caps caps;
} velri_device_entry;

struct velri_context {
    velri_context_config cfg;
    const velri_backend_vtable *backends[VELRI_MAX_BACKENDS];
    uint32_t num_backends;
    velri_device_entry devices[VELRI_MAX_DEVICES];
    uint32_t num_devices;
};

/* ---- Operation Graph ---- */

typedef struct {
    velri_storage_desc desc;
    char debug_name[64];
} velri_graph_storage;

typedef struct {
    velri_op_desc desc;
    velri_device_id device_id;
} velri_graph_node;

struct velri_graph {
    velri_graph_storage *storages;
    uint32_t num_storages;
    uint32_t cap_storages;

    velri_tensor_view *views;
    uint32_t num_views;
    uint32_t cap_views;

    velri_graph_node *nodes;
    uint32_t num_nodes;
    uint32_t cap_nodes;

    velri_kv_binding *kv_bindings;
    uint32_t num_kv_bindings;
    uint32_t cap_kv_bindings;
};

/* ---- Prepared Launch Plan ---- */

typedef enum {
    VELRI_STEP_OP = 0,
    VELRI_STEP_RAM_TRANSFER = 1
} velri_launch_step_kind;

/* Physical storage table entry inside a prepared plan (user storages + shadow transfer copies). */
typedef struct {
    velri_storage_kind kind;
    velri_device_id device_id;
    size_t byte_size;
    size_t align;
    bool in_scratch_arena; /* True => lives inside per-device packed scratch arena */
    size_t arena_offset;   /* Offset inside that device's scratch arena */
    bool is_shadow;        /* True => planner-synthesized copy on a consumer device */
    uint32_t source_storage_id;
} velri_plan_storage;

typedef struct {
    uint32_t plan_storage_idx; /* Index into plan->storages[] */
    size_t view_byte_offset;
    bool present;
} velri_step_arg;

typedef struct {
    velri_launch_step_kind kind;
    union {
        struct {
            velri_device_id device_id;
            velri_prepared_op *prepared;
            uint32_t num_inputs;
            uint32_t num_outputs;
            velri_step_arg inputs[VELRI_OP_MAX_INPUTS];
            velri_step_arg outputs[VELRI_OP_MAX_OUTPUTS];
            bool touches_state;
        } op;
        struct {
            velri_device_id src_device_id;
            velri_device_id dst_device_id;
            uint32_t src_plan_storage_idx;
            uint32_t dst_plan_storage_idx;
            size_t byte_size;
        } transfer;
    } u;
} velri_launch_step;

struct velri_plan {
    velri_context *ctx;
    uint32_t num_user_storages;
    velri_plan_storage *storages;
    uint32_t num_storages;

    size_t scratch_arena_bytes[VELRI_MAX_DEVICES];
    size_t scratch_arena_align[VELRI_MAX_DEVICES];
    size_t ram_staging_bytes;

    velri_launch_step *steps;
    uint32_t num_steps;

    velri_memory_report report;
};

/* Central operation registry validator (src/ops/registry.c). */
velri_status velri_op_validate(const velri_resolved_op *op);

/* CPU backend configuration hook (sets default worker count & variant for cpu:0). */
void velri_cpu_backend_set_defaults(size_t workers, velri_cpu_variant variant);

#ifdef VELRI_CORE_HAS_VULKAN
const velri_backend_vtable *velri_vulkan_backend_vtable(void);
#endif

#endif /* VELRI_CORE_INTERNAL_H */
