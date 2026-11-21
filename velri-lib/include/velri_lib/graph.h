#ifndef VELRI_LIB_GRAPH_H
#define VELRI_LIB_GRAPH_H

#include "velri_lib/backend.h"
#include "velri_lib/tensor.h"

/*
 * Device-agnostic Operation Graph, Memory/Placement Planner, Prepared Launch Plan,
 * and Independent Execution Slots (`velri-lib/docs/EN/architecture.md §4` & `execution.md`).
 *
 * Lifecycle:
 *   1. Build a `velri_graph` (declare storages, zero-copy views, K/V roles, op nodes).
 *   2. `velri_plan_prepare(ctx, graph, &opts, &plan)` validates everything, checks
 *      backend capabilities (no silent CPU fallback), packs scratch arenas by liveness,
 *      inserts explicit Device A -> RAM staging -> Device B transfers, prepares backend ops,
 *      and computes an exact UMA-aware memory report.
 *   3. Create one or more independent `velri_slot`s from the immutable `velri_plan`.
 *      Concurrent threads may run distinct slots sharing immutable weights with 0 B
 *      heap allocations during `velri_slot_run`.
 */

typedef struct velri_graph velri_graph;
typedef struct velri_plan velri_plan;
typedef struct velri_slot velri_slot;

/* ---- Operation Graph Builder ---- */

velri_status velri_graph_create(velri_graph **out_graph);
void velri_graph_destroy(velri_graph *graph);

velri_status velri_graph_add_storage(
    velri_graph *graph,
    const velri_storage_desc *desc,
    velri_storage_id *out_id);

velri_status velri_graph_add_view(
    velri_graph *graph,
    const velri_tensor_view *view,
    velri_view_id *out_id);

/* Record explicit K/V role bindings (allocates 0 extra storage). */
velri_status velri_graph_add_kv_binding(
    velri_graph *graph,
    const velri_kv_binding *binding);

/* Add an operation node targeted at `device_id` (validated immediately by op registry). */
velri_status velri_graph_add_op(
    velri_graph *graph,
    const velri_op_desc *op,
    velri_device_id device_id,
    velri_node_id *out_node_id);

uint32_t velri_graph_storage_count(const velri_graph *graph);
uint32_t velri_graph_view_count(const velri_graph *graph);
uint32_t velri_graph_node_count(const velri_graph *graph);
uint32_t velri_graph_kv_binding_count(const velri_graph *graph);

/* ---- Plan Preparation & Memory Budget Report ---- */

enum { VELRI_MAX_REPORT_DEVICES = 16 };

typedef struct {
    velri_device_id device_id;
    bool is_uma;
    size_t weight_bytes;
    size_t input_output_bytes;
    size_t state_bytes;
    size_t scratch_unpacked_bytes; /* Sum of declared scratch storages on this device */
    size_t scratch_packed_bytes;   /* Actual liveness-packed scratch arena per slot */
    size_t allocator_overhead_bytes;
} velri_device_memory_report;

typedef struct {
    uint32_t num_devices;
    velri_device_memory_report devices[VELRI_MAX_REPORT_DEVICES];
    size_t ram_staging_bytes;          /* Host RAM staging buffer budget for cross-backend transfers */
    uint32_t num_ram_transfers;        /* Count of explicit cross-device RAM transfers in launch plan */
    size_t total_physical_bytes;       /* UMA-aware peak physical bytes (no double-counting UMA) */
    uint32_t kv_bindings_count;
    uint32_t kv_shared_storage_count;  /* K/V bindings sharing one storage with 0 extra allocation */
    uint32_t kv_derived_count;         /* K/V bindings derived via explicit graph op */
} velri_memory_report;

typedef struct {
    bool allow_cpu_fallback; /* Default false: unsupported op on chosen device returns UNSUPPORTED */
} velri_plan_options;

velri_status velri_plan_prepare(
    velri_context *ctx,
    const velri_graph *graph,
    const velri_plan_options *opts,
    velri_plan **out_plan);

void velri_plan_destroy(velri_plan *plan);

/* Inspect exact UMA-aware memory budget, scratch packing, and RAM transfer count. */
velri_status velri_plan_memory_report(const velri_plan *plan, velri_memory_report *out_report);

/* Query total launch steps (compute ops + inserted RAM transfers). */
uint32_t velri_plan_launch_step_count(const velri_plan *plan);

/* ---- Execution Slots (Independent, Reusable, 0 B Hot-Path Allocation) ---- */

velri_status velri_slot_create(const velri_plan *plan, velri_slot **out_slot);
void velri_slot_destroy(velri_slot *slot);

/*
 * Bind host data to an external storage (WEIGHT, INPUT, OUTPUT, STATE).
 * - If the storage resides on a host-memory device (e.g., CPU), binds directly with zero copy.
 * - If the storage resides on a non-host device (e.g., Vulkan / mock async), uploads to the
 *   slot's pre-allocated device buffer (for WEIGHT/INPUT/STATE) or records the host destination
 *   for download after execution (for OUTPUT/STATE).
 */
velri_status velri_slot_bind_host(
    velri_slot *slot,
    velri_storage_id storage_id,
    void *host_ptr,
    size_t byte_size);

/*
 * Execute the prepared launch plan synchronously on this slot with 0 B heap allocations.
 * Enforces pre-accept rejection, post-accept completion draining, and cancellation semantics.
 */
velri_status velri_slot_run(
    velri_slot *slot,
    const velri_step_params *params,
    velri_cancel_token *cancel);

/* True if a post-accept failure or mid-step cancellation left mutable state needing reset. */
bool velri_slot_needs_reset(const velri_slot *slot);

/* Clear `needs_reset` flag and zero slot-local scratch arenas before resuming execution. */
velri_status velri_slot_reset(velri_slot *slot);

/* Query total heap allocations performed inside `velri_slot_run` (always 0 on hot path). */
uint64_t velri_slot_last_run_allocs(const velri_slot *slot);

#endif /* VELRI_LIB_GRAPH_H */
