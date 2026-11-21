#include "core_internal.h"
#include <string.h>

/*
 * Memory/Placement Planner & Launch-Graph Preparation (`velri-lib/docs/EN/architecture.md §4`).
 *
 * Responsibilities during `velri_plan_prepare`:
 *   1. Resolve every graph op against registered devices (`query_operation`).
 *      Never falls back silently to CPU unless `opts->allow_cpu_fallback` is explicitly true.
 *   2. Insert explicit `Device A -> RAM staging -> Device B` transfer steps whenever a
 *      consumer executes on a different device than the storage's current residence.
 *   3. Pack non-overlapping `VELRI_STORAGE_SCRATCH` storages into one per-device arena
 *      using liveness intervals (zero extra storage allocated for K/V views sharing a storage).
 *   4. Prepare backend op handles once and compute a UMA-aware memory budget report.
 */

typedef struct {
    velri_plan *plan;
    uint32_t storage_cap;
    uint32_t step_cap;
} plan_builder;

static bool grow_storages(plan_builder *pb, uint32_t need) {
    if (need <= pb->storage_cap) {
        return true;
    }
    uint32_t cap = pb->storage_cap ? pb->storage_cap * 2U : 32U;
    while (cap < need) {
        cap *= 2U;
    }
    velri_plan_storage *p = (velri_plan_storage *)realloc(pb->plan->storages,
                                                          (size_t)cap * sizeof(*p));
    if (!p) {
        return false;
    }
    pb->plan->storages = p;
    pb->storage_cap = cap;
    return true;
}

static bool grow_steps(plan_builder *pb, uint32_t need) {
    if (need <= pb->step_cap) {
        return true;
    }
    uint32_t cap = pb->step_cap ? pb->step_cap * 2U : 32U;
    while (cap < need) {
        cap *= 2U;
    }
    velri_launch_step *p = (velri_launch_step *)realloc(pb->plan->steps, (size_t)cap * sizeof(*p));
    if (!p) {
        return false;
    }
    pb->plan->steps = p;
    pb->step_cap = cap;
    return true;
}

static velri_status add_shadow_storage(
    plan_builder *pb,
    velri_device_id dst_dev,
    uint32_t source_sid,
    size_t byte_size,
    size_t align,
    uint32_t *out_idx) {
    if (!grow_storages(pb, pb->plan->num_storages + 1U)) {
        return VELRI_ERR_NOMEM;
    }
    uint32_t idx = pb->plan->num_storages++;
    velri_plan_storage *ps = &pb->plan->storages[idx];
    memset(ps, 0, sizeof(*ps));
    ps->kind = VELRI_STORAGE_SCRATCH;
    ps->device_id = dst_dev;
    ps->byte_size = byte_size;
    ps->align = align;
    ps->is_shadow = true;
    ps->source_storage_id = source_sid;
    *out_idx = idx;
    return VELRI_OK;
}

static velri_status emit_ram_transfer(
    plan_builder *pb,
    velri_device_id src_dev,
    uint32_t src_idx,
    velri_device_id dst_dev,
    uint32_t dst_idx,
    size_t byte_size) {
    if (!grow_steps(pb, pb->plan->num_steps + 1U)) {
        return VELRI_ERR_NOMEM;
    }
    velri_launch_step *step = &pb->plan->steps[pb->plan->num_steps++];
    memset(step, 0, sizeof(*step));
    step->kind = VELRI_STEP_RAM_TRANSFER;
    step->u.transfer.src_device_id = src_dev;
    step->u.transfer.dst_device_id = dst_dev;
    step->u.transfer.src_plan_storage_idx = src_idx;
    step->u.transfer.dst_plan_storage_idx = dst_idx;
    step->u.transfer.byte_size = byte_size;

    if (byte_size > pb->plan->ram_staging_bytes) {
        pb->plan->ram_staging_bytes = byte_size;
    }
    return VELRI_OK;
}

static void resolve_graph_node(
    const velri_graph *graph,
    const velri_graph_node *node,
    velri_resolved_op *out) {
    memset(out, 0, sizeof(*out));
    out->kind = node->desc.kind;
    out->num_inputs = node->desc.num_inputs;
    out->num_outputs = node->desc.num_outputs;
    out->raw = node->desc;

    for (uint32_t i = 0U; i < node->desc.num_inputs; ++i) {
        velri_view_id vid = node->desc.inputs[i];
        if (vid != VELRI_INVALID_ID) {
            out->input_present[i] = true;
            out->inputs[i] = graph->views[vid];
        }
    }
    for (uint32_t i = 0U; i < node->desc.num_outputs; ++i) {
        out->outputs[i] = graph->views[node->desc.outputs[i]];
    }
}

/* Query target device; only fall back to cpu:0 if explicitly permitted by plan options. */
static velri_status select_node_device(
    velri_context *ctx,
    const velri_resolved_op *resolved,
    velri_device_id preferred_dev,
    bool allow_cpu_fallback,
    velri_device_id *out_dev) {
    if (preferred_dev >= ctx->num_devices) {
        return VELRI_ERR_NOT_FOUND;
    }
    const velri_device_entry *entry = &ctx->devices[preferred_dev];
    velri_status st = entry->vt->query_operation(entry->dev, resolved);
    if (st == VELRI_OK) {
        *out_dev = preferred_dev;
        return VELRI_OK;
    }
    if (!allow_cpu_fallback) {
        return VELRI_ERR_UNSUPPORTED;
    }

    velri_device_id cpu_dev = 0U;
    if (velri_context_find_device(ctx, "cpu", 0U, &cpu_dev) != VELRI_OK ||
        cpu_dev == preferred_dev) {
        return VELRI_ERR_UNSUPPORTED;
    }
    const velri_device_entry *cpu_entry = &ctx->devices[cpu_dev];
    if (cpu_entry->vt->query_operation(cpu_entry->dev, resolved) != VELRI_OK) {
        return VELRI_ERR_UNSUPPORTED;
    }
    *out_dev = cpu_dev;
    return VELRI_OK;
}

typedef struct {
    size_t start;
    size_t end;
    uint32_t first_step;
    uint32_t last_step;
} placed_interval;

/* Compute exact [first_step, last_step] liveness and pack scratch buffers per device. */
static velri_status pack_scratch_arenas(velri_plan *plan) {
    uint32_t ns = plan->num_storages;
    if (ns == 0U) {
        return VELRI_OK;
    }

    uint32_t *first_use = (uint32_t *)malloc((size_t)ns * sizeof(uint32_t));
    uint32_t *last_use = (uint32_t *)malloc((size_t)ns * sizeof(uint32_t));
    placed_interval *placed = (placed_interval *)malloc((size_t)ns * sizeof(placed_interval));
    if (!first_use || !last_use || !placed) {
        free(first_use);
        free(last_use);
        free(placed);
        return VELRI_ERR_NOMEM;
    }

    for (uint32_t i = 0U; i < ns; ++i) {
        first_use[i] = UINT32_MAX;
        last_use[i] = 0U;
    }

    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        const velri_launch_step *step = &plan->steps[s];
        if (step->kind == VELRI_STEP_OP) {
            for (uint32_t i = 0U; i < step->u.op.num_inputs; ++i) {
                if (!step->u.op.inputs[i].present) {
                    continue;
                }
                uint32_t idx = step->u.op.inputs[i].plan_storage_idx;
                if (s < first_use[idx]) first_use[idx] = s;
                if (s > last_use[idx]) last_use[idx] = s;
            }
            for (uint32_t i = 0U; i < step->u.op.num_outputs; ++i) {
                uint32_t idx = step->u.op.outputs[i].plan_storage_idx;
                if (s < first_use[idx]) first_use[idx] = s;
                if (s > last_use[idx]) last_use[idx] = s;
            }
        } else {
            uint32_t src = step->u.transfer.src_plan_storage_idx;
            uint32_t dst = step->u.transfer.dst_plan_storage_idx;
            if (s < first_use[src]) first_use[src] = s;
            if (s > last_use[src]) last_use[src] = s;
            if (s < first_use[dst]) first_use[dst] = s;
            if (s > last_use[dst]) last_use[dst] = s;
        }
    }

    /* Linear-scan non-overlapping offset packing, independently per device. */
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        uint32_t n_placed = 0U;
        size_t max_align = plan->ctx->devices[d].caps.min_align_bytes;
        if (max_align < 64U) {
            max_align = 64U;
        }
        size_t arena_size = 0U;

        for (uint32_t i = 0U; i < ns; ++i) {
            velri_plan_storage *ps = &plan->storages[i];
            if (ps->kind != VELRI_STORAGE_SCRATCH || ps->device_id != d) {
                continue;
            }
            if (first_use[i] == UINT32_MAX) {
                continue; /* Dead / unreferenced scratch allocates 0 bytes. */
            }

            size_t align = ps->align < max_align ? max_align : ps->align;
            if (align > max_align) {
                max_align = align;
            }
            size_t sz = velri_align_up(ps->byte_size, align);
            if (sz == SIZE_MAX) {
                free(first_use);
                free(last_use);
                free(placed);
                return VELRI_ERR_OVERFLOW;
            }

            /* Find lowest aligned offset that does not intersect any simultaneously live block. */
            size_t cand = 0U;
            bool conflict = true;
            while (conflict) {
                conflict = false;
                for (uint32_t p = 0U; p < n_placed; ++p) {
                    bool time_overlap = !(last_use[i] < placed[p].first_step ||
                                          first_use[i] > placed[p].last_step);
                    bool addr_overlap = (cand < placed[p].end) && (cand + sz > placed[p].start);
                    if (time_overlap && addr_overlap) {
                        cand = velri_align_up(placed[p].end, align);
                        conflict = true;
                        break;
                    }
                }
            }

            ps->in_scratch_arena = true;
            ps->arena_offset = cand;
            placed[n_placed++] = (placed_interval){
                .start = cand,
                .end = cand + sz,
                .first_step = first_use[i],
                .last_step = last_use[i],
            };
            if (cand + sz > arena_size) {
                arena_size = cand + sz;
            }
        }

        plan->scratch_arena_bytes[d] = arena_size;
        plan->scratch_arena_align[d] = max_align;
    }

    free(first_use);
    free(last_use);
    free(placed);
    return VELRI_OK;
}

static void compute_memory_report(velri_plan *plan, const velri_graph *graph) {
    velri_memory_report *rep = &plan->report;
    memset(rep, 0, sizeof(*rep));
    rep->num_devices = plan->ctx->num_devices;
    rep->ram_staging_bytes = plan->ram_staging_bytes;

    for (uint32_t d = 0U; d < rep->num_devices; ++d) {
        rep->devices[d].device_id = d;
        rep->devices[d].is_uma = plan->ctx->devices[d].caps.is_uma;
        rep->devices[d].scratch_packed_bytes = plan->scratch_arena_bytes[d];
    }

    for (uint32_t i = 0U; i < plan->num_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        velri_device_memory_report *dr = &rep->devices[ps->device_id];
        const velri_device_entry *de = &plan->ctx->devices[ps->device_id];

        velri_mem_req req = {0};
        if (de->vt->query_mem_req(de->dev, ps->byte_size, ps->align, &req) == VELRI_OK) {
            if (ps->kind != VELRI_STORAGE_SCRATCH) {
                dr->allocator_overhead_bytes += req.alloc_overhead_bytes;
            }
        }

        switch (ps->kind) {
        case VELRI_STORAGE_WEIGHT:
            dr->weight_bytes += ps->byte_size;
            break;
        case VELRI_STORAGE_INPUT:
        case VELRI_STORAGE_OUTPUT:
            dr->input_output_bytes += ps->byte_size;
            break;
        case VELRI_STORAGE_STATE:
            dr->state_bytes += ps->byte_size;
            break;
        case VELRI_STORAGE_SCRATCH:
            dr->scratch_unpacked_bytes += ps->byte_size;
            break;
        }
    }

    for (uint32_t d = 0U; d < rep->num_devices; ++d) {
        if (plan->scratch_arena_bytes[d] > 0U) {
            const velri_device_entry *de = &plan->ctx->devices[d];
            velri_mem_req req = {0};
            if (de->vt->query_mem_req(de->dev, plan->scratch_arena_bytes[d],
                                      plan->scratch_arena_align[d], &req) == VELRI_OK) {
                rep->devices[d].allocator_overhead_bytes += req.alloc_overhead_bytes;
            }
        }
    }

    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        if (plan->steps[s].kind == VELRI_STEP_RAM_TRANSFER) {
            rep->num_ram_transfers++;
        }
    }

    /* UMA-aware total: sum distinct physical pools once + host RAM staging. */
    size_t total = rep->ram_staging_bytes;
    for (uint32_t d = 0U; d < rep->num_devices; ++d) {
        const velri_device_memory_report *dr = &rep->devices[d];
        total += dr->weight_bytes + dr->input_output_bytes + dr->state_bytes +
                 dr->scratch_packed_bytes + dr->allocator_overhead_bytes;
    }
    rep->total_physical_bytes = total;

    rep->kv_bindings_count = graph->num_kv_bindings;
    for (uint32_t k = 0U; k < graph->num_kv_bindings; ++k) {
        if (graph->kv_bindings[k].kind == VELRI_KV_SHARED) {
            rep->kv_shared_storage_count++;
        } else if (graph->kv_bindings[k].kind == VELRI_KV_DERIVED) {
            rep->kv_derived_count++;
        }
    }
}

velri_status velri_plan_prepare(
    velri_context *ctx,
    const velri_graph *graph,
    const velri_plan_options *opts,
    velri_plan **out_plan) {
    if (!ctx || !graph || !out_plan) {
        return VELRI_ERR_INVALID;
    }
    *out_plan = nullptr;

    bool allow_cpu_fallback = opts ? opts->allow_cpu_fallback : false;

    /* Validate all declared storage devices exist before allocating plan structures. */
    for (uint32_t i = 0U; i < graph->num_storages; ++i) {
        if (graph->storages[i].desc.device_id >= ctx->num_devices) {
            return VELRI_ERR_NOT_FOUND;
        }
    }

    velri_plan *plan = (velri_plan *)calloc(1U, sizeof(*plan));
    if (!plan) {
        return VELRI_ERR_NOMEM;
    }
    plan->ctx = ctx;
    plan->num_user_storages = graph->num_storages;

    plan_builder pb = {.plan = plan, .storage_cap = 0U, .step_cap = 0U};

    /* Copy declared user storages into plan->storages[0 .. num_user_storages-1]. */
    if (graph->num_storages > 0U && !grow_storages(&pb, graph->num_storages)) {
        velri_plan_destroy(plan);
        return VELRI_ERR_NOMEM;
    }
    uint32_t *resident_idx = (uint32_t *)malloc((size_t)(graph->num_storages + 1U) * sizeof(uint32_t));
    velri_device_id *resident_dev =
        (velri_device_id *)malloc((size_t)(graph->num_storages + 1U) * sizeof(velri_device_id));
    if (!resident_idx || !resident_dev) {
        free(resident_idx);
        free(resident_dev);
        velri_plan_destroy(plan);
        return VELRI_ERR_NOMEM;
    }

    for (uint32_t i = 0U; i < graph->num_storages; ++i) {
        const velri_storage_desc *sd = &graph->storages[i].desc;
        velri_plan_storage *ps = &plan->storages[i];
        memset(ps, 0, sizeof(*ps));
        ps->kind = sd->kind;
        ps->device_id = sd->device_id;
        ps->byte_size = sd->byte_size;
        ps->align = sd->align;
        ps->source_storage_id = i;
        resident_idx[i] = i;
        resident_dev[i] = sd->device_id;
    }
    plan->num_storages = graph->num_storages;

    /* Lower each operation node into device-verified steps + explicit RAM transfers. */
    velri_status st = VELRI_OK;
    for (uint32_t n = 0U; n < graph->num_nodes; ++n) {
        const velri_graph_node *node = &graph->nodes[n];
        velri_resolved_op resolved;
        resolve_graph_node(graph, node, &resolved);

        velri_device_id chosen_dev = 0U;
        st = select_node_device(ctx, &resolved, node->device_id, allow_cpu_fallback, &chosen_dev);
        if (st != VELRI_OK) {
            break;
        }

        velri_step_arg in_args[VELRI_OP_MAX_INPUTS];
        velri_step_arg out_args[VELRI_OP_MAX_OUTPUTS];
        memset(in_args, 0, sizeof(in_args));
        memset(out_args, 0, sizeof(out_args));

        /* Ensure every present input resides on `chosen_dev` via explicit RAM transfer if needed. */
        for (uint32_t i = 0U; i < resolved.num_inputs; ++i) {
            if (!resolved.input_present[i]) {
                continue;
            }
            velri_storage_id sid = resolved.inputs[i].storage_id;
            uint32_t src_idx = resident_idx[sid];
            velri_device_id src_dev = resident_dev[sid];
            uint32_t use_idx = src_idx;

            if (src_dev != chosen_dev) {
                uint32_t shadow_idx = 0U;
                st = add_shadow_storage(&pb, chosen_dev, sid, plan->storages[sid].byte_size,
                                        plan->storages[sid].align, &shadow_idx);
                if (st != VELRI_OK) {
                    break;
                }
                st = emit_ram_transfer(&pb, src_dev, src_idx, chosen_dev, shadow_idx,
                                       plan->storages[sid].byte_size);
                if (st != VELRI_OK) {
                    break;
                }
                use_idx = shadow_idx;
            }

            in_args[i].present = true;
            in_args[i].plan_storage_idx = use_idx;
            in_args[i].view_byte_offset = resolved.inputs[i].byte_offset;
        }
        if (st != VELRI_OK) {
            break;
        }

        /* Prepare output slots on `chosen_dev` (handling read-modify-write state/residual cleanly). */
        bool touches_state = false;
        for (uint32_t o = 0U; o < resolved.num_outputs; ++o) {
            velri_storage_id sid = resolved.outputs[o].storage_id;
            velri_storage_kind skind = plan->storages[sid].kind;
            if (skind == VELRI_STORAGE_STATE) {
                touches_state = true;
            }
            bool is_rmw = (resolved.kind == VELRI_OP_FUSED_ADD_RMSNORM && o == 0U);
            velri_device_id home_dev = plan->storages[sid].device_id;
            uint32_t dst_idx = sid;

            if (home_dev != chosen_dev) {
                uint32_t shadow_idx = 0U;
                st = add_shadow_storage(&pb, chosen_dev, sid, plan->storages[sid].byte_size,
                                        plan->storages[sid].align, &shadow_idx);
                if (st != VELRI_OK) {
                    break;
                }
                if (is_rmw && resident_dev[sid] != chosen_dev) {
                    st = emit_ram_transfer(&pb, resident_dev[sid], resident_idx[sid], chosen_dev,
                                           shadow_idx, plan->storages[sid].byte_size);
                    if (st != VELRI_OK) {
                        break;
                    }
                }
                dst_idx = shadow_idx;
            } else if (is_rmw && resident_dev[sid] != chosen_dev) {
                st = emit_ram_transfer(&pb, resident_dev[sid], resident_idx[sid], chosen_dev, sid,
                                       plan->storages[sid].byte_size);
                if (st != VELRI_OK) {
                    break;
                }
            }

            out_args[o].present = true;
            out_args[o].plan_storage_idx = dst_idx;
            out_args[o].view_byte_offset = resolved.outputs[o].byte_offset;
            resident_idx[sid] = dst_idx;
            resident_dev[sid] = chosen_dev;
        }
        if (st != VELRI_OK) {
            break;
        }

        /* Prepare backend-specific op handle once during `prepare`. */
        velri_device_entry *de = &ctx->devices[chosen_dev];
        velri_prepared_op *prepared = nullptr;
        st = de->vt->prepare_op(de->dev, &resolved, &prepared);
        if (st != VELRI_OK) {
            break;
        }

        if (!grow_steps(&pb, plan->num_steps + 1U)) {
            de->vt->destroy_prepared_op(de->dev, prepared);
            st = VELRI_ERR_NOMEM;
            break;
        }

        velri_launch_step *step = &plan->steps[plan->num_steps++];
        memset(step, 0, sizeof(*step));
        step->kind = VELRI_STEP_OP;
        step->u.op.device_id = chosen_dev;
        step->u.op.prepared = prepared;
        step->u.op.num_inputs = resolved.num_inputs;
        step->u.op.num_outputs = resolved.num_outputs;
        step->u.op.touches_state = touches_state;
        memcpy(step->u.op.inputs, in_args, sizeof(in_args));
        memcpy(step->u.op.outputs, out_args, sizeof(out_args));
    }

    /* Transfer any OUTPUT or STATE storage back to its declared home device if produced elsewhere. */
    if (st == VELRI_OK) {
        for (uint32_t sid = 0U; sid < graph->num_storages; ++sid) {
            velri_storage_kind kind = plan->storages[sid].kind;
            if ((kind == VELRI_STORAGE_OUTPUT || kind == VELRI_STORAGE_STATE) &&
                resident_dev[sid] != plan->storages[sid].device_id) {
                st = emit_ram_transfer(&pb, resident_dev[sid], resident_idx[sid],
                                       plan->storages[sid].device_id, sid,
                                       plan->storages[sid].byte_size);
                if (st != VELRI_OK) {
                    break;
                }
            }
        }
    }

    free(resident_idx);
    free(resident_dev);

    if (st == VELRI_OK) {
        st = pack_scratch_arenas(plan);
    }
    if (st != VELRI_OK) {
        velri_plan_destroy(plan);
        return st;
    }

    compute_memory_report(plan, graph);
    *out_plan = plan;
    return VELRI_OK;
}

void velri_plan_destroy(velri_plan *plan) {
    if (!plan) {
        return;
    }
    if (plan->steps && plan->ctx) {
        for (uint32_t i = 0U; i < plan->num_steps; ++i) {
            velri_launch_step *step = &plan->steps[i];
            if (step->kind == VELRI_STEP_OP && step->u.op.prepared) {
                velri_device_entry *de = &plan->ctx->devices[step->u.op.device_id];
                de->vt->destroy_prepared_op(de->dev, step->u.op.prepared);
            }
        }
    }
    free(plan->steps);
    free(plan->storages);
    free(plan);
}

velri_status velri_plan_memory_report(const velri_plan *plan, velri_memory_report *out_report) {
    if (!plan || !out_report) {
        return VELRI_ERR_INVALID;
    }
    *out_report = plan->report;
    return VELRI_OK;
}

uint32_t velri_plan_launch_step_count(const velri_plan *plan) {
    return plan ? plan->num_steps : 0U;
}
