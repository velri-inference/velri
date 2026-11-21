#include "core_internal.h"
#include <string.h>

/*
 * Memory/Placement Planner & Launch-Graph Preparation (`velri-lib/docs/EN/architecture.md §4`).
 *
 * Memory-efficiency guarantees:
 *   - Packs all non-host `VELRI_STORAGE_WEIGHT` storages into contiguous per-device weight slabs
 *     (up to VELRI_WEIGHT_SLAB_LIMIT each), eliminating per-tensor VkDeviceMemory allocations.
 *   - Packs all `VELRI_STORAGE_STATE` storages into one contiguous arena per device per slot.
 *   - Packs all non-host `VELRI_STORAGE_INPUT`/`OUTPUT` storages into one IO arena per device per slot.
 *   - Packs non-overlapping `VELRI_STORAGE_SCRATCH` storages (sorted descending by size, then
 *     lifetime span) into one liveness-packed arena per device per slot.
 *   - Eliminates `ram_staging` allocation whenever either transfer endpoint is host memory.
 */

typedef struct {
    velri_plan *plan;
    uint32_t storage_cap;
    uint32_t step_cap;
} plan_builder;

static velri_status add_shadow_storage(plan_builder *pb, velri_device_id dst_dev,
                                       velri_dtype dtype, size_t byte_size, size_t align,
                                       uint32_t *out_idx) {
    if (pb->plan->num_storages == UINT32_MAX ||
        !velri_vec_grow((void **)&pb->plan->storages, &pb->storage_cap,
                        pb->plan->num_storages + 1U, sizeof(velri_plan_storage))) {
        return VELRI_ERR_NOMEM;
    }
    uint32_t idx = pb->plan->num_storages++;
    velri_plan_storage *ps = &pb->plan->storages[idx];
    memset(ps, 0, sizeof(*ps));
    ps->kind = VELRI_STORAGE_SCRATCH;
    ps->dtype = dtype;
    ps->device_id = dst_dev;
    ps->byte_size = byte_size;
    ps->align = align;
    *out_idx = idx;
    return VELRI_OK;
}

static velri_status emit_ram_transfer(plan_builder *pb, velri_device_id src_dev, uint32_t src_idx,
                                      velri_device_id dst_dev, uint32_t dst_idx, size_t byte_size) {
    if (pb->plan->num_steps == UINT32_MAX ||
        !velri_vec_grow((void **)&pb->plan->steps, &pb->step_cap,
                        pb->plan->num_steps + 1U, sizeof(velri_launch_step))) {
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

    const velri_context *ctx = pb->plan->ctx;
    bool src_host = ctx->devices[src_dev].caps.is_host_memory;
    bool dst_host = ctx->devices[dst_dev].caps.is_host_memory;
    if (!src_host && !dst_host && byte_size > pb->plan->ram_staging_bytes) {
        pb->plan->ram_staging_bytes = byte_size;
    }
    return VELRI_OK;
}

static velri_status select_node_device(velri_context *ctx, const velri_resolved_op *resolved,
                                       velri_device_id preferred_dev, bool allow_cpu_fallback,
                                       velri_device_id *out_dev) {
    if (preferred_dev >= ctx->num_devices) {
        return VELRI_ERR_NOT_FOUND;
    }
    const velri_device_entry *entry = &ctx->devices[preferred_dev];
    if (entry->vt->query_operation(entry->dev, resolved) == VELRI_OK) {
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
    const velri_device_entry *ce = &ctx->devices[cpu_dev];
    if (ce->vt->query_operation(ce->dev, resolved) != VELRI_OK) {
        return VELRI_ERR_UNSUPPORTED;
    }
    *out_dev = cpu_dev;
    return VELRI_OK;
}

static bool view_covers_full_storage(const velri_tensor_view *v, size_t storage_bytes) {
    size_t s0 = 0U, s1 = 0U;
    if (velri_tensor_view_byte_span(v, &s0, &s1) != VELRI_OK) {
        return false;
    }
    return velri_tensor_view_is_contiguous(v) && s0 == 0U && s1 == storage_bytes;
}

static size_t effective_align(const velri_context *ctx, velri_device_id d, size_t req_align) {
    size_t min_a = ctx->devices[d].caps.min_align_bytes;
    if (min_a < 64U) {
        min_a = 64U;
    }
    return req_align < min_a ? min_a : req_align;
}

static velri_status pack_weight_slabs(velri_plan *plan) {
    velri_context *ctx = plan->ctx;
    for (uint32_t d = 0U; d < ctx->num_devices; ++d) {
        if (ctx->devices[d].caps.is_host_memory) {
            continue;
        }
        size_t slab_align[VELRI_MAX_WEIGHT_SLABS];
        memset(slab_align, 0, sizeof(slab_align));
        uint32_t cur_slab = 0U;
        size_t cur_off = 0U;
        size_t cur_align = effective_align(ctx, d, 0U);

        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            velri_plan_storage *ps = &plan->storages[i];
            if (ps->kind != VELRI_STORAGE_WEIGHT || ps->device_id != d) {
                continue;
            }
            size_t align = effective_align(ctx, d, ps->align);
            size_t aligned_off = velri_align_up(cur_off, align);
            if (aligned_off == SIZE_MAX || aligned_off > SIZE_MAX - ps->byte_size) {
                return VELRI_ERR_OVERFLOW;
            }
            if (cur_off > 0U && aligned_off + ps->byte_size > VELRI_WEIGHT_SLAB_LIMIT) {
                plan->weight_slab_bytes[d][cur_slab] = cur_off;
                slab_align[cur_slab] = cur_align;
                cur_slab++;
                if (cur_slab >= VELRI_MAX_WEIGHT_SLABS) {
                    return VELRI_ERR_NOMEM;
                }
                cur_off = 0U;
                cur_align = effective_align(ctx, d, 0U);
                aligned_off = 0U;
            }
            if (align > cur_align) {
                cur_align = align;
            }
            ps->arena_kind = VELRI_ARENA_WEIGHT;
            ps->slab_idx = cur_slab;
            ps->arena_offset = aligned_off;
            cur_off = aligned_off + ps->byte_size;
        }

        if (cur_off > 0U) {
            plan->weight_slab_bytes[d][cur_slab] = cur_off;
            slab_align[cur_slab] = cur_align;
            plan->num_weight_slabs[d] = cur_slab + 1U;
        }

        for (uint32_t s = 0U; s < plan->num_weight_slabs[d]; ++s) {
            size_t sz = velri_align_up(plan->weight_slab_bytes[d][s], slab_align[s]);
            if (sz == SIZE_MAX) {
                return VELRI_ERR_OVERFLOW;
            }
            plan->weight_slab_bytes[d][s] = sz;
            velri_status st = velri_device_alloc_tracked(ctx, d, sz, slab_align[s],
                                                         &plan->weight_slabs[d][s]);
            if (st != VELRI_OK) {
                return st;
            }
        }
    }
    return VELRI_OK;
}

static velri_status pack_state_and_io_arenas(velri_plan *plan) {
    velri_context *ctx = plan->ctx;
    for (uint32_t d = 0U; d < ctx->num_devices; ++d) {
        bool is_host = ctx->devices[d].caps.is_host_memory;
        size_t st_off = 0U, st_align = effective_align(ctx, d, 0U);
        size_t io_off = 0U, io_align = effective_align(ctx, d, 0U);

        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            velri_plan_storage *ps = &plan->storages[i];
            if (ps->device_id != d) {
                continue;
            }
            if (ps->kind == VELRI_STORAGE_STATE) {
                size_t a = effective_align(ctx, d, ps->align);
                if (a > st_align) st_align = a;
                size_t off = velri_align_up(st_off, a);
                if (off == SIZE_MAX || off > SIZE_MAX - ps->byte_size) {
                    return VELRI_ERR_OVERFLOW;
                }
                ps->arena_kind = VELRI_ARENA_STATE;
                ps->arena_offset = off;
                st_off = off + ps->byte_size;
            } else if (!is_host &&
                       (ps->kind == VELRI_STORAGE_INPUT || ps->kind == VELRI_STORAGE_OUTPUT)) {
                size_t a = effective_align(ctx, d, ps->align);
                if (a > io_align) io_align = a;
                size_t off = velri_align_up(io_off, a);
                if (off == SIZE_MAX || off > SIZE_MAX - ps->byte_size) {
                    return VELRI_ERR_OVERFLOW;
                }
                ps->arena_kind = VELRI_ARENA_IO;
                ps->arena_offset = off;
                io_off = off + ps->byte_size;
            }
        }

        if (st_off > 0U) {
            st_off = velri_align_up(st_off, st_align);
            if (st_off == SIZE_MAX) return VELRI_ERR_OVERFLOW;
        }
        if (io_off > 0U) {
            io_off = velri_align_up(io_off, io_align);
            if (io_off == SIZE_MAX) return VELRI_ERR_OVERFLOW;
        }
        plan->state_arena_bytes[d] = st_off;
        plan->state_arena_align[d] = st_align;
        plan->io_arena_bytes[d] = io_off;
        plan->io_arena_align[d] = io_align;
    }
    return VELRI_OK;
}

typedef struct {
    size_t start;
    size_t end;
    uint32_t first_step;
    uint32_t last_step;
} placed_interval;

static velri_status pack_scratch_arenas(velri_plan *plan) {
    uint32_t ns = plan->num_storages;
    if (ns == 0U) {
        return VELRI_OK;
    }

    uint32_t *first_use = (uint32_t *)malloc((size_t)ns * sizeof(uint32_t));
    uint32_t *last_use = (uint32_t *)malloc((size_t)ns * sizeof(uint32_t));
    uint32_t *order = (uint32_t *)malloc((size_t)ns * sizeof(uint32_t));
    placed_interval *placed = (placed_interval *)malloc((size_t)ns * sizeof(placed_interval));
    if (!first_use || !last_use || !order || !placed) {
        free(first_use);
        free(last_use);
        free(order);
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
                if (!step->u.op.inputs[i].present) continue;
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

    velri_status st = VELRI_OK;
    for (uint32_t d = 0U; d < plan->ctx->num_devices && st == VELRI_OK; ++d) {
        uint32_t n_cand = 0U;
        for (uint32_t i = 0U; i < ns; ++i) {
            const velri_plan_storage *ps = &plan->storages[i];
            if (ps->kind == VELRI_STORAGE_SCRATCH && ps->device_id == d &&
                first_use[i] != UINT32_MAX) {
                order[n_cand++] = i;
            }
        }

        /* Sort descending by byte_size, then descending by lifetime span for tighter packing. */
        for (uint32_t a = 1U; a < n_cand; ++a) {
            uint32_t key = order[a];
            size_t key_sz = plan->storages[key].byte_size;
            uint32_t key_span = last_use[key] - first_use[key];
            uint32_t b = a;
            while (b > 0U) {
                uint32_t prev = order[b - 1U];
                size_t prev_sz = plan->storages[prev].byte_size;
                uint32_t prev_span = last_use[prev] - first_use[prev];
                if (prev_sz > key_sz || (prev_sz == key_sz && prev_span >= key_span)) {
                    break;
                }
                order[b] = prev;
                b--;
            }
            order[b] = key;
        }

        uint32_t n_placed = 0U;
        size_t base_align = effective_align(plan->ctx, d, 0U);
        size_t max_align = base_align;
        size_t arena_size = 0U;

        for (uint32_t c = 0U; c < n_cand; ++c) {
            uint32_t i = order[c];
            velri_plan_storage *ps = &plan->storages[i];

            size_t align = ps->align < base_align ? base_align : ps->align;
            if (align > max_align) {
                max_align = align;
            }
            size_t sz = velri_align_up(ps->byte_size, align);
            if (sz == SIZE_MAX) {
                st = VELRI_ERR_OVERFLOW;
                break;
            }

            size_t cand = 0U;
            bool conflict = true;
            while (conflict) {
                if (cand == SIZE_MAX || cand > SIZE_MAX - sz) {
                    st = VELRI_ERR_OVERFLOW;
                    break;
                }
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
            if (st != VELRI_OK) {
                break;
            }

            ps->arena_kind = VELRI_ARENA_SCRATCH;
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

        if (arena_size > 0U) {
            arena_size = velri_align_up(arena_size, max_align);
            if (arena_size == SIZE_MAX) {
                st = VELRI_ERR_OVERFLOW;
                break;
            }
        }
        plan->scratch_arena_bytes[d] = arena_size;
        plan->scratch_arena_align[d] = max_align;
    }

    free(first_use);
    free(last_use);
    free(order);
    free(placed);
    return st;
}

static size_t sat_add(size_t a, size_t b) { return (a > SIZE_MAX - b) ? SIZE_MAX : (a + b); }

static void compute_memory_report(velri_plan *plan) {
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
        switch (ps->kind) {
        case VELRI_STORAGE_WEIGHT:
            dr->weight_bytes = sat_add(dr->weight_bytes, ps->byte_size);
            break;
        case VELRI_STORAGE_INPUT:
        case VELRI_STORAGE_OUTPUT:
            dr->input_output_bytes = sat_add(dr->input_output_bytes, ps->byte_size);
            break;
        case VELRI_STORAGE_STATE:
            dr->state_bytes = sat_add(dr->state_bytes, ps->byte_size);
            break;
        case VELRI_STORAGE_SCRATCH:
            dr->scratch_unpacked_bytes = sat_add(dr->scratch_unpacked_bytes, ps->byte_size);
            break;
        }
    }

    for (uint32_t d = 0U; d < rep->num_devices; ++d) {
        velri_device_memory_report *dr = &rep->devices[d];
        size_t ovh = 0U;
        if (!plan->ctx->devices[d].caps.is_host_memory) {
            size_t packed_w = 0U;
            for (uint32_t s = 0U; s < plan->num_weight_slabs[d]; ++s) {
                packed_w = sat_add(packed_w, plan->weight_slab_bytes[d][s]);
            }
            if (packed_w > dr->weight_bytes) {
                ovh = sat_add(ovh, packed_w - dr->weight_bytes);
            }
            if (plan->io_arena_bytes[d] > dr->input_output_bytes) {
                ovh = sat_add(ovh, plan->io_arena_bytes[d] - dr->input_output_bytes);
            }
        }
        if (plan->state_arena_bytes[d] > dr->state_bytes) {
            ovh = sat_add(ovh, plan->state_arena_bytes[d] - dr->state_bytes);
        }
        dr->allocator_overhead_bytes = ovh;
    }

    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        if (plan->steps[s].kind == VELRI_STEP_RAM_TRANSFER) {
            rep->num_ram_transfers++;
        }
    }

    size_t total = rep->ram_staging_bytes;
    for (uint32_t d = 0U; d < rep->num_devices; ++d) {
        const velri_device_memory_report *dr = &rep->devices[d];
        total = sat_add(total, dr->weight_bytes);
        total = sat_add(total, dr->input_output_bytes);
        total = sat_add(total, dr->state_bytes);
        total = sat_add(total, dr->scratch_packed_bytes);
        total = sat_add(total, dr->allocator_overhead_bytes);
    }
    rep->total_physical_bytes = total;
}

velri_status velri_plan_prepare(velri_context *ctx, const velri_graph *graph,
                                const velri_plan_options *opts, velri_plan **out_plan) {
    if (!ctx || !graph || !out_plan) {
        return VELRI_ERR_INVALID;
    }
    *out_plan = nullptr;

    bool allow_cpu_fallback = opts ? opts->allow_cpu_fallback : false;

    for (uint32_t i = 0U; i < graph->num_storages; ++i) {
        if (graph->storages[i].device_id >= ctx->num_devices) {
            return VELRI_ERR_NOT_FOUND;
        }
    }

    velri_plan *plan = (velri_plan *)calloc(1U, sizeof(*plan));
    if (!plan) {
        return VELRI_ERR_NOMEM;
    }
    atomic_init(&plan->refcount, 1U);
    velri_context_retain(ctx);
    plan->ctx = ctx;
    plan->num_user_storages = graph->num_storages;

    plan_builder pb = {.plan = plan, .storage_cap = 0U, .step_cap = 0U};

    if (graph->num_storages > 0U) {
        if (!velri_vec_grow((void **)&plan->storages, &pb.storage_cap, graph->num_storages,
                            sizeof(velri_plan_storage))) {
            velri_plan_release(plan);
            return VELRI_ERR_NOMEM;
        }
        plan->weight_bindings = (velri_plan_weight_binding *)calloc(
            graph->num_storages, sizeof(*plan->weight_bindings));
        if (!plan->weight_bindings) {
            velri_plan_release(plan);
            return VELRI_ERR_NOMEM;
        }
    }

    size_t n_user = graph->num_storages ? (size_t)graph->num_storages : 1U;
    uint32_t *owner_idx = (uint32_t *)malloc(n_user * sizeof(uint32_t));
    velri_device_id *owner_dev = (velri_device_id *)malloc(n_user * sizeof(velri_device_id));
    uint32_t *valid_copy = (uint32_t *)malloc(n_user * VELRI_MAX_DEVICES * sizeof(uint32_t));
    if (!owner_idx || !owner_dev || !valid_copy) {
        free(owner_idx);
        free(owner_dev);
        free(valid_copy);
        velri_plan_release(plan);
        return VELRI_ERR_NOMEM;
    }

    for (uint32_t i = 0U; i < graph->num_storages; ++i) {
        const velri_storage_desc *sd = &graph->storages[i];
        velri_plan_storage *ps = &plan->storages[i];
        memset(ps, 0, sizeof(*ps));
        ps->kind = sd->kind;
        ps->dtype = sd->dtype;
        ps->device_id = sd->device_id;
        ps->byte_size = sd->byte_size;
        ps->align = sd->align;
        owner_idx[i] = i;
        owner_dev[i] = sd->device_id;
        for (uint32_t d = 0U; d < VELRI_MAX_DEVICES; ++d) {
            valid_copy[(size_t)i * VELRI_MAX_DEVICES + d] = UINT32_MAX;
        }
        valid_copy[(size_t)i * VELRI_MAX_DEVICES + sd->device_id] = i;
    }
    plan->num_storages = graph->num_storages;

    /* Pack & allocate shared non-host WEIGHT slabs, plus STATE/IO arena layouts. */
    velri_status st = pack_weight_slabs(plan);
    if (st == VELRI_OK) {
        st = pack_state_and_io_arenas(plan);
    }

    /* 1-step-adjacent offloaded WEIGHT shadow cache: deduplicates both intra-op subviews
     * (e.g. W_K & W_V inside latent_attention) and immediately consecutive ops on the same
     * device (e.g. q_linear -> latent_attention reading fused qkv_w), without pinning
     * non-adjacent tied weights (e.g. wte / lm_head) across the graph. */
    velri_device_id prev_w_dev = VELRI_INVALID_ID;
    uint32_t prev_w_cnt = 0U;
    velri_storage_id prev_w_sids[VELRI_OP_MAX_INPUTS];
    uint32_t prev_w_idxs[VELRI_OP_MAX_INPUTS];

    /* Lower each operation node into device-verified steps + cached explicit transfers. */
    for (uint32_t n = 0U; n < graph->num_nodes && st == VELRI_OK; ++n) {
        const velri_graph_node *node = &graph->nodes[n];
        velri_resolved_op resolved;
        velri_graph_resolve_node(graph, &node->desc, &resolved);

        velri_device_id chosen_dev = 0U;
        st = select_node_device(ctx, &resolved, node->device_id, allow_cpu_fallback, &chosen_dev);
        if (st != VELRI_OK) {
            break;
        }

        velri_step_arg in_args[VELRI_OP_MAX_INPUTS];
        velri_step_arg out_args[VELRI_OP_MAX_OUTPUTS];
        memset(in_args, 0, sizeof(in_args));
        memset(out_args, 0, sizeof(out_args));

        uint32_t cur_w_cnt = 0U;
        velri_storage_id cur_w_sids[VELRI_OP_MAX_INPUTS];
        uint32_t cur_w_idxs[VELRI_OP_MAX_INPUTS];

        /* 1. Ensure every present input has a valid copy on `chosen_dev` (cached per (sid, dev)). */
        for (uint32_t i = 0U; i < resolved.num_inputs; ++i) {
            if (!resolved.input_present[i]) {
                continue;
            }
            velri_storage_id sid = resolved.inputs[i].storage_id;
            uint32_t use_idx = valid_copy[(size_t)sid * VELRI_MAX_DEVICES + chosen_dev];
            if (use_idx == UINT32_MAX && plan->storages[sid].kind == VELRI_STORAGE_WEIGHT) {
                for (uint32_t k = 0U; k < cur_w_cnt; ++k) {
                    if (cur_w_sids[k] == sid) {
                        use_idx = cur_w_idxs[k];
                        break;
                    }
                }
                if (use_idx == UINT32_MAX && prev_w_dev == chosen_dev) {
                    for (uint32_t k = 0U; k < prev_w_cnt; ++k) {
                        if (prev_w_sids[k] == sid) {
                            use_idx = prev_w_idxs[k];
                            cur_w_sids[cur_w_cnt] = sid;
                            cur_w_idxs[cur_w_cnt] = use_idx;
                            cur_w_cnt++;
                            break;
                        }
                    }
                }
            }
            if (use_idx == UINT32_MAX) {
                st = add_shadow_storage(&pb, chosen_dev, plan->storages[sid].dtype,
                                        plan->storages[sid].byte_size, plan->storages[sid].align,
                                        &use_idx);
                if (st != VELRI_OK) break;
                st = emit_ram_transfer(&pb, owner_dev[sid], owner_idx[sid], chosen_dev, use_idx,
                                       plan->storages[sid].byte_size);
                if (st != VELRI_OK) break;
                if (plan->storages[sid].kind != VELRI_STORAGE_WEIGHT) {
                    valid_copy[(size_t)sid * VELRI_MAX_DEVICES + chosen_dev] = use_idx;
                } else {
                    cur_w_sids[cur_w_cnt] = sid;
                    cur_w_idxs[cur_w_cnt] = use_idx;
                    cur_w_cnt++;
                }
            }
            in_args[i].present = true;
            in_args[i].plan_storage_idx = use_idx;
            in_args[i].view_byte_offset = resolved.inputs[i].byte_offset;
        }
        if (st != VELRI_OK) break;

        /* 2. Prepare output targets on `chosen_dev` (preserving untouched bytes on RMW/subviews
         *    and deduplicating multiple output views targeting the same storage_id). */
        uint32_t out_sids[VELRI_OP_MAX_OUTPUTS];
        uint32_t out_dsts[VELRI_OP_MAX_OUTPUTS];
        for (uint32_t o = 0U; o < resolved.num_outputs; ++o) {
            velri_storage_id sid = resolved.outputs[o].storage_id;
            out_sids[o] = sid;

            uint32_t dst_idx = UINT32_MAX;
            for (uint32_t prev = 0U; prev < o; ++prev) {
                if (out_sids[prev] == sid) {
                    dst_idx = out_dsts[prev];
                    break;
                }
            }

            if (dst_idx == UINT32_MAX) {
                bool is_rmw = (resolved.kind == VELRI_OP_FUSED_ADD_RMSNORM && o == 0U) ||
                              (resolved.kind == VELRI_OP_ATTENTION && (o == 1U || o == 2U)) ||
                              (resolved.kind == VELRI_OP_LATENT_ATTENTION && o == 1U);
                bool need_preserve =
                    is_rmw || !view_covers_full_storage(&resolved.outputs[o],
                                                        plan->storages[sid].byte_size);
                velri_device_id home_dev = plan->storages[sid].device_id;
                uint32_t existing = valid_copy[(size_t)sid * VELRI_MAX_DEVICES + chosen_dev];

                if (existing != UINT32_MAX) {
                    dst_idx = existing;
                } else if (home_dev == chosen_dev) {
                    dst_idx = sid;
                    if (need_preserve) {
                        st = emit_ram_transfer(&pb, owner_dev[sid], owner_idx[sid], chosen_dev,
                                               dst_idx, plan->storages[sid].byte_size);
                        if (st != VELRI_OK) break;
                    }
                } else {
                    st = add_shadow_storage(&pb, chosen_dev, plan->storages[sid].dtype,
                                            plan->storages[sid].byte_size,
                                            plan->storages[sid].align, &dst_idx);
                    if (st != VELRI_OK) break;
                    if (need_preserve) {
                        st = emit_ram_transfer(&pb, owner_dev[sid], owner_idx[sid], chosen_dev,
                                               dst_idx, plan->storages[sid].byte_size);
                        if (st != VELRI_OK) break;
                    }
                }
            }

            out_dsts[o] = dst_idx;
            out_args[o].present = true;
            out_args[o].plan_storage_idx = dst_idx;
            out_args[o].view_byte_offset = resolved.outputs[o].byte_offset;
        }
        if (st != VELRI_OK) break;

        /* 3. Invalidate stale copies on other devices now that all inputs/outputs are resolved. */
        for (uint32_t o = 0U; o < resolved.num_outputs; ++o) {
            velri_storage_id sid = out_sids[o];
            for (uint32_t d = 0U; d < VELRI_MAX_DEVICES; ++d) {
                valid_copy[(size_t)sid * VELRI_MAX_DEVICES + d] = UINT32_MAX;
            }
            valid_copy[(size_t)sid * VELRI_MAX_DEVICES + chosen_dev] = out_dsts[o];
            owner_dev[sid] = chosen_dev;
            owner_idx[sid] = out_dsts[o];
        }

        prev_w_dev = chosen_dev;
        prev_w_cnt = cur_w_cnt;
        for (uint32_t k = 0U; k < cur_w_cnt; ++k) {
            prev_w_sids[k] = cur_w_sids[k];
            prev_w_idxs[k] = cur_w_idxs[k];
        }

        /* 4. Optional backend preparation hook. */
        velri_device_entry *de = &ctx->devices[chosen_dev];
        velri_prepared_op *prepared = nullptr;
        if (de->vt->prepare_op) {
            st = de->vt->prepare_op(de->dev, &resolved, &prepared);
            if (st != VELRI_OK) break;
        }

        if (plan->num_steps == UINT32_MAX ||
            !velri_vec_grow((void **)&plan->steps, &pb.step_cap,
                            plan->num_steps + 1U, sizeof(velri_launch_step))) {
            if (prepared && de->vt->destroy_prepared_op) {
                de->vt->destroy_prepared_op(de->dev, prepared);
            }
            st = VELRI_ERR_NOMEM;
            break;
        }

        velri_launch_step *step = &plan->steps[plan->num_steps++];
        memset(step, 0, sizeof(*step));
        step->kind = VELRI_STEP_OP;
        step->u.op.device_id = chosen_dev;
        step->u.op.resolved = resolved;
        step->u.op.prepared = prepared;
        step->u.op.num_inputs = resolved.num_inputs;
        step->u.op.num_outputs = resolved.num_outputs;
        memcpy(step->u.op.inputs, in_args, sizeof(in_args));
        memcpy(step->u.op.outputs, out_args, sizeof(out_args));
    }

    /* Transfer any OUTPUT or STATE storage back to its declared home device if produced elsewhere. */
    if (st == VELRI_OK) {
        for (uint32_t sid = 0U; sid < graph->num_storages; ++sid) {
            velri_storage_kind kind = plan->storages[sid].kind;
            if ((kind == VELRI_STORAGE_OUTPUT || kind == VELRI_STORAGE_STATE) &&
                owner_dev[sid] != plan->storages[sid].device_id) {
                st = emit_ram_transfer(&pb, owner_dev[sid], owner_idx[sid],
                                       plan->storages[sid].device_id, sid,
                                       plan->storages[sid].byte_size);
                if (st != VELRI_OK) break;
            }
        }
    }

    free(owner_idx);
    free(owner_dev);
    free(valid_copy);

    if (st == VELRI_OK) {
        st = pack_scratch_arenas(plan);
    }
    if (st != VELRI_OK) {
        velri_plan_release(plan);
        return st;
    }

    compute_memory_report(plan);
    *out_plan = plan;
    return VELRI_OK;
}

void velri_plan_retain(velri_plan *plan) {
    if (plan) {
        atomic_fetch_add_explicit(&plan->refcount, 1U, memory_order_relaxed);
    }
}

void velri_plan_release(velri_plan *plan) {
    if (!plan) {
        return;
    }
    if (atomic_fetch_sub_explicit(&plan->refcount, 1U, memory_order_acq_rel) != 1U) {
        return;
    }
    if (plan->ctx) {
        if (plan->steps) {
            for (uint32_t i = 0U; i < plan->num_steps; ++i) {
                velri_launch_step *step = &plan->steps[i];
                if (step->kind == VELRI_STEP_OP && step->u.op.prepared) {
                    velri_device_entry *de = &plan->ctx->devices[step->u.op.device_id];
                    if (de->vt->destroy_prepared_op) {
                        de->vt->destroy_prepared_op(de->dev, step->u.op.prepared);
                    }
                }
            }
        }
        for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
            velri_device_entry *de = &plan->ctx->devices[d];
            for (uint32_t s = 0U; s < plan->num_weight_slabs[d]; ++s) {
                if (plan->weight_slabs[d][s]) {
                    de->vt->free(de->dev, plan->weight_slabs[d][s]);
                }
            }
        }
        velri_context_release(plan->ctx);
    }
    free(plan->weight_bindings);
    free(plan->steps);
    free(plan->storages);
    free(plan);
}

void velri_plan_destroy(velri_plan *plan) { velri_plan_release(plan); }

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
