#include "core_internal.h"
#include <string.h>

/*
 * Execution Slots & Runtime Launch-Graph Executor (`velri-lib/docs/EN/execution.md`).
 *
 * Key invariants:
 *   - Immutable `VELRI_STORAGE_WEIGHT` buffers live once in per-device weight slabs on `velri_plan`
 *     (or zero-copy host pointers on CPU) and are shared read-only across all concurrent slots.
 *   - `VELRI_STORAGE_STATE` lives in one contiguous `state_arena` per device per slot and remains
 *     resident across steps with 0 B shadow RAM required (optional host binding for explicit sync).
 *   - `VELRI_STORAGE_INPUT`/`OUTPUT` on non-host devices share one contiguous `io_arena` per slot.
 *   - Direct CPU <-> non-host transfers bypass `ram_staging` completely (0 B staging RAM).
 *   - Consecutive ops targeting the same device are batched via `submit_batch` (0 B hot-path allocs).
 *   - Dynamic bounds (`seq_len_override`, `pos_offset_override`, `kv_len_override`) are validated
 *     strictly before launching any device work.
 */

typedef struct {
    void *host_ptr;
    size_t bound_bytes;
    bool is_bound;
    bool state_uploaded; /* Initial upload flag for host-bound non-host STATE */
} slot_storage_binding;

struct velri_slot {
    velri_plan *plan;
    void *scratch_arenas[VELRI_MAX_DEVICES];
    void *state_arenas[VELRI_MAX_DEVICES];
    void *io_arenas[VELRI_MAX_DEVICES];
    void *ram_staging;
    slot_storage_binding *user_bindings;
    bool needs_reset;
    uint64_t last_run_allocs;
};

static velri_buffer_binding resolve_storage_binding(const velri_slot *slot,
                                                    uint32_t plan_storage_idx,
                                                    size_t view_byte_offset) {
    const velri_plan *plan = slot->plan;
    const velri_plan_storage *ps = &plan->storages[plan_storage_idx];
    const velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    velri_buffer_binding b;
    memset(&b, 0, sizeof(b));

    switch (ps->arena_kind) {
    case VELRI_ARENA_SCRATCH:
        b.mem_handle = slot->scratch_arenas[ps->device_id];
        b.byte_offset = ps->arena_offset + view_byte_offset;
        b.byte_size = plan->scratch_arena_bytes[ps->device_id];
        b.is_host_ptr = de->caps.is_host_memory;
        return b;

    case VELRI_ARENA_WEIGHT:
        b.mem_handle = plan->weight_slabs[ps->device_id][ps->slab_idx];
        b.byte_offset = ps->arena_offset + view_byte_offset;
        b.byte_size = plan->weight_slab_bytes[ps->device_id][ps->slab_idx];
        b.is_host_ptr = false;
        return b;

    case VELRI_ARENA_STATE:
        b.mem_handle = slot->state_arenas[ps->device_id];
        b.byte_offset = ps->arena_offset + view_byte_offset;
        b.byte_size = plan->state_arena_bytes[ps->device_id];
        b.is_host_ptr = de->caps.is_host_memory;
        return b;

    case VELRI_ARENA_IO:
        b.mem_handle = slot->io_arenas[ps->device_id];
        b.byte_offset = ps->arena_offset + view_byte_offset;
        b.byte_size = plan->io_arena_bytes[ps->device_id];
        b.is_host_ptr = false;
        return b;

    case VELRI_ARENA_NONE:
    default:
        break;
    }

    if (ps->kind == VELRI_STORAGE_WEIGHT) {
        const velri_plan_weight_binding *wb = &plan->weight_bindings[plan_storage_idx];
        b.mem_handle = (void *)wb->host_ptr;
        b.byte_offset = view_byte_offset;
        b.byte_size = wb->bound_bytes;
        b.is_host_ptr = true;
        return b;
    }

    const slot_storage_binding *sb = &slot->user_bindings[plan_storage_idx];
    b.mem_handle = sb->host_ptr;
    b.byte_offset = view_byte_offset;
    b.byte_size = sb->bound_bytes;
    b.is_host_ptr = true;
    return b;
}

static velri_status zero_device_arena(velri_context *ctx, velri_device_id d, void *handle,
                                      size_t sz) {
    if (!handle || sz == 0U) {
        return VELRI_OK;
    }
    velri_device_entry *de = &ctx->devices[d];
    if (de->caps.is_host_memory) {
        memset(handle, 0, sz);
        return VELRI_OK;
    }
    if (de->vt->zero_mem) {
        return de->vt->zero_mem(de->dev, handle, sz);
    }
    /* Portable 0-alloc fallback for non-host backends that omit optional `zero_mem`. */
    static const uint8_t k_zero_chunk[65536] = {0};
    size_t off = 0U;
    while (off < sz) {
        size_t chunk = sz - off;
        if (chunk > sizeof(k_zero_chunk)) {
            chunk = sizeof(k_zero_chunk);
        }
        velri_buffer_binding dst = {
            .mem_handle = handle,
            .byte_offset = off,
            .byte_size = sz,
            .is_host_ptr = false,
        };
        velri_status st = de->vt->copy_from_ram(de->dev, &dst, k_zero_chunk, chunk);
        if (st != VELRI_OK) {
            return st;
        }
        off += chunk;
    }
    return VELRI_OK;
}

velri_status velri_plan_bind_weight_host(velri_plan *plan, velri_storage_id storage_id,
                                         const void *host_ptr, size_t byte_size) {
    if (!plan || !host_ptr || storage_id >= plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &plan->storages[storage_id];
    if (ps->kind != VELRI_STORAGE_WEIGHT || byte_size < ps->byte_size) {
        return VELRI_ERR_INVALID;
    }
    size_t req_align = velri_dtype_align(ps->dtype);
    if (((uintptr_t)host_ptr % req_align) != 0U) {
        return VELRI_ERR_INVALID;
    }

    velri_plan_weight_binding *wb = &plan->weight_bindings[storage_id];
    wb->host_ptr = host_ptr;
    wb->bound_bytes = byte_size;

    velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    if (!de->caps.is_host_memory) {
        velri_buffer_binding dst = {
            .mem_handle = plan->weight_slabs[ps->device_id][ps->slab_idx],
            .byte_offset = ps->arena_offset,
            .byte_size = plan->weight_slab_bytes[ps->device_id][ps->slab_idx],
            .is_host_ptr = false,
        };
        velri_status st = de->vt->copy_from_ram(de->dev, &dst, host_ptr, ps->byte_size);
        if (st != VELRI_OK) {
            wb->is_bound = false;
            return st;
        }
    }
    wb->is_bound = true;
    return VELRI_OK;
}

velri_status velri_slot_create(velri_plan *plan, velri_slot **out_slot) {
    if (!plan || !out_slot) {
        return VELRI_ERR_INVALID;
    }
    *out_slot = nullptr;

    velri_slot *slot = (velri_slot *)calloc(1U, sizeof(*slot));
    if (!slot) {
        return VELRI_ERR_NOMEM;
    }
    velri_plan_retain(plan);
    slot->plan = plan;

    if (plan->num_user_storages > 0U) {
        slot->user_bindings =
            (slot_storage_binding *)calloc(plan->num_user_storages, sizeof(*slot->user_bindings));
        if (!slot->user_bindings) {
            velri_slot_destroy(slot);
            return VELRI_ERR_NOMEM;
        }
    }

    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        if (plan->scratch_arena_bytes[d] > 0U) {
            velri_status st = velri_device_alloc_tracked(plan->ctx, d, plan->scratch_arena_bytes[d],
                                                         plan->scratch_arena_align[d],
                                                         &slot->scratch_arenas[d]);
            if (st != VELRI_OK) {
                velri_slot_destroy(slot);
                return st;
            }
        }
        if (plan->state_arena_bytes[d] > 0U) {
            velri_status st = velri_device_alloc_tracked(plan->ctx, d, plan->state_arena_bytes[d],
                                                         plan->state_arena_align[d],
                                                         &slot->state_arenas[d]);
            if (st != VELRI_OK) {
                velri_slot_destroy(slot);
                return st;
            }
            st = zero_device_arena(plan->ctx, d, slot->state_arenas[d], plan->state_arena_bytes[d]);
            if (st != VELRI_OK) {
                velri_slot_destroy(slot);
                return st;
            }
        }
        if (plan->io_arena_bytes[d] > 0U) {
            velri_status st = velri_device_alloc_tracked(plan->ctx, d, plan->io_arena_bytes[d],
                                                         plan->io_arena_align[d],
                                                         &slot->io_arenas[d]);
            if (st != VELRI_OK) {
                velri_slot_destroy(slot);
                return st;
            }
        }
    }

    if (plan->ram_staging_bytes > 0U) {
        slot->ram_staging = velri_os_aligned_alloc(64U, plan->ram_staging_bytes);
        if (!slot->ram_staging) {
            velri_slot_destroy(slot);
            return VELRI_ERR_NOMEM;
        }
        memset(slot->ram_staging, 0, plan->ram_staging_bytes);
    }

    *out_slot = slot;
    return VELRI_OK;
}

void velri_slot_destroy(velri_slot *slot) {
    if (!slot) {
        return;
    }
    velri_plan *plan = slot->plan;
    if (plan && plan->ctx) {
        for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
            velri_device_entry *de = &plan->ctx->devices[d];
            if (slot->io_arenas[d]) de->vt->free(de->dev, slot->io_arenas[d]);
            if (slot->state_arenas[d]) de->vt->free(de->dev, slot->state_arenas[d]);
            if (slot->scratch_arenas[d]) de->vt->free(de->dev, slot->scratch_arenas[d]);
        }
    }
    velri_os_aligned_free(slot->ram_staging);
    free(slot->user_bindings);
    free(slot);
    velri_plan_release(plan);
}

velri_status velri_slot_bind_host(velri_slot *slot, velri_storage_id storage_id, void *host_ptr,
                                  size_t byte_size) {
    if (!slot || !host_ptr || storage_id >= slot->plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &slot->plan->storages[storage_id];
    if (ps->kind == VELRI_STORAGE_SCRATCH || ps->kind == VELRI_STORAGE_WEIGHT ||
        byte_size < ps->byte_size) {
        return VELRI_ERR_INVALID;
    }
    size_t req_align = velri_dtype_align(ps->dtype);
    if (((uintptr_t)host_ptr % req_align) != 0U) {
        return VELRI_ERR_INVALID;
    }

    slot_storage_binding *sb = &slot->user_bindings[storage_id];
    sb->host_ptr = host_ptr;
    sb->bound_bytes = byte_size;
    sb->is_bound = true;
    sb->state_uploaded = false;
    return VELRI_OK;
}

velri_status velri_slot_sync_state_from_host(velri_slot *slot, velri_storage_id storage_id) {
    if (!slot || storage_id >= slot->plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &slot->plan->storages[storage_id];
    slot_storage_binding *sb = &slot->user_bindings[storage_id];
    if (ps->kind != VELRI_STORAGE_STATE || !sb->is_bound) {
        return VELRI_ERR_INVALID;
    }
    velri_device_entry *de = &slot->plan->ctx->devices[ps->device_id];
    velri_buffer_binding dst = resolve_storage_binding(slot, storage_id, 0U);
    if (de->caps.is_host_memory) {
        if (ps->byte_size > 0U) {
            memcpy((uint8_t *)dst.mem_handle + dst.byte_offset, sb->host_ptr, ps->byte_size);
        }
        sb->state_uploaded = true;
        return VELRI_OK;
    }
    velri_status st = de->vt->copy_from_ram(de->dev, &dst, sb->host_ptr, ps->byte_size);
    if (st == VELRI_OK) {
        sb->state_uploaded = true;
    }
    return st;
}

velri_status velri_slot_sync_state_to_host(velri_slot *slot, velri_storage_id storage_id) {
    if (!slot || storage_id >= slot->plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &slot->plan->storages[storage_id];
    slot_storage_binding *sb = &slot->user_bindings[storage_id];
    if (ps->kind != VELRI_STORAGE_STATE || !sb->is_bound) {
        return VELRI_ERR_INVALID;
    }
    velri_device_entry *de = &slot->plan->ctx->devices[ps->device_id];
    velri_buffer_binding src = resolve_storage_binding(slot, storage_id, 0U);
    if (de->caps.is_host_memory) {
        if (ps->byte_size > 0U) {
            memcpy(sb->host_ptr, (const uint8_t *)src.mem_handle + src.byte_offset, ps->byte_size);
        }
        return VELRI_OK;
    }
    return de->vt->copy_to_ram(de->dev, sb->host_ptr, &src, ps->byte_size);
}

static velri_status validate_dynamic_params(const velri_plan *plan,
                                            const velri_step_params *params) {
    if (!params) {
        return VELRI_OK;
    }
    if (params->has_pos_offset && params->pos_offset_override > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    if (params->has_seq_len) {
        if (params->seq_len_override == 0U) {
            return VELRI_ERR_INVALID;
        }
        if (params->seq_len_override > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    if (params->has_kv_len && params->kv_len_override > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }

    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        const velri_launch_step *step = &plan->steps[s];
        if (step->kind != VELRI_STEP_OP) {
            continue;
        }
        const velri_resolved_op *op = &step->u.op.resolved;

        /* Strictly validate seq_len_override <= static_rows for EVERY dynamic-sequence op. */
        if (params->has_seq_len && velri_op_is_dynamic_seq(op)) {
            size_t static_rows = (op->kind == VELRI_OP_EMBEDDING) ? op->outputs[0].dims[0]
                                                                  : op->inputs[0].dims[0];
            if (params->seq_len_override > static_rows) {
                return VELRI_ERR_INVALID;
            }
        }

        if (op->kind == VELRI_OP_ROPE && (params->has_pos_offset || params->has_seq_len)) {
            size_t active_t = velri_effective_rows(op, op->inputs[0].dims[0], params);
            size_t pos = params->has_pos_offset ? params->pos_offset_override
                                                : op->raw.attr.rope.pos_offset;
            size_t end_pos = 0U, need = 0U;
            if (velri_add_overflow_size(pos, active_t, &end_pos) ||
                velri_mul_overflow_size(end_pos, op->raw.attr.rope.rotary_dim / 2U, &need)) {
                return VELRI_ERR_OVERFLOW;
            }
            size_t ct_n = 0U, st_n = 0U;
            if (velri_tensor_view_num_elements(&op->inputs[1], &ct_n) != VELRI_OK ||
                velri_tensor_view_num_elements(&op->inputs[2], &st_n) != VELRI_OK ||
                ct_n < need || st_n < need) {
                return VELRI_ERR_INVALID;
            }
        } else if (op->kind == VELRI_OP_EMBEDDING) {
            if (!(op->num_inputs >= 2U && op->input_present[1])) {
                size_t active_t = velri_effective_rows(op, op->outputs[0].dims[0], params);
                size_t pos = params->has_pos_offset ? params->pos_offset_override
                                                    : op->raw.attr.embedding.pos_offset;
                size_t end_pos = 0U;
                if (velri_add_overflow_size(pos, active_t, &end_pos)) {
                    return VELRI_ERR_OVERFLOW;
                }
                if (end_pos > op->inputs[0].dims[0]) {
                    return VELRI_ERR_INVALID;
                }
            }
        } else if (op->kind == VELRI_OP_ATTENTION || op->kind == VELRI_OP_LATENT_ATTENTION) {
            size_t active_t = velri_effective_rows(op, op->inputs[0].dims[0], params);
            size_t kv_pos = params->has_kv_len
                                ? params->kv_len_override
                                : (params->has_pos_offset ? params->pos_offset_override : 0U);
            size_t kv_end = 0U;
            if (velri_add_overflow_size(kv_pos, active_t, &kv_end)) {
                return VELRI_ERR_OVERFLOW;
            }
            size_t max_ctx = (op->kind == VELRI_OP_ATTENTION) ? op->outputs[1].dims[1]
                                                              : op->outputs[1].dims[0];
            if (kv_end > max_ctx) {
                return VELRI_ERR_INVALID;
            }
        }
    }
    return VELRI_OK;
}

static velri_status run_ram_transfer(velri_slot *slot, const velri_launch_step *step) {
    const velri_plan *plan = slot->plan;
    velri_device_entry *src_de = &plan->ctx->devices[step->u.transfer.src_device_id];
    velri_device_entry *dst_de = &plan->ctx->devices[step->u.transfer.dst_device_id];

    velri_buffer_binding src_b =
        resolve_storage_binding(slot, step->u.transfer.src_plan_storage_idx, 0U);
    velri_buffer_binding dst_b =
        resolve_storage_binding(slot, step->u.transfer.dst_plan_storage_idx, 0U);
    size_t n = step->u.transfer.byte_size;

    /* Direct zero-staging transfer when either side is already host memory. */
    if (src_b.is_host_ptr && dst_b.is_host_ptr) {
        if (n > 0U) {
            memcpy((uint8_t *)dst_b.mem_handle + dst_b.byte_offset,
                   (const uint8_t *)src_b.mem_handle + src_b.byte_offset, n);
        }
        return VELRI_OK;
    }
    if (src_b.is_host_ptr) {
        const void *src_ptr = (const uint8_t *)src_b.mem_handle + src_b.byte_offset;
        return dst_de->vt->copy_from_ram(dst_de->dev, &dst_b, src_ptr, n);
    }
    if (dst_b.is_host_ptr) {
        void *dst_ptr = (uint8_t *)dst_b.mem_handle + dst_b.byte_offset;
        return src_de->vt->copy_to_ram(src_de->dev, dst_ptr, &src_b, n);
    }

    velri_status st = src_de->vt->copy_to_ram(src_de->dev, slot->ram_staging, &src_b, n);
    if (st != VELRI_OK) {
        return st;
    }
    return dst_de->vt->copy_from_ram(dst_de->dev, &dst_b, slot->ram_staging, n);
}

static velri_status run_compute_batch(velri_slot *slot, uint32_t first_step, uint32_t count,
                                      const velri_step_params *params, bool *out_accepted) {
    *out_accepted = false;
    const velri_plan *plan = slot->plan;
    velri_device_id dev_id = plan->steps[first_step].u.op.device_id;
    velri_device_entry *de = &plan->ctx->devices[dev_id];

    velri_buffer_binding in_b[VELRI_MAX_SUBMIT_BATCH][VELRI_OP_MAX_INPUTS];
    velri_buffer_binding out_b[VELRI_MAX_SUBMIT_BATCH][VELRI_OP_MAX_OUTPUTS];
    velri_submit_item items[VELRI_MAX_SUBMIT_BATCH];

    for (uint32_t k = 0U; k < count; ++k) {
        const velri_launch_step *step = &plan->steps[first_step + k];
        memset(in_b[k], 0, sizeof(in_b[k]));
        memset(out_b[k], 0, sizeof(out_b[k]));
        for (uint32_t i = 0U; i < step->u.op.num_inputs; ++i) {
            if (step->u.op.inputs[i].present) {
                in_b[k][i] = resolve_storage_binding(slot, step->u.op.inputs[i].plan_storage_idx,
                                                     step->u.op.inputs[i].view_byte_offset);
            }
        }
        for (uint32_t o = 0U; o < step->u.op.num_outputs; ++o) {
            out_b[k][o] = resolve_storage_binding(slot, step->u.op.outputs[o].plan_storage_idx,
                                                  step->u.op.outputs[o].view_byte_offset);
        }
        items[k] = (velri_submit_item){
            .op = &step->u.op.resolved,
            .prepared = step->u.op.prepared,
            .inputs = in_b[k],
            .num_inputs = step->u.op.num_inputs,
            .outputs = out_b[k],
            .num_outputs = step->u.op.num_outputs,
            .params = params,
        };
    }

    velri_completion *comp = nullptr;
    velri_status submit_st = (count > 1U && de->vt->submit_batch)
                                 ? de->vt->submit_batch(de->dev, items, count, &comp)
                                 : de->vt->submit(de->dev, &items[0], &comp);

    if (de->vt->wait) {
        if (!comp) {
            return submit_st;
        }
        *out_accepted = true;
        velri_status wait_st = de->vt->wait(de->dev, comp);
        de->vt->destroy_completion(de->dev, comp);
        return (submit_st != VELRI_OK) ? submit_st : wait_st;
    }

    if (submit_st == VELRI_OK || submit_st == VELRI_ERR_EXEC) {
        *out_accepted = true;
    }
    return submit_st;
}

velri_status velri_slot_run(velri_slot *slot, const velri_step_params *params,
                            velri_cancel_token *cancel) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    if (slot->needs_reset) {
        return VELRI_ERR_BUSY;
    }

    const velri_plan *plan = slot->plan;
    uint64_t allocs_before = atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed);
    slot->last_run_allocs = 0U;

    /* Validate all required external storages (STATE may remain purely device-resident). */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        velri_storage_kind kind = plan->storages[i].kind;
        if (kind == VELRI_STORAGE_SCRATCH || kind == VELRI_STORAGE_STATE) {
            continue;
        }
        if (kind == VELRI_STORAGE_WEIGHT) {
            if (!plan->weight_bindings[i].is_bound) return VELRI_ERR_INVALID;
        } else if (!slot->user_bindings[i].is_bound) {
            return VELRI_ERR_INVALID;
        }
    }

    velri_step_params default_params = {0};
    const velri_step_params *eff_params = params ? params : &default_params;
    velri_status pst = validate_dynamic_params(plan, eff_params);
    if (pst != VELRI_OK) {
        return pst;
    }

    if (velri_cancel_is_requested(cancel)) {
        return VELRI_ERR_CANCELLED;
    }

    /* Upload step INPUTs (non-host) and initial host-bound STATE (both host & non-host). */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        bool need_state_up = (ps->kind == VELRI_STORAGE_STATE && slot->user_bindings[i].is_bound &&
                              !slot->user_bindings[i].state_uploaded);
        bool need_input_up = (!de->caps.is_host_memory && ps->kind == VELRI_STORAGE_INPUT);
        if (!need_input_up && !need_state_up) {
            continue;
        }
        if (velri_cancel_is_requested(cancel)) {
            return VELRI_ERR_CANCELLED;
        }
        velri_buffer_binding dst = resolve_storage_binding(slot, i, 0U);
        if (de->caps.is_host_memory) {
            if (ps->byte_size > 0U) {
                memcpy((uint8_t *)dst.mem_handle + dst.byte_offset,
                       slot->user_bindings[i].host_ptr, ps->byte_size);
            }
        } else {
            velri_status st = de->vt->copy_from_ram(
                de->dev, &dst, slot->user_bindings[i].host_ptr, ps->byte_size);
            if (st != VELRI_OK) {
                return st;
            }
        }
        if (ps->kind == VELRI_STORAGE_STATE) {
            slot->user_bindings[i].state_uploaded = true;
        }
    }

    bool any_work_accepted = false;
    uint32_t s = 0U;
    while (s < plan->num_steps) {
        if (velri_cancel_is_requested(cancel)) {
            if (any_work_accepted) slot->needs_reset = true;
            slot->last_run_allocs =
                atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
            return VELRI_ERR_CANCELLED;
        }

        const velri_launch_step *step = &plan->steps[s];
        if (step->kind == VELRI_STEP_RAM_TRANSFER) {
            velri_status st = run_ram_transfer(slot, step);
            if (st != VELRI_OK) {
                if (any_work_accepted) slot->needs_reset = true;
                slot->last_run_allocs =
                    atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) -
                    allocs_before;
                return st;
            }
            s++;
            continue;
        }

        velri_device_id dev_id = step->u.op.device_id;
        uint32_t batch = 1U;
        if (plan->ctx->devices[dev_id].vt->submit_batch) {
            while (s + batch < plan->num_steps && batch < VELRI_MAX_SUBMIT_BATCH &&
                   plan->steps[s + batch].kind == VELRI_STEP_OP &&
                   plan->steps[s + batch].u.op.device_id == dev_id) {
                batch++;
            }
        }

        bool accepted = false;
        velri_status st = run_compute_batch(slot, s, batch, eff_params, &accepted);
        if (accepted) {
            any_work_accepted = true;
        }
        if (st != VELRI_OK) {
            if (any_work_accepted) slot->needs_reset = true;
            slot->last_run_allocs =
                atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
            return st;
        }
        s += batch;
    }

    /* Download ONLY non-host OUTPUTs (STATE remains resident on its home device!). */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        if (de->caps.is_host_memory || ps->kind != VELRI_STORAGE_OUTPUT) {
            continue;
        }
        velri_buffer_binding src = resolve_storage_binding(slot, i, 0U);
        velri_status st =
            de->vt->copy_to_ram(de->dev, slot->user_bindings[i].host_ptr, &src, ps->byte_size);
        if (st != VELRI_OK) {
            slot->needs_reset = true;
            slot->last_run_allocs =
                atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
            return st;
        }
    }

    slot->last_run_allocs =
        atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
    return VELRI_OK;
}

bool velri_slot_needs_reset(const velri_slot *slot) { return slot ? slot->needs_reset : false; }

velri_status velri_slot_reset(velri_slot *slot) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan *plan = slot->plan;
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        velri_status st = zero_device_arena(plan->ctx, d, slot->scratch_arenas[d],
                                            plan->scratch_arena_bytes[d]);
        if (st != VELRI_OK) return st;
        st = zero_device_arena(plan->ctx, d, slot->state_arenas[d], plan->state_arena_bytes[d]);
        if (st != VELRI_OK) return st;
    }
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        slot->user_bindings[i].state_uploaded = false;
    }
    slot->needs_reset = false;
    return VELRI_OK;
}

uint64_t velri_slot_last_run_allocs(const velri_slot *slot) {
    return slot ? slot->last_run_allocs : 0U;
}
