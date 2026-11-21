#include "core_internal.h"
#include <string.h>

/*
 * Execution Slots & Runtime Launch-Graph Executor (`velri-lib/docs/EN/execution.md`).
 *
 * Key invariants:
 *   - All scratch arenas, staging buffers, and non-host device buffers are allocated
 *     once inside `velri_slot_create`; `velri_slot_run` performs 0 B heap allocations.
 *   - Multiple slots sharing one immutable `velri_plan` and shared immutable weights
 *     may execute concurrently on independent threads with full buffer isolation.
 *   - Acceptance != completion: whenever a backend returns a non-null `velri_completion`,
 *     the executor ALWAYS awaits it via `wait` and destroys it before returning, even on
 *     errors or mid-step cancellation.
 */

typedef struct {
    void *host_ptr;
    size_t bound_bytes;
    void *device_handle; /* Non-null only for non-host devices (pre-allocated at slot creation) */
    bool is_bound;
    bool state_uploaded; /* Tracks initial host->device upload for non-host STATE storages */
} slot_storage_binding;

struct velri_slot {
    const velri_plan *plan;
    void *scratch_arenas[VELRI_MAX_DEVICES];
    void *ram_staging;
    slot_storage_binding *user_bindings;
    bool needs_reset;
    uint64_t last_run_allocs;
};

static velri_buffer_binding resolve_storage_binding(
    const velri_slot *slot,
    uint32_t plan_storage_idx,
    size_t view_byte_offset) {
    const velri_plan *plan = slot->plan;
    const velri_plan_storage *ps = &plan->storages[plan_storage_idx];
    const velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    velri_buffer_binding b;
    memset(&b, 0, sizeof(b));

    if (ps->in_scratch_arena) {
        b.mem_handle = slot->scratch_arenas[ps->device_id];
        b.byte_offset = ps->arena_offset + view_byte_offset;
        b.byte_size = plan->scratch_arena_bytes[ps->device_id];
        b.is_host_ptr = de->caps.is_host_memory;
        return b;
    }

    const slot_storage_binding *sb = &slot->user_bindings[plan_storage_idx];
    if (de->caps.is_host_memory) {
        b.mem_handle = sb->host_ptr;
        b.byte_offset = view_byte_offset;
        b.byte_size = sb->bound_bytes;
        b.is_host_ptr = true;
    } else {
        b.mem_handle = sb->device_handle;
        b.byte_offset = view_byte_offset;
        b.byte_size = ps->byte_size;
        b.is_host_ptr = false;
    }
    return b;
}

velri_status velri_slot_create(const velri_plan *plan, velri_slot **out_slot) {
    if (!plan || !out_slot) {
        return VELRI_ERR_INVALID;
    }
    *out_slot = nullptr;

    velri_slot *slot = (velri_slot *)calloc(1U, sizeof(*slot));
    if (!slot) {
        return VELRI_ERR_NOMEM;
    }
    slot->plan = plan;

    if (plan->num_user_storages > 0U) {
        slot->user_bindings =
            (slot_storage_binding *)calloc(plan->num_user_storages, sizeof(*slot->user_bindings));
        if (!slot->user_bindings) {
            velri_slot_destroy(slot);
            return VELRI_ERR_NOMEM;
        }
    }

    /* Pre-allocate one packed scratch arena per device that uses scratch. */
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        size_t sz = plan->scratch_arena_bytes[d];
        if (sz == 0U) {
            continue;
        }
        velri_device_entry *de = &plan->ctx->devices[d];
        velri_status st = de->vt->alloc(de->dev, sz, plan->scratch_arena_align[d],
                                        &slot->scratch_arenas[d]);
        if (st != VELRI_OK) {
            velri_slot_destroy(slot);
            return st;
        }
    }

    /* Pre-allocate host RAM staging buffer for explicit cross-device transfers. */
    if (plan->ram_staging_bytes > 0U) {
        slot->ram_staging = velri_core_aligned_alloc64(plan->ram_staging_bytes);
        if (!slot->ram_staging) {
            velri_slot_destroy(slot);
            return VELRI_ERR_NOMEM;
        }
        memset(slot->ram_staging, 0, plan->ram_staging_bytes);
    }

    /* Pre-allocate device buffers for external storages residing on non-host devices. */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        if (ps->kind == VELRI_STORAGE_SCRATCH) {
            continue;
        }
        velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        if (de->caps.is_host_memory) {
            continue;
        }
        velri_status st = de->vt->alloc(de->dev, ps->byte_size, ps->align,
                                        &slot->user_bindings[i].device_handle);
        if (st != VELRI_OK) {
            velri_slot_destroy(slot);
            return st;
        }
    }

    *out_slot = slot;
    return VELRI_OK;
}

void velri_slot_destroy(velri_slot *slot) {
    if (!slot) {
        return;
    }
    const velri_plan *plan = slot->plan;
    if (plan && plan->ctx) {
        if (slot->user_bindings) {
            for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
                if (slot->user_bindings[i].device_handle) {
                    velri_device_entry *de = &plan->ctx->devices[plan->storages[i].device_id];
                    de->vt->free(de->dev, slot->user_bindings[i].device_handle);
                }
            }
        }
        for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
            if (slot->scratch_arenas[d]) {
                velri_device_entry *de = &plan->ctx->devices[d];
                de->vt->free(de->dev, slot->scratch_arenas[d]);
            }
        }
    }
    velri_core_aligned_free64(slot->ram_staging);
    free(slot->user_bindings);
    free(slot);
}

velri_status velri_slot_bind_host(
    velri_slot *slot,
    velri_storage_id storage_id,
    void *host_ptr,
    size_t byte_size) {
    if (!slot || !host_ptr || storage_id >= slot->plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &slot->plan->storages[storage_id];
    if (ps->kind == VELRI_STORAGE_SCRATCH || byte_size < ps->byte_size) {
        return VELRI_ERR_INVALID;
    }

    slot_storage_binding *sb = &slot->user_bindings[storage_id];
    sb->host_ptr = host_ptr;
    sb->bound_bytes = byte_size;
    sb->is_bound = true;
    sb->state_uploaded = false;

    /* Immutable weights on non-host devices are uploaded once during binding (not per step). */
    velri_device_entry *de = &slot->plan->ctx->devices[ps->device_id];
    if (!de->caps.is_host_memory && ps->kind == VELRI_STORAGE_WEIGHT) {
        velri_buffer_binding dst = resolve_storage_binding(slot, storage_id, 0U);
        velri_status st = de->vt->copy_from_ram(de->dev, &dst, host_ptr, ps->byte_size);
        if (st != VELRI_OK) {
            sb->is_bound = false;
            return st;
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

    velri_status st = src_de->vt->copy_to_ram(src_de->dev, slot->ram_staging, &src_b,
                                              step->u.transfer.byte_size);
    if (st != VELRI_OK) {
        return st;
    }
    return dst_de->vt->copy_from_ram(dst_de->dev, &dst_b, slot->ram_staging,
                                     step->u.transfer.byte_size);
}

static velri_status run_compute_step(
    velri_slot *slot,
    const velri_launch_step *step,
    const velri_step_params *params,
    bool *out_accepted) {
    *out_accepted = false;
    const velri_plan *plan = slot->plan;
    velri_device_entry *de = &plan->ctx->devices[step->u.op.device_id];

    velri_buffer_binding in_b[VELRI_OP_MAX_INPUTS];
    velri_buffer_binding out_b[VELRI_OP_MAX_OUTPUTS];
    memset(in_b, 0, sizeof(in_b));
    memset(out_b, 0, sizeof(out_b));

    for (uint32_t i = 0U; i < step->u.op.num_inputs; ++i) {
        if (step->u.op.inputs[i].present) {
            in_b[i] = resolve_storage_binding(slot, step->u.op.inputs[i].plan_storage_idx,
                                              step->u.op.inputs[i].view_byte_offset);
        }
    }
    for (uint32_t o = 0U; o < step->u.op.num_outputs; ++o) {
        out_b[o] = resolve_storage_binding(slot, step->u.op.outputs[o].plan_storage_idx,
                                           step->u.op.outputs[o].view_byte_offset);
    }

    velri_completion *comp = nullptr;
    velri_status submit_st = de->vt->submit(de->dev, step->u.op.prepared, in_b,
                                            step->u.op.num_inputs, out_b, step->u.op.num_outputs,
                                            params, &comp);
    if (!comp) {
        /* Pre-accept rejection (or synchronous failure before touching buffers). */
        return submit_st;
    }

    /* Work was accepted: ALWAYS wait until the backend finishes touching buffers. */
    *out_accepted = true;
    velri_status wait_st = de->vt->wait(de->dev, comp);
    de->vt->destroy_completion(de->dev, comp);

    if (submit_st != VELRI_OK) {
        return submit_st;
    }
    return wait_st;
}

velri_status velri_slot_run(
    velri_slot *slot,
    const velri_step_params *params,
    velri_cancel_token *cancel) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    if (slot->needs_reset) {
        return VELRI_ERR_BUSY;
    }

    const velri_plan *plan = slot->plan;
    slot->last_run_allocs = 0U;

    /* Validate all external storages are bound before starting any work. */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        if (plan->storages[i].kind != VELRI_STORAGE_SCRATCH && !slot->user_bindings[i].is_bound) {
            return VELRI_ERR_INVALID;
        }
    }

    /* Cancellation before first work is accepted leaves slot state clean (no reset required). */
    if (velri_cancel_is_requested(cancel)) {
        return VELRI_ERR_CANCELLED;
    }

    velri_step_params default_params = {0};
    const velri_step_params *eff_params = params ? params : &default_params;

    /* Upload step inputs (and initial state) to non-host devices before running launch steps. */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        if (de->caps.is_host_memory) {
            continue;
        }
        bool need_up = (ps->kind == VELRI_STORAGE_INPUT) ||
                       (ps->kind == VELRI_STORAGE_STATE && !slot->user_bindings[i].state_uploaded);
        if (!need_up) {
            continue;
        }
        if (velri_cancel_is_requested(cancel)) {
            return VELRI_ERR_CANCELLED;
        }
        velri_buffer_binding dst = resolve_storage_binding(slot, i, 0U);
        velri_status st =
            de->vt->copy_from_ram(de->dev, &dst, slot->user_bindings[i].host_ptr, ps->byte_size);
        if (st != VELRI_OK) {
            return st;
        }
        if (ps->kind == VELRI_STORAGE_STATE) {
            slot->user_bindings[i].state_uploaded = true;
        }
    }

    bool any_work_accepted = false;

    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        if (velri_cancel_is_requested(cancel)) {
            if (any_work_accepted) {
                slot->needs_reset = true;
            }
            return VELRI_ERR_CANCELLED;
        }

        const velri_launch_step *step = &plan->steps[s];
        if (step->kind == VELRI_STEP_RAM_TRANSFER) {
            velri_status st = run_ram_transfer(slot, step);
            if (st != VELRI_OK) {
                if (any_work_accepted) {
                    slot->needs_reset = true;
                }
                return st;
            }
        } else {
            bool accepted = false;
            velri_status st = run_compute_step(slot, step, eff_params, &accepted);
            if (accepted) {
                any_work_accepted = true;
            }
            if (st != VELRI_OK) {
                if (any_work_accepted) {
                    slot->needs_reset = true;
                }
                return st;
            }
        }
    }

    /* Download non-host outputs and updated state back to host bindings. */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        if (de->caps.is_host_memory) {
            continue;
        }
        if (ps->kind == VELRI_STORAGE_OUTPUT || ps->kind == VELRI_STORAGE_STATE) {
            velri_buffer_binding src = resolve_storage_binding(slot, i, 0U);
            velri_status st =
                de->vt->copy_to_ram(de->dev, slot->user_bindings[i].host_ptr, &src, ps->byte_size);
            if (st != VELRI_OK) {
                slot->needs_reset = true;
                return st;
            }
        }
    }

    /* Cancellation arriving after final step completed does not discard the valid result. */
    return VELRI_OK;
}

bool velri_slot_needs_reset(const velri_slot *slot) {
    return slot ? slot->needs_reset : false;
}

velri_status velri_slot_reset(velri_slot *slot) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan *plan = slot->plan;
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        if (slot->scratch_arenas[d] && plan->ctx->devices[d].caps.is_host_memory) {
            memset(slot->scratch_arenas[d], 0, plan->scratch_arena_bytes[d]);
        }
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
