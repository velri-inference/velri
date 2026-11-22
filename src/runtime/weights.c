#include "runtime/runtime_internal.h"

#include <string.h>

/* Open addressing, power-of-two capacity, at most 75% occupancy. An empty name
 * denotes a free bucket. Entries never move while a program holds the model. */
static velri_rt_weight_entry *weight_bucket(velri_rt_weight_entry *table, uint32_t capacity,
                                            const char *name) {
    uint32_t hash = 2166136261U;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        hash = (hash ^ *p) * 16777619U;
    }
    uint32_t mask = capacity - 1U;
    uint32_t i = hash & mask;
    while (table[i].name[0] && strcmp(table[i].name, name) != 0) {
        i = (i + 1U) & mask;
    }
    return &table[i];
}

const velri_rt_weight_entry *velri_rt_model_find_weight(const velri_model *model,
                                                        const char *name) {
    if (!model->cap_weights) {
        return nullptr;
    }
    const velri_rt_weight_entry *entry = weight_bucket(model->weights, model->cap_weights, name);
    return entry->name[0] ? entry : nullptr;
}

static velri_rc grow_weights(velri_model *model) {
    uint32_t capacity = model->cap_weights ? model->cap_weights * 2U : 32U;
    size_t buckets = capacity;
    if (capacity <= model->cap_weights || buckets > SIZE_MAX / sizeof(velri_rt_weight_entry)) {
        velri_rt_ctx_set_error(model->ctx, "model_bind_weight: weight table overflow");
        return VELRI_RC_ERR_OVERFLOW;
    }
    velri_rt_weight_entry *table = calloc(buckets, sizeof(*table));
    if (!table) {
        velri_rt_ctx_set_error(model->ctx, "model_bind_weight: out of memory");
        return VELRI_RC_ERR_NOMEM;
    }
    for (uint32_t i = 0U; i < model->cap_weights; ++i) {
        const velri_rt_weight_entry *old = &model->weights[i];
        if (old->name[0]) {
            *weight_bucket(table, capacity, old->name) = *old;
        }
    }
    free(model->weights);
    model->weights = table;
    model->cap_weights = capacity;
    return VELRI_RC_OK;
}

static velri_rc set_weight(velri_model *model, const char *name, const void *data, size_t bytes) {
    if (!name || !name[0] || !data || !bytes) {
        velri_rt_ctx_set_error(model->ctx,
                               "model_bind_weight: invalid weight name, pointer, or size");
        return VELRI_RC_ERR_INVALID;
    }
    size_t len = strlen(name);
    if (len >= VELRI_MANIFEST_NAME_MAX) {
        velri_rt_ctx_set_error(model->ctx, "model_bind_weight: weight name exceeds %u bytes",
                               (unsigned)(VELRI_MANIFEST_NAME_MAX - 1U));
        return VELRI_RC_ERR_OVERFLOW;
    }
    velri_rt_weight_entry *entry =
        model->cap_weights ? weight_bucket(model->weights, model->cap_weights, name) : nullptr;
    if (!entry || !entry->name[0]) {
        uint32_t count = atomic_load_explicit(&model->num_weights, memory_order_relaxed);
        if (count >= model->cap_weights - model->cap_weights / 4U) {
            velri_rc rc = grow_weights(model);
            if (rc != VELRI_RC_OK) {
                return rc;
            }
            entry = weight_bucket(model->weights, model->cap_weights, name);
        }
        memcpy(entry->name, name, len + 1U);
        atomic_store_explicit(&model->num_weights, count + 1U, memory_order_relaxed);
    }
    entry->host_ptr = data;
    entry->byte_size = bytes;
    return VELRI_RC_OK;
}

velri_rc velri_model_bind_weight(velri_model *model, const char *name, const void *host_data,
                                 size_t byte_size) {
    if (!model) {
        return VELRI_RC_ERR_INVALID;
    }
    if (!velri_rt_try_acquire_exclusive(&model->live_programs)) {
        velri_rt_ctx_set_error(model->ctx, "model_bind_weight: model is in use");
        return VELRI_RC_ERR_BUSY;
    }
    velri_rc rc = set_weight(model, name, host_data, byte_size);
    atomic_store_explicit(&model->live_programs, 0U, memory_order_release);
    return rc;
}

uint32_t velri_model_weight_count(const velri_model *model) {
    return model ? atomic_load_explicit(&model->num_weights, memory_order_relaxed) : 0U;
}
