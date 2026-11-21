#include "velri_lib/blocks.h"
#include "core_internal.h"

#include <stdlib.h>
#include <string.h>

/* ---- Internal Helpers ---- */

static inline velri_op_desc init_op_desc(velri_op_kind kind, uint32_t num_inputs,
                                         uint32_t num_outputs) {
    velri_op_desc op;
    memset(&op, 0, sizeof(op));
    op.kind = kind;
    op.num_inputs = num_inputs;
    op.num_outputs = num_outputs;
    for (uint32_t i = 0U; i < VELRI_OP_MAX_INPUTS; ++i) {
        op.inputs[i] = VELRI_INVALID_ID;
    }
    for (uint32_t i = 0U; i < VELRI_OP_MAX_OUTPUTS; ++i) {
        op.outputs[i] = VELRI_INVALID_ID;
    }
    return op;
}

static size_t current_scope_path_len(const velri_builder *b) {
    size_t len = 0U;
    for (uint32_t i = 0U; i < b->scope_depth; ++i) {
        if (i > 0U) {
            len += 1U;
        }
        len += strlen(b->scopes[i]);
    }
    return len;
}

static void format_scope_path(const velri_builder *b, char out_path[VELRI_BUILDER_MAX_SCOPE_PATH]) {
    size_t pos = 0U;
    out_path[0] = '\0';
    for (uint32_t i = 0U; i < b->scope_depth; ++i) {
        if (i > 0U && pos + 1U < VELRI_BUILDER_MAX_SCOPE_PATH) {
            out_path[pos++] = '/';
        }
        const char *seg = b->scopes[i];
        size_t slen = strlen(seg);
        size_t avail = (VELRI_BUILDER_MAX_SCOPE_PATH - 1U) - pos;
        if (slen > avail) {
            slen = avail;
        }
        if (slen > 0U) {
            memcpy(out_path + pos, seg, slen);
            pos += slen;
        }
    }
    out_path[pos] = '\0';
}

static velri_status manifest_append_entry(velri_manifest_entry **entries, uint32_t *count,
                                          uint32_t *cap, const char *name,
                                          const char *source_name, const velri_tensor *t) {
    if (!entries || !count || !cap || !name || name[0] == '\0' || !t || !t->valid ||
        t->rank == 0U || t->rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t nlen = strlen(name);
    if (nlen >= VELRI_MANIFEST_NAME_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    size_t slen = 0U;
    if (source_name && source_name[0] != '\0') {
        slen = strlen(source_name);
        if (slen >= VELRI_MANIFEST_NAME_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    if (*count == UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }

    void *tmp = *entries;
    if (!velri_vec_grow(&tmp, cap, *count + 1U, sizeof(velri_manifest_entry))) {
        return VELRI_ERR_NOMEM;
    }
    *entries = (velri_manifest_entry *)tmp;

    velri_manifest_entry *e = &(*entries)[*count];
    memset(e, 0, sizeof(*e));
    memcpy(e->name, name, nlen + 1U);
    if (slen > 0U) {
        memcpy(e->source_name, source_name, slen + 1U);
    }
    e->storage_id = t->storage_id;
    e->view_id = t->view_id;
    e->dtype = t->dtype;
    e->rank = t->rank;
    for (uint32_t d = 0U; d < t->rank; ++d) {
        e->dims[d] = t->dims[d];
    }
    e->device_id = t->device_id;
    (*count)++;
    return VELRI_OK;
}

static velri_tensor alloc_tensor(velri_builder *b, velri_storage_kind kind,
                                 velri_device_id device_id, velri_dtype dtype, uint32_t rank,
                                 const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !dims || rank == 0U || rank > VELRI_MAX_RANK) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t elem_bytes = velri_dtype_bytes(dtype);
    if (elem_bytes == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t num_elems = 1U;
    for (uint32_t d = 0U; d < rank; ++d) {
        if (dims[d] == 0U) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        if (velri_mul_overflow_size(num_elems, dims[d], &num_elems)) {
            velri_b_set_error(b, VELRI_ERR_OVERFLOW);
            return VELRI_TENSOR_NONE;
        }
    }
    size_t byte_size = 0U;
    if (velri_mul_overflow_size(num_elems, elem_bytes, &byte_size)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_storage_desc sdesc = {
        .kind = kind,
        .dtype = dtype,
        .byte_size = byte_size,
        .align = 64U,
        .device_id = device_id,
    };
    velri_storage_id sid = VELRI_INVALID_ID;
    velri_status st = velri_graph_add_storage(b->graph, &sdesc, &sid);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor_view tv;
    st = velri_tensor_view_contiguous(sid, dtype, rank, dims, 0U, &tv);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_view_id vid = VELRI_INVALID_ID;
    st = velri_graph_add_view(b->graph, &tv, &vid);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor t = VELRI_TENSOR_NONE;
    t.storage_id = sid;
    t.view_id = vid;
    t.dtype = dtype;
    t.rank = rank;
    for (uint32_t d = 0U; d < rank; ++d) {
        t.dims[d] = tv.dims[d];
        t.strides[d] = tv.strides[d];
    }
    t.byte_offset = 0U;
    t.device_id = device_id;
    t.valid = true;
    return t;
}

static bool emit_op(velri_builder *b, const velri_op_desc *op) {
    velri_device_id compute_dev = 0U;
    velri_b_devices(b, nullptr, &compute_dev);
    velri_status st = velri_graph_add_op(b->graph, op, compute_dev, nullptr);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return false;
    }
    return true;
}

static bool same_shape(const velri_tensor *a, const velri_tensor *b) {
    if (a->rank != b->rank || a->rank == 0U || a->rank > VELRI_MAX_RANK) {
        return false;
    }
    for (uint32_t d = 0U; d < a->rank; ++d) {
        if (a->dims[d] != b->dims[d]) {
            return false;
        }
    }
    return true;
}

/* ---- Named Weight & State Manifest ---- */

void velri_manifest_init(velri_manifest *m) {
    if (!m) {
        return;
    }
    memset(m, 0, sizeof(*m));
    m->token_ids_in = VELRI_INVALID_ID;
    m->logits_out = VELRI_INVALID_ID;
}

void velri_manifest_destroy(velri_manifest *m) {
    if (!m) {
        return;
    }
    free(m->weights);
    free(m->states);
    velri_manifest_init(m);
}

const velri_manifest_entry *velri_manifest_find_weight(const velri_manifest *m, const char *name) {
    if (!m || !name || name[0] == '\0') {
        return nullptr;
    }
    for (uint32_t i = 0U; i < m->num_weights; ++i) {
        if (strcmp(m->weights[i].name, name) == 0) {
            return &m->weights[i];
        }
    }
    return nullptr;
}

const velri_manifest_entry *velri_manifest_find_state(const velri_manifest *m, const char *name) {
    if (!m || !name || name[0] == '\0') {
        return nullptr;
    }
    for (uint32_t i = 0U; i < m->num_states; ++i) {
        if (strcmp(m->states[i].name, name) == 0) {
            return &m->states[i];
        }
    }
    return nullptr;
}

/* ---- Lifecycle & Sticky Error Handling ---- */

void velri_b_init(velri_builder *b, velri_graph *graph, velri_manifest *manifest,
                  velri_device_id default_store_dev, velri_device_id default_compute_dev,
                  velri_placement_resolve_fn resolve_fn, void *resolve_user) {
    if (!b) {
        return;
    }
    memset(b, 0, sizeof(*b));
    b->graph = graph;
    b->manifest = manifest;
    b->err = graph ? VELRI_OK : VELRI_ERR_INVALID;
    b->default_store_dev = default_store_dev;
    b->default_compute_dev = default_compute_dev;
    b->resolve_fn = resolve_fn;
    b->resolve_user = resolve_user;
    b->current_layer = -1;
}

velri_status velri_b_status(const velri_builder *b) {
    return b ? b->err : VELRI_ERR_INVALID;
}

void velri_b_set_error(velri_builder *b, velri_status err) {
    if (b && b->err == VELRI_OK && err != VELRI_OK) {
        b->err = err;
    }
}

/* ---- Scope Stack & Placement Resolution ---- */

void velri_b_push_scope(velri_builder *b, const char *name) {
    if (!b || b->err != VELRI_OK) {
        return;
    }
    if (!name || name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return;
    }
    if (b->scope_depth >= VELRI_BUILDER_MAX_SCOPE_DEPTH) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return;
    }
    size_t cur_len = current_scope_path_len(b);
    size_t add_len = strlen(name) + (b->scope_depth > 0U ? 1U : 0U);
    if (add_len >= VELRI_BUILDER_MAX_SCOPE_PATH ||
        cur_len >= VELRI_BUILDER_MAX_SCOPE_PATH - add_len) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return;
    }
    b->scopes[b->scope_depth++] = name;
}

void velri_b_pop_scope(velri_builder *b) {
    if (!b || b->err != VELRI_OK) {
        return;
    }
    if (b->scope_depth == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return;
    }
    b->scope_depth--;
    b->scopes[b->scope_depth] = nullptr;
}

void velri_b_push_layer(velri_builder *b, int32_t layer_idx) {
    if (!b || b->err != VELRI_OK) {
        return;
    }
    if (layer_idx < 0) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return;
    }
    if (b->layer_depth >= VELRI_BUILDER_MAX_SCOPE_DEPTH) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return;
    }
    b->layer_stack[b->layer_depth++] = b->current_layer;
    b->current_layer = layer_idx;
}

void velri_b_pop_layer(velri_builder *b) {
    if (!b || b->err != VELRI_OK) {
        return;
    }
    if (b->layer_depth == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return;
    }
    b->layer_depth--;
    b->current_layer = b->layer_stack[b->layer_depth];
    b->layer_stack[b->layer_depth] = 0;
}

void velri_b_devices(const velri_builder *b, velri_device_id *out_store,
                     velri_device_id *out_compute) {
    velri_device_id store = b ? b->default_store_dev : 0U;
    velri_device_id compute = b ? b->default_compute_dev : 0U;

    if (b && b->resolve_fn) {
        char path[VELRI_BUILDER_MAX_SCOPE_PATH];
        format_scope_path(b, path);
        b->resolve_fn(path, b->current_layer, b->resolve_user, &store, &compute);
    }

    if (out_store) {
        *out_store = store;
    }
    if (out_compute) {
        *out_compute = compute;
    }
}

/* ---- Tensor Constructors & Named Registration ---- */

velri_tensor velri_b_weight(velri_builder *b, const char *name, velri_dtype dtype, uint32_t rank,
                            const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (name) {
        if (name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        if (strlen(name) >= VELRI_MANIFEST_NAME_MAX) {
            velri_b_set_error(b, VELRI_ERR_OVERFLOW);
            return VELRI_TENSOR_NONE;
        }
    }

    velri_device_id store_dev = 0U;
    velri_b_devices(b, &store_dev, nullptr);
    velri_tensor t = alloc_tensor(b, VELRI_STORAGE_WEIGHT, store_dev, dtype, rank, dims);
    if (!t.valid) {
        return VELRI_TENSOR_NONE;
    }

    if (b->manifest && name) {
        velri_status st = manifest_append_entry(&b->manifest->weights, &b->manifest->num_weights,
                                                &b->manifest->cap_weights, name, "", &t);
        if (st != VELRI_OK) {
            velri_b_set_error(b, st);
            return VELRI_TENSOR_NONE;
        }
    }
    return t;
}

velri_tensor velri_b_weight_tied(velri_builder *b, const char *name, const char *tie_to,
                                 velri_dtype dtype, uint32_t rank, const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!tie_to) {
        return velri_b_weight(b, name, dtype, rank, dims);
    }
    if (!b->manifest || !dims || rank == 0U || rank > VELRI_MAX_RANK || tie_to[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (name && strlen(name) >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    const velri_manifest_entry *src = velri_manifest_find_weight(b->manifest, tie_to);
    if (!src || src->dtype != dtype || src->rank != rank) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    for (uint32_t d = 0U; d < rank; ++d) {
        if (src->dims[d] != dims[d]) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    }

    velri_device_id store_dev = 0U;
    velri_b_devices(b, &store_dev, nullptr);
    if (src->device_id == store_dev) {
        velri_tensor_view tv;
        velri_status st = velri_tensor_view_contiguous(src->storage_id, src->dtype, src->rank,
                                                       src->dims, 0U, &tv);
        if (st != VELRI_OK) {
            velri_b_set_error(b, st);
            return VELRI_TENSOR_NONE;
        }

        velri_tensor t = VELRI_TENSOR_NONE;
        t.storage_id = src->storage_id;
        t.view_id = src->view_id;
        t.dtype = src->dtype;
        t.rank = src->rank;
        for (uint32_t d = 0U; d < src->rank; ++d) {
            t.dims[d] = tv.dims[d];
            t.strides[d] = tv.strides[d];
        }
        t.byte_offset = 0U;
        t.device_id = src->device_id;
        t.is_seq_dynamic = false;
        t.valid = true;
        return t;
    }

    /* Cross-device placement: copy names before any manifest reallocation, then allocate on store_dev. */
    const char *entry_name = (name && name[0] != '\0') ? name : tie_to;
    size_t nlen = strlen(entry_name);
    size_t slen = strlen(tie_to);
    if (nlen >= VELRI_MANIFEST_NAME_MAX || slen >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }
    char name_buf[VELRI_MANIFEST_NAME_MAX];
    char src_buf[VELRI_MANIFEST_NAME_MAX];
    memcpy(name_buf, entry_name, nlen + 1U);
    memcpy(src_buf, tie_to, slen + 1U);

    velri_tensor t = alloc_tensor(b, VELRI_STORAGE_WEIGHT, store_dev, dtype, rank, dims);
    if (!t.valid) {
        return VELRI_TENSOR_NONE;
    }
    velri_status st = manifest_append_entry(&b->manifest->weights, &b->manifest->num_weights,
                                            &b->manifest->cap_weights, name_buf, src_buf, &t);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }
    return t;
}

velri_tensor velri_b_input(velri_builder *b, velri_dtype dtype, uint32_t rank, const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_device_id compute_dev = 0U;
    velri_b_devices(b, nullptr, &compute_dev);
    return alloc_tensor(b, VELRI_STORAGE_INPUT, compute_dev, dtype, rank, dims);
}

velri_tensor velri_b_output(velri_builder *b, velri_dtype dtype, uint32_t rank,
                            const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_device_id compute_dev = 0U;
    velri_b_devices(b, nullptr, &compute_dev);
    return alloc_tensor(b, VELRI_STORAGE_OUTPUT, compute_dev, dtype, rank, dims);
}

velri_tensor velri_b_state(velri_builder *b, const char *name, velri_dtype dtype, uint32_t rank,
                           const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (name) {
        if (name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        if (strlen(name) >= VELRI_MANIFEST_NAME_MAX) {
            velri_b_set_error(b, VELRI_ERR_OVERFLOW);
            return VELRI_TENSOR_NONE;
        }
    }

    velri_device_id compute_dev = 0U;
    velri_b_devices(b, nullptr, &compute_dev);
    velri_tensor t = alloc_tensor(b, VELRI_STORAGE_STATE, compute_dev, dtype, rank, dims);
    if (!t.valid) {
        return VELRI_TENSOR_NONE;
    }

    if (b->manifest && name) {
        velri_status st = manifest_append_entry(&b->manifest->states, &b->manifest->num_states,
                                                &b->manifest->cap_states, name, "", &t);
        if (st != VELRI_OK) {
            velri_b_set_error(b, st);
            return VELRI_TENSOR_NONE;
        }
    }
    return t;
}

velri_tensor velri_b_scratch(velri_builder *b, velri_dtype dtype, uint32_t rank,
                             const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_device_id compute_dev = 0U;
    velri_b_devices(b, nullptr, &compute_dev);
    return alloc_tensor(b, VELRI_STORAGE_SCRATCH, compute_dev, dtype, rank, dims);
}

velri_tensor velri_b_subview_2d_cols(velri_builder *b, velri_tensor base, size_t col_offset,
                                     size_t num_cols) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !base.valid || base.rank != 2U || base.strides[1] != 1U || num_cols == 0U ||
        col_offset > base.dims[1] || num_cols > base.dims[1] - col_offset) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t elem_bytes = velri_dtype_bytes(base.dtype);
    if (elem_bytes == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t extra_byte_offset = 0U;
    if (velri_mul_overflow_size(col_offset, elem_bytes, &extra_byte_offset)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor_view base_view;
    memset(&base_view, 0, sizeof(base_view));
    base_view.storage_id = base.storage_id;
    base_view.dtype = base.dtype;
    base_view.rank = 2U;
    base_view.dims[0] = base.dims[0];
    base_view.dims[1] = base.dims[1];
    base_view.strides[0] = base.strides[0];
    base_view.strides[1] = base.strides[1];
    base_view.byte_offset = base.byte_offset;

    const size_t sub_dims[2] = {base.dims[0], num_cols};
    const size_t sub_strides[2] = {base.strides[0], 1U};
    velri_tensor_view sub_view;
    velri_status st = velri_tensor_view_subview(&base_view, 2U, sub_dims, sub_strides,
                                                extra_byte_offset, &sub_view);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_view_id vid = VELRI_INVALID_ID;
    st = velri_graph_add_view(b->graph, &sub_view, &vid);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = VELRI_TENSOR_NONE;
    out.storage_id = base.storage_id;
    out.view_id = vid;
    out.dtype = base.dtype;
    out.rank = 2U;
    out.dims[0] = sub_view.dims[0];
    out.dims[1] = sub_view.dims[1];
    out.strides[0] = sub_view.strides[0];
    out.strides[1] = sub_view.strides[1];
    out.byte_offset = sub_view.byte_offset;
    out.device_id = base.device_id;
    out.is_seq_dynamic = base.is_seq_dynamic;
    out.valid = true;
    return out;
}

velri_tensor velri_b_subview_2d_rows(velri_builder *b, velri_tensor base, size_t row_offset,
                                     size_t num_rows) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !base.valid || base.rank != 2U || num_rows == 0U ||
        row_offset > base.dims[0] || num_rows > base.dims[0] - row_offset) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t elem_bytes = velri_dtype_bytes(base.dtype);
    if (elem_bytes == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t elem_offset = 0U;
    size_t extra_byte_offset = 0U;
    if (velri_mul_overflow_size(row_offset, base.strides[0], &elem_offset) ||
        velri_mul_overflow_size(elem_offset, elem_bytes, &extra_byte_offset)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor_view base_view;
    memset(&base_view, 0, sizeof(base_view));
    base_view.storage_id = base.storage_id;
    base_view.dtype = base.dtype;
    base_view.rank = 2U;
    base_view.dims[0] = base.dims[0];
    base_view.dims[1] = base.dims[1];
    base_view.strides[0] = base.strides[0];
    base_view.strides[1] = base.strides[1];
    base_view.byte_offset = base.byte_offset;

    const size_t sub_dims[2] = {num_rows, base.dims[1]};
    const size_t sub_strides[2] = {base.strides[0], base.strides[1]};
    velri_tensor_view sub_view;
    velri_status st = velri_tensor_view_subview(&base_view, 2U, sub_dims, sub_strides,
                                                extra_byte_offset, &sub_view);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_view_id vid = VELRI_INVALID_ID;
    st = velri_graph_add_view(b->graph, &sub_view, &vid);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = VELRI_TENSOR_NONE;
    out.storage_id = base.storage_id;
    out.view_id = vid;
    out.dtype = base.dtype;
    out.rank = 2U;
    out.dims[0] = sub_view.dims[0];
    out.dims[1] = sub_view.dims[1];
    out.strides[0] = sub_view.strides[0];
    out.strides[1] = sub_view.strides[1];
    out.byte_offset = sub_view.byte_offset;
    out.device_id = base.device_id;
    out.is_seq_dynamic = (row_offset == 0U && num_rows == base.dims[0]) && base.is_seq_dynamic;
    out.valid = true;
    return out;
}

velri_tensor velri_b_subview_1d(velri_builder *b, velri_tensor base, size_t offset, size_t len) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !base.valid || base.rank != 1U || base.strides[0] != 1U || len == 0U ||
        offset > base.dims[0] || len > base.dims[0] - offset) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t elem_bytes = velri_dtype_bytes(base.dtype);
    if (elem_bytes == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t extra_byte_offset = 0U;
    if (velri_mul_overflow_size(offset, elem_bytes, &extra_byte_offset)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor_view base_view;
    memset(&base_view, 0, sizeof(base_view));
    base_view.storage_id = base.storage_id;
    base_view.dtype = base.dtype;
    base_view.rank = 1U;
    base_view.dims[0] = base.dims[0];
    base_view.strides[0] = base.strides[0];
    base_view.byte_offset = base.byte_offset;

    const size_t sub_dims[1] = {len};
    const size_t sub_strides[1] = {1U};
    velri_tensor_view sub_view;
    velri_status st = velri_tensor_view_subview(&base_view, 1U, sub_dims, sub_strides,
                                                extra_byte_offset, &sub_view);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_view_id vid = VELRI_INVALID_ID;
    st = velri_graph_add_view(b->graph, &sub_view, &vid);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = VELRI_TENSOR_NONE;
    out.storage_id = base.storage_id;
    out.view_id = vid;
    out.dtype = base.dtype;
    out.rank = 1U;
    out.dims[0] = sub_view.dims[0];
    out.strides[0] = sub_view.strides[0];
    out.byte_offset = sub_view.byte_offset;
    out.device_id = base.device_id;
    out.is_seq_dynamic = (offset == 0U && len == base.dims[0]) && base.is_seq_dynamic;
    out.valid = true;
    return out;
}

/* ---- Primitive Operation Builders ---- */

velri_tensor velri_b_embedding(velri_builder *b, velri_tensor table, velri_tensor ids_or_none,
                               size_t pos_offset, size_t max_seq_len) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!table.valid || table.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t seq_len = max_seq_len;
    if (ids_or_none.valid) {
        if (ids_or_none.rank != 1U) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        if (seq_len == 0U) {
            seq_len = ids_or_none.dims[0];
        } else if (seq_len != ids_or_none.dims[0]) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    } else if (seq_len == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    const size_t out_dims[2] = {seq_len, table.dims[1]};
    velri_tensor out = velri_b_scratch(b, table.dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = true;

    velri_op_desc op = init_op_desc(VELRI_OP_EMBEDDING, ids_or_none.valid ? 2U : 1U, 1U);
    op.dynamic_seq = true;
    op.inputs[0] = table.view_id;
    if (ids_or_none.valid) {
        op.inputs[1] = ids_or_none.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.embedding.pos_offset = pos_offset;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_linear_into(velri_builder *b, velri_tensor x, velri_tensor w,
                                 velri_tensor bias_or_none, velri_tensor out) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || !w.valid || !out.valid || x.rank != 2U || w.rank != 2U || out.rank != 2U ||
        x.dims[1] != w.dims[1] || out.dims[0] != x.dims[0] || out.dims[1] != w.dims[0]) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_op_desc op = init_op_desc(VELRI_OP_LINEAR, bias_or_none.valid ? 3U : 2U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    op.inputs[1] = w.view_id;
    if (bias_or_none.valid) {
        op.inputs[2] = bias_or_none.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.linear.desc = (velri_linear_desc){
        .x_dtype = x.dtype,
        .w_dtype = w.dtype,
        .y_dtype = out.dtype,
        .has_x_scale = false,
        .has_w_scale = false,
    };

    out.is_seq_dynamic = x.is_seq_dynamic;
    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_linear(velri_builder *b, velri_tensor x, velri_tensor w,
                            velri_tensor bias_or_none) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || !w.valid || x.rank != 2U || w.rank != 2U || x.dims[1] != w.dims[1]) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_dtype y_dtype =
        (x.dtype == VELRI_DTYPE_F64 && w.dtype == VELRI_DTYPE_F64) ? VELRI_DTYPE_F64
                                                                   : VELRI_DTYPE_F32;
    const size_t out_dims[2] = {x.dims[0], w.dims[0]};
    velri_tensor out = velri_b_scratch(b, y_dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    return velri_b_linear_into(b, x, w, bias_or_none, out);
}

velri_tensor velri_b_layernorm(velri_builder *b, velri_tensor x, velri_tensor w,
                               velri_tensor b_opt, float eps) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = velri_b_scratch(b, x.dtype, 2U, x.dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    uint32_t num_inputs = b_opt.valid ? 3U : (w.valid ? 2U : 1U);
    velri_op_desc op = init_op_desc(VELRI_OP_LAYERNORM, num_inputs, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    if (w.valid) {
        op.inputs[1] = w.view_id;
    }
    if (b_opt.valid) {
        op.inputs[2] = b_opt.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.layernorm.eps = eps;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_rmsnorm(velri_builder *b, velri_tensor x, velri_tensor w, float eps,
                             bool unit_offset) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = velri_b_scratch(b, x.dtype, 2U, x.dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    velri_op_desc op = init_op_desc(VELRI_OP_RMSNORM, w.valid ? 2U : 1U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    if (w.valid) {
        op.inputs[1] = w.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.rmsnorm.eps = eps;
    op.attr.rmsnorm.unit_offset = unit_offset;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_activation(velri_builder *b, velri_tensor x, velri_act_kind act_kind) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank == 0U || x.rank > VELRI_MAX_RANK) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = velri_b_scratch(b, x.dtype, x.rank, x.dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    velri_op_desc op = init_op_desc(VELRI_OP_ACTIVATION, 1U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    op.outputs[0] = out.view_id;
    op.attr.activation.kind = act_kind;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_activation_inplace(velri_builder *b, velri_tensor x, velri_act_kind act_kind) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank == 0U || x.rank > VELRI_MAX_RANK) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    velri_device_id compute_dev = 0U;
    velri_b_devices(b, nullptr, &compute_dev);
    if (x.device_id != compute_dev) {
        return velri_b_activation(b, x, act_kind);
    }

    velri_op_desc op = init_op_desc(VELRI_OP_ACTIVATION, 1U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    op.outputs[0] = x.view_id;
    op.attr.activation.kind = act_kind;

    return emit_op(b, &op) ? x : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_gated_mlp_act(velri_builder *b, velri_tensor gate, velri_tensor up,
                                   velri_act_kind act_kind, float clamp_limit) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!gate.valid || !up.valid || !same_shape(&gate, &up)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = velri_b_scratch(b, gate.dtype, gate.rank, gate.dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    bool dyn = gate.is_seq_dynamic || up.is_seq_dynamic;
    out.is_seq_dynamic = dyn;

    velri_op_desc op = init_op_desc(VELRI_OP_GATED_MLP, 2U, 1U);
    op.dynamic_seq = dyn;
    op.inputs[0] = gate.view_id;
    op.inputs[1] = up.view_id;
    op.outputs[0] = out.view_id;
    op.attr.gated_mlp.act = act_kind;
    op.attr.gated_mlp.clamp_limit = clamp_limit;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_gated_mlp_act_inplace(velri_builder *b, velri_tensor gate, velri_tensor up,
                                           velri_act_kind act_kind, float clamp_limit) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!gate.valid || !up.valid || !same_shape(&gate, &up)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    velri_device_id compute_dev = 0U;
    velri_b_devices(b, nullptr, &compute_dev);
    if (gate.device_id != compute_dev) {
        return velri_b_gated_mlp_act(b, gate, up, act_kind, clamp_limit);
    }

    bool dyn = gate.is_seq_dynamic || up.is_seq_dynamic;
    gate.is_seq_dynamic = dyn;
    velri_op_desc op = init_op_desc(VELRI_OP_GATED_MLP, 2U, 1U);
    op.dynamic_seq = dyn;
    op.inputs[0] = gate.view_id;
    op.inputs[1] = up.view_id;
    op.outputs[0] = gate.view_id;
    op.attr.gated_mlp.act = act_kind;
    op.attr.gated_mlp.clamp_limit = clamp_limit;

    return emit_op(b, &op) ? gate : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_add(velri_builder *b, velri_tensor a, velri_tensor b_in) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!a.valid || !b_in.valid || !same_shape(&a, &b_in)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = velri_b_scratch(b, a.dtype, a.rank, a.dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    bool dyn = a.is_seq_dynamic || b_in.is_seq_dynamic;
    out.is_seq_dynamic = dyn;

    velri_op_desc op = init_op_desc(VELRI_OP_VEC_AXPBY, 2U, 1U);
    op.dynamic_seq = dyn;
    op.inputs[0] = a.view_id;
    op.inputs[1] = b_in.view_id;
    op.outputs[0] = out.view_id;
    op.attr.axpby.alpha = 1.0f;
    op.attr.axpby.beta = 1.0f;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_add_inplace(velri_builder *b, velri_tensor a, velri_tensor b_in) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!a.valid || !b_in.valid || !same_shape(&a, &b_in)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    velri_device_id compute_dev = 0U;
    velri_b_devices(b, nullptr, &compute_dev);
    if (a.device_id != compute_dev) {
        return velri_b_add(b, a, b_in);
    }

    bool dyn = a.is_seq_dynamic || b_in.is_seq_dynamic;
    a.is_seq_dynamic = dyn;
    velri_op_desc op = init_op_desc(VELRI_OP_VEC_AXPBY, 2U, 1U);
    op.dynamic_seq = dyn;
    op.inputs[0] = a.view_id;
    op.inputs[1] = b_in.view_id;
    op.outputs[0] = a.view_id;
    op.attr.axpby.alpha = 1.0f;
    op.attr.axpby.beta = 1.0f;

    return emit_op(b, &op) ? a : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_batched_linear(velri_builder *b, velri_tensor x, velri_tensor w,
                                    velri_tensor bias_or_none, uint32_t n_heads, uint32_t n_groups,
                                    uint32_t in_dim, uint32_t out_dim, bool weight_transposed) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || !w.valid || x.rank != 2U || w.rank != 2U ||
        (bias_or_none.valid && bias_or_none.rank != 1U) ||
        n_heads == 0U || n_groups == 0U || in_dim == 0U || out_dim == 0U ||
        (n_heads % n_groups) != 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t out_cols = 0U;
    if (velri_mul_overflow_size((size_t)n_heads, (size_t)out_dim, &out_cols)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }
    const size_t out_dims[2] = {x.dims[0], out_cols};
    velri_tensor out = velri_b_scratch(b, x.dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    velri_op_desc op = init_op_desc(VELRI_OP_BATCHED_LINEAR, bias_or_none.valid ? 3U : 2U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    op.inputs[1] = w.view_id;
    if (bias_or_none.valid) {
        op.inputs[2] = bias_or_none.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.batched_linear.n_heads = n_heads;
    op.attr.batched_linear.n_groups = n_groups;
    op.attr.batched_linear.in_dim = in_dim;
    op.attr.batched_linear.out_dim = out_dim;
    op.attr.batched_linear.weight_transposed = weight_transposed;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_attention_ex(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                  velri_tensor v_in, velri_tensor k_cache, velri_tensor v_cache,
                                  uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim_k,
                                  uint32_t head_dim_v, float scale, uint32_t sliding_window,
                                  bool causal) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    uint32_t dv = head_dim_v ? head_dim_v : head_dim_k;
    if (!q.valid || !k_in.valid || !v_in.valid || !k_cache.valid || !v_cache.valid ||
        q.rank != 2U || n_heads == 0U || n_kv_heads == 0U || head_dim_k == 0U || dv == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t attn_dim = 0U;
    if (velri_mul_overflow_size((size_t)n_heads, (size_t)dv, &attn_dim)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    const size_t out_dims[2] = {q.dims[0], attn_dim};
    velri_tensor out = velri_b_scratch(b, q.dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = true;

    velri_op_desc op = init_op_desc(VELRI_OP_ATTENTION, 3U, 3U);
    op.dynamic_seq = true;
    op.inputs[0] = q.view_id;
    op.inputs[1] = k_in.view_id;
    op.inputs[2] = v_in.view_id;
    op.outputs[0] = out.view_id;
    op.outputs[1] = k_cache.view_id;
    op.outputs[2] = v_cache.view_id;
    op.attr.attention.n_heads = n_heads;
    op.attr.attention.n_kv_heads = n_kv_heads;
    op.attr.attention.head_dim = head_dim_k;
    op.attr.attention.v_head_dim = dv;
    op.attr.attention.scale = scale;
    op.attr.attention.sliding_window = sliding_window;
    op.attr.attention.causal = causal;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_attention_inplace(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                       velri_tensor v_in, velri_tensor k_cache,
                                       velri_tensor v_cache, uint32_t n_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim, float scale, uint32_t sliding_window,
                                       bool causal) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!q.valid || !k_in.valid || !v_in.valid || !k_cache.valid || !v_cache.valid ||
        q.rank != 2U || n_heads == 0U || n_kv_heads == 0U || head_dim == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_op_desc op = init_op_desc(VELRI_OP_ATTENTION, 3U, 3U);
    op.dynamic_seq = true;
    op.inputs[0] = q.view_id;
    op.inputs[1] = k_in.view_id;
    op.inputs[2] = v_in.view_id;
    op.outputs[0] = q.view_id;
    op.outputs[1] = k_cache.view_id;
    op.outputs[2] = v_cache.view_id;
    op.attr.attention.n_heads = n_heads;
    op.attr.attention.n_kv_heads = n_kv_heads;
    op.attr.attention.head_dim = head_dim;
    op.attr.attention.v_head_dim = head_dim;
    op.attr.attention.scale = scale;
    op.attr.attention.sliding_window = sliding_window;
    op.attr.attention.causal = causal;

    q.is_seq_dynamic = true;
    return emit_op(b, &op) ? q : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_attention(velri_builder *b, velri_tensor q, velri_tensor k_in,
                               velri_tensor v_in, velri_tensor k_cache, velri_tensor v_cache,
                               uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                               float scale, uint32_t sliding_window, bool causal) {
    return velri_b_attention_ex(b, q, k_in, v_in, k_cache, v_cache, n_heads, n_kv_heads, head_dim,
                                head_dim, scale, sliding_window, causal);
}

velri_tensor velri_b_copy_last_row(velri_builder *b, velri_tensor x) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    /* Slice on `x.device_id` so cross-device lm-head only transfers 1 row instead of max_seq. */
    const size_t out_dims[2] = {1U, x.dims[1]};
    velri_tensor out = alloc_tensor(b, VELRI_STORAGE_SCRATCH, x.device_id, x.dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = false;

    velri_op_desc op = init_op_desc(VELRI_OP_COPY_LAST_ROW, 1U, 1U);
    op.dynamic_seq = true;
    op.inputs[0] = x.view_id;
    op.outputs[0] = out.view_id;

    velri_status st = velri_graph_add_op(b->graph, &op, x.device_id, nullptr);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }
    return out;
}
