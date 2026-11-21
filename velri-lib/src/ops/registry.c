#include "core_internal.h"
#include <math.h>

/*
 * Central operation registry validator.
 * Device-agnostic single source of truth for shapes, dtypes, strides, attributes,
 * 32-bit kernel index bounds, and memory aliasing rules across all backends.
 */

static bool views_overlap(const velri_tensor_view *a, const velri_tensor_view *b) {
    if (a->storage_id != b->storage_id) {
        return false;
    }
    size_t a0 = 0U, a1 = 0U, b0 = 0U, b1 = 0U;
    if (velri_tensor_view_byte_span(a, &a0, &a1) != VELRI_OK ||
        velri_tensor_view_byte_span(b, &b0, &b1) != VELRI_OK) {
        return true;
    }
    return (a0 < b1) && (b0 < a1);
}

static bool views_identical_layout(const velri_tensor_view *a, const velri_tensor_view *b) {
    if (a->storage_id != b->storage_id || a->dtype != b->dtype ||
        a->rank != b->rank || a->byte_offset != b->byte_offset) {
        return false;
    }
    for (uint32_t d = 0U; d < a->rank; ++d) {
        if (a->dims[d] != b->dims[d] || a->strides[d] != b->strides[d]) {
            return false;
        }
    }
    return true;
}

/* Reject shifted/partial overlap (which causes parallel races); allow exact in-place or disjoint. */
static bool no_partial_overlap(const velri_tensor_view *in_v, const velri_tensor_view *out_v) {
    return !views_overlap(in_v, out_v) || views_identical_layout(in_v, out_v);
}

static bool fits_u32(size_t v) { return v <= (size_t)UINT32_MAX; }

static bool is_unit_inner_2d(const velri_tensor_view *v) {
    return v->rank == 2U && v->strides[1] == 1U && v->strides[0] >= v->dims[1] &&
           fits_u32(v->dims[0]) && fits_u32(v->dims[1]) && fits_u32(v->strides[0]);
}

static bool is_unit_1d(const velri_tensor_view *v) {
    return v->rank == 1U && v->strides[0] == 1U && fits_u32(v->dims[0]);
}

static bool check_same_2d(const velri_tensor_view *a, const velri_tensor_view *b) {
    return is_unit_inner_2d(a) && is_unit_inner_2d(b) &&
           a->dtype == b->dtype && a->dims[0] == b->dims[0] && a->dims[1] == b->dims[1];
}

static bool check_opt_1d_param(const velri_resolved_op *op, uint32_t idx, velri_dtype dt,
                               size_t len, const velri_tensor_view *out0,
                               const velri_tensor_view *out1) {
    if (idx >= op->num_inputs || !op->input_present[idx]) {
        return true;
    }
    const velri_tensor_view *p = &op->inputs[idx];
    if (!is_unit_1d(p) || p->dtype != dt || p->dims[0] != len) {
        return false;
    }
    if (out0 && views_overlap(p, out0)) {
        return false;
    }
    if (out1 && views_overlap(p, out1)) {
        return false;
    }
    return true;
}

static velri_status check_elemwise_f32_view(const velri_tensor_view *v, size_t *out_n) {
    if (!velri_tensor_view_is_contiguous(v) || v->dtype != VELRI_DTYPE_F32) {
        return VELRI_ERR_INVALID;
    }
    size_t n = 0U;
    velri_status st = velri_tensor_view_num_elements(v, &n);
    if (st != VELRI_OK) {
        return st;
    }
    if (!fits_u32(n)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_n = n;
    return VELRI_OK;
}

static bool valid_linear_combo(const velri_linear_desc *d) {
    if (d->x_dtype == VELRI_DTYPE_I8 && d->w_dtype == VELRI_DTYPE_I8 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return d->has_x_scale && d->has_w_scale;
    }
    if (d->has_x_scale || d->has_w_scale) {
        return false;
    }
    if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_F32 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return true;
    }
    if (d->x_dtype == VELRI_DTYPE_F16 && d->w_dtype == VELRI_DTYPE_F16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return true;
    }
    if (d->x_dtype == VELRI_DTYPE_BF16 && d->w_dtype == VELRI_DTYPE_BF16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return true;
    }
    if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_BF16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return true;
    }
    if (d->x_dtype == VELRI_DTYPE_F64 && d->w_dtype == VELRI_DTYPE_F64 &&
        d->y_dtype == VELRI_DTYPE_F64) {
        return true;
    }
    return false;
}

velri_status velri_op_validate(const velri_resolved_op *op) {
    if (!op || op->num_inputs > VELRI_OP_MAX_INPUTS || op->num_outputs == 0U ||
        op->num_outputs > VELRI_OP_MAX_OUTPUTS) {
        return VELRI_ERR_INVALID;
    }

    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        if (op->num_inputs < 2U || op->num_inputs > 5U || op->num_outputs != 1U ||
            !op->input_present[0] || !op->input_present[1]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *w = &op->inputs[1];
        const velri_tensor_view *y = &op->outputs[0];
        const velri_linear_desc *d = &op->raw.attr.linear.desc;
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(w) || !is_unit_inner_2d(y)) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != d->x_dtype || w->dtype != d->w_dtype || y->dtype != d->y_dtype) {
            return VELRI_ERR_INVALID;
        }
        if (!valid_linear_combo(d)) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (x->dims[1] != w->dims[1] || x->dims[0] != y->dims[0] || w->dims[0] != y->dims[1]) {
            return VELRI_ERR_INVALID;
        }
        if (!check_opt_1d_param(op, 2U, d->y_dtype, w->dims[0], y, nullptr)) {
            return VELRI_ERR_INVALID;
        }
        if (d->has_x_scale) {
            if (op->num_inputs <= 3U || !op->input_present[3] ||
                !check_opt_1d_param(op, 3U, VELRI_DTYPE_F32, x->dims[0], y, nullptr)) {
                return VELRI_ERR_INVALID;
            }
        } else if (op->num_inputs > 3U && op->input_present[3]) {
            return VELRI_ERR_INVALID;
        }
        if (d->has_w_scale) {
            if (op->num_inputs <= 4U || !op->input_present[4] ||
                !check_opt_1d_param(op, 4U, VELRI_DTYPE_F32, w->dims[0], y, nullptr)) {
                return VELRI_ERR_INVALID;
            }
        } else if (op->num_inputs > 4U && op->input_present[4]) {
            return VELRI_ERR_INVALID;
        }
        if (views_overlap(x, y) || views_overlap(w, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_RMSNORM: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (!check_same_2d(x, y) || !no_partial_overlap(x, y) ||
            !(op->raw.attr.rmsnorm.eps > 0.0f) || !isfinite(op->raw.attr.rmsnorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != VELRI_DTYPE_F32 && x->dtype != VELRI_DTYPE_BF16 &&
            x->dtype != VELRI_DTYPE_F16) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (!check_opt_1d_param(op, 1U, x->dtype, x->dims[1], y, nullptr)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_FUSED_ADD_RMSNORM: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 2U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *r = &op->outputs[0];
        const velri_tensor_view *y = &op->outputs[1];
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, r) || !check_same_2d(x, y) ||
            !(op->raw.attr.rmsnorm.eps > 0.0f) || !isfinite(op->raw.attr.rmsnorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        if (views_overlap(r, y) || !no_partial_overlap(x, r) || !no_partial_overlap(x, y) ||
            !check_opt_1d_param(op, 1U, VELRI_DTYPE_F32, x->dims[1], r, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_LAYERNORM: {
        if (op->num_inputs < 1U || op->num_inputs > 3U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, y) || !no_partial_overlap(x, y) ||
            !(op->raw.attr.layernorm.eps > 0.0f) || !isfinite(op->raw.attr.layernorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        if (!check_opt_1d_param(op, 1U, VELRI_DTYPE_F32, x->dims[1], y, nullptr) ||
            !check_opt_1d_param(op, 2U, VELRI_DTYPE_F32, x->dims[1], y, nullptr)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_ROPE: {
        if (op->num_inputs != 3U || op->num_outputs != 1U ||
            !op->input_present[0] || !op->input_present[1] || !op->input_present[2]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *ct = &op->inputs[1];
        const velri_tensor_view *st = &op->inputs[2];
        const velri_tensor_view *y = &op->outputs[0];
        if (x->rank != 3U || y->rank != 3U || x->strides[2] != 1U || y->strides[2] != 1U ||
            x->dtype != y->dtype ||
            x->dims[0] != y->dims[0] || x->dims[1] != y->dims[1] || x->dims[2] != y->dims[2]) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != VELRI_DTYPE_F32 && x->dtype != VELRI_DTYPE_BF16) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (op->raw.attr.rope.mode != VELRI_ROPE_HALF &&
            op->raw.attr.rope.mode != VELRI_ROPE_INTERLEAVED) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (!fits_u32(x->dims[0]) || !fits_u32(x->dims[1]) || !fits_u32(x->dims[2]) ||
            !fits_u32(x->strides[0]) || !fits_u32(x->strides[1]) ||
            !fits_u32(y->strides[0]) || !fits_u32(y->strides[1])) {
            return VELRI_ERR_OVERFLOW;
        }
        /* Non-overlapping output head/token strides prevent parallel write races. */
        size_t min_tok_sy = 0U;
        if (y->strides[1] < y->dims[2] ||
            velri_mul_overflow_size(y->dims[1], y->strides[1], &min_tok_sy) ||
            y->strides[0] < min_tok_sy || x->strides[1] < x->dims[2]) {
            return VELRI_ERR_INVALID;
        }
        size_t rd = op->raw.attr.rope.rotary_dim;
        if (rd == 0U || (rd & 1U) != 0U || rd > x->dims[2] || !fits_u32(rd)) {
            return VELRI_ERR_INVALID;
        }
        if (!velri_tensor_view_is_contiguous(ct) || !velri_tensor_view_is_contiguous(st) ||
            ct->dtype != VELRI_DTYPE_F32 || st->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_INVALID;
        }
        size_t end_pos = 0U, need = 0U;
        if (velri_add_overflow_size(op->raw.attr.rope.pos_offset, x->dims[0], &end_pos) ||
            velri_mul_overflow_size(end_pos, rd / 2U, &need) || !fits_u32(op->raw.attr.rope.pos_offset)) {
            return VELRI_ERR_OVERFLOW;
        }
        size_t ct_n = 0U, st_n = 0U;
        if (velri_tensor_view_num_elements(ct, &ct_n) != VELRI_OK ||
            velri_tensor_view_num_elements(st, &st_n) != VELRI_OK ||
            ct_n < need || st_n < need) {
            return VELRI_ERR_INVALID;
        }
        if (!no_partial_overlap(x, y) || views_overlap(ct, y) || views_overlap(st, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_ACTIVATION: {
        if (op->num_inputs != 1U || op->num_outputs != 1U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        velri_act_kind k = op->raw.attr.activation.kind;
        if (k != VELRI_ACT_SILU && k != VELRI_ACT_GELU_TANH && k != VELRI_ACT_RELU2) {
            return VELRI_ERR_UNSUPPORTED;
        }
        size_t nx = 0U, ny = 0U;
        velri_status st = check_elemwise_f32_view(&op->inputs[0], &nx);
        if (st != VELRI_OK) return st;
        st = check_elemwise_f32_view(&op->outputs[0], &ny);
        if (st != VELRI_OK) return st;
        if (nx != ny || !no_partial_overlap(&op->inputs[0], &op->outputs[0])) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_GATED_MLP:
    case VELRI_OP_VEC_MUL: {
        if (op->num_inputs != 2U || op->num_outputs != 1U ||
            !op->input_present[0] || !op->input_present[1]) {
            return VELRI_ERR_INVALID;
        }
        if (op->kind == VELRI_OP_GATED_MLP) {
            velri_act_kind a = op->raw.attr.gated_mlp.act;
            if (a != VELRI_ACT_SILU && a != VELRI_ACT_GELU_TANH) {
                return VELRI_ERR_UNSUPPORTED;
            }
        }
        size_t na = 0U, nb = 0U, ny = 0U;
        velri_status st = check_elemwise_f32_view(&op->inputs[0], &na);
        if (st != VELRI_OK) return st;
        st = check_elemwise_f32_view(&op->inputs[1], &nb);
        if (st != VELRI_OK) return st;
        st = check_elemwise_f32_view(&op->outputs[0], &ny);
        if (st != VELRI_OK) return st;
        if (na != nb || na != ny ||
            !no_partial_overlap(&op->inputs[0], &op->outputs[0]) ||
            !no_partial_overlap(&op->inputs[1], &op->outputs[0])) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_VEC_AXPBY: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        size_t na = 0U, ny = 0U;
        velri_status st = check_elemwise_f32_view(&op->inputs[0], &na);
        if (st != VELRI_OK) return st;
        st = check_elemwise_f32_view(&op->outputs[0], &ny);
        if (st != VELRI_OK) return st;
        if (na != ny || !no_partial_overlap(&op->inputs[0], &op->outputs[0])) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 1U && op->input_present[1]) {
            size_t nb = 0U;
            st = check_elemwise_f32_view(&op->inputs[1], &nb);
            if (st != VELRI_OK) return st;
            if (nb != na || !no_partial_overlap(&op->inputs[1], &op->outputs[0])) {
                return VELRI_ERR_INVALID;
            }
        } else if (op->raw.attr.axpby.beta != 0.0f) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_SOFTMAX: {
        if (op->num_inputs != 1U || op->num_outputs != 1U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, y) || !no_partial_overlap(x, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_FWHT: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        size_t bs = op->raw.attr.fwht.block_size;
        if (bs != 16U && bs != 32U && bs != 64U && bs != 128U && bs != 256U) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, y) || !no_partial_overlap(x, y) ||
            (x->dims[1] % bs) != 0U) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 1U && op->input_present[1]) {
            velri_dtype sdt = op->raw.attr.fwht.suh_dtype;
            if (sdt != VELRI_DTYPE_F32 && sdt != VELRI_DTYPE_F16) {
                return VELRI_ERR_UNSUPPORTED;
            }
            if (!check_opt_1d_param(op, 1U, sdt, x->dims[1], y, nullptr)) {
                return VELRI_ERR_INVALID;
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_QUANTIZE_ROW: {
        if (op->num_inputs != 1U || op->num_outputs != 2U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *q = &op->outputs[0];
        const velri_tensor_view *s = &op->outputs[1];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(q) || !is_unit_1d(s) ||
            x->dtype != VELRI_DTYPE_F32 || q->dtype != VELRI_DTYPE_I8 ||
            s->dtype != VELRI_DTYPE_F32 ||
            x->dims[0] != q->dims[0] || x->dims[1] != q->dims[1] || s->dims[0] != x->dims[0]) {
            return VELRI_ERR_INVALID;
        }
        if (views_overlap(x, q) || views_overlap(x, s) || views_overlap(q, s)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_EMBEDDING: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *table = &op->inputs[0];
        const velri_tensor_view *out = &op->outputs[0];
        if (!is_unit_inner_2d(table) || !is_unit_inner_2d(out) ||
            table->dtype != VELRI_DTYPE_F32 || out->dtype != VELRI_DTYPE_F32 ||
            table->dims[0] == 0U || out->dims[0] == 0U || table->dims[1] != out->dims[1] ||
            views_overlap(table, out)) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs == 2U && op->input_present[1]) {
            const velri_tensor_view *ids = &op->inputs[1];
            if (!is_unit_1d(ids) || ids->dtype != VELRI_DTYPE_I32 ||
                ids->dims[0] != out->dims[0] || views_overlap(ids, out)) {
                return VELRI_ERR_INVALID;
            }
        } else {
            size_t end_pos = 0U;
            if (!fits_u32(op->raw.attr.embedding.pos_offset) ||
                velri_add_overflow_size(op->raw.attr.embedding.pos_offset, out->dims[0], &end_pos)) {
                return VELRI_ERR_OVERFLOW;
            }
            if (end_pos > table->dims[0]) {
                return VELRI_ERR_INVALID;
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_ATTENTION: {
        if (op->num_inputs != 3U || op->num_outputs != 3U ||
            !op->input_present[0] || !op->input_present[1] || !op->input_present[2]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *q = &op->inputs[0];
        const velri_tensor_view *k_in = &op->inputs[1];
        const velri_tensor_view *v_in = &op->inputs[2];
        const velri_tensor_view *out = &op->outputs[0];
        const velri_tensor_view *k_cache = &op->outputs[1];
        const velri_tensor_view *v_cache = &op->outputs[2];

        uint32_t nh = op->raw.attr.attention.n_heads;
        uint32_t nkvh = op->raw.attr.attention.n_kv_heads;
        uint32_t dk = op->raw.attr.attention.head_dim;
        uint32_t dv = op->raw.attr.attention.v_head_dim ? op->raw.attr.attention.v_head_dim : dk;
        float sc = op->raw.attr.attention.scale;
        if (nh == 0U || nkvh == 0U || dk == 0U || dk > VELRI_ATTN_MAX_HEAD_DIM ||
            dv == 0U || dv > VELRI_ATTN_MAX_HEAD_DIM || (nh % nkvh) != 0U) {
            return VELRI_ERR_INVALID;
        }
        if (sc != 0.0f && (!(sc > 0.0f) || !isfinite(sc))) {
            return VELRI_ERR_INVALID;
        }
        if (!is_unit_inner_2d(q) || !is_unit_inner_2d(k_in) || !is_unit_inner_2d(v_in) ||
            !is_unit_inner_2d(out)) {
            return VELRI_ERR_INVALID;
        }
        if (q->dtype != VELRI_DTYPE_F32 || k_in->dtype != VELRI_DTYPE_F32 ||
            v_in->dtype != VELRI_DTYPE_F32 || out->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_UNSUPPORTED;
        }
        size_t max_t = q->dims[0];
        if (max_t == 0U || k_in->dims[0] != max_t || v_in->dims[0] != max_t ||
            out->dims[0] != max_t) {
            return VELRI_ERR_INVALID;
        }
        size_t q_dim = 0U, o_dim = 0U, k_dim = 0U, v_dim = 0U;
        if (velri_mul_overflow_size((size_t)nh, (size_t)dk, &q_dim) ||
            velri_mul_overflow_size((size_t)nh, (size_t)dv, &o_dim) ||
            velri_mul_overflow_size((size_t)nkvh, (size_t)dk, &k_dim) ||
            velri_mul_overflow_size((size_t)nkvh, (size_t)dv, &v_dim) ||
            !fits_u32(q_dim) || !fits_u32(o_dim) || !fits_u32(k_dim) || !fits_u32(v_dim)) {
            return VELRI_ERR_OVERFLOW;
        }
        if (q->dims[1] != q_dim || out->dims[1] != o_dim ||
            k_in->dims[1] != k_dim || v_in->dims[1] != v_dim) {
            return VELRI_ERR_INVALID;
        }
        if (k_cache->rank != 3U || v_cache->rank != 3U ||
            k_cache->dtype != VELRI_DTYPE_F32 || v_cache->dtype != VELRI_DTYPE_F32 ||
            k_cache->strides[2] != 1U || v_cache->strides[2] != 1U ||
            k_cache->strides[1] < dk || v_cache->strides[1] < dv) {
            return VELRI_ERR_INVALID;
        }
        size_t max_ctx = k_cache->dims[1];
        if (k_cache->dims[0] != nkvh || v_cache->dims[0] != nkvh ||
            k_cache->dims[2] != dk || v_cache->dims[2] != dv ||
            v_cache->dims[1] != max_ctx || max_ctx < max_t ||
            !fits_u32(max_ctx) || !fits_u32(k_cache->strides[0]) ||
            !fits_u32(k_cache->strides[1]) || !fits_u32(v_cache->strides[0]) ||
            !fits_u32(v_cache->strides[1])) {
            return VELRI_ERR_INVALID;
        }
        size_t k_min_hs = 0U, v_min_hs = 0U;
        if (velri_mul_overflow_size(max_ctx, k_cache->strides[1], &k_min_hs) ||
            velri_mul_overflow_size(max_ctx, v_cache->strides[1], &v_min_hs)) {
            return VELRI_ERR_OVERFLOW;
        }
        if (k_cache->strides[0] < k_min_hs || v_cache->strides[0] < v_min_hs) {
            return VELRI_ERR_INVALID;
        }
        if ((views_overlap(out, q) && !(dk == dv && views_identical_layout(q, out))) ||
            views_overlap(out, k_in) || views_overlap(out, v_in) ||
            views_overlap(out, k_cache) || views_overlap(out, v_cache) ||
            views_overlap(k_cache, q) || views_overlap(k_cache, k_in) ||
            views_overlap(k_cache, v_in) || views_overlap(v_cache, q) ||
            views_overlap(v_cache, k_in) || views_overlap(v_cache, v_in)) {
            return VELRI_ERR_INVALID;
        }
        /* Allow exact (K==V==H) or prefix-aliased (K=[c_kv|k_pe], V=c_kv for nkvh==1) shared cache. */
        if (views_overlap(k_cache, v_cache)) {
            bool cache_prefix_ok = (k_cache->storage_id == v_cache->storage_id &&
                                    k_cache->byte_offset == v_cache->byte_offset &&
                                    k_cache->strides[0] == v_cache->strides[0] &&
                                    k_cache->strides[1] == v_cache->strides[1] && dv <= dk);
            bool in_prefix_ok = (k_in->storage_id == v_in->storage_id &&
                                 k_in->byte_offset == v_in->byte_offset &&
                                 k_in->strides[0] == v_in->strides[0] &&
                                 (dv == dk || nkvh == 1U));
            if (!cache_prefix_ok || !in_prefix_ok) {
                return VELRI_ERR_INVALID;
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_COPY_LAST_ROW: {
        if (op->num_inputs != 1U || op->num_outputs != 1U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(y) ||
            x->dtype != VELRI_DTYPE_F32 || y->dtype != VELRI_DTYPE_F32 ||
            x->dims[0] == 0U || y->dims[0] != 1U || x->dims[1] != y->dims[1] ||
            views_overlap(x, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_BATCHED_LINEAR: {
        if (op->num_inputs < 2U || op->num_inputs > 3U || op->num_outputs != 1U ||
            !op->input_present[0] || !op->input_present[1]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *w = &op->inputs[1];
        const velri_tensor_view *y = &op->outputs[0];

        uint32_t b = op->raw.attr.batched_linear.n_heads;
        uint32_t g = op->raw.attr.batched_linear.n_groups;
        uint32_t k = op->raw.attr.batched_linear.in_dim;
        uint32_t m = op->raw.attr.batched_linear.out_dim;
        bool wt = op->raw.attr.batched_linear.weight_transposed;
        if (b == 0U || g == 0U || k == 0U || m == 0U || (b % g) != 0U) {
            return VELRI_ERR_INVALID;
        }
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(w) || !is_unit_inner_2d(y)) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != VELRI_DTYPE_F32 || w->dtype != VELRI_DTYPE_F32 ||
            y->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_UNSUPPORTED;
        }
        size_t bk = 0U, bm = 0U, gm = 0U, gk = 0U;
        if (velri_mul_overflow_size((size_t)b, (size_t)k, &bk) ||
            velri_mul_overflow_size((size_t)b, (size_t)m, &bm) ||
            velri_mul_overflow_size((size_t)g, (size_t)m, &gm) ||
            velri_mul_overflow_size((size_t)g, (size_t)k, &gk) ||
            !fits_u32(bk) || !fits_u32(bm) || !fits_u32(gm) || !fits_u32(gk)) {
            return VELRI_ERR_OVERFLOW;
        }
        size_t max_t = x->dims[0];
        if (max_t == 0U || y->dims[0] != max_t || x->dims[1] != bk || y->dims[1] != bm) {
            return VELRI_ERR_INVALID;
        }
        size_t w_rows = wt ? gk : gm;
        size_t w_cols = wt ? (size_t)m : (size_t)k;
        if (w->dims[0] != w_rows || w->dims[1] != w_cols) {
            return VELRI_ERR_INVALID;
        }
        if (!check_opt_1d_param(op, 2U, VELRI_DTYPE_F32, gm, y, nullptr)) {
            return VELRI_ERR_INVALID;
        }
        if (views_overlap(y, x) || views_overlap(y, w)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    default:
        return VELRI_ERR_UNSUPPORTED;
    }
}
