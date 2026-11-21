#include "core_internal.h"
#include <math.h>

/*
 * Central operation registry validator.
 * Device-agnostic: checks shapes, dtypes, strides, and aliasing rules once across all backends.
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

static bool is_unit_inner_2d(const velri_tensor_view *v) {
    return v->rank == 2U && v->strides[1] == 1U && v->strides[0] >= v->dims[1];
}

static bool is_unit_1d(const velri_tensor_view *v) {
    return v->rank == 1U && v->strides[0] == 1U;
}

velri_status velri_op_validate(const velri_resolved_op *op) {
    if (!op) {
        return VELRI_ERR_INVALID;
    }
    if (op->num_inputs > VELRI_OP_MAX_INPUTS || op->num_outputs == 0U ||
        op->num_outputs > VELRI_OP_MAX_OUTPUTS) {
        return VELRI_ERR_INVALID;
    }

    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        /* inputs: 0=x [rows,k], 1=w [m,k], 2=bias (opt [m]), 3=x_scale (opt [rows]), 4=w_scale (opt [m])
         * outputs: 0=y [rows,m] */
        if (op->num_inputs < 2U || op->num_inputs > 5U || op->num_outputs != 1U) {
            return VELRI_ERR_INVALID;
        }
        if (!op->input_present[0] || !op->input_present[1]) {
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
        if (x->dims[1] != w->dims[1] || x->dims[0] != y->dims[0] || w->dims[0] != y->dims[1]) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 2U && op->input_present[2]) {
            const velri_tensor_view *b = &op->inputs[2];
            if (!is_unit_1d(b) || b->dims[0] != w->dims[0] || b->dtype != d->y_dtype) {
                return VELRI_ERR_INVALID;
            }
        }
        if (d->has_x_scale) {
            if (op->num_inputs <= 3U || !op->input_present[3] ||
                !is_unit_1d(&op->inputs[3]) || op->inputs[3].dtype != VELRI_DTYPE_F32 ||
                op->inputs[3].dims[0] != x->dims[0]) {
                return VELRI_ERR_INVALID;
            }
        }
        if (d->has_w_scale) {
            if (op->num_inputs <= 4U || !op->input_present[4] ||
                !is_unit_1d(&op->inputs[4]) || op->inputs[4].dtype != VELRI_DTYPE_F32 ||
                op->inputs[4].dims[0] != w->dims[0]) {
                return VELRI_ERR_INVALID;
            }
        }
        if (views_overlap(x, y) || views_overlap(w, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_RMSNORM: {
        /* inputs: 0=x [rows,dim], 1=weight (opt [dim]); outputs: 0=y [rows,dim] */
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(y) ||
            x->dtype != y->dtype || x->dims[0] != y->dims[0] || x->dims[1] != y->dims[1]) {
            return VELRI_ERR_INVALID;
        }
        if (!(op->raw.attr.rmsnorm.eps > 0.0f) || !isfinite(op->raw.attr.rmsnorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 1U && op->input_present[1]) {
            const velri_tensor_view *w = &op->inputs[1];
            if (!is_unit_1d(w) || w->dtype != x->dtype || w->dims[0] != x->dims[1]) {
                return VELRI_ERR_INVALID;
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_FUSED_ADD_RMSNORM: {
        /* inputs: 0=x [rows,dim], 1=weight (opt [dim]); outputs: 0=residual_inout [rows,dim], 1=y [rows,dim] */
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 2U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *r = &op->outputs[0];
        const velri_tensor_view *y = &op->outputs[1];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(r) || !is_unit_inner_2d(y)) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != VELRI_DTYPE_F32 || r->dtype != VELRI_DTYPE_F32 || y->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_INVALID;
        }
        if (x->dims[0] != r->dims[0] || x->dims[1] != r->dims[1] ||
            x->dims[0] != y->dims[0] || x->dims[1] != y->dims[1]) {
            return VELRI_ERR_INVALID;
        }
        if (!(op->raw.attr.rmsnorm.eps > 0.0f) || !isfinite(op->raw.attr.rmsnorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 1U && op->input_present[1]) {
            const velri_tensor_view *w = &op->inputs[1];
            if (!is_unit_1d(w) || w->dtype != VELRI_DTYPE_F32 || w->dims[0] != x->dims[1]) {
                return VELRI_ERR_INVALID;
            }
        }
        if (views_overlap(r, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_LAYERNORM: {
        /* inputs: 0=x [rows,dim], 1=w (opt), 2=b (opt); outputs: 0=y [rows,dim] */
        if (op->num_inputs < 1U || op->num_inputs > 3U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(y) ||
            x->dtype != VELRI_DTYPE_F32 || y->dtype != VELRI_DTYPE_F32 ||
            x->dims[0] != y->dims[0] || x->dims[1] != y->dims[1]) {
            return VELRI_ERR_INVALID;
        }
        if (!(op->raw.attr.layernorm.eps > 0.0f) || !isfinite(op->raw.attr.layernorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        for (uint32_t i = 1U; i < op->num_inputs; ++i) {
            if (op->input_present[i]) {
                const velri_tensor_view *p = &op->inputs[i];
                if (!is_unit_1d(p) || p->dtype != VELRI_DTYPE_F32 || p->dims[0] != x->dims[1]) {
                    return VELRI_ERR_INVALID;
                }
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_ROPE: {
        /* inputs: 0=x [n_tokens,n_heads,head_dim], 1=cos_tbl [>=pos+nt, rotary/2], 2=sin_tbl;
         * outputs: 0=y [n_tokens,n_heads,head_dim] */
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
        size_t rd = op->raw.attr.rope.rotary_dim;
        if (rd == 0U || (rd & 1U) != 0U || rd > x->dims[2]) {
            return VELRI_ERR_INVALID;
        }
        if (!velri_tensor_view_is_contiguous(ct) || !velri_tensor_view_is_contiguous(st) ||
            ct->dtype != VELRI_DTYPE_F32 || st->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_INVALID;
        }
        size_t need = (op->raw.attr.rope.pos_offset + x->dims[0]) * (rd / 2U);
        size_t ct_n = 0U, st_n = 0U;
        if (velri_tensor_view_num_elements(ct, &ct_n) != VELRI_OK ||
            velri_tensor_view_num_elements(st, &st_n) != VELRI_OK ||
            ct_n < need || st_n < need) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_ACTIVATION: {
        /* inputs: 0=x; outputs: 0=y (contiguous F32 same count) */
        if (op->num_inputs != 1U || op->num_outputs != 1U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        size_t nx = 0U, ny = 0U;
        if (!velri_tensor_view_is_contiguous(x) || !velri_tensor_view_is_contiguous(y) ||
            x->dtype != VELRI_DTYPE_F32 || y->dtype != VELRI_DTYPE_F32 ||
            velri_tensor_view_num_elements(x, &nx) != VELRI_OK ||
            velri_tensor_view_num_elements(y, &ny) != VELRI_OK || nx != ny) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_GATED_MLP:
    case VELRI_OP_VEC_MUL: {
        /* inputs: 0=a/gate, 1=b/up; outputs: 0=y */
        if (op->num_inputs != 2U || op->num_outputs != 1U ||
            !op->input_present[0] || !op->input_present[1]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *a = &op->inputs[0];
        const velri_tensor_view *b = &op->inputs[1];
        const velri_tensor_view *y = &op->outputs[0];
        size_t na = 0U, nb = 0U, ny = 0U;
        if (!velri_tensor_view_is_contiguous(a) || !velri_tensor_view_is_contiguous(b) ||
            !velri_tensor_view_is_contiguous(y) ||
            a->dtype != VELRI_DTYPE_F32 || b->dtype != VELRI_DTYPE_F32 ||
            y->dtype != VELRI_DTYPE_F32 ||
            velri_tensor_view_num_elements(a, &na) != VELRI_OK ||
            velri_tensor_view_num_elements(b, &nb) != VELRI_OK ||
            velri_tensor_view_num_elements(y, &ny) != VELRI_OK ||
            na != nb || na != ny) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_VEC_AXPBY: {
        /* inputs: 0=a, 1=b (opt if beta == 0); outputs: 0=y */
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *a = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        size_t na = 0U, ny = 0U;
        if (!velri_tensor_view_is_contiguous(a) || !velri_tensor_view_is_contiguous(y) ||
            a->dtype != VELRI_DTYPE_F32 || y->dtype != VELRI_DTYPE_F32 ||
            velri_tensor_view_num_elements(a, &na) != VELRI_OK ||
            velri_tensor_view_num_elements(y, &ny) != VELRI_OK || na != ny) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 1U && op->input_present[1]) {
            const velri_tensor_view *b = &op->inputs[1];
            size_t nb = 0U;
            if (!velri_tensor_view_is_contiguous(b) || b->dtype != VELRI_DTYPE_F32 ||
                velri_tensor_view_num_elements(b, &nb) != VELRI_OK || nb != na) {
                return VELRI_ERR_INVALID;
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_SOFTMAX: {
        /* inputs: 0=x [rows,cols]; outputs: 0=y [rows,cols] */
        if (op->num_inputs != 1U || op->num_outputs != 1U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(y) ||
            x->dtype != VELRI_DTYPE_F32 || y->dtype != VELRI_DTYPE_F32 ||
            x->dims[0] != y->dims[0] || x->dims[1] != y->dims[1]) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_FWHT: {
        /* inputs: 0=x [rows,cols], 1=suh (opt [cols]); outputs: 0=y [rows,cols] */
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        size_t bs = op->raw.attr.fwht.block_size;
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(y) ||
            x->dtype != VELRI_DTYPE_F32 || y->dtype != VELRI_DTYPE_F32 ||
            x->dims[0] != y->dims[0] || x->dims[1] != y->dims[1] ||
            bs == 0U || (bs & (bs - 1U)) != 0U || (x->dims[1] % bs) != 0U) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 1U && op->input_present[1]) {
            const velri_tensor_view *suh = &op->inputs[1];
            if (!is_unit_1d(suh) || suh->dims[0] != x->dims[1] ||
                suh->dtype != op->raw.attr.fwht.suh_dtype) {
                return VELRI_ERR_INVALID;
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_QUANTIZE_ROW: {
        /* inputs: 0=x [rows,k] F32; outputs: 0=x_i8 [rows,k] I8, 1=scale_out [rows] F32 */
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
        return VELRI_OK;
    }

    default:
        return VELRI_ERR_UNSUPPORTED;
    }
}
