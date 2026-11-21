#include "velri_lib/blocks.h"

#include <math.h>
#include <stdio.h>

static bool format_tensor_name(velri_builder *b, char out[VELRI_MANIFEST_NAME_MAX],
                               const char *prefix, const char *sub, const char *suffix) {
    if (!sub || sub[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    const char *suf = suffix ? suffix : "";
    int n = 0;
    if (prefix && prefix[0] != '\0') {
        n = snprintf(out, VELRI_MANIFEST_NAME_MAX, "%s.%s%s", prefix, sub, suf);
    } else {
        n = snprintf(out, VELRI_MANIFEST_NAME_MAX, "%s%s", sub, suf);
    }
    if (n < 0 || (size_t)n >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return false;
    }
    return true;
}

velri_tensor velri_block_lm_head(velri_builder *b, velri_tensor x, const velri_lm_head_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || !spec || spec->vocab_size == 0U || spec->dim == 0U ||
        x.dims[1] != (size_t)spec->dim || velri_dtype_bytes(spec->dtype) == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->tie_to != nullptr) {
        if (spec->tie_to[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    } else if (!spec->weight_name || spec->weight_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->norm_name != nullptr) {
        if (spec->norm_name[0] == '\0' || !(spec->norm.eps > 0.0f) || !isfinite(spec->norm.eps)) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        if (spec->norm.kind != VELRI_NORM_LAYER && spec->norm.kind != VELRI_NORM_RMS) {
            velri_b_set_error(b, VELRI_ERR_UNSUPPORTED);
            return VELRI_TENSOR_NONE;
        }
        if (spec->norm.kind == VELRI_NORM_RMS && spec->norm.has_bias) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    }

    /* Slice the final active sequence row on `x.device_id` before entering `lm-head` scope. */
    velri_tensor h = velri_b_copy_last_row(b, x);

    if (spec->scope) {
        velri_b_push_scope(b, spec->scope);
        if (b->err != VELRI_OK) {
            return VELRI_TENSOR_NONE;
        }
    }

    if (spec->norm_name != nullptr) {
        char w_name[VELRI_MANIFEST_NAME_MAX];
        if (!format_tensor_name(b, w_name, nullptr, spec->norm_name, ".weight")) {
            return VELRI_TENSOR_NONE;
        }
        const size_t dim_norm[1] = {spec->dim};
        velri_tensor norm_w = velri_b_weight(b, w_name, h.dtype, 1U, dim_norm);

        if (spec->norm.kind == VELRI_NORM_LAYER) {
            velri_tensor norm_b = VELRI_TENSOR_NONE;
            if (spec->norm.has_bias) {
                char b_name[VELRI_MANIFEST_NAME_MAX];
                if (!format_tensor_name(b, b_name, nullptr, spec->norm_name, ".bias")) {
                    return VELRI_TENSOR_NONE;
                }
                norm_b = velri_b_weight(b, b_name, h.dtype, 1U, dim_norm);
            }
            h = velri_b_layernorm(b, h, norm_w, norm_b, spec->norm.eps);
        } else {
            h = velri_b_rmsnorm(b, h, norm_w, spec->norm.eps, spec->norm.rms_unit_offset);
        }
    }

    const size_t dim_w[2] = {spec->vocab_size, spec->dim};
    velri_tensor lm_head_w =
        velri_b_weight_tied(b, spec->weight_name, spec->tie_to, spec->dtype, 2U, dim_w);

    velri_dtype out_dtype =
        (h.dtype == VELRI_DTYPE_F64 && spec->dtype == VELRI_DTYPE_F64) ? VELRI_DTYPE_F64
                                                                       : VELRI_DTYPE_F32;
    const size_t dim_logits[2] = {1U, spec->vocab_size};
    velri_tensor logits = velri_b_output(b, out_dtype, 2U, dim_logits);
    velri_tensor out = velri_b_linear_into(b, h, lm_head_w, VELRI_TENSOR_NONE, logits);

    if (spec->scope) {
        velri_b_pop_scope(b);
    }

    if (b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (b->manifest) {
        b->manifest->logits_out = logits.storage_id;
    }
    return out;
}
