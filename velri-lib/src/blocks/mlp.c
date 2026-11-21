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

static bool make_proj_weights(velri_builder *b, const char *prefix, const char *proj_name,
                              velri_dtype w_dtype, velri_dtype b_dtype, size_t out_dim,
                              size_t in_dim, bool has_bias, velri_tensor *out_w,
                              velri_tensor *out_b) {
    *out_w = VELRI_TENSOR_NONE;
    *out_b = VELRI_TENSOR_NONE;

    char w_name[VELRI_MANIFEST_NAME_MAX];
    if (!format_tensor_name(b, w_name, prefix, proj_name, ".weight")) {
        return false;
    }
    const size_t dim_w[2] = {out_dim, in_dim};
    *out_w = velri_b_weight(b, w_name, w_dtype, 2U, dim_w);
    if (!out_w->valid) {
        return false;
    }

    if (has_bias) {
        char b_name[VELRI_MANIFEST_NAME_MAX];
        if (!format_tensor_name(b, b_name, prefix, proj_name, ".bias")) {
            return false;
        }
        const size_t dim_b[1] = {out_dim};
        *out_b = velri_b_weight(b, b_name, b_dtype, 1U, dim_b);
        if (!out_b->valid) {
            return false;
        }
    }
    return true;
}

velri_tensor velri_block_mlp(velri_builder *b, velri_tensor x, const char *prefix,
                             const velri_mlp_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || !spec || spec->intermediate_size == 0U ||
        velri_dtype_bytes(spec->dtype) == 0U || !spec->up_name || spec->up_name[0] == '\0' ||
        !spec->down_name || spec->down_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->is_gated && (!spec->gate_name || spec->gate_name[0] == '\0' ||
                           !isfinite(spec->clamp_limit) || spec->clamp_limit < 0.0f)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    if (spec->scope) {
        velri_b_push_scope(b, spec->scope);
        if (b->err != VELRI_OK) {
            return VELRI_TENSOR_NONE;
        }
    }

    size_t d_model = x.dims[1];
    size_t d_inter = (size_t)spec->intermediate_size;
    velri_dtype act_dtype =
        (x.dtype == VELRI_DTYPE_F64 && spec->dtype == VELRI_DTYPE_F64) ? VELRI_DTYPE_F64
                                                                       : VELRI_DTYPE_F32;

    velri_tensor h = VELRI_TENSOR_NONE;
    if (!spec->is_gated) {
        velri_tensor up_w = VELRI_TENSOR_NONE;
        velri_tensor up_b = VELRI_TENSOR_NONE;
        if (!make_proj_weights(b, prefix, spec->up_name, spec->dtype, act_dtype, d_inter, d_model,
                               spec->has_bias, &up_w, &up_b)) {
            return VELRI_TENSOR_NONE;
        }
        h = velri_b_linear(b, x, up_w, up_b);
        h = velri_b_activation_inplace(b, h, spec->act);
    } else {
        velri_tensor gate_w = VELRI_TENSOR_NONE;
        velri_tensor gate_b = VELRI_TENSOR_NONE;
        velri_tensor up_w = VELRI_TENSOR_NONE;
        velri_tensor up_b = VELRI_TENSOR_NONE;
        if (!make_proj_weights(b, prefix, spec->gate_name, spec->dtype, act_dtype, d_inter, d_model,
                               spec->has_bias, &gate_w, &gate_b) ||
            !make_proj_weights(b, prefix, spec->up_name, spec->dtype, act_dtype, d_inter, d_model,
                               spec->has_bias, &up_w, &up_b)) {
            return VELRI_TENSOR_NONE;
        }
        velri_tensor g = velri_b_linear(b, x, gate_w, gate_b);
        velri_tensor u = velri_b_linear(b, x, up_w, up_b);
        h = velri_b_gated_mlp_act_inplace(b, g, u, spec->act, spec->clamp_limit);
    }

    velri_tensor down_w = VELRI_TENSOR_NONE;
    velri_tensor down_b = VELRI_TENSOR_NONE;
    if (!make_proj_weights(b, prefix, spec->down_name, spec->dtype, act_dtype, d_model, d_inter,
                           spec->has_bias, &down_w, &down_b)) {
        return VELRI_TENSOR_NONE;
    }
    velri_tensor out = velri_b_linear(b, h, down_w, down_b);

    if (spec->scope) {
        velri_b_pop_scope(b);
    }

    return (b->err == VELRI_OK) ? out : VELRI_TENSOR_NONE;
}
