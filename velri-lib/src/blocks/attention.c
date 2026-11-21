#include "velri_lib/blocks.h"
#include "core_internal.h"

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

velri_tensor velri_block_attention(velri_builder *b, velri_tensor x, const char *prefix,
                                   const velri_attention_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || !spec || spec->n_heads == 0U || spec->n_kv_heads == 0U ||
        spec->head_dim == 0U || spec->head_dim > 256U || (spec->n_heads % spec->n_kv_heads) != 0U ||
        spec->max_ctx == 0U || x.dims[0] > (size_t)spec->max_ctx ||
        velri_dtype_bytes(spec->dtype) == 0U || !spec->o_name || spec->o_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->scale != 0.0f && (!(spec->scale > 0.0f) || !isfinite(spec->scale))) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->fused_qkv) {
        if (!spec->qkv_name || spec->qkv_name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    } else {
        if (!spec->q_name || spec->q_name[0] == '\0' || !spec->k_name || spec->k_name[0] == '\0' ||
            !spec->v_name || spec->v_name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    }

    size_t q_dim = 0U;
    size_t kv_dim = 0U;
    if (velri_mul_overflow_size((size_t)spec->n_heads, (size_t)spec->head_dim, &q_dim) ||
        velri_mul_overflow_size((size_t)spec->n_kv_heads, (size_t)spec->head_dim, &kv_dim) ||
        q_dim > (size_t)UINT32_MAX || kv_dim > (size_t)UINT32_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }
    size_t qkv_dim = 0U;
    if (spec->fused_qkv) {
        size_t two_kv = 0U;
        if (velri_mul_overflow_size(2U, kv_dim, &two_kv) ||
            velri_add_overflow_size(q_dim, two_kv, &qkv_dim) || qkv_dim > (size_t)UINT32_MAX) {
            velri_b_set_error(b, VELRI_ERR_OVERFLOW);
            return VELRI_TENSOR_NONE;
        }
    }

    if (spec->scope) {
        velri_b_push_scope(b, spec->scope);
        if (b->err != VELRI_OK) {
            return VELRI_TENSOR_NONE;
        }
    }

    size_t d_model = x.dims[1];
    velri_dtype act_dtype =
        (x.dtype == VELRI_DTYPE_F64 && spec->dtype == VELRI_DTYPE_F64) ? VELRI_DTYPE_F64
                                                                       : VELRI_DTYPE_F32;

    velri_tensor q = VELRI_TENSOR_NONE;
    velri_tensor k = VELRI_TENSOR_NONE;
    velri_tensor v = VELRI_TENSOR_NONE;

    if (spec->fused_qkv) {
        velri_tensor qkv_w = VELRI_TENSOR_NONE;
        velri_tensor qkv_b = VELRI_TENSOR_NONE;
        if (!make_proj_weights(b, prefix, spec->qkv_name, spec->dtype, act_dtype, qkv_dim, d_model,
                               spec->has_bias, &qkv_w, &qkv_b)) {
            return VELRI_TENSOR_NONE;
        }
        velri_tensor qkv = velri_b_linear(b, x, qkv_w, qkv_b);
        q = velri_b_subview_2d_cols(b, qkv, 0U, q_dim);
        k = velri_b_subview_2d_cols(b, qkv, q_dim, kv_dim);
        v = velri_b_subview_2d_cols(b, qkv, q_dim + kv_dim, kv_dim);
    } else {
        velri_tensor q_w = VELRI_TENSOR_NONE;
        velri_tensor q_b = VELRI_TENSOR_NONE;
        velri_tensor k_w = VELRI_TENSOR_NONE;
        velri_tensor k_b = VELRI_TENSOR_NONE;
        velri_tensor v_w = VELRI_TENSOR_NONE;
        velri_tensor v_b = VELRI_TENSOR_NONE;
        if (!make_proj_weights(b, prefix, spec->q_name, spec->dtype, act_dtype, q_dim, d_model,
                               spec->has_bias, &q_w, &q_b) ||
            !make_proj_weights(b, prefix, spec->k_name, spec->dtype, act_dtype, kv_dim, d_model,
                               spec->has_bias, &k_w, &k_b) ||
            !make_proj_weights(b, prefix, spec->v_name, spec->dtype, act_dtype, kv_dim, d_model,
                               spec->has_bias, &v_w, &v_b)) {
            return VELRI_TENSOR_NONE;
        }
        q = velri_b_linear(b, x, q_w, q_b);
        k = velri_b_linear(b, x, k_w, k_b);
        v = velri_b_linear(b, x, v_w, v_b);
    }

    char k_cache_name[VELRI_MANIFEST_NAME_MAX];
    char v_cache_name[VELRI_MANIFEST_NAME_MAX];
    if (!format_tensor_name(b, k_cache_name, prefix, "attn.k_cache", nullptr) ||
        !format_tensor_name(b, v_cache_name, prefix, "attn.v_cache", nullptr)) {
        return VELRI_TENSOR_NONE;
    }

    const size_t dim_kv[3] = {spec->n_kv_heads, spec->max_ctx, spec->head_dim};
    velri_tensor k_cache = velri_b_state(b, k_cache_name, act_dtype, 3U, dim_kv);
    velri_tensor v_cache = velri_b_state(b, v_cache_name, act_dtype, 3U, dim_kv);

    velri_tensor attn = velri_b_attention(b, q, k, v, k_cache, v_cache, spec->n_heads,
                                          spec->n_kv_heads, spec->head_dim, spec->scale,
                                          spec->sliding_window, spec->causal);

    velri_tensor o_w = VELRI_TENSOR_NONE;
    velri_tensor o_b = VELRI_TENSOR_NONE;
    if (!make_proj_weights(b, prefix, spec->o_name, spec->dtype, act_dtype, d_model, q_dim,
                           spec->has_bias, &o_w, &o_b)) {
        return VELRI_TENSOR_NONE;
    }
    velri_tensor out = velri_b_linear(b, attn, o_w, o_b);

    if (spec->scope) {
        velri_b_pop_scope(b);
    }

    return (b->err == VELRI_OK) ? out : VELRI_TENSOR_NONE;
}
