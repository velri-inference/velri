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
    if (out_b) {
        *out_b = VELRI_TENSOR_NONE;
    }

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
        if (!out_b) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return false;
        }
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

static bool validate_attention_common(velri_builder *b, velri_tensor x,
                                      const velri_attention_spec *spec, size_t *out_q_dim,
                                      size_t *out_kv_dim, size_t *out_qkv_dim) {
    if (!x.valid || x.rank != 2U || !spec || spec->n_heads == 0U || spec->n_kv_heads == 0U ||
        spec->head_dim == 0U || spec->head_dim > VELRI_ATTN_MAX_HEAD_DIM ||
        (spec->n_heads % spec->n_kv_heads) != 0U || spec->max_ctx == 0U ||
        x.dims[0] > (size_t)spec->max_ctx || velri_dtype_bytes(spec->dtype) == 0U ||
        !spec->o_name || spec->o_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    if (spec->scale != 0.0f && (!(spec->scale > 0.0f) || !isfinite(spec->scale))) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    if (spec->fused_qkv) {
        if (!spec->qkv_name || spec->qkv_name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return false;
        }
    } else {
        if (!spec->q_name || spec->q_name[0] == '\0' || !spec->k_name || spec->k_name[0] == '\0' ||
            !spec->v_name || spec->v_name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return false;
        }
    }

    size_t q_dim = 0U;
    size_t kv_dim = 0U;
    if (velri_mul_overflow_size((size_t)spec->n_heads, (size_t)spec->head_dim, &q_dim) ||
        velri_mul_overflow_size((size_t)spec->n_kv_heads, (size_t)spec->head_dim, &kv_dim) ||
        q_dim > (size_t)UINT32_MAX || kv_dim > (size_t)UINT32_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return false;
    }
    size_t qkv_dim = 0U;
    if (spec->fused_qkv) {
        size_t two_kv = 0U;
        if (velri_mul_overflow_size(2U, kv_dim, &two_kv) ||
            velri_add_overflow_size(q_dim, two_kv, &qkv_dim) || qkv_dim > (size_t)UINT32_MAX) {
            velri_b_set_error(b, VELRI_ERR_OVERFLOW);
            return false;
        }
    }

    *out_q_dim = q_dim;
    *out_kv_dim = kv_dim;
    *out_qkv_dim = qkv_dim;
    return true;
}

static velri_tensor apply_output_proj(velri_builder *b, velri_tensor attn, const char *prefix,
                                      const velri_attention_spec *spec, velri_dtype act_dtype,
                                      size_t d_model, size_t q_dim) {
    velri_tensor o_w = VELRI_TENSOR_NONE;
    velri_tensor o_b = VELRI_TENSOR_NONE;
    if (!make_proj_weights(b, prefix, spec->o_name, spec->dtype, act_dtype, d_model, q_dim,
                           spec->has_bias, &o_w, &o_b)) {
        return VELRI_TENSOR_NONE;
    }
    return velri_b_linear(b, attn, o_w, o_b);
}

velri_tensor velri_block_latent_attention(velri_builder *b, velri_tensor x, velri_tensor h_lat,
                                          const char *prefix, const velri_attention_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }

    size_t q_dim = 0U, kv_dim = 0U, qkv_dim = 0U;
    if (!validate_attention_common(b, x, spec, &q_dim, &kv_dim, &qkv_dim)) {
        return VELRI_TENSOR_NONE;
    }
    if (!h_lat.valid || h_lat.rank != 2U || h_lat.dims[0] != x.dims[0] || h_lat.dims[1] == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (h_lat.dims[1] > VELRI_LATENT_ATTN_MAX_DLAT) {
        velri_b_set_error(b, VELRI_ERR_UNSUPPORTED);
        return VELRI_TENSOR_NONE;
    }
    /* Fused [Q|K|V] projection expects a single shared input tensor (`h_lat == x`). */
    if (spec->fused_qkv && (h_lat.storage_id != x.storage_id || h_lat.view_id != x.view_id)) {
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
    size_t d_lat = h_lat.dims[1];
    velri_dtype act_dtype =
        (x.dtype == VELRI_DTYPE_F64 && spec->dtype == VELRI_DTYPE_F64) ? VELRI_DTYPE_F64
                                                                       : VELRI_DTYPE_F32;

    velri_tensor q = VELRI_TENSOR_NONE;
    velri_tensor w_k = VELRI_TENSOR_NONE;
    velri_tensor w_v = VELRI_TENSOR_NONE;
    velri_tensor b_v = VELRI_TENSOR_NONE;

    if (spec->fused_qkv) {
        /* Standard fused [Q | K | V] row-concatenated projection: slice W_Q, W_K, W_V zero-copy.
         * Key bias b_K is omitted by exact softmax shift invariance (q . b_K is key-independent). */
        velri_tensor qkv_w = VELRI_TENSOR_NONE;
        velri_tensor qkv_b = VELRI_TENSOR_NONE;
        if (!make_proj_weights(b, prefix, spec->qkv_name, spec->dtype, act_dtype, qkv_dim, d_model,
                               spec->has_bias, &qkv_w, &qkv_b)) {
            return VELRI_TENSOR_NONE;
        }
        velri_tensor w_q = velri_b_subview_2d_rows(b, qkv_w, 0U, q_dim);
        w_k = velri_b_subview_2d_rows(b, qkv_w, q_dim, kv_dim);
        w_v = velri_b_subview_2d_rows(b, qkv_w, q_dim + kv_dim, kv_dim);
        velri_tensor b_q = VELRI_TENSOR_NONE;
        if (spec->has_bias) {
            b_q = velri_b_subview_1d(b, qkv_b, 0U, q_dim);
            b_v = velri_b_subview_1d(b, qkv_b, q_dim + kv_dim, kv_dim);
        }
        q = velri_b_linear(b, x, w_q, b_q);
    } else {
        /* Split Q / K / V (also supports compressed MLA where d_lat != d_model). */
        velri_tensor q_w = VELRI_TENSOR_NONE;
        velri_tensor q_b = VELRI_TENSOR_NONE;
        if (!make_proj_weights(b, prefix, spec->q_name, spec->dtype, act_dtype, q_dim, d_model,
                               spec->has_bias, &q_w, &q_b) ||
            !make_proj_weights(b, prefix, spec->k_name, spec->dtype, act_dtype, kv_dim, d_lat,
                               false, &w_k, nullptr) ||
            !make_proj_weights(b, prefix, spec->v_name, spec->dtype, act_dtype, kv_dim, d_lat,
                               spec->has_bias, &w_v, &b_v)) {
            return VELRI_TENSOR_NONE;
        }
        q = velri_b_linear(b, x, q_w, q_b);
    }

    char h_cache_name[VELRI_MANIFEST_NAME_MAX];
    if (!format_tensor_name(b, h_cache_name, prefix, "attn.h_cache", nullptr)) {
        return VELRI_TENSOR_NONE;
    }
    const size_t dim_h[2] = {spec->max_ctx, d_lat};
    velri_tensor h_cache = velri_b_state(b, h_cache_name, h_lat.dtype, 2U, dim_h);

    velri_tensor attn = velri_b_latent_attention(b, q, h_lat, w_k, w_v, b_v, h_cache, spec->n_heads,
                                                 spec->n_kv_heads, spec->head_dim, spec->scale,
                                                 spec->sliding_window, spec->causal);
    velri_tensor out = apply_output_proj(b, attn, prefix, spec, act_dtype, d_model, q_dim);

    if (spec->scope) {
        velri_b_pop_scope(b);
    }
    return (b->err == VELRI_OK) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_block_attention(velri_builder *b, velri_tensor x, const char *prefix,
                                   const velri_attention_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (spec && spec->latent_kv) {
        return velri_block_latent_attention(b, x, x, prefix, spec);
    }

    size_t q_dim = 0U, kv_dim = 0U, qkv_dim = 0U;
    if (!validate_attention_common(b, x, spec, &q_dim, &kv_dim, &qkv_dim)) {
        return VELRI_TENSOR_NONE;
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
        velri_tensor q_w = VELRI_TENSOR_NONE, q_b = VELRI_TENSOR_NONE;
        velri_tensor k_w = VELRI_TENSOR_NONE, k_b = VELRI_TENSOR_NONE;
        velri_tensor v_w = VELRI_TENSOR_NONE, v_b = VELRI_TENSOR_NONE;
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
    velri_tensor out = apply_output_proj(b, attn, prefix, spec, act_dtype, d_model, q_dim);

    if (spec->scope) {
        velri_b_pop_scope(b);
    }
    return (b->err == VELRI_OK) ? out : VELRI_TENSOR_NONE;
}
