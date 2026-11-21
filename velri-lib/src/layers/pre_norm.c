#include "velri_lib/blocks.h"

#include <limits.h>
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

static velri_tensor apply_named_norm(velri_builder *b, velri_tensor x, const char *prefix,
                                     const char *norm_name, const velri_norm_spec *norm) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }

    char w_name[VELRI_MANIFEST_NAME_MAX];
    if (!format_tensor_name(b, w_name, prefix, norm_name, ".weight")) {
        return VELRI_TENSOR_NONE;
    }
    const size_t dim_norm[1] = {x.dims[1]};
    velri_tensor norm_w = velri_b_weight(b, w_name, x.dtype, 1U, dim_norm);

    if (norm->kind == VELRI_NORM_LAYER) {
        velri_tensor norm_b = VELRI_TENSOR_NONE;
        if (norm->has_bias) {
            char b_name[VELRI_MANIFEST_NAME_MAX];
            if (!format_tensor_name(b, b_name, prefix, norm_name, ".bias")) {
                return VELRI_TENSOR_NONE;
            }
            norm_b = velri_b_weight(b, b_name, x.dtype, 1U, dim_norm);
        }
        return velri_b_layernorm(b, x, norm_w, norm_b, norm->eps);
    }

    return velri_b_rmsnorm(b, x, norm_w, norm->eps, norm->rms_unit_offset);
}

velri_tensor velri_layer_pre_norm(velri_builder *b, velri_tensor x, const char *layer_prefix,
                                  uint32_t layer_idx, const velri_layer_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || !spec || !spec->norm1_name || spec->norm1_name[0] == '\0' ||
        !spec->norm2_name || spec->norm2_name[0] == '\0' || velri_dtype_bytes(spec->dtype) == 0U ||
        !(spec->norm.eps > 0.0f) || !isfinite(spec->norm.eps)) {
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
    if (layer_idx > (uint32_t)INT32_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    char layer_pfx[VELRI_MANIFEST_NAME_MAX];
    int n = 0;
    if (layer_prefix && layer_prefix[0] != '\0') {
        n = snprintf(layer_pfx, sizeof(layer_pfx), "%s.%u", layer_prefix, layer_idx);
    } else {
        n = snprintf(layer_pfx, sizeof(layer_pfx), "%u", layer_idx);
    }
    if (n < 0 || (size_t)n >= sizeof(layer_pfx)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    if (spec->scope) {
        velri_b_push_scope(b, spec->scope);
    }
    velri_b_push_layer(b, (int32_t)layer_idx);
    if (b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }

    /* Attention sub-block: push attention scope once around norm1 + attention + residual add. */
    if (spec->attn.scope) {
        velri_b_push_scope(b, spec->attn.scope);
    }
    velri_tensor h = apply_named_norm(b, x, layer_pfx, spec->norm1_name, &spec->norm);
    velri_attention_spec attn_spec = spec->attn;
    attn_spec.scope = nullptr;
    h = velri_block_attention(b, h, layer_pfx, &attn_spec);
    x = velri_b_add_inplace(b, x, h);
    if (spec->attn.scope) {
        velri_b_pop_scope(b);
    }

    /* MLP sub-block: push MLP scope once around norm2 + mlp + residual add. */
    if (spec->mlp.scope) {
        velri_b_push_scope(b, spec->mlp.scope);
    }
    h = apply_named_norm(b, x, layer_pfx, spec->norm2_name, &spec->norm);
    velri_mlp_spec mlp_spec = spec->mlp;
    mlp_spec.scope = nullptr;
    h = velri_block_mlp(b, h, layer_pfx, &mlp_spec);
    x = velri_b_add_inplace(b, x, h);
    if (spec->mlp.scope) {
        velri_b_pop_scope(b);
    }

    velri_b_pop_layer(b);
    if (spec->scope) {
        velri_b_pop_scope(b);
    }

    return (b->err == VELRI_OK) ? x : VELRI_TENSOR_NONE;
}

/* ---- Metadata-Driven High-Level Layer Composition Helpers ---- */

velri_tensor velri_layer_embedding(velri_builder *b, const velri_model_params *p,
                                   const char *token_weight, const char *pos_weight) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!p) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (pos_weight != nullptr && p->max_ctx > p->max_position_embeddings) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    const velri_embedding_spec spec = {
        .scope = "input-layer",
        .token_weight = token_weight,
        .pos_weight = pos_weight,
        .vocab_size = p->vocab_size,
        .max_pos = p->max_position_embeddings,
        .dim = p->hidden_size,
        .max_seq_len = p->max_seq_len,
        .dtype = p->dtype,
    };
    return velri_block_embedding(b, &spec);
}

velri_tensor velri_layers_pre_norm_repeat(velri_builder *b, velri_tensor x,
                                          const velri_model_params *p,
                                          const velri_pre_norm_layer_desc *desc) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!p || !desc || p->num_hidden_layers == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (p->latent_kv && desc->has_rope) {
        velri_b_set_error(b, VELRI_ERR_UNSUPPORTED);
        return VELRI_TENSOR_NONE;
    }

    const velri_layer_spec spec = {
        .scope = "hidden-layers",
        .norm1_name = desc->norm1,
        .norm2_name = desc->norm2,
        .norm = {
            .kind = desc->norm_kind,
            .eps = p->norm_eps,
            .has_bias = desc->norm_bias,
            .rms_unit_offset = desc->rms_unit_offset,
        },
        .attn = {
            .scope = "attention",
            .qkv_name = desc->qkv,
            .q_name = desc->q_proj,
            .k_name = desc->k_proj,
            .v_name = desc->v_proj,
            .o_name = desc->o_proj,
            .n_heads = p->num_attention_heads,
            .n_kv_heads = p->num_key_value_heads,
            .head_dim = p->head_dim,
            .max_ctx = p->max_ctx,
            .sliding_window = p->sliding_window,
            .causal = true,
            .fused_qkv = desc->fused_qkv,
            .has_bias = desc->attn_bias,
            .latent_kv = p->latent_kv,
            .dtype = p->dtype,
        },
        .mlp = {
            .scope = "dense-mlp",
            .gate_name = desc->gate_proj,
            .up_name = desc->up_proj,
            .down_name = desc->down_proj,
            .intermediate_size = p->intermediate_size,
            .act = p->hidden_act,
            .clamp_limit = desc->mlp_clamp,
            .is_gated = desc->mlp_gated,
            .has_bias = desc->mlp_bias,
            .dtype = p->dtype,
        },
        .dtype = p->dtype,
    };

    for (uint32_t i = 0U; i < p->num_hidden_layers; ++i) {
        x = velri_layer_pre_norm(b, x, desc->prefix, i, &spec);
    }
    return (b->err == VELRI_OK) ? x : VELRI_TENSOR_NONE;
}

velri_tensor velri_layer_lm_head(velri_builder *b, velri_tensor x, const velri_model_params *p,
                                 const char *norm_name, velri_norm_kind norm_kind, bool norm_bias,
                                 bool rms_unit_offset, const char *weight_name,
                                 const char *tie_to_if_tied) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!p) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    const velri_lm_head_spec spec = {
        .scope = "lm-head",
        .norm_name = norm_name,
        .norm = {
            .kind = norm_kind,
            .eps = p->norm_eps,
            .has_bias = norm_bias,
            .rms_unit_offset = rms_unit_offset,
        },
        .weight_name = weight_name,
        .tie_to = p->tie_word_embeddings ? tie_to_if_tied : nullptr,
        .vocab_size = p->vocab_size,
        .dim = p->hidden_size,
        .dtype = p->dtype,
    };
    return velri_block_lm_head(b, x, &spec);
}
