#include "model/layers.h"

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
    /* Reject RoPE layers explicitly until RoPE is wired into the attention block. */
    if (desc->has_rope) {
        velri_b_set_error(b, VELRI_ERR_UNSUPPORTED);
        return VELRI_TENSOR_NONE;
    }

    const velri_layer_spec spec = {
        .scope = "hidden-layers",
        .attn_scope = "attention",
        .mlp_scope = "dense-mlp",
        .norm1_name = desc->norm1,
        .norm2_name = desc->norm2,
        .norm = {
            .kind = desc->norm_kind,
            .eps = p->norm_eps,
            .has_bias = desc->norm_bias,
            .rms_unit_offset = desc->rms_unit_offset,
        },
        .attn = {
            .scope = nullptr,
            .kv_cache_scope = "kv-cache",
            .state_prefix = desc->state_prefix,
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
            .qkv_bias = desc->qkv_bias,
            .o_bias = desc->o_bias,
            .latent_kv = p->latent_kv,
            .dtype = p->dtype,
        },
        .mlp = {
            .scope = nullptr,
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
    };

    for (uint32_t i = 0U; i < p->num_hidden_layers; ++i) {
        x = velri_layer_pre_norm(b, x, desc->prefix, i, &spec);
        if (b->err != VELRI_OK) {
            return VELRI_TENSOR_NONE;
        }
    }
    return (b->err == VELRI_OK) ? x : VELRI_TENSOR_NONE;
}

velri_tensor velri_layer_lm_head(velri_builder *b, velri_tensor x, const velri_model_params *p,
                                 const velri_lm_head_desc *desc) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!p || !desc) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    const velri_lm_head_spec spec = {
        .scope = "lm-head",
        .norm_name = desc->norm_name,
        .norm = {
            .kind = desc->norm_kind,
            .eps = p->norm_eps,
            .has_bias = desc->norm_bias,
            .rms_unit_offset = desc->rms_unit_offset,
        },
        .weight_name = desc->weight_name,
        .tie_to = p->tie_word_embeddings ? desc->tie_to_if_tied : nullptr,
        .vocab_size = p->vocab_size,
        .dim = p->hidden_size,
        .dtype = p->dtype,
        .full_seq_logits = desc->full_seq_logits,
        .has_bias = desc->has_bias,
    };
    return velri_block_lm_head(b, x, &spec);
}
