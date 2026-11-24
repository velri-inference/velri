#include "model/arch_registry.h"

#include <math.h>
#include <stdio.h>

static velri_rc validate_desc(velri_model_desc *norm, char *err_buf, size_t err_cap) {
    if (norm->rope_theta == 0.0f) {
        norm->rope_theta = 10000.0f;
    }
    if (!isfinite(norm->rope_theta) || norm->rope_theta <= 0.0f) {
        snprintf(err_buf, err_cap, "rope_theta must be finite and > 0");
        return VELRI_RC_ERR_INVALID;
    }
    if (norm->rotary_dim == 0U) {
        norm->rotary_dim = norm->head_dim;
    }
    if (norm->rotary_dim > norm->head_dim || (norm->rotary_dim & 1U) != 0U) {
        snprintf(err_buf, err_cap, "rotary_dim (%u) must be even and <= head_dim (%u)",
                 norm->rotary_dim, norm->head_dim);
        return VELRI_RC_ERR_INVALID;
    }
    return VELRI_RC_OK;
}

static const velri_arch_policy arch_policy = {
    .has_auto_rope = true,
    .validate_desc = validate_desc,
};

static const velri_lm_head_desc k_llama_lm_head = {
    .norm_name      = "model.norm",
    .norm_kind      = VELRI_NORM_RMS,
    .norm_bias      = false,
    .weight_name    = "lm_head.weight",
    .tie_to_if_tied = "model.embed_tokens.weight",
};

/*
 * token_ids -> Embedding(embed_tokens)
 *   -> N x Pre-RMSNorm(Split Q/K/V + RoPE + GQA Causal Attention + SwiGLU MLP)
 *   -> LMHead(model.norm, lm_head or tied embed_tokens)
 *
 * Set p->qkv_bias for Qwen2-style Q/K/V biases.
 */
static velri_status build_graph(velri_builder *b, const velri_model_params *p) {
    if (!p) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return velri_b_status(b);
    }
    const velri_pre_norm_layer_desc layer_desc = {
        .norm_kind = VELRI_NORM_RMS,
        .norm_bias = false,
        .has_rope  = true,
        .fused_qkv = false,
        .qkv_bias  = p->qkv_bias,
        .o_bias    = false,
        .mlp_gated = true,
        .mlp_bias  = false,
        .prefix    = "model.layers",
        .state_prefix = "self_attn",
        .norm1     = "input_layernorm",
        .norm2     = "post_attention_layernorm",
        .q_proj    = "self_attn.q_proj",
        .k_proj    = "self_attn.k_proj",
        .v_proj    = "self_attn.v_proj",
        .o_proj    = "self_attn.o_proj",
        .gate_proj = "mlp.gate_proj",
        .up_proj   = "mlp.up_proj",
        .down_proj = "mlp.down_proj",
    };

    velri_tensor x = velri_layer_embedding(b, p, "model.embed_tokens.weight", nullptr);
    x = velri_layers_pre_norm_repeat(b, x, p, &layer_desc, !k_llama_lm_head.full_seq_logits);
    (void)velri_layer_lm_head(b, x, p, &k_llama_lm_head);
    return velri_b_status(b);
}
