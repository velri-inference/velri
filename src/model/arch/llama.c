#include "llama.h"

static const velri_lm_head_desc k_llama_lm_head = {
    .norm_name      = "model.norm",
    .norm_kind      = VELRI_NORM_RMS,
    .norm_bias      = false,
    .weight_name    = "lm_head.weight",
    .tie_to_if_tied = "model.embed_tokens.weight",
};

velri_status velri_arch_llama_build(velri_builder *b, const velri_model_params *p) {
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
