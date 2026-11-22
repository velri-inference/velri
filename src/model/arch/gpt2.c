#include "gpt2.h"

static const velri_pre_norm_layer_desc k_gpt2_layer = {
    .norm_kind = VELRI_NORM_LAYER,
    .norm_bias = true,
    .has_rope  = false,
    .fused_qkv = true,
    .qkv_bias  = true,
    .o_bias    = true,
    .mlp_gated = false,
    .mlp_bias  = true,
    .prefix    = "transformer.h",
    .norm1     = "ln_1",
    .norm2     = "ln_2",
    .qkv       = "attn.c_attn",
    .o_proj    = "attn.c_proj",
    .up_proj   = "mlp.c_fc",
    .down_proj = "mlp.c_proj",
};

static const velri_lm_head_desc k_gpt2_lm_head = {
    .norm_name      = "transformer.ln_f",
    .norm_kind      = VELRI_NORM_LAYER,
    .norm_bias      = true,
    .weight_name    = "lm_head.weight",
    .tie_to_if_tied = "transformer.wte.weight",
};

velri_status velri_arch_gpt2_build(velri_builder *b, const velri_model_params *p) {
    velri_tensor x = velri_layer_embedding(b, p, "transformer.wte.weight", "transformer.wpe.weight");
    x = velri_layers_pre_norm_repeat(b, x, p, &k_gpt2_layer, !k_gpt2_lm_head.full_seq_logits);
    (void)velri_layer_lm_head(b, x, p, &k_gpt2_lm_head);
    return velri_b_status(b);
}
