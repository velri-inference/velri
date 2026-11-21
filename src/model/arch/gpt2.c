#include "gpt2.h"

static const velri_pre_norm_layer_desc k_gpt2_layer = {
    .norm_kind = VELRI_NORM_LAYER,
    .norm_bias = true,
    .fused_qkv = true,
    .attn_bias = true,
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

velri_status velri_arch_gpt2_build(velri_builder *b, const velri_model_params *p) {
    velri_tensor x = velri_layer_embedding(b, p, "transformer.wte.weight", "transformer.wpe.weight");
    x = velri_layers_pre_norm_repeat(b, x, p, &k_gpt2_layer);
    (void)velri_layer_lm_head(b, x, p, "transformer.ln_f", VELRI_NORM_LAYER, true, false,
                              "lm_head.weight", "transformer.wte.weight");
    return velri_b_status(b);
}
