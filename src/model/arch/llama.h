#ifndef VELRI_MODEL_ARCH_LLAMA_H
#define VELRI_MODEL_ARCH_LLAMA_H

#include "model/layers.h"

/*
 * Llama / Qwen2 Architecture Diagram:
 *   token_ids -> Embedding(embed_tokens)
 *     -> N x Pre-RMSNorm(Split Q/K/V + RoPE + GQA Causal Attention + SwiGLU MLP)
 *     -> LMHead(model.norm, lm_head or tied embed_tokens)
 *
 * Set `params->qkv_bias = true` for Qwen2-style Q/K/V biases, `false` for Llama.
 */
velri_status velri_arch_llama_build(velri_builder *b, const velri_model_params *params);

#endif /* VELRI_MODEL_ARCH_LLAMA_H */
