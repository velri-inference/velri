#ifndef VELRI_MODEL_ARCH_GPT2_H
#define VELRI_MODEL_ARCH_GPT2_H

#include "model/layers.h"

/*
 * GPT-2 Architecture Diagram:
 *   token_ids -> Embedding(wte + wpe) -> N x Pre-LN(Fused-QKV Causal MHA + MLP) -> LMHead(ln_f, wte)
 */
velri_status velri_arch_gpt2_build(velri_builder *b, const velri_model_params *params);

#endif /* VELRI_MODEL_ARCH_GPT2_H */
