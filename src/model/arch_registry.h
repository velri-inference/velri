#ifndef VELRI_MODEL_ARCH_REGISTRY_H
#define VELRI_MODEL_ARCH_REGISTRY_H

#include "model/layers.h"
#include "velri/velri.h"

#include <stddef.h>
#include <stdint.h>

/*
 * Declarative Model Architecture Registry.
 *
 * Isolates all architecture-specific rules (validation, max_ctx bounds, RoPE auto-generation,
 * active placement scopes, and graph construction) behind a single descriptor table so `src/runtime/`
 * remains completely architecture-agnostic as new models (Qwen3.5, MoE, Gemma4, etc.) are added.
 */

typedef struct {
    velri_arch arch;
    const char *name;
    bool has_auto_rope;
    bool enforces_max_pos_ctx;

    const char *const *root_scopes;
    uint32_t num_root_scopes;
    const char *const *layer_scopes;
    uint32_t num_layer_scopes;

    velri_rc (*validate_desc)(velri_model_desc *norm, char *err_buf, size_t err_cap);
    velri_status (*build_graph)(velri_builder *b, const velri_model_params *mp);
} velri_arch_ops;

const velri_arch_ops *velri_arch_find(velri_arch arch);

#endif /* VELRI_MODEL_ARCH_REGISTRY_H */
