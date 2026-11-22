#include "model/arch_registry.h"

#include "model/arch/gpt2.h"
#include "model/arch/llama.h"

#include <math.h>
#include <stdio.h>

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))

static velri_rc validate_llama_desc(velri_model_desc *norm, char *err_buf, size_t err_cap) {
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

static const velri_arch_ops k_arch_table[] = {
    {
        .arch = VELRI_ARCH_GPT2,
        .name = "gpt2",
        .has_auto_rope = false,
        .enforces_max_pos_ctx = true,
        .validate_desc = nullptr,
        .build_graph = velri_arch_gpt2_build,
    },
    {
        .arch = VELRI_ARCH_LLAMA,
        .name = "llama",
        .has_auto_rope = true,
        .enforces_max_pos_ctx = false,
        .validate_desc = validate_llama_desc,
        .build_graph = velri_arch_llama_build,
    },
};

const velri_arch_ops *velri_arch_find(velri_arch arch) {
    for (size_t i = 0U; i < ARRAY_COUNT(k_arch_table); ++i) {
        if (k_arch_table[i].arch == arch) {
            return &k_arch_table[i];
        }
    }
    return nullptr;
}
