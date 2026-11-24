#ifndef VELRI_MODEL_ARCH_REGISTRY_H
#define VELRI_MODEL_ARCH_REGISTRY_H

#include "model/layers.h"
#include "velri/velri.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    bool has_auto_rope;
    bool enforces_max_pos_ctx;
    velri_rc (*validate_desc)(velri_model_desc *norm, char *err_buf, size_t err_cap);
} velri_arch_policy;

typedef struct {
    velri_arch arch;
    const char *name;
    const velri_arch_policy *policy;
    velri_status (*build_graph)(velri_builder *b, const velri_model_params *params);
} velri_arch_ops;

const velri_arch_ops *velri_arch_find(velri_arch arch);

#endif /* VELRI_MODEL_ARCH_REGISTRY_H */
