#include "model/arch_registry.h"

#include "models.h"

#include <string.h>

const velri_arch_ops *velri_arch_find(velri_arch arch) {
    for (size_t i = 0U; k_arch_table[i]; ++i) {
        if (k_arch_table[i]->arch == arch) {
            return k_arch_table[i];
        }
    }
    return nullptr;
}

velri_rc velri_arch_from_name(const char *name, velri_arch *out_arch) {
    if (!name || !*name || !out_arch) {
        return VELRI_RC_ERR_INVALID;
    }
    for (size_t i = 0U; k_arch_table[i]; ++i) {
        const velri_arch_ops *ops = k_arch_table[i];
        if (strcmp(name, ops->name) == 0) {
            *out_arch = ops->arch;
            return VELRI_RC_OK;
        }
    }
    return VELRI_RC_ERR_NOT_FOUND;
}
