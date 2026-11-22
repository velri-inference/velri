#ifndef VELRI_CONFIG_PLACEMENT_SCOPES_H
#define VELRI_CONFIG_PLACEMENT_SCOPES_H

/*
 * Canonical Velri Placement Scope Keys & Paths (Single Source of Truth).
 *
 * Shared across config parsing (`placement.c`), cascading resolution (`placement_resolve.c`),
 * inspect formatting (`placement_inspect.c`), model layer builders (`src/model/layers.c`),
 * and the architecture registry (`src/model/arch_registry.c`).
 */

#define VELRI_SCOPE_ROOT            ""
#define VELRI_SCOPE_INPUT_LAYER     "input-layer"
#define VELRI_SCOPE_LM_HEAD         "lm-head"
#define VELRI_SCOPE_PLE             "ple"
#define VELRI_SCOPE_PLE_TABLE       "ple/table"

#define VELRI_SCOPE_HIDDEN_LAYERS   "hidden-layers"
#define VELRI_SCOPE_KEY_ATTENTION   "attention"
#define VELRI_SCOPE_KEY_KV_CACHE    "kv-cache"
#define VELRI_SCOPE_KEY_INDEXER     "indexer"
#define VELRI_SCOPE_KEY_KEYS        "keys"
#define VELRI_SCOPE_KEY_DENSE_MLP   "dense-mlp"
#define VELRI_SCOPE_KEY_MOE         "moe"
#define VELRI_SCOPE_KEY_ROUTER      "router"
#define VELRI_SCOPE_KEY_SHARED_EXP  "shared-experts"
#define VELRI_SCOPE_KEY_ROUTED_EXP  "routed-experts"
#define VELRI_SCOPE_KEY_GDN         "gdn"
#define VELRI_SCOPE_KEY_CONV        "conv"
#define VELRI_SCOPE_KEY_TABLE       "table"

#define VELRI_SCOPE_HL_ATTENTION    "hidden-layers/attention"
#define VELRI_SCOPE_HL_KV_CACHE     "hidden-layers/attention/kv-cache"
#define VELRI_SCOPE_HL_INDEXER      "hidden-layers/attention/indexer"
#define VELRI_SCOPE_HL_INDEXER_KEYS "hidden-layers/attention/indexer/keys"
#define VELRI_SCOPE_HL_DENSE_MLP    "hidden-layers/dense-mlp"
#define VELRI_SCOPE_HL_MOE          "hidden-layers/moe"
#define VELRI_SCOPE_HL_MOE_ROUTER   "hidden-layers/moe/router"
#define VELRI_SCOPE_HL_MOE_SHARED   "hidden-layers/moe/shared-experts"
#define VELRI_SCOPE_HL_MOE_ROUTED   "hidden-layers/moe/routed-experts"
#define VELRI_SCOPE_HL_GDN          "hidden-layers/gdn"
#define VELRI_SCOPE_HL_CONV         "hidden-layers/conv"

#endif /* VELRI_CONFIG_PLACEMENT_SCOPES_H */
