#ifndef VELRI_LIB_OPS_H
#define VELRI_LIB_OPS_H

/*
 * Shared operation parameter enums for Velri-lib MatOps across CPU and Vulkan backends.
 */

typedef enum { VELRI_ROPE_HALF = 0, VELRI_ROPE_INTERLEAVED = 1 } velri_rope_mode;

typedef enum { VELRI_ACT_SILU = 0, VELRI_ACT_GELU_TANH = 1, VELRI_ACT_RELU2 = 2 } velri_act_kind;

#endif /* VELRI_LIB_OPS_H */
