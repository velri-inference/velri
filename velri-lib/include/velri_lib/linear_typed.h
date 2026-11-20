#ifndef VELRI_LIB_LINEAR_TYPED_H
#define VELRI_LIB_LINEAR_TYPED_H

#include "velri_lib/types.h"

/*
 * Typed Linear operation descriptors shared across CPU (Slang AOT) and Vulkan backends:
 *   Computes Y[rows, M] = X[rows, K] * W[M, K]^T + bias[M] (or scaled W8A8 form).
 */
typedef struct {
    velri_dtype x_dtype;
    velri_dtype w_dtype;
    velri_dtype acc_dtype;
    velri_dtype y_dtype;
    bool has_x_scale; /* Per-row float scale[rows] (for W8A8 -> F32) */
    bool has_w_scale; /* Per-output float scale[M] (for W8A8 -> F32 and W8A32 -> F32) */
} velri_linear_desc;

static inline velri_linear_desc velri_desc_f32(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_F32, VELRI_DTYPE_F32,
                               VELRI_DTYPE_F32, false,           false};
}
static inline velri_linear_desc velri_desc_f64(void) {
    return (velri_linear_desc){VELRI_DTYPE_F64, VELRI_DTYPE_F64, VELRI_DTYPE_F64,
                               VELRI_DTYPE_F64, false,           false};
}
static inline velri_linear_desc velri_desc_f16(velri_dtype y_dtype) {
    return (velri_linear_desc){VELRI_DTYPE_F16, VELRI_DTYPE_F16, VELRI_DTYPE_F32,
                               y_dtype,         false,           false};
}
static inline velri_linear_desc velri_desc_f32x_f16w(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_F16, VELRI_DTYPE_F32,
                               VELRI_DTYPE_F32, false,           false};
}
static inline velri_linear_desc velri_desc_bf16(velri_dtype y_dtype) {
    return (velri_linear_desc){VELRI_DTYPE_BF16, VELRI_DTYPE_BF16, VELRI_DTYPE_F32,
                               y_dtype,          false,            false};
}
static inline velri_linear_desc velri_desc_f32x_bf16w(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F32,
                               VELRI_DTYPE_F32, false,            false};
}
static inline velri_linear_desc velri_desc_i8_i32(void) {
    return (velri_linear_desc){VELRI_DTYPE_I8,  VELRI_DTYPE_I8, VELRI_DTYPE_I32,
                               VELRI_DTYPE_I32, false,          false};
}
static inline velri_linear_desc velri_desc_w8a8_f32(void) {
    return (velri_linear_desc){VELRI_DTYPE_I8,  VELRI_DTYPE_I8, VELRI_DTYPE_I32,
                               VELRI_DTYPE_F32, true,           true};
}
static inline velri_linear_desc velri_desc_f32x_i8w(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_I8, VELRI_DTYPE_F32,
                               VELRI_DTYPE_F32, false,          true};
}

#endif /* VELRI_LIB_LINEAR_TYPED_H */
