#include "internal.h"

/*
 * Portable C23 Baseline (`VELRI_CPU_BASE`) Linear Kernels:
 *   - Compiles on any C23 target (`-march=x86-64`, `aarch64`, etc.) with zero target intrinsics.
 *   - Uses `max_columns = 8, k_group = 1` for prepared weights (`0 B` scratch).
 */

static void base_pack(const velri_linear_typed_plan *plan, const void *w, size_t w_stride,
                      const float *w_scale, void *restrict packed) {
    velri_pack_generic_tiles(plan, 1, w, w_stride, w_scale, packed);
}

static inline float load_x_as_f32(velri_dtype dt, const void *base, size_t idx) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        return ((const float *)base)[idx];
    case VELRI_DTYPE_F16:
        return velri_f16_to_f32_scalar(((const velri_fp16 *)base)[idx]);
    case VELRI_DTYPE_BF16:
        return velri_bf16_to_f32_scalar(((const velri_bf16 *)base)[idx]);
    default:
        return 0.0f;
    }
}

static inline float load_w_as_f32(velri_dtype dt, const void *base, size_t idx) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        return ((const float *)base)[idx];
    case VELRI_DTYPE_F16:
        return velri_f16_to_f32_scalar(((const velri_fp16 *)base)[idx]);
    case VELRI_DTYPE_BF16:
        return velri_bf16_to_f32_scalar(((const velri_bf16 *)base)[idx]);
    case VELRI_DTYPE_I8:
        return (float)((const int8_t *)base)[idx];
    default:
        return 0.0f;
    }
}

static inline void store_f32_to_y(velri_dtype y_dt, void *y_base, size_t idx, float val) {
    switch (y_dt) {
    case VELRI_DTYPE_F32:
        ((float *)y_base)[idx] = val;
        break;
    case VELRI_DTYPE_F16:
        ((velri_fp16 *)y_base)[idx] = velri_f32_to_f16_scalar(val);
        break;
    case VELRI_DTYPE_BF16:
        ((velri_bf16 *)y_base)[idx] = velri_f32_to_bf16_scalar(val);
        break;
    default:
        break;
    }
}

static void base_run_prepared(const velri_linear_typed_plan *plan,
                              const velri_linear_run_args *args, size_t r0, size_t r1, size_t c0,
                              size_t c1, void *restrict scratch) {
    (void)scratch;
    const size_t k = plan->k;
    const size_t C = plan->columns;
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const size_t elem_b = velri_dtype_bytes(plan->desc.w_dtype);
    const size_t payload_b = plan->k_aligned * C * elem_b;
    const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
    const size_t scale_off =
        meta_off + (plan->desc.w_dtype == VELRI_DTYPE_I8 ? C * sizeof(int32_t) : 0);

    if (plan->desc.acc_dtype == VELRI_DTYPE_F64) {
        const double *x = (const double *)args->x;
        const double *bias = (const double *)args->bias;
        double *restrict y = (double *)args->y;
        for (size_t col0 = c0; col0 < c1; col0 += C) {
            size_t t = col0 / C;
            size_t width = (c1 - col0 < C) ? (c1 - col0) : C;
            const double *wp = (const double *)(pk_base + t * plan->tile_stride_b);
            for (size_t r = r0; r < r1; ++r) {
                double acc[8] = {0};
                const double *xr = x + r * args->x_stride;
                for (size_t i = 0; i < k; ++i) {
                    double xv = xr[i];
                    for (size_t j = 0; j < width; ++j) {
                        acc[j] += xv * wp[i * C + j];
                    }
                }
                for (size_t j = 0; j < width; ++j) {
                    y[r * args->y_stride + col0 + j] = acc[j] + (bias ? bias[col0 + j] : 0.0);
                }
            }
        }
        return;
    }

    if (plan->desc.acc_dtype == VELRI_DTYPE_I32) {
        const int8_t *x = (const int8_t *)args->x;
        for (size_t col0 = c0; col0 < c1; col0 += C) {
            size_t t = col0 / C;
            size_t width = (c1 - col0 < C) ? (c1 - col0) : C;
            const uint8_t *tile_ptr = pk_base + t * plan->tile_stride_b;
            const int8_t *wp = (const int8_t *)tile_ptr;
            const float *wsc =
                plan->desc.has_w_scale ? (const float *)(tile_ptr + scale_off) : nullptr;
            for (size_t r = r0; r < r1; ++r) {
                int32_t acc[8] = {0};
                const int8_t *xr = x + r * args->x_stride;
                for (size_t i = 0; i < k; ++i) {
                    int32_t xv = (int32_t)xr[i];
                    for (size_t j = 0; j < width; ++j) {
                        acc[j] = velri_i32_add(acc[j], xv * (int32_t)wp[i * C + j]);
                    }
                }
                if (plan->desc.y_dtype == VELRI_DTYPE_I32) {
                    const int32_t *bias = (const int32_t *)args->bias;
                    int32_t *restrict y = (int32_t *)args->y;
                    for (size_t j = 0; j < width; ++j) {
                        y[r * args->y_stride + col0 + j] =
                            velri_i32_add(acc[j], bias ? bias[col0 + j] : 0);
                    }
                } else {
                    const float *bias = (const float *)args->bias;
                    float *restrict y = (float *)args->y;
                    float xs = args->x_scale ? args->x_scale[r] : 1.0f;
                    for (size_t j = 0; j < width; ++j) {
                        float v = (float)acc[j] * (xs * (wsc ? wsc[j] : 1.0f));
                        if (bias) {
                            v += bias[col0 + j];
                        }
                        y[r * args->y_stride + col0 + j] = v;
                    }
                }
            }
        }
        return;
    }

    /* FP32 accumulator family (F32, F16, F32xF16, BF16, F32xBF16, F32xI8) */
    const float *bias = (const float *)args->bias;
    for (size_t col0 = c0; col0 < c1; col0 += C) {
        size_t t = col0 / C;
        size_t width = (c1 - col0 < C) ? (c1 - col0) : C;
        const uint8_t *tile_ptr = pk_base + t * plan->tile_stride_b;
        const float *wsc = plan->desc.has_w_scale ? (const float *)(tile_ptr + scale_off) : nullptr;
        for (size_t r = r0; r < r1; ++r) {
            float acc[8] = {0};
            size_t x_row_off = r * args->x_stride;
            for (size_t i = 0; i < k; ++i) {
                float xv = load_x_as_f32(plan->desc.x_dtype, args->x, x_row_off + i);
                for (size_t j = 0; j < width; ++j) {
                    float wv = load_w_as_f32(plan->desc.w_dtype, tile_ptr, i * C + j);
                    acc[j] += xv * wv;
                }
            }
            for (size_t j = 0; j < width; ++j) {
                float v = acc[j];
                if (wsc) {
                    v *= wsc[j];
                }
                if (bias) {
                    v += bias[col0 + j];
                }
                store_f32_to_y(plan->desc.y_dtype, args->y, r * args->y_stride + col0 + j, v);
            }
        }
    }
}

static void base_run_rowmajor(const velri_linear_typed_plan *plan,
                              const velri_linear_run_args *args, size_t r0, size_t r1, size_t c0,
                              size_t c1, void *restrict scratch) {
    (void)scratch;
    const size_t k = plan->k;

    if (plan->desc.acc_dtype == VELRI_DTYPE_F64) {
        const double *x = (const double *)args->x;
        const double *w = (const double *)args->weights;
        const double *bias = (const double *)args->bias;
        double *restrict y = (double *)args->y;
        for (size_t r = r0; r < r1; ++r) {
            const double *xr = x + r * args->x_stride;
            for (size_t c = c0; c < c1; ++c) {
                const double *wc = w + c * args->w_stride;
                double sum = 0.0;
                for (size_t i = 0; i < k; ++i) {
                    sum += xr[i] * wc[i];
                }
                y[r * args->y_stride + c] = sum + (bias ? bias[c] : 0.0);
            }
        }
        return;
    }

    if (plan->desc.acc_dtype == VELRI_DTYPE_I32) {
        const int8_t *x = (const int8_t *)args->x;
        const int8_t *w = (const int8_t *)args->weights;
        for (size_t r = r0; r < r1; ++r) {
            const int8_t *xr = x + r * args->x_stride;
            float xs = args->x_scale ? args->x_scale[r] : 1.0f;
            for (size_t c = c0; c < c1; ++c) {
                const int8_t *wc = w + c * args->w_stride;
                int32_t sum = 0;
                for (size_t i = 0; i < k; ++i) {
                    sum = velri_i32_add(sum, (int32_t)xr[i] * (int32_t)wc[i]);
                }
                if (plan->desc.y_dtype == VELRI_DTYPE_I32) {
                    const int32_t *bias = (const int32_t *)args->bias;
                    ((int32_t *)args->y)[r * args->y_stride + c] =
                        velri_i32_add(sum, bias ? bias[c] : 0);
                } else {
                    const float *bias = (const float *)args->bias;
                    float ws = args->w_scale ? args->w_scale[c] : 1.0f;
                    float v = (float)sum * (xs * ws) + (bias ? bias[c] : 0.0f);
                    ((float *)args->y)[r * args->y_stride + c] = v;
                }
            }
        }
        return;
    }

    const float *bias = (const float *)args->bias;
    for (size_t r = r0; r < r1; ++r) {
        size_t x_off = r * args->x_stride;
        for (size_t c = c0; c < c1; ++c) {
            size_t w_off = c * args->w_stride;
            float sum = 0.0f;
            for (size_t i = 0; i < k; ++i) {
                float xv = load_x_as_f32(plan->desc.x_dtype, args->x, x_off + i);
                float wv = load_w_as_f32(plan->desc.w_dtype, args->weights, w_off + i);
                sum += xv * wv;
            }
            if (plan->desc.has_w_scale && args->w_scale) {
                sum *= args->w_scale[c];
            }
            if (bias) {
                sum += bias[c];
            }
            store_f32_to_y(plan->desc.y_dtype, args->y, r * args->y_stride + c, sum);
        }
    }
}

static const velri_lin_blocking base_prepared = {0, 4, 128, 4, 16, 2};
static const velri_lin_blocking base_rowmajor = {0, 4, 128, 4, 32, 2};

#define DEF_BASE_ENTRY(knd)                                                                        \
    [(knd)] = {(knd),     VELRI_CPU_BASE,    8,                                                    \
               1,         &base_prepared,    &base_rowmajor,                                       \
               base_pack, base_run_prepared, base_run_rowmajor}

static const velri_lin_impl g_base_impls[] = {
    DEF_BASE_ENTRY(VELRI_LIN_F32),       DEF_BASE_ENTRY(VELRI_LIN_F64),
    DEF_BASE_ENTRY(VELRI_LIN_F16_F32),   DEF_BASE_ENTRY(VELRI_LIN_F16_F16),
    DEF_BASE_ENTRY(VELRI_LIN_F32X_F16W), DEF_BASE_ENTRY(VELRI_LIN_BF16_F32),
    DEF_BASE_ENTRY(VELRI_LIN_BF16_BF16), DEF_BASE_ENTRY(VELRI_LIN_F32X_BF16W),
    DEF_BASE_ENTRY(VELRI_LIN_I8_I32),    DEF_BASE_ENTRY(VELRI_LIN_W8A8_F32),
    DEF_BASE_ENTRY(VELRI_LIN_F32X_I8W),
};

const velri_lin_impl *velri_lin_get_base(velri_lin_kind kind) {
    if ((size_t)kind >= sizeof(g_base_impls) / sizeof(g_base_impls[0])) {
        return nullptr;
    }
    return &g_base_impls[kind];
}
