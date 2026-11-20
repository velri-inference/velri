#include "internal.h"

void velri_cvt_f32_to_f16_base(const float *src, velri_fp16 *restrict dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = velri_f32_to_f16_scalar(src[i]);
    }
}

void velri_cvt_f16_to_f32_base(const velri_fp16 *src, float *restrict dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = velri_f16_to_f32_scalar(src[i]);
    }
}

void velri_cvt_f32_to_bf16_base(const float *src, velri_bf16 *restrict dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = velri_f32_to_bf16_scalar(src[i]);
    }
}

void velri_cvt_bf16_to_f32_base(const velri_bf16 *src, float *restrict dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = velri_bf16_to_f32_scalar(src[i]);
    }
}

void velri_quantize_row_f32_i8_base(size_t rows, size_t k, const float *x, size_t x_stride,
                                    int8_t *restrict x_i8, size_t xi8_stride,
                                    float *restrict scale_out) {
    for (size_t r = 0; r < rows; ++r) {
        const float *xr = x + r * x_stride;
        int8_t *qr = x_i8 + r * xi8_stride;
        float amax = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            float a = fabsf(xr[i]);
            if (a > amax) {
                amax = a;
            }
        }
        if (amax > 0.0f && amax < 127.0f / FLT_MAX) {
            velri_quantize_tiny_row(xr, qr, k, amax, &scale_out[r]);
            continue;
        }
        float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        scale_out[r] = scale;
        for (size_t i = 0; i < k; ++i) {
            float v = rintf(xr[i] * inv_scale);
            if (v > 127.0f) {
                v = 127.0f;
            } else if (v < -127.0f) {
                v = -127.0f;
            }
            qr[i] = (int8_t)(int32_t)v;
        }
    }
}

static velri_status check_cvt_1d(const void *src, size_t src_elem, size_t src_align,
                                 const void *dst, size_t dst_elem, size_t dst_align, size_t count,
                                 velri_cpu_variant *out_v) {
    if (!velri_cpu_variant_supported(*out_v)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    *out_v = velri_cpu_resolve_variant(*out_v);
    size_t sb = 0, db = 0;
    velri_status s = velri_check_span(src, count, 1, count, count, src_elem, src_align, &sb);
    if (s != VELRI_OK) {
        return s;
    }
    s = velri_check_span(dst, count, 1, count, count, dst_elem, dst_align, &db);
    if (s != VELRI_OK) {
        return s;
    }
    if (velri_overlap(src, sb, dst, db)) {
        return VELRI_ERR_INVALID;
    }
    return VELRI_OK;
}

velri_status velri_cvt_f32_to_f16(const float *src, velri_fp16 *restrict dst, size_t count,
                                  velri_cpu_variant variant) {
    velri_status s = check_cvt_1d(src, sizeof(float), alignof(float), dst, sizeof(velri_fp16),
                                  alignof(velri_fp16), count, &variant);
    if (s != VELRI_OK || count == 0) {
        return s;
    }
#if VELRI_X86_VARIANTS
    if (variant == VELRI_CPU_AVX512) {
        velri_cvt_f32_to_f16_avx512(src, dst, count);
        return VELRI_OK;
    }
    if (variant == VELRI_CPU_AVX2) {
        velri_cvt_f32_to_f16_avx2(src, dst, count);
        return VELRI_OK;
    }
#endif
    velri_cvt_f32_to_f16_base(src, dst, count);
    return VELRI_OK;
}

velri_status velri_cvt_f16_to_f32(const velri_fp16 *src, float *restrict dst, size_t count,
                                  velri_cpu_variant variant) {
    velri_status s = check_cvt_1d(src, sizeof(velri_fp16), alignof(velri_fp16), dst, sizeof(float),
                                  alignof(float), count, &variant);
    if (s != VELRI_OK || count == 0) {
        return s;
    }
#if VELRI_X86_VARIANTS
    if (variant == VELRI_CPU_AVX512) {
        velri_cvt_f16_to_f32_avx512(src, dst, count);
        return VELRI_OK;
    }
    if (variant == VELRI_CPU_AVX2) {
        velri_cvt_f16_to_f32_avx2(src, dst, count);
        return VELRI_OK;
    }
#endif
    velri_cvt_f16_to_f32_base(src, dst, count);
    return VELRI_OK;
}

velri_status velri_cvt_f32_to_bf16(const float *src, velri_bf16 *restrict dst, size_t count,
                                   velri_cpu_variant variant) {
    velri_status s = check_cvt_1d(src, sizeof(float), alignof(float), dst, sizeof(velri_bf16),
                                  alignof(velri_bf16), count, &variant);
    if (s != VELRI_OK || count == 0) {
        return s;
    }
#if VELRI_X86_VARIANTS
    if (variant == VELRI_CPU_AVX512) {
        velri_cvt_f32_to_bf16_avx512(src, dst, count);
        return VELRI_OK;
    }
    if (variant == VELRI_CPU_AVX2) {
        velri_cvt_f32_to_bf16_avx2(src, dst, count);
        return VELRI_OK;
    }
#endif
    velri_cvt_f32_to_bf16_base(src, dst, count);
    return VELRI_OK;
}

velri_status velri_cvt_bf16_to_f32(const velri_bf16 *src, float *restrict dst, size_t count,
                                   velri_cpu_variant variant) {
    velri_status s = check_cvt_1d(src, sizeof(velri_bf16), alignof(velri_bf16), dst, sizeof(float),
                                  alignof(float), count, &variant);
    if (s != VELRI_OK || count == 0) {
        return s;
    }
#if VELRI_X86_VARIANTS
    if (variant == VELRI_CPU_AVX512) {
        velri_cvt_bf16_to_f32_avx512(src, dst, count);
        return VELRI_OK;
    }
    if (variant == VELRI_CPU_AVX2) {
        velri_cvt_bf16_to_f32_avx2(src, dst, count);
        return VELRI_OK;
    }
#endif
    velri_cvt_bf16_to_f32_base(src, dst, count);
    return VELRI_OK;
}

velri_status velri_quantize_row_f32_i8(size_t rows, size_t k, const float *x, size_t x_stride,
                                       int8_t *restrict x_i8, size_t xi8_stride,
                                       float *restrict scale_out, velri_cpu_variant variant) {
    if (k == 0) {
        return VELRI_ERR_INVALID;
    }
    if (!velri_cpu_variant_supported(variant)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    variant = velri_cpu_resolve_variant(variant);
    if (rows == 0) {
        return (x_stride >= k && xi8_stride >= k) ? VELRI_OK : VELRI_ERR_INVALID;
    }
    size_t x_cap = 0, q_cap = 0;
    if (!velri_mul_size(rows - 1, x_stride, &x_cap) || !velri_add_size(x_cap, k, &x_cap) ||
        !velri_mul_size(rows - 1, xi8_stride, &q_cap) || !velri_add_size(q_cap, k, &q_cap)) {
        return VELRI_ERR_OVERFLOW;
    }
    size_t xb = 0, qb = 0, sb = 0;
    velri_status s =
        velri_check_span(x, x_cap, rows, k, x_stride, sizeof(float), alignof(float), &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = velri_check_span(x_i8, q_cap, rows, k, xi8_stride, sizeof(int8_t), alignof(int8_t), &qb);
    if (s != VELRI_OK) {
        return s;
    }
    s = velri_check_span(scale_out, rows, 1, rows, rows, sizeof(float), alignof(float), &sb);
    if (s != VELRI_OK) {
        return s;
    }
    if (velri_overlap(x, xb, x_i8, qb) || velri_overlap(x, xb, scale_out, sb) ||
        velri_overlap(x_i8, qb, scale_out, sb)) {
        return VELRI_ERR_INVALID;
    }

#if VELRI_X86_VARIANTS
    if (variant == VELRI_CPU_AVX512) {
        velri_quantize_row_f32_i8_avx512(rows, k, x, x_stride, x_i8, xi8_stride, scale_out);
        return VELRI_OK;
    }
    if (variant == VELRI_CPU_AVX2) {
        velri_quantize_row_f32_i8_avx2(rows, k, x, x_stride, x_i8, xi8_stride, scale_out);
        return VELRI_OK;
    }
#endif
    velri_quantize_row_f32_i8_base(rows, k, x, x_stride, x_i8, xi8_stride, scale_out);
    return VELRI_OK;
}
