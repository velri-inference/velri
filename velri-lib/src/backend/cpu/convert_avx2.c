#include "internal.h"
#include <immintrin.h>

void velri_cvt_f32_to_f16_avx2(const float *src, velri_fp16 *restrict dst, size_t n) {
    size_t i = 0;
    if (n * 6ULL >= velri_nt_io_bytes() && ((uintptr_t)dst & 1ULL) == 0) {
        while (i < n && (((uintptr_t)(dst + i)) & 31ULL) != 0) {
            dst[i] = velri_f32_to_f16_scalar(src[i]);
            ++i;
        }
        for (; i + 32 <= n; i += 32) {
            __m128i h0 = _mm256_cvtps_ph(_mm256_loadu_ps(src + i),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i h1 = _mm256_cvtps_ph(_mm256_loadu_ps(src + i + 8),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i h2 = _mm256_cvtps_ph(_mm256_loadu_ps(src + i + 16),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i h3 = _mm256_cvtps_ph(_mm256_loadu_ps(src + i + 24),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_stream_si256((__m256i *)(dst + i), _mm256_set_m128i(h1, h0));
            _mm256_stream_si256((__m256i *)(dst + i + 16), _mm256_set_m128i(h3, h2));
        }
        _mm_sfence();
    }
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(dst + i), h);
    }
    for (; i < n; ++i) {
        dst[i] = velri_f32_to_f16_scalar(src[i]);
    }
}

void velri_cvt_f16_to_f32_avx2(const velri_fp16 *src, float *restrict dst, size_t n) {
    size_t i = 0;
    if (n * 6ULL >= velri_nt_io_bytes() && ((uintptr_t)dst & 3ULL) == 0) {
        while (i < n && (((uintptr_t)(dst + i)) & 31ULL) != 0) {
            dst[i] = velri_f16_to_f32_scalar(src[i]);
            ++i;
        }
        for (; i + 32 <= n; i += 32) {
            __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i)));
            __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i + 8)));
            __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i + 16)));
            __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i + 24)));
            _mm256_stream_ps(dst + i, v0);
            _mm256_stream_ps(dst + i + 8, v1);
            _mm256_stream_ps(dst + i + 16, v2);
            _mm256_stream_ps(dst + i + 24, v3);
        }
        _mm_sfence();
    }
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
        __m256 v = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; ++i) {
        dst[i] = velri_f16_to_f32_scalar(src[i]);
    }
}

static inline __m256i cvt16_f32_to_bf16_avx2_exact(const float *src) {
    const __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
    const __m256i inf_bits = _mm256_set1_epi32(0x7F800000);
    const __m256i bias = _mm256_set1_epi32(0x00007FFF);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i qnan_bit = _mm256_set1_epi32(0x00400000);

    __m256i u0 = _mm256_loadu_si256((const __m256i *)src);
    __m256i u1 = _mm256_loadu_si256((const __m256i *)(src + 8));

    __m256i nan0 = _mm256_cmpgt_epi32(_mm256_and_si256(u0, abs_mask), inf_bits);
    __m256i nan1 = _mm256_cmpgt_epi32(_mm256_and_si256(u1, abs_mask), inf_bits);

    __m256i lsb0 = _mm256_and_si256(_mm256_srli_epi32(u0, 16), one);
    __m256i lsb1 = _mm256_and_si256(_mm256_srli_epi32(u1, 16), one);

    __m256i r0 = _mm256_add_epi32(u0, _mm256_add_epi32(bias, lsb0));
    __m256i r1 = _mm256_add_epi32(u1, _mm256_add_epi32(bias, lsb1));

    r0 = _mm256_blendv_epi8(r0, _mm256_or_si256(u0, qnan_bit), nan0);
    r1 = _mm256_blendv_epi8(r1, _mm256_or_si256(u1, qnan_bit), nan1);

    r0 = _mm256_srli_epi32(r0, 16);
    r1 = _mm256_srli_epi32(r1, 16);

    __m256i packed = _mm256_packus_epi32(r0, r1);
    return _mm256_permute4x64_epi64(packed, 0xD8);
}

void velri_cvt_f32_to_bf16_avx2(const float *src, velri_bf16 *restrict dst, size_t n) {
    size_t i = 0;
    if (n * 6ULL >= velri_nt_io_bytes() && ((uintptr_t)dst & 1ULL) == 0) {
        while (i < n && (((uintptr_t)(dst + i)) & 31ULL) != 0) {
            dst[i] = velri_f32_to_bf16_scalar(src[i]);
            ++i;
        }
        for (; i + 32 <= n; i += 32) {
            __m256i p0 = cvt16_f32_to_bf16_avx2_exact(src + i);
            __m256i p1 = cvt16_f32_to_bf16_avx2_exact(src + i + 16);
            _mm256_stream_si256((__m256i *)(dst + i), p0);
            _mm256_stream_si256((__m256i *)(dst + i + 16), p1);
        }
        _mm_sfence();
    }
    for (; i + 16 <= n; i += 16) {
        _mm256_storeu_si256((__m256i *)(dst + i), cvt16_f32_to_bf16_avx2_exact(src + i));
    }
    for (; i < n; ++i) {
        dst[i] = velri_f32_to_bf16_scalar(src[i]);
    }
}

void velri_cvt_bf16_to_f32_avx2(const velri_bf16 *src, float *restrict dst, size_t n) {
    size_t i = 0;
    if (n * 6ULL >= velri_nt_io_bytes() && ((uintptr_t)dst & 3ULL) == 0) {
        while (i < n && (((uintptr_t)(dst + i)) & 31ULL) != 0) {
            dst[i] = velri_bf16_to_f32_scalar(src[i]);
            ++i;
        }
        for (; i + 32 <= n; i += 32) {
            __m256i u0 = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(src + i))), 16);
            __m256i u1 = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(src + i + 8))), 16);
            __m256i u2 = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(src + i + 16))), 16);
            __m256i u3 = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(src + i + 24))), 16);
            _mm256_stream_ps(dst + i, _mm256_castsi256_ps(u0));
            _mm256_stream_ps(dst + i + 8, _mm256_castsi256_ps(u1));
            _mm256_stream_ps(dst + i + 16, _mm256_castsi256_ps(u2));
            _mm256_stream_ps(dst + i + 24, _mm256_castsi256_ps(u3));
        }
        _mm_sfence();
    }
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
        __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(u));
    }
    for (; i < n; ++i) {
        dst[i] = velri_bf16_to_f32_scalar(src[i]);
    }
}

void velri_quantize_row_f32_i8_avx2(size_t rows, size_t k, const float *x, size_t x_stride,
                                    int8_t *restrict x_i8, size_t xi8_stride,
                                    float *restrict scale_out) {
    const __m256 sign_clear = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 v_max = _mm256_set1_ps(127.0f);
    const __m256 v_min = _mm256_set1_ps(-127.0f);
    const bool use_nt = (rows * k * 5ULL >= velri_nt_io_bytes());

    for (size_t r = 0; r < rows; ++r) {
        const float *xr = x + r * x_stride;
        int8_t *qr = x_i8 + r * xi8_stride;

        __m256 vm0 = _mm256_setzero_ps(), vm1 = _mm256_setzero_ps();
        __m256 vm2 = _mm256_setzero_ps(), vm3 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 32 <= k; i += 32) {
            vm0 = _mm256_max_ps(vm0, _mm256_and_ps(_mm256_loadu_ps(xr + i), sign_clear));
            vm1 = _mm256_max_ps(vm1, _mm256_and_ps(_mm256_loadu_ps(xr + i + 8), sign_clear));
            vm2 = _mm256_max_ps(vm2, _mm256_and_ps(_mm256_loadu_ps(xr + i + 16), sign_clear));
            vm3 = _mm256_max_ps(vm3, _mm256_and_ps(_mm256_loadu_ps(xr + i + 24), sign_clear));
        }
        __m256 vmax = _mm256_max_ps(_mm256_max_ps(vm0, vm1), _mm256_max_ps(vm2, vm3));
        for (; i + 8 <= k; i += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(xr + i), sign_clear));
        }
        __m128 hi = _mm256_extractf128_ps(vmax, 1);
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(vmax), hi);
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        float amax = _mm_cvtss_f32(m4);
        for (; i < k; ++i) {
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
        __m256 vinv = _mm256_set1_ps(inv_scale);
        bool row_nt = use_nt && (((uintptr_t)qr & 31ULL) == 0);

        i = 0;
        for (; i + 32 <= k; i += 32) {
            __m256 f0 = _mm256_max_ps(
                _mm256_min_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i), vinv), v_max), v_min);
            __m256 f1 = _mm256_max_ps(
                _mm256_min_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i + 8), vinv), v_max), v_min);
            __m256 f2 = _mm256_max_ps(
                _mm256_min_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i + 16), vinv), v_max), v_min);
            __m256 f3 = _mm256_max_ps(
                _mm256_min_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i + 24), vinv), v_max), v_min);
            __m256i s16_0 = _mm256_permute4x64_epi64(
                _mm256_packs_epi32(_mm256_cvtps_epi32(f0), _mm256_cvtps_epi32(f1)), 0xD8);
            __m256i s16_1 = _mm256_permute4x64_epi64(
                _mm256_packs_epi32(_mm256_cvtps_epi32(f2), _mm256_cvtps_epi32(f3)), 0xD8);
            __m256i s8 = _mm256_permute4x64_epi64(_mm256_packs_epi16(s16_0, s16_1), 0xD8);
            if (row_nt) {
                _mm256_stream_si256((__m256i *)(qr + i), s8);
            } else {
                _mm256_storeu_si256((__m256i *)(qr + i), s8);
            }
        }
        for (; i + 16 <= k; i += 16) {
            __m256 f0 = _mm256_mul_ps(_mm256_loadu_ps(xr + i), vinv);
            __m256 f1 = _mm256_mul_ps(_mm256_loadu_ps(xr + i + 8), vinv);
            f0 = _mm256_max_ps(_mm256_min_ps(f0, v_max), v_min);
            f1 = _mm256_max_ps(_mm256_min_ps(f1, v_max), v_min);
            __m256i i0 = _mm256_cvtps_epi32(f0);
            __m256i i1 = _mm256_cvtps_epi32(f1);
            __m256i s16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
            __m128i s8 =
                _mm_packs_epi16(_mm256_castsi256_si128(s16), _mm256_extracti128_si256(s16, 1));
            _mm_storeu_si128((__m128i *)(qr + i), s8);
        }
        for (; i < k; ++i) {
            float v = rintf(xr[i] * inv_scale);
            if (v > 127.0f) {
                v = 127.0f;
            } else if (v < -127.0f) {
                v = -127.0f;
            }
            qr[i] = (int8_t)(int32_t)v;
        }
    }
    if (use_nt) {
        _mm_sfence();
    }
}
