#include "internal.h"
#include <immintrin.h>

void velri_cvt_f32_to_f16_avx512(const float *src, velri_fp16 *restrict dst, size_t n) {
    size_t i = 0;
    if (n * 6ULL >= velri_nt_io_bytes() && ((uintptr_t)dst & 1ULL) == 0) {
        while (i < n && (((uintptr_t)(dst + i)) & 31ULL) != 0) {
            dst[i] = velri_f32_to_f16_scalar(src[i]);
            ++i;
        }
        for (; i + 64 <= n; i += 64) {
            __m256i h0 = _mm512_cvtps_ph(_mm512_loadu_ps(src + i),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i h1 = _mm512_cvtps_ph(_mm512_loadu_ps(src + i + 16),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i h2 = _mm512_cvtps_ph(_mm512_loadu_ps(src + i + 32),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i h3 = _mm512_cvtps_ph(_mm512_loadu_ps(src + i + 48),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_stream_si256((__m256i *)(dst + i), h0);
            _mm256_stream_si256((__m256i *)(dst + i + 16), h1);
            _mm256_stream_si256((__m256i *)(dst + i + 32), h2);
            _mm256_stream_si256((__m256i *)(dst + i + 48), h3);
        }
        _mm_sfence();
    }
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256((__m256i *)(dst + i), h);
    }
    if (i < n) {
        __mmask16 m = (__mmask16)((1U << (n - i)) - 1U);
        __m512 v = _mm512_maskz_loadu_ps(m, src + i);
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_mask_storeu_epi16(dst + i, m, h);
    }
}

void velri_cvt_f16_to_f32_avx512(const velri_fp16 *src, float *restrict dst, size_t n) {
    size_t i = 0;
    if (n * 6ULL >= velri_nt_io_bytes() && ((uintptr_t)dst & 3ULL) == 0) {
        while (i < n && (((uintptr_t)(dst + i)) & 63ULL) != 0) {
            dst[i] = velri_f16_to_f32_scalar(src[i]);
            ++i;
        }
        for (; i + 64 <= n; i += 64) {
            __m512 v0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(src + i)));
            __m512 v1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(src + i + 16)));
            __m512 v2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(src + i + 32)));
            __m512 v3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(src + i + 48)));
            _mm512_stream_ps(dst + i, v0);
            _mm512_stream_ps(dst + i + 16, v1);
            _mm512_stream_ps(dst + i + 32, v2);
            _mm512_stream_ps(dst + i + 48, v3);
        }
        _mm_sfence();
    }
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i *)(src + i));
        __m512 v = _mm512_cvtph_ps(h);
        _mm512_storeu_ps(dst + i, v);
    }
    if (i < n) {
        __mmask16 m = (__mmask16)((1U << (n - i)) - 1U);
        __m256i h = _mm256_maskz_loadu_epi16(m, src + i);
        __m512 v = _mm512_cvtph_ps(h);
        _mm512_mask_storeu_ps(dst + i, m, v);
    }
}

static inline __m256i cvt16_f32_to_bf16_exact(__m512i u) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512i inf_bits = _mm512_set1_epi32(0x7F800000);
    const __m512i bias = _mm512_set1_epi32(0x00007FFF);
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i qnan_bit = _mm512_set1_epi32(0x00400000);

    __mmask16 is_nan = _mm512_cmpgt_epi32_mask(_mm512_and_epi32(u, abs_mask), inf_bits);
    __m512i lsb = _mm512_and_epi32(_mm512_srli_epi32(u, 16), one);
    __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(bias, lsb));
    r = _mm512_mask_mov_epi32(r, is_nan, _mm512_or_epi32(u, qnan_bit));
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16));
}

void velri_cvt_f32_to_bf16_avx512(const float *src, velri_bf16 *restrict dst, size_t n) {
    size_t i = 0;
    if (n * 6ULL >= velri_nt_io_bytes() && ((uintptr_t)dst & 1ULL) == 0) {
        while (i < n && (((uintptr_t)(dst + i)) & 31ULL) != 0) {
            dst[i] = velri_f32_to_bf16_scalar(src[i]);
            ++i;
        }
        for (; i + 64 <= n; i += 64) {
            __m256i b0 = cvt16_f32_to_bf16_exact(_mm512_loadu_si512((const void *)(src + i)));
            __m256i b1 = cvt16_f32_to_bf16_exact(_mm512_loadu_si512((const void *)(src + i + 16)));
            __m256i b2 = cvt16_f32_to_bf16_exact(_mm512_loadu_si512((const void *)(src + i + 32)));
            __m256i b3 = cvt16_f32_to_bf16_exact(_mm512_loadu_si512((const void *)(src + i + 48)));
            _mm256_stream_si256((__m256i *)(dst + i), b0);
            _mm256_stream_si256((__m256i *)(dst + i + 16), b1);
            _mm256_stream_si256((__m256i *)(dst + i + 32), b2);
            _mm256_stream_si256((__m256i *)(dst + i + 48), b3);
        }
        _mm_sfence();
    }
    for (; i + 16 <= n; i += 16) {
        __m256i b16 = cvt16_f32_to_bf16_exact(_mm512_loadu_si512((const void *)(src + i)));
        _mm256_storeu_si256((__m256i *)(dst + i), b16);
    }
    if (i < n) {
        __mmask16 m = (__mmask16)((1U << (n - i)) - 1U);
        __m256i b16 = cvt16_f32_to_bf16_exact(_mm512_maskz_loadu_epi32(m, src + i));
        _mm256_mask_storeu_epi16(dst + i, m, b16);
    }
}

void velri_cvt_bf16_to_f32_avx512(const velri_bf16 *src, float *restrict dst, size_t n) {
    size_t i = 0;
    if (n * 6ULL >= velri_nt_io_bytes() && ((uintptr_t)dst & 3ULL) == 0) {
        while (i < n && (((uintptr_t)(dst + i)) & 63ULL) != 0) {
            dst[i] = velri_bf16_to_f32_scalar(src[i]);
            ++i;
        }
        for (; i + 64 <= n; i += 64) {
            __m512i u0 = _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src + i))), 16);
            __m512i u1 = _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src + i + 16))), 16);
            __m512i u2 = _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src + i + 32))), 16);
            __m512i u3 = _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src + i + 48))), 16);
            _mm512_stream_ps(dst + i, _mm512_castsi512_ps(u0));
            _mm512_stream_ps(dst + i + 16, _mm512_castsi512_ps(u1));
            _mm512_stream_ps(dst + i + 32, _mm512_castsi512_ps(u2));
            _mm512_stream_ps(dst + i + 48, _mm512_castsi512_ps(u3));
        }
        _mm_sfence();
    }
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i *)(src + i));
        __m512i u = _mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(u));
    }
    if (i < n) {
        __mmask16 m = (__mmask16)((1U << (n - i)) - 1U);
        __m256i h = _mm256_maskz_loadu_epi16(m, src + i);
        __m512i u = _mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16);
        _mm512_mask_storeu_ps(dst + i, m, _mm512_castsi512_ps(u));
    }
}

void velri_quantize_row_f32_i8_avx512(size_t rows, size_t k, const float *x, size_t x_stride,
                                      int8_t *restrict x_i8, size_t xi8_stride,
                                      float *restrict scale_out) {
    const __m512 v_max = _mm512_set1_ps(127.0f);
    const __m512 v_min = _mm512_set1_ps(-127.0f);
    /* Permute table restoring linear 64-byte lane order after two cross-128b `packs` stages */
    const __m512i pack_perm =
        _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const bool use_nt = (rows * k * 5ULL >= velri_nt_io_bytes());

    for (size_t r = 0; r < rows; ++r) {
        const float *xr = x + r * x_stride;
        int8_t *qr = x_i8 + r * xi8_stride;

        __m512 vm0 = _mm512_setzero_ps(), vm1 = _mm512_setzero_ps();
        __m512 vm2 = _mm512_setzero_ps(), vm3 = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 64 <= k; i += 64) {
            vm0 = _mm512_max_ps(vm0, _mm512_abs_ps(_mm512_loadu_ps(xr + i)));
            vm1 = _mm512_max_ps(vm1, _mm512_abs_ps(_mm512_loadu_ps(xr + i + 16)));
            vm2 = _mm512_max_ps(vm2, _mm512_abs_ps(_mm512_loadu_ps(xr + i + 32)));
            vm3 = _mm512_max_ps(vm3, _mm512_abs_ps(_mm512_loadu_ps(xr + i + 48)));
        }
        vm0 = _mm512_max_ps(_mm512_max_ps(vm0, vm1), _mm512_max_ps(vm2, vm3));
        for (; i + 16 <= k; i += 16) {
            vm0 = _mm512_max_ps(vm0, _mm512_abs_ps(_mm512_loadu_ps(xr + i)));
        }
        if (i < k) {
            __mmask16 m = (__mmask16)((1U << (k - i)) - 1U);
            vm0 = _mm512_max_ps(vm0, _mm512_abs_ps(_mm512_maskz_loadu_ps(m, xr + i)));
        }
        float amax = _mm512_reduce_max_ps(vm0);
        if (amax > 0.0f && amax < 127.0f / FLT_MAX) {
            velri_quantize_tiny_row(xr, qr, k, amax, &scale_out[r]);
            continue;
        }
        float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        scale_out[r] = scale;

        __m512 vinv = _mm512_set1_ps(inv_scale);
        i = 0;
        bool row_nt = use_nt && (((uintptr_t)qr & 63ULL) == 0);
        const float *xn = (row_nt && r + 1 < rows) ? (x + (r + 1) * x_stride) : nullptr;
        for (; i + 64 <= k; i += 64) {
            if (xn) {
                _mm_prefetch((const char *)(xn + i), _MM_HINT_T0);
                _mm_prefetch((const char *)(xn + i + 16), _MM_HINT_T0);
                _mm_prefetch((const char *)(xn + i + 32), _MM_HINT_T0);
                _mm_prefetch((const char *)(xn + i + 48), _MM_HINT_T0);
            }
            __m512 f0 = _mm512_max_ps(
                _mm512_min_ps(_mm512_mul_ps(_mm512_loadu_ps(xr + i), vinv), v_max), v_min);
            __m512 f1 = _mm512_max_ps(
                _mm512_min_ps(_mm512_mul_ps(_mm512_loadu_ps(xr + i + 16), vinv), v_max), v_min);
            __m512 f2 = _mm512_max_ps(
                _mm512_min_ps(_mm512_mul_ps(_mm512_loadu_ps(xr + i + 32), vinv), v_max), v_min);
            __m512 f3 = _mm512_max_ps(
                _mm512_min_ps(_mm512_mul_ps(_mm512_loadu_ps(xr + i + 48), vinv), v_max), v_min);
            __m512i i0 =
                _mm512_cvt_roundps_epi32(f0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512i i1 =
                _mm512_cvt_roundps_epi32(f1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512i i2 =
                _mm512_cvt_roundps_epi32(f2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512i i3 =
                _mm512_cvt_roundps_epi32(f3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512i p16_0 = _mm512_packs_epi32(i0, i1);
            __m512i p16_1 = _mm512_packs_epi32(i2, i3);
            __m512i p8 = _mm512_permutexvar_epi32(pack_perm, _mm512_packs_epi16(p16_0, p16_1));
            if (row_nt) {
                _mm512_stream_si512((__m512i *)(qr + i), p8);
            } else {
                _mm512_storeu_si512((void *)(qr + i), p8);
            }
        }
        for (; i + 16 <= k; i += 16) {
            __m512 v = _mm512_mul_ps(_mm512_loadu_ps(xr + i), vinv);
            v = _mm512_max_ps(_mm512_min_ps(v, v_max), v_min);
            __m512i vi = _mm512_cvt_roundps_epi32(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i q8 = _mm512_cvtsepi32_epi8(vi);
            _mm_storeu_si128((__m128i *)(qr + i), q8);
        }
        if (i < k) {
            __mmask16 m = (__mmask16)((1U << (k - i)) - 1U);
            __m512 v = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, xr + i), vinv);
            v = _mm512_max_ps(_mm512_min_ps(v, v_max), v_min);
            __m512i vi = _mm512_cvt_roundps_epi32(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i q8 = _mm512_cvtsepi32_epi8(vi);
            _mm_mask_storeu_epi8(qr + i, m, q8);
        }
    }
    if (use_nt) {
        _mm_sfence();
    }
}
