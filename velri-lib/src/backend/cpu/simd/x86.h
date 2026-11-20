#ifndef VELRI_SIMD_X86_H
#define VELRI_SIMD_X86_H

#include "internal.h"
#include <immintrin.h>

/* This header describes vector primitives. Algorithms live in kernels/ and
 * are compiled once for each target; there is no ISA dispatch inside a loop. */
#define VELRI_JOIN_(a, b) a##b
#define VELRI_JOIN(a, b) VELRI_JOIN_(a, b)

#if defined(__AVX512F__)
#define SIMD_NAME(name) VELRI_JOIN(name, _avx512)
#define SIMD_OP(name) VELRI_JOIN(_mm512_, name)
typedef __m512 simd_f32;
typedef __m512d simd_f64;
enum { SIMD_F32_LANES = 16, SIMD_F64_LANES = 8, SIMD_BYTES = 64 };
enum { SIMD_GEMV_UNROLL = 2, SIMD_GEMV_F64_ROWS = 4, SIMD_GEMV_PAIR_TILES = 1 };

static inline float simd_sum_f32(simd_f32 x) { return _mm512_reduce_add_ps(x); }
static inline double simd_sum_f64(simd_f64 x) { return _mm512_reduce_add_pd(x); }

static inline simd_f32 simd_load_f16(const uint16_t *x) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)x));
}
static inline simd_f32 simd_load_bf16(const uint16_t *x) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)x)), 16));
}
static inline simd_f32 simd_load_i8(const int8_t *x) {
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)x)));
}
#else
#define SIMD_NAME(name) VELRI_JOIN(name, _avx2)
#define SIMD_OP(name) VELRI_JOIN(_mm256_, name)
typedef __m256 simd_f32;
typedef __m256d simd_f64;
enum { SIMD_F32_LANES = 8, SIMD_F64_LANES = 4, SIMD_BYTES = 32 };
enum { SIMD_GEMV_UNROLL = 1, SIMD_GEMV_F64_ROWS = 1, SIMD_GEMV_PAIR_TILES = 0 };

static inline float simd_sum_f32(simd_f32 x) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehl_ps(s, s)));
}
static inline double simd_sum_f64(simd_f64 x) {
    __m128d s = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}
static inline simd_f32 simd_load_f16(const uint16_t *x) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)x));
}
static inline simd_f32 simd_load_bf16(const uint16_t *x) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)x)), 16));
}
static inline simd_f32 simd_load_i8(const int8_t *x) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)x)));
}
#endif

static inline float simd_load1_f32(velri_dtype dtype, const void *x, size_t i) {
    switch (dtype) {
    case VELRI_DTYPE_F32: return ((const float *)x)[i];
    case VELRI_DTYPE_F16: return velri_f16_to_f32_scalar(((const velri_fp16 *)x)[i]);
    case VELRI_DTYPE_BF16: return velri_bf16_to_f32_scalar(((const velri_bf16 *)x)[i]);
    case VELRI_DTYPE_I8: return (float)((const int8_t *)x)[i];
    default: return 0.0f;
    }
}

static inline simd_f32 simd_load_as_f32(velri_dtype dtype, const void *x, size_t i) {
    switch (dtype) {
    case VELRI_DTYPE_F32: return SIMD_OP(loadu_ps)((const float *)x + i);
    case VELRI_DTYPE_F16: return simd_load_f16((const uint16_t *)x + i);
    case VELRI_DTYPE_BF16: return simd_load_bf16((const uint16_t *)x + i);
    case VELRI_DTYPE_I8: return simd_load_i8((const int8_t *)x + i);
    default: return SIMD_OP(setzero_ps)();
    }
}

#endif
