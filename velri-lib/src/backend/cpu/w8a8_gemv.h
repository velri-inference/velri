/* Row-major W8A8 decode. The caller supplies a depth divisible by 64. */
#include <immintrin.h>

#if defined(VELRI_W8A8_USE_VNNI)
enum { GEMV_CHANNELS = 16, GEMV_VECTOR_BYTES = 64 };
typedef __m512i gemv_accumulator;
static inline uint32_t gemv_sum(__m512i v) {
    return (uint32_t)_mm512_reduce_add_epi32(v);
}

static uint32_t gemv_correction(const int8_t *x, size_t depth) {
    return 0U - row_correction(x, depth);
}
#else
enum { GEMV_CHANNELS = 4, GEMV_VECTOR_BYTES = 32 };
typedef __m256i gemv_accumulator;
static inline uint32_t gemv_sum(__m256i v) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
    return (uint32_t)_mm_cvtsi128_si32(sum);
}

/* For negative x the kernel multiplies |x| by ~w = -w-1. Adding |x| once per
 * negative input recovers x*w for every signed byte, including -128. */
static uint32_t gemv_correction(const int8_t *x, size_t depth) {
    __m256i sum = _mm256_setzero_si256();
    for (size_t k = 0; k < depth; k += GEMV_VECTOR_BYTES) {
        __m256i a = _mm256_loadu_si256((const __m256i_u *)(x + k));
        __m256i negative = _mm256_cmpgt_epi8(_mm256_setzero_si256(), a);
        __m256i magnitude = _mm256_and_si256(negative, _mm256_abs_epi8(a));
        sum = _mm256_add_epi32(sum, _mm256_sad_epu8(magnitude, _mm256_setzero_si256()));
    }
    return gemv_sum(sum);
}
#endif

[[gnu::always_inline]] static inline void gemv_channels(
    const int8_t *x, const int8_t *w, size_t stride, size_t depth, const float *wscale,
    const float *bias, float *y, float xscale, uint32_t correction, size_t channels) {
    gemv_accumulator acc[GEMV_CHANNELS] = {};
    for (size_t k = 0; k < depth; k += GEMV_VECTOR_BYTES) {
#if defined(VELRI_W8A8_USE_VNNI)
        __m512i a = _mm512_loadu_si512(x + k);
        for (size_t j = 0; j < channels; ++j) {
            __m512i b = _mm512_loadu_si512(w + j * stride + k);
            b = _mm512_xor_si512(b, _mm512_set1_epi8((char)128));
            acc[j] = _mm512_dpbusd_epi32(acc[j], b, a);
        }
#else
        __m256i a = _mm256_loadu_si256((const __m256i_u *)(x + k));
        __m256i negative = _mm256_cmpgt_epi8(_mm256_setzero_si256(), a);
        __m256i magnitude = _mm256_abs_epi8(a);
        for (size_t j = 0; j < channels; ++j) {
            __m256i b = _mm256_loadu_si256((const __m256i_u *)(w + j * stride + k));
            /* Each product is in [-16384, 16256], so the sum of a pair fits int16. */
            __m256i pair = _mm256_maddubs_epi16(magnitude, _mm256_xor_si256(b, negative));
            acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(pair, _mm256_set1_epi16(1)));
        }
#endif
    }
    for (size_t j = 0; j < channels; ++j) {
        float scaled = (float)(int32_t)(gemv_sum(acc[j]) + correction) * (wscale[j] * xscale);
        y[j] = bias ? scaled + bias[j] : scaled;
    }
}

static void raw_gemv(const int8_t *x, const int8_t *w, size_t stride, size_t depth,
                     const float *bias, float xscale, const float *wscale, float *y,
                     size_t c0, size_t c1) {
    uint32_t correction = gemv_correction(x, depth);
    size_t c = c0;
    // A long streaming slice benefits from reading one weight row to completion.
    // Smaller slices reuse the activation across channels in the private cache.
    if (depth < 4096 || (c1 - c0) * depth <= 2U * 1024U * 1024U) {
        for (; c1 - c >= GEMV_CHANNELS; c += GEMV_CHANNELS) {
            gemv_channels(x, w + c * stride, stride, depth, wscale + c, bias ? bias + c : nullptr,
                          y + c, xscale, correction, GEMV_CHANNELS);
        }
    }
    for (; c < c1; ++c) {
        gemv_channels(x, w + c * stride, stride, depth, wscale + c, bias ? bias + c : nullptr,
                      y + c, xscale, correction, 1);
    }
}
