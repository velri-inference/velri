/* Four activation rows by 64 output channels, using unsigned weights and signed activations. */
#include <immintrin.h>

enum { MR = 4 };
static_assert(NR == 64, "Each bit of the output mask selects one panel channel");

static uint32_t row_correction(const int8_t *x, size_t depth) {
    __m512i sum = _mm512_setzero_si512();
    for (size_t k = 0; k < depth; k += 64)
        sum = _mm512_dpbusd_epi32(sum, _mm512_set1_epi8((char)128), _mm512_loadu_si512(x + k));
    return (uint32_t)_mm512_reduce_add_epi32(sum);
}

static inline void transpose16(__m512i rows[16], uint8_t *dst, size_t stride, __m512i flip) {
    __m512i a[16], b[16];
    for (int i = 0; i < 8; ++i) {
        a[2 * i] = _mm512_unpacklo_epi32(rows[2 * i], rows[2 * i + 1]);
        a[2 * i + 1] = _mm512_unpackhi_epi32(rows[2 * i], rows[2 * i + 1]);
    }
    for (int i = 0; i < 4; ++i) {
        b[4 * i] = _mm512_unpacklo_epi64(a[4 * i], a[4 * i + 2]);
        b[4 * i + 1] = _mm512_unpackhi_epi64(a[4 * i], a[4 * i + 2]);
        b[4 * i + 2] = _mm512_unpacklo_epi64(a[4 * i + 1], a[4 * i + 3]);
        b[4 * i + 3] = _mm512_unpackhi_epi64(a[4 * i + 1], a[4 * i + 3]);
    }
    for (int i = 0; i < 4; ++i) {
        __m512i lo = _mm512_shuffle_i32x4(b[i], b[4 + i], 0x44);
        __m512i hi = _mm512_shuffle_i32x4(b[8 + i], b[12 + i], 0x44);
        _mm512_storeu_si512(dst + (i * stride),
                            _mm512_xor_si512(_mm512_shuffle_i32x4(lo, hi, 0x88), flip));
        _mm512_storeu_si512(dst + ((4 + i) * stride),
                            _mm512_xor_si512(_mm512_shuffle_i32x4(lo, hi, 0xdd), flip));
        lo = _mm512_shuffle_i32x4(b[i], b[4 + i], 0xee);
        hi = _mm512_shuffle_i32x4(b[8 + i], b[12 + i], 0xee);
        _mm512_storeu_si512(dst + ((8 + i) * stride),
                            _mm512_xor_si512(_mm512_shuffle_i32x4(lo, hi, 0x88), flip));
        _mm512_storeu_si512(dst + ((12 + i) * stride),
                            _mm512_xor_si512(_mm512_shuffle_i32x4(lo, hi, 0xdd), flip));
    }
}

static void pack_panel(const int8_t *w, size_t ws, uint8_t *dst, size_t n) {
    const __m512i flip = _mm512_set1_epi8((char)128);
    for (size_t j = 0; j < NR; j += 16) {
        for (size_t q = 0; q < n; q += 64) {
            __m512i v[16];
            for (size_t i = 0; i < 16; ++i)
                v[i] = _mm512_loadu_si512(w + (j + i) * ws + q);
            transpose16(v, dst + q * NR + 4 * j, 4 * NR, flip);
        }
    }
}

typedef struct {
    float *dst;
    size_t stride;
    const float *row_scale;
    const float *column_scale;
    const float *bias;
    const uint32_t *correction;
    uint64_t lanes;
} panel_output;

/* Subtract the zero-point correction modulo 2^32 before converting to F32.
   Neighboring workers may own channels in the same panel. */
static inline void store_output(const __m512i acc[NR / 16], const panel_output *out, size_t row) {
    __m512i correction = _mm512_set1_epi32(out->correction[row]);
    __m512 row_scale = _mm512_set1_ps(out->row_scale[row]);
    for (size_t j = 0; j < NR; j += 16) {
        __m512 value = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[j / 16], correction));
        __m512 scale = _mm512_mul_ps(_mm512_loadu_ps(out->column_scale + j), row_scale);
        value = _mm512_mul_ps(value, scale);
        if (out->bias)
            value = _mm512_add_ps(value, _mm512_loadu_ps(out->bias + j));
        _mm512_mask_storeu_ps(out->dst + row * out->stride + j, (__mmask16)(out->lanes >> j),
                              value);
    }
}

[[gnu::always_inline]] static inline void tile(size_t rows, const int8_t *x, size_t xs,
                                               const uint8_t *panel, size_t n, uint32_t *y,
                                               size_t ys, bool first, const panel_output *out,
                                               size_t row) {
    __m512i acc[MR][NR / 16];
    for (size_t r = 0; r < rows; ++r)
        for (size_t j = 0; j < NR / 16; ++j)
            acc[r][j] = first ? _mm512_setzero_si512() : _mm512_loadu_si512(y + r * ys + j * 16);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 2
#endif
    for (size_t k = 0; k < n; k += 4) {
        __m512i weights[NR / 16];
        for (size_t j = 0; j < NR / 16; ++j)
            weights[j] = _mm512_loadu_si512(panel + k * NR + 64 * j);
        for (size_t r = 0; r < rows; ++r) {
            int32_t word;
            memcpy(&word, x + r * xs + k, 4);
            __m512i a = _mm512_set1_epi32(word);
            for (size_t j = 0; j < NR / 16; ++j)
                acc[r][j] = _mm512_dpbusd_epi32(acc[r][j], weights[j], a);
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        if (out) {
            store_output(acc[r], out, row + r);
        } else {
            for (size_t j = 0; j < NR / 16; ++j)
                _mm512_storeu_si512(y + r * ys + j * 16, acc[r][j]);
        }
    }
}

[[gnu::always_inline]] static inline void multiply_panel(const int8_t *x, size_t xs,
                                                         const uint8_t *panel, size_t depth,
                                                         uint32_t *y, size_t ys, size_t rows,
                                                         bool first, const panel_output *out) {
    size_t r = 0;
    for (; rows - r >= MR; r += MR)
        tile(MR, x + r * xs, xs, panel, depth, y + r * ys, ys, first, out, r);
    for (; r < rows; ++r)
        tile(1, x + r * xs, xs, panel, depth, y + r * ys, ys, first, out, r);
}
