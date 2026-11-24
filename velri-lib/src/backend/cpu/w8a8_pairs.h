/* Widen two weights per channel, then reuse the block across activation rows. */
#if defined(__AVX2__) || defined(__SSE2__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__AVX2__)
enum { LANES = 8 };
typedef __m256i sums_vec;
static inline sums_vec add_pairs(sums_vec sum, const int16_t *w, int32_t pair) {
    return _mm256_add_epi32(
        sum, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i_u *)w), _mm256_set1_epi32(pair)));
}
#else
enum { LANES = 4 };
typedef __m128i sums_vec;
static inline sums_vec add_pairs(sums_vec sum, const int16_t *w, int32_t pair) {
    return _mm_add_epi32(
        sum, _mm_madd_epi16(_mm_loadu_si128((const __m128i_u *)w), _mm_set1_epi32(pair)));
}
#endif
#else
enum { LANES = 4 };
typedef struct {
    uint32_t col[LANES];
} sums_vec;
static inline sums_vec add_pairs(sums_vec sum, const int16_t *w, int32_t pair) {
    for (size_t j = 0; j < LANES; ++j)
        sum.col[j] +=
            (uint32_t)((int16_t)pair * w[2 * j] + (int16_t)((uint32_t)pair >> 16) * w[2 * j + 1]);
    return sum;
}
#endif

enum { GROUPS = 4, COLS = LANES * GROUPS, ROWS = 2 };
static_assert(NR % COLS == 0);
static_assert(sizeof(sums_vec) == LANES * sizeof(uint32_t));
static inline void pairs_tile(const int8_t *x, size_t xs, const int16_t *weights, size_t depth,
                              uint32_t *y, size_t ys, size_t rows, bool first) {
    /* Convert each activation pair once so the dot loop can broadcast a single word. */
    int32_t activations[ROWS][KC / 2];
    for (size_t r = 0; r < rows; ++r) {
        for (size_t k = 0; k < depth; k += 2) {
            const int8_t *a = x + r * xs + k;
            activations[r][k / 2] =
                (int32_t)((uint16_t)(int16_t)a[0] | (uint32_t)(uint16_t)(int16_t)a[1] << 16);
        }
    }
    sums_vec sums[ROWS][GROUPS] = {};
    if (!first) {
        for (size_t r = 0; r < rows; ++r)
            memcpy(sums[r], y + r * ys, sizeof(sums[r]));
    }
    for (size_t k = 0; k < depth; k += 2) {
        for (size_t r = 0; r < rows; ++r) {
            int32_t pair = activations[r][k / 2];
            for (size_t j = 0; j < GROUPS; ++j)
                sums[r][j] = add_pairs(sums[r][j], weights + k * COLS + j * 2 * LANES, pair);
        }
    }
    for (size_t r = 0; r < rows; ++r)
        memcpy(y + r * ys, sums[r], sizeof(sums[r]));
}

static void multiply_panel(const int8_t *x, size_t xs, const uint8_t *panel, size_t depth,
                           uint32_t *y, size_t ys, size_t rows, bool first) {
    alignas(64) int16_t weights[KC * COLS];
    for (size_t c = 0; c < NR; c += COLS) {
        for (size_t k = 0; k < depth; k += 2)
            for (size_t j = 0; j < COLS; ++j) {
                const int8_t *pair = (const int8_t *)panel + k * NR + c + j;
                weights[k * COLS + 2 * j] = pair[0];
                weights[k * COLS + 2 * j + 1] = pair[NR];
            }
        size_t r = 0;
        for (; rows - r >= ROWS; r += ROWS)
            pairs_tile(x + r * xs, xs, weights, depth, y + r * ys + c, ys, ROWS, first);
        if (r < rows)
            pairs_tile(x + r * xs, xs, weights, depth, y + r * ys + c, ys, 1, first);
    }
}
