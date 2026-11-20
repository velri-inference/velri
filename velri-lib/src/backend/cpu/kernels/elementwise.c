#include "simd/x86.h"

void SIMD_NAME(velri_vec_axpby_f32)(size_t i0, size_t i1, float alpha, const float *a, float beta,
                               const float *b, float *y) {
    const simd_f32 va = SIMD_OP(set1_ps)(alpha);
    const simd_f32 vb = SIMD_OP(set1_ps)(beta);
    size_t i = i0;

    if (beta == 0.0f) {
        if ((i1 - i0) * 8ULL >= velri_nt_io_bytes() && y != a &&
            (((uintptr_t)(y + i)) & 3ULL) == 0) {
            while (i < i1 && (((uintptr_t)(y + i)) & (SIMD_BYTES - 1)) != 0) {
                y[i] = alpha * a[i];
                ++i;
            }
            for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
                if (SIMD_BYTES == 64 && i1 - i > 256) {
                    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
                    if (beta != 0.0f && b) _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);
                }
                SIMD_OP(stream_ps)(y + i, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), va));
                SIMD_OP(stream_ps)(y + i + SIMD_F32_LANES, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES), va));
                SIMD_OP(stream_ps)(y + i + 2 * SIMD_F32_LANES, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES), va));
                SIMD_OP(stream_ps)(y + i + 3 * SIMD_F32_LANES, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES), va));
            }
            _mm_sfence();
        }
        for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
            SIMD_OP(storeu_ps)(y + i, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), va));
            SIMD_OP(storeu_ps)(y + i + SIMD_F32_LANES, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES), va));
            SIMD_OP(storeu_ps)(y + i + 2 * SIMD_F32_LANES, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES), va));
            SIMD_OP(storeu_ps)(y + i + 3 * SIMD_F32_LANES, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES), va));
        }
        for (; i + SIMD_F32_LANES <= i1; i += SIMD_F32_LANES) {
            SIMD_OP(storeu_ps)(y + i, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), va));
        }
        for (; i < i1; ++i) {
            y[i] = alpha * a[i];
        }
        return;
    }

    const bool use_nt = (i1 - i0) * 12ULL >= velri_nt_io_bytes() && y != a && y != b &&
                        (((uintptr_t)(y + i)) & 3ULL) == 0;
    if (alpha == 1.0f && beta == 1.0f) {
        if (use_nt) {
            while (i < i1 && (((uintptr_t)(y + i)) & (SIMD_BYTES - 1)) != 0) {
                y[i] = a[i] + b[i];
                ++i;
            }
            for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
                if (SIMD_BYTES == 64 && i1 - i > 256) {
                    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
                    if (beta != 0.0f && b) _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);
                }
                SIMD_OP(stream_ps)(y + i,
                                 SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i), SIMD_OP(loadu_ps)(b + i)));
                SIMD_OP(stream_ps)(y + i + SIMD_F32_LANES, SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES),
                                                          SIMD_OP(loadu_ps)(b + i + SIMD_F32_LANES)));
                SIMD_OP(stream_ps)(y + i + 2 * SIMD_F32_LANES, SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES),
                                                           SIMD_OP(loadu_ps)(b + i + 2 * SIMD_F32_LANES)));
                SIMD_OP(stream_ps)(y + i + 3 * SIMD_F32_LANES, SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES),
                                                           SIMD_OP(loadu_ps)(b + i + 3 * SIMD_F32_LANES)));
            }
            _mm_sfence();
        }
        for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
            SIMD_OP(storeu_ps)(y + i, SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i), SIMD_OP(loadu_ps)(b + i)));
            SIMD_OP(storeu_ps)(y + i + SIMD_F32_LANES,
                             SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES), SIMD_OP(loadu_ps)(b + i + SIMD_F32_LANES)));
            SIMD_OP(storeu_ps)(y + i + 2 * SIMD_F32_LANES, SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES),
                                                       SIMD_OP(loadu_ps)(b + i + 2 * SIMD_F32_LANES)));
            SIMD_OP(storeu_ps)(y + i + 3 * SIMD_F32_LANES, SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES),
                                                       SIMD_OP(loadu_ps)(b + i + 3 * SIMD_F32_LANES)));
        }
        for (; i + SIMD_F32_LANES <= i1; i += SIMD_F32_LANES) {
            SIMD_OP(storeu_ps)(y + i, SIMD_OP(add_ps)(SIMD_OP(loadu_ps)(a + i), SIMD_OP(loadu_ps)(b + i)));
        }
        for (; i < i1; ++i) {
            y[i] = a[i] + b[i];
        }
    } else if (alpha == 1.0f) {
        if (use_nt) {
            while (i < i1 && (((uintptr_t)(y + i)) & (SIMD_BYTES - 1)) != 0) {
                y[i] = a[i] + beta * b[i];
                ++i;
            }
            for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
                if (SIMD_BYTES == 64 && i1 - i > 256) {
                    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
                    if (beta != 0.0f && b) _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);
                }
                SIMD_OP(stream_ps)(
                    y + i, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i), vb, SIMD_OP(loadu_ps)(a + i)));
                SIMD_OP(stream_ps)(y + i + SIMD_F32_LANES, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + SIMD_F32_LANES), vb,
                                                            SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES)));
                SIMD_OP(stream_ps)(y + i + 2 * SIMD_F32_LANES, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + 2 * SIMD_F32_LANES), vb,
                                                             SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES)));
                SIMD_OP(stream_ps)(y + i + 3 * SIMD_F32_LANES, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + 3 * SIMD_F32_LANES), vb,
                                                             SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES)));
            }
            _mm_sfence();
        }
        for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
            SIMD_OP(storeu_ps)(y + i,
                             SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i), vb, SIMD_OP(loadu_ps)(a + i)));
            SIMD_OP(storeu_ps)(y + i + SIMD_F32_LANES, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + SIMD_F32_LANES), vb,
                                                        SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES)));
            SIMD_OP(storeu_ps)(y + i + 2 * SIMD_F32_LANES, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + 2 * SIMD_F32_LANES), vb,
                                                         SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES)));
            SIMD_OP(storeu_ps)(y + i + 3 * SIMD_F32_LANES, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + 3 * SIMD_F32_LANES), vb,
                                                         SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES)));
        }
        for (; i + SIMD_F32_LANES <= i1; i += SIMD_F32_LANES) {
            SIMD_OP(storeu_ps)(y + i,
                             SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i), vb, SIMD_OP(loadu_ps)(a + i)));
        }
        for (; i < i1; ++i) {
            y[i] = a[i] + beta * b[i];
        }
    } else {
        if (use_nt) {
            while (i < i1 && (((uintptr_t)(y + i)) & (SIMD_BYTES - 1)) != 0) {
                y[i] = alpha * a[i] + beta * b[i];
                ++i;
            }
            for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
                if (SIMD_BYTES == 64 && i1 - i > 256) {
                    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
                    if (beta != 0.0f && b) _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);
                }
                SIMD_OP(stream_ps)(y + i, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i), vb,
                                                        SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), va)));
                SIMD_OP(stream_ps)(y + i + SIMD_F32_LANES,
                                 SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + SIMD_F32_LANES), vb,
                                                 SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES), va)));
                SIMD_OP(stream_ps)(y + i + 2 * SIMD_F32_LANES,
                                 SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + 2 * SIMD_F32_LANES), vb,
                                                 SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES), va)));
                SIMD_OP(stream_ps)(y + i + 3 * SIMD_F32_LANES,
                                 SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + 3 * SIMD_F32_LANES), vb,
                                                 SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES), va)));
            }
            _mm_sfence();
        }
        for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
            SIMD_OP(storeu_ps)(y + i, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i), vb,
                                                    SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), va)));
            SIMD_OP(storeu_ps)(y + i + SIMD_F32_LANES,
                             SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + SIMD_F32_LANES), vb,
                                             SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES), va)));
            SIMD_OP(storeu_ps)(y + i + 2 * SIMD_F32_LANES,
                             SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + 2 * SIMD_F32_LANES), vb,
                                             SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES), va)));
            SIMD_OP(storeu_ps)(y + i + 3 * SIMD_F32_LANES,
                             SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i + 3 * SIMD_F32_LANES), vb,
                                             SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES), va)));
        }
        for (; i + SIMD_F32_LANES <= i1; i += SIMD_F32_LANES) {
            SIMD_OP(storeu_ps)(y + i, SIMD_OP(fmadd_ps)(SIMD_OP(loadu_ps)(b + i), vb,
                                                    SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), va)));
        }
        for (; i < i1; ++i) {
            y[i] = alpha * a[i] + beta * b[i];
        }
    }
}

void SIMD_NAME(velri_vec_mul_f32)(size_t i0, size_t i1, const float *a, const float *b, float *y) {
    size_t i = i0;
    if ((i1 - i0) * 12ULL >= velri_nt_io_bytes() && y != a && y != b &&
        (((uintptr_t)(y + i)) & 3ULL) == 0) {
        while (i < i1 && (((uintptr_t)(y + i)) & (SIMD_BYTES - 1)) != 0) {
            y[i] = a[i] * b[i];
            ++i;
        }
        for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
            if (SIMD_BYTES == 64 && i1 - i > 256) {
                _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
                if (b) _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);
            }
            SIMD_OP(stream_ps)(y + i, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), SIMD_OP(loadu_ps)(b + i)));
            SIMD_OP(stream_ps)(y + i + SIMD_F32_LANES,
                             SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES), SIMD_OP(loadu_ps)(b + i + SIMD_F32_LANES)));
            SIMD_OP(stream_ps)(y + i + 2 * SIMD_F32_LANES, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES),
                                                       SIMD_OP(loadu_ps)(b + i + 2 * SIMD_F32_LANES)));
            SIMD_OP(stream_ps)(y + i + 3 * SIMD_F32_LANES, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES),
                                                       SIMD_OP(loadu_ps)(b + i + 3 * SIMD_F32_LANES)));
        }
        _mm_sfence();
    }
    for (; i + 4 * SIMD_F32_LANES <= i1; i += 4 * SIMD_F32_LANES) {
        SIMD_OP(storeu_ps)(y + i, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), SIMD_OP(loadu_ps)(b + i)));
        SIMD_OP(storeu_ps)(y + i + SIMD_F32_LANES,
                         SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + SIMD_F32_LANES), SIMD_OP(loadu_ps)(b + i + SIMD_F32_LANES)));
        SIMD_OP(storeu_ps)(y + i + 2 * SIMD_F32_LANES,
                         SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 2 * SIMD_F32_LANES), SIMD_OP(loadu_ps)(b + i + 2 * SIMD_F32_LANES)));
        SIMD_OP(storeu_ps)(y + i + 3 * SIMD_F32_LANES,
                         SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i + 3 * SIMD_F32_LANES), SIMD_OP(loadu_ps)(b + i + 3 * SIMD_F32_LANES)));
    }
    for (; i + SIMD_F32_LANES <= i1; i += SIMD_F32_LANES) {
        SIMD_OP(storeu_ps)(y + i, SIMD_OP(mul_ps)(SIMD_OP(loadu_ps)(a + i), SIMD_OP(loadu_ps)(b + i)));
    }
    for (; i < i1; ++i) {
        y[i] = a[i] * b[i];
    }
}
