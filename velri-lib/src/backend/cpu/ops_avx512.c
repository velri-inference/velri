#include "internal.h"
#include <immintrin.h>

/*
 * Minimax Cody-Waite degree-6 `expf` in 512-bit ZMM (degree-6 approximation;
 * returns exact 0.0f for x <= -105.0f including -INFINITY via `vscalefps`).
 */
static inline __m512 exp512_ps(__m512 orig_x) {
    const __m512 min_x = _mm512_set1_ps(-105.0f);
    const __m512 max_x = _mm512_set1_ps(88.0f);
    __m512 x = _mm512_min_ps(_mm512_max_ps(orig_x, min_x), max_x);

    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693145751953125f);
    const __m512 ln2_lo = _mm512_set1_ps(1.428606765330187045e-6f);

    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, ln2_lo, _mm512_fnmadd_ps(n, ln2_hi, x));

    __m512 p = _mm512_set1_ps(1.38888889e-3f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.33333333e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.16666667e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.66666667e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.00000000e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

static inline void exp512_x4(__m512 *io0, __m512 *io1, __m512 *io2, __m512 *io3) {
    const __m512 min_x = _mm512_set1_ps(-105.0f);
    const __m512 max_x = _mm512_set1_ps(88.0f);
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693145751953125f);
    const __m512 ln2_lo = _mm512_set1_ps(1.428606765330187045e-6f);

    __m512 x0 = _mm512_min_ps(_mm512_max_ps(*io0, min_x), max_x);
    __m512 x1 = _mm512_min_ps(_mm512_max_ps(*io1, min_x), max_x);
    __m512 x2 = _mm512_min_ps(_mm512_max_ps(*io2, min_x), max_x);
    __m512 x3 = _mm512_min_ps(_mm512_max_ps(*io3, min_x), max_x);

    __m512 n0 = _mm512_roundscale_ps(_mm512_mul_ps(x0, log2e),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 n1 = _mm512_roundscale_ps(_mm512_mul_ps(x1, log2e),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 n2 = _mm512_roundscale_ps(_mm512_mul_ps(x2, log2e),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 n3 = _mm512_roundscale_ps(_mm512_mul_ps(x3, log2e),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m512 r0 = _mm512_fnmadd_ps(n0, ln2_lo, _mm512_fnmadd_ps(n0, ln2_hi, x0));
    __m512 r1 = _mm512_fnmadd_ps(n1, ln2_lo, _mm512_fnmadd_ps(n1, ln2_hi, x1));
    __m512 r2 = _mm512_fnmadd_ps(n2, ln2_lo, _mm512_fnmadd_ps(n2, ln2_hi, x2));
    __m512 r3 = _mm512_fnmadd_ps(n3, ln2_lo, _mm512_fnmadd_ps(n3, ln2_hi, x3));

    const __m512 c6 = _mm512_set1_ps(1.38888889e-3f);
    const __m512 c5 = _mm512_set1_ps(8.33333333e-3f);
    const __m512 c4 = _mm512_set1_ps(4.16666667e-2f);
    const __m512 c3 = _mm512_set1_ps(1.66666667e-1f);
    const __m512 c2 = _mm512_set1_ps(5.00000000e-1f);
    const __m512 c1 = _mm512_set1_ps(1.0f);

    __m512 p0 = _mm512_fmadd_ps(c6, r0, c5);
    __m512 p1 = _mm512_fmadd_ps(c6, r1, c5);
    __m512 p2 = _mm512_fmadd_ps(c6, r2, c5);
    __m512 p3 = _mm512_fmadd_ps(c6, r3, c5);

    p0 = _mm512_fmadd_ps(p0, r0, c4);
    p1 = _mm512_fmadd_ps(p1, r1, c4);
    p2 = _mm512_fmadd_ps(p2, r2, c4);
    p3 = _mm512_fmadd_ps(p3, r3, c4);

    p0 = _mm512_fmadd_ps(p0, r0, c3);
    p1 = _mm512_fmadd_ps(p1, r1, c3);
    p2 = _mm512_fmadd_ps(p2, r2, c3);
    p3 = _mm512_fmadd_ps(p3, r3, c3);

    p0 = _mm512_fmadd_ps(p0, r0, c2);
    p1 = _mm512_fmadd_ps(p1, r1, c2);
    p2 = _mm512_fmadd_ps(p2, r2, c2);
    p3 = _mm512_fmadd_ps(p3, r3, c2);

    p0 = _mm512_fmadd_ps(p0, r0, c1);
    p1 = _mm512_fmadd_ps(p1, r1, c1);
    p2 = _mm512_fmadd_ps(p2, r2, c1);
    p3 = _mm512_fmadd_ps(p3, r3, c1);

    p0 = _mm512_fmadd_ps(p0, r0, c1);
    p1 = _mm512_fmadd_ps(p1, r1, c1);
    p2 = _mm512_fmadd_ps(p2, r2, c1);
    p3 = _mm512_fmadd_ps(p3, r3, c1);

    *io0 = _mm512_scalef_ps(p0, n0);
    *io1 = _mm512_scalef_ps(p1, n1);
    *io2 = _mm512_scalef_ps(p2, n2);
    *io3 = _mm512_scalef_ps(p3, n3);
}

static inline float expf_fast_avx512(float orig_x) {
    __m128 vx = _mm_set_ss(orig_x);
    vx = _mm_min_ss(_mm_max_ss(vx, _mm_set_ss(-105.0f)), _mm_set_ss(88.0f));
    __m128 vm = _mm_mul_ss(vx, _mm_set_ss(1.44269504088896341f));
    __m128 vn = _mm_roundscale_ss(vm, vm, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128 vr = _mm_fnmadd_ss(vn, _mm_set_ss(0.693145751953125f), vx);
    vr = _mm_fnmadd_ss(vn, _mm_set_ss(1.428606765330187045e-6f), vr);
    __m128 vp = _mm_set_ss(1.38888889e-3f);
    vp = _mm_fmadd_ss(vp, vr, _mm_set_ss(8.33333333e-3f));
    vp = _mm_fmadd_ss(vp, vr, _mm_set_ss(4.16666667e-2f));
    vp = _mm_fmadd_ss(vp, vr, _mm_set_ss(1.66666667e-1f));
    vp = _mm_fmadd_ss(vp, vr, _mm_set_ss(5.00000000e-1f));
    vp = _mm_fmadd_ss(vp, vr, _mm_set_ss(1.0f));
    vp = _mm_fmadd_ss(vp, vr, _mm_set_ss(1.0f));
    return _mm_cvtss_f32(_mm_scalef_ss(vp, vn));
}

static inline __m512 load16_as_f32_avx512(velri_dtype dt, const void *base, size_t idx) {
    if (dt == VELRI_DTYPE_F32) {
        return _mm512_loadu_ps((const float *)base + idx);
    }
    if (dt == VELRI_DTYPE_F16) {
        return _mm512_cvtph_ps(
            _mm256_loadu_si256((const __m256i *)((const velri_fp16 *)base + idx)));
    }
    __m256i h = _mm256_loadu_si256((const __m256i *)((const velri_bf16 *)base + idx));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

static inline __m256i cvt16_f32_to_bf16_zmm(__m512 v) {
    __m512i u = _mm512_castps_si512(v);
    __mmask16 is_nan = _mm512_cmpgt_epi32_mask(_mm512_and_epi32(u, _mm512_set1_epi32(0x7FFFFFFF)),
                                               _mm512_set1_epi32(0x7F800000));
    __m512i lsb = _mm512_and_epi32(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x00007FFF), lsb));
    r = _mm512_mask_mov_epi32(r, is_nan, _mm512_or_epi32(u, _mm512_set1_epi32(0x00400000)));
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16));
}

static inline void store16_from_f32_avx512(velri_dtype dt, void *base, size_t idx, __m512 v) {
    if (dt == VELRI_DTYPE_F32) {
        _mm512_storeu_ps((float *)base + idx, v);
        return;
    }
    if (dt == VELRI_DTYPE_F16) {
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256((__m256i *)((velri_fp16 *)base + idx), h);
        return;
    }
    _mm256_storeu_si256((__m256i *)((velri_bf16 *)base + idx), cvt16_f32_to_bf16_zmm(v));
}

static inline void stream16_from_f32_avx512(velri_dtype dt, void *base, size_t idx, __m512 v) {
    if (dt == VELRI_DTYPE_F32) {
        float *p = (float *)base + idx;
        if (((uintptr_t)p & 63ULL) == 0) {
            _mm512_stream_ps(p, v);
        } else {
            _mm512_storeu_ps(p, v);
        }
        return;
    }
    if (dt == VELRI_DTYPE_F16) {
        velri_fp16 *p = (velri_fp16 *)base + idx;
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        if (((uintptr_t)p & 31ULL) == 0) {
            _mm256_stream_si256((__m256i *)p, h);
        } else {
            _mm256_storeu_si256((__m256i *)p, h);
        }
        return;
    }
    velri_bf16 *p = (velri_bf16 *)base + idx;
    __m256i b = cvt16_f32_to_bf16_zmm(v);
    if (((uintptr_t)p & 31ULL) == 0) {
        _mm256_stream_si256((__m256i *)p, b);
    } else {
        _mm256_storeu_si256((__m256i *)p, b);
    }
}

static inline float load1_f32(velri_dtype dt, const void *base, size_t idx) {
    if (dt == VELRI_DTYPE_F32) {
        return ((const float *)base)[idx];
    }
    if (dt == VELRI_DTYPE_F16) {
        return velri_f16_to_f32_scalar(((const velri_fp16 *)base)[idx]);
    }
    return velri_bf16_to_f32_scalar(((const velri_bf16 *)base)[idx]);
}

static inline void store1_f32(velri_dtype dt, void *base, size_t idx, float v) {
    if (dt == VELRI_DTYPE_F32) {
        ((float *)base)[idx] = v;
    } else if (dt == VELRI_DTYPE_F16) {
        ((velri_fp16 *)base)[idx] = velri_f32_to_f16_scalar(v);
    } else {
        ((velri_bf16 *)base)[idx] = velri_f32_to_bf16_scalar(v);
    }
}

/* ============================================================================
 * 1. Normalization (`RMSNorm`, `Fused Add+RMSNorm`, `LayerNorm`)
 * ============================================================================ */

static void avx512_rmsnorm(velri_dtype io_dt, velri_dtype w_dt, size_t r0, size_t r1, size_t dim,
                           float eps, bool unit_offset, const void *x, size_t xs, const void *w,
                           void *y, size_t ys) {
    const float inv_dim = 1.0f / (float)dim;
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const bool use_nt =
        (y != x) && ((r1 - r0) * dim * 2ULL * velri_dtype_bytes(io_dt) >= velri_nt_io_bytes());

    for (size_t r = r0; r < r1; ++r) {
        size_t xo = r * xs;
        size_t yo = r * ys;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
        size_t d = 0;
        for (; d + 64 <= dim; d += 64) {
            __m512 v0 = load16_as_f32_avx512(io_dt, x, xo + d);
            __m512 v1 = load16_as_f32_avx512(io_dt, x, xo + d + 16);
            __m512 v2 = load16_as_f32_avx512(io_dt, x, xo + d + 32);
            __m512 v3 = load16_as_f32_avx512(io_dt, x, xo + d + 48);
            acc0 = _mm512_fmadd_ps(v0, v0, acc0);
            acc1 = _mm512_fmadd_ps(v1, v1, acc1);
            acc2 = _mm512_fmadd_ps(v2, v2, acc2);
            acc3 = _mm512_fmadd_ps(v3, v3, acc3);
        }
        acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        for (; d + 16 <= dim; d += 16) {
            __m512 v0 = load16_as_f32_avx512(io_dt, x, xo + d);
            acc0 = _mm512_fmadd_ps(v0, v0, acc0);
        }
        float ss = _mm512_reduce_add_ps(acc0);
        for (; d < dim; ++d) {
            float v = load1_f32(io_dt, x, xo + d);
            ss += v * v;
        }

        float rrms = 1.0f / sqrtf(ss * inv_dim + eps);
        __m512 vrrms = _mm512_set1_ps(rrms);

        d = 0;
        if (use_nt) {
            const uint8_t *xn = (r + 1 < r1)
                                    ? ((const uint8_t *)x + (r + 1) * xs * velri_dtype_bytes(io_dt))
                                    : nullptr;
            const size_t eb = velri_dtype_bytes(io_dt);
            for (; d + 16 <= dim; d += 16) {
                if (xn && ((d * eb) & 63ULL) == 0) {
                    _mm_prefetch((const char *)(xn + d * eb), _MM_HINT_T0);
                }
                __m512 v = _mm512_mul_ps(load16_as_f32_avx512(io_dt, x, xo + d), vrrms);
                if (w) {
                    __m512 wv = load16_as_f32_avx512(w_dt, w, d);
                    if (unit_offset) {
                        wv = _mm512_add_ps(v_one, wv);
                    }
                    v = _mm512_mul_ps(v, wv);
                }
                stream16_from_f32_avx512(io_dt, y, yo + d, v);
            }
        } else {
            for (; d + 16 <= dim; d += 16) {
                __m512 v = _mm512_mul_ps(load16_as_f32_avx512(io_dt, x, xo + d), vrrms);
                if (w) {
                    __m512 wv = load16_as_f32_avx512(w_dt, w, d);
                    if (unit_offset) {
                        wv = _mm512_add_ps(v_one, wv);
                    }
                    v = _mm512_mul_ps(v, wv);
                }
                store16_from_f32_avx512(io_dt, y, yo + d, v);
            }
        }
        for (; d < dim; ++d) {
            float v = load1_f32(io_dt, x, xo + d) * rrms;
            if (w) {
                float wv = load1_f32(w_dt, w, d);
                v *= unit_offset ? (1.0f + wv) : wv;
            }
            store1_f32(io_dt, y, yo + d, v);
        }
    }
    if (use_nt) {
        _mm_sfence();
    }
}

static void avx512_fused_add_rmsnorm_f32(size_t r0, size_t r1, size_t dim, float eps,
                                         bool unit_offset, float *res, size_t rs, const float *x,
                                         size_t xs, const float *w, float *y, size_t ys) {
    const float inv_dim = 1.0f / (float)dim;
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const bool use_nt = (y != res) && ((r1 - r0) * dim * 16ULL >= velri_nt_io_bytes());

    for (size_t r = r0; r < r1; ++r) {
        float *rr = res + r * rs;
        const float *xr = x + r * xs;
        float *yr = y + r * ys;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
        size_t d = 0;
        for (; d + 64 <= dim; d += 64) {
            __m512 s0 = _mm512_add_ps(_mm512_loadu_ps(rr + d), _mm512_loadu_ps(xr + d));
            __m512 s1 = _mm512_add_ps(_mm512_loadu_ps(rr + d + 16), _mm512_loadu_ps(xr + d + 16));
            __m512 s2 = _mm512_add_ps(_mm512_loadu_ps(rr + d + 32), _mm512_loadu_ps(xr + d + 32));
            __m512 s3 = _mm512_add_ps(_mm512_loadu_ps(rr + d + 48), _mm512_loadu_ps(xr + d + 48));
            _mm512_storeu_ps(rr + d, s0);
            _mm512_storeu_ps(rr + d + 16, s1);
            _mm512_storeu_ps(rr + d + 32, s2);
            _mm512_storeu_ps(rr + d + 48, s3);
            acc0 = _mm512_fmadd_ps(s0, s0, acc0);
            acc1 = _mm512_fmadd_ps(s1, s1, acc1);
            acc2 = _mm512_fmadd_ps(s2, s2, acc2);
            acc3 = _mm512_fmadd_ps(s3, s3, acc3);
        }
        acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        for (; d + 16 <= dim; d += 16) {
            __m512 s0 = _mm512_add_ps(_mm512_loadu_ps(rr + d), _mm512_loadu_ps(xr + d));
            _mm512_storeu_ps(rr + d, s0);
            acc0 = _mm512_fmadd_ps(s0, s0, acc0);
        }
        float ss = _mm512_reduce_add_ps(acc0);
        for (; d < dim; ++d) {
            float s = rr[d] + xr[d];
            rr[d] = s;
            ss += s * s;
        }
        float rrms = 1.0f / sqrtf(ss * inv_dim + eps);
        __m512 vrrms = _mm512_set1_ps(rrms);
        d = 0;
        bool row_nt = use_nt && (((uintptr_t)yr & 63ULL) == 0);
        for (; d + 16 <= dim; d += 16) {
            __m512 v = _mm512_mul_ps(_mm512_loadu_ps(rr + d), vrrms);
            if (w) {
                __m512 wv = _mm512_loadu_ps(w + d);
                if (unit_offset) {
                    wv = _mm512_add_ps(v_one, wv);
                }
                v = _mm512_mul_ps(v, wv);
            }
            if (row_nt) {
                _mm512_stream_ps(yr + d, v);
            } else {
                _mm512_storeu_ps(yr + d, v);
            }
        }
        for (; d < dim; ++d) {
            float v = rr[d] * rrms;
            if (w) {
                v *= unit_offset ? (1.0f + w[d]) : w[d];
            }
            yr[d] = v;
        }
    }
    if (use_nt) {
        _mm_sfence();
    }
}

static void avx512_layernorm_f32(size_t r0, size_t r1, size_t dim, float eps, const float *x,
                                 size_t xs, const float *w, const float *b, float *restrict y,
                                 size_t ys) {
    const float inv_dim = 1.0f / (float)dim;
    const bool use_nt = ((r1 - r0) * dim * 8ULL >= velri_nt_io_bytes());
    for (size_t r = r0; r < r1; ++r) {
        const float *xr = x + r * xs;
        float *yr = y + r * ys;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        size_t d = 0;
        for (; d + 64 <= dim; d += 64) {
            s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xr + d));
            s1 = _mm512_add_ps(s1, _mm512_loadu_ps(xr + d + 16));
            s2 = _mm512_add_ps(s2, _mm512_loadu_ps(xr + d + 32));
            s3 = _mm512_add_ps(s3, _mm512_loadu_ps(xr + d + 48));
        }
        s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
        for (; d + 16 <= dim; d += 16) {
            s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xr + d));
        }
        float sum = _mm512_reduce_add_ps(s0);
        for (; d < dim; ++d) {
            sum += xr[d];
        }
        float mean = sum * inv_dim;
        __m512 vmean = _mm512_set1_ps(mean);

        __m512 v0 = _mm512_setzero_ps(), v1 = _mm512_setzero_ps();
        __m512 v2 = _mm512_setzero_ps(), v3 = _mm512_setzero_ps();
        d = 0;
        for (; d + 64 <= dim; d += 64) {
            __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(xr + d), vmean);
            __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(xr + d + 16), vmean);
            __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(xr + d + 32), vmean);
            __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(xr + d + 48), vmean);
            v0 = _mm512_fmadd_ps(d0, d0, v0);
            v1 = _mm512_fmadd_ps(d1, d1, v1);
            v2 = _mm512_fmadd_ps(d2, d2, v2);
            v3 = _mm512_fmadd_ps(d3, d3, v3);
        }
        v0 = _mm512_add_ps(_mm512_add_ps(v0, v1), _mm512_add_ps(v2, v3));
        for (; d + 16 <= dim; d += 16) {
            __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(xr + d), vmean);
            v0 = _mm512_fmadd_ps(d0, d0, v0);
        }
        float var_s = _mm512_reduce_add_ps(v0);
        for (; d < dim; ++d) {
            float df = xr[d] - mean;
            var_s += df * df;
        }
        float inv_std = 1.0f / sqrtf(var_s * inv_dim + eps);
        __m512 vinv = _mm512_set1_ps(inv_std);

        d = 0;
        bool row_nt = use_nt && (((uintptr_t)yr & 63ULL) == 0);
        const float *xn = (row_nt && r + 1 < r1) ? (x + (r + 1) * xs) : nullptr;
        for (; d + 16 <= dim; d += 16) {
            if (xn) {
                _mm_prefetch((const char *)(xn + d), _MM_HINT_T0);
            }
            __m512 val = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(xr + d), vmean), vinv);
            if (w) {
                val = _mm512_mul_ps(val, _mm512_loadu_ps(w + d));
            }
            if (b) {
                val = _mm512_add_ps(val, _mm512_loadu_ps(b + d));
            }
            if (row_nt) {
                _mm512_stream_ps(yr + d, val);
            } else {
                _mm512_storeu_ps(yr + d, val);
            }
        }
        for (; d < dim; ++d) {
            float val = (xr[d] - mean) * inv_std;
            if (w) {
                val *= w[d];
            }
            if (b) {
                val += b[d];
            }
            yr[d] = val;
        }
    }
    if (use_nt) {
        _mm_sfence();
    }
}

/* ============================================================================
 * 2. Rotary Position Embeddings (`RoPE` in 512-bit ZMM)
 * ============================================================================ */

static void avx512_rope(velri_dtype dt, size_t t0, size_t t1, size_t n_heads, size_t head_dim,
                        size_t rotary_dim, size_t pos_offset, const void *x, size_t tsx, size_t hsx,
                        const float *cos_t, const float *sin_t, void *y, size_t tsy, size_t hsy,
                        velri_rope_mode mode) {
    const size_t half = rotary_dim / 2;
    const bool copy_tail = (y != x || tsx != tsy || hsx != hsy);
    const bool use_nt =
        (y != x) &&
        ((t1 - t0) * n_heads * head_dim * 2ULL * velri_dtype_bytes(dt) >= velri_nt_io_bytes());
    /* Index vector to duplicate 8 floats `[c0..c7]` -> `[c0,c0, c1,c1, ..., c7,c7]` in ZMM */
    const __m512i dup_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);

    for (size_t t = t0; t < t1; ++t) {
        const float *cr = rotary_dim ? (cos_t + (pos_offset + t) * half) : nullptr;
        const float *sr = rotary_dim ? (sin_t + (pos_offset + t) * half) : nullptr;
        for (size_t h = 0; h < n_heads; ++h) {
            size_t xo = t * tsx + h * hsx;
            size_t yo = t * tsy + h * hsy;
            if (mode == VELRI_ROPE_HALF) {
                size_t d = 0;
                for (; d + 16 <= half; d += 16) {
                    __m512 x0 = load16_as_f32_avx512(dt, x, xo + d);
                    __m512 x1 = load16_as_f32_avx512(dt, x, xo + half + d);
                    __m512 vc = _mm512_loadu_ps(cr + d);
                    __m512 vs = _mm512_loadu_ps(sr + d);
                    __m512 y0 = _mm512_fmsub_ps(x0, vc, _mm512_mul_ps(x1, vs));
                    __m512 y1 = _mm512_fmadd_ps(x1, vc, _mm512_mul_ps(x0, vs));
                    if (use_nt) {
                        stream16_from_f32_avx512(dt, y, yo + d, y0);
                        stream16_from_f32_avx512(dt, y, yo + half + d, y1);
                    } else {
                        store16_from_f32_avx512(dt, y, yo + d, y0);
                        store16_from_f32_avx512(dt, y, yo + half + d, y1);
                    }
                }
                for (; d < half; ++d) {
                    float x0 = load1_f32(dt, x, xo + d);
                    float x1 = load1_f32(dt, x, xo + half + d);
                    store1_f32(dt, y, yo + d, x0 * cr[d] - x1 * sr[d]);
                    store1_f32(dt, y, yo + half + d, x1 * cr[d] + x0 * sr[d]);
                }
            } else {
                size_t d = 0;
                for (; d + 8 <= half; d += 8) {
                    __m512 xv = load16_as_f32_avx512(dt, x, xo + 2 * d);
                    __m512 x_swap = _mm512_permute_ps(xv, 0xB1);
                    __m512 vc = _mm512_permutexvar_ps(
                        dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(cr + d)));
                    __m512 vs = _mm512_permutexvar_ps(
                        dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(sr + d)));
                    __m512 yv = _mm512_fmaddsub_ps(xv, vc, _mm512_mul_ps(x_swap, vs));
                    if (use_nt) {
                        stream16_from_f32_avx512(dt, y, yo + 2 * d, yv);
                    } else {
                        store16_from_f32_avx512(dt, y, yo + 2 * d, yv);
                    }
                }
                for (; d < half; ++d) {
                    float x0 = load1_f32(dt, x, xo + 2 * d);
                    float x1 = load1_f32(dt, x, xo + 2 * d + 1);
                    store1_f32(dt, y, yo + 2 * d, x0 * cr[d] - x1 * sr[d]);
                    store1_f32(dt, y, yo + 2 * d + 1, x1 * cr[d] + x0 * sr[d]);
                }
            }
            if (copy_tail) {
                size_t d = rotary_dim;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 tv = load16_as_f32_avx512(dt, x, xo + d);
                    if (use_nt) {
                        stream16_from_f32_avx512(dt, y, yo + d, tv);
                    } else {
                        store16_from_f32_avx512(dt, y, yo + d, tv);
                    }
                }
                for (; d < head_dim; ++d) {
                    store1_f32(dt, y, yo + d, load1_f32(dt, x, xo + d));
                }
            }
        }
    }
    if (use_nt) {
        _mm_sfence();
    }
}

/* ============================================================================
 * 3. Activations, GatedMLP & Online Softmax (`AVX-512`)
 * ============================================================================ */

/* Approximate reciprocal refined by one Newton-Raphson step. */
static inline __m512 rcp_nr_512(__m512 d) {
    __m512 r = _mm512_rcp14_ps(d);
    return _mm512_fmadd_ps(r, _mm512_fnmadd_ps(d, r, _mm512_set1_ps(1.0f)), r);
}

static inline __m512 eval_act512(velri_act_kind kind, __m512 x) {
    const __m512 v_one = _mm512_set1_ps(1.0f);
    if (kind == VELRI_ACT_SILU) {
        __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
        __mmask16 keep = _mm512_cmp_ps_mask(x, _mm512_set1_ps(-88.0f), _CMP_GE_OQ);
        return _mm512_maskz_mul_ps(keep, x, rcp_nr_512(_mm512_add_ps(v_one, exp512_ps(neg_x))));
    }
    if (kind == VELRI_ACT_GELU_TANH) {
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 inner = _mm512_fmadd_ps(_mm512_set1_ps(0.044715f), x2, v_one);
        __m512 arg = _mm512_mul_ps(_mm512_set1_ps(-1.59576912f), _mm512_mul_ps(x, inner));
        __mmask16 keep = _mm512_cmp_ps_mask(arg, _mm512_set1_ps(88.0f), _CMP_LE_OQ);
        return _mm512_maskz_mul_ps(keep, x, rcp_nr_512(_mm512_add_ps(v_one, exp512_ps(arg))));
    }
    __m512 r = _mm512_max_ps(x, _mm512_setzero_ps());
    return _mm512_mul_ps(r, r);
}

/* 4x-interleaved ZMM `2^x` minimax degree-5 (polynomial approximation) */
static inline void exp2_512_x4(__m512 *io0, __m512 *io1, __m512 *io2, __m512 *io3) {
    const __m512 min_x = _mm512_set1_ps(-126.0f);
    const __m512 max_x = _mm512_set1_ps(126.0f);
    __m512 x0 = _mm512_min_ps(_mm512_max_ps(*io0, min_x), max_x);
    __m512 x1 = _mm512_min_ps(_mm512_max_ps(*io1, min_x), max_x);
    __m512 x2 = _mm512_min_ps(_mm512_max_ps(*io2, min_x), max_x);
    __m512 x3 = _mm512_min_ps(_mm512_max_ps(*io3, min_x), max_x);

    __m512 n0 = _mm512_roundscale_ps(x0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 n1 = _mm512_roundscale_ps(x1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 n2 = _mm512_roundscale_ps(x2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 n3 = _mm512_roundscale_ps(x3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m512 f0 = _mm512_sub_ps(x0, n0);
    __m512 f1 = _mm512_sub_ps(x1, n1);
    __m512 f2 = _mm512_sub_ps(x2, n2);
    __m512 f3 = _mm512_sub_ps(x3, n3);

    const __m512 c5 = _mm512_set1_ps(1.33335581e-3f);
    const __m512 c4 = _mm512_set1_ps(9.61812911e-3f);
    const __m512 c3 = _mm512_set1_ps(5.55041087e-2f);
    const __m512 c2 = _mm512_set1_ps(2.40226507e-1f);
    const __m512 c1 = _mm512_set1_ps(6.93147182e-1f);
    const __m512 c0 = _mm512_set1_ps(1.0f);

    __m512 p0 = _mm512_fmadd_ps(c5, f0, c4);
    __m512 p1 = _mm512_fmadd_ps(c5, f1, c4);
    __m512 p2 = _mm512_fmadd_ps(c5, f2, c4);
    __m512 p3 = _mm512_fmadd_ps(c5, f3, c4);

    p0 = _mm512_fmadd_ps(p0, f0, c3);
    p1 = _mm512_fmadd_ps(p1, f1, c3);
    p2 = _mm512_fmadd_ps(p2, f2, c3);
    p3 = _mm512_fmadd_ps(p3, f3, c3);

    p0 = _mm512_fmadd_ps(p0, f0, c2);
    p1 = _mm512_fmadd_ps(p1, f1, c2);
    p2 = _mm512_fmadd_ps(p2, f2, c2);
    p3 = _mm512_fmadd_ps(p3, f3, c2);

    p0 = _mm512_fmadd_ps(p0, f0, c1);
    p1 = _mm512_fmadd_ps(p1, f1, c1);
    p2 = _mm512_fmadd_ps(p2, f2, c1);
    p3 = _mm512_fmadd_ps(p3, f3, c1);

    p0 = _mm512_fmadd_ps(p0, f0, c0);
    p1 = _mm512_fmadd_ps(p1, f1, c0);
    p2 = _mm512_fmadd_ps(p2, f2, c0);
    p3 = _mm512_fmadd_ps(p3, f3, c0);

    *io0 = _mm512_scalef_ps(p0, n0);
    *io1 = _mm512_scalef_ps(p1, n1);
    *io2 = _mm512_scalef_ps(p2, n2);
    *io3 = _mm512_scalef_ps(p3, n3);
}

/* 4x-interleaved ZMM SiLU & GELU_tanh via `exp2_512_x4` (zero ZMM stack spills!) */
static inline void silu512_x4(__m512 *x0, __m512 *x1, __m512 *x2, __m512 *x3) {
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_nlog2e = _mm512_set1_ps(-1.44269504088896341f);
    const __m512 v_min = _mm512_set1_ps(-88.0f);
    __mmask16 k0 = _mm512_cmp_ps_mask(*x0, v_min, _CMP_GE_OQ);
    __mmask16 k1 = _mm512_cmp_ps_mask(*x1, v_min, _CMP_GE_OQ);
    __mmask16 k2 = _mm512_cmp_ps_mask(*x2, v_min, _CMP_GE_OQ);
    __mmask16 k3 = _mm512_cmp_ps_mask(*x3, v_min, _CMP_GE_OQ);
    __m512 e0 = _mm512_mul_ps(*x0, v_nlog2e);
    __m512 e1 = _mm512_mul_ps(*x1, v_nlog2e);
    __m512 e2 = _mm512_mul_ps(*x2, v_nlog2e);
    __m512 e3 = _mm512_mul_ps(*x3, v_nlog2e);
    exp2_512_x4(&e0, &e1, &e2, &e3);
    __m512 d0 = _mm512_add_ps(v_one, e0), d1 = _mm512_add_ps(v_one, e1);
    __m512 d2 = _mm512_add_ps(v_one, e2), d3 = _mm512_add_ps(v_one, e3);
    __m512 r0 = _mm512_rcp14_ps(d0), r1 = _mm512_rcp14_ps(d1);
    __m512 r2 = _mm512_rcp14_ps(d2), r3 = _mm512_rcp14_ps(d3);
    r0 = _mm512_fmadd_ps(r0, _mm512_fnmadd_ps(d0, r0, v_one), r0);
    r1 = _mm512_fmadd_ps(r1, _mm512_fnmadd_ps(d1, r1, v_one), r1);
    r2 = _mm512_fmadd_ps(r2, _mm512_fnmadd_ps(d2, r2, v_one), r2);
    r3 = _mm512_fmadd_ps(r3, _mm512_fnmadd_ps(d3, r3, v_one), r3);
    *x0 = _mm512_maskz_mul_ps(k0, *x0, r0);
    *x1 = _mm512_maskz_mul_ps(k1, *x1, r1);
    *x2 = _mm512_maskz_mul_ps(k2, *x2, r2);
    *x3 = _mm512_maskz_mul_ps(k3, *x3, r3);
}

static inline void gelu512_x4(__m512 *x0, __m512 *x1, __m512 *x2, __m512 *x3) {
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_c = _mm512_set1_ps(0.044715f);
    /* `-2 * sqrt(2/pi) * log2(e) = -1.5957691216f * 1.44269504089f` */
    const __m512 v_s2 = _mm512_set1_ps(-2.30220781f);
    const __m512 v_max = _mm512_set1_ps(127.0f);
    __m512 sx0 = _mm512_mul_ps(v_s2, *x0), sq0 = _mm512_mul_ps(*x0, *x0);
    __m512 sx1 = _mm512_mul_ps(v_s2, *x1), sq1 = _mm512_mul_ps(*x1, *x1);
    __m512 sx2 = _mm512_mul_ps(v_s2, *x2), sq2 = _mm512_mul_ps(*x2, *x2);
    __m512 sx3 = _mm512_mul_ps(v_s2, *x3), sq3 = _mm512_mul_ps(*x3, *x3);
    __m512 a0 = _mm512_mul_ps(sx0, _mm512_fmadd_ps(v_c, sq0, v_one));
    __m512 a1 = _mm512_mul_ps(sx1, _mm512_fmadd_ps(v_c, sq1, v_one));
    __m512 a2 = _mm512_mul_ps(sx2, _mm512_fmadd_ps(v_c, sq2, v_one));
    __m512 a3 = _mm512_mul_ps(sx3, _mm512_fmadd_ps(v_c, sq3, v_one));
    __mmask16 k0 = _mm512_cmp_ps_mask(a0, v_max, _CMP_LE_OQ);
    __mmask16 k1 = _mm512_cmp_ps_mask(a1, v_max, _CMP_LE_OQ);
    __mmask16 k2 = _mm512_cmp_ps_mask(a2, v_max, _CMP_LE_OQ);
    __mmask16 k3 = _mm512_cmp_ps_mask(a3, v_max, _CMP_LE_OQ);
    exp2_512_x4(&a0, &a1, &a2, &a3);
    __m512 d0 = _mm512_add_ps(v_one, a0), d1 = _mm512_add_ps(v_one, a1);
    __m512 d2 = _mm512_add_ps(v_one, a2), d3 = _mm512_add_ps(v_one, a3);
    __m512 r0 = _mm512_rcp14_ps(d0), r1 = _mm512_rcp14_ps(d1);
    __m512 r2 = _mm512_rcp14_ps(d2), r3 = _mm512_rcp14_ps(d3);
    r0 = _mm512_fmadd_ps(r0, _mm512_fnmadd_ps(d0, r0, v_one), r0);
    r1 = _mm512_fmadd_ps(r1, _mm512_fnmadd_ps(d1, r1, v_one), r1);
    r2 = _mm512_fmadd_ps(r2, _mm512_fnmadd_ps(d2, r2, v_one), r2);
    r3 = _mm512_fmadd_ps(r3, _mm512_fnmadd_ps(d3, r3, v_one), r3);
    *x0 = _mm512_maskz_mul_ps(k0, *x0, r0);
    *x1 = _mm512_maskz_mul_ps(k1, *x1, r1);
    *x2 = _mm512_maskz_mul_ps(k2, *x2, r2);
    *x3 = _mm512_maskz_mul_ps(k3, *x3, r3);
}

static inline float eval_act1(velri_act_kind kind, float x) {
    if (kind == VELRI_ACT_SILU) {
        if (-x > 88.0f) {
            return 0.0f;
        }
        if (-x < -88.0f) {
            return x;
        }
        return x / (1.0f + expf_fast_avx512(-x));
    }
    if (kind == VELRI_ACT_GELU_TANH) {
        float arg = -1.59576912f * x * (1.0f + 0.044715f * x * x);
        if (arg > 88.0f) {
            return 0.0f;
        }
        if (arg < -88.0f) {
            return x;
        }
        return x / (1.0f + expf_fast_avx512(arg));
    }
    float r = x > 0.0f ? x : 0.0f;
    return r * r;
}

static void avx512_activation_f32(velri_act_kind kind, size_t i0, size_t i1, const float *x,
                                  float *y) {
    size_t i = i0;
    if ((i1 - i0) * 8ULL >= velri_nt_io_bytes() && y != x && (((uintptr_t)(y + i)) & 3ULL) == 0) {
        while (i < i1 && (((uintptr_t)(y + i)) & 63ULL) != 0) {
            y[i] = eval_act1(kind, x[i]);
            ++i;
        }
        if (kind == VELRI_ACT_SILU) {
            for (; i + 64 <= i1; i += 64) {
                __m512 y0 = _mm512_loadu_ps(x + i), y1 = _mm512_loadu_ps(x + i + 16);
                __m512 y2 = _mm512_loadu_ps(x + i + 32), y3 = _mm512_loadu_ps(x + i + 48);
                silu512_x4(&y0, &y1, &y2, &y3);
                _mm512_stream_ps(y + i, y0);
                _mm512_stream_ps(y + i + 16, y1);
                _mm512_stream_ps(y + i + 32, y2);
                _mm512_stream_ps(y + i + 48, y3);
            }
        } else if (kind == VELRI_ACT_GELU_TANH) {
            for (; i + 64 <= i1; i += 64) {
                __m512 y0 = _mm512_loadu_ps(x + i), y1 = _mm512_loadu_ps(x + i + 16);
                __m512 y2 = _mm512_loadu_ps(x + i + 32), y3 = _mm512_loadu_ps(x + i + 48);
                gelu512_x4(&y0, &y1, &y2, &y3);
                _mm512_stream_ps(y + i, y0);
                _mm512_stream_ps(y + i + 16, y1);
                _mm512_stream_ps(y + i + 32, y2);
                _mm512_stream_ps(y + i + 48, y3);
            }
        } else {
            const __m512 vz = _mm512_setzero_ps();
            for (; i + 64 <= i1; i += 64) {
                __m512 r0 = _mm512_max_ps(_mm512_loadu_ps(x + i), vz);
                __m512 r1 = _mm512_max_ps(_mm512_loadu_ps(x + i + 16), vz);
                __m512 r2 = _mm512_max_ps(_mm512_loadu_ps(x + i + 32), vz);
                __m512 r3 = _mm512_max_ps(_mm512_loadu_ps(x + i + 48), vz);
                _mm512_stream_ps(y + i, _mm512_mul_ps(r0, r0));
                _mm512_stream_ps(y + i + 16, _mm512_mul_ps(r1, r1));
                _mm512_stream_ps(y + i + 32, _mm512_mul_ps(r2, r2));
                _mm512_stream_ps(y + i + 48, _mm512_mul_ps(r3, r3));
            }
        }
        _mm_sfence();
    }
    if (kind == VELRI_ACT_SILU) {
        for (; i + 64 <= i1; i += 64) {
            __m512 y0 = _mm512_loadu_ps(x + i), y1 = _mm512_loadu_ps(x + i + 16);
            __m512 y2 = _mm512_loadu_ps(x + i + 32), y3 = _mm512_loadu_ps(x + i + 48);
            silu512_x4(&y0, &y1, &y2, &y3);
            _mm512_storeu_ps(y + i, y0);
            _mm512_storeu_ps(y + i + 16, y1);
            _mm512_storeu_ps(y + i + 32, y2);
            _mm512_storeu_ps(y + i + 48, y3);
        }
    } else if (kind == VELRI_ACT_GELU_TANH) {
        for (; i + 64 <= i1; i += 64) {
            __m512 y0 = _mm512_loadu_ps(x + i), y1 = _mm512_loadu_ps(x + i + 16);
            __m512 y2 = _mm512_loadu_ps(x + i + 32), y3 = _mm512_loadu_ps(x + i + 48);
            gelu512_x4(&y0, &y1, &y2, &y3);
            _mm512_storeu_ps(y + i, y0);
            _mm512_storeu_ps(y + i + 16, y1);
            _mm512_storeu_ps(y + i + 32, y2);
            _mm512_storeu_ps(y + i + 48, y3);
        }
    }
    for (; i + 16 <= i1; i += 16) {
        _mm512_storeu_ps(y + i, eval_act512(kind, _mm512_loadu_ps(x + i)));
    }
    for (; i < i1; ++i) {
        y[i] = eval_act1(kind, x[i]);
    }
}

static void avx512_gated_mlp_f32(velri_act_kind kind, size_t i0, size_t i1, float clamp_limit,
                                 const float *gate, const float *up, float *restrict y) {
    const bool do_clamp = (clamp_limit > 0.0f);
    const __m512 vlo = _mm512_set1_ps(-clamp_limit);
    const __m512 vhi = _mm512_set1_ps(clamp_limit);

    size_t i = i0;
    if ((i1 - i0) * 12ULL >= velri_nt_io_bytes() && (((uintptr_t)(y + i)) & 3ULL) == 0) {
        while (i < i1 && (((uintptr_t)(y + i)) & 63ULL) != 0) {
            float g = gate[i], u = up[i];
            if (do_clamp) {
                g = g < -clamp_limit ? -clamp_limit : (g > clamp_limit ? clamp_limit : g);
                u = u < -clamp_limit ? -clamp_limit : (u > clamp_limit ? clamp_limit : u);
            }
            y[i] = eval_act1(kind, g) * u;
            ++i;
        }
        if (kind == VELRI_ACT_SILU) {
            for (; i + 64 <= i1; i += 64) {
                _mm_prefetch((const char *)(gate + i + 256), _MM_HINT_T0);
                _mm_prefetch((const char *)(up + i + 256), _MM_HINT_T0);
                __m512 g0 = _mm512_loadu_ps(gate + i), g1 = _mm512_loadu_ps(gate + i + 16);
                __m512 g2 = _mm512_loadu_ps(gate + i + 32), g3 = _mm512_loadu_ps(gate + i + 48);
                if (do_clamp) {
                    g0 = _mm512_min_ps(_mm512_max_ps(g0, vlo), vhi);
                    g1 = _mm512_min_ps(_mm512_max_ps(g1, vlo), vhi);
                    g2 = _mm512_min_ps(_mm512_max_ps(g2, vlo), vhi);
                    g3 = _mm512_min_ps(_mm512_max_ps(g3, vlo), vhi);
                }
                silu512_x4(&g0, &g1, &g2, &g3);
                __m512 u0 = _mm512_loadu_ps(up + i), u1 = _mm512_loadu_ps(up + i + 16);
                __m512 u2 = _mm512_loadu_ps(up + i + 32), u3 = _mm512_loadu_ps(up + i + 48);
                if (do_clamp) {
                    u0 = _mm512_min_ps(_mm512_max_ps(u0, vlo), vhi);
                    u1 = _mm512_min_ps(_mm512_max_ps(u1, vlo), vhi);
                    u2 = _mm512_min_ps(_mm512_max_ps(u2, vlo), vhi);
                    u3 = _mm512_min_ps(_mm512_max_ps(u3, vlo), vhi);
                }
                _mm512_stream_ps(y + i, _mm512_mul_ps(g0, u0));
                _mm512_stream_ps(y + i + 16, _mm512_mul_ps(g1, u1));
                _mm512_stream_ps(y + i + 32, _mm512_mul_ps(g2, u2));
                _mm512_stream_ps(y + i + 48, _mm512_mul_ps(g3, u3));
            }
        } else if (kind == VELRI_ACT_GELU_TANH) {
            for (; i + 64 <= i1; i += 64) {
                _mm_prefetch((const char *)(gate + i + 256), _MM_HINT_T0);
                _mm_prefetch((const char *)(up + i + 256), _MM_HINT_T0);
                __m512 g0 = _mm512_loadu_ps(gate + i), g1 = _mm512_loadu_ps(gate + i + 16);
                __m512 g2 = _mm512_loadu_ps(gate + i + 32), g3 = _mm512_loadu_ps(gate + i + 48);
                if (do_clamp) {
                    g0 = _mm512_min_ps(_mm512_max_ps(g0, vlo), vhi);
                    g1 = _mm512_min_ps(_mm512_max_ps(g1, vlo), vhi);
                    g2 = _mm512_min_ps(_mm512_max_ps(g2, vlo), vhi);
                    g3 = _mm512_min_ps(_mm512_max_ps(g3, vlo), vhi);
                }
                gelu512_x4(&g0, &g1, &g2, &g3);
                __m512 u0 = _mm512_loadu_ps(up + i), u1 = _mm512_loadu_ps(up + i + 16);
                __m512 u2 = _mm512_loadu_ps(up + i + 32), u3 = _mm512_loadu_ps(up + i + 48);
                if (do_clamp) {
                    u0 = _mm512_min_ps(_mm512_max_ps(u0, vlo), vhi);
                    u1 = _mm512_min_ps(_mm512_max_ps(u1, vlo), vhi);
                    u2 = _mm512_min_ps(_mm512_max_ps(u2, vlo), vhi);
                    u3 = _mm512_min_ps(_mm512_max_ps(u3, vlo), vhi);
                }
                _mm512_stream_ps(y + i, _mm512_mul_ps(g0, u0));
                _mm512_stream_ps(y + i + 16, _mm512_mul_ps(g1, u1));
                _mm512_stream_ps(y + i + 32, _mm512_mul_ps(g2, u2));
                _mm512_stream_ps(y + i + 48, _mm512_mul_ps(g3, u3));
            }
        } else {
            const __m512 vz = _mm512_setzero_ps();
            for (; i + 64 <= i1; i += 64) {
                __m512 g0 = _mm512_loadu_ps(gate + i), u0 = _mm512_loadu_ps(up + i);
                __m512 g1 = _mm512_loadu_ps(gate + i + 16), u1 = _mm512_loadu_ps(up + i + 16);
                __m512 g2 = _mm512_loadu_ps(gate + i + 32), u2 = _mm512_loadu_ps(up + i + 32);
                __m512 g3 = _mm512_loadu_ps(gate + i + 48), u3 = _mm512_loadu_ps(up + i + 48);
                if (do_clamp) {
                    g0 = _mm512_min_ps(_mm512_max_ps(g0, vlo), vhi);
                    u0 = _mm512_min_ps(_mm512_max_ps(u0, vlo), vhi);
                    g1 = _mm512_min_ps(_mm512_max_ps(g1, vlo), vhi);
                    u1 = _mm512_min_ps(_mm512_max_ps(u1, vlo), vhi);
                    g2 = _mm512_min_ps(_mm512_max_ps(g2, vlo), vhi);
                    u2 = _mm512_min_ps(_mm512_max_ps(u2, vlo), vhi);
                    g3 = _mm512_min_ps(_mm512_max_ps(g3, vlo), vhi);
                    u3 = _mm512_min_ps(_mm512_max_ps(u3, vlo), vhi);
                }
                g0 = _mm512_max_ps(g0, vz);
                g1 = _mm512_max_ps(g1, vz);
                g2 = _mm512_max_ps(g2, vz);
                g3 = _mm512_max_ps(g3, vz);
                _mm512_stream_ps(y + i, _mm512_mul_ps(_mm512_mul_ps(g0, g0), u0));
                _mm512_stream_ps(y + i + 16, _mm512_mul_ps(_mm512_mul_ps(g1, g1), u1));
                _mm512_stream_ps(y + i + 32, _mm512_mul_ps(_mm512_mul_ps(g2, g2), u2));
                _mm512_stream_ps(y + i + 48, _mm512_mul_ps(_mm512_mul_ps(g3, g3), u3));
            }
        }
        _mm_sfence();
    }
    if (kind == VELRI_ACT_SILU) {
        for (; i + 64 <= i1; i += 64) {
            __m512 g0 = _mm512_loadu_ps(gate + i), u0 = _mm512_loadu_ps(up + i);
            __m512 g1 = _mm512_loadu_ps(gate + i + 16), u1 = _mm512_loadu_ps(up + i + 16);
            __m512 g2 = _mm512_loadu_ps(gate + i + 32), u2 = _mm512_loadu_ps(up + i + 32);
            __m512 g3 = _mm512_loadu_ps(gate + i + 48), u3 = _mm512_loadu_ps(up + i + 48);
            if (do_clamp) {
                g0 = _mm512_min_ps(_mm512_max_ps(g0, vlo), vhi);
                u0 = _mm512_min_ps(_mm512_max_ps(u0, vlo), vhi);
                g1 = _mm512_min_ps(_mm512_max_ps(g1, vlo), vhi);
                u1 = _mm512_min_ps(_mm512_max_ps(u1, vlo), vhi);
                g2 = _mm512_min_ps(_mm512_max_ps(g2, vlo), vhi);
                u2 = _mm512_min_ps(_mm512_max_ps(u2, vlo), vhi);
                g3 = _mm512_min_ps(_mm512_max_ps(g3, vlo), vhi);
                u3 = _mm512_min_ps(_mm512_max_ps(u3, vlo), vhi);
            }
            silu512_x4(&g0, &g1, &g2, &g3);
            _mm512_storeu_ps(y + i, _mm512_mul_ps(g0, u0));
            _mm512_storeu_ps(y + i + 16, _mm512_mul_ps(g1, u1));
            _mm512_storeu_ps(y + i + 32, _mm512_mul_ps(g2, u2));
            _mm512_storeu_ps(y + i + 48, _mm512_mul_ps(g3, u3));
        }
    } else if (kind == VELRI_ACT_GELU_TANH) {
        for (; i + 64 <= i1; i += 64) {
            __m512 g0 = _mm512_loadu_ps(gate + i), u0 = _mm512_loadu_ps(up + i);
            __m512 g1 = _mm512_loadu_ps(gate + i + 16), u1 = _mm512_loadu_ps(up + i + 16);
            __m512 g2 = _mm512_loadu_ps(gate + i + 32), u2 = _mm512_loadu_ps(up + i + 32);
            __m512 g3 = _mm512_loadu_ps(gate + i + 48), u3 = _mm512_loadu_ps(up + i + 48);
            if (do_clamp) {
                g0 = _mm512_min_ps(_mm512_max_ps(g0, vlo), vhi);
                u0 = _mm512_min_ps(_mm512_max_ps(u0, vlo), vhi);
                g1 = _mm512_min_ps(_mm512_max_ps(g1, vlo), vhi);
                u1 = _mm512_min_ps(_mm512_max_ps(u1, vlo), vhi);
                g2 = _mm512_min_ps(_mm512_max_ps(g2, vlo), vhi);
                u2 = _mm512_min_ps(_mm512_max_ps(u2, vlo), vhi);
                g3 = _mm512_min_ps(_mm512_max_ps(g3, vlo), vhi);
                u3 = _mm512_min_ps(_mm512_max_ps(u3, vlo), vhi);
            }
            gelu512_x4(&g0, &g1, &g2, &g3);
            _mm512_storeu_ps(y + i, _mm512_mul_ps(g0, u0));
            _mm512_storeu_ps(y + i + 16, _mm512_mul_ps(g1, u1));
            _mm512_storeu_ps(y + i + 32, _mm512_mul_ps(g2, u2));
            _mm512_storeu_ps(y + i + 48, _mm512_mul_ps(g3, u3));
        }
    }
    for (; i + 16 <= i1; i += 16) {
        __m512 g = _mm512_loadu_ps(gate + i);
        __m512 u = _mm512_loadu_ps(up + i);
        if (do_clamp) {
            g = _mm512_min_ps(_mm512_max_ps(g, vlo), vhi);
            u = _mm512_min_ps(_mm512_max_ps(u, vlo), vhi);
        }
        _mm512_storeu_ps(y + i, _mm512_mul_ps(eval_act512(kind, g), u));
    }
    for (; i < i1; ++i) {
        float g = gate[i], u = up[i];
        if (do_clamp) {
            g = g < -clamp_limit ? -clamp_limit : (g > clamp_limit ? clamp_limit : g);
            u = u < -clamp_limit ? -clamp_limit : (u > clamp_limit ? clamp_limit : u);
        }
        y[i] = eval_act1(kind, g) * u;
    }
}

static void avx512_softmax_f32(size_t r0, size_t r1, size_t cols, const float *x, size_t xs,
                               float *y, size_t ys) {
    const bool use_nt = (y != x) && ((r1 - r0) * cols * 8ULL >= velri_nt_io_bytes());

    if (cols <= 16384ULL) {
        /* Single-exp L1/L2 buffer fast path: evaluates `exp512_x4` ONCE per element instead of
         * twice! */
        alignas(64) float exp_buf[16384];
        for (size_t r = r0; r < r1; ++r) {
            const float *xr = x + r * xs;
            float *yr = y + r * ys;

            __m512 vm0 = _mm512_set1_ps(-INFINITY), vm1 = _mm512_set1_ps(-INFINITY);
            __m512 vm2 = _mm512_set1_ps(-INFINITY), vm3 = _mm512_set1_ps(-INFINITY);
            size_t i = 0;
            for (; i + 64 <= cols; i += 64) {
                vm0 = _mm512_max_ps(vm0, _mm512_loadu_ps(xr + i));
                vm1 = _mm512_max_ps(vm1, _mm512_loadu_ps(xr + i + 16));
                vm2 = _mm512_max_ps(vm2, _mm512_loadu_ps(xr + i + 32));
                vm3 = _mm512_max_ps(vm3, _mm512_loadu_ps(xr + i + 48));
            }
            vm0 = _mm512_max_ps(_mm512_max_ps(vm0, vm1), _mm512_max_ps(vm2, vm3));
            for (; i + 16 <= cols; i += 16) {
                vm0 = _mm512_max_ps(vm0, _mm512_loadu_ps(xr + i));
            }
            float m = _mm512_reduce_max_ps(vm0);
            for (; i < cols; ++i) {
                if (xr[i] > m) {
                    m = xr[i];
                }
            }
            if (m == -INFINITY) {
                memset(yr, 0, cols * sizeof(float));
                continue;
            }

            __m512 vmax = _mm512_set1_ps(m);
            __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
            __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
            i = 0;
            for (; i + 64 <= cols; i += 64) {
                __m512 e0 = _mm512_sub_ps(_mm512_loadu_ps(xr + i), vmax);
                __m512 e1 = _mm512_sub_ps(_mm512_loadu_ps(xr + i + 16), vmax);
                __m512 e2 = _mm512_sub_ps(_mm512_loadu_ps(xr + i + 32), vmax);
                __m512 e3 = _mm512_sub_ps(_mm512_loadu_ps(xr + i + 48), vmax);
                exp512_x4(&e0, &e1, &e2, &e3);
                _mm512_store_ps(exp_buf + i, e0);
                _mm512_store_ps(exp_buf + i + 16, e1);
                _mm512_store_ps(exp_buf + i + 32, e2);
                _mm512_store_ps(exp_buf + i + 48, e3);
                s0 = _mm512_add_ps(s0, e0);
                s1 = _mm512_add_ps(s1, e1);
                s2 = _mm512_add_ps(s2, e2);
                s3 = _mm512_add_ps(s3, e3);
            }
            s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
            for (; i + 16 <= cols; i += 16) {
                __m512 e0 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(xr + i), vmax));
                _mm512_store_ps(exp_buf + i, e0);
                s0 = _mm512_add_ps(s0, e0);
            }
            float d_sum = _mm512_reduce_add_ps(s0);
            for (; i < cols; ++i) {
                float e = (xr[i] == -INFINITY) ? 0.0f : expf_fast_avx512(xr[i] - m);
                exp_buf[i] = e;
                d_sum += e;
            }
            if (d_sum <= 0.0f) {
                memset(yr, 0, cols * sizeof(float));
                continue;
            }

            float inv_d = 1.0f / d_sum;
            __m512 vinv = _mm512_set1_ps(inv_d);
            bool row_nt = use_nt && (((uintptr_t)yr & 63ULL) == 0);
            const float *xn = (row_nt && r + 1 < r1) ? (x + (r + 1) * xs) : nullptr;
            i = 0;
            if (row_nt) {
                for (; i + 64 <= cols; i += 64) {
                    if (xn) {
                        _mm_prefetch((const char *)(xn + i), _MM_HINT_T0);
                        _mm_prefetch((const char *)(xn + i + 16), _MM_HINT_T0);
                        _mm_prefetch((const char *)(xn + i + 32), _MM_HINT_T0);
                        _mm_prefetch((const char *)(xn + i + 48), _MM_HINT_T0);
                    }
                    _mm512_stream_ps(yr + i, _mm512_mul_ps(_mm512_load_ps(exp_buf + i), vinv));
                    _mm512_stream_ps(yr + i + 16,
                                     _mm512_mul_ps(_mm512_load_ps(exp_buf + i + 16), vinv));
                    _mm512_stream_ps(yr + i + 32,
                                     _mm512_mul_ps(_mm512_load_ps(exp_buf + i + 32), vinv));
                    _mm512_stream_ps(yr + i + 48,
                                     _mm512_mul_ps(_mm512_load_ps(exp_buf + i + 48), vinv));
                }
                for (; i + 16 <= cols; i += 16) {
                    _mm512_stream_ps(yr + i, _mm512_mul_ps(_mm512_load_ps(exp_buf + i), vinv));
                }
            } else {
                for (; i + 16 <= cols; i += 16) {
                    _mm512_storeu_ps(yr + i, _mm512_mul_ps(_mm512_load_ps(exp_buf + i), vinv));
                }
            }
            for (; i < cols; ++i) {
                yr[i] = exp_buf[i] * inv_d;
            }
        }
        if (use_nt) {
            _mm_sfence();
        }
        return;
    }

    for (size_t r = r0; r < r1; ++r) {
        const float *xr = x + r * xs;
        float *yr = y + r * ys;

        float m = -INFINITY;
        float d_sum = 0.0f;
        size_t i = 0;
        for (; i + 64 <= cols; i += 64) {
            __m512 v0 = _mm512_loadu_ps(xr + i);
            __m512 v1 = _mm512_loadu_ps(xr + i + 16);
            __m512 v2 = _mm512_loadu_ps(xr + i + 32);
            __m512 v3 = _mm512_loadu_ps(xr + i + 48);
            float blk_m =
                _mm512_reduce_max_ps(_mm512_max_ps(_mm512_max_ps(v0, v1), _mm512_max_ps(v2, v3)));
            if (blk_m == -INFINITY) {
                continue;
            }
            float m_new = blk_m > m ? blk_m : m;
            float alpha = (m == -INFINITY) ? 0.0f : expf_fast_avx512(m - m_new);
            __m512 vm = _mm512_set1_ps(m_new);
            __m512 s0 = _mm512_sub_ps(v0, vm);
            __m512 s1 = _mm512_sub_ps(v1, vm);
            __m512 s2 = _mm512_sub_ps(v2, vm);
            __m512 s3 = _mm512_sub_ps(v3, vm);
            exp512_x4(&s0, &s1, &s2, &s3);
            d_sum = d_sum * alpha + _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1),
                                                                       _mm512_add_ps(s2, s3)));
            m = m_new;
        }
        for (; i + 16 <= cols; i += 16) {
            __m512 v0 = _mm512_loadu_ps(xr + i);
            float blk_m = _mm512_reduce_max_ps(v0);
            if (blk_m == -INFINITY) {
                continue;
            }
            float m_new = blk_m > m ? blk_m : m;
            float alpha = (m == -INFINITY) ? 0.0f : expf_fast_avx512(m - m_new);
            d_sum = d_sum * alpha +
                    _mm512_reduce_add_ps(exp512_ps(_mm512_sub_ps(v0, _mm512_set1_ps(m_new))));
            m = m_new;
        }
        for (; i < cols; ++i) {
            float v = xr[i];
            if (v > m) {
                d_sum = (m == -INFINITY) ? 1.0f : (d_sum * expf_fast_avx512(m - v) + 1.0f);
                m = v;
            } else if (v > -INFINITY) {
                d_sum += expf_fast_avx512(v - m);
            }
        }

        if (d_sum <= 0.0f) {
            memset(yr, 0, cols * sizeof(float));
            continue;
        }
        float inv_d = 1.0f / d_sum;
        __m512 vm = _mm512_set1_ps(m);
        __m512 vinv = _mm512_set1_ps(inv_d);
        bool row_nt = use_nt && (((uintptr_t)yr & 63ULL) == 0);
        i = 0;
        for (; i + 16 <= cols; i += 16) {
            __m512 p = _mm512_mul_ps(exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(xr + i), vm)), vinv);
            if (row_nt) {
                _mm512_stream_ps(yr + i, p);
            } else {
                _mm512_storeu_ps(yr + i, p);
            }
        }
        for (; i < cols; ++i) {
            yr[i] = expf_fast_avx512(xr[i] - m) * inv_d;
        }
    }
    if (use_nt) {
        _mm_sfence();
    }
}

/* ============================================================================
 * 4. Orthonormal Block FWHT & Pure Vector Math (`AVX-512`)
 * ============================================================================ */

static inline __m512 load16_fwht_avx512(const float *xr, const float *sc_blk, const void *suh_blk,
                                        velri_dtype suh_dt, size_t j, __m512 vnorm) {
    __m512 vx = _mm512_loadu_ps(xr + j);
    if (sc_blk) {
        return _mm512_mul_ps(vx, _mm512_load_ps(sc_blk + j));
    }
    if (suh_blk) {
        __m512 vs = (suh_dt == VELRI_DTYPE_F32)
                        ? _mm512_loadu_ps((const float *)suh_blk + j)
                        : _mm512_cvtph_ps(_mm256_loadu_si256(
                              (const __m256i *)((const velri_fp16 *)suh_blk + j)));
        return _mm512_mul_ps(vx, _mm512_mul_ps(vs, vnorm));
    }
    return _mm512_mul_ps(vx, vnorm);
}

#define ZMM_BUTTERFLY(a, b)                                                                        \
    do {                                                                                           \
        __m512 ta_ = (a), tb_ = (b);                                                               \
        (a) = _mm512_add_ps(ta_, tb_);                                                             \
        (b) = _mm512_sub_ps(ta_, tb_);                                                             \
    } while (0)

static inline __m512 fwht16_intra_avx512(__m512 z, __m512 vs8, __m512 vs4, __m512 vs2, __m512 vs1) {
    z = _mm512_fmadd_ps(z, vs8, _mm512_shuffle_f32x4(z, z, 0x4E));
    z = _mm512_fmadd_ps(z, vs4, _mm512_shuffle_f32x4(z, z, 0xB1));
    z = _mm512_fmadd_ps(z, vs2, _mm512_permute_ps(z, 0x4E));
    z = _mm512_fmadd_ps(z, vs1, _mm512_permute_ps(z, 0xB1));
    return z;
}

#define FWHT16_INTRA_X4(a, b, c, d, vs8, vs4, vs2, vs1)                                            \
    do {                                                                                           \
        __m512 t0_ = _mm512_shuffle_f32x4((a), (a), 0x4E);                                         \
        __m512 t1_ = _mm512_shuffle_f32x4((b), (b), 0x4E);                                         \
        __m512 t2_ = _mm512_shuffle_f32x4((c), (c), 0x4E);                                         \
        __m512 t3_ = _mm512_shuffle_f32x4((d), (d), 0x4E);                                         \
        (a) = _mm512_fmadd_ps((a), (vs8), t0_);                                                    \
        (b) = _mm512_fmadd_ps((b), (vs8), t1_);                                                    \
        (c) = _mm512_fmadd_ps((c), (vs8), t2_);                                                    \
        (d) = _mm512_fmadd_ps((d), (vs8), t3_);                                                    \
        t0_ = _mm512_shuffle_f32x4((a), (a), 0xB1);                                                \
        t1_ = _mm512_shuffle_f32x4((b), (b), 0xB1);                                                \
        t2_ = _mm512_shuffle_f32x4((c), (c), 0xB1);                                                \
        t3_ = _mm512_shuffle_f32x4((d), (d), 0xB1);                                                \
        (a) = _mm512_fmadd_ps((a), (vs4), t0_);                                                    \
        (b) = _mm512_fmadd_ps((b), (vs4), t1_);                                                    \
        (c) = _mm512_fmadd_ps((c), (vs4), t2_);                                                    \
        (d) = _mm512_fmadd_ps((d), (vs4), t3_);                                                    \
        t0_ = _mm512_permute_ps((a), 0x4E);                                                        \
        t1_ = _mm512_permute_ps((b), 0x4E);                                                        \
        t2_ = _mm512_permute_ps((c), 0x4E);                                                        \
        t3_ = _mm512_permute_ps((d), 0x4E);                                                        \
        (a) = _mm512_fmadd_ps((a), (vs2), t0_);                                                    \
        (b) = _mm512_fmadd_ps((b), (vs2), t1_);                                                    \
        (c) = _mm512_fmadd_ps((c), (vs2), t2_);                                                    \
        (d) = _mm512_fmadd_ps((d), (vs2), t3_);                                                    \
        t0_ = _mm512_permute_ps((a), 0xB1);                                                        \
        t1_ = _mm512_permute_ps((b), 0xB1);                                                        \
        t2_ = _mm512_permute_ps((c), 0xB1);                                                        \
        t3_ = _mm512_permute_ps((d), 0xB1);                                                        \
        (a) = _mm512_fmadd_ps((a), (vs1), t0_);                                                    \
        (b) = _mm512_fmadd_ps((b), (vs1), t1_);                                                    \
        (c) = _mm512_fmadd_ps((c), (vs1), t2_);                                                    \
        (d) = _mm512_fmadd_ps((d), (vs1), t3_);                                                    \
    } while (0)

static void avx512_fwht_f32(size_t b0, size_t b1, size_t cols, size_t block_size, const float *x,
                            size_t xs, velri_dtype suh_dt, const void *suh, float *y, size_t ys) {
    const size_t bpr = cols / block_size;
    const __m512 vnorm = _mm512_set1_ps(1.0f / sqrtf((float)block_size));
    const __m512 vs8 = _mm512_setr_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
                                      -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
    const __m512 vs4 = _mm512_setr_ps(1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
    const __m512 vs2 = _mm512_setr_ps(1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
                                      1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f);
    const __m512 vs1 = _mm512_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
                                      -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
    const size_t suh_eb = (suh_dt == VELRI_DTYPE_F32) ? sizeof(float) : sizeof(velri_fp16);
    const bool use_nt = (y != x) && ((b1 - b0) * block_size * 8ULL >= velri_nt_io_bytes());
    alignas(64) float suh_sc[8192];
    const bool use_sc = (suh != nullptr) && (cols <= 8192ULL) && ((b1 - b0) >= 2ULL * bpr);
    if (use_sc) {
        for (size_t j = 0; j < cols; j += 16) {
            __m512 vs = (suh_dt == VELRI_DTYPE_F32)
                            ? _mm512_loadu_ps((const float *)suh + j)
                            : _mm512_cvtph_ps(_mm256_loadu_si256(
                                  (const __m256i *)((const velri_fp16 *)suh + j)));
            _mm512_store_ps(suh_sc + j, _mm512_mul_ps(vs, vnorm));
        }
    }

    size_t r = (b0 < b1) ? (b0 / bpr) : 0;
    size_t c0 = (b0 < b1) ? ((b0 % bpr) * block_size) : 0;

    if (block_size == 256) {
        /*
         * Register-blocked 16-ZMM FWHT-256:
         * - 256 FP32 values live in `z0..z15` with ZERO stack spills (`24` ZMMs live <= `32`).
         * - Keep the butterfly stages together to reuse register values.
         */
        for (size_t bi = b0; bi < b1; ++bi) {
            const float *xr = x + r * xs + c0;
            float *yr = y + r * ys + c0;
            const float *scb = use_sc ? (suh_sc + c0) : nullptr;
            const void *sb = (!use_sc && suh) ? ((const uint8_t *)suh + c0 * suh_eb) : nullptr;
            c0 += 256;
            if (c0 == cols) {
                c0 = 0;
                ++r;
            }

            __m512 z0 = load16_fwht_avx512(xr, scb, sb, suh_dt, 0, vnorm);
            __m512 z1 = load16_fwht_avx512(xr, scb, sb, suh_dt, 16, vnorm);
            __m512 z2 = load16_fwht_avx512(xr, scb, sb, suh_dt, 32, vnorm);
            __m512 z3 = load16_fwht_avx512(xr, scb, sb, suh_dt, 48, vnorm);
            __m512 z4 = load16_fwht_avx512(xr, scb, sb, suh_dt, 64, vnorm);
            __m512 z5 = load16_fwht_avx512(xr, scb, sb, suh_dt, 80, vnorm);
            __m512 z6 = load16_fwht_avx512(xr, scb, sb, suh_dt, 96, vnorm);
            __m512 z7 = load16_fwht_avx512(xr, scb, sb, suh_dt, 112, vnorm);
            __m512 z8 = load16_fwht_avx512(xr, scb, sb, suh_dt, 128, vnorm);
            __m512 z9 = load16_fwht_avx512(xr, scb, sb, suh_dt, 144, vnorm);
            __m512 z10 = load16_fwht_avx512(xr, scb, sb, suh_dt, 160, vnorm);
            __m512 z11 = load16_fwht_avx512(xr, scb, sb, suh_dt, 176, vnorm);
            __m512 z12 = load16_fwht_avx512(xr, scb, sb, suh_dt, 192, vnorm);
            __m512 z13 = load16_fwht_avx512(xr, scb, sb, suh_dt, 208, vnorm);
            __m512 z14 = load16_fwht_avx512(xr, scb, sb, suh_dt, 224, vnorm);
            __m512 z15 = load16_fwht_avx512(xr, scb, sb, suh_dt, 240, vnorm);

            /* Stage h = 128 */
            ZMM_BUTTERFLY(z0, z8);
            ZMM_BUTTERFLY(z1, z9);
            ZMM_BUTTERFLY(z2, z10);
            ZMM_BUTTERFLY(z3, z11);
            ZMM_BUTTERFLY(z4, z12);
            ZMM_BUTTERFLY(z5, z13);
            ZMM_BUTTERFLY(z6, z14);
            ZMM_BUTTERFLY(z7, z15);

            /* Stage h = 64 */
            ZMM_BUTTERFLY(z0, z4);
            ZMM_BUTTERFLY(z1, z5);
            ZMM_BUTTERFLY(z2, z6);
            ZMM_BUTTERFLY(z3, z7);
            ZMM_BUTTERFLY(z8, z12);
            ZMM_BUTTERFLY(z9, z13);
            ZMM_BUTTERFLY(z10, z14);
            ZMM_BUTTERFLY(z11, z15);

            /* Stage h = 32 */
            ZMM_BUTTERFLY(z0, z2);
            ZMM_BUTTERFLY(z1, z3);
            ZMM_BUTTERFLY(z4, z6);
            ZMM_BUTTERFLY(z5, z7);
            ZMM_BUTTERFLY(z8, z10);
            ZMM_BUTTERFLY(z9, z11);
            ZMM_BUTTERFLY(z12, z14);
            ZMM_BUTTERFLY(z13, z15);

            /* Stage h = 16 */
            ZMM_BUTTERFLY(z0, z1);
            ZMM_BUTTERFLY(z2, z3);
            ZMM_BUTTERFLY(z4, z5);
            ZMM_BUTTERFLY(z6, z7);
            ZMM_BUTTERFLY(z8, z9);
            ZMM_BUTTERFLY(z10, z11);
            ZMM_BUTTERFLY(z12, z13);
            ZMM_BUTTERFLY(z14, z15);

            /* Stages h = 8, 4, 2, 1 */
            FWHT16_INTRA_X4(z0, z1, z2, z3, vs8, vs4, vs2, vs1);
            FWHT16_INTRA_X4(z4, z5, z6, z7, vs8, vs4, vs2, vs1);
            FWHT16_INTRA_X4(z8, z9, z10, z11, vs8, vs4, vs2, vs1);
            FWHT16_INTRA_X4(z12, z13, z14, z15, vs8, vs4, vs2, vs1);

            if (use_nt && (((uintptr_t)yr & 63ULL) == 0)) {
                _mm512_stream_ps(yr + 0, z0);
                _mm512_stream_ps(yr + 16, z1);
                _mm512_stream_ps(yr + 32, z2);
                _mm512_stream_ps(yr + 48, z3);
                _mm512_stream_ps(yr + 64, z4);
                _mm512_stream_ps(yr + 80, z5);
                _mm512_stream_ps(yr + 96, z6);
                _mm512_stream_ps(yr + 112, z7);
                _mm512_stream_ps(yr + 128, z8);
                _mm512_stream_ps(yr + 144, z9);
                _mm512_stream_ps(yr + 160, z10);
                _mm512_stream_ps(yr + 176, z11);
                _mm512_stream_ps(yr + 192, z12);
                _mm512_stream_ps(yr + 208, z13);
                _mm512_stream_ps(yr + 224, z14);
                _mm512_stream_ps(yr + 240, z15);
            } else {
                _mm512_storeu_ps(yr + 0, z0);
                _mm512_storeu_ps(yr + 16, z1);
                _mm512_storeu_ps(yr + 32, z2);
                _mm512_storeu_ps(yr + 48, z3);
                _mm512_storeu_ps(yr + 64, z4);
                _mm512_storeu_ps(yr + 80, z5);
                _mm512_storeu_ps(yr + 96, z6);
                _mm512_storeu_ps(yr + 112, z7);
                _mm512_storeu_ps(yr + 128, z8);
                _mm512_storeu_ps(yr + 144, z9);
                _mm512_storeu_ps(yr + 160, z10);
                _mm512_storeu_ps(yr + 176, z11);
                _mm512_storeu_ps(yr + 192, z12);
                _mm512_storeu_ps(yr + 208, z13);
                _mm512_storeu_ps(yr + 224, z14);
                _mm512_storeu_ps(yr + 240, z15);
            }
        }
        if (use_nt) {
            _mm_sfence();
        }
        return;
    }

    if (block_size == 128) {
        /* 2-block unrolled 16-ZMM FWHT-128 (`z0..z7` + `z8..z15`), zero `divq`, full ILP */
        size_t bi = b0;
        for (; bi + 2 <= b1; bi += 2) {
            const float *xr0 = x + r * xs + c0;
            float *yr0 = y + r * ys + c0;
            const float *scb0 = use_sc ? (suh_sc + c0) : nullptr;
            const void *sb0 = (!use_sc && suh) ? ((const uint8_t *)suh + c0 * suh_eb) : nullptr;
            c0 += 128;
            if (c0 == cols) {
                c0 = 0;
                ++r;
            }
            const float *xr1 = x + r * xs + c0;
            float *yr1 = y + r * ys + c0;
            const float *scb1 = use_sc ? (suh_sc + c0) : nullptr;
            const void *sb1 = (!use_sc && suh) ? ((const uint8_t *)suh + c0 * suh_eb) : nullptr;
            c0 += 128;
            if (c0 == cols) {
                c0 = 0;
                ++r;
            }

            __m512 z0 = load16_fwht_avx512(xr0, scb0, sb0, suh_dt, 0, vnorm);
            __m512 z1 = load16_fwht_avx512(xr0, scb0, sb0, suh_dt, 16, vnorm);
            __m512 z2 = load16_fwht_avx512(xr0, scb0, sb0, suh_dt, 32, vnorm);
            __m512 z3 = load16_fwht_avx512(xr0, scb0, sb0, suh_dt, 48, vnorm);
            __m512 z4 = load16_fwht_avx512(xr0, scb0, sb0, suh_dt, 64, vnorm);
            __m512 z5 = load16_fwht_avx512(xr0, scb0, sb0, suh_dt, 80, vnorm);
            __m512 z6 = load16_fwht_avx512(xr0, scb0, sb0, suh_dt, 96, vnorm);
            __m512 z7 = load16_fwht_avx512(xr0, scb0, sb0, suh_dt, 112, vnorm);
            __m512 z8 = load16_fwht_avx512(xr1, scb1, sb1, suh_dt, 0, vnorm);
            __m512 z9 = load16_fwht_avx512(xr1, scb1, sb1, suh_dt, 16, vnorm);
            __m512 z10 = load16_fwht_avx512(xr1, scb1, sb1, suh_dt, 32, vnorm);
            __m512 z11 = load16_fwht_avx512(xr1, scb1, sb1, suh_dt, 48, vnorm);
            __m512 z12 = load16_fwht_avx512(xr1, scb1, sb1, suh_dt, 64, vnorm);
            __m512 z13 = load16_fwht_avx512(xr1, scb1, sb1, suh_dt, 80, vnorm);
            __m512 z14 = load16_fwht_avx512(xr1, scb1, sb1, suh_dt, 96, vnorm);
            __m512 z15 = load16_fwht_avx512(xr1, scb1, sb1, suh_dt, 112, vnorm);

            ZMM_BUTTERFLY(z0, z4);
            ZMM_BUTTERFLY(z1, z5);
            ZMM_BUTTERFLY(z2, z6);
            ZMM_BUTTERFLY(z3, z7);
            ZMM_BUTTERFLY(z8, z12);
            ZMM_BUTTERFLY(z9, z13);
            ZMM_BUTTERFLY(z10, z14);
            ZMM_BUTTERFLY(z11, z15);

            ZMM_BUTTERFLY(z0, z2);
            ZMM_BUTTERFLY(z1, z3);
            ZMM_BUTTERFLY(z4, z6);
            ZMM_BUTTERFLY(z5, z7);
            ZMM_BUTTERFLY(z8, z10);
            ZMM_BUTTERFLY(z9, z11);
            ZMM_BUTTERFLY(z12, z14);
            ZMM_BUTTERFLY(z13, z15);

            ZMM_BUTTERFLY(z0, z1);
            ZMM_BUTTERFLY(z2, z3);
            ZMM_BUTTERFLY(z4, z5);
            ZMM_BUTTERFLY(z6, z7);
            ZMM_BUTTERFLY(z8, z9);
            ZMM_BUTTERFLY(z10, z11);
            ZMM_BUTTERFLY(z12, z13);
            ZMM_BUTTERFLY(z14, z15);

            FWHT16_INTRA_X4(z0, z1, z2, z3, vs8, vs4, vs2, vs1);
            FWHT16_INTRA_X4(z4, z5, z6, z7, vs8, vs4, vs2, vs1);
            FWHT16_INTRA_X4(z8, z9, z10, z11, vs8, vs4, vs2, vs1);
            FWHT16_INTRA_X4(z12, z13, z14, z15, vs8, vs4, vs2, vs1);

            if (use_nt && (((uintptr_t)yr0 & 63ULL) == 0) && (((uintptr_t)yr1 & 63ULL) == 0)) {
                _mm512_stream_ps(yr0 + 0, z0);
                _mm512_stream_ps(yr0 + 16, z1);
                _mm512_stream_ps(yr0 + 32, z2);
                _mm512_stream_ps(yr0 + 48, z3);
                _mm512_stream_ps(yr0 + 64, z4);
                _mm512_stream_ps(yr0 + 80, z5);
                _mm512_stream_ps(yr0 + 96, z6);
                _mm512_stream_ps(yr0 + 112, z7);
                _mm512_stream_ps(yr1 + 0, z8);
                _mm512_stream_ps(yr1 + 16, z9);
                _mm512_stream_ps(yr1 + 32, z10);
                _mm512_stream_ps(yr1 + 48, z11);
                _mm512_stream_ps(yr1 + 64, z12);
                _mm512_stream_ps(yr1 + 80, z13);
                _mm512_stream_ps(yr1 + 96, z14);
                _mm512_stream_ps(yr1 + 112, z15);
            } else {
                _mm512_storeu_ps(yr0 + 0, z0);
                _mm512_storeu_ps(yr0 + 16, z1);
                _mm512_storeu_ps(yr0 + 32, z2);
                _mm512_storeu_ps(yr0 + 48, z3);
                _mm512_storeu_ps(yr0 + 64, z4);
                _mm512_storeu_ps(yr0 + 80, z5);
                _mm512_storeu_ps(yr0 + 96, z6);
                _mm512_storeu_ps(yr0 + 112, z7);
                _mm512_storeu_ps(yr1 + 0, z8);
                _mm512_storeu_ps(yr1 + 16, z9);
                _mm512_storeu_ps(yr1 + 32, z10);
                _mm512_storeu_ps(yr1 + 48, z11);
                _mm512_storeu_ps(yr1 + 64, z12);
                _mm512_storeu_ps(yr1 + 80, z13);
                _mm512_storeu_ps(yr1 + 96, z14);
                _mm512_storeu_ps(yr1 + 112, z15);
            }
        }
        if (bi < b1) {
            const float *xr = x + r * xs + c0;
            float *yr = y + r * ys + c0;
            const float *scb = use_sc ? (suh_sc + c0) : nullptr;
            const void *sb = (!use_sc && suh) ? ((const uint8_t *)suh + c0 * suh_eb) : nullptr;
            __m512 z0 = load16_fwht_avx512(xr, scb, sb, suh_dt, 0, vnorm);
            __m512 z1 = load16_fwht_avx512(xr, scb, sb, suh_dt, 16, vnorm);
            __m512 z2 = load16_fwht_avx512(xr, scb, sb, suh_dt, 32, vnorm);
            __m512 z3 = load16_fwht_avx512(xr, scb, sb, suh_dt, 48, vnorm);
            __m512 z4 = load16_fwht_avx512(xr, scb, sb, suh_dt, 64, vnorm);
            __m512 z5 = load16_fwht_avx512(xr, scb, sb, suh_dt, 80, vnorm);
            __m512 z6 = load16_fwht_avx512(xr, scb, sb, suh_dt, 96, vnorm);
            __m512 z7 = load16_fwht_avx512(xr, scb, sb, suh_dt, 112, vnorm);
            ZMM_BUTTERFLY(z0, z4);
            ZMM_BUTTERFLY(z1, z5);
            ZMM_BUTTERFLY(z2, z6);
            ZMM_BUTTERFLY(z3, z7);
            ZMM_BUTTERFLY(z0, z2);
            ZMM_BUTTERFLY(z1, z3);
            ZMM_BUTTERFLY(z4, z6);
            ZMM_BUTTERFLY(z5, z7);
            ZMM_BUTTERFLY(z0, z1);
            ZMM_BUTTERFLY(z2, z3);
            ZMM_BUTTERFLY(z4, z5);
            ZMM_BUTTERFLY(z6, z7);
            FWHT16_INTRA_X4(z0, z1, z2, z3, vs8, vs4, vs2, vs1);
            FWHT16_INTRA_X4(z4, z5, z6, z7, vs8, vs4, vs2, vs1);
            _mm512_storeu_ps(yr + 0, z0);
            _mm512_storeu_ps(yr + 16, z1);
            _mm512_storeu_ps(yr + 32, z2);
            _mm512_storeu_ps(yr + 48, z3);
            _mm512_storeu_ps(yr + 64, z4);
            _mm512_storeu_ps(yr + 80, z5);
            _mm512_storeu_ps(yr + 96, z6);
            _mm512_storeu_ps(yr + 112, z7);
        }
        if (use_nt) {
            _mm_sfence();
        }
        return;
    }

    for (size_t bi = b0; bi < b1; ++bi) {
        const float *xr = x + r * xs + c0;
        float *yr = y + r * ys + c0;
        const float *scb = use_sc ? (suh_sc + c0) : nullptr;
        const void *sb = (!use_sc && suh) ? ((const uint8_t *)suh + c0 * suh_eb) : nullptr;
        c0 += block_size;
        if (c0 == cols) {
            c0 = 0;
            ++r;
        }

        if (block_size == 16) {
            __m512 z0 = load16_fwht_avx512(xr, scb, sb, suh_dt, 0, vnorm);
            _mm512_storeu_ps(yr, fwht16_intra_avx512(z0, vs8, vs4, vs2, vs1));
        } else if (block_size == 32) {
            __m512 z0 = load16_fwht_avx512(xr, scb, sb, suh_dt, 0, vnorm);
            __m512 z1 = load16_fwht_avx512(xr, scb, sb, suh_dt, 16, vnorm);
            ZMM_BUTTERFLY(z0, z1);
            _mm512_storeu_ps(yr + 0, fwht16_intra_avx512(z0, vs8, vs4, vs2, vs1));
            _mm512_storeu_ps(yr + 16, fwht16_intra_avx512(z1, vs8, vs4, vs2, vs1));
        } else {
            __m512 z0 = load16_fwht_avx512(xr, scb, sb, suh_dt, 0, vnorm);
            __m512 z1 = load16_fwht_avx512(xr, scb, sb, suh_dt, 16, vnorm);
            __m512 z2 = load16_fwht_avx512(xr, scb, sb, suh_dt, 32, vnorm);
            __m512 z3 = load16_fwht_avx512(xr, scb, sb, suh_dt, 48, vnorm);
            ZMM_BUTTERFLY(z0, z2);
            ZMM_BUTTERFLY(z1, z3);
            ZMM_BUTTERFLY(z0, z1);
            ZMM_BUTTERFLY(z2, z3);
            FWHT16_INTRA_X4(z0, z1, z2, z3, vs8, vs4, vs2, vs1);
            _mm512_storeu_ps(yr + 0, z0);
            _mm512_storeu_ps(yr + 16, z1);
            _mm512_storeu_ps(yr + 32, z2);
            _mm512_storeu_ps(yr + 48, z3);
        }
    }
}

#undef FWHT16_INTRA_X4
#undef ZMM_BUTTERFLY





static const velri_ops_vtable g_ops_avx512 = {
    .rmsnorm = avx512_rmsnorm,
    .fused_add_rmsnorm_f32 = avx512_fused_add_rmsnorm_f32,
    .layernorm_f32 = avx512_layernorm_f32,
    .rope = avx512_rope,
    .activation_f32 = avx512_activation_f32,
    .gated_mlp_f32 = avx512_gated_mlp_f32,
    .softmax_f32 = avx512_softmax_f32,
    .fwht_f32 = avx512_fwht_f32,
    .vec_axpby_f32 = velri_vec_axpby_f32_avx512,
    .vec_mul_f32 = velri_vec_mul_f32_avx512,
};

const velri_ops_vtable *velri_ops_get_avx512(void) { return &g_ops_avx512; }
