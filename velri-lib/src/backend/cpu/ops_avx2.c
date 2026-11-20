#include "internal.h"
#include <immintrin.h>

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline float hmax256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

/*
 * Minimax Cody-Waite degree-6 `expf` in AVX2 (degree-6 approximation;
 * returns exact 0.0f for x < -87.3365f including -INFINITY).
 */
static inline __m256 exp256_ps(__m256 orig_x) {
    const __m256 min_x = _mm256_set1_ps(-87.33654475f);
    const __m256 max_x = _mm256_set1_ps(88.37626266f);
    __m256 underflow_mask = _mm256_cmp_ps(orig_x, min_x, _CMP_LT_OQ);
    __m256 x = _mm256_min_ps(_mm256_max_ps(orig_x, min_x), max_x);

    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125f);
    const __m256 ln2_lo = _mm256_set1_ps(1.428606765330187045e-6f);

    __m256 n =
        _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, ln2_hi, x);
    r = _mm256_fnmadd_ps(n, ln2_lo, r);

    __m256 p = _mm256_set1_ps(1.38888889e-3f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.33333333e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.16666667e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.66666667e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.00000000e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));

    __m256i ni = _mm256_cvtps_epi32(n);
    __m256 scale =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
    __m256 res = _mm256_mul_ps(p, scale);
    return _mm256_andnot_ps(underflow_mask, res);
}

static inline __m256 load8_as_f32_avx2(velri_dtype dt, const void *base, size_t idx) {
    if (dt == VELRI_DTYPE_F32) {
        return _mm256_loadu_ps((const float *)base + idx);
    }
    if (dt == VELRI_DTYPE_F16) {
        return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)((const velri_fp16 *)base + idx)));
    }
    __m128i h = _mm_loadu_si128((const __m128i *)((const velri_bf16 *)base + idx));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

static inline void store8_from_f32_avx2(velri_dtype dt, void *base, size_t idx, __m256 v) {
    if (dt == VELRI_DTYPE_F32) {
        _mm256_storeu_ps((float *)base + idx, v);
        return;
    }
    if (dt == VELRI_DTYPE_F16) {
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)((velri_fp16 *)base + idx), h);
        return;
    }
    __m256i u = _mm256_castps_si256(v);
    __m256i is_nan = _mm256_cmpgt_epi32(_mm256_and_si256(u, _mm256_set1_epi32(0x7FFFFFFF)),
                                        _mm256_set1_epi32(0x7F800000));
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    __m256i r = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x00007FFF), lsb));
    r = _mm256_blendv_epi8(r, _mm256_or_si256(u, _mm256_set1_epi32(0x00400000)), is_nan);
    r = _mm256_srli_epi32(r, 16);
    __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi32(r, _mm256_setzero_si256()), 0xD8);
    _mm_storeu_si128((__m128i *)((velri_bf16 *)base + idx), _mm256_castsi256_si128(packed));
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

static void avx2_rmsnorm(velri_dtype io_dt, velri_dtype w_dt, size_t r0, size_t r1, size_t dim,
                         float eps, bool unit_offset, const void *x, size_t xs, const void *w,
                         void *y, size_t ys) {
    const float inv_dim = 1.0f / (float)dim;
    const __m256 v_one = _mm256_set1_ps(1.0f);

    for (size_t r = r0; r < r1; ++r) {
        size_t xo = r * xs;
        size_t yo = r * ys;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        size_t d = 0;
        for (; d + 16 <= dim; d += 16) {
            __m256 v0 = load8_as_f32_avx2(io_dt, x, xo + d);
            __m256 v1 = load8_as_f32_avx2(io_dt, x, xo + d + 8);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }
        for (; d + 8 <= dim; d += 8) {
            __m256 v0 = load8_as_f32_avx2(io_dt, x, xo + d);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        }
        float ss = hsum256_ps(_mm256_add_ps(acc0, acc1));
        for (; d < dim; ++d) {
            float v = load1_f32(io_dt, x, xo + d);
            ss += v * v;
        }

        float rrms = 1.0f / sqrtf(ss * inv_dim + eps);
        __m256 vrrms = _mm256_set1_ps(rrms);

        d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_mul_ps(load8_as_f32_avx2(io_dt, x, xo + d), vrrms);
            if (w) {
                __m256 wv = load8_as_f32_avx2(w_dt, w, d);
                if (unit_offset) {
                    wv = _mm256_add_ps(v_one, wv);
                }
                v = _mm256_mul_ps(v, wv);
            }
            store8_from_f32_avx2(io_dt, y, yo + d, v);
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
}

static void avx2_fused_add_rmsnorm_f32(size_t r0, size_t r1, size_t dim, float eps,
                                       bool unit_offset, float *res, size_t rs, const float *x,
                                       size_t xs, const float *w, float *y, size_t ys) {
    const float inv_dim = 1.0f / (float)dim;
    const __m256 v_one = _mm256_set1_ps(1.0f);

    for (size_t r = r0; r < r1; ++r) {
        float *rr = res + r * rs;
        const float *xr = x + r * xs;
        float *yr = y + r * ys;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        size_t d = 0;
        for (; d + 16 <= dim; d += 16) {
            __m256 s0 = _mm256_add_ps(_mm256_loadu_ps(rr + d), _mm256_loadu_ps(xr + d));
            __m256 s1 = _mm256_add_ps(_mm256_loadu_ps(rr + d + 8), _mm256_loadu_ps(xr + d + 8));
            _mm256_storeu_ps(rr + d, s0);
            _mm256_storeu_ps(rr + d + 8, s1);
            acc0 = _mm256_fmadd_ps(s0, s0, acc0);
            acc1 = _mm256_fmadd_ps(s1, s1, acc1);
        }
        float ss = hsum256_ps(_mm256_add_ps(acc0, acc1));
        for (; d < dim; ++d) {
            float s = rr[d] + xr[d];
            rr[d] = s;
            ss += s * s;
        }
        float rrms = 1.0f / sqrtf(ss * inv_dim + eps);
        __m256 vrrms = _mm256_set1_ps(rrms);
        d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(rr + d), vrrms);
            if (w) {
                __m256 wv = _mm256_loadu_ps(w + d);
                if (unit_offset) {
                    wv = _mm256_add_ps(v_one, wv);
                }
                v = _mm256_mul_ps(v, wv);
            }
            _mm256_storeu_ps(yr + d, v);
        }
        for (; d < dim; ++d) {
            float v = rr[d] * rrms;
            if (w) {
                v *= unit_offset ? (1.0f + w[d]) : w[d];
            }
            yr[d] = v;
        }
    }
}

static void avx2_layernorm_f32(size_t r0, size_t r1, size_t dim, float eps, const float *x,
                               size_t xs, const float *w, const float *b, float *restrict y,
                               size_t ys) {
    const float inv_dim = 1.0f / (float)dim;
    for (size_t r = r0; r < r1; ++r) {
        const float *xr = x + r * xs;
        float *yr = y + r * ys;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        size_t d = 0;
        for (; d + 16 <= dim; d += 16) {
            s0 = _mm256_add_ps(s0, _mm256_loadu_ps(xr + d));
            s1 = _mm256_add_ps(s1, _mm256_loadu_ps(xr + d + 8));
        }
        float sum = hsum256_ps(_mm256_add_ps(s0, s1));
        for (; d < dim; ++d) {
            sum += xr[d];
        }
        float mean = sum * inv_dim;
        __m256 vmean = _mm256_set1_ps(mean);

        __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
        d = 0;
        for (; d + 16 <= dim; d += 16) {
            __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(xr + d), vmean);
            __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(xr + d + 8), vmean);
            v0 = _mm256_fmadd_ps(d0, d0, v0);
            v1 = _mm256_fmadd_ps(d1, d1, v1);
        }
        float var_s = hsum256_ps(_mm256_add_ps(v0, v1));
        for (; d < dim; ++d) {
            float df = xr[d] - mean;
            var_s += df * df;
        }
        float inv_std = 1.0f / sqrtf(var_s * inv_dim + eps);
        __m256 vinv = _mm256_set1_ps(inv_std);

        d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 val = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + d), vmean), vinv);
            if (w) {
                val = _mm256_mul_ps(val, _mm256_loadu_ps(w + d));
            }
            if (b) {
                val = _mm256_add_ps(val, _mm256_loadu_ps(b + d));
            }
            _mm256_storeu_ps(yr + d, val);
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
}

/* ============================================================================
 * 2. Rotary Position Embeddings (`RoPE`)
 * ============================================================================ */

static void avx2_rope(velri_dtype dt, size_t t0, size_t t1, size_t n_heads, size_t head_dim,
                      size_t rotary_dim, size_t pos_offset, const void *x, size_t tsx, size_t hsx,
                      const float *cos_t, const float *sin_t, void *y, size_t tsy, size_t hsy,
                      velri_rope_mode mode) {
    const size_t half = rotary_dim / 2;
    const bool copy_tail = (y != x || tsx != tsy || hsx != hsy);

    for (size_t t = t0; t < t1; ++t) {
        const float *cr = rotary_dim ? (cos_t + (pos_offset + t) * half) : nullptr;
        const float *sr = rotary_dim ? (sin_t + (pos_offset + t) * half) : nullptr;
        for (size_t h = 0; h < n_heads; ++h) {
            size_t xo = t * tsx + h * hsx;
            size_t yo = t * tsy + h * hsy;
            if (mode == VELRI_ROPE_HALF) {
                size_t d = 0;
                for (; d + 8 <= half; d += 8) {
                    __m256 x0 = load8_as_f32_avx2(dt, x, xo + d);
                    __m256 x1 = load8_as_f32_avx2(dt, x, xo + half + d);
                    __m256 vc = _mm256_loadu_ps(cr + d);
                    __m256 vs = _mm256_loadu_ps(sr + d);
                    __m256 y0 = _mm256_fmsub_ps(x0, vc, _mm256_mul_ps(x1, vs));
                    __m256 y1 = _mm256_fmadd_ps(x1, vc, _mm256_mul_ps(x0, vs));
                    store8_from_f32_avx2(dt, y, yo + d, y0);
                    store8_from_f32_avx2(dt, y, yo + half + d, y1);
                }
                for (; d < half; ++d) {
                    float x0 = load1_f32(dt, x, xo + d);
                    float x1 = load1_f32(dt, x, xo + half + d);
                    store1_f32(dt, y, yo + d, x0 * cr[d] - x1 * sr[d]);
                    store1_f32(dt, y, yo + half + d, x1 * cr[d] + x0 * sr[d]);
                }
            } else {
                size_t d = 0;
                for (; d + 4 <= half; d += 4) {
                    /* Load 8 interleaved elements = 4 complex pairs */
                    __m256 xv = load8_as_f32_avx2(dt, x, xo + 2 * d);
                    __m256 x_swap = _mm256_permute_ps(xv, 0xB1); /* [x1, x0, x3, x2, ...] */
                    __m128 c4 = _mm_loadu_ps(cr + d);
                    __m128 s4 = _mm_loadu_ps(sr + d);
                    __m256 vc = _mm256_set_m128(_mm_unpackhi_ps(c4, c4), _mm_unpacklo_ps(c4, c4));
                    __m256 vs = _mm256_set_m128(_mm_unpackhi_ps(s4, s4), _mm_unpacklo_ps(s4, s4));
                    __m256 yv = _mm256_fmaddsub_ps(xv, vc, _mm256_mul_ps(x_swap, vs));
                    store8_from_f32_avx2(dt, y, yo + 2 * d, yv);
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
                for (; d + 8 <= head_dim; d += 8) {
                    store8_from_f32_avx2(dt, y, yo + d, load8_as_f32_avx2(dt, x, xo + d));
                }
                for (; d < head_dim; ++d) {
                    store1_f32(dt, y, yo + d, load1_f32(dt, x, xo + d));
                }
            }
        }
    }
}

/* ============================================================================
 * 3. Activations, GatedMLP & Online Softmax
 * ============================================================================ */

static inline __m256 rcp_nr_256(__m256 d) {
    __m256 r = _mm256_rcp_ps(d);
    return _mm256_fmadd_ps(r, _mm256_fnmadd_ps(d, r, _mm256_set1_ps(1.0f)), r);
}

static inline __m256 eval_act256(velri_act_kind kind, __m256 x) {
    const __m256 v_one = _mm256_set1_ps(1.0f);
    if (kind == VELRI_ACT_SILU) {
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        __m256 denom = _mm256_add_ps(v_one, exp256_ps(neg_x));
        __m256 res = _mm256_mul_ps(x, rcp_nr_256(denom));
        /* For x < -88.37f, exp(-x) clamped -> force exact 0.0f */
        __m256 zero_mask = _mm256_cmp_ps(x, _mm256_set1_ps(-88.37626f), _CMP_LT_OQ);
        return _mm256_andnot_ps(zero_mask, res);
    }
    if (kind == VELRI_ACT_GELU_TANH) {
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 inner = _mm256_fmadd_ps(_mm256_set1_ps(0.044715f), x2, v_one);
        __m256 arg = _mm256_mul_ps(_mm256_set1_ps(-1.59576912f), _mm256_mul_ps(x, inner));
        __m256 res = _mm256_mul_ps(x, rcp_nr_256(_mm256_add_ps(v_one, exp256_ps(arg))));
        __m256 zero_mask = _mm256_cmp_ps(arg, _mm256_set1_ps(88.37626f), _CMP_GT_OQ);
        return _mm256_andnot_ps(zero_mask, res);
    }
    __m256 r = _mm256_max_ps(x, _mm256_setzero_ps());
    return _mm256_mul_ps(r, r);
}

static inline float eval_act1(velri_act_kind kind, float x) {
    if (kind == VELRI_ACT_SILU) {
        if (-x > 88.0f) {
            return 0.0f;
        }
        if (-x < -88.0f) {
            return x;
        }
        return x / (1.0f + expf(-x));
    }
    if (kind == VELRI_ACT_GELU_TANH) {
        float arg = -1.59576912f * x * (1.0f + 0.044715f * x * x);
        if (arg > 88.0f) {
            return 0.0f;
        }
        if (arg < -88.0f) {
            return x;
        }
        return x / (1.0f + expf(arg));
    }
    float r = x > 0.0f ? x : 0.0f;
    return r * r;
}

static void avx2_activation_f32(velri_act_kind kind, size_t i0, size_t i1, const float *x,
                                float *y) {
    size_t i = i0;
    if ((i1 - i0) * 8ULL >= velri_nt_io_bytes() && y != x && (((uintptr_t)(y + i)) & 3ULL) == 0) {
        while (i < i1 && (((uintptr_t)(y + i)) & 31ULL) != 0) {
            y[i] = eval_act1(kind, x[i]);
            ++i;
        }
        for (; i + 16 <= i1; i += 16) {
            _mm256_stream_ps(y + i, eval_act256(kind, _mm256_loadu_ps(x + i)));
            _mm256_stream_ps(y + i + 8, eval_act256(kind, _mm256_loadu_ps(x + i + 8)));
        }
        _mm_sfence();
    }
    for (; i + 8 <= i1; i += 8) {
        _mm256_storeu_ps(y + i, eval_act256(kind, _mm256_loadu_ps(x + i)));
    }
    for (; i < i1; ++i) {
        y[i] = eval_act1(kind, x[i]);
    }
}

static void avx2_gated_mlp_f32(velri_act_kind kind, size_t i0, size_t i1, float clamp_limit,
                               const float *gate, const float *up, float *restrict y) {
    const bool do_clamp = (clamp_limit > 0.0f);
    const __m256 vlo = _mm256_set1_ps(-clamp_limit);
    const __m256 vhi = _mm256_set1_ps(clamp_limit);

    size_t i = i0;
    if ((i1 - i0) * 12ULL >= velri_nt_io_bytes() && (((uintptr_t)(y + i)) & 3ULL) == 0) {
        while (i < i1 && (((uintptr_t)(y + i)) & 31ULL) != 0) {
            float g = gate[i], u = up[i];
            if (do_clamp) {
                g = g < -clamp_limit ? -clamp_limit : (g > clamp_limit ? clamp_limit : g);
                u = u < -clamp_limit ? -clamp_limit : (u > clamp_limit ? clamp_limit : u);
            }
            y[i] = eval_act1(kind, g) * u;
            ++i;
        }
        for (; i + 16 <= i1; i += 16) {
            __m256 g0 = _mm256_loadu_ps(gate + i), u0 = _mm256_loadu_ps(up + i);
            __m256 g1 = _mm256_loadu_ps(gate + i + 8), u1 = _mm256_loadu_ps(up + i + 8);
            if (do_clamp) {
                g0 = _mm256_min_ps(_mm256_max_ps(g0, vlo), vhi);
                u0 = _mm256_min_ps(_mm256_max_ps(u0, vlo), vhi);
                g1 = _mm256_min_ps(_mm256_max_ps(g1, vlo), vhi);
                u1 = _mm256_min_ps(_mm256_max_ps(u1, vlo), vhi);
            }
            _mm256_stream_ps(y + i, _mm256_mul_ps(eval_act256(kind, g0), u0));
            _mm256_stream_ps(y + i + 8, _mm256_mul_ps(eval_act256(kind, g1), u1));
        }
        _mm_sfence();
    }
    for (; i + 8 <= i1; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        if (do_clamp) {
            g = _mm256_min_ps(_mm256_max_ps(g, vlo), vhi);
            u = _mm256_min_ps(_mm256_max_ps(u, vlo), vhi);
        }
        _mm256_storeu_ps(y + i, _mm256_mul_ps(eval_act256(kind, g), u));
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

static void avx2_softmax_f32(size_t r0, size_t r1, size_t cols, const float *x, size_t xs, float *y,
                             size_t ys) {
    for (size_t r = r0; r < r1; ++r) {
        const float *xr = x + r * xs;
        float *yr = y + r * ys;

        float m = -INFINITY;
        float d_sum = 0.0f;
        size_t i = 0;
        for (; i + 32 <= cols; i += 32) {
            __m256 v0 = _mm256_loadu_ps(xr + i);
            __m256 v1 = _mm256_loadu_ps(xr + i + 8);
            __m256 v2 = _mm256_loadu_ps(xr + i + 16);
            __m256 v3 = _mm256_loadu_ps(xr + i + 24);
            float blk_m = hmax256_ps(_mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3)));
            if (blk_m == -INFINITY) {
                continue;
            }
            float m_new = blk_m > m ? blk_m : m;
            float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
            __m256 vm = _mm256_set1_ps(m_new);
            __m256 s0 = exp256_ps(_mm256_sub_ps(v0, vm));
            __m256 s1 = exp256_ps(_mm256_sub_ps(v1, vm));
            __m256 s2 = exp256_ps(_mm256_sub_ps(v2, vm));
            __m256 s3 = exp256_ps(_mm256_sub_ps(v3, vm));
            d_sum = d_sum * alpha +
                    hsum256_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
            m = m_new;
        }
        for (; i < cols; ++i) {
            float v = xr[i];
            if (v > m) {
                d_sum = (m == -INFINITY) ? 1.0f : (d_sum * expf(m - v) + 1.0f);
                m = v;
            } else if (v > -INFINITY) {
                d_sum += expf(v - m);
            }
        }

        if (d_sum <= 0.0f) {
            memset(yr, 0, cols * sizeof(float));
            continue;
        }
        float inv_d = 1.0f / d_sum;
        __m256 vm = _mm256_set1_ps(m);
        __m256 vinv = _mm256_set1_ps(inv_d);
        i = 0;
        for (; i + 8 <= cols; i += 8) {
            __m256 p = _mm256_mul_ps(exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + i), vm)), vinv);
            _mm256_storeu_ps(yr + i, p);
        }
        for (; i < cols; ++i) {
            yr[i] = expf(xr[i] - m) * inv_d;
        }
    }
}

/* ============================================================================
 * 4. Orthonormal Block FWHT & Pure Vector Math (`AVX2 + FMA + F16C`)
 * ============================================================================ */

static inline __m256 load8_fwht_avx2(const float *xr, const float *sc_blk, const void *suh_blk,
                                     velri_dtype suh_dt, size_t j, __m256 vnorm) {
    __m256 vx = _mm256_loadu_ps(xr + j);
    if (sc_blk) {
        return _mm256_mul_ps(vx, _mm256_load_ps(sc_blk + j));
    }
    if (suh_blk) {
        __m256 vs = (suh_dt == VELRI_DTYPE_F32)
                        ? _mm256_loadu_ps((const float *)suh_blk + j)
                        : _mm256_cvtph_ps(
                              _mm_loadu_si128((const __m128i *)((const velri_fp16 *)suh_blk + j)));
        return _mm256_mul_ps(vx, _mm256_mul_ps(vs, vnorm));
    }
    return _mm256_mul_ps(vx, vnorm);
}

#define AVX2_BUTTERFLY(a, b)                                                                       \
    do {                                                                                           \
        __m256 ta_ = (a), tb_ = (b);                                                               \
        (a) = _mm256_add_ps(ta_, tb_);                                                             \
        (b) = _mm256_sub_ps(ta_, tb_);                                                             \
    } while (0)

static inline __m256 fwht8_intra_avx2(__m256 v, __m256 vs4, __m256 vs2, __m256 vs1) {
    v = _mm256_fmadd_ps(v, vs4, _mm256_permute2f128_ps(v, v, 0x01));
    v = _mm256_fmadd_ps(v, vs2, _mm256_permute_ps(v, 0x4E));
    v = _mm256_fmadd_ps(v, vs1, _mm256_permute_ps(v, 0xB1));
    return v;
}

static inline void fwht64_block_avx2(const float *src, float *dst, __m256 vs4, __m256 vs2,
                                     __m256 vs1) {
    __m256 v0 = _mm256_load_ps(src + 0), v1 = _mm256_load_ps(src + 8);
    __m256 v2 = _mm256_load_ps(src + 16), v3 = _mm256_load_ps(src + 24);
    __m256 v4 = _mm256_load_ps(src + 32), v5 = _mm256_load_ps(src + 40);
    __m256 v6 = _mm256_load_ps(src + 48), v7 = _mm256_load_ps(src + 56);

    AVX2_BUTTERFLY(v0, v4);
    AVX2_BUTTERFLY(v1, v5);
    AVX2_BUTTERFLY(v2, v6);
    AVX2_BUTTERFLY(v3, v7);

    AVX2_BUTTERFLY(v0, v2);
    AVX2_BUTTERFLY(v1, v3);
    AVX2_BUTTERFLY(v4, v6);
    AVX2_BUTTERFLY(v5, v7);

    AVX2_BUTTERFLY(v0, v1);
    AVX2_BUTTERFLY(v2, v3);
    AVX2_BUTTERFLY(v4, v5);
    AVX2_BUTTERFLY(v6, v7);

    _mm256_storeu_ps(dst + 0, fwht8_intra_avx2(v0, vs4, vs2, vs1));
    _mm256_storeu_ps(dst + 8, fwht8_intra_avx2(v1, vs4, vs2, vs1));
    _mm256_storeu_ps(dst + 16, fwht8_intra_avx2(v2, vs4, vs2, vs1));
    _mm256_storeu_ps(dst + 24, fwht8_intra_avx2(v3, vs4, vs2, vs1));
    _mm256_storeu_ps(dst + 32, fwht8_intra_avx2(v4, vs4, vs2, vs1));
    _mm256_storeu_ps(dst + 40, fwht8_intra_avx2(v5, vs4, vs2, vs1));
    _mm256_storeu_ps(dst + 48, fwht8_intra_avx2(v6, vs4, vs2, vs1));
    _mm256_storeu_ps(dst + 56, fwht8_intra_avx2(v7, vs4, vs2, vs1));
}

static void avx2_fwht_f32(size_t b0, size_t b1, size_t cols, size_t block_size, const float *x,
                          size_t xs, velri_dtype suh_dt, const void *suh, float *y, size_t ys) {
    const size_t bpr = cols / block_size;
    const __m256 vnorm = _mm256_set1_ps(1.0f / sqrtf((float)block_size));
    const __m256 vs4 = _mm256_setr_ps(1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
    const __m256 vs2 = _mm256_setr_ps(1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f);
    const __m256 vs1 = _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
    const size_t suh_eb = (suh_dt == VELRI_DTYPE_F32) ? sizeof(float) : sizeof(velri_fp16);
    alignas(32) float buf[256];
    alignas(32) float suh_sc[8192];
    const bool use_sc = (suh != nullptr) && (cols <= 8192ULL) && ((b1 - b0) >= 2ULL * bpr);
    if (use_sc) {
        for (size_t j = 0; j < cols; j += 8) {
            __m256 vs = (suh_dt == VELRI_DTYPE_F32)
                            ? _mm256_loadu_ps((const float *)suh + j)
                            : _mm256_cvtph_ps(
                                  _mm_loadu_si128((const __m128i *)((const velri_fp16 *)suh + j)));
            _mm256_store_ps(suh_sc + j, _mm256_mul_ps(vs, vnorm));
        }
    }

    size_t r = (b0 < b1) ? (b0 / bpr) : 0;
    size_t c0 = (b0 < b1) ? ((b0 % bpr) * block_size) : 0;
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
            __m256 v0 = load8_fwht_avx2(xr, scb, sb, suh_dt, 0, vnorm);
            __m256 v1 = load8_fwht_avx2(xr, scb, sb, suh_dt, 8, vnorm);
            AVX2_BUTTERFLY(v0, v1);
            _mm256_storeu_ps(yr + 0, fwht8_intra_avx2(v0, vs4, vs2, vs1));
            _mm256_storeu_ps(yr + 8, fwht8_intra_avx2(v1, vs4, vs2, vs1));
        } else if (block_size == 32) {
            __m256 v0 = load8_fwht_avx2(xr, scb, sb, suh_dt, 0, vnorm);
            __m256 v1 = load8_fwht_avx2(xr, scb, sb, suh_dt, 8, vnorm);
            __m256 v2 = load8_fwht_avx2(xr, scb, sb, suh_dt, 16, vnorm);
            __m256 v3 = load8_fwht_avx2(xr, scb, sb, suh_dt, 24, vnorm);
            AVX2_BUTTERFLY(v0, v2);
            AVX2_BUTTERFLY(v1, v3);
            AVX2_BUTTERFLY(v0, v1);
            AVX2_BUTTERFLY(v2, v3);
            _mm256_storeu_ps(yr + 0, fwht8_intra_avx2(v0, vs4, vs2, vs1));
            _mm256_storeu_ps(yr + 8, fwht8_intra_avx2(v1, vs4, vs2, vs1));
            _mm256_storeu_ps(yr + 16, fwht8_intra_avx2(v2, vs4, vs2, vs1));
            _mm256_storeu_ps(yr + 24, fwht8_intra_avx2(v3, vs4, vs2, vs1));
        } else if (block_size == 64) {
            for (size_t j = 0; j < 64; j += 8) {
                _mm256_store_ps(buf + j, load8_fwht_avx2(xr, scb, sb, suh_dt, j, vnorm));
            }
            fwht64_block_avx2(buf, yr, vs4, vs2, vs1);
        } else if (block_size == 128) {
            for (size_t j = 0; j < 64; j += 8) {
                __m256 v0 = load8_fwht_avx2(xr, scb, sb, suh_dt, j, vnorm);
                __m256 v1 = load8_fwht_avx2(xr, scb, sb, suh_dt, j + 64, vnorm);
                _mm256_store_ps(buf + j, _mm256_add_ps(v0, v1));
                _mm256_store_ps(buf + j + 64, _mm256_sub_ps(v0, v1));
            }
            fwht64_block_avx2(buf, yr, vs4, vs2, vs1);
            fwht64_block_avx2(buf + 64, yr + 64, vs4, vs2, vs1);
        } else {
            /* block_size == 256: fused load + suh*norm + radix-4 (h=128, h=64) -> 4x fwht64 */
            if (use_sc) {
                for (size_t j = 0; j < 64; j += 8) {
                    __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(xr + j), _mm256_load_ps(scb + j));
                    __m256 v1 =
                        _mm256_mul_ps(_mm256_loadu_ps(xr + j + 64), _mm256_load_ps(scb + j + 64));
                    __m256 v2 =
                        _mm256_mul_ps(_mm256_loadu_ps(xr + j + 128), _mm256_load_ps(scb + j + 128));
                    __m256 v3 =
                        _mm256_mul_ps(_mm256_loadu_ps(xr + j + 192), _mm256_load_ps(scb + j + 192));
                    AVX2_BUTTERFLY(v0, v2);
                    AVX2_BUTTERFLY(v1, v3);
                    AVX2_BUTTERFLY(v0, v1);
                    AVX2_BUTTERFLY(v2, v3);
                    _mm256_store_ps(buf + j, v0);
                    _mm256_store_ps(buf + j + 64, v1);
                    _mm256_store_ps(buf + j + 128, v2);
                    _mm256_store_ps(buf + j + 192, v3);
                }
            } else {
                for (size_t j = 0; j < 64; j += 8) {
                    __m256 v0 = load8_fwht_avx2(xr, nullptr, sb, suh_dt, j, vnorm);
                    __m256 v1 = load8_fwht_avx2(xr, nullptr, sb, suh_dt, j + 64, vnorm);
                    __m256 v2 = load8_fwht_avx2(xr, nullptr, sb, suh_dt, j + 128, vnorm);
                    __m256 v3 = load8_fwht_avx2(xr, nullptr, sb, suh_dt, j + 192, vnorm);
                    AVX2_BUTTERFLY(v0, v2);
                    AVX2_BUTTERFLY(v1, v3);
                    AVX2_BUTTERFLY(v0, v1);
                    AVX2_BUTTERFLY(v2, v3);
                    _mm256_store_ps(buf + j, v0);
                    _mm256_store_ps(buf + j + 64, v1);
                    _mm256_store_ps(buf + j + 128, v2);
                    _mm256_store_ps(buf + j + 192, v3);
                }
            }
            fwht64_block_avx2(buf, yr, vs4, vs2, vs1);
            fwht64_block_avx2(buf + 64, yr + 64, vs4, vs2, vs1);
            fwht64_block_avx2(buf + 128, yr + 128, vs4, vs2, vs1);
            fwht64_block_avx2(buf + 192, yr + 192, vs4, vs2, vs1);
        }
    }
}

#undef AVX2_BUTTERFLY





static const velri_ops_vtable g_ops_avx2 = {
    .rmsnorm = avx2_rmsnorm,
    .fused_add_rmsnorm_f32 = avx2_fused_add_rmsnorm_f32,
    .layernorm_f32 = avx2_layernorm_f32,
    .rope = avx2_rope,
    .activation_f32 = avx2_activation_f32,
    .gated_mlp_f32 = avx2_gated_mlp_f32,
    .softmax_f32 = avx2_softmax_f32,
    .fwht_f32 = avx2_fwht_f32,
    .vec_axpby_f32 = velri_vec_axpby_f32_avx2,
    .vec_mul_f32 = velri_vec_mul_f32_avx2,
};

const velri_ops_vtable *velri_ops_get_avx2(void) { return &g_ops_avx2; }
