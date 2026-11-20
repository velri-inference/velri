#include "internal.h"

static inline float load_elem_f32(velri_dtype dt, const void *base, size_t idx) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        return ((const float *)base)[idx];
    case VELRI_DTYPE_BF16:
        return velri_bf16_to_f32_scalar(((const velri_bf16 *)base)[idx]);
    case VELRI_DTYPE_F16:
        return velri_f16_to_f32_scalar(((const velri_fp16 *)base)[idx]);
    default:
        return 0.0f;
    }
}

static inline void store_elem_f32(velri_dtype dt, void *base, size_t idx, float v) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        ((float *)base)[idx] = v;
        break;
    case VELRI_DTYPE_BF16:
        ((velri_bf16 *)base)[idx] = velri_f32_to_bf16_scalar(v);
        break;
    case VELRI_DTYPE_F16:
        ((velri_fp16 *)base)[idx] = velri_f32_to_f16_scalar(v);
        break;
    default:
        break;
    }
}

static void base_rmsnorm(velri_dtype io_dt, velri_dtype w_dt, size_t r0, size_t r1, size_t dim,
                         float eps, bool unit_offset, const void *x, size_t xs, const void *w,
                         void *y, size_t ys) {
    float inv_dim = 1.0f / (float)dim;
    for (size_t r = r0; r < r1; ++r) {
        size_t x_off = r * xs;
        size_t y_off = r * ys;
        float ss = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float v = load_elem_f32(io_dt, x, x_off + d);
            ss += v * v;
        }
        float rrms = 1.0f / sqrtf(ss * inv_dim + eps);
        for (size_t d = 0; d < dim; ++d) {
            float v = load_elem_f32(io_dt, x, x_off + d) * rrms;
            if (w) {
                float wv = load_elem_f32(w_dt, w, d);
                v *= unit_offset ? (1.0f + wv) : wv;
            }
            store_elem_f32(io_dt, y, y_off + d, v);
        }
    }
}

static void base_fused_add_rmsnorm_f32(size_t r0, size_t r1, size_t dim, float eps,
                                       bool unit_offset, float *res, size_t rs, const float *x,
                                       size_t xs, const float *w, float *y, size_t ys) {
    float inv_dim = 1.0f / (float)dim;
    for (size_t r = r0; r < r1; ++r) {
        float *rr = res + r * rs;
        const float *xr = x + r * xs;
        float *yr = y + r * ys;
        float ss = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float s = rr[d] + xr[d];
            rr[d] = s;
            ss += s * s;
        }
        float rrms = 1.0f / sqrtf(ss * inv_dim + eps);
        for (size_t d = 0; d < dim; ++d) {
            float v = rr[d] * rrms;
            if (w) {
                v *= unit_offset ? (1.0f + w[d]) : w[d];
            }
            yr[d] = v;
        }
    }
}

static void base_layernorm_f32(size_t r0, size_t r1, size_t dim, float eps, const float *x,
                               size_t xs, const float *w, const float *b, float *restrict y,
                               size_t ys) {
    float inv_dim = 1.0f / (float)dim;
    for (size_t r = r0; r < r1; ++r) {
        const float *xr = x + r * xs;
        float *yr = y + r * ys;
        float sum = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            sum += xr[d];
        }
        float mean = sum * inv_dim;
        float var_sum = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float diff = xr[d] - mean;
            var_sum += diff * diff;
        }
        float inv_std = 1.0f / sqrtf(var_sum * inv_dim + eps);
        for (size_t d = 0; d < dim; ++d) {
            float v = (xr[d] - mean) * inv_std;
            if (w) {
                v *= w[d];
            }
            if (b) {
                v += b[d];
            }
            yr[d] = v;
        }
    }
}

static void base_rope(velri_dtype dt, size_t t0, size_t t1, size_t n_heads, size_t head_dim,
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
                for (size_t d = 0; d < half; ++d) {
                    float x0 = load_elem_f32(dt, x, xo + d);
                    float x1 = load_elem_f32(dt, x, xo + half + d);
                    float c = cr[d], s = sr[d];
                    store_elem_f32(dt, y, yo + d, x0 * c - x1 * s);
                    store_elem_f32(dt, y, yo + half + d, x1 * c + x0 * s);
                }
            } else {
                for (size_t d = 0; d < half; ++d) {
                    float x0 = load_elem_f32(dt, x, xo + 2 * d);
                    float x1 = load_elem_f32(dt, x, xo + 2 * d + 1);
                    float c = cr[d], s = sr[d];
                    store_elem_f32(dt, y, yo + 2 * d, x0 * c - x1 * s);
                    store_elem_f32(dt, y, yo + 2 * d + 1, x1 * c + x0 * s);
                }
            }
            if (copy_tail) {
                for (size_t d = rotary_dim; d < head_dim; ++d) {
                    store_elem_f32(dt, y, yo + d, load_elem_f32(dt, x, xo + d));
                }
            }
        }
    }
}

static inline float eval_act_scalar(velri_act_kind kind, float x) {
    if (kind == VELRI_ACT_SILU) {
        float neg_x = -x;
        if (neg_x > 88.0f) {
            return 0.0f;
        }
        if (neg_x < -88.0f) {
            return x;
        }
        return x / (1.0f + expf(neg_x));
    }
    if (kind == VELRI_ACT_GELU_TANH) {
        /* 2 * sqrt(2 / pi) = 1.5957691216057308f */
        float x2 = x * x;
        float arg = -1.59576912f * x * (1.0f + 0.044715f * x2);
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

static void base_activation_f32(velri_act_kind kind, size_t i0, size_t i1, const float *x,
                                float *y) {
    for (size_t i = i0; i < i1; ++i) {
        y[i] = eval_act_scalar(kind, x[i]);
    }
}

static void base_gated_mlp_f32(velri_act_kind kind, size_t i0, size_t i1, float clamp_limit,
                               const float *gate, const float *up, float *restrict y) {
    bool do_clamp = (clamp_limit > 0.0f);
    float lo = -clamp_limit, hi = clamp_limit;
    for (size_t i = i0; i < i1; ++i) {
        float g = gate[i];
        float u = up[i];
        if (do_clamp) {
            g = g < lo ? lo : (g > hi ? hi : g);
            u = u < lo ? lo : (u > hi ? hi : u);
        }
        y[i] = eval_act_scalar(kind, g) * u;
    }
}

static void base_softmax_f32(size_t r0, size_t r1, size_t cols, const float *x, size_t xs, float *y,
                             size_t ys) {
    for (size_t r = r0; r < r1; ++r) {
        const float *xr = x + r * xs;
        float *yr = y + r * ys;
        float m = -INFINITY;
        float d = 0.0f;
        for (size_t i = 0; i < cols; ++i) {
            float v = xr[i];
            if (v > m) {
                d = (m == -INFINITY) ? 1.0f : (d * expf(m - v) + 1.0f);
                m = v;
            } else if (v > -INFINITY) {
                d += expf(v - m);
            }
        }
        float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
        for (size_t i = 0; i < cols; ++i) {
            yr[i] = (d > 0.0f) ? (expf(xr[i] - m) * inv_d) : 0.0f;
        }
    }
}

static void base_fwht_f32(size_t blk0, size_t blk1, size_t cols, size_t block_size, const float *x,
                          size_t xs, velri_dtype suh_dt, const void *suh, float *y, size_t ys) {
    const size_t bpr = cols / block_size;
    const float norm = 1.0f / sqrtf((float)block_size);
    float buf[256];

    for (size_t bi = blk0; bi < blk1; ++bi) {
        size_t r = bi / bpr;
        size_t c0 = (bi % bpr) * block_size;
        const float *xr = x + r * xs + c0;
        float *yr = y + r * ys + c0;

        if (!suh) {
            for (size_t j = 0; j < block_size; ++j) {
                buf[j] = xr[j] * norm;
            }
        } else if (suh_dt == VELRI_DTYPE_F32) {
            const float *sp = (const float *)suh + c0;
            for (size_t j = 0; j < block_size; ++j) {
                buf[j] = xr[j] * (sp[j] * norm);
            }
        } else {
            const velri_fp16 *sp = (const velri_fp16 *)suh + c0;
            for (size_t j = 0; j < block_size; ++j) {
                buf[j] = xr[j] * (velri_f16_to_f32_scalar(sp[j]) * norm);
            }
        }

        for (size_t h = block_size >> 1; h >= 1; h >>= 1) {
            size_t step = h << 1;
            for (size_t i = 0; i < block_size; i += step) {
                for (size_t j = 0; j < h; ++j) {
                    float a = buf[i + j];
                    float b = buf[i + j + h];
                    buf[i + j] = a + b;
                    buf[i + j + h] = a - b;
                }
            }
        }

        memcpy(yr, buf, block_size * sizeof(float));
    }
}

static void base_vec_axpby_f32(size_t i0, size_t i1, float alpha, const float *a, float beta,
                               const float *b, float *y) {
    if (beta == 0.0f) {
        for (size_t i = i0; i < i1; ++i) {
            y[i] = alpha * a[i];
        }
    } else if (alpha == 1.0f && beta == 1.0f) {
        for (size_t i = i0; i < i1; ++i) {
            y[i] = a[i] + b[i];
        }
    } else if (alpha == 1.0f) {
        for (size_t i = i0; i < i1; ++i) {
            y[i] = a[i] + beta * b[i];
        }
    } else {
        for (size_t i = i0; i < i1; ++i) {
            y[i] = alpha * a[i] + beta * b[i];
        }
    }
}

static void base_vec_mul_f32(size_t i0, size_t i1, const float *a, const float *b, float *y) {
    for (size_t i = i0; i < i1; ++i) {
        y[i] = a[i] * b[i];
    }
}

static const velri_ops_vtable g_ops_base = {
    .rmsnorm = base_rmsnorm,
    .fused_add_rmsnorm_f32 = base_fused_add_rmsnorm_f32,
    .layernorm_f32 = base_layernorm_f32,
    .rope = base_rope,
    .activation_f32 = base_activation_f32,
    .gated_mlp_f32 = base_gated_mlp_f32,
    .softmax_f32 = base_softmax_f32,
    .fwht_f32 = base_fwht_f32,
    .vec_axpby_f32 = base_vec_axpby_f32,
    .vec_mul_f32 = base_vec_mul_f32,
};

const velri_ops_vtable *velri_ops_get_base(void) { return &g_ops_base; }
