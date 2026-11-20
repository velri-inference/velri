#include "internal.h"

static const velri_ops_vtable *resolve_ops_vtable(velri_cpu_variant variant) {
    if (!velri_cpu_variant_supported(variant)) {
        return nullptr;
    }
    velri_cpu_variant r = velri_cpu_resolve_variant(variant);
#if VELRI_X86_VARIANTS
    if (r == VELRI_CPU_AVX512) {
        return velri_ops_get_avx512();
    }
    if (r == VELRI_CPU_AVX2) {
        return velri_ops_get_avx2();
    }
#endif
    if (r == VELRI_CPU_BASE) {
        return velri_ops_get_base();
    }
    return nullptr;
}

static bool is_supported_norm_dtype(velri_dtype dt) {
    return dt == VELRI_DTYPE_F32 || dt == VELRI_DTYPE_BF16 || dt == VELRI_DTYPE_F16;
}

static velri_status check_2d_ptr(const void *ptr, size_t rows, size_t cols, size_t stride,
                                 velri_dtype dt, size_t *out_bytes) {
    return velri_check_span(ptr, SIZE_MAX, rows, cols, stride, velri_dtype_bytes(dt),
                            velri_dtype_align(dt), out_bytes);
}

static velri_status check_3d_ptr(const void *ptr, size_t d0, size_t d1, size_t d2, size_t s0,
                                 size_t s1, velri_dtype dt, bool require_non_overlapping_slices,
                                 size_t *out_bytes) {
    if (d2 == 0) {
        return VELRI_ERR_INVALID;
    }
    if (d0 == 0 || d1 == 0) {
        *out_bytes = 0;
        return VELRI_OK;
    }
    if (d1 > 1 && s1 < d2) {
        return VELRI_ERR_INVALID;
    }
    if (d0 > 1 && s0 < d2) {
        return VELRI_ERR_INVALID;
    }
    size_t off0 = 0, off1 = 0, sum01 = 0, used = 0;
    if (!velri_mul_size(d0 - 1, s0, &off0) || !velri_mul_size(d1 - 1, s1, &off1) ||
        !velri_add_size(off0, off1, &sum01) || !velri_add_size(sum01, d2, &used)) {
        return VELRI_ERR_OVERFLOW;
    }
    if (require_non_overlapping_slices && d0 > 1 && d1 > 1) {
        /* Either BSHD (`s0 >= (d1 - 1)*s1 + d2`) or BHSD (`s1 >= (d0 - 1)*s0 + d2`) */
        size_t bshd_min = 0, bhsd_min = 0;
        bool ok_bshd = velri_add_size(off1, d2, &bshd_min) && (s0 >= bshd_min);
        bool ok_bhsd = velri_add_size(off0, d2, &bhsd_min) && (s1 >= bhsd_min);
        if (!ok_bshd && !ok_bhsd) {
            return VELRI_ERR_INVALID;
        }
    }
    size_t eb = velri_dtype_bytes(dt);
    size_t al = velri_dtype_align(dt);
    size_t bytes = 0;
    if (!velri_mul_size(used, eb, &bytes)) {
        return VELRI_ERR_OVERFLOW;
    }
    if (!ptr || ((uintptr_t)ptr % al) != 0) {
        return VELRI_ERR_INVALID;
    }
    if ((uintptr_t)ptr > UINTPTR_MAX - bytes) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_bytes = bytes;
    return VELRI_OK;
}

/* ============================================================================
 * 1. Normalization Dispatch (`RMSNorm`, `Fused Residual+RMSNorm`, `LayerNorm`)
 * ============================================================================ */

typedef struct {
    const velri_ops_vtable *vt;
    velri_dtype io_dt;
    velri_dtype w_dt;
    size_t rows;
    size_t dim;
    float eps;
    bool unit_offset;
    const void *x;
    size_t xs;
    const void *w;
    const float *bias;
    float *res;
    size_t rs;
    void *y;
    size_t ys;
    int op; /* 0 = rmsnorm, 1 = fused_add_rmsnorm, 2 = layernorm */
} norm_par_ctx;

static void norm_parallel_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const norm_par_ctx *c = (const norm_par_ctx *)vctx;
    size_t per = (c->rows + nw - 1) / nw;
    size_t r0 = wid * per;
    size_t r1 = r0 + per < c->rows ? r0 + per : c->rows;
    if (r0 >= r1) {
        return;
    }
    if (c->op == 0) {
        c->vt->rmsnorm(c->io_dt, c->w_dt, r0, r1, c->dim, c->eps, c->unit_offset, c->x, c->xs, c->w,
                       c->y, c->ys);
    } else if (c->op == 1) {
        c->vt->fused_add_rmsnorm_f32(r0, r1, c->dim, c->eps, c->unit_offset, c->res, c->rs,
                                     (const float *)c->x, c->xs, (const float *)c->w, (float *)c->y,
                                     c->ys);
    } else {
        c->vt->layernorm_f32(r0, r1, c->dim, c->eps, (const float *)c->x, c->xs,
                             (const float *)c->w, c->bias, (float *)c->y, c->ys);
    }
}

velri_status velri_rmsnorm(velri_dtype io_dtype, velri_dtype weight_dtype, size_t rows, size_t dim,
                           float eps, bool unit_offset, const void *x, size_t x_stride,
                           const void *weight, void *y, size_t y_stride, velri_cpu_variant variant,
                           velri_pool *pool, size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (!is_supported_norm_dtype(io_dtype) || !is_supported_norm_dtype(weight_dtype)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (dim == 0 || !(eps >= 0.0f) || isinf(eps)) {
        return VELRI_ERR_INVALID;
    }
    size_t xb = 0, yb = 0, wb = 0;
    velri_status s = check_2d_ptr(x, rows, dim, x_stride, io_dtype, &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(y, rows, dim, y_stride, io_dtype, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (weight) {
        s = check_2d_ptr(weight, 1, dim, dim, weight_dtype, &wb);
        if (s != VELRI_OK) {
            return s;
        }
    }
    if (rows == 0) {
        return VELRI_OK;
    }
    bool inplace = (y == x && x_stride == y_stride);
    if ((!inplace && velri_overlap(y, yb, x, xb)) || velri_overlap(y, yb, weight, wb)) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || rows <= 1) {
        vt->rmsnorm(io_dtype, weight_dtype, 0, rows, dim, eps, unit_offset, x, x_stride, weight, y,
                    y_stride);
        return VELRI_OK;
    }
    norm_par_ctx ctx = {.vt = vt,
                        .io_dt = io_dtype,
                        .w_dt = weight_dtype,
                        .rows = rows,
                        .dim = dim,
                        .eps = eps,
                        .unit_offset = unit_offset,
                        .x = x,
                        .xs = x_stride,
                        .w = weight,
                        .y = y,
                        .ys = y_stride,
                        .op = 0};
    size_t nw = active_workers < rows ? active_workers : rows;
    velri_pool_parallel(pool, nw, norm_parallel_worker, &ctx);
    return VELRI_OK;
}

velri_status velri_fused_add_rmsnorm_f32(size_t rows, size_t dim, float eps, bool unit_offset,
                                         float *residual_inout, size_t r_stride, const float *x,
                                         size_t x_stride, const float *weight, float *y,
                                         size_t y_stride, velri_cpu_variant variant,
                                         velri_pool *pool, size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (dim == 0 || !(eps >= 0.0f) || isinf(eps)) {
        return VELRI_ERR_INVALID;
    }
    size_t rb = 0, xb = 0, yb = 0, wb = 0;
    velri_status s = check_2d_ptr(residual_inout, rows, dim, r_stride, VELRI_DTYPE_F32, &rb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(x, rows, dim, x_stride, VELRI_DTYPE_F32, &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(y, rows, dim, y_stride, VELRI_DTYPE_F32, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (weight) {
        s = check_2d_ptr(weight, 1, dim, dim, VELRI_DTYPE_F32, &wb);
        if (s != VELRI_OK) {
            return s;
        }
    }
    if (rows == 0) {
        return VELRI_OK;
    }
    bool y_eq_res = (y == residual_inout && y_stride == r_stride);
    if (velri_overlap(residual_inout, rb, x, xb) || velri_overlap(residual_inout, rb, weight, wb) ||
        (!y_eq_res && velri_overlap(y, yb, residual_inout, rb)) || velri_overlap(y, yb, x, xb) ||
        velri_overlap(y, yb, weight, wb)) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || rows <= 1) {
        vt->fused_add_rmsnorm_f32(0, rows, dim, eps, unit_offset, residual_inout, r_stride, x,
                                  x_stride, weight, y, y_stride);
        return VELRI_OK;
    }
    norm_par_ctx ctx = {.vt = vt,
                        .rows = rows,
                        .dim = dim,
                        .eps = eps,
                        .unit_offset = unit_offset,
                        .x = x,
                        .xs = x_stride,
                        .w = weight,
                        .res = residual_inout,
                        .rs = r_stride,
                        .y = y,
                        .ys = y_stride,
                        .op = 1};
    size_t nw = active_workers < rows ? active_workers : rows;
    velri_pool_parallel(pool, nw, norm_parallel_worker, &ctx);
    return VELRI_OK;
}

velri_status velri_layernorm_f32(size_t rows, size_t dim, float eps, const float *x,
                                 size_t x_stride, const float *weight, const float *bias,
                                 float *restrict y, size_t y_stride, velri_cpu_variant variant,
                                 velri_pool *pool, size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (dim == 0 || !(eps >= 0.0f) || isinf(eps)) {
        return VELRI_ERR_INVALID;
    }
    size_t xb = 0, yb = 0, wb = 0, bb = 0;
    velri_status s = check_2d_ptr(x, rows, dim, x_stride, VELRI_DTYPE_F32, &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(y, rows, dim, y_stride, VELRI_DTYPE_F32, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (weight) {
        s = check_2d_ptr(weight, 1, dim, dim, VELRI_DTYPE_F32, &wb);
        if (s != VELRI_OK) {
            return s;
        }
    }
    if (bias) {
        s = check_2d_ptr(bias, 1, dim, dim, VELRI_DTYPE_F32, &bb);
        if (s != VELRI_OK) {
            return s;
        }
    }
    if (rows == 0) {
        return VELRI_OK;
    }
    if (velri_overlap(y, yb, x, xb) || velri_overlap(y, yb, weight, wb) ||
        velri_overlap(y, yb, bias, bb)) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || rows <= 1) {
        vt->layernorm_f32(0, rows, dim, eps, x, x_stride, weight, bias, y, y_stride);
        return VELRI_OK;
    }
    norm_par_ctx ctx = {.vt = vt,
                        .rows = rows,
                        .dim = dim,
                        .eps = eps,
                        .x = x,
                        .xs = x_stride,
                        .w = weight,
                        .bias = bias,
                        .y = y,
                        .ys = y_stride,
                        .op = 2};
    size_t nw = active_workers < rows ? active_workers : rows;
    velri_pool_parallel(pool, nw, norm_parallel_worker, &ctx);
    return VELRI_OK;
}

/* ============================================================================
 * 2. Rotary Position Embeddings (`RoPE`) Dispatch
 * ============================================================================ */

typedef struct {
    const velri_ops_vtable *vt;
    velri_dtype dt;
    size_t n_tokens;
    size_t n_heads;
    size_t head_dim;
    size_t rotary_dim;
    size_t pos_offset;
    const void *x;
    size_t tsx;
    size_t hsx;
    const float *cos_t;
    const float *sin_t;
    void *y;
    size_t tsy;
    size_t hsy;
    velri_rope_mode mode;
} rope_par_ctx;

static void rope_parallel_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const rope_par_ctx *c = (const rope_par_ctx *)vctx;
    size_t per = (c->n_tokens + nw - 1) / nw;
    size_t t0 = wid * per;
    size_t t1 = t0 + per < c->n_tokens ? t0 + per : c->n_tokens;
    if (t0 >= t1) {
        return;
    }
    c->vt->rope(c->dt, t0, t1, c->n_heads, c->head_dim, c->rotary_dim, c->pos_offset, c->x, c->tsx,
                c->hsx, c->cos_t, c->sin_t, c->y, c->tsy, c->hsy, c->mode);
}

velri_status velri_rope(velri_dtype dtype, size_t n_tokens, size_t n_heads, size_t head_dim,
                        size_t rotary_dim, size_t pos_offset, size_t cos_sin_tokens, const void *x,
                        size_t token_stride_x, size_t head_stride_x, const float *cos_table,
                        const float *sin_table, void *y, size_t token_stride_y,
                        size_t head_stride_y, velri_rope_mode mode, velri_cpu_variant variant,
                        velri_pool *pool, size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt || !is_supported_norm_dtype(dtype)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (mode != VELRI_ROPE_HALF && mode != VELRI_ROPE_INTERLEAVED) {
        return VELRI_ERR_INVALID;
    }
    if (n_heads == 0 || head_dim == 0 || rotary_dim > head_dim || (rotary_dim & 1ULL) != 0) {
        return VELRI_ERR_INVALID;
    }
    if (head_stride_x < head_dim || head_stride_y < head_dim) {
        return VELRI_ERR_INVALID;
    }

    size_t xb = 0, yb = 0, cb = 0, sb = 0;
    velri_status s = check_3d_ptr(x, n_tokens, n_heads, head_dim, token_stride_x, head_stride_x,
                                  dtype, false, &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_3d_ptr(y, n_tokens, n_heads, head_dim, token_stride_y, head_stride_y, dtype, true,
                     &yb);
    if (s != VELRI_OK) {
        return s;
    }

    if (n_tokens == 0) {
        return VELRI_OK;
    }

    if (rotary_dim > 0) {
        size_t max_pos = 0;
        if (!velri_add_size(pos_offset, n_tokens, &max_pos)) {
            return VELRI_ERR_OVERFLOW;
        }
        if (max_pos > cos_sin_tokens) {
            return VELRI_ERR_INVALID;
        }
        size_t half = rotary_dim / 2;
        s = check_2d_ptr(cos_table, cos_sin_tokens, half, half, VELRI_DTYPE_F32, &cb);
        if (s != VELRI_OK) {
            return s;
        }
        s = check_2d_ptr(sin_table, cos_sin_tokens, half, half, VELRI_DTYPE_F32, &sb);
        if (s != VELRI_OK) {
            return s;
        }
    }

    bool inplace = (y == x && token_stride_x == token_stride_y && head_stride_x == head_stride_y);
    if ((!inplace && velri_overlap(y, yb, x, xb)) || velri_overlap(y, yb, cos_table, cb) ||
        velri_overlap(y, yb, sin_table, sb)) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || n_tokens <= 1) {
        vt->rope(dtype, 0, n_tokens, n_heads, head_dim, rotary_dim, pos_offset, x, token_stride_x,
                 head_stride_x, cos_table, sin_table, y, token_stride_y, head_stride_y, mode);
        return VELRI_OK;
    }
    rope_par_ctx ctx = {.vt = vt,
                        .dt = dtype,
                        .n_tokens = n_tokens,
                        .n_heads = n_heads,
                        .head_dim = head_dim,
                        .rotary_dim = rotary_dim,
                        .pos_offset = pos_offset,
                        .x = x,
                        .tsx = token_stride_x,
                        .hsx = head_stride_x,
                        .cos_t = cos_table,
                        .sin_t = sin_table,
                        .y = y,
                        .tsy = token_stride_y,
                        .hsy = head_stride_y,
                        .mode = mode};
    size_t nw = active_workers < n_tokens ? active_workers : n_tokens;
    velri_pool_parallel(pool, nw, rope_parallel_worker, &ctx);
    return VELRI_OK;
}

/* ============================================================================
 * 3. Elementwise Activations, GatedMLP & Online Softmax Dispatch
 * ============================================================================ */

typedef struct {
    const velri_ops_vtable *vt;
    velri_act_kind kind;
    size_t count;
    float clamp_limit;
    const float *a;
    const float *b;
    float *y;
    bool is_gated;
} act_par_ctx;

static void act_parallel_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const act_par_ctx *c = (const act_par_ctx *)vctx;
    size_t blocks = (c->count + 63) / 64;
    size_t b_per = (blocks + nw - 1) / nw;
    size_t i0 = wid * b_per * 64;
    size_t i1 = (wid + 1) * b_per * 64;
    if (i0 >= c->count) {
        return;
    }
    if (i1 > c->count) {
        i1 = c->count;
    }
    if (c->is_gated) {
        c->vt->gated_mlp_f32(c->kind, i0, i1, c->clamp_limit, c->a, c->b, c->y);
    } else {
        c->vt->activation_f32(c->kind, i0, i1, c->a, c->y);
    }
}

static bool valid_act_kind(velri_act_kind k) {
    return k == VELRI_ACT_SILU || k == VELRI_ACT_GELU_TANH || k == VELRI_ACT_RELU2;
}

velri_status velri_activation_f32(velri_act_kind kind, size_t count, const float *x, float *y,
                                  velri_cpu_variant variant, velri_pool *pool,
                                  size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (!valid_act_kind(kind)) {
        return VELRI_ERR_INVALID;
    }
    if (count == 0) {
        return VELRI_OK;
    }
    size_t xb = 0, yb = 0;
    velri_status s = check_2d_ptr(x, 1, count, count, VELRI_DTYPE_F32, &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(y, 1, count, count, VELRI_DTYPE_F32, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (y != x && velri_overlap(y, yb, x, xb)) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || count < 4096) {
        vt->activation_f32(kind, 0, count, x, y);
        return VELRI_OK;
    }
    act_par_ctx ctx = {.vt = vt, .kind = kind, .count = count, .a = x, .y = y, .is_gated = false};
    velri_pool_parallel(pool, active_workers, act_parallel_worker, &ctx);
    return VELRI_OK;
}

velri_status velri_gated_mlp_f32(velri_act_kind gate_act, size_t count, float clamp_limit,
                                 const float *gate, const float *up, float *restrict y,
                                 velri_cpu_variant variant, velri_pool *pool,
                                 size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (!valid_act_kind(gate_act) || !(clamp_limit >= 0.0f) || isinf(clamp_limit)) {
        return VELRI_ERR_INVALID;
    }
    if (count == 0) {
        return VELRI_OK;
    }
    size_t gb = 0, ub = 0, yb = 0;
    velri_status s = check_2d_ptr(gate, 1, count, count, VELRI_DTYPE_F32, &gb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(up, 1, count, count, VELRI_DTYPE_F32, &ub);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(y, 1, count, count, VELRI_DTYPE_F32, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (velri_overlap(y, yb, gate, gb) || velri_overlap(y, yb, up, ub)) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || count < 4096) {
        vt->gated_mlp_f32(gate_act, 0, count, clamp_limit, gate, up, y);
        return VELRI_OK;
    }
    act_par_ctx ctx = {.vt = vt,
                       .kind = gate_act,
                       .count = count,
                       .clamp_limit = clamp_limit,
                       .a = gate,
                       .b = up,
                       .y = y,
                       .is_gated = true};
    velri_pool_parallel(pool, active_workers, act_parallel_worker, &ctx);
    return VELRI_OK;
}

typedef struct {
    const velri_ops_vtable *vt;
    size_t rows;
    size_t cols;
    const float *x;
    size_t xs;
    float *y;
    size_t ys;
} sm_par_ctx;

static void sm_parallel_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const sm_par_ctx *c = (const sm_par_ctx *)vctx;
    size_t per = (c->rows + nw - 1) / nw;
    size_t r0 = wid * per;
    size_t r1 = r0 + per < c->rows ? r0 + per : c->rows;
    if (r0 >= r1) {
        return;
    }
    c->vt->softmax_f32(r0, r1, c->cols, c->x, c->xs, c->y, c->ys);
}

velri_status velri_softmax_f32(size_t rows, size_t cols, const float *x, size_t x_stride, float *y,
                               size_t y_stride, velri_cpu_variant variant, velri_pool *pool,
                               size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (cols == 0) {
        return VELRI_ERR_INVALID;
    }
    size_t xb = 0, yb = 0;
    velri_status s = check_2d_ptr(x, rows, cols, x_stride, VELRI_DTYPE_F32, &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(y, rows, cols, y_stride, VELRI_DTYPE_F32, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (rows == 0) {
        return VELRI_OK;
    }
    bool inplace = (y == x && x_stride == y_stride);
    if (!inplace && velri_overlap(y, yb, x, xb)) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || rows <= 1) {
        vt->softmax_f32(0, rows, cols, x, x_stride, y, y_stride);
        return VELRI_OK;
    }
    sm_par_ctx ctx = {
        .vt = vt, .rows = rows, .cols = cols, .x = x, .xs = x_stride, .y = y, .ys = y_stride};
    size_t nw = active_workers < rows ? active_workers : rows;
    velri_pool_parallel(pool, nw, sm_parallel_worker, &ctx);
    return VELRI_OK;
}

/* ============================================================================
 * 4. Normalized Fast Walsh-Hadamard Transform (`FWHT`)
 * ============================================================================ */

typedef struct {
    const velri_ops_vtable *vt;
    size_t total_blocks;
    size_t cols;
    size_t block_size;
    const float *x;
    size_t xs;
    velri_dtype suh_dt;
    const void *suh;
    float *y;
    size_t ys;
} fwht_par_ctx;

static void fwht_parallel_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const fwht_par_ctx *c = (const fwht_par_ctx *)vctx;
    size_t per = (c->total_blocks + nw - 1) / nw;
    size_t b0 = wid * per;
    size_t b1 = b0 + per < c->total_blocks ? b0 + per : c->total_blocks;
    if (b0 >= b1) {
        return;
    }
    c->vt->fwht_f32(b0, b1, c->cols, c->block_size, c->x, c->xs, c->suh_dt, c->suh, c->y, c->ys);
}

static bool valid_fwht_block_size(size_t b) {
    return b == 16 || b == 32 || b == 64 || b == 128 || b == 256;
}

velri_status velri_fwht_f32(size_t rows, size_t cols, size_t block_size, const float *x,
                            size_t x_stride, velri_dtype suh_dtype, const void *suh, float *y,
                            size_t y_stride, velri_cpu_variant variant, velri_pool *pool,
                            size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (!valid_fwht_block_size(block_size) || cols == 0 || (cols % block_size) != 0) {
        return VELRI_ERR_INVALID;
    }
    size_t xb = 0, yb = 0, sb = 0;
    velri_status s = check_2d_ptr(x, rows, cols, x_stride, VELRI_DTYPE_F32, &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(y, rows, cols, y_stride, VELRI_DTYPE_F32, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (suh) {
        if (suh_dtype != VELRI_DTYPE_F32 && suh_dtype != VELRI_DTYPE_F16) {
            return VELRI_ERR_UNSUPPORTED;
        }
        s = check_2d_ptr(suh, 1, cols, cols, suh_dtype, &sb);
        if (s != VELRI_OK) {
            return s;
        }
    }
    if (rows == 0) {
        return VELRI_OK;
    }
    bool inplace = (y == x && x_stride == y_stride);
    if ((!inplace && velri_overlap(y, yb, x, xb)) || velri_overlap(y, yb, suh, sb)) {
        return VELRI_ERR_INVALID;
    }

    size_t blocks_per_row = cols / block_size;
    size_t total_blocks = 0;
    if (!velri_mul_size(rows, blocks_per_row, &total_blocks)) {
        return VELRI_ERR_OVERFLOW;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || total_blocks <= 1 ||
        total_blocks * block_size < 4096) {
        vt->fwht_f32(0, total_blocks, cols, block_size, x, x_stride, suh_dtype, suh, y, y_stride);
        return VELRI_OK;
    }

    fwht_par_ctx ctx = {.vt = vt,
                        .total_blocks = total_blocks,
                        .cols = cols,
                        .block_size = block_size,
                        .x = x,
                        .xs = x_stride,
                        .suh_dt = suh_dtype,
                        .suh = suh,
                        .y = y,
                        .ys = y_stride};
    size_t nw = active_workers < total_blocks ? active_workers : total_blocks;
    velri_pool_parallel(pool, nw, fwht_parallel_worker, &ctx);
    return VELRI_OK;
}

/* ============================================================================
 * 5. Elementwise Vector Math (`axpby`, `mul`)
 * ============================================================================ */

typedef struct {
    const velri_ops_vtable *vt;
    size_t count;
    float alpha;
    float beta;
    const float *a;
    const float *b;
    float *y;
    bool is_mul;
} vec_par_ctx;

static void vec_parallel_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const vec_par_ctx *c = (const vec_par_ctx *)vctx;
    size_t blocks = (c->count + 63) / 64;
    size_t b_per = (blocks + nw - 1) / nw;
    size_t i0 = wid * b_per * 64;
    size_t i1 = (wid + 1) * b_per * 64;
    if (i0 >= c->count) {
        return;
    }
    if (i1 > c->count) {
        i1 = c->count;
    }
    if (c->is_mul) {
        c->vt->vec_mul_f32(i0, i1, c->a, c->b, c->y);
    } else {
        c->vt->vec_axpby_f32(i0, i1, c->alpha, c->a, c->beta, c->b, c->y);
    }
}

velri_status velri_vec_axpby_f32(size_t count, float alpha, const float *a, float beta,
                                 const float *b, float *y, velri_cpu_variant variant,
                                 velri_pool *pool, size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (isnan(alpha) || isinf(alpha) || isnan(beta) || isinf(beta)) {
        return VELRI_ERR_INVALID;
    }
    size_t ab = 0, bb = 0, yb = 0;
    velri_status s = check_2d_ptr(a, 1, count, count, VELRI_DTYPE_F32, &ab);
    if (s != VELRI_OK) {
        return s;
    }
    if (b != nullptr || beta != 0.0f) {
        s = check_2d_ptr(b, 1, count, count, VELRI_DTYPE_F32, &bb);
        if (s != VELRI_OK) {
            return s;
        }
    }
    s = check_2d_ptr(y, 1, count, count, VELRI_DTYPE_F32, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (count == 0) {
        return VELRI_OK;
    }
    if ((y != a && velri_overlap(y, yb, a, ab)) ||
        (b != nullptr && y != b && velri_overlap(y, yb, b, bb))) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || count < 16384) {
        vt->vec_axpby_f32(0, count, alpha, a, beta, b, y);
        return VELRI_OK;
    }
    vec_par_ctx ctx = {.vt = vt,
                       .count = count,
                       .alpha = alpha,
                       .beta = beta,
                       .a = a,
                       .b = b,
                       .y = y,
                       .is_mul = false};
    velri_pool_parallel(pool, active_workers, vec_parallel_worker, &ctx);
    return VELRI_OK;
}

velri_status velri_vec_mul_f32(size_t count, const float *a, const float *b, float *y,
                               velri_cpu_variant variant, velri_pool *pool, size_t active_workers) {
    const velri_ops_vtable *vt = resolve_ops_vtable(variant);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }
    size_t ab = 0, bb = 0, yb = 0;
    velri_status s = check_2d_ptr(a, 1, count, count, VELRI_DTYPE_F32, &ab);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(b, 1, count, count, VELRI_DTYPE_F32, &bb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(y, 1, count, count, VELRI_DTYPE_F32, &yb);
    if (s != VELRI_OK) {
        return s;
    }
    if (count == 0) {
        return VELRI_OK;
    }
    if ((y != a && velri_overlap(y, yb, a, ab)) || (y != b && velri_overlap(y, yb, b, bb))) {
        return VELRI_ERR_INVALID;
    }

    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || count < 16384) {
        vt->vec_mul_f32(0, count, a, b, y);
        return VELRI_OK;
    }
    vec_par_ctx ctx = {.vt = vt, .count = count, .a = a, .b = b, .y = y, .is_mul = true};
    velri_pool_parallel(pool, active_workers, vec_parallel_worker, &ctx);
    return VELRI_OK;
}

/* ============================================================================
 * 5. Parallel Row Quantization & Parallel Dtype Conversions
 * ============================================================================ */

typedef struct {
    size_t rows;
    size_t k;
    const float *x;
    size_t xs;
    int8_t *x_i8;
    size_t qs;
    float *scale_out;
    velri_cpu_variant variant;
} quant_par_ctx;

static void quant_parallel_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const quant_par_ctx *c = (const quant_par_ctx *)vctx;
    size_t per = (c->rows + nw - 1) / nw;
    size_t r0 = wid * per;
    size_t r1 = r0 + per < c->rows ? r0 + per : c->rows;
    if (r0 >= r1) {
        return;
    }
    (void)velri_quantize_row_f32_i8(r1 - r0, c->k, c->x + r0 * c->xs, c->xs, c->x_i8 + r0 * c->qs,
                                    c->qs, c->scale_out + r0, c->variant);
}

velri_status velri_quantize_row_f32_i8_parallel(size_t rows, size_t k, const float *x,
                                                size_t x_stride, int8_t *restrict x_i8,
                                                size_t xi8_stride, float *restrict scale_out,
                                                velri_cpu_variant variant, velri_pool *pool,
                                                size_t active_workers) {
    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || rows <= 1) {
        return velri_quantize_row_f32_i8(rows, k, x, x_stride, x_i8, xi8_stride, scale_out,
                                         variant);
    }
    if (k == 0) {
        return VELRI_ERR_INVALID;
    }
    if (!velri_cpu_variant_supported(variant)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    velri_cpu_variant resolved = velri_cpu_resolve_variant(variant);
    size_t xb = 0, qb = 0, sb = 0;
    velri_status s = check_2d_ptr(x, rows, k, x_stride, VELRI_DTYPE_F32, &xb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(x_i8, rows, k, xi8_stride, VELRI_DTYPE_I8, &qb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(scale_out, 1, rows, rows, VELRI_DTYPE_F32, &sb);
    if (s != VELRI_OK) {
        return s;
    }
    if (velri_overlap(x, xb, x_i8, qb) || velri_overlap(x, xb, scale_out, sb) ||
        velri_overlap(x_i8, qb, scale_out, sb)) {
        return VELRI_ERR_INVALID;
    }

    quant_par_ctx ctx = {.rows = rows,
                         .k = k,
                         .x = x,
                         .xs = x_stride,
                         .x_i8 = x_i8,
                         .qs = xi8_stride,
                         .scale_out = scale_out,
                         .variant = resolved};
    size_t nw = active_workers < rows ? active_workers : rows;
    velri_pool_parallel(pool, nw, quant_parallel_worker, &ctx);
    return VELRI_OK;
}

typedef struct {
    velri_dtype src_dt;
    velri_dtype dst_dt;
    const void *src;
    void *dst;
    size_t count;
    velri_cpu_variant variant;
} cvt_par_ctx;

static velri_status run_cvt_slice(velri_dtype src_dt, velri_dtype dst_dt, const void *src,
                                  void *dst, size_t i0, size_t n, velri_cpu_variant v) {
    if (src_dt == VELRI_DTYPE_F32 && dst_dt == VELRI_DTYPE_F16) {
        return velri_cvt_f32_to_f16((const float *)src + i0, (velri_fp16 *)dst + i0, n, v);
    }
    if (src_dt == VELRI_DTYPE_F16 && dst_dt == VELRI_DTYPE_F32) {
        return velri_cvt_f16_to_f32((const velri_fp16 *)src + i0, (float *)dst + i0, n, v);
    }
    if (src_dt == VELRI_DTYPE_F32 && dst_dt == VELRI_DTYPE_BF16) {
        return velri_cvt_f32_to_bf16((const float *)src + i0, (velri_bf16 *)dst + i0, n, v);
    }
    if (src_dt == VELRI_DTYPE_BF16 && dst_dt == VELRI_DTYPE_F32) {
        return velri_cvt_bf16_to_f32((const velri_bf16 *)src + i0, (float *)dst + i0, n, v);
    }
    return VELRI_ERR_UNSUPPORTED;
}

static void cvt_parallel_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const cvt_par_ctx *c = (const cvt_par_ctx *)vctx;
    size_t blocks = (c->count + 63) / 64;
    size_t b_per = (blocks + nw - 1) / nw;
    size_t i0 = wid * b_per * 64;
    size_t i1 = (wid + 1) * b_per * 64;
    if (i0 >= c->count) {
        return;
    }
    if (i1 > c->count) {
        i1 = c->count;
    }
    (void)run_cvt_slice(c->src_dt, c->dst_dt, c->src, c->dst, i0, i1 - i0, c->variant);
}

velri_status velri_cvt_parallel(velri_dtype src_dtype, velri_dtype dst_dtype, const void *src,
                                void *restrict dst, size_t count, velri_cpu_variant variant,
                                velri_pool *pool, size_t active_workers) {
    bool valid_pair = (src_dtype == VELRI_DTYPE_F32 &&
                       (dst_dtype == VELRI_DTYPE_F16 || dst_dtype == VELRI_DTYPE_BF16)) ||
                      (dst_dtype == VELRI_DTYPE_F32 &&
                       (src_dtype == VELRI_DTYPE_F16 || src_dtype == VELRI_DTYPE_BF16));
    if (!valid_pair || !velri_cpu_variant_supported(variant)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    velri_cpu_variant resolved = velri_cpu_resolve_variant(variant);
    size_t sb = 0, db = 0;
    velri_status s = check_2d_ptr(src, 1, count, count, src_dtype, &sb);
    if (s != VELRI_OK) {
        return s;
    }
    s = check_2d_ptr(dst, 1, count, count, dst_dtype, &db);
    if (s != VELRI_OK) {
        return s;
    }
    if (velri_overlap(src, sb, dst, db)) {
        return VELRI_ERR_INVALID;
    }
    if (count == 0) {
        return VELRI_OK;
    }
    if (!pool || active_workers <= 1 || velri_pool_num_workers(pool) <= 1 || count < 16384) {
        return run_cvt_slice(src_dtype, dst_dtype, src, dst, 0, count, resolved);
    }
    cvt_par_ctx ctx = {.src_dt = src_dtype,
                       .dst_dt = dst_dtype,
                       .src = src,
                       .dst = dst,
                       .count = count,
                       .variant = resolved};
    velri_pool_parallel(pool, active_workers, cvt_parallel_worker, &ctx);
    return VELRI_OK;
}
