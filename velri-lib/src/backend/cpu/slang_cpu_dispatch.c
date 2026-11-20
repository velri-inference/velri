#include "velri_lib/slang_cpu.h"
#include <stdint.h>

/* Declare all exported Slang AOT symbols for each ISA compiled from entry_cpu.slang. */
#define DECLARE_SLANG_ISA(ISA)                                                                    \
    extern void velri_slang_##ISA##_linear_f32(const float *, uint32_t, const float *, uint32_t,  \
                                               const float *, uint32_t, float *, uint32_t,        \
                                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t); \
    extern void velri_slang_##ISA##_linear_f16_f32(const uint16_t *, uint32_t, const uint16_t *,  \
                                                   uint32_t, const float *, uint32_t, float *,    \
                                                   uint32_t, uint32_t, uint32_t, uint32_t,        \
                                                   uint32_t, uint32_t);                           \
    extern void velri_slang_##ISA##_linear_bf16_f32(const uint16_t *, uint32_t, const uint16_t *, \
                                                    uint32_t, const float *, uint32_t, float *,   \
                                                    uint32_t, uint32_t, uint32_t, uint32_t,       \
                                                    uint32_t, uint32_t);                          \
    extern void velri_slang_##ISA##_linear_f32x_bf16w(                                            \
        const float *, uint32_t, const uint16_t *, uint32_t, const float *, uint32_t, float *,    \
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);                              \
    extern void velri_slang_##ISA##_linear_f64(const double *, uint32_t, const double *,          \
                                               uint32_t, const double *, uint32_t, double *,      \
                                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,  \
                                               uint32_t);                                         \
    extern void velri_slang_##ISA##_linear_w8a8_f32(                                              \
        const int8_t *, uint32_t, const int8_t *, uint32_t, const float *, const float *,         \
        const float *, uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,       \
        uint32_t);                                                                                \
    extern void velri_slang_##ISA##_rmsnorm_f32(const float *, uint32_t, const float *, uint32_t, \
                                                float *, uint32_t, uint32_t, uint32_t, uint32_t,  \
                                                float, uint32_t);                                 \
    extern void velri_slang_##ISA##_rmsnorm_bf16(const uint16_t *, uint32_t, const uint16_t *,    \
                                                 uint32_t, uint16_t *, uint32_t, uint32_t,        \
                                                 uint32_t, uint32_t, float, uint32_t);            \
    extern void velri_slang_##ISA##_rmsnorm_f16(const uint16_t *, uint32_t, const uint16_t *,     \
                                                uint32_t, uint16_t *, uint32_t, uint32_t,         \
                                                uint32_t, uint32_t, float, uint32_t);             \
    extern void velri_slang_##ISA##_fused_add_rmsnorm_f32(                                        \
        float *, uint32_t, const float *, uint32_t, const float *, uint32_t, float *, uint32_t,   \
        uint32_t, uint32_t, uint32_t, float, uint32_t);                                           \
    extern void velri_slang_##ISA##_layernorm_f32(                                                \
        const float *, uint32_t, const float *, uint32_t, const float *, uint32_t, float *,       \
        uint32_t, uint32_t, uint32_t, uint32_t, float);                                           \
    extern void velri_slang_##ISA##_rope_f32(const float *, uint32_t, uint32_t, const float *,    \
                                             const float *, float *, uint32_t, uint32_t,          \
                                             uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,    \
                                             uint32_t, uint32_t);                                 \
    extern void velri_slang_##ISA##_rope_bf16(const uint16_t *, uint32_t, uint32_t,               \
                                              const float *, const float *, uint16_t *, uint32_t, \
                                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,   \
                                              uint32_t, uint32_t, uint32_t);                      \
    extern void velri_slang_##ISA##_activation_silu_f32(const float *, float *, uint32_t,         \
                                                        uint32_t);                                \
    extern void velri_slang_##ISA##_activation_gelu_f32(const float *, float *, uint32_t,         \
                                                        uint32_t);                                \
    extern void velri_slang_##ISA##_activation_relu2_f32(const float *, float *, uint32_t,        \
                                                         uint32_t);                               \
    extern void velri_slang_##ISA##_gated_silu_f32(const float *, const float *, float *,         \
                                                   uint32_t, uint32_t, float);                    \
    extern void velri_slang_##ISA##_gated_gelu_f32(const float *, const float *, float *,         \
                                                   uint32_t, uint32_t, float);                    \
    extern void velri_slang_##ISA##_softmax_f32(const float *, uint32_t, float *, uint32_t,       \
                                                uint32_t, uint32_t, uint32_t);                    \
    extern void velri_slang_##ISA##_vec_axpby_f32(float, const float *, float, const float *,     \
                                                  float *, uint32_t, uint32_t);                   \
    extern void velri_slang_##ISA##_vec_mul_f32(const float *, const float *, float *,            \
                                                uint32_t, uint32_t);                              \
    extern void velri_slang_##ISA##_fwht_f32(const float *, uint32_t, const uint8_t *, uint32_t,  \
                                             float *, uint32_t, uint32_t, uint32_t, uint32_t,     \
                                             uint32_t);                                           \
    extern void velri_slang_##ISA##_quantize_row_f32_i8(const float *, uint32_t, int8_t *,        \
                                                        uint32_t, float *, uint32_t, uint32_t,    \
                                                        uint32_t);

DECLARE_SLANG_ISA(base)
#if VELRI_X86_VARIANTS
DECLARE_SLANG_ISA(avx2)
DECLARE_SLANG_ISA(avx512)
#endif

typedef struct {
    void (*linear_f32)(const float *, uint32_t, const float *, uint32_t, const float *, uint32_t,
                       float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    void (*linear_f16_f32)(const uint16_t *, uint32_t, const uint16_t *, uint32_t, const float *,
                           uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                           uint32_t);
    void (*linear_bf16_f32)(const uint16_t *, uint32_t, const uint16_t *, uint32_t, const float *,
                            uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t);
    void (*linear_f32x_bf16w)(const float *, uint32_t, const uint16_t *, uint32_t, const float *,
                              uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t);
    void (*linear_f64)(const double *, uint32_t, const double *, uint32_t, const double *, uint32_t,
                       double *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    void (*linear_w8a8_f32)(const int8_t *, uint32_t, const int8_t *, uint32_t, const float *,
                            const float *, const float *, uint32_t, float *, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t);
    void (*rmsnorm_f32)(const float *, uint32_t, const float *, uint32_t, float *, uint32_t,
                        uint32_t, uint32_t, uint32_t, float, uint32_t);
    void (*rmsnorm_bf16)(const uint16_t *, uint32_t, const uint16_t *, uint32_t, uint16_t *,
                         uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t);
    void (*rmsnorm_f16)(const uint16_t *, uint32_t, const uint16_t *, uint32_t, uint16_t *,
                        uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t);
    void (*fused_add_rmsnorm_f32)(float *, uint32_t, const float *, uint32_t, const float *,
                                  uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t, float,
                                  uint32_t);
    void (*layernorm_f32)(const float *, uint32_t, const float *, uint32_t, const float *, uint32_t,
                          float *, uint32_t, uint32_t, uint32_t, uint32_t, float);
    void (*rope_f32)(const float *, uint32_t, uint32_t, const float *, const float *, float *,
                     uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                     uint32_t);
    void (*rope_bf16)(const uint16_t *, uint32_t, uint32_t, const float *, const float *,
                      uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t);
    void (*activation_silu_f32)(const float *, float *, uint32_t, uint32_t);
    void (*activation_gelu_f32)(const float *, float *, uint32_t, uint32_t);
    void (*activation_relu2_f32)(const float *, float *, uint32_t, uint32_t);
    void (*gated_silu_f32)(const float *, const float *, float *, uint32_t, uint32_t, float);
    void (*gated_gelu_f32)(const float *, const float *, float *, uint32_t, uint32_t, float);
    void (*softmax_f32)(const float *, uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t);
    void (*vec_axpby_f32)(float, const float *, float, const float *, float *, uint32_t, uint32_t);
    void (*vec_mul_f32)(const float *, const float *, float *, uint32_t, uint32_t);
    void (*fwht_f32)(const float *, uint32_t, const uint8_t *, uint32_t, float *, uint32_t,
                     uint32_t, uint32_t, uint32_t, uint32_t);
    void (*quantize_row_f32_i8)(const float *, uint32_t, int8_t *, uint32_t, float *, uint32_t,
                                uint32_t, uint32_t);
} slang_cpu_vtable;

#define DEFINE_SLANG_VTABLE(ISA)                                           \
    static const slang_cpu_vtable vt_##ISA = {                             \
        .linear_f32 = velri_slang_##ISA##_linear_f32,                      \
        .linear_f16_f32 = velri_slang_##ISA##_linear_f16_f32,              \
        .linear_bf16_f32 = velri_slang_##ISA##_linear_bf16_f32,            \
        .linear_f32x_bf16w = velri_slang_##ISA##_linear_f32x_bf16w,        \
        .linear_f64 = velri_slang_##ISA##_linear_f64,                      \
        .linear_w8a8_f32 = velri_slang_##ISA##_linear_w8a8_f32,            \
        .rmsnorm_f32 = velri_slang_##ISA##_rmsnorm_f32,                    \
        .rmsnorm_bf16 = velri_slang_##ISA##_rmsnorm_bf16,                  \
        .rmsnorm_f16 = velri_slang_##ISA##_rmsnorm_f16,                    \
        .fused_add_rmsnorm_f32 = velri_slang_##ISA##_fused_add_rmsnorm_f32,\
        .layernorm_f32 = velri_slang_##ISA##_layernorm_f32,                \
        .rope_f32 = velri_slang_##ISA##_rope_f32,                          \
        .rope_bf16 = velri_slang_##ISA##_rope_bf16,                        \
        .activation_silu_f32 = velri_slang_##ISA##_activation_silu_f32,    \
        .activation_gelu_f32 = velri_slang_##ISA##_activation_gelu_f32,    \
        .activation_relu2_f32 = velri_slang_##ISA##_activation_relu2_f32,  \
        .gated_silu_f32 = velri_slang_##ISA##_gated_silu_f32,              \
        .gated_gelu_f32 = velri_slang_##ISA##_gated_gelu_f32,              \
        .softmax_f32 = velri_slang_##ISA##_softmax_f32,                    \
        .vec_axpby_f32 = velri_slang_##ISA##_vec_axpby_f32,                \
        .vec_mul_f32 = velri_slang_##ISA##_vec_mul_f32,                    \
        .fwht_f32 = velri_slang_##ISA##_fwht_f32,                          \
        .quantize_row_f32_i8 = velri_slang_##ISA##_quantize_row_f32_i8,    \
    };

DEFINE_SLANG_VTABLE(base)
#if VELRI_X86_VARIANTS
DEFINE_SLANG_VTABLE(avx2)
DEFINE_SLANG_VTABLE(avx512)
#endif

static const slang_cpu_vtable *resolve_slang_vt(velri_cpu_variant variant) {
    if (!velri_cpu_variant_supported(variant)) return nullptr;
    velri_cpu_variant r = velri_cpu_resolve_variant(variant);
#if VELRI_X86_VARIANTS
    if (r == VELRI_CPU_AVX512) return &vt_avx512;
    if (r == VELRI_CPU_AVX2) return &vt_avx2;
#endif
    if (r == VELRI_CPU_BASE) return &vt_base;
    return nullptr;
}

typedef struct {
    const slang_cpu_vtable *vt;
    const velri_linear_desc *desc;
    uint32_t rows, m, k, xs, ws, ys, flags;
    const void *x, *w, *bias;
    const float *x_scale, *w_scale;
    void *y;
    int mode; /* 0=f32, 1=f16_f32, 2=bf16_f32, 3=f32x_bf16w, 4=f64, 5=w8a8_f32 */
} slang_lin_ctx;

static void slang_lin_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc;
    (void)scb;
    const slang_lin_ctx *c = (const slang_lin_ctx *)vctx;
    uint32_t r0 = 0u, r1 = c->rows;
    uint32_t c0 = 0u, c1 = c->m;
    if (c->rows >= 4u && c->m < nw * 16u && c->rows >= nw * 6u) {
        /* Narrow M, tall rows: partition rows in multiples of 6 (MR). */
        size_t rblocks = (c->rows + 5u) / 6u;
        r0 = (uint32_t)(((rblocks * wid) / nw) * 6u);
        r1 = (uint32_t)(((rblocks * (wid + 1u)) / nw) * 6u);
        if (r1 > c->rows) r1 = c->rows;
        if (r0 >= r1) return;
    } else {
        /* Partition columns in multiples of NR (64 for F32/32 for F64) when wide enough, else 4. */
        size_t align = 4u;
        if (c->rows >= 4u) {
            size_t want = (c->mode == 4) ? 32u : 64u;
            if (c->m >= nw * want) align = want;
            else if (c->m >= nw * 16u) align = 16u;
        }
        size_t col_blocks = (c->m + align - 1u) / align;
        size_t b0 = (col_blocks * wid) / nw;
        size_t b1 = (col_blocks * (wid + 1u)) / nw;
        c0 = (uint32_t)(b0 * align);
        c1 = (uint32_t)(b1 * align);
        if (c1 > c->m) c1 = c->m;
        if (c0 >= c1) return;
    }

    switch (c->mode) {
    case 0:
        c->vt->linear_f32((const float *)c->x, c->xs, (const float *)c->w, c->ws,
                          (const float *)c->bias, c->flags, (float *)c->y, c->ys,
                          r0, r1, c0, c1, c->k);
        break;
    case 1:
        c->vt->linear_f16_f32((const uint16_t *)c->x, c->xs, (const uint16_t *)c->w, c->ws,
                              (const float *)c->bias, c->flags, (float *)c->y, c->ys,
                              r0, r1, c0, c1, c->k);
        break;
    case 2:
        c->vt->linear_bf16_f32((const uint16_t *)c->x, c->xs, (const uint16_t *)c->w, c->ws,
                               (const float *)c->bias, c->flags, (float *)c->y, c->ys,
                               r0, r1, c0, c1, c->k);
        break;
    case 3:
        c->vt->linear_f32x_bf16w((const float *)c->x, c->xs, (const uint16_t *)c->w, c->ws,
                                 (const float *)c->bias, c->flags, (float *)c->y, c->ys,
                                 r0, r1, c0, c1, c->k);
        break;
    case 4:
        c->vt->linear_f64((const double *)c->x, c->xs, (const double *)c->w, c->ws,
                          (const double *)c->bias, c->bias ? 1u : 0u, (double *)c->y, c->ys,
                          r0, r1, c0, c1, c->k);
        break;
    case 5:
        c->vt->linear_w8a8_f32((const int8_t *)c->x, c->xs, (const int8_t *)c->w, c->ws,
                               (const float *)c->bias, c->x_scale, c->w_scale, c->flags,
                               (float *)c->y, c->ys, r0, r1, c0, c1, c->k);
        break;
    }
}

velri_status velri_slang_cpu_linear(
    const velri_linear_desc *desc,
    size_t rows, size_t m, size_t k,
    const void *x, size_t x_stride,
    const void *w, size_t w_stride,
    const void *bias, const float *x_scale, const float *w_scale,
    void *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!desc || !x || !w || !y || m == 0 || k == 0) return VELRI_ERR_INVALID;
    if (rows == 0) return VELRI_OK;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;

    int mode = -1;
    if (desc->x_dtype == VELRI_DTYPE_F32 && desc->w_dtype == VELRI_DTYPE_F32 &&
        desc->y_dtype == VELRI_DTYPE_F32) mode = 0;
    else if (desc->x_dtype == VELRI_DTYPE_F16 && desc->w_dtype == VELRI_DTYPE_F16 &&
             desc->y_dtype == VELRI_DTYPE_F32) mode = 1;
    else if (desc->x_dtype == VELRI_DTYPE_BF16 && desc->w_dtype == VELRI_DTYPE_BF16 &&
             desc->y_dtype == VELRI_DTYPE_F32) mode = 2;
    else if (desc->x_dtype == VELRI_DTYPE_F32 && desc->w_dtype == VELRI_DTYPE_BF16 &&
             desc->y_dtype == VELRI_DTYPE_F32) mode = 3;
    else if (desc->x_dtype == VELRI_DTYPE_F64 && desc->w_dtype == VELRI_DTYPE_F64 &&
             desc->y_dtype == VELRI_DTYPE_F64) mode = 4;
    else if (desc->x_dtype == VELRI_DTYPE_I8 && desc->w_dtype == VELRI_DTYPE_I8 &&
             desc->y_dtype == VELRI_DTYPE_F32 && desc->has_x_scale && desc->has_w_scale) mode = 5;
    if (mode < 0) return VELRI_ERR_UNSUPPORTED;

    uint32_t flags = (bias ? 1u : 0u) | (x_scale ? 2u : 0u) | (w_scale ? 4u : 0u);
    slang_lin_ctx ctx = {
        .vt = vt, .desc = desc,
        .rows = (uint32_t)rows, .m = (uint32_t)m, .k = (uint32_t)k,
        .xs = (uint32_t)x_stride, .ws = (uint32_t)w_stride, .ys = (uint32_t)y_stride,
        .flags = flags, .x = x, .w = w, .bias = bias,
        .x_scale = x_scale, .w_scale = w_scale, .y = y, .mode = mode,
    };
    if (pool && active_workers > 1) {
        velri_pool_parallel(pool, active_workers, slang_lin_worker, &ctx);
    } else {
        slang_lin_worker(&ctx, 0, 1, nullptr, 0);
    }
    return VELRI_OK;
}

typedef struct {
    const slang_cpu_vtable *vt;
    velri_dtype dt;
    uint32_t rows, dim, xs, ys, has_w, unit_offset;
    float eps;
    const void *x, *w;
    void *y;
} slang_norm_ctx;

static void slang_norm_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_norm_ctx *c = (const slang_norm_ctx *)vctx;
    uint32_t r0 = (uint32_t)((c->rows * wid) / nw);
    uint32_t r1 = (uint32_t)((c->rows * (wid + 1u)) / nw);
    if (r0 >= r1) return;
    if (c->dt == VELRI_DTYPE_F32)
        c->vt->rmsnorm_f32((const float *)c->x, c->xs, (const float *)c->w, c->has_w,
                           (float *)c->y, c->ys, r0, r1, c->dim, c->eps, c->unit_offset);
    else if (c->dt == VELRI_DTYPE_BF16)
        c->vt->rmsnorm_bf16((const uint16_t *)c->x, c->xs, (const uint16_t *)c->w, c->has_w,
                            (uint16_t *)c->y, c->ys, r0, r1, c->dim, c->eps, c->unit_offset);
    else
        c->vt->rmsnorm_f16((const uint16_t *)c->x, c->xs, (const uint16_t *)c->w, c->has_w,
                           (uint16_t *)c->y, c->ys, r0, r1, c->dim, c->eps, c->unit_offset);
}

velri_status velri_slang_cpu_rmsnorm(
    velri_dtype io_dtype, size_t rows, size_t dim, float eps, bool unit_offset,
    const void *x, size_t x_stride, const void *weight, void *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!x || !y || dim == 0) return VELRI_ERR_INVALID;
    if (io_dtype != VELRI_DTYPE_F32 && io_dtype != VELRI_DTYPE_BF16 && io_dtype != VELRI_DTYPE_F16)
        return VELRI_ERR_UNSUPPORTED;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_norm_ctx ctx = {
        .vt = vt, .dt = io_dtype, .rows = (uint32_t)rows, .dim = (uint32_t)dim,
        .xs = (uint32_t)x_stride, .ys = (uint32_t)y_stride,
        .has_w = weight ? 1u : 0u, .unit_offset = unit_offset ? 1u : 0u,
        .eps = eps, .x = x, .w = weight, .y = y,
    };
    if (pool && active_workers > 1 && rows > 1)
        velri_pool_parallel(pool, active_workers, slang_norm_worker, &ctx);
    else
        slang_norm_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

typedef struct {
    const slang_cpu_vtable *vt;
    velri_dtype dt;
    uint32_t n_tokens, n_heads, head_dim, rotary_dim, pos_offset, tsx, hsx, tsy, hsy, mode;
    const void *x;
    const float *cos_t, *sin_t;
    void *y;
} slang_rope_ctx;

static void slang_rope_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_rope_ctx *c = (const slang_rope_ctx *)vctx;
    uint32_t t0 = (uint32_t)((c->n_tokens * wid) / nw);
    uint32_t t1 = (uint32_t)((c->n_tokens * (wid + 1u)) / nw);
    if (t0 >= t1) return;
    if (c->dt == VELRI_DTYPE_F32)
        c->vt->rope_f32((const float *)c->x, c->tsx, c->hsx, c->cos_t, c->sin_t,
                        (float *)c->y, c->tsy, c->hsy, t0, t1, c->n_heads,
                        c->head_dim, c->rotary_dim, c->pos_offset, c->mode);
    else
        c->vt->rope_bf16((const uint16_t *)c->x, c->tsx, c->hsx, c->cos_t, c->sin_t,
                         (uint16_t *)c->y, c->tsy, c->hsy, t0, t1, c->n_heads,
                         c->head_dim, c->rotary_dim, c->pos_offset, c->mode);
}

velri_status velri_slang_cpu_rope(
    velri_dtype dtype, size_t n_tokens, size_t n_heads, size_t head_dim,
    size_t rotary_dim, size_t pos_offset,
    const void *x, size_t tok_sx, size_t head_sx,
    const float *cos_tbl, const float *sin_tbl,
    void *y, size_t tok_sy, size_t head_sy,
    velri_rope_mode mode, velri_cpu_variant variant,
    velri_pool *pool, size_t active_workers)
{
    if (dtype != VELRI_DTYPE_F32 && dtype != VELRI_DTYPE_BF16) return VELRI_ERR_UNSUPPORTED;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_rope_ctx ctx = {
        .vt = vt, .dt = dtype,
        .n_tokens = (uint32_t)n_tokens, .n_heads = (uint32_t)n_heads,
        .head_dim = (uint32_t)head_dim, .rotary_dim = (uint32_t)rotary_dim,
        .pos_offset = (uint32_t)pos_offset,
        .tsx = (uint32_t)tok_sx, .hsx = (uint32_t)head_sx,
        .tsy = (uint32_t)tok_sy, .hsy = (uint32_t)head_sy,
        .mode = (uint32_t)mode, .x = x, .cos_t = cos_tbl, .sin_t = sin_tbl, .y = y,
    };
    if (pool && active_workers > 1 && n_tokens > 1)
        velri_pool_parallel(pool, active_workers, slang_rope_worker, &ctx);
    else
        slang_rope_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

typedef struct {
    const slang_cpu_vtable *vt;
    velri_act_kind act;
    uint32_t count;
    float clamp_limit;
    const float *gate, *up;
    float *y;
} slang_gated_ctx;

static void slang_gated_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_gated_ctx *c = (const slang_gated_ctx *)vctx;
    uint32_t i0 = (uint32_t)((c->count * wid) / nw);
    uint32_t i1 = (uint32_t)((c->count * (wid + 1u)) / nw);
    if (i0 >= i1) return;
    if (c->act == VELRI_ACT_SILU)
        c->vt->gated_silu_f32(c->gate, c->up, c->y, i0, i1, c->clamp_limit);
    else
        c->vt->gated_gelu_f32(c->gate, c->up, c->y, i0, i1, c->clamp_limit);
}

velri_status velri_slang_cpu_gated_mlp_f32(
    velri_act_kind gate_act, size_t count, float clamp_limit,
    const float *gate, const float *up, float *y,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (gate_act != VELRI_ACT_SILU && gate_act != VELRI_ACT_GELU_TANH) return VELRI_ERR_UNSUPPORTED;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_gated_ctx ctx = {
        .vt = vt, .act = gate_act, .count = (uint32_t)count, .clamp_limit = clamp_limit,
        .gate = gate, .up = up, .y = y,
    };
    if (pool && active_workers > 1)
        velri_pool_parallel(pool, active_workers, slang_gated_worker, &ctx);
    else
        slang_gated_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

typedef struct {
    const slang_cpu_vtable *vt;
    uint32_t rows, cols, xs, ys;
    const float *x;
    float *y;
} slang_sm_ctx;

static void slang_sm_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_sm_ctx *c = (const slang_sm_ctx *)vctx;
    uint32_t r0 = (uint32_t)((c->rows * wid) / nw);
    uint32_t r1 = (uint32_t)((c->rows * (wid + 1u)) / nw);
    if (r0 >= r1) return;
    c->vt->softmax_f32(c->x, c->xs, c->y, c->ys, r0, r1, c->cols);
}

velri_status velri_slang_cpu_softmax_f32(
    size_t rows, size_t cols, const float *x, size_t x_stride,
    float *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_sm_ctx ctx = {
        .vt = vt, .rows = (uint32_t)rows, .cols = (uint32_t)cols,
        .xs = (uint32_t)x_stride, .ys = (uint32_t)y_stride, .x = x, .y = y,
    };
    if (pool && active_workers > 1 && rows > 1)
        velri_pool_parallel(pool, active_workers, slang_sm_worker, &ctx);
    else
        slang_sm_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

/* ---- FusedAddRMSNorm ---- */
typedef struct {
    const slang_cpu_vtable *vt;
    uint32_t rows, dim, rs, xs, ys, has_w, unit_offset;
    float eps;
    float *res;
    const float *x, *w;
    float *y;
} slang_far_ctx;

static void slang_far_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_far_ctx *c = (const slang_far_ctx *)vctx;
    uint32_t r0 = (uint32_t)((c->rows * wid) / nw);
    uint32_t r1 = (uint32_t)((c->rows * (wid + 1u)) / nw);
    if (r0 >= r1) return;
    c->vt->fused_add_rmsnorm_f32(c->res, c->rs, c->x, c->xs, c->w, c->has_w,
                                 c->y, c->ys, r0, r1, c->dim, c->eps, c->unit_offset);
}

velri_status velri_slang_cpu_fused_add_rmsnorm_f32(
    size_t rows, size_t dim, float eps, bool unit_offset,
    float *residual_inout, size_t r_stride, const float *x, size_t x_stride,
    const float *weight, float *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!residual_inout || !x || !y || dim == 0) return VELRI_ERR_INVALID;
    if (rows == 0) return VELRI_OK;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_far_ctx ctx = {
        .vt = vt, .rows = (uint32_t)rows, .dim = (uint32_t)dim,
        .rs = (uint32_t)r_stride, .xs = (uint32_t)x_stride, .ys = (uint32_t)y_stride,
        .has_w = weight ? 1u : 0u, .unit_offset = unit_offset ? 1u : 0u,
        .eps = eps, .res = residual_inout, .x = x, .w = weight, .y = y,
    };
    if (pool && active_workers > 1 && rows > 1)
        velri_pool_parallel(pool, active_workers, slang_far_worker, &ctx);
    else
        slang_far_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

/* ---- LayerNorm ---- */
typedef struct {
    const slang_cpu_vtable *vt;
    uint32_t rows, dim, xs, ys, has_w, has_b;
    float eps;
    const float *x, *w, *b;
    float *y;
} slang_ln_ctx;

static void slang_ln_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_ln_ctx *c = (const slang_ln_ctx *)vctx;
    uint32_t r0 = (uint32_t)((c->rows * wid) / nw);
    uint32_t r1 = (uint32_t)((c->rows * (wid + 1u)) / nw);
    if (r0 >= r1) return;
    c->vt->layernorm_f32(c->x, c->xs, c->w, c->has_w, c->b, c->has_b,
                         c->y, c->ys, r0, r1, c->dim, c->eps);
}

velri_status velri_slang_cpu_layernorm_f32(
    size_t rows, size_t dim, float eps,
    const float *x, size_t x_stride, const float *weight, const float *bias,
    float *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!x || !y || dim == 0) return VELRI_ERR_INVALID;
    if (rows == 0) return VELRI_OK;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_ln_ctx ctx = {
        .vt = vt, .rows = (uint32_t)rows, .dim = (uint32_t)dim,
        .xs = (uint32_t)x_stride, .ys = (uint32_t)y_stride,
        .has_w = weight ? 1u : 0u, .has_b = bias ? 1u : 0u,
        .eps = eps, .x = x, .w = weight, .b = bias, .y = y,
    };
    if (pool && active_workers > 1 && rows > 1)
        velri_pool_parallel(pool, active_workers, slang_ln_worker, &ctx);
    else
        slang_ln_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

/* ---- Standalone Activations ---- */
typedef struct {
    const slang_cpu_vtable *vt;
    velri_act_kind kind;
    uint32_t count;
    const float *x;
    float *y;
} slang_act_ctx;

static void slang_act_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_act_ctx *c = (const slang_act_ctx *)vctx;
    uint32_t i0 = (uint32_t)((c->count * wid) / nw);
    uint32_t i1 = (uint32_t)((c->count * (wid + 1u)) / nw);
    if (i0 >= i1) return;
    if (c->kind == VELRI_ACT_SILU) c->vt->activation_silu_f32(c->x, c->y, i0, i1);
    else if (c->kind == VELRI_ACT_GELU_TANH) c->vt->activation_gelu_f32(c->x, c->y, i0, i1);
    else c->vt->activation_relu2_f32(c->x, c->y, i0, i1);
}

velri_status velri_slang_cpu_activation_f32(
    velri_act_kind kind, size_t count, const float *x, float *y,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!x || !y) return VELRI_ERR_INVALID;
    if (count == 0) return VELRI_OK;
    if (kind != VELRI_ACT_SILU && kind != VELRI_ACT_GELU_TANH && kind != VELRI_ACT_RELU2)
        return VELRI_ERR_UNSUPPORTED;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_act_ctx ctx = { .vt = vt, .kind = kind, .count = (uint32_t)count, .x = x, .y = y };
    if (pool && active_workers > 1)
        velri_pool_parallel(pool, active_workers, slang_act_worker, &ctx);
    else
        slang_act_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

/* ---- vec_axpby & vec_mul ---- */
typedef struct {
    const slang_cpu_vtable *vt;
    uint32_t count;
    float alpha, beta;
    const float *a, *b;
    float *y;
    int is_mul;
} slang_vec_ctx;

static void slang_vec_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_vec_ctx *c = (const slang_vec_ctx *)vctx;
    uint32_t i0 = (uint32_t)((c->count * wid) / nw);
    uint32_t i1 = (uint32_t)((c->count * (wid + 1u)) / nw);
    if (i0 >= i1) return;
    if (c->is_mul) c->vt->vec_mul_f32(c->a, c->b, c->y, i0, i1);
    else c->vt->vec_axpby_f32(c->alpha, c->a, c->beta, c->b, c->y, i0, i1);
}

velri_status velri_slang_cpu_vec_axpby_f32(
    size_t count, float alpha, const float *a, float beta, const float *b, float *y,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!a || !y || (beta != 0.0f && !b)) return VELRI_ERR_INVALID;
    if (count == 0) return VELRI_OK;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_vec_ctx ctx = {
        .vt = vt, .count = (uint32_t)count, .alpha = alpha, .beta = beta,
        .a = a, .b = b ? b : a, .y = y, .is_mul = 0,
    };
    if (pool && active_workers > 1)
        velri_pool_parallel(pool, active_workers, slang_vec_worker, &ctx);
    else
        slang_vec_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

velri_status velri_slang_cpu_vec_mul_f32(
    size_t count, const float *a, const float *b, float *y,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!a || !b || !y) return VELRI_ERR_INVALID;
    if (count == 0) return VELRI_OK;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_vec_ctx ctx = {
        .vt = vt, .count = (uint32_t)count, .a = a, .b = b, .y = y, .is_mul = 1,
    };
    if (pool && active_workers > 1)
        velri_pool_parallel(pool, active_workers, slang_vec_worker, &ctx);
    else
        slang_vec_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

/* ---- FWHT ---- */
typedef struct {
    const slang_cpu_vtable *vt;
    uint32_t rows, cols, block_size, xs, ys, suh_mode;
    const float *x;
    const uint8_t *suh;
    float *y;
} slang_fwht_ctx;

static void slang_fwht_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_fwht_ctx *c = (const slang_fwht_ctx *)vctx;
    uint32_t r0 = (uint32_t)((c->rows * wid) / nw);
    uint32_t r1 = (uint32_t)((c->rows * (wid + 1u)) / nw);
    if (r0 >= r1) return;
    c->vt->fwht_f32(c->x, c->xs, c->suh, c->suh_mode, c->y, c->ys, r0, r1, c->cols, c->block_size);
}

velri_status velri_slang_cpu_fwht_f32(
    size_t rows, size_t cols, size_t block_size,
    const float *x, size_t x_stride, velri_dtype suh_dtype, const void *suh,
    float *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!x || !y || cols == 0) return VELRI_ERR_INVALID;
    if (block_size != 16u && block_size != 32u && block_size != 64u &&
        block_size != 128u && block_size != 256u) return VELRI_ERR_INVALID;
    if ((cols % block_size) != 0u) return VELRI_ERR_INVALID;
    if (rows == 0) return VELRI_OK;
    uint32_t suh_mode = 0u;
    if (suh) {
        if (suh_dtype == VELRI_DTYPE_F32) suh_mode = 1u;
        else if (suh_dtype == VELRI_DTYPE_F16) suh_mode = 2u;
        else return VELRI_ERR_UNSUPPORTED;
    }
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_fwht_ctx ctx = {
        .vt = vt, .rows = (uint32_t)rows, .cols = (uint32_t)cols,
        .block_size = (uint32_t)block_size, .xs = (uint32_t)x_stride, .ys = (uint32_t)y_stride,
        .suh_mode = suh_mode, .x = x, .suh = suh ? (const uint8_t *)suh : (const uint8_t *)x, .y = y,
    };
    if (pool && active_workers > 1 && rows > 1)
        velri_pool_parallel(pool, active_workers, slang_fwht_worker, &ctx);
    else
        slang_fwht_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}

/* ---- Row Quantization F32 -> I8 ---- */
typedef struct {
    const slang_cpu_vtable *vt;
    uint32_t rows, k, xs, qs;
    const float *x;
    int8_t *xi8;
    float *scale_out;
} slang_qrow_ctx;

static void slang_qrow_worker(void *vctx, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    const slang_qrow_ctx *c = (const slang_qrow_ctx *)vctx;
    uint32_t r0 = (uint32_t)((c->rows * wid) / nw);
    uint32_t r1 = (uint32_t)((c->rows * (wid + 1u)) / nw);
    if (r0 >= r1) return;
    c->vt->quantize_row_f32_i8(c->x, c->xs, c->xi8, c->qs, c->scale_out, r0, r1, c->k);
}

velri_status velri_slang_cpu_quantize_row_f32_i8(
    size_t rows, size_t k, const float *x, size_t x_stride,
    int8_t *x_i8, size_t xi8_stride, float *scale_out,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers)
{
    if (!x || !x_i8 || !scale_out || k == 0) return VELRI_ERR_INVALID;
    if (rows == 0) return VELRI_OK;
    const slang_cpu_vtable *vt = resolve_slang_vt(variant);
    if (!vt) return VELRI_ERR_UNSUPPORTED;
    slang_qrow_ctx ctx = {
        .vt = vt, .rows = (uint32_t)rows, .k = (uint32_t)k,
        .xs = (uint32_t)x_stride, .qs = (uint32_t)xi8_stride,
        .x = x, .xi8 = x_i8, .scale_out = scale_out,
    };
    if (pool && active_workers > 1 && rows > 1)
        velri_pool_parallel(pool, active_workers, slang_qrow_worker, &ctx);
    else
        slang_qrow_worker(&ctx, 0, 1, nullptr, 0);
    return VELRI_OK;
}
