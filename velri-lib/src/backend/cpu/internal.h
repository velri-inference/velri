#ifndef VELRI_CPU_INTERNAL_H
#define VELRI_CPU_INTERNAL_H

#include "velri_lib/types.h"
#include "velri_lib/pool.h"
#include "velri_lib/linear_typed.h"
#include "velri_lib/ops.h"
#include <math.h>
#include <float.h>
#include <string.h>

/* A float reciprocal can overflow for tiny, finite rows. Keep the exceptional
 * path outside the vector loop, and never publish a zero quantization scale. */
static inline void velri_quantize_tiny_row(const float *x, int8_t *q, size_t n, float amax,
                                           float *scale) {
    *scale = fmaxf(amax / 127.0f, FLT_TRUE_MIN);
    double inv = 1.0 / (double)*scale;
    for (size_t i = 0; i < n; ++i) {
        double v = rint((double)x[i] * inv);
        q[i] = (int8_t)fmax(-127.0, fmin(127.0, v));
    }
}

velri_status velri_linear_validate_run(const velri_linear_typed_plan *plan,
                                       const velri_linear_run_args *args, bool check_scratch);

/* ============================================================================
 * 1. Overflow-Safe Span & Aliasing Helpers (matching Velri-lib linear.c)
 * ============================================================================ */

static inline bool velri_mul_size(size_t a, size_t b, size_t *out) {
    if (b != 0 && a > SIZE_MAX / b) {
        return false;
    }
    *out = a * b;
    return true;
}

static inline bool velri_add_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static inline bool velri_overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    if (a_bytes == 0 || b_bytes == 0) {
        return false;
    }
    uintptr_t ua = (uintptr_t)a;
    uintptr_t ub = (uintptr_t)b;
    return ua < ub + b_bytes && ub < ua + a_bytes;
}

/* Defined modulo-2^32 integer arithmetic, matching SIMD add/dot instructions.
 * memcpy gives the signed interpretation without an out-of-range signed cast. */
static inline int32_t velri_i32_bits(uint32_t bits) {
    int32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
static inline int32_t velri_i32_add(int32_t a, int32_t b) {
    return velri_i32_bits((uint32_t)a + (uint32_t)b);
}
static inline int32_t velri_i32_mul(int32_t a, int32_t b) {
    return velri_i32_bits((uint32_t)a * (uint32_t)b);
}

/*
 * Validate a 2D view `[rows, cols]` with `stride` (in elements) inside `capacity` elements,
 * checking element alignment and pointer arithmetic overflow.
 * Span bytes cover inter-row gaps `(rows - 1) * stride + cols` so outputs never alias input gaps.
 */
static inline velri_status velri_check_span(const void *ptr, size_t capacity_elems, size_t rows,
                                            size_t cols, size_t stride_elems, size_t elem_bytes,
                                            size_t elem_align, size_t *out_bytes) {
    if (stride_elems < cols) {
        return VELRI_ERR_INVALID;
    }
    size_t used_elems = 0;
    if (rows > 0 && cols > 0) {
        size_t gap = 0;
        if (!velri_mul_size(rows - 1, stride_elems, &gap) ||
            !velri_add_size(gap, cols, &used_elems)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    size_t bytes = 0;
    if (!velri_mul_size(used_elems, elem_bytes, &bytes)) {
        return VELRI_ERR_OVERFLOW;
    }
    if (capacity_elems < used_elems) {
        return VELRI_ERR_INVALID;
    }
    if (used_elems > 0) {
        if (ptr == nullptr) {
            return VELRI_ERR_INVALID;
        }
        if (((uintptr_t)ptr % elem_align) != 0) {
            return VELRI_ERR_INVALID;
        }
        if ((uintptr_t)ptr > UINTPTR_MAX - bytes) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    *out_bytes = bytes;
    return VELRI_OK;
}

/* ============================================================================
 * 2. Portable Bit-Exact IEEE-754 Scalar FP16 & BF16 Primitives
 * ============================================================================ */

static inline uint32_t velri_f32_to_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static inline float velri_bits_to_f32(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static inline float velri_bf16_to_f32_scalar(velri_bf16 h) {
    uint32_t u = (uint32_t)h << 16;
    return velri_bits_to_f32(u);
}

static inline velri_bf16 velri_f32_to_bf16_scalar(float x) {
    uint32_t u = velri_f32_to_bits(x);
    uint32_t abs_u = u & UINT32_C(0x7FFFFFFF);
    if (abs_u > UINT32_C(0x7F800000)) {
        /* Canonicalize signaling/quiet NaN to quiet BF16 NaN */
        return (velri_bf16)((u >> 16) | UINT16_C(0x0040));
    }
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t rounded = u + UINT32_C(0x7FFF) + lsb;
    return (velri_bf16)(rounded >> 16);
}

static inline float velri_f16_to_f32_scalar(velri_fp16 h) {
    uint32_t sign = (uint32_t)(h & UINT16_C(0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    if (exp == 0) {
        if (mant == 0) {
            return velri_bits_to_f32(sign);
        }
        /* Subnormal binary16 -> normalized FP32 */
        uint32_t e = 127U - 14U;
        while ((mant & 0x0400U) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03FFU;
        return velri_bits_to_f32(sign | (e << 23) | (mant << 13));
    }
    if (exp == 0x1FU) {
        if (mant == 0) {
            return velri_bits_to_f32(sign | UINT32_C(0x7F800000));
        }
        return velri_bits_to_f32(sign | UINT32_C(0x7FC00000) | (mant << 13));
    }
    return velri_bits_to_f32(sign | ((exp + (127U - 15U)) << 23) | (mant << 13));
}

static inline velri_fp16 velri_f32_to_f16_scalar(float x) {
    uint32_t u = velri_f32_to_bits(x);
    uint16_t sign = (uint16_t)((u >> 16) & UINT16_C(0x8000));
    uint32_t abs_u = u & UINT32_C(0x7FFFFFFF);

    if (abs_u > UINT32_C(0x7F800000)) {
        /* Quiet NaN preserving payload bits where possible */
        uint16_t payload = (uint16_t)((abs_u >> 13) & 0x03FFU);
        return (velri_fp16)(sign | UINT16_C(0x7E00) | payload);
    }
    if (abs_u >= UINT32_C(0x477FF000)) {
        /* Overflow or Inf: check exact threshold 65504 + 16 = 65520 (0x477FE000 + tie) */
        if (abs_u >= UINT32_C(0x7F800000)) {
            return (velri_fp16)(sign | UINT16_C(0x7C00));
        }
        uint32_t exp = (abs_u >> 23) - 127U + 15U;
        uint32_t mant = abs_u & UINT32_C(0x007FFFFF);
        if (exp > 30U) {
            return (velri_fp16)(sign | UINT16_C(0x7C00));
        }
        uint32_t lsb = (mant >> 13) & 1U;
        uint32_t r = mant + UINT32_C(0x0FFF) + lsb;
        if (r >= UINT32_C(0x00800000)) {
            return (velri_fp16)(sign | UINT16_C(0x7C00));
        }
        return (velri_fp16)(sign | (uint16_t)(exp << 10) | (uint16_t)(r >> 13));
    }
    if (abs_u < UINT32_C(0x33000000)) {
        /* |x| < 2^-25: rounds to +-0 */
        return sign;
    }
    if (abs_u < UINT32_C(0x38800000)) {
        /* Subnormal binary16 range [2^-25, 2^-14) with exact RNE */
        uint32_t f_exp = abs_u >> 23;
        uint32_t sig = (abs_u & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
        uint32_t shift = 126U - f_exp; /* in [14, 25] */
        uint32_t q = sig >> shift;
        uint32_t rem = sig & ((UINT32_C(1) << shift) - 1U);
        uint32_t half = UINT32_C(1) << (shift - 1U);
        if (rem > half || (rem == half && (q & 1U) != 0U)) {
            ++q;
        }
        return (velri_fp16)(sign | (uint16_t)q);
    }
    /* Normal binary16 range */
    uint32_t exp = (abs_u >> 23) - (127U - 15U);
    uint32_t mant = abs_u & UINT32_C(0x007FFFFF);
    uint32_t lsb = (mant >> 13) & 1U;
    uint32_t rounded = mant + UINT32_C(0x0FFF) + lsb;
    if (rounded >= UINT32_C(0x00800000)) {
        rounded -= UINT32_C(0x00800000);
        ++exp;
        if (exp >= 31U) {
            return (velri_fp16)(sign | UINT16_C(0x7C00));
        }
    }
    return (velri_fp16)(sign | (uint16_t)(exp << 10) | (uint16_t)(rounded >> 13));
}

/* ============================================================================
 * 3. Internal Conversion & Quantization Backends
 * ============================================================================ */

void velri_cvt_f32_to_f16_base(const float *src, velri_fp16 *restrict dst, size_t n);
void velri_cvt_f16_to_f32_base(const velri_fp16 *src, float *restrict dst, size_t n);
void velri_cvt_f32_to_bf16_base(const float *src, velri_bf16 *restrict dst, size_t n);
void velri_cvt_bf16_to_f32_base(const velri_bf16 *src, float *restrict dst, size_t n);
void velri_quantize_row_f32_i8_base(size_t rows, size_t k, const float *x, size_t x_stride,
                                    int8_t *restrict x_i8, size_t xi8_stride,
                                    float *restrict scale_out);

#if VELRI_X86_VARIANTS
void velri_cvt_f32_to_f16_avx2(const float *src, velri_fp16 *restrict dst, size_t n);
void velri_cvt_f16_to_f32_avx2(const velri_fp16 *src, float *restrict dst, size_t n);
void velri_cvt_f32_to_bf16_avx2(const float *src, velri_bf16 *restrict dst, size_t n);
void velri_cvt_bf16_to_f32_avx2(const velri_bf16 *src, float *restrict dst, size_t n);
void velri_quantize_row_f32_i8_avx2(size_t rows, size_t k, const float *x, size_t x_stride,
                                    int8_t *restrict x_i8, size_t xi8_stride,
                                    float *restrict scale_out);

void velri_cvt_f32_to_f16_avx512(const float *src, velri_fp16 *restrict dst, size_t n);
void velri_cvt_f16_to_f32_avx512(const velri_fp16 *src, float *restrict dst, size_t n);
void velri_cvt_f32_to_bf16_avx512(const float *src, velri_bf16 *restrict dst, size_t n);
void velri_cvt_bf16_to_f32_avx512(const velri_bf16 *src, float *restrict dst, size_t n);
void velri_quantize_row_f32_i8_avx512(size_t rows, size_t k, const float *x, size_t x_stride,
                                      int8_t *restrict x_i8, size_t xi8_stride,
                                      float *restrict scale_out);
#endif

/* ============================================================================
 * 4. Internal Linear Kernel Mode & Backend Entry Points
 * ============================================================================ */

typedef enum {
    VELRI_LIN_F32 = 0,        /* F32  * F32  -> F32 acc -> F32  */
    VELRI_LIN_F64 = 1,        /* F64  * F64  -> F64 acc -> F64  */
    VELRI_LIN_F16_F32 = 2,    /* F16  * F16  -> F32 acc -> F32  */
    VELRI_LIN_F16_F16 = 3,    /* F16  * F16  -> F32 acc -> F16  */
    VELRI_LIN_F32X_F16W = 4,  /* F32  * F16  -> F32 acc -> F32  */
    VELRI_LIN_BF16_F32 = 5,   /* BF16 * BF16 -> F32 acc -> F32  */
    VELRI_LIN_BF16_BF16 = 6,  /* BF16 * BF16 -> F32 acc -> BF16 */
    VELRI_LIN_F32X_BF16W = 7, /* F32  * BF16 -> F32 acc -> F32  */
    VELRI_LIN_I8_I32 = 8,     /* I8   * I8   -> I32 acc -> I32  */
    VELRI_LIN_W8A8_F32 = 9,   /* I8   * I8   -> I32 acc -> F32  (x_scale * w_scale + bias) */
    VELRI_LIN_F32X_I8W = 10   /* F32  * I8   -> F32 acc -> F32  (dot * w_scale + bias) */
} velri_lin_kind;

/* A kernel owns its workspace and suggested job sizes. The scheduler may split
 * jobs further; these are performance hints, not constraints on public shapes. */
typedef struct {
    size_t scratch_bytes;
    size_t row_grain;
    size_t coarse_rows, coarse_tiles;
    size_t fine_rows, fine_tiles;
} velri_lin_blocking;

typedef void (*velri_lin_pack_fn)(const velri_linear_typed_plan *plan, const void *w,
                                  size_t w_stride, const float *w_scale, void *restrict packed);

typedef void (*velri_lin_slice_fn)(const velri_linear_typed_plan *plan,
                                   const velri_linear_run_args *args, size_t row_first,
                                   size_t row_last, size_t col_first, size_t col_last,
                                   void *restrict scratch);

typedef struct {
    velri_lin_kind kind;
    velri_cpu_variant variant;
    size_t max_columns;
    size_t k_group; /* 1 for F32/F64/F16, 2 for AVX512 BF16, 4 for VNNI INT8, etc. */
    const velri_lin_blocking *prepared;
    const velri_lin_blocking *rowmajor;
    velri_lin_pack_fn pack;
    velri_lin_slice_fn run_prepared;
    velri_lin_slice_fn run_rowmajor;
} velri_lin_impl;

/* Generic tile packer in `linear_dispatch.c` shared by matching tile geometries */
void velri_pack_generic_tiles(const velri_linear_typed_plan *plan, size_t k_group, const void *w,
                              size_t w_stride, const float *w_scale, void *restrict packed);

const velri_lin_impl *velri_lin_get_base(velri_lin_kind kind);
#if VELRI_X86_VARIANTS
void velri_vec_axpby_f32_avx2(size_t, size_t, float, const float *, float, const float *, float *);
void velri_vec_mul_f32_avx2(size_t, size_t, const float *, const float *, float *);
void velri_vec_axpby_f32_avx512(size_t, size_t, float, const float *, float, const float *, float *);
void velri_vec_mul_f32_avx512(size_t, size_t, const float *, const float *, float *);

/* Shared FMA GEMV source, compiled with each target's vector primitives. */
typedef void velri_gemv_fn(const velri_linear_typed_plan *, const velri_linear_run_args *,
                          size_t, size_t, size_t, size_t);
velri_gemv_fn velri_gemv_f32_rowmajor_avx2;
velri_gemv_fn velri_gemv_f32_prepared_avx2;
velri_gemv_fn velri_gemv_f64_rowmajor_avx2;
velri_gemv_fn velri_gemv_f64_prepared_avx2;
velri_gemv_fn velri_gemv_f32_rowmajor_avx512;
velri_gemv_fn velri_gemv_f32_prepared_avx512;
velri_gemv_fn velri_gemv_f64_rowmajor_avx512;
velri_gemv_fn velri_gemv_f64_prepared_avx512;

const velri_lin_impl *velri_lin_get_avx2(velri_lin_kind kind);
const velri_lin_impl *velri_lin_get_avx512_fp(velri_lin_kind kind);
const velri_lin_impl *velri_lin_get_avx512_bf16(velri_lin_kind kind);
const velri_lin_impl *velri_lin_get_avx512_int8(velri_lin_kind kind);
#endif

/* ============================================================================
 * 5. Internal LLM Math Operations Entry Points (`BASE`, `AVX2`, `AVX512`)
 * ============================================================================ */

/* Generic read-only query of current logical CPU's L3 cache domain (`0 .. num_l3_domains - 1`) */
int velri_current_l3_domain(void);

/* Generic read-only query of L3 cache capacity per domain in bytes (fallback: 16 MiB) */
size_t velri_l3_domain_bytes(void);

/*
 * Per-worker I/O byte threshold above which streaming kernels switch to non-temporal stores.
 * Sized to 3/4 of a single L3 domain (capped at 16 MiB) so L1/L2/L3-resident compute tensors
 * stay hot in cache while cold-DRAM streaming sweeps bypass cache RFO traffic.
 */
static inline size_t velri_nt_io_bytes(void) {
    size_t l3 = velri_l3_domain_bytes();
    size_t thr = (l3 * 3ULL) / 4ULL;
    return thr > (16ULL << 20) ? (16ULL << 20) : thr;
}

typedef struct {
    void (*rmsnorm)(velri_dtype io_dt, velri_dtype w_dt, size_t r0, size_t r1, size_t dim,
                    float eps, bool unit_offset, const void *x, size_t xs, const void *w, void *y,
                    size_t ys);
    void (*fused_add_rmsnorm_f32)(size_t r0, size_t r1, size_t dim, float eps, bool unit_offset,
                                  float *res, size_t rs, const float *x, size_t xs, const float *w,
                                  float *y, size_t ys);
    void (*layernorm_f32)(size_t r0, size_t r1, size_t dim, float eps, const float *x, size_t xs,
                          const float *w, const float *b, float *restrict y, size_t ys);
    void (*rope)(velri_dtype dt, size_t t0, size_t t1, size_t n_heads, size_t head_dim,
                 size_t rotary_dim, size_t pos_offset, const void *x, size_t tsx, size_t hsx,
                 const float *cos_t, const float *sin_t, void *y, size_t tsy, size_t hsy,
                 velri_rope_mode mode);
    void (*activation_f32)(velri_act_kind kind, size_t i0, size_t i1, const float *x, float *y);
    void (*gated_mlp_f32)(velri_act_kind kind, size_t i0, size_t i1, float clamp_limit,
                          const float *gate, const float *up, float *restrict y);
    void (*softmax_f32)(size_t r0, size_t r1, size_t cols, const float *x, size_t xs, float *y,
                        size_t ys);
    void (*fwht_f32)(size_t blk0, size_t blk1, size_t cols, size_t block_size, const float *x,
                     size_t xs, velri_dtype suh_dt, const void *suh, float *y, size_t ys);
    void (*vec_axpby_f32)(size_t i0, size_t i1, float alpha, const float *a, float beta,
                          const float *b, float *y);
    void (*vec_mul_f32)(size_t i0, size_t i1, const float *a, const float *b, float *y);
} velri_ops_vtable;

const velri_ops_vtable *velri_ops_get_base(void);
#if VELRI_X86_VARIANTS
const velri_ops_vtable *velri_ops_get_avx2(void);
const velri_ops_vtable *velri_ops_get_avx512(void);
#endif

#endif /* VELRI_CPU_INTERNAL_H */
