#ifndef VELRI_LIB_LINEAR_TYPED_H
#define VELRI_LIB_LINEAR_TYPED_H

#include "velri_lib/types.h"
#include "velri_lib/pool.h"

/*
 * Velri-lib Typed CPU Linear Module:
 *   Computes Y[rows, M] = X[rows, K] * W[M, K]^T + bias[M]  (or scaled quantized form)
 *
 * Architectural & Memory Contracts (matching Velri-lib):
 *   1. Separate storage/accumulator/output dtypes (`velri_linear_desc`):
 *        - F32  * F32  -> F32 acc -> F32          (fast FMA)
 *        - F64  * F64  -> F64 acc -> F64          (native FP64 FMA)
 *        - F16  * F16  -> F32 acc -> F16 or F32
 *        - F32  * F16  -> F32 acc -> F32          (mixed-precision W16A32)
 *        - BF16 * BF16 -> F32 acc -> BF16 or F32  (native AVX512_BF16 vdpbf16ps)
 *        - F32  * BF16 -> F32 acc -> F32          (mixed-precision W16A32)
 *        - I8   * I8   -> I32 acc -> I32          (I32 accumulation modulo 2^32)
 *        - I8   * I8   -> I32 acc -> F32          (W8A8: x_scale[rows] * w_scale[M] * dot + bias)
 *        - F32  * I8   -> F32 acc -> F32          (W8A32 weight-only: dot(x, w) * w_scale[M] +
 * bias)
 *   2. Both Prepared (`packed`) weights AND Row-Major (`unpacked`, e.g. shared embeddings/lm_head)
 *      are supported through the same validated plan contract.
 *   3. Read-only inputs (`x`, `w`/`packed`, `bias`, `x_scale`, `w_scale`) MAY alias each other.
 *      Output `y` (including row-stride gaps) and `scratch` MUST be disjoint from all inputs and
 * plan.
 *   4. Natural element alignment (`alignof(T)`) is sufficient for all caller buffers.
 *   5. Old contents of `y` (including NaN/Inf) are never read; `rows == 0` performs zero writes.
 *   Floating reductions may differ between ISAs and block sizes; serial FP32
 *   bit identity is not promised. INT8 accumulation and I32 bias addition wrap
 *   modulo 2^32. W8A8 reinterprets that sum as signed I32 before scaling.
 *   Query-produced plans are immutable: do not edit fields or deserialize impl.
 *   6. Zero heap allocation in `pack`, `run`, `run_slice`, and `run_parallel`.
 */

typedef struct {
    velri_dtype x_dtype;
    velri_dtype w_dtype;
    velri_dtype acc_dtype;
    velri_dtype y_dtype;
    bool has_x_scale; /* Per-row float scale[rows] (for W8A8 -> F32) */
    bool has_w_scale; /* Per-output float scale[M] (for W8A8 -> F32 and W8A32 -> F32) */
} velri_linear_desc;

/* Standard preset descriptors for convenience */
static inline velri_linear_desc velri_desc_f32(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_F32, VELRI_DTYPE_F32,
                               VELRI_DTYPE_F32, false,           false};
}
static inline velri_linear_desc velri_desc_f64(void) {
    return (velri_linear_desc){VELRI_DTYPE_F64, VELRI_DTYPE_F64, VELRI_DTYPE_F64,
                               VELRI_DTYPE_F64, false,           false};
}
static inline velri_linear_desc velri_desc_f16(velri_dtype y_dtype) {
    return (velri_linear_desc){VELRI_DTYPE_F16, VELRI_DTYPE_F16, VELRI_DTYPE_F32,
                               y_dtype,         false,           false};
}
static inline velri_linear_desc velri_desc_f32x_f16w(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_F16, VELRI_DTYPE_F32,
                               VELRI_DTYPE_F32, false,           false};
}
static inline velri_linear_desc velri_desc_bf16(velri_dtype y_dtype) {
    return (velri_linear_desc){
        VELRI_DTYPE_BF16, VELRI_DTYPE_BF16, VELRI_DTYPE_F32, y_dtype, false, false};
}
static inline velri_linear_desc velri_desc_f32x_bf16w(void) {
    return (velri_linear_desc){
        VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F32, VELRI_DTYPE_F32, false, false};
}
static inline velri_linear_desc velri_desc_i8_i32(void) {
    return (velri_linear_desc){VELRI_DTYPE_I8,  VELRI_DTYPE_I8, VELRI_DTYPE_I32,
                               VELRI_DTYPE_I32, false,          false};
}
static inline velri_linear_desc velri_desc_w8a8_f32(void) {
    return (velri_linear_desc){VELRI_DTYPE_I8,  VELRI_DTYPE_I8, VELRI_DTYPE_I32,
                               VELRI_DTYPE_F32, true,           true};
}
static inline velri_linear_desc velri_desc_f32x_i8w(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_I8, VELRI_DTYPE_F32,
                               VELRI_DTYPE_F32, false,          true};
}

typedef struct velri_linear_typed_plan {
    velri_linear_desc desc;
    size_t m;
    size_t k;
    size_t k_aligned;
    size_t columns;        /* Output column tile width C (last tile can be partial) */
    size_t tile_stride_b;  /* Byte size of one prepared column tile of width `columns` */
    size_t packed_bytes;   /* Exact caller buffer size for prepared weights (0 if unpacked) */
    size_t scratch_bytes;  /* Required per-worker scratch for rows >= 3; rows <= 2 need none */
    bool prepared_weights; /* true = packed layout, false = row-major W[M, K] */
    velri_cpu_variant variant;
    const void *impl; /* Internal immutable kernel table entry */
} velri_linear_typed_plan;

/*
 * Runtime arguments for a Linear call (`run`, `run_slice`, `run_parallel`).
 * Strides and capacities (`*_count`) are in logical ELEMENTS of their respective dtype,
 * except `packed` (`packed_bytes`) and `scratch` (`scratch_bytes`), which are in BYTES.
 * Notice: `bias` has dtype `F64` when `y_dtype == F64`, `I32` when `y_dtype == I32`,
 * and `F32` for all `F32`/`F16`/`BF16` outputs.
 */
typedef struct {
    const void *weights;     /* Packed buffer (if plan->prepared_weights) or row-major W[M, K] */
    size_t weights_capacity; /* Bytes if prepared_weights, else element count of W */
    size_t w_stride;         /* Row stride of W in elements (ignored if prepared_weights) */
    const float *w_scale; /* Optional [M] float weight scales (only if !prepared & has_w_scale) */
    size_t w_scale_count;

    const void *x;        /* Input X[rows, K] */
    size_t x_count;       /* Element capacity of X */
    size_t rows;          /* Number of input rows (0 allowed) */
    size_t x_stride;      /* Row stride of X in elements (>= K) */
    const float *x_scale; /* Optional [rows] float input scales (when desc.has_x_scale) */
    size_t x_scale_count;

    const void *bias; /* Optional [M] bias (nullptr if none; bias_count must be 0 if null) */
    size_t bias_count;

    void *y;         /* Output Y[rows, M] (disjoint from all inputs & plan) */
    size_t y_count;  /* Element capacity of Y */
    size_t y_stride; /* Row stride of Y in elements (>= M) */

    void *scratch; /* Per-worker scratch (`plan->scratch_bytes`), nullptr if 0 needed */
    size_t scratch_bytes;
} velri_linear_run_args;

/* Step 1: Validate descriptor/dimensions/ISA and compute packed & bounded scratch byte sizes. */
velri_status velri_linear_typed_query(const velri_linear_desc *desc, size_t m, size_t k,
                                      bool prepared_weights, velri_cpu_variant variant,
                                      velri_linear_typed_plan *out_plan);

/*
 * Step 2 (only when `plan->prepared_weights == true`):
 * Pack row-major `w[M, K]` (and `w_scale[M]` if `desc.has_w_scale`) into caller-owned `packed`.
 * `packed` is 64-byte-friendly (8-byte natural alignment sufficient) and immutable/shareable.
 */
velri_status velri_linear_typed_pack(const velri_linear_typed_plan *plan, const void *w,
                                     size_t w_count, size_t w_stride, const float *w_scale,
                                     size_t w_scale_count, void *restrict packed,
                                     size_t packed_bytes);

/* Step 3a: Synchronous single-thread execution with full span/overlap/overflow validation. */
velri_status velri_linear_typed_run(const velri_linear_typed_plan *plan,
                                    const velri_linear_run_args *args);

/*
 * Step 3b: Validated output-tile/row slice execution `[row_first, row_last) x [col_first,
 * col_last)` for external schedulers. `col_first` must be a multiple of `plan->columns`, and
 * `col_last` must either be a multiple of `plan->columns` or equal `plan->m`.
 */
velri_status velri_linear_typed_run_slice(const velri_linear_typed_plan *plan,
                                          const velri_linear_run_args *args, size_t row_first,
                                          size_t row_last, size_t col_first, size_t col_last);

/*
 * Step 3c: Full parallel execution across `pool` (`active_workers` clamped to pool size).
 * Automatically partitions 1D output tiles for GEMV/skinny GEMM and 2D `(row_blocks x col_tiles)`
 * for compute-bound GEMM, using each worker's pre-allocated pool scratch (no heap allocation).
 */
velri_status velri_linear_typed_run_parallel(const velri_linear_typed_plan *plan,
                                             const velri_linear_run_args *args, velri_pool *pool,
                                             size_t active_workers);

#endif /* VELRI_LIB_LINEAR_TYPED_H */
