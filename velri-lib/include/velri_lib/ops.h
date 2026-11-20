#ifndef VELRI_LIB_OPS_H
#define VELRI_LIB_OPS_H

#include "velri_lib/types.h"
#include "velri_lib/pool.h"

/*
 * Velri-lib Core LLM Mathematical Operations:
 *   - All functions validate sizes, strides, overflow, and aliasing before the first write.
 *   - Zero heap allocations (`0 B` malloc/free) across all operations.
 *   - Pass `pool == nullptr` (or `active_workers <= 1`) for direct single-thread execution.
 */

/* ============================================================================
 * 1. Normalization: RMSNorm (with `w` or `1 + w` unit-offset!), Fused Residual+RMSNorm, LayerNorm
 *    - Supports F32, BF16, and F16 inputs/weights/outputs (internal FP32 accumulation).
 *    - `weight` may be nullptr (treated as all-ones).
 *    - Read-only `x` and `weight` may alias; `y` may equal `x` (in-place) ONLY when `x_dtype ==
 * y_dtype` and `x_stride == y_stride`, otherwise `y` must be disjoint from inputs.
 * ============================================================================ */

velri_status velri_rmsnorm(velri_dtype io_dtype, velri_dtype weight_dtype, size_t rows, size_t dim,
                           float eps, bool unit_offset, const void *x, size_t x_stride,
                           const void *weight, void *y, size_t y_stride, velri_cpu_variant variant,
                           velri_pool *pool, size_t active_workers);

/*
 * Fused Residual + RMSNorm (avoids a separate pass over the residual sum):
 *   residual_inout[r, d] += x[r, d]
 *   y[r, d] = rmsnorm(residual_inout[r, :], weight, eps, unit_offset)
 * `residual_inout` and `y` must be disjoint (`y != residual_inout`), or `y` may equal
 * `residual_inout` when post-norm overwrite is desired.
 */
velri_status velri_fused_add_rmsnorm_f32(size_t rows, size_t dim, float eps, bool unit_offset,
                                         float *residual_inout, size_t r_stride, const float *x,
                                         size_t x_stride, const float *weight, float *y,
                                         size_t y_stride, velri_cpu_variant variant,
                                         velri_pool *pool, size_t active_workers);

velri_status velri_layernorm_f32(size_t rows, size_t dim, float eps, const float *x,
                                 size_t x_stride, const float *weight, const float *bias,
                                 float *restrict y, size_t y_stride, velri_cpu_variant variant,
                                 velri_pool *pool, size_t active_workers);

/* ============================================================================
 * 2. Rotary Position Embeddings (RoPE)
 *    - Supports partial `rotary_dim <= head_dim` (`rotary_dim` even; features
 * `[rotary_dim..head_dim)` are copied unchanged, matching Velri's block specification).
 *    - Supports both `VELRI_ROPE_HALF` (NeoX / Llama / Qwen rotate-half) and
 *      `VELRI_ROPE_INTERLEAVED` (GPT-J even/odd pairs).
 *    - Supports F32, BF16, and F16 tensors `[n_tokens, n_heads, head_dim]`, in-place (`y == x`)
 *      or disjoint out-of-place (`y != x`).
 *    - `cos_table` and `sin_table` have shape `[cos_sin_tokens, rotary_dim / 2]` in FP32, indexed
 *      at row `pos_offset + token_idx` (checked <= `cos_sin_tokens`).
 * ============================================================================ */

typedef enum { VELRI_ROPE_HALF = 0, VELRI_ROPE_INTERLEAVED = 1 } velri_rope_mode;

velri_status velri_rope(velri_dtype dtype, size_t n_tokens, size_t n_heads, size_t head_dim,
                        size_t rotary_dim, size_t pos_offset, size_t cos_sin_tokens, const void *x,
                        size_t token_stride_x, size_t head_stride_x, const float *cos_table,
                        const float *sin_table, void *y, size_t token_stride_y,
                        size_t head_stride_y, velri_rope_mode mode, velri_cpu_variant variant,
                        velri_pool *pool, size_t active_workers);

/* ============================================================================
 * 3. Activations & Fused Gated MLP (`SwiGLU`, `GeGLU`, `SiLU`, `GELU_tanh`, `ReLU2`)
 *    - SIMD implementations use polynomial approximations to exp/tanh.
 *    - Optional symmetric input clamping `clamp_limit > 0.0f` (`0.0f` = no clamp), matching
 *      element-wise gate/up clamping (`clamp(gate, -limit, +limit)`, `clamp(up, -limit, +limit)`).
 * ============================================================================ */

typedef enum { VELRI_ACT_SILU = 0, VELRI_ACT_GELU_TANH = 1, VELRI_ACT_RELU2 = 2 } velri_act_kind;

velri_status velri_activation_f32(velri_act_kind kind, size_t count, const float *x, float *y,
                                  velri_cpu_variant variant, velri_pool *pool,
                                  size_t active_workers);

/* Fused activation(gate) * up only; matrix projections are separate linear calls. */
velri_status velri_gated_mlp_f32(velri_act_kind gate_act, size_t count, float clamp_limit,
                                 const float *gate, const float *up, float *restrict y,
                                 velri_cpu_variant variant, velri_pool *pool,
                                 size_t active_workers);

/* ============================================================================
 * 4. 1-Pass Online Safe Softmax
 *    - Row-wise softmax over `x[rows, cols] -> y[rows, cols]` (supports in-place `y == x`
 *      when `x_stride == y_stride`, or disjoint `y`).
 * ============================================================================ */

velri_status velri_softmax_f32(size_t rows, size_t cols, const float *x, size_t x_stride, float *y,
                               size_t y_stride, velri_cpu_variant variant, velri_pool *pool,
                               size_t active_workers);

/* ============================================================================
 * 5. Normalized Fast Walsh-Hadamard Transform (`FWHT`)
 *    - Computes block-orthonormal Walsh-Hadamard rotation:
 *        y[r, b*B .. (b+1)*B) = (1 / sqrt(B)) * H_B( (suh * x)[r, b*B .. (b+1)*B) )
 *    - `block_size` (`B`) must be a power of two in `{16, 32, 64, 128, 256}`, with `cols > 0`
 *      and `cols % block_size == 0`.
 *    - `suh` may be `nullptr` (unit scale), or a `cols`-element vector of `VELRI_DTYPE_F32` or
 *      `VELRI_DTYPE_F16`.
 *    - Supports in-place (`y == x` when `x_stride == y_stride`) or disjoint out-of-place `y`.
 * ============================================================================ */

velri_status velri_fwht_f32(size_t rows, size_t cols, size_t block_size, const float *x,
                            size_t x_stride, velri_dtype suh_dtype, const void *suh, float *y,
                            size_t y_stride, velri_cpu_variant variant, velri_pool *pool,
                            size_t active_workers);

/* ============================================================================
 * 6. Elementwise Vector Math (`axpby` for `PreNormResidual` / MoE combine, `mul`)
 *    - `velri_vec_axpby_f32`: `y[i] = alpha * a[i] + beta * b[i]` (`b` may be `nullptr` when
 *      `beta == 0.0f`; `y` may equal `a` or `b` in-place, or be disjoint from inputs).
 *    - `velri_vec_mul_f32`: `y[i] = a[i] * b[i]` (`y` may equal `a` or `b` in-place, or be
 *      disjoint from inputs).
 * ============================================================================ */

velri_status velri_vec_axpby_f32(size_t count, float alpha, const float *a, float beta,
                                 const float *b, float *y, velri_cpu_variant variant,
                                 velri_pool *pool, size_t active_workers);

velri_status velri_vec_mul_f32(size_t count, const float *a, const float *b, float *y,
                               velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

/* ============================================================================
 * 7. Parallel Row Quantization (`W8A8` activations) & Format Conversions
 * ============================================================================ */
velri_status velri_quantize_row_f32_i8_parallel(size_t rows, size_t k, const float *x,
                                                size_t x_stride, int8_t *restrict x_i8,
                                                size_t xi8_stride, float *restrict scale_out,
                                                velri_cpu_variant variant, velri_pool *pool,
                                                size_t active_workers);

velri_status velri_cvt_parallel(velri_dtype src_dtype, velri_dtype dst_dtype, const void *src,
                                void *restrict dst, size_t count, velri_cpu_variant variant,
                                velri_pool *pool, size_t active_workers);

#endif /* VELRI_LIB_OPS_H */
