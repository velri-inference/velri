#ifndef VELRI_LIB_SLANG_CPU_H
#define VELRI_LIB_SLANG_CPU_H

#include "velri_lib/types.h"
#include "velri_lib/pool.h"
#include "velri_lib/ops.h"
#include "velri_lib/linear_typed.h"

/*
 * Slang AOT CPU Kernel Dispatch (Prototype — Stage 2).
 * All ISA variants (base / AVX2 / AVX-512) are compiled from ONE shared Slang
 * source (`src/kernels/slang/entry_cpu.slang`) with zero per-ISA code copies,
 * and dispatched across the existing `velri_pool` with zero heap allocations.
 */

velri_status velri_slang_cpu_linear(
    const velri_linear_desc *desc,
    size_t rows, size_t m, size_t k,
    const void *x, size_t x_stride,
    const void *w, size_t w_stride,
    const void *bias,
    const float *x_scale,
    const float *w_scale,
    void *y, size_t y_stride,
    velri_cpu_variant variant,
    velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_rmsnorm(
    velri_dtype io_dtype, size_t rows, size_t dim, float eps, bool unit_offset,
    const void *x, size_t x_stride, const void *weight, void *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_fused_add_rmsnorm_f32(
    size_t rows, size_t dim, float eps, bool unit_offset,
    float *residual_inout, size_t r_stride, const float *x, size_t x_stride,
    const float *weight, float *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_layernorm_f32(
    size_t rows, size_t dim, float eps,
    const float *x, size_t x_stride, const float *weight, const float *bias,
    float *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_rope(
    velri_dtype dtype, size_t n_tokens, size_t n_heads, size_t head_dim,
    size_t rotary_dim, size_t pos_offset,
    const void *x, size_t tok_sx, size_t head_sx,
    const float *cos_tbl, const float *sin_tbl,
    void *y, size_t tok_sy, size_t head_sy,
    velri_rope_mode mode, velri_cpu_variant variant,
    velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_activation_f32(
    velri_act_kind kind, size_t count, const float *x, float *y,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_gated_mlp_f32(
    velri_act_kind gate_act, size_t count, float clamp_limit,
    const float *gate, const float *up, float *y,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_softmax_f32(
    size_t rows, size_t cols, const float *x, size_t x_stride,
    float *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_vec_axpby_f32(
    size_t count, float alpha, const float *a, float beta, const float *b, float *y,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_vec_mul_f32(
    size_t count, const float *a, const float *b, float *y,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_fwht_f32(
    size_t rows, size_t cols, size_t block_size,
    const float *x, size_t x_stride, velri_dtype suh_dtype, const void *suh,
    float *y, size_t y_stride,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

velri_status velri_slang_cpu_quantize_row_f32_i8(
    size_t rows, size_t k, const float *x, size_t x_stride,
    int8_t *x_i8, size_t xi8_stride, float *scale_out,
    velri_cpu_variant variant, velri_pool *pool, size_t active_workers);

#endif /* VELRI_LIB_SLANG_CPU_H */
