#ifndef VELRI_LIB_VULKAN_H
#define VELRI_LIB_VULKAN_H

#include "velri_lib/types.h"
#include "velri_lib/ops.h"
#include "velri_lib/linear_typed.h"

/*
 * Experimental Vulkan 1.3+ Compute Backend (Stage 2 Slang AOT PoC).
 * - Loads `libvulkan.so.1` dynamically at runtime (zero link-time libvulkan dependency).
 * - Uses Buffer Device Address (BDA) push constants + embedded AOT SPIR-V modules
 *   compiled from `src/kernels/slang/entry_vulkan.slang` and validated by `spirv-val`.
 * - Measures GPU execution with hardware timestamp queries.
 */

typedef struct velri_vk_ctx velri_vk_ctx;
typedef struct velri_vk_buf velri_vk_buf;

typedef struct {
    char device_name[256];
    uint32_t api_version;
    uint32_t subgroup_size;
    uint32_t max_shared_mem;
    float timestamp_period_ns;
    bool has_bda;
    bool has_f16_i8;
    bool has_int_dot;
    bool has_bf16;
    bool has_coopmat;
    bool has_coopmat_f16;
    bool has_coopmat_bf16;
    bool has_coopmat_i8;
} velri_vk_info;

velri_status velri_vk_create(velri_vk_ctx **out_ctx);
void velri_vk_destroy(velri_vk_ctx *ctx);
const velri_vk_info *velri_vk_get_info(const velri_vk_ctx *ctx);

velri_status velri_vk_buf_alloc(velri_vk_ctx *ctx, size_t bytes, velri_vk_buf **out_buf);
void velri_vk_buf_free(velri_vk_ctx *ctx, velri_vk_buf *buf);
velri_status velri_vk_buf_upload(velri_vk_ctx *ctx, velri_vk_buf *dst, const void *src, size_t bytes);
velri_status velri_vk_buf_download(velri_vk_ctx *ctx, void *dst, const velri_vk_buf *src, size_t bytes);
velri_status velri_vk_buf_upload_at(velri_vk_ctx *ctx, velri_vk_buf *dst, size_t dst_offset,
                                    const void *src, size_t bytes);
velri_status velri_vk_buf_download_at(velri_vk_ctx *ctx, void *dst, const velri_vk_buf *src,
                                      size_t src_offset, size_t bytes);

/* Stack-allocated zero-copy subview into a velri_vk_buf at `byte_offset` (0 B heap alloc). */
typedef struct {
    uint64_t opaque[4];
} velri_vk_buf_view;

velri_vk_buf *velri_vk_buf_subview(const velri_vk_buf *base, size_t byte_offset,
                                   velri_vk_buf_view *storage);

/* Timed compute kernels (repeat `iters` times inside one command buffer for accurate GPU timestamps). */
velri_status velri_vk_linear(
    velri_vk_ctx *ctx, const velri_linear_desc *desc,
    uint32_t rows, uint32_t m, uint32_t k,
    const velri_vk_buf *x, uint32_t x_stride,
    const velri_vk_buf *w, uint32_t w_stride,
    const velri_vk_buf *bias,
    const velri_vk_buf *x_scale,
    const velri_vk_buf *w_scale,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_rmsnorm(
    velri_vk_ctx *ctx, velri_dtype io_dtype,
    uint32_t rows, uint32_t dim, float eps, bool unit_offset,
    const velri_vk_buf *x, uint32_t x_stride,
    const velri_vk_buf *weight,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_fused_add_rmsnorm_f32(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t dim, float eps, bool unit_offset,
    velri_vk_buf *residual_inout, uint32_t r_stride,
    const velri_vk_buf *x, uint32_t x_stride,
    const velri_vk_buf *weight,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_layernorm_f32(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t dim, float eps,
    const velri_vk_buf *x, uint32_t x_stride,
    const velri_vk_buf *weight, const velri_vk_buf *bias,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_rope(
    velri_vk_ctx *ctx, velri_dtype dtype,
    uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim, uint32_t rotary_dim,
    uint32_t pos_offset,
    const velri_vk_buf *x, uint32_t tok_sx, uint32_t head_sx,
    const velri_vk_buf *cos_tbl, const velri_vk_buf *sin_tbl,
    velri_vk_buf *y, uint32_t tok_sy, uint32_t head_sy,
    velri_rope_mode mode, uint32_t iters, double *out_avg_ms);

velri_status velri_vk_activation_f32(
    velri_vk_ctx *ctx, velri_act_kind kind, uint32_t count,
    const velri_vk_buf *x, velri_vk_buf *y,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_gated_mlp_f32(
    velri_vk_ctx *ctx, velri_act_kind act, uint32_t count, float clamp_limit,
    const velri_vk_buf *gate, const velri_vk_buf *up, velri_vk_buf *y,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_vec_axpby_f32(
    velri_vk_ctx *ctx, uint32_t count, float alpha, const velri_vk_buf *a,
    float beta, const velri_vk_buf *b, velri_vk_buf *y,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_vec_mul_f32(
    velri_vk_ctx *ctx, uint32_t count, const velri_vk_buf *a,
    const velri_vk_buf *b, velri_vk_buf *y,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_softmax_f32(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t cols,
    const velri_vk_buf *x, uint32_t x_stride,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_fwht_f32(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t cols, uint32_t block_size,
    const velri_vk_buf *x, uint32_t x_stride,
    velri_dtype suh_dtype, const velri_vk_buf *suh,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms);

velri_status velri_vk_quantize_row_f32_i8(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t k,
    const velri_vk_buf *x, uint32_t x_stride,
    velri_vk_buf *x_i8, uint32_t xi8_stride,
    velri_vk_buf *scale_out,
    uint32_t iters, double *out_avg_ms);

/* Hardware ceiling calibrations (measured directly on this GPU via timestamps). */
velri_status velri_vk_measure_bw_gbps(velri_vk_ctx *ctx, size_t copy_bytes, uint32_t iters,
                                      double *out_gbps);
velri_status velri_vk_measure_bw_read_gbps(velri_vk_ctx *ctx, size_t read_bytes, uint32_t iters,
                                           double *out_gbps);
velri_status velri_vk_measure_fma_tflops(velri_vk_ctx *ctx, uint32_t workgroups, uint32_t loop_iters,
                                         double *out_tflops);
velri_status velri_vk_measure_coopmat_peak(velri_vk_ctx *ctx, velri_dtype dtype,
                                           uint32_t workgroups, uint32_t loop_iters,
                                           double *out_tops);

#endif /* VELRI_LIB_VULKAN_H */
