#ifndef VELRI_TEST_REFERENCE_H
#define VELRI_TEST_REFERENCE_H

#include "velri_lib/kernels.h"

/* Exact-end POSIX guard-page allocation (`PROT_NONE` page immediately after buffer) */
typedef struct {
    void *raw_map;
    size_t map_bytes;
    void *ptr;
} velri_guard_buf;

velri_guard_buf velri_guard_alloc(size_t bytes, size_t align);
void velri_guard_free(velri_guard_buf *gb);

/* Deterministic PRNG for reproducible tests */
uint32_t velri_test_rng(uint64_t *state);
float velri_test_rand_f32(uint64_t *state, float lo, float hi);

/* Pure FP64 / int64 reference implementations */
void ref_linear_run(const velri_linear_desc *desc, size_t rows, size_t m, size_t k, const void *x,
                    size_t x_stride, const float *x_scale, const void *w, size_t w_stride,
                    const float *w_scale, const void *bias, void *y, size_t y_stride);

void ref_rmsnorm(velri_dtype io_dt, velri_dtype w_dt, size_t rows, size_t dim, float eps,
                 bool unit_offset, const void *x, size_t xs, const void *w, void *y, size_t ys);

void ref_fused_add_rmsnorm_f32(size_t rows, size_t dim, float eps, bool unit_offset, float *res,
                               size_t rs, const float *x, size_t xs, const float *w, float *y,
                               size_t ys);

void ref_layernorm_f32(size_t rows, size_t dim, float eps, const float *x, size_t xs,
                       const float *w, const float *b, float *y, size_t ys);

void ref_rope(velri_dtype dt, size_t n_tokens, size_t n_heads, size_t head_dim, size_t rotary_dim,
              size_t pos_offset, const void *x, size_t tsx, size_t hsx, const float *cos_t,
              const float *sin_t, void *y, size_t tsy, size_t hsy, velri_rope_mode mode);

void ref_activation_f32(velri_act_kind kind, size_t count, const float *x, float *y);

void ref_gated_mlp_f32(velri_act_kind kind, size_t count, float clamp_limit, const float *gate,
                       const float *up, float *y);

void ref_softmax_f32(size_t rows, size_t cols, const float *x, size_t xs, float *y, size_t ys);

void ref_fwht_f32(size_t rows, size_t cols, size_t block_size, const float *x, size_t xs,
                  velri_dtype suh_dt, const void *suh, float *y, size_t ys);

void ref_vec_axpby_f32(size_t count, float alpha, const float *a, float beta, const float *b,
                       float *y);

void ref_vec_mul_f32(size_t count, const float *a, const float *b, float *y);

void ref_quantize_row_f32_i8(size_t rows, size_t k, const float *x, size_t x_stride,
                             int8_t *x_i8, size_t xi8_stride, float *scale_out);

#endif /* VELRI_TEST_REFERENCE_H */
