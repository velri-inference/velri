#include "internal.h"

/* One descriptor per mathematical mode. ISA selection is a separate decision. */
static bool resolve_lin_kind(const velri_linear_desc *d, velri_lin_kind *out) {
    static const velri_linear_desc modes[] = {
        {VELRI_DTYPE_F32, VELRI_DTYPE_F32, VELRI_DTYPE_F32, VELRI_DTYPE_F32, false, false},
        {VELRI_DTYPE_F64, VELRI_DTYPE_F64, VELRI_DTYPE_F64, VELRI_DTYPE_F64, false, false},
        {VELRI_DTYPE_F16, VELRI_DTYPE_F16, VELRI_DTYPE_F32, VELRI_DTYPE_F32, false, false},
        {VELRI_DTYPE_F16, VELRI_DTYPE_F16, VELRI_DTYPE_F32, VELRI_DTYPE_F16, false, false},
        {VELRI_DTYPE_F32, VELRI_DTYPE_F16, VELRI_DTYPE_F32, VELRI_DTYPE_F32, false, false},
        {VELRI_DTYPE_BF16, VELRI_DTYPE_BF16, VELRI_DTYPE_F32, VELRI_DTYPE_F32, false, false},
        {VELRI_DTYPE_BF16, VELRI_DTYPE_BF16, VELRI_DTYPE_F32, VELRI_DTYPE_BF16, false, false},
        {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F32, VELRI_DTYPE_F32, false, false},
        {VELRI_DTYPE_I8, VELRI_DTYPE_I8, VELRI_DTYPE_I32, VELRI_DTYPE_I32, false, false},
        {VELRI_DTYPE_I8, VELRI_DTYPE_I8, VELRI_DTYPE_I32, VELRI_DTYPE_F32, true, true},
        {VELRI_DTYPE_F32, VELRI_DTYPE_I8, VELRI_DTYPE_F32, VELRI_DTYPE_F32, false, true},
    };
    if (!d)
        return false;
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        const velri_linear_desc *m = &modes[i];
        if (d->x_dtype == m->x_dtype && d->w_dtype == m->w_dtype && d->acc_dtype == m->acc_dtype &&
            d->y_dtype == m->y_dtype && d->has_x_scale == m->has_x_scale &&
            d->has_w_scale == m->has_w_scale) {
            *out = (velri_lin_kind)i;
            return true;
        }
    }
    return false;
}

static const velri_lin_impl *lookup_lin_impl(velri_lin_kind kind, velri_cpu_variant variant) {
#if VELRI_X86_VARIANTS
    if (variant == VELRI_CPU_AVX512) {
        switch (kind) {
        case VELRI_LIN_F32:
        case VELRI_LIN_F64:
        case VELRI_LIN_F16_F32:
        case VELRI_LIN_F16_F16:
        case VELRI_LIN_F32X_F16W:
            return velri_lin_get_avx512_fp(kind);
        case VELRI_LIN_BF16_F32:
        case VELRI_LIN_BF16_BF16:
        case VELRI_LIN_F32X_BF16W:
            return velri_get_cpu_info()->has_avx512_bf16 ? velri_lin_get_avx512_bf16(kind)
                                                         : nullptr;
        case VELRI_LIN_I8_I32:
        case VELRI_LIN_W8A8_F32:
        case VELRI_LIN_F32X_I8W:
            return velri_get_cpu_info()->has_avx512_vnni ? velri_lin_get_avx512_int8(kind)
                                                         : nullptr;
        }
    }
    if (variant == VELRI_CPU_AVX2) {
        return velri_lin_get_avx2(kind);
    }
#endif
    if (variant == VELRI_CPU_BASE) {
        return velri_lin_get_base(kind);
    }
    return nullptr;
}

static velri_dtype bias_dtype_for_y(velri_dtype y_dtype) {
    if (y_dtype == VELRI_DTYPE_F64) {
        return VELRI_DTYPE_F64;
    }
    if (y_dtype == VELRI_DTYPE_I32) {
        return VELRI_DTYPE_I32;
    }
    return VELRI_DTYPE_F32;
}

/*
 * Generic tile packer for layout:
 *   For each tile t in [0, ceil(m / C)):
 *     1. Payload `[k_aligned / G, C, G]` of `w_dtype` (zero-padded for col >= m and k_idx >= k)
 *     2. Padding to 4-byte boundary
 *     3. If `w_dtype == VELRI_DTYPE_I8`: `int32_t bias_corr[C]` (`-128 * sum_k w[col, k]`)
 *     4. If `desc.has_w_scale`: `float tile_w_scale[C]`
 */
void velri_pack_generic_tiles(const velri_linear_typed_plan *plan, size_t k_group, const void *w,
                              size_t w_stride, const float *w_scale, void *restrict packed) {
    const size_t m = plan->m;
    const size_t k = plan->k;
    const size_t C = plan->columns;
    const size_t G = k_group;
    const size_t k_blocks = plan->k_aligned / G;
    const size_t elem_b = velri_dtype_bytes(plan->desc.w_dtype);
    const size_t payload_b = k_blocks * C * G * elem_b;
    const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
    const bool is_i8 = (plan->desc.w_dtype == VELRI_DTYPE_I8);
    const size_t scale_off = meta_off + (is_i8 ? C * sizeof(int32_t) : 0);

    const uint8_t *w_bytes = (const uint8_t *)w;
    uint8_t *dst_base = (uint8_t *)packed;

    for (size_t col0 = 0, t = 0; col0 < m; col0 += C, ++t) {
        uint8_t *tile_dst = dst_base + t * plan->tile_stride_b;
        memset(tile_dst, 0, plan->tile_stride_b);

        for (size_t kb0 = 0; kb0 < k_blocks; kb0 += 64) {
            size_t kb1 = kb0 + 64 < k_blocks ? kb0 + 64 : k_blocks;
            for (size_t lane0 = 0; lane0 < C; lane0 += 8) {
                size_t lane1 = lane0 + 8 < C ? lane0 + 8 : C;
                for (size_t kb = kb0; kb < kb1; ++kb) {
                    for (size_t lane = lane0; lane < lane1; ++lane) {
                        size_t col = col0 + lane;
                        uint8_t *cell = tile_dst + (kb * C + lane) * G * elem_b;
                        if (col < m) {
                            const uint8_t *w_row = w_bytes + col * w_stride * elem_b;
                            size_t ki0 = kb * G;
                            if (ki0 + G <= k) {
                                memcpy(cell, w_row + ki0 * elem_b, G * elem_b);
                            } else {
                                for (size_t g = 0; g < G; ++g) {
                                    size_t ki = ki0 + g;
                                    if (ki < k) {
                                        memcpy(cell + g * elem_b, w_row + ki * elem_b, elem_b);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (is_i8) {
            int32_t *corr = (int32_t *)(tile_dst + meta_off);
            const int8_t *w_i8 = (const int8_t *)w;
            for (size_t lane = 0; lane < C; ++lane) {
                size_t col = col0 + lane;
                int32_t sum = 0;
                if (col < m) {
                    const int8_t *row = w_i8 + col * w_stride;
                    for (size_t i = 0; i < k; ++i) {
                        sum = velri_i32_add(sum, (int32_t)row[i]);
                    }
                }
                corr[lane] = velri_i32_mul(-128, sum);
            }
        }

        if (plan->desc.has_w_scale) {
            float *sc = (float *)(tile_dst + scale_off);
            for (size_t lane = 0; lane < C; ++lane) {
                size_t col = col0 + lane;
                sc[lane] = (col < m && w_scale) ? w_scale[col] : 0.0f;
            }
        }
    }
}

velri_status velri_linear_typed_query(const velri_linear_desc *desc, size_t m, size_t k,
                                      bool prepared_weights, velri_cpu_variant variant,
                                      velri_linear_typed_plan *out_plan) {
    if (!out_plan || !desc || m == 0 || k == 0) {
        return VELRI_ERR_INVALID;
    }
    velri_lin_kind kind;
    if (!resolve_lin_kind(desc, &kind)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (!velri_cpu_variant_supported(variant)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    velri_cpu_variant resolved = velri_cpu_resolve_variant(variant);
    const velri_lin_impl *impl = lookup_lin_impl(kind, resolved);
    if (!impl && variant == VELRI_CPU_AUTO) {
        resolved = velri_cpu_variant_supported(VELRI_CPU_AVX2) ? VELRI_CPU_AVX2 : VELRI_CPU_BASE;
        impl = lookup_lin_impl(kind, resolved);
    }
    if (!impl) {
        return VELRI_ERR_UNSUPPORTED;
    }

    /* Select largest power-of-two tile width C <= min(m, impl->max_columns) */
    size_t C = impl->max_columns;
    while (C > m && C > 1) {
        C >>= 1;
    }

    size_t G = impl->k_group;
    size_t k_rem = k % G;
    size_t k_aligned = k;
    if (k_rem != 0) {
        if (!velri_add_size(k, G - k_rem, &k_aligned)) {
            return VELRI_ERR_OVERFLOW;
        }
    }

    size_t packed_bytes = 0;
    size_t tile_stride_b = 0;
    if (prepared_weights) {
        size_t elem_b = velri_dtype_bytes(desc->w_dtype);
        size_t tile_elems = 0, payload_b = 0;
        if (!velri_mul_size(k_aligned, C, &tile_elems) ||
            !velri_mul_size(tile_elems, elem_b, &payload_b) || payload_b > SIZE_MAX - 3ULL) {
            return VELRI_ERR_OVERFLOW;
        }
        size_t meta_off = (payload_b + 3ULL) & ~3ULL;
        size_t extra_b = 0;
        if (desc->w_dtype == VELRI_DTYPE_I8) {
            extra_b += C * sizeof(int32_t);
        }
        if (desc->has_w_scale) {
            extra_b += C * sizeof(float);
        }
        if (!velri_add_size(meta_off, extra_b, &tile_stride_b)) {
            return VELRI_ERR_OVERFLOW;
        }
        size_t stride_rem = tile_stride_b % 64ULL;
        if (stride_rem != 0) {
            if (!velri_add_size(tile_stride_b, 64ULL - stride_rem, &tile_stride_b)) {
                return VELRI_ERR_OVERFLOW;
            }
        }
        if (tile_stride_b >= 4096ULL && (tile_stride_b & 4095ULL) == 0) {
            /* +64 B skew prevents 4 KiB set-index aliasing across consecutive prepared tiles */
            if (!velri_add_size(tile_stride_b, 64ULL, &tile_stride_b)) {
                return VELRI_ERR_OVERFLOW;
            }
        }
        size_t num_tiles = (m - 1) / C + 1;
        if (!velri_mul_size(num_tiles, tile_stride_b, &packed_bytes)) {
            return VELRI_ERR_OVERFLOW;
        }
    } else {
        /* Still check that logical W[m, k] does not overflow size_t bytes */
        size_t w_elems = 0, w_bytes = 0;
        if (!velri_mul_size(m, k, &w_elems) ||
            !velri_mul_size(w_elems, velri_dtype_bytes(desc->w_dtype), &w_bytes)) {
            return VELRI_ERR_OVERFLOW;
        }
    }

    size_t scratch_b = (prepared_weights ? impl->prepared : impl->rowmajor)->scratch_bytes;

    velri_linear_typed_plan p = {
        .desc = *desc,
        .m = m,
        .k = k,
        .k_aligned = k_aligned,
        .columns = C,
        .tile_stride_b = tile_stride_b,
        .packed_bytes = packed_bytes,
        .scratch_bytes = scratch_b,
        .prepared_weights = prepared_weights,
        .variant = resolved,
        .impl = impl,
    };
    *out_plan = p;
    return VELRI_OK;
}

static bool valid_typed_plan(const velri_linear_typed_plan *p) {
    if (!p || p->m == 0 || p->k == 0 || p->columns == 0 || p->columns > p->m || !p->impl) {
        return false;
    }
    if (p->prepared_weights && (p->packed_bytes == 0 || p->tile_stride_b == 0)) {
        return false;
    }
    return true;
}

velri_status velri_linear_typed_pack(const velri_linear_typed_plan *plan, const void *w,
                                     size_t w_count, size_t w_stride, const float *w_scale,
                                     size_t w_scale_count, void *restrict packed,
                                     size_t packed_bytes) {
    if (!valid_typed_plan(plan) || !plan->prepared_weights) {
        return VELRI_ERR_INVALID;
    }
    const size_t w_elem_b = velri_dtype_bytes(plan->desc.w_dtype);
    const size_t w_align = velri_dtype_align(plan->desc.w_dtype);
    size_t wb = 0, wsb = 0, pb = 0;

    velri_status s =
        velri_check_span(w, w_count, plan->m, plan->k, w_stride, w_elem_b, w_align, &wb);
    if (s != VELRI_OK) {
        return s;
    }
    if (plan->desc.has_w_scale) {
        s = velri_check_span(w_scale, w_scale_count, 1, plan->m, plan->m, sizeof(float),
                             alignof(float), &wsb);
        if (s != VELRI_OK) {
            return s;
        }
    } else if (w_scale != nullptr || w_scale_count != 0) {
        return VELRI_ERR_INVALID;
    }

    size_t p_align = w_align > alignof(float) ? w_align : alignof(float);
    if (plan->desc.w_dtype == VELRI_DTYPE_F64) {
        p_align = alignof(double);
    }
    s = velri_check_span(packed, packed_bytes, 1, plan->packed_bytes, plan->packed_bytes, 1,
                         p_align, &pb);
    if (s != VELRI_OK) {
        return s;
    }

    if (velri_overlap(packed, pb, w, wb) || velri_overlap(packed, pb, w_scale, wsb) ||
        velri_overlap(packed, pb, plan, sizeof(*plan))) {
        return VELRI_ERR_INVALID;
    }

    const velri_lin_impl *impl = (const velri_lin_impl *)plan->impl;
    impl->pack(plan, w, w_stride, w_scale, packed);
    return VELRI_OK;
}

velri_status velri_linear_validate_run(const velri_linear_typed_plan *plan,
                                       const velri_linear_run_args *a,
                                       bool require_caller_scratch) {
    if (!valid_typed_plan(plan) || !a) {
        return VELRI_ERR_INVALID;
    }
    if (!a->bias && a->bias_count != 0) {
        return VELRI_ERR_INVALID;
    }

    size_t wb = 0, wsb = 0, xb = 0, xsb = 0, bb = 0, yb = 0;
    velri_status s;

    if (plan->prepared_weights) {
        size_t w_align = velri_dtype_align(plan->desc.w_dtype);
        size_t p_align = w_align > alignof(float) ? w_align : alignof(float);
        if (plan->desc.w_dtype == VELRI_DTYPE_F64) {
            p_align = alignof(double);
        }
        s = velri_check_span(a->weights, a->weights_capacity, 1, plan->packed_bytes,
                             plan->packed_bytes, 1, p_align, &wb);
        if (s != VELRI_OK) {
            return s;
        }
    } else {
        s = velri_check_span(a->weights, a->weights_capacity, plan->m, plan->k, a->w_stride,
                             velri_dtype_bytes(plan->desc.w_dtype),
                             velri_dtype_align(plan->desc.w_dtype), &wb);
        if (s != VELRI_OK) {
            return s;
        }
        if (plan->desc.has_w_scale) {
            s = velri_check_span(a->w_scale, a->w_scale_count, 1, plan->m, plan->m, sizeof(float),
                                 alignof(float), &wsb);
            if (s != VELRI_OK) {
                return s;
            }
        } else if (a->w_scale != nullptr || a->w_scale_count != 0) {
            return VELRI_ERR_INVALID;
        }
    }

    s = velri_check_span(a->x, a->x_count, a->rows, plan->k, a->x_stride,
                         velri_dtype_bytes(plan->desc.x_dtype),
                         velri_dtype_align(plan->desc.x_dtype), &xb);
    if (s != VELRI_OK) {
        return s;
    }

    if (plan->desc.has_x_scale) {
        s = velri_check_span(a->x_scale, a->x_scale_count, a->rows ? 1 : 0, a->rows, a->rows,
                             sizeof(float), alignof(float), &xsb);
        if (s != VELRI_OK) {
            return s;
        }
    } else if (a->x_scale != nullptr || a->x_scale_count != 0) {
        return VELRI_ERR_INVALID;
    }

    velri_dtype b_dt = bias_dtype_for_y(plan->desc.y_dtype);
    s = velri_check_span(a->bias, a->bias_count, a->bias ? 1 : 0, plan->m, plan->m,
                         velri_dtype_bytes(b_dt), velri_dtype_align(b_dt), &bb);
    if (s != VELRI_OK) {
        return s;
    }

    s = velri_check_span(a->y, a->y_count, a->rows, plan->m, a->y_stride,
                         velri_dtype_bytes(plan->desc.y_dtype),
                         velri_dtype_align(plan->desc.y_dtype), &yb);
    if (s != VELRI_OK) {
        return s;
    }

    /* Output Y must be disjoint from all inputs and from plan */
    if (velri_overlap(a->y, yb, a->weights, wb) || velri_overlap(a->y, yb, a->w_scale, wsb) ||
        velri_overlap(a->y, yb, a->x, xb) || velri_overlap(a->y, yb, a->x_scale, xsb) ||
        velri_overlap(a->y, yb, a->bias, bb) || velri_overlap(a->y, yb, plan, sizeof(*plan)) ||
        velri_overlap(a->y, yb, a, sizeof(*a))) {
        return VELRI_ERR_INVALID;
    }

    if (require_caller_scratch && a->rows >= 3 && plan->scratch_bytes > 0) {
        size_t scb = 0;
        s = velri_check_span(a->scratch, a->scratch_bytes, 1, plan->scratch_bytes,
                             plan->scratch_bytes, 1, 1, &scb);
        if (s != VELRI_OK) {
            return s;
        }
        if (velri_overlap(a->scratch, scb, a->weights, wb) ||
            velri_overlap(a->scratch, scb, a->w_scale, wsb) ||
            velri_overlap(a->scratch, scb, a->x, xb) ||
            velri_overlap(a->scratch, scb, a->x_scale, xsb) ||
            velri_overlap(a->scratch, scb, a->bias, bb) ||
            velri_overlap(a->scratch, scb, a->y, yb) ||
            velri_overlap(a->scratch, scb, plan, sizeof(*plan)) ||
            velri_overlap(a->scratch, scb, a, sizeof(*a))) {
            return VELRI_ERR_INVALID;
        }
    }

    return VELRI_OK;
}

velri_status velri_linear_typed_run_slice(const velri_linear_typed_plan *plan,
                                          const velri_linear_run_args *args, size_t row_first,
                                          size_t row_last, size_t col_first, size_t col_last) {
    velri_status s = velri_linear_validate_run(plan, args, true);
    if (s != VELRI_OK) {
        return s;
    }
    if (row_first > row_last || row_last > args->rows || col_first > col_last ||
        col_last > plan->m) {
        return VELRI_ERR_INVALID;
    }
    if ((col_first % plan->columns) != 0 ||
        (col_last != plan->m && (col_last % plan->columns) != 0)) {
        return VELRI_ERR_INVALID;
    }
    if (row_first == row_last || col_first == col_last) {
        return VELRI_OK;
    }
    const velri_lin_impl *impl = (const velri_lin_impl *)plan->impl;
    velri_lin_slice_fn fn = plan->prepared_weights ? impl->run_prepared : impl->run_rowmajor;
    fn(plan, args, row_first, row_last, col_first, col_last, args->scratch);
    return VELRI_OK;
}

velri_status velri_linear_typed_run(const velri_linear_typed_plan *plan,
                                    const velri_linear_run_args *args) {
    velri_status s = velri_linear_validate_run(plan, args, true);
    if (s != VELRI_OK || args->rows == 0) {
        return s;
    }
    const velri_lin_impl *impl = (const velri_lin_impl *)plan->impl;
    velri_lin_slice_fn fn = plan->prepared_weights ? impl->run_prepared : impl->run_rowmajor;
    fn(plan, args, 0, args->rows, 0, plan->m, args->scratch);
    return VELRI_OK;
}
