/* Included twice by gemv.c. Only type bindings differ; the loops below are shared. */
#if GEMV_F64
#define SCALAR double
#define VECTOR simd_f64
#define LANES SIMD_F64_LANES
#define NAME(name) SIMD_NAME(VELRI_JOIN(velri_gemv_f64_, name))
#define ZERO() SIMD_OP(setzero_pd)()
#define BROADCAST(x) SIMD_OP(set1_pd)(x)
#define ADD(a, b) SIMD_OP(add_pd)(a, b)
#define FMA(a, b, c) SIMD_OP(fmadd_pd)(a, b, c)
#define LOAD(dt, p, i) SIMD_OP(loadu_pd)((const double *)(p) + (i))
#define LOAD1(dt, p, i) (((const double *)(p))[i])
#define STORE(p, x) SIMD_OP(storeu_pd)(p, x)
#define SUM(x) simd_sum_f64(x)
#else
#define SCALAR float
#define VECTOR simd_f32
#define LANES SIMD_F32_LANES
#define NAME(name) SIMD_NAME(VELRI_JOIN(velri_gemv_f32_, name))
#define ZERO() SIMD_OP(setzero_ps)()
#define BROADCAST(x) SIMD_OP(set1_ps)(x)
#define ADD(a, b) SIMD_OP(add_ps)(a, b)
#define FMA(a, b, c) SIMD_OP(fmadd_ps)(a, b, c)
#define LOAD(dt, p, i) simd_load_as_f32(dt, p, i)
#define LOAD1(dt, p, i) simd_load1_f32(dt, p, i)
#define STORE(p, x) SIMD_OP(storeu_ps)(p, x)
#define SUM(x) simd_sum_f32(x)
#endif

static inline void NAME(write)(const velri_linear_typed_plan *p,
                               const velri_linear_run_args *a, size_t row, size_t col,
                               SCALAR *values, size_t n, const float *scales) {
    const SCALAR *bias = (const SCALAR *)a->bias;
    for (size_t j = 0; j < n; ++j) {
        if (scales) values[j] *= scales[j];
#if GEMV_F64
        values[j] += bias ? bias[col + j] : 0.0;
#else
        if (bias) values[j] += bias[col + j];
#endif
    }
    const size_t offset = row * a->y_stride + col;
#if GEMV_F64
    (void)p;
    memcpy((double *)a->y + offset, values, n * sizeof(double));
#else
    if (p->desc.y_dtype == VELRI_DTYPE_F16)
        SIMD_NAME(velri_cvt_f32_to_f16)(values, (velri_fp16 *)a->y + offset, n);
    else if (p->desc.y_dtype == VELRI_DTYPE_BF16)
        SIMD_NAME(velri_cvt_f32_to_bf16)(values, (velri_bf16 *)a->y + offset, n);
    else
        memcpy((float *)a->y + offset, values, n * sizeof(float));
#endif
}

[[gnu::always_inline]] static inline void NAME(rowmajor_impl)(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                    size_t r0, size_t r1, size_t c0, size_t c1,
                    velri_dtype xd, velri_dtype wd) {
    const size_t k = p->k;
    (void)xd; (void)wd;
    const bool group_rows = !GEMV_F64 || SIMD_GEMV_F64_ROWS == 4;
    const bool split = GEMV_F64 || SIMD_GEMV_UNROLL == 2;
    for (size_t r = r0; r < r1; ++r) {
        const size_t xo = r * a->x_stride;
        size_t c = c0;
        for (; group_rows && c + 4 <= c1; c += 4) {
            VECTOR a0 = ZERO(), a1 = ZERO(), a2 = ZERO(), a3 = ZERO();
            VECTOR b0 = ZERO(), b1 = ZERO(), b2 = ZERO(), b3 = ZERO();
            const size_t w0 = c * a->w_stride, w1 = w0 + a->w_stride;
            const size_t w2 = w1 + a->w_stride, w3 = w2 + a->w_stride;
            size_t i = 0;
            for (; i + (split ? 2 : 1) * LANES <= k; i += (split ? 2 : 1) * LANES) {
                VECTOR x = LOAD(xd, a->x, xo + i);
                a0 = FMA(x, LOAD(wd, a->weights, w0 + i), a0);
                a1 = FMA(x, LOAD(wd, a->weights, w1 + i), a1);
                a2 = FMA(x, LOAD(wd, a->weights, w2 + i), a2);
                a3 = FMA(x, LOAD(wd, a->weights, w3 + i), a3);
                if (split) {
                    VECTOR x1 = LOAD(xd, a->x, xo + i + LANES);
                    b0 = FMA(x1, LOAD(wd, a->weights, w0 + i + LANES), b0);
                    b1 = FMA(x1, LOAD(wd, a->weights, w1 + i + LANES), b1);
                    b2 = FMA(x1, LOAD(wd, a->weights, w2 + i + LANES), b2);
                    b3 = FMA(x1, LOAD(wd, a->weights, w3 + i + LANES), b3);
                }
            }
            if (!GEMV_F64 && split && i + LANES <= k) {
                VECTOR x = LOAD(xd, a->x, xo + i);
                a0 = FMA(x, LOAD(wd, a->weights, w0 + i), a0);
                a1 = FMA(x, LOAD(wd, a->weights, w1 + i), a1);
                a2 = FMA(x, LOAD(wd, a->weights, w2 + i), a2);
                a3 = FMA(x, LOAD(wd, a->weights, w3 + i), a3);
                i += LANES;
            }
            SCALAR out[4] = {SUM(split ? ADD(a0, b0) : a0), SUM(split ? ADD(a1, b1) : a1),
                             SUM(split ? ADD(a2, b2) : a2), SUM(split ? ADD(a3, b3) : a3)};
            for (; i < k; ++i) {
                SCALAR x = LOAD1(xd, a->x, xo + i);
                out[0] += x * LOAD1(wd, a->weights, w0 + i);
                out[1] += x * LOAD1(wd, a->weights, w1 + i);
                out[2] += x * LOAD1(wd, a->weights, w2 + i);
                out[3] += x * LOAD1(wd, a->weights, w3 + i);
            }
            NAME(write)(p, a, r, c, out, 4, p->desc.has_w_scale && a->w_scale ? a->w_scale + c : nullptr);
        }
        for (; c < c1; ++c) {
            VECTOR a0 = ZERO(), b0 = ZERO();
            const size_t wo = c * a->w_stride;
            size_t i = 0;
            for (; i + (split ? 2 : 1) * LANES <= k; i += (split ? 2 : 1) * LANES) {
                a0 = FMA(LOAD(xd, a->x, xo + i), LOAD(wd, a->weights, wo + i), a0);
                if (split)
                    b0 = FMA(LOAD(xd, a->x, xo + i + LANES),
                             LOAD(wd, a->weights, wo + i + LANES), b0);
            }
            if (!GEMV_F64 && split && i + LANES <= k) {
                a0 = FMA(LOAD(xd, a->x, xo + i), LOAD(wd, a->weights, wo + i), a0);
                i += LANES;
            }
            SCALAR out = SUM(split ? ADD(a0, b0) : a0);
            for (; i < k; ++i) out += LOAD1(xd, a->x, xo + i) * LOAD1(wd, a->weights, wo + i);
            NAME(write)(p, a, r, c, &out, 1, p->desc.has_w_scale && a->w_scale ? a->w_scale + c : nullptr);
        }
    }
}

[[gnu::always_inline]] static inline void NAME(packed_vectors)(
    size_t k, velri_dtype xd, velri_dtype wd, const void *x, size_t xo,
    const uint8_t *w0, const uint8_t *w1, SCALAR *out, bool pair, bool split) {
    (void)xd; (void)wd;
    VECTOR a0 = ZERO(), a1 = ZERO(), a2 = ZERO(), a3 = ZERO();
    VECTOR b0 = ZERO(), b1 = ZERO(), b2 = ZERO(), b3 = ZERO();
    size_t i = 0;
    for (; i + (split ? 2 : 1) <= k; i += split ? 2 : 1) {
        VECTOR x0 = BROADCAST(LOAD1(xd, x, xo + i));
        a0 = FMA(x0, LOAD(wd, w0, i * 2 * LANES), a0);
        a1 = FMA(x0, LOAD(wd, w0, i * 2 * LANES + LANES), a1);
        if (pair) {
            a2 = FMA(x0, LOAD(wd, w1, i * 2 * LANES), a2);
            a3 = FMA(x0, LOAD(wd, w1, i * 2 * LANES + LANES), a3);
        }
        if (split) {
            VECTOR x1 = BROADCAST(LOAD1(xd, x, xo + i + 1));
            b0 = FMA(x1, LOAD(wd, w0, (i + 1) * 2 * LANES), b0);
            b1 = FMA(x1, LOAD(wd, w0, (i + 1) * 2 * LANES + LANES), b1);
            if (pair) {
                b2 = FMA(x1, LOAD(wd, w1, (i + 1) * 2 * LANES), b2);
                b3 = FMA(x1, LOAD(wd, w1, (i + 1) * 2 * LANES + LANES), b3);
            }
        }
    }
    if (split) { a0 = ADD(a0, b0); a1 = ADD(a1, b1); a2 = ADD(a2, b2); a3 = ADD(a3, b3); }
    if (i < k) {
        VECTOR x0 = BROADCAST(LOAD1(xd, x, xo + i));
        a0 = FMA(x0, LOAD(wd, w0, i * 2 * LANES), a0);
        a1 = FMA(x0, LOAD(wd, w0, i * 2 * LANES + LANES), a1);
        if (pair) {
            a2 = FMA(x0, LOAD(wd, w1, i * 2 * LANES), a2);
            a3 = FMA(x0, LOAD(wd, w1, i * 2 * LANES + LANES), a3);
        }
    }
    STORE(out, a0); STORE(out + LANES, a1);
    if (pair) { STORE(out + 2 * LANES, a2); STORE(out + 3 * LANES, a3); }
}

[[gnu::always_inline]] static inline void NAME(prepared_impl)(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                    size_t r0, size_t r1, size_t c0, size_t c1,
                    velri_dtype xd, velri_dtype wd) {
    const size_t k = p->k, columns = p->columns;
    (void)xd; (void)wd;
    const uint8_t *packed = (const uint8_t *)a->weights;
    const size_t payload = p->k_aligned * columns * velri_dtype_bytes(wd);
    const size_t scale_offset = ((payload + 3) & ~(size_t)3) +
                                (wd == VELRI_DTYPE_I8 ? columns * sizeof(int32_t) : 0);
    const bool split = SIMD_GEMV_UNROLL == 2 && (GEMV_F64 || r1 - r0 == 1);
    for (size_t col = c0; col < c1;) {
        const bool pair = !GEMV_F64 && SIMD_GEMV_PAIR_TILES && r1 - r0 == 1 &&
                          columns == 2 * LANES && col + 2 * columns <= c1;
        const size_t tiles = pair ? 2 : 1;
        const size_t width = c1 - col < columns ? c1 - col : columns;
        const uint8_t *w0 = packed + (col / columns) * p->tile_stride_b;
        const uint8_t *w1 = pair ? w0 + p->tile_stride_b : w0;
        for (size_t r = r0; r < r1; ++r) {
            const size_t xo = r * a->x_stride;
            alignas(64) SCALAR out[4 * LANES] = {0};
            if (columns == 2 * LANES) {
                if (pair) {
                    NAME(packed_vectors)(k, xd, wd, a->x, xo, w0, w1, out, true, true);
                } else if (split) {
                    NAME(packed_vectors)(k, xd, wd, a->x, xo, w0, w0, out, false, true);
                } else {
                    NAME(packed_vectors)(k, xd, wd, a->x, xo, w0, w0, out, false, false);
                }
            } else {
                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < width; ++j)
                        out[j] += LOAD1(xd, a->x, xo + i) * LOAD1(wd, w0, i * columns + j);
            }
            const float *sc0 = p->desc.has_w_scale ? (const float *)(w0 + scale_offset) : nullptr;
            NAME(write)(p, a, r, col, out, width, sc0);
            if (pair) {
                const float *sc1 = p->desc.has_w_scale ? (const float *)(w1 + scale_offset) : nullptr;
                NAME(write)(p, a, r, col + columns, out + 2 * LANES, columns, sc1);
            }
        }
        col += tiles * columns;
    }
}

/* Specialize input/weight formats before entering any reduction loop. */
[[gnu::always_inline]] static inline void NAME(dispatch)(
    const velri_linear_typed_plan *p, const velri_linear_run_args *a,
    size_t r0, size_t r1, size_t c0, size_t c1, bool prepared) {
#define RUN(xd, wd) do { \
    if (prepared) NAME(prepared_impl)(p, a, r0, r1, c0, c1, xd, wd); \
    else NAME(rowmajor_impl)(p, a, r0, r1, c0, c1, xd, wd); \
} while (0)
#if GEMV_F64
    RUN(VELRI_DTYPE_F64, VELRI_DTYPE_F64);
#else
    switch (p->desc.w_dtype) {
    case VELRI_DTYPE_F16:
        if (p->desc.x_dtype == VELRI_DTYPE_F16) { RUN(VELRI_DTYPE_F16, VELRI_DTYPE_F16); }
        else { RUN(VELRI_DTYPE_F32, VELRI_DTYPE_F16); }
        break;
    case VELRI_DTYPE_BF16:
        if (p->desc.x_dtype == VELRI_DTYPE_BF16) { RUN(VELRI_DTYPE_BF16, VELRI_DTYPE_BF16); }
        else { RUN(VELRI_DTYPE_F32, VELRI_DTYPE_BF16); }
        break;
    case VELRI_DTYPE_I8: RUN(VELRI_DTYPE_F32, VELRI_DTYPE_I8); break;
    default: RUN(VELRI_DTYPE_F32, VELRI_DTYPE_F32); break;
    }
#endif
#undef RUN
}

void NAME(rowmajor)(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                    size_t r0, size_t r1, size_t c0, size_t c1) {
    NAME(dispatch)(p, a, r0, r1, c0, c1, false);
}
void NAME(prepared)(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                    size_t r0, size_t r1, size_t c0, size_t c1) {
    NAME(dispatch)(p, a, r0, r1, c0, c1, true);
}

#undef SCALAR
#undef VECTOR
#undef LANES
#undef NAME
#undef ZERO
#undef BROADCAST
#undef ADD
#undef FMA
#undef LOAD
#undef LOAD1
#undef STORE
#undef SUM
