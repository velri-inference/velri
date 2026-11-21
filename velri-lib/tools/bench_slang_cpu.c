#include "os.h"
#include "velri_lib/kernels.h"
#include "velri_lib/slang_cpu.h"
#include "reference.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && VELRI_X86_VARIANTS && !defined(_MSC_VER)
#include <immintrin.h>
#endif

static void die(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(1);
}

static void check_ok(velri_status st, const char *ctx) {
    if (st != VELRI_OK) {
        fprintf(stderr, "ERROR (%s): velri_status=%d\n", ctx, (int)st);
        exit(1);
    }
}

static void *xalloc(size_t bytes) {
    size_t sz = bytes ? ((bytes + 63u) & ~(size_t)63u) : 64u;
    void *p = velri_os_aligned_alloc(64u, sz);
    if (!p) die("aligned allocation failed");
    memset(p, 0, sz);
    return p;
}

static void xfree(void *p) { velri_os_aligned_free(p); }

static double now_sec(void) { return (double)velri_os_monotonic_ns() * 1e-9; }

static int cmp_dbl(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_of(double *arr, size_t n) {
    qsort(arr, n, sizeof(double), cmp_dbl);
    return arr[n / 2u];
}

static void fill_dtype(velri_dtype dt, void *buf, size_t n, uint64_t *rng, float scale) {
    if (dt == VELRI_DTYPE_F32) {
        float *p = (float *)buf;
        for (size_t i = 0; i < n; ++i) p[i] = velri_test_rand_f32(rng, -scale, scale);
    } else if (dt == VELRI_DTYPE_F64) {
        double *p = (double *)buf;
        for (size_t i = 0; i < n; ++i) p[i] = (double)velri_test_rand_f32(rng, -scale, scale);
    } else if (dt == VELRI_DTYPE_I8) {
        int8_t *p = (int8_t *)buf;
        for (size_t i = 0; i < n; ++i) p[i] = (int8_t)((int)(velri_test_rng(rng) % 251u) - 125);
    } else if (dt == VELRI_DTYPE_F16) {
        velri_fp16 *p = (velri_fp16 *)buf;
        for (size_t i = 0; i < n; ++i) {
            p[i] = velri_f32_to_f16_scalar(velri_test_rand_f32(rng, -scale, scale));
        }
    } else if (dt == VELRI_DTYPE_BF16) {
        velri_bf16 *p = (velri_bf16 *)buf;
        for (size_t i = 0; i < n; ++i) {
            p[i] = velri_f32_to_bf16_scalar(velri_test_rand_f32(rng, -scale, scale));
        }
    }
}

static double load_elem_f64(velri_dtype dt, const void *p, size_t idx) {
    if (dt == VELRI_DTYPE_F32) return (double)((const float *)p)[idx];
    if (dt == VELRI_DTYPE_F64) return ((const double *)p)[idx];
    if (dt == VELRI_DTYPE_BF16) return (double)velri_bf16_to_f32_scalar(((const velri_bf16 *)p)[idx]);
    if (dt == VELRI_DTYPE_F16) return (double)velri_f16_to_f32_scalar(((const velri_fp16 *)p)[idx]);
    return NAN;
}

static double max_abs_diff_2d(velri_dtype dt, const void *a, const void *b,
                              size_t rows, size_t cols, size_t stride) {
    double mx = 0.0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            double va = load_elem_f64(dt, a, r * stride + c);
            double vb = load_elem_f64(dt, b, r * stride + c);
            if (!isfinite(va) || !isfinite(vb)) return INFINITY;
            double d = fabs(va - vb);
            if (d > mx) mx = d;
        }
    }
    return mx;
}

/* ============================================================================
 * Part 1: Exhaustive Correctness Verification vs Independent C23 Reference
 * ============================================================================ */
static void verify_linear_case(const char *label, velri_linear_desc desc,
                               velri_cpu_variant var, velri_pool *pool, size_t nw,
                               size_t rows, size_t m, size_t k, bool with_bias, double tol) {
    size_t xs = k + 5u, ws = k + 7u, ys = m + 3u;
    size_t x_b = velri_dtype_bytes(desc.x_dtype);
    size_t w_b = velri_dtype_bytes(desc.w_dtype);
    size_t y_b = velri_dtype_bytes(desc.y_dtype);

    void *x = xalloc(rows * xs * x_b);
    void *w = xalloc(m * ws * w_b);
    void *y_ref = xalloc(rows * ys * y_b);
    void *y_sl = xalloc(rows * ys * y_b);
    void *bias = with_bias ? xalloc(m * y_b) : nullptr;
    float *xscale = desc.has_x_scale ? xalloc(rows * sizeof(float)) : nullptr;
    float *wscale = desc.has_w_scale ? xalloc(m * sizeof(float)) : nullptr;

    uint64_t rng = 0x123456789ABCDEF0ULL ^ ((uint64_t)rows << 16) ^ ((uint64_t)var << 8);
    fill_dtype(desc.x_dtype, x, rows * xs, &rng, 0.5f);
    fill_dtype(desc.w_dtype, w, m * ws, &rng, 0.5f);
    if (bias) fill_dtype(desc.y_dtype, bias, m, &rng, 0.5f);
    if (xscale) fill_dtype(VELRI_DTYPE_F32, xscale, rows, &rng, 0.05f);
    if (wscale) fill_dtype(VELRI_DTYPE_F32, wscale, m, &rng, 0.05f);

    ref_linear_run(&desc, rows, m, k, x, xs, xscale, w, ws, wscale, bias, y_ref, ys);
    check_ok(velri_slang_cpu_linear(&desc, rows, m, k, x, xs, w, ws, bias, xscale, wscale,
                                    y_sl, ys, var, pool, nw),
             label);

    double err = max_abs_diff_2d(desc.y_dtype, y_ref, y_sl, rows, m, ys);
    if (!(err <= tol)) {
        fprintf(stderr, "FAIL %s (isa=%d rows=%zu m=%zu k=%zu bias=%d): max_err=%.3e > tol=%.3e\n",
                label, (int)var, rows, m, k, (int)with_bias, err, tol);
        exit(1);
    }

    xfree(x); xfree(w); xfree(y_ref); xfree(y_sl);
    xfree(bias); xfree(xscale); xfree(wscale);
}

static void run_verification(velri_pool *pool, size_t nw) {
    velri_cpu_variant vars[3] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    const char *vnames[3] = {"base", "avx2", "avx512"};
    size_t nvars = 1;
#if VELRI_X86_VARIANTS
    nvars = 3;
#endif

    printf("=== Verifying Slang CPU kernels vs independent C23 reference ===\n");
    const size_t row_cases[4] = {1u, 2u, 7u, 13u};

    for (size_t vi = 0; vi < nvars; ++vi) {
        velri_cpu_variant var = vars[vi];
        if (!velri_cpu_variant_supported(var)) continue;

        for (size_t ri = 0; ri < 4u; ++ri) {
            size_t rows = row_cases[ri];
            for (int b = 0; b <= 1; ++b) {
                verify_linear_case("linear_f32", velri_desc_f32(), var, pool, nw, rows, 67u, 131u, b != 0, 1e-4);
                verify_linear_case("linear_f64", velri_desc_f64(), var, pool, nw, rows, 67u, 131u, b != 0, 1e-11);
                verify_linear_case("linear_f16", velri_desc_f16(VELRI_DTYPE_F32), var, pool, nw, rows, 67u, 131u, b != 0, 2e-4);
                verify_linear_case("linear_bf16", velri_desc_bf16(VELRI_DTYPE_F32), var, pool, nw, rows, 67u, 131u, b != 0, 2e-4);
                verify_linear_case("linear_f32x_bf16w", velri_desc_f32x_bf16w(), var, pool, nw, rows, 67u, 131u, b != 0, 2e-4);
                verify_linear_case("linear_w8a8", velri_desc_w8a8_f32(), var, pool, nw, rows, 67u, 131u, b != 0, 1e-4);
            }
        }

        /* RMSNorm (F32, BF16, F16, unit_offset false/true, with/without weight) */
        const velri_dtype norm_dts[3] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F16};
        const double norm_tols[3] = {2e-5, 2e-2, 4e-3};
        for (int di = 0; di < 3; ++di) {
            velri_dtype dt = norm_dts[di];
            size_t eb = velri_dtype_bytes(dt);
            size_t rows = 7u, dim = 131u, xs = 137u, ys = 139u;
            void *x = xalloc(rows * xs * eb);
            void *w = xalloc(dim * eb);
            void *y_ref = xalloc(rows * ys * eb);
            void *y_sl = xalloc(rows * ys * eb);
            uint64_t rng = 0xCAFE0001ULL + (uint64_t)di;
            fill_dtype(dt, x, rows * xs, &rng, 2.0f);
            fill_dtype(dt, w, dim, &rng, 1.5f);
            for (int uo = 0; uo <= 1; ++uo) {
                for (int hw = 0; hw <= 1; ++hw) {
                    const void *wp = hw ? w : nullptr;
                    ref_rmsnorm(dt, dt, rows, dim, 1e-5f, uo != 0, x, xs, wp, y_ref, ys);
                    check_ok(velri_slang_cpu_rmsnorm(dt, rows, dim, 1e-5f, uo != 0,
                                                     x, xs, wp, y_sl, ys, var, pool, nw),
                             "rmsnorm");
                    double err = max_abs_diff_2d(dt, y_ref, y_sl, rows, dim, ys);
                    if (!(err <= norm_tols[di])) {
                        fprintf(stderr, "FAIL rmsnorm dt=%d isa=%s err=%.3e\n", (int)dt, vnames[vi], err);
                        exit(1);
                    }
                }
            }
            xfree(x); xfree(w); xfree(y_ref); xfree(y_sl);
        }

        /* RoPE (F32, BF16, HALF & INTERLEAVED, partial rotary_dim) */
        const velri_dtype rope_dts[2] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16};
        const double rope_tols[2] = {1e-5, 2e-2};
        for (int di = 0; di < 2; ++di) {
            velri_dtype dt = rope_dts[di];
            size_t eb = velri_dtype_bytes(dt);
            size_t nt = 7u, nh = 5u, hd = 72u, rd = 48u, po = 5u;
            size_t hsx = hd + 4u, tsx = nh * hsx + 3u;
            size_t hsy = hd + 2u, tsy = nh * hsy + 5u;
            void *x = xalloc(nt * tsx * eb);
            void *y_ref = xalloc(nt * tsy * eb);
            void *y_sl = xalloc(nt * tsy * eb);
            float *ct = xalloc((po + nt) * (rd / 2u) * sizeof(float));
            float *st = xalloc((po + nt) * (rd / 2u) * sizeof(float));
            uint64_t rng = 0xBEEF0001ULL + (uint64_t)di;
            fill_dtype(dt, x, nt * tsx, &rng, 2.0f);
            for (size_t i = 0; i < (po + nt) * (rd / 2u); ++i) {
                float ang = velri_test_rand_f32(&rng, -3.14f, 3.14f);
                ct[i] = cosf(ang);
                st[i] = sinf(ang);
            }
            for (int m = 0; m <= 1; ++m) {
                velri_rope_mode mode = m ? VELRI_ROPE_INTERLEAVED : VELRI_ROPE_HALF;
                ref_rope(dt, nt, nh, hd, rd, po, x, tsx, hsx, ct, st, y_ref, tsy, hsy, mode);
                check_ok(velri_slang_cpu_rope(dt, nt, nh, hd, rd, po, x, tsx, hsx, ct, st,
                                              y_sl, tsy, hsy, mode, var, pool, nw),
                         "rope");
                double err = 0.0;
                for (size_t t = 0; t < nt; ++t) {
                    for (size_t h = 0; h < nh; ++h) {
                        for (size_t d = 0; d < hd; ++d) {
                            size_t off = t * tsy + h * hsy + d;
                            double diff = fabs(load_elem_f64(dt, y_ref, off) - load_elem_f64(dt, y_sl, off));
                            if (diff > err) err = diff;
                        }
                    }
                }
                if (!(err <= rope_tols[di])) {
                    fprintf(stderr, "FAIL rope dt=%d mode=%d isa=%s err=%.3e\n", (int)dt, m, vnames[vi], err);
                    exit(1);
                }
            }
            xfree(x); xfree(y_ref); xfree(y_sl); xfree(ct); xfree(st);
        }

        /* GatedMLP (SwiGLU & GeGLU, clamped and unclamped) */
        {
            size_t n = 1031u;
            float *g = xalloc(n * sizeof(float));
            float *u = xalloc(n * sizeof(float));
            float *y_ref = xalloc(n * sizeof(float));
            float *y_sl = xalloc(n * sizeof(float));
            uint64_t rng = 0x77771234ULL;
            fill_dtype(VELRI_DTYPE_F32, g, n, &rng, 4.0f);
            fill_dtype(VELRI_DTYPE_F32, u, n, &rng, 2.0f);
            const velri_act_kind acts[2] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH};
            const float clamps[2] = {0.0f, 1.5f};
            for (int ai = 0; ai < 2; ++ai) {
                for (int ci = 0; ci < 2; ++ci) {
                    ref_gated_mlp_f32(acts[ai], n, clamps[ci], g, u, y_ref);
                    check_ok(velri_slang_cpu_gated_mlp_f32(acts[ai], n, clamps[ci], g, u, y_sl, var, pool, nw),
                             "gated_mlp");
                    double err = max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, 1u, n, n);
                    if (!(err <= 5e-4)) {
                        fprintf(stderr, "FAIL gated_mlp act=%d clamp=%.1f isa=%s err=%.3e\n",
                                ai, clamps[ci], vnames[vi], err);
                        exit(1);
                    }
                }
            }
            xfree(g); xfree(u); xfree(y_ref); xfree(y_sl);
        }

        /* Softmax */
        {
            size_t rows = 7u, cols = 131u, xs = 137u, ys = 139u;
            float *x = xalloc(rows * xs * sizeof(float));
            float *y_ref = xalloc(rows * ys * sizeof(float));
            float *y_sl = xalloc(rows * ys * sizeof(float));
            uint64_t rng = 0x9999ABCDULL;
            fill_dtype(VELRI_DTYPE_F32, x, rows * xs, &rng, 20.0f);
            ref_softmax_f32(rows, cols, x, xs, y_ref, ys);
            check_ok(velri_slang_cpu_softmax_f32(rows, cols, x, xs, y_sl, ys, var, pool, nw), "softmax");
            double err = max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, rows, cols, ys);
            if (!(err <= 1e-4)) {
                fprintf(stderr, "FAIL softmax isa=%s err=%.3e\n", vnames[vi], err);
                exit(1);
            }
            xfree(x); xfree(y_ref); xfree(y_sl);
        }

        /* FusedAddRMSNorm */
        {
            size_t rows = 7u, dim = 131u, rs = 139u, xs = 137u, ys = 141u;
            float *res0 = xalloc(rows * rs * sizeof(float));
            float *res_ref = xalloc(rows * rs * sizeof(float));
            float *res_sl = xalloc(rows * rs * sizeof(float));
            float *x = xalloc(rows * xs * sizeof(float));
            float *w = xalloc(dim * sizeof(float));
            float *y_ref = xalloc(rows * ys * sizeof(float));
            float *y_sl = xalloc(rows * ys * sizeof(float));
            uint64_t rng = 0xFA001122ULL;
            fill_dtype(VELRI_DTYPE_F32, res0, rows * rs, &rng, 2.0f);
            fill_dtype(VELRI_DTYPE_F32, x, rows * xs, &rng, 2.0f);
            fill_dtype(VELRI_DTYPE_F32, w, dim, &rng, 1.5f);
            for (int uo = 0; uo <= 1; ++uo) {
                for (int hw = 0; hw <= 1; ++hw) {
                    memcpy(res_ref, res0, rows * rs * sizeof(float));
                    memcpy(res_sl, res0, rows * rs * sizeof(float));
                    const float *wp = hw ? w : nullptr;
                    ref_fused_add_rmsnorm_f32(rows, dim, 1e-5f, uo != 0, res_ref, rs, x, xs, wp, y_ref, ys);
                    check_ok(velri_slang_cpu_fused_add_rmsnorm_f32(rows, dim, 1e-5f, uo != 0,
                                                                   res_sl, rs, x, xs, wp, y_sl, ys, var, pool, nw),
                             "fused_add_rmsnorm");
                    double er = max_abs_diff_2d(VELRI_DTYPE_F32, res_ref, res_sl, rows, dim, rs);
                    double ey = max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, rows, dim, ys);
                    if (!(er <= 1e-5 && ey <= 2e-5)) {
                        fprintf(stderr, "FAIL fused_add_rmsnorm isa=%s er=%.3e ey=%.3e\n", vnames[vi], er, ey);
                        exit(1);
                    }
                }
            }
            xfree(res0); xfree(res_ref); xfree(res_sl); xfree(x); xfree(w); xfree(y_ref); xfree(y_sl);
        }

        /* LayerNorm */
        {
            size_t rows = 7u, dim = 131u, xs = 137u, ys = 139u;
            float *x = xalloc(rows * xs * sizeof(float));
            float *w = xalloc(dim * sizeof(float));
            float *b = xalloc(dim * sizeof(float));
            float *y_ref = xalloc(rows * ys * sizeof(float));
            float *y_sl = xalloc(rows * ys * sizeof(float));
            uint64_t rng = 0x1A7E0001ULL;
            fill_dtype(VELRI_DTYPE_F32, x, rows * xs, &rng, 3.0f);
            fill_dtype(VELRI_DTYPE_F32, w, dim, &rng, 1.5f);
            fill_dtype(VELRI_DTYPE_F32, b, dim, &rng, 0.5f);
            for (int hw = 0; hw <= 1; ++hw) {
                for (int hb = 0; hb <= 1; ++hb) {
                    const float *wp = hw ? w : nullptr;
                    const float *bp = hb ? b : nullptr;
                    ref_layernorm_f32(rows, dim, 1e-5f, x, xs, wp, bp, y_ref, ys);
                    check_ok(velri_slang_cpu_layernorm_f32(rows, dim, 1e-5f, x, xs, wp, bp, y_sl, ys, var, pool, nw),
                             "layernorm");
                    double err = max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, rows, dim, ys);
                    if (!(err <= 3e-5)) {
                        fprintf(stderr, "FAIL layernorm hw=%d hb=%d isa=%s err=%.3e\n", hw, hb, vnames[vi], err);
                        exit(1);
                    }
                }
            }
            xfree(x); xfree(w); xfree(b); xfree(y_ref); xfree(y_sl);
        }

        /* Standalone Activations (SiLU, GELU_tanh, ReLU2) */
        {
            size_t n = 1031u;
            float *x = xalloc(n * sizeof(float));
            float *y_ref = xalloc(n * sizeof(float));
            float *y_sl = xalloc(n * sizeof(float));
            uint64_t rng = 0xAC710001ULL;
            fill_dtype(VELRI_DTYPE_F32, x, n, &rng, 5.0f);
            const velri_act_kind kinds[3] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH, VELRI_ACT_RELU2};
            for (int ki = 0; ki < 3; ++ki) {
                ref_activation_f32(kinds[ki], n, x, y_ref);
                check_ok(velri_slang_cpu_activation_f32(kinds[ki], n, x, y_sl, var, pool, nw), "activation");
                double err = max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, 1u, n, n);
                if (!(err <= 2e-4)) {
                    fprintf(stderr, "FAIL activation kind=%d isa=%s err=%.3e\n", ki, vnames[vi], err);
                    exit(1);
                }
            }
            xfree(x); xfree(y_ref); xfree(y_sl);
        }

        /* vec_axpby & vec_mul */
        {
            size_t n = 1031u;
            float *a = xalloc(n * sizeof(float));
            float *b = xalloc(n * sizeof(float));
            float *y_ref = xalloc(n * sizeof(float));
            float *y_sl = xalloc(n * sizeof(float));
            uint64_t rng = 0x5E00AA11ULL;
            fill_dtype(VELRI_DTYPE_F32, a, n, &rng, 3.0f);
            fill_dtype(VELRI_DTYPE_F32, b, n, &rng, 3.0f);
            ref_vec_axpby_f32(n, 1.35f, a, -0.75f, b, y_ref);
            check_ok(velri_slang_cpu_vec_axpby_f32(n, 1.35f, a, -0.75f, b, y_sl, var, pool, nw), "axpby");
            if (!(max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, 1u, n, n) <= 1e-5)) {
                fprintf(stderr, "FAIL vec_axpby isa=%s\n", vnames[vi]);
                exit(1);
            }
            ref_vec_axpby_f32(n, 2.25f, a, 0.0f, nullptr, y_ref);
            check_ok(velri_slang_cpu_vec_axpby_f32(n, 2.25f, a, 0.0f, nullptr, y_sl, var, pool, nw), "axpby0");
            if (!(max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, 1u, n, n) <= 1e-5)) {
                fprintf(stderr, "FAIL vec_axpby(beta=0) isa=%s\n", vnames[vi]);
                exit(1);
            }
            ref_vec_mul_f32(n, a, b, y_ref);
            check_ok(velri_slang_cpu_vec_mul_f32(n, a, b, y_sl, var, pool, nw), "mul");
            if (!(max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, 1u, n, n) <= 1e-5)) {
                fprintf(stderr, "FAIL vec_mul isa=%s\n", vnames[vi]);
                exit(1);
            }
            xfree(a); xfree(b); xfree(y_ref); xfree(y_sl);
        }

        /* FWHT (B in {16, 32, 64, 128, 256} x suh in {none, F32, F16}) */
        {
            const size_t bs[5] = {16u, 32u, 64u, 128u, 256u};
            size_t rows = 5u, cols = 512u, xs = 521u, ys = 523u;
            float *x = xalloc(rows * xs * sizeof(float));
            float *suh32 = xalloc(cols * sizeof(float));
            velri_fp16 *suh16 = xalloc(cols * sizeof(velri_fp16));
            float *y_ref = xalloc(rows * ys * sizeof(float));
            float *y_sl = xalloc(rows * ys * sizeof(float));
            uint64_t rng = 0xF1170001ULL;
            fill_dtype(VELRI_DTYPE_F32, x, rows * xs, &rng, 2.0f);
            fill_dtype(VELRI_DTYPE_F32, suh32, cols, &rng, 1.5f);
            fill_dtype(VELRI_DTYPE_F16, suh16, cols, &rng, 1.5f);
            for (int bi = 0; bi < 5; ++bi) {
                for (int sm = 0; sm < 3; ++sm) {
                    velri_dtype sdt = (sm == 2) ? VELRI_DTYPE_F16 : VELRI_DTYPE_F32;
                    const void *sp = (sm == 0) ? nullptr : ((sm == 1) ? (const void *)suh32 : (const void *)suh16);
                    ref_fwht_f32(rows, cols, bs[bi], x, xs, sdt, sp, y_ref, ys);
                    check_ok(velri_slang_cpu_fwht_f32(rows, cols, bs[bi], x, xs, sdt, sp, y_sl, ys, var, pool, nw),
                             "fwht");
                    double err = max_abs_diff_2d(VELRI_DTYPE_F32, y_ref, y_sl, rows, cols, ys);
                    if (!(err <= 5e-5)) {
                        fprintf(stderr, "FAIL fwht B=%zu sm=%d isa=%s err=%.3e\n", bs[bi], sm, vnames[vi], err);
                        exit(1);
                    }
                }
            }
            xfree(x); xfree(suh32); xfree(suh16); xfree(y_ref); xfree(y_sl);
        }

        /* QuantRow F32 -> I8 (including all-zero row and odd stride) */
        {
            size_t rows = 7u, k = 131u, xs = 137u, qs = 139u;
            float *x = xalloc(rows * xs * sizeof(float));
            int8_t *q_ref = xalloc(rows * qs);
            int8_t *q_sl = xalloc(rows * qs);
            float *s_ref = xalloc(rows * sizeof(float));
            float *s_sl = xalloc(rows * sizeof(float));
            uint64_t rng = 0x0A0B0C0DULL;
            fill_dtype(VELRI_DTYPE_F32, x, rows * xs, &rng, 4.0f);
            memset(x + 3u * xs, 0, k * sizeof(float)); /* all-zero row */
            ref_quantize_row_f32_i8(rows, k, x, xs, q_ref, qs, s_ref);
            check_ok(velri_slang_cpu_quantize_row_f32_i8(rows, k, x, xs, q_sl, qs, s_sl, var, pool, nw), "qsl");
            for (size_t r = 0; r < rows; ++r) {
                if (!(fabsf(s_ref[r] - s_sl[r]) <= 1e-6f)) {
                    fprintf(stderr, "FAIL quantize_row scale r=%zu isa=%s\n", r, vnames[vi]);
                    exit(1);
                }
                for (size_t c = 0; c < k; ++c) {
                    int diff = (int)q_ref[r * qs + c] - (int)q_sl[r * qs + c];
                    if (diff < -1 || diff > 1) {
                        fprintf(stderr, "FAIL quantize_row r=%zu c=%zu isa=%s ref=%d got=%d\n",
                                r, c, vnames[vi], (int)q_ref[r * qs + c], (int)q_sl[r * qs + c]);
                        exit(1);
                    }
                }
            }
            xfree(x); xfree(q_ref); xfree(q_sl); xfree(s_ref); xfree(s_sl);
        }

        printf("  [PASS] ISA=%-6s — all 23 Slang CPU kernels match C23 reference within tolerance.\n",
               vnames[vi]);
    }
}

/* ============================================================================
 * Part 1.5: Direct Physical Silicon Ceiling Calibration on This CPU
 * ============================================================================ */
typedef struct {
    double bw_read_gbps;
    double bw_copy_gbps;
    double fma_f32_gflops;
    double fma_f64_gflops;
} cpu_ceilings;

typedef struct {
    const float *src;
    float *dst;
    size_t n;
    volatile double sink;
} cpu_bw_ctx;

static void bw_read_worker(void *v, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 16u) * wid / nw) * 16u;
    size_t i1 = ((c->n / 16u) * (wid + 1u) / nw) * 16u;
    const float *p = c->src;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (size_t i = i0; i < i1; i += 16u) {
        s0 += p[i + 0u] + p[i + 4u] + p[i + 8u] + p[i + 12u];
        s1 += p[i + 1u] + p[i + 5u] + p[i + 9u] + p[i + 13u];
        s2 += p[i + 2u] + p[i + 6u] + p[i + 10u] + p[i + 14u];
        s3 += p[i + 3u] + p[i + 7u] + p[i + 11u] + p[i + 15u];
    }
    if (wid == 0u) c->sink = (double)(s0 + s1 + s2 + s3);
}

#if defined(__x86_64__) && VELRI_X86_VARIANTS && !defined(_MSC_VER)
__attribute__((target("avx512f")))
static void bw_copy_nt_worker(void *v, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 64u) * wid / nw) * 64u;
    size_t i1 = ((c->n / 64u) * (wid + 1u) / nw) * 64u;
    const float *s = c->src;
    float *d = c->dst;
    for (size_t i = i0; i < i1; i += 64u) {
        __m512 v0 = _mm512_load_ps(s + i + 0u);
        __m512 v1 = _mm512_load_ps(s + i + 16u);
        __m512 v2 = _mm512_load_ps(s + i + 32u);
        __m512 v3 = _mm512_load_ps(s + i + 48u);
        _mm512_stream_ps(d + i + 0u, v0);
        _mm512_stream_ps(d + i + 16u, v1);
        _mm512_stream_ps(d + i + 32u, v2);
        _mm512_stream_ps(d + i + 48u, v3);
    }
    _mm_sfence();
}
#endif

static void bw_copy_worker(void *v, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)sc; (void)scb;
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 16u) * wid / nw) * 16u;
    size_t i1 = ((c->n / 16u) * (wid + 1u) / nw) * 16u;
    memcpy(c->dst + i0, c->src + i0, (i1 - i0) * sizeof(float));
}

#if defined(__x86_64__) && VELRI_X86_VARIANTS && !defined(_MSC_VER)
typedef struct { size_t iters; volatile double sink; } cpu_fma_ctx;

__attribute__((target("avx512f,fma")))
static void ceil_fma_f32_worker(void *v, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)nw; (void)sc; (void)scb;
    cpu_fma_ctx *c = (cpu_fma_ctx *)v;
    __m512 a = _mm512_set1_ps(1.000001f + (float)wid * 1e-6f);
    __m512 b = _mm512_set1_ps(0.999999f);
    __m512 c0 = _mm512_set1_ps(1.f), c1 = _mm512_set1_ps(2.f), c2 = _mm512_set1_ps(3.f), c3 = _mm512_set1_ps(4.f);
    __m512 c4 = _mm512_set1_ps(5.f), c5 = _mm512_set1_ps(6.f), c6 = _mm512_set1_ps(7.f), c7 = _mm512_set1_ps(8.f);
    __m512 c8 = _mm512_set1_ps(9.f), c9 = _mm512_set1_ps(10.f), c10 = _mm512_set1_ps(11.f), c11 = _mm512_set1_ps(12.f);
    for (size_t i = 0; i < c->iters; ++i) {
        c0 = _mm512_fmadd_ps(c0, b, a); c1 = _mm512_fmadd_ps(c1, b, a);
        c2 = _mm512_fmadd_ps(c2, b, a); c3 = _mm512_fmadd_ps(c3, b, a);
        c4 = _mm512_fmadd_ps(c4, b, a); c5 = _mm512_fmadd_ps(c5, b, a);
        c6 = _mm512_fmadd_ps(c6, b, a); c7 = _mm512_fmadd_ps(c7, b, a);
        c8 = _mm512_fmadd_ps(c8, b, a); c9 = _mm512_fmadd_ps(c9, b, a);
        c10 = _mm512_fmadd_ps(c10, b, a); c11 = _mm512_fmadd_ps(c11, b, a);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3)),
                             _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(c4, c5), _mm512_add_ps(c6, c7)),
                                           _mm512_add_ps(_mm512_add_ps(c8, c9), _mm512_add_ps(c10, c11))));
    if (wid == 0u) c->sink = (double)_mm512_reduce_add_ps(s);
}

__attribute__((target("avx512f,fma")))
static void ceil_fma_f64_worker(void *v, size_t wid, size_t nw, void *sc, size_t scb) {
    (void)nw; (void)sc; (void)scb;
    cpu_fma_ctx *c = (cpu_fma_ctx *)v;
    __m512d a = _mm512_set1_pd(1.000001 + (double)wid * 1e-6);
    __m512d b = _mm512_set1_pd(0.999999);
    __m512d c0 = _mm512_set1_pd(1.), c1 = _mm512_set1_pd(2.), c2 = _mm512_set1_pd(3.), c3 = _mm512_set1_pd(4.);
    __m512d c4 = _mm512_set1_pd(5.), c5 = _mm512_set1_pd(6.), c6 = _mm512_set1_pd(7.), c7 = _mm512_set1_pd(8.);
    __m512d c8 = _mm512_set1_pd(9.), c9 = _mm512_set1_pd(10.), c10 = _mm512_set1_pd(11.), c11 = _mm512_set1_pd(12.);
    for (size_t i = 0; i < c->iters; ++i) {
        c0 = _mm512_fmadd_pd(c0, b, a); c1 = _mm512_fmadd_pd(c1, b, a);
        c2 = _mm512_fmadd_pd(c2, b, a); c3 = _mm512_fmadd_pd(c3, b, a);
        c4 = _mm512_fmadd_pd(c4, b, a); c5 = _mm512_fmadd_pd(c5, b, a);
        c6 = _mm512_fmadd_pd(c6, b, a); c7 = _mm512_fmadd_pd(c7, b, a);
        c8 = _mm512_fmadd_pd(c8, b, a); c9 = _mm512_fmadd_pd(c9, b, a);
        c10 = _mm512_fmadd_pd(c10, b, a); c11 = _mm512_fmadd_pd(c11, b, a);
    }
    __m512d s = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(c0, c1), _mm512_add_pd(c2, c3)),
                              _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(c4, c5), _mm512_add_pd(c6, c7)),
                                            _mm512_add_pd(_mm512_add_pd(c8, c9), _mm512_add_pd(c10, c11))));
    if (wid == 0u) c->sink = _mm512_reduce_add_pd(s);
}
#endif

static cpu_ceilings measure_cpu_ceilings(velri_pool *pool, size_t nw) {
    cpu_ceilings out = {0};
    /* 1. Cold-DRAM streaming read-only BW (512 MiB) & copy BW (256 MiB -> 256 MiB) */
    size_t n_rd = (512u * 1024u * 1024u) / sizeof(float);
    float *buf_a = xalloc(n_rd * sizeof(float));
    float *buf_b = xalloc((n_rd / 2u) * sizeof(float));
    for (size_t i = 0; i < n_rd; i += 1024u) buf_a[i] = 1.0f;
    for (size_t i = 0; i < n_rd / 2u; i += 1024u) buf_b[i] = 1.0f;

    cpu_bw_ctx bctx = { .src = buf_a, .dst = buf_b, .n = n_rd };
    velri_pool_parallel(pool, nw, bw_read_worker, &bctx);
    double best_rd = 0.0;
    for (int s = 0; s < 5; ++s) {
        double t0 = now_sec();
        velri_pool_parallel(pool, nw, bw_read_worker, &bctx);
        double gb = ((double)n_rd * sizeof(float)) / (now_sec() - t0) / 1e9;
        if (gb > best_rd) best_rd = gb;
    }
    out.bw_read_gbps = best_rd;

    bctx.n = n_rd / 2u;
    void (*cp_fn)(void *, size_t, size_t, void *, size_t) = bw_copy_worker;
#if defined(__x86_64__) && VELRI_X86_VARIANTS && !defined(_MSC_VER)
    if (velri_get_cpu_info()->has_avx512f) cp_fn = bw_copy_nt_worker;
#endif
    velri_pool_parallel(pool, nw, cp_fn, &bctx);
    double best_cp = 0.0;
    for (int s = 0; s < 5; ++s) {
        double t0 = now_sec();
        velri_pool_parallel(pool, nw, cp_fn, &bctx);
        double gb = (2.0 * (double)bctx.n * sizeof(float)) / (now_sec() - t0) / 1e9;
        if (gb > best_cp) best_cp = gb;
    }
    out.bw_copy_gbps = best_cp;
    xfree(buf_a);
    xfree(buf_b);

#if defined(__x86_64__) && VELRI_X86_VARIANTS && !defined(_MSC_VER)
    if (velri_get_cpu_info()->has_avx512f) {
        cpu_fma_ctx fctx = { .iters = 3000000u };
        /* Warmup until all physical cores hit boost clocks */
        for (int w = 0; w < 6; ++w) velri_pool_parallel(pool, nw, ceil_fma_f32_worker, &fctx);
        double best_f32 = 0.0;
        for (int s = 0; s < 7; ++s) {
            double t0 = now_sec();
            velri_pool_parallel(pool, nw, ceil_fma_f32_worker, &fctx);
            double gf = (double)nw * (double)fctx.iters * 12.0 * 32.0 / (now_sec() - t0) / 1e9;
            if (gf > best_f32) best_f32 = gf;
        }
        out.fma_f32_gflops = best_f32;

        velri_pool_parallel(pool, nw, ceil_fma_f64_worker, &fctx);
        double best_f64 = 0.0;
        for (int s = 0; s < 7; ++s) {
            double t0 = now_sec();
            velri_pool_parallel(pool, nw, ceil_fma_f64_worker, &fctx);
            double gf = (double)nw * (double)fctx.iters * 12.0 * 16.0 / (now_sec() - t0) / 1e9;
            if (gf > best_f64) best_f64 = gf;
        }
        out.fma_f64_gflops = best_f64;
    }
#endif
    printf("\n=== Measured CPU Physical Ceilings (workers=%zu) ===\n", nw);
    printf("  DRAM Read-Only BW (512 MiB)  : %7.1f GB/s\n", out.bw_read_gbps);
    printf("  DRAM Copy BW (256->256 MiB)  : %7.1f GB/s\n", out.bw_copy_gbps);
    if (out.fma_f32_gflops > 0.0) {
        printf("  AVX-512 FP32 FMA Peak        : %7.1f GFLOPS\n", out.fma_f32_gflops);
        printf("  AVX-512 FP64 FMA Peak        : %7.1f GFLOPS\n", out.fma_f64_gflops);
    }
    return out;
}

/* ============================================================================
 * Part 2: Roofline Benchmarks vs Measured CPU Physical Ceilings
 * ============================================================================ */
static void bench_gemv_suite(velri_pool *pool, size_t nw, const char *title,
                             size_t m, size_t k, size_t ring_len, size_t samples,
                             const cpu_ceilings *ceil) {
    bool has_a512 = velri_cpu_variant_supported(VELRI_CPU_AVX512);
    bool has_avx2 = velri_cpu_variant_supported(VELRI_CPU_AVX2);
    velri_cpu_variant primary = has_a512 ? VELRI_CPU_AVX512 : (has_avx2 ? VELRI_CPU_AVX2 : VELRI_CPU_BASE);
    velri_cpu_variant secondary = has_a512 ? VELRI_CPU_AVX2 : VELRI_CPU_BASE;
    const char *p_label = has_a512 ? "Slang A512" : (has_avx2 ? "Slang AVX2" : "Slang Base");
    const char *s_label = has_a512 ? "Slang AVX2" : "Slang Base";

    printf("\n=== %s (rows=1, M=%zu, K=%zu, ring=%zu, workers=%zu) ===\n",
           title, m, k, ring_len, nw);
    if (ring_len > 1u && ceil && ceil->bw_read_gbps > 0.0) {
        printf("%-11s | %11s | %11s | %7s\n", "Mode", p_label, s_label, "%DRAM");
        printf("------------+-------------+-------------+--------\n");
    } else {
        printf("%-11s | %11s | %11s\n", "Mode", p_label, s_label);
        printf("------------+-------------+------------\n");
    }

    struct {
        const char *name;
        velri_linear_desc desc;
    } modes[] = {
        {"F32", velri_desc_f32()},
        {"F64", velri_desc_f64()},
        {"F16->F32", velri_desc_f16(VELRI_DTYPE_F32)},
        {"BF16->F32", velri_desc_bf16(VELRI_DTYPE_F32)},
        {"F32xBF16", velri_desc_f32x_bf16w()},
        {"W8A8->F32", velri_desc_w8a8_f32()},
    };

    for (size_t mi = 0; mi < sizeof(modes) / sizeof(modes[0]); ++mi) {
        velri_linear_desc d = modes[mi].desc;
        size_t x_b = velri_dtype_bytes(d.x_dtype);
        size_t w_b = velri_dtype_bytes(d.w_dtype);
        size_t y_b = velri_dtype_bytes(d.y_dtype);

        void *x = xalloc(k * x_b);
        void *y = xalloc(m * y_b);
        float *xs = d.has_x_scale ? xalloc(sizeof(float)) : nullptr;
        float *ws = d.has_w_scale ? xalloc(m * sizeof(float)) : nullptr;
        uint64_t rng = 0xA0000001ULL + mi;
        fill_dtype(d.x_dtype, x, k, &rng, 0.5f);
        if (xs) xs[0] = 0.02f;
        if (ws) fill_dtype(VELRI_DTYPE_F32, ws, m, &rng, 0.02f);

        void **w_rm = xalloc(ring_len * sizeof(void *));
        for (size_t r = 0; r < ring_len; ++r) {
            w_rm[r] = xalloc(m * k * w_b);
            fill_dtype(d.w_dtype, w_rm[r], m * k, &rng, 0.5f);
        }

        check_ok(velri_slang_cpu_linear(&d, 1u, m, k, x, k, w_rm[0], k, nullptr, xs, ws, y, m, primary, pool, nw), "w1");

        double t_pri[16], t_sec[16];
        size_t iters_per_sample = (ring_len == 1u) ? 40u : ring_len;
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t it = 0; it < iters_per_sample; ++it) {
                check_ok(velri_slang_cpu_linear(&d, 1u, m, k, x, k, w_rm[it % ring_len], k,
                                                nullptr, xs, ws, y, m, primary, pool, nw), "pri");
            }
            t_pri[s] = (now_sec() - t0) / (double)iters_per_sample;

            t0 = now_sec();
            for (size_t it = 0; it < iters_per_sample; ++it) {
                check_ok(velri_slang_cpu_linear(&d, 1u, m, k, x, k, w_rm[it % ring_len], k,
                                                nullptr, xs, ws, y, m, secondary, pool, nw), "sec");
            }
            t_sec[s] = (now_sec() - t0) / (double)iters_per_sample;
        }

        double bytes = (double)(m * k * w_b + k * x_b + m * y_b);
        double gb_pri = bytes / median_of(t_pri, samples) / 1e9;
        double gb_sec = bytes / median_of(t_sec, samples) / 1e9;

        if (ring_len > 1u && ceil && ceil->bw_read_gbps > 0.0) {
            printf("%-11s | %7.1f GB/s | %7.1f GB/s | %6.1f%%\n",
                   modes[mi].name, gb_pri, gb_sec, 100.0 * gb_pri / ceil->bw_read_gbps);
        } else {
            printf("%-11s | %7.1f GB/s | %7.1f GB/s\n", modes[mi].name, gb_pri, gb_sec);
        }

        for (size_t r = 0; r < ring_len; ++r) xfree(w_rm[r]);
        xfree(w_rm); xfree(x); xfree(y); xfree(xs); xfree(ws);
    }
}

static void bench_gemm_suite(velri_pool *pool, size_t nw, size_t rows, size_t m, size_t k,
                             size_t samples, const cpu_ceilings *ceil) {
    bool has_a512 = velri_cpu_variant_supported(VELRI_CPU_AVX512);
    bool has_avx2 = velri_cpu_variant_supported(VELRI_CPU_AVX2);
    velri_cpu_variant primary = has_a512 ? VELRI_CPU_AVX512 : (has_avx2 ? VELRI_CPU_AVX2 : VELRI_CPU_BASE);
    velri_cpu_variant secondary = has_a512 ? VELRI_CPU_AVX2 : VELRI_CPU_BASE;
    const char *p_label = has_a512 ? "Slang A512" : (has_avx2 ? "Slang AVX2" : "Slang Base");
    const char *s_label = has_a512 ? "Slang AVX2" : "Slang Base";

    printf("\n=== GEMM (rows=%zu, M=%zu, K=%zu, workers=%zu) ===\n", rows, m, k, nw);
    printf("%-11s | %12s | %12s | %7s\n", "Mode", p_label, s_label, "%FMA");
    printf("------------+--------------+--------------+--------\n");

    struct {
        const char *name;
        velri_linear_desc desc;
    } modes[] = {
        {"F32", velri_desc_f32()},
        {"F64", velri_desc_f64()},
        {"F16->F32", velri_desc_f16(VELRI_DTYPE_F32)},
        {"BF16->F32", velri_desc_bf16(VELRI_DTYPE_F32)},
        {"F32xBF16", velri_desc_f32x_bf16w()},
        {"W8A8->F32", velri_desc_w8a8_f32()},
    };

    double flops = 2.0 * (double)rows * (double)m * (double)k;
    for (size_t mi = 0; mi < sizeof(modes) / sizeof(modes[0]); ++mi) {
        velri_linear_desc d = modes[mi].desc;
        size_t x_b = velri_dtype_bytes(d.x_dtype);
        size_t w_b = velri_dtype_bytes(d.w_dtype);
        size_t y_b = velri_dtype_bytes(d.y_dtype);

        void *x = xalloc(rows * k * x_b);
        void *w = xalloc(m * k * w_b);
        void *y = xalloc(rows * m * y_b);
        float *xs = d.has_x_scale ? xalloc(rows * sizeof(float)) : nullptr;
        float *ws = d.has_w_scale ? xalloc(m * sizeof(float)) : nullptr;

        uint64_t rng = 0xB0000001ULL + mi;
        fill_dtype(d.x_dtype, x, rows * k, &rng, 0.5f);
        fill_dtype(d.w_dtype, w, m * k, &rng, 0.5f);
        if (xs) fill_dtype(VELRI_DTYPE_F32, xs, rows, &rng, 0.02f);
        if (ws) fill_dtype(VELRI_DTYPE_F32, ws, m, &rng, 0.02f);

        check_ok(velri_slang_cpu_linear(&d, rows, m, k, x, k, w, k, nullptr, xs, ws, y, m, primary, pool, nw), "w1");

        double t_pri[16], t_sec[16];
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            check_ok(velri_slang_cpu_linear(&d, rows, m, k, x, k, w, k, nullptr, xs, ws, y, m, primary, pool, nw), "pri");
            t_pri[s] = now_sec() - t0;

            t0 = now_sec();
            check_ok(velri_slang_cpu_linear(&d, rows, m, k, x, k, w, k, nullptr, xs, ws, y, m, secondary, pool, nw), "sec");
            t_sec[s] = now_sec() - t0;
        }

        double g_pri = flops / median_of(t_pri, samples) / 1e9;
        double g_sec = flops / median_of(t_sec, samples) / 1e9;
        double fma_ref = (d.y_dtype == VELRI_DTYPE_F64)
                             ? (ceil ? ceil->fma_f64_gflops : 0.0)
                             : (ceil ? ceil->fma_f32_gflops : 0.0);
        double pct_fma = (fma_ref > 0.0) ? (100.0 * g_pri / fma_ref) : 0.0;
        printf("%-11s | %6.1f GFLOP | %6.1f GFLOP | %6.1f%%\n",
               modes[mi].name, g_pri, g_sec, pct_fma);

        xfree(x); xfree(w); xfree(y); xfree(xs); xfree(ws);
    }
}

static void bench_non_gemm_suite(velri_pool *pool, size_t nw, size_t samples, const cpu_ceilings *ceil) {
    bool has_a512 = velri_cpu_variant_supported(VELRI_CPU_AVX512);
    bool has_avx2 = velri_cpu_variant_supported(VELRI_CPU_AVX2);
    velri_cpu_variant primary = has_a512 ? VELRI_CPU_AVX512 : (has_avx2 ? VELRI_CPU_AVX2 : VELRI_CPU_BASE);
    velri_cpu_variant secondary = has_a512 ? VELRI_CPU_AVX2 : VELRI_CPU_BASE;
    const char *p_label = has_a512 ? "Slang A512" : (has_avx2 ? "Slang AVX2" : "Slang Base");
    const char *s_label = has_a512 ? "Slang AVX2" : "Slang Base";
    double dram_cp = (ceil && ceil->bw_copy_gbps > 0.0) ? ceil->bw_copy_gbps : 1.0;

    printf("\n=== Non-GEMM LLM Ops (L3-hot vs DRAM copy ceiling, workers=%zu) ===\n", nw);
    printf("%-24s | %11s | %11s | %7s\n", "Operation", p_label, s_label, "xDRAM");
    printf("-------------------------+-------------+-------------+--------\n");

    uint64_t rng = 0xC0000001ULL;

    /* 1. RMSNorm F32 / BF16 / F16 [256 x 4096] */
    const velri_dtype ndts[3] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F16};
    const char *nnames[3] = {"RMSNorm F32  [256x4096]", "RMSNorm BF16 [256x4096]", "RMSNorm F16  [256x4096]"};
    for (int di = 0; di < 3; ++di) {
        size_t rows = 256u, dim = 4096u, eb = velri_dtype_bytes(ndts[di]);
        void *x = xalloc(rows * dim * eb);
        void *w = xalloc(dim * eb);
        void *y = xalloc(rows * dim * eb);
        fill_dtype(ndts[di], x, rows * dim, &rng, 2.0f);
        fill_dtype(ndts[di], w, dim, &rng, 1.0f);

        double t_p[16], t_s[16];
        size_t iters = 20u;
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_rmsnorm(ndts[di], rows, dim, 1e-5f, false, x, dim, w, y, dim, primary, pool, nw), "n");
            t_p[s] = (now_sec() - t0) / (double)iters;

            t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_rmsnorm(ndts[di], rows, dim, 1e-5f, false, x, dim, w, y, dim, secondary, pool, nw), "n");
            t_s[s] = (now_sec() - t0) / (double)iters;
        }
        double bytes = (double)(2u * rows * dim + dim) * (double)eb;
        double gp = bytes / median_of(t_p, samples) / 1e9;
        double gs = bytes / median_of(t_s, samples) / 1e9;
        printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", nnames[di], gp, gs, gp / dram_cp);
        xfree(x); xfree(w); xfree(y);
    }

    /* 2. RoPE F32 / BF16 [256 x 32 x 128] */
    const velri_dtype rdts[2] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16};
    const char *rnames[2] = {"RoPE F32  [256x32x128]", "RoPE BF16 [256x32x128]"};
    for (int di = 0; di < 2; ++di) {
        size_t nt = 256u, nh = 32u, hd = 128u, eb = velri_dtype_bytes(rdts[di]);
        void *x = xalloc(nt * nh * hd * eb);
        void *y = xalloc(nt * nh * hd * eb);
        float *ct = xalloc(nt * (hd / 2u) * sizeof(float));
        float *st = xalloc(nt * (hd / 2u) * sizeof(float));
        fill_dtype(rdts[di], x, nt * nh * hd, &rng, 2.0f);
        for (size_t i = 0; i < nt * (hd / 2u); ++i) { ct[i] = 0.8f; st[i] = 0.6f; }

        double t_p[16], t_s[16];
        size_t iters = 20u;
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_rope(rdts[di], nt, nh, hd, hd, 0u, x, nh * hd, hd, ct, st, y, nh * hd, hd, VELRI_ROPE_HALF, primary, pool, nw), "r");
            t_p[s] = (now_sec() - t0) / (double)iters;

            t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_rope(rdts[di], nt, nh, hd, hd, 0u, x, nh * hd, hd, ct, st, y, nh * hd, hd, VELRI_ROPE_HALF, secondary, pool, nw), "r");
            t_s[s] = (now_sec() - t0) / (double)iters;
        }
        double bytes = (double)(2u * nt * nh * hd) * (double)eb;
        double gp = bytes / median_of(t_p, samples) / 1e9;
        double gs = bytes / median_of(t_s, samples) / 1e9;
        printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", rnames[di], gp, gs, gp / dram_cp);
        xfree(x); xfree(y); xfree(ct); xfree(st);
    }

    /* 3. GatedMLP SwiGLU & GeGLU [256 * 14336] */
    {
        size_t n = 256u * 14336u;
        float *g = xalloc(n * sizeof(float));
        float *u = xalloc(n * sizeof(float));
        float *y = xalloc(n * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, g, n, &rng, 2.0f);
        fill_dtype(VELRI_DTYPE_F32, u, n, &rng, 2.0f);
        const velri_act_kind acts[2] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH};
        const char *anames[2] = {"SwiGLU F32 [256x14336]", "GeGLU  F32 [256x14336]"};
        for (int ai = 0; ai < 2; ++ai) {
            double t_p[16], t_s[16];
            size_t iters = 10u;
            for (size_t s = 0; s < samples; ++s) {
                double t0 = now_sec();
                for (size_t i = 0; i < iters; ++i)
                    check_ok(velri_slang_cpu_gated_mlp_f32(acts[ai], n, 0.0f, g, u, y, primary, pool, nw), "g");
                t_p[s] = (now_sec() - t0) / (double)iters;

                t0 = now_sec();
                for (size_t i = 0; i < iters; ++i)
                    check_ok(velri_slang_cpu_gated_mlp_f32(acts[ai], n, 0.0f, g, u, y, secondary, pool, nw), "g");
                t_s[s] = (now_sec() - t0) / (double)iters;
            }
            double bytes = 3.0 * (double)n * sizeof(float);
            double gp = bytes / median_of(t_p, samples) / 1e9;
            double gs = bytes / median_of(t_s, samples) / 1e9;
            printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", anames[ai], gp, gs, gp / dram_cp);
        }
        xfree(g); xfree(u); xfree(y);
    }

    /* 4. Softmax F32 [1024 x 4096] */
    {
        size_t rows = 1024u, cols = 4096u;
        float *x = xalloc(rows * cols * sizeof(float));
        float *y = xalloc(rows * cols * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, x, rows * cols, &rng, 5.0f);
        double t_p[16], t_s[16];
        size_t iters = 10u;
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_softmax_f32(rows, cols, x, cols, y, cols, primary, pool, nw), "sm");
            t_p[s] = (now_sec() - t0) / (double)iters;

            t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_softmax_f32(rows, cols, x, cols, y, cols, secondary, pool, nw), "sm");
            t_s[s] = (now_sec() - t0) / (double)iters;
        }
        double bytes = 2.0 * (double)(rows * cols) * sizeof(float);
        double gp = bytes / median_of(t_p, samples) / 1e9;
        double gs = bytes / median_of(t_s, samples) / 1e9;
        printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", "Softmax F32 [1024x4096]", gp, gs, gp / dram_cp);
        xfree(x); xfree(y);
    }

    /* 5. FusedAddRMSNorm F32 [256 x 4096] */
    {
        size_t rows = 256u, dim = 4096u;
        float *res = xalloc(rows * dim * sizeof(float));
        float *x = xalloc(rows * dim * sizeof(float));
        float *w = xalloc(dim * sizeof(float));
        float *y = xalloc(rows * dim * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, res, rows * dim, &rng, 1.0f);
        fill_dtype(VELRI_DTYPE_F32, x, rows * dim, &rng, 1.0f);
        fill_dtype(VELRI_DTYPE_F32, w, dim, &rng, 1.0f);
        double t_p[16], t_s[16];
        size_t iters = 20u;
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_fused_add_rmsnorm_f32(rows, dim, 1e-5f, false, res, dim, x, dim, w, y, dim, primary, pool, nw), "far");
            t_p[s] = (now_sec() - t0) / (double)iters;

            t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_fused_add_rmsnorm_f32(rows, dim, 1e-5f, false, res, dim, x, dim, w, y, dim, secondary, pool, nw), "far");
            t_s[s] = (now_sec() - t0) / (double)iters;
        }
        double bytes = (double)(4u * rows * dim + dim) * sizeof(float);
        double gp = bytes / median_of(t_p, samples) / 1e9;
        double gs = bytes / median_of(t_s, samples) / 1e9;
        printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", "FusedAddRMS [256x4096]", gp, gs, gp / dram_cp);
        xfree(res); xfree(x); xfree(w); xfree(y);
    }

    /* 6. LayerNorm F32 [256 x 4096] */
    {
        size_t rows = 256u, dim = 4096u;
        float *x = xalloc(rows * dim * sizeof(float));
        float *w = xalloc(dim * sizeof(float));
        float *b = xalloc(dim * sizeof(float));
        float *y = xalloc(rows * dim * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, x, rows * dim, &rng, 2.0f);
        fill_dtype(VELRI_DTYPE_F32, w, dim, &rng, 1.0f);
        fill_dtype(VELRI_DTYPE_F32, b, dim, &rng, 0.5f);
        double t_p[16], t_s[16];
        size_t iters = 20u;
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_layernorm_f32(rows, dim, 1e-5f, x, dim, w, b, y, dim, primary, pool, nw), "ln");
            t_p[s] = (now_sec() - t0) / (double)iters;

            t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_layernorm_f32(rows, dim, 1e-5f, x, dim, w, b, y, dim, secondary, pool, nw), "ln");
            t_s[s] = (now_sec() - t0) / (double)iters;
        }
        double bytes = (double)(2u * rows * dim + 2u * dim) * sizeof(float);
        double gp = bytes / median_of(t_p, samples) / 1e9;
        double gs = bytes / median_of(t_s, samples) / 1e9;
        printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", "LayerNorm F32 [256x4096]", gp, gs, gp / dram_cp);
        xfree(x); xfree(w); xfree(b); xfree(y);
    }

    /* 7. Standalone SiLU & ReLU2 [256 x 14336] */
    {
        size_t n = 256u * 14336u;
        float *x = xalloc(n * sizeof(float));
        float *y = xalloc(n * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, x, n, &rng, 3.0f);
        const velri_act_kind ks[2] = {VELRI_ACT_SILU, VELRI_ACT_RELU2};
        const char *kn[2] = {"SiLU F32   [256x14336]", "ReLU2 F32  [256x14336]"};
        for (int ki = 0; ki < 2; ++ki) {
            double t_p[16], t_s[16];
            size_t iters = 10u;
            for (size_t s = 0; s < samples; ++s) {
                double t0 = now_sec();
                for (size_t i = 0; i < iters; ++i)
                    check_ok(velri_slang_cpu_activation_f32(ks[ki], n, x, y, primary, pool, nw), "act");
                t_p[s] = (now_sec() - t0) / (double)iters;

                t0 = now_sec();
                for (size_t i = 0; i < iters; ++i)
                    check_ok(velri_slang_cpu_activation_f32(ks[ki], n, x, y, secondary, pool, nw), "act");
                t_s[s] = (now_sec() - t0) / (double)iters;
            }
            double bytes = 2.0 * (double)n * sizeof(float);
            double gp = bytes / median_of(t_p, samples) / 1e9;
            double gs = bytes / median_of(t_s, samples) / 1e9;
            printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", kn[ki], gp, gs, gp / dram_cp);
        }
        xfree(x); xfree(y);
    }

    /* 8. FWHT B=128 & B=256 [256 x 4096] */
    {
        size_t rows = 256u, cols = 4096u;
        float *x = xalloc(rows * cols * sizeof(float));
        float *suh = xalloc(cols * sizeof(float));
        float *y = xalloc(rows * cols * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, x, rows * cols, &rng, 2.0f);
        fill_dtype(VELRI_DTYPE_F32, suh, cols, &rng, 1.0f);
        const size_t bs[2] = {128u, 256u};
        const char *bn[2] = {"FWHT B=128 [256x4096]", "FWHT B=256 [256x4096]"};
        for (int bi = 0; bi < 2; ++bi) {
            double t_p[16], t_s[16];
            size_t iters = 20u;
            for (size_t s = 0; s < samples; ++s) {
                double t0 = now_sec();
                for (size_t i = 0; i < iters; ++i)
                    check_ok(velri_slang_cpu_fwht_f32(rows, cols, bs[bi], x, cols, VELRI_DTYPE_F32, suh, y, cols, primary, pool, nw), "fw");
                t_p[s] = (now_sec() - t0) / (double)iters;

                t0 = now_sec();
                for (size_t i = 0; i < iters; ++i)
                    check_ok(velri_slang_cpu_fwht_f32(rows, cols, bs[bi], x, cols, VELRI_DTYPE_F32, suh, y, cols, secondary, pool, nw), "fw");
                t_s[s] = (now_sec() - t0) / (double)iters;
            }
            double bytes = (double)(2u * rows * cols + cols) * sizeof(float);
            double gp = bytes / median_of(t_p, samples) / 1e9;
            double gs = bytes / median_of(t_s, samples) / 1e9;
            printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", bn[bi], gp, gs, gp / dram_cp);
        }
        xfree(x); xfree(suh); xfree(y);
    }

    /* 9. VecAxpby & VecMul [256 x 14336] */
    {
        size_t n = 256u * 14336u;
        float *a = xalloc(n * sizeof(float));
        float *b = xalloc(n * sizeof(float));
        float *y = xalloc(n * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, a, n, &rng, 2.0f);
        fill_dtype(VELRI_DTYPE_F32, b, n, &rng, 2.0f);
        double t_p[16], t_s[16];
        size_t iters = 10u;
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_vec_axpby_f32(n, 0.7f, a, 0.3f, b, y, primary, pool, nw), "ax");
            t_p[s] = (now_sec() - t0) / (double)iters;

            t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_vec_axpby_f32(n, 0.7f, a, 0.3f, b, y, secondary, pool, nw), "ax");
            t_s[s] = (now_sec() - t0) / (double)iters;
        }
        double bytes = 3.0 * (double)n * sizeof(float);
        double gp = bytes / median_of(t_p, samples) / 1e9;
        double gs = bytes / median_of(t_s, samples) / 1e9;
        printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", "VecAxpby   [256x14336]", gp, gs, gp / dram_cp);

        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_vec_mul_f32(n, a, b, y, primary, pool, nw), "vm");
            t_p[s] = (now_sec() - t0) / (double)iters;

            t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_vec_mul_f32(n, a, b, y, secondary, pool, nw), "vm");
            t_s[s] = (now_sec() - t0) / (double)iters;
        }
        gp = bytes / median_of(t_p, samples) / 1e9;
        gs = bytes / median_of(t_s, samples) / 1e9;
        printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", "VecMul     [256x14336]", gp, gs, gp / dram_cp);
        xfree(a); xfree(b); xfree(y);
    }

    /* 10. QuantRow F32 -> I8 [1024 x 4096] */
    {
        size_t rows = 1024u, k = 4096u;
        float *x = xalloc(rows * k * sizeof(float));
        int8_t *q = xalloc(rows * k);
        float *sc = xalloc(rows * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, x, rows * k, &rng, 3.0f);
        double t_p[16], t_s[16];
        size_t iters = 10u;
        for (size_t s = 0; s < samples; ++s) {
            double t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_quantize_row_f32_i8(rows, k, x, k, q, k, sc, primary, pool, nw), "qr");
            t_p[s] = (now_sec() - t0) / (double)iters;

            t0 = now_sec();
            for (size_t i = 0; i < iters; ++i)
                check_ok(velri_slang_cpu_quantize_row_f32_i8(rows, k, x, k, q, k, sc, secondary, pool, nw), "qr");
            t_s[s] = (now_sec() - t0) / (double)iters;
        }
        double bytes = (double)(rows * k * (sizeof(float) + 1u) + rows * sizeof(float));
        double gp = bytes / median_of(t_p, samples) / 1e9;
        double gs = bytes / median_of(t_s, samples) / 1e9;
        printf("%-24s | %7.1f GB/s | %7.1f GB/s | %6.1fx\n", "QuantRow I8 [1024x4096]", gp, gs, gp / dram_cp);
        xfree(x); xfree(q); xfree(sc);
    }
}

int main(int argc, char **argv) {
    bool verify_only = false;
    const velri_cpu_info *ci = velri_get_cpu_info();
    size_t workers = (ci->hardware_threads >= 2) ? (size_t)(ci->hardware_threads / 2) : 1u;
    if (workers > 16u) workers = 16u;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--verify-only") == 0) verify_only = true;
        else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            workers = (size_t)strtoull(argv[++i], nullptr, 10);
            if (workers == 0) workers = 1u;
        }
    }

    printf("CPU: %d hw threads, L3 domains=%d, AVX2=%d FMA=%d F16C=%d AVX512F=%d BF16=%d VNNI=%d\n",
           ci->hardware_threads, ci->num_l3_domains,
           (int)ci->has_avx2, (int)ci->has_fma, (int)ci->has_f16c,
           (int)ci->has_avx512f, (int)ci->has_avx512_bf16, (int)ci->has_avx512_vnni);

    velri_pool *pool = nullptr;
    check_ok(velri_pool_create(workers, 0u, &pool), "pool_create");

    run_verification(pool, workers);

    if (!verify_only) {
        cpu_ceilings ceil = measure_cpu_ceilings(pool, workers);
        bench_gemv_suite(pool, workers, "Streaming GEMV (cold L3)", 4096u, 4096u, 12u, 7u, &ceil);
        bench_gemv_suite(pool, workers, "Cached GEMV (L2/L3 hot)", 1024u, 1024u, 1u, 9u, &ceil);
        bench_gemm_suite(pool, workers, 64u, 4096u, 4096u, 7u, &ceil);
        bench_gemm_suite(pool, workers, 256u, 4096u, 4096u, 7u, &ceil);
        bench_non_gemm_suite(pool, workers, 9u, &ceil);
    }

    velri_pool_destroy(pool);
    return 0;
}
