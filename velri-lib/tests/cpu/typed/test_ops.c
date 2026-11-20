#include "reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond);                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/* 1. Bit-exact conversion & dynamic symmetric INT8 row quantization */
static void test_conversions_and_quant(void) {
    const size_t n = 519; /* odd length hitting all vector + scalar tails */
    float *src = (float *)malloc(n * sizeof(float));
    uint64_t rng = UINT64_C(0x9876543210FEDCBA);

    for (size_t i = 0; i < n; ++i) {
        src[i] = velri_test_rand_f32(&rng, -25.0f, 25.0f);
    }
    /* Inject subnormals, signed zeros, infinities, and FP16/BF16 RNE tie boundaries */
    src[0] = 0.0f;
    src[1] = -0.0f;
    src[2] = 1.0e-40f;  /* FP32 subnormal! */
    src[3] = -3.5e-41f; /* negative FP32 subnormal! */
    src[4] = 1.5e-6f;   /* FP16 subnormal! */
    src[5] = 65504.0f;  /* FP16 max normal */
    src[6] = 70000.0f;  /* FP16 overflow -> +Inf */
    src[7] = INFINITY;
    src[8] = -INFINITY;

    velri_fp16 *h_base = (velri_fp16 *)malloc(n * sizeof(velri_fp16));
    velri_fp16 *h_simd = (velri_fp16 *)malloc(n * sizeof(velri_fp16));
    velri_bf16 *b_base = (velri_bf16 *)malloc(n * sizeof(velri_bf16));
    velri_bf16 *b_simd = (velri_bf16 *)malloc(n * sizeof(velri_bf16));
    float *rt_base = (float *)malloc(n * sizeof(float));
    float *rt_simd = (float *)malloc(n * sizeof(float));

    CHECK(velri_cvt_f32_to_f16(src, h_base, n, VELRI_CPU_BASE) == VELRI_OK);
    CHECK(velri_cvt_f32_to_bf16(src, b_base, n, VELRI_CPU_BASE) == VELRI_OK);
    CHECK(velri_cvt_f16_to_f32(h_base, rt_base, n, VELRI_CPU_BASE) == VELRI_OK);

    const velri_cpu_variant vars[] = {VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    for (size_t vi = 0; vi < 2; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_cvt_f32_to_f16(src, h_simd, n, vars[vi]) == VELRI_OK);
        CHECK(velri_cvt_f32_to_bf16(src, b_simd, n, vars[vi]) == VELRI_OK);
        CHECK(memcmp(h_base, h_simd, n * sizeof(velri_fp16)) == 0);
        CHECK(memcmp(b_base, b_simd, n * sizeof(velri_bf16)) == 0);

        CHECK(velri_cvt_f16_to_f32(h_base, rt_simd, n, vars[vi]) == VELRI_OK);
        CHECK(memcmp(rt_base, rt_simd, n * sizeof(float)) == 0);
    }

    CHECK(velri_cvt_bf16_to_f32(b_base, rt_base, n, VELRI_CPU_BASE) == VELRI_OK);
    for (size_t vi = 0; vi < 2; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_cvt_bf16_to_f32(b_base, rt_simd, n, vars[vi]) == VELRI_OK);
        CHECK(memcmp(rt_base, rt_simd, n * sizeof(float)) == 0);
    }

    /* Dynamic symmetric INT8 row quantizer bit-exactness across BASE, AVX2, AVX512 */
    const size_t rows = 7, k = 133;
    float *qx = (float *)malloc(rows * (k + 3) * sizeof(float));
    for (size_t i = 0; i < rows * (k + 3); ++i) {
        qx[i] =
            (i < (k + 3)) ? 0.0f : velri_test_rand_f32(&rng, -12.3f, 9.7f); /* row 0 all-zero! */
    }
    int8_t *qi_base = (int8_t *)malloc(rows * k);
    int8_t *qi_simd = (int8_t *)malloc(rows * k);
    float sc_base[7], sc_simd[7];
    CHECK(velri_quantize_row_f32_i8(rows, k, qx, k + 3, qi_base, k, sc_base, VELRI_CPU_BASE) ==
          VELRI_OK);
    for (size_t vi = 0; vi < 2; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_quantize_row_f32_i8(rows, k, qx, k + 3, qi_simd, k, sc_simd, vars[vi]) ==
              VELRI_OK);
        for (size_t r = 0; r < rows; ++r) {
            CHECK(fabsf(sc_base[r] - sc_simd[r]) <= 1e-7f);
            for (size_t j = 0; j < k; ++j) {
                CHECK(abs((int)qi_base[r * k + j] - (int)qi_simd[r * k + j]) <= 1);
            }
        }
    }

    free(src);
    free(h_base);
    free(h_simd);
    free(b_base);
    free(b_simd);
    free(rt_base);
    free(rt_simd);
    free(qx);
    free(qi_base);
    free(qi_simd);
}

/* 2. RMSNorm (`unit_offset`, `F32`/`BF16`/`F16`), Fused Add+RMSNorm & LayerNorm */
static void test_norms(velri_pool *pool) {
    uint64_t rng = UINT64_C(0x1122334455667788);
    const velri_dtype dts[] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F16};
    const velri_cpu_variant vars[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    const size_t rows = 19, dim = 137;

    for (size_t di = 0; di < 3; ++di) {
        velri_dtype dt = dts[di];
        size_t eb = velri_dtype_bytes(dt);
        void *x = malloc(rows * dim * eb);
        void *w = malloc(dim * eb);
        void *y_ref = malloc(rows * dim * eb);
        void *y_got = malloc(rows * dim * eb);

        for (size_t i = 0; i < rows * dim; ++i) {
            float v = velri_test_rand_f32(&rng, -2.0f, 2.0f);
            if (dt == VELRI_DTYPE_F32)
                ((float *)x)[i] = v;
            else if (dt == VELRI_DTYPE_BF16)
                ((velri_bf16 *)x)[i] = velri_f32_to_bf16_scalar(v);
            else
                ((velri_fp16 *)x)[i] = velri_f32_to_f16_scalar(v);
        }
        for (size_t i = 0; i < dim; ++i) {
            float v = velri_test_rand_f32(&rng, -0.3f, 0.5f);
            if (dt == VELRI_DTYPE_F32)
                ((float *)w)[i] = v;
            else if (dt == VELRI_DTYPE_BF16)
                ((velri_bf16 *)w)[i] = velri_f32_to_bf16_scalar(v);
            else
                ((velri_fp16 *)w)[i] = velri_f32_to_f16_scalar(v);
        }

        for (int uo = 0; uo <= 1; ++uo) {
            ref_rmsnorm(dt, dt, rows, dim, 1e-5f, uo != 0, x, dim, w, y_ref, dim);
            for (size_t vi = 0; vi < 3; ++vi) {
                if (!velri_cpu_variant_supported(vars[vi])) {
                    continue;
                }
                /* Out-of-place + parallel */
                CHECK(velri_rmsnorm(dt, dt, rows, dim, 1e-5f, uo != 0, x, dim, w, y_got, dim,
                                    vars[vi], pool, 8) == VELRI_OK);
                for (size_t i = 0; i < rows * dim; ++i) {
                    float g = (dt == VELRI_DTYPE_F32)
                                  ? ((float *)y_got)[i]
                                  : ((dt == VELRI_DTYPE_BF16)
                                         ? velri_bf16_to_f32_scalar(((velri_bf16 *)y_got)[i])
                                         : velri_f16_to_f32_scalar(((velri_fp16 *)y_got)[i]));
                    float e = (dt == VELRI_DTYPE_F32)
                                  ? ((float *)y_ref)[i]
                                  : ((dt == VELRI_DTYPE_BF16)
                                         ? velri_bf16_to_f32_scalar(((velri_bf16 *)y_ref)[i])
                                         : velri_f16_to_f32_scalar(((velri_fp16 *)y_ref)[i]));
                    float tol =
                        (dt == VELRI_DTYPE_F32) ? 2e-5f : ((dt == VELRI_DTYPE_F16) ? 3e-3f : 2e-2f);
                    CHECK(fabsf(g - e) <= tol);
                }

                /* In-place `y == x` */
                memcpy(y_got, x, rows * dim * eb);
                CHECK(velri_rmsnorm(dt, dt, rows, dim, 1e-5f, uo != 0, y_got, dim, w, y_got, dim,
                                    vars[vi], nullptr, 1) == VELRI_OK);
            }
        }
        free(x);
        free(w);
        free(y_ref);
        free(y_got);
    }

    /* Fused Add + RMSNorm & LayerNorm */
    float *res0 = (float *)malloc(rows * dim * sizeof(float));
    float *res_ref = (float *)malloc(rows * dim * sizeof(float));
    float *res_got = (float *)malloc(rows * dim * sizeof(float));
    float *fx = (float *)malloc(rows * dim * sizeof(float));
    float *fw = (float *)malloc(dim * sizeof(float));
    float *fb = (float *)malloc(dim * sizeof(float));
    float *fy_ref = (float *)malloc(rows * dim * sizeof(float));
    float *fy_got = (float *)malloc(rows * dim * sizeof(float));

    for (size_t i = 0; i < rows * dim; ++i) {
        res0[i] = velri_test_rand_f32(&rng, -1.0f, 1.0f);
        fx[i] = velri_test_rand_f32(&rng, -1.0f, 1.0f);
    }
    for (size_t i = 0; i < dim; ++i) {
        fw[i] = velri_test_rand_f32(&rng, 0.2f, 1.5f);
        fb[i] = velri_test_rand_f32(&rng, -0.5f, 0.5f);
    }

    memcpy(res_ref, res0, rows * dim * sizeof(float));
    ref_fused_add_rmsnorm_f32(rows, dim, 1e-5f, true, res_ref, dim, fx, dim, fw, fy_ref, dim);
    for (size_t vi = 0; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        memcpy(res_got, res0, rows * dim * sizeof(float));
        CHECK(velri_fused_add_rmsnorm_f32(rows, dim, 1e-5f, true, res_got, dim, fx, dim, fw, fy_got,
                                          dim, vars[vi], pool, 8) == VELRI_OK);
        for (size_t i = 0; i < rows * dim; ++i) {
            CHECK(fabsf(res_got[i] - res_ref[i]) <= 1e-6f);
            CHECK(fabsf(fy_got[i] - fy_ref[i]) <= 2e-5f);
        }
    }

    ref_layernorm_f32(rows, dim, 1e-5f, fx, dim, fw, fb, fy_ref, dim);
    for (size_t vi = 0; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_layernorm_f32(rows, dim, 1e-5f, fx, dim, fw, fb, fy_got, dim, vars[vi], pool,
                                  8) == VELRI_OK);
        for (size_t i = 0; i < rows * dim; ++i) {
            CHECK(fabsf(fy_got[i] - fy_ref[i]) <= 3e-5f);
        }
    }

    free(res0);
    free(res_ref);
    free(res_got);
    free(fx);
    free(fw);
    free(fb);
    free(fy_ref);
    free(fy_got);
}

/* 3. RoPE (`rotary_dim < head_dim`, `HALF` & `INTERLEAVED`, `F32`/`BF16`/`F16`, in-place) */
static void test_rope(velri_pool *pool) {
    uint64_t rng = UINT64_C(0xABCDEF0123456789);
    const size_t n_tok = 9, n_heads = 5, head_dim = 86, rotary_dim = 54, pos_off = 11;
    const size_t cs_tok = 32, half = rotary_dim / 2;
    float *cos_t = (float *)malloc(cs_tok * half * sizeof(float));
    float *sin_t = (float *)malloc(cs_tok * half * sizeof(float));
    for (size_t i = 0; i < cs_tok * half; ++i) {
        float th = velri_test_rand_f32(&rng, -3.14f, 3.14f);
        cos_t[i] = cosf(th);
        sin_t[i] = sinf(th);
    }

    const velri_dtype dts[] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F16};
    const velri_cpu_variant vars[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    const size_t total = n_tok * n_heads * head_dim;

    for (size_t di = 0; di < 3; ++di) {
        velri_dtype dt = dts[di];
        size_t eb = velri_dtype_bytes(dt);
        void *x = malloc(total * eb);
        void *y_ref = malloc(total * eb);
        void *y_got = malloc(total * eb);

        for (size_t i = 0; i < total; ++i) {
            float v = velri_test_rand_f32(&rng, -2.0f, 2.0f);
            if (dt == VELRI_DTYPE_F32)
                ((float *)x)[i] = v;
            else if (dt == VELRI_DTYPE_BF16)
                ((velri_bf16 *)x)[i] = velri_f32_to_bf16_scalar(v);
            else
                ((velri_fp16 *)x)[i] = velri_f32_to_f16_scalar(v);
        }

        for (int md = 0; md <= 1; ++md) {
            velri_rope_mode mode = (md == 0) ? VELRI_ROPE_HALF : VELRI_ROPE_INTERLEAVED;
            ref_rope(dt, n_tok, n_heads, head_dim, rotary_dim, pos_off, x, n_heads * head_dim,
                     head_dim, cos_t, sin_t, y_ref, n_heads * head_dim, head_dim, mode);

            for (size_t vi = 0; vi < 3; ++vi) {
                if (!velri_cpu_variant_supported(vars[vi])) {
                    continue;
                }
                CHECK(velri_rope(dt, n_tok, n_heads, head_dim, rotary_dim, pos_off, cs_tok, x,
                                 n_heads * head_dim, head_dim, cos_t, sin_t, y_got,
                                 n_heads * head_dim, head_dim, mode, vars[vi], pool,
                                 4) == VELRI_OK);
                for (size_t i = 0; i < total; ++i) {
                    float g = (dt == VELRI_DTYPE_F32)
                                  ? ((float *)y_got)[i]
                                  : ((dt == VELRI_DTYPE_BF16)
                                         ? velri_bf16_to_f32_scalar(((velri_bf16 *)y_got)[i])
                                         : velri_f16_to_f32_scalar(((velri_fp16 *)y_got)[i]));
                    float e = (dt == VELRI_DTYPE_F32)
                                  ? ((float *)y_ref)[i]
                                  : ((dt == VELRI_DTYPE_BF16)
                                         ? velri_bf16_to_f32_scalar(((velri_bf16 *)y_ref)[i])
                                         : velri_f16_to_f32_scalar(((velri_fp16 *)y_ref)[i]));
                    float tol =
                        (dt == VELRI_DTYPE_F32) ? 1e-5f : ((dt == VELRI_DTYPE_F16) ? 3e-3f : 2e-2f);
                    CHECK(fabsf(g - e) <= tol);
                }

                /* Also verify in-place `y == x`! */
                memcpy(y_got, x, total * eb);
                CHECK(velri_rope(dt, n_tok, n_heads, head_dim, rotary_dim, pos_off, cs_tok, y_got,
                                 n_heads * head_dim, head_dim, cos_t, sin_t, y_got,
                                 n_heads * head_dim, head_dim, mode, vars[vi], nullptr,
                                 1) == VELRI_OK);
                for (size_t i = 0; i < total; ++i) {
                    float g = (dt == VELRI_DTYPE_F32)
                                  ? ((float *)y_got)[i]
                                  : ((dt == VELRI_DTYPE_BF16)
                                         ? velri_bf16_to_f32_scalar(((velri_bf16 *)y_got)[i])
                                         : velri_f16_to_f32_scalar(((velri_fp16 *)y_got)[i]));
                    float e = (dt == VELRI_DTYPE_F32)
                                  ? ((float *)y_ref)[i]
                                  : ((dt == VELRI_DTYPE_BF16)
                                         ? velri_bf16_to_f32_scalar(((velri_bf16 *)y_ref)[i])
                                         : velri_f16_to_f32_scalar(((velri_fp16 *)y_ref)[i]));
                    float tol =
                        (dt == VELRI_DTYPE_F32) ? 1e-5f : ((dt == VELRI_DTYPE_F16) ? 3e-3f : 2e-2f);
                    CHECK(fabsf(g - e) <= tol);
                }
            }
        }
        free(x);
        free(y_ref);
        free(y_got);
    }
    free(cos_t);
    free(sin_t);
}

/* 4. Activations, GatedMLP (`clamp_limit`) & Online Softmax (extreme logits + `-INFINITY`) */
static void test_activations_and_softmax(velri_pool *pool) {
    uint64_t rng = UINT64_C(0x55AA55AA12349876);
    const size_t n = 8213;
    float *g = (float *)malloc(n * sizeof(float));
    float *u = (float *)malloc(n * sizeof(float));
    float *y_ref = (float *)malloc(n * sizeof(float));
    float *y_got = (float *)malloc(n * sizeof(float));

    for (size_t i = 0; i < n; ++i) {
        g[i] = velri_test_rand_f32(&rng, -15.0f, 15.0f);
        u[i] = velri_test_rand_f32(&rng, -5.0f, 5.0f);
    }
    g[0] = -120.0f;
    g[1] = 120.0f;

    const velri_act_kind kinds[] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH, VELRI_ACT_RELU2};
    const velri_cpu_variant vars[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};

    for (size_t ki = 0; ki < 3; ++ki) {
        ref_activation_f32(kinds[ki], n, g, y_ref);
        for (size_t vi = 0; vi < 3; ++vi) {
            if (!velri_cpu_variant_supported(vars[vi])) {
                continue;
            }
            CHECK(velri_activation_f32(kinds[ki], n, g, y_got, vars[vi], pool, 8) == VELRI_OK);
            for (size_t i = 0; i < n; ++i) {
                CHECK(!isnan(y_got[i]) &&
                      fabsf(y_got[i] - y_ref[i]) <= 2e-5f * (fabsf(y_ref[i]) + 1.0f));
            }
        }

        /* GatedMLP with symmetric `clamp_limit = 6.5f` */
        ref_gated_mlp_f32(kinds[ki], n, 6.5f, g, u, y_ref);
        for (size_t vi = 0; vi < 3; ++vi) {
            if (!velri_cpu_variant_supported(vars[vi])) {
                continue;
            }
            CHECK(velri_gated_mlp_f32(kinds[ki], n, 6.5f, g, u, y_got, vars[vi], pool, 8) ==
                  VELRI_OK);
            for (size_t i = 0; i < n; ++i) {
                CHECK(!isnan(y_got[i]) &&
                      fabsf(y_got[i] - y_ref[i]) <= 2e-5f * (fabsf(y_ref[i]) + 1.0f));
            }
        }
    }

    /* Online 1-Pass Softmax with large +1000 shift & -INFINITY masked logits */
    const size_t rows = 13, cols = 263;
    float *sx = (float *)malloc(rows * cols * sizeof(float));
    float *sr = (float *)malloc(rows * cols * sizeof(float));
    float *sg = (float *)malloc(rows * cols * sizeof(float));
    for (size_t i = 0; i < rows * cols; ++i) {
        sx[i] = 1000.0f + velri_test_rand_f32(&rng, -8.0f, 8.0f);
    }
    sx[5] = -INFINITY;
    sx[70] = -INFINITY;
    ref_softmax_f32(rows, cols, sx, cols, sr, cols);

    for (size_t vi = 0; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_softmax_f32(rows, cols, sx, cols, sg, cols, vars[vi], pool, 8) == VELRI_OK);
        for (size_t i = 0; i < rows * cols; ++i) {
            CHECK(!isnan(sg[i]) && fabsf(sg[i] - sr[i]) <= 1e-5f);
        }
    }

    free(g);
    free(u);
    free(y_ref);
    free(y_got);
    free(sx);
    free(sr);
    free(sg);
}

/* 5. Orthonormal Block FWHT (`.vtq` / `LTQ v1`, 0-ULP Bit-Exact) & Pure Vector Math */
static void test_fwht_and_vec_math(velri_pool *pool) {
    uint64_t rng = UINT64_C(0x13579BDF2468ACE0);
    const velri_cpu_variant vars[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    const size_t bsizes[] = {16, 32, 64, 128, 256};

    /* A. Orthonormal Block FWHT: 0-ULP bit-exactness across BASE/AVX2/AVX512, F16/F32 suh,
     * involution & Guard Pages */
    const size_t rows = 7, cols = 768, xs = cols + 5, ys = cols + 9;
    float *x = (float *)malloc(rows * xs * sizeof(float));
    float *y_ref = (float *)malloc(rows * ys * sizeof(float));
    float *y_base = (float *)malloc(rows * ys * sizeof(float));
    float *y_simd = (float *)malloc(rows * ys * sizeof(float));
    float *y_inv = (float *)malloc(rows * cols * sizeof(float));
    float *suh_f32 = (float *)malloc(cols * sizeof(float));
    velri_fp16 *suh_f16 = (velri_fp16 *)malloc(cols * sizeof(velri_fp16));
    CHECK(x && y_ref && y_base && y_simd && y_inv && suh_f32 && suh_f16);

    for (size_t i = 0; i < rows * xs; ++i) {
        x[i] = velri_test_rand_f32(&rng, -3.0f, 3.0f);
    }
    for (size_t i = 0; i < cols; ++i) {
        float sign = (velri_test_rng(&rng) & 1U) ? 1.0f : -1.0f;
        float mag = velri_test_rand_f32(&rng, 0.5f, 1.5f);
        suh_f16[i] = velri_f32_to_f16_scalar(sign * mag);
        suh_f32[i] = velri_f16_to_f32_scalar(suh_f16[i]);
    }

    for (size_t bi = 0; bi < 5; ++bi) {
        size_t B = bsizes[bi];
        for (int sm = 0; sm < 3; ++sm) {
            velri_dtype sdt = (sm == 2) ? VELRI_DTYPE_F16 : VELRI_DTYPE_F32;
            const void *sp =
                (sm == 0) ? nullptr : ((sm == 1) ? (const void *)suh_f32 : (const void *)suh_f16);

            ref_fwht_f32(rows, cols, B, x, xs, sdt, sp, y_ref, ys);
            memset(y_base, 0, rows * ys * sizeof(float));
            CHECK(velri_fwht_f32(rows, cols, B, x, xs, sdt, sp, y_base, ys, VELRI_CPU_BASE, nullptr,
                                 1) == VELRI_OK);

            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                    CHECK(fabsf(y_base[r * ys + c] - y_ref[r * ys + c]) <= 2e-5f);
                }
            }

            for (size_t vi = 1; vi < 3; ++vi) {
                if (!velri_cpu_variant_supported(vars[vi])) {
                    continue;
                }
                const size_t nw_list[3] = {1, 4, 16};
                for (size_t nwi = 0; nwi < 3; ++nwi) {
                    size_t nw = nw_list[nwi];
                    memset(y_simd, 0, rows * ys * sizeof(float));
                    CHECK(velri_fwht_f32(rows, cols, B, x, xs, sdt, sp, y_simd, ys, vars[vi],
                                         nw > 1 ? pool : nullptr, nw) == VELRI_OK);
                    /* 0-ULP bit-exact match between BASE, AVX2, and AVX-512! */
                    for (size_t r = 0; r < rows; ++r) {
                        CHECK(memcmp(y_base + r * ys, y_simd + r * ys, cols * sizeof(float)) == 0);
                    }
                }

                /* In-place `y == x` + Involution check H_B(H_B(x)) == x when suh == nullptr */
                if (sm == 0) {
                    for (size_t r = 0; r < rows; ++r) {
                        memcpy(y_inv + r * cols, x + r * xs, cols * sizeof(float));
                    }
                    CHECK(velri_fwht_f32(rows, cols, B, y_inv, cols, VELRI_DTYPE_F32, nullptr,
                                         y_inv, cols, vars[vi], pool, 8) == VELRI_OK);
                    CHECK(velri_fwht_f32(rows, cols, B, y_inv, cols, VELRI_DTYPE_F32, nullptr,
                                         y_inv, cols, vars[vi], pool, 8) == VELRI_OK);
                    for (size_t r = 0; r < rows; ++r) {
                        for (size_t c = 0; c < cols; ++c) {
                            CHECK(fabsf(y_inv[r * cols + c] - x[r * xs + c]) <= 2e-5f);
                        }
                    }
                }
            }
        }
    }

    /* Guard-page & invalid-argument safety checks for FWHT */
    {
        velri_guard_buf gx = velri_guard_alloc(3 * 256 * sizeof(float), alignof(float));
        velri_guard_buf gs = velri_guard_alloc(256 * sizeof(velri_fp16), alignof(velri_fp16));
        velri_guard_buf gy = velri_guard_alloc(3 * 256 * sizeof(float), alignof(float));
        for (size_t i = 0; i < 3 * 256; ++i) {
            ((float *)gx.ptr)[i] = velri_test_rand_f32(&rng, -1.0f, 1.0f);
        }
        for (size_t i = 0; i < 256; ++i) {
            ((velri_fp16 *)gs.ptr)[i] = suh_f16[i];
        }
        for (size_t vi = 0; vi < 3; ++vi) {
            if (!velri_cpu_variant_supported(vars[vi])) {
                continue;
            }
            CHECK(velri_fwht_f32(3, 256, 256, (const float *)gx.ptr, 256, VELRI_DTYPE_F16, gs.ptr,
                                 (float *)gy.ptr, 256, vars[vi], pool, 4) == VELRI_OK);
        }
        CHECK(velri_fwht_f32(3, 256, 24, (const float *)gx.ptr, 256, VELRI_DTYPE_F16, gs.ptr,
                             (float *)gy.ptr, 256, VELRI_CPU_BASE, nullptr,
                             1) == VELRI_ERR_INVALID);
        CHECK(velri_fwht_f32(3, 256, 512, (const float *)gx.ptr, 256, VELRI_DTYPE_F16, gs.ptr,
                             (float *)gy.ptr, 256, VELRI_CPU_BASE, nullptr,
                             1) == VELRI_ERR_INVALID);
        CHECK(velri_fwht_f32(3, 250, 256, (const float *)gx.ptr, 256, VELRI_DTYPE_F16, gs.ptr,
                             (float *)gy.ptr, 256, VELRI_CPU_BASE, nullptr,
                             1) == VELRI_ERR_INVALID);
        CHECK(velri_fwht_f32(2, 256, 256, (const float *)gx.ptr, 256, VELRI_DTYPE_F16, gs.ptr,
                             (float *)gx.ptr + 256, 256, VELRI_CPU_BASE, nullptr,
                             1) == VELRI_ERR_INVALID);
        velri_guard_free(&gx);
        velri_guard_free(&gs);
        velri_guard_free(&gy);
    }

    free(x);
    free(y_ref);
    free(y_base);
    free(y_simd);
    free(y_inv);
    free(suh_f32);
    free(suh_f16);

    /* B. Pure Vector Math (`velri_vec_axpby_f32` & `velri_vec_mul_f32`) + Guard Pages */
    const size_t vn = 519;
    velri_guard_buf ga = velri_guard_alloc(vn * sizeof(float), alignof(float));
    velri_guard_buf gb = velri_guard_alloc(vn * sizeof(float), alignof(float));
    velri_guard_buf gy = velri_guard_alloc(vn * sizeof(float), alignof(float));
    float *vr = (float *)malloc(vn * sizeof(float));
    float *vb_base = (float *)malloc(vn * sizeof(float));
    float *pa = (float *)ga.ptr, *pb = (float *)gb.ptr, *py = (float *)gy.ptr;
    for (size_t i = 0; i < vn; ++i) {
        pa[i] = velri_test_rand_f32(&rng, -4.0f, 4.0f);
        pb[i] = velri_test_rand_f32(&rng, -4.0f, 4.0f);
    }

    const float alphas[5] = {1.0f, 1.0f, 1.0f, 0.5f, 0.7f};
    const float betas[5] = {1.0f, -1.0f, 0.25f, 0.0f, -0.3f};
    for (int k = 0; k < 5; ++k) {
        ref_vec_axpby_f32(vn, alphas[k], pa, betas[k], betas[k] == 0.0f ? nullptr : pb, vr);
        for (size_t vi = 0; vi < 3; ++vi) {
            if (!velri_cpu_variant_supported(vars[vi])) {
                continue;
            }
            CHECK(velri_vec_axpby_f32(vn, alphas[k], pa, betas[k], betas[k] == 0.0f ? nullptr : pb,
                                      py, vars[vi], pool, 8) == VELRI_OK);
            for (size_t i = 0; i < vn; ++i) {
                CHECK(fabsf(py[i] - vr[i]) <= 2e-6f);
            }
            /* In-place `y == a` */
            memcpy(py, pa, vn * sizeof(float));
            CHECK(velri_vec_axpby_f32(vn, alphas[k], py, betas[k], betas[k] == 0.0f ? nullptr : pb,
                                      py, vars[vi], nullptr, 1) == VELRI_OK);
            for (size_t i = 0; i < vn; ++i) {
                CHECK(fabsf(py[i] - vr[i]) <= 2e-6f);
            }
        }
    }

    ref_vec_mul_f32(vn, pa, pb, vr);
    CHECK(velri_vec_mul_f32(vn, pa, pb, vb_base, VELRI_CPU_BASE, nullptr, 1) == VELRI_OK);
    for (size_t vi = 0; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_vec_mul_f32(vn, pa, pb, py, vars[vi], pool, 8) == VELRI_OK);
        CHECK(memcmp(py, vb_base, vn * sizeof(float)) == 0);
        for (size_t i = 0; i < vn; ++i) {
            CHECK(fabsf(py[i] - vr[i]) <= 1e-6f);
        }
        /* In-place `y == a == b` (`x^2`) */
        memcpy(py, pa, vn * sizeof(float));
        CHECK(velri_vec_mul_f32(vn, py, py, py, vars[vi], nullptr, 1) == VELRI_OK);
        for (size_t i = 0; i < vn; ++i) {
            CHECK(fabsf(py[i] - pa[i] * pa[i]) <= 1e-6f);
        }
    }

    /* Partial-overlap rejections */
    CHECK(velri_vec_axpby_f32(vn - 1, 1.0f, py, 1.0f, pb, py + 1, VELRI_CPU_BASE, nullptr, 1) ==
          VELRI_ERR_INVALID);
    CHECK(velri_vec_mul_f32(vn - 1, pa, py, py + 1, VELRI_CPU_BASE, nullptr, 1) ==
          VELRI_ERR_INVALID);

    free(vr);
    free(vb_base);
    velri_guard_free(&ga);
    velri_guard_free(&gb);
    velri_guard_free(&gy);
}

/* 6. Non-temporal streaming paths (`>= 256 Ki` elems/worker & `>= 4 MiB/worker` FWHT-256) */
static void test_nt_streaming_paths(velri_pool *pool) {
    uint64_t rng = UINT64_C(0xCAFEBABE87654321);
    const velri_cpu_variant vars[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};

    /* A. Parallel NT-streaming conversions (`n = 530,117`, unaligned `+3` start & odd tail, `nw =
     * 2`) */
    const size_t n = 530117;
    const size_t alloc_elems = 260ULL * 2048ULL; /* 532,480: multiple of 64 & covers qrows*qk */
    float *f_buf = (float *)aligned_alloc(64, alloc_elems * sizeof(float));
    velri_fp16 *h_ref = (velri_fp16 *)aligned_alloc(64, alloc_elems * sizeof(velri_fp16));
    velri_fp16 *h_got = (velri_fp16 *)aligned_alloc(64, alloc_elems * sizeof(velri_fp16));
    velri_bf16 *b_ref = (velri_bf16 *)aligned_alloc(64, alloc_elems * sizeof(velri_bf16));
    velri_bf16 *b_got = (velri_bf16 *)aligned_alloc(64, alloc_elems * sizeof(velri_bf16));
    float *rt_ref = (float *)aligned_alloc(64, alloc_elems * sizeof(float));
    float *rt_got = (float *)aligned_alloc(64, alloc_elems * sizeof(float));
    CHECK(f_buf && h_ref && h_got && b_ref && b_got && rt_ref && rt_got);

    for (size_t i = 0; i < alloc_elems; ++i) {
        f_buf[i] = velri_test_rand_f32(&rng, -12.0f, 12.0f);
    }
    float *f_src = f_buf + 3; /* deliberately unaligned to 64B to test head-peeling + NT body! */

    CHECK(velri_cvt_parallel(VELRI_DTYPE_F32, VELRI_DTYPE_F16, f_src, h_ref + 3, n, VELRI_CPU_BASE,
                             pool, 2) == VELRI_OK);
    CHECK(velri_cvt_parallel(VELRI_DTYPE_F32, VELRI_DTYPE_BF16, f_src, b_ref + 3, n, VELRI_CPU_BASE,
                             pool, 2) == VELRI_OK);
    CHECK(velri_cvt_parallel(VELRI_DTYPE_F16, VELRI_DTYPE_F32, h_ref + 3, rt_ref + 3, n,
                             VELRI_CPU_BASE, pool, 2) == VELRI_OK);

    for (size_t vi = 1; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_cvt_parallel(VELRI_DTYPE_F32, VELRI_DTYPE_F16, f_src, h_got + 3, n, vars[vi],
                                 pool, 2) == VELRI_OK);
        CHECK(memcmp(h_ref + 3, h_got + 3, n * sizeof(velri_fp16)) == 0);

        CHECK(velri_cvt_parallel(VELRI_DTYPE_F32, VELRI_DTYPE_BF16, f_src, b_got + 3, n, vars[vi],
                                 pool, 2) == VELRI_OK);
        CHECK(memcmp(b_ref + 3, b_got + 3, n * sizeof(velri_bf16)) == 0);

        CHECK(velri_cvt_parallel(VELRI_DTYPE_F16, VELRI_DTYPE_F32, h_ref + 3, rt_got + 3, n,
                                 vars[vi], pool, 2) == VELRI_OK);
        CHECK(memcmp(rt_ref + 3, rt_got + 3, n * sizeof(float)) == 0);
    }

    CHECK(velri_cvt_parallel(VELRI_DTYPE_BF16, VELRI_DTYPE_F32, b_ref + 3, rt_ref + 3, n,
                             VELRI_CPU_BASE, pool, 2) == VELRI_OK);
    for (size_t vi = 1; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_cvt_parallel(VELRI_DTYPE_BF16, VELRI_DTYPE_F32, b_ref + 3, rt_got + 3, n,
                                 vars[vi], pool, 2) == VELRI_OK);
        CHECK(memcmp(rt_ref + 3, rt_got + 3, n * sizeof(float)) == 0);
    }

    /* B. Parallel NT-streaming Vec-AXPBY & Vec-Mul (`n = 530,117`, unaligned `+3`, `nw = 2`) */
    CHECK(velri_vec_axpby_f32(n, 1.0f, f_src, -0.25f, rt_ref + 3, rt_got + 3, VELRI_CPU_BASE, pool,
                              2) == VELRI_OK);
    memcpy(rt_ref + 3, rt_got + 3, n * sizeof(float));
    for (size_t vi = 1; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_vec_axpby_f32(n, 1.0f, f_src, -0.25f, f_src, rt_got + 3, vars[vi], pool, 2) ==
              VELRI_OK);
        CHECK(velri_vec_mul_f32(n, f_src, f_src, rt_got + 3, vars[vi], pool, 2) == VELRI_OK);
        CHECK(velri_vec_mul_f32(n, f_src, f_src, rt_ref + 3, VELRI_CPU_BASE, pool, 2) == VELRI_OK);
        CHECK(memcmp(rt_ref + 3, rt_got + 3, n * sizeof(float)) == 0);
    }

    /* C. Parallel NT-streaming Row Quantization (`rows = 260, k = 2048` -> `266 KiB/worker` at
     * `nw=2`) */
    const size_t qrows = 260, qk = 2048;
    int8_t *qi_ref = (int8_t *)aligned_alloc(64, qrows * qk);
    int8_t *qi_got = (int8_t *)aligned_alloc(64, qrows * qk);
    float *qs_ref = (float *)malloc(qrows * sizeof(float));
    float *qs_got = (float *)malloc(qrows * sizeof(float));
    CHECK(velri_quantize_row_f32_i8_parallel(qrows, qk, f_buf, qk, qi_ref, qk, qs_ref,
                                             VELRI_CPU_BASE, pool, 2) == VELRI_OK);
    for (size_t vi = 1; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_quantize_row_f32_i8_parallel(qrows, qk, f_buf, qk, qi_got, qk, qs_got, vars[vi],
                                                 pool, 2) == VELRI_OK);
        for (size_t r = 0; r < qrows; ++r) {
            CHECK(fabsf(qs_ref[r] - qs_got[r]) <= 1e-7f);
            for (size_t j = 0; j < qk; ++j) {
                CHECK(abs((int)qi_ref[r * qk + j] - (int)qi_got[r * qk + j]) <= 1);
            }
        }
    }

    free(f_buf);
    free(h_ref);
    free(h_got);
    free(b_ref);
    free(b_got);
    free(rt_ref);
    free(rt_got);
    free(qi_ref);
    free(qi_got);
    free(qs_ref);
    free(qs_got);

    /* D. Parallel NT-streaming FWHT-256 (`rows = 520, cols = 4096` -> `8.5 MiB` total, `4.25
     * MiB/worker` at `nw=2`) */
    const size_t frows = 520, fcols = 4096;
    float *fx = (float *)aligned_alloc(64, frows * fcols * sizeof(float));
    float *fy_base = (float *)aligned_alloc(64, frows * fcols * sizeof(float));
    float *fy_simd = (float *)aligned_alloc(64, frows * fcols * sizeof(float));
    velri_fp16 *fsuh = (velri_fp16 *)aligned_alloc(64, fcols * sizeof(velri_fp16));
    CHECK(fx && fy_base && fy_simd && fsuh);
    for (size_t i = 0; i < frows * fcols; ++i) {
        fx[i] = velri_test_rand_f32(&rng, -2.0f, 2.0f);
    }
    for (size_t i = 0; i < fcols; ++i) {
        fsuh[i] = (i & 1ULL) ? UINT16_C(0x3C00) : UINT16_C(0xBC00);
    }
    CHECK(velri_fwht_f32(frows, fcols, 256, fx, fcols, VELRI_DTYPE_F16, fsuh, fy_base, fcols,
                         VELRI_CPU_BASE, pool, 2) == VELRI_OK);
    for (size_t vi = 1; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        CHECK(velri_fwht_f32(frows, fcols, 256, fx, fcols, VELRI_DTYPE_F16, fsuh, fy_simd, fcols,
                             vars[vi], pool, 2) == VELRI_OK);
        CHECK(memcmp(fy_base, fy_simd, frows * fcols * sizeof(float)) == 0);
    }
    free(fx);
    free(fy_base);
    free(fy_simd);
    free(fsuh);
}

int main(void) {
    velri_pool *pool = nullptr;
    CHECK(velri_pool_create(16, 0, &pool) == VELRI_OK);

    printf("  1/6 Testing bit-exact F32<->F16/BF16 conversions & INT8 row quantizer...\n");
    test_conversions_and_quant();

    printf("  2/6 Testing RMSNorm (w & 1+w), Fused Add+RMSNorm & LayerNorm...\n");
    test_norms(pool);

    printf("  3/6 Testing RoPE (partial rotary_dim, HALF & INTERLEAVED, F32/BF16/F16)...\n");
    test_rope(pool);

    printf("  4/6 Testing SiLU, GELU_tanh, ReLU2, Clamped GatedMLP & Online Softmax...\n");
    test_activations_and_softmax(pool);

    printf("  5/6 Testing 0-ULP Orthonormal FWHT (.vtq / LTQ v1) & Vec-AXPBY / Vec-Mul...\n");
    test_fwht_and_vec_math(pool);

    printf("  6/6 Testing NT-streaming conversions, row-quant, FWHT-256 & vector math...\n");
    test_nt_streaming_paths(pool);

    velri_pool_destroy(pool);
    printf("[test_ops] ALL TESTS PASSED!\n");
    return 0;
}
