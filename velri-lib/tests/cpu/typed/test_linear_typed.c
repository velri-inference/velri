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

static void fill_buffer(velri_dtype dt, void *buf, size_t count, uint64_t *rng) {
    for (size_t i = 0; i < count; ++i) {
        float fv = velri_test_rand_f32(rng, -1.5f, 1.5f);
        switch (dt) {
        case VELRI_DTYPE_F32:
            ((float *)buf)[i] = fv;
            break;
        case VELRI_DTYPE_F64:
            ((double *)buf)[i] = (double)fv;
            break;
        case VELRI_DTYPE_F16:
            ((velri_fp16 *)buf)[i] = velri_f32_to_f16_scalar(fv);
            break;
        case VELRI_DTYPE_BF16:
            ((velri_bf16 *)buf)[i] = velri_f32_to_bf16_scalar(fv);
            break;
        case VELRI_DTYPE_I8: {
            /* Explicitly hit full signed int8 range [-128 .. 127] */
            int r = (int)(velri_test_rng(rng) & 0xFFU) - 128;
            ((int8_t *)buf)[i] = (int8_t)r;
            break;
        }
        case VELRI_DTYPE_I32:
            ((int32_t *)buf)[i] = (int32_t)(velri_test_rng(rng) % 2001U) - 1000;
            break;
        }
    }
}

static void fill_canary_y(velri_dtype dt, void *buf, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        switch (dt) {
        case VELRI_DTYPE_F32:
            ((float *)buf)[i] = NAN;
            break;
        case VELRI_DTYPE_F64:
            ((double *)buf)[i] = NAN;
            break;
        case VELRI_DTYPE_F16:
            ((velri_fp16 *)buf)[i] = UINT16_C(0x7E55);
            break;
        case VELRI_DTYPE_BF16:
            ((velri_bf16 *)buf)[i] = UINT16_C(0x7FC5);
            break;
        case VELRI_DTYPE_I32:
            ((int32_t *)buf)[i] = INT32_C(0x5A5A5A5A);
            break;
        default:
            break;
        }
    }
}

static void verify_canary_gap(velri_dtype dt, const void *buf, size_t idx) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        CHECK(isnan(((const float *)buf)[idx]));
        break;
    case VELRI_DTYPE_F64:
        CHECK(isnan(((const double *)buf)[idx]));
        break;
    case VELRI_DTYPE_F16:
        CHECK(((const velri_fp16 *)buf)[idx] == UINT16_C(0x7E55));
        break;
    case VELRI_DTYPE_BF16:
        CHECK(((const velri_bf16 *)buf)[idx] == UINT16_C(0x7FC5));
        break;
    case VELRI_DTYPE_I32:
        CHECK(((const int32_t *)buf)[idx] == INT32_C(0x5A5A5A5A));
        break;
    default:
        break;
    }
}

static int g_cur_di = -1, g_cur_var = -1, g_cur_prep = -1, g_cur_par = 0;

static void compare_outputs(const velri_linear_desc *d, size_t rows, size_t m, size_t k,
                            const void *got, const void *ref, size_t y_stride) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < m; ++c) {
            size_t idx = r * y_stride + c;
            if (d->y_dtype == VELRI_DTYPE_I32) {
                int32_t g = ((const int32_t *)got)[idx];
                int32_t e = ((const int32_t *)ref)[idx];
                if (g != e) {
                    fprintf(stderr, "I32 mismatch at (%zu,%zu) k=%zu: got %d vs ref %d\n", r, c, k,
                            g, e);
                }
                CHECK(g == e); /* 0-ULP BIT-EXACT! */
            } else if (d->y_dtype == VELRI_DTYPE_F64) {
                double g = ((const double *)got)[idx];
                double e = ((const double *)ref)[idx];
                double diff = fabs(g - e);
                double tol = 1e-11 * ((double)k + 1.0) * (fabs(e) + 1.0);
                CHECK(!isnan(g) && diff <= tol);
            } else if (d->y_dtype == VELRI_DTYPE_F32) {
                float g = ((const float *)got)[idx];
                float e = ((const float *)ref)[idx];
                float diff = fabsf(g - e);
                float base_eps = (d->acc_dtype == VELRI_DTYPE_I32) ? 2e-6f : 1.5e-5f;
                float tol = base_eps * sqrtf((float)k + 1.0f) * (fabsf(e) + 1.0f) + 2e-5f;
                if (isnan(g) || diff > tol) {
                    fprintf(stderr,
                            "F32 mismatch [di=%d var=%d prep=%d par=%d] at (%zu,%zu) m=%zu k=%zu: "
                            "got %.7g vs ref %.7g\n",
                            g_cur_di, g_cur_var, g_cur_prep, g_cur_par, r, c, m, k, g, e);
                }
                CHECK(!isnan(g) && diff <= tol);
            } else if (d->y_dtype == VELRI_DTYPE_F16) {
                float g = velri_f16_to_f32_scalar(((const velri_fp16 *)got)[idx]);
                float e = velri_f16_to_f32_scalar(((const velri_fp16 *)ref)[idx]);
                float tol = 2e-3f * (fabsf(e) + 1.0f) + 2e-3f;
                CHECK(!isnan(g) && fabsf(g - e) <= tol);
            } else if (d->y_dtype == VELRI_DTYPE_BF16) {
                float g = velri_bf16_to_f32_scalar(((const velri_bf16 *)got)[idx]);
                float e = velri_bf16_to_f32_scalar(((const velri_bf16 *)ref)[idx]);
                float tol = 1.6e-2f * (fabsf(e) + 1.0f) + 1.6e-2f;
                CHECK(!isnan(g) && fabsf(g - e) <= tol);
            }
        }
        /* Verify inter-row gap canaries were untouched! */
        for (size_t c = m; c < y_stride; ++c) {
            verify_canary_gap(d->y_dtype, got, r * y_stride + c);
        }
    }
}

/* Ask every supported implementation instead of duplicating its workspace budget. */
static size_t test_scratch_bytes(void) {
    const velri_linear_desc descs[] = {
        velri_desc_f32(),
        velri_desc_f64(),
        velri_desc_f16(VELRI_DTYPE_F32),
        velri_desc_f16(VELRI_DTYPE_F16),
        velri_desc_f32x_f16w(),
        velri_desc_bf16(VELRI_DTYPE_F32),
        velri_desc_bf16(VELRI_DTYPE_BF16),
        velri_desc_f32x_bf16w(),
        velri_desc_i8_i32(),
        velri_desc_w8a8_f32(),
        velri_desc_f32x_i8w(),
    };
    size_t bytes = 0;
    const velri_cpu_variant variants[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    for (size_t d = 0; d < sizeof(descs) / sizeof(descs[0]); ++d) {
        for (size_t v = 0; v < sizeof(variants) / sizeof(variants[0]); ++v) {
            if (!velri_cpu_variant_supported(variants[v]))
                continue;
            for (int prepared = 0; prepared < 2; ++prepared) {
                velri_linear_typed_plan plan;
                CHECK(velri_linear_typed_query(&descs[d], 64, 1029, prepared, variants[v], &plan) ==
                      VELRI_OK);
                if (plan.scratch_bytes > bytes)
                    bytes = plan.scratch_bytes;
            }
        }
    }
    return bytes;
}

static void test_all_modes_and_shapes(velri_pool *pool) {
    const velri_linear_desc descs[] = {
        velri_desc_f32(),
        velri_desc_f64(),
        velri_desc_f16(VELRI_DTYPE_F32),
        velri_desc_f16(VELRI_DTYPE_F16),
        velri_desc_f32x_f16w(),
        velri_desc_bf16(VELRI_DTYPE_F32),
        velri_desc_bf16(VELRI_DTYPE_BF16),
        velri_desc_f32x_bf16w(),
        velri_desc_i8_i32(),
        velri_desc_w8a8_f32(),
        velri_desc_f32x_i8w(),
    };

    const struct {
        size_t rows, m, k;
    } shapes[] = {
        {0, 16, 16},  {1, 1, 1},     {1, 7, 3},     {1, 32, 65},    {1, 35, 129},
        {1, 65, 67},  {1, 128, 128},
        {2, 17, 19},  {2, 64, 67},   {3, 15, 17},   {6, 31, 33},    {13, 33, 63},
        {14, 32, 64}, {15, 35, 131}, {29, 65, 259}, {173, 67, 515}, {31, 33, 1029},
    };

    const velri_cpu_variant variants[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512,
                                          VELRI_CPU_AUTO};
    uint64_t rng = UINT64_C(0x123456789ABCDEF0);

    uint8_t *scratch = (uint8_t *)malloc(velri_pool_scratch_bytes(pool) + 64);
    CHECK(scratch != nullptr);

    for (size_t di = 0; di < sizeof(descs) / sizeof(descs[0]); ++di) {
        const velri_linear_desc d = descs[di];
        velri_dtype b_dt =
            (d.y_dtype == VELRI_DTYPE_F64)
                ? VELRI_DTYPE_F64
                : ((d.y_dtype == VELRI_DTYPE_I32) ? VELRI_DTYPE_I32 : VELRI_DTYPE_F32);

        for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); ++si) {
            size_t rows = shapes[si].rows;
            size_t m = shapes[si].m;
            size_t k = shapes[si].k;
            size_t xs = k + 3;
            size_t ws = k + 5;
            size_t ys = m + 4;

            size_t x_cnt = (rows > 0 ? rows : 1) * xs;
            size_t w_cnt = m * ws;
            size_t y_cnt = (rows > 0 ? rows : 1) * ys;

            /* Allocate with intentional +1 element offset to verify natural-alignment-only! */
            size_t x_eb = velri_dtype_bytes(d.x_dtype);
            size_t w_eb = velri_dtype_bytes(d.w_dtype);
            size_t y_eb = velri_dtype_bytes(d.y_dtype);
            size_t b_eb = velri_dtype_bytes(b_dt);

            uint8_t *x_raw =
                (uint8_t *)aligned_alloc(64, (((x_cnt + 8) * x_eb + 64) + 63ULL) & ~63ULL);
            uint8_t *w_raw =
                (uint8_t *)aligned_alloc(64, (((w_cnt + 8) * w_eb + 64) + 63ULL) & ~63ULL);
            uint8_t *y_got_raw =
                (uint8_t *)aligned_alloc(64, (((y_cnt + 8) * y_eb + 64) + 63ULL) & ~63ULL);
            uint8_t *y_ref =
                (uint8_t *)aligned_alloc(64, (((y_cnt + 8) * y_eb + 64) + 63ULL) & ~63ULL);
            uint8_t *bias = (uint8_t *)aligned_alloc(64, (((m + 8) * b_eb + 64) + 63ULL) & ~63ULL);
            float *x_scale = (float *)aligned_alloc(
                64, ((((rows ? rows : 1) + 16) * sizeof(float)) + 63ULL) & ~63ULL);
            float *w_scale =
                (float *)aligned_alloc(64, (((m + 16) * sizeof(float)) + 63ULL) & ~63ULL);

            void *x = x_raw + velri_dtype_align(d.x_dtype);
            void *w = w_raw + velri_dtype_align(d.w_dtype);
            void *y_got = y_got_raw + velri_dtype_align(d.y_dtype);

            fill_buffer(d.x_dtype, x, x_cnt, &rng);
            fill_buffer(d.w_dtype, w, w_cnt, &rng);
            fill_buffer(b_dt, bias, m, &rng);
            for (size_t r = 0; r < (rows ? rows : 1); ++r) {
                x_scale[r] = velri_test_rand_f32(&rng, 0.005f, 0.05f);
            }
            for (size_t c = 0; c < m; ++c) {
                w_scale[c] = velri_test_rand_f32(&rng, 0.005f, 0.05f);
            }

            bool use_bias = ((si + di) % 2 == 0);
            ref_linear_run(&d, rows, m, k, x, xs, d.has_x_scale ? x_scale : nullptr, w, ws,
                           d.has_w_scale ? w_scale : nullptr, use_bias ? bias : nullptr, y_ref, ys);

            for (size_t vi = 0; vi < sizeof(variants) / sizeof(variants[0]); ++vi) {
                velri_cpu_variant var = variants[vi];
                if (!velri_cpu_variant_supported(var)) {
                    continue;
                }
                for (int prep = 0; prep <= 1; ++prep) {
                    g_cur_di = (int)di;
                    g_cur_var = (int)var;
                    g_cur_prep = prep;
                    g_cur_par = 0;
                    velri_linear_typed_plan plan;
                    velri_status query = velri_linear_typed_query(&d, m, k, prep != 0, var, &plan);
                    if (query == VELRI_ERR_UNSUPPORTED && var == VELRI_CPU_AVX512)
                        continue;
                    CHECK(query == VELRI_OK);

                    void *packed = nullptr;
                    if (prep) {
                        packed = aligned_alloc(64, (plan.packed_bytes + 63ULL) & ~63ULL);
                        CHECK(velri_linear_typed_pack(
                                  &plan, w, w_cnt, ws, d.has_w_scale ? w_scale : nullptr,
                                  d.has_w_scale ? m : 0, packed, plan.packed_bytes) == VELRI_OK);
                    }

                    fill_canary_y(d.y_dtype, y_got, y_cnt);
                    velri_linear_run_args args = {
                        .weights = prep ? packed : w,
                        .weights_capacity = prep ? plan.packed_bytes : w_cnt,
                        .w_stride = prep ? 0 : ws,
                        .w_scale = (!prep && d.has_w_scale) ? w_scale : nullptr,
                        .w_scale_count = (!prep && d.has_w_scale) ? m : 0,
                        .x = x,
                        .x_count = rows ? x_cnt : 0,
                        .rows = rows,
                        .x_stride = xs,
                        .x_scale = d.has_x_scale ? x_scale : nullptr,
                        .x_scale_count = (d.has_x_scale && rows) ? rows : 0,
                        .bias = use_bias ? bias : nullptr,
                        .bias_count = use_bias ? m : 0,
                        .y = y_got,
                        .y_count = rows ? y_cnt : 0,
                        .y_stride = ys,
                        .scratch = scratch + 7, /* intentionally unaligned scratch! */
                        .scratch_bytes = plan.scratch_bytes,
                    };

                    CHECK(velri_linear_typed_run(&plan, &args) == VELRI_OK);
                    if (rows == 0) {
                        for (size_t q = 0; q < y_cnt; ++q) {
                            verify_canary_gap(d.y_dtype, y_got, q);
                        }
                    } else {
                        compare_outputs(&d, rows, m, k, y_got, y_ref, ys);
                    }

                    /* Also verify parallel pool execution across 16 workers! */
                    if (rows > 0 && (si % 3 == 0)) {
                        g_cur_par = 1;
                        fill_canary_y(d.y_dtype, y_got, y_cnt);
                        args.scratch = nullptr;
                        args.scratch_bytes = 0;
                        CHECK(velri_linear_typed_run_parallel(&plan, &args, pool, 16) == VELRI_OK);
                        compare_outputs(&d, rows, m, k, y_got, y_ref, ys);
                    }

                    free(packed);
                }
            }

            free(x_raw);
            free(w_raw);
            free(y_got_raw);
            free(y_ref);
            free(bias);
            free(x_scale);
            free(w_scale);
        }
    }
    free(scratch);
}

/* Explicitly verify `-128 * -128` extreme corner across BASE, AVX2, AVX512 (`0-ULP`!) */
static void test_int8_extreme_minus128(void) {
    const size_t rows = 15, m = 33, k = 1029;
    int8_t *x = (int8_t *)malloc(rows * k);
    int8_t *w = (int8_t *)malloc(m * k);
    int32_t *y = (int32_t *)malloc(rows * m * sizeof(int32_t));
    uint8_t *sc = (uint8_t *)malloc(test_scratch_bytes() + 1);
    memset(x, -128, rows * k);
    memset(w, -128, m * k);

    const int32_t expected = (int32_t)k * 16384;
    velri_linear_desc d = velri_desc_i8_i32();
    const velri_cpu_variant vars[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};

    for (size_t vi = 0; vi < 3; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) {
            continue;
        }
        for (int prep = 0; prep <= 1; ++prep) {
            velri_linear_typed_plan plan;
            CHECK(velri_linear_typed_query(&d, m, k, prep != 0, vars[vi], &plan) == VELRI_OK);
            void *pk = prep ? aligned_alloc(64, (plan.packed_bytes + 63ULL) & ~63ULL) : nullptr;
            if (prep) {
                CHECK(velri_linear_typed_pack(&plan, w, m * k, k, nullptr, 0, pk,
                                              plan.packed_bytes) == VELRI_OK);
            }
            const size_t sub_rows_list[2] = {1, 15};
            for (size_t sri = 0; sri < 2; ++sri) {
                size_t sub_r = sub_rows_list[sri];
                memset(y, 0, rows * m * sizeof(int32_t));
                velri_linear_run_args a = {
                    .weights = prep ? pk : w,
                    .weights_capacity = prep ? plan.packed_bytes : m * k,
                    .w_stride = k,
                    .x = x,
                    .x_count = sub_r * k,
                    .rows = sub_r,
                    .x_stride = k,
                    .y = y,
                    .y_count = sub_r * m,
                    .y_stride = m,
                    .scratch = sc,
                    .scratch_bytes = plan.scratch_bytes,
                };
                CHECK(velri_linear_typed_run(&plan, &a) == VELRI_OK);
                for (size_t i = 0; i < sub_r * m; ++i) {
                    CHECK(y[i] == expected);
                }
            }
            free(pk);
        }
    }
    free(x);
    free(w);
    free(y);
    free(sc);
}

/* Guard-page (`PROT_NONE`) and aliasing/overflow validation contracts */
static void test_guard_pages_and_validation(void) {
    const velri_linear_desc descs[] = {velri_desc_f32(), velri_desc_f16(VELRI_DTYPE_F16),
                                       velri_desc_bf16(VELRI_DTYPE_BF16), velri_desc_i8_i32()};
    const velri_cpu_variant vars[] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    uint64_t rng = UINT64_C(0xCAFEBABE12345678);
    uint8_t *sc = (uint8_t *)malloc(test_scratch_bytes() + 1);

    for (size_t di = 0; di < 4; ++di) {
        const velri_linear_desc d = descs[di];
        const size_t rows_list[2] = {1, 15};
        for (size_t rli = 0; rli < 2; ++rli) {
            size_t rows = rows_list[rli];
            size_t m = 35, k = 67;
            velri_guard_buf gx = velri_guard_alloc(rows * k * velri_dtype_bytes(d.x_dtype),
                                                   velri_dtype_align(d.x_dtype));
            velri_guard_buf gw = velri_guard_alloc(m * k * velri_dtype_bytes(d.w_dtype),
                                                   velri_dtype_align(d.w_dtype));
            velri_guard_buf gy = velri_guard_alloc(rows * m * velri_dtype_bytes(d.y_dtype),
                                                   velri_dtype_align(d.y_dtype));
            fill_buffer(d.x_dtype, gx.ptr, rows * k, &rng);
            fill_buffer(d.w_dtype, gw.ptr, m * k, &rng);

            for (size_t vi = 0; vi < 3; ++vi) {
                if (!velri_cpu_variant_supported(vars[vi])) {
                    continue;
                }
                for (int prep = 0; prep <= 1; ++prep) {
                    velri_linear_typed_plan plan;
                    CHECK(velri_linear_typed_query(&d, m, k, prep != 0, vars[vi], &plan) ==
                          VELRI_OK);
                    velri_guard_buf gpk = {0};
                    if (prep) {
                        gpk = velri_guard_alloc(plan.packed_bytes, 8);
                        CHECK(velri_linear_typed_pack(&plan, gw.ptr, m * k, k, nullptr, 0, gpk.ptr,
                                                      plan.packed_bytes) == VELRI_OK);
                    }
                    velri_linear_run_args a = {
                        .weights = prep ? gpk.ptr : gw.ptr,
                        .weights_capacity = prep ? plan.packed_bytes : m * k,
                        .w_stride = k,
                        .x = gx.ptr,
                        .x_count = rows * k,
                        .rows = rows,
                        .x_stride = k,
                        .y = gy.ptr,
                        .y_count = rows * m,
                        .y_stride = m,
                        .scratch = sc,
                        .scratch_bytes = plan.scratch_bytes,
                    };
                    CHECK(velri_linear_typed_run(&plan, &a) == VELRI_OK);
                    velri_guard_free(&gpk);
                }
            }
            velri_guard_free(&gx);
            velri_guard_free(&gw);
            velri_guard_free(&gy);
        }
    }

    /* Verify read-only input aliasing (`x == w == bias`) succeeds while output aliasing fails! */
    float shared_in[64];
    float out_buf[64];
    for (int i = 0; i < 64; ++i) {
        shared_in[i] = 0.25f * (float)(i % 7);
        out_buf[i] = -999.0f;
    }
    velri_linear_desc df32 = velri_desc_f32();
    velri_linear_typed_plan p;
    CHECK(velri_linear_typed_query(&df32, 8, 8, false, VELRI_CPU_AUTO, &p) == VELRI_OK);

    velri_linear_run_args ok_alias = {.weights = shared_in,
                                      .weights_capacity = 64,
                                      .w_stride = 8,
                                      .x = shared_in,
                                      .x_count = 64,
                                      .rows = 8,
                                      .x_stride = 8,
                                      .bias = shared_in,
                                      .bias_count = 8,
                                      .y = out_buf,
                                      .y_count = 64,
                                      .y_stride = 8,
                                      .scratch = sc,
                                      .scratch_bytes = p.scratch_bytes};
    CHECK(velri_linear_typed_run(&p, &ok_alias) == VELRI_OK);

    /* Reject `y == x` before mutating `shared_in[0]`! */
    float saved0 = shared_in[0];
    ok_alias.y = shared_in;
    CHECK(velri_linear_typed_run(&p, &ok_alias) == VELRI_ERR_INVALID);
    CHECK(shared_in[0] == saved0);

    /* Reject size_t overflow */
    CHECK(velri_linear_typed_query(&df32, SIZE_MAX / 2, 16, true, VELRI_CPU_AUTO, &p) ==
          VELRI_ERR_OVERFLOW);

    /* Verify `run_slice` boundary contracts */
    ok_alias.y = out_buf;
    CHECK(velri_linear_typed_run_slice(&p, &ok_alias, 2, 6, 0, 8) == VELRI_OK);
    CHECK(velri_linear_typed_run_slice(&p, &ok_alias, 6, 2, 0, 8) == VELRI_ERR_INVALID);

    free(sc);
}

int main(void) {
    const velri_cpu_info *info = velri_get_cpu_info();
    printf("[test_linear_typed] HW Threads=%d L3 Domains=%d AVX2=%d AVX512F=%d BF16=%d VNNI=%d\n",
           info->hardware_threads, info->num_l3_domains, info->has_avx2, info->has_avx512f,
           info->has_avx512_bf16, info->has_avx512_vnni);

    velri_pool *pool = nullptr;
    CHECK(velri_pool_create(16, test_scratch_bytes(), &pool) == VELRI_OK);

    printf("  1/3 Running 11-mode x 15-shape x {Prepared, RowMajor} x {BASE, AVX2, AVX512}...\n");
    test_all_modes_and_shapes(pool);

    printf("  2/3 Running INT8->INT32 -128x-128 bit-exactness verification...\n");
    test_int8_extreme_minus128();

    printf("  3/3 Running PROT_NONE guard-page, aliasing & overflow contracts...\n");
    test_guard_pages_and_validation();

    velri_pool_destroy(pool);
    printf("[test_linear_typed] ALL TESTS PASSED!\n");
    return 0;
}
