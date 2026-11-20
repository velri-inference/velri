#include "reference.h"
#include "velri_lib/linear.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c)                                                                                   \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c);                                \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

static double read_value(velri_dtype dt, const void *p, size_t i) {
    switch (dt) {
    case VELRI_DTYPE_F64:
        return ((const double *)p)[i];
    case VELRI_DTYPE_F32:
        return ((const float *)p)[i];
    case VELRI_DTYPE_F16:
        return velri_f16_to_f32_scalar(((const velri_fp16 *)p)[i]);
    case VELRI_DTYPE_BF16:
        return velri_bf16_to_f32_scalar(((const velri_bf16 *)p)[i]);
    case VELRI_DTYPE_I8:
        return ((const int8_t *)p)[i];
    case VELRI_DTYPE_I32:
        return ((const int32_t *)p)[i];
    }
    abort();
}
static void fill_rows(void *p, velri_dtype dt, size_t rows, size_t k) {
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < k; ++i) {
            size_t j = r * k + i;
            int sign = (r & 1) ? -1 : 1;
            switch (dt) {
            case VELRI_DTYPE_F64:
                ((double *)p)[j] = sign;
                break;
            case VELRI_DTYPE_F32:
                ((float *)p)[j] = (float)sign;
                break;
            case VELRI_DTYPE_F16:
                ((velri_fp16 *)p)[j] = (sign < 0 ? 0xbc00 : 0x3c00);
                break;
            case VELRI_DTYPE_BF16:
                ((velri_bf16 *)p)[j] = (sign < 0 ? 0xbf80 : 0x3f80);
                break;
            case VELRI_DTYPE_I8:
                ((int8_t *)p)[j] = (int8_t)sign;
                break;
            default:
                abort();
            }
        }
}

/* Cross every blocking boundary with the advertised scratch size, including
 * maximally unaligned caller scratch. SIMD assembly writes need guard pages:
 * compiler sanitizers alone cannot instrument inline assembly memory operands. */
static void workspace_bounds(void) {
    velri_linear_desc modes[] = {velri_desc_f32(),
                                 velri_desc_f64(),
                                 velri_desc_f16(VELRI_DTYPE_F32),
                                 velri_desc_f16(VELRI_DTYPE_F16),
                                 velri_desc_bf16(VELRI_DTYPE_F32),
                                 velri_desc_bf16(VELRI_DTYPE_BF16),
                                 velri_desc_f32x_f16w(),
                                 velri_desc_f32x_bf16w(),
                                 velri_desc_i8_i32(),
                                 velri_desc_w8a8_f32(),
                                 velri_desc_f32x_i8w()};
    const size_t rows = 337, m = 517, k = 1031;
    for (size_t d = 0; d < sizeof(modes) / sizeof(modes[0]); ++d) {
        void *x = malloc(rows * k * velri_dtype_bytes(modes[d].x_dtype));
        void *w = malloc(m * k * velri_dtype_bytes(modes[d].w_dtype));
        float ws[m], xs[rows];
        CHECK(x && w);
        fill_rows(x, modes[d].x_dtype, rows, k);
        fill_rows(w, modes[d].w_dtype, m, k);
        for (size_t i = 0; i < m; ++i)
            ws[i] = 1;
        for (size_t i = 0; i < rows; ++i)
            xs[i] = 1;
        for (int variant = VELRI_CPU_AVX2; variant <= VELRI_CPU_AVX512; ++variant)
            for (int prep = 0; prep < 2; ++prep) {
                velri_linear_typed_plan p;
                velri_status status =
                    velri_linear_typed_query(&modes[d], m, k, prep, (velri_cpu_variant)variant, &p);
                if (status == VELRI_ERR_UNSUPPORTED)
                    continue;
                CHECK(status == VELRI_OK);
                velri_guard_buf output =
                    velri_guard_alloc(rows * m * velri_dtype_bytes(modes[d].y_dtype), 8);
                velri_guard_buf workspace = velri_guard_alloc(p.scratch_bytes + 64, 1);
                memset(workspace.ptr, 0xa5, p.scratch_bytes + 64);
                unsigned char *scratch = (unsigned char *)workspace.ptr + 1;
                void *packed_storage =
                    prep ? aligned_alloc(64, (p.packed_bytes + 71) & ~(size_t)63) : nullptr;
                void *packed = prep ? (unsigned char *)packed_storage + 8 : nullptr;
                if (prep)
                    CHECK(velri_linear_typed_pack(
                              &p, w, m * k, k, modes[d].has_w_scale ? ws : nullptr,
                              modes[d].has_w_scale ? m : 0, packed, p.packed_bytes) == VELRI_OK);
                velri_linear_run_args a = {.weights = prep ? packed : w,
                                           .weights_capacity = prep ? p.packed_bytes : m * k,
                                           .w_stride = k,
                                           .w_scale = !prep && modes[d].has_w_scale ? ws : nullptr,
                                           .w_scale_count = !prep && modes[d].has_w_scale ? m : 0,
                                           .x = x,
                                           .x_count = rows * k,
                                           .rows = rows,
                                           .x_stride = k,
                                           .x_scale = modes[d].has_x_scale ? xs : nullptr,
                                           .x_scale_count = modes[d].has_x_scale ? rows : 0,
                                           .y = output.ptr,
                                           .y_count = rows * m,
                                           .y_stride = m,
                                           .scratch = scratch,
                                           .scratch_bytes = p.scratch_bytes};
                CHECK(velri_linear_typed_run(&p, &a) == VELRI_OK);
                CHECK(((unsigned char *)workspace.ptr)[0] == 0xa5);
                for (size_t i = p.scratch_bytes; i < p.scratch_bytes + 63; ++i)
                    CHECK(scratch[i] == 0xa5);
                for (size_t r = 0; r < rows; ++r)
                    for (size_t c = 0; c < m; ++c) {
                        double expected = (modes[d].y_dtype == VELRI_DTYPE_BF16) ? 1032 : 1031;
                        if ((r ^ c) & 1)
                            expected = -expected;
                        CHECK(read_value(modes[d].y_dtype, output.ptr, r * m + c) == expected);
                    }
                free(packed_storage);
                velri_guard_free(&output);
                velri_guard_free(&workspace);
            }
        free(x);
        free(w);
    }
}

static void integer_wrap(void) {
    const size_t m = 35, k = 131073, rows = 3;
    int8_t *w = malloc(m * k), *x = malloc(rows * k);
    int32_t y[rows * m], bias[m];
    float yf[rows * m], ws[m], xs[rows];
    CHECK(x && w);
    memset(x, 0x80, rows * k);
    memset(w, 0x80, m * k);
    for (size_t c = 0; c < m; ++c) {
        bias[c] = INT32_MAX;
        ws[c] = 1;
    }
    for (size_t r = 0; r < rows; ++r)
        xs[r] = 1;
    const uint32_t dot = (uint32_t)((int64_t)k * 16384);
    int32_t dot_signed, expected;
    memcpy(&dot_signed, &dot, 4);
    uint32_t bits = dot + (uint32_t)INT32_MAX;
    memcpy(&expected, &bits, 4);
    for (int v = VELRI_CPU_BASE; v <= VELRI_CPU_AVX512; ++v)
        for (int prep = 0; prep < 2; ++prep)
            for (int scaled = 0; scaled < 2; ++scaled) {
                velri_linear_desc d = scaled ? velri_desc_w8a8_f32() : velri_desc_i8_i32();
                velri_linear_typed_plan p;
                velri_status status =
                    velri_linear_typed_query(&d, m, k, prep, (velri_cpu_variant)v, &p);
                if (status == VELRI_ERR_UNSUPPORTED)
                    continue;
                CHECK(status == VELRI_OK);
                void *packed = prep ? malloc(p.packed_bytes) : nullptr;
                if (prep)
                    CHECK(velri_linear_typed_pack(&p, w, m * k, k, scaled ? ws : nullptr,
                                                  scaled ? m : 0, packed,
                                                  p.packed_bytes) == VELRI_OK);
                velri_pool *pool;
                CHECK(velri_pool_create(3, p.scratch_bytes, &pool) == VELRI_OK);
                velri_linear_run_args a = {.weights = prep ? packed : w,
                                           .weights_capacity = prep ? p.packed_bytes : m * k,
                                           .w_stride = k,
                                           .w_scale = scaled && !prep ? ws : nullptr,
                                           .w_scale_count = scaled && !prep ? m : 0,
                                           .x = x,
                                           .x_count = rows * k,
                                           .rows = rows,
                                           .x_stride = k,
                                           .x_scale = scaled ? xs : nullptr,
                                           .x_scale_count = scaled ? rows : 0,
                                           .bias = scaled ? nullptr : bias,
                                           .bias_count = scaled ? 0 : m,
                                           .y = scaled ? (void *)yf : (void *)y,
                                           .y_count = rows * m,
                                           .y_stride = m};
                for (size_t n = 1; n <= 3; n += 2) {
                    a.rows = n;
                    CHECK(velri_linear_typed_run_parallel(&p, &a, pool, 3) == VELRI_OK);
                    for (size_t i = 0; i < n * m; ++i)
                        CHECK(scaled ? yf[i] == (float)dot_signed : y[i] == expected);
                }
                velri_pool_destroy(pool);
                free(packed);
            }
    free(x);
    free(w);
}

typedef struct {
    size_t calls[16];
    size_t workers;
} counts;
static void count_worker(void *ptr, size_t id, size_t nw, void *scratch, size_t bytes) {
    (void)scratch;
    (void)bytes;
    counts *c = ptr;
    CHECK(nw == c->workers);
    ++c->calls[id];
}
static void pool_contract(void) {
    velri_pool *pool = nullptr;
    CHECK(velri_pool_create(2, SIZE_MAX, &pool) == VELRI_ERR_OVERFLOW);
    CHECK(!pool);
    CHECK(velri_pool_create(16, 0, &pool) == VELRI_OK);
    for (size_t n = 0; n < 1000; ++n) {
        counts c = {.workers = 1 + (n * 7) % 16};
        velri_pool_parallel(pool, c.workers, count_worker, &c);
        for (size_t i = 0; i < 16; ++i)
            CHECK(c.calls[i] == (i < c.workers));
    }
    velri_pool_destroy(pool);
}

static void tiny_quantization(void) {
    float x[65], scale;
    int8_t q[65];
    for (int v = VELRI_CPU_BASE; v <= VELRI_CPU_AVX512; ++v) {
        if (!velri_cpu_variant_supported((velri_cpu_variant)v))
            continue;
        for (int tiny = 0; tiny < 2; ++tiny) {
            float amplitude = tiny ? FLT_TRUE_MIN : FLT_MIN;
            for (size_t i = 0; i < 65; ++i)
                x[i] = (i % 3 == 0) ? 0 : ((i & 1) ? amplitude : -amplitude);
            CHECK(velri_quantize_row_f32_i8(1, 65, x, 65, q, 65, &scale, (velri_cpu_variant)v) ==
                  VELRI_OK);
            CHECK(isfinite(scale) && scale > 0);
            for (size_t i = 0; i < 65; ++i) {
                int expected = (int)rint((double)x[i] / scale);
                if (expected > 127)
                    expected = 127;
                if (expected < -127)
                    expected = -127;
                CHECK(q[i] == expected);
            }
        }
    }
}

/* Use long-double accumulation over the actual stored values. Unlike a second
 * SIMD implementation, this oracle does not share the kernel reduction tree. */
static void fill_random(void *p, velri_dtype dt, size_t n, uint64_t *rng) {
    for (size_t i = 0; i < n; ++i) {
        double value = (double)velri_test_rand_f32(rng, -1, 1);
        if (dt == VELRI_DTYPE_F64)
            ((double *)p)[i] = value + 1.0 / ((double)i + 3.0);
        else if (dt == VELRI_DTYPE_F32)
            ((float *)p)[i] = (float)value;
        else if (dt == VELRI_DTYPE_F16)
            ((velri_fp16 *)p)[i] = velri_f32_to_f16_scalar((float)value);
        else
            ((velri_bf16 *)p)[i] = velri_f32_to_bf16_scalar((float)value);
    }
}
static void floating_accuracy(void) {
    const size_t rows = 19, m = 37, k = 4099;
    velri_linear_desc modes[] = {velri_desc_f64(), velri_desc_f32(),
                                 velri_desc_f16(VELRI_DTYPE_F32), velri_desc_bf16(VELRI_DTYPE_F32)};
    uint64_t rng = 0x1234567812345678ULL;
    for (size_t d = 0; d < 4; ++d) {
        void *x = malloc(rows * k * velri_dtype_bytes(modes[d].x_dtype));
        void *w = malloc(m * k * velri_dtype_bytes(modes[d].w_dtype));
        void *y = malloc(rows * m * velri_dtype_bytes(modes[d].y_dtype));
        long double reference[rows * m];
        CHECK(x && w && y);
        fill_random(x, modes[d].x_dtype, rows * k, &rng);
        fill_random(w, modes[d].w_dtype, m * k, &rng);
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < m; ++c) {
                long double sum = 0;
                for (size_t j = 0; j < k; ++j)
                    sum += (long double)read_value(modes[d].x_dtype, x, r * k + j) *
                           read_value(modes[d].w_dtype, w, c * k + j);
                reference[r * m + c] = sum;
            }
        for (int v = VELRI_CPU_BASE; v <= VELRI_CPU_AVX512; ++v)
            for (int prep = 0; prep < 2; ++prep) {
                velri_linear_typed_plan p;
                velri_status status =
                    velri_linear_typed_query(&modes[d], m, k, prep, (velri_cpu_variant)v, &p);
                if (status == VELRI_ERR_UNSUPPORTED)
                    continue;
                CHECK(status == VELRI_OK);
                void *packed = prep ? malloc(p.packed_bytes) : nullptr;
                if (prep)
                    CHECK(velri_linear_typed_pack(&p, w, m * k, k, nullptr, 0, packed,
                                                  p.packed_bytes) == VELRI_OK);
                velri_pool *pool;
                CHECK(velri_pool_create(3, p.scratch_bytes, &pool) == VELRI_OK);
                velri_linear_run_args a = {.weights = prep ? packed : w,
                                           .weights_capacity = prep ? p.packed_bytes : m * k,
                                           .w_stride = k,
                                           .x = x,
                                           .x_count = rows * k,
                                           .x_stride = k,
                                           .y = y,
                                           .y_count = rows * m,
                                           .y_stride = m};
                const size_t batches[] = {1, 3, 19};
                for (size_t b = 0; b < 3; ++b) {
                    a.rows = batches[b];
                    CHECK(velri_linear_typed_run_parallel(&p, &a, pool, 3) == VELRI_OK);
                    long double err2 = 0, ref2 = 0;
                    for (size_t i = 0; i < a.rows * m; ++i) {
                        double got = read_value(modes[d].y_dtype, y, i);
                        CHECK(isfinite(got));
                        long double delta = got - reference[i];
                        err2 += delta * delta;
                        ref2 += reference[i] * reference[i];
                    }
                    long double limit = d == 0 ? 1e-12L : 1e-5L;
                    CHECK(err2 <= limit * limit * ref2);
                }
                free(packed);
                velri_pool_destroy(pool);
            }
        free(x);
        free(w);
        free(y);
    }
}

/* Emulated runs must exercise the advertised ISA instead of silently skipping
 * it. Ordinary CTest runs still test every implementation available on the host. */
static void expect_isa(const char *name) {
    velri_cpu_variant expected = VELRI_CPU_BASE;
    if (!strcmp(name, "avx2"))
        expected = VELRI_CPU_AVX2;
    else if (!strcmp(name, "avx512"))
        expected = VELRI_CPU_AVX512;
    else
        CHECK(!strcmp(name, "base"));
    CHECK(velri_cpu_resolve_variant(VELRI_CPU_AUTO) == expected);
    velri_linear_desc d = velri_desc_f64();
    velri_linear_typed_plan plan;
    CHECK(velri_linear_typed_query(&d, 37, 19, true, VELRI_CPU_AUTO, &plan) == VELRI_OK);
    CHECK(plan.variant == expected);
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--dispatch")) {
        /* QEMU TCG's masked-load handling can fault on protected pages even
         * when the corresponding lanes are disabled. Keep this ISA-selection
         * smoke test separate; all guard-page tests still run natively. */
        expect_isa(argv[2]);
        floating_accuracy();
        tiny_quantization();
        puts("CPU dispatch and numerical smoke checks passed");
        return 0;
    }
    CHECK(argc <= 2);
    if (argc == 2)
        expect_isa(argv[1]);
    pool_contract();
    integer_wrap();
    workspace_bounds();
    tiny_quantization();
    floating_accuracy();
    puts("CPU contract: pool transitions, integer overflow, workspace, alignment and accuracy "
         "passed");
}
