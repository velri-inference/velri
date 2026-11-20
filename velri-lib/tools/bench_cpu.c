#define _POSIX_C_SOURCE 200809L
#include "velri_lib/kernels.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The same public-API driver can link the frozen source and the adaptation.
 * Setup/packing is timed separately. Timed calls include validation and the
 * synchronous pool dispatch. No affinity changes or ISA flags in this program. */
static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}
static void check(velri_status s) {
    if (s != VELRI_OK) {
        fprintf(stderr, "status=%d\n", s);
        exit(1);
    }
}
static size_t product(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b)
        fail("size overflow");
    return a * b;
}
static void *alloc(size_t n) {
    if (n > SIZE_MAX - 63)
        fail("allocation overflow");
    void *p = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!p)
        fail("allocation failed");
    return p;
}
static size_t number(const char *s) {
    char *end;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || !*s || *s == '-' || *end || n > SIZE_MAX)
        fail("invalid number");
    return (size_t)n;
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static velri_linear_desc mode(const char *s) {
    if (!strcmp(s, "f32"))
        return velri_desc_f32();
    if (!strcmp(s, "f64"))
        return velri_desc_f64();
    if (!strcmp(s, "f16"))
        return velri_desc_f16(VELRI_DTYPE_F32);
    if (!strcmp(s, "f16out"))
        return velri_desc_f16(VELRI_DTYPE_F16);
    if (!strcmp(s, "bf16"))
        return velri_desc_bf16(VELRI_DTYPE_F32);
    if (!strcmp(s, "bf16out"))
        return velri_desc_bf16(VELRI_DTYPE_BF16);
    if (!strcmp(s, "i8"))
        return velri_desc_i8_i32();
    if (!strcmp(s, "w8a8"))
        return velri_desc_w8a8_f32();
    if (!strcmp(s, "f32f16"))
        return velri_desc_f32x_f16w();
    if (!strcmp(s, "f32bf16"))
        return velri_desc_f32x_bf16w();
    if (!strcmp(s, "f32i8"))
        return velri_desc_f32x_i8w();
    fail("unknown dtype mode");
    return velri_desc_f32();
}
static double load(velri_dtype dt, const void *p, size_t i) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        return ((const float *)p)[i];
    case VELRI_DTYPE_F64:
        return ((const double *)p)[i];
    case VELRI_DTYPE_I8:
        return ((const int8_t *)p)[i];
    case VELRI_DTYPE_I32:
        return ((const int32_t *)p)[i];
    case VELRI_DTYPE_BF16: {
        uint32_t bits = (uint32_t)((const velri_bf16 *)p)[i] << 16;
        float f;
        memcpy(&f, &bits, sizeof f);
        return f;
    }
    case VELRI_DTYPE_F16: {
        uint16_t h = ((const velri_fp16 *)p)[i];
        int exponent = (h >> 10) & 31, fraction = h & 1023;
        double v =
            exponent ? ldexp(1.0 + fraction / 1024.0, exponent - 15) : ldexp((double)fraction, -24);
        return h & 32768 ? -v : v;
    }
    }
    return NAN;
}
static void fill(velri_dtype dt, void *p, size_t n, uint32_t seed) {
    float *temp = (dt == VELRI_DTYPE_F16 || dt == VELRI_DTYPE_BF16)
                      ? alloc(product(n, sizeof(float)))
                      : nullptr;
    for (size_t i = 0; i < n; ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        int v = (int)(seed & 255) - 128;
        double f = (double)v / 128.0;
        if (temp)
            temp[i] = (float)f;
        else if (dt == VELRI_DTYPE_F32)
            ((float *)p)[i] = (float)f;
        else if (dt == VELRI_DTYPE_F64)
            ((double *)p)[i] = f + (double)(seed >> 8) * 0x1p-40;
        else if (dt == VELRI_DTYPE_I8)
            ((int8_t *)p)[i] = (int8_t)v;
    }
    if (dt == VELRI_DTYPE_F16)
        check(velri_cvt_f32_to_f16(temp, p, n, VELRI_CPU_AUTO));
    if (dt == VELRI_DTYPE_BF16)
        check(velri_cvt_f32_to_bf16(temp, p, n, VELRI_CPU_AUTO));
    free(temp);
}
int main(int argc, char **argv) {
    if (argc != 10)
        fail("usage: bench_cpu MODE ISA ROWS M K THREADS PREPARED RING_COUNT ITERATIONS(0=auto)");
    if (strcmp(argv[2], "base") && strcmp(argv[2], "avx2") && strcmp(argv[2], "avx512") &&
        strcmp(argv[2], "auto"))
        fail("unknown ISA");
    velri_linear_desc d = mode(argv[1]);
    velri_cpu_variant v = !strcmp(argv[2], "base")     ? VELRI_CPU_BASE
                          : !strcmp(argv[2], "avx2")   ? VELRI_CPU_AVX2
                          : !strcmp(argv[2], "avx512") ? VELRI_CPU_AVX512
                                                       : VELRI_CPU_AUTO;
    size_t rows = number(argv[3]), m = number(argv[4]), k = number(argv[5]),
           threads = number(argv[6]);
    bool prepared = number(argv[7]) != 0;
    size_t ring = number(argv[8]), iterations = number(argv[9]);
    if (!rows || !m || !k || !threads || threads > 256 || !ring)
        fail("dimensions must be positive");
    velri_linear_typed_plan plan;
    check(velri_linear_typed_query(&d, m, k, prepared, v, &plan));
    size_t wn = product(m, k), xn = product(rows, k), yn = product(rows, m);
    size_t raw_bytes = product(wn, velri_dtype_bytes(d.w_dtype));
    size_t weight_bytes = prepared ? plan.packed_bytes : raw_bytes;
    void *raw = alloc(raw_bytes), *x = alloc(product(xn, velri_dtype_bytes(d.x_dtype)));
    void *y = alloc(product(yn, velri_dtype_bytes(d.y_dtype)));
    float *ws = alloc(product(m, sizeof(float))), *xs = alloc(product(rows, sizeof(float)));
    void **weights = alloc(product(ring, sizeof(void *)));
    fill(d.w_dtype, raw, wn, 23);
    fill(d.x_dtype, x, xn, 47);
    for (size_t i = 0; i < m; ++i)
        ws[i] = 0.02f;
    for (size_t i = 0; i < rows; ++i)
        xs[i] = 0.01f;
    double pack_start = now();
    for (size_t i = 0; i < ring; ++i) {
        weights[i] = alloc(weight_bytes);
        if (prepared)
            check(velri_linear_typed_pack(&plan, raw, wn, k, d.has_w_scale ? ws : nullptr,
                                          d.has_w_scale ? m : 0, weights[i], weight_bytes));
        else
            memcpy(weights[i], raw, weight_bytes);
    }
    double pack_ms = (now() - pack_start) * 1000.0 / ring;
    size_t scratch = rows <= 2 ? 0 : plan.scratch_bytes;
    velri_pool *pool;
    check(velri_pool_create(threads, scratch, &pool));
    velri_linear_run_args a = {.weights = weights[0],
                               .weights_capacity = prepared ? weight_bytes : wn,
                               .w_stride = k,
                               .w_scale = d.has_w_scale && !prepared ? ws : nullptr,
                               .w_scale_count = d.has_w_scale && !prepared ? m : 0,
                               .x = x,
                               .x_count = xn,
                               .rows = rows,
                               .x_stride = k,
                               .x_scale = d.has_x_scale ? xs : nullptr,
                               .x_scale_count = d.has_x_scale ? rows : 0,
                               .y = y,
                               .y_count = yn,
                               .y_stride = m};
    check(velri_linear_typed_run_parallel(&plan, &a, pool, threads));
    for (size_t sample = 0; sample < 8; ++sample) {
        size_t r = (sample * 37) % rows, c = (sample * 149) % m;
        long double sum = 0, magnitude = 0;
        for (size_t j = 0; j < k; ++j) {
            long double term =
                (long double)load(d.x_dtype, x, r * k + j) * load(d.w_dtype, raw, c * k + j);
            sum += term;
            magnitude += fabsl(term);
        }
        double scale = (d.has_x_scale ? xs[r] : 1.0) * (d.has_w_scale ? ws[c] : 1.0);
        double expect = (double)sum * scale, actual = load(d.y_dtype, y, r * m + c);
        double tol = d.y_dtype == VELRI_DTYPE_I32   ? 0
                     : d.y_dtype == VELRI_DTYPE_F64 ? 1e-12 * (double)magnitude
                     : d.y_dtype == VELRI_DTYPE_F16 ? 0.001 * fabs(expect) + 1e-5
                     : d.y_dtype == VELRI_DTYPE_BF16
                         ? 0.008 * fabs(expect) + 1e-5
                         : 2e-5 * (double)magnitude * fabs(scale) + 1e-5;
        if (!isfinite(actual) || fabs(actual - expect) > tol)
            fail("benchmark result failed reference check");
    }
    free(raw);
    double t = now();
    for (size_t i = 0; i < ring; ++i) {
        a.weights = weights[i];
        check(velri_linear_typed_run_parallel(&plan, &a, pool, threads));
    }
    double single = (now() - t) / ring;
    if (!iterations) {
        iterations = (size_t)(0.025 / single);
        if (iterations < ring)
            iterations = ring;
        if (iterations > 20000)
            iterations = 20000;
    }
    double samples[9];
    for (size_t s = 0; s < 9; ++s) {
        t = now();
        for (size_t i = 0; i < iterations; ++i) {
            a.weights = weights[i % ring];
            check(velri_linear_typed_run_parallel(&plan, &a, pool, threads));
        }
        samples[s] = (now() - t) / iterations;
    }
    qsort(samples, 9, sizeof(double), compare);
    printf("%s,%s,%zu,%zu,%zu,%zu,%d,%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%zu,%zu,%.6g\n", argv[1], argv[2],
           rows, m, k, threads, prepared, ring, samples[4] * 1000, samples[0] * 1000,
           samples[8] * 1000, 2.0 * rows * m * k / samples[4] / 1e12,
           (double)weight_bytes / samples[4] / 1e9, weight_bytes, scratch, pack_ms);
    velri_pool_destroy(pool);
    for (size_t i = 0; i < ring; ++i)
        free(weights[i]);
    free(weights);
    free(x);
    free(y);
    free(ws);
    free(xs);
    return 0;
}
