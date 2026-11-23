/* Graph setup follows velri-lib/tools/bench_slang_cpu.c at bf3fcc9d. */
#include "os.h"
#include "velri_lib/velri_lib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check_ok(velri_status st, const char *ctx) {
    if (st != VELRI_OK) {
        fprintf(stderr, "ERROR (%s): velri_status=%d\n", ctx, (int)st);
        exit(1);
    }
}

static void *xalloc(size_t bytes) {
    size_t sz = bytes ? ((bytes + 63U) & ~(size_t)63U) : 64U;
    void *p = velri_os_aligned_alloc(64U, sz);
    if (!p) {
        fprintf(stderr, "FATAL: aligned alloc failed\n");
        exit(1);
    }
    memset(p, 0, sz);
    return p;
}

static double now_sec(void) { return (double)velri_os_monotonic_ns() * 1e-9; }

static int cmp_dbl(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_of(double *arr, size_t n) {
    qsort(arr, n, sizeof(double), cmp_dbl);
    return arr[n / 2U];
}

/* Same signed bytes as fill(VELRI_DTYPE_I8, ...) in the archived bench_cpu. */
static void fill(int8_t *p, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        p[i] = (int)(seed & 255) - 128;
    }
}

static double bench_linear_sec(velri_context *ctx, velri_device_id dev, velri_linear_desc d,
                               size_t rows, size_t m, size_t k, size_t ring_len, size_t iterations) {
    size_t xb = velri_dtype_bytes(d.x_dtype);
    size_t wb = velri_dtype_bytes(d.w_dtype);
    size_t yb = velri_dtype_bytes(d.y_dtype);

    velri_graph *g = nullptr;
    check_ok(velri_graph_create(&g), "g");
    velri_storage_id sx = 0U, sy = 0U, sxs = 0U, sws = 0U;
    velri_storage_id *sw = (velri_storage_id *)calloc(ring_len, sizeof(*sw));
    velri_storage_desc dx = {VELRI_STORAGE_INPUT, d.x_dtype, rows * k * xb, 64U, dev};
    velri_storage_desc dy = {VELRI_STORAGE_OUTPUT, d.y_dtype, rows * m * yb, 64U, dev};
    check_ok(velri_graph_add_storage(g, &dx, &sx), "sx");
    check_ok(velri_graph_add_storage(g, &dy, &sy), "sy");
    if (d.has_x_scale) {
        velri_storage_desc dxs = {VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * sizeof(float), 64U, dev};
        check_ok(velri_graph_add_storage(g, &dxs, &sxs), "sxs");
    }
    if (d.has_w_scale) {
        velri_storage_desc dws = {VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * sizeof(float), 64U, dev};
        check_ok(velri_graph_add_storage(g, &dws, &sws), "sws");
    }

    velri_tensor_view vxt, vyt, vxst, vwst;
    size_t xd[2] = {rows, k}, wd[2] = {m, k}, yd[2] = {rows, m}, xsd[1] = {rows}, wsd[1] = {m};
    velri_view_id vx = 0U, vy = 0U, vxs = VELRI_INVALID_ID, vws = VELRI_INVALID_ID;
    check_ok(velri_tensor_view_contiguous(sx, d.x_dtype, 2U, xd, 0U, &vxt), "vx");
    check_ok(velri_tensor_view_contiguous(sy, d.y_dtype, 2U, yd, 0U, &vyt), "vy");
    check_ok(velri_graph_add_view(g, &vxt, &vx), "avx");
    check_ok(velri_graph_add_view(g, &vyt, &vy), "avy");
    if (d.has_x_scale) {
        check_ok(velri_tensor_view_contiguous(sxs, VELRI_DTYPE_F32, 1U, xsd, 0U, &vxst), "vxs");
        check_ok(velri_graph_add_view(g, &vxst, &vxs), "avxs");
    }
    if (d.has_w_scale) {
        check_ok(velri_tensor_view_contiguous(sws, VELRI_DTYPE_F32, 1U, wsd, 0U, &vwst), "vws");
        check_ok(velri_graph_add_view(g, &vwst, &vws), "avws");
    }

    for (size_t r = 0; r < ring_len; ++r) {
        velri_storage_desc dw = {VELRI_STORAGE_WEIGHT, d.w_dtype, m * k * wb, 64U, dev};
        check_ok(velri_graph_add_storage(g, &dw, &sw[r]), "sw");
        velri_tensor_view vwt;
        velri_view_id vw = 0U;
        check_ok(velri_tensor_view_contiguous(sw[r], d.w_dtype, 2U, wd, 0U, &vwt), "vw");
        check_ok(velri_graph_add_view(g, &vwt, &vw), "avw");
        velri_op_desc op = {
            .kind = VELRI_OP_LINEAR,
            .num_inputs = 5U, .num_outputs = 1U,
            .inputs = {vx, vw, VELRI_INVALID_ID, vxs, vws},
            .outputs = {vy},
            .attr.linear.desc = d,
        };
        check_ok(velri_graph_add_op(g, &op, dev, nullptr), "op");
    }

    velri_plan *plan = nullptr;
    check_ok(velri_plan_prepare(ctx, g, nullptr, &plan), "prep");

    void *hx = xalloc(rows * k * xb);
    void *hy = xalloc(rows * m * yb);
    float *hxs = d.has_x_scale ? (float *)xalloc(rows * sizeof(float)) : nullptr;
    float *hws = d.has_w_scale ? (float *)xalloc(m * sizeof(float)) : nullptr;
    fill(hx, rows * k, 47);
    for (size_t i = 0; i < rows; ++i) hxs[i] = 0.01f;
    for (size_t i = 0; i < m; ++i) hws[i] = 0.02f;
    if (hws) check_ok(velri_plan_bind_weight_host(plan, sws, hws, m * sizeof(float)), "bws");

    void **hw = (void **)calloc(ring_len, sizeof(void *));
    for (size_t r = 0; r < ring_len; ++r) {
        hw[r] = xalloc(m * k * wb);
        fill(hw[r], m * k, 23);
        check_ok(velri_plan_bind_weight_host(plan, sw[r], hw[r], m * k * wb), "bw");
    }

    velri_slot *slot = nullptr;
    check_ok(velri_slot_create(plan, &slot), "slot");
    check_ok(velri_slot_bind_host(slot, sx, hx, rows * k * xb), "bx");
    check_ok(velri_slot_bind_host(slot, sy, hy, rows * m * yb), "by");
    if (hxs) check_ok(velri_slot_bind_host(slot, sxs, hxs, rows * sizeof(float)), "bxs");

    check_ok(velri_slot_run(slot, nullptr, nullptr), "warmup");

    const int8_t *x = hx, *w = hw[0];
    for (size_t sample = 0; sample < 32; ++sample) {
        size_t r = sample * 37 % rows, c = sample * 149 % m;
        int32_t sum = 0;
        for (size_t j = 0; j < k; ++j) sum += (int)x[r * k + j] * w[c * k + j];
        float want = (float)sum * (hxs[r] * hws[c]);
        float got = ((float *)hy)[r * m + c];
        if (!(fabsf(want - got) <= 1e-5f + 1e-6f * fabsf(want))) {
            fprintf(stderr, "bad output r=%zu c=%zu got=%g want=%g\n", r, c, got, want);
            exit(2);
        }
    }
    check_ok(velri_slot_run(slot, nullptr, nullptr), "warmup");
    double times[9];
    for (size_t s = 0; s < 9; ++s) {
        double t0 = now_sec();
        for (size_t i = 0; i < iterations; i += ring_len) {
            check_ok(velri_slot_run(slot, nullptr, nullptr), "run");
        }
        times[s] = (now_sec() - t0) / (double)iterations;
    }

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    for (size_t r = 0; r < ring_len; ++r) velri_os_aligned_free(hw[r]);
    free(hw); free(sw);
    velri_os_aligned_free(hx); velri_os_aligned_free(hy);
    velri_os_aligned_free(hxs); velri_os_aligned_free(hws);
    return median_of(times, 9);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: graph ROWS M K THREADS RING ITERATIONS\n");
        return 1;
    }
    size_t rows = strtoul(argv[1], nullptr, 10), m = strtoul(argv[2], nullptr, 10);
    size_t k = strtoul(argv[3], nullptr, 10), workers = strtoul(argv[4], nullptr, 10);
    size_t ring = strtoul(argv[5], nullptr, 10), iterations = strtoul(argv[6], nullptr, 10);
    if (!rows || !m || !k || !workers || !ring || !iterations || iterations % ring) return 1;
    velri_context_config cfg = {.cpu_workers = workers, .cpu_variant = VELRI_CPU_AUTO};
    velri_context *ctx = nullptr;
    velri_device_id dev;
    check_ok(velri_context_create(&cfg, &ctx), "context");
    check_ok(velri_context_find_device(ctx, "cpu", 0, &dev), "device");
    double sec = bench_linear_sec(ctx, dev, velri_desc_w8a8_f32(), rows, m, k, ring, iterations);
    printf("w8a8,auto,%zu,%zu,%zu,%zu,0,%zu,%.9g,,,%.9g,%.9g\n", rows, m, k, workers, ring,
           1e3 * sec, 2.0 * rows * m * k / sec / 1e12, (double)m * k / sec / 1e9);
    velri_context_destroy(ctx);
    return 0;
}
