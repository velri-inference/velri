#include "os.h"
#include "velri_lib/kernels.h"
#include "velri_lib/vulkan.h"
#include "reference.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void fill_dtype(velri_dtype dt, void *buf, size_t n, uint64_t *rng, float scale) {
    if (dt == VELRI_DTYPE_F32) {
        float *p = (float *)buf;
        for (size_t i = 0; i < n; ++i) p[i] = velri_test_rand_f32(rng, -scale, scale);
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
 * Part 1: Strict Verification of Every Vulkan Kernel vs Independent C23 Ref
 * ============================================================================ */
static void verify_vk_linear_case(velri_vk_ctx *vk, const char *label, velri_linear_desc desc,
                                  uint32_t rows, uint32_t m, uint32_t k,
                                  uint32_t x_pad, uint32_t w_pad, uint32_t y_pad,
                                  bool with_bias, double tol) {
    uint32_t xs = k + x_pad, ws = k + w_pad, ys = m + y_pad;
    size_t xb = velri_dtype_bytes(desc.x_dtype);
    size_t wb = velri_dtype_bytes(desc.w_dtype);
    size_t yb = velri_dtype_bytes(desc.y_dtype);

    void *hx = xalloc((size_t)rows * xs * xb);
    void *hw = xalloc((size_t)m * ws * wb);
    void *hy_ref = xalloc((size_t)rows * ys * yb);
    void *hy_vk = xalloc((size_t)rows * ys * yb);
    void *hbias = with_bias ? xalloc((size_t)m * yb) : nullptr;
    float *hxs = desc.has_x_scale ? xalloc((size_t)rows * sizeof(float)) : nullptr;
    float *hws = desc.has_w_scale ? xalloc((size_t)m * sizeof(float)) : nullptr;

    uint64_t rng = 0x55AA0001ULL ^ ((uint64_t)rows << 16) ^ ((uint64_t)m << 4) ^ (uint64_t)with_bias;
    fill_dtype(desc.x_dtype, hx, (size_t)rows * xs, &rng, 0.5f);
    fill_dtype(desc.w_dtype, hw, (size_t)m * ws, &rng, 0.5f);
    if (hbias) fill_dtype(desc.y_dtype, hbias, m, &rng, 0.5f);
    if (hxs) fill_dtype(VELRI_DTYPE_F32, hxs, rows, &rng, 0.05f);
    if (hws) fill_dtype(VELRI_DTYPE_F32, hws, m, &rng, 0.05f);

    ref_linear_run(&desc, rows, m, k, hx, xs, hxs, hw, ws, hws, hbias, hy_ref, ys);

    velri_vk_buf *dx = nullptr, *dw = nullptr, *dy = nullptr;
    velri_vk_buf *dbias = nullptr, *dxs = nullptr, *dws = nullptr;
    check_ok(velri_vk_buf_alloc(vk, (size_t)rows * xs * xb, &dx), "alloc dx");
    check_ok(velri_vk_buf_alloc(vk, (size_t)m * ws * wb, &dw), "alloc dw");
    check_ok(velri_vk_buf_alloc(vk, (size_t)rows * ys * yb, &dy), "alloc dy");
    check_ok(velri_vk_buf_upload(vk, dx, hx, (size_t)rows * xs * xb), "up dx");
    check_ok(velri_vk_buf_upload(vk, dw, hw, (size_t)m * ws * wb), "up dw");
    if (hbias) {
        check_ok(velri_vk_buf_alloc(vk, (size_t)m * yb, &dbias), "alloc dbias");
        check_ok(velri_vk_buf_upload(vk, dbias, hbias, (size_t)m * yb), "up dbias");
    }
    if (hxs) {
        check_ok(velri_vk_buf_alloc(vk, (size_t)rows * sizeof(float), &dxs), "alloc dxs");
        check_ok(velri_vk_buf_upload(vk, dxs, hxs, (size_t)rows * sizeof(float)), "up dxs");
    }
    if (hws) {
        check_ok(velri_vk_buf_alloc(vk, (size_t)m * sizeof(float), &dws), "alloc dws");
        check_ok(velri_vk_buf_upload(vk, dws, hws, (size_t)m * sizeof(float)), "up dws");
    }

    check_ok(velri_vk_linear(vk, &desc, rows, m, k, dx, xs, dw, ws, dbias, dxs, dws, dy, ys, 1u, nullptr),
             label);
    check_ok(velri_vk_buf_download(vk, hy_vk, dy, (size_t)rows * ys * yb), "down dy");

    double err = max_abs_diff_2d(desc.y_dtype, hy_ref, hy_vk, rows, m, ys);
    if (!(err <= tol)) {
        fprintf(stderr, "FAIL Vulkan %s (rows=%u m=%u k=%u bias=%d): max_err=%.3e > tol=%.3e\n",
                label, rows, m, k, (int)with_bias, err, tol);
        exit(1);
    }

    velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dw); velri_vk_buf_free(vk, dy);
    velri_vk_buf_free(vk, dbias); velri_vk_buf_free(vk, dxs); velri_vk_buf_free(vk, dws);
    xfree(hx); xfree(hw); xfree(hy_ref); xfree(hy_vk);
    xfree(hbias); xfree(hxs); xfree(hws);
}

static void run_vk_verification(velri_vk_ctx *vk) {
    const velri_vk_info *info = velri_vk_get_info(vk);
    printf("\n=== Verifying Vulkan Slang kernels vs independent C23 reference ===\n");

    /* GEMV (aligned + odd/unaligned) & GEMM (aligned full-tile fast path + odd boundary path) */
    struct {
        uint32_t rows, m, k, xp, wp, yp;
    } shapes[] = {
        {1u, 64u, 128u, 0u, 0u, 0u},        /* aligned GEMV */
        {1u, 67u, 131u, 5u, 7u, 3u},        /* unaligned odd GEMV */
        {2u, 67u, 131u, 0u, 0u, 0u},        /* 2-row GEMV */
        {128u, 128u, 64u, 0u, 0u, 0u},      /* 64x64 CoopMat aligned fast-store */
        {140u, 145u, 131u, 5u, 7u, 3u},     /* 64x64 CoopMat odd boundary + unaligned */
        {64u, 512u, 1024u, 0u, 0u, 0u},     /* Split-K aligned fast-store (F32/F16/BF16/W8A8) */
        {67u, 259u, 1031u, 3u, 5u, 2u},     /* Split-K odd boundary + unaligned + odd K */
        {1280u, 1920u, 96u, 0u, 0u, 0u},    /* 128x128 CoopMat aligned fast-store (150 WGs >= 140) */
        {1293u, 1931u, 99u, 3u, 5u, 2u},    /* 128x128 CoopMat odd boundary + unaligned */
    };

    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); ++si) {
        for (int b = 0; b <= 1; ++b) {
            verify_vk_linear_case(vk, "vk_f32", velri_desc_f32(),
                                  shapes[si].rows, shapes[si].m, shapes[si].k,
                                  shapes[si].xp, shapes[si].wp, shapes[si].yp, b != 0, 2e-4);
            if (info->has_f16_i8) {
                verify_vk_linear_case(vk, "vk_f16", velri_desc_f16(VELRI_DTYPE_F32),
                                      shapes[si].rows, shapes[si].m, shapes[si].k,
                                      shapes[si].xp, shapes[si].wp, shapes[si].yp, b != 0, 2e-3);
            }
            verify_vk_linear_case(vk, "vk_bf16", velri_desc_bf16(VELRI_DTYPE_F32),
                                  shapes[si].rows, shapes[si].m, shapes[si].k,
                                  shapes[si].xp, shapes[si].wp, shapes[si].yp, b != 0, 2e-3);
            if (info->has_f16_i8 && info->has_int_dot) {
                verify_vk_linear_case(vk, "vk_w8a8", velri_desc_w8a8_f32(),
                                      shapes[si].rows, shapes[si].m, shapes[si].k,
                                      shapes[si].xp, shapes[si].wp, shapes[si].yp, b != 0, 2e-4);
            }
        }
    }
    printf("  [PASS] Linear GEMV + GEMM (F32, F16, BF16, W8A8; aligned & odd shapes; with/without bias).\n");

    /* RMSNorm (F32, BF16, F16) */
    const velri_dtype ndts[3] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F16};
    const double ntols[3] = {2e-5, 2e-2, 4e-3};
    for (int di = 0; di < 3; ++di) {
        if (ndts[di] == VELRI_DTYPE_F16 && !info->has_f16_i8) continue;
        uint32_t rows = 7u, dim = 131u, xs = 137u, ys = 139u;
        size_t eb = velri_dtype_bytes(ndts[di]);
        void *hx = xalloc(rows * xs * eb), *hw = xalloc(dim * eb);
        void *hy_ref = xalloc(rows * ys * eb), *hy_vk = xalloc(rows * ys * eb);
        uint64_t rng = 0x11110001ULL + (uint64_t)di;
        fill_dtype(ndts[di], hx, rows * xs, &rng, 2.0f);
        fill_dtype(ndts[di], hw, dim, &rng, 1.5f);

        velri_vk_buf *dx = nullptr, *dw = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, rows * xs * eb, &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, dim * eb, &dw), "alloc");
        check_ok(velri_vk_buf_alloc(vk, rows * ys * eb, &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, rows * xs * eb), "up");
        check_ok(velri_vk_buf_upload(vk, dw, hw, dim * eb), "up");

        for (int uo = 0; uo <= 1; ++uo) {
            for (int hw_flag = 0; hw_flag <= 1; ++hw_flag) {
                ref_rmsnorm(ndts[di], ndts[di], rows, dim, 1e-5f, uo != 0,
                            hx, xs, hw_flag ? hw : nullptr, hy_ref, ys);
                check_ok(velri_vk_rmsnorm(vk, ndts[di], rows, dim, 1e-5f, uo != 0,
                                          dx, xs, hw_flag ? dw : nullptr, dy, ys, 1u, nullptr),
                         "vk_rmsnorm");
                check_ok(velri_vk_buf_download(vk, hy_vk, dy, rows * ys * eb), "down");
                double err = max_abs_diff_2d(ndts[di], hy_ref, hy_vk, rows, dim, ys);
                if (!(err <= ntols[di])) {
                    fprintf(stderr, "FAIL Vulkan rmsnorm dt=%d err=%.3e\n", (int)ndts[di], err);
                    exit(1);
                }
            }
        }
        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dw); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hw); xfree(hy_ref); xfree(hy_vk);

        /* Also verify aligned 4096 single-pass register-cached path */
        {
            uint32_t arows = 4u, adim = 4096u;
            void *ahx = xalloc(arows * adim * eb), *ahw = xalloc(adim * eb);
            void *ahy_ref = xalloc(arows * adim * eb), *ahy_vk = xalloc(arows * adim * eb);
            uint64_t arng = 0x11119001ULL + (uint64_t)di;
            fill_dtype(ndts[di], ahx, arows * adim, &arng, 2.0f);
            fill_dtype(ndts[di], ahw, adim, &arng, 1.0f);
            ref_rmsnorm(ndts[di], ndts[di], arows, adim, 1e-5f, di == 1, ahx, adim, ahw, ahy_ref, adim);
            velri_vk_buf *adx = nullptr, *adw = nullptr, *ady = nullptr;
            check_ok(velri_vk_buf_alloc(vk, arows * adim * eb, &adx), "alloc");
            check_ok(velri_vk_buf_alloc(vk, adim * eb, &adw), "alloc");
            check_ok(velri_vk_buf_alloc(vk, arows * adim * eb, &ady), "alloc");
            check_ok(velri_vk_buf_upload(vk, adx, ahx, arows * adim * eb), "up");
            check_ok(velri_vk_buf_upload(vk, adw, ahw, adim * eb), "up");
            check_ok(velri_vk_rmsnorm(vk, ndts[di], arows, adim, 1e-5f, di == 1, adx, adim, adw,
                                      ady, adim, 1u, nullptr),
                     "vk_rmsnorm_aligned");
            check_ok(velri_vk_buf_download(vk, ahy_vk, ady, arows * adim * eb), "down");
            double err = max_abs_diff_2d(ndts[di], ahy_ref, ahy_vk, arows, adim, adim);
            if (!(err <= ntols[di])) {
                fprintf(stderr, "FAIL Vulkan rmsnorm aligned dt=%d err=%.3e\n", (int)ndts[di], err);
                exit(1);
            }
            velri_vk_buf_free(vk, adx); velri_vk_buf_free(vk, adw); velri_vk_buf_free(vk, ady);
            xfree(ahx); xfree(ahw); xfree(ahy_ref); xfree(ahy_vk);
        }
    }
    printf("  [PASS] RMSNorm (F32, BF16, F16; aligned 4096 & odd; unit_offset 0/1; weight on/off).\n");

    /* RoPE (F32, BF16, HALF & INTERLEAVED, partial rotary_dim + aligned 1D fast HALF path) */
    const velri_dtype rdts[2] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16};
    const double rtols[2] = {1e-5, 2e-2};
    for (int di = 0; di < 2; ++di) {
        uint32_t nt = 7u, nh = 5u, hd = 72u, rd = 48u, po = 5u;
        uint32_t hsx = hd + 4u, tsx = nh * hsx + 3u;
        uint32_t hsy = hd + 2u, tsy = nh * hsy + 5u;
        size_t eb = velri_dtype_bytes(rdts[di]);
        void *hx = xalloc(nt * tsx * eb), *hy_ref = xalloc(nt * tsy * eb), *hy_vk = xalloc(nt * tsy * eb);
        float *hcos = xalloc((po + nt) * (rd / 2u) * sizeof(float));
        float *hsin = xalloc((po + nt) * (rd / 2u) * sizeof(float));
        uint64_t rng = 0x22220001ULL + (uint64_t)di;
        fill_dtype(rdts[di], hx, nt * tsx, &rng, 2.0f);
        for (size_t i = 0; i < (po + nt) * (rd / 2u); ++i) {
            float ang = velri_test_rand_f32(&rng, -3.14f, 3.14f);
            hcos[i] = cosf(ang);
            hsin[i] = sinf(ang);
        }
        velri_vk_buf *dx = nullptr, *dcos = nullptr, *dsin = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, nt * tsx * eb, &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (po + nt) * (rd / 2u) * sizeof(float), &dcos), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (po + nt) * (rd / 2u) * sizeof(float), &dsin), "alloc");
        check_ok(velri_vk_buf_alloc(vk, nt * tsy * eb, &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, nt * tsx * eb), "up");
        check_ok(velri_vk_buf_upload(vk, dcos, hcos, (po + nt) * (rd / 2u) * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, dsin, hsin, (po + nt) * (rd / 2u) * sizeof(float)), "up");

        for (int m = 0; m <= 1; ++m) {
            velri_rope_mode mode = m ? VELRI_ROPE_INTERLEAVED : VELRI_ROPE_HALF;
            ref_rope(rdts[di], nt, nh, hd, rd, po, hx, tsx, hsx, hcos, hsin, hy_ref, tsy, hsy, mode);
            check_ok(velri_vk_rope(vk, rdts[di], nt, nh, hd, rd, po, dx, tsx, hsx, dcos, dsin,
                                   dy, tsy, hsy, mode, 1u, nullptr),
                     "vk_rope");
            check_ok(velri_vk_buf_download(vk, hy_vk, dy, nt * tsy * eb), "down");
            double err = 0.0;
            for (size_t t = 0; t < nt; ++t)
                for (size_t h = 0; h < nh; ++h)
                    for (size_t d = 0; d < hd; ++d) {
                        size_t off = t * tsy + h * hsy + d;
                        double df = fabs(load_elem_f64(rdts[di], hy_ref, off) -
                                         load_elem_f64(rdts[di], hy_vk, off));
                        if (df > err) err = df;
                    }
            if (!(err <= rtols[di])) {
                fprintf(stderr, "FAIL Vulkan rope dt=%d mode=%d err=%.3e\n", (int)rdts[di], m, err);
                exit(1);
            }
        }
        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dcos);
        velri_vk_buf_free(vk, dsin); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hy_ref); xfree(hy_vk); xfree(hcos); xfree(hsin);

        /* Also verify aligned 1D fast HALF path (head_dim=rotary_dim=128) */
        {
            uint32_t ant = 9u, anh = 8u, ahd = 128u, apo = 3u;
            void *ahx = xalloc(ant * anh * ahd * eb), *ahy_ref = xalloc(ant * anh * ahd * eb);
            void *ahy_vk = xalloc(ant * anh * ahd * eb);
            float *ahcos = xalloc((apo + ant) * 64u * sizeof(float));
            float *ahsin = xalloc((apo + ant) * 64u * sizeof(float));
            uint64_t arng = 0x22229001ULL + (uint64_t)di;
            fill_dtype(rdts[di], ahx, ant * anh * ahd, &arng, 2.0f);
            for (size_t i = 0; i < (apo + ant) * 64u; ++i) {
                float ang = velri_test_rand_f32(&arng, -3.14f, 3.14f);
                ahcos[i] = cosf(ang);
                ahsin[i] = sinf(ang);
            }
            ref_rope(rdts[di], ant, anh, ahd, ahd, apo, ahx, anh * ahd, ahd, ahcos, ahsin,
                     ahy_ref, anh * ahd, ahd, VELRI_ROPE_HALF);
            velri_vk_buf *adx = nullptr, *adcos = nullptr, *adsin = nullptr, *ady = nullptr;
            check_ok(velri_vk_buf_alloc(vk, ant * anh * ahd * eb, &adx), "alloc");
            check_ok(velri_vk_buf_alloc(vk, (apo + ant) * 64u * sizeof(float), &adcos), "alloc");
            check_ok(velri_vk_buf_alloc(vk, (apo + ant) * 64u * sizeof(float), &adsin), "alloc");
            check_ok(velri_vk_buf_alloc(vk, ant * anh * ahd * eb, &ady), "alloc");
            check_ok(velri_vk_buf_upload(vk, adx, ahx, ant * anh * ahd * eb), "up");
            check_ok(velri_vk_buf_upload(vk, adcos, ahcos, (apo + ant) * 64u * sizeof(float)), "up");
            check_ok(velri_vk_buf_upload(vk, adsin, ahsin, (apo + ant) * 64u * sizeof(float)), "up");
            check_ok(velri_vk_rope(vk, rdts[di], ant, anh, ahd, ahd, apo, adx, anh * ahd, ahd,
                                   adcos, adsin, ady, anh * ahd, ahd, VELRI_ROPE_HALF, 1u, nullptr),
                     "vk_rope_fast");
            check_ok(velri_vk_buf_download(vk, ahy_vk, ady, ant * anh * ahd * eb), "down");
            double err = max_abs_diff_2d(rdts[di], ahy_ref, ahy_vk, ant * anh, ahd, ahd);
            if (!(err <= rtols[di])) {
                fprintf(stderr, "FAIL Vulkan rope aligned dt=%d err=%.3e\n", (int)rdts[di], err);
                exit(1);
            }
            velri_vk_buf_free(vk, adx); velri_vk_buf_free(vk, adcos);
            velri_vk_buf_free(vk, adsin); velri_vk_buf_free(vk, ady);
            xfree(ahx); xfree(ahy_ref); xfree(ahy_vk); xfree(ahcos); xfree(ahsin);
        }
    }
    printf("  [PASS] RoPE (F32, BF16; HALF & INTERLEAVED; partial & aligned 1D fast path).\n");

    /* GatedMLP (SwiGLU & GeGLU) */
    {
        uint32_t n = 1031u;
        float *hg = xalloc(n * sizeof(float)), *hu = xalloc(n * sizeof(float));
        float *hy_ref = xalloc(n * sizeof(float)), *hy_vk = xalloc(n * sizeof(float));
        uint64_t rng = 0x33330001ULL;
        fill_dtype(VELRI_DTYPE_F32, hg, n, &rng, 4.0f);
        fill_dtype(VELRI_DTYPE_F32, hu, n, &rng, 2.0f);
        velri_vk_buf *dg = nullptr, *du = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dg), "alloc");
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &du), "alloc");
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dg, hg, n * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, du, hu, n * sizeof(float)), "up");

        const velri_act_kind acts[2] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH};
        const float clamps[2] = {0.0f, 1.5f};
        for (int ai = 0; ai < 2; ++ai) {
            for (int ci = 0; ci < 2; ++ci) {
                ref_gated_mlp_f32(acts[ai], n, clamps[ci], hg, hu, hy_ref);
                check_ok(velri_vk_gated_mlp_f32(vk, acts[ai], n, clamps[ci], dg, du, dy, 1u, nullptr),
                         "vk_gated");
                check_ok(velri_vk_buf_download(vk, hy_vk, dy, n * sizeof(float)), "down");
                double err = max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, 1u, n, n);
                if (!(err <= 5e-4)) {
                    fprintf(stderr, "FAIL Vulkan gated_mlp act=%d err=%.3e\n", ai, err);
                    exit(1);
                }
            }
        }
        velri_vk_buf_free(vk, dg); velri_vk_buf_free(vk, du); velri_vk_buf_free(vk, dy);
        xfree(hg); xfree(hu); xfree(hy_ref); xfree(hy_vk);
    }
    printf("  [PASS] GatedMLP (SwiGLU & GeGLU; unclamped & symmetric clamp).\n");

    /* Softmax */
    {
        uint32_t rows = 7u, cols = 131u, xs = 137u, ys = 139u;
        float *hx = xalloc(rows * xs * sizeof(float));
        float *hy_ref = xalloc(rows * ys * sizeof(float)), *hy_vk = xalloc(rows * ys * sizeof(float));
        uint64_t rng = 0x44440001ULL;
        fill_dtype(VELRI_DTYPE_F32, hx, rows * xs, &rng, 20.0f);
        ref_softmax_f32(rows, cols, hx, xs, hy_ref, ys);
        velri_vk_buf *dx = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, rows * xs * sizeof(float), &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, rows * ys * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, rows * xs * sizeof(float)), "up");
        check_ok(velri_vk_softmax_f32(vk, rows, cols, dx, xs, dy, ys, 1u, nullptr), "vk_softmax");
        check_ok(velri_vk_buf_download(vk, hy_vk, dy, rows * ys * sizeof(float)), "down");
        double err = max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, rows, cols, ys);
        if (!(err <= 1e-4)) {
            fprintf(stderr, "FAIL Vulkan softmax err=%.3e\n", err);
            exit(1);
        }
        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hy_ref); xfree(hy_vk);
    }
    /* Softmax aligned fast-path (cols=4096) */
    {
        uint32_t rows = 4u, cols = 4096u;
        float *hx = xalloc(rows * cols * sizeof(float));
        float *hy_ref = xalloc(rows * cols * sizeof(float)), *hy_vk = xalloc(rows * cols * sizeof(float));
        uint64_t rng = 0x44440002ULL;
        fill_dtype(VELRI_DTYPE_F32, hx, rows * cols, &rng, 15.0f);
        ref_softmax_f32(rows, cols, hx, cols, hy_ref, cols);
        velri_vk_buf *dx = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, rows * cols * sizeof(float), &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, rows * cols * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, rows * cols * sizeof(float)), "up");
        check_ok(velri_vk_softmax_f32(vk, rows, cols, dx, cols, dy, cols, 1u, nullptr), "vk_softmax");
        check_ok(velri_vk_buf_download(vk, hy_vk, dy, rows * cols * sizeof(float)), "down");
        double err = max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, rows, cols, cols);
        if (!(err <= 1e-4)) {
            fprintf(stderr, "FAIL Vulkan softmax aligned err=%.3e\n", err);
            exit(1);
        }
        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hy_ref); xfree(hy_vk);
    }
    printf("  [PASS] Softmax F32 (wide dynamic range [-20, 20]; aligned & odd).\n");

    /* FusedAddRMSNorm (both aligned 4096 and odd 131) */
    {
        const uint32_t dims[2] = {4096u, 131u};
        const uint32_t strides[2] = {4096u, 137u};
        for (int ci = 0; ci < 2; ++ci) {
            uint32_t rows = 5u, dim = dims[ci], st = strides[ci];
            float *hr = xalloc(rows * st * sizeof(float)), *hr_ref = xalloc(rows * st * sizeof(float));
            float *hr_vk = xalloc(rows * st * sizeof(float)), *hx = xalloc(rows * st * sizeof(float));
            float *hw = xalloc(dim * sizeof(float));
            float *hy_ref = xalloc(rows * st * sizeof(float)), *hy_vk = xalloc(rows * st * sizeof(float));
            uint64_t rng = 0x55550001ULL + (uint64_t)ci;
            fill_dtype(VELRI_DTYPE_F32, hr, rows * st, &rng, 2.0f);
            fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 2.0f);
            fill_dtype(VELRI_DTYPE_F32, hw, dim, &rng, 1.0f);
            memcpy(hr_ref, hr, rows * st * sizeof(float));
            ref_fused_add_rmsnorm_f32(rows, dim, 1e-5f, ci == 0, hr_ref, st, hx, st, hw, hy_ref, st);

            velri_vk_buf *dr = nullptr, *dx = nullptr, *dw = nullptr, *dy = nullptr;
            check_ok(velri_vk_buf_alloc(vk, rows * st * sizeof(float), &dr), "alloc");
            check_ok(velri_vk_buf_alloc(vk, rows * st * sizeof(float), &dx), "alloc");
            check_ok(velri_vk_buf_alloc(vk, dim * sizeof(float), &dw), "alloc");
            check_ok(velri_vk_buf_alloc(vk, rows * st * sizeof(float), &dy), "alloc");
            check_ok(velri_vk_buf_upload(vk, dr, hr, rows * st * sizeof(float)), "up");
            check_ok(velri_vk_buf_upload(vk, dx, hx, rows * st * sizeof(float)), "up");
            check_ok(velri_vk_buf_upload(vk, dw, hw, dim * sizeof(float)), "up");
            check_ok(velri_vk_fused_add_rmsnorm_f32(vk, rows, dim, 1e-5f, ci == 0, dr, st, dx, st,
                                                    dw, dy, st, 1u, nullptr),
                     "vk_fused_add_rmsnorm");
            check_ok(velri_vk_buf_download(vk, hr_vk, dr, rows * st * sizeof(float)), "down");
            check_ok(velri_vk_buf_download(vk, hy_vk, dy, rows * st * sizeof(float)), "down");
            double er = max_abs_diff_2d(VELRI_DTYPE_F32, hr_ref, hr_vk, rows, dim, st);
            double ey = max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, rows, dim, st);
            if (!(er <= 1e-5 && ey <= 1e-4)) {
                fprintf(stderr, "FAIL Vulkan fused_add_rmsnorm ci=%d er=%.3e ey=%.3e\n", ci, er, ey);
                exit(1);
            }
            velri_vk_buf_free(vk, dr); velri_vk_buf_free(vk, dx);
            velri_vk_buf_free(vk, dw); velri_vk_buf_free(vk, dy);
            xfree(hr); xfree(hr_ref); xfree(hr_vk); xfree(hx); xfree(hw); xfree(hy_ref); xfree(hy_vk);
        }
    }
    printf("  [PASS] FusedAddRMSNorm F32 (aligned 4096 & odd 131; unit_offset 0/1).\n");

    /* LayerNorm (aligned 4096 and odd 131) */
    {
        const uint32_t dims[2] = {4096u, 131u};
        const uint32_t strides[2] = {4096u, 139u};
        for (int ci = 0; ci < 2; ++ci) {
            uint32_t rows = 5u, dim = dims[ci], st = strides[ci];
            float *hx = xalloc(rows * st * sizeof(float)), *hw = xalloc(dim * sizeof(float));
            float *hb = xalloc(dim * sizeof(float));
            float *hy_ref = xalloc(rows * st * sizeof(float)), *hy_vk = xalloc(rows * st * sizeof(float));
            uint64_t rng = 0x56560001ULL + (uint64_t)ci;
            fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 3.0f);
            fill_dtype(VELRI_DTYPE_F32, hw, dim, &rng, 1.0f);
            fill_dtype(VELRI_DTYPE_F32, hb, dim, &rng, 1.0f);
            ref_layernorm_f32(rows, dim, 1e-5f, hx, st, hw, hb, hy_ref, st);

            velri_vk_buf *dx = nullptr, *dw = nullptr, *db = nullptr, *dy = nullptr;
            check_ok(velri_vk_buf_alloc(vk, rows * st * sizeof(float), &dx), "alloc");
            check_ok(velri_vk_buf_alloc(vk, dim * sizeof(float), &dw), "alloc");
            check_ok(velri_vk_buf_alloc(vk, dim * sizeof(float), &db), "alloc");
            check_ok(velri_vk_buf_alloc(vk, rows * st * sizeof(float), &dy), "alloc");
            check_ok(velri_vk_buf_upload(vk, dx, hx, rows * st * sizeof(float)), "up");
            check_ok(velri_vk_buf_upload(vk, dw, hw, dim * sizeof(float)), "up");
            check_ok(velri_vk_buf_upload(vk, db, hb, dim * sizeof(float)), "up");
            check_ok(velri_vk_layernorm_f32(vk, rows, dim, 1e-5f, dx, st, dw, db, dy, st, 1u, nullptr),
                     "vk_layernorm");
            check_ok(velri_vk_buf_download(vk, hy_vk, dy, rows * st * sizeof(float)), "down");
            double err = max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, rows, dim, st);
            if (!(err <= 1e-4)) {
                fprintf(stderr, "FAIL Vulkan layernorm ci=%d err=%.3e\n", ci, err);
                exit(1);
            }
            velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dw);
            velri_vk_buf_free(vk, db); velri_vk_buf_free(vk, dy);
            xfree(hx); xfree(hw); xfree(hb); xfree(hy_ref); xfree(hy_vk);
        }
    }
    printf("  [PASS] LayerNorm F32 (aligned 4096 & odd 131; weight + bias).\n");

    /* Activations (SiLU, GELU_tanh, ReLU2) & VecAxpby & VecMul */
    {
        uint32_t n = 1031u;
        float *ha = xalloc(n * sizeof(float)), *hb = xalloc(n * sizeof(float));
        float *hy_ref = xalloc(n * sizeof(float)), *hy_vk = xalloc(n * sizeof(float));
        uint64_t rng = 0x67670001ULL;
        fill_dtype(VELRI_DTYPE_F32, ha, n, &rng, 4.0f);
        fill_dtype(VELRI_DTYPE_F32, hb, n, &rng, 2.0f);
        velri_vk_buf *da = nullptr, *db = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &da), "alloc");
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &db), "alloc");
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, da, ha, n * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, db, hb, n * sizeof(float)), "up");

        const velri_act_kind ak[3] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH, VELRI_ACT_RELU2};
        for (int i = 0; i < 3; ++i) {
            ref_activation_f32(ak[i], n, ha, hy_ref);
            check_ok(velri_vk_activation_f32(vk, ak[i], n, da, dy, 1u, nullptr), "vk_act");
            check_ok(velri_vk_buf_download(vk, hy_vk, dy, n * sizeof(float)), "down");
            double err = max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, 1u, n, n);
            if (!(err <= 5e-4)) { fprintf(stderr, "FAIL Vulkan act=%d err=%.3e\n", i, err); exit(1); }
        }
        ref_vec_axpby_f32(n, 0.75f, ha, -1.25f, hb, hy_ref);
        check_ok(velri_vk_vec_axpby_f32(vk, n, 0.75f, da, -1.25f, db, dy, 1u, nullptr), "vk_axpby");
        check_ok(velri_vk_buf_download(vk, hy_vk, dy, n * sizeof(float)), "down");
        if (!(max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, 1u, n, n) <= 1e-5)) {
            fprintf(stderr, "FAIL Vulkan vec_axpby\n"); exit(1);
        }
        ref_vec_mul_f32(n, ha, hb, hy_ref);
        check_ok(velri_vk_vec_mul_f32(vk, n, da, db, dy, 1u, nullptr), "vk_mul");
        check_ok(velri_vk_buf_download(vk, hy_vk, dy, n * sizeof(float)), "down");
        if (!(max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, 1u, n, n) <= 1e-5)) {
            fprintf(stderr, "FAIL Vulkan vec_mul\n"); exit(1);
        }
        velri_vk_buf_free(vk, da); velri_vk_buf_free(vk, db); velri_vk_buf_free(vk, dy);
        xfree(ha); xfree(hb); xfree(hy_ref); xfree(hy_vk);
    }
    printf("  [PASS] Activations (SiLU, GELU_tanh, ReLU2), VecAxpby, VecMul.\n");

    /* FWHT (ALL B in {16,32,64,128,256} x suh in {none, F32, F16}) */
    {
        const uint32_t bs[5] = {16u, 32u, 64u, 128u, 256u};
        uint32_t rows = 5u, cols = 512u, xs = 528u, ys = 520u;
        float *hx = xalloc(rows * xs * sizeof(float)), *hsf = xalloc(cols * sizeof(float));
        velri_fp16 *hsh = xalloc(cols * sizeof(velri_fp16));
        float *hy_ref = xalloc(rows * ys * sizeof(float)), *hy_vk = xalloc(rows * ys * sizeof(float));
        uint64_t rng = 0x78780001ULL;
        fill_dtype(VELRI_DTYPE_F32, hx, rows * xs, &rng, 2.0f);
        fill_dtype(VELRI_DTYPE_F32, hsf, cols, &rng, 1.0f);
        fill_dtype(VELRI_DTYPE_F16, hsh, cols, &rng, 1.0f);

        velri_vk_buf *dx = nullptr, *dsf = nullptr, *dsh = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, rows * xs * sizeof(float), &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, cols * sizeof(float), &dsf), "alloc");
        check_ok(velri_vk_buf_alloc(vk, cols * sizeof(velri_fp16), &dsh), "alloc");
        check_ok(velri_vk_buf_alloc(vk, rows * ys * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, rows * xs * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, dsf, hsf, cols * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, dsh, hsh, cols * sizeof(velri_fp16)), "up");

        for (int bi = 0; bi < 5; ++bi) {
            for (int sm = 0; sm < 3; ++sm) {
                velri_dtype sdt = (sm == 1) ? VELRI_DTYPE_F32 : ((sm == 2) ? VELRI_DTYPE_F16 : VELRI_DTYPE_F32);
                const void *hsuh = (sm == 1) ? (const void *)hsf : ((sm == 2) ? (const void *)hsh : nullptr);
                const velri_vk_buf *dsuh = (sm == 1) ? dsf : ((sm == 2) ? dsh : nullptr);
                ref_fwht_f32(rows, cols, bs[bi], hx, xs, sdt, hsuh, hy_ref, ys);
                check_ok(velri_vk_fwht_f32(vk, rows, cols, bs[bi], dx, xs, sdt, dsuh, dy, ys, 1u, nullptr),
                         "vk_fwht");
                check_ok(velri_vk_buf_download(vk, hy_vk, dy, rows * ys * sizeof(float)), "down");
                double err = max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy_vk, rows, cols, ys);
                if (!(err <= 5e-4)) {
                    fprintf(stderr, "FAIL Vulkan fwht B=%u sm=%d err=%.3e\n", bs[bi], sm, err);
                    exit(1);
                }
            }
        }
        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dsf);
        velri_vk_buf_free(vk, dsh); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hsf); xfree(hsh); xfree(hy_ref); xfree(hy_vk);
    }
    printf("  [PASS] FWHT F32 (B in {16,32,64,128,256}; suh none/F32/F16).\n");

    /* Row Quantization F32 -> I8 (aligned 4096 and odd 131) */
    {
        const uint32_t ks[2] = {4096u, 131u};
        const uint32_t sts[2] = {4096u, 137u};
        for (int ci = 0; ci < 2; ++ci) {
            uint32_t rows = 7u, k = ks[ci], st = sts[ci];
            float *hx = xalloc(rows * st * sizeof(float));
            int8_t *hq_ref = xalloc(rows * st), *hq_vk = xalloc(rows * st);
            float *hs_ref = xalloc(rows * sizeof(float)), *hs_vk = xalloc(rows * sizeof(float));
            uint64_t rng = 0x89890001ULL + (uint64_t)ci;
            fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 5.0f);
            ref_quantize_row_f32_i8(rows, k, hx, st, hq_ref, st, hs_ref);
            velri_vk_buf *dx = nullptr, *dq = nullptr, *ds = nullptr;
            check_ok(velri_vk_buf_alloc(vk, rows * st * sizeof(float), &dx), "alloc");
            check_ok(velri_vk_buf_alloc(vk, rows * st, &dq), "alloc");
            check_ok(velri_vk_buf_alloc(vk, rows * sizeof(float), &ds), "alloc");
            check_ok(velri_vk_buf_upload(vk, dx, hx, rows * st * sizeof(float)), "up");
            check_ok(velri_vk_quantize_row_f32_i8(vk, rows, k, dx, st, dq, st, ds, 1u, nullptr),
                     "vk_quant");
            check_ok(velri_vk_buf_download(vk, hq_vk, dq, rows * st), "down");
            check_ok(velri_vk_buf_download(vk, hs_vk, ds, rows * sizeof(float)), "down");
            double es = max_abs_diff_2d(VELRI_DTYPE_F32, hs_ref, hs_vk, 1u, rows, rows);
            int max_qdiff = 0;
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < k; ++c) {
                    int d = abs((int)hq_ref[r * st + c] - (int)hq_vk[r * st + c]);
                    if (d > max_qdiff) max_qdiff = d;
                }
            }
            if (!(es <= 1e-6 && max_qdiff <= 1)) {
                fprintf(stderr, "FAIL Vulkan quantize ci=%d es=%.3e max_qdiff=%d\n", ci, es, max_qdiff);
                exit(1);
            }
            velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dq); velri_vk_buf_free(vk, ds);
            xfree(hx); xfree(hq_ref); xfree(hq_vk); xfree(hs_ref); xfree(hs_vk);
        }
    }
    printf("  [PASS] QuantizeRow F32->I8 (aligned 4096 & odd 131).\n");
}

/* ============================================================================
 * Part 2 & 3: Measured Hardware Ceilings & Kernel Benchmarks
 * ============================================================================ */
static void bench_vk_gemv(velri_vk_ctx *vk, const char *label, uint32_t m, uint32_t k,
                          uint32_t iters, double bw_read_gbps, double bw_spec_gbps) {
    printf("\n=== %s (rows=1, M=%u, K=%u) vs Read-Only BW Ceiling ===\n", label, m, k);
    printf("%-11s | %8s | %10s | %12s | %12s\n",
           "Mode", "Time ms", "Eff GB/s", "% Meas Read", "% Spec 896");
    printf("------------+----------+------------+--------------+-------------\n");

    struct {
        const char *name;
        velri_linear_desc desc;
    } modes[] = {
        {"F32", velri_desc_f32()},
        {"F16->F32", velri_desc_f16(VELRI_DTYPE_F32)},
        {"BF16->F32", velri_desc_bf16(VELRI_DTYPE_F32)},
        {"W8A8->F32", velri_desc_w8a8_f32()},
    };

    const velri_vk_info *info = velri_vk_get_info(vk);
    uint64_t rng = 0xD0000001ULL;

    for (size_t mi = 0; mi < sizeof(modes) / sizeof(modes[0]); ++mi) {
        velri_linear_desc d = modes[mi].desc;
        if (d.x_dtype == VELRI_DTYPE_F16 && !info->has_f16_i8) continue;
        if (d.x_dtype == VELRI_DTYPE_I8 && !(info->has_f16_i8 && info->has_int_dot)) continue;

        size_t xb = velri_dtype_bytes(d.x_dtype);
        size_t wb = velri_dtype_bytes(d.w_dtype);
        size_t yb = velri_dtype_bytes(d.y_dtype);

        void *hx = xalloc((size_t)k * xb);
        void *hw = xalloc((size_t)m * k * wb);
        float *hxs = d.has_x_scale ? xalloc(sizeof(float)) : nullptr;
        float *hws = d.has_w_scale ? xalloc((size_t)m * sizeof(float)) : nullptr;
        fill_dtype(d.x_dtype, hx, k, &rng, 0.5f);
        /* Fast tiled pattern fill (keeps GPU out of idle P-state; full PRNG already verified above). */
        size_t seed_elems = (size_t)k * 4u;
        if (seed_elems > (size_t)m * k) seed_elems = (size_t)m * k;
        fill_dtype(d.w_dtype, hw, seed_elems, &rng, 0.5f);
        size_t filled = seed_elems * wb;
        size_t total_w = (size_t)m * k * wb;
        while (filled < total_w) {
            size_t chunk = (total_w - filled < filled) ? (total_w - filled) : filled;
            memcpy((uint8_t *)hw + filled, hw, chunk);
            filled += chunk;
        }
        if (hxs) hxs[0] = 0.02f;
        if (hws) fill_dtype(VELRI_DTYPE_F32, hws, m, &rng, 0.02f);

        velri_vk_buf *dx = nullptr, *dw = nullptr, *dy = nullptr, *dxs = nullptr, *dws = nullptr;
        check_ok(velri_vk_buf_alloc(vk, (size_t)k * xb, &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)m * k * wb, &dw), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)m * yb, &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, (size_t)k * xb), "up");
        check_ok(velri_vk_buf_upload(vk, dw, hw, (size_t)m * k * wb), "up");
        if (hxs) {
            check_ok(velri_vk_buf_alloc(vk, sizeof(float), &dxs), "alloc");
            check_ok(velri_vk_buf_upload(vk, dxs, hxs, sizeof(float)), "up");
        }
        if (hws) {
            check_ok(velri_vk_buf_alloc(vk, (size_t)m * sizeof(float), &dws), "alloc");
            check_ok(velri_vk_buf_upload(vk, dws, hws, (size_t)m * sizeof(float)), "up");
        }

        double avg_ms = 0.0;
        check_ok(velri_vk_linear(vk, &d, 1u, m, k, dx, k, dw, k, nullptr, dxs, dws, dy, m, iters, &avg_ms),
                 "gemv");
        double bytes = (double)((size_t)m * k * wb + (size_t)k * xb + (size_t)m * yb);
        double gbps = bytes / (avg_ms * 1e-3) / 1e9;
        printf("%-11s | %8.4f | %6.1f GB/s | %11.1f%% | %11.1f%%\n",
               modes[mi].name, avg_ms, gbps,
               100.0 * gbps / bw_read_gbps, 100.0 * gbps / bw_spec_gbps);

        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dw); velri_vk_buf_free(vk, dy);
        velri_vk_buf_free(vk, dxs); velri_vk_buf_free(vk, dws);
        xfree(hx); xfree(hw); xfree(hxs); xfree(hws);
    }
}

static void bench_vk_gemm(velri_vk_ctx *vk, uint32_t rows, uint32_t m, uint32_t k,
                          double peak_f32, double peak_f16, double peak_bf16, double peak_i8) {
    printf("\n=== Vulkan GEMM (rows=%u, M=%u, K=%u) vs Compute / Tensor Core Ceiling ===\n",
           rows, m, k);
    printf("%-16s | %8s | %13s | %13s | %12s\n",
           "Mode", "Time ms", "Throughput", "Meas Ceiling", "% of Ceiling");
    printf("-----------------+----------+---------------+---------------+-------------\n");

    struct {
        const char *name;
        velri_linear_desc desc;
        double ceiling;
        const char *unit;
    } modes[] = {
        {"F32 (tiled)",      velri_desc_f32(),                  peak_f32,  "TFLOPS"},
        {"F16 (CoopMat)",    velri_desc_f16(VELRI_DTYPE_F32),   peak_f16,  "TFLOPS"},
        {"BF16 (CoopMat)",   velri_desc_bf16(VELRI_DTYPE_F32),  peak_bf16, "TFLOPS"},
        {"W8A8 (CoopMat)",   velri_desc_w8a8_f32(),             peak_i8,   "TOPS"},
    };

    const velri_vk_info *info = velri_vk_get_info(vk);
    uint64_t rng = 0xE0000001ULL;
    double ops = 2.0 * (double)rows * (double)m * (double)k;

    for (size_t mi = 0; mi < sizeof(modes) / sizeof(modes[0]); ++mi) {
        velri_linear_desc d = modes[mi].desc;
        if (d.x_dtype == VELRI_DTYPE_F16 && !info->has_coopmat_f16) continue;
        if (d.x_dtype == VELRI_DTYPE_BF16 && !info->has_coopmat_bf16) continue;
        if (d.x_dtype == VELRI_DTYPE_I8 && !info->has_coopmat_i8) continue;

        size_t xb = velri_dtype_bytes(d.x_dtype);
        size_t wb = velri_dtype_bytes(d.w_dtype);
        size_t yb = velri_dtype_bytes(d.y_dtype);

        void *hx = xalloc((size_t)rows * k * xb);
        void *hw = xalloc((size_t)m * k * wb);
        float *hxs = d.has_x_scale ? xalloc((size_t)rows * sizeof(float)) : nullptr;
        float *hws = d.has_w_scale ? xalloc((size_t)m * sizeof(float)) : nullptr;
        fill_dtype(d.x_dtype, hx, (size_t)rows * k, &rng, 0.5f);
        fill_dtype(d.w_dtype, hw, (size_t)m * k, &rng, 0.5f);
        if (hxs) fill_dtype(VELRI_DTYPE_F32, hxs, rows, &rng, 0.02f);
        if (hws) fill_dtype(VELRI_DTYPE_F32, hws, m, &rng, 0.02f);

        velri_vk_buf *dx = nullptr, *dw = nullptr, *dy = nullptr, *dxs = nullptr, *dws = nullptr;
        check_ok(velri_vk_buf_alloc(vk, (size_t)rows * k * xb, &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)m * k * wb, &dw), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)rows * m * yb, &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, (size_t)rows * k * xb), "up");
        check_ok(velri_vk_buf_upload(vk, dw, hw, (size_t)m * k * wb), "up");
        if (hxs) {
            check_ok(velri_vk_buf_alloc(vk, (size_t)rows * sizeof(float), &dxs), "alloc");
            check_ok(velri_vk_buf_upload(vk, dxs, hxs, (size_t)rows * sizeof(float)), "up");
        }
        if (hws) {
            check_ok(velri_vk_buf_alloc(vk, (size_t)m * sizeof(float), &dws), "alloc");
            check_ok(velri_vk_buf_upload(vk, dws, hws, (size_t)m * sizeof(float)), "up");
        }

        uint32_t iters = (rows >= 2048u) ? 15u : 40u;
        double avg_ms = 0.0;
        check_ok(velri_vk_linear(vk, &d, rows, m, k, dx, k, dw, k, nullptr, dxs, dws, dy, m, iters, &avg_ms),
                 "gemm");
        double t_ops = ops / (avg_ms * 1e-3) / 1e12;
        double pct = (modes[mi].ceiling > 0.0) ? (100.0 * t_ops / modes[mi].ceiling) : 0.0;
        printf("%-16s | %8.4f | %6.2f %-6s | %6.2f %-6s | %11.1f%%\n",
               modes[mi].name, avg_ms, t_ops, modes[mi].unit,
               modes[mi].ceiling, modes[mi].unit, pct);

        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dw); velri_vk_buf_free(vk, dy);
        velri_vk_buf_free(vk, dxs); velri_vk_buf_free(vk, dws);
        xfree(hx); xfree(hw); xfree(hxs); xfree(hws);
    }
}

static void bench_vk_non_gemm(velri_vk_ctx *vk, double bw_meas_gbps, double bw_spec_gbps) {
    printf("\n=== Vulkan Non-GEMM LLM Ops vs Bandwidth Ceiling ===\n");
    printf("%-26s | %8s | %10s | %11s | %11s\n",
           "Operation", "Time ms", "Eff GB/s", "% Meas BW", "% Spec BW");
    printf("---------------------------+----------+------------+-------------+------------\n");

    const velri_vk_info *info = velri_vk_get_info(vk);
    uint64_t rng = 0xF0000001ULL;

    /* 1. RMSNorm F32 / BF16 / F16 [8192 x 4096] (134-268 MiB >> 48 MiB L2) */
    const velri_dtype ndts[3] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F16};
    const char *nnames[3] = {"RMSNorm F32  [8192x4096]", "RMSNorm BF16 [8192x4096]", "RMSNorm F16  [8192x4096]"};
    for (int di = 0; di < 3; ++di) {
        if (ndts[di] == VELRI_DTYPE_F16 && !info->has_f16_i8) continue;
        uint32_t rows = 8192u, dim = 4096u;
        size_t eb = velri_dtype_bytes(ndts[di]);
        void *hx = xalloc((size_t)rows * dim * eb), *hw = xalloc((size_t)dim * eb);
        fill_dtype(ndts[di], hx, (size_t)rows * dim, &rng, 2.0f);
        fill_dtype(ndts[di], hw, dim, &rng, 1.0f);
        velri_vk_buf *dx = nullptr, *dw = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, (size_t)rows * dim * eb, &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)dim * eb, &dw), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)rows * dim * eb, &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, (size_t)rows * dim * eb), "up");
        check_ok(velri_vk_buf_upload(vk, dw, hw, (size_t)dim * eb), "up");

        double avg_ms = 0.0;
        check_ok(velri_vk_rmsnorm(vk, ndts[di], rows, dim, 1e-5f, false, dx, dim, dw, dy, dim, 25u, &avg_ms),
                 "rmsnorm");
        double bytes = (double)(2u * (size_t)rows * dim + dim) * (double)eb;
        double gbps = bytes / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               nnames[di], avg_ms, gbps, 100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);

        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dw); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hw);
    }

    /* 2. RoPE F32 / BF16 [8192 x 32 x 128] (134-268 MiB >> 48 MiB L2) */
    const velri_dtype rdts[2] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16};
    const char *rnames[2] = {"RoPE F32  [8192x32x128]", "RoPE BF16 [8192x32x128]"};
    for (int di = 0; di < 2; ++di) {
        uint32_t nt = 8192u, nh = 32u, hd = 128u;
        size_t eb = velri_dtype_bytes(rdts[di]);
        void *hx = xalloc((size_t)nt * nh * hd * eb);
        float *hc = xalloc((size_t)nt * (hd / 2u) * sizeof(float));
        float *hs = xalloc((size_t)nt * (hd / 2u) * sizeof(float));
        fill_dtype(rdts[di], hx, (size_t)nt * nh * hd, &rng, 2.0f);
        for (size_t i = 0; i < (size_t)nt * (hd / 2u); ++i) { hc[i] = 0.8f; hs[i] = 0.6f; }
        velri_vk_buf *dx = nullptr, *dc = nullptr, *ds = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, (size_t)nt * nh * hd * eb, &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)nt * (hd / 2u) * sizeof(float), &dc), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)nt * (hd / 2u) * sizeof(float), &ds), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)nt * nh * hd * eb, &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, (size_t)nt * nh * hd * eb), "up");
        check_ok(velri_vk_buf_upload(vk, dc, hc, (size_t)nt * (hd / 2u) * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, ds, hs, (size_t)nt * (hd / 2u) * sizeof(float)), "up");

        double avg_ms = 0.0;
        check_ok(velri_vk_rope(vk, rdts[di], nt, nh, hd, hd, 0u, dx, nh * hd, hd, dc, ds,
                               dy, nh * hd, hd, VELRI_ROPE_HALF, 25u, &avg_ms),
                 "rope");
        double bytes = 2.0 * (double)((size_t)nt * nh * hd) * (double)eb;
        double gbps = bytes / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               rnames[di], avg_ms, gbps, 100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);

        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dc);
        velri_vk_buf_free(vk, ds); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hc); xfree(hs);
    }

    /* 3. GatedMLP SwiGLU & GeGLU [2048 * 14336] (352 MiB total) */
    {
        uint32_t n = 2048u * 14336u;
        float *hg = xalloc((size_t)n * sizeof(float)), *hu = xalloc((size_t)n * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, hg, n, &rng, 2.0f);
        fill_dtype(VELRI_DTYPE_F32, hu, n, &rng, 2.0f);
        velri_vk_buf *dg = nullptr, *du = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, (size_t)n * sizeof(float), &dg), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)n * sizeof(float), &du), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)n * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dg, hg, (size_t)n * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, du, hu, (size_t)n * sizeof(float)), "up");

        const velri_act_kind acts[2] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH};
        const char *anames[2] = {"SwiGLU F32 [2048x14336]", "GeGLU  F32 [2048x14336]"};
        for (int ai = 0; ai < 2; ++ai) {
            double avg_ms = 0.0;
            check_ok(velri_vk_gated_mlp_f32(vk, acts[ai], n, 0.0f, dg, du, dy, 25u, &avg_ms), "gated");
            double bytes = 3.0 * (double)n * sizeof(float);
            double gbps = bytes / (avg_ms * 1e-3) / 1e9;
            printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
                   anames[ai], avg_ms, gbps, 100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);
        }
        velri_vk_buf_free(vk, dg); velri_vk_buf_free(vk, du); velri_vk_buf_free(vk, dy);
        xfree(hg); xfree(hu);
    }

    /* 4. Softmax F32 [8192 x 4096] (268 MiB >> 48 MiB L2) */
    {
        uint32_t rows = 8192u, cols = 4096u;
        float *hx = xalloc((size_t)rows * cols * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, hx, (size_t)rows * cols, &rng, 5.0f);
        velri_vk_buf *dx = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, (size_t)rows * cols * sizeof(float), &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)rows * cols * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, (size_t)rows * cols * sizeof(float)), "up");

        double avg_ms = 0.0;
        check_ok(velri_vk_softmax_f32(vk, rows, cols, dx, cols, dy, cols, 25u, &avg_ms), "softmax");
        double bytes = 2.0 * (double)((size_t)rows * cols) * sizeof(float);
        double gbps = bytes / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               "Softmax F32 [8192x4096]", avg_ms, gbps,
               100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);

        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dy);
        xfree(hx);
    }

    /* 5. FusedAddRMSNorm F32 [8192 x 4096] (4 arrays = 536 MiB >> 48 MiB L2) */
    {
        uint32_t rows = 8192u, dim = 4096u;
        size_t n = (size_t)rows * dim;
        float *hx = xalloc(n * sizeof(float)), *hw = xalloc(dim * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, hx, n, &rng, 1.0f);
        fill_dtype(VELRI_DTYPE_F32, hw, dim, &rng, 1.0f);
        velri_vk_buf *dr = nullptr, *dx = nullptr, *dw = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dr), "alloc");
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, dim * sizeof(float), &dw), "alloc");
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dr, hx, n * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, dx, hx, n * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, dw, hw, dim * sizeof(float)), "up");

        double avg_ms = 0.0;
        check_ok(velri_vk_fused_add_rmsnorm_f32(vk, rows, dim, 1e-5f, false, dr, dim, dx, dim,
                                                dw, dy, dim, 20u, &avg_ms),
                 "fused_add_rmsnorm");
        double bytes = 4.0 * (double)n * sizeof(float);
        double gbps = bytes / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               "FusedAddRMS [8192x4096]", avg_ms, gbps,
               100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);
        velri_vk_buf_free(vk, dr); velri_vk_buf_free(vk, dx);
        velri_vk_buf_free(vk, dw); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hw);
    }

    /* 6. LayerNorm F32 [8192 x 4096] (268 MiB >> 48 MiB L2) */
    {
        uint32_t rows = 8192u, dim = 4096u;
        size_t n = (size_t)rows * dim;
        float *hx = xalloc(n * sizeof(float)), *hw = xalloc(dim * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, hx, n, &rng, 2.0f);
        fill_dtype(VELRI_DTYPE_F32, hw, dim, &rng, 1.0f);
        velri_vk_buf *dx = nullptr, *dw = nullptr, *db = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, dim * sizeof(float), &dw), "alloc");
        check_ok(velri_vk_buf_alloc(vk, dim * sizeof(float), &db), "alloc");
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, n * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, dw, hw, dim * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, db, hw, dim * sizeof(float)), "up");

        double avg_ms = 0.0;
        check_ok(velri_vk_layernorm_f32(vk, rows, dim, 1e-5f, dx, dim, dw, db, dy, dim, 25u, &avg_ms),
                 "layernorm");
        double bytes = 2.0 * (double)n * sizeof(float);
        double gbps = bytes / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               "LayerNorm F32 [8192x4096]", avg_ms, gbps,
               100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);
        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dw);
        velri_vk_buf_free(vk, db); velri_vk_buf_free(vk, dy);
        xfree(hx); xfree(hw);
    }

    /* 7. FWHT F32 (B=128 warp-only & B=256 SMEM) [8192 x 4096] (268 MiB >> 48 MiB L2) */
    {
        uint32_t rows = 8192u, cols = 4096u;
        size_t n = (size_t)rows * cols;
        float *hx = xalloc(n * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, hx, n, &rng, 1.0f);
        velri_vk_buf *dx = nullptr, *dy = nullptr;
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dx), "alloc");
        check_ok(velri_vk_buf_alloc(vk, n * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_upload(vk, dx, hx, n * sizeof(float)), "up");

        const uint32_t bsz[2] = {128u, 256u};
        const char *bnames[2] = {"FWHT B=128 [8192x4096]", "FWHT B=256 [8192x4096]"};
        for (int i = 0; i < 2; ++i) {
            double avg_ms = 0.0;
            check_ok(velri_vk_fwht_f32(vk, rows, cols, bsz[i], dx, cols, VELRI_DTYPE_F32, nullptr,
                                       dy, cols, 25u, &avg_ms),
                     "fwht");
            double bytes = 2.0 * (double)n * sizeof(float);
            double gbps = bytes / (avg_ms * 1e-3) / 1e9;
            printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
                   bnames[i], avg_ms, gbps,
                   100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);
        }
        velri_vk_buf_free(vk, dx); velri_vk_buf_free(vk, dy);
        xfree(hx);
    }

    /* 8. Activations (SiLU, ReLU2), VecAxpby, VecMul, RowQuantize F32->I8 (cold VRAM) */
    {
        uint32_t rows = 8192u, cols = 4096u;
        uint32_t n = rows * cols;
        float *ha = xalloc((size_t)n * sizeof(float)), *hb = xalloc((size_t)n * sizeof(float));
        fill_dtype(VELRI_DTYPE_F32, ha, n, &rng, 2.0f);
        fill_dtype(VELRI_DTYPE_F32, hb, n, &rng, 2.0f);
        velri_vk_buf *da = nullptr, *db = nullptr, *dy = nullptr, *dq = nullptr, *ds = nullptr;
        check_ok(velri_vk_buf_alloc(vk, (size_t)n * sizeof(float), &da), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)n * sizeof(float), &db), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)n * sizeof(float), &dy), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)n, &dq), "alloc");
        check_ok(velri_vk_buf_alloc(vk, (size_t)rows * sizeof(float), &ds), "alloc");
        check_ok(velri_vk_buf_upload(vk, da, ha, (size_t)n * sizeof(float)), "up");
        check_ok(velri_vk_buf_upload(vk, db, hb, (size_t)n * sizeof(float)), "up");

        double avg_ms = 0.0;
        check_ok(velri_vk_activation_f32(vk, VELRI_ACT_SILU, n, da, dy, 25u, &avg_ms), "silu");
        double gbps = (2.0 * (double)n * sizeof(float)) / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               "SiLU F32 [8192x4096]", avg_ms, gbps,
               100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);

        check_ok(velri_vk_vec_axpby_f32(vk, n, 0.8f, da, 0.2f, db, dy, 25u, &avg_ms), "axpby");
        gbps = (3.0 * (double)n * sizeof(float)) / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               "VecAxpby F32 [8192x4096]", avg_ms, gbps,
               100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);

        check_ok(velri_vk_vec_mul_f32(vk, n, da, db, dy, 25u, &avg_ms), "mul");
        gbps = (3.0 * (double)n * sizeof(float)) / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               "VecMul F32 [8192x4096]", avg_ms, gbps,
               100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);

        check_ok(velri_vk_quantize_row_f32_i8(vk, rows, cols, da, cols, dq, cols, ds, 25u, &avg_ms),
                 "quant");
        gbps = ((double)n * 5.0 + (double)rows * 4.0) / (avg_ms * 1e-3) / 1e9;
        printf("%-26s | %8.4f | %6.1f GB/s | %10.1f%% | %10.1f%%\n",
               "QuantRow F32->I8 [8192x4K]", avg_ms, gbps,
               100.0 * gbps / bw_meas_gbps, 100.0 * gbps / bw_spec_gbps);

        velri_vk_buf_free(vk, da); velri_vk_buf_free(vk, db); velri_vk_buf_free(vk, dy);
        velri_vk_buf_free(vk, dq); velri_vk_buf_free(vk, ds);
        xfree(ha); xfree(hb);
    }
}

int main(int argc, char **argv) {
    bool verify_only = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--verify-only") == 0) verify_only = true;
    }

    velri_vk_ctx *vk = nullptr;
    velri_status st = velri_vk_create(&vk);
    if (st != VELRI_OK) {
        fprintf(stderr, "Failed to initialize Vulkan 1.3+ compute context (status=%d)\n", (int)st);
        return 1;
    }

    const velri_vk_info *info = velri_vk_get_info(vk);
    printf("Vulkan GPU: %s\n", info->device_name);
    printf("  API: %u.%u.%u | subgroup=%u | maxShared=%u B | tsPeriod=%.2f ns\n",
           info->api_version >> 22, (info->api_version >> 12) & 0x3FFu, info->api_version & 0xFFFu,
           info->subgroup_size, info->max_shared_mem, info->timestamp_period_ns);
    printf("  Features: BDA=%d F16/I8=%d IntDot=%d BF16=%d CoopMat=%d (F16=%d BF16=%d I8=%d)\n",
           (int)info->has_bda, (int)info->has_f16_i8, (int)info->has_int_dot, (int)info->has_bf16,
           (int)info->has_coopmat, (int)info->has_coopmat_f16, (int)info->has_coopmat_bf16,
           (int)info->has_coopmat_i8);

    run_vk_verification(vk);

    if (!verify_only) {
        /* Theoretical GDDR7 spec for RTX 5070 Ti: 256-bit @ 28 Gbps = 896.0 GB/s. */
        const double bw_spec_gbps = 896.0;
        double bw_read = 0.0, bw_meas = 0.0, peak_f32 = 0.0, peak_f16 = 0.0, peak_bf16 = 0.0, peak_i8 = 0.0;

        printf("\n=== Hardware Ceiling Calibrations (measured directly via GPU timestamps) ===\n");
        /* Run short compute calibration first so RTX 5070 Ti ramps out of P8 idle into P0 boost. */
        check_ok(velri_vk_measure_fma_tflops(vk, 2048u, 512u, &peak_f32), "ceil fma");
        check_ok(velri_vk_measure_bw_read_gbps(vk, 256u * 1024u * 1024u, 20u, &bw_read), "ceil bw read");
        check_ok(velri_vk_measure_bw_gbps(vk, 256u * 1024u * 1024u, 20u, &bw_meas), "ceil bw");
        printf("  Sustained VRAM read (256 MiB): %6.1f GB/s (%.1f%% of %.0f GB/s GDDR7 spec)\n",
               bw_read, 100.0 * bw_read / bw_spec_gbps, bw_spec_gbps);
        printf("  Sustained VRAM copy (256 MiB): %6.1f GB/s (%.1f%% of %.0f GB/s GDDR7 spec)\n",
               bw_meas, 100.0 * bw_meas / bw_spec_gbps, bw_spec_gbps);
        printf("  Sustained FP32 FMA peak      : %6.2f TFLOPS\n", peak_f32);

        if (info->has_coopmat_f16) {
            check_ok(velri_vk_measure_coopmat_peak(vk, VELRI_DTYPE_F16, 1024u, 512u, &peak_f16), "ceil f16");
            printf("  Sustained CoopMat FP16->F32  : %6.2f TFLOPS\n", peak_f16);
        }
        if (info->has_coopmat_bf16) {
            check_ok(velri_vk_measure_coopmat_peak(vk, VELRI_DTYPE_BF16, 1024u, 512u, &peak_bf16), "ceil bf16");
            printf("  Sustained CoopMat BF16->F32  : %6.2f TFLOPS\n", peak_bf16);
        }
        if (info->has_coopmat_i8) {
            check_ok(velri_vk_measure_coopmat_peak(vk, VELRI_DTYPE_I8, 1024u, 512u, &peak_i8), "ceil i8");
            printf("  Sustained CoopMat INT8->I32  : %6.2f TOPS\n", peak_i8);
        }

        bench_vk_gemv(vk, "Streaming GEMV (cold VRAM >> 48 MiB L2)", 65536u, 4096u, 15u, bw_read, bw_spec_gbps);
        bench_vk_gemv(vk, "LLM FFN GEMV (14336x4096, W8A8 partially L2)", 14336u, 4096u, 40u, bw_read, bw_spec_gbps);
        bench_vk_gemv(vk, "Cached GEMV (L2-hot 4096x4096)", 4096u, 4096u, 60u, bw_read, bw_spec_gbps);
        bench_vk_gemm(vk, 256u, 4096u, 4096u, peak_f32, peak_f16, peak_bf16, peak_i8);
        bench_vk_gemm(vk, 4096u, 4096u, 4096u, peak_f32, peak_f16, peak_bf16, peak_i8);
        bench_vk_non_gemm(vk, bw_meas, bw_spec_gbps);
    }

    velri_vk_destroy(vk);
    return 0;
}
