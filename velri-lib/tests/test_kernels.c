#include "reference.h"
#include "velri_lib/velri_lib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Unified Kernel Correctness Suite.
 *
 * Verifies ALL 12 operations and supported dtypes (including FP64 linear), aligned and odd
 * shapes, non-16B-aligned subviews, Split-K & CoopMat tiles, and numeric edge cases
 * (-inf softmax rows, -inf/-1e38 SiLU/GeLU, subnormal-adjacent quantize_row) through the
 * public Graph -> Plan -> Slot API across every available CPU ISA variant and Vulkan GPU.
 */

#define CHECK(cond, msg)                                                                          \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));                     \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

#define CHECK_OK(expr, msg)                                                                       \
    do {                                                                                          \
        velri_status _st = (expr);                                                                \
        if (_st != VELRI_OK) {                                                                    \
            fprintf(stderr, "FAIL [%s:%d]: %s (status=%d)\n", __FILE__, __LINE__, (msg),          \
                    (int)_st);                                                                    \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1U, sz ? sz : 1U);
    CHECK(p != nullptr, "calloc");
    return p;
}

static velri_storage_id add_store(velri_graph *g, velri_storage_kind kind, velri_dtype dt,
                                  size_t elems, velri_device_id dev) {
    velri_storage_desc d = {
        .kind = kind,
        .dtype = dt,
        .byte_size = elems * velri_dtype_bytes(dt),
        .align = 64U,
        .device_id = dev,
    };
    velri_storage_id id = 0U;
    CHECK_OK(velri_graph_add_storage(g, &d, &id), "add_storage");
    return id;
}

static velri_view_id add_view_1d(velri_graph *g, velri_storage_id sid, velri_dtype dt, size_t n,
                                 size_t byte_off) {
    velri_tensor_view v;
    size_t dims[1] = {n};
    CHECK_OK(velri_tensor_view_contiguous(sid, dt, 1U, dims, byte_off, &v), "view1d");
    velri_view_id id = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &id), "add_view1d");
    return id;
}

static velri_view_id add_view_2d_strided(velri_graph *g, velri_storage_id sid, velri_dtype dt,
                                         size_t rows, size_t cols, size_t stride) {
    velri_tensor_view v = {
        .storage_id = sid,
        .dtype = dt,
        .rank = 2U,
        .dims = {rows, cols},
        .strides = {stride, 1U},
        .byte_offset = 0U,
    };
    velri_view_id id = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &id), "add_view2d");
    return id;
}

static velri_view_id add_view_3d_strided(velri_graph *g, velri_storage_id sid, velri_dtype dt,
                                         size_t d0, size_t d1, size_t d2, size_t s0, size_t s1) {
    velri_tensor_view v = {
        .storage_id = sid,
        .dtype = dt,
        .rank = 3U,
        .dims = {d0, d1, d2},
        .strides = {s0, s1, 1U},
        .byte_offset = 0U,
    };
    velri_view_id id = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &id), "add_view3d");
    return id;
}

static void verify_linear_one(velri_context *ctx, velri_device_id dev, const char *dev_tag,
                              const char *label, velri_linear_desc desc, size_t rows, size_t m,
                              size_t k, size_t x_pad, size_t w_pad, size_t y_pad, bool with_bias,
                              double tol) {
    size_t xs = k + x_pad, ws = k + w_pad, ys = m + y_pad;
    size_t xb = velri_dtype_bytes(desc.x_dtype);
    size_t wb = velri_dtype_bytes(desc.w_dtype);
    size_t yb = velri_dtype_bytes(desc.y_dtype);

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "graph");
    velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, desc.x_dtype, rows * xs, dev);
    velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, desc.w_dtype, m * ws, dev);
    velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, desc.y_dtype, rows * ys, dev);
    velri_storage_id sb = with_bias ? add_store(g, VELRI_STORAGE_WEIGHT, desc.y_dtype, m, dev) : 0U;
    velri_storage_id sxs = desc.has_x_scale ? add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows, dev) : 0U;
    velri_storage_id sws = desc.has_w_scale ? add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m, dev) : 0U;

    velri_op_desc op = {
        .kind = VELRI_OP_LINEAR,
        .num_inputs = 5U,
        .num_outputs = 1U,
        .inputs = {
            add_view_2d_strided(g, sx, desc.x_dtype, rows, k, xs),
            add_view_2d_strided(g, sw, desc.w_dtype, m, k, ws),
            with_bias ? add_view_1d(g, sb, desc.y_dtype, m, 0U) : VELRI_INVALID_ID,
            desc.has_x_scale ? add_view_1d(g, sxs, VELRI_DTYPE_F32, rows, 0U) : VELRI_INVALID_ID,
            desc.has_w_scale ? add_view_1d(g, sws, VELRI_DTYPE_F32, m, 0U) : VELRI_INVALID_ID,
        },
        .outputs = {add_view_2d_strided(g, sy, desc.y_dtype, rows, m, ys)},
        .attr.linear.desc = desc,
    };
    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add linear");

    velri_plan *plan = nullptr;
    velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
    if (pst == VELRI_ERR_UNSUPPORTED) {
        velri_graph_destroy(g);
        return; /* Dtype combo intentionally unsupported on this device (e.g. FP64 on Vulkan) */
    }
    CHECK_OK(pst, label);

    void *hx = xcalloc(rows * xs, xb);
    void *hw = xcalloc(m * ws, wb);
    void *hy_ref = xcalloc(rows * ys, yb);
    void *hy = xcalloc(rows * ys, yb);
    void *hbias = with_bias ? xcalloc(m, yb) : nullptr;
    float *hxs = desc.has_x_scale ? (float *)xcalloc(rows, sizeof(float)) : nullptr;
    float *hws = desc.has_w_scale ? (float *)xcalloc(m, sizeof(float)) : nullptr;

    uint64_t rng = 0x55AA0001ULL ^ ((uint64_t)rows << 16) ^ ((uint64_t)m << 4) ^ (uint64_t)with_bias;
    velri_test_fill_dtype(desc.x_dtype, hx, rows * xs, &rng, 0.5f);
    velri_test_fill_dtype(desc.w_dtype, hw, m * ws, &rng, 0.5f);
    if (hbias) velri_test_fill_dtype(desc.y_dtype, hbias, m, &rng, 0.5f);
    if (hxs) velri_test_fill_dtype(VELRI_DTYPE_F32, hxs, rows, &rng, 0.05f);
    if (hws) velri_test_fill_dtype(VELRI_DTYPE_F32, hws, m, &rng, 0.05f);

    ref_linear_run(&desc, rows, m, k, hx, xs, hxs, hw, ws, hws, hbias, hy_ref, ys);

    CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, m * ws * wb), "bind w");
    if (with_bias) CHECK_OK(velri_plan_bind_weight_host(plan, sb, hbias, m * yb), "bind bias");
    if (desc.has_w_scale) {
        CHECK_OK(velri_plan_bind_weight_host(plan, sws, hws, m * sizeof(float)), "bind ws");
    }

    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot), "slot");
    CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * xs * xb), "bind x");
    CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * ys * yb), "bind y");
    if (desc.has_x_scale) {
        CHECK_OK(velri_slot_bind_host(slot, sxs, hxs, rows * sizeof(float)), "bind xs");
    }
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), label);
    CHECK(velri_slot_last_run_allocs(slot) == 0U, "0 hot-path allocs on linear");

    double err = velri_test_max_abs_diff_2d(desc.y_dtype, hy_ref, hy, rows, m, ys);
    if (!(err <= tol)) {
        fprintf(stderr, "FAIL [%s] %s (rows=%zu m=%zu k=%zu bias=%d): err=%.3e > %.3e\n",
                dev_tag, label, rows, m, k, (int)with_bias, err, tol);
        exit(1);
    }

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(hx); free(hw); free(hy_ref); free(hy); free(hbias); free(hxs); free(hws);
}

static void verify_attention_case(velri_context *ctx, velri_device_id dev, const char *dev_tag,
                                  const char *label, uint32_t nh, uint32_t nkvh, uint32_t hd,
                                  size_t max_t, size_t max_ctx, size_t prefill_t,
                                  size_t num_decode_steps, uint32_t sliding_window, bool causal,
                                  bool same_kv_cache, uint64_t seed) {
    const size_t q_dim = (size_t)nh * hd;
    const size_t kv_dim = (size_t)nkvh * hd;
    const size_t qkv_stride = same_kv_cache ? (q_dim + kv_dim) : (q_dim + 2U * kv_dim);

    float *hqkv = (float *)xcalloc(max_t * qkv_stride, sizeof(float));
    float *ho = (float *)xcalloc(max_t * q_dim, sizeof(float));
    float *ho_ref = (float *)xcalloc(max_t * q_dim, sizeof(float));
    float *hkc_ref = (float *)xcalloc((size_t)nkvh * max_ctx * hd, sizeof(float));
    float *hvc_ref =
        same_kv_cache ? hkc_ref : (float *)xcalloc((size_t)nkvh * max_ctx * hd, sizeof(float));
    uint64_t rng = seed;

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "attn g");
    velri_storage_id sqkv =
        add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, max_t * qkv_stride, dev);
    velri_storage_id so = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, max_t * q_dim, dev);
    velri_storage_id skc =
        add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, (size_t)nkvh * max_ctx * hd, dev);
    velri_storage_id svc =
        same_kv_cache
            ? skc
            : add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, (size_t)nkvh * max_ctx * hd, dev);

    velri_tensor_view base_qkv, qv, kv, kc_v;
    size_t base_dims[2] = {max_t, qkv_stride};
    size_t q_dims[2] = {max_t, q_dim}, q_str[2] = {qkv_stride, 1U};
    size_t kv_dims[2] = {max_t, kv_dim}, kv_str[2] = {qkv_stride, 1U};
    size_t cache_dims[3] = {nkvh, max_ctx, hd};

    CHECK_OK(velri_tensor_view_contiguous(sqkv, VELRI_DTYPE_F32, 2U, base_dims, 0U, &base_qkv),
             "attn base");
    CHECK_OK(velri_tensor_view_subview(&base_qkv, 2U, q_dims, q_str, 0U, &qv), "attn qv");
    CHECK_OK(velri_tensor_view_subview(&base_qkv, 2U, kv_dims, kv_str, q_dim * sizeof(float), &kv),
             "attn kv");
    CHECK_OK(velri_tensor_view_contiguous(skc, VELRI_DTYPE_F32, 3U, cache_dims, 0U, &kc_v),
             "attn kcv");

    velri_view_id id_q = 0U, id_k = 0U, id_v = 0U, id_kc = 0U, id_vc = 0U;
    CHECK_OK(velri_graph_add_view(g, &qv, &id_q), "add qv");
    CHECK_OK(velri_graph_add_view(g, &kv, &id_k), "add kv");
    CHECK_OK(velri_graph_add_view(g, &kc_v, &id_kc), "add kcv");
    if (same_kv_cache) {
        id_v = id_k;
        id_vc = id_kc;
    } else {
        velri_tensor_view vv, vc_v;
        CHECK_OK(velri_tensor_view_subview(&base_qkv, 2U, kv_dims, kv_str,
                                           (q_dim + kv_dim) * sizeof(float), &vv),
                 "attn vv");
        CHECK_OK(velri_tensor_view_contiguous(svc, VELRI_DTYPE_F32, 3U, cache_dims, 0U, &vc_v),
                 "attn vcv");
        CHECK_OK(velri_graph_add_view(g, &vv, &id_v), "add vv");
        CHECK_OK(velri_graph_add_view(g, &vc_v, &id_vc), "add vcv");
    }

    velri_op_desc op = {
        .kind = VELRI_OP_ATTENTION,
        .num_inputs = 3U,
        .num_outputs = 3U,
        .inputs = {id_q, id_k, id_v},
        .outputs = {add_view_2d_strided(g, so, VELRI_DTYPE_F32, max_t, q_dim, q_dim), id_kc, id_vc},
        .attr.attention = {
            .n_heads = nh,
            .n_kv_heads = nkvh,
            .head_dim = hd,
            .scale = 0.0f,
            .sliding_window = sliding_window,
            .causal = causal,
        },
    };
    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), label);

    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), label);
    CHECK_OK(velri_slot_create(plan, &slot), "attn slot");
    CHECK_OK(velri_slot_bind_host(slot, sqkv, hqkv, max_t * qkv_stride * sizeof(float)), "qkv");
    CHECK_OK(velri_slot_bind_host(slot, so, ho, max_t * q_dim * sizeof(float)), "o");

    const float *k_in_ptr = hqkv + q_dim;
    const float *v_in_ptr = same_kv_cache ? k_in_ptr : (hqkv + q_dim + kv_dim);

    /* Prefill `prefill_t` tokens at kv_pos = 0 */
    velri_test_fill_dtype(VELRI_DTYPE_F32, hqkv, prefill_t * qkv_stride, &rng, 1.0f);
    velri_step_params sp_pre = {
        .has_seq_len = true,
        .seq_len_override = prefill_t,
        .has_kv_len = true,
        .kv_len_override = 0U,
    };
    CHECK_OK(velri_slot_run(slot, &sp_pre, nullptr), label);
    CHECK(velri_slot_last_run_allocs(slot) == 0U, "attn prefill 0 allocs");
    velri_ref_attention_f32(prefill_t, nh, nkvh, hd, 0U, max_ctx, 0.0f, sliding_window, causal,
                            hqkv, qkv_stride, k_in_ptr, qkv_stride, v_in_ptr, qkv_stride, hkc_ref,
                            hvc_ref, ho_ref, q_dim);
    double err = velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, prefill_t, q_dim, q_dim);
    if (!(err <= 5e-5)) {
        fprintf(stderr, "FAIL [%s] %s prefill(t=%zu): err=%.3e > 5e-5\n", dev_tag, label, prefill_t,
                err);
        exit(1);
    }

    /* Subsequent 1-token decode steps reading resident device KV state (0 B shadow RAM) */
    for (size_t ds = 0U; ds < num_decode_steps; ++ds) {
        size_t kv_pos = prefill_t + ds;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hqkv, qkv_stride, &rng, 1.0f);
        velri_step_params sp_dec = {
            .has_seq_len = true,
            .seq_len_override = 1U,
            .has_kv_len = true,
            .kv_len_override = kv_pos,
        };
        CHECK_OK(velri_slot_run(slot, &sp_dec, nullptr), label);
        CHECK(velri_slot_last_run_allocs(slot) == 0U, "attn decode 0 allocs");
        velri_ref_attention_f32(1U, nh, nkvh, hd, kv_pos, max_ctx, 0.0f, sliding_window, causal,
                                hqkv, qkv_stride, k_in_ptr, qkv_stride, v_in_ptr, qkv_stride,
                                hkc_ref, hvc_ref, ho_ref, q_dim);
        err = velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, 1U, q_dim, q_dim);
        if (!(err <= 5e-5)) {
            fprintf(stderr, "FAIL [%s] %s decode(kv_pos=%zu): err=%.3e > 5e-5\n", dev_tag, label,
                    kv_pos, err);
            exit(1);
        }
    }

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(hqkv);
    free(ho);
    free(ho_ref);
    free(hkc_ref);
    if (!same_kv_cache) {
        free(hvc_ref);
    }
}

static void verify_all_ops_on_device(velri_context *ctx, velri_device_id dev, const char *tag,
                                     bool is_gpu) {
    /* 1. Linear: all dtypes (incl. FP64 on CPU) + GEMV + GEMM + Split-K + CoopMat 64/128 */
    struct {
        size_t rows, m, k, xp, wp, yp;
    } cpu_shapes[] = {
        {1U, 67U, 131U, 5U, 7U, 3U},
        {2U, 67U, 131U, 0U, 0U, 0U},
        {7U, 67U, 131U, 5U, 7U, 3U},
        {13U, 67U, 131U, 5U, 7U, 3U},
    };
    struct {
        size_t rows, m, k, xp, wp, yp;
    } gpu_shapes[] = {
        {1U, 64U, 128U, 0U, 0U, 0U},
        {1U, 67U, 131U, 5U, 7U, 3U},
        {2U, 67U, 131U, 0U, 0U, 0U},
        {128U, 128U, 64U, 0U, 0U, 0U},
        {140U, 145U, 131U, 5U, 7U, 3U},
        {64U, 512U, 1024U, 0U, 0U, 0U},
        {67U, 259U, 1031U, 3U, 5U, 2U},
        {1280U, 1920U, 96U, 0U, 0U, 0U},
    };
    size_t nshapes = is_gpu ? (sizeof(gpu_shapes) / sizeof(gpu_shapes[0]))
                            : (sizeof(cpu_shapes) / sizeof(cpu_shapes[0]));

    for (size_t si = 0U; si < nshapes; ++si) {
        size_t rows = is_gpu ? gpu_shapes[si].rows : cpu_shapes[si].rows;
        size_t m = is_gpu ? gpu_shapes[si].m : cpu_shapes[si].m;
        size_t k = is_gpu ? gpu_shapes[si].k : cpu_shapes[si].k;
        size_t xp = is_gpu ? gpu_shapes[si].xp : cpu_shapes[si].xp;
        size_t wp = is_gpu ? gpu_shapes[si].wp : cpu_shapes[si].wp;
        size_t yp = is_gpu ? gpu_shapes[si].yp : cpu_shapes[si].yp;
        for (int b = 0; b <= 1; ++b) {
            verify_linear_one(ctx, dev, tag, "linear_f32", velri_desc_f32(), rows, m, k, xp, wp, yp,
                              b != 0, 2e-4);
            verify_linear_one(ctx, dev, tag, "linear_f64", velri_desc_f64(), rows, m, k, xp, wp, yp,
                              b != 0, 1e-11);
            verify_linear_one(ctx, dev, tag, "linear_f16", velri_desc_f16(VELRI_DTYPE_F32), rows, m,
                              k, xp, wp, yp, b != 0, 2e-3);
            verify_linear_one(ctx, dev, tag, "linear_bf16", velri_desc_bf16(VELRI_DTYPE_F32), rows,
                              m, k, xp, wp, yp, b != 0, 2e-3);
            verify_linear_one(ctx, dev, tag, "linear_f32x_bf16w", velri_desc_f32x_bf16w(), rows, m,
                              k, xp, wp, yp, b != 0, 2e-4);
            verify_linear_one(ctx, dev, tag, "linear_w8a8", velri_desc_w8a8_f32(), rows, m, k, xp,
                              wp, yp, b != 0, 2e-4);
        }
    }

    /* 2. RMSNorm (F32, BF16, F16; aligned 4096 & odd 131; unit_offset 0/1; weight on/off) */
    const velri_dtype ndts[3] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F16};
    const double ntols[3] = {3e-5, 2e-2, 4e-3};
    const size_t ndims[2] = {131U, 4096U};
    const size_t nstrs[2] = {137U, 4096U};
    for (int di = 0; di < 3; ++di) {
        velri_dtype dt = ndts[di];
        size_t eb = velri_dtype_bytes(dt);
        for (int ci = 0; ci < 2; ++ci) {
            size_t rows = 5U, dim = ndims[ci], st = nstrs[ci];
            void *hx = xcalloc(rows * st, eb), *hw = xcalloc(dim, eb);
            void *hy_ref = xcalloc(rows * st, eb), *hy = xcalloc(rows * st, eb);
            uint64_t rng = 0x11110001ULL + (uint64_t)di * 17U + (uint64_t)ci;
            velri_test_fill_dtype(dt, hx, rows * st, &rng, 2.0f);
            velri_test_fill_dtype(dt, hw, dim, &rng, 1.5f);
            for (int uo = 0; uo <= 1; ++uo) {
                for (int has_w = 0; has_w <= 1; ++has_w) {
                    velri_graph *g = nullptr;
                    CHECK_OK(velri_graph_create(&g), "rmsnorm g");
                    velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, dt, rows * st, dev);
                    velri_storage_id sw = has_w ? add_store(g, VELRI_STORAGE_WEIGHT, dt, dim, dev) : 0U;
                    velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, dt, rows * st, dev);
                    velri_op_desc op = {
                        .kind = VELRI_OP_RMSNORM,
                        .num_inputs = has_w ? 2U : 1U,
                        .num_outputs = 1U,
                        .inputs = {add_view_2d_strided(g, sx, dt, rows, dim, st),
                                   has_w ? add_view_1d(g, sw, dt, dim, 0U) : VELRI_INVALID_ID},
                        .outputs = {add_view_2d_strided(g, sy, dt, rows, dim, st)},
                        .attr.rmsnorm = {.eps = 1e-5f, .unit_offset = (uo != 0)},
                    };
                    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add rmsnorm");
                    velri_plan *plan = nullptr;
                    velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
                    if (pst == VELRI_ERR_UNSUPPORTED) {
                        velri_graph_destroy(g);
                        continue;
                    }
                    CHECK_OK(pst, "prep rmsnorm");
                    if (has_w) CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, dim * eb), "w");
                    velri_slot *slot = nullptr;
                    CHECK_OK(velri_slot_create(plan, &slot), "slot");
                    CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * eb), "x");
                    CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * st * eb), "y");
                    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run rmsnorm");
                    ref_rmsnorm(dt, dt, rows, dim, 1e-5f, uo != 0, hx, st, has_w ? hw : nullptr,
                                hy_ref, st);
                    double err = velri_test_max_abs_diff_2d(dt, hy_ref, hy, rows, dim, st);
                    CHECK(err <= ntols[di], "rmsnorm accuracy");
                    velri_slot_destroy(slot);
                    velri_plan_destroy(plan);
                    velri_graph_destroy(g);
                }
            }
            free(hx); free(hw); free(hy_ref); free(hy);
        }
    }

    /* 3. FusedAddRMSNorm (verifies both updated STATE residual + output y) */
    for (int ci = 0; ci < 2; ++ci) {
        size_t rows = 5U, dim = ndims[ci], st = nstrs[ci];
        float *hr = (float *)xcalloc(rows * st, sizeof(float));
        float *hr_ref = (float *)xcalloc(rows * st, sizeof(float));
        float *hx = (float *)xcalloc(rows * st, sizeof(float));
        float *hw = (float *)xcalloc(dim, sizeof(float));
        float *hy_ref = (float *)xcalloc(rows * st, sizeof(float));
        float *hy = (float *)xcalloc(rows * st, sizeof(float));
        uint64_t rng = 0xFA001122ULL + (uint64_t)ci;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hr, rows * st, &rng, 2.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 2.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hw, dim, &rng, 1.5f);
        memcpy(hr_ref, hr, rows * st * sizeof(float));
        ref_fused_add_rmsnorm_f32(rows, dim, 1e-5f, ci == 1, hr_ref, st, hx, st, hw, hy_ref, st);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "far g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, dim, dev);
        velri_storage_id sr = add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_FUSED_ADD_RMSNORM,
            .num_inputs = 2U, .num_outputs = 2U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, dim, st),
                       add_view_1d(g, sw, VELRI_DTYPE_F32, dim, 0U)},
            .outputs = {add_view_2d_strided(g, sr, VELRI_DTYPE_F32, rows, dim, st),
                        add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, dim, st)},
            .attr.rmsnorm = {.eps = 1e-5f, .unit_offset = (ci == 1)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add far");
        velri_plan *plan = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep far");
        CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, dim * sizeof(float)), "bind w");
        velri_slot *slot = nullptr;
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sr, hr, rows * st * sizeof(float)), "r");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * st * sizeof(float)), "y");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run far");
        CHECK_OK(velri_slot_sync_state_to_host(slot, sr), "sync state r");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hr_ref, hr, rows, dim, st) <= 1e-5, "far res");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, dim, st) <= 1e-4, "far y");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hr); free(hr_ref); free(hx); free(hw); free(hy_ref); free(hy);
    }

    /* 4. LayerNorm (aligned 4096 & odd 131; weight + bias) */
    for (int ci = 0; ci < 2; ++ci) {
        size_t rows = 5U, dim = ndims[ci], st = nstrs[ci];
        float *hx = (float *)xcalloc(rows * st, sizeof(float));
        float *hw = (float *)xcalloc(dim, sizeof(float));
        float *hb = (float *)xcalloc(dim, sizeof(float));
        float *hy_ref = (float *)xcalloc(rows * st, sizeof(float));
        float *hy = (float *)xcalloc(rows * st, sizeof(float));
        uint64_t rng = 0x1A7E0001ULL + (uint64_t)ci;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 3.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hw, dim, &rng, 1.5f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hb, dim, &rng, 0.5f);
        ref_layernorm_f32(rows, dim, 1e-5f, hx, st, hw, hb, hy_ref, st);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "ln g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, dim, dev);
        velri_storage_id sb = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, dim, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_LAYERNORM,
            .num_inputs = 3U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, dim, st),
                       add_view_1d(g, sw, VELRI_DTYPE_F32, dim, 0U),
                       add_view_1d(g, sb, VELRI_DTYPE_F32, dim, 0U)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, dim, st)},
            .attr.layernorm = {.eps = 1e-5f},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add ln");
        velri_plan *plan = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep ln");
        CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, dim * sizeof(float)), "w");
        CHECK_OK(velri_plan_bind_weight_host(plan, sb, hb, dim * sizeof(float)), "b");
        velri_slot *slot = nullptr;
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * st * sizeof(float)), "y");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run ln");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, dim, st) <= 1e-4, "ln");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hw); free(hb); free(hy_ref); free(hy);
    }

    /* 5. RoPE (F32 & BF16, HALF & INTERLEAVED, partial & aligned fast-path + pos_offset_override) */
    const velri_dtype rdts[2] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16};
    const double rtols[2] = {2e-5, 2e-2};
    for (int di = 0; di < 2; ++di) {
        velri_dtype dt = rdts[di];
        size_t eb = velri_dtype_bytes(dt);
        for (int fast = 0; fast <= 1; ++fast) {
            size_t nt = fast ? 9U : 7U, nh = fast ? 8U : 5U;
            size_t hd = fast ? 128U : 72U, rd = fast ? 128U : 48U;
            size_t po_plan = 2U, po_run = 5U;
            size_t hsx = fast ? hd : (hd + 4U), tsx = fast ? (nh * hd) : (nh * hsx + 3U);
            size_t hsy = fast ? hd : (hd + 2U), tsy = fast ? (nh * hd) : (nh * hsy + 5U);
            size_t tbl_len = (po_run + nt) * (rd / 2U);

            void *hx = xcalloc(nt * tsx, eb), *hy_ref = xcalloc(nt * tsy, eb), *hy = xcalloc(nt * tsy, eb);
            float *hcos = (float *)xcalloc(tbl_len, sizeof(float));
            float *hsin = (float *)xcalloc(tbl_len, sizeof(float));
            uint64_t rng = 0x22220001ULL + (uint64_t)di * 31U + (uint64_t)fast;
            velri_test_fill_dtype(dt, hx, nt * tsx, &rng, 2.0f);
            for (size_t i = 0; i < tbl_len; ++i) {
                float ang = velri_test_rand_f32(&rng, -3.14f, 3.14f);
                hcos[i] = cosf(ang);
                hsin[i] = sinf(ang);
            }

            for (int m = 0; m <= (fast ? 0 : 1); ++m) {
                velri_rope_mode mode = m ? VELRI_ROPE_INTERLEAVED : VELRI_ROPE_HALF;
                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "rope g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, dt, nt * tsx, dev);
                velri_storage_id sc = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, tbl_len, dev);
                velri_storage_id ss = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, tbl_len, dev);
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, dt, nt * tsy, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_ROPE,
                    .num_inputs = 3U, .num_outputs = 1U,
                    .inputs = {add_view_3d_strided(g, sx, dt, nt, nh, hd, tsx, hsx),
                               add_view_1d(g, sc, VELRI_DTYPE_F32, tbl_len, 0U),
                               add_view_1d(g, ss, VELRI_DTYPE_F32, tbl_len, 0U)},
                    .outputs = {add_view_3d_strided(g, sy, dt, nt, nh, hd, tsy, hsy)},
                    .attr.rope = {.rotary_dim = rd, .pos_offset = po_plan, .mode = mode},
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add rope");
                velri_plan *plan = nullptr;
                velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
                if (pst == VELRI_ERR_UNSUPPORTED) {
                    velri_graph_destroy(g);
                    continue;
                }
                CHECK_OK(pst, "prep rope");
                CHECK_OK(velri_plan_bind_weight_host(plan, sc, hcos, tbl_len * sizeof(float)), "cos");
                CHECK_OK(velri_plan_bind_weight_host(plan, ss, hsin, tbl_len * sizeof(float)), "sin");
                velri_slot *slot = nullptr;
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, nt * tsx * eb), "x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy, nt * tsy * eb), "y");
                velri_step_params sp = {.pos_offset_override = po_run, .has_pos_offset = true};
                CHECK_OK(velri_slot_run(slot, &sp, nullptr), "run rope");
                ref_rope(dt, nt, nh, hd, rd, po_run, hx, tsx, hsx, hcos, hsin, hy_ref, tsy, hsy, mode);
                for (size_t t = 0; t < nt; ++t)
                    for (size_t h = 0; h < nh; ++h)
                        for (size_t d = 0; d < hd; ++d) {
                            size_t off = t * tsy + h * hsy + d;
                            double diff = fabs(velri_test_load_f64(dt, hy_ref, off) -
                                               velri_test_load_f64(dt, hy, off));
                            CHECK(diff <= rtols[di], "rope accuracy");
                        }
                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
            }
            free(hx); free(hy_ref); free(hy); free(hcos); free(hsin);
        }
    }

    /* 6-9. Elementwise ops (Activations + -inf edge [A17], GatedMLP, VecAxpby, VecMul)
     *      with BOTH 16B-aligned offset 0 AND 4B-aligned non-16B subview offset 4 [A5]! */
    for (int sub = 0; sub <= 1; ++sub) {
        size_t n = 1031U;
        size_t elem_off = sub ? 1U : 0U; /* byte_offset = 4 when sub == 1 (tests A5!) */
        size_t alloc_n = n + elem_off;
        float *ha = (float *)xcalloc(alloc_n, sizeof(float));
        float *hb = (float *)xcalloc(alloc_n, sizeof(float));
        float *hy_ref = (float *)xcalloc(n, sizeof(float));
        float *hy = (float *)xcalloc(alloc_n, sizeof(float));
        uint64_t rng = 0x67670001ULL + (uint64_t)sub;
        velri_test_fill_dtype(VELRI_DTYPE_F32, ha + elem_off, n, &rng, 4.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hb + elem_off, n, &rng, 2.0f);
        /* A17 extreme inputs: -inf, -1e38f, -100.0f, 0.0f */
        ha[elem_off + 0U] = -INFINITY;
        ha[elem_off + 1U] = -1e38f;
        ha[elem_off + 2U] = -100.0f;
        ha[elem_off + 3U] = 0.0f;

        const velri_act_kind ak[3] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH, VELRI_ACT_RELU2};
        for (int ki = 0; ki < 3; ++ki) {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "act g");
            velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_op_desc op = {
                .kind = VELRI_OP_ACTIVATION, .num_inputs = 1U, .num_outputs = 1U,
                .inputs = {add_view_1d(g, sx, VELRI_DTYPE_F32, n, elem_off * sizeof(float))},
                .outputs = {add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float))},
                .attr.activation = {.kind = ak[ki]},
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add act");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep act");
            CHECK_OK(velri_slot_create(plan, &slot), "slot");
            CHECK_OK(velri_slot_bind_host(slot, sx, ha, alloc_n * sizeof(float)), "x");
            CHECK_OK(velri_slot_bind_host(slot, sy, hy, alloc_n * sizeof(float)), "y");
            CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run act");
            ref_activation_f32(ak[ki], n, ha + elem_off, hy_ref);
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy + elem_off, 1U, n, n) <= 5e-4,
                  "activation + A5/A17");
            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
        }

        /* Restore finite inputs for GatedMLP, VecAxpby, VecMul */
        velri_test_fill_dtype(VELRI_DTYPE_F32, ha + elem_off, 4U, &rng, 4.0f);
        for (int gi = 0; gi < 4; ++gi) {
            velri_act_kind gact = (gi < 2) ? VELRI_ACT_SILU : VELRI_ACT_GELU_TANH;
            float clamp = (gi & 1) ? 1.5f : 0.0f;
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "gated g");
            velri_storage_id s0 = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_storage_id s1 = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_op_desc op = {
                .kind = VELRI_OP_GATED_MLP, .num_inputs = 2U, .num_outputs = 1U,
                .inputs = {add_view_1d(g, s0, VELRI_DTYPE_F32, n, elem_off * sizeof(float)),
                           add_view_1d(g, s1, VELRI_DTYPE_F32, n, elem_off * sizeof(float))},
                .outputs = {add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float))},
                .attr.gated_mlp = {.act = gact, .clamp_limit = clamp},
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add gated");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep gated");
            CHECK_OK(velri_slot_create(plan, &slot), "slot");
            CHECK_OK(velri_slot_bind_host(slot, s0, ha, alloc_n * sizeof(float)), "g");
            CHECK_OK(velri_slot_bind_host(slot, s1, hb, alloc_n * sizeof(float)), "u");
            CHECK_OK(velri_slot_bind_host(slot, sy, hy, alloc_n * sizeof(float)), "y");
            CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run gated");
            ref_gated_mlp_f32(gact, n, clamp, ha + elem_off, hb + elem_off, hy_ref);
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy + elem_off, 1U, n, n) <= 5e-4,
                  "gated_mlp");
            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
        }

        /* VecAxpby (beta != 0 and beta == 0) + VecMul */
        for (int mode = 0; mode < 3; ++mode) {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "vec g");
            velri_storage_id s0 = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_storage_id s1 = (mode != 1) ? add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev) : 0U;
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_op_desc op = {0};
            if (mode == 0) {
                op.kind = VELRI_OP_VEC_AXPBY;
                op.num_inputs = 2U; op.num_outputs = 1U;
                op.inputs[0] = add_view_1d(g, s0, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.inputs[1] = add_view_1d(g, s1, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.outputs[0] = add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.attr.axpby.alpha = 1.35f; op.attr.axpby.beta = -0.75f;
                ref_vec_axpby_f32(n, 1.35f, ha + elem_off, -0.75f, hb + elem_off, hy_ref);
            } else if (mode == 1) {
                op.kind = VELRI_OP_VEC_AXPBY;
                op.num_inputs = 1U; op.num_outputs = 1U;
                op.inputs[0] = add_view_1d(g, s0, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.outputs[0] = add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.attr.axpby.alpha = 2.25f; op.attr.axpby.beta = 0.0f;
                ref_vec_axpby_f32(n, 2.25f, ha + elem_off, 0.0f, nullptr, hy_ref);
            } else {
                op.kind = VELRI_OP_VEC_MUL;
                op.num_inputs = 2U; op.num_outputs = 1U;
                op.inputs[0] = add_view_1d(g, s0, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.inputs[1] = add_view_1d(g, s1, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.outputs[0] = add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                ref_vec_mul_f32(n, ha + elem_off, hb + elem_off, hy_ref);
            }
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add vec");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep vec");
            CHECK_OK(velri_slot_create(plan, &slot), "slot");
            CHECK_OK(velri_slot_bind_host(slot, s0, ha, alloc_n * sizeof(float)), "a");
            if (mode != 1) CHECK_OK(velri_slot_bind_host(slot, s1, hb, alloc_n * sizeof(float)), "b");
            CHECK_OK(velri_slot_bind_host(slot, sy, hy, alloc_n * sizeof(float)), "y");
            CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run vec");
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy + elem_off, 1U, n, n) <= 1e-5,
                  "vec_axpby/mul");
            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
        }
        free(ha); free(hb); free(hy_ref); free(hy);
    }

    /* 10. Softmax (wide range, odd & aligned 4096, plus all-(-inf) row [A17]) */
    for (int ci = 0; ci < 2; ++ci) {
        size_t rows = 5U, cols = ndims[ci], st = nstrs[ci];
        float *hx = (float *)xcalloc(rows * st, sizeof(float));
        float *hy_ref = (float *)xcalloc(rows * st, sizeof(float));
        float *hy = (float *)xcalloc(rows * st, sizeof(float));
        uint64_t rng = 0x44440001ULL + (uint64_t)ci;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 20.0f);
        /* Row 2: all -INFINITY -> exact 0.0f; Row 3: causal mask half -INFINITY [A17] */
        for (size_t c = 0; c < cols; ++c) hx[2U * st + c] = -INFINITY;
        for (size_t c = cols / 2U; c < cols; ++c) hx[3U * st + c] = -INFINITY;
        ref_softmax_f32(rows, cols, hx, st, hy_ref, st);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "sm g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_SOFTMAX, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, cols, st)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, cols, st)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add sm");
        velri_plan *plan = nullptr;
        velri_slot *slot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep sm");
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * st * sizeof(float)), "y");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run sm");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, cols, st) <= 1e-4,
              "softmax + A17 -inf row");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hy_ref); free(hy);
    }

    /* 11. FWHT (B in {16,32,64,128,256} x suh in {none, F32, F16}) */
    {
        const size_t bs[5] = {16U, 32U, 64U, 128U, 256U};
        size_t rows = 5U, cols = 512U, xs = 521U, ys = 523U;
        float *hx = (float *)xcalloc(rows * xs, sizeof(float));
        float *hsf = (float *)xcalloc(cols, sizeof(float));
        velri_fp16 *hsh = (velri_fp16 *)xcalloc(cols, sizeof(velri_fp16));
        float *hy_ref = (float *)xcalloc(rows * ys, sizeof(float));
        float *hy = (float *)xcalloc(rows * ys, sizeof(float));
        uint64_t rng = 0xF1170001ULL;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * xs, &rng, 2.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hsf, cols, &rng, 1.5f);
        velri_test_fill_dtype(VELRI_DTYPE_F16, hsh, cols, &rng, 1.5f);
        for (int bi = 0; bi < 5; ++bi) {
            for (int sm = 0; sm < 3; ++sm) {
                velri_dtype sdt = (sm == 2) ? VELRI_DTYPE_F16 : VELRI_DTYPE_F32;
                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "fwht g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * xs, dev);
                velri_storage_id ss = (sm > 0) ? add_store(g, VELRI_STORAGE_WEIGHT, sdt, cols, dev) : 0U;
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * ys, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_FWHT, .num_inputs = (sm > 0) ? 2U : 1U, .num_outputs = 1U,
                    .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, cols, xs),
                               (sm > 0) ? add_view_1d(g, ss, sdt, cols, 0U) : VELRI_INVALID_ID},
                    .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, cols, ys)},
                    .attr.fwht = {.block_size = bs[bi], .suh_dtype = sdt},
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add fwht");
                velri_plan *plan = nullptr;
                velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
                if (pst == VELRI_ERR_UNSUPPORTED) {
                    velri_graph_destroy(g);
                    continue;
                }
                CHECK_OK(pst, "prep fwht");
                if (sm == 1) CHECK_OK(velri_plan_bind_weight_host(plan, ss, hsf, cols * sizeof(float)), "suh32");
                if (sm == 2) CHECK_OK(velri_plan_bind_weight_host(plan, ss, hsh, cols * sizeof(velri_fp16)), "suh16");
                velri_slot *slot = nullptr;
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * xs * sizeof(float)), "x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * ys * sizeof(float)), "y");
                CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run fwht");
                ref_fwht_f32(rows, cols, bs[bi], hx, xs, sdt,
                             (sm == 0) ? nullptr : ((sm == 1) ? (const void *)hsf : (const void *)hsh),
                             hy_ref, ys);
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, cols, ys) <= 5e-4,
                      "fwht accuracy");
                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
            }
        }
        free(hx); free(hsf); free(hsh); free(hy_ref); free(hy);
    }

    /* 12. QuantizeRow F32 -> I8 (odd & 4096, all-zero row + A12 tiny 1e-37f row!) */
    for (int ci = 0; ci < 2; ++ci) {
        size_t rows = 6U, k = ndims[ci], st = nstrs[ci];
        float *hx = (float *)xcalloc(rows * st, sizeof(float));
        int8_t *hq_ref = (int8_t *)xcalloc(rows * st, 1U);
        int8_t *hq = (int8_t *)xcalloc(rows * st, 1U);
        float *hs_ref = (float *)xcalloc(rows, sizeof(float));
        float *hs = (float *)xcalloc(rows, sizeof(float));
        uint64_t rng = 0x89890001ULL + (uint64_t)ci;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 5.0f);
        memset(hx + 2U * st, 0, k * sizeof(float)); /* all-zero row */
        for (size_t c = 0; c < k; ++c) hx[3U * st + c] = 1e-37f; /* A12 tiny subnormal-adj row */
        ref_quantize_row_f32_i8(rows, k, hx, st, hq_ref, st, hs_ref);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "qrow g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sq = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_I8, rows * st, dev);
        velri_storage_id ss = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_QUANTIZE_ROW, .num_inputs = 1U, .num_outputs = 2U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, st)},
            .outputs = {add_view_2d_strided(g, sq, VELRI_DTYPE_I8, rows, k, st),
                        add_view_1d(g, ss, VELRI_DTYPE_F32, rows, 0U)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add qrow");
        velri_plan *plan = nullptr;
        velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
        if (pst == VELRI_ERR_UNSUPPORTED) {
            velri_graph_destroy(g);
            free(hx); free(hq_ref); free(hq); free(hs_ref); free(hs);
            continue;
        }
        CHECK_OK(pst, "prep qrow");
        velri_slot *slot = nullptr;
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sq, hq, rows * st), "q");
        CHECK_OK(velri_slot_bind_host(slot, ss, hs, rows * sizeof(float)), "s");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run qrow");

        for (size_t r = 0; r < rows; ++r) {
            CHECK(isfinite(hs[r]) && hs[r] > 0.0f, "qrow finite positive scale");
            if (r != 3U) {
                CHECK(fabsf(hs_ref[r] - hs[r]) <= 1e-6f, "qrow scale");
                for (size_t c = 0; c < k; ++c) {
                    int diff = abs((int)hq_ref[r * st + c] - (int)hq[r * st + c]);
                    CHECK(diff <= 1, "qrow int8 match");
                }
            }
        }
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hq_ref); free(hq); free(hs_ref); free(hs);
    }

    /* 13. Embedding (token lookup w/ OOB & negative IDs + positional lookup w/ dynamic override) */
    {
        size_t vocab = 37U, dim = 69U, max_t = 8U, t_stride = 73U, o_stride = 71U;
        float *htbl = (float *)xcalloc(vocab * t_stride, sizeof(float));
        int32_t *hids = (int32_t *)xcalloc(max_t, sizeof(int32_t));
        float *hy = (float *)xcalloc(max_t * o_stride, sizeof(float));
        float *hy_ref = (float *)xcalloc(max_t * o_stride, sizeof(float));
        uint64_t rng = 0xE11B0001ULL;
        velri_test_fill_dtype(VELRI_DTYPE_F32, htbl, vocab * t_stride, &rng, 2.0f);
        hids[0] = 0; hids[1] = 36; hids[2] = -5; hids[3] = 999; hids[4] = 17;
        hids[5] = 3; hids[6] = 12; hids[7] = 1;

        /* 13a. Token embedding with dynamic seq_len_override = 5 */
        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "emb g");
        velri_storage_id st = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, vocab * t_stride, dev);
        velri_storage_id si = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_I32, max_t, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, max_t * o_stride, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_EMBEDDING, .num_inputs = 2U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, st, VELRI_DTYPE_F32, vocab, dim, t_stride),
                       add_view_1d(g, si, VELRI_DTYPE_I32, max_t, 0U)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, max_t, dim, o_stride)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add emb");
        velri_plan *plan = nullptr;
        velri_slot *slot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep emb");
        CHECK_OK(velri_plan_bind_weight_host(plan, st, htbl, vocab * t_stride * sizeof(float)), "wt");
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, si, hids, max_t * sizeof(int32_t)), "ids");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, max_t * o_stride * sizeof(float)), "y");
        velri_step_params sp = {.has_seq_len = true, .seq_len_override = 5U};
        CHECK_OK(velri_slot_run(slot, &sp, nullptr), "run emb");
        CHECK(velri_slot_last_run_allocs(slot) == 0U, "emb 0 allocs");
        velri_ref_embedding_f32(vocab, dim, 5U, 0U, htbl, t_stride, hids, hy_ref, o_stride);
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, 5U, dim, o_stride) <= 1e-6,
              "token embedding accuracy");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);

        /* 13b. Positional embedding with dynamic pos_offset_override = 11, seq_len_override = 4 */
        CHECK_OK(velri_graph_create(&g), "pos emb g");
        st = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, vocab * t_stride, dev);
        sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, max_t * o_stride, dev);
        velri_op_desc pop = {
            .kind = VELRI_OP_EMBEDDING, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, st, VELRI_DTYPE_F32, vocab, dim, t_stride)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, max_t, dim, o_stride)},
            .attr.embedding = {.pos_offset = 0U},
        };
        CHECK_OK(velri_graph_add_op(g, &pop, dev, nullptr), "add pos emb");
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep pos emb");
        CHECK_OK(velri_plan_bind_weight_host(plan, st, htbl, vocab * t_stride * sizeof(float)), "wt");
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, max_t * o_stride * sizeof(float)), "y");
        velri_step_params psp = {
            .has_pos_offset = true, .pos_offset_override = 11U,
            .has_seq_len = true, .seq_len_override = 4U,
        };
        CHECK_OK(velri_slot_run(slot, &psp, nullptr), "run pos emb");
        velri_ref_embedding_f32(vocab, dim, 4U, 11U, htbl, t_stride, nullptr, hy_ref, o_stride);
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, 4U, dim, o_stride) <= 1e-6,
              "positional embedding accuracy");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(htbl); free(hids); free(hy); free(hy_ref);
    }

    /* 14. CopyLastRow with dynamic seq_len_override */
    {
        size_t max_t = 9U, dim = 131U, xs = 137U;
        float *hx = (float *)xcalloc(max_t * xs, sizeof(float));
        float *hy = (float *)xcalloc(dim, sizeof(float));
        float *hy_ref = (float *)xcalloc(dim, sizeof(float));
        uint64_t rng = 0xC0910001ULL;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, max_t * xs, &rng, 3.0f);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "clr g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, max_t * xs, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, dim, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_COPY_LAST_ROW, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, max_t, dim, xs)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, 1U, dim, dim)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add clr");
        velri_plan *plan = nullptr;
        velri_slot *slot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep clr");
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, max_t * xs * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, dim * sizeof(float)), "y");
        velri_step_params sp = {.has_seq_len = true, .seq_len_override = 6U};
        CHECK_OK(velri_slot_run(slot, &sp, nullptr), "run clr");
        velri_ref_copy_last_row_f32(6U, dim, hx, xs, hy_ref);
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, 1U, dim, dim) <= 1e-6,
              "copy_last_row accuracy");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hy); free(hy_ref);
    }

    /* 15. Attention (GQA + long decode kv=25..26 + sliding window + bidir + same_kv_cache + hd=64/80/128) */
    {
        /* 15a. GQA + strided fused-QKV + long decode (kv_len=25,26 > 8 -> 4 Vulkan P=8 iters & 2 CPU B_k=16 tiles) */
        verify_attention_case(ctx, dev, tag, "attn_gqa_long_decode_hd64", 6U, 2U, 64U, 24U, 40U,
                              24U, 2U, 0U, true, false, 0xA77E0001ULL);

        /* 15b. Sliding window = 5 (kv_len up to 11 > 5 masks earlier keys) + head_dim = 128 */
        verify_attention_case(ctx, dev, tag, "attn_sliding_win5_hd128", 4U, 2U, 128U, 12U, 24U,
                              9U, 2U, 5U, true, false, 0xA77E0002ULL);

        /* 15c. Bidirectional attention (causal = false) + odd head_dim = 80 */
        verify_attention_case(ctx, dev, tag, "attn_bidir_hd80", 6U, 2U, 80U, 20U, 32U, 18U, 2U,
                              0U, false, false, 0xA77E0003ULL);

        /* 15d. Shared K/V cache (same_kv_cache = true, K_in == V_in, K_cache == V_cache) + head_dim = 128 */
        verify_attention_case(ctx, dev, tag, "attn_same_kv_hd128", 4U, 2U, 128U, 18U, 32U, 17U, 2U,
                              0U, true, true, 0xA77E0004ULL);
    }

    /* 16. Negative contract & dynamic_seq gating tests (Findings #3 & #10) */
    {
        const size_t rows = 4U, m = 8U, k = 16U;

        /* 16a. VELRI_OP_LINEAR rejects unexpected inputs[3]/inputs[4] when has_*_scale == false */
        {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "lin neg g");
            velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
            velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * k, dev);
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * m, dev);
            velri_storage_id sxs = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows, dev);
            velri_storage_id sws = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m, dev);

            velri_view_id vx = add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k);
            velri_view_id vw = add_view_2d_strided(g, sw, VELRI_DTYPE_F32, m, k, k);
            velri_view_id vy = add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, m, m);
            velri_view_id vxs = add_view_1d(g, sxs, VELRI_DTYPE_F32, rows, 0U);
            velri_view_id vws = add_view_1d(g, sws, VELRI_DTYPE_F32, m, 0U);

            velri_op_desc bad_xsc = {
                .kind = VELRI_OP_LINEAR,
                .num_inputs = 4U,
                .num_outputs = 1U,
                .inputs = {vx, vw, VELRI_INVALID_ID, vxs},
                .outputs = {vy},
                .attr.linear.desc = velri_desc_f32(),
            };
            CHECK(velri_graph_add_op(g, &bad_xsc, dev, nullptr) == VELRI_ERR_INVALID,
                  "linear must reject unexpected inputs[3] when has_x_scale == false");

            velri_op_desc bad_wsc = {
                .kind = VELRI_OP_LINEAR,
                .num_inputs = 5U,
                .num_outputs = 1U,
                .inputs = {vx, vw, VELRI_INVALID_ID, VELRI_INVALID_ID, vws},
                .outputs = {vy},
                .attr.linear.desc = velri_desc_f32(),
            };
            CHECK(velri_graph_add_op(g, &bad_wsc, dev, nullptr) == VELRI_ERR_INVALID,
                  "linear must reject unexpected inputs[4] when has_w_scale == false");
            velri_graph_destroy(g);
        }

        /* 16b. dynamic_seq=true rejects seq_len_override > static_rows; dynamic_seq=false is not truncated */
        {
            float *hx = (float *)xcalloc(rows * k, sizeof(float));
            float *hw = (float *)xcalloc(m * k, sizeof(float));
            float *hy = (float *)xcalloc(rows * m, sizeof(float));
            float *hy_ref = (float *)xcalloc(rows * m, sizeof(float));
            for (size_t r = 0U; r < rows; ++r) {
                for (size_t c = 0U; c < k; ++c) {
                    hx[r * k + c] = (float)(r + 1U) * 0.5f + (float)c * 0.01f;
                }
            }
            for (size_t r = 0U; r < m; ++r) {
                for (size_t c = 0U; c < k; ++c) {
                    hw[r * k + c] = 0.25f + (float)r * 0.05f;
                }
            }
            velri_linear_desc ldesc = velri_desc_f32();
            ref_linear_run(&ldesc, rows, m, k, hx, k, nullptr, hw, k, nullptr, nullptr, hy_ref, m);

            /* Part 1: dynamic_seq = true LINEAR strictly rejects seq_len_override > static_rows */
            {
                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "dyn lin g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
                velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * k, dev);
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * m, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_LINEAR,
                    .num_inputs = 2U,
                    .num_outputs = 1U,
                    .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k),
                               add_view_2d_strided(g, sw, VELRI_DTYPE_F32, m, k, k)},
                    .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, m, m)},
                    .dynamic_seq = true,
                    .attr.linear.desc = ldesc,
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add dyn lin");
                velri_plan *plan = nullptr;
                velri_slot *slot = nullptr;
                CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep dyn lin");
                CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, m * k * sizeof(float)), "bind w");
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * k * sizeof(float)), "bind x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * m * sizeof(float)), "bind y");

                velri_step_params sp_oob = {.has_seq_len = true, .seq_len_override = rows + 1U};
                CHECK(velri_slot_run(slot, &sp_oob, nullptr) == VELRI_ERR_INVALID,
                      "dynamic_seq=true must reject seq_len_override > static_rows");
                CHECK(!velri_slot_needs_reset(slot), "pre-accept rejection must not dirty slot");

                velri_step_params sp_ok = {.has_seq_len = true, .seq_len_override = 2U};
                CHECK_OK(velri_slot_run(slot, &sp_ok, nullptr), "dyn lin valid override");
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, 2U, m, m) <= 2e-4,
                      "dyn lin active rows accuracy");

                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
            }

            /* Part 2: dynamic_seq = false LINEAR is NOT truncated when has_seq_len = true */
            {
                memset(hy, 0, rows * m * sizeof(float));
                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "static lin g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
                velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * k, dev);
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * m, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_LINEAR,
                    .num_inputs = 2U,
                    .num_outputs = 1U,
                    .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k),
                               add_view_2d_strided(g, sw, VELRI_DTYPE_F32, m, k, k)},
                    .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, m, m)},
                    .dynamic_seq = false,
                    .attr.linear.desc = ldesc,
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add static lin");
                velri_plan *plan = nullptr;
                velri_slot *slot = nullptr;
                CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep static lin");
                CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, m * k * sizeof(float)), "bind w");
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * k * sizeof(float)), "bind x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * m * sizeof(float)), "bind y");

                velri_step_params sp_small = {.has_seq_len = true, .seq_len_override = 1U};
                CHECK_OK(velri_slot_run(slot, &sp_small, nullptr), "static lin run");
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, m, m) <= 2e-4,
                      "dynamic_seq=false linear must compute all static rows");

                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
            }

            /* Part 3: dynamic_seq = false 2D elementwise op is NOT truncated when has_seq_len = true */
            {
                float *hy_ew = (float *)xcalloc(rows * k, sizeof(float));
                float *hy_ew_ref = (float *)xcalloc(rows * k, sizeof(float));
                ref_vec_axpby_f32(rows * k, 2.0f, hx, 0.0f, hx, hy_ew_ref);

                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "static ew g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * k, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_VEC_AXPBY,
                    .num_inputs = 1U,
                    .num_outputs = 1U,
                    .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k)},
                    .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, k, k)},
                    .dynamic_seq = false,
                    .attr.axpby = {.alpha = 2.0f, .beta = 0.0f},
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add static ew");
                velri_plan *plan = nullptr;
                velri_slot *slot = nullptr;
                CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep static ew");
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * k * sizeof(float)), "bind x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy_ew, rows * k * sizeof(float)), "bind y");

                velri_step_params sp_small = {.has_seq_len = true, .seq_len_override = 1U};
                CHECK_OK(velri_slot_run(slot, &sp_small, nullptr), "static ew run");
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ew_ref, hy_ew, rows, k, k) <= 1e-6,
                      "dynamic_seq=false elemwise must compute all static rows");

                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
                free(hy_ew);
                free(hy_ew_ref);
            }

            free(hx);
            free(hw);
            free(hy);
            free(hy_ref);
        }
    }

    /* 17. Latent Attention (exact h_cache + weight absorption: GPT-2 d_lat=768 + GQA/sliding/bidir) */
    {
        struct {
            const char *name;
            uint32_t nh, nkvh, hd, d_lat, max_t, max_ctx, prefill_t, dec_steps, win;
            bool causal, has_bv;
            uint64_t seed;
        } cases[3] = {
            {"lat_gpt2_d768_bv", 12U, 12U, 64U, 768U, 24U, 40U, 24U, 2U, 0U, true, true, 0x1A7E0001ULL},
            {"lat_gqa_win5_nobv", 6U, 2U, 64U, 300U, 12U, 24U, 9U, 2U, 5U, true, false, 0x1A7E0002ULL},
            {"lat_bidir_odd513", 4U, 2U, 48U, 513U, 18U, 32U, 17U, 2U, 0U, false, true, 0x1A7E0003ULL},
        };
        for (int ci = 0; ci < 3; ++ci) {
            uint32_t nh = cases[ci].nh, nkvh = cases[ci].nkvh, hd = cases[ci].hd;
            uint32_t d_lat = cases[ci].d_lat, max_t = cases[ci].max_t, max_ctx = cases[ci].max_ctx;
            size_t q_dim = (size_t)nh * hd, kv_dim = (size_t)nkvh * hd;
            size_t q_str = q_dim + 3U, h_str = (size_t)d_lat + 5U;
            size_t wk_str = (size_t)d_lat + 3U, wv_str = (size_t)d_lat + 7U, o_str = q_dim + 1U;

            float *hq = (float *)xcalloc((size_t)max_t * q_str, sizeof(float));
            float *hh = (float *)xcalloc((size_t)max_t * h_str, sizeof(float));
            float *hwk = (float *)xcalloc(kv_dim * wk_str, sizeof(float));
            float *hwv = (float *)xcalloc(kv_dim * wv_str, sizeof(float));
            float *hbv = cases[ci].has_bv ? (float *)xcalloc(kv_dim, sizeof(float)) : nullptr;
            float *ho = (float *)xcalloc((size_t)max_t * o_str, sizeof(float));
            float *ho_ref = (float *)xcalloc((size_t)max_t * o_str, sizeof(float));
            float *hhc_ref = (float *)xcalloc((size_t)max_ctx * d_lat, sizeof(float));

            uint64_t rng = cases[ci].seed;
            velri_test_fill_dtype(VELRI_DTYPE_F32, hwk, kv_dim * wk_str, &rng, 0.12f);
            velri_test_fill_dtype(VELRI_DTYPE_F32, hwv, kv_dim * wv_str, &rng, 0.12f);
            if (hbv) velri_test_fill_dtype(VELRI_DTYPE_F32, hbv, kv_dim, &rng, 0.25f);

            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "lat g");
            velri_storage_id sq = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * q_str, dev);
            velri_storage_id sh = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * h_str, dev);
            velri_storage_id swk = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, kv_dim * wk_str, dev);
            velri_storage_id swv = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, kv_dim * wv_str, dev);
            velri_storage_id sbv = cases[ci].has_bv ? add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, kv_dim, dev) : 0U;
            velri_storage_id so = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, (size_t)max_t * o_str, dev);
            velri_storage_id shc = add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, (size_t)max_ctx * d_lat, dev);

            velri_op_desc op = {
                .kind = VELRI_OP_LATENT_ATTENTION,
                .num_inputs = cases[ci].has_bv ? 5U : 4U,
                .num_outputs = 2U,
                .inputs = {
                    add_view_2d_strided(g, sq, VELRI_DTYPE_F32, max_t, q_dim, q_str),
                    add_view_2d_strided(g, sh, VELRI_DTYPE_F32, max_t, d_lat, h_str),
                    add_view_2d_strided(g, swk, VELRI_DTYPE_F32, kv_dim, d_lat, wk_str),
                    add_view_2d_strided(g, swv, VELRI_DTYPE_F32, kv_dim, d_lat, wv_str),
                    cases[ci].has_bv ? add_view_1d(g, sbv, VELRI_DTYPE_F32, kv_dim, 0U) : VELRI_INVALID_ID,
                },
                .outputs = {
                    add_view_2d_strided(g, so, VELRI_DTYPE_F32, max_t, q_dim, o_str),
                    add_view_2d_strided(g, shc, VELRI_DTYPE_F32, max_ctx, d_lat, d_lat),
                },
                .dynamic_seq = true,
                .attr.attention = {
                    .n_heads = nh, .n_kv_heads = nkvh, .head_dim = hd,
                    .scale = 0.0f, .sliding_window = cases[ci].win, .causal = cases[ci].causal,
                },
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add lat");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep lat");
            CHECK_OK(velri_plan_bind_weight_host(plan, swk, hwk, kv_dim * wk_str * sizeof(float)), "wk");
            CHECK_OK(velri_plan_bind_weight_host(plan, swv, hwv, kv_dim * wv_str * sizeof(float)), "wv");
            if (hbv) CHECK_OK(velri_plan_bind_weight_host(plan, sbv, hbv, kv_dim * sizeof(float)), "bv");
            CHECK_OK(velri_slot_create(plan, &slot), "slot lat");
            CHECK_OK(velri_slot_bind_host(slot, sq, hq, (size_t)max_t * q_str * sizeof(float)), "q");
            CHECK_OK(velri_slot_bind_host(slot, sh, hh, (size_t)max_t * h_str * sizeof(float)), "h");
            CHECK_OK(velri_slot_bind_host(slot, so, ho, (size_t)max_t * o_str * sizeof(float)), "o");

            /* Prefill step */
            velri_test_fill_dtype(VELRI_DTYPE_F32, hq, (size_t)cases[ci].prefill_t * q_str, &rng, 0.5f);
            velri_test_fill_dtype(VELRI_DTYPE_F32, hh, (size_t)cases[ci].prefill_t * h_str, &rng, 0.5f);
            velri_step_params sp = {.has_seq_len = true, .seq_len_override = cases[ci].prefill_t,
                                    .has_kv_len = true, .kv_len_override = 0U};
            CHECK_OK(velri_slot_run(slot, &sp, nullptr), cases[ci].name);
            CHECK(velri_slot_last_run_allocs(slot) == 0U, "lat prefill 0 allocs");
            velri_ref_latent_attention_f32(cases[ci].prefill_t, nh, nkvh, hd, d_lat, 0U, max_ctx,
                                           0.0f, cases[ci].win, cases[ci].causal,
                                           hq, q_str, hh, h_str, hwk, wk_str, hwv, wv_str, hbv,
                                           hhc_ref, ho_ref, o_str);
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, cases[ci].prefill_t, q_dim, o_str) <= 2e-4,
                  cases[ci].name);

            /* Incremental decode steps */
            for (uint32_t ds = 0U; ds < cases[ci].dec_steps; ++ds) {
                uint32_t kv_pos = cases[ci].prefill_t + ds;
                velri_test_fill_dtype(VELRI_DTYPE_F32, hq, q_str, &rng, 0.5f);
                velri_test_fill_dtype(VELRI_DTYPE_F32, hh, h_str, &rng, 0.5f);
                velri_step_params dsp = {.has_seq_len = true, .seq_len_override = 1U,
                                         .has_kv_len = true, .kv_len_override = kv_pos};
                CHECK_OK(velri_slot_run(slot, &dsp, nullptr), cases[ci].name);
                CHECK(velri_slot_last_run_allocs(slot) == 0U, "lat decode 0 allocs");
                velri_ref_latent_attention_f32(1U, nh, nkvh, hd, d_lat, kv_pos, max_ctx,
                                               0.0f, cases[ci].win, cases[ci].causal,
                                               hq, q_str, hh, h_str, hwk, wk_str, hwv, wv_str, hbv,
                                               hhc_ref, ho_ref, o_str);
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, 1U, q_dim, o_str) <= 2e-4,
                      cases[ci].name);
            }

            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
            free(hq); free(hh); free(hwk); free(hwv); free(hbv); free(ho); free(ho_ref); free(hhc_ref);
        }
    }

    printf("  [PASS] %-14s — all 16 ops + edge cases verified vs FP64 reference.\n", tag);
}

int main(void) {
    printf("=== Velri-lib Unified Kernel Correctness Suite ===\n");

    const velri_cpu_variant vars[3] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    const char *vnames[3] = {"cpu:base", "cpu:avx2", "cpu:avx512"};
    size_t nvars = 1U;
#if VELRI_X86_VARIANTS
    nvars = 3U;
#endif

    for (size_t vi = 0U; vi < nvars; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) continue;
        velri_context_config cfg = {
            .cpu_workers = 4U,
            .cpu_variant = vars[vi],
            .enable_vulkan = false,
        };
        velri_context *ctx = nullptr;
        CHECK_OK(velri_context_create(&cfg, &ctx), "cpu ctx");
        velri_device_id cpu_dev = 0U;
        CHECK_OK(velri_context_find_device(ctx, "cpu", 0U, &cpu_dev), "find cpu");
        verify_all_ops_on_device(ctx, cpu_dev, vnames[vi], false);
        velri_context_destroy(ctx);
    }

#ifdef VELRI_CORE_HAS_VULKAN
    velri_context_config vcfg = {
        .cpu_workers = 1U,
        .cpu_variant = VELRI_CPU_AUTO,
        .enable_vulkan = true,
    };
    velri_context *vctx = nullptr;
    CHECK_OK(velri_context_create(&vcfg, &vctx), "vk ctx");
    velri_device_id vk_dev = 0U;
    if (velri_context_find_device(vctx, "vulkan", 0U, &vk_dev) == VELRI_OK) {
        velri_device_caps caps;
        CHECK_OK(velri_context_device_caps(vctx, vk_dev, &caps), "vk caps");
        printf("  Vulkan GPU: %s (UMA=%d, VRAM=%.1f GiB, align=%zu)\n", caps.device_name,
               (int)caps.is_uma, (double)caps.mem_total_bytes / (1024.0 * 1024.0 * 1024.0),
               caps.min_align_bytes);
        verify_all_ops_on_device(vctx, vk_dev, "vulkan:0", true);
    } else {
        printf("  [SKIP] vulkan:0 (no Vulkan 1.3 GPU available at runtime).\n");
    }
    velri_context_destroy(vctx);
#endif

    printf("=== All Unified Kernel Tests PASSED ===\n");
    return 0;
}
