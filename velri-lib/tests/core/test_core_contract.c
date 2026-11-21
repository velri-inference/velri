#include "mock_async_backend.h"
#include "os.h"
#include "reference.h"
#include "velri_lib/velri_lib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Stage 1 Core Contract Test Suite (`docs/EN/roadmap.md` Stage 1 acceptance):
 *   1. TensorView validation, zero-copy subviews, and explicit K/V shared/derived roles.
 *   2. Standalone client graph (SwiGLU MLP block) with scratch arena liveness reuse and
 *      TWO independent slots executing concurrently on separate OS threads (0 B run allocs).
 *   3. No silent CPU fallback when a chosen device does not support an op.
 *   4. Cross-backend RAM transfer bus (`cpu:0` -> RAM staging -> `mock_async:0` -> RAM -> `cpu:0`).
 *   5. Pre-accept rejection (zero work started, slot stays clean).
 *   6. Post-accept failure & partial-accept failure (runtime drains in-flight work before return).
 *   7. Cancellation before accept, mid-flight after partial accept, and after completion.
 *   8. Hybrid CPU + Vulkan graph test when compiled with `-DVELRI_ENABLE_VULKAN=ON` and GPU present.
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

static velri_storage_id add_f32_storage(
    velri_graph *g,
    velri_storage_kind kind,
    size_t elems,
    velri_device_id dev,
    const char *name) {
    velri_storage_desc d = {
        .kind = kind,
        .dtype = VELRI_DTYPE_F32,
        .byte_size = elems * sizeof(float),
        .align = 64U,
        .device_id = dev,
        .debug_name = name,
    };
    velri_storage_id id = 0U;
    CHECK_OK(velri_graph_add_storage(g, &d, &id), name);
    return id;
}

static velri_view_id add_2d_view(velri_graph *g, velri_storage_id sid, size_t r, size_t c) {
    size_t dims[2] = {r, c};
    velri_tensor_view v;
    CHECK_OK(velri_tensor_view_contiguous(sid, VELRI_DTYPE_F32, 2U, dims, 0U, &v), "view2d");
    velri_view_id vid = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &vid), "add_view2d");
    return vid;
}

static velri_view_id add_1d_view(velri_graph *g, velri_storage_id sid, size_t n) {
    size_t dims[1] = {n};
    velri_tensor_view v;
    CHECK_OK(velri_tensor_view_contiguous(sid, VELRI_DTYPE_F32, 1U, dims, 0U, &v), "view1d");
    velri_view_id vid = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &vid), "add_view1d");
    return vid;
}

/* ---- Test 1: TensorView validation & K/V shared/derived roles (0 extra storage) ---- */
static void test_tensor_views_and_kv_roles(velri_context *ctx, velri_device_id cpu_dev) {
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "graph_create");

    /* Single shared KV storage: K and V views alias the exact same storage_id (0 extra bytes). */
    velri_storage_id kv_store = add_f32_storage(g, VELRI_STORAGE_STATE, 16U * 32U, cpu_dev, "kv");
    velri_view_id v_view = add_2d_view(g, kv_store, 16U, 32U);
    velri_view_id k_shared_view = add_2d_view(g, kv_store, 16U, 32U);

    velri_kv_binding shared_kv = {
        .kind = VELRI_KV_SHARED,
        .k_view = k_shared_view,
        .v_view = v_view,
        .derive_node = VELRI_INVALID_ID,
    };
    CHECK_OK(velri_graph_add_kv_binding(g, &shared_kv), "kv_shared");

    /* Out-of-bounds view must be rejected cleanly. */
    size_t oob_dims[2] = {32U, 32U};
    velri_tensor_view bad_v;
    CHECK_OK(velri_tensor_view_contiguous(kv_store, VELRI_DTYPE_F32, 2U, oob_dims, 0U, &bad_v),
             "bad_v_ctor");
    velri_view_id dummy_id = 0U;
    CHECK(velri_graph_add_view(g, &bad_v, &dummy_id) == VELRI_ERR_INVALID, "oob view rejected");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare kv_shared");
    velri_memory_report rep;
    CHECK_OK(velri_plan_memory_report(plan, &rep), "report kv_shared");

    CHECK(rep.kv_bindings_count == 1U && rep.kv_shared_storage_count == 1U, "kv role counted");
    CHECK(rep.devices[cpu_dev].state_bytes == 16U * 32U * sizeof(float),
          "shared K=V allocates exactly ONE storage, not two");

    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    printf("  [PASS] TensorView bounds & shared K/V role (0 extra storage).\n");
}

/* ---- Test 2: SwiGLU MLP graph + scratch reuse + 2 concurrent isolated slots ---- */
typedef struct {
    velri_slot *slot;
    const float *x;
    float *y;
    velri_status st;
} slot_thread_args;

static void slot_runner_thread(void *arg) {
    slot_thread_args *a = (slot_thread_args *)arg;
    a->st = velri_slot_run(a->slot, nullptr, nullptr);
}

static void ref_swiglu_mlp_block(
    size_t rows, size_t d, size_t ff, float eps,
    const float *x, const float *norm_w,
    const float *w_gate, const float *w_up, const float *w_down,
    float *y_out) {
    float *xn = (float *)calloc(rows * d, sizeof(float));
    float *g = (float *)calloc(rows * ff, sizeof(float));
    float *u = (float *)calloc(rows * ff, sizeof(float));
    float *h = (float *)calloc(rows * ff, sizeof(float));
    float *down = (float *)calloc(rows * d, sizeof(float));

    velri_linear_desc f32_lin = velri_desc_f32();
    ref_rmsnorm(VELRI_DTYPE_F32, VELRI_DTYPE_F32, rows, d, eps, false, x, d, norm_w, xn, d);
    ref_linear_run(&f32_lin, rows, ff, d, xn, d, nullptr, w_gate, d, nullptr, nullptr, g, ff);
    ref_linear_run(&f32_lin, rows, ff, d, xn, d, nullptr, w_up, d, nullptr, nullptr, u, ff);
    ref_gated_mlp_f32(VELRI_ACT_SILU, rows * ff, 0.0f, g, u, h);
    ref_linear_run(&f32_lin, rows, d, ff, h, ff, nullptr, w_down, ff, nullptr, nullptr, down, d);
    ref_vec_axpby_f32(rows * d, 1.0f, x, 1.0f, down, y_out);

    free(xn); free(g); free(u); free(h); free(down);
}

static void test_standalone_mlp_two_concurrent_slots(velri_context *ctx, velri_device_id cpu_dev) {
    const size_t rows = 8U;
    const size_t d = 64U;
    const size_t ff = 128U;
    const float eps = 1e-5f;

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "mlp graph_create");

    /* Declare storages: immutable weights, inputs/outputs, and 5 temporary scratch buffers. */
    velri_storage_id s_x = add_f32_storage(g, VELRI_STORAGE_INPUT, rows * d, cpu_dev, "x");
    velri_storage_id s_nw = add_f32_storage(g, VELRI_STORAGE_WEIGHT, d, cpu_dev, "norm_w");
    velri_storage_id s_wg = add_f32_storage(g, VELRI_STORAGE_WEIGHT, ff * d, cpu_dev, "w_gate");
    velri_storage_id s_wu = add_f32_storage(g, VELRI_STORAGE_WEIGHT, ff * d, cpu_dev, "w_up");
    velri_storage_id s_wd = add_f32_storage(g, VELRI_STORAGE_WEIGHT, d * ff, cpu_dev, "w_down");
    velri_storage_id s_xn = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * d, cpu_dev, "xn");
    velri_storage_id s_g = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * ff, cpu_dev, "gate");
    velri_storage_id s_u = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * ff, cpu_dev, "up");
    velri_storage_id s_h = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * ff, cpu_dev, "act");
    velri_storage_id s_dn = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * d, cpu_dev, "down");
    velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, rows * d, cpu_dev, "y");

    velri_view_id vx = add_2d_view(g, s_x, rows, d);
    velri_view_id vnw = add_1d_view(g, s_nw, d);
    velri_view_id vwg = add_2d_view(g, s_wg, ff, d);
    velri_view_id vwu = add_2d_view(g, s_wu, ff, d);
    velri_view_id vwd = add_2d_view(g, s_wd, d, ff);
    velri_view_id vxn = add_2d_view(g, s_xn, rows, d);
    velri_view_id vg = add_2d_view(g, s_g, rows, ff);
    velri_view_id vu = add_2d_view(g, s_u, rows, ff);
    velri_view_id vh = add_2d_view(g, s_h, rows, ff);
    velri_view_id vdn = add_2d_view(g, s_dn, rows, d);
    velri_view_id vy = add_2d_view(g, s_y, rows, d);

    /* 1. RMSNorm(x, norm_w) -> xn */
    velri_op_desc op_norm = {
        .kind = VELRI_OP_RMSNORM,
        .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vx, vnw}, .outputs = {vxn},
        .attr.rmsnorm = {.eps = eps, .unit_offset = false},
    };
    CHECK_OK(velri_graph_add_op(g, &op_norm, cpu_dev, nullptr), "add rmsnorm");

    /* 2. gate = Linear(xn, w_gate); 3. up = Linear(xn, w_up) */
    velri_op_desc op_gate = {
        .kind = VELRI_OP_LINEAR,
        .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vxn, vwg}, .outputs = {vg},
        .attr.linear.desc = velri_desc_f32(),
    };
    velri_op_desc op_up = {
        .kind = VELRI_OP_LINEAR,
        .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vxn, vwu}, .outputs = {vu},
        .attr.linear.desc = velri_desc_f32(),
    };
    CHECK_OK(velri_graph_add_op(g, &op_gate, cpu_dev, nullptr), "add gate");
    CHECK_OK(velri_graph_add_op(g, &op_up, cpu_dev, nullptr), "add up");

    /* 4. h = SwiGLU(gate, up) */
    velri_op_desc op_act = {
        .kind = VELRI_OP_GATED_MLP,
        .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vg, vu}, .outputs = {vh},
        .attr.gated_mlp = {.act = VELRI_ACT_SILU, .clamp_limit = 0.0f},
    };
    CHECK_OK(velri_graph_add_op(g, &op_act, cpu_dev, nullptr), "add swiglu");

    /* 5. down = Linear(h, w_down) */
    velri_op_desc op_down = {
        .kind = VELRI_OP_LINEAR,
        .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vh, vwd}, .outputs = {vdn},
        .attr.linear.desc = velri_desc_f32(),
    };
    CHECK_OK(velri_graph_add_op(g, &op_down, cpu_dev, nullptr), "add down");

    /* 6. y = x + down */
    velri_op_desc op_res = {
        .kind = VELRI_OP_VEC_AXPBY,
        .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vx, vdn}, .outputs = {vy},
        .attr.axpby = {.alpha = 1.0f, .beta = 1.0f},
    };
    CHECK_OK(velri_graph_add_op(g, &op_res, cpu_dev, nullptr), "add residual");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare mlp plan");

    velri_memory_report rep;
    CHECK_OK(velri_plan_memory_report(plan, &rep), "mlp memory report");
    CHECK(rep.devices[cpu_dev].scratch_packed_bytes < rep.devices[cpu_dev].scratch_unpacked_bytes,
          "scratch liveness packer must reuse non-overlapping scratch buffers");
    CHECK(rep.num_ram_transfers == 0U, "single-device CPU graph has 0 RAM transfers");

    /* Shared immutable weights + two distinct slot inputs/outputs. */
    uint64_t rng = 0x123456789ABCDEFULL;
    float *nw = (float *)calloc(d, sizeof(float));
    float *wg = (float *)calloc(ff * d, sizeof(float));
    float *wu = (float *)calloc(ff * d, sizeof(float));
    float *wd = (float *)calloc(d * ff, sizeof(float));
    float *x0 = (float *)calloc(rows * d, sizeof(float));
    float *x1 = (float *)calloc(rows * d, sizeof(float));
    float *y0 = (float *)calloc(rows * d, sizeof(float));
    float *y1 = (float *)calloc(rows * d, sizeof(float));
    float *ref0 = (float *)calloc(rows * d, sizeof(float));
    float *ref1 = (float *)calloc(rows * d, sizeof(float));

    for (size_t i = 0; i < d; ++i) nw[i] = velri_test_rand_f32(&rng, 0.5f, 1.5f);
    for (size_t i = 0; i < ff * d; ++i) {
        wg[i] = velri_test_rand_f32(&rng, -0.25f, 0.25f);
        wu[i] = velri_test_rand_f32(&rng, -0.25f, 0.25f);
        wd[i] = velri_test_rand_f32(&rng, -0.25f, 0.25f);
    }
    for (size_t i = 0; i < rows * d; ++i) {
        x0[i] = velri_test_rand_f32(&rng, -1.0f, 1.0f);
        x1[i] = velri_test_rand_f32(&rng, -2.0f, 2.0f);
    }

    ref_swiglu_mlp_block(rows, d, ff, eps, x0, nw, wg, wu, wd, ref0);
    ref_swiglu_mlp_block(rows, d, ff, eps, x1, nw, wg, wu, wd, ref1);

    /* Create TWO independent slots sharing the same immutable plan and weights. */
    velri_slot *slot0 = nullptr, *slot1 = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot0), "slot0 create");
    CHECK_OK(velri_slot_create(plan, &slot1), "slot1 create");

    velri_slot *slots[2] = {slot0, slot1};
    float *xs[2] = {x0, x1};
    float *ys[2] = {y0, y1};
    for (int s = 0; s < 2; ++s) {
        CHECK_OK(velri_slot_bind_host(slots[s], s_nw, nw, d * sizeof(float)), "bind nw");
        CHECK_OK(velri_slot_bind_host(slots[s], s_wg, wg, ff * d * sizeof(float)), "bind wg");
        CHECK_OK(velri_slot_bind_host(slots[s], s_wu, wu, ff * d * sizeof(float)), "bind wu");
        CHECK_OK(velri_slot_bind_host(slots[s], s_wd, wd, d * ff * sizeof(float)), "bind wd");
        CHECK_OK(velri_slot_bind_host(slots[s], s_x, xs[s], rows * d * sizeof(float)), "bind x");
        CHECK_OK(velri_slot_bind_host(slots[s], s_y, ys[s], rows * d * sizeof(float)), "bind y");
    }

    /* Run both slots concurrently on two OS threads across 4 iterations. */
    for (int iter = 0; iter < 4; ++iter) {
        slot_thread_args a0 = {.slot = slot0, .x = x0, .y = y0, .st = VELRI_ERR_EXEC};
        slot_thread_args a1 = {.slot = slot1, .x = x1, .y = y1, .st = VELRI_ERR_EXEC};
        velri_os_thread *t0 = nullptr, *t1 = nullptr;
        CHECK(velri_os_thread_spawn(&t0, slot_runner_thread, &a0), "spawn t0");
        CHECK(velri_os_thread_spawn(&t1, slot_runner_thread, &a1), "spawn t1");
        velri_os_thread_join(t0);
        velri_os_thread_join(t1);
        CHECK_OK(a0.st, "slot0 concurrent run");
        CHECK_OK(a1.st, "slot1 concurrent run");
        CHECK(velri_slot_last_run_allocs(slot0) == 0U && velri_slot_last_run_allocs(slot1) == 0U,
              "0 B heap allocs on hot execution path");
    }

    double err0 = 0.0, err1 = 0.0;
    for (size_t i = 0; i < rows * d; ++i) {
        double d0 = fabs((double)y0[i] - (double)ref0[i]);
        double d1 = fabs((double)y1[i] - (double)ref1[i]);
        if (d0 > err0) err0 = d0;
        if (d1 > err1) err1 = d1;
    }
    CHECK(err0 < 1e-4 && err1 < 1e-4, "both concurrent slots match FP64 reference");

    velri_slot_destroy(slot0);
    velri_slot_destroy(slot1);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(nw); free(wg); free(wu); free(wd);
    free(x0); free(x1); free(y0); free(y1); free(ref0); free(ref1);
    printf("  [PASS] Standalone SwiGLU MLP graph + scratch reuse (%zu -> %zu B) + 2 concurrent slots (err=%.2e/%.2e).\n",
           rep.devices[cpu_dev].scratch_unpacked_bytes, rep.devices[cpu_dev].scratch_packed_bytes,
           err0, err1);
}

/* ---- Test 3: No silent CPU fallback when target device rejects an op ---- */
static void test_no_silent_cpu_fallback(velri_context *ctx, velri_device_id mock_dev) {
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "fallback graph");
    velri_storage_id sx = add_f32_storage(g, VELRI_STORAGE_INPUT, 4U * 16U, mock_dev, "x");
    velri_storage_id sy = add_f32_storage(g, VELRI_STORAGE_OUTPUT, 4U * 16U, mock_dev, "y");
    velri_view_id vx = add_2d_view(g, sx, 4U, 16U);
    velri_view_id vy = add_2d_view(g, sy, 4U, 16U);

    velri_op_desc op = {
        .kind = VELRI_OP_RMSNORM, /* mock_async supports ONLY VEC_AXPBY */
        .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vx}, .outputs = {vy},
        .attr.rmsnorm = {.eps = 1e-5f, .unit_offset = false},
    };
    CHECK_OK(velri_graph_add_op(g, &op, mock_dev, nullptr), "add unsupported op");

    velri_plan *plan = nullptr;
    velri_plan_options strict_opts = {.allow_cpu_fallback = false};
    CHECK(velri_plan_prepare(ctx, g, &strict_opts, &plan) == VELRI_ERR_UNSUPPORTED,
          "must reject unsupported op without silent CPU fallback");
    CHECK(plan == nullptr, "failed prepare returns nullptr");

    /* Explicit opt-in fallback must succeed. */
    velri_plan_options fb_opts = {.allow_cpu_fallback = true};
    CHECK_OK(velri_plan_prepare(ctx, g, &fb_opts, &plan), "explicit fallback succeeds");
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    printf("  [PASS] No silent CPU fallback (strict reject + explicit opt-in).\n");
}

/* ---- Test 4: Cross-backend RAM transfer bus (cpu:0 -> RAM -> mock_async:0 -> RAM -> cpu:0) ---- */
static void test_cross_backend_ram_transfer(
    velri_context *ctx,
    velri_device_id cpu_dev,
    velri_device_id mock_dev) {
    velri_mock_async_reset_ctrl();
    const size_t rows = 4U, dim = 32U, n = rows * dim;

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "hybrid graph");
    velri_storage_id s_x = add_f32_storage(g, VELRI_STORAGE_INPUT, n, cpu_dev, "x");
    velri_storage_id s_b = add_f32_storage(g, VELRI_STORAGE_INPUT, n, cpu_dev, "b");
    velri_storage_id s_n = add_f32_storage(g, VELRI_STORAGE_SCRATCH, n, cpu_dev, "normed");
    velri_storage_id s_m = add_f32_storage(g, VELRI_STORAGE_SCRATCH, n, mock_dev, "scaled");
    velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, n, cpu_dev, "y");

    velri_view_id vx = add_2d_view(g, s_x, rows, dim);
    velri_view_id vb = add_2d_view(g, s_b, rows, dim);
    velri_view_id vn = add_2d_view(g, s_n, rows, dim);
    velri_view_id vm = add_2d_view(g, s_m, rows, dim);
    velri_view_id vy = add_2d_view(g, s_y, rows, dim);

    velri_op_desc op0 = {
        .kind = VELRI_OP_RMSNORM, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vx}, .outputs = {vn}, .attr.rmsnorm = {.eps = 1e-5f, .unit_offset = false},
    };
    velri_op_desc op1 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vn, vb}, .outputs = {vm}, .attr.axpby = {.alpha = 2.0f, .beta = -0.5f},
    };
    velri_op_desc op2 = {
        .kind = VELRI_OP_ACTIVATION, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vm}, .outputs = {vy}, .attr.activation = {.kind = VELRI_ACT_SILU},
    };
    CHECK_OK(velri_graph_add_op(g, &op0, cpu_dev, nullptr), "cpu norm");
    CHECK_OK(velri_graph_add_op(g, &op1, mock_dev, nullptr), "mock axpby");
    CHECK_OK(velri_graph_add_op(g, &op2, cpu_dev, nullptr), "cpu silu");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare hybrid");
    velri_memory_report rep;
    CHECK_OK(velri_plan_memory_report(plan, &rep), "hybrid report");
    CHECK(rep.num_ram_transfers == 3U, "expected 3 RAM transfers (vn->mock, vb->mock, vm->cpu)");
    CHECK(rep.ram_staging_bytes == n * sizeof(float), "RAM staging sized to transfer");

    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot), "hybrid slot");

    float *hx = (float *)calloc(n, sizeof(float));
    float *hb = (float *)calloc(n, sizeof(float));
    float *hy = (float *)calloc(n, sizeof(float));
    float *t0 = (float *)calloc(n, sizeof(float));
    float *t1 = (float *)calloc(n, sizeof(float));
    float *href = (float *)calloc(n, sizeof(float));

    uint64_t rng = 0xCAFEBABEULL;
    for (size_t i = 0; i < n; ++i) {
        hx[i] = velri_test_rand_f32(&rng, -2.0f, 2.0f);
        hb[i] = velri_test_rand_f32(&rng, -1.0f, 1.0f);
    }
    ref_rmsnorm(VELRI_DTYPE_F32, VELRI_DTYPE_F32, rows, dim, 1e-5f, false, hx, dim, nullptr, t0, dim);
    ref_vec_axpby_f32(n, 2.0f, t0, -0.5f, hb, t1);
    ref_activation_f32(VELRI_ACT_SILU, n, t1, href);

    CHECK_OK(velri_slot_bind_host(slot, s_x, hx, n * sizeof(float)), "bind hx");
    CHECK_OK(velri_slot_bind_host(slot, s_b, hb, n * sizeof(float)), "bind hb");
    CHECK_OK(velri_slot_bind_host(slot, s_y, hy, n * sizeof(float)), "bind hy");
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run hybrid");

    velri_mock_async_ctrl *ctrl = velri_mock_async_control();
    CHECK(atomic_load(&ctrl->total_uploads) == 2U, "mock received 2 RAM uploads");
    CHECK(atomic_load(&ctrl->total_downloads) == 1U, "mock performed 1 RAM download");
    CHECK(atomic_load(&ctrl->in_flight) == 0, "no work left in flight");

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double e = fabs((double)hy[i] - (double)href[i]);
        if (e > max_err) max_err = e;
    }
    CHECK(max_err < 1e-5, "hybrid CPU <-> RAM <-> mock_async output matches reference");

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(hx); free(hb); free(hy); free(t0); free(t1); free(href);
    printf("  [PASS] Cross-backend RAM transfer bus (3 transfers, max_err=%.2e).\n", max_err);
}

/* ---- Tests 5, 6, 7: Pre-accept rejection, post-accept error draining, and cancellation ---- */
typedef struct {
    velri_cancel_token *tok;
    uint32_t delay_ms;
} delayed_cancel_args;

static void delayed_cancel_thread(void *arg) {
    delayed_cancel_args *a = (delayed_cancel_args *)arg;
    velri_os_sleep_ms(a->delay_ms);
    velri_cancel_request(a->tok);
}

static void test_errors_and_cancellation(velri_context *ctx, velri_device_id mock_dev) {
    const size_t n = 64U;
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "err graph");
    velri_storage_id s_a = add_f32_storage(g, VELRI_STORAGE_INPUT, n, mock_dev, "a");
    velri_storage_id s_mid = add_f32_storage(g, VELRI_STORAGE_SCRATCH, n, mock_dev, "mid");
    velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, n, mock_dev, "y");
    velri_view_id va = add_1d_view(g, s_a, n);
    velri_view_id vm = add_1d_view(g, s_mid, n);
    velri_view_id vy = add_1d_view(g, s_y, n);

    velri_op_desc op0 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {va}, .outputs = {vm}, .attr.axpby = {.alpha = 2.0f, .beta = 0.0f},
    };
    velri_op_desc op1 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vm}, .outputs = {vy}, .attr.axpby = {.alpha = 3.0f, .beta = 0.0f},
    };
    CHECK_OK(velri_graph_add_op(g, &op0, mock_dev, nullptr), "async op0");
    CHECK_OK(velri_graph_add_op(g, &op1, mock_dev, nullptr), "async op1");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare async plan");
    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot), "async slot");

    float ha[64], hy[64];
    for (size_t i = 0; i < n; ++i) {
        ha[i] = (float)(i + 1U);
        hy[i] = 0.0f;
    }
    CHECK_OK(velri_slot_bind_host(slot, s_a, ha, sizeof(ha)), "bind a");
    CHECK_OK(velri_slot_bind_host(slot, s_y, hy, sizeof(hy)), "bind y");

    velri_mock_async_ctrl *ctrl = velri_mock_async_control();

    /* 5. Pre-accept rejection: completion == nullptr, zero work run, slot stays clean. */
    velri_mock_async_reset_ctrl();
    atomic_store(&ctrl->fail_pre_accept, true);
    CHECK(velri_slot_run(slot, nullptr, nullptr) == VELRI_ERR_EXEC, "pre-accept returns EXEC");
    CHECK(atomic_load(&ctrl->total_submitted) == 0U && atomic_load(&ctrl->in_flight) == 0,
          "pre-accept starts no work");
    CHECK(!velri_slot_needs_reset(slot), "pre-accept leaves slot clean (no reset needed)");

    /* 6a. Post-accept failure: async worker runs for 20ms then fails; runtime MUST wait! */
    velri_mock_async_reset_ctrl();
    atomic_store(&ctrl->delay_ms, 20U);
    atomic_store(&ctrl->fail_post_accept, true);
    CHECK(velri_slot_run(slot, nullptr, nullptr) == VELRI_ERR_EXEC, "post-accept returns EXEC");
    CHECK(atomic_load(&ctrl->in_flight) == 0 && atomic_load(&ctrl->total_completed) == 1U,
          "post-accept failure awaited in-flight worker before returning");
    CHECK(velri_slot_needs_reset(slot), "post-accept failure marks slot needing reset");
    CHECK(velri_slot_run(slot, nullptr, nullptr) == VELRI_ERR_BUSY,
          "dirty slot refuses new run until explicit reset");
    CHECK_OK(velri_slot_reset(slot), "slot reset");

    /* 6b. Partial-accept submit error: submit returns error WITH non-null completion. */
    velri_mock_async_reset_ctrl();
    atomic_store(&ctrl->delay_ms, 20U);
    atomic_store(&ctrl->fail_partial_submit, true);
    CHECK(velri_slot_run(slot, nullptr, nullptr) == VELRI_ERR_EXEC, "partial-accept returns EXEC");
    CHECK(atomic_load(&ctrl->in_flight) == 0 && atomic_load(&ctrl->total_completed) == 1U,
          "partial-accept error still awaited in-flight worker");
    CHECK(velri_slot_needs_reset(slot), "partial-accept marks slot needing reset");
    CHECK_OK(velri_slot_reset(slot), "slot reset 2");

    /* 7a. Cancellation BEFORE first accept: zero ops submitted, slot stays clean. */
    velri_mock_async_reset_ctrl();
    velri_cancel_token tok;
    velri_cancel_init(&tok);
    velri_cancel_request(&tok);
    CHECK(velri_slot_run(slot, nullptr, &tok) == VELRI_ERR_CANCELLED, "pre-cancel returns CANCELLED");
    CHECK(atomic_load(&ctrl->total_submitted) == 0U && !velri_slot_needs_reset(slot),
          "pre-cancel leaves slot clean");

    /* 7b. Cancellation MID-FLIGHT (during 35ms op0): op0 drained, op1 skipped, needs_reset=true. */
    velri_mock_async_reset_ctrl();
    atomic_store(&ctrl->delay_ms, 35U);
    velri_cancel_init(&tok);
    delayed_cancel_args ca = {.tok = &tok, .delay_ms = 10U};
    velri_os_thread *ct = nullptr;
    CHECK(velri_os_thread_spawn(&ct, delayed_cancel_thread, &ca), "spawn cancel thread");
    CHECK(velri_slot_run(slot, nullptr, &tok) == VELRI_ERR_CANCELLED, "mid-flight cancel");
    velri_os_thread_join(ct);
    CHECK(atomic_load(&ctrl->total_submitted) == 1U, "op1 must not be submitted after cancel");
    CHECK(atomic_load(&ctrl->in_flight) == 0 && atomic_load(&ctrl->total_completed) == 1U,
          "op0 must be fully drained before returning from cancelled run");
    CHECK(velri_slot_needs_reset(slot), "mid-flight cancel requires explicit slot reset");
    CHECK_OK(velri_slot_reset(slot), "slot reset after mid-flight cancel");

    /* 7c. Clean execution after reset (and cancellation after completion preserves output). */
    velri_mock_async_reset_ctrl();
    velri_cancel_init(&tok);
    CHECK_OK(velri_slot_run(slot, nullptr, &tok), "clean run after reset");
    velri_cancel_request(&tok); /* Late cancel after completion does not invalidate output. */
    for (size_t i = 0; i < n; ++i) {
        CHECK(fabs((double)hy[i] - (double)(6.0f * ha[i])) < 1e-5, "result intact");
    }

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    printf("  [PASS] Pre-accept rejection, post-accept/partial-accept drain, and cancellation.\n");
}

#ifdef VELRI_CORE_HAS_VULKAN
/* ---- Test 8: Optional real Hybrid CPU <-> RAM staging <-> Vulkan graph ---- */
static void test_hybrid_cpu_vulkan(velri_context *ctx, velri_device_id cpu_dev) {
    velri_device_id vk_dev = 0U;
    if (velri_context_find_device(ctx, "vulkan", 0U, &vk_dev) != VELRI_OK) {
        printf("  [SKIP] Hybrid CPU <-> Vulkan test (no Vulkan 1.3 GPU available at runtime).\n");
        return;
    }

    const size_t rows = 4U, k = 64U, m = 32U;
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "vk hybrid graph");

    velri_storage_id s_x = add_f32_storage(g, VELRI_STORAGE_INPUT, rows * k, cpu_dev, "x");
    velri_storage_id s_w = add_f32_storage(g, VELRI_STORAGE_WEIGHT, m * k, vk_dev, "w");
    velri_storage_id s_n = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * k, cpu_dev, "norm");
    velri_storage_id s_l = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * m, vk_dev, "logits");
    velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, rows * m, cpu_dev, "probs");

    velri_view_id vx = add_2d_view(g, s_x, rows, k);
    velri_view_id vw = add_2d_view(g, s_w, m, k);
    velri_view_id vn = add_2d_view(g, s_n, rows, k);
    velri_view_id vl = add_2d_view(g, s_l, rows, m);
    velri_view_id vy = add_2d_view(g, s_y, rows, m);

    velri_op_desc op0 = {
        .kind = VELRI_OP_RMSNORM, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vx}, .outputs = {vn}, .attr.rmsnorm = {.eps = 1e-5f, .unit_offset = false},
    };
    velri_op_desc op1 = {
        .kind = VELRI_OP_LINEAR, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vn, vw}, .outputs = {vl}, .attr.linear.desc = velri_desc_f32(),
    };
    velri_op_desc op2 = {
        .kind = VELRI_OP_SOFTMAX, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vl}, .outputs = {vy},
    };
    CHECK_OK(velri_graph_add_op(g, &op0, cpu_dev, nullptr), "cpu rmsnorm");
    CHECK_OK(velri_graph_add_op(g, &op1, vk_dev, nullptr), "vk linear");
    CHECK_OK(velri_graph_add_op(g, &op2, cpu_dev, nullptr), "cpu softmax");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare cpu+vk");
    velri_memory_report rep;
    CHECK_OK(velri_plan_memory_report(plan, &rep), "cpu+vk report");
    CHECK(rep.num_ram_transfers == 2U, "expected 2 RAM transfers (cpu->vk and vk->cpu)");

    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot), "cpu+vk slot");

    float *hx = (float *)calloc(rows * k, sizeof(float));
    float *hw = (float *)calloc(m * k, sizeof(float));
    float *hy = (float *)calloc(rows * m, sizeof(float));
    float *t0 = (float *)calloc(rows * k, sizeof(float));
    float *t1 = (float *)calloc(rows * m, sizeof(float));
    float *href = (float *)calloc(rows * m, sizeof(float));

    uint64_t rng = 0xDEADBEEF1234ULL;
    for (size_t i = 0; i < rows * k; ++i) hx[i] = velri_test_rand_f32(&rng, -1.5f, 1.5f);
    for (size_t i = 0; i < m * k; ++i) hw[i] = velri_test_rand_f32(&rng, -0.3f, 0.3f);

    velri_linear_desc ld = velri_desc_f32();
    ref_rmsnorm(VELRI_DTYPE_F32, VELRI_DTYPE_F32, rows, k, 1e-5f, false, hx, k, nullptr, t0, k);
    ref_linear_run(&ld, rows, m, k, t0, k, nullptr, hw, k, nullptr, nullptr, t1, m);
    ref_softmax_f32(rows, m, t1, m, href, m);

    CHECK_OK(velri_slot_bind_host(slot, s_x, hx, rows * k * sizeof(float)), "bind hx");
    CHECK_OK(velri_slot_bind_host(slot, s_w, hw, m * k * sizeof(float)), "bind hw");
    CHECK_OK(velri_slot_bind_host(slot, s_y, hy, rows * m * sizeof(float)), "bind hy");
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run cpu+vk");

    double max_err = 0.0;
    for (size_t i = 0; i < rows * m; ++i) {
        double e = fabs((double)hy[i] - (double)href[i]);
        if (e > max_err) max_err = e;
    }
    CHECK(max_err < 1e-4, "hybrid CPU <-> RAM <-> Vulkan output matches FP64 reference");

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(hx); free(hw); free(hy); free(t0); free(t1); free(href);
    printf("  [PASS] Hybrid CPU <-> RAM staging <-> Vulkan graph (max_err=%.2e).\n", max_err);
}
#endif

int main(void) {
    printf("=== Velri-lib Stage 1 Core & Backend Contract Suite ===\n");
    velri_context_config cfg = {
        .cpu_workers = 4U,
        .cpu_variant = VELRI_CPU_AUTO,
        .enable_vulkan = true,
    };
    velri_context *ctx = nullptr;
    CHECK_OK(velri_context_create(&cfg, &ctx), "context_create");
    CHECK_OK(velri_context_register_backend(ctx, velri_mock_async_backend_vtable()),
             "register mock_async");

    velri_device_id cpu_dev = 0U, mock_dev = 0U;
    CHECK_OK(velri_context_find_device(ctx, "cpu", 0U, &cpu_dev), "find cpu:0");
    CHECK_OK(velri_context_find_device(ctx, "mock_async", 0U, &mock_dev), "find mock_async:0");

    test_tensor_views_and_kv_roles(ctx, cpu_dev);
    test_standalone_mlp_two_concurrent_slots(ctx, cpu_dev);
    test_no_silent_cpu_fallback(ctx, mock_dev);
    test_cross_backend_ram_transfer(ctx, cpu_dev, mock_dev);
    test_errors_and_cancellation(ctx, mock_dev);
#ifdef VELRI_CORE_HAS_VULKAN
    test_hybrid_cpu_vulkan(ctx, cpu_dev);
#endif

    velri_context_destroy(ctx);
    printf("=== All Stage 1 Core Contract Tests PASSED ===\n");
    return 0;
}
