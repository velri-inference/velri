#include <math.h>
#include <velri_lib/velri_lib.h>

/* Check public C23 headers, exported `VelriLib::velri` target, and end-to-end
 * Graph -> Plan -> Slot execution for both FP64 and FP32 Linear. */
static int check_linear_graph(velri_dtype dt, velri_linear_desc desc, const void *x,
                              size_t x_bytes, const void *w, size_t w_bytes, const void *bias,
                              size_t b_bytes, void *y, size_t y_bytes) {
    velri_context *ctx = nullptr;
    if (velri_context_create(nullptr, &ctx) != VELRI_OK) return 1;

    velri_device_id cpu_dev = 0U;
    if (velri_context_find_device(ctx, "cpu", 0U, &cpu_dev) != VELRI_OK) {
        velri_context_destroy(ctx);
        return 2;
    }

    velri_graph *g = nullptr;
    if (velri_graph_create(&g) != VELRI_OK) {
        velri_context_destroy(ctx);
        return 3;
    }

    velri_storage_id sx = 0U, sw = 0U, sb = 0U, sy = 0U;
    velri_storage_desc dx = {VELRI_STORAGE_INPUT, dt, x_bytes, 64U, cpu_dev};
    velri_storage_desc dw = {VELRI_STORAGE_WEIGHT, dt, w_bytes, 64U, cpu_dev};
    velri_storage_desc db = {VELRI_STORAGE_WEIGHT, dt, b_bytes, 64U, cpu_dev};
    velri_storage_desc dy = {VELRI_STORAGE_OUTPUT, dt, y_bytes, 64U, cpu_dev};
    if (velri_graph_add_storage(g, &dx, &sx) != VELRI_OK ||
        velri_graph_add_storage(g, &dw, &sw) != VELRI_OK ||
        velri_graph_add_storage(g, &dy, &sy) != VELRI_OK ||
        (bias && velri_graph_add_storage(g, &db, &sb) != VELRI_OK)) {
        velri_graph_destroy(g);
        velri_context_destroy(ctx);
        return 4;
    }

    velri_tensor_view vx_t, vw_t, vb_t, vy_t;
    size_t dim_x[2] = {1U, 3U}, dim_w[2] = {2U, 3U}, dim_b[1] = {2U}, dim_y[2] = {1U, 2U};
    velri_view_id vx = 0U, vw = 0U, vb = VELRI_INVALID_ID, vy = 0U;
    if (velri_tensor_view_contiguous(sx, dt, 2U, dim_x, 0U, &vx_t) != VELRI_OK ||
        velri_tensor_view_contiguous(sw, dt, 2U, dim_w, 0U, &vw_t) != VELRI_OK ||
        velri_tensor_view_contiguous(sy, dt, 2U, dim_y, 0U, &vy_t) != VELRI_OK ||
        velri_graph_add_view(g, &vx_t, &vx) != VELRI_OK ||
        velri_graph_add_view(g, &vw_t, &vw) != VELRI_OK ||
        velri_graph_add_view(g, &vy_t, &vy) != VELRI_OK) {
        velri_graph_destroy(g);
        velri_context_destroy(ctx);
        return 5;
    }
    if (bias && (velri_tensor_view_contiguous(sb, dt, 1U, dim_b, 0U, &vb_t) != VELRI_OK ||
                 velri_graph_add_view(g, &vb_t, &vb) != VELRI_OK)) {
        velri_graph_destroy(g);
        velri_context_destroy(ctx);
        return 6;
    }

    velri_op_desc op = {
        .kind = VELRI_OP_LINEAR,
        .num_inputs = bias ? 3U : 2U,
        .num_outputs = 1U,
        .inputs = {vx, vw, vb},
        .outputs = {vy},
        .attr.linear.desc = desc,
    };

    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    int rc = 0;
    if (velri_graph_add_op(g, &op, cpu_dev, nullptr) != VELRI_OK ||
        velri_plan_prepare(ctx, g, nullptr, &plan) != VELRI_OK ||
        velri_plan_bind_weight_host(plan, sw, w, w_bytes) != VELRI_OK ||
        (bias && velri_plan_bind_weight_host(plan, sb, bias, b_bytes) != VELRI_OK) ||
        velri_slot_create(plan, &slot) != VELRI_OK ||
        velri_slot_bind_host(slot, sx, (void *)x, x_bytes) != VELRI_OK ||
        velri_slot_bind_host(slot, sy, y, y_bytes) != VELRI_OK ||
        velri_slot_run(slot, nullptr, nullptr) != VELRI_OK) {
        rc = 7;
    }

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    velri_context_destroy(ctx);
    return rc;
}

int main(void) {
    /* 1. FP64 Linear with bias */
    const double x64[3] = {1.0, 2.0, 3.0};
    const double w64[6] = {4.0, 5.0, 6.0, -1.0, 0.0, 1.0};
    const double b64[2] = {0.5, -0.5};
    double y64[2] = {0.0, 0.0};
    int rc = check_linear_graph(VELRI_DTYPE_F64, velri_desc_f64(), x64, sizeof(x64), w64,
                                sizeof(w64), b64, sizeof(b64), y64, sizeof(y64));
    if (rc != 0 || fabs(y64[0] - 32.5) > 1e-12 || fabs(y64[1] - 1.5) > 1e-12) {
        return rc ? rc : 10;
    }

    /* 2. FP32 Linear without bias */
    const float x32[3] = {1.0f, 2.0f, 3.0f};
    const float w32[6] = {4.0f, 5.0f, 6.0f, -1.0f, 0.0f, 1.0f};
    float y32[2] = {0.0f, 0.0f};
    rc = check_linear_graph(VELRI_DTYPE_F32, velri_desc_f32(), x32, sizeof(x32), w32, sizeof(w32),
                            nullptr, 0U, y32, sizeof(y32));
    if (rc != 0 || fabsf(y32[0] - 32.0f) > 1e-5f || fabsf(y32[1] - 2.0f) > 1e-5f) {
        return rc ? rc : 20;
    }
    return 0;
}
