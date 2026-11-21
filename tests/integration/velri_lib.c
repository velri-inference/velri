#include <math.h>
#include <velri_lib/velri_lib.h>

/* Check public C23 headers, exported CMake targets (VelriLib::core + VelriLib::cpu),
 * direct Slang AOT Linear call, and end-to-end Graph -> Prepare -> Slot execution. */
static int check_direct_linear(void) {
    const double input[] = {1.0, 2.0, 3.0};
    const double weights[] = {4.0, 5.0, 6.0, -1.0, 0.0, 1.0};
    const double bias[] = {0.5, -0.5};
    double output[2] = {0.0, 0.0};
    const velri_linear_desc desc = velri_desc_f64();
    if (velri_slang_cpu_linear(&desc, 1, 2, 3, input, 3, weights, 3, bias, nullptr, nullptr,
                               output, 2, VELRI_CPU_AUTO, nullptr, 1) != VELRI_OK) {
        return 1;
    }
    return (output[0] == 32.5 && output[1] == 1.5) ? 0 : 2;
}

static int check_graph_pipeline(void) {
    velri_context *ctx = nullptr;
    if (velri_context_create(nullptr, &ctx) != VELRI_OK) return 10;

    velri_device_id cpu_dev = 0U;
    if (velri_context_find_device(ctx, "cpu", 0U, &cpu_dev) != VELRI_OK) {
        velri_context_destroy(ctx);
        return 11;
    }

    velri_graph *g = nullptr;
    if (velri_graph_create(&g) != VELRI_OK) {
        velri_context_destroy(ctx);
        return 12;
    }

    velri_storage_id sx = 0U, sw = 0U, sy = 0U;
    velri_storage_desc dx = {VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, 3U * sizeof(float), 64U, cpu_dev, "x"};
    velri_storage_desc dw = {VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, 6U * sizeof(float), 64U, cpu_dev, "w"};
    velri_storage_desc dy = {VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, 2U * sizeof(float), 64U, cpu_dev, "y"};
    if (velri_graph_add_storage(g, &dx, &sx) != VELRI_OK ||
        velri_graph_add_storage(g, &dw, &sw) != VELRI_OK ||
        velri_graph_add_storage(g, &dy, &sy) != VELRI_OK) {
        velri_graph_destroy(g);
        velri_context_destroy(ctx);
        return 13;
    }

    velri_tensor_view vx_t, vw_t, vy_t;
    size_t dim_x[2] = {1U, 3U}, dim_w[2] = {2U, 3U}, dim_y[2] = {1U, 2U};
    velri_view_id vx = 0U, vw = 0U, vy = 0U;
    if (velri_tensor_view_contiguous(sx, VELRI_DTYPE_F32, 2U, dim_x, 0U, &vx_t) != VELRI_OK ||
        velri_tensor_view_contiguous(sw, VELRI_DTYPE_F32, 2U, dim_w, 0U, &vw_t) != VELRI_OK ||
        velri_tensor_view_contiguous(sy, VELRI_DTYPE_F32, 2U, dim_y, 0U, &vy_t) != VELRI_OK ||
        velri_graph_add_view(g, &vx_t, &vx) != VELRI_OK ||
        velri_graph_add_view(g, &vw_t, &vw) != VELRI_OK ||
        velri_graph_add_view(g, &vy_t, &vy) != VELRI_OK) {
        velri_graph_destroy(g);
        velri_context_destroy(ctx);
        return 14;
    }

    velri_op_desc op = {
        .kind = VELRI_OP_LINEAR,
        .num_inputs = 2U,
        .num_outputs = 1U,
        .inputs = {vx, vw},
        .outputs = {vy},
        .attr.linear.desc = velri_desc_f32(),
    };
    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    float x[3] = {1.0f, 2.0f, 3.0f};
    float w[6] = {4.0f, 5.0f, 6.0f, -1.0f, 0.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};

    int rc = 0;
    if (velri_graph_add_op(g, &op, cpu_dev, nullptr) != VELRI_OK ||
        velri_plan_prepare(ctx, g, nullptr, &plan) != VELRI_OK ||
        velri_slot_create(plan, &slot) != VELRI_OK ||
        velri_slot_bind_host(slot, sx, x, sizeof(x)) != VELRI_OK ||
        velri_slot_bind_host(slot, sw, w, sizeof(w)) != VELRI_OK ||
        velri_slot_bind_host(slot, sy, y, sizeof(y)) != VELRI_OK ||
        velri_slot_run(slot, nullptr, nullptr) != VELRI_OK) {
        rc = 15;
    } else if (fabsf(y[0] - 32.0f) > 1e-5f || fabsf(y[1] - 2.0f) > 1e-5f) {
        rc = 16;
    }

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    velri_context_destroy(ctx);
    return rc;
}

int main(void) {
    int rc = check_direct_linear();
    if (rc != 0) return rc;
    return check_graph_pipeline();
}
