// Shared GPU subgroup + workgroup reduction primitives (no hardcoded subgroup size).
// Reused by Vulkan RMSNorm, LayerNorm, Softmax, and GEMV kernels.
module reductions;

// Maximum supported subgroups per workgroup (256 threads / min subgroup 4 = 64).
static const uint kMaxSubgroups = 64u;
groupshared float g_reduce_buf[kMaxSubgroups];

public float wg_reduce_sum(float val, uint tid, uint wg_size) {
    float wave_sum = WaveActiveSum(val);
    uint lane_count = WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint wave_id = tid / lane_count;
    uint num_waves = (wg_size + lane_count - 1u) / lane_count;

    if (num_waves <= 1u) {
        return wave_sum;
    }
    if (lane_id == 0u) {
        g_reduce_buf[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0f;
    if (wave_id == 0u) {
        float v = (lane_id < num_waves) ? g_reduce_buf[lane_id] : 0.0f;
        total = WaveActiveSum(v);
        if (lane_id == 0u) {
            g_reduce_buf[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return g_reduce_buf[0];
}

public float wg_reduce_max(float val, uint tid, uint wg_size) {
    float wave_max = WaveActiveMax(val);
    uint lane_count = WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint wave_id = tid / lane_count;
    uint num_waves = (wg_size + lane_count - 1u) / lane_count;

    if (num_waves <= 1u) {
        return wave_max;
    }
    if (lane_id == 0u) {
        g_reduce_buf[wave_id] = wave_max;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        float v = (lane_id < num_waves) ? g_reduce_buf[lane_id] : -3.402823466e+38f;
        float total = WaveActiveMax(v);
        if (lane_id == 0u) {
            g_reduce_buf[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return g_reduce_buf[0];
}
