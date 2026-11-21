// Vulkan Embedding, CopyLastRow, KV-Cache Append, and Online-Softmax Attention (F32).
// Subgroup-size agnostic (Wave4..Wave128), zero global score scratch, zero divergent barriers.
import numeric;
import reductions;
import vk_common;

// ============================================================================
// 1. Embedding Lookup (Token ID table lookup or positional offset lookup)
// ============================================================================
struct EmbeddingArgs {
    Ptr<float, Access.Read> table;
    Ptr<int32_t, Access.Read> ids;
    Ptr<float> y;
    uint rows;
    uint dim;
    uint vocab;
    uint table_stride;
    uint out_stride;
    uint pos_offset;
    uint has_ids;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_embedding_f32(uniform EmbeddingArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint t = gid.x * 256u + gtid.x;
    uint base = t << 2u;
    uint row = gid.y;
    if (row >= a.rows || base >= a.dim) return;

    bool valid = false;
    uint idx = 0u;
    if (a.has_ids != 0u) {
        int32_t raw_id = a.ids[row];
        valid = (raw_id >= 0) && (uint(raw_id) < a.vocab);
        idx = uint(raw_id);
    } else {
        uint64_t pos64 = uint64_t(a.pos_offset) + uint64_t(row);
        valid = pos64 < uint64_t(a.vocab);
        idx = uint(pos64);
    }

    Ptr<float> dst = a.y + uint64_t(row) * uint64_t(a.out_stride);
    if (!valid) {
        if (base + 4u <= a.dim && is_ptr_aligned16(dst)) {
            store_a16(Ptr<float4>(dst) + uint64_t(t), float4(0.0f));
        } else {
            uint end = min(base + 4u, a.dim);
            for (uint c = base; c < end; ++c) dst[c] = 0.0f;
        }
        return;
    }

    Ptr<float, Access.Read> src = a.table + uint64_t(idx) * uint64_t(a.table_stride);
    if (base + 4u <= a.dim && is_ptr_aligned16(src) && is_ptr_aligned16(dst)) {
        float4 v = load_a16(Ptr<float4, Access.Read>(src) + uint64_t(t));
        store_a16(Ptr<float4>(dst) + uint64_t(t), v);
    } else {
        uint end = min(base + 4u, a.dim);
        for (uint c = base; c < end; ++c) dst[c] = src[c];
    }
}

// ============================================================================
// 2. Copy Last Active Row ([rows, dim] -> [1, dim])
// ============================================================================
struct CopyLastRowArgs {
    Ptr<float, Access.Read> x;
    Ptr<float> y;
    uint src_row;
    uint dim;
    uint x_stride;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_copy_last_row_f32(uniform CopyLastRowArgs a, uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    uint base = t << 2u;
    if (base >= a.dim) return;

    Ptr<float, Access.Read> src = a.x + uint64_t(a.src_row) * uint64_t(a.x_stride);
    if (base + 4u <= a.dim && is_ptr_aligned16(src) && is_ptr_aligned16(a.y)) {
        float4 v = load_a16(Ptr<float4, Access.Read>(src) + uint64_t(t));
        store_a16(Ptr<float4>(a.y) + uint64_t(t), v);
    } else {
        uint end = min(base + 4u, a.dim);
        for (uint c = base; c < end; ++c) a.y[c] = src[c];
    }
}

// ============================================================================
// 3. KV Cache Append ([rows, n_kv_heads * head_dim] -> [n_kv_heads, max_ctx, head_dim])
// ============================================================================
struct KvAppendArgs {
    Ptr<float, Access.Read> k_in;
    Ptr<float, Access.Read> v_in;
    Ptr<float> k_cache;
    Ptr<float> v_cache;
    uint rows;
    uint n_kv_heads;
    uint head_dim;
    uint max_ctx;
    uint kv_pos;
    uint k_stride;
    uint v_stride;
    uint same_kv_cache;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_kv_append_f32(uniform KvAppendArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint idx = gid.x * 256u + gtid.x;
    uint t = gid.y;
    uint kv_dim = a.n_kv_heads * a.head_dim;
    if (t >= a.rows || idx >= kv_dim) return;

    uint64_t pos64 = uint64_t(a.kv_pos) + uint64_t(t);
    if (pos64 >= uint64_t(a.max_ctx)) return;

    uint kvh = idx / a.head_dim;
    uint d = idx - kvh * a.head_dim;
    uint64_t dst_off =
        (uint64_t(kvh) * uint64_t(a.max_ctx) + pos64) * uint64_t(a.head_dim) + uint64_t(d);

    a.k_cache[dst_off] = a.k_in[uint64_t(t) * uint64_t(a.k_stride) + uint64_t(idx)];
    if (a.same_kv_cache == 0u) {
        a.v_cache[dst_off] = a.v_in[uint64_t(t) * uint64_t(a.v_stride) + uint64_t(idx)];
    }
}

// ============================================================================
// 4 & 5. Online-Softmax Multi-Head / GQA Attention (Decode & Prefill, 0 B scratch)
// ============================================================================
struct AttentionArgs {
    Ptr<float, Access.Read> q;
    Ptr<float, Access.Read> k_cache;
    Ptr<float, Access.Read> v_cache;
    Ptr<float> y;
    uint rows;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint max_ctx;
    uint kv_pos;
    uint q_stride;
    uint o_stride;
    uint sliding_window;
    uint causal;
    float scale;
    uint pad;
};

groupshared float s_q[256];
groupshared float s_red[256];
groupshared float s_m[8];
groupshared float s_l[8];

// Subgroup-size-agnostic sum + broadcast within each 32-thread partition.
// Zero barriers / zero smem traffic on Wave32; portable tree reduction on all other wave sizes.
// Caller MUST invoke uniformly across the entire workgroup (never inside divergent branches).
float reduce_sum_part32(float val, uint tid, uint part, uint lane) {
    if (WaveGetLaneCount() == 32u) {
        return WaveActiveSum(val);
    }
    s_red[tid] = val;
    GroupMemoryBarrierWithGroupSync();
    if (lane < 16u) s_red[tid] += s_red[tid + 16u];
    GroupMemoryBarrierWithGroupSync();
    if (lane < 8u)  s_red[tid] += s_red[tid + 8u];
    GroupMemoryBarrierWithGroupSync();
    if (lane < 4u)  s_red[tid] += s_red[tid + 4u];
    GroupMemoryBarrierWithGroupSync();
    if (lane < 2u)  s_red[tid] += s_red[tid + 2u];
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) s_red[tid] += s_red[tid + 1u];
    GroupMemoryBarrierWithGroupSync();
    float res = s_red[part << 5u];
    GroupMemoryBarrierWithGroupSync();
    return res;
}

void attention_online_impl<let P : uint>(AttentionArgs a, uint q_row, uint h, uint tid) {
    if (q_row >= a.rows || h >= a.n_heads) return;

    static const float kNegInfInit = -3.4028234e38f;
    static const uint kWgSize = P << 5u;

    uint64_t kv_end64 = (a.causal != 0u)
                            ? (uint64_t(a.kv_pos) + uint64_t(q_row) + 1ull)
                            : (uint64_t(a.kv_pos) + uint64_t(a.rows));
    uint kv_end = uint(min(kv_end64, uint64_t(a.max_ctx)));
    uint kv_start = (a.sliding_window > 0u && kv_end > a.sliding_window)
                        ? (kv_end - a.sliding_window)
                        : 0u;
    uint kv_len = (kv_end > kv_start) ? (kv_end - kv_start) : 0u;

    Ptr<float> o_ptr =
        a.y + uint64_t(q_row) * uint64_t(a.o_stride) + uint64_t(h) * uint64_t(a.head_dim);
    if (kv_len == 0u) {
        for (uint d = tid; d < a.head_dim; d += kWgSize) {
            o_ptr[d] = 0.0f;
        }
        return;
    }

    Ptr<float, Access.Read> q_ptr =
        a.q + uint64_t(q_row) * uint64_t(a.q_stride) + uint64_t(h) * uint64_t(a.head_dim);
    for (uint d = tid; d < a.head_dim; d += kWgSize) {
        s_q[d] = q_ptr[d];
    }
    GroupMemoryBarrierWithGroupSync();

    uint part = tid >> 5u;
    uint lane = tid & 31u;

    float q_reg[8];
    float acc[8];
    [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
        uint d = lane + (j << 5u);
        q_reg[j] = (d < a.head_dim) ? s_q[d] : 0.0f;
        acc[j] = 0.0f;
    }

    uint group_size = a.n_heads / a.n_kv_heads;
    uint kvh = h / group_size;
    uint64_t kv_head_base = uint64_t(kvh) * uint64_t(a.max_ctx) * uint64_t(a.head_dim);
    Ptr<float, Access.Read> k_head = a.k_cache + kv_head_base;
    Ptr<float, Access.Read> v_head = a.v_cache + kv_head_base;

    float m_p = kNegInfInit;
    float l_p = 0.0f;
    uint n_iters = (kv_len - 1u) / P + 1u;

    for (uint it = 0u; it < n_iters; ++it) {
        uint64_t k_rel = uint64_t(it) * uint64_t(P) + uint64_t(part);
        bool valid = (k_rel < uint64_t(kv_len));
        uint k_idx = kv_start + uint(k_rel);

        float partial_dot = 0.0f;
        if (valid) {
            Ptr<float, Access.Read> k_row = k_head + uint64_t(k_idx) * uint64_t(a.head_dim);
            [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
                if ((j << 5u) < a.head_dim) {
                    uint d = lane + (j << 5u);
                    if (d < a.head_dim) {
                        partial_dot = fma(q_reg[j], k_row[d], partial_dot);
                    }
                }
            }
        }

        float score = reduce_sum_part32(partial_dot, tid, part, lane) * a.scale;

        if (valid) {
            float alpha = 1.0f;
            float p_w = 1.0f;
            if (l_p == 0.0f) {
                alpha = 0.0f;
                m_p = score;
                l_p = 1.0f;
            } else if (score > m_p) {
                alpha = exp(m_p - score);
                l_p = l_p * alpha + 1.0f;
                m_p = score;
            } else {
                p_w = exp(score - m_p);
                l_p += p_w;
            }

            Ptr<float, Access.Read> v_row = v_head + uint64_t(k_idx) * uint64_t(a.head_dim);
            [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
                if ((j << 5u) < a.head_dim) {
                    uint d = lane + (j << 5u);
                    if (d < a.head_dim) {
                        acc[j] = fma(acc[j], alpha, p_w * v_row[d]);
                    }
                }
            }
        }
    }

    if (lane == 0u) {
        s_m[part] = m_p;
        s_l[part] = l_p;
    }
    GroupMemoryBarrierWithGroupSync();

    // Partition 0 always has >= 1 token because kv_len >= 1.
    float m_all = s_m[0];
    [ForceUnroll] for (uint p = 1u; p < P; ++p) {
        if (s_l[p] > 0.0f) {
            m_all = max(m_all, s_m[p]);
        }
    }

    float l_all = 0.0f;
    [ForceUnroll] for (uint p = 0u; p < P; ++p) {
        if (s_l[p] > 0.0f) {
            l_all += s_l[p] * exp(s_m[p] - m_all);
        }
    }

    float w_p = (l_p > 0.0f) ? exp(m_p - m_all) : 0.0f;
    float inv_l = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;

    [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
        if ((j << 5u) < a.head_dim) {
            uint d = lane + (j << 5u);
            s_red[tid] = (d < a.head_dim) ? (acc[j] * w_p) : 0.0f;
            GroupMemoryBarrierWithGroupSync();
            if (part == 0u && d < a.head_dim) {
                float sum = 0.0f;
                [ForceUnroll] for (uint p = 0u; p < P; ++p) {
                    sum += s_red[(p << 5u) + lane];
                }
                o_ptr[d] = sum * inv_l;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_decode_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    attention_online_impl<8u>(a, 0u, gid.x, gtid.x);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_attention_prefill_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    attention_online_impl<4u>(a, gid.x, gid.y, gtid.x);
}

// ============================================================================
// 6. Latent Cache Append ([rows, d_lat] -> [max_ctx, d_lat])
// ============================================================================
struct LatentAppendArgs {
    Ptr<float, Access.Read> h_in;
    Ptr<float> h_cache;
    uint rows;
    uint d_lat;
    uint kv_pos;
    uint h_stride;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_latent_append_f32(uniform LatentAppendArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint t = gid.x * 256u + gtid.x;
    uint base = t << 2u;
    uint row = gid.y;
    if (row >= a.rows || base >= a.d_lat) return;

    Ptr<float, Access.Read> src = a.h_in + uint64_t(row) * uint64_t(a.h_stride);
    Ptr<float> dst = a.h_cache + (uint64_t(a.kv_pos) + uint64_t(row)) * uint64_t(a.d_lat);
    if (a.d_lat - base >= 4u && is_ptr_aligned16(src) && is_ptr_aligned16(dst)) {
        float4 v = load_a16(Ptr<float4, Access.Read>(src) + uint64_t(t));
        store_a16(Ptr<float4>(dst) + uint64_t(t), v);
    } else {
        uint end = (a.d_lat - base < 4u) ? a.d_lat : (base + 4u);
        for (uint c = base; c < end; ++c) {
            dst[c] = src[c];
        }
    }
}

// ============================================================================
// 7. Latent Attention with Exact Weight Absorption (d_lat <= 2048, 0 B scratch)
// ============================================================================
struct LatentAttentionArgs {
    Ptr<float, Access.Read> q;
    Ptr<float, Access.Read> wk;
    Ptr<float, Access.Read> wv;
    Ptr<float, Access.Read> bv;
    Ptr<float, Access.Read> h_cache;
    Ptr<float> y;
    uint rows;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint d_lat;
    uint max_ctx;
    uint kv_pos;
    uint q_stride;
    uint wk_stride;
    uint wv_stride;
    uint o_stride;
    uint sliding_window;
    float scale;
    uint flags;
};

// Loop-safe 4-wide workgroup sum reduction reusing s_red[256] (4 x 64 waves max):
// 2 barriers per 4 items (0.5 barriers/item, 6x fewer barriers than scalar 3-barrier loop reduction).
void wg_reduce_sum4_loop(float v0, float v1, float v2, float v3, uint tid,
                         out float r0, out float r1, out float r2, out float r3) {
    float w0 = WaveActiveSum(v0);
    float w1 = WaveActiveSum(v1);
    float w2 = WaveActiveSum(v2);
    float w3 = WaveActiveSum(v3);

    uint lane_cnt = WaveGetLaneCount();
    if (lane_cnt >= 256u) {
        r0 = w0;
        r1 = w1;
        r2 = w2;
        r3 = w3;
        return;
    }

    uint nwaves = 256u / lane_cnt;
    uint wid = tid / lane_cnt;
    if (WaveIsFirstLane()) {
        s_red[wid] = w0;
        s_red[64u + wid] = w1;
        s_red[128u + wid] = w2;
        s_red[192u + wid] = w3;
    }
    GroupMemoryBarrierWithGroupSync();

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint w = 0u; w < nwaves; ++w) {
        a0 += s_red[w];
        a1 += s_red[64u + w];
        a2 += s_red[128u + w];
        a3 += s_red[192u + w];
    }
    GroupMemoryBarrierWithGroupSync();
    r0 = a0;
    r1 = a1;
    r2 = a2;
    r3 = a3;
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_latent_attention_f32(uniform LatentAttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint q_row = gid.x;
    uint h = gid.y;
    uint tid = gtid.x;
    if (q_row >= a.rows || h >= a.n_heads) return;

    uint64_t kv_end64 = ((a.flags & 1u) != 0u)
                            ? (uint64_t(a.kv_pos) + uint64_t(q_row) + 1ull)
                            : (uint64_t(a.kv_pos) + uint64_t(a.rows));
    uint kv_end = uint(min(kv_end64, uint64_t(a.max_ctx)));
    uint kv_start = (a.sliding_window > 0u && kv_end > a.sliding_window)
                        ? (kv_end - a.sliding_window)
                        : 0u;
    uint kv_len = (kv_end > kv_start) ? (kv_end - kv_start) : 0u;

    Ptr<float> o_ptr =
        a.y + uint64_t(q_row) * uint64_t(a.o_stride) + uint64_t(h) * uint64_t(a.head_dim);
    if (kv_len == 0u) {
        if (tid < a.head_dim) {
            o_ptr[tid] = 0.0f;
        }
        return;
    }

    Ptr<float, Access.Read> q_ptr =
        a.q + uint64_t(q_row) * uint64_t(a.q_stride) + uint64_t(h) * uint64_t(a.head_dim);
    if (tid < a.head_dim) {
        s_q[tid] = q_ptr[tid];
    }
    GroupMemoryBarrierWithGroupSync();

    uint group_size = a.n_heads / a.n_kv_heads;
    uint kvh = h / group_size;
    uint64_t kv_row_base = uint64_t(kvh) * uint64_t(a.head_dim);

    // Step A: Weight Absorption (qt = q_h * W_{K,kvh}), pure SSA registers across u = 0..7.
    float qt[8];
    float hbar[8];
    [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
        qt[u] = 0.0f;
        hbar[u] = 0.0f;
    }

    uint64_t wk_s64 = uint64_t(a.wk_stride);
    Ptr<float, Access.Read> wk_head = a.wk + kv_row_base * wk_s64;
    for (uint r = 0u; r < a.head_dim; ++r) {
        float qr = s_q[r];
        Ptr<float, Access.Read> wk_r = wk_head + uint64_t(r) * wk_s64;
        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_lat) {
                uint d = tid + (u << 8u);
                if (d < a.d_lat) {
                    qt[u] = fma(qr, wk_r[d], qt[u]);
                }
            }
        }
    }

    // Step B: Online Softmax over H_cache[kv_start .. kv_end-1], 4-key tiled (2 barriers per 4 keys).
    static const float kNegInfInit = -3.4028234e38f;
    float m = kNegInfInit;
    float l = 0.0f;
    uint64_t d_lat64 = uint64_t(a.d_lat);

    for (uint j = kv_start; j < kv_end; j += 4u) {
        uint cnt = min(4u, kv_end - j);
        Ptr<float, Access.Read> hj0 = a.h_cache + uint64_t(j) * d_lat64;
        Ptr<float, Access.Read> hj1 = (cnt > 1u) ? (hj0 + d_lat64) : hj0;
        Ptr<float, Access.Read> hj2 = (cnt > 2u) ? (hj1 + d_lat64) : hj0;
        Ptr<float, Access.Read> hj3 = (cnt > 3u) ? (hj2 + d_lat64) : hj0;

        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_lat) {
                uint d = tid + (u << 8u);
                if (d < a.d_lat) {
                    float qtu = qt[u];
                    d0 = fma(qtu, hj0[d], d0);
                    if (cnt > 1u) d1 = fma(qtu, hj1[d], d1);
                    if (cnt > 2u) d2 = fma(qtu, hj2[d], d2);
                    if (cnt > 3u) d3 = fma(qtu, hj3[d], d3);
                }
            }
        }

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        wg_reduce_sum4_loop(d0, d1, d2, d3, tid, s0, s1, s2, s3);
        s0 *= a.scale;
        s1 *= a.scale;
        s2 *= a.scale;
        s3 *= a.scale;

        float m_tile = s0;
        if (cnt > 1u) m_tile = max(m_tile, s1);
        if (cnt > 2u) m_tile = max(m_tile, s2);
        if (cnt > 3u) m_tile = max(m_tile, s3);

        float m_new = (l > 0.0f) ? max(m, m_tile) : m_tile;
        float alpha = (l > 0.0f) ? exp(m - m_new) : 0.0f;
        float p0 = exp(s0 - m_new);
        float p1 = (cnt > 1u) ? exp(s1 - m_new) : 0.0f;
        float p2 = (cnt > 2u) ? exp(s2 - m_new) : 0.0f;
        float p3 = (cnt > 3u) ? exp(s3 - m_new) : 0.0f;
        l = l * alpha + (p0 + p1 + p2 + p3);
        m = m_new;

        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_lat) {
                uint d = tid + (u << 8u);
                if (d < a.d_lat) {
                    float acc = fma(hbar[u], alpha, p0 * hj0[d]);
                    if (cnt > 1u) acc = fma(p1, hj1[d], acc);
                    if (cnt > 2u) acc = fma(p2, hj2[d], acc);
                    if (cnt > 3u) acc = fma(p3, hj3[d], acc);
                    hbar[u] = acc;
                }
            }
        }
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
        if ((u << 8u) < a.d_lat) {
            hbar[u] *= inv_l;
        }
    }

    // Step C: Output projection (O = hbar * W_{V,kvh}^T + b_V), 4-row tiled (2 barriers per 4 rows).
    uint64_t wv_s64 = uint64_t(a.wv_stride);
    Ptr<float, Access.Read> wv_head = a.wv + kv_row_base * wv_s64;
    for (uint r = 0u; r < a.head_dim; r += 4u) {
        uint cnt = min(4u, a.head_dim - r);
        Ptr<float, Access.Read> wv0 = wv_head + uint64_t(r) * wv_s64;
        Ptr<float, Access.Read> wv1 = (cnt > 1u) ? (wv0 + wv_s64) : wv0;
        Ptr<float, Access.Read> wv2 = (cnt > 2u) ? (wv1 + wv_s64) : wv0;
        Ptr<float, Access.Read> wv3 = (cnt > 3u) ? (wv2 + wv_s64) : wv0;

        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_lat) {
                uint d = tid + (u << 8u);
                if (d < a.d_lat) {
                    float hbu = hbar[u];
                    v0 = fma(hbu, wv0[d], v0);
                    if (cnt > 1u) v1 = fma(hbu, wv1[d], v1);
                    if (cnt > 2u) v2 = fma(hbu, wv2[d], v2);
                    if (cnt > 3u) v3 = fma(hbu, wv3[d], v3);
                }
            }
        }

        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        wg_reduce_sum4_loop(v0, v1, v2, v3, tid, o0, o1, o2, o3);
        if (tid == 0u) {
            bool has_bv = (a.flags & 2u) != 0u;
            o_ptr[r] = has_bv ? (o0 + a.bv[kv_row_base + uint64_t(r)]) : o0;
            if (cnt > 1u) {
                o_ptr[r + 1u] = has_bv ? (o1 + a.bv[kv_row_base + uint64_t(r + 1u)]) : o1;
            }
            if (cnt > 2u) {
                o_ptr[r + 2u] = has_bv ? (o2 + a.bv[kv_row_base + uint64_t(r + 2u)]) : o2;
            }
            if (cnt > 3u) {
                o_ptr[r + 3u] = has_bv ? (o3 + a.bv[kv_row_base + uint64_t(r + 3u)]) : o3;
            }
        }
    }
}
