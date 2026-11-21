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
