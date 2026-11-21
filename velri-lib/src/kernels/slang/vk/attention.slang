// Vulkan Embedding, CopyLastRow, KV-Cache Append, Unified Online-Softmax Attention (F32,
// d_k/d_v up to 2048, compile-time IKeyIter generic), and Grouped Batched Linear (F32).
// Subgroup-size agnostic (Wave4..Wave128), zero global score scratch, zero local-memory spills.
import numeric;
import reductions;
import vk_common;

interface IKeyIter {
    uint count();
    uint key_index(uint i);
}

struct DenseRangeIter : IKeyIter {
    uint start;
    uint len;

    uint count() { return len; }
    uint key_index(uint i) { return start + i; }
}

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
// 3. Unified KV Cache Append (supports d_k != d_v, strided caches, shared K==V)
// ============================================================================
struct KvAppendArgs {
    Ptr<float, Access.Read> k_in;
    Ptr<float, Access.Read> v_in;
    Ptr<float> k_cache;
    Ptr<float> v_cache;
    uint rows;
    uint n_kv_heads;
    uint d_k;
    uint d_v;
    uint max_ctx;
    uint kv_pos;
    uint k_stride;
    uint v_stride;
    uint k_head_stride;
    uint k_row_stride;
    uint v_head_stride;
    uint v_row_stride;
    uint same_kv_cache;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_kv_append_f32(uniform KvAppendArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint idx = gid.x * 256u + gtid.x;
    uint t = gid.y;
    if (t >= a.rows) return;

    uint64_t pos64 = uint64_t(a.kv_pos) + uint64_t(t);
    if (pos64 >= uint64_t(a.max_ctx)) return;

    uint k_dim = a.n_kv_heads * a.d_k;
    if (idx < k_dim) {
        uint kvh = idx / a.d_k;
        uint d = idx - kvh * a.d_k;
        uint64_t k_dst = uint64_t(kvh) * uint64_t(a.k_head_stride) +
                         pos64 * uint64_t(a.k_row_stride) + uint64_t(d);
        a.k_cache[k_dst] = a.k_in[uint64_t(t) * uint64_t(a.k_stride) + uint64_t(idx)];
    }

    if (a.same_kv_cache == 0u) {
        uint v_dim = a.n_kv_heads * a.d_v;
        if (idx < v_dim) {
            uint kvh = idx / a.d_v;
            uint d = idx - kvh * a.d_v;
            uint64_t v_dst = uint64_t(kvh) * uint64_t(a.v_head_stride) +
                             pos64 * uint64_t(a.v_row_stride) + uint64_t(d);
            a.v_cache[v_dst] = a.v_in[uint64_t(t) * uint64_t(a.v_stride) + uint64_t(idx)];
        }
    }
}

// ============================================================================
// 4, 5, 6. Unified Online-Softmax Attention (d_k, d_v <= 256 fast & <= 2048 wide)
// ============================================================================
struct AttentionArgs {
    Ptr<float, Access.Read> q;
    Ptr<float, Access.Read> k_cache;
    Ptr<float, Access.Read> v_cache;
    Ptr<float> y;
    uint rows;
    uint n_heads;
    uint n_kv_heads;
    uint d_k;
    uint d_v;
    uint max_ctx;
    uint kv_pos;
    uint q_stride;
    uint o_stride;
    uint k_head_stride;
    uint k_row_stride;
    uint v_head_stride;
    uint v_row_stride;
    uint sliding_window;
    uint causal;
    float scale;
};

groupshared float s_q[256];
groupshared float s_red[256];
groupshared float s_m[8];
groupshared float s_l[8];

DenseRangeIter make_dense_iter(AttentionArgs a, uint q_row) {
    uint64_t kv_end64 = (a.causal != 0u)
                            ? (uint64_t(a.kv_pos) + uint64_t(q_row) + 1ull)
                            : (uint64_t(a.kv_pos) + uint64_t(a.rows));
    uint kv_end = uint(min(kv_end64, uint64_t(a.max_ctx)));
    uint kv_start = (a.sliding_window > 0u && kv_end > a.sliding_window)
                        ? (kv_end - a.sliding_window)
                        : 0u;
    DenseRangeIter it;
    it.start = kv_start;
    it.len = (kv_end > kv_start) ? (kv_end - kv_start) : 0u;
    return it;
}

// Subgroup-size-agnostic sum + broadcast within each 32-thread partition.
// Zero barriers / zero smem traffic on Wave32; portable tree reduction on all other wave sizes.
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

// Loop-safe 4-wide workgroup sum reduction reusing s_red[256] (4 x 64 waves max):
// 2 barriers per 4 keys (0.5 barriers/key, 0 local-memory spills).
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

// Fast partition-parallel path for d_k <= 256 && d_v <= 256 (0 barriers per key on Wave32).
void attention_online_impl<let P : uint, TIter : IKeyIter>(
    AttentionArgs a, uint q_row, uint h, uint tid, TIter iter)
{
    if (q_row >= a.rows || h >= a.n_heads) return;

    static const float kNegInfInit = -3.4028234e38f;
    static const uint kWgSize = P << 5u;

    uint kv_len = iter.count();
    Ptr<float> o_ptr =
        a.y + uint64_t(q_row) * uint64_t(a.o_stride) + uint64_t(h) * uint64_t(a.d_v);
    if (kv_len == 0u) {
        for (uint d = tid; d < a.d_v; d += kWgSize) {
            o_ptr[d] = 0.0f;
        }
        return;
    }

    Ptr<float, Access.Read> q_ptr =
        a.q + uint64_t(q_row) * uint64_t(a.q_stride) + uint64_t(h) * uint64_t(a.d_k);
    for (uint d = tid; d < a.d_k; d += kWgSize) {
        s_q[d] = q_ptr[d];
    }
    GroupMemoryBarrierWithGroupSync();

    uint part = tid >> 5u;
    uint lane = tid & 31u;

    float q_reg[8];
    float acc[8];
    [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
        uint d = lane + (j << 5u);
        q_reg[j] = (d < a.d_k) ? s_q[d] : 0.0f;
        acc[j] = 0.0f;
    }

    uint group_size = a.n_heads / a.n_kv_heads;
    uint kvh = h / group_size;
    Ptr<float, Access.Read> k_head = a.k_cache + uint64_t(kvh) * uint64_t(a.k_head_stride);
    Ptr<float, Access.Read> v_head = a.v_cache + uint64_t(kvh) * uint64_t(a.v_head_stride);
    uint64_t krs64 = uint64_t(a.k_row_stride);
    uint64_t vrs64 = uint64_t(a.v_row_stride);

    float m_p = kNegInfInit;
    float l_p = 0.0f;
    uint n_iters = (kv_len - 1u) / P + 1u;

    for (uint it = 0u; it < n_iters; ++it) {
        uint64_t k_rel = uint64_t(it) * uint64_t(P) + uint64_t(part);
        bool valid = (k_rel < uint64_t(kv_len));
        uint k_idx = valid ? iter.key_index(uint(k_rel)) : 0u;

        float partial_dot = 0.0f;
        if (valid) {
            Ptr<float, Access.Read> k_row = k_head + uint64_t(k_idx) * krs64;
            [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
                if ((j << 5u) < a.d_k) {
                    uint d = lane + (j << 5u);
                    if (d < a.d_k) {
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

            Ptr<float, Access.Read> v_row = v_head + uint64_t(k_idx) * vrs64;
            [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
                if ((j << 5u) < a.d_v) {
                    uint d = lane + (j << 5u);
                    if (d < a.d_v) {
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
        if ((j << 5u) < a.d_v) {
            uint d = lane + (j << 5u);
            s_red[tid] = (d < a.d_v) ? (acc[j] * w_p) : 0.0f;
            GroupMemoryBarrierWithGroupSync();
            if (part == 0u && d < a.d_v) {
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

// Wide path for d_k > 256 || d_v > 256 (up to 2048): 4-key tiled online softmax, 0 spills.
void attention_wide_impl<TIter : IKeyIter>(
    AttentionArgs a, uint q_row, uint h, uint tid, TIter iter)
{
    if (q_row >= a.rows || h >= a.n_heads) return;

    uint kv_len = iter.count();
    Ptr<float> o_ptr =
        a.y + uint64_t(q_row) * uint64_t(a.o_stride) + uint64_t(h) * uint64_t(a.d_v);
    if (kv_len == 0u) {
        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_v) {
                uint d = tid + (u << 8u);
                if (d < a.d_v) {
                    o_ptr[d] = 0.0f;
                }
            }
        }
        return;
    }

    Ptr<float, Access.Read> q_ptr =
        a.q + uint64_t(q_row) * uint64_t(a.q_stride) + uint64_t(h) * uint64_t(a.d_k);

    float qt[8];
    float acc[8];
    [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
        uint d = tid + (u << 8u);
        qt[u] = (d < a.d_k) ? q_ptr[d] : 0.0f;
        acc[u] = 0.0f;
    }

    uint group_size = a.n_heads / a.n_kv_heads;
    uint kvh = h / group_size;
    Ptr<float, Access.Read> k_head = a.k_cache + uint64_t(kvh) * uint64_t(a.k_head_stride);
    Ptr<float, Access.Read> v_head = a.v_cache + uint64_t(kvh) * uint64_t(a.v_head_stride);
    uint64_t krs64 = uint64_t(a.k_row_stride);
    uint64_t vrs64 = uint64_t(a.v_row_stride);

    static const float kNegInfInit = -3.4028234e38f;
    float m = kNegInfInit;
    float l = 0.0f;

    uint j = 0u;
    while (j < kv_len) {
        uint cnt = min(4u, kv_len - j);
        uint idx0 = iter.key_index(j);
        uint idx1 = (cnt > 1u) ? iter.key_index(j + 1u) : idx0;
        uint idx2 = (cnt > 2u) ? iter.key_index(j + 2u) : idx0;
        uint idx3 = (cnt > 3u) ? iter.key_index(j + 3u) : idx0;

        Ptr<float, Access.Read> kj0 = k_head + uint64_t(idx0) * krs64;
        Ptr<float, Access.Read> kj1 = k_head + uint64_t(idx1) * krs64;
        Ptr<float, Access.Read> kj2 = k_head + uint64_t(idx2) * krs64;
        Ptr<float, Access.Read> kj3 = k_head + uint64_t(idx3) * krs64;

        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_k) {
                uint d = tid + (u << 8u);
                if (d < a.d_k) {
                    float qtu = qt[u];
                    d0 = fma(qtu, kj0[d], d0);
                    if (cnt > 1u) d1 = fma(qtu, kj1[d], d1);
                    if (cnt > 2u) d2 = fma(qtu, kj2[d], d2);
                    if (cnt > 3u) d3 = fma(qtu, kj3[d], d3);
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

        Ptr<float, Access.Read> vj0 = v_head + uint64_t(idx0) * vrs64;
        Ptr<float, Access.Read> vj1 = v_head + uint64_t(idx1) * vrs64;
        Ptr<float, Access.Read> vj2 = v_head + uint64_t(idx2) * vrs64;
        Ptr<float, Access.Read> vj3 = v_head + uint64_t(idx3) * vrs64;

        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_v) {
                uint d = tid + (u << 8u);
                if (d < a.d_v) {
                    float v_acc = fma(acc[u], alpha, p0 * vj0[d]);
                    if (cnt > 1u) v_acc = fma(p1, vj1[d], v_acc);
                    if (cnt > 2u) v_acc = fma(p2, vj2[d], v_acc);
                    if (cnt > 3u) v_acc = fma(p3, vj3[d], v_acc);
                    acc[u] = v_acc;
                }
            }
        }
        j += cnt;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
        if ((u << 8u) < a.d_v) {
            uint d = tid + (u << 8u);
            if (d < a.d_v) {
                o_ptr[d] = acc[u] * inv_l;
            }
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_decode_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    attention_online_impl<8u, DenseRangeIter>(a, 0u, gid.x, gtid.x, make_dense_iter(a, 0u));
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_attention_prefill_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    attention_online_impl<4u, DenseRangeIter>(a, gid.x, gid.y, gtid.x, make_dense_iter(a, gid.x));
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_wide_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    attention_wide_impl<DenseRangeIter>(a, gid.x, gid.y, gtid.x, make_dense_iter(a, gid.x));
}

// ============================================================================
// 7 & 8. Grouped Batched Linear (Y = X @ W^T + b and Y = X @ W + b, 0 spills)
// ============================================================================
struct BatchedLinearArgs {
    Ptr<float, Access.Read> x;
    Ptr<float, Access.Read> w;
    Ptr<float, Access.Read> bias;
    Ptr<float> y;
    uint rows;
    uint n_heads;
    uint n_groups;
    uint in_dim;
    uint out_dim;
    uint x_stride;
    uint w_stride;
    uint y_stride;
    uint flags;
    uint pad;
};

// Non-transposed weight layout W [G * out_dim, in_dim]: Y = X @ W^T + bias.
// Fixed 4 x 32-thread partitions per 128-thread WG (immune to varying Vulkan 1.3 subgroup sizes;
// 0 barriers / single WaveActiveSum on Wave32; vectorized float4 loads when 16B-aligned).
[shader("compute")] [numthreads(128, 1, 1)]
void vk_batched_linear_nt_f32(uniform BatchedLinearArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint t = gid.y;
    uint h = gid.z;
    if (t >= a.rows || h >= a.n_heads) return;

    uint tid = gtid.x;
    uint part = tid >> 5u;
    uint lane = tid & 31u;
    uint m = gid.x * 4u + part;
    bool valid = (m < a.out_dim);

    uint g = h / (a.n_heads / a.n_groups);
    float acc = 0.0f;
    if (valid) {
        Ptr<float, Access.Read> x_ptr =
            a.x + uint64_t(t) * uint64_t(a.x_stride) + uint64_t(h) * uint64_t(a.in_dim);
        Ptr<float, Access.Read> w_row =
            a.w + (uint64_t(g) * uint64_t(a.out_dim) + uint64_t(m)) * uint64_t(a.w_stride);

        uint k4 = a.in_dim >> 2u;
        if (k4 > 0u && is_ptr_aligned16(x_ptr) && is_ptr_aligned16(w_row)) {
            Ptr<float4, Access.Read> x4 = Ptr<float4, Access.Read>(x_ptr);
            Ptr<float4, Access.Read> w4 = Ptr<float4, Access.Read>(w_row);
            for (uint64_t i = uint64_t(lane); i < uint64_t(k4); i += 32ull) {
                acc += dot(load_a16(x4 + i), load_a16(w4 + i));
            }
            uint tail_k = (k4 << 2u) + lane;
            if (tail_k < a.in_dim) {
                acc = fma(x_ptr[tail_k], w_row[tail_k], acc);
            }
        } else {
            for (uint64_t k = uint64_t(lane); k < uint64_t(a.in_dim); k += 32ull) {
                acc = fma(x_ptr[k], w_row[k], acc);
            }
        }
    }

    float sum = reduce_sum_part32(acc, tid, part, lane);
    if (valid && lane == 0u) {
        if ((a.flags & 1u) != 0u) {
            sum += a.bias[uint64_t(g) * uint64_t(a.out_dim) + uint64_t(m)];
        }
        a.y[uint64_t(t) * uint64_t(a.y_stride) + uint64_t(h) * uint64_t(a.out_dim) + uint64_t(m)] =
            sum;
    }
}

// Transposed weight layout W [G * in_dim, out_dim]: Y = X @ W + bias.
// One thread per output element m; 100% coalesced reads of W and writes of Y, 0 barriers, 0 smem.
[shader("compute")] [numthreads(256, 1, 1)]
void vk_batched_linear_nn_f32(uniform BatchedLinearArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint m = gid.x * 256u + gtid.x;
    uint t = gid.y;
    uint h = gid.z;
    if (t >= a.rows || h >= a.n_heads || m >= a.out_dim) return;

    uint g = h / (a.n_heads / a.n_groups);
    Ptr<float, Access.Read> x_ptr =
        a.x + uint64_t(t) * uint64_t(a.x_stride) + uint64_t(h) * uint64_t(a.in_dim);
    uint64_t ws64 = uint64_t(a.w_stride);
    Ptr<float, Access.Read> w_grp = a.w + (uint64_t(g) * uint64_t(a.in_dim)) * ws64;

    float acc = ((a.flags & 1u) != 0u)
                    ? a.bias[uint64_t(g) * uint64_t(a.out_dim) + uint64_t(m)]
                    : 0.0f;
    uint in_dim4 = a.in_dim & ~3u;
    uint k = 0u;
    for (; k < in_dim4; k += 4u) {
        acc = fma(x_ptr[k + 0u], w_grp[uint64_t(k + 0u) * ws64 + uint64_t(m)], acc);
        acc = fma(x_ptr[k + 1u], w_grp[uint64_t(k + 1u) * ws64 + uint64_t(m)], acc);
        acc = fma(x_ptr[k + 2u], w_grp[uint64_t(k + 2u) * ws64 + uint64_t(m)], acc);
        acc = fma(x_ptr[k + 3u], w_grp[uint64_t(k + 3u) * ws64 + uint64_t(m)], acc);
    }
    for (; k < a.in_dim; ++k) {
        acc = fma(x_ptr[k], w_grp[uint64_t(k) * ws64 + uint64_t(m)], acc);
    }
    a.y[uint64_t(t) * uint64_t(a.y_stride) + uint64_t(h) * uint64_t(a.out_dim) + uint64_t(m)] = acc;
}
