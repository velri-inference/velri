// Shared CPU kernels for Embedding lookup, CopyLastRow, KV-cache append,
// Unified Online-Softmax Scaled Dot-Product Attention (MHA / GQA / MQA / wide latent / MLA,
// d_k & d_v up to 2048, compile-time IKeyIter generic), and Grouped Batched Linear.
module attention;

import numeric;
import activations;

public interface IKeyIter {
    uint count();
    uint key_index(uint i);
}

public struct DenseRangeIter : IKeyIter {
    public uint start;
    public uint len;

    public uint count() { return len; }
    public uint key_index(uint i) { return start + i; }
}

public void embedding_rows_cpu(
    Ptr<float, Access.Read> table, uint table_stride,
    Ptr<int32_t, Access.Read> ids, uint has_ids,
    uint vocab, uint pos_offset,
    Ptr<float> out, uint out_stride,
    uint r0, uint r1, uint dim)
{
    uint64_t ts64 = uint64_t(table_stride);
    uint64_t os64 = uint64_t(out_stride);
    uint64_t vocab64 = uint64_t(vocab);

    if (has_ids != 0u) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<float> dst = out + uint64_t(r) * os64;
            int32_t raw_id = ids[r];
            if (raw_id >= 0 && uint64_t(uint(raw_id)) < vocab64) {
                Ptr<float, Access.Read> src = table + uint64_t(uint(raw_id)) * ts64;
                for (uint c = 0u; c < dim; ++c) {
                    dst[c] = src[c];
                }
            } else {
                for (uint c = 0u; c < dim; ++c) {
                    dst[c] = 0.0f;
                }
            }
        }
    } else {
        for (uint r = r0; r < r1; ++r) {
            Ptr<float> dst = out + uint64_t(r) * os64;
            uint64_t idx64 = uint64_t(pos_offset) + uint64_t(r);
            if (idx64 < vocab64) {
                Ptr<float, Access.Read> src = table + idx64 * ts64;
                for (uint c = 0u; c < dim; ++c) {
                    dst[c] = src[c];
                }
            } else {
                for (uint c = 0u; c < dim; ++c) {
                    dst[c] = 0.0f;
                }
            }
        }
    }
}

public void copy_last_row_cpu(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<float> y, uint src_row, uint dim)
{
    Ptr<float, Access.Read> src = x + uint64_t(src_row) * uint64_t(x_stride);
    for (uint c = 0u; c < dim; ++c) {
        y[c] = src[c];
    }
}

public void kv_append_cpu(
    Ptr<float, Access.Read> k_in, uint k_stride,
    Ptr<float, Access.Read> v_in, uint v_stride,
    Ptr<float> k_cache, uint k_head_stride, uint k_row_stride,
    Ptr<float> v_cache, uint v_head_stride, uint v_row_stride,
    uint same_kv_cache, uint kvh0, uint kvh1,
    uint active_t, uint kv_pos, uint d_k, uint d_v)
{
    uint64_t ks64 = uint64_t(k_stride);
    uint64_t vs64 = uint64_t(v_stride);
    uint64_t khs64 = uint64_t(k_head_stride);
    uint64_t krs64 = uint64_t(k_row_stride);
    uint64_t vhs64 = uint64_t(v_head_stride);
    uint64_t vrs64 = uint64_t(v_row_stride);
    uint64_t dk64 = uint64_t(d_k);
    uint64_t dv64 = uint64_t(d_v);
    uint64_t kvp64 = uint64_t(kv_pos);

    for (uint kvh = kvh0; kvh < kvh1; ++kvh) {
        uint64_t k_head_base = uint64_t(kvh) * khs64;
        uint64_t k_in_head = uint64_t(kvh) * dk64;
        for (uint t = 0u; t < active_t; ++t) {
            Ptr<float, Access.Read> k_src = k_in + uint64_t(t) * ks64 + k_in_head;
            Ptr<float> k_dst = k_cache + k_head_base + (kvp64 + uint64_t(t)) * krs64;
            for (uint d = 0u; d < d_k; ++d) {
                k_dst[d] = k_src[d];
            }
        }
        if (same_kv_cache == 0u) {
            uint64_t v_head_base = uint64_t(kvh) * vhs64;
            uint64_t v_in_head = uint64_t(kvh) * dv64;
            for (uint t = 0u; t < active_t; ++t) {
                Ptr<float, Access.Read> v_src = v_in + uint64_t(t) * vs64 + v_in_head;
                Ptr<float> v_dst = v_cache + v_head_base + (kvp64 + uint64_t(t)) * vrs64;
                for (uint d = 0u; d < d_v; ++d) {
                    v_dst[d] = v_src[d];
                }
            }
        }
    }
}

void attention_online_core_cpu<let MAX_DK : uint, let MAX_DV : uint, TIter : IKeyIter>(
    Ptr<float, Access.Read> q_ptr,
    Ptr<float, Access.Read> k_head, uint64_t krs64,
    Ptr<float, Access.Read> v_head, uint64_t vrs64,
    Ptr<float> o_ptr,
    uint d_k, uint d_v, float scale, TIter iter)
{
    uint n_keys = iter.count();
    if (n_keys == 0u) {
        for (uint d = 0u; d < d_v; ++d) {
            o_ptr[d] = 0.0f;
        }
        return;
    }

    // Direct array indexing (never take &qv[0] as Ptr) so Slang DSE preserves stores.
    float qv[MAX_DK];
    float acc[MAX_DV];
    float scores[16];
    float probs[16];

    for (uint d = 0u; d < d_k; ++d) {
        qv[d] = q_ptr[d];
    }
    for (uint d = 0u; d < d_v; ++d) {
        acc[d] = 0.0f;
    }

    // Finite -ffast-math safe sentinel (never -INFINITY).
    float m = -3.4028234e38f;
    float l = 0.0f;

    // Block-16 tiled online softmax (B_k = 16): amortizes acc rescaling across 16 keys
    // and vectorizes fast_expf cleanly across 16 AVX-512 lanes.
    uint pos = 0u;
    while (pos < n_keys) {
        uint bk = min(16u, n_keys - pos);

        // 1. Compute up to 16 QK^T scores (4-way key-unrolled dot products reusing qv) & blk_max.
        float blk_max = -3.4028234e38f;
        uint kj = 0u;
        for (; kj + 4u <= bk; kj += 4u) {
            Ptr<float, Access.Read> k0 = k_head + uint64_t(iter.key_index(pos + kj + 0u)) * krs64;
            Ptr<float, Access.Read> k1 = k_head + uint64_t(iter.key_index(pos + kj + 1u)) * krs64;
            Ptr<float, Access.Read> k2 = k_head + uint64_t(iter.key_index(pos + kj + 2u)) * krs64;
            Ptr<float, Access.Read> k3 = k_head + uint64_t(iter.key_index(pos + kj + 3u)) * krs64;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (uint d = 0u; d < d_k; ++d) {
                float qd = qv[d];
                d0 += qd * k0[d];
                d1 += qd * k1[d];
                d2 += qd * k2[d];
                d3 += qd * k3[d];
            }
            float s0 = d0 * scale;
            float s1 = d1 * scale;
            float s2 = d2 * scale;
            float s3 = d3 * scale;
            scores[kj + 0u] = s0;
            scores[kj + 1u] = s1;
            scores[kj + 2u] = s2;
            scores[kj + 3u] = s3;
            float m01 = select(s0 > s1, s0, s1);
            float m23 = select(s2 > s3, s2, s3);
            float m0123 = select(m01 > m23, m01, m23);
            blk_max = select(m0123 > blk_max, m0123, blk_max);
        }
        for (; kj < bk; ++kj) {
            Ptr<float, Access.Read> k_row = k_head + uint64_t(iter.key_index(pos + kj)) * krs64;
            float dot = 0.0f;
            for (uint d = 0u; d < d_k; ++d) {
                dot += qv[d] * k_row[d];
            }
            float s = dot * scale;
            scores[kj] = s;
            blk_max = select(s > blk_max, s, blk_max);
        }

        // 2. Block-wide running maximum.
        float m_new = select(blk_max > m, blk_max, m);

        // 3. Rescale existing accumulator ONCE for the entire block of up to 16 keys.
        if (l > 0.0f && m_new > m) {
            float alpha = fast_expf(m - m_new);
            l *= alpha;
            for (uint d = 0u; d < d_v; ++d) {
                acc[d] *= alpha;
            }
        }

        // 4. Vectorizable fast_expf across 16 AVX-512 lanes + 4-way unrolled V accumulation.
        float blk_l = 0.0f;
        if (bk == 16u) {
            for (uint j = 0u; j < 16u; ++j) {
                float p = fast_expf(scores[j] - m_new);
                probs[j] = p;
                blk_l += p;
            }
        } else {
            for (uint j = 0u; j < bk; ++j) {
                float p = fast_expf(scores[j] - m_new);
                probs[j] = p;
                blk_l += p;
            }
        }
        l += blk_l;

        uint vj = 0u;
        for (; vj + 4u <= bk; vj += 4u) {
            float p0 = probs[vj + 0u];
            float p1 = probs[vj + 1u];
            float p2 = probs[vj + 2u];
            float p3 = probs[vj + 3u];
            Ptr<float, Access.Read> v0 = v_head + uint64_t(iter.key_index(pos + vj + 0u)) * vrs64;
            Ptr<float, Access.Read> v1 = v_head + uint64_t(iter.key_index(pos + vj + 1u)) * vrs64;
            Ptr<float, Access.Read> v2 = v_head + uint64_t(iter.key_index(pos + vj + 2u)) * vrs64;
            Ptr<float, Access.Read> v3 = v_head + uint64_t(iter.key_index(pos + vj + 3u)) * vrs64;
            for (uint d = 0u; d < d_v; ++d) {
                float a = acc[d];
                a += p0 * v0[d];
                a += p1 * v1[d];
                a += p2 * v2[d];
                a += p3 * v3[d];
                acc[d] = a;
            }
        }
        for (; vj < bk; ++vj) {
            float pj = probs[vj];
            Ptr<float, Access.Read> v_row = v_head + uint64_t(iter.key_index(pos + vj)) * vrs64;
            for (uint d = 0u; d < d_v; ++d) {
                acc[d] += pj * v_row[d];
            }
        }

        m = m_new;
        pos += bk;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint d = 0u; d < d_v; ++d) {
        o_ptr[d] = acc[d] * inv_l;
    }
}

public void attention_rows_cpu_impl<let MAX_DK : uint, let MAX_DV : uint>(
    Ptr<float, Access.Read> q, uint q_stride,
    Ptr<float, Access.Read> k_cache, uint k_head_stride, uint k_row_stride,
    Ptr<float, Access.Read> v_cache, uint v_head_stride, uint v_row_stride,
    Ptr<float> out, uint o_stride,
    uint w0, uint w1, uint active_t, uint n_heads, uint n_kv_heads,
    uint d_k, uint d_v, uint kv_pos, float scale,
    uint sliding_window, uint causal)
{
    if (d_k == 0u || d_k > MAX_DK || d_v == 0u || d_v > MAX_DV ||
        n_heads == 0u || n_kv_heads == 0u) {
        return;
    }

    uint heads_per_kv = n_heads / n_kv_heads;
    uint64_t qs64 = uint64_t(q_stride);
    uint64_t os64 = uint64_t(o_stride);
    uint64_t khs64 = uint64_t(k_head_stride);
    uint64_t krs64 = uint64_t(k_row_stride);
    uint64_t vhs64 = uint64_t(v_head_stride);
    uint64_t vrs64 = uint64_t(v_row_stride);
    uint64_t dk64 = uint64_t(d_k);
    uint64_t dv64 = uint64_t(d_v);

    for (uint w = w0; w < w1; ++w) {
        uint t = w / n_heads;
        uint h = w - t * n_heads;
        uint kvh = h / heads_per_kv;

        uint kv_end = (causal != 0u) ? (kv_pos + t + 1u) : (kv_pos + active_t);
        uint kv_start = (sliding_window > 0u && kv_end > sliding_window)
                            ? (kv_end - sliding_window)
                            : 0u;
        DenseRangeIter iter;
        iter.start = kv_start;
        iter.len = (kv_end > kv_start) ? (kv_end - kv_start) : 0u;

        Ptr<float, Access.Read> q_ptr = q + uint64_t(t) * qs64 + uint64_t(h) * dk64;
        Ptr<float, Access.Read> k_head = k_cache + uint64_t(kvh) * khs64;
        Ptr<float, Access.Read> v_head = v_cache + uint64_t(kvh) * vhs64;
        Ptr<float> o_ptr = out + uint64_t(t) * os64 + uint64_t(h) * dv64;

        attention_online_core_cpu<MAX_DK, MAX_DV, DenseRangeIter>(
            q_ptr, k_head, krs64, v_head, vrs64, o_ptr, d_k, d_v, scale, iter);
    }
}

public void batched_linear_rows_cpu(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<float, Access.Read> w, uint w_stride,
    Ptr<float, Access.Read> bias, uint has_bias,
    Ptr<float> y, uint y_stride,
    uint w0, uint w1, uint n_heads, uint n_groups,
    uint in_dim, uint out_dim, uint weight_transposed)
{
    if (n_heads == 0u || n_groups == 0u || in_dim == 0u || out_dim == 0u) {
        return;
    }

    uint heads_per_group = n_heads / n_groups;
    uint64_t xs64 = uint64_t(x_stride);
    uint64_t ws64 = uint64_t(w_stride);
    uint64_t ys64 = uint64_t(y_stride);
    uint64_t k64 = uint64_t(in_dim);
    uint64_t m64 = uint64_t(out_dim);
    uint out_dim4 = out_dim & ~3u;
    uint in_dim4 = in_dim & ~3u;

    for (uint work = w0; work < w1; ++work) {
        uint t = work / n_heads;
        uint h = work - t * n_heads;
        uint g = h / heads_per_group;

        Ptr<float, Access.Read> x_ptr = x + uint64_t(t) * xs64 + uint64_t(h) * k64;
        Ptr<float> y_ptr = y + uint64_t(t) * ys64 + uint64_t(h) * m64;
        uint64_t b_base = uint64_t(g) * m64;

        if (weight_transposed == 0u) {
            // Y = X @ W^T + bias, W is [G * out_dim, in_dim]
            Ptr<float, Access.Read> w_grp = w + (uint64_t(g) * m64) * ws64;
            uint m = 0u;
            for (; m < out_dim4; m += 4u) {
                Ptr<float, Access.Read> w0p = w_grp + uint64_t(m + 0u) * ws64;
                Ptr<float, Access.Read> w1p = w_grp + uint64_t(m + 1u) * ws64;
                Ptr<float, Access.Read> w2p = w_grp + uint64_t(m + 2u) * ws64;
                Ptr<float, Access.Read> w3p = w_grp + uint64_t(m + 3u) * ws64;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (uint k = 0u; k < in_dim; ++k) {
                    float xv = x_ptr[k];
                    a0 += xv * w0p[k];
                    a1 += xv * w1p[k];
                    a2 += xv * w2p[k];
                    a3 += xv * w3p[k];
                }
                if (has_bias != 0u) {
                    a0 += bias[b_base + uint64_t(m + 0u)];
                    a1 += bias[b_base + uint64_t(m + 1u)];
                    a2 += bias[b_base + uint64_t(m + 2u)];
                    a3 += bias[b_base + uint64_t(m + 3u)];
                }
                y_ptr[m + 0u] = a0;
                y_ptr[m + 1u] = a1;
                y_ptr[m + 2u] = a2;
                y_ptr[m + 3u] = a3;
            }
            for (; m < out_dim; ++m) {
                Ptr<float, Access.Read> wm = w_grp + uint64_t(m) * ws64;
                float acc = 0.0f;
                for (uint k = 0u; k < in_dim; ++k) {
                    acc += x_ptr[k] * wm[k];
                }
                if (has_bias != 0u) {
                    acc += bias[b_base + uint64_t(m)];
                }
                y_ptr[m] = acc;
            }
        } else {
            // Y = X @ W + bias, W is [G * in_dim, out_dim] (unit-stride AXPY across out_dim)
            if (has_bias != 0u) {
                for (uint m = 0u; m < out_dim; ++m) {
                    y_ptr[m] = bias[b_base + uint64_t(m)];
                }
            } else {
                for (uint m = 0u; m < out_dim; ++m) {
                    y_ptr[m] = 0.0f;
                }
            }
            Ptr<float, Access.Read> w_grp = w + (uint64_t(g) * k64) * ws64;
            uint k = 0u;
            for (; k < in_dim4; k += 4u) {
                float x0 = x_ptr[k + 0u];
                float x1 = x_ptr[k + 1u];
                float x2 = x_ptr[k + 2u];
                float x3 = x_ptr[k + 3u];
                Ptr<float, Access.Read> wk0 = w_grp + uint64_t(k + 0u) * ws64;
                Ptr<float, Access.Read> wk1 = w_grp + uint64_t(k + 1u) * ws64;
                Ptr<float, Access.Read> wk2 = w_grp + uint64_t(k + 2u) * ws64;
                Ptr<float, Access.Read> wk3 = w_grp + uint64_t(k + 3u) * ws64;
                for (uint m = 0u; m < out_dim; ++m) {
                    float a = y_ptr[m];
                    a += x0 * wk0[m];
                    a += x1 * wk1[m];
                    a += x2 * wk2[m];
                    a += x3 * wk3[m];
                    y_ptr[m] = a;
                }
            }
            for (; k < in_dim; ++k) {
                float xk = x_ptr[k];
                Ptr<float, Access.Read> wkp = w_grp + uint64_t(k) * ws64;
                for (uint m = 0u; m < out_dim; ++m) {
                    y_ptr[m] += xk * wkp[m];
                }
            }
        }
    }
}
