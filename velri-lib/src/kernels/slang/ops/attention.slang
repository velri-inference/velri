// Shared CPU kernels for Embedding lookup, CopyLastRow, KV-cache append,
// Online-Softmax Scaled Dot-Product Attention (MHA / GQA / MQA, causal & sliding window),
// and Latent-Cache Weight-Absorbed Attention (MLA / Level B).
module attention;

import numeric;
import activations;

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
    Ptr<float> k_cache, Ptr<float> v_cache,
    uint same_kv_cache, uint kvh0, uint kvh1,
    uint active_t, uint kv_pos, uint max_ctx, uint head_dim)
{
    uint64_t ks64 = uint64_t(k_stride);
    uint64_t vs64 = uint64_t(v_stride);
    uint64_t mc64 = uint64_t(max_ctx);
    uint64_t hd64 = uint64_t(head_dim);
    uint64_t kvp64 = uint64_t(kv_pos);

    if (same_kv_cache == 0u) {
        for (uint kvh = kvh0; kvh < kvh1; ++kvh) {
            uint64_t cache_head_base = uint64_t(kvh) * mc64;
            uint64_t in_head_off = uint64_t(kvh) * hd64;
            for (uint t = 0u; t < active_t; ++t) {
                uint64_t dst_off = (cache_head_base + kvp64 + uint64_t(t)) * hd64;
                Ptr<float, Access.Read> k_src = k_in + uint64_t(t) * ks64 + in_head_off;
                Ptr<float, Access.Read> v_src = v_in + uint64_t(t) * vs64 + in_head_off;
                Ptr<float> k_dst = k_cache + dst_off;
                Ptr<float> v_dst = v_cache + dst_off;
                for (uint d = 0u; d < head_dim; ++d) {
                    k_dst[d] = k_src[d];
                    v_dst[d] = v_src[d];
                }
            }
        }
    } else {
        for (uint kvh = kvh0; kvh < kvh1; ++kvh) {
            uint64_t cache_head_base = uint64_t(kvh) * mc64;
            uint64_t in_head_off = uint64_t(kvh) * hd64;
            for (uint t = 0u; t < active_t; ++t) {
                uint64_t dst_off = (cache_head_base + kvp64 + uint64_t(t)) * hd64;
                Ptr<float, Access.Read> k_src = k_in + uint64_t(t) * ks64 + in_head_off;
                Ptr<float> k_dst = k_cache + dst_off;
                for (uint d = 0u; d < head_dim; ++d) {
                    k_dst[d] = k_src[d];
                }
            }
        }
    }
}

public void attention_rows_cpu(
    Ptr<float, Access.Read> q, uint q_stride,
    Ptr<float, Access.Read> k_cache, Ptr<float, Access.Read> v_cache,
    Ptr<float> out, uint o_stride,
    uint w0, uint w1, uint active_t, uint n_heads, uint n_kv_heads,
    uint head_dim, uint kv_pos, uint max_ctx, float scale,
    uint sliding_window, uint causal)
{
    if (head_dim == 0u || head_dim > 256u || n_heads == 0u || n_kv_heads == 0u) {
        return;
    }

    uint heads_per_kv = n_heads / n_kv_heads;
    uint64_t qs64 = uint64_t(q_stride);
    uint64_t os64 = uint64_t(o_stride);
    uint64_t mc64 = uint64_t(max_ctx);
    uint64_t hd64 = uint64_t(head_dim);

    float qv[256];
    float acc[256];
    float scores[16];
    float probs[16];

    for (uint w = w0; w < w1; ++w) {
        uint t = w / n_heads;
        uint h = w - t * n_heads;
        uint kvh = h / heads_per_kv;

        uint kv_end = (causal != 0u) ? (kv_pos + t + 1u) : (kv_pos + active_t);
        uint kv_start = (sliding_window > 0u && kv_end > sliding_window)
                            ? (kv_end - sliding_window)
                            : 0u;

        Ptr<float, Access.Read> q_ptr = q + uint64_t(t) * qs64 + uint64_t(h) * hd64;
        Ptr<float> o_ptr = out + uint64_t(t) * os64 + uint64_t(h) * hd64;

        if (kv_start >= kv_end) {
            for (uint d = 0u; d < head_dim; ++d) {
                o_ptr[d] = 0.0f;
            }
            continue;
        }

        for (uint d = 0u; d < head_dim; ++d) {
            qv[d] = q_ptr[d];
            acc[d] = 0.0f;
        }

        uint64_t cache_head_base = uint64_t(kvh) * mc64;
        // Finite -ffast-math safe sentinel (never -INFINITY).
        float m = -3.4028234e38f;
        float l = 0.0f;

        // Block-16 tiled online softmax (B_k = 16): amortizes acc rescaling across 16 keys
        // and vectorizes fast_expf cleanly across 16 AVX-512 lanes.
        uint k_idx = kv_start;
        while (k_idx < kv_end) {
            uint bk = min(16u, kv_end - k_idx);
            uint64_t blk_off = (cache_head_base + uint64_t(k_idx)) * hd64;
            Ptr<float, Access.Read> k_blk = k_cache + blk_off;
            Ptr<float, Access.Read> v_blk = v_cache + blk_off;

            // 1. Compute up to 16 QK^T scores (4-way key-unrolled dot products reusing qv) & blk_max.
            float blk_max = -3.4028234e38f;
            uint kj = 0u;
            for (; kj + 4u <= bk; kj += 4u) {
                Ptr<float, Access.Read> k0 = k_blk + uint64_t(kj + 0u) * hd64;
                Ptr<float, Access.Read> k1 = k_blk + uint64_t(kj + 1u) * hd64;
                Ptr<float, Access.Read> k2 = k_blk + uint64_t(kj + 2u) * hd64;
                Ptr<float, Access.Read> k3 = k_blk + uint64_t(kj + 3u) * hd64;
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                for (uint d = 0u; d < head_dim; ++d) {
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
                Ptr<float, Access.Read> k_row = k_blk + uint64_t(kj) * hd64;
                float dot = 0.0f;
                for (uint d = 0u; d < head_dim; ++d) {
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
                for (uint d = 0u; d < head_dim; ++d) {
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
                Ptr<float, Access.Read> v0 = v_blk + uint64_t(vj + 0u) * hd64;
                Ptr<float, Access.Read> v1 = v_blk + uint64_t(vj + 1u) * hd64;
                Ptr<float, Access.Read> v2 = v_blk + uint64_t(vj + 2u) * hd64;
                Ptr<float, Access.Read> v3 = v_blk + uint64_t(vj + 3u) * hd64;
                for (uint d = 0u; d < head_dim; ++d) {
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
                Ptr<float, Access.Read> v_row = v_blk + uint64_t(vj) * hd64;
                for (uint d = 0u; d < head_dim; ++d) {
                    acc[d] += pj * v_row[d];
                }
            }

            // 5. Commit running max and advance to next tile.
            m = m_new;
            k_idx += bk;
        }

        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (uint d = 0u; d < head_dim; ++d) {
            o_ptr[d] = acc[d] * inv_l;
        }
    }
}

public void latent_append_cpu(
    Ptr<float, Access.Read> h_in, uint h_stride,
    Ptr<float> h_cache,
    uint t0, uint t1, uint kv_pos, uint d_lat)
{
    uint64_t hs64 = uint64_t(h_stride);
    uint64_t d64 = uint64_t(d_lat);
    uint64_t kvp64 = uint64_t(kv_pos);

    for (uint t = t0; t < t1; ++t) {
        Ptr<float, Access.Read> src = h_in + uint64_t(t) * hs64;
        Ptr<float> dst = h_cache + (kvp64 + uint64_t(t)) * d64;
        for (uint d = 0u; d < d_lat; ++d) {
            dst[d] = src[d];
        }
    }
}

public void latent_attention_rows_cpu(
    Ptr<float, Access.Read> q, uint q_stride,
    Ptr<float, Access.Read> h_cache,
    Ptr<float, Access.Read> w_k, uint wk_stride,
    Ptr<float, Access.Read> w_v, uint wv_stride,
    Ptr<float, Access.Read> b_v, uint has_bv,
    Ptr<float> out, uint o_stride,
    uint w0, uint w1, uint active_t, uint n_heads, uint n_kv_heads,
    uint head_dim, uint d_lat, uint kv_pos, float scale,
    uint sliding_window, uint causal)
{
    if (head_dim == 0u || head_dim > 256u || d_lat == 0u || d_lat > 2048u ||
        n_heads == 0u || n_kv_heads == 0u) {
        return;
    }

    uint heads_per_kv = n_heads / n_kv_heads;
    uint64_t qs64 = uint64_t(q_stride);
    uint64_t os64 = uint64_t(o_stride);
    uint64_t wks64 = uint64_t(wk_stride);
    uint64_t wvs64 = uint64_t(wv_stride);
    uint64_t hd64 = uint64_t(head_dim);
    uint64_t d64 = uint64_t(d_lat);

    // 16 KiB stack buffers (0 B heap allocs). Always index directly (never take &qt[0]
    // as Ptr<float, Access.Read>) so Slang DSE never eliminates stores to local arrays.
    float qt[2048];
    float hbar[2048];
    float scores[16];
    float probs[16];

    for (uint w = w0; w < w1; ++w) {
        uint t = w / n_heads;
        uint h = w - t * n_heads;
        uint kvh = h / heads_per_kv;

        uint kv_end = (causal != 0u) ? (kv_pos + t + 1u) : (kv_pos + active_t);
        uint kv_start = (sliding_window > 0u && kv_end > sliding_window)
                            ? (kv_end - sliding_window)
                            : 0u;

        Ptr<float, Access.Read> q_ptr = q + uint64_t(t) * qs64 + uint64_t(h) * hd64;
        Ptr<float> o_ptr = out + uint64_t(t) * os64 + uint64_t(h) * hd64;

        if (kv_start >= kv_end) {
            for (uint r = 0u; r < head_dim; ++r) {
                o_ptr[r] = 0.0f;
            }
            continue;
        }

        // Step A: Weight Absorption (q_tilde = q_h * W_{K,kvh}).
        for (uint d = 0u; d < d_lat; ++d) {
            qt[d] = 0.0f;
            hbar[d] = 0.0f;
        }
        uint64_t kvh_row_base = uint64_t(kvh) * hd64;
        uint r_wk = 0u;
        for (; r_wk + 4u <= head_dim; r_wk += 4u) {
            float q0 = q_ptr[r_wk + 0u];
            float q1 = q_ptr[r_wk + 1u];
            float q2 = q_ptr[r_wk + 2u];
            float q3 = q_ptr[r_wk + 3u];
            Ptr<float, Access.Read> wk0 = w_k + (kvh_row_base + uint64_t(r_wk + 0u)) * wks64;
            Ptr<float, Access.Read> wk1 = w_k + (kvh_row_base + uint64_t(r_wk + 1u)) * wks64;
            Ptr<float, Access.Read> wk2 = w_k + (kvh_row_base + uint64_t(r_wk + 2u)) * wks64;
            Ptr<float, Access.Read> wk3 = w_k + (kvh_row_base + uint64_t(r_wk + 3u)) * wks64;
            for (uint d = 0u; d < d_lat; ++d) {
                float a = qt[d];
                a += q0 * wk0[d];
                a += q1 * wk1[d];
                a += q2 * wk2[d];
                a += q3 * wk3[d];
                qt[d] = a;
            }
        }
        for (; r_wk < head_dim; ++r_wk) {
            float qr = q_ptr[r_wk];
            Ptr<float, Access.Read> wk_r = w_k + (kvh_row_base + uint64_t(r_wk)) * wks64;
            for (uint d = 0u; d < d_lat; ++d) {
                qt[d] += qr * wk_r[d];
            }
        }

        // Step B: Block-16 tiled online softmax over H_cache[kv_start .. kv_end-1].
        // Finite -ffast-math safe sentinel (never -INFINITY).
        float m = -3.4028234e38f;
        float l = 0.0f;

        uint k_idx = kv_start;
        while (k_idx < kv_end) {
            uint bk = min(16u, kv_end - k_idx);
            Ptr<float, Access.Read> h_blk = h_cache + uint64_t(k_idx) * d64;

            // 1. Compute up to 16 scores = dot(qt, H_cache[k_idx + j]) * scale & blk_max.
            float blk_max = -3.4028234e38f;
            uint kj = 0u;
            for (; kj + 4u <= bk; kj += 4u) {
                Ptr<float, Access.Read> h0 = h_blk + uint64_t(kj + 0u) * d64;
                Ptr<float, Access.Read> h1 = h_blk + uint64_t(kj + 1u) * d64;
                Ptr<float, Access.Read> h2 = h_blk + uint64_t(kj + 2u) * d64;
                Ptr<float, Access.Read> h3 = h_blk + uint64_t(kj + 3u) * d64;
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                for (uint d = 0u; d < d_lat; ++d) {
                    float qd = qt[d];
                    d0 += qd * h0[d];
                    d1 += qd * h1[d];
                    d2 += qd * h2[d];
                    d3 += qd * h3[d];
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
                Ptr<float, Access.Read> h_row = h_blk + uint64_t(kj) * d64;
                float dot = 0.0f;
                for (uint d = 0u; d < d_lat; ++d) {
                    dot += qt[d] * h_row[d];
                }
                float s = dot * scale;
                scores[kj] = s;
                blk_max = select(s > blk_max, s, blk_max);
            }

            // 2. Block-wide running maximum.
            float m_new = select(blk_max > m, blk_max, m);

            // 3. Rescale existing hbar accumulator ONCE for the entire block of up to 16 keys.
            if (l > 0.0f && m_new > m) {
                float alpha = fast_expf(m - m_new);
                l *= alpha;
                for (uint d = 0u; d < d_lat; ++d) {
                    hbar[d] *= alpha;
                }
            }

            // 4. Vectorizable fast_expf across 16 AVX-512 lanes + 4-way unrolled H accumulation.
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

            uint hj = 0u;
            for (; hj + 4u <= bk; hj += 4u) {
                float p0 = probs[hj + 0u];
                float p1 = probs[hj + 1u];
                float p2 = probs[hj + 2u];
                float p3 = probs[hj + 3u];
                Ptr<float, Access.Read> h0 = h_blk + uint64_t(hj + 0u) * d64;
                Ptr<float, Access.Read> h1 = h_blk + uint64_t(hj + 1u) * d64;
                Ptr<float, Access.Read> h2 = h_blk + uint64_t(hj + 2u) * d64;
                Ptr<float, Access.Read> h3 = h_blk + uint64_t(hj + 3u) * d64;
                for (uint d = 0u; d < d_lat; ++d) {
                    float a = hbar[d];
                    a += p0 * h0[d];
                    a += p1 * h1[d];
                    a += p2 * h2[d];
                    a += p3 * h3[d];
                    hbar[d] = a;
                }
            }
            for (; hj < bk; ++hj) {
                float pj = probs[hj];
                Ptr<float, Access.Read> h_row = h_blk + uint64_t(hj) * d64;
                for (uint d = 0u; d < d_lat; ++d) {
                    hbar[d] += pj * h_row[d];
                }
            }

            // 5. Commit running max and advance to next tile.
            m = m_new;
            k_idx += bk;
        }

        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (uint d = 0u; d < d_lat; ++d) {
            hbar[d] *= inv_l;
        }

        // Step C: Output projection (O = hbar * W_{V,kvh}^T + b_V).
        uint r_wv = 0u;
        for (; r_wv + 4u <= head_dim; r_wv += 4u) {
            Ptr<float, Access.Read> wv0 = w_v + (kvh_row_base + uint64_t(r_wv + 0u)) * wvs64;
            Ptr<float, Access.Read> wv1 = w_v + (kvh_row_base + uint64_t(r_wv + 1u)) * wvs64;
            Ptr<float, Access.Read> wv2 = w_v + (kvh_row_base + uint64_t(r_wv + 2u)) * wvs64;
            Ptr<float, Access.Read> wv3 = w_v + (kvh_row_base + uint64_t(r_wv + 3u)) * wvs64;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (uint d = 0u; d < d_lat; ++d) {
                float hd_v = hbar[d];
                a0 += hd_v * wv0[d];
                a1 += hd_v * wv1[d];
                a2 += hd_v * wv2[d];
                a3 += hd_v * wv3[d];
            }
            if (has_bv != 0u) {
                a0 += b_v[kvh_row_base + uint64_t(r_wv + 0u)];
                a1 += b_v[kvh_row_base + uint64_t(r_wv + 1u)];
                a2 += b_v[kvh_row_base + uint64_t(r_wv + 2u)];
                a3 += b_v[kvh_row_base + uint64_t(r_wv + 3u)];
            }
            o_ptr[r_wv + 0u] = a0;
            o_ptr[r_wv + 1u] = a1;
            o_ptr[r_wv + 2u] = a2;
            o_ptr[r_wv + 3u] = a3;
        }
        for (; r_wv < head_dim; ++r_wv) {
            Ptr<float, Access.Read> wv_r = w_v + (kvh_row_base + uint64_t(r_wv)) * wvs64;
            float acc = 0.0f;
            for (uint d = 0u; d < d_lat; ++d) {
                acc += hbar[d] * wv_r[d];
            }
            if (has_bv != 0u) {
                acc += b_v[kvh_row_base + uint64_t(r_wv)];
            }
            o_ptr[r_wv] = acc;
        }
    }
}
