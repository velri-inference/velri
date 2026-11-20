// Unified Vulkan compute entrypoints for Velri-lib Slang kernels.
// Uses Buffer Device Address (Ptr<T>) in push constants (zero descriptor sets)
// and reuses common/{numeric,activations,epilogue,reductions}.slang + ops/norm.slang.
import numeric;
import activations;
import epilogue;
import reductions;
import norm;

// True 128-bit PhysicalStorageBuffer (BDA) loads/stores: Slang scalar layout defaults to
// Aligned 4 (which forces 4x LDG.32/STG.32 on NVIDIA); explicit Aligned 16 emits LDG.128/STG.128!
bool is_ptr_aligned16<T, let A : Access>(Ptr<T, A> p) {
    uint64_t addr = spirv_asm { result : $$uint64_t = OpConvertPtrToU $p; };
    return (uint(addr) & 15u) == 0u;
}
T load_a16<T>(Ptr<T, Access.Read> p) {
    return spirv_asm { result : $$T = OpLoad $p Aligned 16; };
}
void store_a16<T>(Ptr<T> p, T v) {
    spirv_asm { OpStore $p $v Aligned 16; };
}

EpilogueParams make_vk_ep(Ptr<float, Access.Read> bias, Ptr<float, Access.Read> xs,
                          Ptr<float, Access.Read> ws, uint flags)
{
    EpilogueParams ep;
    ep.bias = bias;
    ep.x_scale = xs;
    ep.w_scale = ws;
    ep.has_bias = flags & 1u;
    ep.has_x_scale = flags & 2u;
    ep.has_w_scale = flags & 4u;
    return ep;
}

// ============================================================================
// Reusable 128-bit Vectorized Storage Helpers (shared by Norm, RoPE, Softmax, FWHT, Quant)
// ============================================================================
interface IVecIO128<T : IElement> {
    static const uint kElems; // 4 for F32, 8 for F16/BF16
    static void unpack(uint4 raw, out float4 lo, out float4 hi);
    static uint4 pack(float4 lo, float4 hi);
}

struct VecIOF32 : IVecIO128<ElemF32> {
    static const uint kElems = 4u;
    static void unpack(uint4 raw, out float4 lo, out float4 hi) {
        lo = asfloat(raw);
        hi = float4(0.0f);
    }
    static uint4 pack(float4 lo, float4 hi) { return asuint(lo); }
}

struct VecIOF16 : IVecIO128<ElemF16> {
    static const uint kElems = 8u;
    static void unpack(uint4 r, out float4 lo, out float4 hi) {
        lo = float4(
            float(bit_cast<half>(uint16_t(r.x & 0xFFFFu))),
            float(bit_cast<half>(uint16_t(r.x >> 16u))),
            float(bit_cast<half>(uint16_t(r.y & 0xFFFFu))),
            float(bit_cast<half>(uint16_t(r.y >> 16u))));
        hi = float4(
            float(bit_cast<half>(uint16_t(r.z & 0xFFFFu))),
            float(bit_cast<half>(uint16_t(r.z >> 16u))),
            float(bit_cast<half>(uint16_t(r.w & 0xFFFFu))),
            float(bit_cast<half>(uint16_t(r.w >> 16u))));
    }
    static uint4 pack(float4 lo, float4 hi) {
        uint x = uint(bit_cast<uint16_t>(half(lo.x))) | (uint(bit_cast<uint16_t>(half(lo.y))) << 16u);
        uint y = uint(bit_cast<uint16_t>(half(lo.z))) | (uint(bit_cast<uint16_t>(half(lo.w))) << 16u);
        uint z = uint(bit_cast<uint16_t>(half(hi.x))) | (uint(bit_cast<uint16_t>(half(hi.y))) << 16u);
        uint w = uint(bit_cast<uint16_t>(half(hi.z))) | (uint(bit_cast<uint16_t>(half(hi.w))) << 16u);
        return uint4(x, y, z, w);
    }
}

struct VecIOBF16 : IVecIO128<ElemBF16> {
    static const uint kElems = 8u;
    static void unpack(uint4 r, out float4 lo, out float4 hi) {
        lo = float4(
            asfloat((r.x & 0xFFFFu) << 16u), asfloat(r.x & 0xFFFF0000u),
            asfloat((r.y & 0xFFFFu) << 16u), asfloat(r.y & 0xFFFF0000u));
        hi = float4(
            asfloat((r.z & 0xFFFFu) << 16u), asfloat(r.z & 0xFFFF0000u),
            asfloat((r.w & 0xFFFFu) << 16u), asfloat(r.w & 0xFFFF0000u));
    }
    static uint4 pack(float4 lo, float4 hi) {
        uint x = uint(ElemBF16::fromF32(lo.x)) | (uint(ElemBF16::fromF32(lo.y)) << 16u);
        uint y = uint(ElemBF16::fromF32(lo.z)) | (uint(ElemBF16::fromF32(lo.w)) << 16u);
        uint z = uint(ElemBF16::fromF32(hi.x)) | (uint(ElemBF16::fromF32(hi.y)) << 16u);
        uint w = uint(ElemBF16::fromF32(hi.z)) | (uint(ElemBF16::fromF32(hi.w)) << 16u);
        return uint4(x, y, z, w);
    }
}

float4 apply_act4<A : IActivation>(float4 v) {
    return float4(A::apply(v.x), A::apply(v.y), A::apply(v.z), A::apply(v.w));
}

float4 apply_clamp4(float4 v, float lim) {
    if (lim <= 0.0f) return v;
    return clamp(v, float4(-lim), float4(lim));
}

// ============================================================================
// 1. Normalization: RMSNorm (F32/BF16/F16), FusedAddRMSNorm (F32), LayerNorm (F32)
//    - 128-bit LDG.128/STG.128 + single-pass register cache for dim <= 4096!
// ============================================================================
struct RmsNormArgs<TIO : IElement, TW : IElement> {
    Ptr<TIO.Storage, Access.Read> x;
    Ptr<TW.Storage, Access.Read> w;
    Ptr<TIO.Storage> y;
    uint rows;
    uint dim;
    uint x_stride;
    uint y_stride;
    float eps;
    uint has_weight;
    uint unit_offset;
};

void rmsnorm_gpu_impl<TIO : IElement, TW : IElement, VIO : IVecIO128<TIO>, VW : IVecIO128<TW>>(
    RmsNormArgs<TIO, TW> a, uint row, uint tid)
{
    if (row >= a.rows) return;
    Ptr<TIO.Storage, Access.Read> xr = a.x + row * a.x_stride;
    Ptr<TIO.Storage> yr = a.y + row * a.y_stride;

    uint vec_elems = VIO.kElems;
    bool aligned_vec = ((a.dim & (vec_elems - 1u)) == 0u) &&
                       is_ptr_aligned16(xr) && is_ptr_aligned16(yr) &&
                       ((a.has_weight == 0u) || is_ptr_aligned16(a.w));

    if (aligned_vec && a.dim <= 4096u) {
        // Single-pass register-cached path: reads x ONCE from VRAM!
        uint n_vec = a.dim / vec_elems;
        Ptr<uint4, Access.Read> xr4 = Ptr<uint4, Access.Read>(xr);
        Ptr<uint4, Access.Read> w4 = Ptr<uint4, Access.Read>(a.w);
        Ptr<uint4> yr4 = Ptr<uint4>(yr);

        float4 reg[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        float partial = 0.0f;
        if (VIO.kElems == 4u) {
            [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
                uint idx = s * 256u + tid;
                if (idx < n_vec) {
                    float4 lo, hi;
                    VIO::unpack(load_a16(xr4 + idx), lo, hi);
                    reg[s] = lo;
                    partial += dot(lo, lo);
                }
            }
        } else {
            [ForceUnroll] for (uint s = 0u; s < 2u; ++s) {
                uint idx = s * 256u + tid;
                if (idx < n_vec) {
                    float4 lo, hi;
                    VIO::unpack(load_a16(xr4 + idx), lo, hi);
                    reg[s * 2u + 0u] = lo;
                    reg[s * 2u + 1u] = hi;
                    partial += dot(lo, lo) + dot(hi, hi);
                }
            }
        }
        float total = wg_reduce_sum(partial, tid, 256u);
        float inv_rms = rsqrt(total / float(a.dim) + a.eps);
        float uoff = (a.unit_offset != 0u) ? 1.0f : 0.0f;

        if (VIO.kElems == 4u) {
            [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
                uint idx = s * 256u + tid;
                if (idx < n_vec) {
                    float4 wlo = float4(1.0f), whi = float4(1.0f);
                    if (a.has_weight != 0u) {
                        VW::unpack(load_a16(w4 + idx), wlo, whi);
                        wlo += float4(uoff);
                    }
                    store_a16(yr4 + idx, VIO::pack(reg[s] * (inv_rms * wlo), float4(0.0f)));
                }
            }
        } else {
            [ForceUnroll] for (uint s = 0u; s < 2u; ++s) {
                uint idx = s * 256u + tid;
                if (idx < n_vec) {
                    float4 wlo = float4(1.0f), whi = float4(1.0f);
                    if (a.has_weight != 0u) {
                        VW::unpack(load_a16(w4 + idx), wlo, whi);
                        wlo += float4(uoff);
                        whi += float4(uoff);
                    }
                    store_a16(yr4 + idx, VIO::pack(
                        reg[s * 2u + 0u] * (inv_rms * wlo),
                        reg[s * 2u + 1u] * (inv_rms * whi)));
                }
            }
        }
        return;
    }

    float partial = 0.0f;
    for (uint d = tid; d < a.dim; d += 256u) {
        float v = TIO::toF32(xr[d]);
        partial += v * v;
    }
    float total = wg_reduce_sum(partial, tid, 256u);
    float inv_rms = rsqrt(total / float(a.dim) + a.eps);

    for (uint d = tid; d < a.dim; d += 256u) {
        float w = rmsnorm_scale_weight<TW>(a.w, a.has_weight, a.unit_offset, d);
        yr[d] = TIO::fromF32(TIO::toF32(xr[d]) * inv_rms * w);
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rmsnorm_f32(uniform RmsNormArgs<ElemF32, ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rmsnorm_gpu_impl<ElemF32, ElemF32, VecIOF32, VecIOF32>(a, gid.x, gtid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rmsnorm_bf16(uniform RmsNormArgs<ElemBF16, ElemBF16> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rmsnorm_gpu_impl<ElemBF16, ElemBF16, VecIOBF16, VecIOBF16>(a, gid.x, gtid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rmsnorm_f16(uniform RmsNormArgs<ElemF16, ElemF16> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rmsnorm_gpu_impl<ElemF16, ElemF16, VecIOF16, VecIOF16>(a, gid.x, gtid.x);
}

struct FusedAddRmsNormArgs {
    Ptr<float> residual_inout;
    Ptr<float, Access.Read> x;
    Ptr<float, Access.Read> w;
    Ptr<float> y;
    uint rows;
    uint dim;
    uint r_stride;
    uint x_stride;
    uint y_stride;
    float eps;
    uint has_weight;
    uint unit_offset;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_fused_add_rmsnorm_f32(uniform FusedAddRmsNormArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    Ptr<float> rr = a.residual_inout + row * a.r_stride;
    Ptr<float, Access.Read> xr = a.x + row * a.x_stride;
    Ptr<float> yr = a.y + row * a.y_stride;

    bool aligned = ((a.dim & 3u) == 0u) && is_ptr_aligned16(rr) && is_ptr_aligned16(xr) &&
                   is_ptr_aligned16(yr) && ((a.has_weight == 0u) || is_ptr_aligned16(a.w));
    if (aligned && a.dim <= 4096u) {
        uint n4 = a.dim >> 2u;
        Ptr<float4> rr4 = Ptr<float4>(rr);
        Ptr<float4, Access.Read> rr4_r = Ptr<float4, Access.Read>(rr);
        Ptr<float4, Access.Read> xr4 = Ptr<float4, Access.Read>(xr);
        Ptr<float4, Access.Read> w4 = Ptr<float4, Access.Read>(a.w);
        Ptr<float4> yr4 = Ptr<float4>(yr);

        float4 reg[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        float partial = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                float4 v = load_a16(rr4_r + idx) + load_a16(xr4 + idx);
                store_a16(rr4 + idx, v);
                reg[s] = v;
                partial += dot(v, v);
            }
        }
        float inv_rms = rsqrt(wg_reduce_sum(partial, tid, 256u) / float(a.dim) + a.eps);
        float uoff = (a.unit_offset != 0u) ? 1.0f : 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                float4 wv = (a.has_weight != 0u) ? (load_a16(w4 + idx) + float4(uoff)) : float4(1.0f);
                store_a16(yr4 + idx, reg[s] * (inv_rms * wv));
            }
        }
        return;
    }

    float partial = 0.0f;
    for (uint d = tid; d < a.dim; d += 256u) {
        float v = rr[d] + xr[d];
        rr[d] = v;
        partial += v * v;
    }
    float inv_rms = rsqrt(wg_reduce_sum(partial, tid, 256u) / float(a.dim) + a.eps);
    for (uint d = tid; d < a.dim; d += 256u) {
        float w = rmsnorm_scale_weight<ElemF32>(a.w, a.has_weight, a.unit_offset, d);
        yr[d] = rr[d] * inv_rms * w;
    }
}

struct LayerNormArgs {
    Ptr<float, Access.Read> x;
    Ptr<float, Access.Read> w;
    Ptr<float, Access.Read> b;
    Ptr<float> y;
    uint rows;
    uint dim;
    uint x_stride;
    uint y_stride;
    float eps;
    uint has_weight;
    uint has_bias;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_layernorm_f32(uniform LayerNormArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    Ptr<float, Access.Read> xr = a.x + row * a.x_stride;
    Ptr<float> yr = a.y + row * a.y_stride;
    float inv_n = 1.0f / float(a.dim);

    bool aligned = ((a.dim & 3u) == 0u) && is_ptr_aligned16(xr) && is_ptr_aligned16(yr) &&
                   ((a.has_weight == 0u) || is_ptr_aligned16(a.w)) &&
                   ((a.has_bias == 0u) || is_ptr_aligned16(a.b));
    if (aligned && a.dim <= 4096u) {
        uint n4 = a.dim >> 2u;
        Ptr<float4, Access.Read> xr4 = Ptr<float4, Access.Read>(xr);
        Ptr<float4, Access.Read> w4 = Ptr<float4, Access.Read>(a.w);
        Ptr<float4, Access.Read> b4 = Ptr<float4, Access.Read>(a.b);
        Ptr<float4> yr4 = Ptr<float4>(yr);

        float4 reg[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        float psum = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                float4 v = load_a16(xr4 + idx);
                reg[s] = v;
                psum += v.x + v.y + v.z + v.w;
            }
        }
        float mean = wg_reduce_sum(psum, tid, 256u) * inv_n;

        float pvar = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                float4 d = reg[s] - float4(mean);
                reg[s] = d;
                pvar += dot(d, d);
            }
        }
        float inv_std = rsqrt(wg_reduce_sum(pvar, tid, 256u) * inv_n + a.eps);

        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                float4 outv = reg[s] * inv_std;
                if (a.has_weight != 0u) outv *= load_a16(w4 + idx);
                if (a.has_bias != 0u) outv += load_a16(b4 + idx);
                store_a16(yr4 + idx, outv);
            }
        }
        return;
    }

    float psum = 0.0f;
    for (uint d = tid; d < a.dim; d += 256u) psum += xr[d];
    float mean = wg_reduce_sum(psum, tid, 256u) * inv_n;
    float pvar = 0.0f;
    for (uint d = tid; d < a.dim; d += 256u) {
        float diff = xr[d] - mean;
        pvar += diff * diff;
    }
    float inv_std = rsqrt(wg_reduce_sum(pvar, tid, 256u) * inv_n + a.eps);
    for (uint d = tid; d < a.dim; d += 256u) {
        float v = (xr[d] - mean) * inv_std;
        if (a.has_weight != 0u) v *= a.w[d];
        if (a.has_bias != 0u) v += a.b[d];
        yr[d] = v;
    }
}

// ============================================================================
// 2. RoPE (HALF & INTERLEAVED, 1 warp/head = 4 heads/WG, 0 idle lanes, 128-bit HALF path)
// ============================================================================
struct RopeArgs<T : IElement> {
    Ptr<T.Storage, Access.Read> x;
    Ptr<float, Access.Read> cos_tbl;
    Ptr<float, Access.Read> sin_tbl;
    Ptr<T.Storage> y;
    uint n_tokens;
    uint n_heads;
    uint head_dim;
    uint rotary_dim;
    uint pos_offset;
    uint tok_sx;
    uint head_sx;
    uint tok_sy;
    uint head_sy;
    uint mode;
};

void rope_gpu_impl<T : IElement, VIO : IVecIO128<T>>(RopeArgs<T> a, uint3 gid, uint3 gtid) {
    uint half_rot = a.rotary_dim >> 1u;
    if ((a.mode & 2u) != 0u) {
        // Flattened 1D fast HALF path (rotary_dim == head_dim, 16B aligned): 100% lane utilization!
        uint v_elems = VIO.kElems;
        uint n_vec = half_rot / v_elems;
        uint idx = gid.x * 256u + gtid.x;
        uint pair = idx / n_vec;
        uint i = idx - pair * n_vec;
        uint t = pair / a.n_heads;
        if (t >= a.n_tokens) return;
        uint h = pair - t * a.n_heads;
        uint cs_off = (a.pos_offset + t) * half_rot;
        Ptr<T.Storage, Access.Read> xh = a.x + t * a.tok_sx + h * a.head_sx;
        Ptr<T.Storage> yh = a.y + t * a.tok_sy + h * a.head_sy;

        Ptr<uint4, Access.Read> x0_4 = Ptr<uint4, Access.Read>(xh);
        Ptr<uint4, Access.Read> x1_4 = Ptr<uint4, Access.Read>(xh + half_rot);
        Ptr<uint4> y0_4 = Ptr<uint4>(yh);
        Ptr<uint4> y1_4 = Ptr<uint4>(yh + half_rot);
        Ptr<float4, Access.Read> c4 = Ptr<float4, Access.Read>(a.cos_tbl + cs_off);
        Ptr<float4, Access.Read> s4 = Ptr<float4, Access.Read>(a.sin_tbl + cs_off);

        float4 x0l, x0h, x1l, x1h;
        VIO::unpack(load_a16(x0_4 + i), x0l, x0h);
        VIO::unpack(load_a16(x1_4 + i), x1l, x1h);
        if (VIO.kElems == 4u) {
            float4 cv = load_a16(c4 + i);
            float4 sv = load_a16(s4 + i);
            store_a16(y0_4 + i, VIO::pack(x0l * cv - x1l * sv, float4(0.0f)));
            store_a16(y1_4 + i, VIO::pack(x1l * cv + x0l * sv, float4(0.0f)));
        } else {
            float4 cl = load_a16(c4 + (i * 2u + 0u));
            float4 sl = load_a16(s4 + (i * 2u + 0u));
            float4 ch = load_a16(c4 + (i * 2u + 1u));
            float4 sh = load_a16(s4 + (i * 2u + 1u));
            store_a16(y0_4 + i, VIO::pack(x0l * cl - x1l * sl, x0h * ch - x1h * sh));
            store_a16(y1_4 + i, VIO::pack(x1l * cl + x0l * sl, x1h * ch + x0h * sh));
        }
        return;
    }

    uint lane = gtid.x & 31u;
    uint warp = gtid.x >> 5u;
    uint t = gid.y;
    uint h = gid.x * 8u + warp;
    if (t >= a.n_tokens || h >= a.n_heads) return;
    uint cs_off = (a.pos_offset + t) * half_rot;
    Ptr<T.Storage, Access.Read> xh = a.x + t * a.tok_sx + h * a.head_sx;
    Ptr<T.Storage> yh = a.y + t * a.tok_sy + h * a.head_sy;

    for (uint i = lane; i < half_rot; i += 32u) {
        float c = a.cos_tbl[cs_off + i];
        float s = a.sin_tbl[cs_off + i];
        uint i0 = ((a.mode & 1u) == 0u) ? i : (i << 1u);
        uint i1 = ((a.mode & 1u) == 0u) ? (i + half_rot) : ((i << 1u) + 1u);
        float x0 = T::toF32(xh[i0]);
        float x1 = T::toF32(xh[i1]);
        yh[i0] = T::fromF32(x0 * c - x1 * s);
        yh[i1] = T::fromF32(x1 * c + x0 * s);
    }
    for (uint i = a.rotary_dim + lane; i < a.head_dim; i += 32u) {
        yh[i] = xh[i];
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rope_f32(uniform RopeArgs<ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rope_gpu_impl<ElemF32, VecIOF32>(a, gid, gtid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rope_bf16(uniform RopeArgs<ElemBF16> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rope_gpu_impl<ElemBF16, VecIOBF16>(a, gid, gtid);
}

// ============================================================================
// 3. Elementwise Vectorized MatOps: Activations, GatedMLP, Axpby, Mul, Softmax, FWHT, Quant
// ============================================================================
struct ActivationArgs {
    Ptr<float, Access.Read> x;
    Ptr<float> y;
    uint count;
    uint pad;
};

void activation_gpu_impl<A : IActivation>(ActivationArgs a, uint tid) {
    uint base = tid << 2u;
    if (base + 4u <= a.count) {
        float4 v = load_a16(Ptr<float4, Access.Read>(a.x) + tid);
        store_a16(Ptr<float4>(a.y) + tid, apply_act4<A>(v));
    } else {
        for (uint i = base; i < a.count; ++i) a.y[i] = A::apply(a.x[i]);
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_activation_silu_f32(uniform ActivationArgs a, uint3 tid : SV_DispatchThreadID) {
    activation_gpu_impl<ActSiLU>(a, tid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_activation_gelu_f32(uniform ActivationArgs a, uint3 tid : SV_DispatchThreadID) {
    activation_gpu_impl<ActGeluTanh>(a, tid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_activation_relu2_f32(uniform ActivationArgs a, uint3 tid : SV_DispatchThreadID) {
    activation_gpu_impl<ActReLU2>(a, tid.x);
}

struct GatedMlpArgs {
    Ptr<float, Access.Read> gate;
    Ptr<float, Access.Read> up;
    Ptr<float> y;
    uint count;
    float clamp_limit;
};

void gated_mlp_gpu_impl<A : IActivation>(GatedMlpArgs a, uint tid) {
    uint base = tid << 2u;
    if (base + 4u <= a.count) {
        float4 g = apply_clamp4(load_a16(Ptr<float4, Access.Read>(a.gate) + tid), a.clamp_limit);
        float4 u = apply_clamp4(load_a16(Ptr<float4, Access.Read>(a.up) + tid), a.clamp_limit);
        store_a16(Ptr<float4>(a.y) + tid, apply_act4<A>(g) * u);
    } else {
        for (uint i = base; i < a.count; ++i) {
            float g = apply_symmetric_clamp(a.gate[i], a.clamp_limit);
            float u = apply_symmetric_clamp(a.up[i], a.clamp_limit);
            a.y[i] = A::apply(g) * u;
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gated_silu_f32(uniform GatedMlpArgs a, uint3 tid : SV_DispatchThreadID) {
    gated_mlp_gpu_impl<ActSiLU>(a, tid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gated_gelu_f32(uniform GatedMlpArgs a, uint3 tid : SV_DispatchThreadID) {
    gated_mlp_gpu_impl<ActGeluTanh>(a, tid.x);
}

struct VecAxpbyArgs {
    Ptr<float, Access.Read> a;
    Ptr<float, Access.Read> b;
    Ptr<float> y;
    uint count;
    float alpha;
    float beta;
    uint has_b;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_vec_axpby_f32(uniform VecAxpbyArgs p, uint3 tid : SV_DispatchThreadID) {
    uint base = tid.x << 2u;
    if (base + 4u <= p.count) {
        float4 outv = float4(p.alpha) * load_a16(Ptr<float4, Access.Read>(p.a) + tid.x);
        if (p.has_b != 0u) outv = fma(float4(p.beta), load_a16(Ptr<float4, Access.Read>(p.b) + tid.x), outv);
        store_a16(Ptr<float4>(p.y) + tid.x, outv);
    } else {
        for (uint i = base; i < p.count; ++i) {
            float v = p.alpha * p.a[i];
            if (p.has_b != 0u) v = fma(p.beta, p.b[i], v);
            p.y[i] = v;
        }
    }
}

struct VecMulArgs {
    Ptr<float, Access.Read> a;
    Ptr<float, Access.Read> b;
    Ptr<float> y;
    uint count;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_vec_mul_f32(uniform VecMulArgs p, uint3 tid : SV_DispatchThreadID) {
    uint base = tid.x << 2u;
    if (base + 4u <= p.count) {
        float4 av = load_a16(Ptr<float4, Access.Read>(p.a) + tid.x);
        float4 bv = load_a16(Ptr<float4, Access.Read>(p.b) + tid.x);
        store_a16(Ptr<float4>(p.y) + tid.x, av * bv);
    } else {
        for (uint i = base; i < p.count; ++i) p.y[i] = p.a[i] * p.b[i];
    }
}

struct SoftmaxArgs {
    Ptr<float, Access.Read> x;
    Ptr<float> y;
    uint rows;
    uint cols;
    uint x_stride;
    uint y_stride;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_softmax_f32(uniform SoftmaxArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    Ptr<float, Access.Read> xr = a.x + row * a.x_stride;
    Ptr<float> yr = a.y + row * a.y_stride;

    bool aligned = ((a.cols & 3u) == 0u) && is_ptr_aligned16(xr) && is_ptr_aligned16(yr);
    if (aligned && a.cols <= 4096u) {
        // Single-pass register-cached Softmax: reads x ONCE and writes y ONCE!
        uint n4 = a.cols >> 2u;
        Ptr<float4, Access.Read> xr4 = Ptr<float4, Access.Read>(xr);
        Ptr<float4> yr4 = Ptr<float4>(yr);

        float4 reg[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        float local_max = -3.402823466e+38f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                float4 v = load_a16(xr4 + idx);
                reg[s] = v;
                local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
            }
        }
        float row_max = wg_reduce_max(local_max, tid, 256u);

        float local_sum = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                float4 v = reg[s] - float4(row_max);
                float4 e = float4(fast_expf(v.x), fast_expf(v.y), fast_expf(v.z), fast_expf(v.w));
                reg[s] = e;
                local_sum += e.x + e.y + e.z + e.w;
            }
        }
        float inv_sum = 1.0f / wg_reduce_sum(local_sum, tid, 256u);

        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                store_a16(yr4 + idx, reg[s] * inv_sum);
            }
        }
        return;
    }

    float local_max = -3.402823466e+38f;
    for (uint c = tid; c < a.cols; c += 256u) {
        local_max = max(local_max, xr[c]);
    }
    float row_max = wg_reduce_max(local_max, tid, 256u);

    float local_sum = 0.0f;
    for (uint c = tid; c < a.cols; c += 256u) {
        float e = fast_expf(xr[c] - row_max);
        yr[c] = e;
        local_sum += e;
    }
    float inv_sum = 1.0f / wg_reduce_sum(local_sum, tid, 256u);

    for (uint c = tid; c < a.cols; c += 256u) {
        yr[c] *= inv_sum;
    }
}

// ============================================================================
// Normalized FWHT (ONE unified kernel for B in {16,32,64,128,256} x suh in {none,F32,F16})
// - float4 in-register H_4 (strides 1,2) + warp-shuffle (strides 4..64) + 1 SMEM stage (B=256)
// ============================================================================
struct FwhtArgs {
    Ptr<float, Access.Read> x;
    Ptr<uint8_t, Access.Read> suh;
    Ptr<float> y;
    uint rows;
    uint cols;
    uint block_size;
    uint x_stride;
    uint y_stride;
    uint suh_mode; // 0=none, 1=f32, 2=f16
    float inv_sqrt_b;
    uint pad;
};

groupshared float4 s_fwht[256];

float4 fwht_warp_step(float4 v, uint lane, uint bit) {
    float4 p = WaveReadLaneAt(v, lane ^ bit);
    return ((lane & bit) == 0u) ? (v + p) : (p - v);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_fwht_f32(uniform FwhtArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.y;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    uint elem0 = gid.x * 1024u + tid * 4u;
    Ptr<float, Access.Read> xr = a.x + row * a.x_stride;
    Ptr<float> yr = a.y + row * a.y_stride;

    float4 v = float4(0.0f);
    bool active = (elem0 < a.cols);
    if (active) {
        if (is_ptr_aligned16(xr)) {
            v = load_a16(Ptr<float4, Access.Read>(xr + elem0));
        } else {
            v = float4(xr[elem0 + 0u], xr[elem0 + 1u], xr[elem0 + 2u], xr[elem0 + 3u]);
        }
        if (a.suh_mode == 1u) {
            Ptr<float, Access.Read> sf = Ptr<float, Access.Read>(a.suh);
            float4 sv = is_ptr_aligned16(sf) ? load_a16(Ptr<float4, Access.Read>(sf + elem0))
                                             : float4(sf[elem0], sf[elem0 + 1u], sf[elem0 + 2u], sf[elem0 + 3u]);
            v *= sv;
        } else if (a.suh_mode == 2u) {
            Ptr<half, Access.Read> sh = Ptr<half, Access.Read>(a.suh);
            v *= float4(float(sh[elem0]), float(sh[elem0 + 1u]), float(sh[elem0 + 2u]), float(sh[elem0 + 3u]));
        }
        // In-register H_4 (strides 1 and 2)
        float a0 = v.x + v.y, a1 = v.x - v.y;
        float a2 = v.z + v.w, a3 = v.z - v.w;
        v = float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);
    }

    uint lane = tid & 31u;
    v = fwht_warp_step(v, lane, 1u); // stride 4 (B >= 16)
    v = fwht_warp_step(v, lane, 2u); // stride 8 (B >= 16)
    if (a.block_size >= 32u)  v = fwht_warp_step(v, lane, 4u);  // stride 16
    if (a.block_size >= 64u)  v = fwht_warp_step(v, lane, 8u);  // stride 32
    if (a.block_size >= 128u) v = fwht_warp_step(v, lane, 16u); // stride 64
    if (a.block_size >= 256u) {
        s_fwht[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        float4 p = s_fwht[tid ^ 32u];
        v = ((tid & 32u) == 0u) ? (v + p) : (p - v);
    }

    if (active) {
        v *= a.inv_sqrt_b;
        if (is_ptr_aligned16(yr)) {
            store_a16(Ptr<float4>(yr + elem0), v);
        } else {
            yr[elem0 + 0u] = v.x; yr[elem0 + 1u] = v.y;
            yr[elem0 + 2u] = v.z; yr[elem0 + 3u] = v.w;
        }
    }
}

// ============================================================================
// Parallel Dynamic Symmetric Row Quantization (F32 -> I8, single-pass cached for k <= 4096)
// ============================================================================
struct QuantizeRowArgs {
    Ptr<float, Access.Read> x;
    Ptr<int8_t> x_i8;
    Ptr<float> scale_out;
    uint rows;
    uint k;
    uint x_stride;
    uint xi8_stride;
};

uint pack_i8x4(float4 v, float inv_scale) {
    float4 s = clamp(round(v * inv_scale), float4(-127.0f), float4(127.0f));
    uint b0 = uint(int32_t(s.x)) & 0xFFu;
    uint b1 = uint(int32_t(s.y)) & 0xFFu;
    uint b2 = uint(int32_t(s.z)) & 0xFFu;
    uint b3 = uint(int32_t(s.w)) & 0xFFu;
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_quantize_row_f32_i8(uniform QuantizeRowArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    Ptr<float, Access.Read> xr = a.x + row * a.x_stride;
    Ptr<int8_t> qr = a.x_i8 + row * a.xi8_stride;

    bool aligned = ((a.k & 3u) == 0u) && is_ptr_aligned16(xr) && ((a.xi8_stride & 3u) == 0u);
    if (aligned && a.k <= 4096u) {
        uint n4 = a.k >> 2u;
        Ptr<float4, Access.Read> xr4 = Ptr<float4, Access.Read>(xr);
        Ptr<uint> qr4 = Ptr<uint>(qr);

        float4 reg[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        float local_max = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) {
                float4 v = load_a16(xr4 + idx);
                reg[s] = v;
                float4 av = abs(v);
                local_max = max(local_max, max(max(av.x, av.y), max(av.z, av.w)));
            }
        }
        float amax = wg_reduce_max(local_max, tid, 256u);
        float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        if (tid == 0u) a.scale_out[row] = scale;

        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * 256u + tid;
            if (idx < n4) qr4[idx] = pack_i8x4(reg[s], inv_scale);
        }
        return;
    }

    float local_max = 0.0f;
    for (uint c = tid; c < a.k; c += 256u) {
        local_max = max(local_max, abs(xr[c]));
    }
    float amax = wg_reduce_max(local_max, tid, 256u);
    float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    if (tid == 0u) a.scale_out[row] = scale;

    for (uint c = tid; c < a.k; c += 256u) {
        float v = clamp(round(xr[c] * inv_scale), -127.0f, 127.0f);
        qr[c] = int8_t(int32_t(v));
    }
}

// ============================================================================
// 4. Unified High-Bandwidth GEMV (1 subgroup/row, 4 rows/WG, 128-bit coalesced)
//    ONE generic skeleton parameterized by an IVecDot128 interface!
// ============================================================================
struct GemvArgs<TX : IElement, TW : IElement, TY : IElement> {
    Ptr<TX.Storage, Access.Read> x;
    Ptr<TW.Storage, Access.Read> w;
    Ptr<float, Access.Read> bias;
    Ptr<float, Access.Read> x_scale;
    Ptr<float, Access.Read> w_scale;
    Ptr<TY.Storage> y;
    uint rows;
    uint m;
    uint k;
    uint x_stride;
    uint w_stride;
    uint y_stride;
    uint flags;
    uint split_k;
};

interface IGemvDot128<TX : IElement, TW : IElement> {
    static const uint kElemsPer128;
    static float dot128(uint4 xv, uint4 wv);
}

struct GemvDotF32 : IGemvDot128<ElemF32, ElemF32> {
    static const uint kElemsPer128 = 4u;
    static float dot128(uint4 xv, uint4 wv) {
        float4 xf = asfloat(xv);
        float4 wf = asfloat(wv);
        return dot(xf, wf);
    }
}

struct GemvDotF16 : IGemvDot128<ElemF16, ElemF16> {
    static const uint kElemsPer128 = 8u;
    static float dot128(uint4 xv, uint4 wv) {
        float s = 0.0f;
        [ForceUnroll] for (uint e = 0u; e < 4u; ++e) {
            uint xw = (e == 0u) ? xv.x : ((e == 1u) ? xv.y : ((e == 2u) ? xv.z : xv.w));
            uint ww = (e == 0u) ? wv.x : ((e == 1u) ? wv.y : ((e == 2u) ? wv.z : wv.w));
            s = fma(float(bit_cast<half>(uint16_t(xw & 0xFFFFu))), float(bit_cast<half>(uint16_t(ww & 0xFFFFu))), s);
            s = fma(float(bit_cast<half>(uint16_t(xw >> 16u))),    float(bit_cast<half>(uint16_t(ww >> 16u))),    s);
        }
        return s;
    }
}

struct GemvDotBF16 : IGemvDot128<ElemBF16, ElemBF16> {
    static const uint kElemsPer128 = 8u;
    static float dot128(uint4 xv, uint4 wv) {
        float s = 0.0f;
        s = fma(asfloat((xv.x & 0xFFFFu) << 16u), asfloat((wv.x & 0xFFFFu) << 16u), s);
        s = fma(asfloat(xv.x & 0xFFFF0000u),      asfloat(wv.x & 0xFFFF0000u),      s);
        s = fma(asfloat((xv.y & 0xFFFFu) << 16u), asfloat((wv.y & 0xFFFFu) << 16u), s);
        s = fma(asfloat(xv.y & 0xFFFF0000u),      asfloat(wv.y & 0xFFFF0000u),      s);
        s = fma(asfloat((xv.z & 0xFFFFu) << 16u), asfloat((wv.z & 0xFFFFu) << 16u), s);
        s = fma(asfloat(xv.z & 0xFFFF0000u),      asfloat(wv.z & 0xFFFF0000u),      s);
        s = fma(asfloat((xv.w & 0xFFFFu) << 16u), asfloat((wv.w & 0xFFFFu) << 16u), s);
        s = fma(asfloat(xv.w & 0xFFFF0000u),      asfloat(wv.w & 0xFFFF0000u),      s);
        return s;
    }
}

// ONE generic float-accumulating GEMV kernel shared by F32, F16, and BF16!
void gemv_float_gpu_impl<TX : IElement, TW : IElement, Dot : IGemvDot128<TX, TW>, Ep : IEpilogue>(
    GemvArgs<TX, TW, ElemF32> a, uint3 gtid, uint3 gid)
{
    uint lane = WaveGetLaneIndex();
    uint lane_cnt = WaveGetLaneCount();
    uint warp_in_wg = gtid.x / lane_cnt;
    uint warps_per_wg = 128u / lane_cnt;
    uint col = gid.x * warps_per_wg + warp_in_wg;
    uint row = gid.y;
    if (row >= a.rows || col >= a.m) return;

    Ptr<TX.Storage, Access.Read> xr = a.x + row * a.x_stride;
    Ptr<TW.Storage, Access.Read> wc = a.w + col * a.w_stride;
    uint mask = Dot.kElemsPer128 - 1u;
    float acc = 0.0f;
    if (is_ptr_aligned16(xr) && is_ptr_aligned16(wc)) {
        uint k_vec = a.k & ~mask;
        uint n_vec = k_vec / Dot.kElemsPer128;
        Ptr<uint4, Access.Read> xr4 = Ptr<uint4, Access.Read>(xr);
        Ptr<uint4, Access.Read> wc4 = Ptr<uint4, Access.Read>(wc);
        for (uint i = lane; i < n_vec; i += lane_cnt) {
            acc += Dot::dot128(load_a16(xr4 + i), load_a16(wc4 + i));
        }
        for (uint i = k_vec + lane; i < a.k; i += lane_cnt) {
            acc = fma(TX::toF32(xr[i]), TW::toF32(wc[i]), acc);
        }
    } else {
        for (uint i = lane; i < a.k; i += lane_cnt) {
            acc = fma(TX::toF32(xr[i]), TW::toF32(wc[i]), acc);
        }
    }
    float total = WaveActiveSum(acc);
    if (lane == 0u) {
        a.y[row * a.y_stride + col] = Ep::apply(total, row, col, make_vk_ep(a.bias, a.x_scale, a.w_scale, a.flags));
    }
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_f32(uniform GemvArgs<ElemF32, ElemF32, ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    gemv_float_gpu_impl<ElemF32, ElemF32, GemvDotF32, EpBias>(a, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_f16(uniform GemvArgs<ElemF16, ElemF16, ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    gemv_float_gpu_impl<ElemF16, ElemF16, GemvDotF16, EpBias>(a, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_bf16(uniform GemvArgs<ElemBF16, ElemBF16, ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    gemv_float_gpu_impl<ElemBF16, ElemBF16, GemvDotBF16, EpBias>(a, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_w8a8(uniform GemvArgs<ElemI8, ElemI8, ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint lane = WaveGetLaneIndex();
    uint lane_cnt = WaveGetLaneCount();
    uint warp_in_wg = gtid.x / lane_cnt;
    uint warps_per_wg = 128u / lane_cnt;
    uint col = gid.x * warps_per_wg + warp_in_wg;
    uint row = gid.y;
    if (row >= a.rows || col >= a.m) return;

    Ptr<int8_t, Access.Read> xr = a.x + row * a.x_stride;
    Ptr<int8_t, Access.Read> wc = a.w + col * a.w_stride;
    int32_t acc = 0;
    uint k16 = a.k & ~15u;
    if (is_ptr_aligned16(xr) && is_ptr_aligned16(wc)) {
        Ptr<uint4, Access.Read> xr16 = Ptr<uint4, Access.Read>(xr);
        Ptr<uint4, Access.Read> wc16 = Ptr<uint4, Access.Read>(wc);
        uint n16 = k16 >> 4u;
        for (uint i = lane; i < n16; i += lane_cnt) {
            uint4 xv = load_a16(xr16 + i);
            uint4 wv = load_a16(wc16 + i);
            acc = dot4add_i8packed(xv.x, wv.x, acc);
            acc = dot4add_i8packed(xv.y, wv.y, acc);
            acc = dot4add_i8packed(xv.z, wv.z, acc);
            acc = dot4add_i8packed(xv.w, wv.w, acc);
        }
        for (uint i = k16 + lane; i < a.k; i += lane_cnt) {
            acc += int32_t(xr[i]) * int32_t(wc[i]);
        }
    } else {
        for (uint i = lane; i < a.k; i += lane_cnt) {
            acc += int32_t(xr[i]) * int32_t(wc[i]);
        }
    }
    int32_t total = WaveActiveSum(acc);
    if (lane == 0u) {
        a.y[row * a.y_stride + col] = EpQuantScale::apply(float(total), row, col, make_vk_ep(a.bias, a.x_scale, a.w_scale, a.flags));
    }
}

// ============================================================================
// 5. High-Throughput Double-Buffered BK=8 Tiled F32 GEMM (128x128 WG, 256T, 8x8)
//    - 2-stage double-buffered BK=8 -> 16.5 KiB SMEM -> 2 WGs/SM (16 warps/SM)!
//    - Scalar STS.32 transpose into groupshared float4[] -> 0 bank conflicts & native LDS.128!
//    - Unroll-by-2 LDS/FMA + split kk=0..3 / STS / kk=4..7 overlap -> 0 spills (<128 regs)!
// ============================================================================
groupshared float4 s_f32_a[2 * 8 * 33];
groupshared float4 s_f32_b[2 * 8 * 33];

uint2 swizzle_tile_2d(uint3 gid, uint grid_x, uint grid_y) {
    uint linear = gid.y * grid_x + gid.x;
    uint tiles_per_grp = 8u * grid_x;
    uint grp = linear / tiles_per_grp;
    uint rem = linear - grp * tiles_per_grp;
    uint first_y = grp * 8u;
    uint grp_h = min(8u, grid_y - first_y);
    return uint2(rem / grp_h, first_y + (rem % grp_h));
}

float4 load_f32x4_safe(Ptr<float, Access.Read> base, uint row, uint row_lim,
                       uint k, uint k_lim, uint stride, bool aligned4)
{
    if (row >= row_lim || k >= k_lim) return float4(0.0f);
    if (aligned4 && k + 4u <= k_lim) {
        return load_a16(Ptr<float4, Access.Read>(base + row * stride + k));
    }
    uint off = row * stride + k;
    return float4(
        (k + 0u < k_lim) ? base[off + 0u] : 0.0f,
        (k + 1u < k_lim) ? base[off + 1u] : 0.0f,
        (k + 2u < k_lim) ? base[off + 2u] : 0.0f,
        (k + 3u < k_lim) ? base[off + 3u] : 0.0f);
}

void store_f32_transposed_a(uint sb, uint comp, float4 v) {
    s_f32_a[sb +  0u][comp] = v.x;
    s_f32_a[sb + 33u][comp] = v.y;
    s_f32_a[sb + 66u][comp] = v.z;
    s_f32_a[sb + 99u][comp] = v.w;
}

void store_f32_transposed_b(uint sb, uint comp, float4 v) {
    s_f32_b[sb +  0u][comp] = v.x;
    s_f32_b[sb + 33u][comp] = v.y;
    s_f32_b[sb + 66u][comp] = v.z;
    s_f32_b[sb + 99u][comp] = v.w;
}

void f32_outer_8x8(inout float4 acc0[8], inout float4 acc1[8],
                   float4 a0, float4 a1, float4 b0, float4 b1)
{
    acc0[0] = fma(float4(a0.x), b0, acc0[0]); acc1[0] = fma(float4(a0.x), b1, acc1[0]);
    acc1[1] = fma(float4(a0.y), b1, acc1[1]); acc0[1] = fma(float4(a0.y), b0, acc0[1]);
    acc0[2] = fma(float4(a0.z), b0, acc0[2]); acc1[2] = fma(float4(a0.z), b1, acc1[2]);
    acc1[3] = fma(float4(a0.w), b1, acc1[3]); acc0[3] = fma(float4(a0.w), b0, acc0[3]);
    acc0[4] = fma(float4(a1.x), b0, acc0[4]); acc1[4] = fma(float4(a1.x), b1, acc1[4]);
    acc1[5] = fma(float4(a1.y), b1, acc1[5]); acc0[5] = fma(float4(a1.y), b0, acc0[5]);
    acc0[6] = fma(float4(a1.z), b0, acc0[6]); acc1[6] = fma(float4(a1.z), b1, acc1[6]);
    acc1[7] = fma(float4(a1.w), b1, acc1[7]); acc0[7] = fma(float4(a1.w), b0, acc0[7]);
}

void f32_gemm_mma8(inout float4 acc0[8], inout float4 acc1[8], uint sbase, uint a_col, uint b_col) {
    [ForceUnroll] for (uint kk = 0u; kk < 8u; kk += 2u) {
        uint off0 = sbase + kk * 33u;
        uint off1 = off0 + 33u;
        float4 a0_0 = s_f32_a[off0 + a_col];
        float4 a1_0 = s_f32_a[off0 + a_col + 4u];
        float4 b0_0 = s_f32_b[off0 + b_col];
        float4 b1_0 = s_f32_b[off0 + b_col + 8u];
        float4 a0_1 = s_f32_a[off1 + a_col];
        float4 a1_1 = s_f32_a[off1 + a_col + 4u];
        float4 b0_1 = s_f32_b[off1 + b_col];
        float4 b1_1 = s_f32_b[off1 + b_col + 8u];
        f32_outer_8x8(acc0, acc1, a0_0, a1_0, b0_0, b1_0);
        f32_outer_8x8(acc0, acc1, a0_1, a1_1, b0_1, b1_1);
    }
}

void f32_gemm_store_epilogue(
    uniform GemvArgs<ElemF32, ElemF32, ElemF32> a,
    Ptr<float> y_out, uint flags_eff,
    float4 acc0[8], float4 acc1[8],
    uint row0, uint col0, uint a_col, uint b_col)
{
    EpilogueParams ep = make_vk_ep(a.bias, a.x_scale, a.w_scale, flags_eff);
    bool y_vec = is_ptr_aligned16(y_out) && ((a.y_stride & 3u) == 0u);
    bool has_bias = ep.has_bias != 0u;
    bool bias_vec = has_bias && is_ptr_aligned16(a.bias);

    uint r_base0 = row0 + a_col * 4u;
    uint r_base1 = r_base0 + 16u;
    uint c0 = col0 + b_col * 4u;
    uint c1 = c0 + 32u;

    float4 bias0 = float4(0.0f);
    float4 bias1 = float4(0.0f);
    if (has_bias) {
        if (c0 + 4u <= a.m) {
            bias0 = bias_vec ? load_a16(Ptr<float4, Access.Read>(a.bias + c0))
                             : float4(a.bias[c0], a.bias[c0 + 1u], a.bias[c0 + 2u], a.bias[c0 + 3u]);
        }
        if (c1 + 4u <= a.m) {
            bias1 = bias_vec ? load_a16(Ptr<float4, Access.Read>(a.bias + c1))
                             : float4(a.bias[c1], a.bias[c1 + 1u], a.bias[c1 + 2u], a.bias[c1 + 3u]);
        }
    }

    [ForceUnroll] for (uint i = 0u; i < 8u; ++i) {
        uint r = (i < 4u) ? (r_base0 + i) : (r_base1 + (i - 4u));
        if (r >= a.rows) continue;
        Ptr<float> yr = y_out + r * a.y_stride;
        if (y_vec && c0 + 4u <= a.m) {
            store_a16(Ptr<float4>(yr + c0), acc0[i] + bias0);
        } else {
            float4 v0 = acc0[i];
            if (c0 + 0u < a.m) yr[c0 + 0u] = EpBias::apply(v0.x, r, c0 + 0u, ep);
            if (c0 + 1u < a.m) yr[c0 + 1u] = EpBias::apply(v0.y, r, c0 + 1u, ep);
            if (c0 + 2u < a.m) yr[c0 + 2u] = EpBias::apply(v0.z, r, c0 + 2u, ep);
            if (c0 + 3u < a.m) yr[c0 + 3u] = EpBias::apply(v0.w, r, c0 + 3u, ep);
        }
        if (y_vec && c1 + 4u <= a.m) {
            store_a16(Ptr<float4>(yr + c1), acc1[i] + bias1);
        } else {
            float4 v1 = acc1[i];
            if (c1 + 0u < a.m) yr[c1 + 0u] = EpBias::apply(v1.x, r, c1 + 0u, ep);
            if (c1 + 1u < a.m) yr[c1 + 1u] = EpBias::apply(v1.y, r, c1 + 1u, ep);
            if (c1 + 2u < a.m) yr[c1 + 2u] = EpBias::apply(v1.z, r, c1 + 2u, ep);
            if (c1 + 3u < a.m) yr[c1 + 3u] = EpBias::apply(v1.w, r, c1 + 3u, ep);
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_f32(uniform GemvArgs<ElemF32, ElemF32, ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint tid = gtid.x;
    uint grid_x = (a.m + 127u) >> 7u;
    uint grid_y = (a.rows + 127u) >> 7u;
    uint2 tile = swizzle_tile_2d(gid, grid_x, grid_y);
    uint row0 = tile.y * 128u;
    uint col0 = tile.x * 128u;

    uint warp_id = tid >> 5u;
    uint lane = tid & 31u;
    uint wr = warp_id >> 1u;
    uint wc = warp_id & 1u;
    uint lr_w = lane >> 3u;
    uint lc_w = lane & 7u;
    uint a_col = wr * 8u + lr_w;
    uint b_col = wc * 16u + lc_w;

    // 256 threads load 128 rows x 2 float4s (BK=8): 1 float4 A + 1 float4 B per thread!
    // Even lanes: l_lk=0 -> banks 0..15; odd lanes: l_lk=4 -> 4*132=528=16 (mod 32) -> banks 16..31 (0 conflicts!)
    uint l_lk = (tid & 1u) << 2u;       // 0 or 4
    uint lr = tid >> 1u;                // 0..127
    uint s_off = l_lk * 33u + (lr >> 2u);
    uint s_comp = lr & 3u;

    float4 acc0[8];
    float4 acc1[8];
    [ForceUnroll] for (uint i = 0u; i < 8u; ++i) {
        acc0[i] = float4(0.0f);
        acc1[i] = float4(0.0f);
    }

    uint k_start = 0u;
    uint k_len = a.k;
    Ptr<float> y_out = a.y;
    uint flags_eff = a.flags;
    if (a.split_k > 1u) {
        uint total_kt = (a.k + 7u) >> 3u;
        uint kt_per = (total_kt + a.split_k - 1u) / a.split_k;
        uint k_chunk = kt_per << 3u;
        k_start = min(a.k, gid.z * k_chunk);
        uint k_end = min(a.k, k_start + k_chunk);
        k_len = k_end - k_start;
        y_out = a.y + gid.z * (a.rows * a.y_stride);
        flags_eff = 0u;
    }

    Ptr<float, Access.Read> x_base = a.x + k_start;
    Ptr<float, Access.Read> w_base = a.w + k_start;
    bool aligned4 = is_ptr_aligned16(x_base) && is_ptr_aligned16(w_base) &&
                    ((a.x_stride & 3u) == 0u) && ((a.w_stride & 3u) == 0u);
    bool fast_interior = aligned4 && ((k_len & 7u) == 0u) && (row0 + 128u <= a.rows) && (col0 + 128u <= a.m);
    uint num_k_tiles = (k_len + 7u) >> 3u;

    if (fast_interior && num_k_tiles > 0u) {
        Ptr<float4, Access.Read> xp = Ptr<float4, Access.Read>(x_base + (row0 + lr) * a.x_stride + l_lk);
        Ptr<float4, Access.Read> wp = Ptr<float4, Access.Read>(w_base + (col0 + lr) * a.w_stride + l_lk);

        store_f32_transposed_a(s_off, s_comp, load_a16(xp));
        store_f32_transposed_b(s_off, s_comp, load_a16(wp));
        GroupMemoryBarrierWithGroupSync();

        uint stage = 0u;
        for (uint kt = 0u; kt + 1u < num_k_tiles; ++kt) {
            uint k_u4 = (kt + 1u) << 1u;
            float4 pA = load_a16(xp + k_u4);
            float4 pB = load_a16(wp + k_u4);
            uint sbase = stage * (8u * 33u);
            f32_gemm_mma8(acc0, acc1, sbase, a_col, b_col);
            uint nbase = (stage ^ 1u) * (8u * 33u);
            store_f32_transposed_a(nbase + s_off, s_comp, pA);
            store_f32_transposed_b(nbase + s_off, s_comp, pB);
            GroupMemoryBarrierWithGroupSync();
            stage ^= 1u;
        }
        f32_gemm_mma8(acc0, acc1, stage * (8u * 33u), a_col, b_col);
    } else {
        for (uint kt = 0u; kt < num_k_tiles; ++kt) {
            uint k0 = kt << 3u;
            float4 va = load_f32x4_safe(x_base, row0 + lr, a.rows, k0 + l_lk, k_len, a.x_stride, aligned4);
            float4 vb = load_f32x4_safe(w_base, col0 + lr, a.m,    k0 + l_lk, k_len, a.w_stride, aligned4);
            store_f32_transposed_a(s_off, s_comp, va);
            store_f32_transposed_b(s_off, s_comp, vb);
            GroupMemoryBarrierWithGroupSync();
            f32_gemm_mma8(acc0, acc1, 0u, a_col, b_col);
            GroupMemoryBarrierWithGroupSync();
        }
    }

    f32_gemm_store_epilogue(a, y_out, flags_eff, acc0, acc1, row0, col0, a_col, b_col);
}

// ============================================================================
// 5b. Universal Split-K Reduction + Epilogue (shared by F32, F16, BF16, W8A8!)
//     Sums `split_k` contiguous partial slices (float4 or bit-exact int4 for W8A8)
//     and applies quantized scales + bias in ONE coalesced 128-bit streaming pass.
// ============================================================================
struct SplitKReduceArgs {
    Ptr<uint4, Access.Read> partials;
    Ptr<float, Access.Read> bias;
    Ptr<float, Access.Read> x_scale;
    Ptr<float, Access.Read> w_scale;
    Ptr<float> y;
    uint rows;
    uint m;
    uint part_stride_u4;
    uint y_stride;
    uint split_k;
    uint flags;
    uint is_int32;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_splitk_reduce(uniform SplitKReduceArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint idx = gid.x * 256u + gtid.x;
    uint slice_u4 = a.rows * a.part_stride_u4;
    if (idx >= slice_u4) return;

    uint gr = idx / a.part_stride_u4;
    uint c_u4 = idx - gr * a.part_stride_u4;
    uint gc = c_u4 << 2u;

    float4 fv;
    if (a.is_int32 != 0u) {
        int4 sum_i = int4(0);
        for (uint s = 0u; s < a.split_k; ++s) {
            sum_i += int4(load_a16(a.partials + s * slice_u4 + idx));
        }
        fv = float4(sum_i);
    } else {
        float4 sum_f = float4(0.0f);
        for (uint s = 0u; s < a.split_k; ++s) {
            sum_f += asfloat(load_a16(a.partials + s * slice_u4 + idx));
        }
        fv = sum_f;
    }

    EpilogueParams ep = make_vk_ep(a.bias, a.x_scale, a.w_scale, a.flags);
    bool y_vec = is_ptr_aligned16(a.y) && ((a.y_stride & 3u) == 0u);
    if (y_vec && gc + 4u <= a.m) {
        float4 outv = fv;
        if (ep.has_x_scale != 0u || ep.has_w_scale != 0u) {
            float xs = (ep.has_x_scale != 0u) ? ep.x_scale[gr] : 1.0f;
            float4 ws = float4(1.0f);
            if (ep.has_w_scale != 0u) {
                ws = is_ptr_aligned16(ep.w_scale)
                     ? load_a16(Ptr<float4, Access.Read>(ep.w_scale + gc))
                     : float4(ep.w_scale[gc], ep.w_scale[gc + 1u], ep.w_scale[gc + 2u], ep.w_scale[gc + 3u]);
            }
            outv = fv * (ws * xs);
        }
        if (ep.has_bias != 0u) {
            float4 bv = is_ptr_aligned16(ep.bias)
                        ? load_a16(Ptr<float4, Access.Read>(ep.bias + gc))
                        : float4(ep.bias[gc], ep.bias[gc + 1u], ep.bias[gc + 2u], ep.bias[gc + 3u]);
            outv += bv;
        }
        store_a16(Ptr<float4>(a.y + gr * a.y_stride + gc), outv);
    } else {
        Ptr<float> yr = a.y + gr * a.y_stride;
        if (gc + 0u < a.m) yr[gc + 0u] = EpQuantScale::apply(fv.x, gr, gc + 0u, ep);
        if (gc + 1u < a.m) yr[gc + 1u] = EpQuantScale::apply(fv.y, gr, gc + 1u, ep);
        if (gc + 2u < a.m) yr[gc + 2u] = EpQuantScale::apply(fv.z, gr, gc + 2u, ep);
        if (gc + 3u < a.m) yr[gc + 3u] = EpQuantScale::apply(fv.w, gr, gc + 3u, ep);
    }
}

// ============================================================================
// 6. ONE Unified High-Occupancy Shared-Memory uint4 Cooperative Matrix GEMM
//    - Single-buffered 20 KiB (128x128 -> 2 WGs/SM = 16 warps/SM!) and
//      10 KiB (64x64 -> 4 WGs/SM = 16 warps/SM!) + register prefetch!
//    - Hoisted fast_interior loop (zero branch/fallback register bloat!)
//    - Reused across ALL 6 kernels: {FP16, BF16, W8A8} x {128x128, 64x64}
// ============================================================================
struct CoopGemmArgs<TMat, TY> {
    Ptr<TMat> x;
    Ptr<TMat> w;
    Ptr<float, Access.Read> bias;
    Ptr<float, Access.Read> x_scale;
    Ptr<float, Access.Read> w_scale;
    Ptr<TY> y;
    uint rows;
    uint m;
    uint k;
    uint x_stride;
    uint w_stride;
    uint y_stride;
    uint flags;
    uint split_k;
};

// 128x128 single-buffered + reg prefetch + STS/MMA overlap: 20 KiB -> 2 WGs/SM (16 warps/SM)!
groupshared uint4 s_cm128_a[128 * 5];
groupshared uint4 s_cm128_b[128 * 5];
// 64x64 single-buffered + reg prefetch + STS/MMA overlap: 10 KiB -> 4 WGs/SM (16 warps/SM)!
groupshared uint4 s_cm64_a[64 * 5];
groupshared uint4 s_cm64_b[64 * 5];

uint4 load_u4_safe<let ELEM_BYTES : int>(
    Ptr<uint8_t, Access.Read> base_b, uint row, uint row_lim,
    uint k_elem, uint k_lim, uint stride_elems, bool aligned16)
{
    if (row >= row_lim || k_elem >= k_lim) return uint4(0u);
    uint elems_per_u4 = uint(16 / ELEM_BYTES);
    if (aligned16 && (k_elem + elems_per_u4 <= k_lim)) {
        uint byte_off = (row * stride_elems + k_elem) * uint(ELEM_BYTES);
        return load_a16(Ptr<uint4, Access.Read>(base_b + byte_off));
    }
    uint start = (row * stride_elems + k_elem) * uint(ELEM_BYTES);
    uint max_bytes = min(16u, (k_lim - k_elem) * uint(ELEM_BYTES));
    uint w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
    [ForceUnroll] for (uint b = 0u; b < 16u; ++b) {
        if (b < max_bytes) {
            uint v = uint(base_b[start + b]) << ((b & 3u) << 3u);
            if (b < 4u) w0 |= v;
            else if (b < 8u) w1 |= v;
            else if (b < 12u) w2 |= v;
            else w3 |= v;
        }
    }
    return uint4(w0, w1, w2, w3);
}

interface ICoopSmemTile<let WG_M : int, let WG_N : int, let WARP_M : int, let WARP_N : int> {
    static void storeA(uint idx, uint4 v);
    static void storeB(uint idx, uint4 v);
    static uint4 readA(uint idx);
    static linalg::CoopMat<T, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA>
        loadMatA<T : ICoopElement, let K_STEP : int>(uint off_u4);
    static linalg::CoopMat<T, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB>
        loadMatB<T : ICoopElement, let K_STEP : int>(uint off_u4);
    static void storeAcc<TAcc : ICoopElement>(
        linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> mat,
        uint off_u4);
}

struct CoopSmem128 : ICoopSmemTile<128, 128, 4, 2> {
    static void storeA(uint idx, uint4 v) { s_cm128_a[idx] = v; }
    static void storeB(uint idx, uint4 v) { s_cm128_b[idx] = v; }
    static uint4 readA(uint idx) { return s_cm128_a[idx]; }
    static linalg::CoopMat<T, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA>
        loadMatA<T : ICoopElement, let K_STEP : int>(uint off_u4)
    {
        uint lay = 0u, str = 5u;
        return spirv_asm {
            OpExtension "SPV_KHR_cooperative_matrix";
            OpCapability CooperativeMatrixKHR;
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &s_cm128_a $off_u4;
            result : $$linalg::CoopMat<T, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA> =
                OpCooperativeMatrixLoadKHR %ptr $lay $str;
        };
    }
    static linalg::CoopMat<T, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB>
        loadMatB<T : ICoopElement, let K_STEP : int>(uint off_u4)
    {
        uint lay = 1u, str = 5u;
        return spirv_asm {
            OpExtension "SPV_KHR_cooperative_matrix";
            OpCapability CooperativeMatrixKHR;
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &s_cm128_b $off_u4;
            result : $$linalg::CoopMat<T, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB> =
                OpCooperativeMatrixLoadKHR %ptr $lay $str;
        };
    }
    static void storeAcc<TAcc : ICoopElement>(
        linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> mat,
        uint off_u4)
    {
        uint lay = 0u, str = 4u;
        spirv_asm {
            OpExtension "SPV_KHR_cooperative_matrix";
            OpCapability CooperativeMatrixKHR;
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &s_cm128_a $off_u4;
            OpCooperativeMatrixStoreKHR %ptr $mat $lay $str;
        };
    }
}

struct CoopSmem64 : ICoopSmemTile<64, 64, 2, 2> {
    static void storeA(uint idx, uint4 v) { s_cm64_a[idx] = v; }
    static void storeB(uint idx, uint4 v) { s_cm64_b[idx] = v; }
    static uint4 readA(uint idx) { return s_cm64_a[idx]; }
    static linalg::CoopMat<T, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA>
        loadMatA<T : ICoopElement, let K_STEP : int>(uint off_u4)
    {
        uint lay = 0u, str = 5u;
        return spirv_asm {
            OpExtension "SPV_KHR_cooperative_matrix";
            OpCapability CooperativeMatrixKHR;
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &s_cm64_a $off_u4;
            result : $$linalg::CoopMat<T, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA> =
                OpCooperativeMatrixLoadKHR %ptr $lay $str;
        };
    }
    static linalg::CoopMat<T, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB>
        loadMatB<T : ICoopElement, let K_STEP : int>(uint off_u4)
    {
        uint lay = 1u, str = 5u;
        return spirv_asm {
            OpExtension "SPV_KHR_cooperative_matrix";
            OpCapability CooperativeMatrixKHR;
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &s_cm64_b $off_u4;
            result : $$linalg::CoopMat<T, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB> =
                OpCooperativeMatrixLoadKHR %ptr $lay $str;
        };
    }
    static void storeAcc<TAcc : ICoopElement>(
        linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> mat,
        uint off_u4)
    {
        uint lay = 0u, str = 4u;
        spirv_asm {
            OpExtension "SPV_KHR_cooperative_matrix";
            OpCapability CooperativeMatrixKHR;
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &s_cm64_a $off_u4;
            OpCooperativeMatrixStoreKHR %ptr $mat $lay $str;
        };
    }
}

void subgroup_smem_barrier() {
    uint sc = 3u;
    uint sem = 0x108u; // AcquireRelease | WorkgroupMemory
    spirv_asm { OpControlBarrier $sc $sc $sem; };
}

void write_staged_ep_tile<let IS_QUANT : int, let WG_M : int, let WG_N : int, let WARP_M : int, let WARP_N : int,
                          S : ICoopSmemTile<WG_M, WG_N, WARP_M, WARP_N>>(
    uint base_u4, Ptr<float> y, uint y_stride, uint rows, uint m,
    uint tr, uint tc, uint lane_id, bool y_aligned4, EpilogueParams ep)
{
    if (tr >= rows || tc >= m) return;
    [ForceUnroll] for (uint u = 0u; u < 2u; ++u) {
        uint idx_u4 = u * 32u + lane_id;
        uint gr = tr + (idx_u4 >> 2u);
        uint gc = tc + ((idx_u4 & 3u) << 2u);
        if (gr >= rows) continue;
        uint4 raw = S::readA(base_u4 + idx_u4);
        float4 fv = (IS_QUANT != 0) ? float4(int4(raw)) : asfloat(raw);
        if (y_aligned4 && (gc + 4u <= m)) {
            float4 outv;
            if (IS_QUANT != 0) {
                outv = float4(
                    EpQuantScale::apply(fv.x, gr, gc + 0u, ep),
                    EpQuantScale::apply(fv.y, gr, gc + 1u, ep),
                    EpQuantScale::apply(fv.z, gr, gc + 2u, ep),
                    EpQuantScale::apply(fv.w, gr, gc + 3u, ep));
            } else {
                outv = float4(
                    EpBias::apply(fv.x, gr, gc + 0u, ep),
                    EpBias::apply(fv.y, gr, gc + 1u, ep),
                    EpBias::apply(fv.z, gr, gc + 2u, ep),
                    EpBias::apply(fv.w, gr, gc + 3u, ep));
            }
            store_a16(Ptr<float4>(y + gr * y_stride + gc), outv);
        } else {
            if (IS_QUANT != 0) {
                if (gc + 0u < m) y[gr * y_stride + gc + 0u] = EpQuantScale::apply(fv.x, gr, gc + 0u, ep);
                if (gc + 1u < m) y[gr * y_stride + gc + 1u] = EpQuantScale::apply(fv.y, gr, gc + 1u, ep);
                if (gc + 2u < m) y[gr * y_stride + gc + 2u] = EpQuantScale::apply(fv.z, gr, gc + 2u, ep);
                if (gc + 3u < m) y[gr * y_stride + gc + 3u] = EpQuantScale::apply(fv.w, gr, gc + 3u, ep);
            } else {
                if (gc + 0u < m) y[gr * y_stride + gc + 0u] = EpBias::apply(fv.x, gr, gc + 0u, ep);
                if (gc + 1u < m) y[gr * y_stride + gc + 1u] = EpBias::apply(fv.y, gr, gc + 1u, ep);
                if (gc + 2u < m) y[gr * y_stride + gc + 2u] = EpBias::apply(fv.z, gr, gc + 2u, ep);
                if (gc + 3u < m) y[gr * y_stride + gc + 3u] = EpBias::apply(fv.w, gr, gc + 3u, ep);
            }
        }
    }
}

void coopmat_mma_step<TMat : ICoopElement, TAcc : ICoopElement, let K_STEP : int,
                      let WG_M : int, let WG_N : int, let WARP_M : int, let WARP_N : int,
                      S : ICoopSmemTile<WG_M, WG_N, WARP_M, WARP_N>>(
    inout linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> acc[WARP_M][WARP_N],
    uint wr, uint wc, uint kk_u4)
{
    linalg::CoopMat<TMat, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA> mA[WARP_M];
    linalg::CoopMat<TMat, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB> mB[WARP_N];
    [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
        mA[i] = S::loadMatA<TMat, K_STEP>((wr * uint(WARP_M * 16) + i * 16u) * 5u + kk_u4);
    [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
        mB[j] = S::loadMatB<TMat, K_STEP>((wc * uint(WARP_N * 16) + j * 16u) * 5u + kk_u4);
    [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
        [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
            acc[i][j] = linalg::coopMatMulAdd<TAcc, false>(mA[i], mB[j], acc[i][j]);
}

void coopmat_gemm_smem<TMat : ICoopElement, TAcc : ICoopElement,
                       let K_STEP : int, let ELEM_BYTES : int, let IS_QUANT : int,
                       let WG_M : int, let WG_N : int, let WARP_M : int, let WARP_N : int,
                       let NTHREADS : int,
                       S : ICoopSmemTile<WG_M, WG_N, WARP_M, WARP_N>>(
    CoopGemmArgs<TMat, float> a, TAcc zero_acc, uint3 gtid, uint3 gid)
{
    uint tid = gtid.x;
    uint warp_id = tid >> 5u;
    uint lane_id = tid & 31u;
    uint n_col_warps = uint(WG_N / (WARP_N * 16));
    uint wr = warp_id / n_col_warps;
    uint wc = warp_id - wr * n_col_warps;

    uint wg_m_u = uint(WG_M);
    uint wg_n_u = uint(WG_N);
    uint grid_x = (a.m + wg_n_u - 1u) / wg_n_u;
    uint grid_y = (a.rows + wg_m_u - 1u) / wg_m_u;
    uint2 tile = swizzle_tile_2d(gid, grid_x, grid_y);
    uint row0 = tile.y * wg_m_u;
    uint col0 = tile.x * wg_n_u;

    linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> acc[WARP_M][WARP_N];
    [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
        [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
            acc[i][j] = linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator>(zero_acc);

    Ptr<uint8_t, Access.Read> xb = Ptr<uint8_t, Access.Read>(a.x);
    Ptr<uint8_t, Access.Read> wb = Ptr<uint8_t, Access.Read>(a.w);
    uint elems_per_u4 = uint(16 / ELEM_BYTES);
    uint bk_elems = elems_per_u4 * 4u;
    uint num_k_tiles = (a.k + bk_elems - 1u) / bk_elems;
    bool aligned16 = is_ptr_aligned16(a.x) && is_ptr_aligned16(a.w) &&
                     (((a.x_stride * uint(ELEM_BYTES)) & 15u) == 0u) &&
                     (((a.w_stride * uint(ELEM_BYTES)) & 15u) == 0u);
    bool fast_interior = aligned16 && ((a.k & (bk_elems - 1u)) == 0u) &&
                         (row0 + wg_m_u <= a.rows) && (col0 + wg_n_u <= a.m);

    static const uint kVecA = uint((WG_M * 4) / NTHREADS);
    static const uint kVecB = uint((WG_N * 4) / NTHREADS);
    static const uint kRowStep = uint(NTHREADS / 4);
    uint lr0 = tid >> 2u;
    uint lu0 = tid & 3u;

    if (fast_interior && num_k_tiles > 0u) {
        Ptr<uint4, Access.Read> xp0 = Ptr<uint4, Access.Read>(xb + ((row0 + lr0) * a.x_stride + lu0 * elems_per_u4) * uint(ELEM_BYTES));
        Ptr<uint4, Access.Read> wp0 = Ptr<uint4, Access.Read>(wb + ((col0 + lr0) * a.w_stride + lu0 * elems_per_u4) * uint(ELEM_BYTES));
        uint x_step_u4 = (kRowStep * a.x_stride) / elems_per_u4;
        uint w_step_u4 = (kRowStep * a.w_stride) / elems_per_u4;
        uint s0 = lr0 * 5u + lu0;

        [ForceUnroll] for (uint v = 0u; v < kVecA; ++v)
            S::storeA(s0 + v * (kRowStep * 5u), load_a16(xp0 + v * x_step_u4));
        [ForceUnroll] for (uint v = 0u; v < kVecB; ++v)
            S::storeB(s0 + v * (kRowStep * 5u), load_a16(wp0 + v * w_step_u4));
        GroupMemoryBarrierWithGroupSync();

        for (uint kt = 0u; kt < num_k_tiles; ++kt) {
            uint4 pA[kVecA];
            uint4 pB[kVecB];
            [ForceUnroll] for (uint v = 0u; v < kVecA; ++v) pA[v] = uint4(0u);
            [ForceUnroll] for (uint v = 0u; v < kVecB; ++v) pB[v] = uint4(0u);
            bool has_next = (kt + 1u < num_k_tiles);
            if (has_next) {
                uint k_u4 = (kt + 1u) << 2u;
                [ForceUnroll] for (uint v = 0u; v < kVecA; ++v)
                    pA[v] = load_a16(xp0 + (v * x_step_u4 + k_u4));
                [ForceUnroll] for (uint v = 0u; v < kVecB; ++v)
                    pB[v] = load_a16(wp0 + (v * w_step_u4 + k_u4));
            }
            // Step ks=0: load + MMA (hides LDG.128 latency of pA/pB)
            coopmat_mma_step<TMat, TAcc, K_STEP, WG_M, WG_N, WARP_M, WARP_N, S>(acc, wr, wc, 0u);
            // Step ks=1: load mA1/mB1 BEFORE barrier so SMEM is free immediately!
            linalg::CoopMat<TMat, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA> mA1[WARP_M];
            linalg::CoopMat<TMat, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB> mB1[WARP_N];
            [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
                mA1[i] = S::loadMatA<TMat, K_STEP>((wr * uint(WARP_M * 16) + i * 16u) * 5u + 2u);
            [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
                mB1[j] = S::loadMatB<TMat, K_STEP>((wc * uint(WARP_N * 16) + j * 16u) * 5u + 2u);
            if (has_next) {
                GroupMemoryBarrierWithGroupSync();
                [ForceUnroll] for (uint v = 0u; v < kVecA; ++v)
                    S::storeA(s0 + v * (kRowStep * 5u), pA[v]);
                [ForceUnroll] for (uint v = 0u; v < kVecB; ++v)
                    S::storeB(s0 + v * (kRowStep * 5u), pB[v]);
            }
            // Overlap ks=1 MMAs with STS.128 writes (zero extra registers, 20 KiB = 2 WGs/SM!)
            [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
                [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
                    acc[i][j] = linalg::coopMatMulAdd<TAcc, false>(mA1[i], mB1[j], acc[i][j]);
            if (has_next) {
                GroupMemoryBarrierWithGroupSync();
            }
        }
    } else {
        for (uint kt = 0u; kt < num_k_tiles; ++kt) {
            uint k0 = kt * bk_elems;
            [ForceUnroll] for (uint v = 0u; v < kVecA; ++v) {
                uint lr = lr0 + v * kRowStep;
                S::storeA(lr * 5u + lu0, load_u4_safe<ELEM_BYTES>(xb, row0 + lr, a.rows, k0 + lu0 * elems_per_u4, a.k, a.x_stride, aligned16));
            }
            [ForceUnroll] for (uint v = 0u; v < kVecB; ++v) {
                uint lr = lr0 + v * kRowStep;
                S::storeB(lr * 5u + lu0, load_u4_safe<ELEM_BYTES>(wb, col0 + lr, a.m, k0 + lu0 * elems_per_u4, a.k, a.w_stride, aligned16));
            }
            GroupMemoryBarrierWithGroupSync();
            coopmat_mma_step<TMat, TAcc, K_STEP, WG_M, WG_N, WARP_M, WARP_N, S>(acc, wr, wc, 0u);
            coopmat_mma_step<TMat, TAcc, K_STEP, WG_M, WG_N, WARP_M, WARP_N, S>(acc, wr, wc, 2u);
            GroupMemoryBarrierWithGroupSync();
        }
    }

    EpilogueParams ep = make_vk_ep(a.bias, a.x_scale, a.w_scale, a.flags);
    bool y_aligned4 = is_ptr_aligned16(a.y) && ((a.y_stride & 3u) == 0u);
    if ((IS_QUANT == 0) && (a.flags == 0u) && y_aligned4 && (row0 + wg_m_u <= a.rows) && (col0 + wg_n_u <= a.m)) {
        Ptr<TAcc> y_direct = Ptr<TAcc>(a.y);
        [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i) {
            [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j) {
                uint tr = row0 + wr * uint(WARP_M * 16) + i * 16u;
                uint tc = col0 + wc * uint(WARP_N * 16) + j * 16u;
                acc[i][j].Store<linalg::CoopMatMatrixLayout.RowMajor>(y_direct, tr * a.y_stride + tc, a.y_stride);
            }
        }
    } else {
        // Reuse s_cm_a for warp-private vectorized float4 epilogue (ZERO cross-warp stalls!).
        GroupMemoryBarrierWithGroupSync();
        uint warp_ep_base = warp_id * 64u;
        [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i) {
            uint tr = row0 + wr * uint(WARP_M * 16) + i * 16u;
            [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j) {
                uint tc = col0 + wc * uint(WARP_N * 16) + j * 16u;
                if (tr < a.rows && tc < a.m) {
                    S::storeAcc<TAcc>(acc[i][j], warp_ep_base);
                    subgroup_smem_barrier();
                    write_staged_ep_tile<IS_QUANT, WG_M, WG_N, WARP_M, WARP_N, S>(
                        warp_ep_base, a.y, a.y_stride, a.rows, a.m, tr, tc, lane_id, y_aligned4, ep);
                    subgroup_smem_barrier();
                }
            }
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_coopmat_f16(uniform CoopGemmArgs<half, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<half, float, 16, 2, 0, 128, 128, 4, 2, 256, CoopSmem128>(a, 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_coopmat_bf16(uniform CoopGemmArgs<BFloat16, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<BFloat16, float, 16, 2, 0, 128, 128, 4, 2, 256, CoopSmem128>(a, 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_coopmat_w8a8(uniform CoopGemmArgs<int8_t, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<int8_t, int32_t, 32, 1, 1, 128, 128, 4, 2, 256, CoopSmem128>(a, 0, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemm_coopmat_f16_64(uniform CoopGemmArgs<half, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<half, float, 16, 2, 0, 64, 64, 2, 2, 128, CoopSmem64>(a, 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemm_coopmat_bf16_64(uniform CoopGemmArgs<BFloat16, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<BFloat16, float, 16, 2, 0, 64, 64, 2, 2, 128, CoopSmem64>(a, 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemm_coopmat_w8a8_64(uniform CoopGemmArgs<int8_t, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<int8_t, int32_t, 32, 1, 1, 64, 64, 2, 2, 128, CoopSmem64>(a, 0, gtid, gid);
}

// ============================================================================
// 7. Hardware Ceiling Calibration Shaders (100% Coalesced Read/Copy BW, FMA, CoopMat)
// ============================================================================
struct CeilingBwArgs {
    Ptr<float4, Access.Read> src;
    Ptr<float4> dst;
    uint count_vec4;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_bw_f32x4(uniform CeilingBwArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint base = gid.x * 1024u + gtid.x;
    [ForceUnroll] for (uint u = 0u; u < 4u; ++u) {
        uint i = base + u * 256u;
        if (i < a.count_vec4) {
            store_a16(a.dst + i, load_a16(a.src + i));
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_bw_read_f32x4(uniform CeilingBwArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    // Grid-stride 2048-WG read-only sweep: zero sink cache-line contention!
    uint stride = 2048u * 256u;
    float4 s = float4(0.0f);
    for (uint i = gid.x * 256u + gtid.x; i < a.count_vec4; i += stride) {
        s += load_a16(a.src + i);
    }
    float r = WaveActiveSum(s.x + s.y + s.z + s.w);
    if (WaveGetLaneIndex() == 0u) {
        uint lane_cnt = WaveGetLaneCount();
        uint warp_global = gid.x * (256u / lane_cnt) + (gtid.x / lane_cnt);
        Ptr<float> sink = Ptr<float>(a.dst);
        sink[warp_global] = r;
    }
}

struct CeilingComputeArgs {
    Ptr<float> sink;
    uint iters;
    float seed;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_fma_f32(uniform CeilingComputeArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    float base = a.seed + float(gtid.x) * 1e-4f;
    float4 c0 = float4(base + 0.01f), c1 = float4(base + 0.02f);
    float4 c2 = float4(base + 0.03f), c3 = float4(base + 0.04f);
    float4 m0 = float4(0.999991f), m1 = float4(0.000013f);
    for (uint i = 0u; i < a.iters; ++i) {
        [ForceUnroll] for (uint u = 0u; u < 16u; ++u) {
            c0 = fma(c0, m0, m1);
            c1 = fma(c1, m0, m1);
            c2 = fma(c2, m0, m1);
            c3 = fma(c3, m0, m1);
        }
    }
    float4 s = c0 + c1 + c2 + c3;
    float r = WaveActiveSum(s.x + s.y + s.z + s.w);
    if (WaveGetLaneIndex() == 0u) {
        uint lane_cnt = WaveGetLaneCount();
        uint warp_global = gid.x * (256u / lane_cnt) + (gtid.x / lane_cnt);
        a.sink[warp_global & 4095u] = r;
    }
}

void ceiling_coopmat_impl<TMat : ICoopElement, TAcc : ICoopElement, let K_STEP : int>(
    CeilingComputeArgs a, TMat va, TMat vb, TAcc zero_acc, uint3 gtid, uint3 gid)
{
    var mA = linalg::CoopMat<TMat, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA>(va);
    var mB = linalg::CoopMat<TMat, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB>(vb);
    linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> c[8];
    [ForceUnroll] for (uint j = 0u; j < 8u; ++j)
        c[j] = linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator>(zero_acc);
    for (uint i = 0u; i < a.iters; ++i) {
        [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
            c[j] = linalg::coopMatMulAdd<TAcc, false>(mA, mB, c[j]);
        }
    }
    // Store all 8 accumulators per warp into sink (128 KiB ring) so neither DCE nor loop sinking is possible.
    uint warp_global = gid.x * 8u + (gtid.x >> 5u);
    Ptr<TAcc> sink_ptr = Ptr<TAcc>(a.sink);
    [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
        c[j].Store<linalg::CoopMatMatrixLayout.RowMajor>(sink_ptr, ((warp_global & 15u) * 8u + j) * 256u, 16u);
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_coopmat_f16(uniform CeilingComputeArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    ceiling_coopmat_impl<half, float, 16>(a, half(a.seed), half(0.999f), 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_coopmat_bf16(uniform CeilingComputeArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    ceiling_coopmat_impl<BFloat16, float, 16>(a, BFloat16(a.seed), BFloat16(0.999f), 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_coopmat_i8(uniform CeilingComputeArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    int s = int(a.seed);
    ceiling_coopmat_impl<int8_t, int32_t, 32>(a, int8_t(s), int8_t(s + 1), int32_t(0), gtid, gid);
}
