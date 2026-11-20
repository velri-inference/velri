// Unified generic CPU Linear (GEMV + L1-packed BLIS outer-product GEMM) in Slang.
// ONE parameterized source compiles across baseline, AVX2, and AVX-512 via
// compiler flags (-llvm-features and -DVELRI_MR/NR_F32/NR_F64) and across
// F32, F16->F32, BF16->F32, F32xBF16, F64, and exact-int32 W8A8->F32!
// Reuses common/numeric.slang and common/epilogue.slang.
module linear_cpu;

import numeric;
import epilogue;

#ifndef VELRI_KC
#define VELRI_KC 128
#endif
static const uint kKC = uint(VELRI_KC);

// ============================================================================
// 1. Streaming GEMV (rows < 4): 4-col register blocking, zero packing overhead
// ============================================================================
void gemv_4rows<TX : IElement, TW : IElement, TY : IElement, Ep : IEpilogue>(
    Ptr<TX.Storage, Access.Read> xr,
    Ptr<TW.Storage, Access.Read> w, uint w_stride,
    Ptr<TY.Storage> yr,
    uint row, uint col, uint k, EpilogueParams ep)
{
    Ptr<TW.Storage, Access.Read> w0 = w + (col + 0u) * w_stride;
    Ptr<TW.Storage, Access.Read> w1 = w + (col + 1u) * w_stride;
    Ptr<TW.Storage, Access.Read> w2 = w + (col + 2u) * w_stride;
    Ptr<TW.Storage, Access.Read> w3 = w + (col + 3u) * w_stride;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint i = 0u; i < k; ++i) {
        float xv = TX::toF32(xr[i]);
        s0 += xv * TW::toF32(w0[i]);
        s1 += xv * TW::toF32(w1[i]);
        s2 += xv * TW::toF32(w2[i]);
        s3 += xv * TW::toF32(w3[i]);
    }
    yr[col + 0u] = TY::fromF32(Ep::apply(s0, row, col + 0u, ep));
    yr[col + 1u] = TY::fromF32(Ep::apply(s1, row, col + 1u, ep));
    yr[col + 2u] = TY::fromF32(Ep::apply(s2, row, col + 2u, ep));
    yr[col + 3u] = TY::fromF32(Ep::apply(s3, row, col + 3u, ep));
}

void gemv_1row<TX : IElement, TW : IElement, TY : IElement, Ep : IEpilogue>(
    Ptr<TX.Storage, Access.Read> xr,
    Ptr<TW.Storage, Access.Read> wc,
    Ptr<TY.Storage> yr,
    uint row, uint col, uint k, EpilogueParams ep)
{
    float s = 0.0f;
    for (uint i = 0u; i < k; ++i) s += TX::toF32(xr[i]) * TW::toF32(wc[i]);
    yr[col] = TY::fromF32(Ep::apply(s, row, col, ep));
}

// ============================================================================
// 2. Scatter-Free & Gather-Free Packers:
//    - pack_b_f32/f64: unaligned-safe 4x4 block transpose -> aligned Ptr<float4/double4>
//      stores (emits pure vunpck/vshuf + vmovaps, ZERO vgather and ZERO vscatter!).
//    - pack_a_f32/f64: contiguous row-major copy/convert into ap[MR * kKC] (ZERO transpose!).
// ============================================================================
float4 load_f32x4<T : IElement>(Ptr<T.Storage, Access.Read> p) {
    return T::load4F32(p);
}

void pack_b_f32<T : IElement, let NR : int>(
    Ptr<T.Storage, Access.Read> src, uint stride, uint col0, uint nr,
    uint k0, uint kc, Ptr<float4> dst4)
{
    static const uint W4 = uint(NR) >> 2u;
    Ptr<float> dst = Ptr<float>(dst4);
    uint j = 0u;
    for (; j + 4u <= nr; j += 4u) {
        Ptr<T.Storage, Access.Read> s0 = src + (col0 + j + 0u) * stride + k0;
        Ptr<T.Storage, Access.Read> s1 = src + (col0 + j + 1u) * stride + k0;
        Ptr<T.Storage, Access.Read> s2 = src + (col0 + j + 2u) * stride + k0;
        Ptr<T.Storage, Access.Read> s3 = src + (col0 + j + 3u) * stride + k0;
        uint j4 = j >> 2u;
        uint p = 0u;
        for (; p + 4u <= kc; p += 4u) {
            float4 r0 = load_f32x4<T>(s0 + p);
            float4 r1 = load_f32x4<T>(s1 + p);
            float4 r2 = load_f32x4<T>(s2 + p);
            float4 r3 = load_f32x4<T>(s3 + p);
            dst4[(p + 0u) * W4 + j4] = float4(r0.x, r1.x, r2.x, r3.x);
            dst4[(p + 1u) * W4 + j4] = float4(r0.y, r1.y, r2.y, r3.y);
            dst4[(p + 2u) * W4 + j4] = float4(r0.z, r1.z, r2.z, r3.z);
            dst4[(p + 3u) * W4 + j4] = float4(r0.w, r1.w, r2.w, r3.w);
        }
        for (; p < kc; ++p) {
            dst4[p * W4 + j4] = float4(T::toF32(s0[p]), T::toF32(s1[p]),
                                       T::toF32(s2[p]), T::toF32(s3[p]));
        }
    }
    for (; j < nr; ++j) {
        Ptr<T.Storage, Access.Read> sj = src + (col0 + j) * stride + k0;
        for (uint p = 0u; p < kc; ++p) dst[p * uint(NR) + j] = T::toF32(sj[p]);
    }
    if (nr < uint(NR)) {
        for (uint p = 0u; p < kc; ++p) {
            for (uint jj = nr; jj < uint(NR); ++jj) dst[p * uint(NR) + jj] = 0.0f;
        }
    }
}

void pack_a_f32<T : IElement, let MR : int>(
    Ptr<T.Storage, Access.Read> src, uint stride, uint row0, uint mr,
    uint k0, uint kc, Ptr<float> dst)
{
    for (uint r = 0u; r < mr; ++r) {
        Ptr<T.Storage, Access.Read> sr = src + (row0 + r) * stride + k0;
        Ptr<float> dr = dst + r * kKC;
        for (uint p = 0u; p < kc; ++p) dr[p] = T::toF32(sr[p]);
    }
    if (mr < uint(MR)) {
        for (uint r = mr; r < uint(MR); ++r) {
            Ptr<float> dr = dst + r * kKC;
            for (uint p = 0u; p < kc; ++p) dr[p] = 0.0f;
        }
    }
}

void pack_b_f64<let NR : int>(
    Ptr<double, Access.Read> src, uint stride, uint col0, uint nr,
    uint k0, uint kc, Ptr<double4> dst4)
{
    static const uint W4 = uint(NR) >> 2u;
    Ptr<double> dst = Ptr<double>(dst4);
    uint j = 0u;
    for (; j + 4u <= nr; j += 4u) {
        Ptr<double, Access.Read> s0 = src + (col0 + j + 0u) * stride + k0;
        Ptr<double, Access.Read> s1 = src + (col0 + j + 1u) * stride + k0;
        Ptr<double, Access.Read> s2 = src + (col0 + j + 2u) * stride + k0;
        Ptr<double, Access.Read> s3 = src + (col0 + j + 3u) * stride + k0;
        uint j4 = j >> 2u;
        uint p = 0u;
        for (; p + 4u <= kc; p += 4u) {
            double4 r0 = double4(s0[p], s0[p+1u], s0[p+2u], s0[p+3u]);
            double4 r1 = double4(s1[p], s1[p+1u], s1[p+2u], s1[p+3u]);
            double4 r2 = double4(s2[p], s2[p+1u], s2[p+2u], s2[p+3u]);
            double4 r3 = double4(s3[p], s3[p+1u], s3[p+2u], s3[p+3u]);
            dst4[(p + 0u) * W4 + j4] = double4(r0.x, r1.x, r2.x, r3.x);
            dst4[(p + 1u) * W4 + j4] = double4(r0.y, r1.y, r2.y, r3.y);
            dst4[(p + 2u) * W4 + j4] = double4(r0.z, r1.z, r2.z, r3.z);
            dst4[(p + 3u) * W4 + j4] = double4(r0.w, r1.w, r2.w, r3.w);
        }
        for (; p < kc; ++p) {
            dst4[p * W4 + j4] = double4(s0[p], s1[p], s2[p], s3[p]);
        }
    }
    for (; j < nr; ++j) {
        Ptr<double, Access.Read> sj = src + (col0 + j) * stride + k0;
        for (uint p = 0u; p < kc; ++p) dst[p * uint(NR) + j] = sj[p];
    }
    if (nr < uint(NR)) {
        for (uint p = 0u; p < kc; ++p) {
            for (uint jj = nr; jj < uint(NR); ++jj) dst[p * uint(NR) + jj] = 0.0;
        }
    }
}

void pack_a_f64<let MR : int>(
    Ptr<double, Access.Read> src, uint stride, uint row0, uint mr,
    uint k0, uint kc, Ptr<double> dst)
{
    for (uint r = 0u; r < mr; ++r) {
        Ptr<double, Access.Read> sr = src + (row0 + r) * stride + k0;
        Ptr<double> dr = dst + r * kKC;
        for (uint p = 0u; p < kc; ++p) dr[p] = sr[p];
    }
    if (mr < uint(MR)) {
        for (uint r = mr; r < uint(MR); ++r) {
            Ptr<double> dr = dst + r * kKC;
            for (uint p = 0u; p < kc; ++p) dr[p] = 0.0;
        }
    }
}

// ============================================================================
// 3. Unified Zero-Spill L1-Packed BLIS Outer-Product Microkernels
//    - AVX-512: MR=6, NR=64 F32 / 32 F64 (24 ZMM acc + 6 A + 1 B = 31/32 ZMMs!)
//    - AVX2   : MR=3, NR=32 F32 / 16 F64 (12 YMM acc + 3 A + 1 B = 16/16 YMMs!)
//    - Base   : MR=3, NR=16 F32 /  8 F64 (12 XMM acc + 3 A + 1 B = 16/16 XMMs!)
//    - Shared by F32, F16->F32, BF16->F32, F32xBF16, AND exact-int32 W8A8->F32!
// ============================================================================
void gemm_mk_f32<let MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<float, Access.Read> ap, uint lda, Ptr<float> bp,
    Ptr<float> c_tile, uint ldc, uint kc)
{
    float c[MR * NR];
    if (ZERO_INIT) {
        for (uint i = 0u; i < uint(MR * NR); ++i) c[i] = 0.0f;
    } else {
        for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<float> cr = c_tile + r * ldc;
            for (uint j = 0u; j < uint(NR); ++j) c[r * uint(NR) + j] = cr[j];
        }
    }
    for (uint p = 0u; p < kc; ++p) {
        Ptr<float> brow = bp + p * uint(NR);
        float a[MR];
        [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = ap[r * lda + p];
        for (uint j = 0u; j < uint(NR); ++j) {
            float b = brow[j];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                c[r * uint(NR) + j] += a[r] * b;
            }
        }
    }
    for (uint r = 0u; r < uint(MR); ++r) {
        Ptr<float> cr = c_tile + r * ldc;
        for (uint j = 0u; j < uint(NR); ++j) cr[j] = c[r * uint(NR) + j];
    }
}

void gemm_mk_f64<let MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<double, Access.Read> ap, uint lda, Ptr<double> bp,
    Ptr<double> c_tile, uint ldc, uint kc)
{
    if (NR >= 32) {
        double c[MR * NR];
        if (ZERO_INIT) {
            for (uint i = 0u; i < uint(MR * NR); ++i) c[i] = 0.0;
        } else {
            for (uint r = 0u; r < uint(MR); ++r) {
                Ptr<double> cr = c_tile + r * ldc;
                for (uint j = 0u; j < uint(NR); ++j) c[r * uint(NR) + j] = cr[j];
            }
        }
        for (uint p = 0u; p < kc; ++p) {
            Ptr<double> brow = bp + p * uint(NR);
            double a[MR];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = ap[r * lda + p];
            for (uint j = 0u; j < uint(NR); ++j) {
                double b = brow[j];
                [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                    c[r * uint(NR) + j] += a[r] * b;
                }
            }
        }
        for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<double> cr = c_tile + r * ldc;
            for (uint j = 0u; j < uint(NR); ++j) cr[j] = c[r * uint(NR) + j];
        }
    } else {
        // Explicit 4-wide double4 (= 1 full YMM on AVX2 / 2 XMM on base) prevents
        // LLVM FullLoopUnroll from scalarizing NR=8/16 FP64 microkernels!
        static const uint W4 = uint(NR) >> 2u;
        double4 c[MR * (NR / 4)];
        if (ZERO_INIT) {
            [ForceUnroll] for (uint i = 0u; i < uint(MR) * W4; ++i) c[i] = double4(0.0);
        } else {
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                Ptr<double> cr = c_tile + r * ldc;
                [ForceUnroll] for (uint j = 0u; j < W4; ++j)
                    c[r * W4 + j] = double4(cr[j*4u], cr[j*4u+1u], cr[j*4u+2u], cr[j*4u+3u]);
            }
        }
        Ptr<double4> bp4 = Ptr<double4>(bp);
        for (uint p = 0u; p < kc; ++p) {
            Ptr<double4> brow4 = bp4 + p * W4;
            double a[MR];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = ap[r * lda + p];
            [ForceUnroll] for (uint j = 0u; j < W4; ++j) {
                double4 b = brow4[j];
                [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                    c[r * W4 + j] += a[r] * b;
                }
            }
        }
        [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<double> cr = c_tile + r * ldc;
            [ForceUnroll] for (uint j = 0u; j < W4; ++j) {
                double4 v = c[r * W4 + j];
                cr[j*4u+0u] = v.x; cr[j*4u+1u] = v.y; cr[j*4u+2u] = v.z; cr[j*4u+3u] = v.w;
            }
        }
    }
}

// ============================================================================
// 4. Unified Floating-Point Linear (F32, F16->F32, BF16->F32, F32xBF16)
// ============================================================================
public void linear_slice_cpu<let MR : int, let NR : int,
                             TX : IElement, TW : IElement, TY : IElement, Ep : IEpilogue>(
    Ptr<TX.Storage, Access.Read> x, uint x_stride,
    Ptr<TW.Storage, Access.Read> w, uint w_stride,
    Ptr<TY.Storage> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep)
{
    uint nrows = r1 - r0;
    if (nrows < 4u) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<TX.Storage, Access.Read> xr = x + r * x_stride;
            Ptr<TY.Storage> yr = y + r * y_stride;
            uint c = c0;
            for (; c + 4u <= c1; c += 4u) {
                gemv_4rows<TX, TW, TY, Ep>(xr, w, w_stride, yr, r, c, k, ep);
            }
            for (; c < c1; ++c) {
                gemv_1row<TX, TW, TY, Ep>(xr, w + c * w_stride, yr, r, c, k, ep);
            }
        }
        return;
    }

    static const uint kMC = 264u; // multiple of 6 & 3, covers rows=256 in 1 pass (66 KiB)
    float4 bp4[VELRI_KC * (NR / 4)];
    float ap[MR * VELRI_KC];
    float c_buf[264 * NR];
    bp4[0] = float4(0.0f); ap[0] = 0.0f; c_buf[0] = 0.0f;
    Ptr<float> bp = Ptr<float>(&bp4[0]);
    Ptr<float, Access.Read> ap_rd = Ptr<float, Access.Read>(&ap[0]);

    for (uint cb = c0; cb < c1; cb += uint(NR)) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1; rb += kMC) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0u; k0 < k; k0 += kKC) {
                uint kc = min(kKC, k - k0);
                pack_b_f32<TW, NR>(w, w_stride, cb, nr, k0, kc, &bp4[0]);
                if (k0 == 0u) {
                    for (uint rm = 0u; rm < mc; rm += uint(MR)) {
                        uint mr = min(uint(MR), mc - rm);
                        if (sizeof(TX.Storage) == 4 && mr == uint(MR)) {
                            Ptr<float, Access.Read> xp = Ptr<float, Access.Read>(x) + (rb + rm) * x_stride + k0;
                            gemm_mk_f32<MR, NR, true>(xp, x_stride, bp, &c_buf[rm * uint(NR)], uint(NR), kc);
                        } else {
                            pack_a_f32<TX, MR>(x, x_stride, rb + rm, mr, k0, kc, &ap[0]);
                            gemm_mk_f32<MR, NR, true>(ap_rd, kKC, bp, &c_buf[rm * uint(NR)], uint(NR), kc);
                        }
                    }
                } else {
                    for (uint rm = 0u; rm < mc; rm += uint(MR)) {
                        uint mr = min(uint(MR), mc - rm);
                        if (sizeof(TX.Storage) == 4 && mr == uint(MR)) {
                            Ptr<float, Access.Read> xp = Ptr<float, Access.Read>(x) + (rb + rm) * x_stride + k0;
                            gemm_mk_f32<MR, NR, false>(xp, x_stride, bp, &c_buf[rm * uint(NR)], uint(NR), kc);
                        } else {
                            pack_a_f32<TX, MR>(x, x_stride, rb + rm, mr, k0, kc, &ap[0]);
                            gemm_mk_f32<MR, NR, false>(ap_rd, kKC, bp, &c_buf[rm * uint(NR)], uint(NR), kc);
                        }
                    }
                }
            }
            for (uint rm = 0u; rm < mc; ++rm) {
                uint rr = rb + rm;
                Ptr<TY.Storage> yr = y + rr * y_stride + cb;
                for (uint j = 0u; j < nr; ++j) {
                    yr[j] = TY::fromF32(Ep::apply(c_buf[rm * uint(NR) + j], rr, cb + j, ep));
                }
            }
        }
    }
}

// ============================================================================
// 5. Exact INT8 * INT8 -> INT32 (modulo 2^32) & W8A8 -> F32
//    REUSES gemm_mk_f32<MR, NR> over KC=256 blocks (|sum| <= 2^22 << 2^24 exact!).
// ============================================================================
public void linear_w8a8_f32_slice_cpu<let MR : int, let NR : int>(
    Ptr<int8_t, Access.Read> x, uint x_stride,
    Ptr<int8_t, Access.Read> w, uint w_stride,
    Ptr<float> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep)
{
    uint nrows = r1 - r0;
    if (nrows < 4u) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<int8_t, Access.Read> xr = x + r * x_stride;
            Ptr<float> yr = y + r * y_stride;
            uint c = c0;
            for (; c + 4u <= c1; c += 4u) {
                Ptr<int8_t, Access.Read> w0 = w + (c + 0u) * w_stride;
                Ptr<int8_t, Access.Read> w1 = w + (c + 1u) * w_stride;
                Ptr<int8_t, Access.Read> w2 = w + (c + 2u) * w_stride;
                Ptr<int8_t, Access.Read> w3 = w + (c + 3u) * w_stride;
                uint32_t s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
                for (uint i = 0u; i < k; ++i) {
                    int16_t xv = int16_t(xr[i]);
                    s0 += uint32_t(int32_t(xv * int16_t(w0[i])));
                    s1 += uint32_t(int32_t(xv * int16_t(w1[i])));
                    s2 += uint32_t(int32_t(xv * int16_t(w2[i])));
                    s3 += uint32_t(int32_t(xv * int16_t(w3[i])));
                }
                yr[c + 0u] = EpQuantScale::apply(float(int32_t(s0)), r, c + 0u, ep);
                yr[c + 1u] = EpQuantScale::apply(float(int32_t(s1)), r, c + 1u, ep);
                yr[c + 2u] = EpQuantScale::apply(float(int32_t(s2)), r, c + 2u, ep);
                yr[c + 3u] = EpQuantScale::apply(float(int32_t(s3)), r, c + 3u, ep);
            }
            for (; c < c1; ++c) {
                Ptr<int8_t, Access.Read> wc = w + c * w_stride;
                uint32_t s = 0u;
                for (uint i = 0u; i < k; ++i) s += uint32_t(int32_t(int16_t(xr[i]) * int16_t(wc[i])));
                yr[c] = EpQuantScale::apply(float(int32_t(s)), r, c, ep);
            }
        }
        return;
    }

    static const uint kMC = 264u;
    float4 bp4[VELRI_KC * (NR / 4)];
    float ap[MR * VELRI_KC];
    float c_tmp[MR * NR];
    uint32_t c_i32[264 * NR];
    bp4[0] = float4(0.0f); ap[0] = 0.0f; c_tmp[0] = 0.0f; c_i32[0] = 0u;
    Ptr<float> bp = Ptr<float>(&bp4[0]);

    for (uint cb = c0; cb < c1; cb += uint(NR)) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1; rb += kMC) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0u; k0 < k; k0 += kKC) {
                uint kc = min(kKC, k - k0);
                pack_b_f32<ElemI8, NR>(w, w_stride, cb, nr, k0, kc, &bp4[0]);
                bool first = (k0 == 0u);
                for (uint rm = 0u; rm < mc; rm += uint(MR)) {
                    uint mr = min(uint(MR), mc - rm);
                    pack_a_f32<ElemI8, MR>(x, x_stride, rb + rm, mr, k0, kc, &ap[0]);
                    gemm_mk_f32<MR, NR, true>(Ptr<float, Access.Read>(&ap[0]), kKC, bp, &c_tmp[0], uint(NR), kc);
                    if (first) {
                        for (uint r = 0u; r < mr; ++r) {
                            uint dst_off = (rm + r) * uint(NR);
                            uint src_off = r * uint(NR);
                            for (uint j = 0u; j < uint(NR); ++j)
                                c_i32[dst_off + j] = uint32_t(int32_t(c_tmp[src_off + j]));
                        }
                    } else {
                        for (uint r = 0u; r < mr; ++r) {
                            uint dst_off = (rm + r) * uint(NR);
                            uint src_off = r * uint(NR);
                            for (uint j = 0u; j < uint(NR); ++j)
                                c_i32[dst_off + j] += uint32_t(int32_t(c_tmp[src_off + j]));
                        }
                    }
                }
            }
            for (uint rm = 0u; rm < mc; ++rm) {
                uint rr = rb + rm;
                Ptr<float> yr = y + rr * y_stride + cb;
                for (uint j = 0u; j < nr; ++j) {
                    yr[j] = EpQuantScale::apply(float(int32_t(c_i32[rm * uint(NR) + j])), rr, cb + j, ep);
                }
            }
        }
    }
}

// ============================================================================
// 6. Native FP64 Linear (Streaming GEMV + Scatter-Free L1-Packed BLIS GEMM)
// ============================================================================
public void linear_f64_slice_cpu<let MR : int, let NR : int>(
    Ptr<double, Access.Read> x, uint x_stride,
    Ptr<double, Access.Read> w, uint w_stride,
    Ptr<double, Access.Read> bias, uint has_bias,
    Ptr<double> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    uint nrows = r1 - r0;
    if (nrows < 4u) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<double, Access.Read> xr = x + r * x_stride;
            Ptr<double> yr = y + r * y_stride;
            uint c = c0;
            for (; c + 4u <= c1; c += 4u) {
                Ptr<double, Access.Read> w0 = w + (c + 0u) * w_stride;
                Ptr<double, Access.Read> w1 = w + (c + 1u) * w_stride;
                Ptr<double, Access.Read> w2 = w + (c + 2u) * w_stride;
                Ptr<double, Access.Read> w3 = w + (c + 3u) * w_stride;
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (uint i = 0u; i < k; ++i) {
                    double xv = xr[i];
                    s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
                }
                yr[c + 0u] = s0 + (has_bias != 0u ? bias[c + 0u] : 0.0);
                yr[c + 1u] = s1 + (has_bias != 0u ? bias[c + 1u] : 0.0);
                yr[c + 2u] = s2 + (has_bias != 0u ? bias[c + 2u] : 0.0);
                yr[c + 3u] = s3 + (has_bias != 0u ? bias[c + 3u] : 0.0);
            }
            for (; c < c1; ++c) {
                Ptr<double, Access.Read> wc = w + c * w_stride;
                double s = 0.0;
                for (uint i = 0u; i < k; ++i) s += xr[i] * wc[i];
                yr[c] = s + (has_bias != 0u ? bias[c] : 0.0);
            }
        }
        return;
    }

    static const uint kMC = 264u;
    double4 bp4[VELRI_KC * (NR / 4)];
    double ap[MR * VELRI_KC];
    double c_buf[264 * NR];
    bp4[0] = double4(0.0); ap[0] = 0.0; c_buf[0] = 0.0;
    Ptr<double> bp = Ptr<double>(&bp4[0]);
    Ptr<double, Access.Read> ap_rd = Ptr<double, Access.Read>(&ap[0]);

    for (uint cb = c0; cb < c1; cb += uint(NR)) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1; rb += kMC) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0u; k0 < k; k0 += kKC) {
                uint kc = min(kKC, k - k0);
                pack_b_f64<NR>(w, w_stride, cb, nr, k0, kc, &bp4[0]);
                if (k0 == 0u) {
                    for (uint rm = 0u; rm < mc; rm += uint(MR)) {
                        uint mr = min(uint(MR), mc - rm);
                        if (mr == uint(MR)) {
                            gemm_mk_f64<MR, NR, true>(x + (rb + rm) * x_stride + k0, x_stride, bp, &c_buf[rm * uint(NR)], uint(NR), kc);
                        } else {
                            pack_a_f64<MR>(x, x_stride, rb + rm, mr, k0, kc, &ap[0]);
                            gemm_mk_f64<MR, NR, true>(ap_rd, kKC, bp, &c_buf[rm * uint(NR)], uint(NR), kc);
                        }
                    }
                } else {
                    for (uint rm = 0u; rm < mc; rm += uint(MR)) {
                        uint mr = min(uint(MR), mc - rm);
                        if (mr == uint(MR)) {
                            gemm_mk_f64<MR, NR, false>(x + (rb + rm) * x_stride + k0, x_stride, bp, &c_buf[rm * uint(NR)], uint(NR), kc);
                        } else {
                            pack_a_f64<MR>(x, x_stride, rb + rm, mr, k0, kc, &ap[0]);
                            gemm_mk_f64<MR, NR, false>(ap_rd, kKC, bp, &c_buf[rm * uint(NR)], uint(NR), kc);
                        }
                    }
                }
            }
            for (uint rm = 0u; rm < mc; ++rm) {
                uint rr = rb + rm;
                Ptr<double> yr = y + rr * y_stride + cb;
                for (uint j = 0u; j < nr; ++j) {
                    yr[j] = c_buf[rm * uint(NR) + j] + (has_bias != 0u ? bias[cb + j] : 0.0);
                }
            }
        }
    }
}
