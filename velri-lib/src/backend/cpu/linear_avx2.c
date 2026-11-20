#include "internal.h"
#include <immintrin.h>
#include "linear_workspace.h"
#include "gemm_fma.h"

/*
 * AVX2 + FMA + F16C Linear Backend (`VELRI_CPU_AVX2`):
 *   - 6x16 4x-unrolled register-blocked FP32 inline-asm microkernel (16/16 YMMs, 0 spills)
 *   - 6x8  4x-unrolled register-blocked FP64 inline-asm microkernel (16/16 YMMs, 0 spills)
 *   - 6x16 2x-unrolled overflow-free INT8 inline-asm microkernel (`vpmovsxbw + vpmaddwd`, 0-ULP!)
 *   - 32B/64B-aligned `a_pack` (8-lane stride) + SIMD 8x8 / 4x4 transposes for `pack_a` & `pack_b`
 *   - Hoisted conflict-free 8-row grouped `pack_b` for Row-Major GEMM (`0 B` heap allocation).
 */

/* MR/NR match the fixed SIMD microkernels. MC, KC and TILES describe
 * cache blocking; workspace offsets and scheduler hints derive from them.
 * A panels pad the microkernel's row count to a vector-friendly stride. */
enum { FP_MR = 6, FP_NR = 16, FP_A_ROWS = 8 };
static_assert(FP_MR == AVX2_F32_STEP_ROWS &&
              FP_NR * sizeof(float) == 2 * AVX2_F32_STEP_VECTOR_BYTES &&
              sizeof(float) == AVX2_F32_STEP_ELEMENT_BYTES);
enum { FP_P_MC = 192, FP_P_KC = 256, FP_P_TILES = 16, FP_P_BAND = 960 };
static_assert(FP_P_MC % FP_MR == 0);
VELRI_LINEAR_WORKSPACE(FP_P, FP_P_MC * FP_P_TILES * FP_NR * sizeof(float),
                       FP_A_ROWS * FP_P_KC * sizeof(float),
                       FP_P_TILES * FP_NR * FP_P_KC * sizeof(float), 0);
static const velri_lin_blocking fp_p = {FP_P_BYTES, FP_MR, FP_P_MC, FP_P_TILES, 4 * FP_MR, 8};
enum { FP_R_MC = 336, FP_R_KC = 256, FP_R_TILES = 8, FP_R_BAND = 0 };
static_assert(FP_R_MC % FP_MR == 0);
VELRI_LINEAR_WORKSPACE(FP_R, FP_R_MC * FP_R_TILES * FP_NR * sizeof(float),
                       FP_A_ROWS * FP_R_KC * sizeof(float),
                       FP_R_TILES * FP_NR * FP_R_KC * sizeof(float), 0);
static const velri_lin_blocking fp_r = {FP_R_BYTES, FP_MR, FP_R_MC, FP_R_TILES, 16 * FP_MR, 4};
enum { F64_MR = 6, F64_NR = 8, F64_A_ROWS = 8 };
static_assert(F64_MR == AVX2_F64_STEP_ROWS &&
              F64_NR * sizeof(double) == 2 * AVX2_F64_STEP_VECTOR_BYTES &&
              sizeof(double) == AVX2_F64_STEP_ELEMENT_BYTES);
enum { F64_P_MC = 96, F64_P_KC = 256, F64_P_TILES = 16, F64_P_BAND = 480 };
static_assert(F64_P_MC % F64_MR == 0);
VELRI_LINEAR_WORKSPACE(F64_P, F64_P_MC * F64_P_TILES * F64_NR * sizeof(double),
                       F64_A_ROWS * F64_P_KC * sizeof(double), F64_NR * F64_P_KC * sizeof(double),
                       0);
static const velri_lin_blocking f64_p = {F64_P_BYTES, F64_MR, F64_P_MC, F64_P_TILES, 4 * F64_MR, 8};
enum { F64_R_MC = 192, F64_R_KC = 256, F64_R_TILES = 16, F64_R_BAND = 0 };
static_assert(F64_R_MC % F64_MR == 0);
VELRI_LINEAR_WORKSPACE(F64_R, F64_R_MC * F64_R_TILES * F64_NR * sizeof(double),
                       F64_A_ROWS * F64_R_KC * sizeof(double),
                       F64_R_TILES * F64_NR * F64_R_KC * sizeof(double), 0);
static const velri_lin_blocking f64_r = {F64_R_BYTES, F64_MR,      F64_R_MC,
                                         F64_R_TILES, 16 * F64_MR, 8};
enum { I8_MR = 6, I8_NR = 16, I8_A_ROWS = 8 };
enum { I8_P_MC = 192, I8_P_KC = 512, I8_P_TILES = 16, I8_P_BAND = 960 };
static_assert(I8_P_MC % I8_MR == 0);
VELRI_LINEAR_WORKSPACE(I8_P, I8_P_MC * I8_P_TILES * I8_NR * sizeof(int32_t),
                       I8_A_ROWS * I8_P_KC * sizeof(int16_t),
                       I8_P_TILES * I8_NR * I8_P_KC * sizeof(int16_t), 0);
static const velri_lin_blocking i8_p = {I8_P_BYTES, I8_MR, I8_P_MC, I8_P_TILES, 4 * I8_MR, 8};
enum { I8_R_MC = 336, I8_R_KC = 512, I8_R_TILES = 8, I8_R_BAND = 0 };
static_assert(I8_R_MC % I8_MR == 0);
VELRI_LINEAR_WORKSPACE(I8_R, I8_R_MC * I8_R_TILES * I8_NR * sizeof(int32_t),
                       I8_A_ROWS * I8_R_KC * sizeof(int16_t),
                       I8_R_TILES * I8_NR * I8_R_KC * sizeof(int16_t), 0);
static const velri_lin_blocking i8_r = {I8_R_BYTES, I8_MR, I8_R_MC, I8_R_TILES, 16 * I8_MR, 4};

static_assert(I8_P_KC % 2 == 0);
static_assert(I8_R_KC % 2 == 0);

static inline uint8_t *align64_ptr(void *p) {
    uintptr_t u = ((uintptr_t)p + 63ULL) & ~63ULL;
    return (uint8_t *)u;
}





static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

/* ============================================================================
 * In-Register SIMD Transpose Primitives (8x8 `f32`/`i32`, 4x4 `f64`, 8x8 `i16`)
 * ============================================================================ */

static inline void transpose_8x8_ps(__m256 *r0, __m256 *r1, __m256 *r2, __m256 *r3, __m256 *r4,
                                    __m256 *r5, __m256 *r6, __m256 *r7) {
    __m256 t0 = _mm256_unpacklo_ps(*r0, *r1);
    __m256 t1 = _mm256_unpackhi_ps(*r0, *r1);
    __m256 t2 = _mm256_unpacklo_ps(*r2, *r3);
    __m256 t3 = _mm256_unpackhi_ps(*r2, *r3);
    __m256 t4 = _mm256_unpacklo_ps(*r4, *r5);
    __m256 t5 = _mm256_unpackhi_ps(*r4, *r5);
    __m256 t6 = _mm256_unpacklo_ps(*r6, *r7);
    __m256 t7 = _mm256_unpackhi_ps(*r6, *r7);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    *r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    *r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    *r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    *r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    *r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    *r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    *r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    *r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

static inline void transpose_4x4_pd(__m256d *r0, __m256d *r1, __m256d *r2, __m256d *r3) {
    __m256d t0 = _mm256_unpacklo_pd(*r0, *r1);
    __m256d t1 = _mm256_unpackhi_pd(*r0, *r1);
    __m256d t2 = _mm256_unpacklo_pd(*r2, *r3);
    __m256d t3 = _mm256_unpackhi_pd(*r2, *r3);

    *r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    *r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    *r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    *r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

static inline void transpose_8x8_epi16(__m128i *r0, __m128i *r1, __m128i *r2, __m128i *r3,
                                       __m128i *r4, __m128i *r5, __m128i *r6, __m128i *r7) {
    __m128i t0 = _mm_unpacklo_epi16(*r0, *r1);
    __m128i t1 = _mm_unpackhi_epi16(*r0, *r1);
    __m128i t2 = _mm_unpacklo_epi16(*r2, *r3);
    __m128i t3 = _mm_unpackhi_epi16(*r2, *r3);
    __m128i t4 = _mm_unpacklo_epi16(*r4, *r5);
    __m128i t5 = _mm_unpackhi_epi16(*r4, *r5);
    __m128i t6 = _mm_unpacklo_epi16(*r6, *r7);
    __m128i t7 = _mm_unpackhi_epi16(*r6, *r7);

    __m128i s0 = _mm_unpacklo_epi32(t0, t2);
    __m128i s1 = _mm_unpackhi_epi32(t0, t2);
    __m128i s2 = _mm_unpacklo_epi32(t1, t3);
    __m128i s3 = _mm_unpackhi_epi32(t1, t3);
    __m128i s4 = _mm_unpacklo_epi32(t4, t6);
    __m128i s5 = _mm_unpackhi_epi32(t4, t6);
    __m128i s6 = _mm_unpacklo_epi32(t5, t7);
    __m128i s7 = _mm_unpackhi_epi32(t5, t7);

    *r0 = _mm_unpacklo_epi64(s0, s4);
    *r1 = _mm_unpackhi_epi64(s0, s4);
    *r2 = _mm_unpacklo_epi64(s1, s5);
    *r3 = _mm_unpackhi_epi64(s1, s5);
    *r4 = _mm_unpacklo_epi64(s2, s6);
    *r5 = _mm_unpackhi_epi64(s2, s6);
    *r6 = _mm_unpacklo_epi64(s3, s7);
    *r7 = _mm_unpackhi_epi64(s3, s7);
}

/* ============================================================================
 * 1. Core 6x16 FP32 4x-Unrolled Inline-ASM Microkernel (`a` stride = 8 floats!)
 * ============================================================================ */


static inline void ukernel_6x16_f32_avx2(size_t kc, const float *a, const float *b,
                                         float *restrict acc, int first_k) {
    if (kc == 0) {
        return;
    }
    size_t kc4 = kc >> 2;
    size_t kc1 = kc & 3ULL;
    /* clang-format off */
    __asm__ volatile(
        "test %[fk], %[fk]\n\t"
        "jz 1f\n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0\n\tvxorps %%ymm1, %%ymm1, %%ymm1\n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2\n\tvxorps %%ymm3, %%ymm3, %%ymm3\n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4\n\tvxorps %%ymm5, %%ymm5, %%ymm5\n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6\n\tvxorps %%ymm7, %%ymm7, %%ymm7\n\t"
        "vxorps %%ymm8, %%ymm8, %%ymm8\n\tvxorps %%ymm9, %%ymm9, %%ymm9\n\t"
        "vxorps %%ymm10, %%ymm10, %%ymm10\n\tvxorps %%ymm11, %%ymm11, %%ymm11\n\t"
        "jmp 2f\n\t"
        "1:\n\t"
        "vmovaps 0(%[acc]),   %%ymm0\n\tvmovaps 32(%[acc]),  %%ymm1\n\t"
        "vmovaps 64(%[acc]),  %%ymm2\n\tvmovaps 96(%[acc]),  %%ymm3\n\t"
        "vmovaps 128(%[acc]), %%ymm4\n\tvmovaps 160(%[acc]), %%ymm5\n\t"
        "vmovaps 192(%[acc]), %%ymm6\n\tvmovaps 224(%[acc]), %%ymm7\n\t"
        "vmovaps 256(%[acc]), %%ymm8\n\tvmovaps 288(%[acc]), %%ymm9\n\t"
        "vmovaps 320(%[acc]), %%ymm10\n\tvmovaps 352(%[acc]), %%ymm11\n\t"
        "2:\n\t"
        "test %[kc4], %[kc4]\n\t"
        "jz 4f\n\t"
        ".p2align 4\n\t"
        "3:\n\t"
        AVX2_F32_STEP(0, 0)
        AVX2_F32_STEP(32, 64)
        AVX2_F32_STEP(64, 128)
        AVX2_F32_STEP(96, 192)
        "add $128, %[a]\n\t"
        "add $256, %[b]\n\t"
        "dec %[kc4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kc1], %[kc1]\n\t"
        "jz 6f\n\t"
        ".p2align 4\n\t"
        "5:\n\t"
        AVX2_F32_STEP(0, 0)
        "add $32, %[a]\n\t"
        "add $64, %[b]\n\t"
        "dec %[kc1]\n\t"
        "jnz 5b\n\t"
        "6:\n\t"
        "vmovaps %%ymm0,  0(%[acc])\n\tvmovaps %%ymm1,  32(%[acc])\n\t"
        "vmovaps %%ymm2,  64(%[acc])\n\tvmovaps %%ymm3,  96(%[acc])\n\t"
        "vmovaps %%ymm4,  128(%[acc])\n\tvmovaps %%ymm5,  160(%[acc])\n\t"
        "vmovaps %%ymm6,  192(%[acc])\n\tvmovaps %%ymm7,  224(%[acc])\n\t"
        "vmovaps %%ymm8,  256(%[acc])\n\tvmovaps %%ymm9,  288(%[acc])\n\t"
        "vmovaps %%ymm10, 320(%[acc])\n\tvmovaps %%ymm11, 352(%[acc])\n\t"
        : [a] "+&r"(a), [b] "+&r"(b), [kc4] "+&r"(kc4), [kc1] "+&r"(kc1)
        : [acc] "r"(acc), [fk] "r"(first_k)
        : "cc", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15");
    /* clang-format on */
}

static inline float load_elem_f32_avx2(velri_dtype dt, const void *base, size_t idx) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        return ((const float *)base)[idx];
    case VELRI_DTYPE_F16:
        return velri_f16_to_f32_scalar(((const velri_fp16 *)base)[idx]);
    case VELRI_DTYPE_BF16:
        return velri_bf16_to_f32_scalar(((const velri_bf16 *)base)[idx]);
    case VELRI_DTYPE_I8:
        return (float)((const int8_t *)base)[idx];
    default:
        return 0.0f;
    }
}

static inline __m256 load8_as_f32_avx2(velri_dtype dt, const void *base, size_t idx) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        return _mm256_loadu_ps((const float *)base + idx);
    case VELRI_DTYPE_F16:
        return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)((const velri_fp16 *)base + idx)));
    case VELRI_DTYPE_BF16: {
        __m128i h = _mm_loadu_si128((const __m128i *)((const velri_bf16 *)base + idx));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
    }
    case VELRI_DTYPE_I8: {
        __m128i q = _mm_loadl_epi64((const __m128i *)((const int8_t *)base + idx));
        return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
    }
    default:
        return _mm256_setzero_ps();
    }
}

static inline void store_row_y_avx2(velri_dtype y_dt, void *y_base, size_t y_off, const float *vals,
                                    size_t width) {
    if (y_dt == VELRI_DTYPE_F32) {
        float *dst = (float *)y_base + y_off;
        if (width == 16) {
            _mm256_storeu_ps(dst, _mm256_loadu_ps(vals));
            _mm256_storeu_ps(dst + 8, _mm256_loadu_ps(vals + 8));
        } else {
            for (size_t j = 0; j < width; ++j) {
                dst[j] = vals[j];
            }
        }
    } else if (y_dt == VELRI_DTYPE_F16) {
        velri_fp16 *dst = (velri_fp16 *)y_base + y_off;
        if (width == 16) {
            __m128i h0 = _mm256_cvtps_ph(_mm256_loadu_ps(vals),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i h1 = _mm256_cvtps_ph(_mm256_loadu_ps(vals + 8),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i *)dst, h0);
            _mm_storeu_si128((__m128i *)(dst + 8), h1);
        } else {
            for (size_t j = 0; j < width; ++j) {
                dst[j] = velri_f32_to_f16_scalar(vals[j]);
            }
        }
    } else if (y_dt == VELRI_DTYPE_BF16) {
        velri_bf16 *dst = (velri_bf16 *)y_base + y_off;
        if (width == 16) {
            velri_cvt_f32_to_bf16_avx2(vals, dst, 16);
        } else {
            for (size_t j = 0; j < width; ++j) {
                dst[j] = velri_f32_to_bf16_scalar(vals[j]);
            }
        }
    }
}

/* Pack `mr <= 6` rows into 32B-aligned `[kc, 8]` float micro-panel via 8x8 YMM transpose */
static void pack_a_6_f32_avx2(size_t mr, size_t kc, velri_dtype x_dt, const void *x,
                              size_t x_stride, size_t r_base, size_t k_base, float *restrict dst) {
    size_t p = 0;
    if (mr == 6) {
        const size_t o0 = (r_base + 0) * x_stride + k_base;
        const size_t o1 = (r_base + 1) * x_stride + k_base;
        const size_t o2 = (r_base + 2) * x_stride + k_base;
        const size_t o3 = (r_base + 3) * x_stride + k_base;
        const size_t o4 = (r_base + 4) * x_stride + k_base;
        const size_t o5 = (r_base + 5) * x_stride + k_base;
        for (; p + 8 <= kc; p += 8) {
            __m256 r0 = load8_as_f32_avx2(x_dt, x, o0 + p);
            __m256 r1 = load8_as_f32_avx2(x_dt, x, o1 + p);
            __m256 r2 = load8_as_f32_avx2(x_dt, x, o2 + p);
            __m256 r3 = load8_as_f32_avx2(x_dt, x, o3 + p);
            __m256 r4 = load8_as_f32_avx2(x_dt, x, o4 + p);
            __m256 r5 = load8_as_f32_avx2(x_dt, x, o5 + p);
            __m256 r6 = _mm256_setzero_ps();
            __m256 r7 = _mm256_setzero_ps();
            transpose_8x8_ps(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            float *d = dst + p * 8;
            _mm256_storeu_ps(d + 0, r0);
            _mm256_storeu_ps(d + 8, r1);
            _mm256_storeu_ps(d + 16, r2);
            _mm256_storeu_ps(d + 24, r3);
            _mm256_storeu_ps(d + 32, r4);
            _mm256_storeu_ps(d + 40, r5);
            _mm256_storeu_ps(d + 48, r6);
            _mm256_storeu_ps(d + 56, r7);
        }
    }
    for (; p < kc; ++p) {
        for (size_t r = 0; r < mr; ++r) {
            dst[p * 8 + r] = load_elem_f32_avx2(x_dt, x, (r_base + r) * x_stride + k_base + p);
        }
        for (size_t r = mr; r < 8; ++r) {
            dst[p * 8 + r] = 0.0f;
        }
    }
}

/* Conflict-free 8-row grouped `pack_b` for 16-column panels (`Prepared` non-F32 or `RowMajor`) */
static void pack_b_tile16_f32_avx2(size_t nr, size_t kc, velri_dtype w_dt, const void *w_src,
                                   size_t w_stride, size_t c_base, size_t k_base, bool from_packed,
                                   size_t packed_C, float *restrict dst) {
    if (from_packed) {
        if (w_dt == VELRI_DTYPE_F32 && packed_C == 16) {
            memcpy(dst, (const float *)w_src + k_base * 16, kc * 16 * sizeof(float));
            return;
        }
        if (packed_C == 16) {
            for (size_t p = 0; p < kc; ++p) {
                __m256 v0 = load8_as_f32_avx2(w_dt, w_src, (k_base + p) * 16);
                __m256 v1 = load8_as_f32_avx2(w_dt, w_src, (k_base + p) * 16 + 8);
                _mm256_storeu_ps(dst + p * 16, v0);
                _mm256_storeu_ps(dst + p * 16 + 8, v1);
            }
            return;
        }
        for (size_t p = 0; p < kc; ++p) {
            for (size_t j = 0; j < packed_C; ++j) {
                dst[p * 16 + j] = load_elem_f32_avx2(w_dt, w_src, (k_base + p) * packed_C + j);
            }
            for (size_t j = packed_C; j < 16; ++j) {
                dst[p * 16 + j] = 0.0f;
            }
        }
        return;
    }

    size_t p_done = 0;
    if (nr == 16 && kc >= 8) {
        const size_t kc8 = kc & ~7ULL;
        for (size_t jh = 0; jh < 16; jh += 8) {
            const size_t c0 = (c_base + jh + 0) * w_stride + k_base;
            const size_t c1 = (c_base + jh + 1) * w_stride + k_base;
            const size_t c2 = (c_base + jh + 2) * w_stride + k_base;
            const size_t c3 = (c_base + jh + 3) * w_stride + k_base;
            const size_t c4 = (c_base + jh + 4) * w_stride + k_base;
            const size_t c5 = (c_base + jh + 5) * w_stride + k_base;
            const size_t c6 = (c_base + jh + 6) * w_stride + k_base;
            const size_t c7 = (c_base + jh + 7) * w_stride + k_base;
            for (size_t p = 0; p < kc8; p += 8) {
                __m256 r0 = load8_as_f32_avx2(w_dt, w_src, c0 + p);
                __m256 r1 = load8_as_f32_avx2(w_dt, w_src, c1 + p);
                __m256 r2 = load8_as_f32_avx2(w_dt, w_src, c2 + p);
                __m256 r3 = load8_as_f32_avx2(w_dt, w_src, c3 + p);
                __m256 r4 = load8_as_f32_avx2(w_dt, w_src, c4 + p);
                __m256 r5 = load8_as_f32_avx2(w_dt, w_src, c5 + p);
                __m256 r6 = load8_as_f32_avx2(w_dt, w_src, c6 + p);
                __m256 r7 = load8_as_f32_avx2(w_dt, w_src, c7 + p);
                transpose_8x8_ps(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
                float *d = dst + p * 16 + jh;
                _mm256_storeu_ps(d + 0 * 16, r0);
                _mm256_storeu_ps(d + 1 * 16, r1);
                _mm256_storeu_ps(d + 2 * 16, r2);
                _mm256_storeu_ps(d + 3 * 16, r3);
                _mm256_storeu_ps(d + 4 * 16, r4);
                _mm256_storeu_ps(d + 5 * 16, r5);
                _mm256_storeu_ps(d + 6 * 16, r6);
                _mm256_storeu_ps(d + 7 * 16, r7);
            }
        }
        p_done = kc8;
    }
    for (size_t p = p_done; p < kc; ++p) {
        for (size_t j = 0; j < nr; ++j) {
            dst[p * 16 + j] = load_elem_f32_avx2(w_dt, w_src, (c_base + j) * w_stride + k_base + p);
        }
        for (size_t j = nr; j < 16; ++j) {
            dst[p * 16 + j] = 0.0f;
        }
    }
}

/* ============================================================================
 * 2. FP32-Accumulated Family (`F32`, `F16`, `F32xF16`, `BF16`, `F32xBF16`, `F32xI8`)
 * ============================================================================ */





static void epilogue_6x16_fp_avx2(const velri_linear_typed_plan *plan,
                                  const velri_linear_run_args *args, size_t rb, size_t mb,
                                  size_t mr_blocks, size_t max_mr_blocks, size_t cb, size_t cb_end,
                                  size_t C, size_t scale_off, bool from_packed,
                                  const float *acc_base) {
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const float *bias = (const float *)args->bias;
    size_t tile_idx = 0;
    for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
        size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
        const float *wsc = nullptr;
        if (plan->desc.has_w_scale) {
            if (from_packed) {
                const uint8_t *tp = pk_base + (col0 / C) * plan->tile_stride_b;
                wsc = (const float *)(tp + scale_off);
            } else if (args->w_scale) {
                wsc = args->w_scale + col0;
            }
        }
        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
            size_t r_sub = mr_i * 6;
            size_t mr = (mb - r_sub < 6) ? (mb - r_sub) : 6;
            const float *acc_tile = acc_base + (tile_idx * max_mr_blocks + mr_i) * 96;
            for (size_t r = 0; r < mr; ++r) {
                const float *row_vals = acc_tile + r * 16;
                if (width == 16) {
                    __m256 v0 = _mm256_load_ps(row_vals);
                    __m256 v1 = _mm256_load_ps(row_vals + 8);
                    if (wsc) {
                        v0 = _mm256_mul_ps(v0, _mm256_loadu_ps(wsc));
                        v1 = _mm256_mul_ps(v1, _mm256_loadu_ps(wsc + 8));
                    }
                    if (bias) {
                        v0 = _mm256_add_ps(v0, _mm256_loadu_ps(bias + col0));
                        v1 = _mm256_add_ps(v1, _mm256_loadu_ps(bias + col0 + 8));
                    }
                    if (plan->desc.y_dtype == VELRI_DTYPE_F32) {
                        float *ydst = (float *)args->y + (rb + r_sub + r) * args->y_stride + col0;
                        _mm256_storeu_ps(ydst, v0);
                        _mm256_storeu_ps(ydst + 8, v1);
                    } else {
                        alignas(32) float tmp[16];
                        _mm256_store_ps(tmp, v0);
                        _mm256_store_ps(tmp + 8, v1);
                        store_row_y_avx2(plan->desc.y_dtype, args->y,
                                         (rb + r_sub + r) * args->y_stride + col0, tmp, 16);
                    }
                } else {
                    float tmp[16];
                    for (size_t j = 0; j < width; ++j) {
                        float v = row_vals[j];
                        if (wsc) {
                            v *= wsc[j];
                        }
                        if (bias) {
                            v += bias[col0 + j];
                        }
                        tmp[j] = v;
                    }
                    store_row_y_avx2(plan->desc.y_dtype, args->y,
                                     (rb + r_sub + r) * args->y_stride + col0, tmp, width);
                }
            }
        }
    }
}

static void avx2_fp_gemm(const velri_linear_typed_plan *plan, const velri_linear_run_args *args,
                         size_t r0, size_t r1, size_t c0, size_t c1, void *restrict scratch,
                         bool from_packed) {
    if (r1 - r0 <= 2) {
        if (from_packed) {
            velri_gemv_f32_prepared_avx2(plan, args, r0, r1, c0, c1);
        } else {
            velri_gemv_f32_rowmajor_avx2(plan, args, r0, r1, c0, c1);
        }
        return;
    }

    const size_t k = plan->k;
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const size_t elem_b = velri_dtype_bytes(plan->desc.w_dtype);
    const size_t payload_b = plan->k_aligned * plan->columns * elem_b;
    const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
    const size_t scale_off =
        meta_off + (plan->desc.w_dtype == VELRI_DTYPE_I8 ? plan->columns * sizeof(int32_t) : 0);

    if (from_packed) {
        /* Panel offsets and bounds come from the workspace declarations above. */
        enum { MC_L3 = FP_P_BAND };
        enum { MC = FP_P_MC };
        enum { KC = FP_P_KC };
        enum { NC_TILES = FP_P_TILES };

        uint8_t *sc = align64_ptr(scratch);
        float *acc_base = (float *)sc;
        float *a_pack = (float *)(sc + FP_P_A);
        float *b_pack = (float *)(sc + FP_P_B);

        const size_t C = plan->columns;
        const size_t step_c = NC_TILES * C;
        const bool direct_f32 = (plan->desc.w_dtype == VELRI_DTYPE_F32 && C == 16);

        for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
            size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
            for (size_t cb = c0; cb < c1; cb += step_c) {
                size_t cb_end = (c1 - cb < step_c) ? c1 : (cb + step_c);
                for (size_t rb = r_l3; rb < r_l3_end; rb += MC) {
                    size_t mb = (r_l3_end - rb < MC) ? (r_l3_end - rb) : MC;
                    size_t mr_blocks = (mb + 5) / 6;

                    for (size_t kb = 0; kb < k; kb += KC) {
                        size_t kc = (k - kb < KC) ? (k - kb) : KC;
                        int first_k = (kb == 0) ? 1 : 0;

                        if (!direct_f32) {
                            size_t ti = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++ti) {
                                size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                                const void *wsrc =
                                    (const void *)(pk_base + (col0 / C) * plan->tile_stride_b);
                                pack_b_tile16_f32_avx2(width, kc, plan->desc.w_dtype, wsrc, 0, col0,
                                                       kb, true, C, b_pack + ti * 16 * kc);
                            }
                        }

                        /* `mr_i` OUTER, `col0` INNER: 8 KiB `a_pack` is reused across output tiles.
                         */
                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 6;
                            size_t mr = (mb - r_sub < 6) ? (mb - r_sub) : 6;
                            pack_a_6_f32_avx2(mr, kc, plan->desc.x_dtype, args->x, args->x_stride,
                                              rb + r_sub, kb, a_pack);

                            size_t tile_idx = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                                const float *b_panel =
                                    direct_f32
                                        ? ((const float *)(pk_base +
                                                           (col0 / 16) * plan->tile_stride_b) +
                                           kb * 16)
                                        : (b_pack + tile_idx * 16 * kc);
                                float *acc_tile =
                                    acc_base + (tile_idx * (FP_P_MC / FP_MR) + mr_i) * 96;
                                ukernel_6x16_f32_avx2(kc, a_pack, b_panel, acc_tile, first_k);
                            }
                        }
                    }
                    epilogue_6x16_fp_avx2(plan, args, rb, mb, mr_blocks, FP_P_MC / FP_MR, cb,
                                          cb_end, C, scale_off, true, acc_base);
                }
            }
        }
        return;
    }

    /* Panel offsets and bounds come from the workspace declarations above. */
    enum { MC_RM = FP_R_MC };
    enum { KC_RM = FP_R_KC };
    enum { NC_TILES_RM = FP_R_TILES };
    enum { STEP_C_RM = FP_R_TILES * FP_NR };

    uint8_t *sc = align64_ptr(scratch);
    float *acc_rm = (float *)sc;
    float *a_pack = (float *)(sc + FP_R_A);
    float *b_pack = (float *)(sc + FP_R_B);

    for (size_t rb0 = r0; rb0 < r1; rb0 += MC_RM) {
        size_t mb_big = (r1 - rb0 < MC_RM) ? (r1 - rb0) : MC_RM;
        size_t mr_total = (mb_big + 5) / 6;

        for (size_t cb = c0; cb < c1; cb += STEP_C_RM) {
            size_t cb_end = (c1 - cb < STEP_C_RM) ? c1 : (cb + STEP_C_RM);

            for (size_t kb = 0; kb < k; kb += KC_RM) {
                size_t kc = (k - kb < KC_RM) ? (k - kb) : KC_RM;
                int first_k = (kb == 0) ? 1 : 0;

                size_t n_tiles = 0;
                for (size_t col0 = cb; col0 < cb_end; col0 += 16, ++n_tiles) {
                    size_t width = (cb_end - col0 < 16) ? (cb_end - col0) : 16;
                    pack_b_tile16_f32_avx2(width, kc, plan->desc.w_dtype, args->weights,
                                           args->w_stride, col0, kb, false, 16,
                                           b_pack + n_tiles * 16 * kc);
                }

                for (size_t mr_i = 0; mr_i < mr_total; ++mr_i) {
                    size_t r_sub = mr_i * 6;
                    size_t mr = (mb_big - r_sub < 6) ? (mb_big - r_sub) : 6;
                    pack_a_6_f32_avx2(mr, kc, plan->desc.x_dtype, args->x, args->x_stride,
                                      rb0 + r_sub, kb, a_pack);
                    for (size_t ti = 0; ti < n_tiles; ++ti) {
                        float *acc_tile = acc_rm + (ti * (FP_R_MC / FP_MR) + mr_i) * 96;
                        ukernel_6x16_f32_avx2(kc, a_pack, b_pack + ti * 16 * kc, acc_tile, first_k);
                    }
                }
            }

            epilogue_6x16_fp_avx2(plan, args, rb0, mb_big, mr_total, FP_R_MC / FP_MR, cb, cb_end,
                                  16, 0, false, acc_rm);
        }
    }
}

/* ============================================================================
 * 3. FP64 (`F64 * F64 -> F64`) AVX2 4x-Unrolled 6x8 Inline-ASM Microkernel
 * ============================================================================ */


static inline void ukernel_6x8_f64_avx2(size_t kc, const double *a, const double *b,
                                        double *restrict acc, int first_k) {
    if (kc == 0) {
        return;
    }
    size_t kc4 = kc >> 2;
    size_t kc1 = kc & 3ULL;
    /* clang-format off */
    __asm__ volatile(
        "test %[fk], %[fk]\n\t"
        "jz 1f\n\t"
        "vxorpd %%ymm0, %%ymm0, %%ymm0\n\tvxorpd %%ymm1, %%ymm1, %%ymm1\n\t"
        "vxorpd %%ymm2, %%ymm2, %%ymm2\n\tvxorpd %%ymm3, %%ymm3, %%ymm3\n\t"
        "vxorpd %%ymm4, %%ymm4, %%ymm4\n\tvxorpd %%ymm5, %%ymm5, %%ymm5\n\t"
        "vxorpd %%ymm6, %%ymm6, %%ymm6\n\tvxorpd %%ymm7, %%ymm7, %%ymm7\n\t"
        "vxorpd %%ymm8, %%ymm8, %%ymm8\n\tvxorpd %%ymm9, %%ymm9, %%ymm9\n\t"
        "vxorpd %%ymm10, %%ymm10, %%ymm10\n\tvxorpd %%ymm11, %%ymm11, %%ymm11\n\t"
        "jmp 2f\n\t"
        "1:\n\t"
        "vmovapd 0(%[acc]),   %%ymm0\n\tvmovapd 32(%[acc]),  %%ymm1\n\t"
        "vmovapd 64(%[acc]),  %%ymm2\n\tvmovapd 96(%[acc]),  %%ymm3\n\t"
        "vmovapd 128(%[acc]), %%ymm4\n\tvmovapd 160(%[acc]), %%ymm5\n\t"
        "vmovapd 192(%[acc]), %%ymm6\n\tvmovapd 224(%[acc]), %%ymm7\n\t"
        "vmovapd 256(%[acc]), %%ymm8\n\tvmovapd 288(%[acc]), %%ymm9\n\t"
        "vmovapd 320(%[acc]), %%ymm10\n\tvmovapd 352(%[acc]), %%ymm11\n\t"
        "2:\n\t"
        "test %[kc4], %[kc4]\n\t"
        "jz 4f\n\t"
        ".p2align 4\n\t"
        "3:\n\t"
        AVX2_F64_STEP(0, 0)
        AVX2_F64_STEP(64, 64)
        AVX2_F64_STEP(128, 128)
        AVX2_F64_STEP(192, 192)
        "add $256, %[a]\n\t"
        "add $256, %[b]\n\t"
        "dec %[kc4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kc1], %[kc1]\n\t"
        "jz 6f\n\t"
        ".p2align 4\n\t"
        "5:\n\t"
        AVX2_F64_STEP(0, 0)
        "add $64, %[a]\n\t"
        "add $64, %[b]\n\t"
        "dec %[kc1]\n\t"
        "jnz 5b\n\t"
        "6:\n\t"
        "vmovapd %%ymm0,  0(%[acc])\n\tvmovapd %%ymm1,  32(%[acc])\n\t"
        "vmovapd %%ymm2,  64(%[acc])\n\tvmovapd %%ymm3,  96(%[acc])\n\t"
        "vmovapd %%ymm4,  128(%[acc])\n\tvmovapd %%ymm5,  160(%[acc])\n\t"
        "vmovapd %%ymm6,  192(%[acc])\n\tvmovapd %%ymm7,  224(%[acc])\n\t"
        "vmovapd %%ymm8,  256(%[acc])\n\tvmovapd %%ymm9,  288(%[acc])\n\t"
        "vmovapd %%ymm10, 320(%[acc])\n\tvmovapd %%ymm11, 352(%[acc])\n\t"
        : [a] "+&r"(a), [b] "+&r"(b), [kc4] "+&r"(kc4), [kc1] "+&r"(kc1)
        : [acc] "r"(acc), [fk] "r"(first_k)
        : "cc", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15");
    /* clang-format on */
}

static void pack_a_6_f64_avx2(size_t mr, size_t kc, const double *x, size_t x_stride, size_t r_base,
                              size_t k_base, double *restrict dst) {
    size_t p = 0;
    if (mr == 6) {
        const double *x0 = x + (r_base + 0) * x_stride + k_base;
        const double *x1 = x + (r_base + 1) * x_stride + k_base;
        const double *x2 = x + (r_base + 2) * x_stride + k_base;
        const double *x3 = x + (r_base + 3) * x_stride + k_base;
        const double *x4 = x + (r_base + 4) * x_stride + k_base;
        const double *x5 = x + (r_base + 5) * x_stride + k_base;
        for (; p + 4 <= kc; p += 4) {
            __m256d r0 = _mm256_loadu_pd(x0 + p);
            __m256d r1 = _mm256_loadu_pd(x1 + p);
            __m256d r2 = _mm256_loadu_pd(x2 + p);
            __m256d r3 = _mm256_loadu_pd(x3 + p);
            transpose_4x4_pd(&r0, &r1, &r2, &r3);

            __m256d r4 = _mm256_loadu_pd(x4 + p);
            __m256d r5 = _mm256_loadu_pd(x5 + p);
            __m256d r6 = _mm256_setzero_pd();
            __m256d r7 = _mm256_setzero_pd();
            transpose_4x4_pd(&r4, &r5, &r6, &r7);

            double *d = dst + p * 8;
            _mm256_storeu_pd(d + 0, r0);
            _mm256_storeu_pd(d + 4, r4);
            _mm256_storeu_pd(d + 8, r1);
            _mm256_storeu_pd(d + 12, r5);
            _mm256_storeu_pd(d + 16, r2);
            _mm256_storeu_pd(d + 20, r6);
            _mm256_storeu_pd(d + 24, r3);
            _mm256_storeu_pd(d + 28, r7);
        }
    }
    for (; p < kc; ++p) {
        for (size_t r = 0; r < mr; ++r) {
            dst[p * 8 + r] = x[(r_base + r) * x_stride + k_base + p];
        }
        for (size_t r = mr; r < 8; ++r) {
            dst[p * 8 + r] = 0.0;
        }
    }
}

static void pack_b_8_rm_f64_avx2(size_t nr, size_t kc, const double *w, size_t w_stride,
                                 size_t c_base, size_t k_base, double *restrict dst) {
    size_t p_done = 0;
    if (nr == 8 && kc >= 4) {
        const size_t kc4 = kc & ~3ULL;
        for (size_t jh = 0; jh < 8; jh += 4) {
            const double *w0 = w + (c_base + jh + 0) * w_stride + k_base;
            const double *w1 = w + (c_base + jh + 1) * w_stride + k_base;
            const double *w2 = w + (c_base + jh + 2) * w_stride + k_base;
            const double *w3 = w + (c_base + jh + 3) * w_stride + k_base;
            for (size_t p = 0; p < kc4; p += 4) {
                __m256d r0 = _mm256_loadu_pd(w0 + p);
                __m256d r1 = _mm256_loadu_pd(w1 + p);
                __m256d r2 = _mm256_loadu_pd(w2 + p);
                __m256d r3 = _mm256_loadu_pd(w3 + p);
                transpose_4x4_pd(&r0, &r1, &r2, &r3);
                double *d = dst + p * 8 + jh;
                _mm256_storeu_pd(d + 0 * 8, r0);
                _mm256_storeu_pd(d + 1 * 8, r1);
                _mm256_storeu_pd(d + 2 * 8, r2);
                _mm256_storeu_pd(d + 3 * 8, r3);
            }
        }
        p_done = kc4;
    }
    for (size_t p = p_done; p < kc; ++p) {
        for (size_t j = 0; j < nr; ++j) {
            dst[p * 8 + j] = w[(c_base + j) * w_stride + k_base + p];
        }
        for (size_t j = nr; j < 8; ++j) {
            dst[p * 8 + j] = 0.0;
        }
    }
}

static void avx2_f64_run(const velri_linear_typed_plan *plan, const velri_linear_run_args *args,
                         size_t r0, size_t r1, size_t c0, size_t c1, void *restrict scratch,
                         bool from_packed) {
    const size_t k = plan->k;
    const double *x = (const double *)args->x;
    const double *bias = (const double *)args->bias;
    double *restrict y = (double *)args->y;

    if (r1 - r0 <= 2) {
        if (from_packed)
            velri_gemv_f64_prepared_avx2(plan, args, r0, r1, c0, c1);
        else
            velri_gemv_f64_rowmajor_avx2(plan, args, r0, r1, c0, c1);
        return;
    }

    const uint8_t *pk_base = (const uint8_t *)args->weights;

    if (from_packed) {
        /* Panel offsets and bounds come from the workspace declarations above. */
        enum { MC_L3 = F64_P_BAND };
        enum { MC = F64_P_MC };
        enum { KC = F64_P_KC };
        enum { NC_TILES = F64_P_TILES };

        uint8_t *sc = align64_ptr(scratch);
        double *acc_base = (double *)sc;
        double *a_pack = (double *)(sc + F64_P_A);
        double *b_pack = (double *)(sc + F64_P_B);

        const size_t C = plan->columns;
        const size_t step_c = (C == 8) ? (NC_TILES * C) : C;

        for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
            size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
            for (size_t cb = c0; cb < c1; cb += step_c) {
                size_t cb_end = (c1 - cb < step_c) ? c1 : (cb + step_c);
                for (size_t rb = r_l3; rb < r_l3_end; rb += MC) {
                    size_t mb = (r_l3_end - rb < MC) ? (r_l3_end - rb) : MC;
                    size_t mr_blocks = (mb + 5) / 6;
                    for (size_t kb = 0; kb < k; kb += KC) {
                        size_t kc = (k - kb < KC) ? (k - kb) : KC;
                        int first_k = (kb == 0) ? 1 : 0;

                        if (C < 8) {
                            const double *wp =
                                (const double *)(pk_base + (cb / C) * plan->tile_stride_b);
                            for (size_t p = 0; p < kc; ++p) {
                                for (size_t j = 0; j < C; ++j) {
                                    b_pack[p * 8 + j] = wp[(kb + p) * C + j];
                                }
                                for (size_t j = C; j < 8; ++j) {
                                    b_pack[p * 8 + j] = 0.0;
                                }
                            }
                        }

                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 6;
                            size_t mr = (mb - r_sub < 6) ? (mb - r_sub) : 6;
                            pack_a_6_f64_avx2(mr, kc, x, args->x_stride, rb + r_sub, kb, a_pack);

                            size_t tile_idx = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                                const double *b_panel =
                                    (C == 8) ? ((const double *)(pk_base +
                                                                 (col0 / 8) * plan->tile_stride_b) +
                                                kb * 8)
                                             : b_pack;
                                double *acc_tile =
                                    acc_base + (tile_idx * (F64_P_MC / F64_MR) + mr_i) * 48;
                                ukernel_6x8_f64_avx2(kc, a_pack, b_panel, acc_tile, first_k);
                            }
                        }
                    }

                    size_t tile_idx = 0;
                    for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                        size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 6;
                            size_t mr = (mb - r_sub < 6) ? (mb - r_sub) : 6;
                            const double *acc_tile =
                                acc_base + (tile_idx * (F64_P_MC / F64_MR) + mr_i) * 48;
                            for (size_t r = 0; r < mr; ++r) {
                                double *ydst = y + (rb + r_sub + r) * args->y_stride + col0;
                                if (width == 8) {
                                    __m256d v0 = _mm256_load_pd(acc_tile + r * 8);
                                    __m256d v1 = _mm256_load_pd(acc_tile + r * 8 + 4);
                                    if (bias) {
                                        v0 = _mm256_add_pd(v0, _mm256_loadu_pd(bias + col0));
                                        v1 = _mm256_add_pd(v1, _mm256_loadu_pd(bias + col0 + 4));
                                    }
                                    _mm256_storeu_pd(ydst, v0);
                                    _mm256_storeu_pd(ydst + 4, v1);
                                } else {
                                    for (size_t j = 0; j < width; ++j) {
                                        ydst[j] =
                                            acc_tile[r * 8 + j] + (bias ? bias[col0 + j] : 0.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    /* Panel offsets and bounds come from the workspace declarations above. */
    enum { MC_RM = F64_R_MC };
    enum { KC_RM = F64_R_KC };
    enum { STEP_C_RM = F64_R_TILES * F64_NR };

    uint8_t *sc = align64_ptr(scratch);
    double *acc_rm = (double *)sc;
    double *a_pack = (double *)(sc + F64_R_A);
    double *b_pack = (double *)(sc + F64_R_B);
    const double *w = (const double *)args->weights;

    for (size_t rb0 = r0; rb0 < r1; rb0 += MC_RM) {
        size_t mb_big = (r1 - rb0 < MC_RM) ? (r1 - rb0) : MC_RM;
        size_t mr_total = (mb_big + 5) / 6;
        for (size_t cb = c0; cb < c1; cb += STEP_C_RM) {
            size_t cb_end = (c1 - cb < STEP_C_RM) ? c1 : (cb + STEP_C_RM);
            for (size_t kb = 0; kb < k; kb += KC_RM) {
                size_t kc = (k - kb < KC_RM) ? (k - kb) : KC_RM;
                int first_k = (kb == 0) ? 1 : 0;

                size_t n_tiles = 0;
                for (size_t col0 = cb; col0 < cb_end; col0 += 8, ++n_tiles) {
                    size_t width = (cb_end - col0 < 8) ? (cb_end - col0) : 8;
                    pack_b_8_rm_f64_avx2(width, kc, w, args->w_stride, col0, kb,
                                         b_pack + n_tiles * 8 * kc);
                }

                for (size_t mr_i = 0; mr_i < mr_total; ++mr_i) {
                    size_t r_sub = mr_i * 6;
                    size_t mr = (mb_big - r_sub < 6) ? (mb_big - r_sub) : 6;
                    pack_a_6_f64_avx2(mr, kc, x, args->x_stride, rb0 + r_sub, kb, a_pack);
                    for (size_t ti = 0; ti < n_tiles; ++ti) {
                        double *acc_tile = acc_rm + (ti * (F64_R_MC / F64_MR) + mr_i) * 48;
                        ukernel_6x8_f64_avx2(kc, a_pack, b_pack + ti * 8 * kc, acc_tile, first_k);
                    }
                }
            }

            size_t ti = 0;
            for (size_t col0 = cb; col0 < cb_end; col0 += 8, ++ti) {
                size_t width = (cb_end - col0 < 8) ? (cb_end - col0) : 8;
                for (size_t mr_i = 0; mr_i < mr_total; ++mr_i) {
                    size_t r_sub = mr_i * 6;
                    size_t mr = (mb_big - r_sub < 6) ? (mb_big - r_sub) : 6;
                    const double *acc_tile = acc_rm + (ti * (F64_R_MC / F64_MR) + mr_i) * 48;
                    for (size_t r = 0; r < mr; ++r) {
                        double *ydst = y + (rb0 + r_sub + r) * args->y_stride + col0;
                        if (width == 8) {
                            __m256d v0 = _mm256_load_pd(acc_tile + r * 8);
                            __m256d v1 = _mm256_load_pd(acc_tile + r * 8 + 4);
                            if (bias) {
                                v0 = _mm256_add_pd(v0, _mm256_loadu_pd(bias + col0));
                                v1 = _mm256_add_pd(v1, _mm256_loadu_pd(bias + col0 + 4));
                            }
                            _mm256_storeu_pd(ydst, v0);
                            _mm256_storeu_pd(ydst + 4, v1);
                        } else {
                            for (size_t j = 0; j < width; ++j) {
                                ydst[j] = acc_tile[r * 8 + j] + (bias ? bias[col0 + j] : 0.0);
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * 4. Saturation-Free Bit-Exact AVX2 INT8 (`vpmovsxbw + vpmaddwd`, k_group = 2)
 *    `a_pack` stride = 8 `int32_t` (`16` `int16_t`, 32B aligned!).
 *    Prepared tile stores `[k_aligned / 2, C, 2]` `int8_t` (1 byte/weight!).
 * ============================================================================ */

#define AVX2_I16_STEP(AOFF, BOFF)                                                                  \
    "vmovdqa " #BOFF "(%[b]), %%ymm12\n\t"                                                         \
    "vmovdqa " #BOFF "+32(%[b]), %%ymm13\n\t"                                                      \
    "vpbroadcastd " #AOFF "+0(%[a]), %%ymm14\n\t"                                                  \
    "vpmaddwd %%ymm12, %%ymm14, %%ymm15\n\tvpaddd %%ymm15, %%ymm0, %%ymm0\n\t"                     \
    "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\tvpaddd %%ymm14, %%ymm1, %%ymm1\n\t"                     \
    "vpbroadcastd " #AOFF "+4(%[a]), %%ymm14\n\t"                                                  \
    "vpmaddwd %%ymm12, %%ymm14, %%ymm15\n\tvpaddd %%ymm15, %%ymm2, %%ymm2\n\t"                     \
    "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\tvpaddd %%ymm14, %%ymm3, %%ymm3\n\t"                     \
    "vpbroadcastd " #AOFF "+8(%[a]), %%ymm14\n\t"                                                  \
    "vpmaddwd %%ymm12, %%ymm14, %%ymm15\n\tvpaddd %%ymm15, %%ymm4, %%ymm4\n\t"                     \
    "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\tvpaddd %%ymm14, %%ymm5, %%ymm5\n\t"                     \
    "vpbroadcastd " #AOFF "+12(%[a]), %%ymm14\n\t"                                                 \
    "vpmaddwd %%ymm12, %%ymm14, %%ymm15\n\tvpaddd %%ymm15, %%ymm6, %%ymm6\n\t"                     \
    "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\tvpaddd %%ymm14, %%ymm7, %%ymm7\n\t"                     \
    "vpbroadcastd " #AOFF "+16(%[a]), %%ymm14\n\t"                                                 \
    "vpmaddwd %%ymm12, %%ymm14, %%ymm15\n\tvpaddd %%ymm15, %%ymm8, %%ymm8\n\t"                     \
    "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\tvpaddd %%ymm14, %%ymm9, %%ymm9\n\t"                     \
    "vpbroadcastd " #AOFF "+20(%[a]), %%ymm14\n\t"                                                 \
    "vpmaddwd %%ymm12, %%ymm14, %%ymm15\n\tvpaddd %%ymm15, %%ymm10, %%ymm10\n\t"                   \
    "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\tvpaddd %%ymm14, %%ymm11, %%ymm11\n\t"

static inline void ukernel_6x16_i16_avx2(size_t kc_pairs, const int16_t *a_pair16,
                                         const int16_t *b_pair16, int32_t *restrict acc,
                                         int first_k) {
    if (kc_pairs == 0) {
        return;
    }
    size_t kc4 = kc_pairs >> 2;
    size_t kc1 = kc_pairs & 3ULL;
    /* clang-format off */
    __asm__ volatile(
        "test %[fk], %[fk]\n\t"
        "jz 1f\n\t"
        "vpxor %%ymm0, %%ymm0, %%ymm0\n\tvpxor %%ymm1, %%ymm1, %%ymm1\n\t"
        "vpxor %%ymm2, %%ymm2, %%ymm2\n\tvpxor %%ymm3, %%ymm3, %%ymm3\n\t"
        "vpxor %%ymm4, %%ymm4, %%ymm4\n\tvpxor %%ymm5, %%ymm5, %%ymm5\n\t"
        "vpxor %%ymm6, %%ymm6, %%ymm6\n\tvpxor %%ymm7, %%ymm7, %%ymm7\n\t"
        "vpxor %%ymm8, %%ymm8, %%ymm8\n\tvpxor %%ymm9, %%ymm9, %%ymm9\n\t"
        "vpxor %%ymm10, %%ymm10, %%ymm10\n\tvpxor %%ymm11, %%ymm11, %%ymm11\n\t"
        "jmp 2f\n\t"
        "1:\n\t"
        "vmovdqa 0(%[acc]),   %%ymm0\n\tvmovdqa 32(%[acc]),  %%ymm1\n\t"
        "vmovdqa 64(%[acc]),  %%ymm2\n\tvmovdqa 96(%[acc]),  %%ymm3\n\t"
        "vmovdqa 128(%[acc]), %%ymm4\n\tvmovdqa 160(%[acc]), %%ymm5\n\t"
        "vmovdqa 192(%[acc]), %%ymm6\n\tvmovdqa 224(%[acc]), %%ymm7\n\t"
        "vmovdqa 256(%[acc]), %%ymm8\n\tvmovdqa 288(%[acc]), %%ymm9\n\t"
        "vmovdqa 320(%[acc]), %%ymm10\n\tvmovdqa 352(%[acc]), %%ymm11\n\t"
        "2:\n\t"
        "test %[kc4], %[kc4]\n\t"
        "jz 4f\n\t"
        ".p2align 4\n\t"
        "3:\n\t"
        AVX2_I16_STEP(0, 0)
        AVX2_I16_STEP(32, 64)
        AVX2_I16_STEP(64, 128)
        AVX2_I16_STEP(96, 192)
        "add $128, %[a]\n\t"
        "add $256, %[b]\n\t"
        "dec %[kc4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kc1], %[kc1]\n\t"
        "jz 6f\n\t"
        ".p2align 4\n\t"
        "5:\n\t"
        AVX2_I16_STEP(0, 0)
        "add $32, %[a]\n\t"
        "add $64, %[b]\n\t"
        "dec %[kc1]\n\t"
        "jnz 5b\n\t"
        "6:\n\t"
        "vmovdqa %%ymm0,  0(%[acc])\n\tvmovdqa %%ymm1,  32(%[acc])\n\t"
        "vmovdqa %%ymm2,  64(%[acc])\n\tvmovdqa %%ymm3,  96(%[acc])\n\t"
        "vmovdqa %%ymm4,  128(%[acc])\n\tvmovdqa %%ymm5,  160(%[acc])\n\t"
        "vmovdqa %%ymm6,  192(%[acc])\n\tvmovdqa %%ymm7,  224(%[acc])\n\t"
        "vmovdqa %%ymm8,  256(%[acc])\n\tvmovdqa %%ymm9,  288(%[acc])\n\t"
        "vmovdqa %%ymm10, 320(%[acc])\n\tvmovdqa %%ymm11, 352(%[acc])\n\t"
        : [a] "+&r"(a_pair16), [b] "+&r"(b_pair16), [kc4] "+&r"(kc4), [kc1] "+&r"(kc1)
        : [acc] "r"(acc), [fk] "r"(first_k)
        : "cc", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15");
    /* clang-format on */
}

static inline void widen_b_16_i8_to_i16_avx2(size_t kc_pairs, const int8_t *src,
                                             int16_t *restrict dst) {
    size_t p = 0;
    for (; p + 4 <= kc_pairs; p += 4) {
        const int8_t *s = src + p * 32;
        int16_t *d = dst + p * 32;
        _mm256_storeu_si256((__m256i *)(d + 0),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 0))));
        _mm256_storeu_si256((__m256i *)(d + 16),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 16))));
        _mm256_storeu_si256((__m256i *)(d + 32),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 32))));
        _mm256_storeu_si256((__m256i *)(d + 48),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 48))));
        _mm256_storeu_si256((__m256i *)(d + 64),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 64))));
        _mm256_storeu_si256((__m256i *)(d + 80),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 80))));
        _mm256_storeu_si256((__m256i *)(d + 96),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 96))));
        _mm256_storeu_si256((__m256i *)(d + 112),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 112))));
    }
    for (; p < kc_pairs; ++p) {
        const int8_t *s = src + p * 32;
        int16_t *d = dst + p * 32;
        _mm256_storeu_si256((__m256i *)(d + 0),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 0))));
        _mm256_storeu_si256((__m256i *)(d + 16),
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(s + 16))));
    }
}

/* Conflict-free 8-row grouped `pack_b` directly into pre-widened `int16_t` `[kc_pairs, 32]` */
static void pack_b_16_rm_i8_to_i16_avx2(size_t nr, size_t kc, size_t kc_pairs, const int8_t *w,
                                        size_t w_stride, size_t c_base, size_t k_base,
                                        int16_t *restrict bp16) {
    size_t p_done = 0;
    const size_t full_pairs = kc / 2;
    if (nr == 16 && full_pairs >= 8) {
        const size_t fp8 = full_pairs & ~7ULL;
        for (size_t jh = 0; jh < 16; jh += 8) {
            const int8_t *w0 = w + (c_base + jh + 0) * w_stride + k_base;
            const int8_t *w1 = w + (c_base + jh + 1) * w_stride + k_base;
            const int8_t *w2 = w + (c_base + jh + 2) * w_stride + k_base;
            const int8_t *w3 = w + (c_base + jh + 3) * w_stride + k_base;
            const int8_t *w4 = w + (c_base + jh + 4) * w_stride + k_base;
            const int8_t *w5 = w + (c_base + jh + 5) * w_stride + k_base;
            const int8_t *w6 = w + (c_base + jh + 6) * w_stride + k_base;
            const int8_t *w7 = w + (c_base + jh + 7) * w_stride + k_base;
            for (size_t p = 0; p < fp8; p += 8) {
                __m128i r0 = _mm_loadu_si128((const __m128i *)(w0 + p * 2));
                __m128i r1 = _mm_loadu_si128((const __m128i *)(w1 + p * 2));
                __m128i r2 = _mm_loadu_si128((const __m128i *)(w2 + p * 2));
                __m128i r3 = _mm_loadu_si128((const __m128i *)(w3 + p * 2));
                __m128i r4 = _mm_loadu_si128((const __m128i *)(w4 + p * 2));
                __m128i r5 = _mm_loadu_si128((const __m128i *)(w5 + p * 2));
                __m128i r6 = _mm_loadu_si128((const __m128i *)(w6 + p * 2));
                __m128i r7 = _mm_loadu_si128((const __m128i *)(w7 + p * 2));
                transpose_8x8_epi16(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
                int16_t *d = bp16 + p * 32 + jh * 2;
                _mm256_storeu_si256((__m256i *)(d + 0 * 32), _mm256_cvtepi8_epi16(r0));
                _mm256_storeu_si256((__m256i *)(d + 1 * 32), _mm256_cvtepi8_epi16(r1));
                _mm256_storeu_si256((__m256i *)(d + 2 * 32), _mm256_cvtepi8_epi16(r2));
                _mm256_storeu_si256((__m256i *)(d + 3 * 32), _mm256_cvtepi8_epi16(r3));
                _mm256_storeu_si256((__m256i *)(d + 4 * 32), _mm256_cvtepi8_epi16(r4));
                _mm256_storeu_si256((__m256i *)(d + 5 * 32), _mm256_cvtepi8_epi16(r5));
                _mm256_storeu_si256((__m256i *)(d + 6 * 32), _mm256_cvtepi8_epi16(r6));
                _mm256_storeu_si256((__m256i *)(d + 7 * 32), _mm256_cvtepi8_epi16(r7));
            }
        }
        p_done = fp8;
    }
    for (size_t p = p_done; p < kc_pairs; ++p) {
        for (size_t j = 0; j < nr; ++j) {
            const int8_t *wj = w + (c_base + j) * w_stride + k_base;
            bp16[(p * 16 + j) * 2 + 0] = (int16_t)wj[p * 2];
            bp16[(p * 16 + j) * 2 + 1] = (p * 2 + 1 < kc) ? (int16_t)wj[p * 2 + 1] : 0;
        }
        for (size_t j = nr; j < 16; ++j) {
            bp16[(p * 16 + j) * 2 + 0] = 0;
            bp16[(p * 16 + j) * 2 + 1] = 0;
        }
    }
}

/* Pack `mr <= 6` rows of `int8_t` into `[kc_pairs, 8]` `int32_t` (`16` `int16_t` pairs, 32B
 * aligned) */
static void pack_a_6_i8_avx2(size_t mr, size_t kc, size_t kc_pairs, const int8_t *x,
                             size_t x_stride, size_t r_base, size_t k_base, int16_t *restrict ap) {
    size_t p = 0;
    const size_t full_pairs = kc / 2;
    if (mr == 6 && full_pairs >= 8) {
        const int8_t *x0 = x + (r_base + 0) * x_stride + k_base;
        const int8_t *x1 = x + (r_base + 1) * x_stride + k_base;
        const int8_t *x2 = x + (r_base + 2) * x_stride + k_base;
        const int8_t *x3 = x + (r_base + 3) * x_stride + k_base;
        const int8_t *x4 = x + (r_base + 4) * x_stride + k_base;
        const int8_t *x5 = x + (r_base + 5) * x_stride + k_base;
        for (; p + 8 <= full_pairs; p += 8) {
            __m256 r0 = _mm256_castsi256_ps(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x0 + p * 2))));
            __m256 r1 = _mm256_castsi256_ps(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x1 + p * 2))));
            __m256 r2 = _mm256_castsi256_ps(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x2 + p * 2))));
            __m256 r3 = _mm256_castsi256_ps(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x3 + p * 2))));
            __m256 r4 = _mm256_castsi256_ps(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x4 + p * 2))));
            __m256 r5 = _mm256_castsi256_ps(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x5 + p * 2))));
            __m256 r6 = _mm256_setzero_ps();
            __m256 r7 = _mm256_setzero_ps();
            transpose_8x8_ps(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            float *d = (float *)(ap + p * 16);
            _mm256_storeu_ps(d + 0, r0);
            _mm256_storeu_ps(d + 8, r1);
            _mm256_storeu_ps(d + 16, r2);
            _mm256_storeu_ps(d + 24, r3);
            _mm256_storeu_ps(d + 32, r4);
            _mm256_storeu_ps(d + 40, r5);
            _mm256_storeu_ps(d + 48, r6);
            _mm256_storeu_ps(d + 56, r7);
        }
    }
    for (; p < kc_pairs; ++p) {
        for (size_t r = 0; r < mr; ++r) {
            const int8_t *xr = x + (r_base + r) * x_stride + k_base;
            ap[(p * 8 + r) * 2 + 0] = (int16_t)xr[p * 2];
            ap[(p * 8 + r) * 2 + 1] = (p * 2 + 1 < kc) ? (int16_t)xr[p * 2 + 1] : 0;
        }
        for (size_t r = mr; r < 8; ++r) {
            ap[(p * 8 + r) * 2 + 0] = 0;
            ap[(p * 8 + r) * 2 + 1] = 0;
        }
    }
}

/* Conflict-free 8-row grouped `pack_b` for 16-column `int8_t` pair panels (`[kc_pairs, 16, 2]`) */
static void pack_b_16_rm_i8_avx2(size_t nr, size_t kc, size_t kc_pairs, const int8_t *w,
                                 size_t w_stride, size_t c_base, size_t k_base,
                                 int8_t *restrict bp) {
    size_t p_done = 0;
    const size_t full_pairs = kc / 2;
    if (nr == 16 && full_pairs >= 8) {
        const size_t fp8 = full_pairs & ~7ULL;
        for (size_t jh = 0; jh < 16; jh += 8) {
            const int8_t *w0 = w + (c_base + jh + 0) * w_stride + k_base;
            const int8_t *w1 = w + (c_base + jh + 1) * w_stride + k_base;
            const int8_t *w2 = w + (c_base + jh + 2) * w_stride + k_base;
            const int8_t *w3 = w + (c_base + jh + 3) * w_stride + k_base;
            const int8_t *w4 = w + (c_base + jh + 4) * w_stride + k_base;
            const int8_t *w5 = w + (c_base + jh + 5) * w_stride + k_base;
            const int8_t *w6 = w + (c_base + jh + 6) * w_stride + k_base;
            const int8_t *w7 = w + (c_base + jh + 7) * w_stride + k_base;
            for (size_t p = 0; p < fp8; p += 8) {
                __m128i r0 = _mm_loadu_si128((const __m128i *)(w0 + p * 2));
                __m128i r1 = _mm_loadu_si128((const __m128i *)(w1 + p * 2));
                __m128i r2 = _mm_loadu_si128((const __m128i *)(w2 + p * 2));
                __m128i r3 = _mm_loadu_si128((const __m128i *)(w3 + p * 2));
                __m128i r4 = _mm_loadu_si128((const __m128i *)(w4 + p * 2));
                __m128i r5 = _mm_loadu_si128((const __m128i *)(w5 + p * 2));
                __m128i r6 = _mm_loadu_si128((const __m128i *)(w6 + p * 2));
                __m128i r7 = _mm_loadu_si128((const __m128i *)(w7 + p * 2));
                transpose_8x8_epi16(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
                int8_t *d = bp + p * 32 + jh * 2;
                _mm_storeu_si128((__m128i *)(d + 0 * 32), r0);
                _mm_storeu_si128((__m128i *)(d + 1 * 32), r1);
                _mm_storeu_si128((__m128i *)(d + 2 * 32), r2);
                _mm_storeu_si128((__m128i *)(d + 3 * 32), r3);
                _mm_storeu_si128((__m128i *)(d + 4 * 32), r4);
                _mm_storeu_si128((__m128i *)(d + 5 * 32), r5);
                _mm_storeu_si128((__m128i *)(d + 6 * 32), r6);
                _mm_storeu_si128((__m128i *)(d + 7 * 32), r7);
            }
        }
        p_done = fp8;
    }
    for (size_t p = p_done; p < kc_pairs; ++p) {
        for (size_t j = 0; j < nr; ++j) {
            const int8_t *wj = w + (c_base + j) * w_stride + k_base;
            bp[(p * 16 + j) * 2 + 0] = wj[p * 2];
            bp[(p * 16 + j) * 2 + 1] = (p * 2 + 1 < kc) ? wj[p * 2 + 1] : 0;
        }
        for (size_t j = nr; j < 16; ++j) {
            bp[(p * 16 + j) * 2 + 0] = 0;
            bp[(p * 16 + j) * 2 + 1] = 0;
        }
    }
}

static inline void store_row_i8_epilogue_avx2(const velri_linear_typed_plan *plan,
                                              const velri_linear_run_args *args, size_t row_idx,
                                              size_t col0, size_t width, const int32_t *row_acc,
                                              const float *wsc) {
    if (plan->desc.y_dtype == VELRI_DTYPE_I32) {
        const int32_t *bias = (const int32_t *)args->bias;
        int32_t *ydst = (int32_t *)args->y + row_idx * args->y_stride + col0;
        if (width == 16) {
            __m256i v0 = _mm256_load_si256((const __m256i *)row_acc);
            __m256i v1 = _mm256_load_si256((const __m256i *)(row_acc + 8));
            if (bias) {
                v0 = _mm256_add_epi32(v0, _mm256_loadu_si256((const __m256i *)(bias + col0)));
                v1 = _mm256_add_epi32(v1, _mm256_loadu_si256((const __m256i *)(bias + col0 + 8)));
            }
            _mm256_storeu_si256((__m256i *)ydst, v0);
            _mm256_storeu_si256((__m256i *)(ydst + 8), v1);
        } else {
            for (size_t j = 0; j < width; ++j) {
                ydst[j] = velri_i32_add(row_acc[j], bias ? bias[col0 + j] : 0);
            }
        }
    } else {
        const float *bias = (const float *)args->bias;
        float *ydst = (float *)args->y + row_idx * args->y_stride + col0;
        float xs = args->x_scale ? args->x_scale[row_idx] : 1.0f;
        if (width == 16) {
            __m256 f0 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i *)row_acc));
            __m256 f1 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i *)(row_acc + 8)));
            __m256 vxs = _mm256_set1_ps(xs);
            __m256 sc0 = wsc ? _mm256_mul_ps(vxs, _mm256_loadu_ps(wsc)) : vxs;
            __m256 sc1 = wsc ? _mm256_mul_ps(vxs, _mm256_loadu_ps(wsc + 8)) : vxs;
            if (bias) {
                f0 = _mm256_fmadd_ps(f0, sc0, _mm256_loadu_ps(bias + col0));
                f1 = _mm256_fmadd_ps(f1, sc1, _mm256_loadu_ps(bias + col0 + 8));
            } else {
                f0 = _mm256_mul_ps(f0, sc0);
                f1 = _mm256_mul_ps(f1, sc1);
            }
            _mm256_storeu_ps(ydst, f0);
            _mm256_storeu_ps(ydst + 8, f1);
        } else {
            for (size_t j = 0; j < width; ++j) {
                ydst[j] = (float)row_acc[j] * (xs * (wsc ? wsc[j] : 1.0f)) +
                          (bias ? bias[col0 + j] : 0.0f);
            }
        }
    }
}

static void avx2_i8_run(const velri_linear_typed_plan *plan, const velri_linear_run_args *args,
                        size_t r0, size_t r1, size_t c0, size_t c1, void *restrict scratch,
                        bool from_packed) {
    const size_t k = plan->k;
    const int8_t *x = (const int8_t *)args->x;
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const size_t payload_b = plan->k_aligned * plan->columns * sizeof(int8_t);
    const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
    const size_t scale_off = meta_off + plan->columns * sizeof(int32_t);

    if (r1 - r0 <= 2) {
        if (from_packed) {
            const size_t C = plan->columns;
            const size_t k_pairs = plan->k_aligned / 2;
            for (size_t col0 = c0; col0 < c1; col0 += C) {
                size_t width = (c1 - col0 < C) ? (c1 - col0) : C;
                const uint8_t *tp = pk_base + (col0 / C) * plan->tile_stride_b;
                const int8_t *wp = (const int8_t *)tp;
                const float *wsc =
                    plan->desc.has_w_scale ? (const float *)(tp + scale_off) : nullptr;
                for (size_t r = r0; r < r1; ++r) {
                    const int8_t *xr = x + r * args->x_stride;
                    alignas(32) int32_t out_i32[16] = {0};
                    if (C == 16) {
                        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
                        for (size_t p = 0; p < k_pairs; ++p) {
                            int16_t xp0 = (int16_t)xr[p * 2];
                            int16_t xp1 = (p * 2 + 1 < k) ? (int16_t)xr[p * 2 + 1] : 0;
                            uint32_t xpair = (uint16_t)xp0 | ((uint32_t)(uint16_t)xp1 << 16);
                            __m256i xv = _mm256_set1_epi32((int)xpair);
                            __m256i b0 = _mm256_cvtepi8_epi16(
                                _mm_loadu_si128((const __m128i *)(wp + p * 32)));
                            __m256i b1 = _mm256_cvtepi8_epi16(
                                _mm_loadu_si128((const __m128i *)(wp + p * 32 + 16)));
                            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(xv, b0));
                            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(xv, b1));
                        }
                        _mm256_store_si256((__m256i *)out_i32, a0);
                        _mm256_store_si256((__m256i *)(out_i32 + 8), a1);
                    } else {
                        for (size_t p = 0; p < k_pairs; ++p) {
                            int32_t x0 = (int32_t)xr[p * 2];
                            int32_t x1 = (p * 2 + 1 < k) ? (int32_t)xr[p * 2 + 1] : 0;
                            for (size_t j = 0; j < width; ++j) {
                                out_i32[j] += x0 * (int32_t)wp[(p * C + j) * 2 + 0] +
                                              x1 * (int32_t)wp[(p * C + j) * 2 + 1];
                            }
                        }
                    }
                    store_row_i8_epilogue_avx2(plan, args, r, col0, width, out_i32, wsc);
                }
            }
        } else {
            const int8_t *w = (const int8_t *)args->weights;
            for (size_t r = r0; r < r1; ++r) {
                const int8_t *xr = x + r * args->x_stride;
                float xs = args->x_scale ? args->x_scale[r] : 1.0f;
                size_t c = c0;
                for (; c + 4 <= c1; c += 4) {
                    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
                    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
                    const int8_t *w0 = w + (c + 0) * args->w_stride;
                    const int8_t *w1 = w + (c + 1) * args->w_stride;
                    const int8_t *w2 = w + (c + 2) * args->w_stride;
                    const int8_t *w3 = w + (c + 3) * args->w_stride;
                    size_t i = 0;
                    for (; i + 16 <= k; i += 16) {
                        __m256i vx =
                            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xr + i)));
                        a0 = _mm256_add_epi32(
                            a0, _mm256_madd_epi16(vx, _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                                          (const __m128i *)(w0 + i)))));
                        a1 = _mm256_add_epi32(
                            a1, _mm256_madd_epi16(vx, _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                                          (const __m128i *)(w1 + i)))));
                        a2 = _mm256_add_epi32(
                            a2, _mm256_madd_epi16(vx, _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                                          (const __m128i *)(w2 + i)))));
                        a3 = _mm256_add_epi32(
                            a3, _mm256_madd_epi16(vx, _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                                          (const __m128i *)(w3 + i)))));
                    }
                    int32_t s[4] = {hsum256_epi32(a0), hsum256_epi32(a1), hsum256_epi32(a2),
                                    hsum256_epi32(a3)};
                    for (; i < k; ++i) {
                        int32_t xv = (int32_t)xr[i];
                        s[0] = velri_i32_add(s[0], xv * (int32_t)w0[i]);
                        s[1] = velri_i32_add(s[1], xv * (int32_t)w1[i]);
                        s[2] = velri_i32_add(s[2], xv * (int32_t)w2[i]);
                        s[3] = velri_i32_add(s[3], xv * (int32_t)w3[i]);
                    }
                    for (size_t j = 0; j < 4; ++j) {
                        if (plan->desc.y_dtype == VELRI_DTYPE_I32) {
                            const int32_t *bias = (const int32_t *)args->bias;
                            ((int32_t *)args->y)[r * args->y_stride + c + j] =
                                velri_i32_add(s[j], bias ? bias[c + j] : 0);
                        } else {
                            const float *bias = (const float *)args->bias;
                            float ws = args->w_scale ? args->w_scale[c + j] : 1.0f;
                            ((float *)args->y)[r * args->y_stride + c + j] =
                                (float)s[j] * (xs * ws) + (bias ? bias[c + j] : 0.0f);
                        }
                    }
                }
                for (; c < c1; ++c) {
                    const int8_t *wc = w + c * args->w_stride;
                    __m256i a0 = _mm256_setzero_si256();
                    size_t i = 0;
                    for (; i + 16 <= k; i += 16) {
                        a0 = _mm256_add_epi32(
                            a0,
                            _mm256_madd_epi16(
                                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xr + i))),
                                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(wc + i)))));
                    }
                    int32_t sum = hsum256_epi32(a0);
                    for (; i < k; ++i) {
                        sum = velri_i32_add(sum, (int32_t)xr[i] * (int32_t)wc[i]);
                    }
                    if (plan->desc.y_dtype == VELRI_DTYPE_I32) {
                        const int32_t *bias = (const int32_t *)args->bias;
                        ((int32_t *)args->y)[r * args->y_stride + c] =
                            velri_i32_add(sum, bias ? bias[c] : 0);
                    } else {
                        const float *bias = (const float *)args->bias;
                        float ws = args->w_scale ? args->w_scale[c] : 1.0f;
                        ((float *)args->y)[r * args->y_stride + c] =
                            (float)sum * (xs * ws) + (bias ? bias[c] : 0.0f);
                    }
                }
            }
        }
        return;
    }

    if (from_packed) {
        /* Panel offsets and bounds come from the workspace declarations above. */
        enum { MC_L3 = I8_P_BAND };
        enum { MC = I8_P_MC };
        enum { KC = I8_P_KC };
        enum { NC_TILES = I8_P_TILES };

        uint8_t *sc = align64_ptr(scratch);
        int32_t *acc_base = (int32_t *)sc;
        int16_t *a_pack = (int16_t *)(sc + I8_P_A);
        int16_t *b_pack16 = (int16_t *)(sc + I8_P_B);

        const size_t C = plan->columns;
        const size_t step_c = (C == 16) ? (NC_TILES * C) : C;

        for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
            size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
            for (size_t cb = c0; cb < c1; cb += step_c) {
                size_t cb_end = (c1 - cb < step_c) ? c1 : (cb + step_c);
                for (size_t rb = r_l3; rb < r_l3_end; rb += MC) {
                    size_t mb = (r_l3_end - rb < MC) ? (r_l3_end - rb) : MC;
                    size_t mr_blocks = (mb + 5) / 6;

                    for (size_t kb = 0; kb < k; kb += KC) {
                        size_t kc = (k - kb < KC) ? (k - kb) : KC;
                        size_t kc_pairs = (kc + 1) / 2;
                        int first_k = (kb == 0) ? 1 : 0;

                        /* Pre-widen `int8 -> int16` (`vpmovsxbw`) ONCE per `(cb, rb, kb)` block! */
                        size_t n_tiles = 0;
                        for (size_t col0 = cb; col0 < cb_end; col0 += C, ++n_tiles) {
                            int16_t *bp_dst = b_pack16 + n_tiles * 32 * kc_pairs;
                            if (C == 16) {
                                const int8_t *src8 =
                                    (const int8_t *)(pk_base + (col0 / 16) * plan->tile_stride_b) +
                                    (kb / 2) * 32;
                                widen_b_16_i8_to_i16_avx2(kc_pairs, src8, bp_dst);
                            } else {
                                const int8_t *wp =
                                    (const int8_t *)(pk_base + (col0 / C) * plan->tile_stride_b);
                                for (size_t p = 0; p < kc_pairs; ++p) {
                                    size_t kp = (kb / 2) + p;
                                    for (size_t j = 0; j < C; ++j) {
                                        bp_dst[(p * 16 + j) * 2 + 0] =
                                            (int16_t)wp[(kp * C + j) * 2 + 0];
                                        bp_dst[(p * 16 + j) * 2 + 1] =
                                            (int16_t)wp[(kp * C + j) * 2 + 1];
                                    }
                                    for (size_t j = C; j < 16; ++j) {
                                        bp_dst[(p * 16 + j) * 2 + 0] = 0;
                                        bp_dst[(p * 16 + j) * 2 + 1] = 0;
                                    }
                                }
                            }
                        }

                        /* `mr_i` OUTER, `ti` INNER: 8 KiB `a_pack` is reused across output tiles.
                         */
                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 6;
                            size_t mr = (mb - r_sub < 6) ? (mb - r_sub) : 6;
                            pack_a_6_i8_avx2(mr, kc, kc_pairs, x, args->x_stride, rb + r_sub, kb,
                                             a_pack);
                            for (size_t ti = 0; ti < n_tiles; ++ti) {
                                int32_t *acc_tile = acc_base + (ti * (I8_P_MC / I8_MR) + mr_i) * 96;
                                ukernel_6x16_i16_avx2(kc_pairs, a_pack,
                                                      b_pack16 + ti * 32 * kc_pairs, acc_tile,
                                                      first_k);
                            }
                        }
                    }

                    size_t tile_idx = 0;
                    for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                        size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                        const float *wsc =
                            plan->desc.has_w_scale
                                ? (const float *)(pk_base + (col0 / C) * plan->tile_stride_b +
                                                  scale_off)
                                : nullptr;
                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 6;
                            size_t mr = (mb - r_sub < 6) ? (mb - r_sub) : 6;
                            const int32_t *acc_tile =
                                acc_base + (tile_idx * (I8_P_MC / I8_MR) + mr_i) * 96;
                            for (size_t r = 0; r < mr; ++r) {
                                store_row_i8_epilogue_avx2(plan, args, rb + r_sub + r, col0, width,
                                                           acc_tile + r * 16, wsc);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    /* Panel offsets and bounds come from the workspace declarations above. */
    enum { MC_RM = I8_R_MC };
    enum { KC_RM = I8_R_KC };
    enum { STEP_C_RM = I8_R_TILES * I8_NR };

    uint8_t *sc = align64_ptr(scratch);
    int32_t *acc_rm = (int32_t *)sc;
    int16_t *a_pack = (int16_t *)(sc + I8_R_A);
    int16_t *b_pack16 = (int16_t *)(sc + I8_R_B);
    const int8_t *w = (const int8_t *)args->weights;

    for (size_t rb0 = r0; rb0 < r1; rb0 += MC_RM) {
        size_t mb_big = (r1 - rb0 < MC_RM) ? (r1 - rb0) : MC_RM;
        size_t mr_total = (mb_big + 5) / 6;
        for (size_t cb = c0; cb < c1; cb += STEP_C_RM) {
            size_t cb_end = (c1 - cb < STEP_C_RM) ? c1 : (cb + STEP_C_RM);
            for (size_t kb = 0; kb < k; kb += KC_RM) {
                size_t kc = (k - kb < KC_RM) ? (k - kb) : KC_RM;
                size_t kc_pairs = (kc + 1) / 2;
                int first_k = (kb == 0) ? 1 : 0;

                size_t n_tiles = 0;
                for (size_t col0 = cb; col0 < cb_end; col0 += 16, ++n_tiles) {
                    size_t width = (cb_end - col0 < 16) ? (cb_end - col0) : 16;
                    pack_b_16_rm_i8_to_i16_avx2(width, kc, kc_pairs, w, args->w_stride, col0, kb,
                                                b_pack16 + n_tiles * 32 * kc_pairs);
                }

                for (size_t mr_i = 0; mr_i < mr_total; ++mr_i) {
                    size_t r_sub = mr_i * 6;
                    size_t mr = (mb_big - r_sub < 6) ? (mb_big - r_sub) : 6;
                    pack_a_6_i8_avx2(mr, kc, kc_pairs, x, args->x_stride, rb0 + r_sub, kb, a_pack);
                    for (size_t ti = 0; ti < n_tiles; ++ti) {
                        int32_t *acc_tile = acc_rm + (ti * (I8_R_MC / I8_MR) + mr_i) * 96;
                        ukernel_6x16_i16_avx2(kc_pairs, a_pack, b_pack16 + ti * 32 * kc_pairs,
                                              acc_tile, first_k);
                    }
                }
            }

            size_t ti = 0;
            for (size_t col0 = cb; col0 < cb_end; col0 += 16, ++ti) {
                size_t width = (cb_end - col0 < 16) ? (cb_end - col0) : 16;
                const float *wsc =
                    (plan->desc.has_w_scale && args->w_scale) ? (args->w_scale + col0) : nullptr;
                for (size_t mr_i = 0; mr_i < mr_total; ++mr_i) {
                    size_t r_sub = mr_i * 6;
                    size_t mr = (mb_big - r_sub < 6) ? (mb_big - r_sub) : 6;
                    const int32_t *acc_tile = acc_rm + (ti * (I8_R_MC / I8_MR) + mr_i) * 96;
                    for (size_t r = 0; r < mr; ++r) {
                        store_row_i8_epilogue_avx2(plan, args, rb0 + r_sub + r, col0, width,
                                                   acc_tile + r * 16, wsc);
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * 5. SIMD Weight Packers & Dispatch Table (`VELRI_CPU_AVX2`)
 * ============================================================================ */

static void avx2_pack_g1(const velri_linear_typed_plan *p, const void *w, size_t ws,
                         const float *wsc, void *restrict pk) {
    const size_t m = p->m;
    const size_t k = p->k;
    const size_t C = p->columns;
    if (p->desc.w_dtype == VELRI_DTYPE_F32 && C == 16) {
        const float *wf = (const float *)w;
        uint8_t *base = (uint8_t *)pk;
        const size_t payload_b = p->k_aligned * C * sizeof(float);
        const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
        const size_t n_tiles = (m + C - 1) / C;
        for (size_t t = 0; t < n_tiles; ++t) {
            size_t col0 = t * C;
            size_t width = (m - col0 < C) ? (m - col0) : C;
            uint8_t *tp = base + t * p->tile_stride_b;
            memset(tp, 0, p->tile_stride_b);
            pack_b_tile16_f32_avx2(width, k, VELRI_DTYPE_F32, wf, ws, col0, 0, false, 16,
                                   (float *)tp);
            if (p->desc.has_w_scale) {
                float *sdst = (float *)(tp + meta_off);
                for (size_t j = 0; j < width; ++j) {
                    sdst[j] = wsc ? wsc[col0 + j] : 1.0f;
                }
            }
        }
        return;
    }
    if (p->desc.w_dtype == VELRI_DTYPE_F64 && C == 8) {
        const double *wd = (const double *)w;
        uint8_t *base = (uint8_t *)pk;
        const size_t n_tiles = (m + C - 1) / C;
        for (size_t t = 0; t < n_tiles; ++t) {
            size_t col0 = t * C;
            size_t width = (m - col0 < C) ? (m - col0) : C;
            uint8_t *tp = base + t * p->tile_stride_b;
            memset(tp, 0, p->tile_stride_b);
            pack_b_8_rm_f64_avx2(width, k, wd, ws, col0, 0, (double *)tp);
        }
        return;
    }
    velri_pack_generic_tiles(p, 1, w, ws, wsc, pk);
}

static void avx2_pack_g2(const velri_linear_typed_plan *p, const void *w, size_t ws,
                         const float *wsc, void *restrict pk) {
    const size_t m = p->m;
    const size_t k = p->k;
    const size_t C = p->columns;
    if (p->desc.w_dtype == VELRI_DTYPE_I8 && C == 16) {
        const int8_t *wi = (const int8_t *)w;
        uint8_t *base = (uint8_t *)pk;
        const size_t kc_pairs = p->k_aligned / 2;
        const size_t payload_b = p->k_aligned * C * sizeof(int8_t);
        const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
        const size_t scale_off = meta_off + C * sizeof(int32_t);
        const size_t n_tiles = (m + C - 1) / C;
        for (size_t t = 0; t < n_tiles; ++t) {
            size_t col0 = t * C;
            size_t width = (m - col0 < C) ? (m - col0) : C;
            uint8_t *tp = base + t * p->tile_stride_b;
            memset(tp, 0, p->tile_stride_b);
            pack_b_16_rm_i8_avx2(width, k, kc_pairs, wi, ws, col0, 0, (int8_t *)tp);
            int32_t *corr = (int32_t *)(tp + meta_off);
            for (size_t j = 0; j < width; ++j) {
                const int8_t *wj = wi + (col0 + j) * ws;
                int32_t s = 0;
                for (size_t i = 0; i < k; ++i) {
                    s = velri_i32_add(s, (int32_t)wj[i]);
                }
                corr[j] = velri_i32_mul(s, 128);
            }
            if (p->desc.has_w_scale) {
                float *sdst = (float *)(tp + scale_off);
                for (size_t j = 0; j < width; ++j) {
                    sdst[j] = wsc ? wsc[col0 + j] : 1.0f;
                }
            }
        }
        return;
    }
    velri_pack_generic_tiles(p, 2, w, ws, wsc, pk);
}

static void avx2_fp_prep_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                               size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx2_fp_gemm(p, a, r0, r1, c0, c1, sc, true);
}
static void avx2_fp_rm_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                             size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx2_fp_gemm(p, a, r0, r1, c0, c1, sc, false);
}
static void avx2_f64_prep_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx2_f64_run(p, a, r0, r1, c0, c1, sc, true);
}
static void avx2_f64_rm_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                              size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx2_f64_run(p, a, r0, r1, c0, c1, sc, false);
}
static void avx2_i8_prep_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                               size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx2_i8_run(p, a, r0, r1, c0, c1, sc, true);
}
static void avx2_i8_rm_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                             size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx2_i8_run(p, a, r0, r1, c0, c1, sc, false);
}

static const velri_lin_impl g_avx2_impls[] = {
    [VELRI_LIN_F32] = {VELRI_LIN_F32, VELRI_CPU_AVX2, 16, 1, &fp_p, &fp_r, avx2_pack_g1,
                       avx2_fp_prep_entry, avx2_fp_rm_entry},
    [VELRI_LIN_F64] = {VELRI_LIN_F64, VELRI_CPU_AVX2, 8, 1, &f64_p, &f64_r, avx2_pack_g1,
                       avx2_f64_prep_entry, avx2_f64_rm_entry},
    [VELRI_LIN_F16_F32] = {VELRI_LIN_F16_F32, VELRI_CPU_AVX2, 16, 1, &fp_p, &fp_r, avx2_pack_g1,
                           avx2_fp_prep_entry, avx2_fp_rm_entry},
    [VELRI_LIN_F16_F16] = {VELRI_LIN_F16_F16, VELRI_CPU_AVX2, 16, 1, &fp_p, &fp_r, avx2_pack_g1,
                           avx2_fp_prep_entry, avx2_fp_rm_entry},
    [VELRI_LIN_F32X_F16W] = {VELRI_LIN_F32X_F16W, VELRI_CPU_AVX2, 16, 1, &fp_p, &fp_r, avx2_pack_g1,
                             avx2_fp_prep_entry, avx2_fp_rm_entry},
    [VELRI_LIN_BF16_F32] = {VELRI_LIN_BF16_F32, VELRI_CPU_AVX2, 16, 1, &fp_p, &fp_r, avx2_pack_g1,
                            avx2_fp_prep_entry, avx2_fp_rm_entry},
    [VELRI_LIN_BF16_BF16] = {VELRI_LIN_BF16_BF16, VELRI_CPU_AVX2, 16, 1, &fp_p, &fp_r, avx2_pack_g1,
                             avx2_fp_prep_entry, avx2_fp_rm_entry},
    [VELRI_LIN_F32X_BF16W] = {VELRI_LIN_F32X_BF16W, VELRI_CPU_AVX2, 16, 1, &fp_p, &fp_r,
                              avx2_pack_g1, avx2_fp_prep_entry, avx2_fp_rm_entry},
    [VELRI_LIN_I8_I32] = {VELRI_LIN_I8_I32, VELRI_CPU_AVX2, 16, 2, &i8_p, &i8_r, avx2_pack_g2,
                          avx2_i8_prep_entry, avx2_i8_rm_entry},
    [VELRI_LIN_W8A8_F32] = {VELRI_LIN_W8A8_F32, VELRI_CPU_AVX2, 16, 2, &i8_p, &i8_r, avx2_pack_g2,
                            avx2_i8_prep_entry, avx2_i8_rm_entry},
    [VELRI_LIN_F32X_I8W] = {VELRI_LIN_F32X_I8W, VELRI_CPU_AVX2, 16, 1, &fp_p, &fp_r, avx2_pack_g1,
                            avx2_fp_prep_entry, avx2_fp_rm_entry},
};

const velri_lin_impl *velri_lin_get_avx2(velri_lin_kind kind) {
    if ((size_t)kind >= sizeof(g_avx2_impls) / sizeof(g_avx2_impls[0])) {
        return nullptr;
    }
    return &g_avx2_impls[kind];
}
