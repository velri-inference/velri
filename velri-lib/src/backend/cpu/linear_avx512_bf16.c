#include "internal.h"
#include <immintrin.h>
#include "linear_workspace.h"

/*
 * AVX-512 Brain Float 16 Linear Backend (`BF16 * BF16` via `vdpbf16ps`, plus `F32 * BF16`):
 *   - Native `AVX512_BF16` (`vdpbf16ps`): 32 BF16 pairs -> 16 FP32 sums = 64 FLOPs/insn.
 *   - 14x32 inline-asm microkernel (28 ZMM accumulators, 4x K-pair unrolled, 64B-aligned
 *     `stride = 16` pair A-panels, fused `last_k` bias + F32 output epilogue).
 *   - `F32 * BF16` (`VELRI_LIN_F32X_BF16W`): 14x32 inline-asm microkernel expanding BF16 weights
 *     in-register (`vpmovzxwd + vpslld $16`) while keeping full 24-bit FP32 `X` precision.
 *   - Conflict-free 8-row grouped ZMM/YMM transposes for A/B panel packing.
 */

/* MR/NR match the fixed SIMD microkernels. MC, KC and TILES describe
 * cache blocking; workspace offsets and scheduler hints derive from them.
 * A panels pad the microkernel's row count to a vector-friendly stride. */
enum { BF16_MR = 14, BF16_NR = 32, BF16_A_ROWS = 16 };
enum { BF16_P_MC = 126, BF16_P_KC = 512, BF16_P_TILES = 16, BF16_P_BAND = 1008 };
static_assert(BF16_P_MC % BF16_MR == 0);
VELRI_LINEAR_WORKSPACE(BF16_P, BF16_P_MC * BF16_P_TILES * BF16_NR * sizeof(float),
                       BF16_A_ROWS * BF16_P_KC * sizeof(velri_bf16),
                       BF16_NR * BF16_P_KC * sizeof(velri_bf16), 0);
static const velri_lin_blocking bf16_p = {BF16_P_BYTES, BF16_MR,     BF16_P_MC,
                                          BF16_P_TILES, 4 * BF16_MR, 8};
enum { BF16_R_MC = 252, BF16_R_KC = 512, BF16_R_TILES = 8, BF16_R_BAND = 1008 };
static_assert(BF16_R_MC % BF16_MR == 0);
VELRI_LINEAR_WORKSPACE(BF16_R, BF16_R_MC * BF16_R_TILES * BF16_NR * sizeof(float),
                       BF16_A_ROWS * BF16_R_KC * sizeof(velri_bf16),
                       BF16_R_TILES * BF16_NR * BF16_R_KC * sizeof(velri_bf16), 0);
static const velri_lin_blocking bf16_r = {BF16_R_BYTES, BF16_MR,     BF16_R_MC,
                                          BF16_R_TILES, 9 * BF16_MR, 4};
enum { F32_BF16_MR = 14, F32_BF16_NR = 32, F32_BF16_A_ROWS = 16 };
enum { F32_BF16_P_MC = 126, F32_BF16_P_KC = 256, F32_BF16_P_TILES = 16, F32_BF16_P_BAND = 1008 };
static_assert(F32_BF16_P_MC % F32_BF16_MR == 0);
VELRI_LINEAR_WORKSPACE(F32_BF16_P, F32_BF16_P_MC * F32_BF16_P_TILES * F32_BF16_NR * sizeof(float),
                       F32_BF16_A_ROWS * F32_BF16_P_KC * sizeof(float),
                       F32_BF16_NR * F32_BF16_P_KC * sizeof(velri_bf16), 0);
static const velri_lin_blocking f32_bf16_p = {F32_BF16_P_BYTES, F32_BF16_MR,     F32_BF16_P_MC,
                                              F32_BF16_P_TILES, 4 * F32_BF16_MR, 8};
enum { F32_BF16_R_MC = 252, F32_BF16_R_KC = 256, F32_BF16_R_TILES = 8, F32_BF16_R_BAND = 1008 };
static_assert(F32_BF16_R_MC % F32_BF16_MR == 0);
VELRI_LINEAR_WORKSPACE(F32_BF16_R, F32_BF16_R_MC * F32_BF16_R_TILES * F32_BF16_NR * sizeof(float),
                       F32_BF16_A_ROWS * F32_BF16_R_KC * sizeof(float),
                       F32_BF16_R_TILES * F32_BF16_NR * F32_BF16_R_KC * sizeof(velri_bf16), 0);
static const velri_lin_blocking f32_bf16_r = {F32_BF16_R_BYTES, F32_BF16_MR,     F32_BF16_R_MC,
                                              F32_BF16_R_TILES, 9 * F32_BF16_MR, 4};

static_assert(BF16_P_KC % 2 == 0);
static_assert(BF16_R_KC % 2 == 0);

static inline uint8_t *align64_ptr(void *p) {
    uintptr_t u = ((uintptr_t)p + 63ULL) & ~63ULL;
    return (uint8_t *)u;
}

typedef struct {
    float *acc;        /* offset 0  */
    float *y;          /* offset 8  */
    const float *bias; /* offset 16 */
    size_t y_stride_b; /* offset 24 */
    int32_t first_k;   /* offset 32 */
    int32_t fuse_f32;  /* offset 36: 0 = store to acc, 1 = add bias & store F32 to y */
} bf16_epilogue;

static_assert(offsetof(bf16_epilogue, acc) == 0);
static_assert(offsetof(bf16_epilogue, y) == 8);
static_assert(offsetof(bf16_epilogue, bias) == 16);
static_assert(offsetof(bf16_epilogue, y_stride_b) == 24);
static_assert(offsetof(bf16_epilogue, first_k) == 32);
static_assert(offsetof(bf16_epilogue, fuse_f32) == 36);

static inline void transpose_16x16_u32_zmm(__m512 r[16]) {
    __m512 t[16], u[16], v[16];
    for (int i = 0; i < 8; ++i) {
        t[2 * i + 0] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }
    for (int i = 0; i < 4; ++i) {
        u[4 * i + 0] = _mm512_shuffle_ps(t[4 * i + 0], t[4 * i + 2], 0x44);
        u[4 * i + 1] = _mm512_shuffle_ps(t[4 * i + 0], t[4 * i + 2], 0xEE);
        u[4 * i + 2] = _mm512_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0x44);
        u[4 * i + 3] = _mm512_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0xEE);
    }
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = _mm512_shuffle_f32x4(u[j], u[4 + j], 0x44);
        v[j + 4] = _mm512_shuffle_f32x4(u[8 + j], u[12 + j], 0x44);
        v[j + 8] = _mm512_shuffle_f32x4(u[j], u[4 + j], 0xEE);
        v[j + 12] = _mm512_shuffle_f32x4(u[8 + j], u[12 + j], 0xEE);
    }
    for (int j = 0; j < 4; ++j) {
        r[j + 0] = _mm512_shuffle_f32x4(v[j], v[j + 4], 0x88);
        r[j + 4] = _mm512_shuffle_f32x4(v[j], v[j + 4], 0xDD);
        r[j + 8] = _mm512_shuffle_f32x4(v[j + 8], v[j + 12], 0x88);
        r[j + 12] = _mm512_shuffle_f32x4(v[j + 8], v[j + 12], 0xDD);
    }
}

static inline void transpose_16x16_u16_ymm(__m256i r[16]) {
    __m256i t[16], u[16], v[16];
    for (int i = 0; i < 8; ++i) {
        t[2 * i + 0] = _mm256_unpacklo_epi16(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm256_unpackhi_epi16(r[2 * i], r[2 * i + 1]);
    }
    for (int i = 0; i < 4; ++i) {
        u[4 * i + 0] = _mm256_unpacklo_epi32(t[4 * i + 0], t[4 * i + 2]);
        u[4 * i + 1] = _mm256_unpackhi_epi32(t[4 * i + 0], t[4 * i + 2]);
        u[4 * i + 2] = _mm256_unpacklo_epi32(t[4 * i + 1], t[4 * i + 3]);
        u[4 * i + 3] = _mm256_unpackhi_epi32(t[4 * i + 1], t[4 * i + 3]);
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j + 0] = _mm256_unpacklo_epi64(u[8 * i + j], u[8 * i + 4 + j]);
            v[8 * i + 2 * j + 1] = _mm256_unpackhi_epi64(u[8 * i + j], u[8 * i + 4 + j]);
        }
    }
    for (int j = 0; j < 8; ++j) {
        r[j + 0] = _mm256_permute2x128_si256(v[j], v[8 + j], 0x20);
        r[j + 8] = _mm256_permute2x128_si256(v[j], v[8 + j], 0x31);
    }
}

#define BF16_PROLOGUE                                                                              \
    "cmpl $0, 32(%[ep])\n\t"                                                                       \
    "je 1f\n\t"                                                                                    \
    "vpxord %%zmm0,  %%zmm0,  %%zmm0\n\tvpxord %%zmm1,  %%zmm1,  %%zmm1\n\t"                       \
    "vpxord %%zmm2,  %%zmm2,  %%zmm2\n\tvpxord %%zmm3,  %%zmm3,  %%zmm3\n\t"                       \
    "vpxord %%zmm4,  %%zmm4,  %%zmm4\n\tvpxord %%zmm5,  %%zmm5,  %%zmm5\n\t"                       \
    "vpxord %%zmm6,  %%zmm6,  %%zmm6\n\tvpxord %%zmm7,  %%zmm7,  %%zmm7\n\t"                       \
    "vpxord %%zmm8,  %%zmm8,  %%zmm8\n\tvpxord %%zmm9,  %%zmm9,  %%zmm9\n\t"                       \
    "vpxord %%zmm10, %%zmm10, %%zmm10\n\tvpxord %%zmm11, %%zmm11, %%zmm11\n\t"                     \
    "vpxord %%zmm12, %%zmm12, %%zmm12\n\tvpxord %%zmm13, %%zmm13, %%zmm13\n\t"                     \
    "vpxord %%zmm14, %%zmm14, %%zmm14\n\tvpxord %%zmm15, %%zmm15, %%zmm15\n\t"                     \
    "vpxord %%zmm16, %%zmm16, %%zmm16\n\tvpxord %%zmm17, %%zmm17, %%zmm17\n\t"                     \
    "vpxord %%zmm18, %%zmm18, %%zmm18\n\tvpxord %%zmm19, %%zmm19, %%zmm19\n\t"                     \
    "vpxord %%zmm20, %%zmm20, %%zmm20\n\tvpxord %%zmm21, %%zmm21, %%zmm21\n\t"                     \
    "vpxord %%zmm22, %%zmm22, %%zmm22\n\tvpxord %%zmm23, %%zmm23, %%zmm23\n\t"                     \
    "vpxord %%zmm24, %%zmm24, %%zmm24\n\tvpxord %%zmm25, %%zmm25, %%zmm25\n\t"                     \
    "vpxord %%zmm26, %%zmm26, %%zmm26\n\tvpxord %%zmm27, %%zmm27, %%zmm27\n\t"                     \
    "jmp 2f\n\t"                                                                                   \
    "1:\n\t"                                                                                       \
    "movq 0(%[ep]), %[ytmp]\n\t"                                                                   \
    "vmovaps 0(%[ytmp]),    %%zmm0\n\tvmovaps 64(%[ytmp]),   %%zmm1\n\t"                           \
    "vmovaps 128(%[ytmp]),  %%zmm2\n\tvmovaps 192(%[ytmp]),  %%zmm3\n\t"                           \
    "vmovaps 256(%[ytmp]),  %%zmm4\n\tvmovaps 320(%[ytmp]),  %%zmm5\n\t"                           \
    "vmovaps 384(%[ytmp]),  %%zmm6\n\tvmovaps 448(%[ytmp]),  %%zmm7\n\t"                           \
    "vmovaps 512(%[ytmp]),  %%zmm8\n\tvmovaps 576(%[ytmp]),  %%zmm9\n\t"                           \
    "vmovaps 640(%[ytmp]),  %%zmm10\n\tvmovaps 704(%[ytmp]),  %%zmm11\n\t"                         \
    "vmovaps 768(%[ytmp]),  %%zmm12\n\tvmovaps 832(%[ytmp]),  %%zmm13\n\t"                         \
    "vmovaps 896(%[ytmp]),  %%zmm14\n\tvmovaps 960(%[ytmp]),  %%zmm15\n\t"                         \
    "vmovaps 1024(%[ytmp]), %%zmm16\n\tvmovaps 1088(%[ytmp]), %%zmm17\n\t"                         \
    "vmovaps 1152(%[ytmp]), %%zmm18\n\tvmovaps 1216(%[ytmp]), %%zmm19\n\t"                         \
    "vmovaps 1280(%[ytmp]), %%zmm20\n\tvmovaps 1344(%[ytmp]), %%zmm21\n\t"                         \
    "vmovaps 1408(%[ytmp]), %%zmm22\n\tvmovaps 1472(%[ytmp]), %%zmm23\n\t"                         \
    "vmovaps 1536(%[ytmp]), %%zmm24\n\tvmovaps 1600(%[ytmp]), %%zmm25\n\t"                         \
    "vmovaps 1664(%[ytmp]), %%zmm26\n\tvmovaps 1728(%[ytmp]), %%zmm27\n\t"                         \
    "2:\n\t"

#define BF16_EPILOGUE(CNT_REG)                                                                     \
    "cmpl $0, 36(%[ep])\n\t"                                                                       \
    "jne 7f\n\t"                                                                                   \
    "movq 0(%[ep]), %[ytmp]\n\t"                                                                   \
    "vmovaps %%zmm0,  0(%[ytmp])\n\tvmovaps %%zmm1,  64(%[ytmp])\n\t"                              \
    "vmovaps %%zmm2,  128(%[ytmp])\n\tvmovaps %%zmm3,  192(%[ytmp])\n\t"                           \
    "vmovaps %%zmm4,  256(%[ytmp])\n\tvmovaps %%zmm5,  320(%[ytmp])\n\t"                           \
    "vmovaps %%zmm6,  384(%[ytmp])\n\tvmovaps %%zmm7,  448(%[ytmp])\n\t"                           \
    "vmovaps %%zmm8,  512(%[ytmp])\n\tvmovaps %%zmm9,  576(%[ytmp])\n\t"                           \
    "vmovaps %%zmm10, 640(%[ytmp])\n\tvmovaps %%zmm11, 704(%[ytmp])\n\t"                           \
    "vmovaps %%zmm12, 768(%[ytmp])\n\tvmovaps %%zmm13, 832(%[ytmp])\n\t"                           \
    "vmovaps %%zmm14, 896(%[ytmp])\n\tvmovaps %%zmm15, 960(%[ytmp])\n\t"                           \
    "vmovaps %%zmm16, 1024(%[ytmp])\n\tvmovaps %%zmm17, 1088(%[ytmp])\n\t"                         \
    "vmovaps %%zmm18, 1152(%[ytmp])\n\tvmovaps %%zmm19, 1216(%[ytmp])\n\t"                         \
    "vmovaps %%zmm20, 1280(%[ytmp])\n\tvmovaps %%zmm21, 1344(%[ytmp])\n\t"                         \
    "vmovaps %%zmm22, 1408(%[ytmp])\n\tvmovaps %%zmm23, 1472(%[ytmp])\n\t"                         \
    "vmovaps %%zmm24, 1536(%[ytmp])\n\tvmovaps %%zmm25, 1600(%[ytmp])\n\t"                         \
    "vmovaps %%zmm26, 1664(%[ytmp])\n\tvmovaps %%zmm27, 1728(%[ytmp])\n\t"                         \
    "jmp 9f\n\t"                                                                                   \
    "7:\n\t"                                                                                       \
    "movq 16(%[ep]), %[ytmp]\n\t"                                                                  \
    "test %[ytmp], %[ytmp]\n\t"                                                                    \
    "jz 8f\n\t"                                                                                    \
    "vmovups 0(%[ytmp]),  %%zmm28\n\tvmovups 64(%[ytmp]), %%zmm29\n\t"                             \
    "vaddps %%zmm28, %%zmm0,  %%zmm0\n\tvaddps %%zmm29, %%zmm1,  %%zmm1\n\t"                       \
    "vaddps %%zmm28, %%zmm2,  %%zmm2\n\tvaddps %%zmm29, %%zmm3,  %%zmm3\n\t"                       \
    "vaddps %%zmm28, %%zmm4,  %%zmm4\n\tvaddps %%zmm29, %%zmm5,  %%zmm5\n\t"                       \
    "vaddps %%zmm28, %%zmm6,  %%zmm6\n\tvaddps %%zmm29, %%zmm7,  %%zmm7\n\t"                       \
    "vaddps %%zmm28, %%zmm8,  %%zmm8\n\tvaddps %%zmm29, %%zmm9,  %%zmm9\n\t"                       \
    "vaddps %%zmm28, %%zmm10, %%zmm10\n\tvaddps %%zmm29, %%zmm11, %%zmm11\n\t"                     \
    "vaddps %%zmm28, %%zmm12, %%zmm12\n\tvaddps %%zmm29, %%zmm13, %%zmm13\n\t"                     \
    "vaddps %%zmm28, %%zmm14, %%zmm14\n\tvaddps %%zmm29, %%zmm15, %%zmm15\n\t"                     \
    "vaddps %%zmm28, %%zmm16, %%zmm16\n\tvaddps %%zmm29, %%zmm17, %%zmm17\n\t"                     \
    "vaddps %%zmm28, %%zmm18, %%zmm18\n\tvaddps %%zmm29, %%zmm19, %%zmm19\n\t"                     \
    "vaddps %%zmm28, %%zmm20, %%zmm20\n\tvaddps %%zmm29, %%zmm21, %%zmm21\n\t"                     \
    "vaddps %%zmm28, %%zmm22, %%zmm22\n\tvaddps %%zmm29, %%zmm23, %%zmm23\n\t"                     \
    "vaddps %%zmm28, %%zmm24, %%zmm24\n\tvaddps %%zmm29, %%zmm25, %%zmm25\n\t"                     \
    "vaddps %%zmm28, %%zmm26, %%zmm26\n\tvaddps %%zmm29, %%zmm27, %%zmm27\n\t"                     \
    "8:\n\t"                                                                                       \
    "movq 8(%[ep]),  %[ytmp]\n\t"                                                                  \
    "movq 24(%[ep]), %[" #CNT_REG "]\n\t"                                                          \
    "vmovups %%zmm0,  0(%[ytmp])\n\tvmovups %%zmm1,  64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm2,  0(%[ytmp])\n\tvmovups %%zmm3,  64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm4,  0(%[ytmp])\n\tvmovups %%zmm5,  64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm6,  0(%[ytmp])\n\tvmovups %%zmm7,  64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm8,  0(%[ytmp])\n\tvmovups %%zmm9,  64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm10, 0(%[ytmp])\n\tvmovups %%zmm11, 64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm12, 0(%[ytmp])\n\tvmovups %%zmm13, 64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm14, 0(%[ytmp])\n\tvmovups %%zmm15, 64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm16, 0(%[ytmp])\n\tvmovups %%zmm17, 64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm18, 0(%[ytmp])\n\tvmovups %%zmm19, 64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm20, 0(%[ytmp])\n\tvmovups %%zmm21, 64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm22, 0(%[ytmp])\n\tvmovups %%zmm23, 64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm24, 0(%[ytmp])\n\tvmovups %%zmm25, 64(%[ytmp])\n\tadd %[" #CNT_REG               \
    "], %[ytmp]\n\t"                                                                               \
    "vmovups %%zmm26, 0(%[ytmp])\n\tvmovups %%zmm27, 64(%[ytmp])\n\t"                              \
    "9:\n\t"

/* ============================================================================
 * 1. Core 14x32 Native `vdpbf16ps` Inline-ASM Microkernel (`stride = 16` pairs = 64 B)
 * ============================================================================ */
static inline void ukernel_14x32_bf16_avx512(size_t kc_pairs, const velri_bf16 *a,
                                             const velri_bf16 *b, const bf16_epilogue *ep) {
    if (kc_pairs == 0) {
        return;
    }
    size_t kp4 = kc_pairs >> 2;
    size_t kp1 = kc_pairs & 3ULL;
    void *ytmp = nullptr;

    /* clang-format off */
    __asm__ volatile(
        BF16_PROLOGUE
        "test %[kp4], %[kp4]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        #define STEP_14X32_BF16(A_OFF, B_OFF)                                                      \
            "vmovdqu64 " #B_OFF "(%[b]), %%zmm28\n\t"                                              \
            "vmovdqu64 " #B_OFF "+64(%[b]), %%zmm29\n\t"                                           \
            "vbroadcastss " #A_OFF "+0(%[a]),  %%zmm30\n\tvbroadcastss " #A_OFF "+4(%[a]),  %%zmm31\n\t" \
            "vdpbf16ps %%zmm28, %%zmm30, %%zmm0\n\tvdpbf16ps %%zmm29, %%zmm30, %%zmm1\n\t"         \
            "vdpbf16ps %%zmm28, %%zmm31, %%zmm2\n\tvdpbf16ps %%zmm29, %%zmm31, %%zmm3\n\t"         \
            "vbroadcastss " #A_OFF "+8(%[a]),  %%zmm30\n\tvbroadcastss " #A_OFF "+12(%[a]), %%zmm31\n\t" \
            "vdpbf16ps %%zmm28, %%zmm30, %%zmm4\n\tvdpbf16ps %%zmm29, %%zmm30, %%zmm5\n\t"         \
            "vdpbf16ps %%zmm28, %%zmm31, %%zmm6\n\tvdpbf16ps %%zmm29, %%zmm31, %%zmm7\n\t"         \
            "vbroadcastss " #A_OFF "+16(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+20(%[a]), %%zmm31\n\t" \
            "vdpbf16ps %%zmm28, %%zmm30, %%zmm8\n\tvdpbf16ps %%zmm29, %%zmm30, %%zmm9\n\t"         \
            "vdpbf16ps %%zmm28, %%zmm31, %%zmm10\n\tvdpbf16ps %%zmm29, %%zmm31, %%zmm11\n\t"       \
            "vbroadcastss " #A_OFF "+24(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+28(%[a]), %%zmm31\n\t" \
            "vdpbf16ps %%zmm28, %%zmm30, %%zmm12\n\tvdpbf16ps %%zmm29, %%zmm30, %%zmm13\n\t"       \
            "vdpbf16ps %%zmm28, %%zmm31, %%zmm14\n\tvdpbf16ps %%zmm29, %%zmm31, %%zmm15\n\t"       \
            "vbroadcastss " #A_OFF "+32(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+36(%[a]), %%zmm31\n\t" \
            "vdpbf16ps %%zmm28, %%zmm30, %%zmm16\n\tvdpbf16ps %%zmm29, %%zmm30, %%zmm17\n\t"       \
            "vdpbf16ps %%zmm28, %%zmm31, %%zmm18\n\tvdpbf16ps %%zmm29, %%zmm31, %%zmm19\n\t"       \
            "vbroadcastss " #A_OFF "+40(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+44(%[a]), %%zmm31\n\t" \
            "vdpbf16ps %%zmm28, %%zmm30, %%zmm20\n\tvdpbf16ps %%zmm29, %%zmm30, %%zmm21\n\t"       \
            "vdpbf16ps %%zmm28, %%zmm31, %%zmm22\n\tvdpbf16ps %%zmm29, %%zmm31, %%zmm23\n\t"       \
            "vbroadcastss " #A_OFF "+48(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+52(%[a]), %%zmm31\n\t" \
            "vdpbf16ps %%zmm28, %%zmm30, %%zmm24\n\tvdpbf16ps %%zmm29, %%zmm30, %%zmm25\n\t"       \
            "vdpbf16ps %%zmm28, %%zmm31, %%zmm26\n\tvdpbf16ps %%zmm29, %%zmm31, %%zmm27\n\t"
        STEP_14X32_BF16(0, 0)
        STEP_14X32_BF16(64, 128)
        STEP_14X32_BF16(128, 256)
        STEP_14X32_BF16(192, 384)
        "add $256, %[a]\n\t"
        "add $512, %[b]\n\t"
        "dec %[kp4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kp1], %[kp1]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        STEP_14X32_BF16(0, 0)
        "add $64, %[a]\n\t"
        "add $128, %[b]\n\t"
        "dec %[kp1]\n\t"
        "jnz 5b\n\t"
        #undef STEP_14X32_BF16
        "6:\n\t"
        BF16_EPILOGUE(kp4)
        : [a] "+&r"(a), [b] "+&r"(b), [kp4] "+&r"(kp4), [kp1] "+&r"(kp1), [ytmp] "=&r"(ytmp)
        : [ep] "r"(ep)
        : "cc", "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
          "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18",
          "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
          "zmm29", "zmm30", "zmm31");
    /* clang-format on */
}

/* 14x32 Mixed `F32 * BF16 -> F32` Microkernel (in-register BF16->FP32 expansion, 28 ZMMs!) */
static inline void ukernel_14x32_bf16w_f32_avx512(size_t kc, const float *a, const velri_bf16 *b,
                                                  const bf16_epilogue *ep) {
    if (kc == 0) {
        return;
    }
    size_t kc4 = kc >> 2;
    size_t kc1 = kc & 3ULL;
    void *ytmp = nullptr;

    /* clang-format off */
    __asm__ volatile(
        BF16_PROLOGUE
        "test %[kc4], %[kc4]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        #define STEP_BF16W_F32(A_OFF, B_OFF)                                                       \
            "vpmovzxwd " #B_OFF "(%[b]), %%zmm28\n\tvpslld $16, %%zmm28, %%zmm28\n\t"              \
            "vpmovzxwd " #B_OFF "+32(%[b]), %%zmm29\n\tvpslld $16, %%zmm29, %%zmm29\n\t"           \
            "vbroadcastss " #A_OFF "+0(%[a]),  %%zmm30\n\tvbroadcastss " #A_OFF "+4(%[a]),  %%zmm31\n\t" \
            "vfmadd231ps %%zmm28, %%zmm30, %%zmm0\n\tvfmadd231ps %%zmm29, %%zmm30, %%zmm1\n\t"     \
            "vfmadd231ps %%zmm28, %%zmm31, %%zmm2\n\tvfmadd231ps %%zmm29, %%zmm31, %%zmm3\n\t"     \
            "vbroadcastss " #A_OFF "+8(%[a]),  %%zmm30\n\tvbroadcastss " #A_OFF "+12(%[a]), %%zmm31\n\t" \
            "vfmadd231ps %%zmm28, %%zmm30, %%zmm4\n\tvfmadd231ps %%zmm29, %%zmm30, %%zmm5\n\t"     \
            "vfmadd231ps %%zmm28, %%zmm31, %%zmm6\n\tvfmadd231ps %%zmm29, %%zmm31, %%zmm7\n\t"     \
            "vbroadcastss " #A_OFF "+16(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+20(%[a]), %%zmm31\n\t" \
            "vfmadd231ps %%zmm28, %%zmm30, %%zmm8\n\tvfmadd231ps %%zmm29, %%zmm30, %%zmm9\n\t"     \
            "vfmadd231ps %%zmm28, %%zmm31, %%zmm10\n\tvfmadd231ps %%zmm29, %%zmm31, %%zmm11\n\t"   \
            "vbroadcastss " #A_OFF "+24(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+28(%[a]), %%zmm31\n\t" \
            "vfmadd231ps %%zmm28, %%zmm30, %%zmm12\n\tvfmadd231ps %%zmm29, %%zmm30, %%zmm13\n\t"   \
            "vfmadd231ps %%zmm28, %%zmm31, %%zmm14\n\tvfmadd231ps %%zmm29, %%zmm31, %%zmm15\n\t"   \
            "vbroadcastss " #A_OFF "+32(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+36(%[a]), %%zmm31\n\t" \
            "vfmadd231ps %%zmm28, %%zmm30, %%zmm16\n\tvfmadd231ps %%zmm29, %%zmm30, %%zmm17\n\t"   \
            "vfmadd231ps %%zmm28, %%zmm31, %%zmm18\n\tvfmadd231ps %%zmm29, %%zmm31, %%zmm19\n\t"   \
            "vbroadcastss " #A_OFF "+40(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+44(%[a]), %%zmm31\n\t" \
            "vfmadd231ps %%zmm28, %%zmm30, %%zmm20\n\tvfmadd231ps %%zmm29, %%zmm30, %%zmm21\n\t"   \
            "vfmadd231ps %%zmm28, %%zmm31, %%zmm22\n\tvfmadd231ps %%zmm29, %%zmm31, %%zmm23\n\t"   \
            "vbroadcastss " #A_OFF "+48(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+52(%[a]), %%zmm31\n\t" \
            "vfmadd231ps %%zmm28, %%zmm30, %%zmm24\n\tvfmadd231ps %%zmm29, %%zmm30, %%zmm25\n\t"   \
            "vfmadd231ps %%zmm28, %%zmm31, %%zmm26\n\tvfmadd231ps %%zmm29, %%zmm31, %%zmm27\n\t"
        STEP_BF16W_F32(0, 0)
        STEP_BF16W_F32(64, 64)
        STEP_BF16W_F32(128, 128)
        STEP_BF16W_F32(192, 192)
        "add $256, %[a]\n\t"
        "add $256, %[b]\n\t"
        "dec %[kc4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kc1], %[kc1]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        STEP_BF16W_F32(0, 0)
        "add $64, %[a]\n\t"
        "add $64, %[b]\n\t"
        "dec %[kc1]\n\t"
        "jnz 5b\n\t"
        #undef STEP_BF16W_F32
        "6:\n\t"
        BF16_EPILOGUE(kc4)
        : [a] "+&r"(a), [b] "+&r"(b), [kc4] "+&r"(kc4), [kc1] "+&r"(kc1), [ytmp] "=&r"(ytmp)
        : [ep] "r"(ep)
        : "cc", "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
          "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18",
          "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
          "zmm29", "zmm30", "zmm31");
    /* clang-format on */
}

/* Pack 14 rows of BF16 X into 64B-aligned `stride = 16` `u32` pairs (`32` BF16/step = 64 B) */
static void pack_a_14_bf16_avx512(size_t mr, size_t kc, const velri_bf16 *x, size_t x_stride,
                                  size_t r_base, size_t k_base, velri_bf16 *restrict dst) {
    size_t p = 0;
    if (mr == 14) {
        for (; p + 32 <= kc; p += 32) {
            __m512 z[16];
            for (size_t r = 0; r < 8; ++r) {
                z[r] = _mm512_loadu_ps((const void *)(x + (r_base + r) * x_stride + k_base + p));
            }
            for (size_t r = 8; r < 14; ++r) {
                z[r] = _mm512_loadu_ps((const void *)(x + (r_base + r) * x_stride + k_base + p));
            }
            z[14] = _mm512_setzero_ps();
            z[15] = _mm512_setzero_ps();
            transpose_16x16_u32_zmm(z);
            float *dst_f = (float *)dst + (p / 2) * 16;
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_ps(dst_f + q * 16, z[q]);
            }
        }
    }
    size_t kc_pairs = (kc + 1) / 2;
    for (size_t qp = p / 2; qp < kc_pairs; ++qp) {
        size_t i0 = qp * 2, i1 = qp * 2 + 1;
        for (size_t r = 0; r < mr; ++r) {
            const velri_bf16 *xr = x + (r_base + r) * x_stride + k_base;
            dst[(qp * 16 + r) * 2 + 0] = xr[i0];
            dst[(qp * 16 + r) * 2 + 1] = (i1 < kc) ? xr[i1] : 0;
        }
        for (size_t r = mr; r < 16; ++r) {
            dst[(qp * 16 + r) * 2 + 0] = 0;
            dst[(qp * 16 + r) * 2 + 1] = 0;
        }
    }
}

static void pack_a_14_f32_for_bf16w(size_t mr, size_t kc, const float *x, size_t x_stride,
                                    size_t r_base, size_t k_base, float *restrict dst) {
    size_t p = 0;
    if (mr == 14) {
        for (; p + 16 <= kc; p += 16) {
            __m512 z[16];
            for (size_t r = 0; r < 8; ++r) {
                z[r] = _mm512_loadu_ps(x + (r_base + r) * x_stride + k_base + p);
            }
            for (size_t r = 8; r < 14; ++r) {
                z[r] = _mm512_loadu_ps(x + (r_base + r) * x_stride + k_base + p);
            }
            z[14] = _mm512_setzero_ps();
            z[15] = _mm512_setzero_ps();
            transpose_16x16_u32_zmm(z);
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_ps(dst + (p + q) * 16, z[q]);
            }
        }
    }
    for (; p < kc; ++p) {
        for (size_t r = 0; r < mr; ++r) {
            dst[p * 16 + r] = x[(r_base + r) * x_stride + k_base + p];
        }
        for (size_t r = mr; r < 16; ++r) {
            dst[p * 16 + r] = 0.0f;
        }
    }
}

static void pack_b_32_bf16_avx512(size_t nr, size_t kc, const velri_bf16 *w_src, size_t w_stride,
                                  size_t c_base, size_t k_base, bool from_packed, size_t packed_C,
                                  velri_bf16 *restrict dst) {
    size_t kc_pairs = (kc + 1) / 2;
    if (from_packed) {
        size_t kp_base = k_base / 2;
        for (size_t qp = 0; qp < kc_pairs; ++qp) {
            for (size_t j = 0; j < packed_C; ++j) {
                dst[(qp * 32 + j) * 2 + 0] = w_src[((kp_base + qp) * packed_C + j) * 2 + 0];
                dst[(qp * 32 + j) * 2 + 1] = w_src[((kp_base + qp) * packed_C + j) * 2 + 1];
            }
            for (size_t j = packed_C; j < 32; ++j) {
                dst[(qp * 32 + j) * 2 + 0] = 0;
                dst[(qp * 32 + j) * 2 + 1] = 0;
            }
        }
        return;
    }

    size_t p = 0;
    if (nr == 32) {
        /* Split into two 16-row groups (read 8 rows at a time) to avoid 4K L1d set thrashing */
        for (; p + 32 <= kc; p += 32) {
            __m512 lo[16];
            for (size_t j = 0; j < 8; ++j) {
                lo[j] =
                    _mm512_loadu_ps((const void *)(w_src + (c_base + j) * w_stride + k_base + p));
            }
            for (size_t j = 8; j < 16; ++j) {
                lo[j] =
                    _mm512_loadu_ps((const void *)(w_src + (c_base + j) * w_stride + k_base + p));
            }
            transpose_16x16_u32_zmm(lo);
            uint32_t *dst32 = (uint32_t *)dst + (p / 2) * 32;
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_ps((void *)(dst32 + q * 32), lo[q]);
            }

            __m512 hi[16];
            for (size_t j = 0; j < 8; ++j) {
                hi[j] = _mm512_loadu_ps(
                    (const void *)(w_src + (c_base + 16 + j) * w_stride + k_base + p));
            }
            for (size_t j = 8; j < 16; ++j) {
                hi[j] = _mm512_loadu_ps(
                    (const void *)(w_src + (c_base + 16 + j) * w_stride + k_base + p));
            }
            transpose_16x16_u32_zmm(hi);
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_ps((void *)(dst32 + q * 32 + 16), hi[q]);
            }
        }
    }
    for (size_t qp = p / 2; qp < kc_pairs; ++qp) {
        size_t i0 = qp * 2, i1 = qp * 2 + 1;
        for (size_t j = 0; j < nr; ++j) {
            const velri_bf16 *wj = w_src + (c_base + j) * w_stride + k_base;
            dst[(qp * 32 + j) * 2 + 0] = wj[i0];
            dst[(qp * 32 + j) * 2 + 1] = (i1 < kc) ? wj[i1] : 0;
        }
        for (size_t j = nr; j < 32; ++j) {
            dst[(qp * 32 + j) * 2 + 0] = 0;
            dst[(qp * 32 + j) * 2 + 1] = 0;
        }
    }
}

static void pack_b_32_bf16_g1_avx512(size_t nr, size_t kc, const velri_bf16 *w_src, size_t w_stride,
                                     size_t c_base, size_t k_base, velri_bf16 *restrict dst) {
    size_t p = 0;
    if (nr == 32) {
        for (; p + 16 <= kc; p += 16) {
            __m256i lo[16];
            for (size_t j = 0; j < 8; ++j) {
                lo[j] = _mm256_loadu_si256(
                    (const __m256i *)(w_src + (c_base + j) * w_stride + k_base + p));
            }
            for (size_t j = 8; j < 16; ++j) {
                lo[j] = _mm256_loadu_si256(
                    (const __m256i *)(w_src + (c_base + j) * w_stride + k_base + p));
            }
            transpose_16x16_u16_ymm(lo);
            for (size_t q = 0; q < 16; ++q) {
                _mm256_storeu_si256((__m256i *)(dst + (p + q) * 32), lo[q]);
            }

            __m256i hi[16];
            for (size_t j = 0; j < 8; ++j) {
                hi[j] = _mm256_loadu_si256(
                    (const __m256i *)(w_src + (c_base + 16 + j) * w_stride + k_base + p));
            }
            for (size_t j = 8; j < 16; ++j) {
                hi[j] = _mm256_loadu_si256(
                    (const __m256i *)(w_src + (c_base + 16 + j) * w_stride + k_base + p));
            }
            transpose_16x16_u16_ymm(hi);
            for (size_t q = 0; q < 16; ++q) {
                _mm256_storeu_si256((__m256i *)(dst + (p + q) * 32 + 16), hi[q]);
            }
        }
    }
    for (; p < kc; ++p) {
        for (size_t j = 0; j < nr; ++j) {
            dst[p * 32 + j] = w_src[(c_base + j) * w_stride + k_base + p];
        }
        for (size_t j = nr; j < 32; ++j) {
            dst[p * 32 + j] = 0;
        }
    }
}

/* Vectorized packer for prepared BF16 weights `[ceil(M/C), ceil(K/2), C, 2]` */
static void avx512_bf16_pack_g2(const velri_linear_typed_plan *plan, const void *w_void,
                                size_t w_stride, const float *w_scale, void *restrict packed) {
    if (plan->columns != 32 || plan->k < 32) {
        velri_pack_generic_tiles(plan, 2, w_void, w_stride, w_scale, packed);
        return;
    }
    const velri_bf16 *w = (const velri_bf16 *)w_void;
    uint8_t *dst_base = (uint8_t *)packed;
    for (size_t col0 = 0, t = 0; col0 < plan->m; col0 += 32, ++t) {
        uint8_t *tile_dst = dst_base + t * plan->tile_stride_b;
        size_t nr = (plan->m - col0 < 32) ? (plan->m - col0) : 32;
        memset(tile_dst, 0, plan->tile_stride_b);
        pack_b_32_bf16_avx512(nr, plan->k, w, w_stride, col0, 0, false, 32, (velri_bf16 *)tile_dst);
    }
}

static void avx512_bf16_pack_g1(const velri_linear_typed_plan *plan, const void *w_void,
                                size_t w_stride, const float *w_scale, void *restrict packed) {
    if (plan->columns != 32 || plan->k < 16) {
        velri_pack_generic_tiles(plan, 1, w_void, w_stride, w_scale, packed);
        return;
    }
    const velri_bf16 *w = (const velri_bf16 *)w_void;
    uint8_t *dst_base = (uint8_t *)packed;
    for (size_t col0 = 0, t = 0; col0 < plan->m; col0 += 32, ++t) {
        uint8_t *tile_dst = dst_base + t * plan->tile_stride_b;
        size_t nr = (plan->m - col0 < 32) ? (plan->m - col0) : 32;
        memset(tile_dst, 0, plan->tile_stride_b);
        pack_b_32_bf16_g1_avx512(nr, plan->k, w, w_stride, col0, 0, (velri_bf16 *)tile_dst);
    }
}

static inline __m256i cvt16_f32_to_bf16_local(__m512 v) {
    const __m512i u = _mm512_castps_si512(v);
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512i inf_bits = _mm512_set1_epi32(0x7F800000);
    const __m512i bias = _mm512_set1_epi32(0x00007FFF);
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i qnan_bit = _mm512_set1_epi32(0x00400000);

    __mmask16 is_nan = _mm512_cmpgt_epi32_mask(_mm512_and_epi32(u, abs_mask), inf_bits);
    __m512i lsb = _mm512_and_epi32(_mm512_srli_epi32(u, 16), one);
    __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(bias, lsb));
    r = _mm512_mask_mov_epi32(r, is_nan, _mm512_or_epi32(u, qnan_bit));
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16));
}

static inline void store_y_bf16_or_f32(velri_dtype y_dt, void *y_base, size_t y_off,
                                       const float *vals, const float *bias, size_t col0,
                                       size_t width) {
    if (width == 32) {
        __m512 v0 = _mm512_loadu_ps(vals);
        __m512 v1 = _mm512_loadu_ps(vals + 16);
        if (bias) {
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias + col0));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias + col0 + 16));
        }
        if (y_dt == VELRI_DTYPE_F32) {
            float *yd = (float *)y_base + y_off;
            _mm512_storeu_ps(yd, v0);
            _mm512_storeu_ps(yd + 16, v1);
        } else {
            velri_bf16 *yd = (velri_bf16 *)y_base + y_off;
            _mm256_storeu_si256((__m256i *)yd, cvt16_f32_to_bf16_local(v0));
            _mm256_storeu_si256((__m256i *)(yd + 16), cvt16_f32_to_bf16_local(v1));
        }
        return;
    }

    if (y_dt == VELRI_DTYPE_F32) {
        float *yd = (float *)y_base + y_off;
        for (size_t j = 0; j < width; ++j) {
            yd[j] = vals[j] + (bias ? bias[col0 + j] : 0.0f);
        }
    } else {
        velri_bf16 *yd = (velri_bf16 *)y_base + y_off;
        for (size_t j = 0; j < width; ++j) {
            yd[j] = velri_f32_to_bf16_scalar(vals[j] + (bias ? bias[col0 + j] : 0.0f));
        }
    }
}

/* ============================================================================
 * 2. Pure BF16 GEMV & Blocked GEMM (`VELRI_LIN_BF16_F32` & `VELRI_LIN_BF16_BF16`)
 * ============================================================================ */
static void avx512_bf16_run(const velri_linear_typed_plan *plan, const velri_linear_run_args *args,
                            size_t r0, size_t r1, size_t c0, size_t c1, void *restrict scratch,
                            bool from_packed) {
    const size_t k = plan->k;
    const velri_bf16 *x = (const velri_bf16 *)args->x;
    const float *bias = (const float *)args->bias;
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const velri_dtype y_dt = plan->desc.y_dtype;
    const size_t y_stride_b = args->y_stride * sizeof(float);

    if (r1 - r0 <= 2) {
        if (from_packed) {
            const size_t C = plan->columns;
            const size_t k_pairs = plan->k_aligned / 2;
            for (size_t col0 = c0; col0 < c1; col0 += C) {
                size_t width = (c1 - col0 < C) ? (c1 - col0) : C;
                const velri_bf16 *wp =
                    (const velri_bf16 *)(pk_base + (col0 / C) * plan->tile_stride_b);
                for (size_t r = r0; r < r1; ++r) {
                    const velri_bf16 *xr = x + r * args->x_stride;
                    alignas(64) float out[32] = {0};
                    if (C == 32) {
                        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
                        for (size_t qp = 0; qp < k_pairs; ++qp) {
                            uint32_t xp =
                                (uint32_t)xr[qp * 2] |
                                ((qp * 2 + 1 < k) ? ((uint32_t)xr[qp * 2 + 1] << 16) : 0U);
                            __m512bh vx = (__m512bh)_mm512_set1_epi32((int)xp);
                            __m512bh wb0 =
                                (__m512bh)_mm512_loadu_si512((const void *)(wp + qp * 64));
                            __m512bh wb1 =
                                (__m512bh)_mm512_loadu_si512((const void *)(wp + qp * 64 + 32));
                            a0 = _mm512_dpbf16_ps(a0, vx, wb0);
                            a1 = _mm512_dpbf16_ps(a1, vx, wb1);
                        }
                        _mm512_store_ps(out, a0);
                        _mm512_store_ps(out + 16, a1);
                    } else {
                        for (size_t qp = 0; qp < k_pairs; ++qp) {
                            float x0 = velri_bf16_to_f32_scalar(xr[qp * 2]);
                            float x1 =
                                (qp * 2 + 1 < k) ? velri_bf16_to_f32_scalar(xr[qp * 2 + 1]) : 0.0f;
                            for (size_t j = 0; j < width; ++j) {
                                out[j] += x0 * velri_bf16_to_f32_scalar(wp[(qp * C + j) * 2 + 0]) +
                                          x1 * velri_bf16_to_f32_scalar(wp[(qp * C + j) * 2 + 1]);
                            }
                        }
                    }
                    store_y_bf16_or_f32(y_dt, args->y, r * args->y_stride + col0, out, bias, col0,
                                        width);
                }
            }
        } else {
            const velri_bf16 *w = (const velri_bf16 *)args->weights;
            for (size_t r = r0; r < r1; ++r) {
                const velri_bf16 *xr = x + r * args->x_stride;
                size_t c = c0;
                for (; c + 4 <= c1; c += 4) {
                    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
                    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
                    const velri_bf16 *w0 = w + (c + 0) * args->w_stride;
                    const velri_bf16 *w1 = w + (c + 1) * args->w_stride;
                    const velri_bf16 *w2 = w + (c + 2) * args->w_stride;
                    const velri_bf16 *w3 = w + (c + 3) * args->w_stride;
                    size_t i = 0;
                    for (; i + 32 <= k; i += 32) {
                        __m512bh vx = (__m512bh)_mm512_loadu_si512((const void *)(xr + i));
                        a0 = _mm512_dpbf16_ps(a0, vx,
                                              (__m512bh)_mm512_loadu_si512((const void *)(w0 + i)));
                        a1 = _mm512_dpbf16_ps(a1, vx,
                                              (__m512bh)_mm512_loadu_si512((const void *)(w1 + i)));
                        a2 = _mm512_dpbf16_ps(a2, vx,
                                              (__m512bh)_mm512_loadu_si512((const void *)(w2 + i)));
                        a3 = _mm512_dpbf16_ps(a3, vx,
                                              (__m512bh)_mm512_loadu_si512((const void *)(w3 + i)));
                    }
                    float s[4] = {_mm512_reduce_add_ps(a0), _mm512_reduce_add_ps(a1),
                                  _mm512_reduce_add_ps(a2), _mm512_reduce_add_ps(a3)};
                    for (; i < k; ++i) {
                        float xv = velri_bf16_to_f32_scalar(xr[i]);
                        s[0] += xv * velri_bf16_to_f32_scalar(w0[i]);
                        s[1] += xv * velri_bf16_to_f32_scalar(w1[i]);
                        s[2] += xv * velri_bf16_to_f32_scalar(w2[i]);
                        s[3] += xv * velri_bf16_to_f32_scalar(w3[i]);
                    }
                    store_y_bf16_or_f32(y_dt, args->y, r * args->y_stride + c, s, bias, c, 4);
                }
                for (; c < c1; ++c) {
                    const velri_bf16 *wc = w + c * args->w_stride;
                    __m512 a0 = _mm512_setzero_ps();
                    size_t i = 0;
                    for (; i + 32 <= k; i += 32) {
                        __m512bh vx = (__m512bh)_mm512_loadu_si512((const void *)(xr + i));
                        a0 = _mm512_dpbf16_ps(a0, vx,
                                              (__m512bh)_mm512_loadu_si512((const void *)(wc + i)));
                    }
                    float sum = _mm512_reduce_add_ps(a0);
                    for (; i < k; ++i) {
                        sum += velri_bf16_to_f32_scalar(xr[i]) * velri_bf16_to_f32_scalar(wc[i]);
                    }
                    store_y_bf16_or_f32(y_dt, args->y, r * args->y_stride + c, &sum, bias, c, 1);
                }
            }
        }
        return;
    }

    uint8_t *sc = align64_ptr(scratch);

    if (from_packed) {
        /* Panel offsets and bounds come from the workspace declarations above. */
        const size_t MC_L3 = BF16_P_BAND;
        const size_t MC = BF16_P_MC;
        const size_t KC = BF16_P_KC;
        const size_t C = plan->columns;
        const size_t max_tiles = (C == 32) ? BF16_P_TILES : 1;
        const size_t step_c = max_tiles * C;

        float *acc_base = (float *)sc;
        velri_bf16 *a_pack = (velri_bf16 *)(sc + BF16_P_A);
        velri_bf16 *b_scratch = (velri_bf16 *)(sc + BF16_P_B);
        alignas(64) float local_acc_p[448];

        for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
            size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
            for (size_t cb = c0; cb < c1; cb += step_c) {
                size_t cb_end = (c1 - cb < step_c) ? c1 : (cb + step_c);
                for (size_t rb = r_l3; rb < r_l3_end; rb += MC) {
                    size_t mb = (r_l3_end - rb < MC) ? (r_l3_end - rb) : MC;
                    size_t mr_blocks = (mb + 13) / 14;

                    for (size_t kb = 0; kb < k; kb += KC) {
                        size_t kc = (k - kb < KC) ? (k - kb) : KC;
                        size_t kc_pairs = (kc + 1) / 2;
                        int32_t first_k = (kb == 0) ? 1 : 0;
                        bool last_k = (kb + kc >= k);

                        if (C < 32) {
                            size_t width0 = (cb_end - cb < C) ? (cb_end - cb) : C;
                            const velri_bf16 *wsrc =
                                (const velri_bf16 *)(pk_base + (cb / C) * plan->tile_stride_b);
                            pack_b_32_bf16_avx512(width0, kc, wsrc, 0, cb, kb, true, C, b_scratch);
                        }

                        /* `mr_i` OUTER, `col0` INNER: 16 KiB `a_pack` is reused across output
                         * tiles. */
                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 14;
                            size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                            pack_a_14_bf16_avx512(mr, kc, x, args->x_stride, rb + r_sub, kb,
                                                  a_pack);

                            size_t tile_idx = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                                size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                                const velri_bf16 *b_panel =
                                    (C == 32)
                                        ? ((const velri_bf16 *)(pk_base +
                                                                (col0 / 32) * plan->tile_stride_b) +
                                           (kb / 2) * 64)
                                        : b_scratch;
                                float *acc_tile =
                                    (first_k && last_k)
                                        ? local_acc_p
                                        : (acc_base +
                                           (tile_idx * (BF16_P_MC / BF16_MR) + mr_i) * 448);
                                bool fuse_f32 =
                                    (last_k && y_dt == VELRI_DTYPE_F32 && mr == 14 && width == 32);
                                bf16_epilogue ep = {
                                    .acc = acc_tile,
                                    .y = (y_dt == VELRI_DTYPE_F32)
                                             ? ((float *)args->y + (rb + r_sub) * args->y_stride +
                                                col0)
                                             : nullptr,
                                    .bias = bias ? (bias + col0) : nullptr,
                                    .y_stride_b = y_stride_b,
                                    .first_k = first_k,
                                    .fuse_f32 = fuse_f32 ? 1 : 0,
                                };
                                ukernel_14x32_bf16_avx512(kc_pairs, a_pack, b_panel, &ep);
                                if (last_k && !fuse_f32) {
                                    for (size_t r = 0; r < mr; ++r) {
                                        store_y_bf16_or_f32(
                                            y_dt, args->y, (rb + r_sub + r) * args->y_stride + col0,
                                            acc_tile + r * 32, bias, col0, width);
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
    const size_t MC_L3 = BF16_R_BAND;
    const size_t MC_RM = BF16_R_MC;
    const size_t NC_RM = BF16_R_TILES * BF16_NR;
    const size_t KC_RM = BF16_R_KC;
    float *acc_base = (float *)sc;
    velri_bf16 *a_pack = (velri_bf16 *)(sc + BF16_R_A);
    velri_bf16 *b_scratch = (velri_bf16 *)(sc + BF16_R_B);
    const velri_bf16 *w = (const velri_bf16 *)args->weights;
    alignas(64) float local_acc[448];

    for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
        size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
        for (size_t cb = c0; cb < c1; cb += NC_RM) {
            size_t cb_end = (c1 - cb < NC_RM) ? c1 : (cb + NC_RM);
            size_t n_tiles = (cb_end - cb + 31) / 32;

            for (size_t rb = r_l3; rb < r_l3_end; rb += MC_RM) {
                size_t mb = (r_l3_end - rb < MC_RM) ? (r_l3_end - rb) : MC_RM;

                for (size_t kb = 0; kb < k; kb += KC_RM) {
                    size_t kc = (k - kb < KC_RM) ? (k - kb) : KC_RM;
                    size_t kc_pairs = (kc + 1) / 2;
                    int32_t first_k = (kb == 0) ? 1 : 0;
                    bool last_k = (kb + kc >= k);

                    for (size_t ti = 0; ti < n_tiles; ++ti) {
                        size_t col0 = cb + ti * 32;
                        size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                        pack_b_32_bf16_avx512(width, kc, w, args->w_stride, col0, kb, false, 32,
                                              b_scratch + ti * 64 * kc_pairs);
                    }

                    for (size_t r_sub = 0, mr_idx = 0; r_sub < mb; r_sub += 14, ++mr_idx) {
                        size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                        size_t r_row = rb + r_sub;
                        pack_a_14_bf16_avx512(mr, kc, x, args->x_stride, r_row, kb, a_pack);

                        for (size_t ti = 0; ti < n_tiles; ++ti) {
                            size_t col0 = cb + ti * 32;
                            size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                            const velri_bf16 *b_panel = b_scratch + ti * 64 * kc_pairs;
                            float *acc_tile =
                                (first_k && last_k)
                                    ? local_acc
                                    : (acc_base + (ti * (BF16_R_MC / BF16_MR) + mr_idx) * 448);
                            bool fuse_f32 =
                                (last_k && y_dt == VELRI_DTYPE_F32 && mr == 14 && width == 32);
                            bf16_epilogue ep = {
                                .acc = acc_tile,
                                .y = (y_dt == VELRI_DTYPE_F32)
                                         ? ((float *)args->y + r_row * args->y_stride + col0)
                                         : nullptr,
                                .bias = bias ? (bias + col0) : nullptr,
                                .y_stride_b = y_stride_b,
                                .first_k = first_k,
                                .fuse_f32 = fuse_f32 ? 1 : 0,
                            };
                            ukernel_14x32_bf16_avx512(kc_pairs, a_pack, b_panel, &ep);
                            if (last_k && !fuse_f32) {
                                for (size_t r = 0; r < mr; ++r) {
                                    store_y_bf16_or_f32(y_dt, args->y,
                                                        (r_row + r) * args->y_stride + col0,
                                                        acc_tile + r * 32, bias, col0, width);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * 3. Mixed `F32 * BF16 -> F32` (`VELRI_LIN_F32X_BF16W`):
 *    Full 24-bit FP32 `X` mantissa + compact 16-bit BF16 weight storage (`k_group = 1`)!
 * ============================================================================ */


static void avx512_f32x_bf16w_run(const velri_linear_typed_plan *plan,
                                  const velri_linear_run_args *args, size_t r0, size_t r1,
                                  size_t c0, size_t c1, void *restrict scratch, bool from_packed) {
    const size_t k = plan->k;
    const float *x = (const float *)args->x;
    const float *bias = (const float *)args->bias;
    float *restrict y = (float *)args->y;
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const size_t y_stride_b = args->y_stride * sizeof(float);

    if (r1 - r0 <= 2) {
        if (from_packed)
            velri_gemv_f32_prepared_avx512(plan, args, r0, r1, c0, c1);
        else
            velri_gemv_f32_rowmajor_avx512(plan, args, r0, r1, c0, c1);
        return;
    }

    uint8_t *sc = align64_ptr(scratch);

    if (from_packed) {
        const size_t MC_L3 = F32_BF16_P_BAND;
        const size_t MC = F32_BF16_P_MC;
        const size_t KC = F32_BF16_P_KC;
        const size_t C = plan->columns;
        const size_t max_tiles = (C == 32) ? F32_BF16_P_TILES : 1;
        const size_t step_c = max_tiles * C;

        float *acc_base = (float *)sc;
        float *a_pack = (float *)(sc + F32_BF16_P_A);
        velri_bf16 *b_scratch = (velri_bf16 *)(sc + F32_BF16_P_B);
        alignas(64) float local_acc_p[448];

        for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
            size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
            for (size_t cb = c0; cb < c1; cb += step_c) {
                size_t cb_end = (c1 - cb < step_c) ? c1 : (cb + step_c);
                for (size_t rb = r_l3; rb < r_l3_end; rb += MC) {
                    size_t mb = (r_l3_end - rb < MC) ? (r_l3_end - rb) : MC;
                    size_t mr_blocks = (mb + 13) / 14;
                    for (size_t kb = 0; kb < k; kb += KC) {
                        size_t kc = (k - kb < KC) ? (k - kb) : KC;
                        int32_t first_k = (kb == 0) ? 1 : 0;
                        bool last_k = (kb + kc >= k);

                        if (C < 32) {
                            const velri_bf16 *wp =
                                (const velri_bf16 *)(pk_base + (cb / C) * plan->tile_stride_b);
                            for (size_t p = 0; p < kc; ++p) {
                                for (size_t j = 0; j < C; ++j) {
                                    b_scratch[p * 32 + j] = wp[(kb + p) * C + j];
                                }
                                for (size_t j = C; j < 32; ++j) {
                                    b_scratch[p * 32 + j] = 0;
                                }
                            }
                        }

                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 14;
                            size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                            pack_a_14_f32_for_bf16w(mr, kc, x, args->x_stride, rb + r_sub, kb,
                                                    a_pack);

                            size_t tile_idx = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                                size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                                const velri_bf16 *b_panel =
                                    (C == 32)
                                        ? ((const velri_bf16 *)(pk_base +
                                                                (col0 / 32) * plan->tile_stride_b) +
                                           kb * 32)
                                        : b_scratch;
                                float *acc_tile =
                                    (first_k && last_k)
                                        ? local_acc_p
                                        : (acc_base +
                                           (tile_idx * (F32_BF16_P_MC / F32_BF16_MR) + mr_i) * 448);
                                bool fuse_f32 = (last_k && mr == 14 && width == 32);
                                bf16_epilogue ep = {
                                    .acc = acc_tile,
                                    .y = y + (rb + r_sub) * args->y_stride + col0,
                                    .bias = bias ? (bias + col0) : nullptr,
                                    .y_stride_b = y_stride_b,
                                    .first_k = first_k,
                                    .fuse_f32 = fuse_f32 ? 1 : 0,
                                };
                                ukernel_14x32_bf16w_f32_avx512(kc, a_pack, b_panel, &ep);
                                if (last_k && !fuse_f32) {
                                    for (size_t r = 0; r < mr; ++r) {
                                        store_y_bf16_or_f32(VELRI_DTYPE_F32, y,
                                                            (rb + r_sub + r) * args->y_stride +
                                                                col0,
                                                            acc_tile + r * 32, bias, col0, width);
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

    const size_t MC_L3 = F32_BF16_R_BAND;
    const size_t MC_RM = F32_BF16_R_MC;
    const size_t NC_RM = F32_BF16_R_TILES * F32_BF16_NR;
    const size_t KC_RM = F32_BF16_R_KC;
    float *acc_base = (float *)sc;
    float *a_pack = (float *)(sc + F32_BF16_R_A);
    velri_bf16 *b_scratch = (velri_bf16 *)(sc + F32_BF16_R_B);
    const velri_bf16 *w = (const velri_bf16 *)args->weights;
    alignas(64) float local_acc[448];

    for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
        size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
        for (size_t cb = c0; cb < c1; cb += NC_RM) {
            size_t cb_end = (c1 - cb < NC_RM) ? c1 : (cb + NC_RM);
            size_t n_tiles = (cb_end - cb + 31) / 32;
            for (size_t rb = r_l3; rb < r_l3_end; rb += MC_RM) {
                size_t mb = (r_l3_end - rb < MC_RM) ? (r_l3_end - rb) : MC_RM;
                for (size_t kb = 0; kb < k; kb += KC_RM) {
                    size_t kc = (k - kb < KC_RM) ? (k - kb) : KC_RM;
                    int32_t first_k = (kb == 0) ? 1 : 0;
                    bool last_k = (kb + kc >= k);

                    for (size_t ti = 0; ti < n_tiles; ++ti) {
                        size_t col0 = cb + ti * 32;
                        size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                        pack_b_32_bf16_g1_avx512(width, kc, w, args->w_stride, col0, kb,
                                                 b_scratch + ti * 32 * kc);
                    }

                    for (size_t r_sub = 0, mr_idx = 0; r_sub < mb; r_sub += 14, ++mr_idx) {
                        size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                        size_t r_row = rb + r_sub;
                        pack_a_14_f32_for_bf16w(mr, kc, x, args->x_stride, r_row, kb, a_pack);

                        for (size_t ti = 0; ti < n_tiles; ++ti) {
                            size_t col0 = cb + ti * 32;
                            size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                            const velri_bf16 *b_panel = b_scratch + ti * 32 * kc;
                            float *acc_tile =
                                (first_k && last_k)
                                    ? local_acc
                                    : (acc_base +
                                       (ti * (F32_BF16_R_MC / F32_BF16_MR) + mr_idx) * 448);
                            bool fuse_f32 = (last_k && mr == 14 && width == 32);
                            bf16_epilogue ep = {
                                .acc = acc_tile,
                                .y = y + r_row * args->y_stride + col0,
                                .bias = bias ? (bias + col0) : nullptr,
                                .y_stride_b = y_stride_b,
                                .first_k = first_k,
                                .fuse_f32 = fuse_f32 ? 1 : 0,
                            };
                            ukernel_14x32_bf16w_f32_avx512(kc, a_pack, b_panel, &ep);
                            if (last_k && !fuse_f32) {
                                for (size_t r = 0; r < mr; ++r) {
                                    store_y_bf16_or_f32(VELRI_DTYPE_F32, y,
                                                        (r_row + r) * args->y_stride + col0,
                                                        acc_tile + r * 32, bias, col0, width);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void avx512_bf16_prep_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                   size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_bf16_run(p, a, r0, r1, c0, c1, sc, true);
}
static void avx512_bf16_rm_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                 size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_bf16_run(p, a, r0, r1, c0, c1, sc, false);
}
static void avx512_f32x_bf16w_prep(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                   size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_f32x_bf16w_run(p, a, r0, r1, c0, c1, sc, true);
}
static void avx512_f32x_bf16w_rm(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                 size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_f32x_bf16w_run(p, a, r0, r1, c0, c1, sc, false);
}

static const velri_lin_impl g_avx512_bf16_impls[] = {
    [VELRI_LIN_BF16_F32] = {VELRI_LIN_BF16_F32, VELRI_CPU_AVX512, 32, 2, &bf16_p, &bf16_r,
                            avx512_bf16_pack_g2, avx512_bf16_prep_entry, avx512_bf16_rm_entry},
    [VELRI_LIN_BF16_BF16] = {VELRI_LIN_BF16_BF16, VELRI_CPU_AVX512, 32, 2, &bf16_p, &bf16_r,
                             avx512_bf16_pack_g2, avx512_bf16_prep_entry, avx512_bf16_rm_entry},
    [VELRI_LIN_F32X_BF16W] = {VELRI_LIN_F32X_BF16W, VELRI_CPU_AVX512, 32, 1, &f32_bf16_p,
                              &f32_bf16_r, avx512_bf16_pack_g1, avx512_f32x_bf16w_prep,
                              avx512_f32x_bf16w_rm},
};

const velri_lin_impl *velri_lin_get_avx512_bf16(velri_lin_kind kind) {
    if (kind < VELRI_LIN_BF16_F32 || kind > VELRI_LIN_F32X_BF16W) {
        return nullptr;
    }
    return &g_avx512_bf16_impls[kind];
}
