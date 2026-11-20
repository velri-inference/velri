#include "internal.h"
#include <immintrin.h>
#include "linear_workspace.h"
#include "gemm_fma.h"

/*
 * AVX-512 Floating-Point Linear Backend (`F32`, `F64`, `F16`, `F32xF16`):
 *   - 14x32 FP32 & 14x32 FP16-weight inline-asm microkernels (28 ZMM accumulators, 4x K-unrolled,
 *     64B-aligned `stride = 16` A-panels, fused `last_k` bias + F32/F16 output epilogue)
 *   - 14x16 FP64 inline-asm microkernel (28 ZMM accumulators, 4x K-unrolled, `stride = 16`
 * A-panels, fused `last_k` bias + F64 output epilogue)
 *   - Conflict-free 8-row grouped 16x16 / 8x8 ZMM transposes for A/B panel packing
 *   - Dedicated skinny-M (`rows == 1..2`) fast paths
 */

/* MR/NR match the fixed SIMD microkernels. MC, KC and TILES describe
 * cache blocking; workspace offsets and scheduler hints derive from them.
 * A panels pad the microkernel's row count to a vector-friendly stride. */
enum { FP_MR = 14, FP_NR = 32, FP_A_ROWS = 16 };
static_assert(FP_MR == FMA_BODY_14X32_PS_ROWS &&
              FP_NR * sizeof(float) == 2 * FMA_BODY_14X32_PS_VECTOR_BYTES &&
              sizeof(float) == FMA_BODY_14X32_PS_ELEMENT_BYTES);
enum { FP_P_MC = 126, FP_P_KC = 256, FP_P_TILES = 16, FP_P_BAND = 1008 };
static_assert(FP_P_MC % FP_MR == 0);
VELRI_LINEAR_WORKSPACE(FP_P, FP_P_MC * FP_P_TILES * FP_NR * sizeof(float),
                       FP_A_ROWS * FP_P_KC * sizeof(float), FP_NR * FP_P_KC * sizeof(float), 0);
static const velri_lin_blocking fp_p = {FP_P_BYTES, FP_MR, FP_P_MC, FP_P_TILES, 4 * FP_MR, 8};
enum { FP_R_MC = 252, FP_R_KC = 256, FP_R_TILES = 8, FP_R_BAND = 1008 };
static_assert(FP_R_MC % FP_MR == 0);
VELRI_LINEAR_WORKSPACE(FP_R, FP_R_MC * FP_R_TILES * FP_NR * sizeof(float),
                       FP_A_ROWS * FP_R_KC * sizeof(float),
                       FP_R_TILES * FP_NR * FP_R_KC * sizeof(float), 0);
static const velri_lin_blocking fp_r = {FP_R_BYTES, FP_MR, FP_R_MC, FP_R_TILES, 9 * FP_MR, 4};
enum { F64_MR = 14, F64_NR = 16, F64_A_ROWS = 16 };
static_assert(F64_MR == STEP_14X16_F64_ROWS &&
              F64_NR * sizeof(double) == 2 * STEP_14X16_F64_VECTOR_BYTES &&
              sizeof(double) == STEP_14X16_F64_ELEMENT_BYTES);
enum { F64_P_MC = 126, F64_P_KC = 128, F64_P_TILES = 8, F64_P_BAND = 504 };
static_assert(F64_P_MC % F64_MR == 0);
/* Keep an extra accumulator-sized span before A. Removing this spacing
 * regressed parallel prepared F64 GEMM despite identical arithmetic code. */
enum { F64_P_ACC_SPANS = 2 };
VELRI_LINEAR_WORKSPACE(F64_P, F64_P_ACC_SPANS * F64_P_MC * F64_P_TILES * F64_NR * sizeof(double),
                       F64_A_ROWS * F64_P_KC * sizeof(double), F64_NR * F64_P_KC * sizeof(double),
                       0);
static const velri_lin_blocking f64_p = {F64_P_BYTES,     F64_MR,     F64_P_MC,
                                         2 * F64_P_TILES, 4 * F64_MR, 8};
enum { F64_R_MC = 252, F64_R_KC = 128, F64_R_TILES = 8, F64_R_BAND = 504 };
static_assert(F64_R_MC % F64_MR == 0);
VELRI_LINEAR_WORKSPACE(F64_R, F64_R_MC * F64_R_TILES * F64_NR * sizeof(double),
                       F64_A_ROWS * F64_R_KC * sizeof(double),
                       F64_R_TILES * F64_NR * F64_R_KC * sizeof(double), 0);
static const velri_lin_blocking f64_r = {F64_R_BYTES, F64_MR, F64_R_MC, F64_R_TILES, 9 * F64_MR, 4};

static inline uint8_t *align64_ptr(void *p) {
    uintptr_t u = ((uintptr_t)p + 63ULL) & ~63ULL;
    return (uint8_t *)u;
}

typedef struct {
    void *acc;         /* offset 0  */
    void *y;           /* offset 8  */
    const void *bias;  /* offset 16 */
    size_t y_stride_b; /* offset 24 */
    int32_t first_k;   /* offset 32 */
    int32_t y_mode;    /* offset 36: 0 = acc, 1 = y_f32/f64, 2 = y_f16 */
} fp_epilogue;

static_assert(offsetof(fp_epilogue, acc) == 0);
static_assert(offsetof(fp_epilogue, y) == 8);
static_assert(offsetof(fp_epilogue, bias) == 16);
static_assert(offsetof(fp_epilogue, y_stride_b) == 24);
static_assert(offsetof(fp_epilogue, first_k) == 32);
static_assert(offsetof(fp_epilogue, y_mode) == 36);

/* ============================================================================
 * 1. Core 14x32 FP32 & 14x32 FP16-Weight Inline-ASM Microkernels
 * ============================================================================ */

#define INIT_OR_LOAD_28_PS                                                                         \
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


#define EPILOGUE_28_PS                                                                             \
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
    "jmp 10f\n\t"                                                                                  \
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
    "movq 24(%[ep]), %[kc4]\n\t"                                                                   \
    "cmpl $2, 36(%[ep])\n\t"                                                                       \
    "je 9f\n\t"                                                                                    \
    "vmovups %%zmm0,  0(%[ytmp])\n\tvmovups %%zmm1,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm2,  0(%[ytmp])\n\tvmovups %%zmm3,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm4,  0(%[ytmp])\n\tvmovups %%zmm5,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm6,  0(%[ytmp])\n\tvmovups %%zmm7,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm8,  0(%[ytmp])\n\tvmovups %%zmm9,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm10, 0(%[ytmp])\n\tvmovups %%zmm11, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm12, 0(%[ytmp])\n\tvmovups %%zmm13, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm14, 0(%[ytmp])\n\tvmovups %%zmm15, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm16, 0(%[ytmp])\n\tvmovups %%zmm17, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm18, 0(%[ytmp])\n\tvmovups %%zmm19, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm20, 0(%[ytmp])\n\tvmovups %%zmm21, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm22, 0(%[ytmp])\n\tvmovups %%zmm23, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm24, 0(%[ytmp])\n\tvmovups %%zmm25, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"       \
    "vmovups %%zmm26, 0(%[ytmp])\n\tvmovups %%zmm27, 64(%[ytmp])\n\t"                              \
    "jmp 10f\n\t"                                                                                  \
    "9:\n\t"                                                                                       \
    "vcvtps2ph $4, %%zmm0,  0(%[ytmp])\n\tvcvtps2ph $4, %%zmm1,  32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm2,  0(%[ytmp])\n\tvcvtps2ph $4, %%zmm3,  32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm4,  0(%[ytmp])\n\tvcvtps2ph $4, %%zmm5,  32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm6,  0(%[ytmp])\n\tvcvtps2ph $4, %%zmm7,  32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm8,  0(%[ytmp])\n\tvcvtps2ph $4, %%zmm9,  32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm10, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm11, 32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm12, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm13, 32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm14, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm15, 32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm16, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm17, 32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm18, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm19, 32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm20, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm21, 32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm22, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm23, 32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm24, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm25, 32(%[ytmp])\n\tadd %[kc4], "      \
    "%[ytmp]\n\t"                                                                                  \
    "vcvtps2ph $4, %%zmm26, 0(%[ytmp])\n\tvcvtps2ph $4, %%zmm27, 32(%[ytmp])\n\t"                  \
    "10:\n\t"

static inline void ukernel_14x32_f32_avx512(size_t kc, const float *a, const float *b,
                                            const fp_epilogue *ep) {
    if (kc == 0) {
        return;
    }
    size_t kc4 = kc >> 2;
    size_t kc1 = kc & 3ULL;
    void *ytmp = nullptr;

    /* clang-format off */
    __asm__ volatile(
        INIT_OR_LOAD_28_PS
        "test %[kc4], %[kc4]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        #define STEP_F32(A_OFF, B_OFF)                                                             \
            "vmovups " #B_OFF "(%[b]), %%zmm28\n\t"                                                \
            "vmovups " #B_OFF "+64(%[b]), %%zmm29\n\t"                                             \
            FMA_BODY_14X32_PS(A_OFF)
        STEP_F32(0, 0)
        STEP_F32(64, 128)
        STEP_F32(128, 256)
        STEP_F32(192, 384)
        "add $256, %[a]\n\t"
        "add $512, %[b]\n\t"
        "dec %[kc4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kc1], %[kc1]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        STEP_F32(0, 0)
        "add $64, %[a]\n\t"
        "add $128, %[b]\n\t"
        "dec %[kc1]\n\t"
        "jnz 5b\n\t"
        #undef STEP_F32
        "6:\n\t"
        EPILOGUE_28_PS
        : [a] "+&r"(a), [b] "+&r"(b), [kc4] "+&r"(kc4), [kc1] "+&r"(kc1), [ytmp] "=&r"(ytmp)
        : [ep] "r"(ep)
        : "cc", "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
          "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18",
          "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
          "zmm29", "zmm30", "zmm31");
    /* clang-format on */
}

static inline void ukernel_14x32_f16w_avx512(size_t kc, const float *a, const velri_fp16 *b,
                                             const fp_epilogue *ep) {
    if (kc == 0) {
        return;
    }
    size_t kc4 = kc >> 2;
    size_t kc1 = kc & 3ULL;
    void *ytmp = nullptr;

    /* clang-format off */
    __asm__ volatile(
        INIT_OR_LOAD_28_PS
        "test %[kc4], %[kc4]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        #define STEP_F16W(A_OFF, B_OFF)                                                            \
            "vcvtph2ps " #B_OFF "(%[b]), %%zmm28\n\t"                                              \
            "vcvtph2ps " #B_OFF "+32(%[b]), %%zmm29\n\t"                                           \
            FMA_BODY_14X32_PS(A_OFF)
        STEP_F16W(0, 0)
        STEP_F16W(64, 64)
        STEP_F16W(128, 128)
        STEP_F16W(192, 192)
        "add $256, %[a]\n\t"
        "add $256, %[b]\n\t"
        "dec %[kc4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kc1], %[kc1]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        STEP_F16W(0, 0)
        "add $64, %[a]\n\t"
        "add $64, %[b]\n\t"
        "dec %[kc1]\n\t"
        "jnz 5b\n\t"
        #undef STEP_F16W
        "6:\n\t"
        EPILOGUE_28_PS
        : [a] "+&r"(a), [b] "+&r"(b), [kc4] "+&r"(kc4), [kc1] "+&r"(kc1), [ytmp] "=&r"(ytmp)
        : [ep] "r"(ep)
        : "cc", "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
          "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18",
          "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
          "zmm29", "zmm30", "zmm31");
    /* clang-format on */
}

/* ============================================================================
 * 2. Conflict-Free 16x16 ZMM Transposes & Fast SIMD Panel Packers (`F32` & `F16`)
 * ============================================================================ */
static inline void transpose_16x16_zmm(__m512 r[16]) {
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

static inline __m512 load16_f32_or_f16(velri_dtype dt, const void *base, size_t idx) {
    if (dt == VELRI_DTYPE_F32) {
        return _mm512_loadu_ps((const float *)base + idx);
    }
    __m256i h = _mm256_loadu_si256((const __m256i *)((const velri_fp16 *)base + idx));
    return _mm512_cvtph_ps(h);
}

static inline float load1_f32_or_f16(velri_dtype dt, const void *base, size_t idx) {
    if (dt == VELRI_DTYPE_F32) {
        return ((const float *)base)[idx];
    }
    return velri_f16_to_f32_scalar(((const velri_fp16 *)base)[idx]);
}

/* Pack 14 rows of X into 64B-aligned `stride = 16` floats per K-step */
static void pack_a_14_fp_avx512(size_t mr, size_t kc, velri_dtype x_dt, const void *x,
                                size_t x_stride, size_t r_base, size_t k_base,
                                float *restrict dst) {
    size_t p = 0;
    if (mr == 14) {
        for (; p + 16 <= kc; p += 16) {
            __m512 z[16];
            for (size_t r = 0; r < 8; ++r) {
                z[r] = load16_f32_or_f16(x_dt, x, (r_base + r) * x_stride + k_base + p);
            }
            for (size_t r = 8; r < 14; ++r) {
                z[r] = load16_f32_or_f16(x_dt, x, (r_base + r) * x_stride + k_base + p);
            }
            z[14] = _mm512_setzero_ps();
            z[15] = _mm512_setzero_ps();
            transpose_16x16_zmm(z);
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_ps(dst + (p + q) * 16, z[q]);
            }
        }
    }
    for (; p < kc; ++p) {
        for (size_t r = 0; r < mr; ++r) {
            dst[p * 16 + r] = load1_f32_or_f16(x_dt, x, (r_base + r) * x_stride + k_base + p);
        }
        for (size_t r = mr; r < 16; ++r) {
            dst[p * 16 + r] = 0.0f;
        }
    }
}

static void pack_b_32_fp_avx512(size_t nr, size_t kc, velri_dtype w_dt, const void *w_src,
                                size_t w_stride, size_t c_base, size_t k_base, bool from_packed,
                                size_t packed_C, float *restrict dst) {
    if (from_packed) {
        for (size_t p = 0; p < kc; ++p) {
            for (size_t j = 0; j < packed_C; ++j) {
                dst[p * 32 + j] = load1_f32_or_f16(w_dt, w_src, (k_base + p) * packed_C + j);
            }
            for (size_t j = packed_C; j < 32; ++j) {
                dst[p * 32 + j] = 0.0f;
            }
        }
        return;
    }

    size_t p = 0;
    if (nr == 32) {
        /* Split into two 16-row groups (each read 8 rows at a time) to avoid 4K L1d set thrashing
         */
        for (; p + 16 <= kc; p += 16) {
            __m512 lo[16];
            for (size_t j = 0; j < 8; ++j) {
                lo[j] = load16_f32_or_f16(w_dt, w_src, (c_base + j) * w_stride + k_base + p);
            }
            for (size_t j = 8; j < 16; ++j) {
                lo[j] = load16_f32_or_f16(w_dt, w_src, (c_base + j) * w_stride + k_base + p);
            }
            transpose_16x16_zmm(lo);
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_ps(dst + (p + q) * 32, lo[q]);
            }

            __m512 hi[16];
            for (size_t j = 0; j < 8; ++j) {
                hi[j] = load16_f32_or_f16(w_dt, w_src, (c_base + 16 + j) * w_stride + k_base + p);
            }
            for (size_t j = 8; j < 16; ++j) {
                hi[j] = load16_f32_or_f16(w_dt, w_src, (c_base + 16 + j) * w_stride + k_base + p);
            }
            transpose_16x16_zmm(hi);
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_ps(dst + (p + q) * 32 + 16, hi[q]);
            }
        }
    }
    for (; p < kc; ++p) {
        for (size_t j = 0; j < nr; ++j) {
            dst[p * 32 + j] = load1_f32_or_f16(w_dt, w_src, (c_base + j) * w_stride + k_base + p);
        }
        for (size_t j = nr; j < 32; ++j) {
            dst[p * 32 + j] = 0.0f;
        }
    }
}

/* Native 16x16 uint16 transpose so Row-Major & offline FP16 packing keep B in compact FP16! */
static inline void transpose_16x16_epi16(__m256i r[16]) {
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

static void pack_b_32_f16_raw_avx512(size_t nr, size_t kc, const velri_fp16 *w_src, size_t w_stride,
                                     size_t c_base, size_t k_base, velri_fp16 *restrict dst) {
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
            transpose_16x16_epi16(lo);
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
            transpose_16x16_epi16(hi);
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

static inline void store_y_fp_avx512(velri_dtype y_dt, void *y_base, size_t y_off,
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
            velri_fp16 *yd = (velri_fp16 *)y_base + y_off;
            _mm256_storeu_si256((__m256i *)yd,
                                _mm512_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            _mm256_storeu_si256((__m256i *)(yd + 16),
                                _mm512_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
        return;
    }

    if (y_dt == VELRI_DTYPE_F32) {
        float *yd = (float *)y_base + y_off;
        for (size_t j = 0; j < width; ++j) {
            yd[j] = vals[j] + (bias ? bias[col0 + j] : 0.0f);
        }
    } else {
        velri_fp16 *yd = (velri_fp16 *)y_base + y_off;
        for (size_t j = 0; j < width; ++j) {
            yd[j] = velri_f32_to_f16_scalar(vals[j] + (bias ? bias[col0 + j] : 0.0f));
        }
    }
}

/* ============================================================================
 * 3. Fast GEMV & Skinny-Batch (`rows == 1..2`) for Prepared & Row-Major FP32/FP16
 * ============================================================================ */




static void avx512_fp_gemm(const velri_linear_typed_plan *plan, const velri_linear_run_args *args,
                           size_t r0, size_t r1, size_t c0, size_t c1, void *restrict scratch,
                           bool from_packed) {
    if (r1 - r0 <= 2) {
        if (from_packed) {
            velri_gemv_f32_prepared_avx512(plan, args, r0, r1, c0, c1);
        } else {
            velri_gemv_f32_rowmajor_avx512(plan, args, r0, r1, c0, c1);
        }
        return;
    }

    const size_t k = plan->k;
    const velri_dtype w_dt = plan->desc.w_dtype;
    const velri_dtype y_dt = plan->desc.y_dtype;
    const float *bias = (const float *)args->bias;
    const size_t y_elem_b = velri_dtype_bytes(y_dt);
    const size_t y_stride_b = args->y_stride * y_elem_b;
    const int full_y_mode = (y_dt == VELRI_DTYPE_F32) ? 1 : 2;
    uint8_t *sc = align64_ptr(scratch);

    if (from_packed) {
        /* Panel offsets and bounds come from the workspace declarations above. */
        const size_t MC_L3 = FP_P_BAND;
        const size_t MC = FP_P_MC;
        const size_t KC = FP_P_KC;
        const size_t C = plan->columns;
        const size_t max_tiles = (C == 32) ? FP_P_TILES : 1;
        const size_t step_c = max_tiles * C;

        float *acc_base = (float *)sc;
        float *a_pack = (float *)(sc + FP_P_A);
        float *b_scratch = (float *)(sc + FP_P_B);
        const uint8_t *pk_base = (const uint8_t *)args->weights;
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
                            pack_b_32_fp_avx512(cb_end - cb, kc, w_dt,
                                                pk_base + (cb / C) * plan->tile_stride_b, 0, cb, kb,
                                                true, C, b_scratch);
                        }

                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 14;
                            size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                            pack_a_14_fp_avx512(mr, kc, plan->desc.x_dtype, args->x, args->x_stride,
                                                rb + r_sub, kb, a_pack);

                            size_t tile_idx = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                                size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                                const uint8_t *tile_ptr =
                                    pk_base + (col0 / C) * plan->tile_stride_b;
                                float *acc_tile =
                                    (first_k && last_k)
                                        ? local_acc_p
                                        : (acc_base + (tile_idx * (FP_P_MC / FP_MR) + mr_i) * 448);

                                if (C == 32 && w_dt == VELRI_DTYPE_F32) {
                                    const float *b_panel = (const float *)tile_ptr + kb * 32;
                                    bool fuse_y = (last_k && mr == 14 && width == 32);
                                    fp_epilogue ep = {
                                        .acc = acc_tile,
                                        .y = (uint8_t *)args->y +
                                             ((rb + r_sub) * args->y_stride + col0) * y_elem_b,
                                        .bias = bias ? (bias + col0) : nullptr,
                                        .y_stride_b = y_stride_b,
                                        .first_k = first_k,
                                        .y_mode = fuse_y ? full_y_mode : 0,
                                    };
                                    ukernel_14x32_f32_avx512(kc, a_pack, b_panel, &ep);
                                    if (last_k && !fuse_y) {
                                        for (size_t r = 0; r < mr; ++r) {
                                            store_y_fp_avx512(y_dt, args->y,
                                                              (rb + r_sub + r) * args->y_stride +
                                                                  col0,
                                                              acc_tile + r * 32, bias, col0, width);
                                        }
                                    }
                                } else if (C == 32 && w_dt == VELRI_DTYPE_F16) {
                                    const velri_fp16 *b_panel =
                                        (const velri_fp16 *)tile_ptr + kb * 32;
                                    bool fuse_y = (last_k && mr == 14 && width == 32);
                                    fp_epilogue ep = {
                                        .acc = acc_tile,
                                        .y = (uint8_t *)args->y +
                                             ((rb + r_sub) * args->y_stride + col0) * y_elem_b,
                                        .bias = bias ? (bias + col0) : nullptr,
                                        .y_stride_b = y_stride_b,
                                        .first_k = first_k,
                                        .y_mode = fuse_y ? full_y_mode : 0,
                                    };
                                    ukernel_14x32_f16w_avx512(kc, a_pack, b_panel, &ep);
                                    if (last_k && !fuse_y) {
                                        for (size_t r = 0; r < mr; ++r) {
                                            store_y_fp_avx512(y_dt, args->y,
                                                              (rb + r_sub + r) * args->y_stride +
                                                                  col0,
                                                              acc_tile + r * 32, bias, col0, width);
                                        }
                                    }
                                } else {
                                    fp_epilogue ep = {
                                        .acc = acc_tile, .first_k = first_k, .y_mode = 0};
                                    ukernel_14x32_f32_avx512(kc, a_pack, b_scratch, &ep);
                                    if (last_k) {
                                        for (size_t r = 0; r < mr; ++r) {
                                            store_y_fp_avx512(y_dt, args->y,
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
        }
        return;
    }

    /* Panel offsets and bounds come from the workspace declarations above. */
    const size_t MC_L3 = FP_R_BAND;
    const size_t MC_RM = FP_R_MC;
    const size_t NC_RM = FP_R_TILES * FP_NR;
    const size_t KC_RM = FP_R_KC;
    float *acc_base = (float *)sc;
    float *a_pack = (float *)(sc + FP_R_A);
    void *b_scratch = (void *)(sc + FP_R_B);
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
                        if (w_dt == VELRI_DTYPE_F16) {
                            pack_b_32_f16_raw_avx512(width, kc, (const velri_fp16 *)args->weights,
                                                     args->w_stride, col0, kb,
                                                     (velri_fp16 *)b_scratch + ti * 32 * kc);
                        } else {
                            pack_b_32_fp_avx512(width, kc, VELRI_DTYPE_F32, args->weights,
                                                args->w_stride, col0, kb, false, 32,
                                                (float *)b_scratch + ti * 32 * kc);
                        }
                    }

                    for (size_t r_sub = 0, mr_idx = 0; r_sub < mb; r_sub += 14, ++mr_idx) {
                        size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                        size_t r_row = rb + r_sub;
                        pack_a_14_fp_avx512(mr, kc, plan->desc.x_dtype, args->x, args->x_stride,
                                            r_row, kb, a_pack);

                        for (size_t ti = 0; ti < n_tiles; ++ti) {
                            size_t col0 = cb + ti * 32;
                            size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                            float *acc_tile =
                                (first_k && last_k)
                                    ? local_acc
                                    : (acc_base + (ti * (FP_R_MC / FP_MR) + mr_idx) * 448);
                            bool fuse_y = (last_k && mr == 14 && width == 32);
                            fp_epilogue ep = {
                                .acc = acc_tile,
                                .y =
                                    (uint8_t *)args->y + (r_row * args->y_stride + col0) * y_elem_b,
                                .bias = bias ? (bias + col0) : nullptr,
                                .y_stride_b = y_stride_b,
                                .first_k = first_k,
                                .y_mode = fuse_y ? full_y_mode : 0,
                            };
                            if (w_dt == VELRI_DTYPE_F16) {
                                ukernel_14x32_f16w_avx512(
                                    kc, a_pack, (const velri_fp16 *)b_scratch + ti * 32 * kc, &ep);
                            } else {
                                ukernel_14x32_f32_avx512(
                                    kc, a_pack, (const float *)b_scratch + ti * 32 * kc, &ep);
                            }
                            if (last_k && !fuse_y) {
                                for (size_t r = 0; r < mr; ++r) {
                                    store_y_fp_avx512(y_dt, args->y,
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
 * 4. Native AVX-512 FP64 (`14x16` ZMM Inline-ASM Microkernel, 4x K-Unrolled)
 * ============================================================================ */
static inline void transpose_8x8_pd(__m512d r[8]) {
    __m512d t[8];
    for (int i = 0; i < 4; ++i) {
        t[2 * i + 0] = _mm512_unpacklo_pd(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_pd(r[2 * i], r[2 * i + 1]);
    }
    for (int s = 0; s < 2; ++s) {
        __m512d v0 = _mm512_shuffle_f64x2(t[0 + s], t[2 + s], 0x44);
        __m512d v1 = _mm512_shuffle_f64x2(t[4 + s], t[6 + s], 0x44);
        __m512d v2 = _mm512_shuffle_f64x2(t[0 + s], t[2 + s], 0xEE);
        __m512d v3 = _mm512_shuffle_f64x2(t[4 + s], t[6 + s], 0xEE);
        r[0 + s] = _mm512_shuffle_f64x2(v0, v1, 0x88);
        r[2 + s] = _mm512_shuffle_f64x2(v0, v1, 0xDD);
        r[4 + s] = _mm512_shuffle_f64x2(v2, v3, 0x88);
        r[6 + s] = _mm512_shuffle_f64x2(v2, v3, 0xDD);
    }
}

/* Pack 14 rows of FP64 X into 128B-aligned `stride = 16` doubles per K-step */
static void pack_a_14_f64_avx512(size_t mr, size_t kc, const double *x, size_t xs, size_t r_base,
                                 size_t k_base, double *restrict dst) {
    size_t p = 0;
    if (mr == 14) {
        for (; p + 8 <= kc; p += 8) {
            __m512d lo[8], hi[8];
            for (size_t r = 0; r < 8; ++r) {
                lo[r] = _mm512_loadu_pd(x + (r_base + r) * xs + k_base + p);
            }
            for (size_t r = 0; r < 6; ++r) {
                hi[r] = _mm512_loadu_pd(x + (r_base + 8 + r) * xs + k_base + p);
            }
            hi[6] = _mm512_setzero_pd();
            hi[7] = _mm512_setzero_pd();
            transpose_8x8_pd(lo);
            transpose_8x8_pd(hi);
            for (size_t q = 0; q < 8; ++q) {
                _mm512_storeu_pd(dst + (p + q) * 16, lo[q]);
                _mm512_storeu_pd(dst + (p + q) * 16 + 8, hi[q]);
            }
        }
    }
    for (; p < kc; ++p) {
        for (size_t r = 0; r < mr; ++r) {
            dst[p * 16 + r] = x[(r_base + r) * xs + k_base + p];
        }
        for (size_t r = mr; r < 16; ++r) {
            dst[p * 16 + r] = 0.0;
        }
    }
}

static void pack_b_16_f64_raw_avx512(size_t nr, size_t kc, const double *w, size_t ws,
                                     size_t c_base, size_t k_base, double *restrict bp) {
    size_t p = 0;
    if (nr == 16) {
        for (; p + 8 <= kc; p += 8) {
            __m512d lo[8];
            for (size_t j = 0; j < 8; ++j) {
                lo[j] = _mm512_loadu_pd(w + (c_base + j) * ws + k_base + p);
            }
            transpose_8x8_pd(lo);
            for (size_t q = 0; q < 8; ++q) {
                _mm512_storeu_pd(bp + (p + q) * 16, lo[q]);
            }

            __m512d hi[8];
            for (size_t j = 0; j < 8; ++j) {
                hi[j] = _mm512_loadu_pd(w + (c_base + 8 + j) * ws + k_base + p);
            }
            transpose_8x8_pd(hi);
            for (size_t q = 0; q < 8; ++q) {
                _mm512_storeu_pd(bp + (p + q) * 16 + 8, hi[q]);
            }
        }
    }
    for (; p < kc; ++p) {
        for (size_t j = 0; j < nr; ++j) {
            bp[p * 16 + j] = w[(c_base + j) * ws + k_base + p];
        }
        for (size_t j = nr; j < 16; ++j) {
            bp[p * 16 + j] = 0.0;
        }
    }
}

static inline void ukernel_14x16_f64_avx512(size_t kc, const double *a, const double *b,
                                            const fp_epilogue *ep) {
    if (kc == 0) {
        return;
    }
    size_t kc4 = kc >> 2;
    size_t kc1 = kc & 3ULL;
    void *ytmp = nullptr;

    /* clang-format off */
    __asm__ volatile(
        "cmpl $0, 32(%[ep])\n\t"
        "je 1f\n\t"
        "vpxord %%zmm0,  %%zmm0,  %%zmm0\n\tvpxord %%zmm1,  %%zmm1,  %%zmm1\n\t"
        "vpxord %%zmm2,  %%zmm2,  %%zmm2\n\tvpxord %%zmm3,  %%zmm3,  %%zmm3\n\t"
        "vpxord %%zmm4,  %%zmm4,  %%zmm4\n\tvpxord %%zmm5,  %%zmm5,  %%zmm5\n\t"
        "vpxord %%zmm6,  %%zmm6,  %%zmm6\n\tvpxord %%zmm7,  %%zmm7,  %%zmm7\n\t"
        "vpxord %%zmm8,  %%zmm8,  %%zmm8\n\tvpxord %%zmm9,  %%zmm9,  %%zmm9\n\t"
        "vpxord %%zmm10, %%zmm10, %%zmm10\n\tvpxord %%zmm11, %%zmm11, %%zmm11\n\t"
        "vpxord %%zmm12, %%zmm12, %%zmm12\n\tvpxord %%zmm13, %%zmm13, %%zmm13\n\t"
        "vpxord %%zmm14, %%zmm14, %%zmm14\n\tvpxord %%zmm15, %%zmm15, %%zmm15\n\t"
        "vpxord %%zmm16, %%zmm16, %%zmm16\n\tvpxord %%zmm17, %%zmm17, %%zmm17\n\t"
        "vpxord %%zmm18, %%zmm18, %%zmm18\n\tvpxord %%zmm19, %%zmm19, %%zmm19\n\t"
        "vpxord %%zmm20, %%zmm20, %%zmm20\n\tvpxord %%zmm21, %%zmm21, %%zmm21\n\t"
        "vpxord %%zmm22, %%zmm22, %%zmm22\n\tvpxord %%zmm23, %%zmm23, %%zmm23\n\t"
        "vpxord %%zmm24, %%zmm24, %%zmm24\n\tvpxord %%zmm25, %%zmm25, %%zmm25\n\t"
        "vpxord %%zmm26, %%zmm26, %%zmm26\n\tvpxord %%zmm27, %%zmm27, %%zmm27\n\t"
        "jmp 2f\n\t"
        "1:\n\t"
        "movq 0(%[ep]), %[ytmp]\n\t"
        "vmovapd 0(%[ytmp]),    %%zmm0\n\tvmovapd 64(%[ytmp]),   %%zmm1\n\t"
        "vmovapd 128(%[ytmp]),  %%zmm2\n\tvmovapd 192(%[ytmp]),  %%zmm3\n\t"
        "vmovapd 256(%[ytmp]),  %%zmm4\n\tvmovapd 320(%[ytmp]),  %%zmm5\n\t"
        "vmovapd 384(%[ytmp]),  %%zmm6\n\tvmovapd 448(%[ytmp]),  %%zmm7\n\t"
        "vmovapd 512(%[ytmp]),  %%zmm8\n\tvmovapd 576(%[ytmp]),  %%zmm9\n\t"
        "vmovapd 640(%[ytmp]),  %%zmm10\n\tvmovapd 704(%[ytmp]),  %%zmm11\n\t"
        "vmovapd 768(%[ytmp]),  %%zmm12\n\tvmovapd 832(%[ytmp]),  %%zmm13\n\t"
        "vmovapd 896(%[ytmp]),  %%zmm14\n\tvmovapd 960(%[ytmp]),  %%zmm15\n\t"
        "vmovapd 1024(%[ytmp]), %%zmm16\n\tvmovapd 1088(%[ytmp]), %%zmm17\n\t"
        "vmovapd 1152(%[ytmp]), %%zmm18\n\tvmovapd 1216(%[ytmp]), %%zmm19\n\t"
        "vmovapd 1280(%[ytmp]), %%zmm20\n\tvmovapd 1344(%[ytmp]), %%zmm21\n\t"
        "vmovapd 1408(%[ytmp]), %%zmm22\n\tvmovapd 1472(%[ytmp]), %%zmm23\n\t"
        "vmovapd 1536(%[ytmp]), %%zmm24\n\tvmovapd 1600(%[ytmp]), %%zmm25\n\t"
        "vmovapd 1664(%[ytmp]), %%zmm26\n\tvmovapd 1728(%[ytmp]), %%zmm27\n\t"
        "2:\n\t"
        "test %[kc4], %[kc4]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        STEP_14X16_F64(0, 0)
        STEP_14X16_F64(128, 128)
        STEP_14X16_F64(256, 256)
        STEP_14X16_F64(384, 384)
        "add $512, %[a]\n\t"
        "add $512, %[b]\n\t"
        "dec %[kc4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kc1], %[kc1]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        STEP_14X16_F64(0, 0)
        "add $128, %[a]\n\t"
        "add $128, %[b]\n\t"
        "dec %[kc1]\n\t"
        "jnz 5b\n\t"
        #undef STEP_14X16_F64
        "6:\n\t"
        "cmpl $0, 36(%[ep])\n\t"
        "jne 7f\n\t"
        "movq 0(%[ep]), %[ytmp]\n\t"
        "vmovapd %%zmm0,  0(%[ytmp])\n\tvmovapd %%zmm1,  64(%[ytmp])\n\t"
        "vmovapd %%zmm2,  128(%[ytmp])\n\tvmovapd %%zmm3,  192(%[ytmp])\n\t"
        "vmovapd %%zmm4,  256(%[ytmp])\n\tvmovapd %%zmm5,  320(%[ytmp])\n\t"
        "vmovapd %%zmm6,  384(%[ytmp])\n\tvmovapd %%zmm7,  448(%[ytmp])\n\t"
        "vmovapd %%zmm8,  512(%[ytmp])\n\tvmovapd %%zmm9,  576(%[ytmp])\n\t"
        "vmovapd %%zmm10, 640(%[ytmp])\n\tvmovapd %%zmm11, 704(%[ytmp])\n\t"
        "vmovapd %%zmm12, 768(%[ytmp])\n\tvmovapd %%zmm13, 832(%[ytmp])\n\t"
        "vmovapd %%zmm14, 896(%[ytmp])\n\tvmovapd %%zmm15, 960(%[ytmp])\n\t"
        "vmovapd %%zmm16, 1024(%[ytmp])\n\tvmovapd %%zmm17, 1088(%[ytmp])\n\t"
        "vmovapd %%zmm18, 1152(%[ytmp])\n\tvmovapd %%zmm19, 1216(%[ytmp])\n\t"
        "vmovapd %%zmm20, 1280(%[ytmp])\n\tvmovapd %%zmm21, 1344(%[ytmp])\n\t"
        "vmovapd %%zmm22, 1408(%[ytmp])\n\tvmovapd %%zmm23, 1472(%[ytmp])\n\t"
        "vmovapd %%zmm24, 1536(%[ytmp])\n\tvmovapd %%zmm25, 1600(%[ytmp])\n\t"
        "vmovapd %%zmm26, 1664(%[ytmp])\n\tvmovapd %%zmm27, 1728(%[ytmp])\n\t"
        "jmp 9f\n\t"
        "7:\n\t"
        "movq 16(%[ep]), %[ytmp]\n\t"
        "test %[ytmp], %[ytmp]\n\t"
        "jz 8f\n\t"
        "vmovupd 0(%[ytmp]),  %%zmm28\n\tvmovupd 64(%[ytmp]), %%zmm29\n\t"
        "vaddpd %%zmm28, %%zmm0,  %%zmm0\n\tvaddpd %%zmm29, %%zmm1,  %%zmm1\n\t"
        "vaddpd %%zmm28, %%zmm2,  %%zmm2\n\tvaddpd %%zmm29, %%zmm3,  %%zmm3\n\t"
        "vaddpd %%zmm28, %%zmm4,  %%zmm4\n\tvaddpd %%zmm29, %%zmm5,  %%zmm5\n\t"
        "vaddpd %%zmm28, %%zmm6,  %%zmm6\n\tvaddpd %%zmm29, %%zmm7,  %%zmm7\n\t"
        "vaddpd %%zmm28, %%zmm8,  %%zmm8\n\tvaddpd %%zmm29, %%zmm9,  %%zmm9\n\t"
        "vaddpd %%zmm28, %%zmm10, %%zmm10\n\tvaddpd %%zmm29, %%zmm11, %%zmm11\n\t"
        "vaddpd %%zmm28, %%zmm12, %%zmm12\n\tvaddpd %%zmm29, %%zmm13, %%zmm13\n\t"
        "vaddpd %%zmm28, %%zmm14, %%zmm14\n\tvaddpd %%zmm29, %%zmm15, %%zmm15\n\t"
        "vaddpd %%zmm28, %%zmm16, %%zmm16\n\tvaddpd %%zmm29, %%zmm17, %%zmm17\n\t"
        "vaddpd %%zmm28, %%zmm18, %%zmm18\n\tvaddpd %%zmm29, %%zmm19, %%zmm19\n\t"
        "vaddpd %%zmm28, %%zmm20, %%zmm20\n\tvaddpd %%zmm29, %%zmm21, %%zmm21\n\t"
        "vaddpd %%zmm28, %%zmm22, %%zmm22\n\tvaddpd %%zmm29, %%zmm23, %%zmm23\n\t"
        "vaddpd %%zmm28, %%zmm24, %%zmm24\n\tvaddpd %%zmm29, %%zmm25, %%zmm25\n\t"
        "vaddpd %%zmm28, %%zmm26, %%zmm26\n\tvaddpd %%zmm29, %%zmm27, %%zmm27\n\t"
        "8:\n\t"
        "movq 8(%[ep]),  %[ytmp]\n\t"
        "movq 24(%[ep]), %[kc4]\n\t"
        "vmovupd %%zmm0,  0(%[ytmp])\n\tvmovupd %%zmm1,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm2,  0(%[ytmp])\n\tvmovupd %%zmm3,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm4,  0(%[ytmp])\n\tvmovupd %%zmm5,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm6,  0(%[ytmp])\n\tvmovupd %%zmm7,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm8,  0(%[ytmp])\n\tvmovupd %%zmm9,  64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm10, 0(%[ytmp])\n\tvmovupd %%zmm11, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm12, 0(%[ytmp])\n\tvmovupd %%zmm13, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm14, 0(%[ytmp])\n\tvmovupd %%zmm15, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm16, 0(%[ytmp])\n\tvmovupd %%zmm17, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm18, 0(%[ytmp])\n\tvmovupd %%zmm19, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm20, 0(%[ytmp])\n\tvmovupd %%zmm21, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm22, 0(%[ytmp])\n\tvmovupd %%zmm23, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm24, 0(%[ytmp])\n\tvmovupd %%zmm25, 64(%[ytmp])\n\tadd %[kc4], %[ytmp]\n\t"
        "vmovupd %%zmm26, 0(%[ytmp])\n\tvmovupd %%zmm27, 64(%[ytmp])\n\t"
        "9:\n\t"
        : [a] "+&r"(a), [b] "+&r"(b), [kc4] "+&r"(kc4), [kc1] "+&r"(kc1), [ytmp] "=&r"(ytmp)
        : [ep] "r"(ep)
        : "cc", "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
          "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18",
          "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
          "zmm29", "zmm30", "zmm31");
    /* clang-format on */
}

static inline void store_y_f64_row(double *yd, const double *vals, const double *bias, size_t col0,
                                   size_t width) {
    if (width == 16) {
        __m512d v0 = _mm512_load_pd(vals);
        __m512d v1 = _mm512_load_pd(vals + 8);
        if (bias) {
            v0 = _mm512_add_pd(v0, _mm512_loadu_pd(bias + col0));
            v1 = _mm512_add_pd(v1, _mm512_loadu_pd(bias + col0 + 8));
        }
        _mm512_storeu_pd(yd, v0);
        _mm512_storeu_pd(yd + 8, v1);
    } else {
        for (size_t j = 0; j < width; ++j) {
            yd[j] = vals[j] + (bias ? bias[col0 + j] : 0.0);
        }
    }
}

static void avx512_f64_run(const velri_linear_typed_plan *plan, const velri_linear_run_args *args,
                           size_t r0, size_t r1, size_t c0, size_t c1, void *restrict scratch,
                           bool from_packed) {
    const size_t k = plan->k;
    const double *x = (const double *)args->x;
    const double *bias = (const double *)args->bias;
    double *restrict y = (double *)args->y;
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const size_t y_stride_b = args->y_stride * sizeof(double);

    if (r1 - r0 <= 2) {
        if (from_packed)
            velri_gemv_f64_prepared_avx512(plan, args, r0, r1, c0, c1);
        else
            velri_gemv_f64_rowmajor_avx512(plan, args, r0, r1, c0, c1);
        return;
    }

    uint8_t *sc = align64_ptr(scratch);

    if (from_packed) {
        /* Panel offsets and bounds come from the workspace declarations above. */
        const size_t MC_L3 = F64_P_BAND;
        const size_t MC = F64_P_MC;
        const size_t KC = F64_P_KC;
        const size_t C = plan->columns;
        const size_t max_tiles = (C == 16) ? F64_P_TILES : 1;
        const size_t step_c = max_tiles * C;

        double *acc_base = (double *)sc;
        double *a_pack = (double *)(sc + F64_P_A);
        double *b_scratch = (double *)(sc + F64_P_B);
        alignas(64) double local_acc[224];

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

                        if (C < 16) {
                            const double *wp =
                                (const double *)(pk_base + (cb / C) * plan->tile_stride_b);
                            for (size_t p = 0; p < kc; ++p) {
                                for (size_t j = 0; j < C; ++j) {
                                    b_scratch[p * 16 + j] = wp[(kb + p) * C + j];
                                }
                                for (size_t j = C; j < 16; ++j) {
                                    b_scratch[p * 16 + j] = 0.0;
                                }
                            }
                        }

                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 14;
                            size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                            pack_a_14_f64_avx512(mr, kc, x, args->x_stride, rb + r_sub, kb, a_pack);

                            size_t tile_idx = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                                size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                                const double *b_panel =
                                    (C == 16)
                                        ? ((const double *)(pk_base +
                                                            (col0 / 16) * plan->tile_stride_b) +
                                           kb * 16)
                                        : b_scratch;
                                double *acc_tile =
                                    (first_k && last_k)
                                        ? local_acc
                                        : (acc_base +
                                           (tile_idx * (F64_P_MC / F64_MR) + mr_i) * 224);
                                bool fuse_y = (last_k && mr == 14 && width == 16);
                                fp_epilogue ep = {
                                    .acc = acc_tile,
                                    .y = y + (rb + r_sub) * args->y_stride + col0,
                                    .bias = bias ? (bias + col0) : nullptr,
                                    .y_stride_b = y_stride_b,
                                    .first_k = first_k,
                                    .y_mode = fuse_y ? 1 : 0,
                                };
                                ukernel_14x16_f64_avx512(kc, a_pack, b_panel, &ep);
                                if (last_k && !fuse_y) {
                                    for (size_t r = 0; r < mr; ++r) {
                                        store_y_f64_row(y + (rb + r_sub + r) * args->y_stride +
                                                            col0,
                                                        acc_tile + r * 16, bias, col0, width);
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
    const size_t MC_L3 = F64_R_BAND;
    const size_t MC_RM = F64_R_MC;
    const size_t NC_RM = F64_R_TILES * F64_NR;
    const size_t KC_RM = F64_R_KC;
    double *acc_base = (double *)sc;
    double *a_pack = (double *)(sc + F64_R_A);
    double *b_scratch = (double *)(sc + F64_R_B);
    const double *w = (const double *)args->weights;
    alignas(64) double local_acc_rm[224];

    for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
        size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
        for (size_t cb = c0; cb < c1; cb += NC_RM) {
            size_t cb_end = (c1 - cb < NC_RM) ? c1 : (cb + NC_RM);
            size_t n_tiles = (cb_end - cb + 15) / 16;

            for (size_t rb = r_l3; rb < r_l3_end; rb += MC_RM) {
                size_t mb = (r_l3_end - rb < MC_RM) ? (r_l3_end - rb) : MC_RM;

                for (size_t kb = 0; kb < k; kb += KC_RM) {
                    size_t kc = (k - kb < KC_RM) ? (k - kb) : KC_RM;
                    int32_t first_k = (kb == 0) ? 1 : 0;
                    bool last_k = (kb + kc >= k);

                    for (size_t ti = 0; ti < n_tiles; ++ti) {
                        size_t col0 = cb + ti * 16;
                        size_t width = (cb_end - col0 < 16) ? (cb_end - col0) : 16;
                        pack_b_16_f64_raw_avx512(width, kc, w, args->w_stride, col0, kb,
                                                 b_scratch + ti * 16 * kc);
                    }

                    for (size_t r_sub = 0, mr_idx = 0; r_sub < mb; r_sub += 14, ++mr_idx) {
                        size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                        size_t r_row = rb + r_sub;
                        pack_a_14_f64_avx512(mr, kc, x, args->x_stride, r_row, kb, a_pack);

                        for (size_t ti = 0; ti < n_tiles; ++ti) {
                            size_t col0 = cb + ti * 16;
                            size_t width = (cb_end - col0 < 16) ? (cb_end - col0) : 16;
                            const double *b_panel = b_scratch + ti * 16 * kc;
                            double *acc_tile =
                                (first_k && last_k)
                                    ? local_acc_rm
                                    : (acc_base + (ti * (F64_R_MC / F64_MR) + mr_idx) * 224);
                            bool fuse_y = (last_k && mr == 14 && width == 16);
                            fp_epilogue ep = {
                                .acc = acc_tile,
                                .y = y + r_row * args->y_stride + col0,
                                .bias = bias ? (bias + col0) : nullptr,
                                .y_stride_b = y_stride_b,
                                .first_k = first_k,
                                .y_mode = fuse_y ? 1 : 0,
                            };
                            ukernel_14x16_f64_avx512(kc, a_pack, b_panel, &ep);
                            if (last_k && !fuse_y) {
                                for (size_t r = 0; r < mr; ++r) {
                                    store_y_f64_row(y + (r_row + r) * args->y_stride + col0,
                                                    acc_tile + r * 16, bias, col0, width);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void avx512_fp_pack(const velri_linear_typed_plan *p, const void *w, size_t ws,
                           const float *wsc, void *restrict pk) {
    const size_t C = p->columns;
    if (C == 32 && p->desc.w_dtype == VELRI_DTYPE_F32 && !p->desc.has_w_scale) {
        uint8_t *dst_base = (uint8_t *)pk;
        for (size_t col0 = 0, t = 0; col0 < p->m; col0 += 32, ++t) {
            uint8_t *tile_dst = dst_base + t * p->tile_stride_b;
            memset(tile_dst, 0, p->tile_stride_b);
            size_t nr = (p->m - col0 < 32) ? (p->m - col0) : 32;
            pack_b_32_fp_avx512(nr, p->k, VELRI_DTYPE_F32, w, ws, col0, 0, false, 32,
                                (float *)tile_dst);
        }
        return;
    }
    if (C == 32 && p->desc.w_dtype == VELRI_DTYPE_F16 && !p->desc.has_w_scale) {
        uint8_t *dst_base = (uint8_t *)pk;
        for (size_t col0 = 0, t = 0; col0 < p->m; col0 += 32, ++t) {
            uint8_t *tile_dst = dst_base + t * p->tile_stride_b;
            memset(tile_dst, 0, p->tile_stride_b);
            size_t nr = (p->m - col0 < 32) ? (p->m - col0) : 32;
            pack_b_32_f16_raw_avx512(nr, p->k, (const velri_fp16 *)w, ws, col0, 0,
                                     (velri_fp16 *)tile_dst);
        }
        return;
    }
    if (C == 16 && p->desc.w_dtype == VELRI_DTYPE_F64 && !p->desc.has_w_scale) {
        uint8_t *dst_base = (uint8_t *)pk;
        for (size_t col0 = 0, t = 0; col0 < p->m; col0 += 16, ++t) {
            uint8_t *tile_dst = dst_base + t * p->tile_stride_b;
            memset(tile_dst, 0, p->tile_stride_b);
            size_t nr = (p->m - col0 < 16) ? (p->m - col0) : 16;
            pack_b_16_f64_raw_avx512(nr, p->k, (const double *)w, ws, col0, 0, (double *)tile_dst);
        }
        return;
    }
    velri_pack_generic_tiles(p, 1, w, ws, wsc, pk);
}

static void avx512_fp_prep_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                 size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_fp_gemm(p, a, r0, r1, c0, c1, sc, true);
}
static void avx512_fp_rm_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                               size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_fp_gemm(p, a, r0, r1, c0, c1, sc, false);
}
static void avx512_f64_prep_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                  size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_f64_run(p, a, r0, r1, c0, c1, sc, true);
}
static void avx512_f64_rm_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_f64_run(p, a, r0, r1, c0, c1, sc, false);
}

static const velri_lin_impl g_avx512_fp_impls[] = {
    [VELRI_LIN_F32] = {VELRI_LIN_F32, VELRI_CPU_AVX512, 32, 1, &fp_p, &fp_r, avx512_fp_pack,
                       avx512_fp_prep_entry, avx512_fp_rm_entry},
    [VELRI_LIN_F64] = {VELRI_LIN_F64, VELRI_CPU_AVX512, 16, 1, &f64_p, &f64_r, avx512_fp_pack,
                       avx512_f64_prep_entry, avx512_f64_rm_entry},
    [VELRI_LIN_F16_F32] = {VELRI_LIN_F16_F32, VELRI_CPU_AVX512, 32, 1, &fp_p, &fp_r, avx512_fp_pack,
                           avx512_fp_prep_entry, avx512_fp_rm_entry},
    [VELRI_LIN_F16_F16] = {VELRI_LIN_F16_F16, VELRI_CPU_AVX512, 32, 1, &fp_p, &fp_r, avx512_fp_pack,
                           avx512_fp_prep_entry, avx512_fp_rm_entry},
    [VELRI_LIN_F32X_F16W] = {VELRI_LIN_F32X_F16W, VELRI_CPU_AVX512, 32, 1, &fp_p, &fp_r,
                             avx512_fp_pack, avx512_fp_prep_entry, avx512_fp_rm_entry},
};

const velri_lin_impl *velri_lin_get_avx512_fp(velri_lin_kind kind) {
    if ((size_t)kind > VELRI_LIN_F32X_F16W) {
        return nullptr;
    }
    return &g_avx512_fp_impls[kind];
}
