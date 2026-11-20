#include "internal.h"
#include <immintrin.h>
#include "linear_workspace.h"

/*
 * AVX-512 VNNI INT8 Linear Backend (`vpdpbusd`, plus Weight-Only `F32 * I8`):
 *   - Native `AVX512_VNNI` (`vpdpbusd`): 64 INT8 MACs = 128 INT8 OPs/insn.
 *   - 0-ULP Bit-Exact Signed x Signed Identity (including `-128 * -128`):
 *       u_i = (uint8_t)x_i ^ 0x80 = x_i + 128   for all x_i in [-128, 127]
 *       sum_i x_i * w_i = sum_i u_i * w_i - 128 * sum_i w_i
 *   - 14x32 inline-asm microkernels (`vpdpbusd` for W8A8/I8_I32, `vpmovsxbd + vcvtdq2ps` for
 * F32*I8), 64B-aligned `stride = 16` A-panels, conflict-free 8-row grouped SIMD B-packers.
 */

/* MR/NR match the fixed SIMD microkernels. MC, KC and TILES describe
 * cache blocking; workspace offsets and scheduler hints derive from them.
 * A panels pad the microkernel's row count to a vector-friendly stride. */
enum { I8_MR = 14, I8_NR = 32, I8_A_ROWS = 16 };
enum { I8_P_MC = 126, I8_P_KC = 1024, I8_P_TILES = 16, I8_P_BAND = 1008 };
static_assert(I8_P_MC % I8_MR == 0);
VELRI_LINEAR_WORKSPACE(I8_P, I8_P_MC * I8_P_TILES * I8_NR * sizeof(int32_t),
                       I8_A_ROWS * I8_P_KC * sizeof(uint8_t), I8_NR * I8_P_KC * sizeof(int8_t), 0);
static const velri_lin_blocking i8_p = {I8_P_BYTES, I8_MR, I8_P_MC, I8_P_TILES, 4 * I8_MR, 8};
enum { I8_R_MC = 252, I8_R_KC = 512, I8_R_TILES = 8, I8_R_BAND = 1008 };
static_assert(I8_R_MC % I8_MR == 0);
/* Preserve panel spacing: compact A/B regions slowed parallel row-major
 * INT8 GEMM in the before/after comparison. The unused spans are deliberate. */
enum { I8_R_A_SPANS = 2, I8_R_B_SPANS = 2 };
VELRI_LINEAR_WORKSPACE(I8_R, I8_R_MC * I8_R_TILES * I8_NR * sizeof(int32_t),
                       I8_R_A_SPANS * I8_A_ROWS * I8_R_KC * sizeof(uint8_t),
                       I8_R_B_SPANS * I8_R_TILES * I8_NR * I8_R_KC * sizeof(int8_t),
                       I8_R_TILES * I8_NR * sizeof(int32_t));
static const velri_lin_blocking i8_r = {I8_R_BYTES, I8_MR, I8_R_MC, I8_R_TILES, 9 * I8_MR, 4};
enum { F32_I8_MR = 14, F32_I8_NR = 32, F32_I8_A_ROWS = 16 };
enum { F32_I8_P_MC = 126, F32_I8_P_KC = 256, F32_I8_P_TILES = 16, F32_I8_P_BAND = 1008 };
static_assert(F32_I8_P_MC % F32_I8_MR == 0);
VELRI_LINEAR_WORKSPACE(F32_I8_P, F32_I8_P_MC * F32_I8_P_TILES * F32_I8_NR * sizeof(float),
                       F32_I8_A_ROWS * F32_I8_P_KC * sizeof(float),
                       F32_I8_NR * F32_I8_P_KC * sizeof(int8_t), 0);
static const velri_lin_blocking f32_i8_p = {F32_I8_P_BYTES, F32_I8_MR,     F32_I8_P_MC,
                                            F32_I8_P_TILES, 4 * F32_I8_MR, 8};
enum { F32_I8_R_MC = 252, F32_I8_R_KC = 256, F32_I8_R_TILES = 8, F32_I8_R_BAND = 1008 };
static_assert(F32_I8_R_MC % F32_I8_MR == 0);
VELRI_LINEAR_WORKSPACE(F32_I8_R, F32_I8_R_MC * F32_I8_R_TILES * F32_I8_NR * sizeof(float),
                       F32_I8_A_ROWS * F32_I8_R_KC * sizeof(float),
                       F32_I8_R_TILES * F32_I8_NR * F32_I8_R_KC * sizeof(int8_t), 0);
static const velri_lin_blocking f32_i8_r = {F32_I8_R_BYTES, F32_I8_MR,     F32_I8_R_MC,
                                            F32_I8_R_TILES, 9 * F32_I8_MR, 4};

static_assert(I8_P_KC % 4 == 0);
static_assert(I8_R_KC % 4 == 0);

static inline uint8_t *align64_ptr(void *p) {
    uintptr_t u = ((uintptr_t)p + 63ULL) & ~63ULL;
    return (uint8_t *)u;
}

typedef struct {
    int32_t *acc;        /* offset 0  */
    const int32_t *corr; /* offset 8  */
    int32_t *y_i32;      /* offset 16 */
    const int32_t *bias; /* offset 24 */
    size_t y_stride_b;   /* offset 32 */
    int32_t first_k;     /* offset 40 */
    int32_t fuse_i32;    /* offset 44: 0 = store to acc, 1 = add bias & store I32 to y_i32 */
} i8_epilogue;

static_assert(offsetof(i8_epilogue, acc) == 0);
static_assert(offsetof(i8_epilogue, corr) == 8);
static_assert(offsetof(i8_epilogue, y_i32) == 16);
static_assert(offsetof(i8_epilogue, bias) == 24);
static_assert(offsetof(i8_epilogue, y_stride_b) == 32);
static_assert(offsetof(i8_epilogue, first_k) == 40);
static_assert(offsetof(i8_epilogue, fuse_i32) == 44);

static inline void transpose_16x16_u32_zmm_i32(__m512i r[16]) {
    __m512 f[16], t[16], u[16], v[16];
    for (int i = 0; i < 16; ++i) {
        f[i] = _mm512_castsi512_ps(r[i]);
    }
    for (int i = 0; i < 8; ++i) {
        t[2 * i + 0] = _mm512_unpacklo_ps(f[2 * i], f[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(f[2 * i], f[2 * i + 1]);
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
        r[j + 0] = _mm512_castps_si512(_mm512_shuffle_f32x4(v[j], v[j + 4], 0x88));
        r[j + 4] = _mm512_castps_si512(_mm512_shuffle_f32x4(v[j], v[j + 4], 0xDD));
        r[j + 8] = _mm512_castps_si512(_mm512_shuffle_f32x4(v[j + 8], v[j + 12], 0x88));
        r[j + 12] = _mm512_castps_si512(_mm512_shuffle_f32x4(v[j + 8], v[j + 12], 0xDD));
    }
}

static inline void transpose_16x16_u8_xmm(__m128i r[16]) {
    __m128i tmp[16];
    for (int step = 0; step < 2; ++step) {
        for (int i = 0; i < 8; ++i) {
            tmp[2 * i + 0] = _mm_unpacklo_epi8(r[i], r[i + 8]);
            tmp[2 * i + 1] = _mm_unpackhi_epi8(r[i], r[i + 8]);
        }
        for (int i = 0; i < 8; ++i) {
            r[2 * i + 0] = _mm_unpacklo_epi8(tmp[i], tmp[i + 8]);
            r[2 * i + 1] = _mm_unpackhi_epi8(tmp[i], tmp[i + 8]);
        }
    }
}

/* ============================================================================
 * 1. Core 14x32 Native `vpdpbusd` Inline-ASM Microkernel (`stride = 16` quads = 64 B)
 * ============================================================================ */
static inline void ukernel_14x32_i8_avx512(size_t kc_quads, const uint8_t *a_u8, const int8_t *b_s8,
                                           const i8_epilogue *ep) {
    if (kc_quads == 0) {
        return;
    }
    static const alignas(64) int32_t zero_corr[32] = {0};
    i8_epilogue local_ep = *ep;
    if (!local_ep.corr) {
        local_ep.corr = zero_corr;
    }

    size_t kq4 = kc_quads >> 2;
    size_t kq1 = kc_quads & 3ULL;
    void *ytmp = nullptr;

    /* clang-format off */
    __asm__ volatile(
        "movq 8(%[ep]), %[ytmp]\n\t"
        "cmpl $0, 40(%[ep])\n\t"
        "je 1f\n\t"
        "vmovdqu32 0(%[ytmp]),  %%zmm0\n\tvmovdqu32 64(%[ytmp]), %%zmm1\n\t"
        "vmovdqa64 %%zmm0, %%zmm2\n\tvmovdqa64 %%zmm1, %%zmm3\n\t"
        "vmovdqa64 %%zmm0, %%zmm4\n\tvmovdqa64 %%zmm1, %%zmm5\n\t"
        "vmovdqa64 %%zmm0, %%zmm6\n\tvmovdqa64 %%zmm1, %%zmm7\n\t"
        "vmovdqa64 %%zmm0, %%zmm8\n\tvmovdqa64 %%zmm1, %%zmm9\n\t"
        "vmovdqa64 %%zmm0, %%zmm10\n\tvmovdqa64 %%zmm1, %%zmm11\n\t"
        "vmovdqa64 %%zmm0, %%zmm12\n\tvmovdqa64 %%zmm1, %%zmm13\n\t"
        "vmovdqa64 %%zmm0, %%zmm14\n\tvmovdqa64 %%zmm1, %%zmm15\n\t"
        "vmovdqa64 %%zmm0, %%zmm16\n\tvmovdqa64 %%zmm1, %%zmm17\n\t"
        "vmovdqa64 %%zmm0, %%zmm18\n\tvmovdqa64 %%zmm1, %%zmm19\n\t"
        "vmovdqa64 %%zmm0, %%zmm20\n\tvmovdqa64 %%zmm1, %%zmm21\n\t"
        "vmovdqa64 %%zmm0, %%zmm22\n\tvmovdqa64 %%zmm1, %%zmm23\n\t"
        "vmovdqa64 %%zmm0, %%zmm24\n\tvmovdqa64 %%zmm1, %%zmm25\n\t"
        "vmovdqa64 %%zmm0, %%zmm26\n\tvmovdqa64 %%zmm1, %%zmm27\n\t"
        "jmp 2f\n\t"
        "1:\n\t"
        "vmovdqu32 0(%[ytmp]),  %%zmm28\n\tvmovdqu32 64(%[ytmp]), %%zmm29\n\t"
        "movq 0(%[ep]), %[ytmp]\n\t"
        "vpaddd 0(%[ytmp]),     %%zmm28, %%zmm0\n\tvpaddd 64(%[ytmp]),    %%zmm29, %%zmm1\n\t"
        "vpaddd 128(%[ytmp]),   %%zmm28, %%zmm2\n\tvpaddd 192(%[ytmp]),   %%zmm29, %%zmm3\n\t"
        "vpaddd 256(%[ytmp]),   %%zmm28, %%zmm4\n\tvpaddd 320(%[ytmp]),   %%zmm29, %%zmm5\n\t"
        "vpaddd 384(%[ytmp]),   %%zmm28, %%zmm6\n\tvpaddd 448(%[ytmp]),   %%zmm29, %%zmm7\n\t"
        "vpaddd 512(%[ytmp]),   %%zmm28, %%zmm8\n\tvpaddd 576(%[ytmp]),   %%zmm29, %%zmm9\n\t"
        "vpaddd 640(%[ytmp]),   %%zmm28, %%zmm10\n\tvpaddd 704(%[ytmp]),   %%zmm29, %%zmm11\n\t"
        "vpaddd 768(%[ytmp]),   %%zmm28, %%zmm12\n\tvpaddd 832(%[ytmp]),   %%zmm29, %%zmm13\n\t"
        "vpaddd 896(%[ytmp]),   %%zmm28, %%zmm14\n\tvpaddd 960(%[ytmp]),   %%zmm29, %%zmm15\n\t"
        "vpaddd 1024(%[ytmp]),  %%zmm28, %%zmm16\n\tvpaddd 1088(%[ytmp]),  %%zmm29, %%zmm17\n\t"
        "vpaddd 1152(%[ytmp]),  %%zmm28, %%zmm18\n\tvpaddd 1216(%[ytmp]),  %%zmm29, %%zmm19\n\t"
        "vpaddd 1280(%[ytmp]),  %%zmm28, %%zmm20\n\tvpaddd 1344(%[ytmp]),  %%zmm29, %%zmm21\n\t"
        "vpaddd 1408(%[ytmp]),  %%zmm28, %%zmm22\n\tvpaddd 1472(%[ytmp]),  %%zmm29, %%zmm23\n\t"
        "vpaddd 1536(%[ytmp]),  %%zmm28, %%zmm24\n\tvpaddd 1600(%[ytmp]),  %%zmm29, %%zmm25\n\t"
        "vpaddd 1664(%[ytmp]),  %%zmm28, %%zmm26\n\tvpaddd 1728(%[ytmp]),  %%zmm29, %%zmm27\n\t"
        "2:\n\t"
        "test %[kq4], %[kq4]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        #define STEP_14X32_I8(A_OFF, B_OFF)                                                        \
            "vmovdqu64 " #B_OFF "(%[b]), %%zmm28\n\t"                                              \
            "vmovdqu64 " #B_OFF "+64(%[b]), %%zmm29\n\t"                                           \
            "vbroadcastss " #A_OFF "+0(%[a]),  %%zmm30\n\tvbroadcastss " #A_OFF "+4(%[a]),  %%zmm31\n\t" \
            "vpdpbusd %%zmm28, %%zmm30, %%zmm0\n\tvpdpbusd %%zmm29, %%zmm30, %%zmm1\n\t"           \
            "vpdpbusd %%zmm28, %%zmm31, %%zmm2\n\tvpdpbusd %%zmm29, %%zmm31, %%zmm3\n\t"           \
            "vbroadcastss " #A_OFF "+8(%[a]),  %%zmm30\n\tvbroadcastss " #A_OFF "+12(%[a]), %%zmm31\n\t" \
            "vpdpbusd %%zmm28, %%zmm30, %%zmm4\n\tvpdpbusd %%zmm29, %%zmm30, %%zmm5\n\t"           \
            "vpdpbusd %%zmm28, %%zmm31, %%zmm6\n\tvpdpbusd %%zmm29, %%zmm31, %%zmm7\n\t"           \
            "vbroadcastss " #A_OFF "+16(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+20(%[a]), %%zmm31\n\t" \
            "vpdpbusd %%zmm28, %%zmm30, %%zmm8\n\tvpdpbusd %%zmm29, %%zmm30, %%zmm9\n\t"           \
            "vpdpbusd %%zmm28, %%zmm31, %%zmm10\n\tvpdpbusd %%zmm29, %%zmm31, %%zmm11\n\t"         \
            "vbroadcastss " #A_OFF "+24(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+28(%[a]), %%zmm31\n\t" \
            "vpdpbusd %%zmm28, %%zmm30, %%zmm12\n\tvpdpbusd %%zmm29, %%zmm30, %%zmm13\n\t"         \
            "vpdpbusd %%zmm28, %%zmm31, %%zmm14\n\tvpdpbusd %%zmm29, %%zmm31, %%zmm15\n\t"         \
            "vbroadcastss " #A_OFF "+32(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+36(%[a]), %%zmm31\n\t" \
            "vpdpbusd %%zmm28, %%zmm30, %%zmm16\n\tvpdpbusd %%zmm29, %%zmm30, %%zmm17\n\t"         \
            "vpdpbusd %%zmm28, %%zmm31, %%zmm18\n\tvpdpbusd %%zmm29, %%zmm31, %%zmm19\n\t"         \
            "vbroadcastss " #A_OFF "+40(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+44(%[a]), %%zmm31\n\t" \
            "vpdpbusd %%zmm28, %%zmm30, %%zmm20\n\tvpdpbusd %%zmm29, %%zmm30, %%zmm21\n\t"         \
            "vpdpbusd %%zmm28, %%zmm31, %%zmm22\n\tvpdpbusd %%zmm29, %%zmm31, %%zmm23\n\t"         \
            "vbroadcastss " #A_OFF "+48(%[a]), %%zmm30\n\tvbroadcastss " #A_OFF "+52(%[a]), %%zmm31\n\t" \
            "vpdpbusd %%zmm28, %%zmm30, %%zmm24\n\tvpdpbusd %%zmm29, %%zmm30, %%zmm25\n\t"         \
            "vpdpbusd %%zmm28, %%zmm31, %%zmm26\n\tvpdpbusd %%zmm29, %%zmm31, %%zmm27\n\t"
        STEP_14X32_I8(0, 0)
        STEP_14X32_I8(64, 128)
        STEP_14X32_I8(128, 256)
        STEP_14X32_I8(192, 384)
        "add $256, %[a]\n\t"
        "add $512, %[b]\n\t"
        "dec %[kq4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kq1], %[kq1]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        STEP_14X32_I8(0, 0)
        "add $64, %[a]\n\t"
        "add $128, %[b]\n\t"
        "dec %[kq1]\n\t"
        "jnz 5b\n\t"
        #undef STEP_14X32_I8
        "6:\n\t"
        "cmpl $0, 44(%[ep])\n\t"
        "jne 7f\n\t"
        "movq 0(%[ep]), %[ytmp]\n\t"
        "vmovdqa64 %%zmm0,  0(%[ytmp])\n\tvmovdqa64 %%zmm1,  64(%[ytmp])\n\t"
        "vmovdqa64 %%zmm2,  128(%[ytmp])\n\tvmovdqa64 %%zmm3,  192(%[ytmp])\n\t"
        "vmovdqa64 %%zmm4,  256(%[ytmp])\n\tvmovdqa64 %%zmm5,  320(%[ytmp])\n\t"
        "vmovdqa64 %%zmm6,  384(%[ytmp])\n\tvmovdqa64 %%zmm7,  448(%[ytmp])\n\t"
        "vmovdqa64 %%zmm8,  512(%[ytmp])\n\tvmovdqa64 %%zmm9,  576(%[ytmp])\n\t"
        "vmovdqa64 %%zmm10, 640(%[ytmp])\n\tvmovdqa64 %%zmm11, 704(%[ytmp])\n\t"
        "vmovdqa64 %%zmm12, 768(%[ytmp])\n\tvmovdqa64 %%zmm13, 832(%[ytmp])\n\t"
        "vmovdqa64 %%zmm14, 896(%[ytmp])\n\tvmovdqa64 %%zmm15, 960(%[ytmp])\n\t"
        "vmovdqa64 %%zmm16, 1024(%[ytmp])\n\tvmovdqa64 %%zmm17, 1088(%[ytmp])\n\t"
        "vmovdqa64 %%zmm18, 1152(%[ytmp])\n\tvmovdqa64 %%zmm19, 1216(%[ytmp])\n\t"
        "vmovdqa64 %%zmm20, 1280(%[ytmp])\n\tvmovdqa64 %%zmm21, 1344(%[ytmp])\n\t"
        "vmovdqa64 %%zmm22, 1408(%[ytmp])\n\tvmovdqa64 %%zmm23, 1472(%[ytmp])\n\t"
        "vmovdqa64 %%zmm24, 1536(%[ytmp])\n\tvmovdqa64 %%zmm25, 1600(%[ytmp])\n\t"
        "vmovdqa64 %%zmm26, 1664(%[ytmp])\n\tvmovdqa64 %%zmm27, 1728(%[ytmp])\n\t"
        "jmp 9f\n\t"
        "7:\n\t"
        "movq 24(%[ep]), %[ytmp]\n\t"
        "test %[ytmp], %[ytmp]\n\t"
        "jz 8f\n\t"
        "vmovdqu32 0(%[ytmp]),  %%zmm28\n\tvmovdqu32 64(%[ytmp]), %%zmm29\n\t"
        "vpaddd %%zmm28, %%zmm0,  %%zmm0\n\tvpaddd %%zmm29, %%zmm1,  %%zmm1\n\t"
        "vpaddd %%zmm28, %%zmm2,  %%zmm2\n\tvpaddd %%zmm29, %%zmm3,  %%zmm3\n\t"
        "vpaddd %%zmm28, %%zmm4,  %%zmm4\n\tvpaddd %%zmm29, %%zmm5,  %%zmm5\n\t"
        "vpaddd %%zmm28, %%zmm6,  %%zmm6\n\tvpaddd %%zmm29, %%zmm7,  %%zmm7\n\t"
        "vpaddd %%zmm28, %%zmm8,  %%zmm8\n\tvpaddd %%zmm29, %%zmm9,  %%zmm9\n\t"
        "vpaddd %%zmm28, %%zmm10, %%zmm10\n\tvpaddd %%zmm29, %%zmm11, %%zmm11\n\t"
        "vpaddd %%zmm28, %%zmm12, %%zmm12\n\tvpaddd %%zmm29, %%zmm13, %%zmm13\n\t"
        "vpaddd %%zmm28, %%zmm14, %%zmm14\n\tvpaddd %%zmm29, %%zmm15, %%zmm15\n\t"
        "vpaddd %%zmm28, %%zmm16, %%zmm16\n\tvpaddd %%zmm29, %%zmm17, %%zmm17\n\t"
        "vpaddd %%zmm28, %%zmm18, %%zmm18\n\tvpaddd %%zmm29, %%zmm19, %%zmm19\n\t"
        "vpaddd %%zmm28, %%zmm20, %%zmm20\n\tvpaddd %%zmm29, %%zmm21, %%zmm21\n\t"
        "vpaddd %%zmm28, %%zmm22, %%zmm22\n\tvpaddd %%zmm29, %%zmm23, %%zmm23\n\t"
        "vpaddd %%zmm28, %%zmm24, %%zmm24\n\tvpaddd %%zmm29, %%zmm25, %%zmm25\n\t"
        "vpaddd %%zmm28, %%zmm26, %%zmm26\n\tvpaddd %%zmm29, %%zmm27, %%zmm27\n\t"
        "8:\n\t"
        "movq 16(%[ep]), %[ytmp]\n\t"
        "movq 32(%[ep]), %[kq4]\n\t"
        "vmovdqu32 %%zmm0,  0(%[ytmp])\n\tvmovdqu32 %%zmm1,  64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm2,  0(%[ytmp])\n\tvmovdqu32 %%zmm3,  64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm4,  0(%[ytmp])\n\tvmovdqu32 %%zmm5,  64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm6,  0(%[ytmp])\n\tvmovdqu32 %%zmm7,  64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm8,  0(%[ytmp])\n\tvmovdqu32 %%zmm9,  64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm10, 0(%[ytmp])\n\tvmovdqu32 %%zmm11, 64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm12, 0(%[ytmp])\n\tvmovdqu32 %%zmm13, 64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm14, 0(%[ytmp])\n\tvmovdqu32 %%zmm15, 64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm16, 0(%[ytmp])\n\tvmovdqu32 %%zmm17, 64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm18, 0(%[ytmp])\n\tvmovdqu32 %%zmm19, 64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm20, 0(%[ytmp])\n\tvmovdqu32 %%zmm21, 64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm22, 0(%[ytmp])\n\tvmovdqu32 %%zmm23, 64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm24, 0(%[ytmp])\n\tvmovdqu32 %%zmm25, 64(%[ytmp])\n\tadd %[kq4], %[ytmp]\n\t"
        "vmovdqu32 %%zmm26, 0(%[ytmp])\n\tvmovdqu32 %%zmm27, 64(%[ytmp])\n\t"
        "9:\n\t"
        : [a] "+&r"(a_u8), [b] "+&r"(b_s8), [kq4] "+&r"(kq4), [kq1] "+&r"(kq1), [ytmp] "=&r"(ytmp)
        : [ep] "r"(&local_ep)
        : "cc", "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
          "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18",
          "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
          "zmm29", "zmm30", "zmm31");
    /* clang-format on */
}

/* 14x32 Weight-Only `F32 * I8 -> F32` Inline-ASM Microkernel (`vpmovsxbd + vcvtdq2ps` in-reg!) */
static inline void ukernel_14x32_i8w_f32_avx512(size_t kc, const float *a, const int8_t *b,
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
        "vmovaps 0(%[acc]),    %%zmm0\n\tvmovaps 64(%[acc]),   %%zmm1\n\t"
        "vmovaps 128(%[acc]),  %%zmm2\n\tvmovaps 192(%[acc]),  %%zmm3\n\t"
        "vmovaps 256(%[acc]),  %%zmm4\n\tvmovaps 320(%[acc]),  %%zmm5\n\t"
        "vmovaps 384(%[acc]),  %%zmm6\n\tvmovaps 448(%[acc]),  %%zmm7\n\t"
        "vmovaps 512(%[acc]),  %%zmm8\n\tvmovaps 576(%[acc]),  %%zmm9\n\t"
        "vmovaps 640(%[acc]),  %%zmm10\n\tvmovaps 704(%[acc]),  %%zmm11\n\t"
        "vmovaps 768(%[acc]),  %%zmm12\n\tvmovaps 832(%[acc]),  %%zmm13\n\t"
        "vmovaps 896(%[acc]),  %%zmm14\n\tvmovaps 960(%[acc]),  %%zmm15\n\t"
        "vmovaps 1024(%[acc]), %%zmm16\n\tvmovaps 1088(%[acc]), %%zmm17\n\t"
        "vmovaps 1152(%[acc]), %%zmm18\n\tvmovaps 1216(%[acc]), %%zmm19\n\t"
        "vmovaps 1280(%[acc]), %%zmm20\n\tvmovaps 1344(%[acc]), %%zmm21\n\t"
        "vmovaps 1408(%[acc]), %%zmm22\n\tvmovaps 1472(%[acc]), %%zmm23\n\t"
        "vmovaps 1536(%[acc]), %%zmm24\n\tvmovaps 1600(%[acc]), %%zmm25\n\t"
        "vmovaps 1664(%[acc]), %%zmm26\n\tvmovaps 1728(%[acc]), %%zmm27\n\t"
        "2:\n\t"
        "test %[kc4], %[kc4]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        #define STEP_I8W_F32(A_OFF, B_OFF)                                                         \
            "vpmovsxbd " #B_OFF "(%[b]), %%zmm28\n\tvcvtdq2ps %%zmm28, %%zmm28\n\t"                 \
            "vpmovsxbd " #B_OFF "+16(%[b]), %%zmm29\n\tvcvtdq2ps %%zmm29, %%zmm29\n\t"              \
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
        STEP_I8W_F32(0, 0)
        STEP_I8W_F32(64, 32)
        STEP_I8W_F32(128, 64)
        STEP_I8W_F32(192, 96)
        "add $256, %[a]\n\t"
        "add $128, %[b]\n\t"
        "dec %[kc4]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[kc1], %[kc1]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        STEP_I8W_F32(0, 0)
        "add $64, %[a]\n\t"
        "add $32, %[b]\n\t"
        "dec %[kc1]\n\t"
        "jnz 5b\n\t"
        #undef STEP_I8W_F32
        "6:\n\t"
        "vmovaps %%zmm0,  0(%[acc])\n\tvmovaps %%zmm1,  64(%[acc])\n\t"
        "vmovaps %%zmm2,  128(%[acc])\n\tvmovaps %%zmm3,  192(%[acc])\n\t"
        "vmovaps %%zmm4,  256(%[acc])\n\tvmovaps %%zmm5,  320(%[acc])\n\t"
        "vmovaps %%zmm6,  384(%[acc])\n\tvmovaps %%zmm7,  448(%[acc])\n\t"
        "vmovaps %%zmm8,  512(%[acc])\n\tvmovaps %%zmm9,  576(%[acc])\n\t"
        "vmovaps %%zmm10, 640(%[acc])\n\tvmovaps %%zmm11, 704(%[acc])\n\t"
        "vmovaps %%zmm12, 768(%[acc])\n\tvmovaps %%zmm13, 832(%[acc])\n\t"
        "vmovaps %%zmm14, 896(%[acc])\n\tvmovaps %%zmm15, 960(%[acc])\n\t"
        "vmovaps %%zmm16, 1024(%[acc])\n\tvmovaps %%zmm17, 1088(%[acc])\n\t"
        "vmovaps %%zmm18, 1152(%[acc])\n\tvmovaps %%zmm19, 1216(%[acc])\n\t"
        "vmovaps %%zmm20, 1280(%[acc])\n\tvmovaps %%zmm21, 1344(%[acc])\n\t"
        "vmovaps %%zmm22, 1408(%[acc])\n\tvmovaps %%zmm23, 1472(%[acc])\n\t"
        "vmovaps %%zmm24, 1536(%[acc])\n\tvmovaps %%zmm25, 1600(%[acc])\n\t"
        "vmovaps %%zmm26, 1664(%[acc])\n\tvmovaps %%zmm27, 1728(%[acc])\n\t"
        : [a] "+&r"(a), [b] "+&r"(b), [kc4] "+&r"(kc4), [kc1] "+&r"(kc1)
        : [acc] "r"(acc), [fk] "r"(first_k)
        : "cc", "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
          "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18",
          "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
          "zmm29", "zmm30", "zmm31");
    /* clang-format on */
}

/* Pack 14 rows of INT8 X (`^ 0x80`) into 64B-aligned `stride = 16` quads (`64 B`/step) */
static void pack_a_14_i8_u8_avx512(size_t mr, size_t kc, const int8_t *x, size_t x_stride,
                                   size_t r_base, size_t k_base, uint8_t *restrict dst) {
    size_t p = 0;
    if (mr == 14) {
        const __m512i v80 = _mm512_set1_epi8((char)0x80);
        for (; p + 64 <= kc; p += 64) {
            __m512i z[16];
            for (size_t r = 0; r < 8; ++r) {
                __m512i v =
                    _mm512_loadu_si512((const void *)(x + (r_base + r) * x_stride + k_base + p));
                z[r] = _mm512_xor_si512(v, v80);
            }
            for (size_t r = 8; r < 14; ++r) {
                __m512i v =
                    _mm512_loadu_si512((const void *)(x + (r_base + r) * x_stride + k_base + p));
                z[r] = _mm512_xor_si512(v, v80);
            }
            z[14] = _mm512_setzero_si512();
            z[15] = _mm512_setzero_si512();
            transpose_16x16_u32_zmm_i32(z);
            uint32_t *dst32 = (uint32_t *)dst + (p / 4) * 16;
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_si512((void *)(dst32 + q * 16), z[q]);
            }
        }
    }
    size_t kc_quads = (kc + 3) / 4;
    for (size_t qq = p / 4; qq < kc_quads; ++qq) {
        size_t base_i = qq * 4;
        for (size_t r = 0; r < mr; ++r) {
            const int8_t *xr = x + (r_base + r) * x_stride + k_base;
            uint8_t *cell = dst + (qq * 16 + r) * 4;
            cell[0] = (base_i + 0 < kc) ? (uint8_t)((uint8_t)xr[base_i + 0] ^ 0x80U) : 0U;
            cell[1] = (base_i + 1 < kc) ? (uint8_t)((uint8_t)xr[base_i + 1] ^ 0x80U) : 0U;
            cell[2] = (base_i + 2 < kc) ? (uint8_t)((uint8_t)xr[base_i + 2] ^ 0x80U) : 0U;
            cell[3] = (base_i + 3 < kc) ? (uint8_t)((uint8_t)xr[base_i + 3] ^ 0x80U) : 0U;
        }
        if (mr < 16) {
            memset(dst + (qq * 16 + mr) * 4, 0, (16 - mr) * 4);
        }
    }
}

static void pack_a_14_f32_for_i8w(size_t mr, size_t kc, const float *x, size_t x_stride,
                                  size_t r_base, size_t k_base, float *restrict dst) {
    size_t p = 0;
    if (mr == 14) {
        for (; p + 16 <= kc; p += 16) {
            __m512i z[16];
            for (size_t r = 0; r < 8; ++r) {
                z[r] = _mm512_loadu_si512((const void *)(x + (r_base + r) * x_stride + k_base + p));
            }
            for (size_t r = 8; r < 14; ++r) {
                z[r] = _mm512_loadu_si512((const void *)(x + (r_base + r) * x_stride + k_base + p));
            }
            z[14] = _mm512_setzero_si512();
            z[15] = _mm512_setzero_si512();
            transpose_16x16_u32_zmm_i32(z);
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_si512((void *)(dst + (p + q) * 16), z[q]);
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

/* Compute `-128 * sum_{p=0}^{kc-1} B[p, col]` across a packed `[kc_quads, 32, 4]` panel */
static inline void compute_panel_corr_32(size_t kc_quads, const int8_t *b_panel,
                                         int32_t *restrict corr_out) {
    const __m512i v128_u8 = _mm512_set1_epi8((char)0x80);
    __m512i s0 = _mm512_setzero_si512();
    __m512i s1 = _mm512_setzero_si512();
    for (size_t qq = 0; qq < kc_quads; ++qq) {
        __m512i b0 = _mm512_loadu_si512((const void *)(b_panel + qq * 128));
        __m512i b1 = _mm512_loadu_si512((const void *)(b_panel + qq * 128 + 64));
        s0 = _mm512_dpbusd_epi32(s0, v128_u8, b0);
        s1 = _mm512_dpbusd_epi32(s1, v128_u8, b1);
    }
    const __m512i zero = _mm512_setzero_si512();
    _mm512_storeu_si512((void *)corr_out, _mm512_sub_epi32(zero, s0));
    _mm512_storeu_si512((void *)(corr_out + 16), _mm512_sub_epi32(zero, s1));
}

static void pack_b_32_i8_avx512(size_t nr, size_t kc, const int8_t *w_src, size_t w_stride,
                                size_t c_base, size_t k_base, bool from_packed, size_t packed_C,
                                int8_t *restrict dst, int32_t *restrict out_panel_corr) {
    size_t kc_quads = (kc + 3) / 4;
    if (from_packed) {
        size_t kq_base = k_base / 4;
        for (size_t qq = 0; qq < kc_quads; ++qq) {
            for (size_t j = 0; j < packed_C; ++j) {
                memcpy(dst + (qq * 32 + j) * 4, w_src + ((kq_base + qq) * packed_C + j) * 4, 4);
            }
            if (packed_C < 32) {
                memset(dst + (qq * 32 + packed_C) * 4, 0, (32 - packed_C) * 4);
            }
        }
        return;
    }

    size_t p = 0;
    if (nr == 32) {
        for (; p + 64 <= kc; p += 64) {
            __m512i lo[16];
            for (size_t j = 0; j < 8; ++j) {
                lo[j] = _mm512_loadu_si512(
                    (const void *)(w_src + (c_base + j) * w_stride + k_base + p));
            }
            for (size_t j = 8; j < 16; ++j) {
                lo[j] = _mm512_loadu_si512(
                    (const void *)(w_src + (c_base + j) * w_stride + k_base + p));
            }
            transpose_16x16_u32_zmm_i32(lo);
            uint32_t *dst32 = (uint32_t *)dst + (p / 4) * 32;
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_si512((void *)(dst32 + q * 32), lo[q]);
            }

            __m512i hi[16];
            for (size_t j = 0; j < 8; ++j) {
                hi[j] = _mm512_loadu_si512(
                    (const void *)(w_src + (c_base + 16 + j) * w_stride + k_base + p));
            }
            for (size_t j = 8; j < 16; ++j) {
                hi[j] = _mm512_loadu_si512(
                    (const void *)(w_src + (c_base + 16 + j) * w_stride + k_base + p));
            }
            transpose_16x16_u32_zmm_i32(hi);
            for (size_t q = 0; q < 16; ++q) {
                _mm512_storeu_si512((void *)(dst32 + q * 32 + 16), hi[q]);
            }
        }
    }
    for (size_t qq = p / 4; qq < kc_quads; ++qq) {
        size_t base_i = qq * 4;
        for (size_t j = 0; j < nr; ++j) {
            const int8_t *wj = w_src + (c_base + j) * w_stride + k_base;
            int8_t *cell = dst + (qq * 32 + j) * 4;
            cell[0] = (base_i + 0 < kc) ? wj[base_i + 0] : 0;
            cell[1] = (base_i + 1 < kc) ? wj[base_i + 1] : 0;
            cell[2] = (base_i + 2 < kc) ? wj[base_i + 2] : 0;
            cell[3] = (base_i + 3 < kc) ? wj[base_i + 3] : 0;
        }
        if (nr < 32) {
            memset(dst + (qq * 32 + nr) * 4, 0, (32 - nr) * 4);
        }
    }
    if (out_panel_corr) {
        compute_panel_corr_32(kc_quads, dst, out_panel_corr);
    }
}

static void pack_b_32_i8_g1_avx512(size_t nr, size_t kc, const int8_t *w_src, size_t w_stride,
                                   size_t c_base, size_t k_base, int8_t *restrict dst) {
    size_t p = 0;
    if (nr == 32) {
        for (; p + 16 <= kc; p += 16) {
            __m128i lo[16];
            for (size_t j = 0; j < 8; ++j) {
                lo[j] = _mm_loadu_si128(
                    (const __m128i *)(w_src + (c_base + j) * w_stride + k_base + p));
            }
            for (size_t j = 8; j < 16; ++j) {
                lo[j] = _mm_loadu_si128(
                    (const __m128i *)(w_src + (c_base + j) * w_stride + k_base + p));
            }
            transpose_16x16_u8_xmm(lo);
            for (size_t q = 0; q < 16; ++q) {
                _mm_storeu_si128((__m128i *)(dst + (p + q) * 32), lo[q]);
            }

            __m128i hi[16];
            for (size_t j = 0; j < 8; ++j) {
                hi[j] = _mm_loadu_si128(
                    (const __m128i *)(w_src + (c_base + 16 + j) * w_stride + k_base + p));
            }
            for (size_t j = 8; j < 16; ++j) {
                hi[j] = _mm_loadu_si128(
                    (const __m128i *)(w_src + (c_base + 16 + j) * w_stride + k_base + p));
            }
            transpose_16x16_u8_xmm(hi);
            for (size_t q = 0; q < 16; ++q) {
                _mm_storeu_si128((__m128i *)(dst + (p + q) * 32 + 16), hi[q]);
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

/* Vectorized packer for prepared INT8 VNNI weights `[ceil(M/32), ceil(K/4), 32, 4] + corr + scale`
 */
static void avx512_i8_pack_g4(const velri_linear_typed_plan *plan, const void *w_void,
                              size_t w_stride, const float *w_scale, void *restrict packed) {
    if (plan->columns != 32 || plan->k < 64 || plan->k > 65536) {
        velri_pack_generic_tiles(plan, 4, w_void, w_stride, w_scale, packed);
        return;
    }
    const int8_t *w = (const int8_t *)w_void;
    uint8_t *dst_base = (uint8_t *)packed;
    const size_t m = plan->m;
    const size_t k = plan->k;
    const size_t payload_b = plan->k_aligned * 32;
    const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
    const size_t scale_off = meta_off + 32 * sizeof(int32_t);

    for (size_t col0 = 0, t = 0; col0 < m; col0 += 32, ++t) {
        uint8_t *tile_dst = dst_base + t * plan->tile_stride_b;
        size_t nr = (m - col0 < 32) ? (m - col0) : 32;
        memset(tile_dst, 0, plan->tile_stride_b);
        int32_t *corr = (int32_t *)(tile_dst + meta_off);
        pack_b_32_i8_avx512(nr, k, w, w_stride, col0, 0, false, 32, (int8_t *)tile_dst, corr);
        if (plan->desc.has_w_scale) {
            float *sc = (float *)(tile_dst + scale_off);
            for (size_t j = 0; j < nr; ++j) {
                sc[j] = w_scale ? w_scale[col0 + j] : 0.0f;
            }
        }
    }
}

static void avx512_i8_pack_g1(const velri_linear_typed_plan *plan, const void *w_void,
                              size_t w_stride, const float *w_scale, void *restrict packed) {
    if (plan->columns != 32 || plan->k < 16 || plan->k > 65536) {
        velri_pack_generic_tiles(plan, 1, w_void, w_stride, w_scale, packed);
        return;
    }
    const int8_t *w = (const int8_t *)w_void;
    uint8_t *dst_base = (uint8_t *)packed;
    const size_t m = plan->m;
    const size_t k = plan->k;
    const size_t payload_b = plan->k_aligned * 32;
    const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
    const size_t scale_off = meta_off + 32 * sizeof(int32_t);
    const size_t k_quads = (k + 3) / 4;

    for (size_t col0 = 0, t = 0; col0 < m; col0 += 32, ++t) {
        uint8_t *tile_dst = dst_base + t * plan->tile_stride_b;
        size_t nr = (m - col0 < 32) ? (m - col0) : 32;
        memset(tile_dst, 0, plan->tile_stride_b);
        pack_b_32_i8_g1_avx512(nr, k, w, w_stride, col0, 0, (int8_t *)tile_dst);
        int32_t *corr = (int32_t *)(tile_dst + meta_off);
        compute_panel_corr_32(k_quads, (const int8_t *)tile_dst, corr);
        if (plan->desc.has_w_scale) {
            float *sc = (float *)(tile_dst + scale_off);
            for (size_t j = 0; j < nr; ++j) {
                sc[j] = w_scale ? w_scale[col0 + j] : 0.0f;
            }
        }
    }
}

static inline void store_y_i32_or_w8a8(velri_dtype y_dt, void *y_base, size_t y_off,
                                       const int32_t *vals, float xs, const float *wsc,
                                       const void *bias, size_t col0, size_t width) {
    if (y_dt == VELRI_DTYPE_I32) {
        int32_t *yd = (int32_t *)y_base + y_off;
        const int32_t *bi = (const int32_t *)bias;
        if (width == 32) {
            __m512i v0 = _mm512_loadu_si512((const void *)vals);
            __m512i v1 = _mm512_loadu_si512((const void *)(vals + 16));
            if (bi) {
                v0 = _mm512_add_epi32(v0, _mm512_loadu_si512((const void *)(bi + col0)));
                v1 = _mm512_add_epi32(v1, _mm512_loadu_si512((const void *)(bi + col0 + 16)));
            }
            _mm512_storeu_si512((void *)yd, v0);
            _mm512_storeu_si512((void *)(yd + 16), v1);
        } else {
            for (size_t j = 0; j < width; ++j) {
                yd[j] = velri_i32_add(vals[j], bi ? bi[col0 + j] : 0);
            }
        }
        return;
    }

    float *yd = (float *)y_base + y_off;
    const float *bf = (const float *)bias;
    if (width == 32) {
        __m512 f0 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const void *)vals));
        __m512 f1 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const void *)(vals + 16)));
        __m512 vxs = _mm512_set1_ps(xs);
        __m512 ws0 = wsc ? _mm512_mul_ps(vxs, _mm512_loadu_ps(wsc)) : vxs;
        __m512 ws1 = wsc ? _mm512_mul_ps(vxs, _mm512_loadu_ps(wsc + 16)) : vxs;
        if (bf) {
            f0 = _mm512_fmadd_ps(f0, ws0, _mm512_loadu_ps(bf + col0));
            f1 = _mm512_fmadd_ps(f1, ws1, _mm512_loadu_ps(bf + col0 + 16));
        } else {
            f0 = _mm512_mul_ps(f0, ws0);
            f1 = _mm512_mul_ps(f1, ws1);
        }
        _mm512_storeu_ps(yd, f0);
        _mm512_storeu_ps(yd + 16, f1);
    } else {
        for (size_t j = 0; j < width; ++j) {
            float s = xs * (wsc ? wsc[j] : 1.0f);
            yd[j] = (float)vals[j] * s + (bf ? bf[col0 + j] : 0.0f);
        }
    }
}

static inline void store_y_w8a8_14x32(float *y_base, size_t y_stride, size_t r0_tile, size_t col0,
                                      const int32_t *acc_tile, const float *x_scale,
                                      const float *wsc, const float *bf) {
    const __m512 vw0 = wsc ? _mm512_loadu_ps(wsc) : _mm512_set1_ps(1.0f);
    const __m512 vw1 = wsc ? _mm512_loadu_ps(wsc + 16) : _mm512_set1_ps(1.0f);
    if (bf) {
        const __m512 vb0 = _mm512_loadu_ps(bf + col0);
        const __m512 vb1 = _mm512_loadu_ps(bf + col0 + 16);
        for (size_t r = 0; r < 14; ++r) {
            __m512 vxs = _mm512_set1_ps(x_scale ? x_scale[r0_tile + r] : 1.0f);
            __m512 ws0 = wsc ? _mm512_mul_ps(vxs, vw0) : vxs;
            __m512 ws1 = wsc ? _mm512_mul_ps(vxs, vw1) : vxs;
            __m512 f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const void *)(acc_tile + r * 32)));
            __m512 f1 =
                _mm512_cvtepi32_ps(_mm512_load_si512((const void *)(acc_tile + r * 32 + 16)));
            float *yd = y_base + (r0_tile + r) * y_stride + col0;
            _mm512_storeu_ps(yd, _mm512_fmadd_ps(f0, ws0, vb0));
            _mm512_storeu_ps(yd + 16, _mm512_fmadd_ps(f1, ws1, vb1));
        }
    } else {
        for (size_t r = 0; r < 14; ++r) {
            __m512 vxs = _mm512_set1_ps(x_scale ? x_scale[r0_tile + r] : 1.0f);
            __m512 ws0 = wsc ? _mm512_mul_ps(vxs, vw0) : vxs;
            __m512 ws1 = wsc ? _mm512_mul_ps(vxs, vw1) : vxs;
            __m512 f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const void *)(acc_tile + r * 32)));
            __m512 f1 =
                _mm512_cvtepi32_ps(_mm512_load_si512((const void *)(acc_tile + r * 32 + 16)));
            float *yd = y_base + (r0_tile + r) * y_stride + col0;
            _mm512_storeu_ps(yd, _mm512_mul_ps(f0, ws0));
            _mm512_storeu_ps(yd + 16, _mm512_mul_ps(f1, ws1));
        }
    }
}

/* ============================================================================
 * 2. Native VNNI INT8 GEMV & Blocked GEMM (`VELRI_LIN_I8_I32` & `VELRI_LIN_W8A8_F32`)
 * ============================================================================ */
static void avx512_i8_run(const velri_linear_typed_plan *plan, const velri_linear_run_args *args,
                          size_t r0, size_t r1, size_t c0, size_t c1, void *restrict scratch,
                          bool from_packed) {
    const size_t k = plan->k;
    const int8_t *x = (const int8_t *)args->x;
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const size_t C = from_packed ? plan->columns : 32;
    const size_t payload_b = from_packed ? (plan->k_aligned * C) : 0;
    const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
    const size_t scale_off = meta_off + C * sizeof(int32_t);
    const velri_dtype y_dt = plan->desc.y_dtype;
    const size_t y_stride_b = args->y_stride * sizeof(int32_t);

    if (r1 - r0 <= 2) {
        if (from_packed) {
            const size_t k_quads = plan->k_aligned / 4;
            for (size_t col0 = c0; col0 < c1; col0 += C) {
                size_t width = (c1 - col0 < C) ? (c1 - col0) : C;
                const uint8_t *tile_ptr = pk_base + (col0 / C) * plan->tile_stride_b;
                const int8_t *wp = (const int8_t *)tile_ptr;
                const int32_t *corr = (const int32_t *)(tile_ptr + meta_off);
                const float *wsc =
                    plan->desc.has_w_scale ? (const float *)(tile_ptr + scale_off) : nullptr;

                for (size_t r = r0; r < r1; ++r) {
                    const int8_t *xr = x + r * args->x_stride;
                    float xs = args->x_scale ? args->x_scale[r] : 1.0f;
                    alignas(64) int32_t out[32] = {0};

                    if (C == 32) {
                        __m512i a0 = _mm512_loadu_si512((const void *)corr);
                        __m512i a1 = _mm512_loadu_si512((const void *)(corr + 16));
                        for (size_t qq = 0; qq < k_quads; ++qq) {
                            size_t bi = qq * 4;
                            uint32_t uq = 0;
                            if (bi + 3 < k) {
                                uint32_t raw;
                                memcpy(&raw, xr + bi, 4);
                                uq = raw ^ UINT32_C(0x80808080);
                            } else {
                                uint8_t b0 =
                                    (bi + 0 < k) ? (uint8_t)((uint8_t)xr[bi + 0] ^ 0x80U) : 0U;
                                uint8_t b1 =
                                    (bi + 1 < k) ? (uint8_t)((uint8_t)xr[bi + 1] ^ 0x80U) : 0U;
                                uint8_t b2 =
                                    (bi + 2 < k) ? (uint8_t)((uint8_t)xr[bi + 2] ^ 0x80U) : 0U;
                                uq = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16);
                            }
                            __m512i vx = _mm512_set1_epi32((int)uq);
                            __m512i wb0 = _mm512_loadu_si512((const void *)(wp + qq * 128));
                            __m512i wb1 = _mm512_loadu_si512((const void *)(wp + qq * 128 + 64));
                            a0 = _mm512_dpbusd_epi32(a0, vx, wb0);
                            a1 = _mm512_dpbusd_epi32(a1, vx, wb1);
                        }
                        _mm512_store_si512((void *)out, a0);
                        _mm512_store_si512((void *)(out + 16), a1);
                    } else {
                        for (size_t j = 0; j < width; ++j) {
                            out[j] = corr[j];
                        }
                        for (size_t qq = 0; qq < k_quads; ++qq) {
                            size_t bi = qq * 4;
                            int32_t u0 = (bi + 0 < k) ? ((uint8_t)xr[bi + 0] ^ 0x80U) : 0;
                            int32_t u1 = (bi + 1 < k) ? ((uint8_t)xr[bi + 1] ^ 0x80U) : 0;
                            int32_t u2 = (bi + 2 < k) ? ((uint8_t)xr[bi + 2] ^ 0x80U) : 0;
                            int32_t u3 = (bi + 3 < k) ? ((uint8_t)xr[bi + 3] ^ 0x80U) : 0;
                            for (size_t j = 0; j < width; ++j) {
                                const int8_t *cell = wp + (qq * C + j) * 4;
                                out[j] += u0 * cell[0] + u1 * cell[1] + u2 * cell[2] + u3 * cell[3];
                            }
                        }
                    }
                    store_y_i32_or_w8a8(y_dt, args->y, r * args->y_stride + col0, out, xs, wsc,
                                        args->bias, col0, width);
                }
            }
        } else {
            const int8_t *w = (const int8_t *)args->weights;
            const __m512i v80 = _mm512_set1_epi8((char)0x80);
            for (size_t r = r0; r < r1; ++r) {
                const int8_t *xr = x + r * args->x_stride;
                float xs = args->x_scale ? args->x_scale[r] : 1.0f;

                size_t k_vec = k & ~63ULL;
                __m512i x_pos = _mm512_setzero_si512();
                for (size_t i = 0; i < k_vec; i += 64) {
                    x_pos =
                        _mm512_dpbusd_epi32(x_pos, v80, _mm512_loadu_si512((const void *)(xr + i)));
                }
                int32_t row_corr = velri_i32_mul(-1, _mm512_reduce_add_epi32(x_pos));

                size_t c = c0;
                for (; c + 4 <= c1; c += 4) {
                    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
                    const int8_t *w0 = w + (c + 0) * args->w_stride;
                    const int8_t *w1 = w + (c + 1) * args->w_stride;
                    const int8_t *w2 = w + (c + 2) * args->w_stride;
                    const int8_t *w3 = w + (c + 3) * args->w_stride;
                    for (size_t i = 0; i < k_vec; i += 64) {
                        __m512i vx = _mm512_loadu_si512((const void *)(xr + i));
                        a0 = _mm512_dpbusd_epi32(
                            a0, _mm512_xor_si512(_mm512_loadu_si512((const void *)(w0 + i)), v80),
                            vx);
                        a1 = _mm512_dpbusd_epi32(
                            a1, _mm512_xor_si512(_mm512_loadu_si512((const void *)(w1 + i)), v80),
                            vx);
                        a2 = _mm512_dpbusd_epi32(
                            a2, _mm512_xor_si512(_mm512_loadu_si512((const void *)(w2 + i)), v80),
                            vx);
                        a3 = _mm512_dpbusd_epi32(
                            a3, _mm512_xor_si512(_mm512_loadu_si512((const void *)(w3 + i)), v80),
                            vx);
                    }
                    int32_t s[4] = {velri_i32_add(row_corr, _mm512_reduce_add_epi32(a0)),
                                    velri_i32_add(row_corr, _mm512_reduce_add_epi32(a1)),
                                    velri_i32_add(row_corr, _mm512_reduce_add_epi32(a2)),
                                    velri_i32_add(row_corr, _mm512_reduce_add_epi32(a3))};
                    for (size_t i = k_vec; i < k; ++i) {
                        int32_t xv = (int32_t)xr[i];
                        s[0] = velri_i32_add(s[0], xv * (int32_t)w0[i]);
                        s[1] = velri_i32_add(s[1], xv * (int32_t)w1[i]);
                        s[2] = velri_i32_add(s[2], xv * (int32_t)w2[i]);
                        s[3] = velri_i32_add(s[3], xv * (int32_t)w3[i]);
                    }
                    const float *wsc = args->w_scale ? (args->w_scale + c) : nullptr;
                    store_y_i32_or_w8a8(y_dt, args->y, r * args->y_stride + c, s, xs, wsc,
                                        args->bias, c, 4);
                }
                for (; c < c1; ++c) {
                    const int8_t *wc = w + c * args->w_stride;
                    __m512i a0 = _mm512_setzero_si512();
                    for (size_t i = 0; i < k_vec; i += 64) {
                        __m512i vx = _mm512_loadu_si512((const void *)(xr + i));
                        a0 = _mm512_dpbusd_epi32(
                            a0, _mm512_xor_si512(_mm512_loadu_si512((const void *)(wc + i)), v80),
                            vx);
                    }
                    int32_t sum = velri_i32_add(row_corr, _mm512_reduce_add_epi32(a0));
                    for (size_t i = k_vec; i < k; ++i) {
                        sum = velri_i32_add(sum, (int32_t)xr[i] * (int32_t)wc[i]);
                    }
                    const float *wsc = args->w_scale ? (args->w_scale + c) : nullptr;
                    store_y_i32_or_w8a8(y_dt, args->y, r * args->y_stride + c, &sum, xs, wsc,
                                        args->bias, c, 1);
                }
            }
        }
        return;
    }

    uint8_t *sc = align64_ptr(scratch);

    if (from_packed) {
        /* Panel offsets and bounds come from the workspace declarations above. */
        const size_t MC_L3 = I8_P_BAND;
        const size_t MC = I8_P_MC;
        const size_t KC = I8_P_KC;
        const size_t max_tiles = (C == 32) ? I8_P_TILES : 1;
        const size_t step_c = max_tiles * C;

        int32_t *acc_base = (int32_t *)sc;
        uint8_t *a_pack = sc + I8_P_A;
        int8_t *b_scratch = (int8_t *)(sc + I8_P_B);
        alignas(64) int32_t local_acc[448];

        for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
            size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
            for (size_t cb = c0; cb < c1; cb += step_c) {
                size_t cb_end = (c1 - cb < step_c) ? c1 : (cb + step_c);
                for (size_t rb = r_l3; rb < r_l3_end; rb += MC) {
                    size_t mb = (r_l3_end - rb < MC) ? (r_l3_end - rb) : MC;
                    size_t mr_blocks = (mb + 13) / 14;

                    for (size_t kb = 0; kb < k; kb += KC) {
                        size_t kc = (k - kb < KC) ? (k - kb) : KC;
                        size_t kc_quads = (kc + 3) / 4;
                        int32_t first_k = (kb == 0) ? 1 : 0;
                        bool last_k = (kb + kc >= k);

                        alignas(64) int32_t corr_tmp[32] = {0};
                        if (C < 32) {
                            size_t width0 = (cb_end - cb < C) ? (cb_end - cb) : C;
                            const uint8_t *tile_ptr0 = pk_base + (cb / C) * plan->tile_stride_b;
                            pack_b_32_i8_avx512(width0, kc, (const int8_t *)tile_ptr0, 0, cb, kb,
                                                true, C, b_scratch, nullptr);
                            if (first_k) {
                                const int32_t *tc = (const int32_t *)(tile_ptr0 + meta_off);
                                for (size_t j = 0; j < C; ++j) {
                                    corr_tmp[j] = tc[j];
                                }
                            }
                        }

                        /* `mr_i` OUTER, `col0` INNER: 16 KiB `a_pack` is reused across output
                         * tiles. */
                        for (size_t mr_i = 0; mr_i < mr_blocks; ++mr_i) {
                            size_t r_sub = mr_i * 14;
                            size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                            pack_a_14_i8_u8_avx512(mr, kc, x, args->x_stride, rb + r_sub, kb,
                                                   a_pack);

                            size_t tile_idx = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                                size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                                const uint8_t *tile_ptr =
                                    pk_base + (col0 / C) * plan->tile_stride_b;
                                const int8_t *b_panel =
                                    (C == 32) ? ((const int8_t *)tile_ptr + (kb / 4) * 128)
                                              : b_scratch;
                                const int32_t *corr_ptr = nullptr;
                                if (first_k) {
                                    corr_ptr = (C == 32) ? (const int32_t *)(tile_ptr + meta_off)
                                                         : corr_tmp;
                                }
                                const float *wsc = plan->desc.has_w_scale
                                                       ? (const float *)(tile_ptr + scale_off)
                                                       : nullptr;

                                int32_t *acc_tile =
                                    (first_k && last_k)
                                        ? local_acc
                                        : (acc_base + (tile_idx * (I8_P_MC / I8_MR) + mr_i) * 448);
                                bool fuse_i32 =
                                    (last_k && y_dt == VELRI_DTYPE_I32 && mr == 14 && width == 32);
                                i8_epilogue ep = {
                                    .acc = acc_tile,
                                    .corr = corr_ptr,
                                    .y_i32 = (y_dt == VELRI_DTYPE_I32)
                                                 ? ((int32_t *)args->y +
                                                    (rb + r_sub) * args->y_stride + col0)
                                                 : nullptr,
                                    .bias = (y_dt == VELRI_DTYPE_I32 && args->bias)
                                                ? ((const int32_t *)args->bias + col0)
                                                : nullptr,
                                    .y_stride_b = y_stride_b,
                                    .first_k = first_k,
                                    .fuse_i32 = fuse_i32 ? 1 : 0,
                                };
                                ukernel_14x32_i8_avx512(kc_quads, a_pack, b_panel, &ep);
                                if (last_k && !fuse_i32) {
                                    if (y_dt == VELRI_DTYPE_F32 && mr == 14 && width == 32) {
                                        store_y_w8a8_14x32((float *)args->y, args->y_stride,
                                                           rb + r_sub, col0, acc_tile,
                                                           args->x_scale, wsc,
                                                           (const float *)args->bias);
                                    } else {
                                        for (size_t r = 0; r < mr; ++r) {
                                            size_t row = rb + r_sub + r;
                                            float xs = args->x_scale ? args->x_scale[row] : 1.0f;
                                            store_y_i32_or_w8a8(y_dt, args->y,
                                                                row * args->y_stride + col0,
                                                                acc_tile + r * 32, xs, wsc,
                                                                args->bias, col0, width);
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
    const size_t MC_L3 = I8_R_BAND;
    const size_t MC_RM = I8_R_MC;
    const size_t NC_RM = I8_R_TILES * I8_NR;
    const size_t KC_RM = I8_R_KC;
    int32_t *acc_base = (int32_t *)sc;
    uint8_t *a_pack = sc + I8_R_A;
    int8_t *b_scratch = (int8_t *)(sc + I8_R_B);
    int32_t *panel_corr_buf = (int32_t *)(sc + I8_R_AUX);
    alignas(64) int32_t local_acc[448];

    for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
        size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
        for (size_t cb = c0; cb < c1; cb += NC_RM) {
            size_t cb_end = (c1 - cb < NC_RM) ? c1 : (cb + NC_RM);
            size_t n_tiles = (cb_end - cb + 31) / 32;

            for (size_t rb = r_l3; rb < r_l3_end; rb += MC_RM) {
                size_t mb = (r_l3_end - rb < MC_RM) ? (r_l3_end - rb) : MC_RM;

                for (size_t kb = 0; kb < k; kb += KC_RM) {
                    size_t kc = (k - kb < KC_RM) ? (k - kb) : KC_RM;
                    size_t kc_quads = (kc + 3) / 4;
                    int32_t first_k = (kb == 0) ? 1 : 0;
                    bool last_k = (kb + kc >= k);

                    for (size_t ti = 0; ti < n_tiles; ++ti) {
                        size_t col0 = cb + ti * 32;
                        size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                        pack_b_32_i8_avx512(
                            width, kc, (const int8_t *)args->weights, args->w_stride, col0, kb,
                            false, 32, b_scratch + ti * 128 * kc_quads, panel_corr_buf + ti * 32);
                    }

                    for (size_t r_sub = 0, mr_idx = 0; r_sub < mb; r_sub += 14, ++mr_idx) {
                        size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                        size_t r_row = rb + r_sub;
                        pack_a_14_i8_u8_avx512(mr, kc, x, args->x_stride, r_row, kb, a_pack);

                        for (size_t ti = 0; ti < n_tiles; ++ti) {
                            size_t col0 = cb + ti * 32;
                            size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                            const int8_t *b_panel = b_scratch + ti * 128 * kc_quads;
                            const int32_t *corr_ptr = panel_corr_buf + ti * 32;
                            const float *wsc = (plan->desc.has_w_scale && args->w_scale)
                                                   ? (args->w_scale + col0)
                                                   : nullptr;

                            int32_t *acc_tile =
                                (first_k && last_k)
                                    ? local_acc
                                    : (acc_base + (ti * (I8_R_MC / I8_MR) + mr_idx) * 448);
                            bool fuse_i32 =
                                (last_k && y_dt == VELRI_DTYPE_I32 && mr == 14 && width == 32);
                            i8_epilogue ep = {
                                .acc = acc_tile,
                                .corr = corr_ptr,
                                .y_i32 = (y_dt == VELRI_DTYPE_I32)
                                             ? ((int32_t *)args->y + r_row * args->y_stride + col0)
                                             : nullptr,
                                .bias = (y_dt == VELRI_DTYPE_I32 && args->bias)
                                            ? ((const int32_t *)args->bias + col0)
                                            : nullptr,
                                .y_stride_b = y_stride_b,
                                .first_k = first_k,
                                .fuse_i32 = fuse_i32 ? 1 : 0,
                            };
                            ukernel_14x32_i8_avx512(kc_quads, a_pack, b_panel, &ep);
                            if (last_k && !fuse_i32) {
                                if (y_dt == VELRI_DTYPE_F32 && mr == 14 && width == 32) {
                                    store_y_w8a8_14x32((float *)args->y, args->y_stride, r_row,
                                                       col0, acc_tile, args->x_scale, wsc,
                                                       (const float *)args->bias);
                                } else {
                                    for (size_t r = 0; r < mr; ++r) {
                                        size_t row = r_row + r;
                                        float xs = args->x_scale ? args->x_scale[row] : 1.0f;
                                        store_y_i32_or_w8a8(
                                            y_dt, args->y, row * args->y_stride + col0,
                                            acc_tile + r * 32, xs, wsc, args->bias, col0, width);
                                    }
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
 * 3. Weight-Only INT8 (`F32 * I8 * w_scale + bias -> F32`, `VELRI_LIN_F32X_I8W`)
 * ============================================================================ */


static inline void store_y_f32x_i8w_row(float *yd, const float *ar, const float *wsc,
                                        const float *bias, size_t col0, size_t width) {
    if (width == 32 && wsc) {
        __m512 v0 = _mm512_mul_ps(_mm512_load_ps(ar), _mm512_loadu_ps(wsc));
        __m512 v1 = _mm512_mul_ps(_mm512_load_ps(ar + 16), _mm512_loadu_ps(wsc + 16));
        if (bias) {
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias + col0));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias + col0 + 16));
        }
        _mm512_storeu_ps(yd, v0);
        _mm512_storeu_ps(yd + 16, v1);
    } else {
        for (size_t j = 0; j < width; ++j) {
            float ws = wsc ? wsc[j] : 1.0f;
            yd[j] = ar[j] * ws + (bias ? bias[col0 + j] : 0.0f);
        }
    }
}

static void avx512_f32x_i8w_run(const velri_linear_typed_plan *plan,
                                const velri_linear_run_args *args, size_t r0, size_t r1, size_t c0,
                                size_t c1, void *restrict scratch, bool from_packed) {
    const size_t k = plan->k;
    const float *x = (const float *)args->x;
    const float *bias = (const float *)args->bias;
    float *restrict y = (float *)args->y;
    const uint8_t *pk_base = (const uint8_t *)args->weights;
    const size_t C = from_packed ? plan->columns : 32;
    const size_t payload_b = from_packed ? (plan->k_aligned * C) : 0;
    const size_t meta_off = (payload_b + 3ULL) & ~3ULL;
    const size_t scale_off = meta_off + C * sizeof(int32_t);

    if (r1 - r0 <= 2) {
        if (from_packed)
            velri_gemv_f32_prepared_avx512(plan, args, r0, r1, c0, c1);
        else
            velri_gemv_f32_rowmajor_avx512(plan, args, r0, r1, c0, c1);
        return;
    }

    uint8_t *sc = align64_ptr(scratch);

    if (from_packed) {
        const size_t MC_L3 = F32_I8_P_BAND;
        const size_t MC = F32_I8_P_MC;
        const size_t KC = F32_I8_P_KC;
        const size_t max_tiles = (C == 32) ? F32_I8_P_TILES : 1;
        const size_t step_c = max_tiles * C;

        float *acc_base = (float *)sc;
        float *a_pack = (float *)(sc + F32_I8_P_A);
        int8_t *b_scratch = (int8_t *)(sc + F32_I8_P_B);
        alignas(64) float local_acc[448];

        for (size_t r_l3 = r0; r_l3 < r1; r_l3 += MC_L3) {
            size_t r_l3_end = (r1 - r_l3 < MC_L3) ? r1 : (r_l3 + MC_L3);
            for (size_t cb = c0; cb < c1; cb += step_c) {
                size_t cb_end = (c1 - cb < step_c) ? c1 : (cb + step_c);
                for (size_t rb = r_l3; rb < r_l3_end; rb += MC) {
                    size_t mb = (r_l3_end - rb < MC) ? (r_l3_end - rb) : MC;
                    size_t mr_blocks = (mb + 13) / 14;
                    for (size_t kb = 0; kb < k; kb += KC) {
                        size_t kc = (k - kb < KC) ? (k - kb) : KC;
                        int first_k = (kb == 0) ? 1 : 0;
                        bool last_k = (kb + kc >= k);

                        if (C < 32) {
                            const int8_t *wp =
                                (const int8_t *)(pk_base + (cb / C) * plan->tile_stride_b);
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
                            pack_a_14_f32_for_i8w(mr, kc, x, args->x_stride, rb + r_sub, kb,
                                                  a_pack);

                            size_t tile_idx = 0;
                            for (size_t col0 = cb; col0 < cb_end; col0 += C, ++tile_idx) {
                                size_t width = (cb_end - col0 < C) ? (cb_end - col0) : C;
                                const uint8_t *tile_ptr =
                                    pk_base + (col0 / C) * plan->tile_stride_b;
                                const float *wsc = (const float *)(tile_ptr + scale_off);
                                const int8_t *b_panel =
                                    (C == 32) ? ((const int8_t *)tile_ptr + kb * 32) : b_scratch;

                                float *acc_tile =
                                    (first_k && last_k)
                                        ? local_acc
                                        : (acc_base +
                                           (tile_idx * (F32_I8_P_MC / F32_I8_MR) + mr_i) * 448);
                                ukernel_14x32_i8w_f32_avx512(kc, a_pack, b_panel, acc_tile,
                                                             first_k);
                                if (last_k) {
                                    for (size_t r = 0; r < mr; ++r) {
                                        store_y_f32x_i8w_row(
                                            y + (rb + r_sub + r) * args->y_stride + col0,
                                            acc_tile + r * 32, wsc, bias, col0, width);
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

    const size_t MC_L3 = F32_I8_R_BAND;
    const size_t MC_RM = F32_I8_R_MC;
    const size_t NC_RM = F32_I8_R_TILES * F32_I8_NR;
    const size_t KC_RM = F32_I8_R_KC;
    float *acc_base = (float *)sc;
    float *a_pack = (float *)(sc + F32_I8_R_A);
    int8_t *b_scratch = (int8_t *)(sc + F32_I8_R_B);
    const int8_t *w = (const int8_t *)args->weights;
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
                    int first_k = (kb == 0) ? 1 : 0;
                    bool last_k = (kb + kc >= k);

                    for (size_t ti = 0; ti < n_tiles; ++ti) {
                        size_t col0 = cb + ti * 32;
                        size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                        pack_b_32_i8_g1_avx512(width, kc, w, args->w_stride, col0, kb,
                                               b_scratch + ti * 32 * kc);
                    }

                    for (size_t r_sub = 0, mr_idx = 0; r_sub < mb; r_sub += 14, ++mr_idx) {
                        size_t mr = (mb - r_sub < 14) ? (mb - r_sub) : 14;
                        size_t r_row = rb + r_sub;
                        pack_a_14_f32_for_i8w(mr, kc, x, args->x_stride, r_row, kb, a_pack);

                        for (size_t ti = 0; ti < n_tiles; ++ti) {
                            size_t col0 = cb + ti * 32;
                            size_t width = (cb_end - col0 < 32) ? (cb_end - col0) : 32;
                            const int8_t *b_panel = b_scratch + ti * 32 * kc;
                            const float *wsc = args->w_scale ? (args->w_scale + col0) : nullptr;
                            float *acc_tile =
                                (first_k && last_k)
                                    ? local_acc
                                    : (acc_base + (ti * (F32_I8_R_MC / F32_I8_MR) + mr_idx) * 448);
                            ukernel_14x32_i8w_f32_avx512(kc, a_pack, b_panel, acc_tile, first_k);
                            if (last_k) {
                                for (size_t r = 0; r < mr; ++r) {
                                    store_y_f32x_i8w_row(y + (r_row + r) * args->y_stride + col0,
                                                         acc_tile + r * 32, wsc, bias, col0, width);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void avx512_i8_prep_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                 size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_i8_run(p, a, r0, r1, c0, c1, sc, true);
}
static void avx512_i8_rm_entry(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                               size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_i8_run(p, a, r0, r1, c0, c1, sc, false);
}
static void avx512_f32x_i8w_prep(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                                 size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_f32x_i8w_run(p, a, r0, r1, c0, c1, sc, true);
}
static void avx512_f32x_i8w_rm(const velri_linear_typed_plan *p, const velri_linear_run_args *a,
                               size_t r0, size_t r1, size_t c0, size_t c1, void *restrict sc) {
    avx512_f32x_i8w_run(p, a, r0, r1, c0, c1, sc, false);
}

static const velri_lin_impl g_avx512_int8_impls[] = {
    [VELRI_LIN_I8_I32] = {VELRI_LIN_I8_I32, VELRI_CPU_AVX512, 32, 4, &i8_p, &i8_r,
                          avx512_i8_pack_g4, avx512_i8_prep_entry, avx512_i8_rm_entry},
    [VELRI_LIN_W8A8_F32] = {VELRI_LIN_W8A8_F32, VELRI_CPU_AVX512, 32, 4, &i8_p, &i8_r,
                            avx512_i8_pack_g4, avx512_i8_prep_entry, avx512_i8_rm_entry},
    [VELRI_LIN_F32X_I8W] = {VELRI_LIN_F32X_I8W, VELRI_CPU_AVX512, 32, 1, &f32_i8_p, &f32_i8_r,
                            avx512_i8_pack_g1, avx512_f32x_i8w_prep, avx512_f32x_i8w_rm},
};

const velri_lin_impl *velri_lin_get_avx512_int8(velri_lin_kind kind) {
    if (kind < VELRI_LIN_I8_I32 || kind > VELRI_LIN_F32X_I8W) {
        return nullptr;
    }
    return &g_avx512_int8_impls[kind];
}
