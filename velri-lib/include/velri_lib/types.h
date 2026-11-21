#ifndef VELRI_LIB_TYPES_H
#define VELRI_LIB_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Explicit 16-bit floating-point storage representations in standard C23:
 *   - velri_fp16: IEEE-754 binary16 (1 sign, 5 exponent, 10 fraction bits)
 *   - velri_bf16: Brain Float 16    (1 sign, 8 exponent,  7 fraction bits)
 * Using uint16_t storage keeps public headers portable across x86-64, ARM64,
 * and RISC-V without requiring target-specific _Float16 / __bf16 ABI support.
 */
typedef uint16_t velri_fp16;
typedef uint16_t velri_bf16;

typedef enum {
    VELRI_OK = 0,
    VELRI_ERR_INVALID = 1,
    VELRI_ERR_OVERFLOW = 2,
    VELRI_ERR_UNSUPPORTED = 3,
    VELRI_ERR_NOMEM = 4,
    VELRI_ERR_CANCELLED = 5,
    VELRI_ERR_EXEC = 6,
    VELRI_ERR_BUSY = 7,
    VELRI_ERR_NOT_FOUND = 8
} velri_status;

typedef enum {
    VELRI_CPU_AUTO = 0,
    VELRI_CPU_BASE = 1,
    VELRI_CPU_AVX2 = 2,
    VELRI_CPU_AVX512 = 3
} velri_cpu_variant;

typedef enum {
    VELRI_DTYPE_F32 = 0,
    VELRI_DTYPE_F64 = 1,
    VELRI_DTYPE_F16 = 2,
    VELRI_DTYPE_BF16 = 3,
    VELRI_DTYPE_I8 = 4,
    VELRI_DTYPE_I32 = 5
} velri_dtype;

typedef struct {
    int hardware_threads;
    int num_l3_domains;
    bool has_avx2;
    bool has_fma;
    bool has_f16c;
    bool has_avx_vnni;
    bool has_avx512f;
    bool has_avx512bw;
    bool has_avx512dq;
    bool has_avx512vl;
    bool has_avx512_bf16;
    bool has_avx512_vnni;
} velri_cpu_info;

/* Query element storage size in bytes (returns 0 for unknown enum values). */
size_t velri_dtype_bytes(velri_dtype dtype);

/* Query required natural alignment in bytes for a dtype. */
size_t velri_dtype_align(velri_dtype dtype);

/* Check whether BOTH the CPU and OS extended-state support `variant`. */
bool velri_cpu_variant_supported(velri_cpu_variant variant);

/* Resolve VELRI_CPU_AUTO to the widest supported variant on this host. */
velri_cpu_variant velri_cpu_resolve_variant(velri_cpu_variant variant);

/* Return immutable detected CPU ISA capability summary. */
const velri_cpu_info *velri_get_cpu_info(void);

/* ============================================================================
 * Portable Bit-Exact IEEE-754 Scalar FP16 & BF16 Conversion Helpers
 * ============================================================================ */

static inline uint32_t velri_f32_to_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static inline float velri_bits_to_f32(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static inline float velri_bf16_to_f32_scalar(velri_bf16 h) {
    return velri_bits_to_f32((uint32_t)h << 16);
}

static inline velri_bf16 velri_f32_to_bf16_scalar(float x) {
    uint32_t u = velri_f32_to_bits(x);
    uint32_t abs_u = u & UINT32_C(0x7FFFFFFF);
    if (abs_u > UINT32_C(0x7F800000)) {
        return (velri_bf16)((u >> 16) | UINT16_C(0x0040));
    }
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t rounded = u + UINT32_C(0x7FFF) + lsb;
    return (velri_bf16)(rounded >> 16);
}

static inline float velri_f16_to_f32_scalar(velri_fp16 h) {
    uint32_t sign = (uint32_t)(h & UINT16_C(0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    if (exp == 0) {
        if (mant == 0) {
            return velri_bits_to_f32(sign);
        }
        uint32_t e = 127U - 14U;
        while ((mant & 0x0400U) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03FFU;
        return velri_bits_to_f32(sign | (e << 23) | (mant << 13));
    }
    if (exp == 0x1FU) {
        if (mant == 0) {
            return velri_bits_to_f32(sign | UINT32_C(0x7F800000));
        }
        return velri_bits_to_f32(sign | UINT32_C(0x7FC00000) | (mant << 13));
    }
    return velri_bits_to_f32(sign | ((exp + (127U - 15U)) << 23) | (mant << 13));
}

static inline velri_fp16 velri_f32_to_f16_scalar(float x) {
    uint32_t u = velri_f32_to_bits(x);
    uint16_t sign = (uint16_t)((u >> 16) & UINT16_C(0x8000));
    uint32_t abs_u = u & UINT32_C(0x7FFFFFFF);

    if (abs_u > UINT32_C(0x7F800000)) {
        uint16_t payload = (uint16_t)((abs_u >> 13) & 0x03FFU);
        return (velri_fp16)(sign | UINT16_C(0x7E00) | payload);
    }
    if (abs_u >= UINT32_C(0x477FF000)) {
        if (abs_u >= UINT32_C(0x7F800000)) {
            return (velri_fp16)(sign | UINT16_C(0x7C00));
        }
        uint32_t exp = (abs_u >> 23) - 127U + 15U;
        uint32_t mant = abs_u & UINT32_C(0x007FFFFF);
        if (exp > 30U) {
            return (velri_fp16)(sign | UINT16_C(0x7C00));
        }
        uint32_t lsb = (mant >> 13) & 1U;
        uint32_t r = mant + UINT32_C(0x0FFF) + lsb;
        if (r >= UINT32_C(0x00800000)) {
            return (velri_fp16)(sign | UINT16_C(0x7C00));
        }
        return (velri_fp16)(sign | (uint16_t)(exp << 10) | (uint16_t)(r >> 13));
    }
    if (abs_u < UINT32_C(0x33000000)) {
        return sign;
    }
    if (abs_u < UINT32_C(0x38800000)) {
        uint32_t f_exp = abs_u >> 23;
        uint32_t sig = (abs_u & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
        uint32_t shift = 126U - f_exp;
        uint32_t q = sig >> shift;
        uint32_t rem = sig & ((UINT32_C(1) << shift) - 1U);
        uint32_t half = UINT32_C(1) << (shift - 1U);
        if (rem > half || (rem == half && (q & 1U) != 0U)) {
            ++q;
        }
        return (velri_fp16)(sign | (uint16_t)q);
    }
    uint32_t exp = (abs_u >> 23) - (127U - 15U);
    uint32_t mant = abs_u & UINT32_C(0x007FFFFF);
    uint32_t lsb = (mant >> 13) & 1U;
    uint32_t rounded = mant + UINT32_C(0x0FFF) + lsb;
    if (rounded >= UINT32_C(0x00800000)) {
        rounded -= UINT32_C(0x00800000);
        ++exp;
        if (exp >= 31U) {
            return (velri_fp16)(sign | UINT16_C(0x7C00));
        }
    }
    return (velri_fp16)(sign | (uint16_t)(exp << 10) | (uint16_t)(rounded >> 13));
}

#endif /* VELRI_LIB_TYPES_H */
