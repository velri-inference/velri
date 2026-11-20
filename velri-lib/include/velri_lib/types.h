#ifndef VELRI_LIB_TYPES_H
#define VELRI_LIB_TYPES_H

#include <stddef.h>
#include <stdint.h>

/*
 * Explicit 16-bit floating-point storage representations in standard C23:
 *   - velri_fp16: IEEE-754 binary16 (1 sign, 5 exponent, 10 fraction bits)
 *   - velri_bf16: Brain Float 16    (1 sign, 8 exponent,  7 fraction bits)
 * Using uint16_t storage keeps public headers portable across x86-64 and ARM64
 * without requiring target-specific _Float16 / __bf16 ABI support in callers.
 */
typedef uint16_t velri_fp16;
typedef uint16_t velri_bf16;

typedef enum {
    VELRI_OK = 0,
    VELRI_ERR_INVALID = 1,
    VELRI_ERR_OVERFLOW = 2,
    VELRI_ERR_UNSUPPORTED = 3,
    VELRI_ERR_NOMEM = 4
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
 * Vectorized Format Conversions & Dynamic Symmetric INT8 Row Quantization
 * Read-only `src` and mutable `dst` must not overlap unless `count == 0`.
 * ============================================================================ */

velri_status velri_cvt_f32_to_f16(const float *src, velri_fp16 *restrict dst, size_t count,
                                  velri_cpu_variant variant);
velri_status velri_cvt_f16_to_f32(const velri_fp16 *src, float *restrict dst, size_t count,
                                  velri_cpu_variant variant);

velri_status velri_cvt_f32_to_bf16(const float *src, velri_bf16 *restrict dst, size_t count,
                                   velri_cpu_variant variant);
velri_status velri_cvt_bf16_to_f32(const velri_bf16 *src, float *restrict dst, size_t count,
                                   velri_cpu_variant variant);

/*
 * Dynamic symmetric per-row INT8 quantization for W8A8 activations:
 * For each row r in [0, rows):
 *   amax = max_{i in [0, k)} |x[r * x_stride + i]|
 *   scale_out[r] = amax > 0 ? max(amax / 127, FLT_TRUE_MIN) : 1
 *   Quantize with reciprocal multiplication, round to nearest even and clamp
 *   to [-127, 127]. Tiny rows use double division to avoid reciprocal overflow.
 * Inputs must be finite. The calling thread and pool workers require default
 * round-to-nearest mode with subnormal flushing disabled.
 */
velri_status velri_quantize_row_f32_i8(size_t rows, size_t k, const float *x, size_t x_stride,
                                       int8_t *restrict x_i8, size_t xi8_stride,
                                       float *restrict scale_out, velri_cpu_variant variant);

#endif /* VELRI_LIB_TYPES_H */
