#ifndef VELRI_LINEAR_WORKSPACE_H
#define VELRI_LINEAR_WORKSPACE_H

/* Compile-time offsets for four contiguous scratch regions: accumulators,
 * packed activations (A), packed weights (B), and optional metadata. Kernels
 * supply byte counts derived from their block dimensions. The same constants
 * address the buffers and determine the public plan's required allocation.
 * Include 63 bytes so the caller's scratch pointer need not be aligned. */
#define VELRI_ALIGN64_BYTES(n) (((n) + 63) & ~63)
#define VELRI_LINEAR_WORKSPACE(name, acc_bytes, a_bytes, b_bytes, aux_bytes)                       \
    enum {                                                                                         \
        name##_A = VELRI_ALIGN64_BYTES(acc_bytes),                                                 \
        name##_B = VELRI_ALIGN64_BYTES(name##_A + (a_bytes)),                                      \
        name##_AUX = VELRI_ALIGN64_BYTES(name##_B + (b_bytes)),                                    \
        name##_BYTES = name##_AUX + (aux_bytes) + 63                                               \
    }

#endif
