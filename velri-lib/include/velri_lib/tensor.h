#ifndef VELRI_LIB_TENSOR_H
#define VELRI_LIB_TENSOR_H

#include "velri_lib/types.h"

/*
 * Storage, TensorView, and explicit K/V input role descriptors.
 *
 * Core invariant (velri-lib/docs/EN/architecture.md §3):
 *   - `velri_storage_desc` owns physical memory allocation, device, and lifetime.
 *   - `velri_tensor_view` is a zero-copy view (shape, dtype, offset, strides) into a storage.
 *   - K and V are attention input roles, never an implicit mandate to allocate two buffers.
 *     Only explicitly declared storages are allocated by the memory planner.
 */

enum {
    VELRI_MAX_RANK = 4,
    VELRI_INVALID_ID = 0xFFFFFFFFU
};

typedef uint32_t velri_storage_id;
typedef uint32_t velri_view_id;
typedef uint32_t velri_node_id;
typedef uint32_t velri_device_id;

typedef enum {
    VELRI_STORAGE_WEIGHT = 0,  /* Immutable weights; shared across concurrent execution slots. */
    VELRI_STORAGE_INPUT = 1,   /* Step input bound per slot before execution. */
    VELRI_STORAGE_OUTPUT = 2,  /* Step output bound per slot; kept alive through step end. */
    VELRI_STORAGE_STATE = 3,   /* Mutable session state (KV, GDN, conv) preserved across steps. */
    VELRI_STORAGE_SCRATCH = 4  /* Temporary slot-local buffer packed/reused by lifetime planner. */
} velri_storage_kind;

typedef struct {
    velri_storage_kind kind;
    velri_dtype dtype;
    size_t byte_size;
    size_t align;                  /* 0 => natural alignment >= 64 B */
    velri_device_id device_id;     /* Target device owning this storage */
    const char *debug_name;        /* Optional human-readable label (copied by graph) */
} velri_storage_desc;

typedef struct {
    velri_storage_id storage_id;
    velri_dtype dtype;
    uint32_t rank;                 /* 1..VELRI_MAX_RANK */
    size_t dims[VELRI_MAX_RANK];
    size_t strides[VELRI_MAX_RANK]; /* Element strides (innermost typically 1) */
    size_t byte_offset;            /* Byte offset inside `storage_id` */
} velri_tensor_view;

/*
 * Explicit K/V input role binding for attention blocks.
 * Declaring a K/V role never creates hidden allocations; both `k_view` and `v_view`
 * reference views already declared on existing storages (or derived via an op node).
 */
typedef enum {
    VELRI_KV_SEPARATE = 0, /* K and V views reference distinct storages */
    VELRI_KV_SHARED = 1,   /* K and V views alias the exact same storage_id (0 extra bytes) */
    VELRI_KV_DERIVED = 2   /* K is derived from V via an explicit graph op (e.g. K = RoPE(V*w_k)) */
} velri_kv_source_kind;

typedef struct {
    velri_kv_source_kind kind;
    velri_view_id k_view;
    velri_view_id v_view;
    velri_node_id derive_node; /* Used when kind == VELRI_KV_DERIVED, else VELRI_INVALID_ID */
} velri_kv_binding;

/* Build a row-major contiguous tensor view at `byte_offset`. */
velri_status velri_tensor_view_contiguous(
    velri_storage_id storage_id,
    velri_dtype dtype,
    uint32_t rank,
    const size_t *dims,
    size_t byte_offset,
    velri_tensor_view *out_view);

/* Create a zero-copy subview slicing the same storage_id at an additional element offset. */
velri_status velri_tensor_view_subview(
    const velri_tensor_view *base,
    uint32_t rank,
    const size_t *dims,
    const size_t *strides,
    size_t extra_byte_offset,
    velri_tensor_view *out_view);

/* Compute total logical element count (overflow-checked). */
velri_status velri_tensor_view_num_elements(const velri_tensor_view *view, size_t *out_elems);

/* Compute [byte_offset, byte_end) covered by `view` (overflow-checked). */
velri_status velri_tensor_view_byte_span(
    const velri_tensor_view *view,
    size_t *out_start,
    size_t *out_end);

/* Validate `view` shape, dtype, alignment, strides, and bounds against `storage`. */
velri_status velri_tensor_view_validate(
    const velri_tensor_view *view,
    const velri_storage_desc *storage);

/* True when the innermost dimension is unit-strided and outer dimensions are contiguous. */
bool velri_tensor_view_is_contiguous(const velri_tensor_view *view);

#endif /* VELRI_LIB_TENSOR_H */
