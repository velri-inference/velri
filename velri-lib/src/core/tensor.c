#include "velri_lib/tensor.h"
#include <stdint.h>
#include <string.h>

static bool mul_overflow_size(size_t a, size_t b, size_t *out) {
    if (a != 0U && b > SIZE_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
}

static bool add_overflow_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) {
        return true;
    }
    *out = a + b;
    return false;
}

velri_status velri_tensor_view_contiguous(
    velri_storage_id storage_id,
    velri_dtype dtype,
    uint32_t rank,
    const size_t *dims,
    size_t byte_offset,
    velri_tensor_view *out_view) {
    if (!dims || !out_view || rank == 0U || rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(dtype);
    if (eb == 0U) {
        return VELRI_ERR_INVALID;
    }
    size_t ea = velri_dtype_align(dtype);
    if ((byte_offset % ea) != 0U) {
        return VELRI_ERR_INVALID;
    }

    velri_tensor_view v;
    memset(&v, 0, sizeof(v));
    v.storage_id = storage_id;
    v.dtype = dtype;
    v.rank = rank;
    v.byte_offset = byte_offset;

    size_t stride = 1U;
    for (uint32_t i = rank; i > 0U; --i) {
        uint32_t d = i - 1U;
        if (dims[d] == 0U) {
            return VELRI_ERR_INVALID;
        }
        v.dims[d] = dims[d];
        v.strides[d] = stride;
        if (mul_overflow_size(stride, dims[d], &stride)) {
            return VELRI_ERR_OVERFLOW;
        }
    }

    size_t total_bytes = 0U;
    if (mul_overflow_size(stride, eb, &total_bytes)) {
        return VELRI_ERR_OVERFLOW;
    }
    size_t end = 0U;
    if (add_overflow_size(byte_offset, total_bytes, &end)) {
        return VELRI_ERR_OVERFLOW;
    }

    *out_view = v;
    return VELRI_OK;
}

velri_status velri_tensor_view_subview(
    const velri_tensor_view *base,
    uint32_t rank,
    const size_t *dims,
    const size_t *strides,
    size_t extra_byte_offset,
    velri_tensor_view *out_view) {
    if (!base || !dims || !out_view || rank == 0U || rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(base->dtype);
    size_t ea = velri_dtype_align(base->dtype);
    if (eb == 0U) {
        return VELRI_ERR_INVALID;
    }
    size_t new_offset = 0U;
    if (add_overflow_size(base->byte_offset, extra_byte_offset, &new_offset)) {
        return VELRI_ERR_OVERFLOW;
    }
    if ((new_offset % ea) != 0U) {
        return VELRI_ERR_INVALID;
    }

    velri_tensor_view v;
    memset(&v, 0, sizeof(v));
    v.storage_id = base->storage_id;
    v.dtype = base->dtype;
    v.rank = rank;
    v.byte_offset = new_offset;

    if (strides) {
        for (uint32_t d = 0U; d < rank; ++d) {
            if (dims[d] == 0U) {
                return VELRI_ERR_INVALID;
            }
            v.dims[d] = dims[d];
            v.strides[d] = strides[d];
        }
    } else {
        size_t s = 1U;
        for (uint32_t i = rank; i > 0U; --i) {
            uint32_t d = i - 1U;
            if (dims[d] == 0U) {
                return VELRI_ERR_INVALID;
            }
            v.dims[d] = dims[d];
            v.strides[d] = s;
            if (mul_overflow_size(s, dims[d], &s)) {
                return VELRI_ERR_OVERFLOW;
            }
        }
    }

    size_t s0 = 0U, s1 = 0U;
    velri_status st = velri_tensor_view_byte_span(&v, &s0, &s1);
    if (st != VELRI_OK) {
        return st;
    }
    *out_view = v;
    return VELRI_OK;
}

velri_status velri_tensor_view_num_elements(const velri_tensor_view *view, size_t *out_elems) {
    if (!view || !out_elems || view->rank == 0U || view->rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t n = 1U;
    for (uint32_t d = 0U; d < view->rank; ++d) {
        if (view->dims[d] == 0U) {
            return VELRI_ERR_INVALID;
        }
        if (mul_overflow_size(n, view->dims[d], &n)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    *out_elems = n;
    return VELRI_OK;
}

velri_status velri_tensor_view_byte_span(
    const velri_tensor_view *view,
    size_t *out_start,
    size_t *out_end) {
    if (!view || !out_start || !out_end || view->rank == 0U || view->rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(view->dtype);
    if (eb == 0U) {
        return VELRI_ERR_INVALID;
    }

    /* Maximum accessed element index = sum_d (dims[d] - 1) * strides[d]. */
    size_t max_elem = 0U;
    for (uint32_t d = 0U; d < view->rank; ++d) {
        if (view->dims[d] == 0U) {
            return VELRI_ERR_INVALID;
        }
        size_t term = 0U;
        if (mul_overflow_size(view->dims[d] - 1U, view->strides[d], &term)) {
            return VELRI_ERR_OVERFLOW;
        }
        if (add_overflow_size(max_elem, term, &max_elem)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    size_t span_elems = 0U;
    if (add_overflow_size(max_elem, 1U, &span_elems)) {
        return VELRI_ERR_OVERFLOW;
    }
    size_t span_bytes = 0U;
    if (mul_overflow_size(span_elems, eb, &span_bytes)) {
        return VELRI_ERR_OVERFLOW;
    }
    size_t end = 0U;
    if (add_overflow_size(view->byte_offset, span_bytes, &end)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_start = view->byte_offset;
    *out_end = end;
    return VELRI_OK;
}

velri_status velri_tensor_view_validate(
    const velri_tensor_view *view,
    const velri_storage_desc *storage) {
    if (!view || !storage) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(view->dtype);
    size_t ea = velri_dtype_align(view->dtype);
    if (eb == 0U || (view->byte_offset % ea) != 0U) {
        return VELRI_ERR_INVALID;
    }
    size_t start = 0U, end = 0U;
    velri_status st = velri_tensor_view_byte_span(view, &start, &end);
    if (st != VELRI_OK) {
        return st;
    }
    if (end > storage->byte_size) {
        return VELRI_ERR_INVALID;
    }
    return VELRI_OK;
}

bool velri_tensor_view_is_contiguous(const velri_tensor_view *view) {
    if (!view || view->rank == 0U || view->rank > VELRI_MAX_RANK) {
        return false;
    }
    size_t expected = 1U;
    for (uint32_t i = view->rank; i > 0U; --i) {
        uint32_t d = i - 1U;
        if (view->dims[d] > 1U && view->strides[d] != expected) {
            return false;
        }
        if (mul_overflow_size(expected, view->dims[d], &expected)) {
            return false;
        }
    }
    return true;
}
