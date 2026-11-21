#ifndef VELRI_FORMAT_JSON_H
#define VELRI_FORMAT_JSON_H

#include <velri_lib/types.h>

/*
 * Minimal Zero-Dependency Streaming C23 JSON Scanner (RFC 8259).
 *
 * Designed for:
 *   1. Cascading placement & engine configuration files (`src/config/placement.{h,c}`).
 *   2. Unbounded JSON footers in `.vtq` weight containers (`mmap`-backed buffers,
 *      skipping large unused sections like `tokenizer` in O(n) with 0 B heap allocations).
 *
 * Guarantees:
 *   - 0 B heap allocations across all operations.
 *   - Works directly on non-null-terminated `(data, len)` buffers.
 *   - Non-recursive `velri_json_skip` bounded by `VELRI_JSON_MAX_DEPTH`.
 *   - Strict RFC 8259 validation (strings, UTF-16 surrogates, numbers, trailing commas/garbage).
 */

enum {
    VELRI_JSON_MAX_DEPTH = 64,
    VELRI_JSON_ERR_MSG_MAX = 160
};

typedef enum {
    VELRI_JSON_INVALID = 0,
    VELRI_JSON_OBJECT,
    VELRI_JSON_ARRAY,
    VELRI_JSON_STRING,
    VELRI_JSON_NUMBER,
    VELRI_JSON_BOOL,
    VELRI_JSON_NULL
} velri_json_type;

/* Zero-copy raw JSON string slice (content between quotes, unescaped on demand). */
typedef struct {
    const char *ptr;
    size_t len;
    bool has_escapes;
} velri_json_str;

typedef struct {
    bool first;
    bool done;
} velri_json_iter;

typedef struct {
    const char *data;
    size_t len;
    size_t pos;
    uint32_t depth;
    uint32_t line;
    uint32_t col;
    size_t err_offset;
    char err_msg[VELRI_JSON_ERR_MSG_MAX];
} velri_json_reader;

void velri_json_init(velri_json_reader *r, const char *data, size_t len);

/* Inspect the type of the next JSON value (skipping leading whitespace). */
velri_json_type velri_json_peek(velri_json_reader *r);

/* Object iteration: begin_object -> loop next_member until !has_more. */
velri_status velri_json_begin_object(velri_json_reader *r, velri_json_iter *it);
velri_status velri_json_next_member(velri_json_reader *r, velri_json_iter *it,
                                    velri_json_str *out_key, bool *out_has_more);

/* Array iteration: begin_array -> loop next_elem until !has_more. */
velri_status velri_json_begin_array(velri_json_reader *r, velri_json_iter *it);
velri_status velri_json_next_elem(velri_json_reader *r, velri_json_iter *it, bool *out_has_more);

/* Scalar readers. */
velri_status velri_json_read_string(velri_json_reader *r, velri_json_str *out);
velri_status velri_json_read_string_buf(velri_json_reader *r, char *dst, size_t dst_cap,
                                        size_t *out_len);
velri_status velri_json_read_i64(velri_json_reader *r, int64_t *out);
velri_status velri_json_read_u64(velri_json_reader *r, uint64_t *out);
velri_status velri_json_read_f64(velri_json_reader *r, double *out);
velri_status velri_json_read_bool(velri_json_reader *r, bool *out);
velri_status velri_json_read_null(velri_json_reader *r);

/* Non-recursive O(n), 0 B-allocation skip of any valid JSON value. */
velri_status velri_json_skip(velri_json_reader *r);

/* Ensure only whitespace remains after the top-level JSON value. */
velri_status velri_json_finish(velri_json_reader *r);

/* String helpers (handling escape decoding when `s.has_escapes` is true). */
velri_status velri_json_str_decode(velri_json_str s, char *dst, size_t dst_cap, size_t *out_len);
bool velri_json_str_eq(velri_json_str s, const char *lit);

#endif /* VELRI_FORMAT_JSON_H */
