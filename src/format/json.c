#include "format/json.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Internal Cursor, UTF-8 & Escape Helpers ---- */

static velri_status json_fail(velri_json_reader *r, velri_status st, const char *msg) {
    if (r && r->err_msg[0] == '\0') {
        r->err_offset = r->pos;
        snprintf(r->err_msg, sizeof(r->err_msg), "JSON error at line %u, col %u: %s", r->line,
                 r->col, msg ? msg : "invalid syntax");
    }
    return st;
}

static inline void advance_n(velri_json_reader *r, size_t n) {
    while (n-- > 0U && r->pos < r->len) {
        if (r->data[r->pos++] == '\n') {
            r->line++;
            r->col = 1U;
        } else {
            r->col++;
        }
    }
}

static void skip_ws(velri_json_reader *r) {
    while (r->pos < r->len) {
        char c = r->data[r->pos];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        advance_n(r, 1U);
    }
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (c - 'A');
    }
    return -1;
}

static bool parse_hex4(const char *p, uint32_t *out) {
    uint32_t v = 0U;
    for (int i = 0; i < 4; ++i) {
        int h = hex_val(p[i]);
        if (h < 0) {
            return false;
        }
        v = (v << 4U) | (uint32_t)h;
    }
    *out = v;
    return true;
}

static size_t encode_utf8(uint32_t cp, char out[4]) {
    if (cp <= 0x7FU) {
        out[0] = (char)cp;
        return 1U;
    }
    if (cp <= 0x7FFU) {
        out[0] = (char)(0xC0U | (cp >> 6U));
        out[1] = (char)(0x80U | (cp & 0x3FU));
        return 2U;
    }
    if (cp <= 0xFFFFU) {
        out[0] = (char)(0xE0U | (cp >> 12U));
        out[1] = (char)(0x80U | ((cp >> 6U) & 0x3FU));
        out[2] = (char)(0x80U | (cp & 0x3FU));
        return 3U;
    }
    if (cp <= 0x10FFFFU) {
        out[0] = (char)(0xF0U | (cp >> 18U));
        out[1] = (char)(0x80U | ((cp >> 12U) & 0x3FU));
        out[2] = (char)(0x80U | ((cp >> 6U) & 0x3FU));
        out[3] = (char)(0x80U | (cp & 0x3FU));
        return 4U;
    }
    return 0U;
}

/* Decode one raw byte or escape sequence (incl. UTF-16 surrogate pair) into 1..4 UTF-8 bytes. */
static bool decode_next_unit(const char *s, size_t len, size_t *pos, char out[4], size_t *out_n,
                             bool reject_nul) {
    char c = s[(*pos)++];
    if (c != '\\') {
        out[0] = c;
        *out_n = 1U;
        return true;
    }
    if (*pos >= len) return false;
    char esc = s[(*pos)++];
    switch (esc) {
    case '"':  out[0] = '"';  *out_n = 1U; return true;
    case '\\': out[0] = '\\'; *out_n = 1U; return true;
    case '/':  out[0] = '/';  *out_n = 1U; return true;
    case 'b':  out[0] = '\b'; *out_n = 1U; return true;
    case 'f':  out[0] = '\f'; *out_n = 1U; return true;
    case 'n':  out[0] = '\n'; *out_n = 1U; return true;
    case 'r':  out[0] = '\r'; *out_n = 1U; return true;
    case 't':  out[0] = '\t'; *out_n = 1U; return true;
    case 'u': {
        uint32_t u1 = 0U;
        if (len - *pos < 4U || !parse_hex4(s + *pos, &u1)) return false;
        *pos += 4U;
        uint32_t cp = u1;
        if (u1 >= 0xD800U && u1 <= 0xDBFFU) {
            uint32_t u2 = 0U;
            if (len - *pos < 6U || s[*pos] != '\\' || s[*pos + 1U] != 'u' ||
                !parse_hex4(s + *pos + 2U, &u2) || u2 < 0xDC00U || u2 > 0xDFFFU) {
                return false;
            }
            *pos += 6U;
            cp = 0x10000U + (((u1 - 0xD800U) << 10U) | (u2 - 0xDC00U));
        } else if (u1 >= 0xDC00U && u1 <= 0xDFFFU) {
            return false;
        }
        if (reject_nul && cp == 0U) return false;
        *out_n = encode_utf8(cp, out);
        return *out_n > 0U;
    }
    default:
        return false;
    }
}

/* Validate RFC 3629 multi-byte UTF-8 sequence starting at `p[0]` (lead >= 0x80); returns 0 on error. */
static size_t valid_utf8_len(const unsigned char *p, size_t avail) {
    unsigned char c = p[0];
    if (c >= 0xC2U && c <= 0xDFU) {
        return (avail >= 2U && (p[1] & 0xC0U) == 0x80U) ? 2U : 0U;
    }
    if (c >= 0xE0U && c <= 0xEFU) {
        if (avail < 3U || (p[1] & 0xC0U) != 0x80U || (p[2] & 0xC0U) != 0x80U) return 0U;
        if ((c == 0xE0U && p[1] < 0xA0U) || (c == 0xEDU && p[1] >= 0xA0U)) return 0U;
        return 3U;
    }
    if (c >= 0xF0U && c <= 0xF4U) {
        if (avail < 4U || (p[1] & 0xC0U) != 0x80U || (p[2] & 0xC0U) != 0x80U ||
            (p[3] & 0xC0U) != 0x80U) {
            return 0U;
        }
        if ((c == 0xF0U && p[1] < 0x90U) || (c == 0xF4U && p[1] > 0x8FU)) return 0U;
        return 4U;
    }
    return 0U;
}

static velri_status scan_number(velri_json_reader *r, size_t *out_start, size_t *out_len,
                                bool *out_is_float) {
    skip_ws(r);
    size_t start = r->pos;
    bool is_float = false;

    if (r->pos < r->len && r->data[r->pos] == '-') advance_n(r, 1U);
    if (r->pos >= r->len) return json_fail(r, VELRI_ERR_INVALID, "unexpected end of input in number");

    if (r->data[r->pos] == '0') {
        advance_n(r, 1U);
        if (r->pos < r->len && r->data[r->pos] >= '0' && r->data[r->pos] <= '9') {
            return json_fail(r, VELRI_ERR_INVALID, "leading zero in number");
        }
    } else if (r->data[r->pos] >= '1' && r->data[r->pos] <= '9') {
        while (r->pos < r->len && r->data[r->pos] >= '0' && r->data[r->pos] <= '9') advance_n(r, 1U);
    } else {
        return json_fail(r, VELRI_ERR_INVALID, "expected number");
    }

    if (r->pos < r->len && r->data[r->pos] == '.') {
        is_float = true;
        advance_n(r, 1U);
        if (r->pos >= r->len || r->data[r->pos] < '0' || r->data[r->pos] > '9') {
            return json_fail(r, VELRI_ERR_INVALID, "expected digit after decimal point");
        }
        while (r->pos < r->len && r->data[r->pos] >= '0' && r->data[r->pos] <= '9') advance_n(r, 1U);
    }

    if (r->pos < r->len && (r->data[r->pos] == 'e' || r->data[r->pos] == 'E')) {
        is_float = true;
        advance_n(r, 1U);
        if (r->pos < r->len && (r->data[r->pos] == '+' || r->data[r->pos] == '-')) advance_n(r, 1U);
        if (r->pos >= r->len || r->data[r->pos] < '0' || r->data[r->pos] > '9') {
            return json_fail(r, VELRI_ERR_INVALID, "expected digit in exponent");
        }
        while (r->pos < r->len && r->data[r->pos] >= '0' && r->data[r->pos] <= '9') advance_n(r, 1U);
    }

    if (out_start) *out_start = start;
    if (out_len) *out_len = r->pos - start;
    if (out_is_float) *out_is_float = is_float;
    return VELRI_OK;
}

static velri_status begin_container(velri_json_reader *r, velri_json_iter *it, char open_ch,
                                    const char *err_msg) {
    if (!r || !it || !r->data) return VELRI_ERR_INVALID;
    skip_ws(r);
    if (r->pos >= r->len || r->data[r->pos] != open_ch) {
        return json_fail(r, VELRI_ERR_INVALID, err_msg);
    }
    if (r->depth >= VELRI_JSON_MAX_DEPTH) {
        return json_fail(r, VELRI_ERR_OVERFLOW, "maximum JSON nesting depth exceeded");
    }
    advance_n(r, 1U);
    r->depth++;
    it->first = true;
    it->done = false;
    return VELRI_OK;
}

static velri_status next_container_item(velri_json_reader *r, velri_json_iter *it, char close_ch,
                                        bool *out_has_more) {
    if (!r || !it || !out_has_more || it->done) return VELRI_ERR_INVALID;
    skip_ws(r);
    if (r->pos >= r->len) return json_fail(r, VELRI_ERR_INVALID, "unexpected end of input in container");
    if (r->data[r->pos] == close_ch) {
        advance_n(r, 1U);
        if (r->depth > 0U) r->depth--;
        it->done = true;
        *out_has_more = false;
        return VELRI_OK;
    }
    if (!it->first) {
        if (r->data[r->pos] != ',') return json_fail(r, VELRI_ERR_INVALID, "expected ',' or closing delimiter");
        advance_n(r, 1U);
        skip_ws(r);
        if (r->pos >= r->len || r->data[r->pos] == close_ch) {
            return json_fail(r, VELRI_ERR_INVALID, "trailing comma in container");
        }
    }
    it->first = false;
    *out_has_more = true;
    return VELRI_OK;
}

/* Locale-independent ASCII decimal-to-IEEE-754 double parser (immune to LC_NUMERIC). */
static velri_status parse_ascii_f64(const char *s, size_t nlen, double *out) {
    size_t i = 0U;
    bool neg = (i < nlen && s[i] == '-');
    if (neg) i++;

    uint64_t sig = 0U;
    uint32_t sig_digits = 0U;
    int64_t exp10 = 0;
    bool in_frac = false;

    while (i < nlen) {
        char c = s[i];
        if (c == '.') {
            in_frac = true;
            i++;
            continue;
        }
        if (c < '0' || c > '9') break;
        uint32_t d = (uint32_t)(c - '0');
        i++;
        if (sig == 0U && d == 0U) {
            if (in_frac && exp10 > -1000000) exp10--;
            continue;
        }
        if (sig_digits < 19U) {
            sig = sig * 10U + (uint64_t)d;
            sig_digits++;
            if (in_frac && exp10 > -1000000) exp10--;
        } else if (!in_frac && exp10 < 1000000) {
            exp10++;
        }
    }

    if (i < nlen && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        bool eneg = (i < nlen && s[i] == '-');
        if (i < nlen && (s[i] == '+' || s[i] == '-')) i++;
        int64_t eval = 0;
        while (i < nlen && s[i] >= '0' && s[i] <= '9') {
            if (eval < 1000000) eval = eval * 10 + (int64_t)(s[i] - '0');
            i++;
        }
        exp10 += eneg ? -eval : eval;
    }

    if (sig == 0U) {
        *out = neg ? -0.0 : 0.0;
        return VELRI_OK;
    }
    int64_t sci_exp = exp10 + (int64_t)sig_digits - 1;
    if (sci_exp > 309) return VELRI_ERR_OVERFLOW;
    if (sci_exp < -326) {
        *out = neg ? -0.0 : 0.0;
        return VELRI_OK;
    }

    double v = (double)sig;
    if (exp10 > 300) {
        v = v * pow(10.0, (double)(exp10 - 300)) * 1e300;
    } else if (exp10 > 0) {
        v *= pow(10.0, (double)exp10);
    } else if (exp10 >= -22) {
        v /= pow(10.0, (double)(-exp10));
    } else if (exp10 < -300) {
        v = v * pow(10.0, (double)(exp10 + 300)) * 1e-300;
    } else {
        v *= pow(10.0, (double)exp10);
    }

    if (!isfinite(v)) return VELRI_ERR_OVERFLOW;
    *out = neg ? -v : v;
    return VELRI_OK;
}

static velri_status read_int_buf(velri_json_reader *r, bool allow_neg, char buf[64],
                                 size_t *out_len) {
    size_t start = 0U, nlen = 0U;
    bool is_float = false;
    velri_status st = scan_number(r, &start, &nlen, &is_float);
    if (st != VELRI_OK) return st;
    if (is_float || (!allow_neg && r->data[start] == '-')) {
        return json_fail(r, VELRI_ERR_INVALID,
                         allow_neg ? "expected integer" : "expected non-negative integer");
    }
    if (nlen >= 64U) return json_fail(r, VELRI_ERR_OVERFLOW, "integer literal too long");
    memcpy(buf, r->data + start, nlen);
    buf[nlen] = '\0';
    *out_len = nlen;
    return VELRI_OK;
}

/* ---- Public API ---- */

void velri_json_init(velri_json_reader *r, const char *data, size_t len) {
    if (!r) return;
    memset(r, 0, sizeof(*r));
    r->data = data;
    r->len = data ? len : 0U;
    r->line = 1U;
    r->col = 1U;
}

velri_json_type velri_json_peek(velri_json_reader *r) {
    if (!r || !r->data) return VELRI_JSON_INVALID;
    skip_ws(r);
    if (r->pos >= r->len) return VELRI_JSON_INVALID;
    char c = r->data[r->pos];
    if (c == '{') return VELRI_JSON_OBJECT;
    if (c == '[') return VELRI_JSON_ARRAY;
    if (c == '"') return VELRI_JSON_STRING;
    if (c == 't' || c == 'f') return VELRI_JSON_BOOL;
    if (c == 'n') return VELRI_JSON_NULL;
    if (c == '-' || (c >= '0' && c <= '9')) return VELRI_JSON_NUMBER;
    return VELRI_JSON_INVALID;
}

velri_status velri_json_begin_object(velri_json_reader *r, velri_json_iter *it) {
    return begin_container(r, it, '{', "expected '{'");
}

velri_status velri_json_begin_array(velri_json_reader *r, velri_json_iter *it) {
    return begin_container(r, it, '[', "expected '['");
}

velri_status velri_json_next_member(velri_json_reader *r, velri_json_iter *it,
                                    velri_json_str *out_key, bool *out_has_more) {
    if (!out_key) return VELRI_ERR_INVALID;
    velri_status st = next_container_item(r, it, '}', out_has_more);
    if (st != VELRI_OK || !*out_has_more) {
        memset(out_key, 0, sizeof(*out_key));
        return st;
    }
    st = velri_json_read_string(r, out_key);
    if (st != VELRI_OK) return st;
    skip_ws(r);
    if (r->pos >= r->len || r->data[r->pos] != ':') {
        return json_fail(r, VELRI_ERR_INVALID, "expected ':' after object key");
    }
    advance_n(r, 1U);
    skip_ws(r);
    return VELRI_OK;
}

velri_status velri_json_next_elem(velri_json_reader *r, velri_json_iter *it, bool *out_has_more) {
    return next_container_item(r, it, ']', out_has_more);
}

velri_status velri_json_read_string(velri_json_reader *r, velri_json_str *out) {
    if (!r || !r->data) return VELRI_ERR_INVALID;
    skip_ws(r);
    if (r->pos >= r->len || r->data[r->pos] != '"') {
        return json_fail(r, VELRI_ERR_INVALID, "expected string");
    }
    advance_n(r, 1U);
    size_t start = r->pos;
    bool has_escapes = false;

    while (r->pos < r->len) {
        unsigned char c = (unsigned char)r->data[r->pos];
        if (c == '"') {
            size_t slen = r->pos - start;
            advance_n(r, 1U);
            if (out) {
                out->ptr = r->data + start;
                out->len = slen;
                out->has_escapes = has_escapes;
            }
            return VELRI_OK;
        }
        if (c < 0x20U) return json_fail(r, VELRI_ERR_INVALID, "unescaped control character in string");
        if (c == '\\') {
            has_escapes = true;
            size_t before = r->pos;
            char tmp[4];
            size_t tn = 0U;
            size_t cur = before;
            if (!decode_next_unit(r->data, r->len, &cur, tmp, &tn, false)) {
                return json_fail(r, VELRI_ERR_INVALID, "invalid escape sequence in string");
            }
            advance_n(r, cur - before);
        } else if (c >= 0x80U) {
            size_t ulen = valid_utf8_len((const unsigned char *)(r->data + r->pos), r->len - r->pos);
            if (ulen == 0U) return json_fail(r, VELRI_ERR_INVALID, "invalid UTF-8 sequence");
            advance_n(r, ulen);
        } else {
            advance_n(r, 1U);
        }
    }
    return json_fail(r, VELRI_ERR_INVALID, "unterminated string");
}

velri_status velri_json_str_decode(velri_json_str s, char *dst, size_t dst_cap, size_t *out_len) {
    if (!dst || dst_cap == 0U || (!s.ptr && s.len > 0U)) return VELRI_ERR_INVALID;
    if (!s.has_escapes) {
        if (s.len >= dst_cap) return VELRI_ERR_OVERFLOW;
        if (s.len > 0U) memcpy(dst, s.ptr, s.len);
        dst[s.len] = '\0';
        if (out_len) *out_len = s.len;
        return VELRI_OK;
    }

    size_t i = 0U, out = 0U;
    while (i < s.len) {
        char unit[4];
        size_t un = 0U;
        if (!decode_next_unit(s.ptr, s.len, &i, unit, &un, false)) return VELRI_ERR_INVALID;
        if (un >= dst_cap - out) return VELRI_ERR_OVERFLOW;
        memcpy(dst + out, unit, un);
        out += un;
    }
    dst[out] = '\0';
    if (out_len) *out_len = out;
    return VELRI_OK;
}

bool velri_json_str_eq(velri_json_str s, const char *lit) {
    if (!lit || (!s.ptr && s.len > 0U)) return false;
    size_t lit_len = strlen(lit);
    if (!s.has_escapes) {
        return s.len == lit_len && (lit_len == 0U || memcmp(s.ptr, lit, lit_len) == 0);
    }

    size_t i = 0U, lpos = 0U;
    while (i < s.len) {
        char unit[4];
        size_t un = 0U;
        if (!decode_next_unit(s.ptr, s.len, &i, unit, &un, true) || un > lit_len - lpos ||
            memcmp(lit + lpos, unit, un) != 0) {
            return false;
        }
        lpos += un;
    }
    return lpos == lit_len;
}

velri_status velri_json_read_string_buf(velri_json_reader *r, char *dst, size_t dst_cap,
                                        size_t *out_len) {
    velri_json_str s;
    velri_status st = velri_json_read_string(r, &s);
    if (st != VELRI_OK) return st;
    st = velri_json_str_decode(s, dst, dst_cap, out_len);
    return (st == VELRI_OK) ? VELRI_OK : json_fail(r, st, "string exceeds destination buffer");
}

velri_status velri_json_read_i64(velri_json_reader *r, int64_t *out) {
    if (!r || !out) return VELRI_ERR_INVALID;
    char buf[64];
    size_t nlen = 0U;
    velri_status st = read_int_buf(r, true, buf, &nlen);
    if (st != VELRI_OK) return st;
    errno = 0;
    char *endp = nullptr;
    long long v = strtoll(buf, &endp, 10);
    if (errno == ERANGE || !endp || (size_t)(endp - buf) != nlen) {
        return json_fail(r, VELRI_ERR_OVERFLOW, "integer out of int64 range");
    }
    *out = (int64_t)v;
    return VELRI_OK;
}

velri_status velri_json_read_u64(velri_json_reader *r, uint64_t *out) {
    if (!r || !out) return VELRI_ERR_INVALID;
    char buf[64];
    size_t nlen = 0U;
    velri_status st = read_int_buf(r, false, buf, &nlen);
    if (st != VELRI_OK) return st;
    errno = 0;
    char *endp = nullptr;
    unsigned long long v = strtoull(buf, &endp, 10);
    if (errno == ERANGE || !endp || (size_t)(endp - buf) != nlen) {
        return json_fail(r, VELRI_ERR_OVERFLOW, "integer out of uint64 range");
    }
    *out = (uint64_t)v;
    return VELRI_OK;
}

velri_status velri_json_read_f64(velri_json_reader *r, double *out) {
    if (!r || !out) return VELRI_ERR_INVALID;
    size_t start = 0U, nlen = 0U;
    velri_status st = scan_number(r, &start, &nlen, nullptr);
    if (st != VELRI_OK) return st;
    st = parse_ascii_f64(r->data + start, nlen, out);
    return (st == VELRI_OK) ? VELRI_OK : json_fail(r, st, "number out of IEEE-754 double range");
}

velri_status velri_json_read_bool(velri_json_reader *r, bool *out) {
    if (!r || !r->data || !out) return VELRI_ERR_INVALID;
    skip_ws(r);
    if (r->len - r->pos >= 4U && memcmp(r->data + r->pos, "true", 4U) == 0) {
        advance_n(r, 4U);
        *out = true;
        return VELRI_OK;
    }
    if (r->len - r->pos >= 5U && memcmp(r->data + r->pos, "false", 5U) == 0) {
        advance_n(r, 5U);
        *out = false;
        return VELRI_OK;
    }
    return json_fail(r, VELRI_ERR_INVALID, "expected boolean");
}

velri_status velri_json_read_null(velri_json_reader *r) {
    if (!r || !r->data) return VELRI_ERR_INVALID;
    skip_ws(r);
    if (r->len - r->pos >= 4U && memcmp(r->data + r->pos, "null", 4U) == 0) {
        advance_n(r, 4U);
        return VELRI_OK;
    }
    return json_fail(r, VELRI_ERR_INVALID, "expected null");
}

velri_status velri_json_skip(velri_json_reader *r) {
    if (!r || !r->data) return VELRI_ERR_INVALID;
    typedef struct {
        bool is_object;
        velri_json_iter it;
    } skip_frame;
    skip_frame stack[VELRI_JSON_MAX_DEPTH];
    uint32_t sp = 0U;

    for (;;) {
        velri_json_type t = velri_json_peek(r);
        velri_status st = VELRI_OK;
        if (t == VELRI_JSON_OBJECT || t == VELRI_JSON_ARRAY) {
            if (sp >= VELRI_JSON_MAX_DEPTH) {
                return json_fail(r, VELRI_ERR_OVERFLOW, "maximum JSON nesting depth exceeded");
            }
            stack[sp].is_object = (t == VELRI_JSON_OBJECT);
            st = stack[sp].is_object ? velri_json_begin_object(r, &stack[sp].it)
                                     : velri_json_begin_array(r, &stack[sp].it);
            if (st != VELRI_OK) return st;
            sp++;
        } else {
            bool b = false;
            if (t == VELRI_JSON_STRING)      st = velri_json_read_string(r, nullptr);
            else if (t == VELRI_JSON_NUMBER) st = scan_number(r, nullptr, nullptr, nullptr);
            else if (t == VELRI_JSON_BOOL)   st = velri_json_read_bool(r, &b);
            else if (t == VELRI_JSON_NULL)   st = velri_json_read_null(r);
            else return json_fail(r, VELRI_ERR_INVALID, "unexpected token");
            if (st != VELRI_OK) return st;
        }

        while (sp > 0U) {
            bool more = false;
            velri_json_str k;
            st = stack[sp - 1U].is_object
                     ? velri_json_next_member(r, &stack[sp - 1U].it, &k, &more)
                     : velri_json_next_elem(r, &stack[sp - 1U].it, &more);
            if (st != VELRI_OK) return st;
            if (more) break;
            sp--;
        }
        if (sp == 0U) return VELRI_OK;
    }
}

velri_status velri_json_finish(velri_json_reader *r) {
    if (!r || !r->data) return VELRI_ERR_INVALID;
    skip_ws(r);
    if (r->pos != r->len) return json_fail(r, VELRI_ERR_INVALID, "unexpected trailing data after JSON");
    if (r->depth != 0U) return json_fail(r, VELRI_ERR_INVALID, "unclosed JSON container");
    return VELRI_OK;
}
