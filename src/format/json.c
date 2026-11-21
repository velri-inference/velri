#include "format/json.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Internal Cursor & Diagnostic Helpers ---- */

static velri_status json_fail(velri_json_reader *r, velri_status st, const char *msg) {
    if (r && r->err_msg[0] == '\0') {
        r->err_offset = r->pos;
        snprintf(r->err_msg, sizeof(r->err_msg), "JSON error at line %u, col %u: %s", r->line,
                 r->col, msg ? msg : "invalid syntax");
    }
    return st;
}

static inline void advance_byte(velri_json_reader *r) {
    if (r->pos < r->len) {
        if (r->data[r->pos] == '\n') {
            r->line++;
            r->col = 1U;
        } else {
            r->col++;
        }
        r->pos++;
    }
}

static void skip_ws(velri_json_reader *r) {
    while (r->pos < r->len) {
        char c = r->data[r->pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance_byte(r);
        } else {
            break;
        }
    }
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
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

static bool append_utf8(uint32_t cp, char *dst, size_t dst_cap, size_t *pos) {
    char tmp[4];
    size_t n = 0U;
    if (cp <= 0x7FU) {
        tmp[0] = (char)cp;
        n = 1U;
    } else if (cp <= 0x7FFU) {
        tmp[0] = (char)(0xC0U | (cp >> 6U));
        tmp[1] = (char)(0x80U | (cp & 0x3FU));
        n = 2U;
    } else if (cp <= 0xFFFFU) {
        tmp[0] = (char)(0xE0U | (cp >> 12U));
        tmp[1] = (char)(0x80U | ((cp >> 6U) & 0x3FU));
        tmp[2] = (char)(0x80U | (cp & 0x3FU));
        n = 3U;
    } else if (cp <= 0x10FFFFU) {
        tmp[0] = (char)(0xF0U | (cp >> 18U));
        tmp[1] = (char)(0x80U | ((cp >> 12U) & 0x3FU));
        tmp[2] = (char)(0x80U | ((cp >> 6U) & 0x3FU));
        tmp[3] = (char)(0x80U | (cp & 0x3FU));
        n = 4U;
    } else {
        return false;
    }
    if (*pos + n >= dst_cap) {
        return false;
    }
    memcpy(dst + *pos, tmp, n);
    *pos += n;
    return true;
}

/* Scan and strictly validate a RFC 8259 number token starting at r->pos (after skip_ws). */
static velri_status scan_number(velri_json_reader *r, size_t *out_start, size_t *out_len,
                                bool *out_is_float) {
    skip_ws(r);
    size_t start = r->pos;
    bool is_float = false;

    if (r->pos < r->len && r->data[r->pos] == '-') {
        advance_byte(r);
    }
    if (r->pos >= r->len) {
        return json_fail(r, VELRI_ERR_INVALID, "unexpected end of input in number");
    }
    if (r->data[r->pos] == '0') {
        advance_byte(r);
        if (r->pos < r->len && r->data[r->pos] >= '0' && r->data[r->pos] <= '9') {
            return json_fail(r, VELRI_ERR_INVALID, "leading zero in number");
        }
    } else if (r->data[r->pos] >= '1' && r->data[r->pos] <= '9') {
        advance_byte(r);
        while (r->pos < r->len && r->data[r->pos] >= '0' && r->data[r->pos] <= '9') {
            advance_byte(r);
        }
    } else {
        return json_fail(r, VELRI_ERR_INVALID, "expected number");
    }

    if (r->pos < r->len && r->data[r->pos] == '.') {
        is_float = true;
        advance_byte(r);
        if (r->pos >= r->len || r->data[r->pos] < '0' || r->data[r->pos] > '9') {
            return json_fail(r, VELRI_ERR_INVALID, "expected digit after decimal point");
        }
        while (r->pos < r->len && r->data[r->pos] >= '0' && r->data[r->pos] <= '9') {
            advance_byte(r);
        }
    }

    if (r->pos < r->len && (r->data[r->pos] == 'e' || r->data[r->pos] == 'E')) {
        is_float = true;
        advance_byte(r);
        if (r->pos < r->len && (r->data[r->pos] == '+' || r->data[r->pos] == '-')) {
            advance_byte(r);
        }
        if (r->pos >= r->len || r->data[r->pos] < '0' || r->data[r->pos] > '9') {
            return json_fail(r, VELRI_ERR_INVALID, "expected digit in exponent");
        }
        while (r->pos < r->len && r->data[r->pos] >= '0' && r->data[r->pos] <= '9') {
            advance_byte(r);
        }
    }

    if (out_start) *out_start = start;
    if (out_len) *out_len = r->pos - start;
    if (out_is_float) *out_is_float = is_float;
    return VELRI_OK;
}

/* ---- Public API ---- */

void velri_json_init(velri_json_reader *r, const char *data, size_t len) {
    if (!r) {
        return;
    }
    memset(r, 0, sizeof(*r));
    r->data = data;
    r->len = data ? len : 0U;
    r->line = 1U;
    r->col = 1U;
}

velri_json_type velri_json_peek(velri_json_reader *r) {
    if (!r || !r->data) {
        return VELRI_JSON_INVALID;
    }
    skip_ws(r);
    if (r->pos >= r->len) {
        return VELRI_JSON_INVALID;
    }
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
    if (!r || !it || !r->data) {
        return VELRI_ERR_INVALID;
    }
    skip_ws(r);
    if (r->pos >= r->len || r->data[r->pos] != '{') {
        return json_fail(r, VELRI_ERR_INVALID, "expected '{'");
    }
    if (r->depth >= VELRI_JSON_MAX_DEPTH) {
        return json_fail(r, VELRI_ERR_OVERFLOW, "maximum JSON nesting depth exceeded");
    }
    advance_byte(r);
    r->depth++;
    it->first = true;
    it->done = false;
    return VELRI_OK;
}

velri_status velri_json_next_member(velri_json_reader *r, velri_json_iter *it,
                                    velri_json_str *out_key, bool *out_has_more) {
    if (!r || !it || !out_key || !out_has_more || it->done) {
        return VELRI_ERR_INVALID;
    }
    skip_ws(r);
    if (r->pos >= r->len) {
        return json_fail(r, VELRI_ERR_INVALID, "unexpected end of input inside object");
    }
    if (r->data[r->pos] == '}') {
        advance_byte(r);
        if (r->depth > 0U) {
            r->depth--;
        }
        it->done = true;
        *out_has_more = false;
        memset(out_key, 0, sizeof(*out_key));
        return VELRI_OK;
    }
    if (!it->first) {
        if (r->data[r->pos] != ',') {
            return json_fail(r, VELRI_ERR_INVALID, "expected ',' or '}' in object");
        }
        advance_byte(r);
        skip_ws(r);
        if (r->pos >= r->len || r->data[r->pos] == '}') {
            return json_fail(r, VELRI_ERR_INVALID, "trailing comma in object");
        }
    }
    it->first = false;

    velri_status st = velri_json_read_string(r, out_key);
    if (st != VELRI_OK) {
        return st;
    }
    skip_ws(r);
    if (r->pos >= r->len || r->data[r->pos] != ':') {
        return json_fail(r, VELRI_ERR_INVALID, "expected ':' after object key");
    }
    advance_byte(r);
    skip_ws(r);
    *out_has_more = true;
    return VELRI_OK;
}

velri_status velri_json_begin_array(velri_json_reader *r, velri_json_iter *it) {
    if (!r || !it || !r->data) {
        return VELRI_ERR_INVALID;
    }
    skip_ws(r);
    if (r->pos >= r->len || r->data[r->pos] != '[') {
        return json_fail(r, VELRI_ERR_INVALID, "expected '['");
    }
    if (r->depth >= VELRI_JSON_MAX_DEPTH) {
        return json_fail(r, VELRI_ERR_OVERFLOW, "maximum JSON nesting depth exceeded");
    }
    advance_byte(r);
    r->depth++;
    it->first = true;
    it->done = false;
    return VELRI_OK;
}

velri_status velri_json_next_elem(velri_json_reader *r, velri_json_iter *it, bool *out_has_more) {
    if (!r || !it || !out_has_more || it->done) {
        return VELRI_ERR_INVALID;
    }
    skip_ws(r);
    if (r->pos >= r->len) {
        return json_fail(r, VELRI_ERR_INVALID, "unexpected end of input inside array");
    }
    if (r->data[r->pos] == ']') {
        advance_byte(r);
        if (r->depth > 0U) {
            r->depth--;
        }
        it->done = true;
        *out_has_more = false;
        return VELRI_OK;
    }
    if (!it->first) {
        if (r->data[r->pos] != ',') {
            return json_fail(r, VELRI_ERR_INVALID, "expected ',' or ']' in array");
        }
        advance_byte(r);
        skip_ws(r);
        if (r->pos >= r->len || r->data[r->pos] == ']') {
            return json_fail(r, VELRI_ERR_INVALID, "trailing comma in array");
        }
    }
    it->first = false;
    *out_has_more = true;
    return VELRI_OK;
}

velri_status velri_json_read_string(velri_json_reader *r, velri_json_str *out) {
    if (!r || !r->data) {
        return VELRI_ERR_INVALID;
    }
    skip_ws(r);
    if (r->pos >= r->len || r->data[r->pos] != '"') {
        return json_fail(r, VELRI_ERR_INVALID, "expected string");
    }
    advance_byte(r);
    size_t start = r->pos;
    bool has_escapes = false;

    while (r->pos < r->len) {
        unsigned char c = (unsigned char)r->data[r->pos];
        if (c == '"') {
            size_t slen = r->pos - start;
            advance_byte(r);
            if (out) {
                out->ptr = r->data + start;
                out->len = slen;
                out->has_escapes = has_escapes;
            }
            return VELRI_OK;
        }
        if (c < 0x20U) {
            return json_fail(r, VELRI_ERR_INVALID, "unescaped control character in string");
        }
        if (c == '\\') {
            has_escapes = true;
            advance_byte(r);
            if (r->pos >= r->len) {
                return json_fail(r, VELRI_ERR_INVALID, "unterminated escape in string");
            }
            char esc = r->data[r->pos];
            switch (esc) {
            case '"':
            case '\\':
            case '/':
            case 'b':
            case 'f':
            case 'n':
            case 'r':
            case 't':
                advance_byte(r);
                break;
            case 'u': {
                advance_byte(r);
                if (r->len - r->pos < 4U) {
                    return json_fail(r, VELRI_ERR_INVALID, "truncated \\u escape");
                }
                uint32_t u1 = 0U;
                if (!parse_hex4(r->data + r->pos, &u1)) {
                    return json_fail(r, VELRI_ERR_INVALID, "invalid hex in \\u escape");
                }
                for (int i = 0; i < 4; ++i) advance_byte(r);
                if (u1 >= 0xD800U && u1 <= 0xDBFFU) {
                    if (r->len - r->pos < 6U || r->data[r->pos] != '\\' ||
                        r->data[r->pos + 1U] != 'u') {
                        return json_fail(r, VELRI_ERR_INVALID, "missing low surrogate after high");
                    }
                    uint32_t u2 = 0U;
                    if (!parse_hex4(r->data + r->pos + 2U, &u2) || u2 < 0xDC00U || u2 > 0xDFFFU) {
                        return json_fail(r, VELRI_ERR_INVALID, "invalid low surrogate");
                    }
                    for (int i = 0; i < 6; ++i) advance_byte(r);
                } else if (u1 >= 0xDC00U && u1 <= 0xDFFFU) {
                    return json_fail(r, VELRI_ERR_INVALID, "unexpected lone low surrogate");
                }
                break;
            }
            default:
                return json_fail(r, VELRI_ERR_INVALID, "invalid escape sequence");
            }
        } else if (c >= 0x80U) {
            /* Strict RFC 8259 / RFC 3629 UTF-8 sequence validation. */
            size_t avail = r->len - r->pos;
            if (c >= 0xC2U && c <= 0xDFU) {
                if (avail < 2U || ((unsigned char)r->data[r->pos + 1U] & 0xC0U) != 0x80U) {
                    return json_fail(r, VELRI_ERR_INVALID, "invalid 2-byte UTF-8 sequence");
                }
                advance_byte(r);
                advance_byte(r);
            } else if (c >= 0xE0U && c <= 0xEFU) {
                if (avail < 3U) {
                    return json_fail(r, VELRI_ERR_INVALID, "truncated 3-byte UTF-8 sequence");
                }
                unsigned char c1 = (unsigned char)r->data[r->pos + 1U];
                unsigned char c2 = (unsigned char)r->data[r->pos + 2U];
                if ((c1 & 0xC0U) != 0x80U || (c2 & 0xC0U) != 0x80U ||
                    (c == 0xE0U && c1 < 0xA0U) || (c == 0xEDU && c1 >= 0xA0U)) {
                    return json_fail(r, VELRI_ERR_INVALID, "invalid 3-byte UTF-8 sequence");
                }
                advance_byte(r);
                advance_byte(r);
                advance_byte(r);
            } else if (c >= 0xF0U && c <= 0xF4U) {
                if (avail < 4U) {
                    return json_fail(r, VELRI_ERR_INVALID, "truncated 4-byte UTF-8 sequence");
                }
                unsigned char c1 = (unsigned char)r->data[r->pos + 1U];
                unsigned char c2 = (unsigned char)r->data[r->pos + 2U];
                unsigned char c3 = (unsigned char)r->data[r->pos + 3U];
                if ((c1 & 0xC0U) != 0x80U || (c2 & 0xC0U) != 0x80U || (c3 & 0xC0U) != 0x80U ||
                    (c == 0xF0U && c1 < 0x90U) || (c == 0xF4U && c1 > 0x8FU)) {
                    return json_fail(r, VELRI_ERR_INVALID, "invalid 4-byte UTF-8 sequence");
                }
                advance_byte(r);
                advance_byte(r);
                advance_byte(r);
                advance_byte(r);
            } else {
                return json_fail(r, VELRI_ERR_INVALID, "invalid UTF-8 lead byte");
            }
        } else {
            advance_byte(r);
        }
    }
    return json_fail(r, VELRI_ERR_INVALID, "unterminated string");
}

velri_status velri_json_str_decode(velri_json_str s, char *dst, size_t dst_cap, size_t *out_len) {
    if (!dst || dst_cap == 0U || (!s.ptr && s.len > 0U)) {
        return VELRI_ERR_INVALID;
    }
    if (!s.has_escapes) {
        if (s.len >= dst_cap) {
            return VELRI_ERR_OVERFLOW;
        }
        if (s.len > 0U) {
            memcpy(dst, s.ptr, s.len);
        }
        dst[s.len] = '\0';
        if (out_len) *out_len = s.len;
        return VELRI_OK;
    }

    size_t i = 0U, out = 0U;
    while (i < s.len) {
        char c = s.ptr[i++];
        if (c != '\\') {
            if (out + 1U >= dst_cap) return VELRI_ERR_OVERFLOW;
            dst[out++] = c;
            continue;
        }
        if (i >= s.len) return VELRI_ERR_INVALID;
        char esc = s.ptr[i++];
        char plain = 0;
        switch (esc) {
        case '"':  plain = '"'; break;
        case '\\': plain = '\\'; break;
        case '/':  plain = '/'; break;
        case 'b':  plain = '\b'; break;
        case 'f':  plain = '\f'; break;
        case 'n':  plain = '\n'; break;
        case 'r':  plain = '\r'; break;
        case 't':  plain = '\t'; break;
        case 'u': {
            if (s.len - i < 4U) return VELRI_ERR_INVALID;
            uint32_t u1 = 0U;
            if (!parse_hex4(s.ptr + i, &u1)) return VELRI_ERR_INVALID;
            i += 4U;
            uint32_t cp = u1;
            if (u1 >= 0xD800U && u1 <= 0xDBFFU) {
                if (s.len - i < 6U || s.ptr[i] != '\\' || s.ptr[i + 1U] != 'u') {
                    return VELRI_ERR_INVALID;
                }
                uint32_t u2 = 0U;
                if (!parse_hex4(s.ptr + i + 2U, &u2) || u2 < 0xDC00U || u2 > 0xDFFFU) {
                    return VELRI_ERR_INVALID;
                }
                i += 6U;
                cp = 0x10000U + (((u1 - 0xD800U) << 10U) | (u2 - 0xDC00U));
            } else if (u1 >= 0xDC00U && u1 <= 0xDFFFU) {
                return VELRI_ERR_INVALID;
            }
            if (!append_utf8(cp, dst, dst_cap, &out)) return VELRI_ERR_OVERFLOW;
            continue;
        }
        default:
            return VELRI_ERR_INVALID;
        }
        if (out + 1U >= dst_cap) return VELRI_ERR_OVERFLOW;
        dst[out++] = plain;
    }
    dst[out] = '\0';
    if (out_len) *out_len = out;
    return VELRI_OK;
}

bool velri_json_str_eq(velri_json_str s, const char *lit) {
    if (!lit || (!s.ptr && s.len > 0U)) {
        return false;
    }
    size_t lit_len = strlen(lit);
    if (!s.has_escapes) {
        return s.len == lit_len && (lit_len == 0U || memcmp(s.ptr, lit, lit_len) == 0);
    }

    /* Streaming escape-aware comparison (0 B heap allocs, no fixed length cap). */
    size_t i = 0U, lpos = 0U;
    while (i < s.len) {
        char chunk[4];
        size_t clen = 0U;
        char c = s.ptr[i++];
        if (c != '\\') {
            chunk[0] = c;
            clen = 1U;
        } else {
            if (i >= s.len) return false;
            char esc = s.ptr[i++];
            switch (esc) {
            case '"':  chunk[0] = '"'; clen = 1U; break;
            case '\\': chunk[0] = '\\'; clen = 1U; break;
            case '/':  chunk[0] = '/'; clen = 1U; break;
            case 'b':  chunk[0] = '\b'; clen = 1U; break;
            case 'f':  chunk[0] = '\f'; clen = 1U; break;
            case 'n':  chunk[0] = '\n'; clen = 1U; break;
            case 'r':  chunk[0] = '\r'; clen = 1U; break;
            case 't':  chunk[0] = '\t'; clen = 1U; break;
            case 'u': {
                if (s.len - i < 4U) return false;
                uint32_t u1 = 0U;
                if (!parse_hex4(s.ptr + i, &u1)) return false;
                i += 4U;
                uint32_t cp = u1;
                if (u1 >= 0xD800U && u1 <= 0xDBFFU) {
                    if (s.len - i < 6U || s.ptr[i] != '\\' || s.ptr[i + 1U] != 'u') return false;
                    uint32_t u2 = 0U;
                    if (!parse_hex4(s.ptr + i + 2U, &u2) || u2 < 0xDC00U || u2 > 0xDFFFU) {
                        return false;
                    }
                    i += 6U;
                    cp = 0x10000U + (((u1 - 0xD800U) << 10U) | (u2 - 0xDC00U));
                } else if (u1 >= 0xDC00U && u1 <= 0xDFFFU) {
                    return false;
                }
                if (cp == 0U) return false; /* NUL cannot match a C-string literal */
                char tmp[5];
                size_t tlen = 0U;
                if (!append_utf8(cp, tmp, sizeof(tmp), &tlen)) return false;
                memcpy(chunk, tmp, tlen);
                clen = tlen;
                break;
            }
            default:
                return false;
            }
        }
        if (clen > lit_len - lpos || memcmp(lit + lpos, chunk, clen) != 0) {
            return false;
        }
        lpos += clen;
    }
    return lpos == lit_len;
}

velri_status velri_json_read_string_buf(velri_json_reader *r, char *dst, size_t dst_cap,
                                        size_t *out_len) {
    velri_json_str s;
    velri_status st = velri_json_read_string(r, &s);
    if (st != VELRI_OK) {
        return st;
    }
    st = velri_json_str_decode(s, dst, dst_cap, out_len);
    if (st != VELRI_OK) {
        return json_fail(r, st, "string exceeds destination buffer");
    }
    return VELRI_OK;
}

velri_status velri_json_read_i64(velri_json_reader *r, int64_t *out) {
    if (!r || !out) {
        return VELRI_ERR_INVALID;
    }
    size_t start = 0U, nlen = 0U;
    bool is_float = false;
    velri_status st = scan_number(r, &start, &nlen, &is_float);
    if (st != VELRI_OK) {
        return st;
    }
    if (is_float) {
        return json_fail(r, VELRI_ERR_INVALID, "expected integer, got floating-point number");
    }
    if (nlen >= 64U) {
        return json_fail(r, VELRI_ERR_OVERFLOW, "integer literal too long");
    }
    char buf[64];
    memcpy(buf, r->data + start, nlen);
    buf[nlen] = '\0';
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
    if (!r || !out) {
        return VELRI_ERR_INVALID;
    }
    size_t start = 0U, nlen = 0U;
    bool is_float = false;
    velri_status st = scan_number(r, &start, &nlen, &is_float);
    if (st != VELRI_OK) {
        return st;
    }
    if (is_float || r->data[start] == '-') {
        return json_fail(r, VELRI_ERR_INVALID, "expected non-negative integer");
    }
    if (nlen >= 64U) {
        return json_fail(r, VELRI_ERR_OVERFLOW, "unsigned integer literal too long");
    }
    char buf[64];
    memcpy(buf, r->data + start, nlen);
    buf[nlen] = '\0';
    errno = 0;
    char *endp = nullptr;
    unsigned long long v = strtoull(buf, &endp, 10);
    if (errno == ERANGE || !endp || (size_t)(endp - buf) != nlen) {
        return json_fail(r, VELRI_ERR_OVERFLOW, "integer out of uint64 range");
    }
    *out = (uint64_t)v;
    return VELRI_OK;
}

/* Locale-independent ASCII decimal-to-IEEE-754 double parser (immune to LC_NUMERIC,
 * preserves IEEE-754 subnormals down to DBL_TRUE_MIN and underflow-to-zero cleanly). */
static velri_status parse_ascii_f64(const char *s, size_t nlen, double *out) {
    size_t i = 0U;
    bool neg = false;
    if (i < nlen && s[i] == '-') {
        neg = true;
        i++;
    }

    uint64_t sig = 0U;
    uint32_t sig_digits = 0U;
    int64_t exp10 = 0;

    while (i < nlen && s[i] >= '0' && s[i] <= '9') {
        uint32_t d = (uint32_t)(s[i++] - '0');
        if (sig == 0U && d == 0U) {
            continue;
        }
        if (sig_digits < 19U) {
            sig = sig * 10U + (uint64_t)d;
            sig_digits++;
        } else if (exp10 < 1000000) {
            exp10++;
        }
    }

    if (i < nlen && s[i] == '.') {
        i++;
        while (i < nlen && s[i] >= '0' && s[i] <= '9') {
            uint32_t d = (uint32_t)(s[i++] - '0');
            if (sig == 0U && d == 0U) {
                if (exp10 > -1000000) {
                    exp10--;
                }
                continue;
            }
            if (sig_digits < 19U) {
                sig = sig * 10U + (uint64_t)d;
                sig_digits++;
                if (exp10 > -1000000) {
                    exp10--;
                }
            }
        }
    }

    if (i < nlen && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        bool eneg = false;
        if (i < nlen && (s[i] == '+' || s[i] == '-')) {
            eneg = (s[i++] == '-');
        }
        int64_t eval = 0;
        while (i < nlen && s[i] >= '0' && s[i] <= '9') {
            if (eval < 1000000) {
                eval = eval * 10 + (int64_t)(s[i] - '0');
            }
            i++;
        }
        exp10 += eneg ? -eval : eval;
    }

    if (sig == 0U) {
        *out = neg ? -0.0 : 0.0;
        return VELRI_OK;
    }

    int64_t sci_exp = exp10 + (int64_t)sig_digits - 1;
    if (sci_exp > 309) {
        return VELRI_ERR_OVERFLOW;
    }
    if (sci_exp < -326) {
        *out = neg ? -0.0 : 0.0;
        return VELRI_OK;
    }

    double v = (double)sig;
    if (exp10 > 0) {
        if (exp10 > 300) {
            v *= pow(10.0, (double)(exp10 - 300));
            v *= 1e300;
        } else {
            v *= pow(10.0, (double)exp10);
        }
    } else if (exp10 < 0) {
        if (exp10 >= -22) {
            v /= pow(10.0, (double)(-exp10));
        } else if (exp10 < -300) {
            v *= pow(10.0, (double)(exp10 + 300));
            v *= 1e-300;
        } else {
            v *= pow(10.0, (double)exp10);
        }
    }

    if (!isfinite(v)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out = neg ? -v : v;
    return VELRI_OK;
}

velri_status velri_json_read_f64(velri_json_reader *r, double *out) {
    if (!r || !out) {
        return VELRI_ERR_INVALID;
    }
    size_t start = 0U, nlen = 0U;
    velri_status st = scan_number(r, &start, &nlen, nullptr);
    if (st != VELRI_OK) {
        return st;
    }
    st = parse_ascii_f64(r->data + start, nlen, out);
    if (st != VELRI_OK) {
        return json_fail(r, st, "number out of IEEE-754 double range");
    }
    return VELRI_OK;
}

velri_status velri_json_read_bool(velri_json_reader *r, bool *out) {
    if (!r || !r->data || !out) {
        return VELRI_ERR_INVALID;
    }
    skip_ws(r);
    if (r->len - r->pos >= 4U && memcmp(r->data + r->pos, "true", 4U) == 0) {
        for (int i = 0; i < 4; ++i) advance_byte(r);
        *out = true;
        return VELRI_OK;
    }
    if (r->len - r->pos >= 5U && memcmp(r->data + r->pos, "false", 5U) == 0) {
        for (int i = 0; i < 5; ++i) advance_byte(r);
        *out = false;
        return VELRI_OK;
    }
    return json_fail(r, VELRI_ERR_INVALID, "expected boolean");
}

velri_status velri_json_read_null(velri_json_reader *r) {
    if (!r || !r->data) {
        return VELRI_ERR_INVALID;
    }
    skip_ws(r);
    if (r->len - r->pos >= 4U && memcmp(r->data + r->pos, "null", 4U) == 0) {
        for (int i = 0; i < 4; ++i) advance_byte(r);
        return VELRI_OK;
    }
    return json_fail(r, VELRI_ERR_INVALID, "expected null");
}

velri_status velri_json_skip(velri_json_reader *r) {
    if (!r || !r->data) {
        return VELRI_ERR_INVALID;
    }
    typedef struct {
        bool is_object;
        velri_json_iter it;
    } skip_frame;

    skip_frame stack[VELRI_JSON_MAX_DEPTH];
    uint32_t sp = 0U;

    for (;;) {
        velri_json_type t = velri_json_peek(r);
        velri_status st = VELRI_OK;
        switch (t) {
        case VELRI_JSON_OBJECT: {
            if (sp >= VELRI_JSON_MAX_DEPTH) {
                return json_fail(r, VELRI_ERR_OVERFLOW, "maximum JSON nesting depth exceeded");
            }
            stack[sp].is_object = true;
            st = velri_json_begin_object(r, &stack[sp].it);
            if (st != VELRI_OK) return st;
            sp++;
            goto next_in_container;
        }
        case VELRI_JSON_ARRAY: {
            if (sp >= VELRI_JSON_MAX_DEPTH) {
                return json_fail(r, VELRI_ERR_OVERFLOW, "maximum JSON nesting depth exceeded");
            }
            stack[sp].is_object = false;
            st = velri_json_begin_array(r, &stack[sp].it);
            if (st != VELRI_OK) return st;
            sp++;
            goto next_in_container;
        }
        case VELRI_JSON_STRING:
            st = velri_json_read_string(r, nullptr);
            break;
        case VELRI_JSON_NUMBER:
            st = scan_number(r, nullptr, nullptr, nullptr);
            break;
        case VELRI_JSON_BOOL: {
            bool dummy = false;
            st = velri_json_read_bool(r, &dummy);
            break;
        }
        case VELRI_JSON_NULL:
            st = velri_json_read_null(r);
            break;
        default:
            return json_fail(r, VELRI_ERR_INVALID, "expected JSON value");
        }
        if (st != VELRI_OK) {
            return st;
        }

    next_in_container:
        while (sp > 0U) {
            skip_frame *f = &stack[sp - 1U];
            bool has_more = false;
            if (f->is_object) {
                velri_json_str key;
                st = velri_json_next_member(r, &f->it, &key, &has_more);
            } else {
                st = velri_json_next_elem(r, &f->it, &has_more);
            }
            if (st != VELRI_OK) {
                return st;
            }
            if (has_more) {
                break; /* Parse the member value / array element on the next outer iteration. */
            }
            sp--;
        }
        if (sp == 0U) {
            return VELRI_OK;
        }
    }
}

velri_status velri_json_finish(velri_json_reader *r) {
    if (!r) {
        return VELRI_ERR_INVALID;
    }
    skip_ws(r);
    if (r->depth != 0U) {
        return json_fail(r, VELRI_ERR_INVALID, "unclosed JSON container at end of input");
    }
    if (r->pos != r->len) {
        return json_fail(r, VELRI_ERR_INVALID, "unexpected trailing content after JSON root");
    }
    return VELRI_OK;
}
