#include "config/placement.h"
#include "format/json.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Diagnostic & Allocation Helpers ---- */

static void set_errf(char *err_msg, size_t err_cap, const char *fmt, ...) {
    if (!err_msg || err_cap == 0U || err_msg[0] != '\0') {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err_msg, err_cap, fmt, ap);
    va_end(ap);
}

/* Decode a JSON string into a C-string buffer, rejecting embedded NUL bytes (`\u0000`). */
static velri_status decode_cstr_buf(velri_json_str s, char *dst, size_t dst_cap, const char *path,
                                    char *err_msg, size_t err_cap) {
    size_t len = 0U;
    velri_status st = velri_json_str_decode(s, dst, dst_cap, &len);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: string exceeds maximum length", path);
        return st;
    }
    if (memchr(dst, '\0', len) != nullptr) {
        set_errf(err_msg, err_cap, "%s: embedded NUL character not allowed", path);
        return VELRI_ERR_INVALID;
    }
    return VELRI_OK;
}

static velri_status read_cstr_buf(velri_json_reader *r, char *dst, size_t dst_cap, const char *path,
                                  char *err_msg, size_t err_cap) {
    velri_json_str s;
    velri_status st = velri_json_read_string(r, &s);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", path, r->err_msg);
        return st;
    }
    return decode_cstr_buf(s, dst, dst_cap, path, err_msg, err_cap);
}

/* Always writes a safe printable preview (truncated if > 63 bytes) for unknown-key diagnostics. */
static void format_key_preview(velri_json_str key, char out[64]) {
    size_t len = 0U;
    if (velri_json_str_decode(key, out, 64U, &len) == VELRI_OK &&
        memchr(out, '\0', len) == nullptr) {
        return;
    }
    size_t n = (key.len < 60U) ? key.len : 60U;
    for (size_t i = 0U; i < n; ++i) {
        unsigned char c = (unsigned char)key.ptr[i];
        out[i] = (c >= 0x20U && c < 0x7FU) ? (char)c : '?';
    }
    if (key.len > n) {
        out[n++] = '.';
        out[n++] = '.';
        out[n++] = '.';
    }
    out[n] = '\0';
}

static bool grow_array(void *old_ptr, void **out_ptr, uint32_t *cap, uint32_t count,
                       size_t elem_size) {
    if (count < *cap) {
        *out_ptr = old_ptr;
        return true;
    }
    if (count == UINT32_MAX) {
        return false;
    }
    uint32_t new_cap = (*cap == 0U) ? 8U : ((*cap <= UINT32_MAX / 2U) ? (*cap * 2U) : UINT32_MAX);
    if (new_cap < count + 1U) {
        new_cap = count + 1U;
    }
    if ((size_t)new_cap > SIZE_MAX / elem_size) {
        return false;
    }
    void *np = realloc(old_ptr, (size_t)new_cap * elem_size);
    if (!np) {
        return false;
    }
    *out_ptr = np;
    *cap = new_cap;
    return true;
}

static velri_status read_u32_field(velri_json_reader *r, uint32_t *out, const char *path,
                                   char *err_msg, size_t err_cap) {
    uint64_t v = 0U;
    velri_status st = velri_json_read_u64(r, &v);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", path, r->err_msg);
        return st;
    }
    if (v > (uint64_t)UINT32_MAX) {
        set_errf(err_msg, err_cap, "%s: value exceeds uint32 range", path);
        return VELRI_ERR_OVERFLOW;
    }
    *out = (uint32_t)v;
    return VELRI_OK;
}

/* ---- Device Spec Parsing & Formatting ---- */

velri_status velri_device_spec_parse(const char *text, bool allow_store_only,
                                     velri_device_spec *out_spec, char *err_msg, size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!text || text[0] == '\0' || !out_spec) {
        set_errf(err_msg, err_cap, "empty device specification");
        return VELRI_ERR_INVALID;
    }
    memset(out_spec, 0, sizeof(*out_spec));

    if (strcmp(text, "disk") == 0) {
        if (!allow_store_only) {
            set_errf(err_msg, err_cap, "'disk' is store-only and cannot be used for compute");
            return VELRI_ERR_INVALID;
        }
        out_spec->kind = VELRI_LOC_DISK;
        return VELRI_OK;
    }

    const char *colon = strchr(text, ':');
    if (!colon || colon == text || colon[1] == '\0' || strchr(colon + 1, ':') != nullptr) {
        set_errf(err_msg, err_cap,
                 "invalid device spec '%s': expected '<backend>:<index>', 'ram:0', or 'disk'",
                 text);
        return VELRI_ERR_INVALID;
    }

    size_t blen = (size_t)(colon - text);
    if (blen >= VELRI_CFG_BACKEND_NAME_MAX) {
        set_errf(err_msg, err_cap, "backend name too long in '%s'", text);
        return VELRI_ERR_OVERFLOW;
    }
    char bname[VELRI_CFG_BACKEND_NAME_MAX];
    for (size_t i = 0U; i < blen; ++i) {
        char c = text[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) {
            set_errf(err_msg, err_cap, "invalid character in backend name '%s'", text);
            return VELRI_ERR_INVALID;
        }
        bname[i] = c;
    }
    bname[blen] = '\0';

    /* Parse non-negative decimal device index without leading junk. */
    const char *idx_s = colon + 1;
    uint64_t idx = 0U;
    if (idx_s[0] == '0' && idx_s[1] != '\0') {
        set_errf(err_msg, err_cap, "invalid device index in '%s' (leading zero)", text);
        return VELRI_ERR_INVALID;
    }
    for (const char *p = idx_s; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            set_errf(err_msg, err_cap, "invalid device index in '%s'", text);
            return VELRI_ERR_INVALID;
        }
        idx = idx * 10U + (uint64_t)(*p - '0');
        if (idx > (uint64_t)UINT32_MAX) {
            set_errf(err_msg, err_cap, "device index overflow in '%s'", text);
            return VELRI_ERR_OVERFLOW;
        }
    }

    /* Explicit migration diagnostics for legacy prototype device names. */
    if (strcmp(bname, "cuda") == 0) {
        set_errf(err_msg, err_cap,
                 "legacy device name '%s' is not supported; migrate to 'vulkan:%u' "
                 "(or a registered compute backend '<backend>:<index>')",
                 text, (uint32_t)idx);
        return VELRI_ERR_INVALID;
    }
    if (strcmp(bname, "vram") == 0) {
        set_errf(err_msg, err_cap,
                 "legacy device name '%s' is not supported; migrate to 'vulkan:%u'", text,
                 (uint32_t)idx);
        return VELRI_ERR_INVALID;
    }
    if (strcmp(bname, "device") == 0) {
        set_errf(err_msg, err_cap,
                 "ambiguous device name '%s' is disallowed; specify explicit '<backend>:%u' "
                 "(e.g. 'vulkan:%u' or 'cpu:%u')",
                 text, (uint32_t)idx, (uint32_t)idx, (uint32_t)idx);
        return VELRI_ERR_INVALID;
    }

    if (strcmp(bname, "disk") == 0) {
        set_errf(err_msg, err_cap, "use 'disk' without a device index (got '%s')", text);
        return VELRI_ERR_INVALID;
    }

    if (strcmp(bname, "ram") == 0) {
        if (idx != 0U) {
            set_errf(err_msg, err_cap, "only 'ram:0' is supported for host RAM storage (got '%s')",
                     text);
            return VELRI_ERR_INVALID;
        }
        if (!allow_store_only) {
            set_errf(err_msg, err_cap,
                     "'ram:0' is store-only; use 'cpu:0' (or another backend) for compute");
            return VELRI_ERR_INVALID;
        }
        out_spec->kind = VELRI_LOC_RAM;
        out_spec->local_index = 0U;
        return VELRI_OK;
    }

    out_spec->kind = VELRI_LOC_BACKEND;
    memcpy(out_spec->backend_name, bname, blen + 1U);
    out_spec->local_index = (uint32_t)idx;
    return VELRI_OK;
}

void velri_device_spec_format(const velri_device_spec *spec, char *out, size_t out_cap) {
    if (!out || out_cap == 0U) {
        return;
    }
    if (!spec || spec->kind == VELRI_LOC_UNSET) {
        snprintf(out, out_cap, "<unset>");
    } else if (spec->kind == VELRI_LOC_RAM) {
        snprintf(out, out_cap, "ram:0");
    } else if (spec->kind == VELRI_LOC_DISK) {
        snprintf(out, out_cap, "disk");
    } else {
        snprintf(out, out_cap, "%s:%u", spec->backend_name, spec->local_index);
    }
}

/* ---- Layer Selector Parsing & Overlap Validation ---- */

static bool parse_u32_slice(const char *s, size_t len, uint32_t *out) {
    if (len == 0U || (len > 1U && s[0] == '0')) {
        return false;
    }
    uint64_t v = 0U;
    for (size_t i = 0U; i < len; ++i) {
        if (s[i] < '0' || s[i] > '9') {
            return false;
        }
        v = v * 10U + (uint64_t)(s[i] - '0');
        if (v > 65535U) {
            return false;
        }
    }
    *out = (uint32_t)v;
    return true;
}

static bool ranges_overlap(velri_layer_range a, velri_layer_range b, uint32_t *out_layer) {
    uint32_t lo = (a.start > b.start) ? a.start : b.start;
    uint32_t hi = (a.end < b.end) ? a.end : b.end;
    if (lo <= hi) {
        if (out_layer) *out_layer = lo;
        return true;
    }
    return false;
}

static velri_status parse_layer_selector(const char *key, velri_engine_config *cfg,
                                         int32_t *out_sel_idx, char *err_msg, size_t err_cap) {
    size_t klen = strlen(key);
    if (klen == 0U || klen >= sizeof(cfg->selectors[0].raw_key)) {
        set_errf(err_msg, err_cap, "invalid layer selector '%s'", key);
        return VELRI_ERR_INVALID;
    }

    velri_layer_range *ranges = nullptr;
    uint32_t num_ranges = 0U;
    uint32_t cap_ranges = 0U;

    size_t pos = 0U;
    while (pos < klen) {
        size_t seg_start = pos;
        while (pos < klen && key[pos] != ',') {
            pos++;
        }
        size_t seg_len = pos - seg_start;
        if (seg_len == 0U) {
            free(ranges);
            set_errf(err_msg, err_cap, "empty segment in layer selector '%s'", key);
            return VELRI_ERR_INVALID;
        }

        const char *seg = key + seg_start;
        const char *dash = (const char *)memchr(seg, '-', seg_len);
        uint32_t lo = 0U, hi = 0U;
        if (!dash) {
            if (!parse_u32_slice(seg, seg_len, &lo)) {
                free(ranges);
                set_errf(err_msg, err_cap, "invalid layer index in selector '%s'", key);
                return VELRI_ERR_INVALID;
            }
            hi = lo;
        } else {
            size_t left_len = (size_t)(dash - seg);
            size_t right_len = seg_len - left_len - 1U;
            if (memchr(dash + 1, '-', right_len) != nullptr ||
                !parse_u32_slice(seg, left_len, &lo) ||
                !parse_u32_slice(dash + 1, right_len, &hi) || lo > hi) {
                free(ranges);
                set_errf(err_msg, err_cap, "invalid layer range in selector '%s'", key);
                return VELRI_ERR_INVALID;
            }
        }

        velri_layer_range nr = {.start = lo, .end = hi};
        /* Check intra-selector overlap. */
        for (uint32_t i = 0U; i < num_ranges; ++i) {
            uint32_t dup = 0U;
            if (ranges_overlap(ranges[i], nr, &dup)) {
                free(ranges);
                set_errf(err_msg, err_cap, "layer %u duplicated inside selector '%s'", dup, key);
                return VELRI_ERR_INVALID;
            }
        }
        /* Check inter-selector overlap against all previously parsed selectors. */
        for (uint32_t s = 0U; s < cfg->num_selectors; ++s) {
            const velri_layer_selector *other = &cfg->selectors[s];
            for (uint32_t j = 0U; j < other->num_ranges; ++j) {
                uint32_t dup = 0U;
                if (ranges_overlap(other->ranges[j], nr, &dup)) {
                    free(ranges);
                    set_errf(err_msg, err_cap,
                             "overlapping layer selectors '%s' and '%s' at layer %u",
                             other->raw_key, key, dup);
                    return VELRI_ERR_INVALID;
                }
            }
        }

        void *np = nullptr;
        if (!grow_array(ranges, &np, &cap_ranges, num_ranges, sizeof(*ranges))) {
            free(ranges);
            return VELRI_ERR_NOMEM;
        }
        ranges = (velri_layer_range *)np;
        ranges[num_ranges++] = nr;

        if (pos < klen) {
            pos++; /* consume ',' */
            if (pos == klen) {
                free(ranges);
                set_errf(err_msg, err_cap, "trailing comma in layer selector '%s'", key);
                return VELRI_ERR_INVALID;
            }
        }
    }

    void *np = nullptr;
    if (!grow_array(cfg->selectors, &np, &cfg->cap_selectors, cfg->num_selectors,
                    sizeof(*cfg->selectors))) {
        free(ranges);
        return VELRI_ERR_NOMEM;
    }
    cfg->selectors = (velri_layer_selector *)np;
    velri_layer_selector *sel = &cfg->selectors[cfg->num_selectors];
    memset(sel, 0, sizeof(*sel));
    memcpy(sel->raw_key, key, klen + 1U);
    sel->ranges = ranges;
    sel->num_ranges = num_ranges;
    *out_sel_idx = (int32_t)cfg->num_selectors;
    cfg->num_selectors++;
    return VELRI_OK;
}

static bool selector_matches_layer(const velri_layer_selector *sel, uint32_t layer) {
    for (uint32_t i = 0U; i < sel->num_ranges; ++i) {
        if (layer >= sel->ranges[i].start && layer <= sel->ranges[i].end) {
            return true;
        }
    }
    return false;
}

/* ---- Rule Storage Helpers ---- */

static velri_status append_rule(velri_engine_config *cfg, const char *scope_path,
                                const char *display_rule, int32_t selector_idx,
                                uint32_t *out_idx) {
    void *np = nullptr;
    if (!grow_array(cfg->rules, &np, &cfg->cap_rules, cfg->num_rules, sizeof(*cfg->rules))) {
        return VELRI_ERR_NOMEM;
    }
    cfg->rules = (velri_placement_rule *)np;
    velri_placement_rule *rule = &cfg->rules[cfg->num_rules];
    memset(rule, 0, sizeof(*rule));
    snprintf(rule->scope_path, sizeof(rule->scope_path), "%s", scope_path);
    snprintf(rule->display_rule, sizeof(rule->display_rule), "%s", display_rule);
    rule->selector_idx = selector_idx;
    *out_idx = cfg->num_rules;
    cfg->num_rules++;
    return VELRI_OK;
}

static velri_status read_placement_field(velri_json_reader *r, const char *field_name,
                                         bool allow_store_only, velri_device_spec *target,
                                         const char *display_rule, char *err_msg, size_t err_cap) {
    if (target->kind != VELRI_LOC_UNSET) {
        set_errf(err_msg, err_cap, "%s: duplicate '%s' key", display_rule, field_name);
        return VELRI_ERR_INVALID;
    }
    char path[VELRI_CFG_RULE_LABEL_MAX + 32];
    snprintf(path, sizeof(path), "%s.%s", display_rule, field_name);
    char val[96];
    velri_status st = read_cstr_buf(r, val, sizeof(val), path, err_msg, err_cap);
    if (st != VELRI_OK) {
        return st;
    }
    char spec_err[VELRI_CFG_ERR_MAX] = {0};
    st = velri_device_spec_parse(val, allow_store_only, target, spec_err, sizeof(spec_err));
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", path, spec_err);
        return st;
    }
    return VELRI_OK;
}

/* ---- Structured Subtree Parsers ---- */

/* Parse a leaf placement object allowing only `store`, `compute`, `prefill-compute`. */
static velri_status parse_leaf_placement(velri_json_reader *r, velri_engine_config *cfg,
                                         const char *scope_path, const char *display_rule,
                                         int32_t selector_idx, char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", display_rule, r->err_msg);
        return st;
    }

    uint32_t rule_idx = 0U;
    st = append_rule(cfg, scope_path, display_rule, selector_idx, &rule_idx);
    if (st != VELRI_OK) {
        return st;
    }

    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        velri_placement_rule *rule = &cfg->rules[rule_idx];
        if (velri_json_str_eq(key, "store")) {
            st = read_placement_field(r, "store", true, &rule->store, display_rule, err_msg,
                                      err_cap);
        } else if (velri_json_str_eq(key, "compute")) {
            st = read_placement_field(r, "compute", false, &rule->compute, display_rule, err_msg,
                                      err_cap);
        } else if (velri_json_str_eq(key, "prefill-compute")) {
            st = read_placement_field(r, "prefill-compute", false, &rule->prefill_compute,
                                      display_rule, err_msg, err_cap);
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "%s: unknown key '%s'", display_rule, kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) {
            return st;
        }
    }
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", display_rule, r->err_msg);
    }
    return st;
}

static void join_label4(char *dst, size_t cap, const char *a, const char *b, const char *c,
                        const char *d) {
    if (!dst || cap == 0U) {
        return;
    }
    size_t pos = 0U;
    const char *parts[4] = {a, b, c, d};
    for (size_t i = 0U; i < 4U; ++i) {
        if (!parts[i]) continue;
        size_t n = strlen(parts[i]);
        size_t avail = (cap - 1U) - pos;
        if (n > avail) n = avail;
        if (n > 0U) {
            memcpy(dst + pos, parts[i], n);
            pos += n;
        }
    }
    dst[pos] = '\0';
}

static velri_status parse_indexer_object(velri_json_reader *r, velri_engine_config *cfg,
                                         const char *scope_prefix, const char *display_prefix,
                                         int32_t selector_idx, char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", display_prefix, r->err_msg);
        return st;
    }
    uint32_t rule_idx = 0U;
    st = append_rule(cfg, scope_prefix, display_prefix, selector_idx, &rule_idx);
    if (st != VELRI_OK) return st;

    bool seen_keys = false;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "store")) {
            st = read_placement_field(r, "store", true, &cfg->rules[rule_idx].store, display_prefix,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "compute")) {
            st = read_placement_field(r, "compute", false, &cfg->rules[rule_idx].compute,
                                      display_prefix, err_msg, err_cap);
        } else if (velri_json_str_eq(key, "prefill-compute")) {
            st = read_placement_field(r, "prefill-compute", false,
                                      &cfg->rules[rule_idx].prefill_compute, display_prefix,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "keys")) {
            if (seen_keys) {
                set_errf(err_msg, err_cap, "%s: duplicate 'keys' object", display_prefix);
                return VELRI_ERR_INVALID;
            }
            seen_keys = true;
            char sub_scope[VELRI_CFG_SCOPE_MAX], sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_scope, sizeof(sub_scope), scope_prefix, "/keys", nullptr, nullptr);
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".keys", nullptr, nullptr);
            st = parse_leaf_placement(r, cfg, sub_scope, sub_disp, selector_idx, err_msg, err_cap);
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "%s: unknown key '%s'", display_prefix, kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "%s: %s", display_prefix, r->err_msg);
    return st;
}

static velri_status parse_attention_object(velri_json_reader *r, velri_engine_config *cfg,
                                           const char *scope_prefix, const char *display_prefix,
                                           int32_t selector_idx, char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", display_prefix, r->err_msg);
        return st;
    }
    uint32_t rule_idx = 0U;
    st = append_rule(cfg, scope_prefix, display_prefix, selector_idx, &rule_idx);
    if (st != VELRI_OK) return st;

    uint32_t seen_children = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "store")) {
            st = read_placement_field(r, "store", true, &cfg->rules[rule_idx].store, display_prefix,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "compute")) {
            st = read_placement_field(r, "compute", false, &cfg->rules[rule_idx].compute,
                                      display_prefix, err_msg, err_cap);
        } else if (velri_json_str_eq(key, "prefill-compute")) {
            st = read_placement_field(r, "prefill-compute", false,
                                      &cfg->rules[rule_idx].prefill_compute, display_prefix,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "kv-cache")) {
            if (seen_children & 1U) {
                set_errf(err_msg, err_cap, "%s: duplicate 'kv-cache' object", display_prefix);
                return VELRI_ERR_INVALID;
            }
            seen_children |= 1U;
            char sub_scope[VELRI_CFG_SCOPE_MAX], sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_scope, sizeof(sub_scope), scope_prefix, "/kv-cache", nullptr, nullptr);
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".kv-cache", nullptr, nullptr);
            st = parse_leaf_placement(r, cfg, sub_scope, sub_disp, selector_idx, err_msg, err_cap);
        } else if (velri_json_str_eq(key, "indexer")) {
            if (seen_children & 2U) {
                set_errf(err_msg, err_cap, "%s: duplicate 'indexer' object", display_prefix);
                return VELRI_ERR_INVALID;
            }
            seen_children |= 2U;
            char sub_scope[VELRI_CFG_SCOPE_MAX], sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_scope, sizeof(sub_scope), scope_prefix, "/indexer", nullptr, nullptr);
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".indexer", nullptr, nullptr);
            st = parse_indexer_object(r, cfg, sub_scope, sub_disp, selector_idx, err_msg, err_cap);
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "%s: unknown key '%s'", display_prefix, kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "%s: %s", display_prefix, r->err_msg);
    return st;
}

static velri_status parse_moe_object(velri_json_reader *r, velri_engine_config *cfg,
                                     const char *scope_prefix, const char *display_prefix,
                                     int32_t selector_idx, char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", display_prefix, r->err_msg);
        return st;
    }
    uint32_t rule_idx = 0U;
    st = append_rule(cfg, scope_prefix, display_prefix, selector_idx, &rule_idx);
    if (st != VELRI_OK) return st;

    uint32_t seen_children = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        const char *child = nullptr;
        uint32_t bit = 0U;
        if (velri_json_str_eq(key, "store")) {
            st = read_placement_field(r, "store", true, &cfg->rules[rule_idx].store, display_prefix,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "compute")) {
            st = read_placement_field(r, "compute", false, &cfg->rules[rule_idx].compute,
                                      display_prefix, err_msg, err_cap);
        } else if (velri_json_str_eq(key, "prefill-compute")) {
            st = read_placement_field(r, "prefill-compute", false,
                                      &cfg->rules[rule_idx].prefill_compute, display_prefix,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "router")) {
            child = "router";
            bit = 1U;
        } else if (velri_json_str_eq(key, "shared-experts")) {
            child = "shared-experts";
            bit = 2U;
        } else if (velri_json_str_eq(key, "routed-experts")) {
            child = "routed-experts";
            bit = 4U;
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "%s: unknown key '%s'", display_prefix, kbuf);
            return VELRI_ERR_INVALID;
        }
        if (child) {
            if (seen_children & bit) {
                set_errf(err_msg, err_cap, "%s: duplicate '%s' object", display_prefix, child);
                return VELRI_ERR_INVALID;
            }
            seen_children |= bit;
            char sub_scope[VELRI_CFG_SCOPE_MAX], sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_scope, sizeof(sub_scope), scope_prefix, "/", child, nullptr);
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".", child, nullptr);
            st = parse_leaf_placement(r, cfg, sub_scope, sub_disp, selector_idx, err_msg, err_cap);
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "%s: %s", display_prefix, r->err_msg);
    return st;
}

static velri_status parse_ple_object(velri_json_reader *r, velri_engine_config *cfg,
                                     char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "model.ple: %s", r->err_msg);
        return st;
    }
    uint32_t rule_idx = 0U;
    st = append_rule(cfg, "ple", "model.ple", -1, &rule_idx);
    if (st != VELRI_OK) return st;

    bool seen_table = false;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "store")) {
            st = read_placement_field(r, "store", true, &cfg->rules[rule_idx].store, "model.ple",
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "compute")) {
            st = read_placement_field(r, "compute", false, &cfg->rules[rule_idx].compute,
                                      "model.ple", err_msg, err_cap);
        } else if (velri_json_str_eq(key, "prefill-compute")) {
            st = read_placement_field(r, "prefill-compute", false,
                                      &cfg->rules[rule_idx].prefill_compute, "model.ple", err_msg,
                                      err_cap);
        } else if (velri_json_str_eq(key, "table")) {
            if (seen_table) {
                set_errf(err_msg, err_cap, "model.ple: duplicate 'table' object");
                return VELRI_ERR_INVALID;
            }
            seen_table = true;
            st = parse_leaf_placement(r, cfg, "ple/table", "model.ple.table", -1, err_msg, err_cap);
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "model.ple: unknown key '%s'", kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "model.ple: %s", r->err_msg);
    return st;
}

static velri_status parse_hidden_layers_body(velri_json_reader *r, velri_engine_config *cfg,
                                             const char *display_prefix, int32_t selector_idx,
                                             bool allow_selectors, char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s: %s", display_prefix, r->err_msg);
        return st;
    }
    uint32_t rule_idx = 0U;
    st = append_rule(cfg, "hidden-layers", display_prefix, selector_idx, &rule_idx);
    if (st != VELRI_OK) return st;

    uint32_t seen_children = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "store")) {
            st = read_placement_field(r, "store", true, &cfg->rules[rule_idx].store, display_prefix,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "compute")) {
            st = read_placement_field(r, "compute", false, &cfg->rules[rule_idx].compute,
                                      display_prefix, err_msg, err_cap);
        } else if (velri_json_str_eq(key, "prefill-compute")) {
            st = read_placement_field(r, "prefill-compute", false,
                                      &cfg->rules[rule_idx].prefill_compute, display_prefix,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "attention")) {
            if (seen_children & 1U) {
                set_errf(err_msg, err_cap, "%s: duplicate 'attention' object", display_prefix);
                return VELRI_ERR_INVALID;
            }
            seen_children |= 1U;
            char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".attention", nullptr, nullptr);
            st = parse_attention_object(r, cfg, "hidden-layers/attention", sub_disp, selector_idx,
                                        err_msg, err_cap);
        } else if (velri_json_str_eq(key, "dense-mlp")) {
            if (seen_children & 2U) {
                set_errf(err_msg, err_cap, "%s: duplicate 'dense-mlp' object", display_prefix);
                return VELRI_ERR_INVALID;
            }
            seen_children |= 2U;
            char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".dense-mlp", nullptr, nullptr);
            st = parse_leaf_placement(r, cfg, "hidden-layers/dense-mlp", sub_disp, selector_idx,
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "moe")) {
            if (seen_children & 4U) {
                set_errf(err_msg, err_cap, "%s: duplicate 'moe' object", display_prefix);
                return VELRI_ERR_INVALID;
            }
            seen_children |= 4U;
            char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".moe", nullptr, nullptr);
            st = parse_moe_object(r, cfg, "hidden-layers/moe", sub_disp, selector_idx, err_msg,
                                  err_cap);
        } else if (velri_json_str_eq(key, "gdn")) {
            if (seen_children & 8U) {
                set_errf(err_msg, err_cap, "%s: duplicate 'gdn' object", display_prefix);
                return VELRI_ERR_INVALID;
            }
            seen_children |= 8U;
            char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".gdn", nullptr, nullptr);
            st = parse_leaf_placement(r, cfg, "hidden-layers/gdn", sub_disp, selector_idx, err_msg,
                                      err_cap);
        } else if (velri_json_str_eq(key, "conv")) {
            if (seen_children & 16U) {
                set_errf(err_msg, err_cap, "%s: duplicate 'conv' object", display_prefix);
                return VELRI_ERR_INVALID;
            }
            seen_children |= 16U;
            char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            join_label4(sub_disp, sizeof(sub_disp), display_prefix, ".conv", nullptr, nullptr);
            st = parse_leaf_placement(r, cfg, "hidden-layers/conv", sub_disp, selector_idx, err_msg,
                                      err_cap);
        } else {
            char kbuf[64] = {0};
            if (allow_selectors && key.len > 0U && key.ptr[0] >= '0' && key.ptr[0] <= '9') {
                st = decode_cstr_buf(key, kbuf, sizeof(kbuf), display_prefix, err_msg, err_cap);
                if (st != VELRI_OK) return st;
                int32_t sub_sel = -1;
                st = parse_layer_selector(kbuf, cfg, &sub_sel, err_msg, err_cap);
                if (st != VELRI_OK) return st;
                char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
                join_label4(sub_disp, sizeof(sub_disp), display_prefix, "[", kbuf, "]");
                st = parse_hidden_layers_body(r, cfg, sub_disp, sub_sel, false, err_msg, err_cap);
            } else {
                /* Also support escaped digit-leading selector keys like "\u0030-3". */
                if (allow_selectors && key.has_escapes &&
                    decode_cstr_buf(key, kbuf, sizeof(kbuf), display_prefix, err_msg, err_cap) ==
                        VELRI_OK &&
                    kbuf[0] >= '0' && kbuf[0] <= '9') {
                    int32_t sub_sel = -1;
                    st = parse_layer_selector(kbuf, cfg, &sub_sel, err_msg, err_cap);
                    if (st != VELRI_OK) return st;
                    char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
                    join_label4(sub_disp, sizeof(sub_disp), display_prefix, "[", kbuf, "]");
                    st = parse_hidden_layers_body(r, cfg, sub_disp, sub_sel, false, err_msg,
                                                  err_cap);
                } else {
                    if (err_msg && err_cap > 0U) err_msg[0] = '\0';
                    format_key_preview(key, kbuf);
                    set_errf(err_msg, err_cap, "%s: unknown key '%s'", display_prefix, kbuf);
                    return VELRI_ERR_INVALID;
                }
            }
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "%s: %s", display_prefix, r->err_msg);
    return st;
}

static velri_status parse_sampling(velri_json_reader *r, velri_sampling_config *out, char *err_msg,
                                   size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "model.sampling: %s", r->err_msg);
        return st;
    }
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "temperature")) {
            if (seen & 1U) {
                set_errf(err_msg, err_cap, "model.sampling: duplicate 'temperature'");
                return VELRI_ERR_INVALID;
            }
            seen |= 1U;
            double v = 0.0;
            st = velri_json_read_f64(r, &v);
            float fv = (float)v;
            if (st != VELRI_OK || !isfinite(fv) || fv < 0.0f) {
                set_errf(err_msg, err_cap, "model.sampling.temperature: expected finite float >= 0");
                return VELRI_ERR_INVALID;
            }
            out->temperature = fv;
        } else if (velri_json_str_eq(key, "top-p")) {
            if (seen & 2U) {
                set_errf(err_msg, err_cap, "model.sampling: duplicate 'top-p'");
                return VELRI_ERR_INVALID;
            }
            seen |= 2U;
            double v = 0.0;
            st = velri_json_read_f64(r, &v);
            float fv = (float)v;
            if (st != VELRI_OK || !isfinite(fv) || fv <= 0.0f || fv > 1.0f) {
                set_errf(err_msg, err_cap, "model.sampling.top-p: expected finite float in (0, 1]");
                return VELRI_ERR_INVALID;
            }
            out->top_p = fv;
        } else if (velri_json_str_eq(key, "top-k")) {
            if (seen & 4U) {
                set_errf(err_msg, err_cap, "model.sampling: duplicate 'top-k'");
                return VELRI_ERR_INVALID;
            }
            seen |= 4U;
            st = read_u32_field(r, &out->top_k, "model.sampling.top-k", err_msg, err_cap);
        } else if (velri_json_str_eq(key, "min-p")) {
            if (seen & 8U) {
                set_errf(err_msg, err_cap, "model.sampling: duplicate 'min-p'");
                return VELRI_ERR_INVALID;
            }
            seen |= 8U;
            double v = 0.0;
            st = velri_json_read_f64(r, &v);
            float fv = (float)v;
            if (st != VELRI_OK || !isfinite(fv) || fv < 0.0f || fv > 1.0f) {
                set_errf(err_msg, err_cap, "model.sampling.min-p: expected finite float in [0, 1]");
                return VELRI_ERR_INVALID;
            }
            out->min_p = fv;
        } else if (velri_json_str_eq(key, "seed")) {
            if (seen & 16U) {
                set_errf(err_msg, err_cap, "model.sampling: duplicate 'seed'");
                return VELRI_ERR_INVALID;
            }
            seen |= 16U;
            st = velri_json_read_u64(r, &out->seed);
            if (st != VELRI_OK) {
                set_errf(err_msg, err_cap, "model.sampling.seed: %s", r->err_msg);
                return st;
            }
            out->has_seed = true;
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "model.sampling: unknown key '%s'", kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "model.sampling: %s", r->err_msg);
    return st;
}

static velri_status parse_model_object(velri_json_reader *r, velri_engine_config *cfg,
                                       char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "model: %s", r->err_msg);
        return st;
    }
    uint32_t root_idx = 0U;
    st = append_rule(cfg, "", "model", -1, &root_idx);
    if (st != VELRI_OK) return st;

    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "path")) {
            if (seen & 1U) {
                set_errf(err_msg, err_cap, "model: duplicate 'path'");
                return VELRI_ERR_INVALID;
            }
            seen |= 1U;
            st = read_cstr_buf(r, cfg->model_path, sizeof(cfg->model_path), "model.path", err_msg,
                               err_cap);
        } else if (velri_json_str_eq(key, "store")) {
            st = read_placement_field(r, "store", true, &cfg->rules[root_idx].store, "model",
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "compute")) {
            st = read_placement_field(r, "compute", false, &cfg->rules[root_idx].compute, "model",
                                      err_msg, err_cap);
        } else if (velri_json_str_eq(key, "prefill-compute")) {
            st = read_placement_field(r, "prefill-compute", false,
                                      &cfg->rules[root_idx].prefill_compute, "model", err_msg,
                                      err_cap);
        } else if (velri_json_str_eq(key, "input-layer")) {
            if (seen & 2U) {
                set_errf(err_msg, err_cap, "model: duplicate 'input-layer'");
                return VELRI_ERR_INVALID;
            }
            seen |= 2U;
            st = parse_leaf_placement(r, cfg, "input-layer", "model.input-layer", -1, err_msg,
                                      err_cap);
        } else if (velri_json_str_eq(key, "lm-head")) {
            if (seen & 4U) {
                set_errf(err_msg, err_cap, "model: duplicate 'lm-head'");
                return VELRI_ERR_INVALID;
            }
            seen |= 4U;
            st = parse_leaf_placement(r, cfg, "lm-head", "model.lm-head", -1, err_msg, err_cap);
        } else if (velri_json_str_eq(key, "ple")) {
            if (seen & 8U) {
                set_errf(err_msg, err_cap, "model: duplicate 'ple'");
                return VELRI_ERR_INVALID;
            }
            seen |= 8U;
            st = parse_ple_object(r, cfg, err_msg, err_cap);
        } else if (velri_json_str_eq(key, "hidden-layers")) {
            if (seen & 16U) {
                set_errf(err_msg, err_cap, "model: duplicate 'hidden-layers'");
                return VELRI_ERR_INVALID;
            }
            seen |= 16U;
            st = parse_hidden_layers_body(r, cfg, "model.hidden-layers", -1, true, err_msg,
                                          err_cap);
        } else if (velri_json_str_eq(key, "sampling")) {
            if (seen & 32U) {
                set_errf(err_msg, err_cap, "model: duplicate 'sampling'");
                return VELRI_ERR_INVALID;
            }
            seen |= 32U;
            st = parse_sampling(r, &cfg->sampling, err_msg, err_cap);
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "model: unknown key '%s'", kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "model: %s", r->err_msg);
    return st;
}

static velri_status parse_backends_object(velri_json_reader *r, velri_engine_config *cfg,
                                          char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "backends: %s", r->err_msg);
        return st;
    }
    uint32_t seen_b = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "cpu")) {
            if (seen_b & 1U) {
                set_errf(err_msg, err_cap, "backends: duplicate 'cpu'");
                return VELRI_ERR_INVALID;
            }
            seen_b |= 1U;
            velri_json_iter cit;
            st = velri_json_begin_object(r, &cit);
            if (st != VELRI_OK) {
                set_errf(err_msg, err_cap, "backends.cpu: %s", r->err_msg);
                return st;
            }
            uint32_t seen_c = 0U;
            bool cmore = false;
            velri_json_str ckey;
            while ((st = velri_json_next_member(r, &cit, &ckey, &cmore)) == VELRI_OK && cmore) {
                if (velri_json_str_eq(ckey, "threads")) {
                    if (seen_c & 1U) {
                        set_errf(err_msg, err_cap, "backends.cpu: duplicate 'threads'");
                        return VELRI_ERR_INVALID;
                    }
                    seen_c |= 1U;
                    st = read_u32_field(r, &cfg->cpu_backend.threads, "backends.cpu.threads",
                                        err_msg, err_cap);
                } else if (velri_json_str_eq(ckey, "reserve-cores")) {
                    if (seen_c & 2U) {
                        set_errf(err_msg, err_cap, "backends.cpu: duplicate 'reserve-cores'");
                        return VELRI_ERR_INVALID;
                    }
                    seen_c |= 2U;
                    st = read_u32_field(r, &cfg->cpu_backend.reserve_cores,
                                        "backends.cpu.reserve-cores", err_msg, err_cap);
                } else if (velri_json_str_eq(ckey, "affinity")) {
                    if (seen_c & 4U) {
                        set_errf(err_msg, err_cap, "backends.cpu: duplicate 'affinity'");
                        return VELRI_ERR_INVALID;
                    }
                    seen_c |= 4U;
                    char abuf[32];
                    st = read_cstr_buf(r, abuf, sizeof(abuf), "backends.cpu.affinity", err_msg,
                                       err_cap);
                    if (st != VELRI_OK) {
                        return st;
                    }
                    if (strcmp(abuf, "auto") == 0) {
                        cfg->cpu_backend.affinity = VELRI_AFFINITY_AUTO;
                    } else if (strcmp(abuf, "none") == 0) {
                        cfg->cpu_backend.affinity = VELRI_AFFINITY_NONE;
                    } else if (strcmp(abuf, "required") == 0) {
                        cfg->cpu_backend.affinity = VELRI_AFFINITY_REQUIRED;
                    } else {
                        set_errf(err_msg, err_cap,
                                 "backends.cpu.affinity: expected 'auto', 'none', or 'required' "
                                 "(got '%s')",
                                 abuf);
                        return VELRI_ERR_INVALID;
                    }
                } else {
                    char kbuf[64] = {0};
                    format_key_preview(ckey, kbuf);
                    set_errf(err_msg, err_cap, "backends.cpu: unknown key '%s'", kbuf);
                    return VELRI_ERR_INVALID;
                }
                if (st != VELRI_OK) return st;
            }
            if (st != VELRI_OK) set_errf(err_msg, err_cap, "backends.cpu: %s", r->err_msg);
        } else if (velri_json_str_eq(key, "vulkan")) {
            if (seen_b & 2U) {
                set_errf(err_msg, err_cap, "backends: duplicate 'vulkan'");
                return VELRI_ERR_INVALID;
            }
            seen_b |= 2U;
            velri_json_iter vit;
            st = velri_json_begin_object(r, &vit);
            if (st != VELRI_OK) {
                set_errf(err_msg, err_cap, "backends.vulkan: %s", r->err_msg);
                return st;
            }
            bool seen_chunk = false;
            bool vmore = false;
            velri_json_str vkey;
            while ((st = velri_json_next_member(r, &vit, &vkey, &vmore)) == VELRI_OK && vmore) {
                if (velri_json_str_eq(vkey, "prefill-chunk-size")) {
                    if (seen_chunk) {
                        set_errf(err_msg, err_cap, "backends.vulkan: duplicate 'prefill-chunk-size'");
                        return VELRI_ERR_INVALID;
                    }
                    seen_chunk = true;
                    st = read_u32_field(r, &cfg->vulkan_backend.prefill_chunk_size,
                                        "backends.vulkan.prefill-chunk-size", err_msg, err_cap);
                    if (st == VELRI_OK && cfg->vulkan_backend.prefill_chunk_size == 0U) {
                        set_errf(err_msg, err_cap,
                                 "backends.vulkan.prefill-chunk-size: must be > 0");
                        return VELRI_ERR_INVALID;
                    }
                } else {
                    char kbuf[64] = {0};
                    format_key_preview(vkey, kbuf);
                    set_errf(err_msg, err_cap, "backends.vulkan: unknown key '%s'", kbuf);
                    return VELRI_ERR_INVALID;
                }
                if (st != VELRI_OK) return st;
            }
            if (st != VELRI_OK) set_errf(err_msg, err_cap, "backends.vulkan: %s", r->err_msg);
        } else {
            /* Forward-compatible backend section: validate identifier + JSON object body
             * without hardcoding a closed backend list in the config parser. */
            char bname[VELRI_CFG_BACKEND_NAME_MAX];
            st = decode_cstr_buf(key, bname, sizeof(bname), "backends", err_msg, err_cap);
            if (st != VELRI_OK) return st;
            bool valid_id = (bname[0] != '\0');
            for (size_t i = 0U; valid_id && bname[i] != '\0'; ++i) {
                char c = bname[i];
                if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) {
                    valid_id = false;
                }
            }
            if (!valid_id) {
                set_errf(err_msg, err_cap, "backends: invalid backend section name '%s'", bname);
                return VELRI_ERR_INVALID;
            }
            if (velri_json_peek(r) != VELRI_JSON_OBJECT) {
                set_errf(err_msg, err_cap, "backends.%s: expected JSON object", bname);
                return VELRI_ERR_INVALID;
            }
            st = velri_json_skip(r);
            if (st != VELRI_OK) set_errf(err_msg, err_cap, "backends.%s: %s", bname, r->err_msg);
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "backends: %s", r->err_msg);
    return st;
}

static velri_status parse_context_object(velri_json_reader *r, velri_engine_config *cfg,
                                         char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "context: %s", r->err_msg);
        return st;
    }
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "size")) {
            if (seen & 1U) {
                set_errf(err_msg, err_cap, "context: duplicate 'size'");
                return VELRI_ERR_INVALID;
            }
            seen |= 1U;
            st = read_u32_field(r, &cfg->context_size, "context.size", err_msg, err_cap);
            if (st == VELRI_OK && cfg->context_size == 0U) {
                set_errf(err_msg, err_cap, "context.size: must be > 0");
                return VELRI_ERR_INVALID;
            }
        } else if (velri_json_str_eq(key, "max-seq-len")) {
            if (seen & 2U) {
                set_errf(err_msg, err_cap, "context: duplicate 'max-seq-len'");
                return VELRI_ERR_INVALID;
            }
            seen |= 2U;
            st = read_u32_field(r, &cfg->max_seq_len, "context.max-seq-len", err_msg, err_cap);
            if (st == VELRI_OK && cfg->max_seq_len == 0U) {
                set_errf(err_msg, err_cap, "context.max-seq-len: must be > 0");
                return VELRI_ERR_INVALID;
            }
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "context: unknown key '%s'", kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "context: %s", r->err_msg);
    return st;
}

static velri_status parse_kv_quant_object(velri_json_reader *r, velri_engine_config *cfg,
                                          char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "kv-quantization: %s", r->err_msg);
        return st;
    }
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "k-bits")) {
            if (seen & 1U) {
                set_errf(err_msg, err_cap, "kv-quantization: duplicate 'k-bits'");
                return VELRI_ERR_INVALID;
            }
            seen |= 1U;
            st = read_u32_field(r, &cfg->kv_k_bits, "kv-quantization.k-bits", err_msg, err_cap);
            if (st == VELRI_OK && (cfg->kv_k_bits == 0U || cfg->kv_k_bits > 16U)) {
                set_errf(err_msg, err_cap, "kv-quantization.k-bits: must be in 1..16");
                return VELRI_ERR_INVALID;
            }
        } else if (velri_json_str_eq(key, "v-bits")) {
            if (seen & 2U) {
                set_errf(err_msg, err_cap, "kv-quantization: duplicate 'v-bits'");
                return VELRI_ERR_INVALID;
            }
            seen |= 2U;
            st = read_u32_field(r, &cfg->kv_v_bits, "kv-quantization.v-bits", err_msg, err_cap);
            if (st == VELRI_OK && (cfg->kv_v_bits == 0U || cfg->kv_v_bits > 16U)) {
                set_errf(err_msg, err_cap, "kv-quantization.v-bits: must be in 1..16");
                return VELRI_ERR_INVALID;
            }
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "kv-quantization: unknown key '%s'", kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "kv-quantization: %s", r->err_msg);
    return st;
}

static velri_status parse_disk_object(velri_json_reader *r, velri_engine_config *cfg, char *err_msg,
                                      size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "disk: %s", r->err_msg);
        return st;
    }
    bool seen_io = false;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "io-threads")) {
            if (seen_io) {
                set_errf(err_msg, err_cap, "disk: duplicate 'io-threads'");
                return VELRI_ERR_INVALID;
            }
            seen_io = true;
            st = read_u32_field(r, &cfg->disk_io_threads, "disk.io-threads", err_msg, err_cap);
            if (st == VELRI_OK && cfg->disk_io_threads == 0U) {
                set_errf(err_msg, err_cap, "disk.io-threads: must be > 0");
                return VELRI_ERR_INVALID;
            }
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "disk: unknown key '%s'", kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "disk: %s", r->err_msg);
    return st;
}

static velri_status parse_server_object(velri_json_reader *r, char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "server: %s", r->err_msg);
        return st;
    }
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "host")) {
            if (seen & 1U) {
                set_errf(err_msg, err_cap, "server: duplicate 'host'");
                return VELRI_ERR_INVALID;
            }
            seen |= 1U;
            char hbuf[256];
            st = read_cstr_buf(r, hbuf, sizeof(hbuf), "server.host", err_msg, err_cap);
        } else if (velri_json_str_eq(key, "port")) {
            if (seen & 2U) {
                set_errf(err_msg, err_cap, "server: duplicate 'port'");
                return VELRI_ERR_INVALID;
            }
            seen |= 2U;
            uint32_t port = 0U;
            st = read_u32_field(r, &port, "server.port", err_msg, err_cap);
            if (st == VELRI_OK && (port == 0U || port > 65535U)) {
                set_errf(err_msg, err_cap, "server.port: must be in 1..65535");
                return VELRI_ERR_INVALID;
            }
        } else if (velri_json_str_eq(key, "max-concurrent-requests")) {
            if (seen & 4U) {
                set_errf(err_msg, err_cap, "server: duplicate 'max-concurrent-requests'");
                return VELRI_ERR_INVALID;
            }
            seen |= 4U;
            uint32_t mcr = 0U;
            st = read_u32_field(r, &mcr, "server.max-concurrent-requests", err_msg, err_cap);
            if (st == VELRI_OK && mcr == 0U) {
                set_errf(err_msg, err_cap, "server.max-concurrent-requests: must be > 0");
                return VELRI_ERR_INVALID;
            }
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "server: unknown key '%s'", kbuf);
            return VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) return st;
    }
    if (st != VELRI_OK) set_errf(err_msg, err_cap, "server: %s", r->err_msg);
    return st;
}

/* ---- Public Lifecycle & Parse API ---- */

void velri_engine_config_init(velri_engine_config *cfg) {
    if (!cfg) {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->sampling.temperature = 0.0f;
    cfg->sampling.top_p = 1.0f;
    cfg->sampling.top_k = 0U;
    cfg->sampling.min_p = 0.0f;
    cfg->cpu_backend.reserve_cores = 1U;
    cfg->cpu_backend.affinity = VELRI_AFFINITY_AUTO;
}

void velri_engine_config_destroy(velri_engine_config *cfg) {
    if (!cfg) {
        return;
    }
    for (uint32_t i = 0U; i < cfg->num_selectors; ++i) {
        free(cfg->selectors[i].ranges);
    }
    free(cfg->selectors);
    free(cfg->rules);
    velri_engine_config_init(cfg);
}

velri_status velri_engine_config_parse(const char *json_data, size_t json_len,
                                       velri_engine_config *out_cfg, char *err_msg,
                                       size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!json_data || !out_cfg) {
        set_errf(err_msg, err_cap, "invalid arguments to velri_engine_config_parse");
        return VELRI_ERR_INVALID;
    }

    velri_engine_config tmp;
    velri_engine_config_init(&tmp);

    velri_json_reader r;
    velri_json_init(&r, json_data, json_len);

    velri_json_iter it;
    velri_status st = velri_json_begin_object(&r, &it);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s", r.err_msg);
        velri_engine_config_destroy(&tmp);
        return st;
    }

    uint32_t seen_top = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(&r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "model")) {
            if (seen_top & 1U) {
                set_errf(err_msg, err_cap, "duplicate top-level 'model' object");
                st = VELRI_ERR_INVALID;
            } else {
                seen_top |= 1U;
                st = parse_model_object(&r, &tmp, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "chat-template")) {
            if (seen_top & 2U) {
                set_errf(err_msg, err_cap, "duplicate top-level 'chat-template'");
                st = VELRI_ERR_INVALID;
            } else if (velri_json_peek(&r) != VELRI_JSON_OBJECT) {
                set_errf(err_msg, err_cap, "chat-template: expected JSON object");
                st = VELRI_ERR_INVALID;
            } else {
                seen_top |= 2U;
                st = velri_json_skip(&r);
                if (st != VELRI_OK) set_errf(err_msg, err_cap, "chat-template: %s", r.err_msg);
            }
        } else if (velri_json_str_eq(key, "context")) {
            if (seen_top & 4U) {
                set_errf(err_msg, err_cap, "duplicate top-level 'context'");
                st = VELRI_ERR_INVALID;
            } else {
                seen_top |= 4U;
                st = parse_context_object(&r, &tmp, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "kv-quantization")) {
            if (seen_top & 8U) {
                set_errf(err_msg, err_cap, "duplicate top-level 'kv-quantization'");
                st = VELRI_ERR_INVALID;
            } else {
                seen_top |= 8U;
                st = parse_kv_quant_object(&r, &tmp, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "backends")) {
            if (seen_top & 16U) {
                set_errf(err_msg, err_cap, "duplicate top-level 'backends'");
                st = VELRI_ERR_INVALID;
            } else {
                seen_top |= 16U;
                st = parse_backends_object(&r, &tmp, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "disk")) {
            if (seen_top & 32U) {
                set_errf(err_msg, err_cap, "duplicate top-level 'disk'");
                st = VELRI_ERR_INVALID;
            } else {
                seen_top |= 32U;
                st = parse_disk_object(&r, &tmp, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "server")) {
            if (seen_top & 64U) {
                set_errf(err_msg, err_cap, "duplicate top-level 'server'");
                st = VELRI_ERR_INVALID;
            } else {
                seen_top |= 64U;
                st = parse_server_object(&r, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "attention-mechanism")) {
            if (seen_top & 128U) {
                set_errf(err_msg, err_cap, "duplicate top-level 'attention-mechanism'");
                st = VELRI_ERR_INVALID;
            } else {
                seen_top |= 128U;
                char mbuf[48];
                st = read_cstr_buf(&r, mbuf, sizeof(mbuf), "attention-mechanism", err_msg, err_cap);
                if (st != VELRI_OK) {
                    /* err_msg already populated */
                } else if (strcmp(mbuf, "flash-attention") == 0) {
                    tmp.attn_mechanism = VELRI_ATTN_MECH_FLASH;
                } else if (strcmp(mbuf, "latent-kv") == 0) {
                    tmp.attn_mechanism = VELRI_ATTN_MECH_LATENT_KV;
                } else if (strcmp(mbuf, "auto") == 0) {
                    tmp.attn_mechanism = VELRI_ATTN_MECH_AUTO;
                } else {
                    set_errf(err_msg, err_cap,
                             "attention-mechanism: unsupported value '%s' "
                             "(expected 'flash-attention', 'latent-kv', or 'auto')",
                             mbuf);
                    st = VELRI_ERR_INVALID;
                }
            }
        } else {
            char kbuf[64] = {0};
            format_key_preview(key, kbuf);
            set_errf(err_msg, err_cap, "unknown top-level config key '%s'", kbuf);
            st = VELRI_ERR_INVALID;
        }
        if (st != VELRI_OK) {
            velri_engine_config_destroy(&tmp);
            return st;
        }
    }
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s", r.err_msg);
        velri_engine_config_destroy(&tmp);
        return st;
    }

    st = velri_json_finish(&r);
    if (st != VELRI_OK) {
        set_errf(err_msg, err_cap, "%s", r.err_msg);
        velri_engine_config_destroy(&tmp);
        return st;
    }
    if ((seen_top & 1U) == 0U) {
        set_errf(err_msg, err_cap, "missing required top-level 'model' object");
        velri_engine_config_destroy(&tmp);
        return VELRI_ERR_INVALID;
    }

    velri_engine_config_destroy(out_cfg);
    *out_cfg = tmp;
    return VELRI_OK;
}

velri_status velri_engine_config_validate_layers(const velri_engine_config *cfg,
                                                 uint32_t num_hidden_layers, bool strict,
                                                 char *err_msg, size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!cfg || num_hidden_layers == 0U) {
        set_errf(err_msg, err_cap, "invalid arguments to validate_layers");
        return VELRI_ERR_INVALID;
    }
    if (!strict) {
        return VELRI_OK;
    }
    for (uint32_t i = 0U; i < cfg->num_selectors; ++i) {
        const velri_layer_selector *sel = &cfg->selectors[i];
        for (uint32_t r = 0U; r < sel->num_ranges; ++r) {
            if (sel->ranges[r].end >= num_hidden_layers) {
                set_errf(err_msg, err_cap,
                         "layer selector '%s' references layer %u, out of range for %u layers",
                         sel->raw_key, sel->ranges[r].end, num_hidden_layers);
                return VELRI_ERR_INVALID;
            }
        }
    }
    return VELRI_OK;
}

/* ---- Context Binding & Cascading Resolution ---- */

static velri_status resolve_spec_to_dev(const velri_context *ctx, velri_device_id host_ram_dev,
                                        bool has_host_ram_dev, const velri_device_spec *spec,
                                        const char *rule_label, const char *field_name,
                                        velri_device_id *out_dev, bool *out_is_disk, char *err_msg,
                                        size_t err_cap) {
    if (out_is_disk) *out_is_disk = false;
    if (spec->kind == VELRI_LOC_UNSET) {
        return VELRI_OK;
    }
    if (spec->kind == VELRI_LOC_DISK) {
        if (out_is_disk) *out_is_disk = true;
        if (out_dev) *out_dev = VELRI_INVALID_ID;
        return VELRI_OK;
    }
    if (spec->kind == VELRI_LOC_RAM) {
        if (!has_host_ram_dev) {
            set_errf(err_msg, err_cap, "%s.%s: 'ram:0' requires a host-memory device in context",
                     rule_label, field_name);
            return VELRI_ERR_NOT_FOUND;
        }
        if (out_dev) *out_dev = host_ram_dev;
        return VELRI_OK;
    }

    velri_device_id dev_id = VELRI_INVALID_ID;
    velri_status st =
        velri_context_find_device(ctx, spec->backend_name, spec->local_index, &dev_id);
    if (st != VELRI_OK) {
        /* Distinguish missing backend vs out-of-range device index for clear diagnostics. */
        uint32_t n_dev = velri_context_device_count(ctx);
        uint32_t backend_count = 0U;
        for (uint32_t d = 0U; d < n_dev; ++d) {
            velri_device_caps caps;
            if (velri_context_device_caps(ctx, d, &caps) == VELRI_OK &&
                strcmp(caps.backend_name, spec->backend_name) == 0) {
                backend_count++;
            }
        }
        if (backend_count == 0U) {
            set_errf(err_msg, err_cap,
                     "%s.%s: backend '%s' is not available or not enabled in this build/context",
                     rule_label, field_name, spec->backend_name);
        } else {
            set_errf(err_msg, err_cap,
                     "%s.%s: device '%s:%u' not found (%u '%s' device(s) available)", rule_label,
                     field_name, spec->backend_name, spec->local_index, backend_count,
                     spec->backend_name);
        }
        return VELRI_ERR_NOT_FOUND;
    }
    if (out_dev) *out_dev = dev_id;
    return VELRI_OK;
}

velri_status velri_placement_plan_bind(const velri_engine_config *cfg, const velri_context *ctx,
                                       velri_device_id fallback_dev, velri_placement_plan *out_plan,
                                       char *err_msg, size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!cfg || !ctx || !out_plan) {
        set_errf(err_msg, err_cap, "invalid arguments to velri_placement_plan_bind");
        return VELRI_ERR_INVALID;
    }

    uint32_t n_dev = velri_context_device_count(ctx);
    if (fallback_dev >= n_dev) {
        set_errf(err_msg, err_cap, "fallback device %u out of range (context has %u devices)",
                 fallback_dev, n_dev);
        return VELRI_ERR_NOT_FOUND;
    }

    velri_device_id host_dev = VELRI_INVALID_ID;
    bool has_host = false;
    for (uint32_t d = 0U; d < n_dev; ++d) {
        velri_device_caps caps;
        if (velri_context_device_caps(ctx, d, &caps) == VELRI_OK && caps.is_host_memory) {
            host_dev = d;
            has_host = true;
            break;
        }
    }

    bool has_disk = false;
    for (uint32_t i = 0U; i < cfg->num_rules; ++i) {
        const velri_placement_rule *rule = &cfg->rules[i];
        bool is_disk = false;
        velri_status st = resolve_spec_to_dev(ctx, host_dev, has_host, &rule->store,
                                              rule->display_rule, "store", nullptr, &is_disk,
                                              err_msg, err_cap);
        if (st != VELRI_OK) return st;
        if (is_disk) has_disk = true;

        st = resolve_spec_to_dev(ctx, host_dev, has_host, &rule->compute, rule->display_rule,
                                 "compute", nullptr, nullptr, err_msg, err_cap);
        if (st != VELRI_OK) return st;

        st = resolve_spec_to_dev(ctx, host_dev, has_host, &rule->prefill_compute,
                                 rule->display_rule, "prefill-compute", nullptr, nullptr, err_msg,
                                 err_cap);
        if (st != VELRI_OK) return st;
    }

    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->cfg = cfg;
    out_plan->ctx = ctx;
    out_plan->default_store_dev = fallback_dev;
    out_plan->default_compute_dev = fallback_dev;
    out_plan->host_ram_dev = host_dev;
    out_plan->has_host_ram_dev = has_host;
    out_plan->has_disk_rules = has_disk;

    /* Resolve root defaults from `model` rule if present. */
    velri_resolved_placement root_res;
    if (velri_placement_resolve(out_plan, "", -1, &root_res, err_msg, err_cap) == VELRI_OK) {
        if (!root_res.store_is_disk && root_res.store_dev != VELRI_INVALID_ID) {
            out_plan->default_store_dev = root_res.store_dev;
        }
        if (root_res.compute_dev != VELRI_INVALID_ID) {
            out_plan->default_compute_dev = root_res.compute_dev;
        }
    }
    return VELRI_OK;
}

static const velri_placement_rule *find_rule(const velri_engine_config *cfg, const char *scope_path,
                                             size_t scope_len, int32_t selector_idx) {
    for (uint32_t i = 0U; i < cfg->num_rules; ++i) {
        const velri_placement_rule *rule = &cfg->rules[i];
        if (rule->selector_idx == selector_idx && strlen(rule->scope_path) == scope_len &&
            (scope_len == 0U || memcmp(rule->scope_path, scope_path, scope_len) == 0)) {
            return rule;
        }
    }
    return nullptr;
}

static void apply_rule_cascade(const velri_placement_rule *rule, bool *has_store,
                               velri_device_spec *store_spec, char *store_src, size_t src_cap,
                               bool *has_comp, velri_device_spec *comp_spec, char *comp_src,
                               bool *has_pre, velri_device_spec *pre_spec, char *pre_src) {
    if (!rule) {
        return;
    }
    if (!*has_store && rule->store.kind != VELRI_LOC_UNSET) {
        *has_store = true;
        *store_spec = rule->store;
        join_label4(store_src, src_cap, rule->display_rule, ".store", nullptr, nullptr);
    }
    if (!*has_comp && rule->compute.kind != VELRI_LOC_UNSET) {
        *has_comp = true;
        *comp_spec = rule->compute;
        join_label4(comp_src, src_cap, rule->display_rule, ".compute", nullptr, nullptr);
    }
    if (!*has_pre) {
        if (rule->prefill_compute.kind != VELRI_LOC_UNSET) {
            *has_pre = true;
            *pre_spec = rule->prefill_compute;
            join_label4(pre_src, src_cap, rule->display_rule, ".prefill-compute", nullptr, nullptr);
        } else if (rule->compute.kind != VELRI_LOC_UNSET) {
            /* Optional `prefill-compute` inherits from `compute` at the winning specificity level. */
            *has_pre = true;
            *pre_spec = rule->compute;
            join_label4(pre_src, src_cap, rule->display_rule, ".compute", nullptr, nullptr);
        }
    }
}

velri_status velri_placement_resolve(const velri_placement_plan *plan, const char *scope_path,
                                     int32_t layer_idx, velri_resolved_placement *out,
                                     char *err_msg, size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!plan || !plan->cfg || !plan->ctx || !out) {
        set_errf(err_msg, err_cap, "invalid arguments to velri_placement_resolve");
        return VELRI_ERR_INVALID;
    }
    memset(out, 0, sizeof(*out));
    const char *path = scope_path ? scope_path : "";
    size_t plen = strlen(path);

    /* Collect prefix lengths of `scope_path` from deepest to shallowest. */
    size_t prefix_lens[16];
    uint32_t num_prefixes = 0U;
    if (plen > 0U) {
        size_t cur = plen;
        while (cur > 0U && num_prefixes < 16U) {
            prefix_lens[num_prefixes++] = cur;
            while (cur > 0U && path[cur - 1U] != '/') {
                cur--;
            }
            if (cur > 0U) {
                cur--; /* step before '/' */
            }
        }
    }

    bool has_store = false, has_comp = false, has_pre = false;

    /* 1. Highest priority: matching layer selector under hidden-layers (deepest -> shallowest). */
    bool in_hidden = (plen >= 13U && memcmp(path, "hidden-layers", 13U) == 0 &&
                      (plen == 13U || path[13] == '/'));
    if (in_hidden && layer_idx >= 0) {
        int32_t matched_sel = -1;
        for (uint32_t s = 0U; s < plan->cfg->num_selectors; ++s) {
            if (selector_matches_layer(&plan->cfg->selectors[s], (uint32_t)layer_idx)) {
                matched_sel = (int32_t)s;
                break;
            }
        }
        if (matched_sel >= 0) {
            for (uint32_t p = 0U; p < num_prefixes; ++p) {
                const velri_placement_rule *rule =
                    find_rule(plan->cfg, path, prefix_lens[p], matched_sel);
                apply_rule_cascade(rule, &has_store, &out->store_spec, out->store_source,
                                   sizeof(out->store_source), &has_comp, &out->compute_spec,
                                   out->compute_source, &has_pre, &out->prefill_spec,
                                   out->prefill_source);
            }
        }
    }

    /* 2. General tree prefixes (deepest -> shallowest). */
    for (uint32_t p = 0U; p < num_prefixes; ++p) {
        const velri_placement_rule *rule = find_rule(plan->cfg, path, prefix_lens[p], -1);
        apply_rule_cascade(rule, &has_store, &out->store_spec, out->store_source,
                           sizeof(out->store_source), &has_comp, &out->compute_spec,
                           out->compute_source, &has_pre, &out->prefill_spec, out->prefill_source);
    }

    /* 3. Model root defaults (""). */
    const velri_placement_rule *root = find_rule(plan->cfg, "", 0U, -1);
    apply_rule_cascade(root, &has_store, &out->store_spec, out->store_source,
                       sizeof(out->store_source), &has_comp, &out->compute_spec,
                       out->compute_source, &has_pre, &out->prefill_spec, out->prefill_source);

    /* 4. Context fallback when root omitted `store` or `compute`. */
    velri_status st = VELRI_OK;
    if (has_store) {
        st = resolve_spec_to_dev(plan->ctx, plan->host_ram_dev, plan->has_host_ram_dev,
                                 &out->store_spec, out->store_source, "store", &out->store_dev,
                                 &out->store_is_disk, err_msg, err_cap);
        if (st != VELRI_OK) return st;
    } else {
        out->store_dev = plan->default_store_dev;
        velri_device_caps caps;
        if (velri_context_device_caps(plan->ctx, out->store_dev, &caps) == VELRI_OK) {
            out->store_spec.kind = VELRI_LOC_BACKEND;
            snprintf(out->store_spec.backend_name, sizeof(out->store_spec.backend_name), "%s",
                     caps.backend_name);
            out->store_spec.local_index = caps.local_index;
        }
        snprintf(out->store_source, sizeof(out->store_source), "<default>");
    }

    if (has_comp) {
        st = resolve_spec_to_dev(plan->ctx, plan->host_ram_dev, plan->has_host_ram_dev,
                                 &out->compute_spec, out->compute_source, "compute",
                                 &out->compute_dev, nullptr, err_msg, err_cap);
        if (st != VELRI_OK) return st;
    } else {
        out->compute_dev = plan->default_compute_dev;
        velri_device_caps caps;
        if (velri_context_device_caps(plan->ctx, out->compute_dev, &caps) == VELRI_OK) {
            out->compute_spec.kind = VELRI_LOC_BACKEND;
            snprintf(out->compute_spec.backend_name, sizeof(out->compute_spec.backend_name), "%s",
                     caps.backend_name);
            out->compute_spec.local_index = caps.local_index;
        }
        snprintf(out->compute_source, sizeof(out->compute_source), "<default>");
    }

    if (has_pre) {
        st = resolve_spec_to_dev(plan->ctx, plan->host_ram_dev, plan->has_host_ram_dev,
                                 &out->prefill_spec, out->prefill_source, "prefill-compute",
                                 &out->prefill_dev, nullptr, err_msg, err_cap);
        if (st != VELRI_OK) return st;
    } else {
        out->prefill_spec = out->compute_spec;
        out->prefill_dev = out->compute_dev;
        snprintf(out->prefill_source, sizeof(out->prefill_source), "%s", out->compute_source);
    }

    return VELRI_OK;
}

bool velri_placement_has_distinct_prefill(const velri_placement_plan *plan,
                                          uint32_t num_hidden_layers) {
    if (!plan || !plan->cfg) {
        return false;
    }
    static const char *const k_root_scopes[] = {"", "input-layer", "lm-head", "ple", "ple/table"};
    for (size_t i = 0U; i < sizeof(k_root_scopes) / sizeof(k_root_scopes[0]); ++i) {
        velri_resolved_placement res;
        if (velri_placement_resolve(plan, k_root_scopes[i], -1, &res, nullptr, 0U) == VELRI_OK &&
            res.prefill_dev != res.compute_dev) {
            return true;
        }
    }

    static const char *const k_layer_scopes[] = {
        "hidden-layers",
        "hidden-layers/attention",
        "hidden-layers/attention/kv-cache",
        "hidden-layers/attention/indexer",
        "hidden-layers/attention/indexer/keys",
        "hidden-layers/dense-mlp",
        "hidden-layers/moe",
        "hidden-layers/moe/router",
        "hidden-layers/moe/shared-experts",
        "hidden-layers/moe/routed-experts",
        "hidden-layers/gdn",
        "hidden-layers/conv",
    };
    for (uint32_t l = 0U; l < num_hidden_layers; ++l) {
        for (size_t i = 0U; i < sizeof(k_layer_scopes) / sizeof(k_layer_scopes[0]); ++i) {
            velri_resolved_placement res;
            if (velri_placement_resolve(plan, k_layer_scopes[i], (int32_t)l, &res, nullptr, 0U) ==
                    VELRI_OK &&
                res.prefill_dev != res.compute_dev) {
                return true;
            }
        }
    }
    return false;
}

void velri_placement_builder_cb(const char *scope_path, int32_t layer_idx, void *user,
                                velri_device_id *out_store, velri_device_id *out_compute) {
    velri_placement_builder_adapter *ad = (velri_placement_builder_adapter *)user;
    if (!ad || !ad->plan || ad->err != VELRI_OK) {
        return;
    }
    velri_resolved_placement res;
    velri_status st = velri_placement_resolve(ad->plan, scope_path, layer_idx, &res, ad->err_msg,
                                              sizeof(ad->err_msg));
    if (st != VELRI_OK) {
        ad->err = st;
        if (ad->builder) {
            velri_b_set_error(ad->builder, st);
        }
        return;
    }
    if (res.store_is_disk) {
        ad->err = VELRI_ERR_UNSUPPORTED;
        set_errf(ad->err_msg, sizeof(ad->err_msg),
                 "scope '%s' (layer %d): 'disk' storage streaming is not yet implemented",
                 scope_path ? scope_path : "", layer_idx);
        if (ad->builder) {
            velri_b_set_error(ad->builder, ad->err);
        }
        return;
    }
    if (out_store) {
        *out_store = res.store_dev;
    }
    if (out_compute) {
        *out_compute = ad->is_prefill ? res.prefill_dev : res.compute_dev;
    }
}

/* ---- Inspect Report Formatter ---- */

static void appendf(char *buf, size_t cap, size_t *pos, const char *fmt, ...) {
    if (!buf || cap == 0U || *pos >= cap) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *pos, cap - *pos, fmt, ap);
    va_end(ap);
    if (n > 0) {
        size_t avail = cap - 1U - *pos;
        *pos += ((size_t)n < avail) ? (size_t)n : avail;
    }
}

static void format_resolved_line(char *buf, size_t cap, size_t *pos, const char *label,
                                 const velri_resolved_placement *r) {
    char s_buf[48], c_buf[48], p_buf[48];
    velri_device_spec_format(&r->store_spec, s_buf, sizeof(s_buf));
    velri_device_spec_format(&r->compute_spec, c_buf, sizeof(c_buf));
    velri_device_spec_format(&r->prefill_spec, p_buf, sizeof(p_buf));
    appendf(buf, cap, pos, "  %-32s store=%-10s (%s) | compute=%-10s (%s) | prefill=%-10s (%s)\n",
            label, s_buf, r->store_source, c_buf, r->compute_source, p_buf, r->prefill_source);
}

static bool device_spec_eq(const velri_device_spec *a, const velri_device_spec *b) {
    if (a->kind != b->kind || a->local_index != b->local_index) {
        return false;
    }
    if (a->kind == VELRI_LOC_BACKEND && strcmp(a->backend_name, b->backend_name) != 0) {
        return false;
    }
    return true;
}

static bool resolved_placement_eq(const velri_resolved_placement *a,
                                  const velri_resolved_placement *b) {
    return a->store_dev == b->store_dev && a->compute_dev == b->compute_dev &&
           a->prefill_dev == b->prefill_dev && a->store_is_disk == b->store_is_disk &&
           device_spec_eq(&a->store_spec, &b->store_spec) &&
           device_spec_eq(&a->compute_spec, &b->compute_spec) &&
           device_spec_eq(&a->prefill_spec, &b->prefill_spec) &&
           strcmp(a->store_source, b->store_source) == 0 &&
           strcmp(a->compute_source, b->compute_source) == 0 &&
           strcmp(a->prefill_source, b->prefill_source) == 0;
}

static bool config_has_scope_rule(const velri_engine_config *cfg, const char *scope_path) {
    for (uint32_t i = 0U; i < cfg->num_rules; ++i) {
        if (strcmp(cfg->rules[i].scope_path, scope_path) == 0) {
            return true;
        }
    }
    return false;
}

size_t velri_placement_format_inspect(const velri_placement_plan *plan, uint32_t num_hidden_layers,
                                      const velri_memory_report *mem_report, char *out_buf,
                                      size_t out_cap) {
    if (!plan || !plan->cfg || !plan->ctx || !out_buf || out_cap == 0U) {
        return 0U;
    }
    size_t pos = 0U;
    out_buf[0] = '\0';

    appendf(out_buf, out_cap, &pos, "=== Velri Placement & Context Inspect ===\n");
    if (plan->cfg->model_path[0] != '\0') {
        appendf(out_buf, out_cap, &pos, "Model path: %s\n", plan->cfg->model_path);
    }

    uint32_t n_dev = velri_context_device_count(plan->ctx);
    appendf(out_buf, out_cap, &pos, "Devices (%u):\n", n_dev);
    for (uint32_t d = 0U; d < n_dev; ++d) {
        velri_device_caps caps;
        if (velri_context_device_caps(plan->ctx, d, &caps) == VELRI_OK) {
            appendf(out_buf, out_cap, &pos, "  [%u] %s:%u - %s (host=%s, uma=%s)\n", d,
                    caps.backend_name, caps.local_index, caps.device_name,
                    caps.is_host_memory ? "yes" : "no", caps.is_uma ? "yes" : "no");
        }
    }

    const char *aff_str = (plan->cfg->cpu_backend.affinity == VELRI_AFFINITY_NONE)
                              ? "none"
                              : ((plan->cfg->cpu_backend.affinity == VELRI_AFFINITY_REQUIRED)
                                     ? "required"
                                     : "auto");
    appendf(out_buf, out_cap, &pos,
            "Backends: cpu(threads=%u, reserve=%u, affinity=%s), vulkan(prefill_chunk=%u)\n",
            plan->cfg->cpu_backend.threads, plan->cfg->cpu_backend.reserve_cores, aff_str,
            plan->cfg->vulkan_backend.prefill_chunk_size);

    /* Flag any out-of-range layer selectors when non-strict validation is used. */
    if (num_hidden_layers > 0U) {
        for (uint32_t s = 0U; s < plan->cfg->num_selectors; ++s) {
            const velri_layer_selector *sel = &plan->cfg->selectors[s];
            for (uint32_t r = 0U; r < sel->num_ranges; ++r) {
                if (sel->ranges[r].end >= num_hidden_layers) {
                    appendf(out_buf, out_cap, &pos,
                            "  [note] layer selector '%s' exceeds %u model layers\n", sel->raw_key,
                            num_hidden_layers);
                    break;
                }
            }
        }
    }

    appendf(out_buf, out_cap, &pos, "Resolved Placement:\n");
    velri_resolved_placement res;
    if (velri_placement_resolve(plan, "input-layer", -1, &res, nullptr, 0U) == VELRI_OK) {
        format_resolved_line(out_buf, out_cap, &pos, "input-layer", &res);
    }
    if (velri_placement_resolve(plan, "lm-head", -1, &res, nullptr, 0U) == VELRI_OK) {
        format_resolved_line(out_buf, out_cap, &pos, "lm-head", &res);
    }
    if (config_has_scope_rule(plan->cfg, "ple") &&
        velri_placement_resolve(plan, "ple", -1, &res, nullptr, 0U) == VELRI_OK) {
        format_resolved_line(out_buf, out_cap, &pos, "ple", &res);
    }
    if (config_has_scope_rule(plan->cfg, "ple/table") &&
        velri_placement_resolve(plan, "ple/table", -1, &res, nullptr, 0U) == VELRI_OK) {
        format_resolved_line(out_buf, out_cap, &pos, "ple.table", &res);
    }

    /* Select active layer sub-scopes: always show dense core + any configured optional scopes. */
    typedef struct {
        const char *scope;
        const char *suffix;
        bool always_show;
    } inspect_layer_scope;
    static const inspect_layer_scope k_lscopes[] = {
        {"hidden-layers/attention", "attention", true},
        {"hidden-layers/attention/kv-cache", "kv-cache", true},
        {"hidden-layers/attention/indexer", "attention.indexer", false},
        {"hidden-layers/attention/indexer/keys", "attention.indexer.keys", false},
        {"hidden-layers/dense-mlp", "dense-mlp", true},
        {"hidden-layers/moe", "moe", false},
        {"hidden-layers/moe/router", "moe.router", false},
        {"hidden-layers/moe/shared-experts", "moe.shared-experts", false},
        {"hidden-layers/moe/routed-experts", "moe.routed-experts", false},
        {"hidden-layers/gdn", "gdn", false},
        {"hidden-layers/conv", "conv", false},
    };
    enum { K_MAX_LSCOPES = sizeof(k_lscopes) / sizeof(k_lscopes[0]) };

    uint32_t active_idx[K_MAX_LSCOPES];
    uint32_t num_active = 0U;
    for (uint32_t i = 0U; i < K_MAX_LSCOPES; ++i) {
        if (k_lscopes[i].always_show || config_has_scope_rule(plan->cfg, k_lscopes[i].scope)) {
            active_idx[num_active++] = i;
        }
    }

    /* Group adjacent layers ONLY when ALL displayed sub-scopes match across every field. */
    for (uint32_t l = 0U; l < num_hidden_layers; ++l) {
        velri_resolved_placement base[K_MAX_LSCOPES];
        for (uint32_t k = 0U; k < num_active; ++k) {
            velri_placement_resolve(plan, k_lscopes[active_idx[k]].scope, (int32_t)l, &base[k],
                                    nullptr, 0U);
        }

        uint32_t end_l = l;
        while (end_l + 1U < num_hidden_layers) {
            bool all_same = true;
            for (uint32_t k = 0U; k < num_active; ++k) {
                velri_resolved_placement next_r;
                velri_placement_resolve(plan, k_lscopes[active_idx[k]].scope,
                                        (int32_t)(end_l + 1U), &next_r, nullptr, 0U);
                if (!resolved_placement_eq(&base[k], &next_r)) {
                    all_same = false;
                    break;
                }
            }
            if (!all_same) break;
            end_l++;
        }

        char lbl[64];
        for (uint32_t k = 0U; k < num_active; ++k) {
            snprintf(lbl, sizeof(lbl), "layers[%u..%u].%s", l, end_l,
                     k_lscopes[active_idx[k]].suffix);
            format_resolved_line(out_buf, out_cap, &pos, lbl, &base[k]);
        }
        l = end_l;
    }

    if (mem_report) {
        appendf(out_buf, out_cap, &pos,
                "Memory Plan: total=%zu B | ram_staging=%zu B | ram_transfers=%u\n",
                mem_report->total_physical_bytes, mem_report->ram_staging_bytes,
                mem_report->num_ram_transfers);
        uint32_t nd = (mem_report->num_devices <= VELRI_MAX_REPORT_DEVICES)
                          ? mem_report->num_devices
                          : (uint32_t)VELRI_MAX_REPORT_DEVICES;
        for (uint32_t d = 0U; d < nd; ++d) {
            const velri_device_memory_report *dr = &mem_report->devices[d];
            velri_device_caps caps;
            const char *bname = "dev";
            uint32_t lidx = dr->device_id;
            if (velri_context_device_caps(plan->ctx, dr->device_id, &caps) == VELRI_OK) {
                bname = caps.backend_name;
                lidx = caps.local_index;
            }
            appendf(out_buf, out_cap, &pos,
                    "  %s:%u | weights=%zu B | state=%zu B | io=%zu B | scratch=%zu B "
                    "(unpacked=%zu B) | ovh=%zu B\n",
                    bname, lidx, dr->weight_bytes, dr->state_bytes, dr->input_output_bytes,
                    dr->scratch_packed_bytes, dr->scratch_unpacked_bytes,
                    dr->allocator_overhead_bytes);
        }
    }

    return pos;
}
