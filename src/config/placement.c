#include "config/placement_internal.h"
#include "config/sampling_limits.h"
#include "format/json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))

/* ---- Shared Enum Tables & Helpers ---- */

const velri_cfg_enum_entry velri_cfg_affinity_table[] = {
    {"auto", VELRI_AFFINITY_AUTO},
    {"none", VELRI_AFFINITY_NONE},
    {"required", VELRI_AFFINITY_REQUIRED},
};
const size_t velri_cfg_affinity_table_len = ARRAY_COUNT(velri_cfg_affinity_table);

static const velri_cfg_enum_entry k_attn_mech_table[] = {
    {"flash-attention", VELRI_ATTN_MECH_FLASH},
    {"latent-kv", VELRI_ATTN_MECH_LATENT_KV},
    {"auto", VELRI_ATTN_MECH_AUTO},
};

const char *velri_cfg_enum_name(const velri_cfg_enum_entry *table, size_t count, int value,
                                const char *fallback) {
    for (size_t i = 0U; i < count; ++i) {
        if (table[i].value == value) {
            return table[i].name;
        }
    }
    return fallback;
}

static bool lookup_enum_str(const velri_cfg_enum_entry *table, size_t count, const char *text,
                            int *out_value) {
    for (size_t i = 0U; i < count; ++i) {
        if (strcmp(table[i].name, text) == 0) {
            *out_value = table[i].value;
            return true;
        }
    }
    return false;
}

void velri_cfg_set_errf(char *err_msg, size_t err_cap, const char *fmt, ...) {
    if (!err_msg || err_cap == 0U || err_msg[0] != '\0') {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err_msg, err_cap, fmt, ap);
    va_end(ap);
}

void velri_cfg_join_label4(char *dst, size_t cap, const char *a, const char *b, const char *c,
                           const char *d) {
    if (!dst || cap == 0U) {
        return;
    }
    size_t pos = 0U;
    const char *parts[4] = {a, b, c, d};
    for (size_t i = 0U; i < 4U; ++i) {
        if (!parts[i]) {
            continue;
        }
        size_t n = strlen(parts[i]);
        size_t avail = (cap - 1U) - pos;
        if (n > avail) {
            n = avail;
        }
        if (n > 0U) {
            memcpy(dst + pos, parts[i], n);
            pos += n;
        }
    }
    dst[pos] = '\0';
}

/* ---- Diagnostic, String, Numeric & Allocation Helpers ---- */

static velri_status decode_cstr_buf(velri_json_str s, char *dst, size_t dst_cap, const char *path,
                                    char *err_msg, size_t err_cap) {
    size_t len = 0U;
    velri_status st = velri_json_str_decode(s, dst, dst_cap, &len);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: string exceeds maximum length", path);
        return st;
    }
    if (memchr(dst, '\0', len) != nullptr) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: embedded NUL character not allowed", path);
        return VELRI_ERR_INVALID;
    }
    return VELRI_OK;
}

static velri_status read_cstr_buf(velri_json_reader *r, char *dst, size_t dst_cap, const char *path,
                                  char *err_msg, size_t err_cap) {
    velri_json_str s;
    velri_status st = velri_json_read_string(r, &s);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: %s", path, r->err_msg);
        return st;
    }
    return decode_cstr_buf(s, dst, dst_cap, path, err_msg, err_cap);
}

static velri_status read_enum_str(velri_json_reader *r, const velri_cfg_enum_entry *table,
                                  size_t count, int *out_value, const char *path, char *err_msg,
                                  size_t err_cap) {
    char buf[48];
    velri_status st = read_cstr_buf(r, buf, sizeof(buf), path, err_msg, err_cap);
    if (st != VELRI_OK) {
        return st;
    }
    if (!lookup_enum_str(table, count, buf, out_value)) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: unsupported value '%s'", path, buf);
        return VELRI_ERR_INVALID;
    }
    return VELRI_OK;
}

static velri_status fail_unknown_key(velri_json_str key, const char *prefix, char *err_msg,
                                     size_t err_cap) {
    char kbuf[64] = {0};
    size_t len = 0U;
    if (velri_json_str_decode(key, kbuf, sizeof(kbuf), &len) != VELRI_OK ||
        memchr(kbuf, '\0', len) != nullptr) {
        size_t n = (key.len < 60U) ? key.len : 60U;
        for (size_t i = 0U; i < n; ++i) {
            unsigned char c = (unsigned char)key.ptr[i];
            kbuf[i] = (c >= 0x20U && c < 0x7FU) ? (char)c : '?';
        }
        if (key.len > n) {
            memcpy(kbuf + n, "...", 3U);
            n += 3U;
        }
        kbuf[n] = '\0';
    }
    if (prefix && prefix[0] != '\0') {
        velri_cfg_set_errf(err_msg, err_cap, "%s: unknown key '%s'", prefix, kbuf);
    } else {
        velri_cfg_set_errf(err_msg, err_cap, "unknown top-level config key '%s'", kbuf);
    }
    return VELRI_ERR_INVALID;
}

static bool check_dup(uint32_t *seen, uint32_t bit, const char *obj, const char *key,
                      char *err_msg, size_t err_cap) {
    if ((*seen & bit) != 0U) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: duplicate '%s'", obj, key);
        return false;
    }
    *seen |= bit;
    return true;
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
    uint32_t nc = (*cap == 0U) ? 8U : ((*cap <= UINT32_MAX / 2U) ? (*cap * 2U) : UINT32_MAX);
    if (nc < count + 1U) {
        nc = count + 1U;
    }
    size_t nc_sz = nc;
    if (nc_sz > SIZE_MAX / elem_size) {
        return false;
    }
    void *np = realloc(old_ptr, nc_sz * elem_size);
    if (!np) {
        return false;
    }
    *out_ptr = np;
    *cap = nc;
    return true;
}

static velri_status read_u32_range(velri_json_reader *r, uint32_t *out, uint32_t min_v,
                                   uint32_t max_v, const char *path, char *err_msg,
                                   size_t err_cap) {
    uint64_t v = 0U;
    velri_status st = velri_json_read_u64(r, &v);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: %s", path, r->err_msg);
        return st;
    }
    if (v > (uint64_t)UINT32_MAX) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: value exceeds uint32 range", path);
        return VELRI_ERR_OVERFLOW;
    }
    uint32_t u = (uint32_t)v;
    if (u < min_v || u > max_v) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: value %u out of range %u..%u", path, u, min_v,
                           max_v);
        return VELRI_ERR_INVALID;
    }
    *out = u;
    return VELRI_OK;
}

static velri_status read_sampling_float(velri_json_reader *r, float *out, bool (*valid)(double),
                                        const char *path, char *err_msg, size_t err_cap) {
    double v = 0.0;
    velri_status st = velri_json_read_f64(r, &v);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: %s", path, r->err_msg);
        return st;
    }
    if (!valid(v)) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: expected finite float in range", path);
        return VELRI_ERR_INVALID;
    }
    *out = (float)v;
    return VELRI_OK;
}

static bool is_valid_ident(const char *s, size_t len) {
    if (len == 0U) {
        return false;
    }
    for (size_t i = 0U; i < len; ++i) {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) {
            return false;
        }
    }
    return true;
}

/* ---- Device Spec Parsing & Formatting ---- */

velri_status velri_device_spec_parse(const char *text, bool allow_store_only,
                                     velri_device_spec *out_spec, char *err_msg, size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!text || text[0] == '\0' || !out_spec) {
        velri_cfg_set_errf(err_msg, err_cap, "empty device specification");
        return VELRI_ERR_INVALID;
    }
    memset(out_spec, 0, sizeof(*out_spec));

    if (strcmp(text, "disk") == 0) {
        if (!allow_store_only) {
            velri_cfg_set_errf(err_msg, err_cap, "'disk' is store-only and cannot be used for compute");
            return VELRI_ERR_INVALID;
        }
        out_spec->kind = VELRI_LOC_DISK;
        return VELRI_OK;
    }

    const char *colon = strchr(text, ':');
    if (!colon || colon == text || colon[1] == '\0' || strchr(colon + 1, ':') != nullptr) {
        velri_cfg_set_errf(err_msg, err_cap,
                           "invalid device spec '%s': expected '<backend>:<index>', 'ram:0', or 'disk'",
                           text);
        return VELRI_ERR_INVALID;
    }

    size_t blen = (size_t)(colon - text);
    if (blen >= VELRI_CFG_BACKEND_NAME_MAX) {
        velri_cfg_set_errf(err_msg, err_cap, "backend name too long in '%s'", text);
        return VELRI_ERR_OVERFLOW;
    }
    if (!is_valid_ident(text, blen)) {
        velri_cfg_set_errf(err_msg, err_cap, "invalid character in backend name '%s'", text);
        return VELRI_ERR_INVALID;
    }
    char bname[VELRI_CFG_BACKEND_NAME_MAX];
    memcpy(bname, text, blen);
    bname[blen] = '\0';

    const char *idx_s = colon + 1;
    uint64_t idx = 0U;
    if (idx_s[0] == '0' && idx_s[1] != '\0') {
        velri_cfg_set_errf(err_msg, err_cap, "invalid device index in '%s' (leading zero)", text);
        return VELRI_ERR_INVALID;
    }
    for (const char *p = idx_s; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            velri_cfg_set_errf(err_msg, err_cap, "invalid device index in '%s'", text);
            return VELRI_ERR_INVALID;
        }
        idx = idx * 10U + (uint64_t)(*p - '0');
        if (idx > (uint64_t)UINT32_MAX) {
            velri_cfg_set_errf(err_msg, err_cap, "device index overflow in '%s'", text);
            return VELRI_ERR_OVERFLOW;
        }
    }

    if (strcmp(bname, "cuda") == 0 || strcmp(bname, "vram") == 0) {
        velri_cfg_set_errf(err_msg, err_cap,
                           "legacy device name '%s' is not supported; migrate to 'vulkan:%u'", text,
                           (uint32_t)idx);
        return VELRI_ERR_INVALID;
    }
    if (strcmp(bname, "device") == 0) {
        velri_cfg_set_errf(err_msg, err_cap,
                           "ambiguous device name '%s' is disallowed; specify explicit '<backend>:%u'",
                           text, (uint32_t)idx);
        return VELRI_ERR_INVALID;
    }
    if (strcmp(bname, "disk") == 0) {
        velri_cfg_set_errf(err_msg, err_cap, "use 'disk' without a device index (got '%s')", text);
        return VELRI_ERR_INVALID;
    }
    if (strcmp(bname, "ram") == 0) {
        if (idx != 0U) {
            velri_cfg_set_errf(err_msg, err_cap,
                               "only 'ram:0' is supported for host RAM storage (got '%s')", text);
            return VELRI_ERR_INVALID;
        }
        if (!allow_store_only) {
            velri_cfg_set_errf(err_msg, err_cap,
                               "'ram:0' is store-only; use 'cpu:0' (or another backend) for compute");
            return VELRI_ERR_INVALID;
        }
        out_spec->kind = VELRI_LOC_RAM;
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
        if (out_layer) {
            *out_layer = lo;
        }
        return true;
    }
    return false;
}

static velri_status parse_layer_range_segment(const char *seg, size_t seg_len, const char *key,
                                              velri_layer_range *out_range, char *err_msg,
                                              size_t err_cap) {
    if (seg_len == 0U) {
        velri_cfg_set_errf(err_msg, err_cap, "empty segment in layer selector '%s'", key);
        return VELRI_ERR_INVALID;
    }
    const char *dash = (const char *)memchr(seg, '-', seg_len);
    uint32_t lo = 0U;
    uint32_t hi = 0U;
    if (!dash) {
        if (!parse_u32_slice(seg, seg_len, &lo)) {
            velri_cfg_set_errf(err_msg, err_cap, "invalid layer index in selector '%s'", key);
            return VELRI_ERR_INVALID;
        }
        hi = lo;
    } else {
        size_t left = (size_t)(dash - seg);
        size_t right = seg_len - left - 1U;
        if (memchr(dash + 1, '-', right) != nullptr || !parse_u32_slice(seg, left, &lo) ||
            !parse_u32_slice(dash + 1, right, &hi) || lo > hi) {
            velri_cfg_set_errf(err_msg, err_cap, "invalid layer range in selector '%s'", key);
            return VELRI_ERR_INVALID;
        }
    }
    *out_range = (velri_layer_range){.start = lo, .end = hi};
    return VELRI_OK;
}

static velri_status check_range_overlaps(const velri_engine_config *cfg,
                                         const velri_layer_range *ranges, uint32_t num_ranges,
                                         velri_layer_range nr, const char *key, char *err_msg,
                                         size_t err_cap) {
    uint32_t dup = 0U;
    for (uint32_t i = 0U; i < num_ranges; ++i) {
        if (ranges_overlap(ranges[i], nr, &dup)) {
            velri_cfg_set_errf(err_msg, err_cap, "layer %u duplicated inside selector '%s'", dup,
                               key);
            return VELRI_ERR_INVALID;
        }
    }
    for (uint32_t s = 0U; s < cfg->num_selectors; ++s) {
        const velri_layer_selector *other = &cfg->selectors[s];
        for (uint32_t j = 0U; j < other->num_ranges; ++j) {
            if (ranges_overlap(other->ranges[j], nr, &dup)) {
                velri_cfg_set_errf(err_msg, err_cap,
                                   "overlapping layer selectors '%s' and '%s' at layer %u",
                                   other->raw_key, key, dup);
                return VELRI_ERR_INVALID;
            }
        }
    }
    return VELRI_OK;
}

static velri_status parse_layer_selector(const char *key, velri_engine_config *cfg,
                                         int32_t *out_sel_idx, char *err_msg, size_t err_cap) {
    size_t klen = strlen(key);
    if (klen == 0U || klen >= sizeof(cfg->selectors[0].raw_key)) {
        velri_cfg_set_errf(err_msg, err_cap, "invalid layer selector '%s'", key);
        return VELRI_ERR_INVALID;
    }

    velri_layer_range *ranges = nullptr;
    uint32_t num_ranges = 0U;
    uint32_t cap_ranges = 0U;
    velri_status st = VELRI_OK;

    size_t pos = 0U;
    while (pos < klen) {
        size_t seg_start = pos;
        while (pos < klen && key[pos] != ',') {
            pos++;
        }
        velri_layer_range nr;
        st = parse_layer_range_segment(key + seg_start, pos - seg_start, key, &nr, err_msg, err_cap);
        if (st == VELRI_OK) {
            st = check_range_overlaps(cfg, ranges, num_ranges, nr, key, err_msg, err_cap);
        }
        if (st != VELRI_OK) {
            break;
        }

        void *np = nullptr;
        if (!grow_array(ranges, &np, &cap_ranges, num_ranges, sizeof(*ranges))) {
            st = VELRI_ERR_NOMEM;
            break;
        }
        ranges = (velri_layer_range *)np;
        ranges[num_ranges++] = nr;

        if (pos < klen) {
            pos++;
            if (pos == klen) {
                velri_cfg_set_errf(err_msg, err_cap, "trailing comma in layer selector '%s'", key);
                st = VELRI_ERR_INVALID;
                break;
            }
        }
    }

    void *np = nullptr;
    if (st == VELRI_OK && !grow_array(cfg->selectors, &np, &cfg->cap_selectors, cfg->num_selectors,
                                      sizeof(*cfg->selectors))) {
        st = VELRI_ERR_NOMEM;
    }
    if (st != VELRI_OK) {
        free(ranges);
        return st;
    }

    cfg->selectors = (velri_layer_selector *)np;
    velri_layer_selector *sel = &cfg->selectors[cfg->num_selectors];
    memset(sel, 0, sizeof(*sel));
    memcpy(sel->raw_key, key, klen + 1U);
    sel->ranges = ranges;
    sel->num_ranges = num_ranges;
    *out_sel_idx = (int32_t)cfg->num_selectors++;
    return VELRI_OK;
}

bool velri_cfg_selector_matches_layer(const velri_layer_selector *sel, uint32_t layer) {
    for (uint32_t i = 0U; i < sel->num_ranges; ++i) {
        if (layer >= sel->ranges[i].start && layer <= sel->ranges[i].end) {
            return true;
        }
    }
    return false;
}

/* ---- Declarative Table-Driven Placement Subtree Parser ---- */

static velri_status append_rule(velri_engine_config *cfg, const char *scope_path,
                                const char *display_rule, int32_t selector_idx, uint32_t *out_idx) {
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
    *out_idx = cfg->num_rules++;
    return VELRI_OK;
}

static velri_status read_placement_field(velri_json_reader *r, const char *field_name,
                                         bool allow_store_only, velri_device_spec *target,
                                         const char *display_rule, char *err_msg, size_t err_cap) {
    if (target->kind != VELRI_LOC_UNSET) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: duplicate '%s' key", display_rule, field_name);
        return VELRI_ERR_INVALID;
    }
    char path[VELRI_CFG_RULE_LABEL_MAX + 32];
    char val[96];
    char spec_err[VELRI_CFG_ERR_MAX] = {0};
    velri_cfg_join_label4(path, sizeof(path), display_rule, ".", field_name, nullptr);
    velri_status st = read_cstr_buf(r, val, sizeof(val), path, err_msg, err_cap);
    if (st != VELRI_OK) {
        return st;
    }
    st = velri_device_spec_parse(val, allow_store_only, target, spec_err, sizeof(spec_err));
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: %s", path, spec_err);
    }
    return st;
}

static bool try_read_placement_field(velri_json_reader *r, velri_json_str key,
                                     velri_placement_rule *rule, const char *display_rule,
                                     velri_status *out_st, char *err_msg, size_t err_cap) {
    if (velri_json_str_eq(key, "store")) {
        *out_st = read_placement_field(r, "store", true, &rule->store, display_rule, err_msg, err_cap);
        return true;
    }
    if (velri_json_str_eq(key, "compute")) {
        *out_st = read_placement_field(r, "compute", false, &rule->compute, display_rule, err_msg, err_cap);
        return true;
    }
    if (velri_json_str_eq(key, "prefill-compute")) {
        *out_st = read_placement_field(r, "prefill-compute", false, &rule->prefill_compute,
                                       display_rule, err_msg, err_cap);
        return true;
    }
    return false;
}

typedef struct placement_schema placement_schema;
typedef struct {
    const char *key;
    const placement_schema *child;
} placement_child;

struct placement_schema {
    const placement_child *children;
    uint32_t num_children;
    bool allow_selectors;
};

static const placement_schema k_schema_leaf = {nullptr, 0U, false};
static const placement_child k_indexer_ch[] = {
    {VELRI_SCOPE_KEY_KEYS, &k_schema_leaf},
};
static const placement_schema k_schema_indexer = {k_indexer_ch, ARRAY_COUNT(k_indexer_ch), false};
static const placement_child k_attn_ch[] = {
    {VELRI_SCOPE_KEY_KV_CACHE, &k_schema_leaf},
    {VELRI_SCOPE_KEY_INDEXER, &k_schema_indexer},
};
static const placement_schema k_schema_attn = {k_attn_ch, ARRAY_COUNT(k_attn_ch), false};
static const placement_child k_moe_ch[] = {
    {VELRI_SCOPE_KEY_ROUTER, &k_schema_leaf},
    {VELRI_SCOPE_KEY_SHARED_EXP, &k_schema_leaf},
    {VELRI_SCOPE_KEY_ROUTED_EXP, &k_schema_leaf},
};
static const placement_schema k_schema_moe = {k_moe_ch, ARRAY_COUNT(k_moe_ch), false};
static const placement_child k_ple_ch[] = {
    {VELRI_SCOPE_KEY_TABLE, &k_schema_leaf},
};
static const placement_schema k_schema_ple = {k_ple_ch, ARRAY_COUNT(k_ple_ch), false};
static const placement_child k_hl_ch[] = {
    {VELRI_SCOPE_KEY_ATTENTION, &k_schema_attn},
    {VELRI_SCOPE_KEY_DENSE_MLP, &k_schema_leaf},
    {VELRI_SCOPE_KEY_MOE, &k_schema_moe},
    {VELRI_SCOPE_KEY_GDN, &k_schema_leaf},
    {VELRI_SCOPE_KEY_CONV, &k_schema_leaf},
};
static const placement_schema k_schema_hl_sub = {k_hl_ch, ARRAY_COUNT(k_hl_ch), false};
static const placement_schema k_schema_hl_root = {k_hl_ch, ARRAY_COUNT(k_hl_ch), true};
static const placement_child k_model_ch[] = {
    {VELRI_SCOPE_INPUT_LAYER, &k_schema_leaf},
    {VELRI_SCOPE_LM_HEAD, &k_schema_leaf},
    {VELRI_SCOPE_PLE, &k_schema_ple},
    {VELRI_SCOPE_HIDDEN_LAYERS, &k_schema_hl_root},
};

static_assert(ARRAY_COUNT(k_hl_ch) < 32U, "seen_ch bitmask fits uint32_t");
static_assert(ARRAY_COUNT(k_model_ch) < 32U, "model seen_ch bitmask fits uint32_t");

/* Probe whether a JSON member key starts with an ASCII digit without touching `err_msg`. */
static bool key_starts_with_digit(velri_json_str key) {
    if (key.len == 0U) {
        return false;
    }
    if (key.ptr[0] >= '0' && key.ptr[0] <= '9') {
        return true;
    }
    if (!key.has_escapes) {
        return false;
    }
    char probe[64];
    size_t plen = 0U;
    return velri_json_str_decode(key, probe, sizeof(probe), &plen) == VELRI_OK && plen > 0U &&
           probe[0] >= '0' && probe[0] <= '9';
}

static velri_status parse_placement_node(velri_json_reader *r, velri_engine_config *cfg,
                                         const placement_schema *schema, const char *scope_path,
                                         const char *display_rule, int32_t selector_idx,
                                         char *err_msg, size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: %s", display_rule, r->err_msg);
        return st;
    }
    uint32_t rule_idx = 0U;
    st = append_rule(cfg, scope_path, display_rule, selector_idx, &rule_idx);
    if (st != VELRI_OK) {
        return st;
    }

    uint32_t seen_ch = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (try_read_placement_field(r, key, &cfg->rules[rule_idx], display_rule, &st, err_msg,
                                     err_cap)) {
            if (st != VELRI_OK) {
                return st;
            }
            continue;
        }

        bool matched = false;
        for (uint32_t c = 0U; c < schema->num_children; ++c) {
            if (velri_json_str_eq(key, schema->children[c].key)) {
                if (!check_dup(&seen_ch, 1U << c, display_rule, schema->children[c].key, err_msg,
                               err_cap)) {
                    return VELRI_ERR_INVALID;
                }
                char sub_scope[VELRI_CFG_SCOPE_MAX];
                char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
                const char *sep = (scope_path[0] != '\0') ? "/" : nullptr;
                velri_cfg_join_label4(sub_scope, sizeof(sub_scope), scope_path, sep,
                                      schema->children[c].key, nullptr);
                velri_cfg_join_label4(sub_disp, sizeof(sub_disp), display_rule, ".",
                                      schema->children[c].key, nullptr);
                st = parse_placement_node(r, cfg, schema->children[c].child, sub_scope, sub_disp,
                                          selector_idx, err_msg, err_cap);
                matched = true;
                break;
            }
        }
        if (matched) {
            if (st != VELRI_OK) {
                return st;
            }
            continue;
        }

        if (schema->allow_selectors && key_starts_with_digit(key)) {
            char kbuf[64] = {0};
            st = decode_cstr_buf(key, kbuf, sizeof(kbuf), display_rule, err_msg, err_cap);
            if (st != VELRI_OK) {
                return st;
            }
            int32_t sub_sel = -1;
            st = parse_layer_selector(kbuf, cfg, &sub_sel, err_msg, err_cap);
            if (st != VELRI_OK) {
                return st;
            }
            char sub_disp[VELRI_CFG_RULE_LABEL_MAX];
            velri_cfg_join_label4(sub_disp, sizeof(sub_disp), display_rule, "[", kbuf, "]");
            st = parse_placement_node(r, cfg, &k_schema_hl_sub, scope_path, sub_disp, sub_sel,
                                      err_msg, err_cap);
            if (st != VELRI_OK) {
                return st;
            }
            continue;
        }
        return fail_unknown_key(key, display_rule, err_msg, err_cap);
    }
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: %s", display_rule, r->err_msg);
    }
    return st;
}

/* ---- Top-Level Config Section Parsers ---- */

typedef struct {
    const char *key;
    uint32_t *target;
    uint32_t min_v;
    uint32_t max_v;
    const char *path;
    bool endpoints_only; /* Discrete choice between min_v and max_v. */
} u32_obj_field;

static velri_status parse_u32_object(velri_json_reader *r, const char *obj_name,
                                     const u32_obj_field *fields, uint32_t n_fields, char *err_msg,
                                     size_t err_cap) {
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: %s", obj_name, r->err_msg);
        return st;
    }
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        bool hit = false;
        for (uint32_t i = 0U; i < n_fields; ++i) {
            if (velri_json_str_eq(key, fields[i].key)) {
                if (!check_dup(&seen, 1U << i, obj_name, fields[i].key, err_msg, err_cap)) {
                    return VELRI_ERR_INVALID;
                }
                st = read_u32_range(r, fields[i].target, fields[i].min_v, fields[i].max_v,
                                    fields[i].path, err_msg, err_cap);
                if (st == VELRI_OK && fields[i].endpoints_only &&
                    *fields[i].target != fields[i].min_v && *fields[i].target != fields[i].max_v) {
                    velri_cfg_set_errf(err_msg, err_cap, "%s must be %u or %u", fields[i].path,
                                       fields[i].min_v, fields[i].max_v);
                    return VELRI_ERR_INVALID;
                }
                hit = true;
                break;
            }
        }
        if (!hit) {
            return fail_unknown_key(key, obj_name, err_msg, err_cap);
        }
        if (st != VELRI_OK) {
            return st;
        }
    }
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s: %s", obj_name, r->err_msg);
    }
    return st;
}

static velri_status parse_server_object(velri_json_reader *r, char *err_msg, size_t err_cap) {
    enum {
        SRV_SEEN_HOST = 1U << 0,
        SRV_SEEN_PORT = 1U << 1,
        SRV_SEEN_MAX_CONC = 1U << 2,
    };
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "server: %s", r->err_msg);
        return st;
    }
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "host")) {
            if (!check_dup(&seen, SRV_SEEN_HOST, "server", "host", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            char host[256];
            st = read_cstr_buf(r, host, sizeof(host), "server.host", err_msg, err_cap);
        } else if (velri_json_str_eq(key, "port")) {
            if (!check_dup(&seen, SRV_SEEN_PORT, "server", "port", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            uint32_t port = 0U;
            st = read_u32_range(r, &port, 1U, 65535U, "server.port", err_msg, err_cap);
        } else if (velri_json_str_eq(key, "max-concurrent-requests")) {
            if (!check_dup(&seen, SRV_SEEN_MAX_CONC, "server", "max-concurrent-requests", err_msg,
                           err_cap)) {
                return VELRI_ERR_INVALID;
            }
            uint32_t max_conc = 0U;
            st = read_u32_range(r, &max_conc, 1U, UINT32_MAX, "server.max-concurrent-requests",
                                err_msg, err_cap);
        } else {
            return fail_unknown_key(key, "server", err_msg, err_cap);
        }
        if (st != VELRI_OK) {
            return st;
        }
    }
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "server: %s", r->err_msg);
    }
    return st;
}

static velri_status parse_sampling(velri_json_reader *r, velri_sampling_config *out, char *err_msg,
                                   size_t err_cap) {
    enum {
        SAMP_SEEN_TEMP = 1U << 0,
        SAMP_SEEN_TOP_P = 1U << 1,
        SAMP_SEEN_TOP_K = 1U << 2,
        SAMP_SEEN_MIN_P = 1U << 3,
        SAMP_SEEN_SEED = 1U << 4,
    };
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "model.sampling: %s", r->err_msg);
        return st;
    }
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "temperature")) {
            if (!check_dup(&seen, SAMP_SEEN_TEMP, "model.sampling", "temperature", err_msg,
                           err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = read_sampling_float(r, &out->temperature, velri_sampling_valid_temperature,
                                "model.sampling.temperature", err_msg, err_cap);
        } else if (velri_json_str_eq(key, "top-p")) {
            if (!check_dup(&seen, SAMP_SEEN_TOP_P, "model.sampling", "top-p", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = read_sampling_float(r, &out->top_p, velri_sampling_valid_top_p,
                                     "model.sampling.top-p", err_msg, err_cap);
        } else if (velri_json_str_eq(key, "top-k")) {
            if (!check_dup(&seen, SAMP_SEEN_TOP_K, "model.sampling", "top-k", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = read_u32_range(r, &out->top_k, 0U, UINT32_MAX, "model.sampling.top-k", err_msg,
                                err_cap);
        } else if (velri_json_str_eq(key, "min-p")) {
            if (!check_dup(&seen, SAMP_SEEN_MIN_P, "model.sampling", "min-p", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = read_sampling_float(r, &out->min_p, velri_sampling_valid_probability,
                                     "model.sampling.min-p", err_msg, err_cap);
        } else if (velri_json_str_eq(key, "seed")) {
            if (!check_dup(&seen, SAMP_SEEN_SEED, "model.sampling", "seed", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = velri_json_read_u64(r, &out->seed);
            if (st != VELRI_OK) {
                velri_cfg_set_errf(err_msg, err_cap, "model.sampling.seed: %s", r->err_msg);
            } else {
                out->has_seed = true;
            }
        } else {
            return fail_unknown_key(key, "model.sampling", err_msg, err_cap);
        }
        if (st != VELRI_OK) {
            return st;
        }
    }
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "model.sampling: %s", r->err_msg);
    }
    return st;
}

static velri_status parse_model_object(velri_json_reader *r, velri_engine_config *cfg,
                                       char *err_msg, size_t err_cap) {
    enum {
        MODEL_SEEN_PATH = 1U << 0,
        MODEL_SEEN_SAMPLING = 1U << 1,
    };
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "model: %s", r->err_msg);
        return st;
    }
    uint32_t root_idx = 0U;
    st = append_rule(cfg, VELRI_SCOPE_ROOT, "model", -1, &root_idx);
    if (st != VELRI_OK) {
        return st;
    }

    uint32_t seen_ch = 0U;
    uint32_t seen_meta = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (try_read_placement_field(r, key, &cfg->rules[root_idx], "model", &st, err_msg, err_cap)) {
            if (st != VELRI_OK) {
                return st;
            }
            continue;
        }
        bool hit = false;
        for (uint32_t c = 0U; c < (uint32_t)ARRAY_COUNT(k_model_ch); ++c) {
            if (velri_json_str_eq(key, k_model_ch[c].key)) {
                if (!check_dup(&seen_ch, 1U << c, "model", k_model_ch[c].key, err_msg, err_cap)) {
                    return VELRI_ERR_INVALID;
                }
                char disp[VELRI_CFG_RULE_LABEL_MAX];
                velri_cfg_join_label4(disp, sizeof(disp), "model.", k_model_ch[c].key, nullptr,
                                      nullptr);
                st = parse_placement_node(r, cfg, k_model_ch[c].child, k_model_ch[c].key, disp, -1,
                                          err_msg, err_cap);
                hit = true;
                break;
            }
        }
        if (!hit && velri_json_str_eq(key, "path")) {
            if (!check_dup(&seen_meta, MODEL_SEEN_PATH, "model", "path", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = read_cstr_buf(r, cfg->model_path, sizeof(cfg->model_path), "model.path", err_msg,
                               err_cap);
            hit = true;
        } else if (!hit && velri_json_str_eq(key, "sampling")) {
            if (!check_dup(&seen_meta, MODEL_SEEN_SAMPLING, "model", "sampling", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = parse_sampling(r, &cfg->sampling, err_msg, err_cap);
            hit = true;
        }
        if (!hit) {
            return fail_unknown_key(key, "model", err_msg, err_cap);
        }
        if (st != VELRI_OK) {
            return st;
        }
    }
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "model: %s", r->err_msg);
    }
    return st;
}

static velri_status parse_cpu_backend(velri_json_reader *r, velri_cpu_backend_cfg *out,
                                      char *err_msg, size_t err_cap) {
    enum {
        CPU_SEEN_THREADS = 1U << 0,
        CPU_SEEN_RESERVE = 1U << 1,
        CPU_SEEN_AFFINITY = 1U << 2,
    };
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "backends.cpu: %s", r->err_msg);
        return st;
    }
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "threads")) {
            if (!check_dup(&seen, CPU_SEEN_THREADS, "backends.cpu", "threads", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = read_u32_range(r, &out->threads, 0U, UINT32_MAX, "backends.cpu.threads", err_msg,
                                err_cap);
            out->has_threads = true;
        } else if (velri_json_str_eq(key, "reserve-cores")) {
            if (!check_dup(&seen, CPU_SEEN_RESERVE, "backends.cpu", "reserve-cores", err_msg,
                           err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = read_u32_range(r, &out->reserve_cores, 0U, UINT32_MAX,
                                "backends.cpu.reserve-cores", err_msg, err_cap);
            out->has_reserve_cores = true;
        } else if (velri_json_str_eq(key, "affinity")) {
            if (!check_dup(&seen, CPU_SEEN_AFFINITY, "backends.cpu", "affinity", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            int aff = VELRI_AFFINITY_AUTO;
            st = read_enum_str(r, velri_cfg_affinity_table, velri_cfg_affinity_table_len, &aff,
                               "backends.cpu.affinity", err_msg, err_cap);
            if (st == VELRI_OK) {
                out->affinity = (velri_cpu_affinity_mode)aff;
            }
        } else {
            return fail_unknown_key(key, "backends.cpu", err_msg, err_cap);
        }
        if (st != VELRI_OK) {
            return st;
        }
    }
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "backends.cpu: %s", r->err_msg);
    }
    return st;
}

static velri_status parse_backends_object(velri_json_reader *r, velri_engine_config *cfg,
                                          char *err_msg, size_t err_cap) {
    enum {
        BACKEND_SEEN_CPU = 1U << 0,
        BACKEND_SEEN_VULKAN = 1U << 1,
    };
    velri_json_iter it;
    velri_status st = velri_json_begin_object(r, &it);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "backends: %s", r->err_msg);
        return st;
    }
    const u32_obj_field vk_fields[] = {
        {"prefill-chunk-size", &cfg->vulkan_backend.prefill_chunk_size, 1U, UINT32_MAX,
         "backends.vulkan.prefill-chunk-size", false},
        {"prefill-matmul-bits", &cfg->vulkan_backend.prefill_matmul_bits, 16U, 32U,
         "backends.vulkan.prefill-matmul-bits", true},
        {"prefill-attention-bits", &cfg->vulkan_backend.prefill_attention_bits, 16U, 32U,
         "backends.vulkan.prefill-attention-bits", true},
    };
    uint32_t seen = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "cpu")) {
            if (!check_dup(&seen, BACKEND_SEEN_CPU, "backends", "cpu", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = parse_cpu_backend(r, &cfg->cpu_backend, err_msg, err_cap);
        } else if (velri_json_str_eq(key, "vulkan")) {
            if (!check_dup(&seen, BACKEND_SEEN_VULKAN, "backends", "vulkan", err_msg, err_cap)) {
                return VELRI_ERR_INVALID;
            }
            st = parse_u32_object(r, "backends.vulkan", vk_fields, ARRAY_COUNT(vk_fields), err_msg,
                                  err_cap);
        } else {
            char bname[VELRI_CFG_BACKEND_NAME_MAX];
            st = decode_cstr_buf(key, bname, sizeof(bname), "backends", err_msg, err_cap);
            if (st != VELRI_OK) {
                return st;
            }
            if (!is_valid_ident(bname, strlen(bname))) {
                velri_cfg_set_errf(err_msg, err_cap, "backends: invalid backend section name '%s'",
                                   bname);
                return VELRI_ERR_INVALID;
            }
            if (velri_json_peek(r) != VELRI_JSON_OBJECT) {
                velri_cfg_set_errf(err_msg, err_cap, "backends.%s: expected JSON object", bname);
                return VELRI_ERR_INVALID;
            }
            st = velri_json_skip(r);
            if (st != VELRI_OK) {
                velri_cfg_set_errf(err_msg, err_cap, "backends.%s: %s", bname, r->err_msg);
            }
        }
        if (st != VELRI_OK) {
            return st;
        }
    }
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "backends: %s", r->err_msg);
    }
    return st;
}

/* ---- Public Lifecycle & Parse API ---- */

void velri_engine_config_init(velri_engine_config *cfg) {
    if (!cfg) {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->sampling.top_p = 1.0f;
    cfg->cpu_backend.reserve_cores = VELRI_DEFAULT_CPU_RESERVE_CORES;
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

static velri_status parse_config(const char *json_data, size_t json_len, bool require_model,
                                 velri_engine_config *out_cfg, char *err_msg, size_t err_cap) {
    enum {
        TOP_SEEN_MODEL = 1U << 0,
        TOP_SEEN_CHAT_TEMPLATE = 1U << 1,
        TOP_SEEN_CONTEXT = 1U << 2,
        TOP_SEEN_KV_QUANT = 1U << 3,
        TOP_SEEN_BACKENDS = 1U << 4,
        TOP_SEEN_DISK = 1U << 5,
        TOP_SEEN_SERVER = 1U << 6,
        TOP_SEEN_ATTN_MECH = 1U << 7,
    };

    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!json_data || !out_cfg) {
        velri_cfg_set_errf(err_msg, err_cap, "invalid arguments to velri_engine_config_parse");
        return VELRI_ERR_INVALID;
    }

    velri_engine_config tmp;
    velri_engine_config_init(&tmp);
    velri_json_reader r;
    velri_json_init(&r, json_data, json_len);

    velri_json_iter it;
    velri_status st = velri_json_begin_object(&r, &it);
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s", r.err_msg);
        velri_engine_config_destroy(&tmp);
        return st;
    }

    const u32_obj_field ctx_fields[] = {
        {"size", &tmp.context_size, 1U, UINT32_MAX, "context.size", false},
        {"max-seq-len", &tmp.max_seq_len, 1U, UINT32_MAX, "context.max-seq-len", false},
    };
    const u32_obj_field kvq_fields[] = {
        {"k-bits", &tmp.kv_k_bits, 1U, 16U, "kv-quantization.k-bits", false},
        {"v-bits", &tmp.kv_v_bits, 1U, 16U, "kv-quantization.v-bits", false},
    };
    const u32_obj_field disk_fields[] = {
        {"io-threads", &tmp.disk_io_threads, 1U, UINT32_MAX, "disk.io-threads", false},
    };

    uint32_t seen_top = 0U;
    bool more = false;
    velri_json_str key;
    while ((st = velri_json_next_member(&r, &it, &key, &more)) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "model")) {
            if (!check_dup(&seen_top, TOP_SEEN_MODEL, "root", "model", err_msg, err_cap)) {
                st = VELRI_ERR_INVALID;
            } else {
                st = parse_model_object(&r, &tmp, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "chat-template")) {
            if (!check_dup(&seen_top, TOP_SEEN_CHAT_TEMPLATE, "root", "chat-template", err_msg,
                           err_cap)) {
                st = VELRI_ERR_INVALID;
            } else if (velri_json_peek(&r) != VELRI_JSON_OBJECT) {
                velri_cfg_set_errf(err_msg, err_cap, "chat-template: expected JSON object");
                st = VELRI_ERR_INVALID;
            } else {
                st = velri_json_skip(&r);
                if (st != VELRI_OK) {
                    velri_cfg_set_errf(err_msg, err_cap, "chat-template: %s", r.err_msg);
                }
            }
        } else if (velri_json_str_eq(key, "context")) {
            if (!check_dup(&seen_top, TOP_SEEN_CONTEXT, "root", "context", err_msg, err_cap)) {
                st = VELRI_ERR_INVALID;
            } else {
                st = parse_u32_object(&r, "context", ctx_fields, ARRAY_COUNT(ctx_fields), err_msg,
                                      err_cap);
            }
        } else if (velri_json_str_eq(key, "kv-quantization")) {
            if (!check_dup(&seen_top, TOP_SEEN_KV_QUANT, "root", "kv-quantization", err_msg,
                           err_cap)) {
                st = VELRI_ERR_INVALID;
            } else {
                st = parse_u32_object(&r, "kv-quantization", kvq_fields, ARRAY_COUNT(kvq_fields),
                                      err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "backends")) {
            if (!check_dup(&seen_top, TOP_SEEN_BACKENDS, "root", "backends", err_msg, err_cap)) {
                st = VELRI_ERR_INVALID;
            } else {
                st = parse_backends_object(&r, &tmp, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "disk")) {
            if (!check_dup(&seen_top, TOP_SEEN_DISK, "root", "disk", err_msg, err_cap)) {
                st = VELRI_ERR_INVALID;
            } else {
                st = parse_u32_object(&r, "disk", disk_fields, ARRAY_COUNT(disk_fields), err_msg,
                                      err_cap);
            }
        } else if (velri_json_str_eq(key, "server")) {
            if (!check_dup(&seen_top, TOP_SEEN_SERVER, "root", "server", err_msg, err_cap)) {
                st = VELRI_ERR_INVALID;
            } else {
                st = parse_server_object(&r, err_msg, err_cap);
            }
        } else if (velri_json_str_eq(key, "attention-mechanism")) {
            if (!check_dup(&seen_top, TOP_SEEN_ATTN_MECH, "root", "attention-mechanism", err_msg,
                           err_cap)) {
                st = VELRI_ERR_INVALID;
            } else {
                int mech = VELRI_ATTN_MECH_AUTO;
                st = read_enum_str(&r, k_attn_mech_table, ARRAY_COUNT(k_attn_mech_table), &mech,
                                   "attention-mechanism", err_msg, err_cap);
                if (st == VELRI_OK) {
                    tmp.attn_mechanism = (velri_attn_mechanism)mech;
                }
            }
        } else {
            st = fail_unknown_key(key, nullptr, err_msg, err_cap);
        }
        if (st != VELRI_OK) {
            break;
        }
    }

    if (st == VELRI_OK) {
        st = velri_json_finish(&r);
    }
    if (st != VELRI_OK) {
        velri_cfg_set_errf(err_msg, err_cap, "%s", r.err_msg);
        velri_engine_config_destroy(&tmp);
        return st;
    }
    if (require_model && (seen_top & TOP_SEEN_MODEL) == 0U) {
        velri_cfg_set_errf(err_msg, err_cap, "missing required top-level 'model' object");
        velri_engine_config_destroy(&tmp);
        return VELRI_ERR_INVALID;
    }

    velri_engine_config_destroy(out_cfg);
    *out_cfg = tmp;
    return VELRI_OK;
}

velri_status velri_engine_config_parse(const char *json_data, size_t json_len,
                                       velri_engine_config *out_cfg, char *err_msg, size_t err_cap) {
    return parse_config(json_data, json_len, true, out_cfg, err_msg, err_cap);
}

velri_status velri_cpu_config_parse(const char *json_data, size_t json_len,
                                    velri_cpu_backend_cfg *out, char *err_msg, size_t err_cap) {
    if (!out) {
        return VELRI_ERR_INVALID;
    }
    velri_engine_config cfg;
    velri_engine_config_init(&cfg);
    velri_status st = parse_config(json_data, json_len, false, &cfg, err_msg, err_cap);
    if (st == VELRI_OK) {
        *out = cfg.cpu_backend;
    }
    velri_engine_config_destroy(&cfg);
    return st;
}

velri_status velri_engine_config_validate_layers(const velri_engine_config *cfg,
                                                 uint32_t num_hidden_layers, bool strict,
                                                 char *err_msg, size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!cfg || num_hidden_layers == 0U) {
        velri_cfg_set_errf(err_msg, err_cap, "invalid arguments to validate_layers");
        return VELRI_ERR_INVALID;
    }
    if (!strict) {
        return VELRI_OK;
    }
    for (uint32_t i = 0U; i < cfg->num_selectors; ++i) {
        const velri_layer_selector *sel = &cfg->selectors[i];
        for (uint32_t r = 0U; r < sel->num_ranges; ++r) {
            if (sel->ranges[r].end >= num_hidden_layers) {
                velri_cfg_set_errf(err_msg, err_cap,
                                   "layer selector '%s' references layer %u, out of range for %u layers",
                                   sel->raw_key, sel->ranges[r].end, num_hidden_layers);
                return VELRI_ERR_INVALID;
            }
        }
    }
    return VELRI_OK;
}
