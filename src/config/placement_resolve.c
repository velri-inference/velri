#include "config/placement_internal.h"

#include <stdio.h>
#include <string.h>

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))

/* ---- Shared Canonical Layer Scope Metadata ---- */

const velri_cfg_layer_scope_meta velri_cfg_layer_scopes[] = {
    {VELRI_SCOPE_HIDDEN_LAYERS, nullptr, false},
    {VELRI_SCOPE_HL_ATTENTION, "attention", true},
    {VELRI_SCOPE_HL_KV_CACHE, "kv-cache", true},
    {VELRI_SCOPE_HL_INDEXER, "attention.indexer", false},
    {VELRI_SCOPE_HL_INDEXER_KEYS, "attention.indexer.keys", false},
    {VELRI_SCOPE_HL_DENSE_MLP, "dense-mlp", true},
    {VELRI_SCOPE_HL_MOE, "moe", false},
    {VELRI_SCOPE_HL_MOE_ROUTER, "moe.router", false},
    {VELRI_SCOPE_HL_MOE_SHARED, "moe.shared-experts", false},
    {VELRI_SCOPE_HL_MOE_ROUTED, "moe.routed-experts", false},
    {VELRI_SCOPE_HL_GDN, "gdn", false},
    {VELRI_SCOPE_HL_CONV, "conv", false},
};
const size_t velri_cfg_layer_scopes_len = ARRAY_COUNT(velri_cfg_layer_scopes);
static_assert(ARRAY_COUNT(velri_cfg_layer_scopes) <= VELRI_CFG_MAX_LAYER_SCOPES,
              "velri_cfg_layer_scopes exceeds VELRI_CFG_MAX_LAYER_SCOPES");

/* ---- Context Binding & Cascading Resolution ---- */

static velri_status resolve_spec_to_dev(const velri_context *ctx, velri_device_id host_ram_dev,
                                        bool has_host_ram_dev, const velri_device_spec *spec,
                                        const char *rule_label, const char *field_name,
                                        velri_device_id *out_dev, bool *out_is_disk, char *err_msg,
                                        size_t err_cap) {
    if (out_is_disk) {
        *out_is_disk = false;
    }
    if (spec->kind == VELRI_LOC_UNSET) {
        return VELRI_OK;
    }
    if (spec->kind == VELRI_LOC_DISK) {
        if (out_is_disk) {
            *out_is_disk = true;
        }
        if (out_dev) {
            *out_dev = VELRI_INVALID_ID;
        }
        return VELRI_OK;
    }
    if (spec->kind == VELRI_LOC_RAM) {
        if (!has_host_ram_dev) {
            velri_cfg_set_errf(err_msg, err_cap,
                               "%s.%s: 'ram:0' requires a host-memory device in context",
                               rule_label, field_name);
            return VELRI_ERR_NOT_FOUND;
        }
        if (out_dev) {
            *out_dev = host_ram_dev;
        }
        return VELRI_OK;
    }

    velri_device_id dev_id = VELRI_INVALID_ID;
    if (velri_context_find_device(ctx, spec->backend_name, spec->local_index, &dev_id) != VELRI_OK) {
        uint32_t n_dev = velri_context_device_count(ctx);
        uint32_t cnt = 0U;
        for (uint32_t d = 0U; d < n_dev; ++d) {
            velri_device_caps caps;
            if (velri_context_device_caps(ctx, d, &caps) == VELRI_OK &&
                strcmp(caps.backend_name, spec->backend_name) == 0) {
                cnt++;
            }
        }
        if (cnt == 0U) {
            velri_cfg_set_errf(err_msg, err_cap,
                               "%s.%s: backend '%s' is not available or not enabled in this build/context",
                               rule_label, field_name, spec->backend_name);
        } else {
            velri_cfg_set_errf(err_msg, err_cap,
                               "%s.%s: device '%s:%u' not found (%u '%s' device(s) available)",
                               rule_label, field_name, spec->backend_name, spec->local_index, cnt,
                               spec->backend_name);
        }
        return VELRI_ERR_NOT_FOUND;
    }
    if (out_dev) {
        *out_dev = dev_id;
    }
    return VELRI_OK;
}

velri_status velri_placement_plan_bind(const velri_engine_config *cfg, const velri_context *ctx,
                                       velri_device_id fallback_dev, velri_placement_plan *out_plan,
                                       char *err_msg, size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!cfg || !ctx || !out_plan) {
        velri_cfg_set_errf(err_msg, err_cap, "invalid arguments to velri_placement_plan_bind");
        return VELRI_ERR_INVALID;
    }
    uint32_t n_dev = velri_context_device_count(ctx);
    if (fallback_dev >= n_dev) {
        velri_cfg_set_errf(err_msg, err_cap,
                           "fallback device %u out of range (context has %u devices)", fallback_dev,
                           n_dev);
        return VELRI_ERR_NOT_FOUND;
    }

    velri_device_id host_dev = VELRI_INVALID_ID;
    bool has_host = false;
    bool has_disk = false;
    for (uint32_t d = 0U; d < n_dev; ++d) {
        velri_device_caps caps;
        if (velri_context_device_caps(ctx, d, &caps) == VELRI_OK && caps.is_host_memory) {
            host_dev = d;
            has_host = true;
            break;
        }
    }

    for (uint32_t i = 0U; i < cfg->num_rules; ++i) {
        const velri_placement_rule *rule = &cfg->rules[i];
        bool is_disk = false;
        velri_status st = resolve_spec_to_dev(ctx, host_dev, has_host, &rule->store,
                                              rule->display_rule, "store", nullptr, &is_disk,
                                              err_msg, err_cap);
        if (st == VELRI_OK) {
            st = resolve_spec_to_dev(ctx, host_dev, has_host, &rule->compute, rule->display_rule,
                                     "compute", nullptr, nullptr, err_msg, err_cap);
        }
        if (st == VELRI_OK) {
            st = resolve_spec_to_dev(ctx, host_dev, has_host, &rule->prefill_compute,
                                     rule->display_rule, "prefill-compute", nullptr, nullptr,
                                     err_msg, err_cap);
        }
        if (st != VELRI_OK) {
            return st;
        }
        if (is_disk) {
            has_disk = true;
        }
    }

    *out_plan = (velri_placement_plan){
        .cfg = cfg,
        .ctx = ctx,
        .default_store_dev = fallback_dev,
        .default_compute_dev = fallback_dev,
        .host_ram_dev = host_dev,
        .has_host_ram_dev = has_host,
        .has_disk_rules = has_disk,
    };

    velri_resolved_placement root_res;
    if (velri_placement_resolve(out_plan, VELRI_SCOPE_ROOT, -1, &root_res, err_msg, err_cap) ==
        VELRI_OK) {
        if (!root_res.store_is_disk && root_res.store_dev != VELRI_INVALID_ID) {
            out_plan->default_store_dev = root_res.store_dev;
        }
        if (root_res.compute_dev != VELRI_INVALID_ID) {
            out_plan->default_compute_dev = root_res.compute_dev;
        }
    }
    return VELRI_OK;
}

typedef struct {
    bool store;
    bool compute;
    bool prefill;
} placement_found;

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

static void apply_rule_cascade(const velri_placement_rule *rule, velri_resolved_placement *out,
                               placement_found *found) {
    if (!rule) {
        return;
    }
    if (!found->store && rule->store.kind != VELRI_LOC_UNSET) {
        found->store = true;
        out->store_spec = rule->store;
        velri_cfg_join_label4(out->store_source, sizeof(out->store_source), rule->display_rule,
                              ".store", nullptr, nullptr);
    }
    if (!found->compute && rule->compute.kind != VELRI_LOC_UNSET) {
        found->compute = true;
        out->compute_spec = rule->compute;
        velri_cfg_join_label4(out->compute_source, sizeof(out->compute_source), rule->display_rule,
                              ".compute", nullptr, nullptr);
    }
    if (!found->prefill) {
        if (rule->prefill_compute.kind != VELRI_LOC_UNSET) {
            found->prefill = true;
            out->prefill_spec = rule->prefill_compute;
            velri_cfg_join_label4(out->prefill_source, sizeof(out->prefill_source),
                                  rule->display_rule, ".prefill-compute", nullptr, nullptr);
        } else if (rule->compute.kind != VELRI_LOC_UNSET) {
            found->prefill = true;
            out->prefill_spec = rule->compute;
            velri_cfg_join_label4(out->prefill_source, sizeof(out->prefill_source),
                                  rule->display_rule, ".compute", nullptr, nullptr);
        }
    }
}

static void fill_default_spec(const velri_context *ctx, velri_device_id dev,
                              velri_device_spec *spec, char *src, size_t src_cap) {
    velri_device_caps caps;
    if (velri_context_device_caps(ctx, dev, &caps) == VELRI_OK) {
        spec->kind = VELRI_LOC_BACKEND;
        snprintf(spec->backend_name, sizeof(spec->backend_name), "%s", caps.backend_name);
        spec->local_index = caps.local_index;
    }
    snprintf(src, src_cap, "<default>");
}

velri_status velri_placement_resolve(const velri_placement_plan *plan, const char *scope_path,
                                     int32_t layer_idx, velri_resolved_placement *out,
                                     char *err_msg, size_t err_cap) {
    if (err_msg && err_cap > 0U) {
        err_msg[0] = '\0';
    }
    if (!plan || !plan->cfg || !plan->ctx || !out) {
        velri_cfg_set_errf(err_msg, err_cap, "invalid arguments to velri_placement_resolve");
        return VELRI_ERR_INVALID;
    }
    memset(out, 0, sizeof(*out));
    const char *path = scope_path ? scope_path : VELRI_SCOPE_ROOT;
    size_t plen = strlen(path);

    size_t prefix_lens[16];
    uint32_t num_prefixes = 0U;
    for (size_t cur = plen; cur > 0U && num_prefixes < ARRAY_COUNT(prefix_lens);) {
        prefix_lens[num_prefixes++] = cur;
        while (cur > 0U && path[cur - 1U] != '/') {
            cur--;
        }
        if (cur > 0U) {
            cur--;
        }
    }

    placement_found found = {false, false, false};
    const size_t k_hl_len = sizeof(VELRI_SCOPE_HIDDEN_LAYERS) - 1U;
    bool in_hidden = (plen >= k_hl_len && memcmp(path, VELRI_SCOPE_HIDDEN_LAYERS, k_hl_len) == 0 &&
                      (plen == k_hl_len || path[k_hl_len] == '/'));
    if (in_hidden && layer_idx >= 0) {
        for (uint32_t s = 0U; s < plan->cfg->num_selectors; ++s) {
            if (velri_cfg_selector_matches_layer(&plan->cfg->selectors[s], (uint32_t)layer_idx)) {
                for (uint32_t p = 0U; p < num_prefixes; ++p) {
                    apply_rule_cascade(find_rule(plan->cfg, path, prefix_lens[p], (int32_t)s), out,
                                       &found);
                }
                break;
            }
        }
    }
    for (uint32_t p = 0U; p < num_prefixes; ++p) {
        apply_rule_cascade(find_rule(plan->cfg, path, prefix_lens[p], -1), out, &found);
    }
    apply_rule_cascade(find_rule(plan->cfg, VELRI_SCOPE_ROOT, 0U, -1), out, &found);

    velri_status st = VELRI_OK;
    if (found.store) {
        st = resolve_spec_to_dev(plan->ctx, plan->host_ram_dev, plan->has_host_ram_dev,
                                 &out->store_spec, out->store_source, "store", &out->store_dev,
                                 &out->store_is_disk, err_msg, err_cap);
        if (st != VELRI_OK) {
            return st;
        }
    } else {
        out->store_dev = plan->default_store_dev;
        fill_default_spec(plan->ctx, out->store_dev, &out->store_spec, out->store_source,
                          sizeof(out->store_source));
    }

    if (found.compute) {
        st = resolve_spec_to_dev(plan->ctx, plan->host_ram_dev, plan->has_host_ram_dev,
                                 &out->compute_spec, out->compute_source, "compute",
                                 &out->compute_dev, nullptr, err_msg, err_cap);
        if (st != VELRI_OK) {
            return st;
        }
    } else {
        out->compute_dev = plan->default_compute_dev;
        fill_default_spec(plan->ctx, out->compute_dev, &out->compute_spec, out->compute_source,
                          sizeof(out->compute_source));
    }

    if (found.prefill) {
        return resolve_spec_to_dev(plan->ctx, plan->host_ram_dev, plan->has_host_ram_dev,
                                   &out->prefill_spec, out->prefill_source, "prefill-compute",
                                   &out->prefill_dev, nullptr, err_msg, err_cap);
    }
    out->prefill_spec = out->compute_spec;
    out->prefill_dev = out->compute_dev;
    snprintf(out->prefill_source, sizeof(out->prefill_source), "%s", out->compute_source);
    return VELRI_OK;
}

bool velri_placement_has_distinct_prefill_for_scopes(const velri_placement_plan *plan,
                                                     const char *const *root_scopes,
                                                     uint32_t num_root_scopes,
                                                     const char *const *layer_scopes,
                                                     uint32_t num_layer_scopes,
                                                     uint32_t num_hidden_layers) {
    if (!plan || !plan->cfg) {
        return false;
    }
    velri_resolved_placement res;
    for (uint32_t i = 0U; i < num_root_scopes; ++i) {
        if (velri_placement_resolve(plan, root_scopes[i], -1, &res, nullptr, 0U) == VELRI_OK &&
            res.prefill_dev != res.compute_dev) {
            return true;
        }
    }
    for (uint32_t l = 0U; l < num_hidden_layers; ++l) {
        for (uint32_t i = 0U; i < num_layer_scopes; ++i) {
            if (velri_placement_resolve(plan, layer_scopes[i], (int32_t)l, &res, nullptr, 0U) ==
                    VELRI_OK &&
                res.prefill_dev != res.compute_dev) {
                return true;
            }
        }
    }
    return false;
}

bool velri_placement_has_distinct_prefill(const velri_placement_plan *plan,
                                          uint32_t num_hidden_layers) {
    static const char *const k_all_roots[] = {
        VELRI_SCOPE_ROOT,      VELRI_SCOPE_INPUT_LAYER, VELRI_SCOPE_LM_HEAD,
        VELRI_SCOPE_PLE,       VELRI_SCOPE_PLE_TABLE,
    };
    const char *all_layers[VELRI_CFG_MAX_LAYER_SCOPES];
    uint32_t n_layers = (uint32_t)velri_cfg_layer_scopes_len;
    for (uint32_t i = 0U; i < n_layers; ++i) {
        all_layers[i] = velri_cfg_layer_scopes[i].scope;
    }
    return velri_placement_has_distinct_prefill_for_scopes(
        plan, k_all_roots, (uint32_t)ARRAY_COUNT(k_all_roots), all_layers, n_layers,
        num_hidden_layers);
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
    if (st == VELRI_OK && res.store_is_disk) {
        st = VELRI_ERR_UNSUPPORTED;
        velri_cfg_set_errf(ad->err_msg, sizeof(ad->err_msg),
                           "scope '%s' (layer %d): 'disk' storage streaming is not yet implemented",
                           scope_path ? scope_path : "", layer_idx);
    }
    if (st != VELRI_OK) {
        ad->err = st;
        if (ad->builder) {
            velri_b_set_error(ad->builder, st);
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
