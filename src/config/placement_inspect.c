#include "config/placement_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))

void velri_cfg_sb_appendf(velri_cfg_str_buf *sb, const char *fmt, ...) {
    if (!sb || !sb->buf || sb->cap == 0U || sb->pos >= sb->cap) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(sb->buf + sb->pos, sb->cap - sb->pos, fmt, ap);
    va_end(ap);
    if (n > 0) {
        size_t avail = sb->cap - 1U - sb->pos;
        sb->pos += ((size_t)n < avail) ? (size_t)n : avail;
    }
}

static bool spec_eq(const velri_device_spec *a, const velri_device_spec *b) {
    return a->kind == b->kind && a->local_index == b->local_index &&
           strcmp(a->backend_name, b->backend_name) == 0;
}

static bool resolved_eq(const velri_resolved_placement *a, const velri_resolved_placement *b) {
    return a->store_dev == b->store_dev && a->compute_dev == b->compute_dev &&
           a->prefill_dev == b->prefill_dev && a->store_is_disk == b->store_is_disk &&
           spec_eq(&a->store_spec, &b->store_spec) &&
           spec_eq(&a->compute_spec, &b->compute_spec) &&
           spec_eq(&a->prefill_spec, &b->prefill_spec) &&
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

static void format_resolved_line(velri_cfg_str_buf *sb, const char *label,
                                 const velri_resolved_placement *r) {
    char s_buf[48];
    char c_buf[48];
    char p_buf[48];
    velri_device_spec_format(&r->store_spec, s_buf, sizeof(s_buf));
    velri_device_spec_format(&r->compute_spec, c_buf, sizeof(c_buf));
    velri_device_spec_format(&r->prefill_spec, p_buf, sizeof(p_buf));
    velri_cfg_sb_appendf(sb,
                         "  %-32s store=%-10s (%s) | compute=%-10s (%s) | prefill=%-10s (%s)\n",
                         label, s_buf, r->store_source, c_buf, r->compute_source, p_buf,
                         r->prefill_source);
}

static void format_devices_and_backends(velri_cfg_str_buf *sb, const velri_placement_plan *plan,
                                        uint32_t num_hidden_layers) {
    velri_cfg_sb_appendf(sb, "=== Velri Placement & Context Inspect ===\n");
    if (plan->cfg->model_path[0] != '\0') {
        velri_cfg_sb_appendf(sb, "Model path: %s\n", plan->cfg->model_path);
    }

    uint32_t n_dev = velri_context_device_count(plan->ctx);
    velri_cfg_sb_appendf(sb, "Devices (%u):\n", n_dev);
    for (uint32_t d = 0U; d < n_dev; ++d) {
        velri_device_caps caps;
        if (velri_context_device_caps(plan->ctx, d, &caps) == VELRI_OK) {
            velri_cfg_sb_appendf(sb, "  [%u] %s:%u - %s (host=%s, uma=%s)\n", d, caps.backend_name,
                                 caps.local_index, caps.device_name,
                                 caps.is_host_memory ? "yes" : "no", caps.is_uma ? "yes" : "no");
        }
    }

    const char *aff_str = velri_cfg_enum_name(velri_cfg_affinity_table, velri_cfg_affinity_table_len,
                                              (int)plan->cfg->cpu_backend.affinity, "auto");
    velri_cfg_sb_appendf(
        sb, "Backends: cpu(threads=%u, reserve=%u, affinity=%s), vulkan(prefill_chunk=%u)\n",
        plan->cfg->cpu_backend.threads, plan->cfg->cpu_backend.reserve_cores, aff_str,
        plan->cfg->vulkan_backend.prefill_chunk_size);

    if (num_hidden_layers > 0U) {
        for (uint32_t s = 0U; s < plan->cfg->num_selectors; ++s) {
            const velri_layer_selector *sel = &plan->cfg->selectors[s];
            for (uint32_t r = 0U; r < sel->num_ranges; ++r) {
                if (sel->ranges[r].end >= num_hidden_layers) {
                    velri_cfg_sb_appendf(
                        sb, "  [note] layer selector '%s' exceeds %u model layers\n", sel->raw_key,
                        num_hidden_layers);
                    break;
                }
            }
        }
    }
}

typedef struct {
    const char *scope;
    const char *label;
    bool optional;
} root_display_entry;

static void format_resolved_scopes(velri_cfg_str_buf *sb, const velri_placement_plan *plan,
                                   uint32_t num_hidden_layers) {
    static const root_display_entry k_root_disp[] = {
        {VELRI_SCOPE_INPUT_LAYER, "input-layer", false},
        {VELRI_SCOPE_LM_HEAD, "lm-head", false},
        {VELRI_SCOPE_PLE, "ple", true},
        {VELRI_SCOPE_PLE_TABLE, "ple.table", true},
    };

    velri_cfg_sb_appendf(sb, "Resolved Placement:\n");
    velri_resolved_placement res;
    for (size_t i = 0U; i < ARRAY_COUNT(k_root_disp); ++i) {
        if ((!k_root_disp[i].optional || config_has_scope_rule(plan->cfg, k_root_disp[i].scope)) &&
            velri_placement_resolve(plan, k_root_disp[i].scope, -1, &res, nullptr, 0U) == VELRI_OK) {
            format_resolved_line(sb, k_root_disp[i].label, &res);
        }
    }

    uint32_t active[VELRI_CFG_MAX_LAYER_SCOPES];
    uint32_t num_active = 0U;
    for (uint32_t i = 0U;
         i < (uint32_t)velri_cfg_layer_scopes_len && num_active < VELRI_CFG_MAX_LAYER_SCOPES; ++i) {
        if (velri_cfg_layer_scopes[i].suffix &&
            (velri_cfg_layer_scopes[i].always_show ||
             config_has_scope_rule(plan->cfg, velri_cfg_layer_scopes[i].scope))) {
            active[num_active++] = i;
        }
    }

    for (uint32_t l = 0U; l < num_hidden_layers; ++l) {
        velri_resolved_placement base[VELRI_CFG_MAX_LAYER_SCOPES];
        for (uint32_t k = 0U; k < num_active; ++k) {
            velri_placement_resolve(plan, velri_cfg_layer_scopes[active[k]].scope, (int32_t)l,
                                    &base[k], nullptr, 0U);
        }
        uint32_t end_l = l;
        while (end_l + 1U < num_hidden_layers) {
            bool same = true;
            for (uint32_t k = 0U; k < num_active && same; ++k) {
                velri_resolved_placement nr;
                velri_placement_resolve(plan, velri_cfg_layer_scopes[active[k]].scope,
                                        (int32_t)(end_l + 1U), &nr, nullptr, 0U);
                same = resolved_eq(&base[k], &nr);
            }
            if (!same) {
                break;
            }
            end_l++;
        }
        char lbl[64];
        for (uint32_t k = 0U; k < num_active; ++k) {
            snprintf(lbl, sizeof(lbl), "layers[%u..%u].%s", l, end_l,
                     velri_cfg_layer_scopes[active[k]].suffix);
            format_resolved_line(sb, lbl, &base[k]);
        }
        l = end_l;
    }
}

static void format_memory_plan(velri_cfg_str_buf *sb, const velri_context *ctx,
                               const velri_memory_report *mem_report) {
    velri_cfg_sb_appendf(sb, "Memory Plan: total=%zu B | ram_staging=%zu B | ram_transfers=%u\n",
                         mem_report->total_physical_bytes, mem_report->ram_staging_bytes,
                         mem_report->num_ram_transfers);
    uint32_t nd = (mem_report->num_devices <= VELRI_MAX_REPORT_DEVICES)
                      ? mem_report->num_devices
                      : (uint32_t)VELRI_MAX_REPORT_DEVICES;
    for (uint32_t d = 0U; d < nd; ++d) {
        const velri_device_memory_report *dr = &mem_report->devices[d];
        velri_device_caps caps;
        bool ok = (velri_context_device_caps(ctx, dr->device_id, &caps) == VELRI_OK);
        velri_cfg_sb_appendf(sb,
                             "  %s:%u | weights=%zu B | state=%zu B | io=%zu B | scratch=%zu B "
                             "(unpacked=%zu B) | ovh=%zu B\n",
                             ok ? caps.backend_name : "dev", ok ? caps.local_index : dr->device_id,
                             dr->weight_bytes, dr->state_bytes, dr->input_output_bytes,
                             dr->scratch_packed_bytes, dr->scratch_unpacked_bytes,
                             dr->allocator_overhead_bytes);
    }
}

size_t velri_placement_format_inspect_placement(const velri_placement_plan *plan,
                                                uint32_t num_hidden_layers, char *out_buf,
                                                size_t out_cap) {
    if (!plan || !plan->cfg || !plan->ctx || !out_buf || out_cap == 0U) {
        return 0U;
    }
    out_buf[0] = '\0';
    velri_cfg_str_buf sb = {.buf = out_buf, .cap = out_cap, .pos = 0U};
    format_devices_and_backends(&sb, plan, num_hidden_layers);
    format_resolved_scopes(&sb, plan, num_hidden_layers);
    return sb.pos;
}

size_t velri_placement_format_inspect(const velri_placement_plan *plan, uint32_t num_hidden_layers,
                                      const velri_memory_report *mem_report, char *out_buf,
                                      size_t out_cap) {
    if (!plan || !plan->cfg || !plan->ctx || !out_buf || out_cap == 0U) {
        return 0U;
    }
    out_buf[0] = '\0';
    velri_cfg_str_buf sb = {.buf = out_buf, .cap = out_cap, .pos = 0U};
    format_devices_and_backends(&sb, plan, num_hidden_layers);
    format_resolved_scopes(&sb, plan, num_hidden_layers);
    if (mem_report) {
        format_memory_plan(&sb, plan->ctx, mem_report);
    }
    return sb.pos;
}
