#ifndef VELRI_CONFIG_PLACEMENT_H
#define VELRI_CONFIG_PLACEMENT_H

#include <velri_lib/velri_lib.h>

/*
 * Velri Engine Configuration & Cascading Device Placement (`docs/EN/blocks-and-config.md §2`).
 *
 * Parses JSON configs (via `src/format/json.{h,c}` with 0 external dependencies), validates
 * schema keys, device specs (`<backend>:<index>`, `ram:0`, `disk`, migration errors for
 * legacy `cuda:N`/`vram:N`/`device:N`), and non-overlapping layer selectors (`"N"`, `"N-M"`).
 *
 * Resolves cascading `store`, `compute`, and optional `prefill-compute` (which inherits
 * from `compute` when omitted) against a live `velri_context`, provides a direct adapter
 * for `velri_builder`, and formats human-readable `inspect` reports.
 */

enum {
    VELRI_CFG_PATH_MAX = 512,
    VELRI_CFG_BACKEND_NAME_MAX = 32,
    VELRI_CFG_SCOPE_MAX = 128,
    VELRI_CFG_RULE_LABEL_MAX = 160,
    VELRI_CFG_ERR_MAX = 256,
    VELRI_DEFAULT_CPU_RESERVE_CORES = 1
};

typedef enum {
    VELRI_LOC_UNSET = 0,
    VELRI_LOC_BACKEND = 1, /* "<backend>:<index>" e.g. "cpu:0", "vulkan:0" */
    VELRI_LOC_RAM = 2,     /* "ram:0" (store-only; maps to host-memory device) */
    VELRI_LOC_DISK = 3     /* "disk"  (store-only; streaming storage) */
} velri_loc_kind;

typedef struct {
    velri_loc_kind kind;
    char backend_name[VELRI_CFG_BACKEND_NAME_MAX];
    uint32_t local_index;
} velri_device_spec;

typedef enum {
    VELRI_AFFINITY_AUTO = 0,
    VELRI_AFFINITY_NONE = 1,
    VELRI_AFFINITY_REQUIRED = 2
} velri_cpu_affinity_mode;

typedef enum {
    VELRI_ATTN_MECH_AUTO = 0,
    VELRI_ATTN_MECH_FLASH = 1,
    VELRI_ATTN_MECH_LATENT_KV = 2
} velri_attn_mechanism;

typedef struct {
    float temperature;
    float top_p;
    uint32_t top_k;
    float min_p;
    uint64_t seed;
    bool has_seed;
} velri_sampling_config;

typedef struct {
    uint32_t threads;       /* 0 => auto */
    uint32_t reserve_cores; /* default 1 */
    velri_cpu_affinity_mode affinity;
    bool has_threads;
    bool has_reserve_cores;
} velri_cpu_backend_cfg;

typedef struct {
    uint32_t prefill_chunk_size; /* 0 => default */
} velri_vulkan_backend_cfg;

typedef struct {
    uint32_t start;
    uint32_t end; /* inclusive */
} velri_layer_range;

typedef struct {
    char raw_key[64];
    velri_layer_range *ranges;
    uint32_t num_ranges;
} velri_layer_selector;

typedef struct {
    char scope_path[VELRI_CFG_SCOPE_MAX];
    char display_rule[VELRI_CFG_RULE_LABEL_MAX];
    int32_t selector_idx; /* -1 for non-selector rules, >= 0 into selectors[] */
    velri_device_spec store;
    velri_device_spec compute;
    velri_device_spec prefill_compute;
} velri_placement_rule;

typedef struct {
    char model_path[VELRI_CFG_PATH_MAX];
    velri_sampling_config sampling;
    uint32_t context_size;    /* 0 => unset */
    uint32_t max_seq_len;     /* 0 => unset */
    uint32_t kv_k_bits;       /* 0 => unset / full precision */
    uint32_t kv_v_bits;       /* 0 => unset / full precision */
    velri_attn_mechanism attn_mechanism;

    velri_cpu_backend_cfg cpu_backend;
    velri_vulkan_backend_cfg vulkan_backend;
    uint32_t disk_io_threads;

    velri_layer_selector *selectors;
    uint32_t num_selectors;
    uint32_t cap_selectors;

    velri_placement_rule *rules;
    uint32_t num_rules;
    uint32_t cap_rules;
} velri_engine_config;

/* Resolved placement for one (scope_path, layer_idx) query. */
typedef struct {
    velri_device_spec store_spec;
    velri_device_spec compute_spec;
    velri_device_spec prefill_spec;

    velri_device_id store_dev;
    velri_device_id compute_dev;
    velri_device_id prefill_dev;
    bool store_is_disk;

    char store_source[VELRI_CFG_RULE_LABEL_MAX];
    char compute_source[VELRI_CFG_RULE_LABEL_MAX];
    char prefill_source[VELRI_CFG_RULE_LABEL_MAX];
} velri_resolved_placement;

/* Placement plan bound to a live `velri_context` (device IDs verified against registry). */
typedef struct {
    const velri_engine_config *cfg; /* Borrowed; must outlive plan */
    const velri_context *ctx;       /* Borrowed; must outlive plan */
    velri_device_id default_store_dev;
    velri_device_id default_compute_dev;
    velri_device_id host_ram_dev;
    bool has_host_ram_dev;
    bool has_disk_rules;
} velri_placement_plan;

/* Reports whether compute placements actually requested during graph construction differ. */
typedef struct {
    const velri_placement_plan *plan;
    bool is_prefill;
    bool distinct_compute;
    char err_msg[VELRI_CFG_ERR_MAX];
} velri_placement_builder_adapter;

/* ---- Lifecycle & Parsing ---- */

void velri_engine_config_init(velri_engine_config *cfg);
void velri_engine_config_destroy(velri_engine_config *cfg);

/* Parse JSON text buffer into an initialized `out_cfg` (must be prepared with
 * `velri_engine_config_init` beforehand; any prior contents in `out_cfg` are replaced on success).
 * On failure, `out_cfg` is left untouched and a diagnostic is written to `err_msg` (if non-null). */
velri_status velri_engine_config_parse(const char *json_data, size_t json_len,
                                       velri_engine_config *out_cfg, char *err_msg,
                                       size_t err_cap);

/* The same configuration schema before context creation; the model section is optional. */
velri_status velri_cpu_config_parse(const char *json_data, size_t json_len,
                                    velri_cpu_backend_cfg *out, char *err_msg, size_t err_cap);

/* Parse a device spec string ("cpu:0", "vulkan:0", "ram:0", "disk"). */
velri_status velri_device_spec_parse(const char *text, bool allow_store_only,
                                     velri_device_spec *out_spec, char *err_msg, size_t err_cap);

/* Format a device spec into a human-readable string (e.g. "vulkan:0", "ram:0", "disk", "<unset>"). */
void velri_device_spec_format(const velri_device_spec *spec, char *out, size_t out_cap);

/* Validate layer selectors against a concrete model's `num_hidden_layers`. */
velri_status velri_engine_config_validate_layers(const velri_engine_config *cfg,
                                                 uint32_t num_hidden_layers, bool strict,
                                                 char *err_msg, size_t err_cap);

/* ---- Context Binding, Resolution & Inspection ---- */

/* Verify all referenced devices exist in `ctx` and bind defaults for `plan`. */
velri_status velri_placement_plan_bind(const velri_engine_config *cfg, const velri_context *ctx,
                                       velri_device_id fallback_dev, velri_placement_plan *out_plan,
                                       char *err_msg, size_t err_cap);

/* Resolve cascading `store`, `compute`, and `prefill-compute` (inheriting from `compute`). */
velri_status velri_placement_resolve(const velri_placement_plan *plan, const char *scope_path,
                                     int32_t layer_idx, velri_resolved_placement *out,
                                     char *err_msg, size_t err_cap);

/* Checked builder callback. Each non-null output is a placement used by the graph. */
velri_status velri_placement_builder_cb(const char *scope_path, int32_t layer_idx, void *user,
                                        velri_device_id *out_store, velri_device_id *out_compute);

/* Format a human-readable `inspect` report (devices, resolved placement + provenance, optional mem report). */
size_t velri_placement_format_inspect(const velri_placement_plan *plan, uint32_t num_hidden_layers,
                                      const velri_memory_report *mem_report, char *out_buf,
                                      size_t out_cap);

#endif /* VELRI_CONFIG_PLACEMENT_H */
