#ifndef VELRI_H
#define VELRI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Velri Engine Public C23 API (`docs/EN/api.md`).
 *
 * Design guarantees:
 *   - Pure C23 header with zero backend, GPU, SIMD, Slang, or third-party types.
 *   - Strict opaque handle hierarchy: `velri_ctx` -> `velri_model` -> `velri_program` ->
 *     `velri_session`. Destroying a parent with live dependents returns `VELRI_RC_ERR_BUSY`.
 *   - Never calls `abort`/`exit` and never uses `longjmp`. Every fallible function returns a
 *     `velri_rc` status code and records thread-safe diagnostic context on `velri_ctx`.
 *   - Synchronous `velri_session_step` with 0 B hot-path heap/device allocations and explicit
 *     buffer ownership/lifetime rules.
 *
 * Thread safety:
 *   - `velri_ctx`: safe to query and create child `velri_model`/`velri_program` concurrently;
 *     `velri_ctx_destroy` must be serialized against all operations on `ctx`.
 *   - `velri_model`: immutable once a `velri_program` is prepared from it; safe to share across
 *     multiple programs and threads. `velri_model_bind_weight` is rejected with `VELRI_RC_ERR_BUSY`
 *     while any dependent `velri_program` is alive. Binding a weight also takes exclusive access.
 *     A competing bind or prepare may return `VELRI_RC_ERR_BUSY`. Serialize destruction with
 *     all other operations on the same model.
 *   - `velri_program`: immutable after `velri_program_prepare`; safe to share across multiple
 *     concurrent `velri_session`s.
 *   - `velri_session`: at most one call (`step` / `reset`) may run concurrently on a given
 *     session (concurrent `step`/`reset` returns `VELRI_RC_ERR_BUSY`); `velri_session_destroy`
 *     must be serialized against all other calls on that session. Independent sessions on the
 *     same `velri_program` / `velri_ctx` may execute concurrently.
 *   - `velri_cancel`: `velri_cancel_trigger` and `velri_cancel_triggered` are lock-free and safe
 *     to invoke from any thread while a `velri_session_step` is in flight. `velri_cancel_reset`
 *     is atomic (`memory_order_release`).
 */

typedef struct velri_ctx velri_ctx;
typedef struct velri_model velri_model;
typedef struct velri_program velri_program;
typedef struct velri_session velri_session;
typedef struct velri_cancel velri_cancel;

/* ---- Status Codes & Diagnostics ---- */

typedef enum {
    VELRI_RC_OK = 0,
    VELRI_RC_ERR_INVALID = 1,
    VELRI_RC_ERR_OVERFLOW = 2,
    VELRI_RC_ERR_UNSUPPORTED = 3,
    VELRI_RC_ERR_NOMEM = 4,
    VELRI_RC_ERR_CANCELLED = 5,
    VELRI_RC_ERR_EXEC = 6,
    VELRI_RC_ERR_BUSY = 7,
    VELRI_RC_ERR_NOT_FOUND = 8,
    VELRI_RC_ERR_IO = 9
} velri_rc;

const char *velri_rc_str(velri_rc rc);

/* ---- Cancellation Handle ---- */

velri_rc velri_cancel_create(velri_cancel **out_cancel);
void velri_cancel_destroy(velri_cancel *cancel);
void velri_cancel_trigger(velri_cancel *cancel);
void velri_cancel_reset(velri_cancel *cancel);
bool velri_cancel_triggered(const velri_cancel *cancel);

/* ---- Context (Devices, Shared CPU Pool, Diagnostics) ---- */

typedef struct {
    uint32_t cpu_threads; /* 0 => automatic physical-core selection or the JSON setting */
    bool enable_vulkan;   /* Opt-in Vulkan backend when compiled into VelriLib */
} velri_ctx_options;

typedef struct {
    uint32_t device_id;
    uint32_t local_index;
    char backend_name[32];
    char device_name[128];
    bool is_host_memory;
    bool is_uma;
    uint64_t total_memory_bytes; /* 0 when unknown */
    uint32_t cpu_workers;        /* Pool size including the caller; 0 for other devices */
} velri_device_info;

/* `opts` may be nullptr for defaults (auto CPU threads, Vulkan disabled). */
velri_rc velri_ctx_create(const velri_ctx_options *opts, velri_ctx **out_ctx);

/* Apply backends.cpu before opening the shared pool. Accepts a full engine config or a
 * backend-only object. Nonzero opts->cpu_threads and explicitly specified JSON threads
 * must agree. JSON is borrowed for the call; err_msg is an optional diagnostic buffer. */
velri_rc velri_ctx_create_from_json(const velri_ctx_options *opts, const char *json, size_t json_len,
                                    velri_ctx **out_ctx, char *err_msg, size_t err_cap);

/* Fails with `VELRI_RC_ERR_BUSY` (leaving `ctx` alive) if any Model or Program still depends on it. */
velri_rc velri_ctx_destroy(velri_ctx *ctx);

uint32_t velri_ctx_device_count(const velri_ctx *ctx);
velri_rc velri_ctx_device_info(const velri_ctx *ctx, uint32_t device_id,
                               velri_device_info *out_info);

/* Returns a thread-local copy of the last diagnostic message recorded on `ctx` (never nullptr). */
const char *velri_ctx_last_error(const velri_ctx *ctx);

/* ---- Model (Topology, Metadata, Immutable Weight Registry) ---- */

typedef enum {
    VELRI_ARCH_GPT2 = 0,
    VELRI_ARCH_LLAMA = 1,
    VELRI_ARCH_ID_MAX = INT32_MAX /* Upper bound for architecture IDs. */
} velri_arch;

/* Look up a compiled-in architecture by its exact name. Returns VELRI_RC_ERR_NOT_FOUND
 * for an unknown name and writes out_arch only on success. */
velri_rc velri_arch_from_name(const char *name, velri_arch *out_arch);

typedef enum {
    VELRI_TYPE_F32 = 0,
    VELRI_TYPE_F16 = 1,
    VELRI_TYPE_BF16 = 2
} velri_type;

typedef enum {
    VELRI_ACTIVATION_SILU = 0,
    VELRI_ACTIVATION_GELU_TANH = 1,
    VELRI_ACTIVATION_RELU2 = 2
} velri_activation;

typedef enum {
    VELRI_ROTARY_HALF = 0,
    VELRI_ROTARY_INTERLEAVED = 1
} velri_rotary_mode;

typedef struct {
    velri_arch arch;
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    uint32_t hidden_size;
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads; /* 0 => num_attention_heads */
    uint32_t head_dim;            /* 0 => hidden_size / num_attention_heads */
    uint32_t intermediate_size;
    uint32_t sliding_window;      /* 0 => full causal context */
    float norm_eps;               /* 0.0f => 1e-5f */
    velri_activation hidden_act;
    bool tie_word_embeddings;
    velri_type weight_type;

    /* RoPE and Q/K/V bias options (used by `VELRI_ARCH_LLAMA`). */
    float rope_theta;             /* 0.0f => 10000.0f */
    uint32_t rotary_dim;          /* 0 => head_dim */
    velri_rotary_mode rope_mode;
    bool qkv_bias;
} velri_model_desc;

velri_rc velri_model_create_synthetic(velri_ctx *ctx, const velri_model_desc *desc,
                                      velri_model **out_model);

/* Fails with `VELRI_RC_ERR_BUSY` (leaving `model` alive) if any Program still depends on it. */
velri_rc velri_model_destroy(velri_model *model);

/*
 * Register an immutable host weight buffer by canonical tensor name.
 *
 * Ownership & lifetime: `host_data` is BORROWED (zero-copy) and must remain valid and unmodified
 * for the entire lifetime of `model` and all `velri_program`s prepared from it.
 * Fails with `VELRI_RC_ERR_BUSY` if any `velri_program` currently depends on `model`.
 */
velri_rc velri_model_bind_weight(velri_model *model, const char *name, const void *host_data,
                                 size_t byte_size);

uint32_t velri_model_weight_count(const velri_model *model);
velri_rc velri_model_get_desc(const velri_model *model, velri_model_desc *out_desc);

/* ---- Program (Validated Execution Plan, Cascading Placement, Memory Accounting) ---- */

enum { VELRI_PROGRAM_MAX_DEVICES = 16 };

typedef struct {
    uint32_t max_ctx;            /* 0 => from JSON config or model `max_position_embeddings` */
    uint32_t prefill_chunk_size; /* 0 => from JSON config or default `min(max_ctx, 64)` */
    bool allow_cpu_fallback;     /* Default false: unsupported op on chosen device returns error */
    bool strict_layer_selectors; /* Default false: selectors beyond model layers are noted, not fatal */
    bool latent_kv;              /* Enable weight-absorbed single-cache latent KV attention */
    const char *placement_json;  /* Optional UTF-8 JSON config (borrowed during `prepare` only) */
    size_t placement_json_len;   /* Byte length of `placement_json` (0 when nullptr) */
} velri_program_options;

typedef struct {
    uint32_t device_id;
    bool is_uma;
    size_t weight_bytes;           /* Physical weight bytes (1x, shared across decode+prefill) */
    size_t state_bytes;            /* Per-session persistent state bytes (1x, shared across decode+prefill) */
    size_t input_output_bytes;     /* Per-session IO arena bytes (sum across concurrently live plans) */
    size_t scratch_packed_bytes;   /* Per-session packed scratch bytes (sum across live plans) */
    size_t scratch_unpacked_bytes; /* Sum of declared scratch storages */
    size_t allocator_overhead_bytes;
} velri_program_device_mem;

typedef struct {
    uint32_t num_devices;
    velri_program_device_mem devices[VELRI_PROGRAM_MAX_DEVICES];
    size_t ram_staging_bytes;
    uint32_t num_ram_transfers_decode;
    uint32_t num_ram_transfers_prefill;
    size_t total_physical_bytes; /* Exact de-duplicated concurrent budget (1x weights + 1x state) */
    bool has_distinct_prefill_plan;
    uint32_t max_ctx;
    uint32_t prefill_chunk_size;
} velri_program_mem_report;

typedef struct {
    float temperature; /* 0.0f => deterministic greedy argmax; > 0.0f => stochastic sampling */
    float top_p;       /* [0, 1]; 0.0f or 1.0f => disabled */
    uint32_t top_k;    /* 0 => disabled */
    float min_p;       /* [0, 1]; 0.0f => disabled */
    uint64_t seed;     /* Applied only when `reseed == true` (otherwise advances session PRNG) */
    bool reseed;
} velri_sampling_params;

/* `opts` may be nullptr for default placement & workload limits. */
velri_rc velri_program_prepare(velri_ctx *ctx, velri_model *model,
                               const velri_program_options *opts, velri_program **out_prog);

/* Fails with `VELRI_RC_ERR_BUSY` (leaving `prog` alive) if any Session still depends on it. */
velri_rc velri_program_destroy(velri_program *prog);

velri_rc velri_program_memory_report(const velri_program *prog,
                                     velri_program_mem_report *out_report);

/* Format a human-readable inspect summary (devices, placement provenance, memory plan).
 * Writes at most `out_cap` bytes (NUL-terminated) and returns bytes written (excluding NUL). */
size_t velri_program_inspect(const velri_program *prog, char *out_buf, size_t out_cap);

bool velri_program_has_distinct_prefill(const velri_program *prog);
velri_rc velri_program_default_sampling(const velri_program *prog,
                                        velri_sampling_params *out_params);

/* ---- Session (Mutable Sequence State, Synchronous Step, Sampling) ---- */

typedef struct {
    uint64_t prng_seed; /* Initial PRNG seed when `has_prng_seed == true` */
    bool has_prng_seed;
} velri_session_options;

typedef enum {
    VELRI_STEP_STOP_OK = 0,
    VELRI_STEP_STOP_CANCELLED = 1,
    VELRI_STEP_STOP_ERROR = 2
} velri_step_stop;

typedef struct {
    velri_step_stop stop_reason;
    const float *logits;              /* Borrowed view [vocab_size]; valid until next mutating call or destroy */
    uint32_t vocab_size;
    int32_t sampled_token;            /* Sampled token ID (>= 0) when `sampling != nullptr`, else -1 */
    uint32_t kv_len;                  /* Total sequence tokens committed in this session */
    uint32_t tokens_processed;        /* Input tokens committed during this step */
    uint64_t last_step_allocs;        /* Backend device allocations during this step (0 on hot path) */
    uint64_t last_step_graph_replays; /* Persistent captured launch-graph replays during this step */
} velri_step_result;

/* `opts` may be nullptr to inherit seed from JSON config (or deterministic default). */
velri_rc velri_session_create(velri_program *prog, const velri_session_options *opts,
                              velri_session **out_sess);

/* Fails with `VELRI_RC_ERR_BUSY` if a step is concurrently executing on `sess`. */
velri_rc velri_session_destroy(velri_session *sess);

/*
 * Execute a synchronous forward step over `token_ids[0..num_tokens-1]`.
 *
 * Ownership & lifetime:
 *   - `token_ids` is borrowed only until this function returns.
 *   - `out_result->logits` is a borrowed view owned by `sess`, valid until the next call to
 *     `velri_session_step`, `velri_session_reset`, or `velri_session_destroy`.
 *   - `sampling` is optional (nullptr skips sampling and sets `sampled_token = -1`).
 *   - `cancel` is optional (nullptr disables cancellation polling).
 *
 * Multi-token inputs are automatically chunked by `prefill_chunk_size`. When `prog` has a distinct
 * `prefill-compute` placement, `num_tokens > 1` executes on the prefill plan and `num_tokens == 1`
 * executes on the decode plan over the SAME shared weight slabs and persistent KV-cache arenas.
 */
velri_rc velri_session_step(velri_session *sess, const int32_t *token_ids, uint32_t num_tokens,
                            const velri_sampling_params *sampling, velri_cancel *cancel,
                            velri_step_result *out_result);

/* Zero persistent sequence state (KV cache), clear `needs_reset`, and restore initial PRNG seed. */
velri_rc velri_session_reset(velri_session *sess);

bool velri_session_needs_reset(const velri_session *sess);
uint32_t velri_session_kv_len(const velri_session *sess);
uint32_t velri_session_max_ctx(const velri_session *sess);

#ifdef __cplusplus
}
#endif

#endif /* VELRI_H */
