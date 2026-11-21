#ifndef VELRI_LIB_BLOCKS_H
#define VELRI_LIB_BLOCKS_H

#include "velri_lib/graph.h"

/*
 * Composable Neural Network Graph Builder, Named Manifest, and Reusable Blocks/Layers.
 *
 * Composition hierarchy:
 *   Operations (graph.h) -> Builder & Blocks (blocks.h) -> Architectures (Velri: ~50 lines each)
 *
 * Design guarantees:
 *   - Zero model-family branches (`if (family == ...)`).
 *   - Automatic named weight & state registration in `velri_manifest` so weight loaders
 *     bind tensors by name without per-model boilerplate structs.
 *   - Scope & layer stack with optional `velri_placement_resolve_fn` callback for cascading
 *     per-component/per-layer `(store, compute)` device placement.
 *   - Memory-efficient by default: zero-copy QKV column subviews, in-place activations and
 *     residual additions, automatic tied-weight sharing, and last-row slicing before lm_head.
 */

enum {
    VELRI_BUILDER_MAX_SCOPE_DEPTH = 8,
    VELRI_BUILDER_MAX_SCOPE_PATH = 256,
    VELRI_MANIFEST_NAME_MAX = 128
};

/* ---- Typed Tensor Handle ---- */

typedef struct {
    velri_storage_id storage_id;
    velri_view_id view_id;
    velri_dtype dtype;
    uint32_t rank;
    size_t dims[VELRI_MAX_RANK];
    size_t strides[VELRI_MAX_RANK];
    size_t byte_offset;
    velri_device_id device_id;
    bool is_seq_dynamic;
    bool valid;
} velri_tensor;

#define VELRI_TENSOR_NONE                                                                          \
    ((velri_tensor){.storage_id = VELRI_INVALID_ID, .view_id = VELRI_INVALID_ID, .valid = false})

/* ---- Named Weight & State Manifest (populated automatically by blocks) ---- */

typedef struct {
    char name[VELRI_MANIFEST_NAME_MAX];        /* Logical tensor name (e.g. "transformer.wte.weight") */
    char source_name[VELRI_MANIFEST_NAME_MAX]; /* Optional alias source (e.g. tied wte on another dev) */
    velri_storage_id storage_id;
    velri_view_id view_id;
    velri_dtype dtype;
    uint32_t rank;
    size_t dims[VELRI_MAX_RANK];
    velri_device_id device_id;
} velri_manifest_entry;

typedef struct {
    velri_storage_id token_ids_in;
    velri_storage_id logits_out;

    velri_manifest_entry *weights;
    uint32_t num_weights;
    uint32_t cap_weights;

    velri_manifest_entry *states;
    uint32_t num_states;
    uint32_t cap_states;
} velri_manifest;

void velri_manifest_init(velri_manifest *m);
void velri_manifest_destroy(velri_manifest *m);
const velri_manifest_entry *velri_manifest_find_weight(const velri_manifest *m, const char *name);
const velri_manifest_entry *velri_manifest_find_state(const velri_manifest *m, const char *name);

/* ---- Placement Callback & Graph Builder ---- */

typedef void (*velri_placement_resolve_fn)(const char *scope_path, int32_t layer_idx, void *user,
                                           velri_device_id *out_store,
                                           velri_device_id *out_compute);

typedef struct {
    velri_graph *graph;
    velri_manifest *manifest; /* Optional; when non-null, named weights/states/IO are recorded */
    velri_status err;

    velri_device_id default_store_dev;
    velri_device_id default_compute_dev;
    velri_placement_resolve_fn resolve_fn;
    void *resolve_user;

    const char *scopes[VELRI_BUILDER_MAX_SCOPE_DEPTH];
    uint32_t scope_depth;
    int32_t current_layer;
    int32_t layer_stack[VELRI_BUILDER_MAX_SCOPE_DEPTH];
    uint32_t layer_depth;
} velri_builder;

void velri_b_init(velri_builder *b, velri_graph *graph, velri_manifest *manifest,
                  velri_device_id default_store_dev, velri_device_id default_compute_dev,
                  velri_placement_resolve_fn resolve_fn, void *resolve_user);
velri_status velri_b_status(const velri_builder *b);
void velri_b_set_error(velri_builder *b, velri_status err);

void velri_b_push_scope(velri_builder *b, const char *name);
void velri_b_pop_scope(velri_builder *b);
void velri_b_push_layer(velri_builder *b, int32_t layer_idx);
void velri_b_pop_layer(velri_builder *b);
void velri_b_devices(const velri_builder *b, velri_device_id *out_store,
                     velri_device_id *out_compute);

/* ---- Tensor Constructors & Named Registration ---- */

velri_tensor velri_b_weight(velri_builder *b, const char *name, velri_dtype dtype, uint32_t rank,
                            const size_t *dims);
velri_tensor velri_b_weight_tied(velri_builder *b, const char *name, const char *tie_to,
                                 velri_dtype dtype, uint32_t rank, const size_t *dims);
velri_tensor velri_b_input(velri_builder *b, velri_dtype dtype, uint32_t rank, const size_t *dims);
velri_tensor velri_b_output(velri_builder *b, velri_dtype dtype, uint32_t rank,
                            const size_t *dims);
velri_tensor velri_b_state(velri_builder *b, const char *name, velri_dtype dtype, uint32_t rank,
                           const size_t *dims);
velri_tensor velri_b_scratch(velri_builder *b, velri_dtype dtype, uint32_t rank,
                             const size_t *dims);
velri_tensor velri_b_subview_2d_cols(velri_builder *b, velri_tensor base, size_t col_offset,
                                     size_t num_cols);
velri_tensor velri_b_subview_2d_rows(velri_builder *b, velri_tensor base, size_t row_offset,
                                     size_t num_rows);
velri_tensor velri_b_subview_1d(velri_builder *b, velri_tensor base, size_t offset, size_t len);

/* ---- Primitive Operation Builders ---- */

velri_tensor velri_b_embedding(velri_builder *b, velri_tensor table, velri_tensor ids_or_none,
                               size_t pos_offset, size_t max_seq_len);
velri_tensor velri_b_linear(velri_builder *b, velri_tensor x, velri_tensor w,
                            velri_tensor bias_or_none);
velri_tensor velri_b_linear_into(velri_builder *b, velri_tensor x, velri_tensor w,
                                 velri_tensor bias_or_none, velri_tensor out);
velri_tensor velri_b_batched_linear(velri_builder *b, velri_tensor x, velri_tensor w,
                                    velri_tensor bias_or_none, uint32_t n_heads, uint32_t n_groups,
                                    uint32_t in_dim, uint32_t out_dim, bool weight_transposed);
velri_tensor velri_b_layernorm(velri_builder *b, velri_tensor x, velri_tensor w,
                               velri_tensor b_opt, float eps);
velri_tensor velri_b_rmsnorm(velri_builder *b, velri_tensor x, velri_tensor w, float eps,
                             bool unit_offset);
velri_tensor velri_b_activation(velri_builder *b, velri_tensor x, velri_act_kind act_kind);
velri_tensor velri_b_activation_inplace(velri_builder *b, velri_tensor x, velri_act_kind act_kind);
velri_tensor velri_b_gated_mlp_act(velri_builder *b, velri_tensor gate, velri_tensor up,
                                   velri_act_kind act_kind, float clamp_limit);
velri_tensor velri_b_gated_mlp_act_inplace(velri_builder *b, velri_tensor gate, velri_tensor up,
                                           velri_act_kind act_kind, float clamp_limit);
velri_tensor velri_b_add(velri_builder *b, velri_tensor a, velri_tensor b_in);
velri_tensor velri_b_add_inplace(velri_builder *b, velri_tensor a, velri_tensor b_in);
velri_tensor velri_b_attention(velri_builder *b, velri_tensor q, velri_tensor k_in,
                               velri_tensor v_in, velri_tensor k_cache, velri_tensor v_cache,
                               uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                               float scale, uint32_t sliding_window, bool causal);
velri_tensor velri_b_attention_ex(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                  velri_tensor v_in, velri_tensor k_cache, velri_tensor v_cache,
                                  uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim_k,
                                  uint32_t head_dim_v, float scale, uint32_t sliding_window,
                                  bool causal);
velri_tensor velri_b_attention_inplace(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                       velri_tensor v_in, velri_tensor k_cache,
                                       velri_tensor v_cache, uint32_t n_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim, float scale, uint32_t sliding_window,
                                       bool causal);
velri_tensor velri_b_copy_last_row(velri_builder *b, velri_tensor x);

/* ---- High-Level Reusable Blocks & Layers ---- */

typedef enum {
    VELRI_NORM_LAYER = 0,
    VELRI_NORM_RMS = 1
} velri_norm_kind;

typedef struct {
    velri_norm_kind kind;
    float eps;
    bool has_bias;
    bool rms_unit_offset;
} velri_norm_spec;

typedef struct {
    const char *scope;        /* Optional scope name (e.g. "input-layer") */
    const char *token_weight; /* Required weight tensor name */
    const char *pos_weight;   /* Optional learned positional embedding table name */
    uint32_t vocab_size;
    uint32_t max_pos;         /* Required if pos_weight != nullptr */
    uint32_t dim;
    uint32_t max_seq_len;
    velri_dtype dtype;
} velri_embedding_spec;

typedef struct {
    const char *scope;       /* Optional sub-scope (e.g. "attention") */
    const char *qkv_name;    /* Prefix for fused QKV (expects ".weight", optional ".bias") */
    const char *q_name;      /* Prefixes for split Q/K/V (when !fused_qkv) */
    const char *k_name;
    const char *v_name;
    const char *o_name;      /* Output projection prefix */
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_ctx;
    uint32_t sliding_window;
    float scale;             /* 0.0f => 1.0f / sqrt(head_dim) */
    bool causal;
    bool fused_qkv;
    bool has_bias;
    bool latent_kv;          /* True => cache single input `h_cache` + exact weight absorption (2x less STATE) */
    velri_dtype dtype;
} velri_attention_spec;

typedef struct {
    const char *scope;       /* Optional sub-scope (e.g. "dense-mlp") */
    const char *gate_name;   /* Prefix for gate proj (when is_gated) */
    const char *up_name;     /* Prefix for up/fc proj */
    const char *down_name;   /* Prefix for down/output proj */
    uint32_t intermediate_size;
    velri_act_kind act;
    float clamp_limit;
    bool is_gated;
    bool has_bias;
    velri_dtype dtype;
} velri_mlp_spec;

typedef struct {
    const char *scope;       /* Optional container scope (e.g. "hidden-layers") */
    const char *norm1_name;  /* Prefix inside layer (e.g. "ln_1") */
    const char *norm2_name;  /* Prefix inside layer (e.g. "ln_2") */
    velri_norm_spec norm;
    velri_attention_spec attn;
    velri_mlp_spec mlp;
    velri_dtype dtype;
} velri_layer_spec;

typedef struct {
    const char *scope;       /* Optional scope name (e.g. "lm-head") */
    const char *norm_name;   /* Optional final norm prefix (e.g. "transformer.ln_f") */
    velri_norm_spec norm;
    const char *weight_name; /* Weight tensor name if untied (e.g. "lm_head.weight") */
    const char *tie_to;      /* Optional existing weight name to share (e.g. "transformer.wte.weight") */
    uint32_t vocab_size;
    uint32_t dim;
    velri_dtype dtype;
} velri_lm_head_spec;

typedef struct {
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim_k;     /* Key/query dim d_k (1..VELRI_ATTN_MAX_HEAD_DIM) */
    uint32_t head_dim_v;     /* Value/output dim d_v (0 => head_dim_k) */
    uint32_t sliding_window;
    float scale;             /* 0.0f => 1.0f / sqrt(head_dim_k) */
    bool causal;
    bool inplace_q;          /* Write output in-place into `q` when head_dim_k == head_dim_v (halves scratch) */
} velri_attention_core_spec;

/* Creates INPUT token_ids [max_seq_len] (recorded in manifest->token_ids_in) and embeds tokens (+ optional pos). */
velri_tensor velri_block_embedding(velri_builder *b, const velri_embedding_spec *spec);

/* Unified attention core block: appends K/V into cache (supports d_k != d_v & shared K==V cache)
 * and computes scaled dot-product online-softmax attention. Shared by dense, latent, and future
 * QSA / DSA / LSA / MLA wrappers. */
velri_tensor velri_block_attention_core(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                        velri_tensor v_in, velri_tensor k_cache,
                                        velri_tensor v_cache,
                                        const velri_attention_core_spec *spec);

/* Dense attention wrapper: fused or split Q/K/V proj -> k/v_cache -> attention_core -> o_proj. */
velri_tensor velri_block_dense_attention(velri_builder *b, velri_tensor x, const char *prefix,
                                         const velri_attention_spec *spec);

/* Latent weight-absorbed attention wrapper: Q proj -> BatchedLinear(Q, W_K) -> single h_cache ->
 * unified attention_core (K==V==h_cache) -> BatchedLinear(h_bar, W_V, b_V) -> o_proj.
 * Works for both direct input caching (`h_lat == x`) and compressed MLA (`h_lat == c_kv`). */
velri_tensor velri_block_latent_attention(velri_builder *b, velri_tensor x, velri_tensor h_lat,
                                          const char *prefix, const velri_attention_spec *spec);

/* Convenience attention dispatcher selecting dense vs latent wrapper based on `spec->latent_kv`. */
velri_tensor velri_block_attention(velri_builder *b, velri_tensor x, const char *prefix,
                                   const velri_attention_spec *spec);

/* MLP block: standard 2-linear (in-place act) or gated 3-linear (in-place gated act). */
velri_tensor velri_block_mlp(velri_builder *b, velri_tensor x, const char *prefix,
                             const velri_mlp_spec *spec);

/* Pre-norm transformer layer: norm1 -> attention -> residual += -> norm2 -> mlp -> residual +=. */
velri_tensor velri_layer_pre_norm(velri_builder *b, velri_tensor x, const char *layer_prefix,
                                  uint32_t layer_idx, const velri_layer_spec *spec);

/* Output LM head: copy_last_row (on x's device) -> optional final norm -> linear into OUTPUT logits [1, vocab]. */
velri_tensor velri_block_lm_head(velri_builder *b, velri_tensor x, const velri_lm_head_spec *spec);

/* ---- Model-Agnostic Metadata Parameters & Composed Layer Stacks (`src/layers/`) ---- */

typedef struct {
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    uint32_t hidden_size;
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t sliding_window;
    float norm_eps;
    velri_act_kind hidden_act;
    bool tie_word_embeddings;
    velri_dtype dtype;

    /* Runtime execution & memory-policy knobs (from placement/session config). */
    uint32_t max_seq_len;
    uint32_t max_ctx;
    bool latent_kv; /* Cache single latent input H_cache + exact W_K/W_V absorption (no RoPE, d_lat <= 2048). */
} velri_model_params;

typedef struct {
    velri_norm_kind norm_kind;
    bool norm_bias;
    bool rms_unit_offset;
    bool has_rope;         /* True when attention applies RoPE to Q/K (disallows raw-input latent_kv) */
    bool fused_qkv;
    bool attn_bias;
    bool mlp_gated;
    bool mlp_bias;
    float mlp_clamp;

    const char *prefix;    /* e.g. "transformer.h" or "model.layers" */
    const char *norm1;     /* e.g. "ln_1" or "input_layernorm" */
    const char *norm2;     /* e.g. "ln_2" or "post_attention_layernorm" */
    const char *qkv;       /* Fused QKV projection prefix (when fused_qkv) */
    const char *q_proj;    /* Split Q/K/V prefixes (when !fused_qkv) */
    const char *k_proj;
    const char *v_proj;
    const char *o_proj;    /* Attention output projection prefix */
    const char *gate_proj; /* MLP gate projection prefix (when mlp_gated) */
    const char *up_proj;   /* MLP up / fc projection prefix */
    const char *down_proj; /* MLP down / output projection prefix */
} velri_pre_norm_layer_desc;

/* High-level metadata-driven layer helpers (`src/layers/pre_norm.c`):
 * Architectures only supply topology + tensor naming; all dimensions/dtypes/eps come from `p`. */
velri_tensor velri_layer_embedding(velri_builder *b, const velri_model_params *p,
                                   const char *token_weight, const char *pos_weight);

velri_tensor velri_layers_pre_norm_repeat(velri_builder *b, velri_tensor x,
                                          const velri_model_params *p,
                                          const velri_pre_norm_layer_desc *desc);

velri_tensor velri_layer_lm_head(velri_builder *b, velri_tensor x, const velri_model_params *p,
                                 const char *norm_name, velri_norm_kind norm_kind, bool norm_bias,
                                 bool rms_unit_offset, const char *weight_name,
                                 const char *tie_to_if_tied);

#endif /* VELRI_LIB_BLOCKS_H */
