#ifndef VELRI_MODEL_LAYERS_H
#define VELRI_MODEL_LAYERS_H

#include <velri_lib/velri_lib.h>

/* Canonical synthetic/precomputed RoPE table weight names shared between layers & runtime. */
#define VELRI_WEIGHT_ROPE_COS "model.rope.cos"
#define VELRI_WEIGHT_ROPE_SIN "model.rope.sin"

/*
 * Velri Engine Model Metadata Parameters, Placement Scope Conventions, and
 * Metadata-Driven Layer Composition Helpers.
 *
 * Lives strictly in `Velri` (`src/model/`), keeping engine JSON placement schema keys
 * ("input-layer", "hidden-layers", "attention", "kv-cache", "dense-mlp", "lm-head")
 * and HuggingFace-style model config out of the standalone `VelriLib` library.
 */

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

    /* RoPE & optional Q/K/V bias parameters (Llama / Qwen / Mistral). */
    float rope_theta;          /* 0.0f => 10000.0f */
    uint32_t rotary_dim;       /* 0 => head_dim */
    velri_rope_mode rope_mode; /* VELRI_ROPE_HALF (0) or VELRI_ROPE_INTERLEAVED (1) */
    bool qkv_bias;             /* Optional Q/K/V projection bias (e.g. true for Qwen2, false for Llama) */

    /* Runtime execution & memory-policy knobs (from placement/session config). */
    uint32_t max_seq_len;
    uint32_t max_ctx;
    bool latent_kv; /* Cache single latent input H_cache + exact W_K/W_V absorption (no RoPE, d_lat <= 2048). */
    bool f16_kv;
} velri_model_params;

typedef struct {
    velri_norm_kind norm_kind;
    bool norm_bias;
    bool rms_unit_offset;
    bool has_rope;         /* True when attention applies Rotary Position Embeddings to Q and K */
    bool fused_qkv;
    bool qkv_bias;
    bool o_bias;
    bool mlp_gated;
    bool mlp_bias;
    float mlp_clamp;

    const char *prefix;       /* e.g. "transformer.h" or "model.layers" */
    const char *state_prefix; /* Optional KV state prefix inside layer (nullptr => "attn") */
    const char *norm1;        /* e.g. "ln_1" or "input_layernorm" */
    const char *norm2;        /* e.g. "ln_2" or "post_attention_layernorm" */
    const char *qkv;          /* Fused QKV projection prefix (when fused_qkv) */
    const char *q_proj;       /* Split Q/K/V prefixes (when !fused_qkv) */
    const char *k_proj;
    const char *v_proj;
    const char *o_proj;       /* Attention output projection prefix */
    const char *gate_proj;    /* MLP gate projection prefix (when mlp_gated) */
    const char *up_proj;      /* MLP up / fc projection prefix */
    const char *down_proj;    /* MLP down / output projection prefix */
} velri_pre_norm_layer_desc;

typedef struct {
    const char *norm_name;      /* Optional final norm prefix (e.g. "transformer.ln_f") */
    velri_norm_kind norm_kind;
    bool norm_bias;
    bool rms_unit_offset;
    const char *weight_name;    /* Untied weight name (e.g. "lm_head.weight") */
    const char *tie_to_if_tied; /* Source weight name when `p->tie_word_embeddings` is true */
    bool full_seq_logits;
    bool has_bias;
} velri_lm_head_desc;

velri_tensor velri_layer_embedding(velri_builder *b, const velri_model_params *p,
                                   const char *token_weight, const char *pos_weight);

velri_tensor velri_layers_pre_norm_repeat(velri_builder *b, velri_tensor x,
                                          const velri_model_params *p,
                                          const velri_pre_norm_layer_desc *desc, bool last_row_only);

velri_tensor velri_layer_lm_head(velri_builder *b, velri_tensor x, const velri_model_params *p,
                                 const velri_lm_head_desc *desc);

/* Populate precomputed RoPE cos/sin tables (`[max_ctx, rotary_dim / 2]` F32) in FP64 precision. */
velri_status velri_fill_rope_tables(float *cos_out, float *sin_out, uint32_t max_ctx,
                                    uint32_t rotary_dim, float theta);

#endif /* VELRI_MODEL_LAYERS_H */
