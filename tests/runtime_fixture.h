#ifndef VELRI_TEST_RUNTIME_FIXTURE_H
#define VELRI_TEST_RUNTIME_FIXTURE_H

#include "velri/velri.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CHECK_RC(expr, msg)                                                                        \
    do {                                                                                           \
        velri_rc _rc = (expr);                                                                     \
        if (_rc != VELRI_RC_OK) {                                                                  \
            fprintf(stderr, "FAIL [%s:%d]: %s (rc=%d: %s)\n", __FILE__, __LINE__, (msg), (int)_rc, \
                    velri_rc_str(_rc));                                                            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/* ---- Deterministic Synthetic Weight Pool ---- */

typedef struct {
    float **ptrs;
    uint32_t count;
    uint32_t cap;
} weight_pool;

static inline void pool_destroy(weight_pool *p) {
    for (uint32_t i = 0U; i < p->count; ++i) {
        free(p->ptrs[i]);
    }
    free(p->ptrs);
    memset(p, 0, sizeof(*p));
}

static inline float *pool_alloc_f32(weight_pool *p, size_t count, uint32_t seed, float scale, float bias) {
    if (p->count == p->cap) {
        uint32_t nc = p->cap ? p->cap * 2U : 64U;
        float **np = (float **)realloc(p->ptrs, (size_t)nc * sizeof(float *));
        CHECK(np != nullptr, "pool realloc");
        p->ptrs = np;
        p->cap = nc;
    }
    float *buf = (float *)calloc(count, sizeof(float));
    CHECK(buf != nullptr, "pool calloc");
    uint32_t s = seed * 2654435761U + 17U;
    for (size_t i = 0U; i < count; ++i) {
        s = s * 1664525U + 1013904223U;
        float u = ((float)((s >> 8U) & 0xFFFFU) / 32767.5f) - 1.0f;
        buf[i] = u * scale + bias;
    }
    p->ptrs[p->count++] = buf;
    return buf;
}

static inline void bind_w(velri_model *m, weight_pool *p, const char *name, size_t elems, uint32_t seed,
                   float scale, float bias) {
    float *w = pool_alloc_f32(p, elems, seed, scale, bias);
    CHECK_RC(velri_model_bind_weight(m, name, w, elems * sizeof(float)), name);
}

static inline void bind_wf(velri_model *m, weight_pool *p, uint32_t *seed, float scale, float bias,
                    size_t elems, const char *fmt, uint32_t layer) {
    char name[128];
    snprintf(name, sizeof(name), fmt, layer);
    bind_w(m, p, name, elems, (*seed)++, scale, bias);
}

static inline void populate_llama_weights(velri_model *m, const velri_model_desc *d, weight_pool *p) {
    uint32_t V = d->vocab_size, D = d->hidden_size, I = d->intermediate_size;
    uint32_t nh = d->num_attention_heads, nkvh = d->num_key_value_heads ? d->num_key_value_heads : nh;
    uint32_t hd = d->head_dim ? d->head_dim : (D / nh);
    uint32_t q_dim = nh * hd, kv_dim = nkvh * hd, seed = 500U;

    bind_w(m, p, "model.embed_tokens.weight", (size_t)V * D, seed++, 0.12f, 0.0f);
    for (uint32_t l = 0U; l < d->num_hidden_layers; ++l) {
        bind_wf(m, p, &seed, 0.05f, 1.0f, D,                 "model.layers.%u.input_layernorm.weight", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)q_dim*D,   "model.layers.%u.self_attn.q_proj.weight", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)kv_dim*D,  "model.layers.%u.self_attn.k_proj.weight", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)kv_dim*D,  "model.layers.%u.self_attn.v_proj.weight", l);
        if (d->qkv_bias) {
            bind_wf(m, p, &seed, 0.02f, 0.0f, q_dim,         "model.layers.%u.self_attn.q_proj.bias", l);
            bind_wf(m, p, &seed, 0.02f, 0.0f, kv_dim,        "model.layers.%u.self_attn.k_proj.bias", l);
            bind_wf(m, p, &seed, 0.02f, 0.0f, kv_dim,        "model.layers.%u.self_attn.v_proj.bias", l);
        }
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)D*q_dim,   "model.layers.%u.self_attn.o_proj.weight", l);
        bind_wf(m, p, &seed, 0.05f, 1.0f, D,                 "model.layers.%u.post_attention_layernorm.weight", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)I * D,     "model.layers.%u.mlp.gate_proj.weight", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)I * D,     "model.layers.%u.mlp.up_proj.weight", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)D * I,     "model.layers.%u.mlp.down_proj.weight", l);
    }
    bind_w(m, p, "model.norm.weight", D, seed++, 0.05f, 1.0f);
    if (!d->tie_word_embeddings) bind_w(m, p, "lm_head.weight", (size_t)V * D, seed++, 0.08f, 0.0f);
}

#endif
