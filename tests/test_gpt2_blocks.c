#include "model/arch_registry.h"
#include "model/manifest.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * End-to-End Synthetic GPT-2 Verification:
 *   - Composes GPT-2 purely from metadata (`velri_model_params`) and reusable VelriLib blocks
 *     (`velri_block_embedding`, `velri_layer_pre_norm`, `velri_block_lm_head`).
 *   - Verifies automatic `velri_manifest` recording (`token_ids_in`, `logits_out`, zero
 *     `lm_head.weight` duplication when `tie_word_embeddings` is true) and binds ALL weights
 *     generically by iterating `manifest.weights` and matching tensor names.
 *   - Runs on BOTH CPU and Vulkan (when available).
 *   - Verifies exact numerical agreement against an independent FP64 GPT-2 reference.
 *   - Verifies exact equivalence between chunked prefill+decode (Slot 0) and
 *     token-by-token decode (Slot 1) sharing ONE plan & ONE packed weight arena.
 *   - Verifies `velri_slot_last_run_allocs() == 0`, `velri_slot_reset()`, zero host
 *     RAM shadow for KV cache, and tight memory-report accounting.
 */

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CHECK_OK(expr, msg)                                                                        \
    do {                                                                                           \
        velri_status _st = (expr);                                                                 \
        if (_st != VELRI_OK) {                                                                     \
            fprintf(stderr, "FAIL [%s:%d]: %s (status=%d)\n", __FILE__, __LINE__, (msg),           \
                    (int)_st);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static velri_status build_arch(velri_arch arch, velri_builder *b, const velri_model_params *params) {
    const velri_arch_ops *ops = velri_arch_find(arch);
    CHECK(ops && ops->build_graph, "registered architecture has a graph builder");
    return ops->build_graph(b, params);
}

static float next_rand(uint64_t *rng, float scale) {
    uint64_t x = *rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng = x;
    uint32_t u = (uint32_t)((x * 2685821657736338717ULL) >> 40);
    return ((float)u * (1.0f / 8388608.0f) - 1.0f) * scale;
}

static float *alloc_rand(size_t n, uint64_t *rng, float scale) {
    float *p = (float *)calloc(n ? n : 1U, sizeof(float));
    CHECK(p != nullptr, "oom");
    for (size_t i = 0; i < n; ++i) {
        p[i] = next_rand(rng, scale);
    }
    return p;
}

static float *alloc_ones_jitter(size_t n, uint64_t *rng, float jitter) {
    float *p = (float *)calloc(n ? n : 1U, sizeof(float));
    CHECK(p != nullptr, "oom");
    for (size_t i = 0; i < n; ++i) {
        p[i] = 1.0f + next_rand(rng, jitter);
    }
    return p;
}

typedef struct {
    float *ln_1_w, *ln_1_b;
    float *c_attn_w, *c_attn_b;
    float *c_proj_w, *c_proj_b;
    float *ln_2_w, *ln_2_b;
    float *mlp_fc_w, *mlp_fc_b;
    float *mlp_proj_w, *mlp_proj_b;
} host_layer_weights;

typedef struct {
    float *wte;
    float *wpe;
    float *ln_f_w;
    float *ln_f_b;
    host_layer_weights *layers;
    uint32_t n_layer;
} host_gpt2_weights;

static void init_host_weights(const velri_model_params *meta, host_gpt2_weights *w) {
    memset(w, 0, sizeof(*w));
    size_t d = meta->hidden_size;
    size_t inner = meta->intermediate_size;
    uint64_t rng = 0x69722026ULL;
    float s = 0.08f;

    w->wte = alloc_rand((size_t)meta->vocab_size * d, &rng, 0.25f);
    w->wpe = alloc_rand((size_t)meta->max_position_embeddings * d, &rng, 0.15f);
    w->ln_f_w = alloc_ones_jitter(d, &rng, 0.05f);
    w->ln_f_b = alloc_rand(d, &rng, 0.05f);
    w->n_layer = meta->num_hidden_layers;
    w->layers = (host_layer_weights *)calloc(meta->num_hidden_layers, sizeof(*w->layers));
    CHECK(w->layers != nullptr, "oom layers");

    for (uint32_t i = 0U; i < meta->num_hidden_layers; ++i) {
        host_layer_weights *lw = &w->layers[i];
        lw->ln_1_w = alloc_ones_jitter(d, &rng, 0.05f);
        lw->ln_1_b = alloc_rand(d, &rng, 0.05f);
        lw->c_attn_w = alloc_rand(3U * d * d, &rng, s);
        lw->c_attn_b = alloc_rand(3U * d, &rng, 0.05f);
        lw->c_proj_w = alloc_rand(d * d, &rng, s);
        lw->c_proj_b = alloc_rand(d, &rng, 0.05f);
        lw->ln_2_w = alloc_ones_jitter(d, &rng, 0.05f);
        lw->ln_2_b = alloc_rand(d, &rng, 0.05f);
        lw->mlp_fc_w = alloc_rand(inner * d, &rng, s);
        lw->mlp_fc_b = alloc_rand(inner, &rng, 0.05f);
        lw->mlp_proj_w = alloc_rand(d * inner, &rng, s);
        lw->mlp_proj_b = alloc_rand(d, &rng, 0.05f);
    }
}

static void free_host_weights(host_gpt2_weights *w) {
    if (!w) return;
    free(w->wte);
    free(w->wpe);
    free(w->ln_f_w);
    free(w->ln_f_b);
    for (uint32_t i = 0U; i < w->n_layer; ++i) {
        free(w->layers[i].ln_1_w);
        free(w->layers[i].ln_1_b);
        free(w->layers[i].c_attn_w);
        free(w->layers[i].c_attn_b);
        free(w->layers[i].c_proj_w);
        free(w->layers[i].c_proj_b);
        free(w->layers[i].ln_2_w);
        free(w->layers[i].ln_2_b);
        free(w->layers[i].mlp_fc_w);
        free(w->layers[i].mlp_fc_b);
        free(w->layers[i].mlp_proj_w);
        free(w->layers[i].mlp_proj_b);
    }
    free(w->layers);
}

/* Independent pure-FP64 GPT-2 full-sequence reference producing logits for the last token. */
static void ref_gpt2_fp64(const velri_model_params *meta, const host_gpt2_weights *w,
                          const int32_t *tokens, size_t n_tok, float *out_logits) {
    size_t d = meta->hidden_size;
    size_t inner = meta->intermediate_size;
    size_t nh = meta->num_attention_heads;
    size_t hd = meta->head_dim;
    CHECK(nh * hd == d, "ref head_dim matches hidden_size");
    double eps = (double)meta->norm_eps;

    double *x = (double *)calloc(n_tok * d, sizeof(double));
    double *h = (double *)calloc(n_tok * d, sizeof(double));
    double *qkv = (double *)calloc(n_tok * 3U * d, sizeof(double));
    double *attn = (double *)calloc(n_tok * d, sizeof(double));
    double *mlp = (double *)calloc(n_tok * inner, sizeof(double));
    double *scores = (double *)calloc(n_tok, sizeof(double));
    CHECK(x && h && qkv && attn && mlp && scores, "ref oom");

    for (size_t t = 0U; t < n_tok; ++t) {
        size_t tid = (size_t)tokens[t];
        for (size_t c = 0U; c < d; ++c) {
            x[t * d + c] = (double)w->wte[tid * d + c] + (double)w->wpe[t * d + c];
        }
    }

    for (uint32_t l = 0U; l < meta->num_hidden_layers; ++l) {
        const host_layer_weights *lw = &w->layers[l];

        /* LN1 */
        for (size_t t = 0U; t < n_tok; ++t) {
            double mean = 0.0;
            for (size_t c = 0U; c < d; ++c) mean += x[t * d + c];
            mean /= (double)d;
            double var = 0.0;
            for (size_t c = 0U; c < d; ++c) {
                double df = x[t * d + c] - mean;
                var += df * df;
            }
            double inv = 1.0 / sqrt(var / (double)d + eps);
            for (size_t c = 0U; c < d; ++c) {
                h[t * d + c] =
                    (x[t * d + c] - mean) * inv * (double)lw->ln_1_w[c] + (double)lw->ln_1_b[c];
            }
        }

        /* QKV = Linear(h, c_attn) */
        for (size_t t = 0U; t < n_tok; ++t) {
            for (size_t m = 0U; m < 3U * d; ++m) {
                double acc = (double)lw->c_attn_b[m];
                for (size_t k = 0U; k < d; ++k) {
                    acc += h[t * d + k] * (double)lw->c_attn_w[m * d + k];
                }
                qkv[t * 3U * d + m] = acc;
            }
        }

        /* Causal MHA */
        double scale = 1.0 / sqrt((double)hd);
        for (size_t t = 0U; t < n_tok; ++t) {
            for (size_t head = 0U; head < nh; ++head) {
                const double *q = qkv + t * 3U * d + head * hd;
                double smax = -1e300;
                for (size_t j = 0U; j <= t; ++j) {
                    const double *k = qkv + j * 3U * d + d + head * hd;
                    double dot = 0.0;
                    for (size_t c = 0U; c < hd; ++c) dot += q[c] * k[c];
                    scores[j] = dot * scale;
                    if (scores[j] > smax) smax = scores[j];
                }
                double ssum = 0.0;
                for (size_t j = 0U; j <= t; ++j) {
                    scores[j] = exp(scores[j] - smax);
                    ssum += scores[j];
                }
                for (size_t c = 0U; c < hd; ++c) {
                    double acc = 0.0;
                    for (size_t j = 0U; j <= t; ++j) {
                        const double *v = qkv + j * 3U * d + 2U * d + head * hd;
                        acc += (scores[j] / ssum) * v[c];
                    }
                    attn[t * d + head * hd + c] = acc;
                }
            }
        }

        /* c_proj + residual */
        for (size_t t = 0U; t < n_tok; ++t) {
            for (size_t m = 0U; m < d; ++m) {
                double acc = (double)lw->c_proj_b[m];
                for (size_t k = 0U; k < d; ++k) {
                    acc += attn[t * d + k] * (double)lw->c_proj_w[m * d + k];
                }
                x[t * d + m] += acc;
            }
        }

        /* LN2 */
        for (size_t t = 0U; t < n_tok; ++t) {
            double mean = 0.0;
            for (size_t c = 0U; c < d; ++c) mean += x[t * d + c];
            mean /= (double)d;
            double var = 0.0;
            for (size_t c = 0U; c < d; ++c) {
                double df = x[t * d + c] - mean;
                var += df * df;
            }
            double inv = 1.0 / sqrt(var / (double)d + eps);
            for (size_t c = 0U; c < d; ++c) {
                h[t * d + c] =
                    (x[t * d + c] - mean) * inv * (double)lw->ln_2_w[c] + (double)lw->ln_2_b[c];
            }
        }

        /* MLP: GELU_tanh(c_fc(h)) -> c_proj + residual */
        const double k_sqrt_2_over_pi = 0.79788456080286535588;
        for (size_t t = 0U; t < n_tok; ++t) {
            for (size_t m = 0U; m < inner; ++m) {
                double acc = (double)lw->mlp_fc_b[m];
                for (size_t k = 0U; k < d; ++k) {
                    acc += h[t * d + k] * (double)lw->mlp_fc_w[m * d + k];
                }
                double inner_v = k_sqrt_2_over_pi * (acc + 0.044715 * acc * acc * acc);
                mlp[t * inner + m] = 0.5 * acc * (1.0 + tanh(inner_v));
            }
            for (size_t m = 0U; m < d; ++m) {
                double acc = (double)lw->mlp_proj_b[m];
                for (size_t k = 0U; k < inner; ++k) {
                    acc += mlp[t * inner + k] * (double)lw->mlp_proj_w[m * inner + k];
                }
                x[t * d + m] += acc;
            }
        }
    }

    /* Final LN on last row + tied lm_head */
    const double *last = x + (n_tok - 1U) * d;
    double mean = 0.0;
    for (size_t c = 0U; c < d; ++c) mean += last[c];
    mean /= (double)d;
    double var = 0.0;
    for (size_t c = 0U; c < d; ++c) {
        double df = last[c] - mean;
        var += df * df;
    }
    double inv = 1.0 / sqrt(var / (double)d + eps);
    for (size_t c = 0U; c < d; ++c) {
        h[c] = (last[c] - mean) * inv * (double)w->ln_f_w[c] + (double)w->ln_f_b[c];
    }

    for (uint32_t v = 0U; v < meta->vocab_size; ++v) {
        double acc = 0.0;
        for (size_t c = 0U; c < d; ++c) {
            acc += h[c] * (double)w->wte[(size_t)v * d + c];
        }
        out_logits[v] = (float)acc;
    }

    free(x);
    free(h);
    free(qkv);
    free(attn);
    free(mlp);
    free(scores);
}

static size_t manifest_entry_bytes(const velri_manifest_entry *e) {
    size_t bytes = velri_dtype_bytes(e->dtype);
    CHECK(bytes > 0U && e->rank >= 1U && e->rank <= VELRI_MAX_RANK, "valid manifest entry");
    for (uint32_t r = 0U; r < e->rank; ++r) {
        CHECK(e->dims[r] > 0U && bytes <= SIZE_MAX / e->dims[r], "manifest entry byte overflow");
        bytes *= e->dims[r];
    }
    return bytes;
}

/* Generic name lookup simulating a weight loader matching `manifest.weights[i].name`. */
static const void *resolve_host_weight(const velri_model_params *meta, const host_gpt2_weights *w,
                                       const char *name, size_t *out_bytes) {
    size_t d = meta->hidden_size;
    size_t inner = meta->intermediate_size;

    if (strcmp(name, "transformer.wte.weight") == 0) {
        *out_bytes = (size_t)meta->vocab_size * d * sizeof(float);
        return w->wte;
    }
    if (strcmp(name, "transformer.wpe.weight") == 0) {
        *out_bytes = (size_t)meta->max_position_embeddings * d * sizeof(float);
        return w->wpe;
    }
    if (strcmp(name, "transformer.ln_f.weight") == 0) {
        *out_bytes = d * sizeof(float);
        return w->ln_f_w;
    }
    if (strcmp(name, "transformer.ln_f.bias") == 0) {
        *out_bytes = d * sizeof(float);
        return w->ln_f_b;
    }

    const char *prefix = "transformer.h.";
    size_t plen = strlen(prefix);
    if (strncmp(name, prefix, plen) == 0) {
        const char *num = name + plen;
        if (*num >= '0' && *num <= '9') {
            char *end = nullptr;
            unsigned long layer = strtoul(num, &end, 10);
            if (end && *end == '.' && layer < meta->num_hidden_layers) {
                const host_layer_weights *lw = &w->layers[layer];
                const char *sub = end + 1;
                if (strcmp(sub, "ln_1.weight") == 0) {
                    *out_bytes = d * sizeof(float);
                    return lw->ln_1_w;
                }
                if (strcmp(sub, "ln_1.bias") == 0) {
                    *out_bytes = d * sizeof(float);
                    return lw->ln_1_b;
                }
                if (strcmp(sub, "attn.c_attn.weight") == 0) {
                    *out_bytes = 3U * d * d * sizeof(float);
                    return lw->c_attn_w;
                }
                if (strcmp(sub, "attn.c_attn.bias") == 0) {
                    *out_bytes = 3U * d * sizeof(float);
                    return lw->c_attn_b;
                }
                if (strcmp(sub, "attn.c_proj.weight") == 0) {
                    *out_bytes = d * d * sizeof(float);
                    return lw->c_proj_w;
                }
                if (strcmp(sub, "attn.c_proj.bias") == 0) {
                    *out_bytes = d * sizeof(float);
                    return lw->c_proj_b;
                }
                if (strcmp(sub, "ln_2.weight") == 0) {
                    *out_bytes = d * sizeof(float);
                    return lw->ln_2_w;
                }
                if (strcmp(sub, "ln_2.bias") == 0) {
                    *out_bytes = d * sizeof(float);
                    return lw->ln_2_b;
                }
                if (strcmp(sub, "mlp.c_fc.weight") == 0) {
                    *out_bytes = inner * d * sizeof(float);
                    return lw->mlp_fc_w;
                }
                if (strcmp(sub, "mlp.c_fc.bias") == 0) {
                    *out_bytes = inner * sizeof(float);
                    return lw->mlp_fc_b;
                }
                if (strcmp(sub, "mlp.c_proj.weight") == 0) {
                    *out_bytes = d * inner * sizeof(float);
                    return lw->mlp_proj_w;
                }
                if (strcmp(sub, "mlp.c_proj.bias") == 0) {
                    *out_bytes = d * sizeof(float);
                    return lw->mlp_proj_b;
                }
            }
        }
    }
    return nullptr;
}

static void bind_all_weights(velri_plan *plan, const velri_manifest *manifest,
                             const velri_model_params *meta, const host_gpt2_weights *w) {
    for (uint32_t i = 0U; i < manifest->num_weights; ++i) {
        const velri_manifest_entry *e = &manifest->weights[i];
        const char *key = e->source_name[0] ? e->source_name : e->name;
        size_t host_bytes = 0U;
        const void *ptr = resolve_host_weight(meta, w, key, &host_bytes);
        CHECK(ptr != nullptr, "every manifest weight name must resolve");
        CHECK(manifest_entry_bytes(e) == host_bytes, "manifest weight shape matches host tensor");
        CHECK_OK(velri_plan_bind_weight_host(plan, e->storage_id, ptr, host_bytes), e->name);
    }
}

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double mx = 0.0;
    for (size_t i = 0U; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void run_gpt2_suite_ex(velri_context *ctx, velri_device_id store_dev,
                              velri_device_id compute_dev, velri_placement_resolve_fn resolve_fn,
                              void *resolve_user, velri_device_id expected_state_dev,
                              bool verify_kv_shadow_reuse, const char *tag,
                              const velri_model_params *meta, const host_gpt2_weights *w,
                              const int32_t *seq, size_t seq_len, const float *ref_4,
                              const float *ref_6, float *out_final_logits) {
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "graph create");

    velri_manifest manifest;
    velri_manifest_init(&manifest);

    velri_builder b;
    velri_b_init(&b, g, &manifest, store_dev, compute_dev, resolve_fn, resolve_user);

    CHECK_OK(build_arch(VELRI_ARCH_GPT2, &b, meta), "build gpt2");
    CHECK(manifest.token_ids_in != VELRI_INVALID_ID, "manifest recorded token_ids_in");
    CHECK(manifest.logits_out != VELRI_INVALID_ID, "manifest recorded logits_out");
    CHECK(velri_manifest_find_weight(&manifest, "transformer.wte.weight") != nullptr,
          "manifest contains transformer.wte.weight");
    CHECK(velri_manifest_find_weight(&manifest, "lm_head.weight") == nullptr,
          "tied wte must not duplicate lm_head.weight in manifest");
    CHECK(manifest.num_weights == 4U + 12U * meta->num_hidden_layers,
          "exact GPT-2 tied weight count in manifest");
    if (meta->latent_kv) {
        CHECK(manifest.num_states == meta->num_hidden_layers,
              "exact GPT-2 latent h_cache state count in manifest");
        CHECK(velri_manifest_find_state(&manifest, "transformer.h.0.attn.h_cache") != nullptr,
              "manifest contains transformer.h.0.attn.h_cache");
    } else {
        CHECK(manifest.num_states == 2U * meta->num_hidden_layers,
              "exact GPT-2 standard k/v_cache state count in manifest");
        CHECK(velri_manifest_find_state(&manifest, "transformer.h.0.attn.k_cache") != nullptr,
              "manifest contains transformer.h.0.attn.k_cache");
        CHECK(velri_manifest_find_state(&manifest, "transformer.h.0.attn.v_cache") != nullptr,
              "manifest contains transformer.h.0.attn.v_cache");
    }
    for (uint32_t i = 0U; i < manifest.num_weights; ++i) {
        CHECK(manifest.weights[i].device_id == store_dev, "weight placed on store_dev");
    }
    for (uint32_t i = 0U; i < manifest.num_states; ++i) {
        CHECK(manifest.states[i].device_id == expected_state_dev,
              "state placed on expected_state_dev");
    }

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare gpt2");
    bind_all_weights(plan, &manifest, meta, w);

    uint32_t dag_ops = 0U, dag_elided = 0U;
    CHECK_OK(velri_plan_barrier_stats(plan, &dag_ops, &dag_elided), "gpt2 barrier stats");
    CHECK(dag_ops > 0U, "gpt2 has ops");

    velri_device_caps comp_caps;
    CHECK_OK(velri_context_device_caps(ctx, compute_dev, &comp_caps), "compute caps");
    bool expect_vk_replay = (!comp_caps.is_host_memory && store_dev == compute_dev &&
                             expected_state_dev == compute_dev);

    velri_memory_report rep;
    CHECK_OK(velri_plan_memory_report(plan, &rep), "memory report");
    const velri_device_memory_report *dr_store = &rep.devices[store_dev];
    const velri_device_memory_report *dr_state = &rep.devices[expected_state_dev];
    const velri_device_memory_report *dr_comp = &rep.devices[compute_dev];
    const size_t per_cache_bytes =
        (size_t)meta->num_hidden_layers * (size_t)meta->max_ctx * (size_t)meta->hidden_size *
        sizeof(float);
    const size_t expected_state_bytes = meta->latent_kv ? per_cache_bytes : (2U * per_cache_bytes);
    CHECK(dr_state->state_bytes == expected_state_bytes,
          "STATE footprint matches standard (2x) vs latent (1x) KV mode");
    if (expected_state_dev != compute_dev) {
        CHECK(dr_comp->state_bytes == 0U, "compute_dev holds 0 B resident STATE when offloaded");
    }
    if (verify_kv_shadow_reuse) {
        CHECK(dr_comp->scratch_packed_bytes < expected_state_bytes,
              "offloaded N-layer KV shadows must reuse GPU scratch across layers");
    }
    CHECK(rep.ram_staging_bytes == 0U, "0 B ram_staging on single-dev or host<->dev offload");
    CHECK(dr_comp->scratch_packed_bytes < dr_comp->scratch_unpacked_bytes,
          "liveness scratch packing must reuse cells tightly");

    printf("  [MEM] %-18s | weights=%zu B (%u tensors) | state=%zu B | scratch=%zu B "
           "(unpacked=%zu B, %.1fx saved) | ovh=%zu B\n",
           tag, dr_store->weight_bytes, manifest.num_weights, dr_state->state_bytes,
           dr_comp->scratch_packed_bytes, dr_comp->scratch_unpacked_bytes,
           (double)dr_comp->scratch_unpacked_bytes / (double)dr_comp->scratch_packed_bytes,
           dr_comp->allocator_overhead_bytes);

    /* Two concurrent-capable slots sharing ONE plan and ONE weight arena.
     * Note: ZERO host shadow buffers are bound for KV-cache STATE storages! */
    velri_slot *slot_prefill = nullptr;
    velri_slot *slot_decode = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot_prefill), "slot prefill");
    CHECK_OK(velri_slot_create(plan, &slot_decode), "slot decode");

    size_t tok_bytes = (size_t)meta->max_seq_len * sizeof(int32_t);
    size_t logits_bytes = (size_t)meta->vocab_size * sizeof(float);
    int32_t *tok_buf0 = (int32_t *)calloc(meta->max_seq_len, sizeof(int32_t));
    int32_t *tok_buf1 = (int32_t *)calloc(meta->max_seq_len, sizeof(int32_t));
    float *logits0 = (float *)calloc(meta->vocab_size, sizeof(float));
    float *logits1 = (float *)calloc(meta->vocab_size, sizeof(float));
    float *logits_after_reset = (float *)calloc(meta->vocab_size, sizeof(float));
    CHECK(tok_buf0 && tok_buf1 && logits0 && logits1 && logits_after_reset, "io oom");

    CHECK_OK(velri_slot_bind_host(slot_prefill, manifest.token_ids_in, tok_buf0, tok_bytes), "in0");
    CHECK_OK(velri_slot_bind_host(slot_prefill, manifest.logits_out, logits0, logits_bytes), "out0");
    CHECK_OK(velri_slot_bind_host(slot_decode, manifest.token_ids_in, tok_buf1, tok_bytes), "in1");
    CHECK_OK(velri_slot_bind_host(slot_decode, manifest.logits_out, logits1, logits_bytes), "out1");

    /* Slot 0: prefill 4 tokens in one step, then decode tokens 4 and 5 one by one */
    memcpy(tok_buf0, seq, 4U * sizeof(int32_t));
    velri_step_params sp_pre = {
        .has_pos_offset = true, .pos_offset_override = 0U,
        .has_seq_len = true, .seq_len_override = 4U,
        .has_kv_len = true, .kv_len_override = 0U,
    };
    CHECK_OK(velri_slot_run(slot_prefill, &sp_pre, nullptr), "prefill 4");
    CHECK(velri_slot_last_run_allocs(slot_prefill) == 0U, "prefill 0 allocs");
    double err_pre4 = max_abs_diff(logits0, ref_4, meta->vocab_size);
    CHECK(err_pre4 < 1e-4, "prefill(4) logits match FP64 reference");

    for (size_t t = 4U; t < seq_len; ++t) {
        tok_buf0[0] = seq[t];
        velri_step_params sp_dec = {
            .has_pos_offset = true, .pos_offset_override = t,
            .has_seq_len = true, .seq_len_override = 1U,
            .has_kv_len = true, .kv_len_override = t,
        };
        CHECK_OK(velri_slot_run(slot_prefill, &sp_dec, nullptr), "decode after prefill");
        CHECK(velri_slot_last_run_allocs(slot_prefill) == 0U, "decode 0 allocs");
    }
    double err_pre_dec6 = max_abs_diff(logits0, ref_6, meta->vocab_size);
    CHECK(err_pre_dec6 < 1e-4, "prefill(4)+decode(2) logits match FP64 reference");

    /* Slot 1: pure token-by-token decode across all 6 tokens */
    uint64_t total_vk_replays = 0U;
    for (size_t t = 0U; t < seq_len; ++t) {
        tok_buf1[0] = seq[t];
        velri_step_params sp_dec = {
            .has_pos_offset = true, .pos_offset_override = t,
            .has_seq_len = true, .seq_len_override = 1U,
            .has_kv_len = true, .kv_len_override = t,
        };
        CHECK_OK(velri_slot_run(slot_decode, &sp_dec, nullptr), "pure decode");
        CHECK(velri_slot_last_run_allocs(slot_decode) == 0U, "pure decode 0 allocs");
        total_vk_replays += velri_slot_last_run_graph_replays(slot_decode);
    }
    if (expect_vk_replay) {
        CHECK(total_vk_replays >= seq_len,
              "pure Vulkan decode must replay persistent captured command buffer every step");
    }
    double err_pre_vs_dec = max_abs_diff(logits0, logits1, meta->vocab_size);
    CHECK(err_pre_vs_dec < 1e-4, "prefill+decode matches pure decode");

    /* Reset Slot 0 and re-run prefill(4) -> must reproduce clean prefill! */
    CHECK_OK(velri_slot_reset(slot_prefill), "reset slot0");
    memcpy(tok_buf0, seq, 4U * sizeof(int32_t));
    CHECK_OK(velri_slot_bind_host(slot_prefill, manifest.logits_out, logits_after_reset,
                                  logits_bytes),
             "rebind out0");
    CHECK_OK(velri_slot_run(slot_prefill, &sp_pre, nullptr), "prefill after reset");
    CHECK(velri_slot_last_run_allocs(slot_prefill) == 0U, "post-reset prefill 0 allocs");
    CHECK(max_abs_diff(logits_after_reset, ref_4, meta->vocab_size) < 1e-4,
          "reset slot reproduces clean prefill");

    memcpy(out_final_logits, logits0, logits_bytes);
    printf("  [PASS] %-18s | pre(4) err=%.2e | pre+dec(6)=%.2e | pre==dec=%.2e | "
           "dag_elided=%u/%u | vk_replays=%llu | 0 allocs\n",
           tag, err_pre4, err_pre_dec6, err_pre_vs_dec, dag_elided, dag_ops,
           (unsigned long long)total_vk_replays);

    free(tok_buf0);
    free(tok_buf1);
    free(logits0);
    free(logits1);
    free(logits_after_reset);
    velri_slot_destroy(slot_decode);
    velri_slot_destroy(slot_prefill);
    velri_plan_destroy(plan);
    velri_manifest_destroy(&manifest);
    velri_graph_destroy(g);
}

static void run_gpt2_suite_on_device(velri_context *ctx, velri_device_id store_dev,
                                     velri_device_id compute_dev, const char *tag,
                                     const velri_model_params *meta, const host_gpt2_weights *w,
                                     const int32_t *seq, size_t seq_len, const float *ref_4,
                                     const float *ref_6, float *out_final_logits) {
    run_gpt2_suite_ex(ctx, store_dev, compute_dev, nullptr, nullptr, store_dev, false, tag, meta, w,
                      seq, seq_len, ref_4, ref_6, out_final_logits);
}

static void kv_offload_placement_cb(const char *scope_path, int32_t layer_idx, void *user,
                                    velri_device_id *out_store, velri_device_id *out_compute) {
    (void)layer_idx;
    (void)out_compute;
    if (strcmp(scope_path, "hidden-layers/attention/kv-cache") == 0) {
        *out_store = *(const velri_device_id *)user;
    }
}

typedef struct {
    bool saw_input_layer;
    bool saw_attention;
    bool saw_kv_cache;
    bool saw_dense_mlp;
    bool saw_lm_head;
} scope_trace;

static void trace_placement_cb(const char *scope_path, int32_t layer_idx, void *user,
                               velri_device_id *out_store, velri_device_id *out_compute) {
    (void)out_store;
    (void)out_compute;
    scope_trace *tr = (scope_trace *)user;
    if (strcmp(scope_path, "input-layer") == 0 && layer_idx == -1) {
        tr->saw_input_layer = true;
    } else if (strcmp(scope_path, "hidden-layers/attention") == 0 && layer_idx >= 0) {
        tr->saw_attention = true;
    } else if (strcmp(scope_path, "hidden-layers/attention/kv-cache") == 0 && layer_idx >= 0) {
        tr->saw_kv_cache = true;
    } else if (strcmp(scope_path, "hidden-layers/dense-mlp") == 0 && layer_idx >= 0) {
        tr->saw_dense_mlp = true;
    } else if (strcmp(scope_path, "lm-head") == 0 && layer_idx == -1) {
        tr->saw_lm_head = true;
    }
}

static void test_tied_replicas(velri_context *ctx, velri_device_id cpu) {
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "tied replicas graph");
    velri_manifest m;
    velri_manifest_init(&m);
    velri_builder b;
    velri_b_init(&b, g, &m, cpu, cpu, nullptr, nullptr);
    const size_t dims[] = {4, 4};
    velri_tensor root = velri_b_weight(&b, "emb", VELRI_DTYPE_F32, 2, dims);
    velri_b_weight_tied(&b, "c", "emb", VELRI_DTYPE_F32, 2, dims);
    velri_b_weight_tied(&b, "third", "emb", VELRI_DTYPE_F32, 2, dims);
    CHECK(root.valid && !velri_manifest_find_weight(&m, "c") && m.num_weights == 1,
          "local names add no loader bindings");
    velri_device_id other = cpu + 1U;
    bool has_vk = velri_context_find_device(ctx, "vulkan", 0U, &other) == VELRI_OK;
    b.default_store_dev = has_vk ? other : cpu + 1U;
    velri_tensor a = velri_b_weight_tied(&b, "a", "emb", VELRI_DTYPE_F32, 2, dims);
    velri_tensor c = velri_b_weight_tied(&b, "c", "emb", VELRI_DTYPE_F32, 2, dims);
    CHECK(a.valid && c.valid && a.storage_id == c.storage_id && a.view_id == c.view_id &&
              a.storage_id != root.storage_id, "aliases on one device must reuse its replica");
    const velri_manifest_entry *alias = velri_manifest_find_weight(&m, "c");
    CHECK(alias && strcmp(alias->name, "c") == 0 && alias->storage_id == a.storage_id &&
              strcmp(velri_manifest_weight_source(alias), "emb") == 0,
          "both remote names remain available to manifest consumers");
    b.default_store_dev = cpu;
    velri_tensor back = velri_b_weight_tied(&b, "back", "c", VELRI_DTYPE_F32, 2, dims);
    CHECK(back.valid && back.storage_id == root.storage_id && m.num_weights == 2 &&
              velri_graph_storage_count(g) == 2, "chain returns to the original storage");

    if (has_vk) {
        velri_plan *plan = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare Vulkan weight replica");
        alignas(64) float values[16] = {};
        velri_memory_report memory;
        CHECK_OK(velri_plan_memory_report(plan, &memory), "replica memory report");
        CHECK(memory.devices[cpu].weight_bytes == sizeof(values) &&
                  memory.devices[other].weight_bytes == sizeof(values), "one weight per device");
        for (uint32_t i = 0; i < m.num_weights; ++i) {
            const velri_manifest_entry *e = &m.weights[i];
            CHECK(strcmp(e->source_name[0] ? e->source_name : e->name, "emb") == 0,
                  "every physical weight binds the original source");
            CHECK_OK(velri_plan_bind_weight_host(plan, e->storage_id, values, sizeof(values)),
                     "bind canonical weight");
        }
        velri_plan_destroy(plan);
    }

    b.default_store_dev = a.device_id + 1U;
    velri_tensor third = velri_b_weight_tied(&b, "third", "c", VELRI_DTYPE_F32, 2, dims);
    const velri_manifest_entry *e = velri_manifest_find_weight(&m, "third");
    CHECK(third.valid && e && strcmp(e->source_name, "emb") == 0 && m.num_weights == 3,
          "a replica created through an alias binds the canonical name");
    CHECK(velri_b_weight_shared(&b, "back", VELRI_DTYPE_F32, 2, dims).storage_id == third.storage_id,
          "shared and tied constructors reuse the same replica");
    velri_tensor different = velri_b_weight(&b, "different", VELRI_DTYPE_F32, 2, dims);
    velri_tensor conflict = velri_b_weight_tied(&b, "back", "different", VELRI_DTYPE_F32, 2, dims);
    CHECK(different.valid && !conflict.valid && velri_b_status(&b) == VELRI_ERR_INVALID,
          "an alias cannot be redirected to another source");
    velri_manifest_destroy(&m);
    velri_graph_destroy(g);
}

static void test_builder_and_layer_contracts(velri_device_id cpu_dev,
                                             const velri_model_params *meta) {
    /* 1. RoPE + latent_kv combination must fail cleanly with VELRI_ERR_UNSUPPORTED. */
    {
        velri_model_params bad_meta = *meta;
        bad_meta.latent_kv = true;
        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "rope+latent neg graph");
        velri_manifest m;
        velri_manifest_init(&m);
        velri_builder b;
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, nullptr, nullptr);

        velri_tensor x = velri_layer_embedding(&b, &bad_meta, "tok.weight", nullptr);
        CHECK(x.valid && velri_b_status(&b) == VELRI_OK, "embed before rope+latent check");

        const velri_pre_norm_layer_desc rope_desc = {
            .norm_kind = VELRI_NORM_RMS,
            .has_rope = true,
            .fused_qkv = false,
            .mlp_gated = true,
            .prefix = "model.layers",
            .norm1 = "input_layernorm",
            .norm2 = "post_attention_layernorm",
            .q_proj = "self_attn.q_proj",
            .k_proj = "self_attn.k_proj",
            .v_proj = "self_attn.v_proj",
            .o_proj = "self_attn.o_proj",
            .gate_proj = "mlp.gate_proj",
            .up_proj = "mlp.up_proj",
            .down_proj = "mlp.down_proj",
        };
        velri_tensor out = velri_layers_pre_norm_repeat(&b, x, &bad_meta, &rope_desc, false);
        CHECK(!out.valid && velri_b_status(&b) == VELRI_ERR_UNSUPPORTED,
              "has_rope=true && latent_kv=true must fail cleanly with VELRI_ERR_UNSUPPORTED");

        velri_manifest_destroy(&m);
        velri_graph_destroy(g);
    }

    /* 1b. Shared weight constructor reuses identical manifest entry, rejects shape mismatch & null manifest. */
    {
        const size_t d1[2] = {16U, 8U};
        const size_t d2[2] = {16U, 4U};
        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "shared weight g");
        velri_manifest m;
        velri_manifest_init(&m);
        velri_builder b;
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, nullptr, nullptr);
        velri_tensor a = velri_b_weight_shared(&b, "model.rope.cos", VELRI_DTYPE_F32, 2U, d1);
        velri_tensor a2 = velri_b_weight_shared(&b, "model.rope.cos", VELRI_DTYPE_F32, 2U, d1);
        CHECK(a.valid && a2.valid && a.storage_id == a2.storage_id && m.num_weights == 1U,
              "velri_b_weight_shared reuses storage without duplicating manifest");
        velri_tensor bad = velri_b_weight_shared(&b, "model.rope.cos", VELRI_DTYPE_F32, 2U, d2);
        CHECK(!bad.valid && velri_b_status(&b) == VELRI_ERR_INVALID,
              "velri_b_weight_shared rejects shape mismatch");
        velri_manifest_destroy(&m);
        velri_graph_destroy(g);

        CHECK_OK(velri_graph_create(&g), "shared null manifest g");
        velri_b_init(&b, g, nullptr, cpu_dev, cpu_dev, nullptr, nullptr);
        velri_tensor nm = velri_b_weight_shared(&b, "model.rope.cos", VELRI_DTYPE_F32, 2U, d1);
        CHECK(!nm.valid && velri_b_status(&b) == VELRI_ERR_INVALID,
              "velri_b_weight_shared requires manifest");
        velri_graph_destroy(g);

        /* H4: velri_fill_rope_tables must reject negative, NaN, and Inf theta. */
        float c_tmp[8], s_tmp[8];
        CHECK(velri_fill_rope_tables(c_tmp, s_tmp, 2U, 8U, -10000.0f) == VELRI_ERR_INVALID,
              "negative theta rejected");
        CHECK(velri_fill_rope_tables(c_tmp, s_tmp, 2U, 8U, NAN) == VELRI_ERR_INVALID,
              "NaN theta rejected");
        CHECK(velri_fill_rope_tables(c_tmp, s_tmp, 2U, 8U, INFINITY) == VELRI_ERR_INVALID,
              "Inf theta rejected");
        CHECK_OK(velri_fill_rope_tables(c_tmp, s_tmp, 2U, 8U, 0.0f), "theta=0 defaults cleanly");

        /* H1: velri_block_dense_attention must reject rope.enabled without explicit table names. */
        CHECK_OK(velri_graph_create(&g), "rope missing name g");
        velri_manifest_init(&m);
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, nullptr, nullptr);
        const size_t xdims[2] = {4U, 32U};
        velri_tensor x = velri_b_input(&b, VELRI_DTYPE_F32, 2U, xdims);
        const velri_attention_spec bad_rope = {
            .q_name = "q", .k_name = "k", .v_name = "v", .o_name = "o",
            .n_heads = 2U, .n_kv_heads = 2U, .head_dim = 16U, .max_ctx = 8U,
            .causal = true, .dtype = VELRI_DTYPE_F32,
            .rope = {.enabled = true, .cos_name = nullptr, .sin_name = nullptr},
        };
        velri_tensor bad_a = velri_block_dense_attention(&b, x, "l0", &bad_rope);
        CHECK(!bad_a.valid && velri_b_status(&b) == VELRI_ERR_INVALID,
              "dense_attention must require explicit rope cos_name/sin_name");
        velri_manifest_destroy(&m);
        velri_graph_destroy(g);
    }

    /* 2. Manifest duplicate weight/state rejection and tied-weight validation. */
    {
        const size_t dim_w[2] = {4U, 4U};
        const size_t dim_s[3] = {1U, 4U, 4U};
        velri_graph *g = nullptr;
        velri_manifest m;
        velri_builder b;

        CHECK_OK(velri_graph_create(&g), "dup weight graph");
        velri_manifest_init(&m);
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, nullptr, nullptr);
        velri_tensor w0 = velri_b_weight(&b, "dup.weight", VELRI_DTYPE_F32, 2U, dim_w);
        CHECK(w0.valid && velri_b_status(&b) == VELRI_OK, "first weight succeeds");
        velri_tensor w1 = velri_b_weight(&b, "dup.weight", VELRI_DTYPE_F32, 2U, dim_w);
        CHECK(!w1.valid && velri_b_status(&b) == VELRI_ERR_INVALID,
              "duplicate manifest weight name must fail with VELRI_ERR_INVALID");
        velri_manifest_destroy(&m);
        velri_graph_destroy(g);

        CHECK_OK(velri_graph_create(&g), "dup state graph");
        velri_manifest_init(&m);
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, nullptr, nullptr);
        velri_tensor s0 = velri_b_state(&b, "dup.state", VELRI_DTYPE_F32, 3U, dim_s);
        CHECK(s0.valid && velri_b_status(&b) == VELRI_OK, "first state succeeds");
        velri_tensor s1 = velri_b_state(&b, "dup.state", VELRI_DTYPE_F32, 3U, dim_s);
        CHECK(!s1.valid && velri_b_status(&b) == VELRI_ERR_INVALID,
              "duplicate manifest state name must fail with VELRI_ERR_INVALID");
        velri_manifest_destroy(&m);
        velri_graph_destroy(g);

        CHECK_OK(velri_graph_create(&g), "missing tie graph");
        velri_manifest_init(&m);
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, nullptr, nullptr);
        velri_tensor wt =
            velri_b_weight_tied(&b, "lm_head.weight", "missing.weight", VELRI_DTYPE_F32, 2U, dim_w);
        CHECK(!wt.valid && velri_b_status(&b) == VELRI_ERR_NOT_FOUND,
              "missing tie_to must fail with VELRI_ERR_NOT_FOUND");
        velri_manifest_destroy(&m);
        velri_graph_destroy(g);

        CHECK_OK(velri_graph_create(&g), "empty name tie graph");
        velri_manifest_init(&m);
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, nullptr, nullptr);
        velri_tensor wbase = velri_b_weight(&b, "base.weight", VELRI_DTYPE_F32, 2U, dim_w);
        CHECK(wbase.valid, "base weight succeeds");
        velri_tensor wbad = velri_b_weight_tied(&b, "", "base.weight", VELRI_DTYPE_F32, 2U, dim_w);
        CHECK(!wbad.valid && velri_b_status(&b) == VELRI_ERR_INVALID,
              "empty name in weight_tied must fail with VELRI_ERR_INVALID");
        velri_manifest_destroy(&m);
        velri_graph_destroy(g);
    }

    /* 3. Engine placement scopes (incl. kv-cache) are visited and cleanly balanced. */
    {
        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "scope trace graph");
        velri_manifest m;
        velri_manifest_init(&m);
        scope_trace tr = {0};
        velri_builder b;
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, trace_placement_cb, &tr);
        CHECK_OK(build_arch(VELRI_ARCH_GPT2, &b, meta), "build gpt2 scope trace");
        CHECK(tr.saw_input_layer && tr.saw_attention && tr.saw_kv_cache && tr.saw_dense_mlp &&
                  tr.saw_lm_head,
              "all engine placement scopes visited");
        CHECK(b.scope_depth == 0U && b.layer_depth == 0U && b.current_layer == -1,
              "builder scope/layer stacks cleanly balanced");
        velri_manifest_destroy(&m);
        velri_graph_destroy(g);
    }

    /* 4. Sticky-error scope/layer unwinding & safe velri_b_devices after error. */
    {
        const size_t dim_w[2] = {4U, 4U};
        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "unwind graph");
        velri_manifest m;
        velri_manifest_init(&m);
        scope_trace tr = {0};
        velri_builder b;
        velri_b_init(&b, g, &m, cpu_dev, cpu_dev, trace_placement_cb, &tr);

        char stack_scope[] = "input-layer";
        velri_b_push_scope(&b, stack_scope);
        velri_b_push_layer(&b, 0);
        velri_tensor w0 = velri_b_weight(&b, "dup.w", VELRI_DTYPE_F32, 2U, dim_w);
        CHECK(w0.valid, "unwind first weight ok");
        velri_tensor w1 = velri_b_weight(&b, "dup.w", VELRI_DTYPE_F32, 2U, dim_w);
        CHECK(!w1.valid && velri_b_status(&b) == VELRI_ERR_INVALID, "unwind triggers sticky err");

        /* Nested push/pop after sticky error must balance, and outer pop must clear stack_scope. */
        velri_b_push_scope(&b, "nested-after-err");
        velri_b_pop_scope(&b);
        velri_b_pop_layer(&b);
        velri_b_pop_scope(&b);
        CHECK(b.scope_depth == 0U && b.scopes[0] == nullptr && b.skipped_scope_pushes == 0U,
              "pop_scope unwinds and clears pointers even after sticky error");
        CHECK(b.layer_depth == 0U && b.current_layer == -1 && b.skipped_layer_pushes == 0U,
              "pop_layer unwinds even after sticky error");

        tr.saw_input_layer = false;
        velri_device_id sdev = 99U, cdev = 99U;
        velri_b_devices(&b, &sdev, &cdev);
        CHECK(!tr.saw_input_layer && sdev == cpu_dev && cdev == cpu_dev,
              "velri_b_devices must not invoke resolve_fn after sticky error");

        velri_manifest_destroy(&m);
        velri_graph_destroy(g);
    }

    printf("  [PASS] builder & layer contracts (RoPE+latent guard, shared weight, dup manifest, tie_to, scopes, unwind)\n");
}

/* ============================================================================
 * Synthetic Llama / Qwen2 End-to-End Suite (Pre-RMSNorm + RoPE + GQA + SwiGLU)
 * ============================================================================ */

typedef struct {
    float *in_norm_w;
    float *q_w, *q_b;
    float *k_w, *k_b;
    float *v_w, *v_b;
    float *o_w;
    float *post_norm_w;
    float *gate_w;
    float *up_w;
    float *down_w;
} host_llama_layer;

typedef struct {
    float *embed;
    float *norm_w;
    float *lm_head;
    float *rope_cos;
    float *rope_sin;
    host_llama_layer *layers;
    uint32_t n_layer;
} host_llama_weights;

static void init_llama_weights(const velri_model_params *p, host_llama_weights *w) {
    memset(w, 0, sizeof(*w));
    size_t d = p->hidden_size;
    size_t q_dim = (size_t)p->num_attention_heads * p->head_dim;
    size_t kv_dim = (size_t)p->num_key_value_heads * p->head_dim;
    size_t inner = p->intermediate_size;
    uint32_t rdim = p->rotary_dim ? p->rotary_dim : p->head_dim;
    uint64_t rng = 0x11A11A26ULL;
    float s = 0.08f;

    w->embed = alloc_rand((size_t)p->vocab_size * d, &rng, 0.25f);
    w->norm_w = alloc_ones_jitter(d, &rng, 0.05f);
    w->lm_head = p->tie_word_embeddings ? nullptr : alloc_rand((size_t)p->vocab_size * d, &rng, 0.1f);
    w->rope_cos = (float *)calloc((size_t)p->max_ctx * (size_t)(rdim / 2U), sizeof(float));
    w->rope_sin = (float *)calloc((size_t)p->max_ctx * (size_t)(rdim / 2U), sizeof(float));
    CHECK(w->rope_cos && w->rope_sin, "llama rope oom");
    CHECK_OK(velri_fill_rope_tables(w->rope_cos, w->rope_sin, p->max_ctx, rdim, p->rope_theta),
             "fill rope tables");

    w->n_layer = p->num_hidden_layers;
    w->layers = (host_llama_layer *)calloc(p->num_hidden_layers, sizeof(*w->layers));
    CHECK(w->layers != nullptr, "llama layers oom");
    for (uint32_t i = 0U; i < p->num_hidden_layers; ++i) {
        host_llama_layer *lw = &w->layers[i];
        lw->in_norm_w = alloc_ones_jitter(d, &rng, 0.05f);
        lw->q_w = alloc_rand(q_dim * d, &rng, s);
        lw->k_w = alloc_rand(kv_dim * d, &rng, s);
        lw->v_w = alloc_rand(kv_dim * d, &rng, s);
        if (p->qkv_bias) {
            lw->q_b = alloc_rand(q_dim, &rng, 0.05f);
            lw->k_b = alloc_rand(kv_dim, &rng, 0.05f);
            lw->v_b = alloc_rand(kv_dim, &rng, 0.05f);
        }
        lw->o_w = alloc_rand(d * q_dim, &rng, s);
        lw->post_norm_w = alloc_ones_jitter(d, &rng, 0.05f);
        lw->gate_w = alloc_rand(inner * d, &rng, s);
        lw->up_w = alloc_rand(inner * d, &rng, s);
        lw->down_w = alloc_rand(d * inner, &rng, s);
    }
}

static void free_llama_weights(host_llama_weights *w) {
    if (!w) return;
    free(w->embed);
    free(w->norm_w);
    free(w->lm_head);
    free(w->rope_cos);
    free(w->rope_sin);
    for (uint32_t i = 0U; i < w->n_layer; ++i) {
        free(w->layers[i].in_norm_w);
        free(w->layers[i].q_w);
        free(w->layers[i].q_b);
        free(w->layers[i].k_w);
        free(w->layers[i].k_b);
        free(w->layers[i].v_w);
        free(w->layers[i].v_b);
        free(w->layers[i].o_w);
        free(w->layers[i].post_norm_w);
        free(w->layers[i].gate_w);
        free(w->layers[i].up_w);
        free(w->layers[i].down_w);
    }
    free(w->layers);
}

static void ref_apply_rope_fp64(double *buf, size_t n_tok, size_t n_heads, size_t hd,
                                size_t rdim, velri_rope_mode mode, const float *cos_t,
                                const float *sin_t) {
    size_t half = rdim / 2U;
    for (size_t t = 0U; t < n_tok; ++t) {
        const float *c = cos_t + t * half;
        const float *s = sin_t + t * half;
        for (size_t h = 0U; h < n_heads; ++h) {
            double *row = buf + (t * n_heads + h) * hd;
            for (size_t i = 0U; i < half; ++i) {
                size_t i0 = (mode == VELRI_ROPE_HALF) ? i : (i * 2U);
                size_t i1 = (mode == VELRI_ROPE_HALF) ? (i + half) : (i * 2U + 1U);
                double x0 = row[i0];
                double x1 = row[i1];
                row[i0] = x0 * (double)c[i] - x1 * (double)s[i];
                row[i1] = x1 * (double)c[i] + x0 * (double)s[i];
            }
        }
    }
}

static void ref_llama_fp64(const velri_model_params *p, const host_llama_weights *w,
                           const int32_t *tokens, size_t n_tok, float *out_logits) {
    size_t d = p->hidden_size;
    size_t nh = p->num_attention_heads;
    size_t nkvh = p->num_key_value_heads;
    size_t hd = p->head_dim;
    size_t q_dim = nh * hd;
    size_t kv_dim = nkvh * hd;
    size_t inner = p->intermediate_size;
    size_t rdim = p->rotary_dim ? p->rotary_dim : hd;
    double eps = (double)p->norm_eps;

    double *x = (double *)calloc(n_tok * d, sizeof(double));
    double *h = (double *)calloc(n_tok * d, sizeof(double));
    double *q = (double *)calloc(n_tok * q_dim, sizeof(double));
    double *k = (double *)calloc(n_tok * kv_dim, sizeof(double));
    double *v = (double *)calloc(n_tok * kv_dim, sizeof(double));
    double *attn = (double *)calloc(n_tok * q_dim, sizeof(double));
    double *mlp = (double *)calloc(n_tok * inner, sizeof(double));
    double *scores = (double *)calloc(n_tok, sizeof(double));
    CHECK(x && h && q && k && v && attn && mlp && scores, "ref llama oom");

    for (size_t t = 0U; t < n_tok; ++t) {
        size_t tid = (size_t)tokens[t];
        for (size_t c = 0U; c < d; ++c) {
            x[t * d + c] = (double)w->embed[tid * d + c];
        }
    }

    for (uint32_t l = 0U; l < p->num_hidden_layers; ++l) {
        const host_llama_layer *lw = &w->layers[l];

        /* Pre-RMSNorm 1 */
        for (size_t t = 0U; t < n_tok; ++t) {
            double ss = 0.0;
            for (size_t c = 0U; c < d; ++c) ss += x[t * d + c] * x[t * d + c];
            double inv = 1.0 / sqrt(ss / (double)d + eps);
            for (size_t c = 0U; c < d; ++c) h[t * d + c] = x[t * d + c] * inv * (double)lw->in_norm_w[c];
        }

        /* Split Q, K, V */
        for (size_t t = 0U; t < n_tok; ++t) {
            for (size_t m = 0U; m < q_dim; ++m) {
                double acc = lw->q_b ? (double)lw->q_b[m] : 0.0;
                for (size_t c = 0U; c < d; ++c) acc += h[t * d + c] * (double)lw->q_w[m * d + c];
                q[t * q_dim + m] = acc;
            }
            for (size_t m = 0U; m < kv_dim; ++m) {
                double ak = lw->k_b ? (double)lw->k_b[m] : 0.0;
                double av = lw->v_b ? (double)lw->v_b[m] : 0.0;
                for (size_t c = 0U; c < d; ++c) {
                    ak += h[t * d + c] * (double)lw->k_w[m * d + c];
                    av += h[t * d + c] * (double)lw->v_w[m * d + c];
                }
                k[t * kv_dim + m] = ak;
                v[t * kv_dim + m] = av;
            }
        }

        /* RoPE on Q and K */
        ref_apply_rope_fp64(q, n_tok, nh, hd, rdim, p->rope_mode, w->rope_cos, w->rope_sin);
        ref_apply_rope_fp64(k, n_tok, nkvh, hd, rdim, p->rope_mode, w->rope_cos, w->rope_sin);

        /* Causal GQA Attention */
        double scale = 1.0 / sqrt((double)hd);
        size_t gsize = nh / nkvh;
        for (size_t t = 0U; t < n_tok; ++t) {
            for (size_t head = 0U; head < nh; ++head) {
                size_t kvh = head / gsize;
                const double *qh = q + t * q_dim + head * hd;
                double smax = -1e300;
                for (size_t j = 0U; j <= t; ++j) {
                    const double *kh = k + j * kv_dim + kvh * hd;
                    double dot = 0.0;
                    for (size_t c = 0U; c < hd; ++c) dot += qh[c] * kh[c];
                    scores[j] = dot * scale;
                    if (scores[j] > smax) smax = scores[j];
                }
                double ssum = 0.0;
                for (size_t j = 0U; j <= t; ++j) {
                    scores[j] = exp(scores[j] - smax);
                    ssum += scores[j];
                }
                for (size_t c = 0U; c < hd; ++c) {
                    double acc = 0.0;
                    for (size_t j = 0U; j <= t; ++j) {
                        const double *vh = v + j * kv_dim + kvh * hd;
                        acc += (scores[j] / ssum) * vh[c];
                    }
                    attn[t * q_dim + head * hd + c] = acc;
                }
            }
        }

        /* o_proj + residual */
        for (size_t t = 0U; t < n_tok; ++t) {
            for (size_t m = 0U; m < d; ++m) {
                double acc = 0.0;
                for (size_t c = 0U; c < q_dim; ++c) acc += attn[t * q_dim + c] * (double)lw->o_w[m * q_dim + c];
                x[t * d + m] += acc;
            }
        }

        /* Pre-RMSNorm 2 */
        for (size_t t = 0U; t < n_tok; ++t) {
            double ss = 0.0;
            for (size_t c = 0U; c < d; ++c) ss += x[t * d + c] * x[t * d + c];
            double inv = 1.0 / sqrt(ss / (double)d + eps);
            for (size_t c = 0U; c < d; ++c) h[t * d + c] = x[t * d + c] * inv * (double)lw->post_norm_w[c];
        }

        /* SwiGLU MLP + residual */
        for (size_t t = 0U; t < n_tok; ++t) {
            for (size_t m = 0U; m < inner; ++m) {
                double ag = 0.0, au = 0.0;
                for (size_t c = 0U; c < d; ++c) {
                    ag += h[t * d + c] * (double)lw->gate_w[m * d + c];
                    au += h[t * d + c] * (double)lw->up_w[m * d + c];
                }
                double silu = ag / (1.0 + exp(-ag));
                mlp[t * inner + m] = silu * au;
            }
            for (size_t m = 0U; m < d; ++m) {
                double acc = 0.0;
                for (size_t c = 0U; c < inner; ++c) acc += mlp[t * inner + c] * (double)lw->down_w[m * inner + c];
                x[t * d + m] += acc;
            }
        }
    }

    /* Final RMSNorm on last row + LM head */
    const double *last = x + (n_tok - 1U) * d;
    double ss = 0.0;
    for (size_t c = 0U; c < d; ++c) ss += last[c] * last[c];
    double inv = 1.0 / sqrt(ss / (double)d + eps);
    for (size_t c = 0U; c < d; ++c) h[c] = last[c] * inv * (double)w->norm_w[c];

    const float *lm_w = p->tie_word_embeddings ? w->embed : w->lm_head;
    for (uint32_t vo = 0U; vo < p->vocab_size; ++vo) {
        double acc = 0.0;
        for (size_t c = 0U; c < d; ++c) acc += h[c] * (double)lm_w[(size_t)vo * d + c];
        out_logits[vo] = (float)acc;
    }

    free(x); free(h); free(q); free(k); free(v); free(attn); free(mlp); free(scores);
}

static const void *resolve_llama_weight(const velri_model_params *p, const host_llama_weights *w,
                                        const char *name, size_t *out_bytes) {
    size_t d = p->hidden_size;
    size_t q_dim = (size_t)p->num_attention_heads * p->head_dim;
    size_t kv_dim = (size_t)p->num_key_value_heads * p->head_dim;
    size_t inner = p->intermediate_size;
    uint32_t rdim = p->rotary_dim ? p->rotary_dim : p->head_dim;

    if (strcmp(name, "model.embed_tokens.weight") == 0) {
        *out_bytes = (size_t)p->vocab_size * d * sizeof(float);
        return w->embed;
    }
    if (strcmp(name, "model.norm.weight") == 0) {
        *out_bytes = d * sizeof(float);
        return w->norm_w;
    }
    if (strcmp(name, "lm_head.weight") == 0) {
        *out_bytes = (size_t)p->vocab_size * d * sizeof(float);
        return w->lm_head;
    }
    if (strcmp(name, "model.rope.cos") == 0) {
        *out_bytes = (size_t)p->max_ctx * (size_t)(rdim / 2U) * sizeof(float);
        return w->rope_cos;
    }
    if (strcmp(name, "model.rope.sin") == 0) {
        *out_bytes = (size_t)p->max_ctx * (size_t)(rdim / 2U) * sizeof(float);
        return w->rope_sin;
    }

    const char *pfx = "model.layers.";
    size_t plen = strlen(pfx);
    if (strncmp(name, pfx, plen) == 0) {
        const char *num = name + plen;
        if (*num >= '0' && *num <= '9') {
            char *end = nullptr;
            unsigned long layer = strtoul(num, &end, 10);
            if (end && *end == '.' && layer < p->num_hidden_layers) {
                const host_llama_layer *lw = &w->layers[layer];
                const char *sub = end + 1;
                if (strcmp(sub, "input_layernorm.weight") == 0) { *out_bytes = d * sizeof(float); return lw->in_norm_w; }
                if (strcmp(sub, "post_attention_layernorm.weight") == 0) { *out_bytes = d * sizeof(float); return lw->post_norm_w; }
                if (strcmp(sub, "self_attn.q_proj.weight") == 0) { *out_bytes = q_dim * d * sizeof(float); return lw->q_w; }
                if (strcmp(sub, "self_attn.q_proj.bias") == 0) { *out_bytes = q_dim * sizeof(float); return lw->q_b; }
                if (strcmp(sub, "self_attn.k_proj.weight") == 0) { *out_bytes = kv_dim * d * sizeof(float); return lw->k_w; }
                if (strcmp(sub, "self_attn.k_proj.bias") == 0) { *out_bytes = kv_dim * sizeof(float); return lw->k_b; }
                if (strcmp(sub, "self_attn.v_proj.weight") == 0) { *out_bytes = kv_dim * d * sizeof(float); return lw->v_w; }
                if (strcmp(sub, "self_attn.v_proj.bias") == 0) { *out_bytes = kv_dim * sizeof(float); return lw->v_b; }
                if (strcmp(sub, "self_attn.o_proj.weight") == 0) { *out_bytes = d * q_dim * sizeof(float); return lw->o_w; }
                if (strcmp(sub, "mlp.gate_proj.weight") == 0) { *out_bytes = inner * d * sizeof(float); return lw->gate_w; }
                if (strcmp(sub, "mlp.up_proj.weight") == 0) { *out_bytes = inner * d * sizeof(float); return lw->up_w; }
                if (strcmp(sub, "mlp.down_proj.weight") == 0) { *out_bytes = d * inner * sizeof(float); return lw->down_w; }
            }
        }
    }
    return nullptr;
}

typedef struct {
    velri_device_id cpu_dev;
    velri_device_id vk_dev;
} hybrid_llama_cfg;

static void hybrid_llama_placement_cb(const char *scope_path, int32_t layer_idx, void *user,
                                      velri_device_id *out_store, velri_device_id *out_compute) {
    (void)scope_path;
    const hybrid_llama_cfg *hc = (const hybrid_llama_cfg *)user;
    if (layer_idx == 0) {
        *out_store = hc->cpu_dev;
        *out_compute = hc->cpu_dev;
    } else if (layer_idx >= 1) {
        *out_store = hc->vk_dev;
        *out_compute = hc->vk_dev;
    }
}

static void run_llama_suite_ex(velri_context *ctx, velri_device_id default_dev,
                               velri_placement_resolve_fn resolve_fn, void *resolve_user,
                               uint32_t expected_rope_replicas, const char *tag,
                               const velri_model_params *p, const host_llama_weights *w,
                               const int32_t *seq, size_t seq_len, const float *ref_4,
                               const float *ref_6, float *out_logits) {
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "llama g");
    velri_manifest manifest;
    velri_manifest_init(&manifest);
    velri_builder b;
    velri_b_init(&b, g, &manifest, default_dev, default_dev, resolve_fn, resolve_user);
    CHECK_OK(build_arch(VELRI_ARCH_LLAMA, &b, p), "build llama");

    CHECK(velri_manifest_find_weight(&manifest, "model.rope.cos") != nullptr &&
              velri_manifest_find_weight(&manifest, "model.rope.sin") != nullptr,
          "Llama manifest has shared RoPE tables");
    uint32_t per_layer_w = p->qkv_bias ? 12U : 9U;
    uint32_t base_w = (p->tie_word_embeddings ? 2U : 3U) + 2U * expected_rope_replicas;
    uint32_t expected_w = base_w + per_layer_w * p->num_hidden_layers;
    CHECK(manifest.num_weights == expected_w, "shared RoPE tables registered once per device");
    CHECK(manifest.num_states == 2U * p->num_hidden_layers, "exact Llama GQA KV state count");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep llama");
    for (uint32_t i = 0U; i < manifest.num_weights; ++i) {
        const velri_manifest_entry *e = &manifest.weights[i];
        const char *key = e->source_name[0] ? e->source_name : e->name;
        size_t bytes = 0U;
        const void *ptr = resolve_llama_weight(p, w, key, &bytes);
        CHECK(ptr != nullptr && bytes == manifest_entry_bytes(e), e->name);
        CHECK_OK(velri_plan_bind_weight_host(plan, e->storage_id, ptr, bytes), e->name);
    }

    uint32_t dag_ops = 0U, dag_elided = 0U;
    CHECK_OK(velri_plan_barrier_stats(plan, &dag_ops, &dag_elided), "llama barrier stats");
    /* Every layer has parallel Q/K/V (2), parallel RoPE Q/K (1), and parallel Gate/Up (1) => >= 4/layer! */
    CHECK(dag_elided >= 4U * p->num_hidden_layers,
          "Llama DAG analyzer must elide barriers across parallel Q/K/V, RoPE Q/K, and Gate/Up");

    velri_device_caps caps;
    CHECK_OK(velri_context_device_caps(ctx, default_dev, &caps), "llama dev caps");

    velri_slot *slot0 = nullptr;
    velri_slot *slot1 = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot0), "llama slot0");
    CHECK_OK(velri_slot_create(plan, &slot1), "llama slot1");

    int32_t *tok0 = (int32_t *)calloc(p->max_seq_len, sizeof(int32_t));
    int32_t *tok1 = (int32_t *)calloc(p->max_seq_len, sizeof(int32_t));
    float *log0 = (float *)calloc(p->vocab_size, sizeof(float));
    float *log1 = (float *)calloc(p->vocab_size, sizeof(float));
    CHECK(tok0 && tok1 && log0 && log1, "llama io oom");

    CHECK_OK(velri_slot_bind_host(slot0, manifest.token_ids_in, tok0, p->max_seq_len * sizeof(int32_t)), "t0");
    CHECK_OK(velri_slot_bind_host(slot0, manifest.logits_out, log0, p->vocab_size * sizeof(float)), "l0");
    CHECK_OK(velri_slot_bind_host(slot1, manifest.token_ids_in, tok1, p->max_seq_len * sizeof(int32_t)), "t1");
    CHECK_OK(velri_slot_bind_host(slot1, manifest.logits_out, log1, p->vocab_size * sizeof(float)), "l1");

    /* Prefill(4) + Decode(2) */
    memcpy(tok0, seq, 4U * sizeof(int32_t));
    velri_step_params sp_pre = {
        .has_pos_offset = true, .pos_offset_override = 0U,
        .has_seq_len = true, .seq_len_override = 4U,
        .has_kv_len = true, .kv_len_override = 0U,
    };
    CHECK_OK(velri_slot_run(slot0, &sp_pre, nullptr), "llama prefill 4");
    CHECK(velri_slot_last_run_allocs(slot0) == 0U, "llama prefill 0 allocs");
    double err_pre4 = max_abs_diff(log0, ref_4, p->vocab_size);
    CHECK(err_pre4 < 1e-4, "llama prefill(4) matches FP64 reference");

    for (size_t t = 4U; t < seq_len; ++t) {
        tok0[0] = seq[t];
        velri_step_params sp_dec = {
            .has_pos_offset = true, .pos_offset_override = t,
            .has_seq_len = true, .seq_len_override = 1U,
            .has_kv_len = true, .kv_len_override = t,
        };
        CHECK_OK(velri_slot_run(slot0, &sp_dec, nullptr), "llama dec after prefill");
        CHECK(velri_slot_last_run_allocs(slot0) == 0U, "llama dec 0 allocs");
    }
    double err_pre_dec6 = max_abs_diff(log0, ref_6, p->vocab_size);
    CHECK(err_pre_dec6 < 1e-4, "llama prefill(4)+decode(2) matches FP64 reference");

    /* Pure token-by-token decode(6) */
    uint64_t replays = 0U;
    for (size_t t = 0U; t < seq_len; ++t) {
        tok1[0] = seq[t];
        velri_step_params sp_dec = {
            .has_pos_offset = true, .pos_offset_override = t,
            .has_seq_len = true, .seq_len_override = 1U,
            .has_kv_len = true, .kv_len_override = t,
        };
        CHECK_OK(velri_slot_run(slot1, &sp_dec, nullptr), "llama pure dec");
        CHECK(velri_slot_last_run_allocs(slot1) == 0U, "llama pure dec 0 allocs");
        replays += velri_slot_last_run_graph_replays(slot1);
    }
    if (!caps.is_host_memory && !resolve_fn) {
        CHECK(replays == seq_len, "Llama Vulkan decode must replay persistent graph every step");
    }
    double err_pvd = max_abs_diff(log0, log1, p->vocab_size);
    CHECK(err_pvd < 1e-4, "llama prefill+decode == pure decode");

    memcpy(out_logits, log0, p->vocab_size * sizeof(float));
    printf("  [PASS] %-18s | pre(4) err=%.2e | pre+dec(6)=%.2e | pre==dec=%.2e | "
           "dag_elided=%u/%u | vk_replays=%llu | 0 allocs\n",
           tag, err_pre4, err_pre_dec6, err_pvd, dag_elided, dag_ops,
           (unsigned long long)replays);

    free(tok0); free(tok1); free(log0); free(log1);
    velri_slot_destroy(slot1);
    velri_slot_destroy(slot0);
    velri_plan_destroy(plan);
    velri_manifest_destroy(&manifest);
    velri_graph_destroy(g);
}

static void run_llama_suite_on_device(velri_context *ctx, velri_device_id dev, const char *tag,
                                      const velri_model_params *p, const host_llama_weights *w,
                                      const int32_t *seq, size_t seq_len, const float *ref_4,
                                      const float *ref_6, float *out_logits) {
    run_llama_suite_ex(ctx, dev, nullptr, nullptr, 1U, tag, p, w, seq, seq_len, ref_4, ref_6,
                       out_logits);
}

static void test_shared_manifest(void) {
    velri_manifest_entry entries[] = {
        {.name = "key",   .rank = 2, .dims = {1, 4}, .device_id = 0},
        {.name = "other", .rank = 1, .dims = {8},    .device_id = 1},
        {.name = "value", .rank = 2, .dims = {1, 4}, .device_id = 0},
    };
    velri_manifest_entry peer_entries[] = {entries[1], entries[0], entries[2]};
    velri_manifest owner = {.weights = entries, .num_weights = 3};
    velri_manifest peer = {.weights = peer_entries, .num_weights = 3};
    char err[192];
    CHECK(velri_model_check_shared_manifest(&owner, &peer, 2, err, sizeof(err)),
          "independent devices can interleave declarations differently");
    strcpy(peer_entries[1].name, "prefill.alias");
    strcpy(peer_entries[1].source_name, "key");
    CHECK(velri_model_check_shared_manifest(&owner, &peer, 2, err, sizeof(err)),
          "two aliases for one binding can share the same weight slab");
    strcpy(peer_entries[1].source_name, "wrong-tied-weight");
    CHECK(!velri_model_check_shared_manifest(&owner, &peer, 2, err, sizeof(err)),
          "tied weights must resolve to the same source in both plans");
    peer_entries[1] = entries[2];
    peer_entries[2] = entries[0];
    owner = (velri_manifest){.states = entries, .num_states = 3};
    peer = (velri_manifest){.states = peer_entries, .num_states = 3};
    CHECK(!velri_model_check_shared_manifest(&owner, &peer, 2, err, sizeof(err)) &&
              strstr(err, "key") && strstr(err, "value"), "reject same-size swapped KV roles");
    peer_entries[1] = entries[0];
    peer_entries[2] = entries[2];
    peer_entries[2].dims[0] = 4;
    peer_entries[2].dims[1] = 1;
    CHECK(!velri_model_check_shared_manifest(&owner, &peer, 2, err, sizeof(err)),
          "equal bytes with different state shape");
}

int main(void) {
    test_shared_manifest();
    printf("=== Velri End-to-End GPT-2 & Llama/Qwen Architecture Suite ===\n");

    const velri_model_params meta_std = {
        .vocab_size = 257U,
        .max_position_embeddings = 64U,
        .hidden_size = 128U,
        .num_hidden_layers = 4U,
        .num_attention_heads = 4U,
        .num_key_value_heads = 4U,
        .head_dim = 32U,
        .intermediate_size = 512U,
        .sliding_window = 0U,
        .norm_eps = 1e-5f,
        .hidden_act = VELRI_ACT_GELU_TANH,
        .tie_word_embeddings = true,
        .dtype = VELRI_DTYPE_F32,
        .max_seq_len = 16U,
        .max_ctx = 64U,
        .latent_kv = false,
    };
    velri_model_params meta_lat = meta_std;
    meta_lat.latent_kv = true;

    host_gpt2_weights w;
    init_host_weights(&meta_std, &w);

    const int32_t seq[6] = {12, 200, 7, 256, 42, 0};
    float *ref_4 = (float *)calloc(meta_std.vocab_size, sizeof(float));
    float *ref_6 = (float *)calloc(meta_std.vocab_size, sizeof(float));
    float *cpu_std_logits = (float *)calloc(meta_std.vocab_size, sizeof(float));
    float *cpu_lat_logits = (float *)calloc(meta_std.vocab_size, sizeof(float));
    float *vk_std_logits = (float *)calloc(meta_std.vocab_size, sizeof(float));
    float *vk_lat_logits = (float *)calloc(meta_std.vocab_size, sizeof(float));
    float *offload_lat_logits = (float *)calloc(meta_std.vocab_size, sizeof(float));
    float *kv_offload_logits = (float *)calloc(meta_std.vocab_size, sizeof(float));
    CHECK(ref_4 && ref_6 && cpu_std_logits && cpu_lat_logits && vk_std_logits && vk_lat_logits &&
              offload_lat_logits && kv_offload_logits,
          "oom");

    ref_gpt2_fp64(&meta_std, &w, seq, 4U, ref_4);
    ref_gpt2_fp64(&meta_std, &w, seq, 6U, ref_6);

    velri_context_config ccfg = {
        .cpu_workers = 4U,
        .cpu_variant = VELRI_CPU_AUTO,
        .enable_vulkan = true,
    };
    velri_context *ctx = nullptr;
    CHECK_OK(velri_context_create(&ccfg, &ctx), "ctx");

    velri_device_id cpu_dev = 0U;
    CHECK_OK(velri_context_find_device(ctx, "cpu", 0U, &cpu_dev), "find cpu");
    test_tied_replicas(ctx, cpu_dev);
    test_builder_and_layer_contracts(cpu_dev, &meta_std);
    run_gpt2_suite_on_device(ctx, cpu_dev, cpu_dev, "cpu:0 [std-kv]", &meta_std, &w, seq, 6U,
                             ref_4, ref_6, cpu_std_logits);
    run_gpt2_suite_on_device(ctx, cpu_dev, cpu_dev, "cpu:0 [latent-kv]", &meta_lat, &w, seq, 6U,
                             ref_4, ref_6, cpu_lat_logits);
    double cpu_std_vs_lat = max_abs_diff(cpu_std_logits, cpu_lat_logits, meta_std.vocab_size);
    CHECK(cpu_std_vs_lat < 1e-4, "CPU standard KV and latent KV logits match");
    printf("  [PASS] cpu:0 std-kv == latent-kv max_abs_diff=%.2e\n", cpu_std_vs_lat);

    velri_device_id vk_dev = 0U;
    if (velri_context_find_device(ctx, "vulkan", 0U, &vk_dev) == VELRI_OK) {
        run_gpt2_suite_on_device(ctx, vk_dev, vk_dev, "vk:0  [std-kv]", &meta_std, &w, seq, 6U,
                                 ref_4, ref_6, vk_std_logits);
        run_gpt2_suite_on_device(ctx, vk_dev, vk_dev, "vk:0  [latent-kv]", &meta_lat, &w, seq, 6U,
                                 ref_4, ref_6, vk_lat_logits);
        run_gpt2_suite_on_device(ctx, cpu_dev, vk_dev, "cpu->vk [latent]", &meta_lat, &w, seq, 6U,
                                 ref_4, ref_6, offload_lat_logits);
        run_gpt2_suite_ex(ctx, vk_dev, vk_dev, kv_offload_placement_cb, &cpu_dev, cpu_dev, true,
                          "vk:0 [kv->cpu]", &meta_std, &w, seq, 6U, ref_4, ref_6,
                          kv_offload_logits);
        double vk_std_vs_lat = max_abs_diff(vk_std_logits, vk_lat_logits, meta_std.vocab_size);
        double cpu_vk_std = max_abs_diff(cpu_std_logits, vk_std_logits, meta_std.vocab_size);
        double cpu_vk_lat = max_abs_diff(cpu_lat_logits, vk_lat_logits, meta_std.vocab_size);
        double offload_diff = max_abs_diff(vk_lat_logits, offload_lat_logits, meta_std.vocab_size);
        double kv_offload_diff = max_abs_diff(vk_std_logits, kv_offload_logits, meta_std.vocab_size);
        CHECK(vk_std_vs_lat < 1e-4, "Vulkan standard KV and latent KV logits match");
        CHECK(cpu_vk_std < 1e-4, "CPU and Vulkan standard KV GPT-2 logits match");
        CHECK(cpu_vk_lat < 1e-4, "CPU and Vulkan latent KV GPT-2 logits match");
        CHECK(offload_diff < 1e-4, "Offloaded CPU->Vulkan latent KV GPT-2 logits match");
        CHECK(kv_offload_diff < 1e-4, "Vulkan weights + CPU KV-cache offload GPT-2 logits match");
        printf("  [PASS] vk std==lat=%.2e | cpu==vk std=%.2e | cpu==vk lat=%.2e | offload=%.2e | kv->cpu=%.2e\n",
               vk_std_vs_lat, cpu_vk_std, cpu_vk_lat, offload_diff, kv_offload_diff);
    } else {
        printf("  [SKIP] vulkan:0 (not enabled or no GPU available)\n");
    }

    /* Synthetic Llama / Qwen2 (Pre-RMSNorm + RoPE + GQA + SwiGLU) E2E Suite */
    {
        const velri_model_params llama_p = {
            .vocab_size = 257U,
            .max_position_embeddings = 64U,
            .hidden_size = 128U,
            .num_hidden_layers = 3U,
            .num_attention_heads = 4U,
            .num_key_value_heads = 2U,
            .head_dim = 32U,
            .intermediate_size = 256U,
            .sliding_window = 0U,
            .norm_eps = 1e-5f,
            .hidden_act = VELRI_ACT_SILU,
            .tie_word_embeddings = false,
            .dtype = VELRI_DTYPE_F32,
            .rope_theta = 500000.0f,
            .rotary_dim = 32U,
            .rope_mode = VELRI_ROPE_HALF,
            .qkv_bias = true, /* exercises Qwen2 Q/K/V biases + Llama SwiGLU/GQA/RoPE */
            .max_seq_len = 16U,
            .max_ctx = 64U,
            .latent_kv = false,
        };
        host_llama_weights lw;
        init_llama_weights(&llama_p, &lw);
        float *lref_4 = (float *)calloc(llama_p.vocab_size, sizeof(float));
        float *lref_6 = (float *)calloc(llama_p.vocab_size, sizeof(float));
        float *lcpu = (float *)calloc(llama_p.vocab_size, sizeof(float));
        float *lvk = (float *)calloc(llama_p.vocab_size, sizeof(float));
        CHECK(lref_4 && lref_6 && lcpu && lvk, "llama ref oom");
        ref_llama_fp64(&llama_p, &lw, seq, 4U, lref_4);
        ref_llama_fp64(&llama_p, &lw, seq, 6U, lref_6);

        run_llama_suite_on_device(ctx, cpu_dev, "cpu:0 [llama/qwen]", &llama_p, &lw, seq, 6U,
                                  lref_4, lref_6, lcpu);
        if (velri_context_find_device(ctx, "vulkan", 0U, &vk_dev) == VELRI_OK) {
            run_llama_suite_on_device(ctx, vk_dev, "vk:0  [llama/qwen]", &llama_p, &lw, seq, 6U,
                                      lref_4, lref_6, lvk);
            double ldiff = max_abs_diff(lcpu, lvk, llama_p.vocab_size);
            CHECK(ldiff < 1e-4, "CPU and Vulkan Llama/Qwen logits match");

            /* H2 regression: hybrid Llama (layer 0 on cpu:0, layers 1..2 on vulkan:0) replicates RoPE once/dev. */
            float *lhyb = (float *)calloc(llama_p.vocab_size, sizeof(float));
            CHECK(lhyb != nullptr, "hybrid llama oom");
            hybrid_llama_cfg hcfg = {.cpu_dev = cpu_dev, .vk_dev = vk_dev};
            run_llama_suite_ex(ctx, vk_dev, hybrid_llama_placement_cb, &hcfg, 2U,
                               "hyb   [llama/qwen]", &llama_p, &lw, seq, 6U, lref_4, lref_6, lhyb);
            double hdiff = max_abs_diff(lvk, lhyb, llama_p.vocab_size);
            CHECK(hdiff < 1e-4, "Hybrid CPU+Vulkan Llama logits match pure Vulkan");
            printf("  [PASS] llama/qwen cpu==vk=%.2e | hybrid==vk=%.2e\n", ldiff, hdiff);
            free(lhyb);
        }

        /* Partial rotary_dim (16 < 32) + VELRI_ROPE_INTERLEAVED verification */
        velri_model_params part_p = llama_p;
        part_p.num_hidden_layers = 2U;
        part_p.rotary_dim = 16U;
        part_p.rope_mode = VELRI_ROPE_INTERLEAVED;
        part_p.qkv_bias = false;
        part_p.tie_word_embeddings = true;
        host_llama_weights pw;
        init_llama_weights(&part_p, &pw);
        ref_llama_fp64(&part_p, &pw, seq, 4U, lref_4);
        ref_llama_fp64(&part_p, &pw, seq, 6U, lref_6);
        run_llama_suite_on_device(ctx, cpu_dev, "cpu:0 [part-rope]", &part_p, &pw, seq, 6U,
                                  lref_4, lref_6, lcpu);
        if (velri_context_find_device(ctx, "vulkan", 0U, &vk_dev) == VELRI_OK) {
            run_llama_suite_on_device(ctx, vk_dev, "vk:0  [part-rope]", &part_p, &pw, seq, 6U,
                                      lref_4, lref_6, lvk);
            CHECK(max_abs_diff(lcpu, lvk, part_p.vocab_size) < 1e-4,
                  "Partial interleaved RoPE CPU==Vulkan");
        }
        free_llama_weights(&pw);

        free(lref_4); free(lref_6); free(lcpu); free(lvk);
        free_llama_weights(&lw);
    }

    velri_context_destroy(ctx);
    free(ref_4);
    free(ref_6);
    free(cpu_std_logits);
    free(cpu_lat_logits);
    free(vk_std_logits);
    free(vk_lat_logits);
    free(offload_lat_logits);
    free(kv_offload_logits);
    free_host_weights(&w);
    printf("=== GPT-2 & Llama/Qwen End-to-End Suite PASSED ===\n");
    return 0;
}
