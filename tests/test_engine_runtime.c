#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "velri/velri.h"

#include "runtime/sampler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#endif

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

static void pool_destroy(weight_pool *p) {
    for (uint32_t i = 0U; i < p->count; ++i) {
        free(p->ptrs[i]);
    }
    free(p->ptrs);
    memset(p, 0, sizeof(*p));
}

static float *pool_alloc_f32(weight_pool *p, size_t count, uint32_t seed, float scale, float bias) {
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

static void bind_w(velri_model *m, weight_pool *p, const char *name, size_t elems, uint32_t seed,
                   float scale, float bias) {
    float *w = pool_alloc_f32(p, elems, seed, scale, bias);
    CHECK_RC(velri_model_bind_weight(m, name, w, elems * sizeof(float)), name);
}

static void bind_wf(velri_model *m, weight_pool *p, uint32_t *seed, float scale, float bias,
                    size_t elems, const char *fmt, uint32_t layer) {
    char name[128];
    snprintf(name, sizeof(name), fmt, layer);
    bind_w(m, p, name, elems, (*seed)++, scale, bias);
}

static void populate_gpt2_weights(velri_model *m, const velri_model_desc *d, weight_pool *p) {
    uint32_t V = d->vocab_size, P = d->max_position_embeddings;
    uint32_t D = d->hidden_size, I = d->intermediate_size, seed = 100U;

    bind_w(m, p, "transformer.wte.weight", (size_t)V * D, seed++, 0.12f, 0.0f);
    bind_w(m, p, "transformer.wpe.weight", (size_t)P * D, seed++, 0.04f, 0.0f);
    for (uint32_t l = 0U; l < d->num_hidden_layers; ++l) {
        bind_wf(m, p, &seed, 0.05f, 1.0f, D,                "transformer.h.%u.ln_1.weight", l);
        bind_wf(m, p, &seed, 0.02f, 0.0f, D,                "transformer.h.%u.ln_1.bias", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)3U*D*D,   "transformer.h.%u.attn.c_attn.weight", l);
        bind_wf(m, p, &seed, 0.02f, 0.0f, 3U * D,           "transformer.h.%u.attn.c_attn.bias", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)D * D,    "transformer.h.%u.attn.c_proj.weight", l);
        bind_wf(m, p, &seed, 0.02f, 0.0f, D,                "transformer.h.%u.attn.c_proj.bias", l);
        bind_wf(m, p, &seed, 0.05f, 1.0f, D,                "transformer.h.%u.ln_2.weight", l);
        bind_wf(m, p, &seed, 0.02f, 0.0f, D,                "transformer.h.%u.ln_2.bias", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)I * D,    "transformer.h.%u.mlp.c_fc.weight", l);
        bind_wf(m, p, &seed, 0.02f, 0.0f, I,                "transformer.h.%u.mlp.c_fc.bias", l);
        bind_wf(m, p, &seed, 0.08f, 0.0f, (size_t)D * I,    "transformer.h.%u.mlp.c_proj.weight", l);
        bind_wf(m, p, &seed, 0.02f, 0.0f, D,                "transformer.h.%u.mlp.c_proj.bias", l);
    }
    bind_w(m, p, "transformer.ln_f.weight", D, seed++, 0.05f, 1.0f);
    bind_w(m, p, "transformer.ln_f.bias", D, seed++, 0.02f, 0.0f);
    if (!d->tie_word_embeddings) bind_w(m, p, "lm_head.weight", (size_t)V * D, seed++, 0.08f, 0.0f);
}

static void populate_llama_weights(velri_model *m, const velri_model_desc *d, weight_pool *p) {
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

static float max_abs_diff(const float *a, const float *b, uint32_t n) {
    float mx = 0.0f;
    for (uint32_t i = 0U; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) {
            mx = d;
        }
    }
    return mx;
}

/* ---- Test 1: Zero-Alloc Deterministic Sampler ---- */

static void test_sampler_unit(void) {
    velri_sampler s1, s2;
    CHECK_RC(velri_sampler_init(&s1, 8U, 12345ULL), "sampler init 1");
    CHECK_RC(velri_sampler_init(&s2, 8U, 12345ULL), "sampler init 2");

    /* Greedy argmax with deterministic smallest-index tie-break. */
    const float tied[8] = {1.0f, 3.5f, 2.0f, 3.5f, -INFINITY, 0.0f, 3.5f, -2.0f};
    velri_sampling_params greedy = {.temperature = 0.0f};
    int32_t tok = -1;
    CHECK_RC(velri_sampler_sample(&s1, tied, &greedy, &tok), "greedy sample");
    CHECK(tok == 1, "greedy breaks ties to smallest index");

    /* Reject NaN, +Inf, and all--Inf logits, and verify `reseed` is NOT applied on error! */
    const float nan_logits[8] = {0.0f, NAN, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    CHECK(velri_sampler_sample(&s1, nan_logits, &greedy, &tok) == VELRI_RC_ERR_EXEC,
          "NaN logits fail");
    const float pos_inf_logits[8] = {1.0f, INFINITY, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint64_t rng_before = s1.rng_state;
    velri_sampling_params reseed_p = {.temperature = 1.0f, .seed = 999999ULL, .reseed = true};
    CHECK(velri_sampler_sample(&s1, pos_inf_logits, &reseed_p, &tok) == VELRI_RC_ERR_EXEC,
          "+Inf logits rejected as EXEC error");
    CHECK(s1.rng_state == rng_before, "error does not mutate RNG state");
    const float inf_logits[8] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY,
                                 -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    CHECK(velri_sampler_sample(&s1, inf_logits, &greedy, &tok) == VELRI_RC_ERR_EXEC,
          "all -Inf logits fail");

    /* Extreme underflow / zero-weight tail MUST NEVER be sampled! */
    const float extreme[8] = {0.0f, -1000.0f, -1000.0f, -1000.0f,
                              -1000.0f, -1000.0f, -1000.0f, -1000.0f};
    velri_sampling_params unit_t = {.temperature = 1.0f};
    for (int i = 0; i < 500; ++i) {
        CHECK_RC(velri_sampler_sample(&s1, extreme, &unit_t, &tok), "extreme underflow sample");
        CHECK(tok == 0, "zero-probability tail token never drawn");
    }

    /* Combined top_k + top_p (uses O(n log k) heap path) restricts strictly to top-2 nucleus. */
    const float combo[8] = {-5.0f, -4.0f, 2.0f, -3.0f, 2.5f, 1.5f, -1.0f, -6.0f};
    velri_sampling_params p_combo = {.temperature = 1.0f, .top_k = 3U, .top_p = 0.70f};
    for (int i = 0; i < 200; ++i) {
        CHECK_RC(velri_sampler_sample(&s1, combo, &p_combo, &tok), "top_k+top_p sample");
        CHECK(tok == 4 || tok == 2, "top_k=3 + top_p=0.70 selects strictly within top-2 nucleus");
    }

    /* Invalid sampling params rejected. */
    velri_sampling_params bad_t = {.temperature = -0.1f};
    velri_sampling_params bad_p = {.temperature = 1.0f, .top_p = 1.2f};
    velri_sampling_params bad_m = {.temperature = 1.0f, .min_p = -0.05f};
    CHECK(velri_sampler_validate_params(&bad_t) == VELRI_RC_ERR_INVALID, "neg temp rejected");
    CHECK(velri_sampler_validate_params(&bad_p) == VELRI_RC_ERR_INVALID, "top_p > 1 rejected");
    CHECK(velri_sampler_validate_params(&bad_m) == VELRI_RC_ERR_INVALID, "neg min_p rejected");

    /* top_k = 2 restricts draws strictly to indices {4, 2}. */
    const float logits[8] = {-5.0f, -4.0f, 2.0f, -3.0f, 3.0f, -2.0f, -1.0f, -6.0f};
    velri_sampling_params p_topk = {.temperature = 1.0f, .top_k = 2U};
    uint32_t seen_2 = 0U, seen_4 = 0U;
    for (int i = 0; i < 200; ++i) {
        CHECK_RC(velri_sampler_sample(&s1, logits, &p_topk, &tok), "top_k sample");
        CHECK(tok == 2 || tok == 4, "top_k=2 only emits indices 2 or 4");
        if (tok == 2) seen_2++;
        if (tok == 4) seen_4++;
    }
    CHECK(seen_2 > 10U && seen_4 > 10U, "top_k=2 samples both top candidates");

    /* top_p = 0.55 on probs ~ [0.665, 0.245, 0.090] keeps ONLY index 4! */
    const float sharp[8] = {0.0f, -20.0f, 1.0f, -20.0f, 2.0f, -20.0f, -20.0f, -20.0f};
    velri_sampling_params p_topp = {.temperature = 1.0f, .top_p = 0.55f};
    for (int i = 0; i < 64; ++i) {
        CHECK_RC(velri_sampler_sample(&s1, sharp, &p_topp, &tok), "top_p narrow");
        CHECK(tok == 4, "top_p=0.55 truncates to top-1 token covering 0.665");
    }

    /* min_p = 0.25 with exp(1-2)=0.368 >= 0.25 and exp(0-2)=0.135 < 0.25 keeps {4, 2} only! */
    velri_sampling_params p_minp = {.temperature = 1.0f, .min_p = 0.25f};
    for (int i = 0; i < 200; ++i) {
        CHECK_RC(velri_sampler_sample(&s1, sharp, &p_minp, &tok), "min_p sample");
        CHECK(tok == 4 || tok == 2, "min_p=0.25 filters index 0 (0.135 < 0.25)");
    }

    /* Exact cross-instance and reset determinism. */
    velri_sampler_reset(&s1);
    velri_sampler_reset(&s2);
    velri_sampling_params stoch = {.temperature = 0.9f, .top_k = 5U, .top_p = 0.95f, .min_p = 0.05f};
    int32_t seq[64];
    for (int i = 0; i < 64; ++i) {
        int32_t a = -1, b = -1;
        CHECK_RC(velri_sampler_sample(&s1, logits, &stoch, &a), "det s1");
        CHECK_RC(velri_sampler_sample(&s2, logits, &stoch, &b), "det s2");
        CHECK(a == b, "identical seeds produce identical streams");
        seq[i] = a;
    }
    velri_sampler_reset(&s1);
    for (int i = 0; i < 64; ++i) {
        int32_t a = -1;
        CHECK_RC(velri_sampler_sample(&s1, logits, &stoch, &a), "det reset");
        CHECK(a == seq[i], "reset reproduces exact stream");
    }

    velri_sampler_destroy(&s1);
    velri_sampler_destroy(&s2);
    printf("  [PASS] Zero-alloc deterministic sampler (greedy, top_k, top_p, min_p, NaN/Inf, seed).\n");
}

/* ---- Test 2: Parent-Before-Child Lifecycle & Diagnostic Errors ---- */

static void test_cpu_pool_configuration(void) {
    const char fixed[] = "{\"model\":{},\"backends\":{\"cpu\":{\"threads\":2,\"affinity\":\"none\"}}}";
    const char automatic[] = "{\"backends\":{\"cpu\":{\"threads\":0,\"reserve-cores\":4294967295}}}";
    const char required[] = "{\"backends\":{\"cpu\":{\"affinity\":\"required\"}}}";
    char err[256];
    velri_ctx *ctx = nullptr;
    velri_ctx_options three = {.cpu_threads = 3U};
    CHECK(velri_ctx_create_from_json(&three, fixed, sizeof(fixed) - 1U, &ctx, err, sizeof(err)) ==
              VELRI_RC_ERR_INVALID && !ctx && strstr(err, "cpu_threads"),
          "reject conflicting CPU options during context creation");
    CHECK(velri_ctx_create_from_json(nullptr, required, sizeof(required) - 1U, &ctx, err, sizeof(err)) ==
              VELRI_RC_ERR_UNSUPPORTED && !ctx && strstr(err, "pinning"),
          "required affinity rejected before pool creation");
    CHECK(velri_ctx_create_from_json(nullptr, "{", 1U, &ctx, err, sizeof(err)) ==
              VELRI_RC_ERR_INVALID && !ctx && err[0],
          "return JSON diagnostic before context exists");
    CHECK_RC(velri_ctx_create_from_json(nullptr, automatic, sizeof(automatic) - 1U, &ctx,
                                        err, sizeof(err)), "auto context from backend-only object");
    velri_device_info info;
    CHECK_RC(velri_ctx_device_info(ctx, 0U, &info), "get actual automatic CPU pool");
    CHECK(info.cpu_workers == 1U, "reserving all cores leaves one worker");

    velri_model_desc md = {
        .arch = VELRI_ARCH_GPT2, .vocab_size = 16U, .max_position_embeddings = 8U,
        .hidden_size = 16U, .num_hidden_layers = 1U, .num_attention_heads = 2U,
        .intermediate_size = 32U, .hidden_act = VELRI_ACTIVATION_GELU_TANH,
        .tie_word_embeddings = true,
    };
    velri_model *model = nullptr;
    velri_program *prog = nullptr;
    CHECK_RC(velri_model_create_synthetic(ctx, &md, &model), "auto context model");
    const char reserve[] = "{\"model\":{},\"backends\":{\"cpu\":{\"reserve-cores\":0}}}";
    velri_program_options opts = {.placement_json = reserve, .placement_json_len = sizeof(reserve) - 1U};
    CHECK(velri_program_prepare(ctx, model, &opts, &prog) == VELRI_RC_ERR_INVALID &&
              strstr(velri_ctx_last_error(ctx), "reserve-cores"),
          "prepare rejects changes to an automatic context's reserve");
    opts = (velri_program_options){.placement_json = fixed, .placement_json_len = sizeof(fixed) - 1U};
    CHECK(velri_program_prepare(ctx, model, &opts, &prog) == VELRI_RC_ERR_INVALID &&
              strstr(velri_ctx_last_error(ctx), "cpu.threads"),
          "prepare rejects different CPU policy");
    CHECK_RC(velri_model_destroy(model), "destroy automatic context model");
    CHECK_RC(velri_ctx_destroy(ctx), "destroy automatic context");

    CHECK_RC(velri_ctx_create_from_json(nullptr, fixed, sizeof(fixed) - 1U, &ctx, err, sizeof(err)),
             "open pool before preparing program");
    CHECK_RC(velri_ctx_device_info(ctx, 0U, &info), "fixed CPU device info");
    CHECK(info.cpu_workers == 2U, "JSON controls actual worker count");
    CHECK_RC(velri_model_create_synthetic(ctx, &md, &model), "fixed context model");
    weight_pool pool = {0};
    populate_gpt2_weights(model, &md, &pool);
    CHECK_RC(velri_program_prepare(ctx, model, &opts, &prog), "reuse context config at prepare");
    CHECK_RC(velri_program_destroy(prog), "destroy program with explicit CPU settings");
    CHECK_RC(velri_program_prepare(ctx, model, nullptr, &prog), "omitted CPU settings inherit pool");
    char report[4096];
    CHECK(velri_program_inspect(prog, report, sizeof(report)) > 0U &&
              strstr(report, "CPU pool: threads=2, affinity=none"),
          "inspection reports actual context settings without program JSON");
    CHECK_RC(velri_program_destroy(prog), "destroy fixed program");
    CHECK_RC(velri_model_destroy(model), "destroy fixed model");
    CHECK_RC(velri_ctx_destroy(ctx), "destroy fixed context");
    pool_destroy(&pool);
}

typedef struct {
    velri_ctx *ctx;
    velri_model *model;
} prepare_thread_args;

#if defined(_WIN32)
static DWORD WINAPI prepare_while_binding(void *arg) {
#else
static void *prepare_while_binding(void *arg) {
#endif
    prepare_thread_args *args = arg;
    for (unsigned i = 0; i < 40; ++i) {
        velri_program *prog = nullptr;
        velri_rc rc = velri_program_prepare(args->ctx, args->model, nullptr, &prog);
        CHECK(rc == VELRI_RC_OK || rc == VELRI_RC_ERR_BUSY, "prepare during binding");
        CHECK_RC(velri_program_destroy(prog), "release concurrent program");
    }
    return 0;
}

static void exercise_weight_registry(velri_ctx *ctx, velri_model *model) {
    static const float weight = 1.0f;
    uint32_t initial_count = velri_model_weight_count(model);
    prepare_thread_args args = {ctx, model};
#if defined(_WIN32)
    HANDLE thread = CreateThread(nullptr, 0, prepare_while_binding, &args, 0, nullptr);
    CHECK(thread != nullptr, "start concurrent prepare");
#else
    pthread_t thread;
    CHECK(pthread_create(&thread, nullptr, prepare_while_binding, &args) == 0, "start concurrent prepare");
#endif
    /* Grow through several rehashes while another thread reads the original model weights. */
    for (unsigned i = 0; i < 256; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "unused.weight.%u", i);
        velri_rc rc;
        do {
            rc = velri_model_bind_weight(model, name, &weight, sizeof(weight));
        } while (rc == VELRI_RC_ERR_BUSY);
        CHECK_RC(rc, "register weight during concurrent preparation");
    }
#if defined(_WIN32)
    CHECK(WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0, "join concurrent prepare");
    CloseHandle(thread);
#else
    CHECK(pthread_join(thread, nullptr) == 0, "join concurrent prepare");
#endif
    CHECK_RC(velri_model_bind_weight(model, "unused.weight.0", &weight, sizeof(weight)), "rebind");
    CHECK(velri_model_weight_count(model) == initial_count + 256U, "distinct weight names");
}

static void test_lifecycle_and_diagnostics(void) {
    velri_ctx *ctx = nullptr;
    velri_ctx_options copts = {.cpu_threads = 2U, .enable_vulkan = false};
    CHECK_RC(velri_ctx_create(&copts, &ctx), "ctx create");
    CHECK(velri_ctx_device_count(ctx) >= 1U, "at least cpu:0 present");

    velri_device_info dinfo;
    CHECK_RC(velri_ctx_device_info(ctx, 0U, &dinfo), "device 0 info");
    CHECK(strcmp(dinfo.backend_name, "cpu") == 0 && dinfo.is_host_memory, "device 0 is cpu");
    CHECK(velri_ctx_device_info(ctx, 999U, &dinfo) == VELRI_RC_ERR_NOT_FOUND, "oob device");

    velri_model_desc md = {
        .arch = VELRI_ARCH_GPT2,
        .vocab_size = 32U,
        .max_position_embeddings = 16U,
        .hidden_size = 32U,
        .num_hidden_layers = 1U,
        .num_attention_heads = 2U,
        .intermediate_size = 64U,
        .norm_eps = 1e-5f,
        .hidden_act = VELRI_ACTIVATION_GELU_TANH,
        .tie_word_embeddings = true,
        .weight_type = VELRI_TYPE_F32,
    };
    velri_model *model = nullptr;
    CHECK_RC(velri_model_create_synthetic(ctx, &md, &model), "model create");

    /* Missing weights must fail with NOT_FOUND + descriptive diagnostic naming the weight. */
    velri_program *prog = nullptr;
    CHECK(velri_program_prepare(ctx, model, nullptr, &prog) == VELRI_RC_ERR_NOT_FOUND,
          "missing weight fails prepare");
    CHECK(strstr(velri_ctx_last_error(ctx), "transformer.wte.weight") != nullptr,
          "diagnostic names missing weight");

    weight_pool pool = {0};
    populate_gpt2_weights(model, &md, &pool);
    exercise_weight_registry(ctx, model);

    velri_program_options popts = {.max_ctx = 16U, .prefill_chunk_size = 4U};
    CHECK_RC(velri_program_prepare(ctx, model, &popts, &prog), "prog prepare");

    velri_session *sess = nullptr;
    CHECK_RC(velri_session_create(prog, nullptr, &sess), "sess create");

    /* Parent-before-child destruction must return BUSY without destroying parents! */
    CHECK(velri_ctx_destroy(ctx) == VELRI_RC_ERR_BUSY, "ctx destroy busy with live model/prog");
    CHECK(velri_ctx_last_error(ctx)[0] != '\0', "ctx busy diagnostic set");
    CHECK(velri_model_destroy(model) == VELRI_RC_ERR_BUSY, "model destroy busy with live prog");
    CHECK(velri_program_destroy(prog) == VELRI_RC_ERR_BUSY, "prog destroy busy with live sess");

    /* Mutating model weights while a Program depends on it must fail with BUSY. */
    float dummy = 1.0f;
    CHECK(velri_model_bind_weight(model, "extra.w", &dummy, sizeof(dummy)) == VELRI_RC_ERR_BUSY,
          "model immutable while program alive");

    /* Clean child-before-parent destruction succeeds. */
    CHECK_RC(velri_session_destroy(sess), "sess destroy");
    CHECK_RC(velri_program_destroy(prog), "prog destroy");
    CHECK_RC(velri_model_destroy(model), "model destroy");
    CHECK_RC(velri_ctx_destroy(ctx), "ctx destroy");
    pool_destroy(&pool);
    printf("  [PASS] Strict lifecycle parent-before-child BUSY guards & error diagnostics.\n");
}

/* ---- Test 3: End-to-End Synthetic GPT-2 (Chunked Prefill, Decode, Isolation, Validation) ---- */

static void test_gpt2_runtime_e2e(void) {
    velri_ctx *ctx = nullptr;
    velri_ctx_options copts = {.cpu_threads = 2U, .enable_vulkan = false};
    CHECK_RC(velri_ctx_create(&copts, &ctx), "ctx create");

    velri_model_desc md = {
        .arch = VELRI_ARCH_GPT2,
        .vocab_size = 64U,
        .max_position_embeddings = 32U,
        .hidden_size = 32U,
        .num_hidden_layers = 2U,
        .num_attention_heads = 4U,
        .intermediate_size = 64U,
        .norm_eps = 1e-5f,
        .hidden_act = VELRI_ACTIVATION_GELU_TANH,
        .tie_word_embeddings = true,
        .weight_type = VELRI_TYPE_F32,
    };
    velri_model *model = nullptr;
    CHECK_RC(velri_model_create_synthetic(ctx, &md, &model), "gpt2 model");
    weight_pool pool = {0};
    populate_gpt2_weights(model, &md, &pool);

    velri_program_options popts = {.max_ctx = 24U, .prefill_chunk_size = 4U};
    velri_program *prog = nullptr;
    CHECK_RC(velri_program_prepare(ctx, model, &popts, &prog), "gpt2 prog");
    CHECK(!velri_program_has_distinct_prefill(prog), "default cpu plan is unified");

    velri_session *s_chunk = nullptr, *s_tok = nullptr, *s_alt = nullptr;
    CHECK_RC(velri_session_create(prog, nullptr, &s_chunk), "s_chunk");
    CHECK_RC(velri_session_create(prog, nullptr, &s_tok), "s_tok");
    CHECK_RC(velri_session_create(prog, nullptr, &s_alt), "s_alt");

    const int32_t prompt[10] = {3, 17, 42, 5, 61, 0, 29, 11, 8, 50};
    const int32_t alt_prompt[6] = {60, 55, 1, 2, 33, 44};
    velri_sampling_params greedy = {.temperature = 0.0f};

    /* Pre-accept validation errors leave session clean & kv_len == 0. */
    velri_step_result res;
    int32_t bad_id = 64;
    CHECK(velri_session_step(s_chunk, &bad_id, 1U, &greedy, nullptr, &res) == VELRI_RC_ERR_INVALID,
          "oob token rejected");
    CHECK(!velri_session_needs_reset(s_chunk) && velri_session_kv_len(s_chunk) == 0U,
          "oob token left session clean");
    int32_t neg_id = -1;
    CHECK(velri_session_step(s_chunk, &neg_id, 1U, &greedy, nullptr, &res) == VELRI_RC_ERR_INVALID,
          "negative token rejected");

    /* Session A: chunked prefill 10 tokens (4 + 4 + 2) in one call. */
    velri_step_result r_chunk;
    CHECK_RC(velri_session_step(s_chunk, prompt, 10U, &greedy, nullptr, &r_chunk), "chunk prefill");
    CHECK(r_chunk.stop_reason == VELRI_STEP_STOP_OK && r_chunk.tokens_processed == 10U &&
              r_chunk.kv_len == 10U && r_chunk.last_step_allocs == 0U,
          "chunk prefill metadata & 0 allocs");
    float logits_chunk[64];
    memcpy(logits_chunk, r_chunk.logits, sizeof(logits_chunk));

    /* Interleave Session C with completely different prompt to test multi-session isolation! */
    velri_step_result r_alt;
    CHECK_RC(velri_session_step(s_alt, alt_prompt, 6U, &greedy, nullptr, &r_alt), "alt prefill");
    CHECK(r_alt.kv_len == 6U && r_alt.last_step_allocs == 0U, "alt prefill ok");
    float logits_alt_first[64];
    memcpy(logits_alt_first, r_alt.logits, sizeof(logits_alt_first));

    /* Session B: token-by-token prefill of the same 10 tokens. */
    velri_step_result r_tok = {0};
    for (uint32_t i = 0U; i < 10U; ++i) {
        CHECK_RC(velri_session_step(s_tok, &prompt[i], 1U, &greedy, nullptr, &r_tok), "tok step");
        CHECK(r_tok.last_step_allocs == 0U, "tok step 0 allocs");
    }
    float diff_pre = max_abs_diff(logits_chunk, r_tok.logits, 64U);
    CHECK(diff_pre < 1e-4f, "GPT-2 chunked prefill matches token-by-token");
    CHECK(r_chunk.sampled_token == r_tok.sampled_token, "GPT-2 greedy token matches");

    /* 4 autoregressive decode steps: both sessions stay locked in sync despite Session C! */
    int32_t next_tok = r_chunk.sampled_token;
    float max_dec_diff = 0.0f;
    for (int step = 0; step < 4; ++step) {
        CHECK_RC(velri_session_step(s_chunk, &next_tok, 1U, &greedy, nullptr, &r_chunk), "dec A");
        CHECK_RC(velri_session_step(s_tok, &next_tok, 1U, &greedy, nullptr, &r_tok), "dec B");
        CHECK(r_chunk.last_step_allocs == 0U && r_tok.last_step_allocs == 0U, "decode 0 allocs");
        float d = max_abs_diff(r_chunk.logits, r_tok.logits, 64U);
        if (d > max_dec_diff) max_dec_diff = d;
        CHECK(d < 1e-4f && r_chunk.sampled_token == r_tok.sampled_token, "decode parity");
        next_tok = r_chunk.sampled_token;
    }

    /* Reset Session C and re-run `alt_prompt` -> bitwise identical logits! */
    CHECK_RC(velri_session_reset(s_alt), "reset alt");
    CHECK(velri_session_kv_len(s_alt) == 0U && !velri_session_needs_reset(s_alt), "alt reset state");
    CHECK_RC(velri_session_step(s_alt, alt_prompt, 6U, &greedy, nullptr, &r_alt), "alt rerun");
    CHECK(max_abs_diff(logits_alt_first, r_alt.logits, 64U) < 1e-6f, "reset exact reproducibility");

    /* Context overflow check (14 + 11 > 24 -> OVERFLOW, session stays clean). */
    int32_t many[11] = {0};
    CHECK(velri_session_step(s_chunk, many, 11U, &greedy, nullptr, &res) == VELRI_RC_ERR_OVERFLOW,
          "context overflow rejected");
    CHECK(!velri_session_needs_reset(s_chunk) && velri_session_kv_len(s_chunk) == 14U,
          "context overflow leaves session intact");

    CHECK_RC(velri_session_destroy(s_alt), "destroy s_alt");
    CHECK_RC(velri_session_destroy(s_tok), "destroy s_tok");
    CHECK_RC(velri_session_destroy(s_chunk), "destroy s_chunk");
    CHECK_RC(velri_program_destroy(prog), "destroy prog");
    CHECK_RC(velri_model_destroy(model), "destroy model");
    CHECK_RC(velri_ctx_destroy(ctx), "destroy ctx");
    pool_destroy(&pool);
    printf("  [PASS] GPT-2 E2E runtime (prefill_diff=%.2e, dec_diff=%.2e, 3 isolated sessions, 0 allocs).\n",
           diff_pre, max_dec_diff);
}

/* ---- Test 4: Llama/Qwen Auto-RoPE + Cancellation & Reset Semantics ---- */

typedef struct {
    velri_cancel *cancel;
    uint32_t delay_ms;
} cancel_delay_ctx;

#if defined(_WIN32)
static DWORD WINAPI cancel_thread_fn(LPVOID arg) {
    cancel_delay_ctx *c = (cancel_delay_ctx *)arg;
    Sleep(c->delay_ms);
    velri_cancel_trigger(c->cancel);
    return 0;
}
#else
static void *cancel_thread_fn(void *arg) {
    cancel_delay_ctx *c = (cancel_delay_ctx *)arg;
    struct timespec ts = {
        .tv_sec = (time_t)(c->delay_ms / 1000U),
        .tv_nsec = (long)((c->delay_ms % 1000U) * 1000000U),
    };
    nanosleep(&ts, nullptr);
    velri_cancel_trigger(c->cancel);
    return nullptr;
}
#endif

static void test_llama_auto_rope_and_cancellation(void) {
    velri_ctx *ctx = nullptr;
    velri_ctx_options copts = {.cpu_threads = 2U, .enable_vulkan = false};
    CHECK_RC(velri_ctx_create(&copts, &ctx), "ctx create");

    velri_model_desc md = {
        .arch = VELRI_ARCH_LLAMA,
        .vocab_size = 64U,
        .max_position_embeddings = 64U,
        .hidden_size = 64U,
        .num_hidden_layers = 2U,
        .num_attention_heads = 4U,
        .num_key_value_heads = 2U, /* GQA 2:1 */
        .head_dim = 16U,
        .intermediate_size = 128U,
        .norm_eps = 1e-6f,
        .hidden_act = VELRI_ACTIVATION_SILU,
        .tie_word_embeddings = false,
        .weight_type = VELRI_TYPE_F32,
        .rope_theta = 100000.0f,
        .rotary_dim = 16U,
        .rope_mode = VELRI_ROTARY_HALF,
        .qkv_bias = true, /* Qwen2-style QKV bias */
    };
    velri_model *model = nullptr;
    CHECK_RC(velri_model_create_synthetic(ctx, &md, &model), "llama model");
    weight_pool pool = {0};
    populate_llama_weights(model, &md, &pool); /* Intentionally omits model.rope.cos/sin -> auto! */

    const char *cfg_json =
        "{\n"
        "  \"context\": { \"size\": 64 },\n"
        "  \"model\": {\n"
        "    \"store\": \"cpu:0\",\n"
        "    \"compute\": \"cpu:0\",\n"
        "    \"sampling\": { \"temperature\": 0.0, \"top-p\": 0, \"seed\": 777 }\n"
        "  }\n"
        "}\n";
    velri_program_options popts = {
        .prefill_chunk_size = 4U,
        .placement_json = cfg_json,
        .placement_json_len = strlen(cfg_json),
    };
    velri_program *prog = nullptr;

    /* Strict config & validation guards (M4). */
    {
        velri_program *bad_p = nullptr;
        velri_program_options empty_len_opts = {.placement_json = cfg_json, .placement_json_len = 0U};
        CHECK(velri_program_prepare(ctx, model, &empty_len_opts, &bad_p) == VELRI_RC_ERR_INVALID,
              "non-null placement_json with len=0 rejected");

        const char *kvq_json =
            "{\"model\": {\"store\": \"cpu:0\", \"compute\": \"cpu:0\"},"
            " \"kv-quantization\": {\"k-bits\": 8, \"v-bits\": 8}}";
        velri_program_options kvq_opts = {.placement_json = kvq_json,
                                          .placement_json_len = strlen(kvq_json)};
        CHECK(velri_program_prepare(ctx, model, &kvq_opts, &bad_p) == VELRI_RC_ERR_UNSUPPORTED,
              "unsupported kv-quantization rejected explicitly");

        const char *aff_json =
            "{\"model\": {\"store\": \"cpu:0\", \"compute\": \"cpu:0\"},"
            " \"backends\": {\"cpu\": {\"affinity\": \"required\"}}}";
        velri_program_options aff_opts = {.placement_json = aff_json,
                                          .placement_json_len = strlen(aff_json)};
        CHECK(velri_program_prepare(ctx, model, &aff_opts, &bad_p) == VELRI_RC_ERR_UNSUPPORTED,
              "unsupported cpu affinity='required' rejected explicitly");

        /* Partial custom RoPE binding (only cos, missing sin) must fail with INVALID. */
        velri_model *partial_m = nullptr;
        CHECK_RC(velri_model_create_synthetic(ctx, &md, &partial_m), "partial rope model");
        weight_pool partial_pool = {0};
        populate_llama_weights(partial_m, &md, &partial_pool);
        bind_w(partial_m, &partial_pool, "model.rope.cos", (size_t)64U * 8U, 999U, 0.1f, 0.0f);
        CHECK(velri_program_prepare(ctx, partial_m, &popts, &bad_p) == VELRI_RC_ERR_INVALID,
              "partial custom RoPE (cos without sin) rejected");
        CHECK_RC(velri_model_destroy(partial_m), "destroy partial_m");
        pool_destroy(&partial_pool);
    }

    CHECK_RC(velri_program_prepare(ctx, model, &popts, &prog), "llama prog");

    /* Inspect output check. */
    char inspect_buf[4096];
    size_t in_len = velri_program_inspect(prog, inspect_buf, sizeof(inspect_buf));
    CHECK(in_len > 0U && strstr(inspect_buf, "Resolved Placement:") != nullptr &&
              strstr(inspect_buf, "Memory Plan:") != nullptr &&
              strstr(inspect_buf, "Program: arch=llama") != nullptr,
          "program inspect report formatted cleanly");

    velri_sampling_params def_samp;
    CHECK_RC(velri_program_default_sampling(prog, &def_samp), "default sampling");
    CHECK(def_samp.temperature == 0.0f && def_samp.top_p == 0.0f && def_samp.seed == 777ULL, "sampling inherited from JSON");

    velri_session *s1 = nullptr, *s2 = nullptr;
    CHECK_RC(velri_session_create(prog, nullptr, &s1), "llama s1");
    CHECK_RC(velri_session_create(prog, nullptr, &s2), "llama s2");

    const int32_t prompt[6] = {5, 19, 31, 47, 2, 60};
    velri_step_result r1, r2;
    CHECK_RC(velri_session_step(s1, prompt, 6U, &def_samp, nullptr, &r1), "llama prefill");
    CHECK(r1.last_step_allocs == 0U && r1.kv_len == 6U, "llama prefill 0 allocs");
    for (uint32_t i = 0U; i < 6U; ++i) {
        CHECK_RC(velri_session_step(s2, &prompt[i], 1U, &def_samp, nullptr, &r2), "llama tok");
    }
    float diff = max_abs_diff(r1.logits, r2.logits, 64U);
    CHECK(diff < 1e-4f && r1.sampled_token == r2.sampled_token,
          "Llama auto-RoPE GQA prefill matches token-by-token");

    /* Pre-accept cancellation: leaves session completely intact (`needs_reset == false`). */
    velri_cancel *cancel = nullptr;
    CHECK_RC(velri_cancel_create(&cancel), "cancel create");
    velri_cancel_trigger(cancel);
    int32_t next_t = r1.sampled_token;
    velri_step_result r_can;
    CHECK(velri_session_step(s1, &next_t, 1U, &def_samp, cancel, &r_can) == VELRI_RC_ERR_CANCELLED,
          "pre-accept cancel returns CANCELLED");
    CHECK(r_can.stop_reason == VELRI_STEP_STOP_CANCELLED && r_can.tokens_processed == 0U &&
              !velri_session_needs_reset(s1) && velri_session_kv_len(s1) == 6U,
          "pre-accept cancel leaves session clean");

    /* Reset cancel handle and continue cleanly! */
    velri_cancel_reset(cancel);
    CHECK_RC(velri_session_step(s1, &next_t, 1U, &def_samp, cancel, &r1), "continue after pre-cancel");
    CHECK_RC(velri_session_step(s2, &next_t, 1U, &def_samp, nullptr, &r2), "s2 decode");
    CHECK(max_abs_diff(r1.logits, r2.logits, 64U) < 1e-4f, "clean continuation parity");

    /* Mid-step cancellation over a multi-chunk step to trigger post-accept cancel -> reset. */
    velri_model_desc big_md = md;
    big_md.hidden_size = 192U;
    big_md.num_attention_heads = 6U;
    big_md.num_key_value_heads = 3U;
    big_md.head_dim = 32U;
    big_md.rotary_dim = 32U;
    big_md.intermediate_size = 512U;
    big_md.num_hidden_layers = 6U;
    big_md.max_position_embeddings = 128U;
    velri_model *big_m = nullptr;
    CHECK_RC(velri_model_create_synthetic(ctx, &big_md, &big_m), "big llama");
    weight_pool big_pool = {0};
    populate_llama_weights(big_m, &big_md, &big_pool);
    velri_program_options big_opts = {.max_ctx = 128U, .prefill_chunk_size = 2U};
    velri_program *big_p = nullptr;
    CHECK_RC(velri_program_prepare(ctx, big_m, &big_opts, &big_p), "big prog");
    velri_session *big_s = nullptr;
    CHECK_RC(velri_session_create(big_p, nullptr, &big_s), "big sess");

    int32_t long_prompt[128];
    for (uint32_t i = 0U; i < 128U; ++i) long_prompt[i] = (int32_t)(i % 64U);

    bool observed_dirty_cancel = false;
    for (uint32_t delay = 1U; delay <= 8U && !observed_dirty_cancel; ++delay) {
        CHECK_RC(velri_session_reset(big_s), "big reset");
        velri_cancel_reset(cancel);
        cancel_delay_ctx dctx = {.cancel = cancel, .delay_ms = delay};
#if defined(_WIN32)
        HANDLE th = CreateThread(nullptr, 0, cancel_thread_fn, &dctx, 0, nullptr);
        CHECK(th != nullptr, "CreateThread");
        velri_rc crc = velri_session_step(big_s, long_prompt, 128U, &def_samp, cancel, &r_can);
        WaitForSingleObject(th, INFINITE);
        CloseHandle(th);
#else
        pthread_t th;
        CHECK(pthread_create(&th, nullptr, cancel_thread_fn, &dctx) == 0, "pthread_create");
        velri_rc crc = velri_session_step(big_s, long_prompt, 128U, &def_samp, cancel, &r_can);
        pthread_join(th, nullptr);
#endif
        if (crc == VELRI_RC_ERR_CANCELLED && velri_session_needs_reset(big_s)) {
            observed_dirty_cancel = true;
            CHECK(r_can.stop_reason == VELRI_STEP_STOP_CANCELLED, "dirty cancel stop_reason");
            /* Next step MUST return BUSY until velri_session_reset() is called! */
            int32_t t0 = 1;
            CHECK(velri_session_step(big_s, &t0, 1U, &def_samp, nullptr, &r_can) == VELRI_RC_ERR_BUSY,
                  "dirty session rejects step with BUSY");
            CHECK_RC(velri_session_reset(big_s), "reset clears dirty session");
            CHECK(!velri_session_needs_reset(big_s) && velri_session_kv_len(big_s) == 0U,
                  "session clean after reset");
            CHECK_RC(velri_session_step(big_s, prompt, 4U, &def_samp, nullptr, &r_can),
                     "session runs cleanly after reset");
        }
    }

    velri_cancel_destroy(cancel);
    CHECK_RC(velri_session_destroy(big_s), "destroy big_s");
    CHECK_RC(velri_program_destroy(big_p), "destroy big_p");
    CHECK_RC(velri_model_destroy(big_m), "destroy big_m");
    pool_destroy(&big_pool);

    CHECK_RC(velri_session_destroy(s2), "destroy s2");
    CHECK_RC(velri_session_destroy(s1), "destroy s1");
    CHECK_RC(velri_program_destroy(prog), "destroy prog");
    CHECK_RC(velri_model_destroy(model), "destroy model");
    CHECK_RC(velri_ctx_destroy(ctx), "destroy ctx");
    pool_destroy(&pool);
    printf("  [PASS] Llama/Qwen auto-RoPE GQA + inspect + cancel/reset (diff=%.2e, dirty_cancel=%s).\n",
           diff, observed_dirty_cancel ? "verified" : "completed-fast");
}

/* ---- Test 5: Dual-Plan `prefill-compute != compute` (0 Duplication + Vulkan Graph Replay) ---- */

static void test_dual_plan_prefill_compute_and_vulkan(void) {
    velri_ctx *ctx = nullptr;
    velri_ctx_options copts = {.cpu_threads = 2U, .enable_vulkan = true};
    CHECK_RC(velri_ctx_create(&copts, &ctx), "ctx vk");

    bool has_vk = false;
    uint32_t vk_dev_id = 0U;
    uint32_t n_dev = velri_ctx_device_count(ctx);
    for (uint32_t d = 0U; d < n_dev; ++d) {
        velri_device_info di;
        CHECK_RC(velri_ctx_device_info(ctx, d, &di), "dev info");
        if (strcmp(di.backend_name, "vulkan") == 0 && di.local_index == 0U) {
            has_vk = true;
            vk_dev_id = d;
            break;
        }
    }
    if (!has_vk) {
        CHECK_RC(velri_ctx_destroy(ctx), "ctx destroy no-vk");
        printf("  [SKIP] Dual-plan CPU/Vulkan test (Vulkan device unavailable in this build).\n");
        return;
    }

    velri_model_desc md = {
        .arch = VELRI_ARCH_LLAMA,
        .vocab_size = 64U,
        .max_position_embeddings = 32U,
        .hidden_size = 64U,
        .num_hidden_layers = 2U,
        .num_attention_heads = 4U,
        .num_key_value_heads = 2U,
        .head_dim = 16U,
        .intermediate_size = 128U,
        .norm_eps = 1e-6f,
        .hidden_act = VELRI_ACTIVATION_SILU,
        .tie_word_embeddings = false,
        .weight_type = VELRI_TYPE_F32,
        .rope_theta = 10000.0f,
        .rotary_dim = 16U,
        .rope_mode = VELRI_ROTARY_HALF,
        .qkv_bias = false,
    };
    velri_model *model = nullptr;
    CHECK_RC(velri_model_create_synthetic(ctx, &md, &model), "vk llama model");
    weight_pool pool = {0};
    populate_llama_weights(model, &md, &pool);

    /* Reference Program: unified Vulkan (prefill-compute omitted -> inherits vulkan:0). */
    const char *ref_json =
        "{\n"
        "  \"model\": {\n"
        "    \"store\": \"vulkan:0\",\n"
        "    \"compute\": \"vulkan:0\"\n"
        "  }\n"
        "}\n";
    velri_program_options ref_opts = {
        .max_ctx = 32U,
        .prefill_chunk_size = 4U,
        .placement_json = ref_json,
        .placement_json_len = strlen(ref_json),
    };
    velri_program *ref_prog = nullptr;
    CHECK_RC(velri_program_prepare(ctx, model, &ref_opts, &ref_prog), "ref vk prog");
    CHECK(!velri_program_has_distinct_prefill(ref_prog), "ref prog is unified");

    /* Dual-Plan Program: weights & KV stay on vulkan:0 (0 duplication!), prefill MLP runs on
     * vulkan:0, while decode MLP runs on cpu:0 (explicit `prefill-compute != compute`)! */
    const char *dual_json =
        "{\n"
        "  \"model\": {\n"
        "    \"store\": \"vulkan:0\",\n"
        "    \"compute\": \"vulkan:0\",\n"
        "    \"hidden-layers\": {\n"
        "      \"dense-mlp\": {\n"
        "        \"compute\": \"cpu:0\",\n"
        "        \"prefill-compute\": \"vulkan:0\"\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n";
    velri_program_options dual_opts = {
        .max_ctx = 32U,
        .prefill_chunk_size = 4U,
        .placement_json = dual_json,
        .placement_json_len = strlen(dual_json),
    };
    velri_program *dual_prog = nullptr;
    CHECK_RC(velri_program_prepare(ctx, model, &dual_opts, &dual_prog), "dual prog");
    CHECK(velri_program_has_distinct_prefill(dual_prog), "dual prog has distinct prefill plan");

    /* Spurious dual-plan guards: max_ctx==1 OR unused MoE prefill-compute must stay unified! */
    {
        velri_program *u1 = nullptr, *u2 = nullptr;
        velri_program_options ctx1_opts = dual_opts;
        ctx1_opts.max_ctx = 1U;
        ctx1_opts.prefill_chunk_size = 1U;
        CHECK_RC(velri_program_prepare(ctx, model, &ctx1_opts, &u1), "ctx1 prog");
        CHECK(!velri_program_has_distinct_prefill(u1), "max_ctx==1 avoids spurious dual plan");
        CHECK_RC(velri_program_destroy(u1), "destroy u1");

        const char *moe_only_json =
            "{\n"
            "  \"model\": {\n"
            "    \"store\": \"vulkan:0\",\n"
            "    \"compute\": \"vulkan:0\",\n"
            "    \"hidden-layers\": {\n"
            "      \"moe\": {\n"
            "        \"routed-experts\": { \"compute\": \"cpu:0\", \"prefill-compute\": \"vulkan:0\" }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n";
        velri_program_options moe_opts = {
            .max_ctx = 32U,
            .prefill_chunk_size = 4U,
            .placement_json = moe_only_json,
            .placement_json_len = strlen(moe_only_json),
        };
        CHECK_RC(velri_program_prepare(ctx, model, &moe_opts, &u2), "moe_only prog");
        CHECK(!velri_program_has_distinct_prefill(u2),
              "dense Llama ignores unused moe.routed-experts prefill-compute");
        CHECK_RC(velri_program_destroy(u2), "destroy u2");

        const char *inactive_compute[] = {
            "{\"model\":{\"compute\":\"cpu:0\",\"hidden-layers\":{\"attention\":{"
            "\"kv-cache\":{\"prefill-compute\":\"vulkan:0\"}}}}}",
            "{\"model\":{\"compute\":\"cpu:0\",\"prefill-compute\":\"vulkan:0\","
            "\"input-layer\":{\"compute\":\"cpu:0\"},\"lm-head\":{\"compute\":\"cpu:0\"},"
            "\"hidden-layers\":{\"attention\":{\"compute\":\"cpu:0\"},"
            "\"dense-mlp\":{\"compute\":\"cpu:0\"}}}}",
        };
        for (size_t i = 0; i < sizeof(inactive_compute) / sizeof(inactive_compute[0]); ++i) {
            velri_program_options opts = dual_opts;
            opts.placement_json = inactive_compute[i];
            opts.placement_json_len = strlen(inactive_compute[i]);
            velri_program *unified = nullptr;
            CHECK_RC(velri_program_prepare(ctx, model, &opts, &unified), "inactive compute scope");
            CHECK(!velri_program_has_distinct_prefill(unified),
                  "storage-only KV or fully overridden parent compute keeps one plan");
            CHECK_RC(velri_program_destroy(unified), "destroy unified program");
        }
    }

    /* Prove 0 weight duplication AND 0 KV-cache duplication in memory report! */
    velri_program_mem_report ref_rep, dual_rep;
    CHECK_RC(velri_program_memory_report(ref_prog, &ref_rep), "ref mem report");
    CHECK_RC(velri_program_memory_report(dual_prog, &dual_rep), "dual mem report");
    CHECK(!ref_rep.has_distinct_prefill_plan && dual_rep.has_distinct_prefill_plan,
          "mem report flags dual plan");
    CHECK(dual_rep.devices[vk_dev_id].weight_bytes == ref_rep.devices[vk_dev_id].weight_bytes &&
              dual_rep.devices[vk_dev_id].weight_bytes > 0U,
          "dual plan has ZERO weight duplication on vulkan:0");
    CHECK(dual_rep.devices[vk_dev_id].state_bytes == ref_rep.devices[vk_dev_id].state_bytes &&
              dual_rep.devices[vk_dev_id].state_bytes > 0U,
          "dual plan has ZERO KV-cache state duplication on vulkan:0");

    char insp[4096];
    CHECK(velri_program_inspect(dual_prog, insp, sizeof(insp)) > 0U &&
              strstr(insp, "mode=dual-plan (shared weights+state)") != nullptr,
          "dual inspect reports shared weights+state");

    velri_session *s_ref = nullptr, *s_dual = nullptr;
    CHECK_RC(velri_session_create(ref_prog, nullptr, &s_ref), "ref sess");
    CHECK_RC(velri_session_create(dual_prog, nullptr, &s_dual), "dual sess");

    const int32_t prompt[6] = {1, 9, 23, 41, 52, 7};
    velri_sampling_params greedy = {.temperature = 0.0f};
    velri_step_result r_ref, r_dual;

    /* Prefill 6 tokens (chunks 4 + 2 on prefill_slot -> Vulkan MLP!). */
    CHECK_RC(velri_session_step(s_ref, prompt, 6U, &greedy, nullptr, &r_ref), "ref prefill");
    CHECK_RC(velri_session_step(s_dual, prompt, 6U, &greedy, nullptr, &r_dual), "dual prefill");
    CHECK(r_ref.last_step_allocs == 0U && r_dual.last_step_allocs == 0U, "prefill 0 allocs");
    float pre_diff = max_abs_diff(r_ref.logits, r_dual.logits, 64U);
    CHECK(pre_diff < 1e-4f && r_ref.sampled_token == r_dual.sampled_token,
          "dual prefill matches unified Vulkan reference");

    /* Decode 4 tokens (on decode_slot -> CPU MLP reading shared Vulkan KV cache!). */
    int32_t tok = r_ref.sampled_token;
    float max_dec_diff = 0.0f;
    uint64_t ref_replays = 0U, dual_replays = 0U;
    for (int step = 0; step < 4; ++step) {
        CHECK_RC(velri_session_step(s_ref, &tok, 1U, &greedy, nullptr, &r_ref), "ref decode");
        CHECK_RC(velri_session_step(s_dual, &tok, 1U, &greedy, nullptr, &r_dual), "dual decode");
        CHECK(r_ref.last_step_allocs == 0U && r_dual.last_step_allocs == 0U, "decode 0 allocs");
        ref_replays += r_ref.last_step_graph_replays;
        dual_replays += r_dual.last_step_graph_replays;
        float d = max_abs_diff(r_ref.logits, r_dual.logits, 64U);
        if (d > max_dec_diff) max_dec_diff = d;
        CHECK(d < 1e-4f && r_ref.sampled_token == r_dual.sampled_token,
              "dual decode matches unified Vulkan reference");
        tok = r_ref.sampled_token;
    }
    CHECK(ref_replays >= 4U, "unified Vulkan decode replays persistent captured graph");
    CHECK(dual_replays >= 4U, "dual-plan Vulkan sub-batches also replay persistent captured graphs");

    /* Reset dual session and verify clean rerun. */
    CHECK_RC(velri_session_reset(s_dual), "dual reset");
    CHECK_RC(velri_session_step(s_dual, prompt, 6U, &greedy, nullptr, &r_dual), "dual rerun");
    CHECK(r_dual.kv_len == 6U && r_dual.last_step_allocs == 0U, "dual rerun clean");

    CHECK_RC(velri_session_destroy(s_dual), "destroy s_dual");
    CHECK_RC(velri_session_destroy(s_ref), "destroy s_ref");
    CHECK_RC(velri_program_destroy(dual_prog), "destroy dual_prog");
    CHECK_RC(velri_program_destroy(ref_prog), "destroy ref_prog");
    CHECK_RC(velri_model_destroy(model), "destroy model");
    CHECK_RC(velri_ctx_destroy(ctx), "destroy ctx");
    pool_destroy(&pool);
    printf("  [PASS] Dual-plan prefill-compute!=compute (pre_diff=%.2e, dec_diff=%.2e, "
           "vk_w=%zu B [1x], vk_kv=%zu B [1x], replays ref=%llu dual=%llu, 0 allocs).\n",
           pre_diff, max_dec_diff, dual_rep.devices[vk_dev_id].weight_bytes,
           dual_rep.devices[vk_dev_id].state_bytes, (unsigned long long)ref_replays,
           (unsigned long long)dual_replays);
}

int main(void) {
    printf("=== Velri Engine Public C23 Runtime Suite ===\n");
    test_sampler_unit();
    test_cpu_pool_configuration();
    test_lifecycle_and_diagnostics();
    test_gpt2_runtime_e2e();
    test_llama_auto_rope_and_cancellation();
    test_dual_plan_prefill_compute_and_vulkan();
    printf("=== All Engine Runtime Tests PASSED ===\n");
    return 0;
}
