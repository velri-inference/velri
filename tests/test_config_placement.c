#include "config/placement.h"
#include "format/json.h"
#include "model/arch/gpt2.h"

#include <math.h>
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

#define CHECK_OK(st, msg)                                                                          \
    do {                                                                                           \
        velri_status _st = (st);                                                                   \
        if (_st != VELRI_OK) {                                                                     \
            fprintf(stderr, "FAIL [%s:%d]: %s (status=%d)\n", __FILE__, __LINE__, (msg),           \
                    (int)_st);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static char *read_entire_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    CHECK(f != nullptr, "open example json");
    CHECK(fseek(f, 0, SEEK_END) == 0, "fseek end");
    long sz = ftell(f);
    CHECK(sz >= 0, "ftell");
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1U);
    CHECK(buf != nullptr, "malloc file");
    if (sz > 0) {
        CHECK(fread(buf, 1U, (size_t)sz, f) == (size_t)sz, "fread");
    }
    buf[sz] = '\0';
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

/* ---- 1. Streaming C23 JSON Scanner Tests ---- */

static void test_json_scanner(void) {
    /* Valid escapes, UTF-16 surrogate pair, numbers, booleans, null, non-NUL-terminated buffer. */
    const char raw[] =
        "{\"s\":\"hi\\n\\t\\u0041\\uD83D\\uDE00\",\"i\":-42,\"u\":18446744073709551615,"
        "\"f\":3.5e-2,\"t\":true,\"n\":null,\"skip\":{\"a\":[1,2,{\"b\":[true,false,null]}]},"
        "\"arr\":[10,20]}";
    size_t raw_len = sizeof(raw) - 1U;
    /* Allocate exact length with NO trailing NUL to catch OOB reads under ASan. */
    char *exact = (char *)malloc(raw_len);
    CHECK(exact != nullptr, "malloc exact");
    memcpy(exact, raw, raw_len);

    velri_json_reader r;
    velri_json_init(&r, exact, raw_len);
    velri_json_iter it;
    CHECK_OK(velri_json_begin_object(&r, &it), "begin root");

    bool more = false;
    velri_json_str key;
    int seen = 0;
    while (velri_json_next_member(&r, &it, &key, &more) == VELRI_OK && more) {
        if (velri_json_str_eq(key, "s")) {
            char buf[64];
            size_t len = 0U;
            CHECK_OK(velri_json_read_string_buf(&r, buf, sizeof(buf), &len), "read str");
            /* "hi\n\tA" + U+1F600 (F0 9F 98 80) = 9 bytes */
            CHECK(len == 9U, "decoded string length");
            CHECK(memcmp(buf, "hi\n\tA\xF0\x9F\x98\x80", 9U) == 0, "UTF-8 surrogate decode");
            seen |= 1;
        } else if (velri_json_str_eq(key, "i")) {
            int64_t v = 0;
            CHECK_OK(velri_json_read_i64(&r, &v), "read i64");
            CHECK(v == -42, "i64 value");
            seen |= 2;
        } else if (velri_json_str_eq(key, "u")) {
            uint64_t v = 0U;
            CHECK_OK(velri_json_read_u64(&r, &v), "read u64");
            CHECK(v == UINT64_MAX, "u64 max");
            seen |= 4;
        } else if (velri_json_str_eq(key, "f")) {
            double v = 0.0;
            CHECK_OK(velri_json_read_f64(&r, &v), "read f64");
            CHECK(fabs(v - 0.035) < 1e-12, "f64 value");
            seen |= 8;
        } else if (velri_json_str_eq(key, "t")) {
            bool b = false;
            CHECK_OK(velri_json_read_bool(&r, &b), "read bool");
            CHECK(b == true, "bool true");
            seen |= 16;
        } else if (velri_json_str_eq(key, "n")) {
            CHECK_OK(velri_json_read_null(&r), "read null");
            seen |= 32;
        } else if (velri_json_str_eq(key, "skip")) {
            CHECK_OK(velri_json_skip(&r), "skip nested object");
            seen |= 64;
        } else if (velri_json_str_eq(key, "arr")) {
            velri_json_iter ait;
            CHECK_OK(velri_json_begin_array(&r, &ait), "begin arr");
            bool amore = false;
            int64_t sum = 0;
            while (velri_json_next_elem(&r, &ait, &amore) == VELRI_OK && amore) {
                int64_t e = 0;
                CHECK_OK(velri_json_read_i64(&r, &e), "arr elem");
                sum += e;
            }
            CHECK(sum == 30, "arr sum");
            seen |= 128;
        } else {
            CHECK(false, "unexpected key");
        }
    }
    CHECK(seen == 255, "all fields parsed");
    CHECK_OK(velri_json_finish(&r), "json finish");
    free(exact);

    /* Negative syntax tests */
    const char *bad_cases[] = {
        "{\"a\": 1,}",            /* trailing comma */
        "[1, 2,]",                /* array trailing comma */
        "{\"a\": 01}",            /* leading zero */
        "{\"a\": 1.}",            /* dangling dot */
        "{\"a\": \"\\uD800\"}",   /* lone high surrogate */
        "{\"a\": \"\\uDC00\"}",   /* lone low surrogate */
        "{\"a\": \"raw\nline\"}", /* unescaped control char */
        "{} extra",               /* trailing garbage */
    };
    for (size_t i = 0U; i < sizeof(bad_cases) / sizeof(bad_cases[0]); ++i) {
        velri_json_reader br;
        velri_json_init(&br, bad_cases[i], strlen(bad_cases[i]));
        velri_status st = velri_json_skip(&br);
        if (st == VELRI_OK) {
            st = velri_json_finish(&br);
        }
        CHECK(st != VELRI_OK, "invalid JSON must be rejected");
    }

    /* Depth overflow test (VELRI_JSON_MAX_DEPTH + 2 nested arrays) */
    char deep[VELRI_JSON_MAX_DEPTH * 2 + 16];
    size_t dpos = 0U;
    for (int i = 0; i < VELRI_JSON_MAX_DEPTH + 2; ++i) deep[dpos++] = '[';
    for (int i = 0; i < VELRI_JSON_MAX_DEPTH + 2; ++i) deep[dpos++] = ']';
    velri_json_reader dr;
    velri_json_init(&dr, deep, dpos);
    CHECK(velri_json_skip(&dr) == VELRI_ERR_OVERFLOW, "excessive nesting must return OVERFLOW");

    /* IEEE-754 subnormals, underflow-to-zero, overflow, -0, and invalid raw UTF-8. */
    {
        velri_json_reader nr;
        double dv = -1.0;
        velri_json_init(&nr, "1e-310", 6U);
        CHECK_OK(velri_json_read_f64(&nr, &dv), "subnormal 1e-310 must succeed");
        CHECK(dv > 0.0 && dv < 1e-308, "subnormal 1e-310 positive");

        velri_json_init(&nr, "5e-324", 6U);
        CHECK_OK(velri_json_read_f64(&nr, &dv), "DBL_TRUE_MIN 5e-324 must succeed");
        CHECK(dv > 0.0, "DBL_TRUE_MIN nonzero");

        velri_json_init(&nr, "1e-400", 6U);
        CHECK_OK(velri_json_read_f64(&nr, &dv), "underflow 1e-400 rounds to 0.0");
        CHECK(dv == 0.0, "1e-400 == 0.0");

        velri_json_init(&nr, "1e309", 5U);
        CHECK(velri_json_read_f64(&nr, &dv) == VELRI_ERR_OVERFLOW, "1e309 overflows double");

        int64_t iv = 99;
        uint64_t uv = 99U;
        velri_json_init(&nr, "-0", 2U);
        CHECK_OK(velri_json_read_i64(&nr, &iv), "-0 valid i64");
        CHECK(iv == 0, "-0 == 0");
        velri_json_init(&nr, "-0", 2U);
        CHECK(velri_json_read_u64(&nr, &uv) == VELRI_ERR_INVALID, "-0 rejected for u64");

        /* Invalid raw UTF-8 (0xFF, overlong 0xC0 0x80, raw UTF-8 surrogate ED A0 80). */
        const char bad_utf8_1[] = "\"\xFF\"";
        const char bad_utf8_2[] = "\"\xC0\x80\"";
        const char bad_utf8_3[] = "\"\xED\xA0\x80\"";
        velri_json_init(&nr, bad_utf8_1, sizeof(bad_utf8_1) - 1U);
        CHECK(velri_json_read_string(&nr, nullptr) == VELRI_ERR_INVALID, "0xFF UTF-8 rejected");
        velri_json_init(&nr, bad_utf8_2, sizeof(bad_utf8_2) - 1U);
        CHECK(velri_json_read_string(&nr, nullptr) == VELRI_ERR_INVALID, "overlong UTF-8 rejected");
        velri_json_init(&nr, bad_utf8_3, sizeof(bad_utf8_3) - 1U);
        CHECK(velri_json_read_string(&nr, nullptr) == VELRI_ERR_INVALID, "raw surrogate rejected");

        /* Streaming escaped string eq > 256 bytes (no fixed buffer cap). */
        char long_lit[300];
        char long_json[320];
        memset(long_lit, 'a', 290U);
        long_lit[290] = '\0';
        long_json[0] = '"';
        memcpy(long_json + 1, "\\u0061", 6U);
        memset(long_json + 7, 'a', 289U);
        long_json[296] = '"';
        velri_json_init(&nr, long_json, 297U);
        velri_json_str ls;
        CHECK_OK(velri_json_read_string(&nr, &ls), "read long escaped str");
        CHECK(ls.has_escapes && velri_json_str_eq(ls, long_lit),
              "velri_json_str_eq handles >256-byte escaped strings");
    }

    printf("  [PASS] streaming C23 JSON scanner (escapes, surrogates, subnormals, UTF-8, skip)\n");
}

/* ---- 2. Example Configs & Cascading Placement Tests ---- */

static void test_example_configs_and_cascade(velri_context *ctx, velri_device_id cpu_dev,
                                             bool has_vk, velri_device_id vk_dev) {
    char err[VELRI_CFG_ERR_MAX] = {0};

    /* Parse real examples/example-dense.json and examples/example-moe.json from disk. */
    char dense_path[512], moe_path[512];
    snprintf(dense_path, sizeof(dense_path), "%s/examples/example-dense.json", VELRI_SOURCE_DIR);
    snprintf(moe_path, sizeof(moe_path), "%s/examples/example-moe.json", VELRI_SOURCE_DIR);

    size_t dense_len = 0U, moe_len = 0U;
    char *dense_txt = read_entire_file(dense_path, &dense_len);
    char *moe_txt = read_entire_file(moe_path, &moe_len);

    velri_engine_config dense_cfg, moe_cfg;
    velri_engine_config_init(&dense_cfg);
    velri_engine_config_init(&moe_cfg);
    CHECK_OK(velri_engine_config_parse(dense_txt, dense_len, &dense_cfg, err, sizeof(err)),
             "parse example-dense.json");
    CHECK_OK(velri_engine_config_parse(moe_txt, moe_len, &moe_cfg, err, sizeof(err)),
             "parse example-moe.json");

    CHECK(strcmp(dense_cfg.model_path, "models/qwen3.8-27b") == 0, "dense model path");
    CHECK(dense_cfg.context_size == 262144U, "dense context size");
    CHECK(dense_cfg.cpu_backend.threads == 16U && dense_cfg.cpu_backend.reserve_cores == 1U,
          "dense cpu backend cfg");
    CHECK(dense_cfg.vulkan_backend.prefill_chunk_size == 512U, "dense vk chunk");
    CHECK(dense_cfg.kv_k_bits == 8U && dense_cfg.kv_v_bits == 5U, "dense kv quant");
    CHECK(dense_cfg.attn_mechanism == VELRI_ATTN_MECH_FLASH, "dense attn mech");
    CHECK(fabsf(dense_cfg.sampling.temperature - 0.6f) < 1e-5f && dense_cfg.sampling.top_k == 20U,
          "dense sampling cfg");

    CHECK(strcmp(moe_cfg.model_path, "models/qwen3.8-flash-next") == 0, "moe model path");
    CHECK(moe_cfg.disk_io_threads == 4U, "moe disk io threads");
    CHECK(moe_cfg.vulkan_backend.prefill_chunk_size == 4096U, "moe vk chunk");

    /* Strict vs non-strict layer range validation. */
    CHECK(velri_engine_config_validate_layers(&dense_cfg, 16U, true, err, sizeof(err)) ==
              VELRI_ERR_INVALID,
          "strict validation rejects selector 36-63 for 16-layer model");
    CHECK_OK(velri_engine_config_validate_layers(&dense_cfg, 16U, false, err, sizeof(err)),
             "non-strict allows out-of-range selector");
    CHECK_OK(velri_engine_config_validate_layers(&dense_cfg, 64U, true, err, sizeof(err)),
             "strict passes for 64-layer model");

    if (has_vk) {
        velri_placement_plan plan;
        CHECK_OK(velri_placement_plan_bind(&dense_cfg, ctx, vk_dev, &plan, err, sizeof(err)),
                 "bind dense plan");
        CHECK(velri_placement_has_distinct_prefill(&plan, 64U), "dense has distinct prefill");

        velri_resolved_placement r;
        /* input-layer: store=ram:0 (cpu_dev), compute=vulkan:0, prefill=vulkan:0 (inherited) */
        CHECK_OK(velri_placement_resolve(&plan, "input-layer", -1, &r, err, sizeof(err)),
                 "resolve input-layer");
        CHECK(r.store_dev == cpu_dev && r.compute_dev == vk_dev && r.prefill_dev == vk_dev,
              "input-layer devices");

        /* layer 0 dense-mlp: store=ram:0, compute=cpu:0, prefill=vulkan:0 */
        CHECK_OK(velri_placement_resolve(&plan, "hidden-layers/dense-mlp", 0, &r, err, sizeof(err)),
                 "resolve l0 mlp");
        CHECK(r.store_dev == cpu_dev && r.compute_dev == cpu_dev && r.prefill_dev == vk_dev,
              "l0 dense-mlp split prefill/decode");

        /* layer 36 dense-mlp: selector 36-63 overrides compute=vulkan:0 -> prefill inherits vulkan:0 */
        CHECK_OK(
            velri_placement_resolve(&plan, "hidden-layers/dense-mlp", 36, &r, err, sizeof(err)),
            "resolve l36 mlp");
        CHECK(r.store_dev == vk_dev && r.compute_dev == vk_dev && r.prefill_dev == vk_dev,
              "l36 dense-mlp selector wins");
        CHECK(strstr(r.compute_source, "[36-63]") != nullptr, "provenance records [36-63]");

        /* MoE plan with disk rule: bind succeeds, builder cb rejects disk cleanly. */
        velri_placement_plan moe_plan;
        CHECK_OK(velri_placement_plan_bind(&moe_cfg, ctx, vk_dev, &moe_plan, err, sizeof(err)),
                 "bind moe plan");
        CHECK(moe_plan.has_disk_rules, "moe plan flags disk rules");
        CHECK_OK(velri_placement_resolve(&moe_plan, "ple/table", -1, &r, err, sizeof(err)),
                 "resolve ple/table");
        CHECK(r.store_is_disk, "ple/table resolves to disk");

        velri_graph *dg = nullptr;
        CHECK_OK(velri_graph_create(&dg), "disk builder graph");
        velri_builder db;
        velri_placement_builder_adapter ad = {.plan = &moe_plan, .builder = &db, .is_prefill = false};
        velri_b_init(&db, dg, nullptr, vk_dev, vk_dev, velri_placement_builder_cb, &ad);
        velri_device_id sd = 0U, cd = 0U;
        velri_placement_builder_cb("ple/table", -1, &ad, &sd, &cd);
        CHECK(ad.err == VELRI_ERR_UNSUPPORTED && velri_b_status(&db) == VELRI_ERR_UNSUPPORTED,
              "disk scope fails with UNSUPPORTED and propagates sticky error to builder");
        velri_graph_destroy(dg);

        /* Inspect on example-moe.json must display PLE, indexer, and MoE routed-experts scopes. */
        char moe_inspect[8192];
        size_t mlen = velri_placement_format_inspect(&moe_plan, 48U, nullptr, moe_inspect,
                                                     sizeof(moe_inspect));
        CHECK(mlen > 0U && strstr(moe_inspect, "ple.table") != nullptr &&
                  strstr(moe_inspect, "attention.indexer.keys") != nullptr &&
                  strstr(moe_inspect, "layers[0..31].moe.routed-experts") != nullptr &&
                  strstr(moe_inspect, "layers[32..47].moe.routed-experts") != nullptr,
              "inspect displays PLE, indexer, and MoE selector groups");
    }

    /* Inspect grouping must NOT merge ram:0 vs cpu:0 or differing provenance/compute. */
    {
        const char *split_json =
            "{"
            "  \"model\": {"
            "    \"store\": \"cpu:0\","
            "    \"compute\": \"cpu:0\","
            "    \"hidden-layers\": {"
            "      \"0\": { \"dense-mlp\": { \"store\": \"cpu:0\" } },"
            "      \"1\": { \"dense-mlp\": { \"store\": \"ram:0\" } }"
            "    }"
            "  }"
            "}";
        velri_engine_config scfg;
        velri_engine_config_init(&scfg);
        CHECK_OK(velri_engine_config_parse(split_json, strlen(split_json), &scfg, err, sizeof(err)),
                 "parse ram vs cpu inspect test");
        velri_placement_plan splan;
        CHECK_OK(velri_placement_plan_bind(&scfg, ctx, cpu_dev, &splan, err, sizeof(err)),
                 "bind ram vs cpu");
        char ibuf[4096];
        velri_placement_format_inspect(&splan, 2U, nullptr, ibuf, sizeof(ibuf));
        CHECK(strstr(ibuf, "layers[0..0].dense-mlp") != nullptr &&
                  strstr(ibuf, "layers[1..1].dense-mlp") != nullptr,
              "inspect must not falsely merge ram:0 and cpu:0 layers");
        velri_engine_config_destroy(&scfg);
    }

    /* Optional prefill-compute inheritance: when omitted, prefill always equals compute. */
    {
        const char *no_pre_json =
            "{"
            "  \"model\": {"
            "    \"store\": \"ram:0\","
            "    \"compute\": \"cpu:0\","
            "    \"hidden-layers\": {"
            "      \"0,2-3\": { \"dense-mlp\": { \"store\": \"cpu:0\" } }"
            "    }"
            "  }"
            "}";
        velri_engine_config cfg;
        velri_engine_config_init(&cfg);
        CHECK_OK(velri_engine_config_parse(no_pre_json, strlen(no_pre_json), &cfg, err, sizeof(err)),
                 "parse no-prefill json");
        velri_placement_plan plan;
        CHECK_OK(velri_placement_plan_bind(&cfg, ctx, cpu_dev, &plan, err, sizeof(err)),
                 "bind no-prefill");
        CHECK(!velri_placement_has_distinct_prefill(&plan, 4U),
              "omitted prefill-compute must not create distinct prefill plan");
        velri_resolved_placement r;
        CHECK_OK(velri_placement_resolve(&plan, "hidden-layers/dense-mlp", 2, &r, err, sizeof(err)),
                 "resolve l2");
        CHECK(r.compute_dev == cpu_dev && r.prefill_dev == cpu_dev,
              "prefill inherits from compute");
        velri_engine_config_destroy(&cfg);
    }

    velri_engine_config_destroy(&dense_cfg);
    velri_engine_config_destroy(&moe_cfg);
    free(dense_txt);
    free(moe_txt);
    printf("  [PASS] example-dense.json, example-moe.json & cascading placement resolution\n");
}

/* ---- 3. Negative Validation & Migration Diagnostics ---- */

static void test_placement_validation_errors(velri_context *ctx, velri_device_id cpu_dev) {
    struct {
        const char *json;
        const char *expected_substr;
    } cases[] = {
        {"{\"model\":{\"compute\":\"cuda:0\"}}", "migrate"},
        {"{\"model\":{\"store\":\"vram:0\"}}", "migrate"},
        {"{\"model\":{\"compute\":\"device:0\"}}", "ambiguous"},
        {"{\"model\":{\"compute\":\"ram:0\"}}", "store-only"},
        {"{\"model\":{\"prefill-compute\":\"disk\"}}", "store-only"},
        {"{\"model\":{\"store\":\"ram:1\"}}", "ram:0"},
        {"{\"model\":{\"store\":\"disk:0\"}}", "without a device index"},
        {"{\"model\":{\"compute\":\"cpu:0\\u0000evil\"}}", "NUL"},
        {"{\"model\":{\"path\":\"ok\\u0000evil\"}}", "NUL"},
        {"{\"model\":{\"hidden-layers\":{\"0-3\\u0000x\":{}}}}", "NUL"},
        {"{\"model\":{\"sampling\":{\"temperature\":1e39}}}", "temperature"},
        {"{\"model\":{\"sampling\":{\"top-p\":1e-46}}}", "top-p"},
        {"{\"model\":{},\"kv-quantization\":{\"k-bits\":99}}", "1..16"},
        {"{\"model\":{\"hidden-layers\":{\"0-3\":{},\"3-5\":{}}}}", "overlapping"},
        {"{\"model\":{\"hidden-layers\":{\"1,1-2\":{}}}}", "duplicated"},
        {"{\"model\":{\"hidden-layers\":{\"5-2\":{}}}}", "invalid layer range"},
        {"{\"model\":{\"dense-mlpp\":{}}}", "unknown key"},
        {"{\"model\":{\"store\":\"cpu:0\",\"store\":\"ram:0\"}}", "duplicate"},
        {"{\"unknown-top\":{}}", "unknown top-level"},
        {"{}", "missing required"},
        {"{\"model\":{},\"backends\":{\"cpu\":{\"affinity\":\"bogus\"}}}", "affinity"},
    };
    for (size_t i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        velri_engine_config cfg;
        velri_engine_config_init(&cfg);
        char err[VELRI_CFG_ERR_MAX] = {0};
        velri_status st = velri_engine_config_parse(cases[i].json, strlen(cases[i].json), &cfg, err,
                                                    sizeof(err));
        CHECK(st == VELRI_ERR_INVALID, "invalid config must return VELRI_ERR_INVALID");
        CHECK(strstr(err, cases[i].expected_substr) != nullptr,
              "diagnostic message must explain the error clearly");
        velri_engine_config_destroy(&cfg);
    }

    /* Missing backend / device index out of range at bind time. */
    {
        const char *bad_dev = "{\"model\":{\"compute\":\"nonexistent:0\"}}";
        velri_engine_config cfg;
        velri_engine_config_init(&cfg);
        char err[VELRI_CFG_ERR_MAX] = {0};
        CHECK_OK(velri_engine_config_parse(bad_dev, strlen(bad_dev), &cfg, err, sizeof(err)),
                 "parse unknown backend");
        velri_placement_plan plan;
        CHECK(velri_placement_plan_bind(&cfg, ctx, cpu_dev, &plan, err, sizeof(err)) ==
                  VELRI_ERR_NOT_FOUND,
              "missing backend fails at bind time");
        CHECK(strstr(err, "not available") != nullptr, "missing backend diagnostic");
        velri_engine_config_destroy(&cfg);
    }
    {
        const char *bad_idx = "{\"model\":{\"compute\":\"cpu:99\"}}";
        velri_engine_config cfg;
        velri_engine_config_init(&cfg);
        char err[VELRI_CFG_ERR_MAX] = {0};
        CHECK_OK(velri_engine_config_parse(bad_idx, strlen(bad_idx), &cfg, err, sizeof(err)),
                 "parse bad cpu index");
        velri_placement_plan plan;
        CHECK(velri_placement_plan_bind(&cfg, ctx, cpu_dev, &plan, err, sizeof(err)) ==
                  VELRI_ERR_NOT_FOUND,
              "out-of-range device index fails at bind time");
        CHECK(strstr(err, "cpu:99") != nullptr, "bad index diagnostic");
        velri_engine_config_destroy(&cfg);
    }

    printf("  [PASS] placement schema validation, overlap rejection & migration diagnostics\n");
}

/* ---- 4. End-to-End JSON-Driven Hybrid GPT-2 Execution + Inspect ---- */

static float det_val(uint64_t seed, size_t idx, float scale) {
    uint64_t x = seed ^ ((uint64_t)idx * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    double u = (double)(x >> 11) * (1.0 / 9007199254740992.0);
    return (float)((u * 2.0 - 1.0) * (double)scale);
}

static void run_json_gpt2(velri_context *ctx, const velri_placement_plan *pplan, bool is_prefill,
                          const velri_model_params *meta, const int32_t *seq, size_t seq_len,
                          float *out_logits, velri_memory_report *out_rep) {
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "graph create");
    velri_manifest manifest;
    velri_manifest_init(&manifest);

    velri_builder b;
    velri_placement_builder_adapter ad = {
        .plan = pplan,
        .builder = &b,
        .is_prefill = is_prefill,
        .err = VELRI_OK,
    };
    velri_b_init(&b, g, &manifest, pplan->default_store_dev, pplan->default_compute_dev,
                 velri_placement_builder_cb, &ad);
    CHECK_OK(velri_arch_gpt2_build(&b, meta), "build gpt2 from placement plan");
    CHECK_OK(ad.err, "placement adapter ok");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare gpt2");

    /* Deterministically populate every weight in manifest (honoring cross-device `source_name`
     * for replicated tied weights such as wte on ram:0 + lm_head on vulkan:0). */
    float **w_bufs = (float **)calloc(manifest.num_weights, sizeof(float *));
    CHECK(w_bufs != nullptr, "w_bufs");
    for (uint32_t i = 0U; i < manifest.num_weights; ++i) {
        const velri_manifest_entry *e = &manifest.weights[i];
        size_t n = 1U;
        for (uint32_t d = 0U; d < e->rank; ++d) n *= e->dims[d];
        const float *bind_ptr = nullptr;
        if (e->source_name[0] != '\0') {
            for (uint32_t j = 0U; j < i; ++j) {
                if (strcmp(manifest.weights[j].name, e->source_name) == 0) {
                    bind_ptr = w_bufs[j];
                    break;
                }
            }
            CHECK(bind_ptr != nullptr, "tied weight source_name must resolve to earlier entry");
        } else {
            w_bufs[i] = (float *)malloc(n * sizeof(float));
            CHECK(w_bufs[i] != nullptr, "w_buf");
            uint64_t seed = 0xCAFE0001ULL + (uint64_t)i * 131U;
            for (size_t k = 0U; k < n; ++k) {
                w_bufs[i][k] = det_val(seed, k, 0.04f);
            }
            bind_ptr = w_bufs[i];
        }
        CHECK_OK(velri_plan_bind_weight_host(plan, e->storage_id, bind_ptr, n * sizeof(float)),
                 "bind weight");
    }

    if (out_rep) {
        CHECK_OK(velri_plan_memory_report(plan, out_rep), "memory report");
    }

    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot), "slot create");
    int32_t *toks = (int32_t *)calloc(meta->max_seq_len, sizeof(int32_t));
    CHECK(toks != nullptr, "toks");
    memcpy(toks, seq, seq_len * sizeof(int32_t));
    CHECK_OK(velri_slot_bind_host(slot, manifest.token_ids_in, toks,
                                  meta->max_seq_len * sizeof(int32_t)),
             "bind in");
    CHECK_OK(velri_slot_bind_host(slot, manifest.logits_out, out_logits,
                                  meta->vocab_size * sizeof(float)),
             "bind out");

    velri_step_params sp = {
        .has_pos_offset = true, .pos_offset_override = 0U,
        .has_seq_len = true, .seq_len_override = seq_len,
        .has_kv_len = true, .kv_len_override = 0U,
    };
    CHECK_OK(velri_slot_run(slot, &sp, nullptr), "slot run");
    CHECK(velri_slot_last_run_allocs(slot) == 0U, "0 hot-path allocs");

    free(toks);
    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    for (uint32_t i = 0U; i < manifest.num_weights; ++i) free(w_bufs[i]);
    free(w_bufs);
    velri_manifest_destroy(&manifest);
    velri_graph_destroy(g);
}

static void test_json_driven_gpt2_hybrid(velri_context *ctx, velri_device_id cpu_dev, bool has_vk,
                                         velri_device_id vk_dev) {
    const velri_model_params meta = {
        .vocab_size = 128U,
        .max_position_embeddings = 32U,
        .hidden_size = 64U,
        .num_hidden_layers = 4U,
        .num_attention_heads = 4U,
        .num_key_value_heads = 4U,
        .head_dim = 16U,
        .intermediate_size = 256U,
        .norm_eps = 1e-5f,
        .hidden_act = VELRI_ACT_GELU_TANH,
        .tie_word_embeddings = true,
        .dtype = VELRI_DTYPE_F32,
        .max_seq_len = 8U,
        .max_ctx = 32U,
        .latent_kv = false,
    };
    const int32_t seq[4] = {3, 41, 12, 99};

    /* Baseline pure CPU JSON config. */
    const char *cpu_json = "{\"model\":{\"store\":\"ram:0\",\"compute\":\"cpu:0\"}}";
    velri_engine_config cpu_cfg;
    velri_engine_config_init(&cpu_cfg);
    char err[VELRI_CFG_ERR_MAX] = {0};
    CHECK_OK(velri_engine_config_parse(cpu_json, strlen(cpu_json), &cpu_cfg, err, sizeof(err)),
             "parse cpu json");
    velri_placement_plan cpu_pplan;
    CHECK_OK(velri_placement_plan_bind(&cpu_cfg, ctx, cpu_dev, &cpu_pplan, err, sizeof(err)),
             "bind cpu pplan");

    float *cpu_logits = (float *)calloc(meta.vocab_size, sizeof(float));
    CHECK(cpu_logits != nullptr, "cpu_logits");
    velri_memory_report cpu_rep;
    run_json_gpt2(ctx, &cpu_pplan, false, &meta, seq, 4U, cpu_logits, &cpu_rep);

    char inspect_buf[4096];
    size_t ilen = velri_placement_format_inspect(&cpu_pplan, meta.num_hidden_layers, &cpu_rep,
                                                 inspect_buf, sizeof(inspect_buf));
    CHECK(ilen > 0U && strstr(inspect_buf, "Memory Plan:") != nullptr, "inspect format cpu");

    if (has_vk) {
        /* Hybrid JSON: layers 0-1 on vulkan:0, layers 2-3 dense-mlp split (decode=cpu, prefill=vk),
         * input-layer & kv-cache stored in ram:0. */
        const char *hyb_json =
            "{"
            "  \"model\": {"
            "    \"store\": \"vulkan:0\","
            "    \"compute\": \"vulkan:0\","
            "    \"input-layer\": { \"store\": \"ram:0\" },"
            "    \"hidden-layers\": {"
            "      \"attention\": { \"kv-cache\": { \"store\": \"ram:0\" } },"
            "      \"2-3\": {"
            "        \"dense-mlp\": {"
            "          \"store\": \"ram:0\","
            "          \"compute\": \"cpu:0\","
            "          \"prefill-compute\": \"vulkan:0\""
            "        }"
            "      }"
            "    }"
            "  }"
            "}";
        velri_engine_config hyb_cfg;
        velri_engine_config_init(&hyb_cfg);
        CHECK_OK(velri_engine_config_parse(hyb_json, strlen(hyb_json), &hyb_cfg, err, sizeof(err)),
                 "parse hybrid json");
        CHECK_OK(velri_engine_config_validate_layers(&hyb_cfg, meta.num_hidden_layers, true, err,
                                                     sizeof(err)),
                 "validate hybrid layers");
        velri_placement_plan hyb_pplan;
        CHECK_OK(velri_placement_plan_bind(&hyb_cfg, ctx, vk_dev, &hyb_pplan, err, sizeof(err)),
                 "bind hybrid pplan");
        CHECK(velri_placement_has_distinct_prefill(&hyb_pplan, meta.num_hidden_layers),
              "hybrid has distinct prefill");

        float *dec_logits = (float *)calloc(meta.vocab_size, sizeof(float));
        float *pre_logits = (float *)calloc(meta.vocab_size, sizeof(float));
        CHECK(dec_logits && pre_logits, "hyb logits");
        velri_memory_report dec_rep, pre_rep;
        run_json_gpt2(ctx, &hyb_pplan, false, &meta, seq, 4U, dec_logits, &dec_rep);
        run_json_gpt2(ctx, &hyb_pplan, true, &meta, seq, 4U, pre_logits, &pre_rep);

        CHECK(dec_rep.ram_staging_bytes == 0U && pre_rep.ram_staging_bytes == 0U,
              "0 B ram_staging on CPU<->Vulkan hybrid");
        CHECK(dec_rep.devices[vk_dev].state_bytes == 0U &&
                  dec_rep.devices[cpu_dev].state_bytes > 0U,
              "kv-cache stored on ram:0 (cpu_dev)");

        double max_dec = 0.0, max_pre = 0.0;
        for (uint32_t i = 0U; i < meta.vocab_size; ++i) {
            double dd = fabs((double)dec_logits[i] - (double)cpu_logits[i]);
            double dp = fabs((double)pre_logits[i] - (double)cpu_logits[i]);
            if (dd > max_dec) max_dec = dd;
            if (dp > max_pre) max_pre = dp;
        }
        CHECK(max_dec < 1e-4 && max_pre < 1e-4, "hybrid JSON GPT-2 logits match pure CPU");

        ilen = velri_placement_format_inspect(&hyb_pplan, meta.num_hidden_layers, &dec_rep,
                                              inspect_buf, sizeof(inspect_buf));
        CHECK(ilen > 0U && strstr(inspect_buf, "layers[2..3].dense-mlp") != nullptr,
              "inspect groups layers[2..3]");
        printf("  [PASS] JSON-driven hybrid GPT-2 (decode err=%.2e | prefill err=%.2e | "
               "ram_staging=0 B)\n",
               max_dec, max_pre);

        free(dec_logits);
        free(pre_logits);
        velri_engine_config_destroy(&hyb_cfg);
    } else {
        printf("  [PASS] JSON-driven CPU GPT-2 (vulkan skipped in this build)\n");
    }

    free(cpu_logits);
    velri_engine_config_destroy(&cpu_cfg);
}

int main(void) {
    printf("=== Velri Config, Streaming JSON & Cascading Placement Suite ===\n");
    test_json_scanner();

    velri_context_config ccfg = {
        .cpu_workers = 4U,
        .cpu_variant = VELRI_CPU_AUTO,
        .enable_vulkan = true,
    };
    velri_context *ctx = nullptr;
    CHECK_OK(velri_context_create(&ccfg, &ctx), "ctx create");

    velri_device_id cpu_dev = 0U, vk_dev = 0U;
    CHECK_OK(velri_context_find_device(ctx, "cpu", 0U, &cpu_dev), "find cpu");
    bool has_vk = (velri_context_find_device(ctx, "vulkan", 0U, &vk_dev) == VELRI_OK);

    test_example_configs_and_cascade(ctx, cpu_dev, has_vk, vk_dev);
    test_placement_validation_errors(ctx, cpu_dev);
    test_json_driven_gpt2_hybrid(ctx, cpu_dev, has_vk, vk_dev);

    velri_context_destroy(ctx);
    printf("=== Config & Placement Suite PASSED ===\n");
    return 0;
}
