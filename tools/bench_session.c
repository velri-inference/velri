#include "os.h"
#include "runtime_fixture.h"

#include <errno.h>
#include <inttypes.h>

enum { PROMPT = 128, DECODE = 64 };

static unsigned parse_number(const char *word, unsigned minimum, unsigned maximum) {
    char *end;
    errno = 0;
    unsigned long value = strtoul(word, &end, 10);
    CHECK(!errno && *word && !*end && value >= minimum && value <= maximum, "invalid numeric option");
    return (unsigned)value;
}

/* prepare_session_weights.py emits one text descriptor followed by length-prefixed
 * little-endian payloads. The public binding API borrows each allocation. */
static velri_model *load_weights(velri_ctx *ctx, const char *path, velri_model_desc *d, weight_pool *pool) {
    FILE *file = fopen(path, "rb");
    char line[512], precision[8], flag[8] = "", extra;
    uint32_t count;
    CHECK(file && fgets(line, sizeof(line), file), "open checkpoint fixture");
    const uint32_t one = 1;
    CHECK(*(const unsigned char *)&one == 1, "checkpoint fixture expects a little-endian host");
    int fields = sscanf(line, "VELRI-BENCH-1 %7s %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32
                 " %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32 " %f %f %" SCNu32 " %7s %c",
                 precision, &d->vocab_size, &d->max_position_embeddings, &d->hidden_size,
                 &d->num_hidden_layers, &d->num_attention_heads, &d->num_key_value_heads, &d->head_dim,
                 &d->intermediate_size, &d->norm_eps, &d->rope_theta, &count, flag, &extra);
    CHECK(fields == 12 || (fields == 13 && !strcmp(flag, "tied")), "bad fixture descriptor or flags");
    d->tie_word_embeddings = fields == 13;
    CHECK(!strcmp(precision, "w8a8") || !strcmp(precision, "f32"), "unknown fixture precision");
    d->weight_type = !strcmp(precision, "w8a8") ? VELRI_TYPE_W8A8 : VELRI_TYPE_F32;
    d->rotary_dim = d->head_dim;
    velri_model *model;
    CHECK_RC(velri_model_create_synthetic(ctx, d, &model), "checkpoint descriptor");
    pool->ptrs = calloc(count, sizeof(*pool->ptrs));
    CHECK(pool->ptrs, "weight array");
    for (uint32_t i = 0; i < count; ++i) {
        char name[192];
        size_t size;
        CHECK(fgets(line, sizeof(line), file) && sscanf(line, "%191s %zu", name, &size) == 2 && size,
              "invalid weight header");
        void *data = malloc(size);
        CHECK(data && fread(data, 1, size, file) == size, "incomplete weight payload");
        pool->ptrs[pool->count++] = data;
        CHECK_RC(velri_model_bind_weight(model, name, data, size), name);
    }
    CHECK(fgetc(file) == EOF && !ferror(file) && !fclose(file), "bad fixture end");
    CHECK(velri_model_weight_count(model) == count, "duplicate weight name");
    return model;
}

static void save_logits(FILE *file, const velri_step_result *result) {
    if (file) CHECK(fwrite(result->logits, sizeof(float), result->vocab_size, file) == result->vocab_size,
                    "save logits");
}

int main(int argc, char **argv) {
    if (argc < 2 || (strcmp(argv[1], "cpu") && strcmp(argv[1], "vulkan"))) {
        fprintf(stderr, "usage: %s cpu|vulkan [cpu-workers] [--weights FILE] [--isa auto|base|avx2|avx512]"
                        " [--warmups N] [--runs N] [--logits FILE]\n", argv[0]);
        return 2;
    }
    bool gpu = strcmp(argv[1], "vulkan") == 0;
    unsigned workers = 0, warmups = 2, runs = 3;
    int i = 2;
    if (i < argc && argv[i][0] != '-') workers = parse_number(argv[i++], 1, 256);
    const char *weights = nullptr, *logits_path = nullptr, *isa = "auto";
    for (; i < argc; i += 2) {
        CHECK(i + 1 < argc, "missing option value");
        const char *key = argv[i], *value = argv[i + 1];
        if (!strcmp(key, "--weights")) weights = value;
        else if (!strcmp(key, "--logits")) logits_path = value;
        else if (!strcmp(key, "--isa")) isa = value;
        else if (!strcmp(key, "--runs")) runs = parse_number(value, 1, 10000);
        else if (!strcmp(key, "--warmups")) warmups = parse_number(value, 0, 10000);
        else CHECK(false, "unknown benchmark option");
    }
    CHECK(!weights || !logits_path || strcmp(weights, logits_path), "weights and logits need distinct paths");
    CHECK(!strcmp(isa, "auto") || !strcmp(isa, "base") || !strcmp(isa, "avx2") || !strcmp(isa, "avx512"),
          "unknown CPU ISA");
    velri_ctx *ctx;
    char config[128], error[256];
    snprintf(config, sizeof(config), "{\"backends\":{\"cpu\":{\"variant\":\"%s\"}}}", isa);
    velri_ctx_options options = {.cpu_threads = workers, .enable_vulkan = gpu};
    velri_rc rc = velri_ctx_create_from_json(&options, config, strlen(config), &ctx, error, sizeof(error));
    CHECK_RC(rc, error);
    velri_model_desc desc = {
        .arch = VELRI_ARCH_LLAMA, .vocab_size = 8192U, .max_position_embeddings = 9216U,
        .hidden_size = 768U, .num_hidden_layers = 8U, .num_attention_heads = 12U,
        .num_key_value_heads = 4U, .head_dim = 64U, .intermediate_size = 2048U,
        .norm_eps = 1e-6f, .hidden_act = VELRI_ACTIVATION_SILU, .weight_type = VELRI_TYPE_F32,
        .rope_theta = 10000.0f, .rotary_dim = 64U, .rope_mode = VELRI_ROTARY_HALF,
    };
    velri_model *model;
    weight_pool pool = {0};
    if (weights) model = load_weights(ctx, weights, &desc, &pool);
    else {
        CHECK_RC(velri_model_create_synthetic(ctx, &desc, &model), "synthetic Llama");
        populate_llama_weights(model, &desc, &pool);
    }
    CHECK(desc.max_position_embeddings >= PROMPT + DECODE, "checkpoint context too short");
    const char *placement = gpu ? "{\"model\":{\"store\":\"vulkan:0\",\"compute\":\"vulkan:0\"}}"
                                : "{\"model\":{\"store\":\"ram:0\",\"compute\":\"cpu:0\"}}";
    velri_program_options opts = {.max_ctx = weights ? PROMPT + DECODE : desc.max_position_embeddings,
                                  .prefill_chunk_size = weights ? PROMPT : 512U,
                                  .placement_json = placement, .placement_json_len = strlen(placement)};
    velri_program *program;
    rc = velri_program_prepare(ctx, model, &opts, &program);
    CHECK_RC(rc, velri_ctx_last_error(ctx));
    velri_session *session;
    CHECK_RC(velri_session_create(program, nullptr, &session), "create session");

    int32_t prompt[PROMPT], reference[DECODE + 1];
    for (uint32_t j = 0; j < PROMPT; ++j) prompt[j] = (int32_t)((37U * j + 11U) % desc.vocab_size);
    const velri_sampling_params greedy = {.temperature = 0.0f};
    FILE *oracle = logits_path ? fopen(logits_path, "wb") : nullptr;
    CHECK(!logits_path || oracle, "open logits snapshot");
    if (oracle && warmups == 0) warmups = 1;
    double total_ms = 0.0;
    puts("run,tokens,total_ms,ms_per_token,tokens_per_s");
    for (unsigned run = 0; run < warmups + runs; ++run) {
        if (run) CHECK_RC(velri_session_reset(session), "reset");
        velri_step_result result;
        CHECK_RC(velri_session_step(session, prompt, PROMPT, &greedy, nullptr, &result), "prefill");
        save_logits(oracle, &result);
        int32_t trajectory[DECODE + 1] = {result.sampled_token};
        uint64_t start = velri_os_monotonic_ns();
        for (unsigned j = 0; j < DECODE; ++j) {
            CHECK_RC(velri_session_step(session, &trajectory[j], 1U, &greedy, nullptr, &result), "decode");
            trajectory[j + 1] = result.sampled_token;
            save_logits(oracle, &result);
        }
        double ms = (double)(velri_os_monotonic_ns() - start) * 1e-6;
        if (oracle) {
            CHECK(!fclose(oracle), "close logits snapshot");
            oracle = nullptr;
        }
        CHECK(result.kv_len == PROMPT + DECODE, "unexpected final KV length");
        if (!run) memcpy(reference, trajectory, sizeof(reference));
        else CHECK(!memcmp(reference, trajectory, sizeof(reference)), "greedy token sequence changed");
        if (run >= warmups) {
            printf("%u,%d,%.4f,%.6f,%.3f\n", run - warmups + 1U, DECODE, ms, ms / DECODE,
                   DECODE * 1000.0 / ms);
            total_ms += ms;
        }
    }
    printf("all,%u,%.4f,%.6f,%.3f\n", runs * DECODE, total_ms, total_ms / (runs * DECODE),
           runs * DECODE * 1000.0 / total_ms);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned j = 0; j <= DECODE; ++j)
        for (unsigned shift = 0; shift < 32; shift += 8)
            hash = (hash ^ (((uint32_t)reference[j] >> shift) & 255U)) * UINT64_C(1099511628211);
    fprintf(stderr, "trajectory_fnv1a64=%016" PRIx64 "\n", hash);
    CHECK_RC(velri_session_destroy(session), "destroy session");
    CHECK_RC(velri_program_destroy(program), "destroy program");
    CHECK_RC(velri_model_destroy(model), "destroy model");
    pool_destroy(&pool);
    CHECK_RC(velri_ctx_destroy(ctx), "destroy context");
    return 0;
}
