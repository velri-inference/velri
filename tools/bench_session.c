#include "os.h"
#include "runtime_fixture.h"

enum { PROMPT = 128, DECODE = 64, WARMUPS = 2, RUNS = 3 };

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (strcmp(argv[1], "cpu") && strcmp(argv[1], "vulkan"))) {
        fprintf(stderr, "usage: %s cpu|vulkan [cpu-workers]\n", argv[0]);
        return 2;
    }
    bool gpu = strcmp(argv[1], "vulkan") == 0;
    long workers = 0;
    if (argc == 3) {
        char *end;
        workers = strtol(argv[2], &end, 10);
        CHECK(*argv[2] && !*end && workers >= 1 && workers <= 256, "invalid worker count");
    }
    velri_ctx *ctx;
    velri_model *model;
    velri_program *program;
    velri_session *session;
    velri_ctx_options options = {.cpu_threads = (uint32_t)workers, .enable_vulkan = gpu};
    CHECK_RC(velri_ctx_create(&options, &ctx), "context");
    const velri_model_desc desc = {
        .arch = VELRI_ARCH_LLAMA, .vocab_size = 8192U, .max_position_embeddings = 9216U,
        .hidden_size = 768U, .num_hidden_layers = 8U, .num_attention_heads = 12U,
        .num_key_value_heads = 4U, .head_dim = 64U, .intermediate_size = 2048U,
        .norm_eps = 1e-6f, .hidden_act = VELRI_ACTIVATION_SILU, .weight_type = VELRI_TYPE_F32,
        .rope_theta = 10000.0f, .rotary_dim = 64U, .rope_mode = VELRI_ROTARY_HALF,
    };
    CHECK_RC(velri_model_create_synthetic(ctx, &desc, &model), "synthetic Llama");
    weight_pool pool = {0};
    populate_llama_weights(model, &desc, &pool);
    const char *placement = gpu ? "{\"model\":{\"store\":\"vulkan:0\",\"compute\":\"vulkan:0\"}}"
                                : "{\"model\":{\"store\":\"ram:0\",\"compute\":\"cpu:0\"}}";
    velri_program_options opts = {.max_ctx = desc.max_position_embeddings, .prefill_chunk_size = 512U,
                                  .placement_json = placement, .placement_json_len = strlen(placement)};
    CHECK_RC(velri_program_prepare(ctx, model, &opts, &program), "prepare");
    CHECK_RC(velri_session_create(program, nullptr, &session), "create session");

    int32_t prompt[PROMPT], reference[DECODE];
    for (uint32_t j = 0; j < PROMPT; ++j) prompt[j] = (int32_t)((37U * j + 11U) % desc.vocab_size);
    const velri_sampling_params greedy = {.temperature = 0.0f};
    double total_ms = 0.0;
    puts("run,tokens,total_ms,ms_per_token,tokens_per_s");
    for (unsigned run = 0; run < WARMUPS + RUNS; ++run) {
        if (run) CHECK_RC(velri_session_reset(session), "reset");
        velri_step_result result;
        CHECK_RC(velri_session_step(session, prompt, PROMPT, &greedy, nullptr, &result), "prefill");
        int32_t trajectory[DECODE], next = result.sampled_token;
        uint64_t start = velri_os_monotonic_ns();
        for (unsigned j = 0; j < DECODE; ++j) {
            CHECK_RC(velri_session_step(session, &next, 1U, &greedy, nullptr, &result), "decode");
            trajectory[j] = next = result.sampled_token;
        }
        double ms = (double)(velri_os_monotonic_ns() - start) * 1e-6;
        CHECK(result.kv_len == PROMPT + DECODE, "unexpected final KV length");
        if (!run) memcpy(reference, trajectory, sizeof(reference));
        else CHECK(!memcmp(reference, trajectory, sizeof(reference)), "greedy token sequence changed");
        if (run >= WARMUPS) {
            printf("%u,%d,%.4f,%.6f,%.3f\n", run - WARMUPS + 1U, DECODE, ms, ms / DECODE,
                   DECODE * 1000.0 / ms);
            total_ms += ms;
        }
    }
    printf("all,%d,%.4f,%.6f,%.3f\n", RUNS * DECODE, total_ms, total_ms / (RUNS * DECODE),
           RUNS * DECODE * 1000.0 / total_ms);

    CHECK_RC(velri_session_destroy(session), "destroy session");
    CHECK_RC(velri_program_destroy(program), "destroy program");
    CHECK_RC(velri_model_destroy(model), "destroy model");
    pool_destroy(&pool);
    CHECK_RC(velri_ctx_destroy(ctx), "destroy context");
    return 0;
}
