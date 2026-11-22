#define main velri_unused_runtime_tests
#include "test_engine_runtime.c"
#undef main
#include "runtime/runtime_internal.h"

int main(int argc, char **argv) {
    CHECK(argc == 3, "expected raw and index output paths");
    velri_ctx *ctx = nullptr;
    velri_ctx_options options = {.cpu_threads = 1U, .enable_vulkan = false};
    CHECK_RC(velri_ctx_create(&options, &ctx), "create CPU context for weight binding");
    const velri_model_desc d = {
        .arch = VELRI_ARCH_LLAMA,
        .vocab_size = 8192,
        .max_position_embeddings = 9216,
        .hidden_size = 768,
        .num_hidden_layers = 8,
        .num_attention_heads = 12,
        .num_key_value_heads = 4,
        .head_dim = 64,
        .intermediate_size = 2048,
        .norm_eps = 1e-6f,
        .hidden_act = VELRI_ACTIVATION_SILU,
        .weight_type = VELRI_TYPE_F32,
        .rope_theta = 10000.0f,
        .rotary_dim = 64,
        .rope_mode = VELRI_ROTARY_HALF,
    };
    weight_pool pool = {0};
    velri_model *model = nullptr;
    CHECK_RC(velri_model_create_synthetic(ctx, &d, &model), "create benchmark model");
    populate_llama_weights(model, &d, &pool);
    FILE *raw = fopen(argv[1], "wb"), *index = fopen(argv[2], "w");
    CHECK(raw && index, "open output");
    for (uint32_t i = 0; i < model->cap_weights; ++i) {
        const velri_rt_weight_entry *w = &model->weights[i];
        if (!w->name[0]) {
            continue;
        }
        CHECK(fprintf(index, "%s\t%zu\n", w->name, w->byte_size) > 0, "tensor index");
        CHECK(fwrite(w->host_ptr, 1, w->byte_size, raw) == w->byte_size, "tensor bytes");
    }
    CHECK(!fclose(raw) && !fclose(index), "close outputs");
    CHECK_RC(velri_model_destroy(model), "destroy model");
    pool_destroy(&pool);
    CHECK_RC(velri_ctx_destroy(ctx), "destroy context");
    return 0;
}
