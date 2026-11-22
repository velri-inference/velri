#define main original_runtime_tests
#include "test_engine_runtime.c"
#undef main
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "gguf_index.h"

static void *bind_gguf(velri_model *model) {
    const char *path = getenv("BENCH_GGUF");
    if (!path) path = BENCH_DEFAULT_GGUF;
    int fd = open(path, O_RDONLY);
    struct stat statbuf;
    CHECK(fd >= 0 && !fstat(fd, &statbuf) && statbuf.st_size == BENCH_GGUF_BYTES, "open GGUF");
    char *mapping = mmap(NULL, BENCH_GGUF_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(mapping != MAP_FAILED && !close(fd), "map weights");
    const unsigned count = sizeof(gguf_weights) / sizeof(gguf_weights[0]);
    for (unsigned i = 0; i < count; ++i)
        CHECK_RC(velri_model_bind_weight(model, gguf_weights[i].name, mapping + gguf_weights[i].offset,
                                         gguf_weights[i].bytes), gguf_weights[i].name);
    CHECK(count == velri_model_weight_count(model), "bind entire GGUF to Velri");
    return mapping;
}


enum {VOCAB = 8192, MODEL_CONTEXT = 9216, CHUNK = 1024};

static unsigned parse_number(const char *word, unsigned lo, unsigned hi) {
    char *end;
    errno = 0;
    unsigned long n = strtoul(word, &end, 10);
    CHECK(!errno && *word && !*end && n >= lo && n <= hi, "bad numeric argument");
    return (unsigned)n;
}

static uint64_t now_ns(void) {
    struct timespec t;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "clock_gettime");
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

typedef struct {
    double us;
    uint64_t replay;
    int32_t token;
} observation;

int main(int argc, char **argv) {
    CHECK(argc >= 2 && argc <= 7, "usage: session_bench cpu|vulkan|hybrid [reps=4] [workers=8] [warmups=2] [decode=32] [prompt=128]");
    bool cpu = !strcmp(argv[1], "cpu"), hybrid = !strcmp(argv[1], "hybrid");
    CHECK(cpu || hybrid || !strcmp(argv[1], "vulkan"), "unknown placement");
    const unsigned CONTEXT = getenv("BENCH_CONTEXT") ? parse_number(getenv("BENCH_CONTEXT"), 1024, 1000000) : MODEL_CONTEXT;
    unsigned reps = argc > 2 ? parse_number(argv[2], 1, 10000) : 4;
    unsigned workers = argc > 3 ? parse_number(argv[3], 1, 256) : 8;
    unsigned warmups = argc > 4 ? parse_number(argv[4], 0, 1000) : 2;
    unsigned decode_count = argc > 5 ? parse_number(argv[5], 0, CONTEXT - 1) : 32;
    unsigned prompt_count = argc > 6 ? parse_number(argv[6], 2, CONTEXT - 1) : 128;
    CHECK(prompt_count + decode_count <= CONTEXT, "prompt plus decode exceeds context");
    unsigned prime = getenv("BENCH_HISTORY") ? parse_number(getenv("BENCH_HISTORY"), 0, CONTEXT-1) : 0;
    CHECK(prime + prompt_count + decode_count <= CONTEXT, "history exceeds context");
    uint64_t a = now_ns();
    velri_ctx *ctx = nullptr;
    velri_ctx_options co = {.cpu_threads = workers, .enable_vulkan = !cpu};
    CHECK_RC(velri_ctx_create(&co, &ctx), "context create");
    uint64_t b = now_ns();

    const velri_model_desc d = {
        .arch=VELRI_ARCH_LLAMA, .vocab_size=VOCAB, .max_position_embeddings=MODEL_CONTEXT,
        .hidden_size=768, .num_hidden_layers=8, .num_attention_heads=12, .num_key_value_heads=4,
        .head_dim=64, .intermediate_size=2048, .norm_eps=1e-6f, .hidden_act=VELRI_ACTIVATION_SILU,
        .weight_type=VELRI_TYPE_F32, .rope_theta=10000.0f, .rotary_dim=64, .rope_mode=VELRI_ROTARY_INTERLEAVED
    };
    velri_model *model = nullptr;

    CHECK_RC(velri_model_create_synthetic(ctx, &d, &model), "model create");
    void *mapping = bind_gguf(model);
    uint64_t c = now_ns();
    CHECK(cpu && !hybrid, "matched CPU-only placement");
    const char *json = "{\"model\":{\"store\":\"cpu:0\",\"compute\":\"cpu:0\"}}";
    velri_program_options po = {.max_ctx=CONTEXT, .prefill_chunk_size=CHUNK, .placement_json=json,
                                .placement_json_len=strlen(json), .strict_layer_selectors=true};
    velri_program *prog = nullptr;
    velri_rc rc = velri_program_prepare(ctx, model, &po, &prog);
    if (rc != VELRI_RC_OK) fprintf(stderr, "program error: %s\n", velri_ctx_last_error(ctx));
    CHECK_RC(rc, "program prepare");
    CHECK(!madvise(mapping, BENCH_GGUF_BYTES, MADV_DONTNEED), "release clean host weight pages");
    fprintf(stderr, "gguf_bytes=%u tensors=%u mmap=1 madv_dontneed=1 rope=interleaved kv_bits=%d "
                    "prefill_matmul_bits=%d prefill_attention_bits=%d cpu_weights=1\n",
            BENCH_GGUF_BYTES, velri_model_weight_count(model), 32, 32, 32);
    uint64_t e = now_ns();
    velri_session *session = nullptr;
    CHECK_RC(velri_session_create(prog, nullptr, &session), "session create");
    uint64_t f = now_ns();
    fprintf(stderr, "setup_us ctx=%.3f gguf_bind=%.3f program=%.3f session=%.3f\n",
            (b-a)/1e3, (c-b)/1e3,(e-c)/1e3,(f-e)/1e3);
    CHECK(velri_program_has_distinct_prefill(prog) == hybrid, "unexpected phase placement");
    velri_program_mem_report report;
    CHECK_RC(velri_program_memory_report(prog, &report), "memory report");
    size_t cpu_weights = 0;
    for (unsigned i = 0; i < report.num_devices; ++i) {
        const velri_program_device_mem *bytes = &report.devices[i];
        velri_device_info info;
        CHECK_RC(velri_ctx_device_info(ctx, bytes->device_id, &info), "device info");
        fprintf(stderr, "placement device=%s backend=%s weight_bytes=%zu state_bytes=%zu\n",
                info.device_name, info.backend_name, bytes->weight_bytes, bytes->state_bytes);
        if (bytes->weight_bytes) {
            CHECK(!strcmp(info.backend_name, "cpu"), "weights physically on CPU");
            cpu_weights += bytes->weight_bytes;
        }
    }
    CHECK(cpu_weights >= 251710464ULL && report.max_ctx == CONTEXT && report.prefill_chunk_size == CHUNK &&
          !report.num_ram_transfers_decode && !report.num_ram_transfers_prefill, "full CPU plan");
    fprintf(stderr, "mode=%s workers=%u warmups=%u reps=%u prompt=%u chunk=%d decode=%u plan_bytes=%zu"
            " xfers_decode=%u xfers_prefill=%u\n",argv[1],workers,warmups,reps,prompt_count,
            CHUNK,decode_count,report.total_physical_bytes,report.num_ram_transfers_decode,
            report.num_ram_transfers_prefill);
    int32_t prompt[CONTEXT];
    for(unsigned i=0;i<prime+prompt_count;i++) prompt[i]=(int32_t)((37U*i+11U)%VOCAB);
    unsigned nsteps=decode_count+1;
    observation *records=calloc((size_t)reps*nsteps,sizeof(*records));
    CHECK(records,"allocate observations");
    const char *gold_path=getenv("VELRI_SPEED_LOGITS");
    float *gold=gold_path?malloc((size_t)nsteps*VOCAB*sizeof(*gold)):nullptr;
    CHECK(!gold_path || gold,"allocate logits snapshot");
    const velri_sampling_params sampling={.temperature=0.0f};
    const char *teacher_path=getenv("VELRI_FORCE_TOKENS");
    int32_t *teacher=nullptr;
    if (teacher_path) {
        teacher=calloc(decode_count,sizeof(*teacher));
        FILE *inputs=fopen(teacher_path,"r");
        CHECK(teacher && inputs,"open teacher inputs");
        for (unsigned j=0;j<decode_count;j++) {
            CHECK(fscanf(inputs,"%" SCNd32,&teacher[j])==1 && teacher[j]>=0 && teacher[j]<VOCAB,
                  "invalid teacher token");
        }
        CHECK(!fclose(inputs),"close teacher inputs");
    }
    const unsigned requested_prompt = prompt_count;
    const unsigned warm_prompt = getenv("BENCH_WARM_PROMPT") ? parse_number(getenv("BENCH_WARM_PROMPT"), 2, prompt_count) : prompt_count;
    fprintf(stderr, "capacity=%u warm_prompt=%u\n", CONTEXT, warm_prompt);
    for(unsigned i=0;i<warmups+reps;i++) {
        prompt_count = i < warmups ? warm_prompt : requested_prompt;
        if(i) CHECK_RC(velri_session_reset(session),"session reset");
        if (prime) {
            velri_step_result history;
            CHECK_RC(velri_session_step(session, prompt, prime, nullptr, nullptr, &history), "prime KV");
            CHECK(history.kv_len == prime, "primed KV length");
        }
        int32_t next=0;
        for(unsigned j=0;j<nsteps;j++) {
            velri_step_result boundaries[(CONTEXT + CHUNK - 1U) / CHUNK];
            unsigned count = j ? 1U : prompt_count;
            const int32_t *tokens = j ? &next : prompt + prime;
            unsigned n = 0;
            uint64_t begin = now_ns();
            for (unsigned pos = 0; pos < count; pos += CHUNK) {
                unsigned batch = count - pos < CHUNK ? count - pos : CHUNK;
                const velri_sampling_params *sample = pos + batch == count ? &sampling : nullptr;
                CHECK_RC(velri_session_step(session, tokens + pos, batch, sample, nullptr, &boundaries[n]), "session step");
                ++n;
            }
            uint64_t end = now_ns();
            unsigned consumed = j ? prime + prompt_count + j - 1U : prime;
            for (unsigned i = 0; i < n; ++i) {
                unsigned expected = count - i * CHUNK < CHUNK ? count - i * CHUNK : CHUNK;
                consumed += expected;
                CHECK(boundaries[i].tokens_processed == expected && boundaries[i].kv_len == consumed &&
                      boundaries[i].vocab_size == VOCAB && boundaries[i].logits &&
                      !boundaries[i].last_step_allocs, "public step boundary");
                if (i + 1U < n) CHECK(boundaries[i].sampled_token == -1, "no intermediate greedy");
            }
            velri_step_result result = boundaries[n - 1U];
            CHECK(result.sampled_token >= 0 && result.sampled_token < VOCAB, "greedy");
            next=result.sampled_token;
            if(i>=warmups) {
                if(i>warmups) CHECK(records[j].token==next,"greedy trajectory changed across repetitions");
                records[(size_t)(i-warmups)*nsteps+j]=(observation){(end-begin)/1e3,
                    result.last_step_graph_replays, result.sampled_token};
                if(gold && i+1==warmups+reps) memcpy(gold+(size_t)j*VOCAB,result.logits,VOCAB*sizeof(*gold));
            }
            if (teacher && j<decode_count) next=teacher[j];
        }
    }
    puts("phase,rep,step,us,replays,token");
    for(unsigned i=0;i<reps;i++) {
        for(unsigned j=0;j<nsteps;j++) {
            observation *o=&records[(size_t)i*nsteps+j];
            printf("%s,%u,%u,%.3f,%" PRIu64 ",%d\n",j?"decode":"prefill",i,j,o->us,o->replay,o->token);
        }
    }
    if(gold) {
        FILE *fp=fopen(gold_path,"wb");
        CHECK(fp && fwrite(gold,sizeof(*gold),(size_t)nsteps*VOCAB,fp)==(size_t)nsteps*VOCAB && !fclose(fp),"save logits");
        free(gold);
    }
    free(records);
    free(teacher);
    CHECK_RC(velri_session_destroy(session),"session destroy");
    CHECK_RC(velri_program_destroy(prog),"program destroy");
    CHECK_RC(velri_model_destroy(model),"model destroy");
    CHECK(!munmap(mapping, BENCH_GGUF_BYTES), "unmap GGUF");
    CHECK_RC(velri_ctx_destroy(ctx),"context destroy");
    return 0;
}
