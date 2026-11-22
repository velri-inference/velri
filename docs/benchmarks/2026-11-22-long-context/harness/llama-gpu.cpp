#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

constexpr int vocab = 8192;
static void check(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "matched bench: %s\n", why); std::exit(1); }
}

static int number(const char * text, int minimum, int maximum) {
    char * end;
    errno = 0;
    long n = std::strtol(text, &end, 10);
    check(!errno && *text && !*end && n >= minimum && n <= maximum, "invalid numeric argument");
    return int(n);
}
static const char * setting(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return value ? value : fallback;
}

static int greedy(const float * x) {
    float best = -std::numeric_limits<float>::infinity();
#if defined(__AVX2__)
    __m256 highest = _mm256_set1_ps(best);
    for (int i = 0; i < vocab; i += 8) {
        highest = _mm256_max_ps(_mm256_loadu_ps(x + i), highest);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, highest);
    for (float lane : lanes) best = std::max(best, lane);
    __m256 target = _mm256_set1_ps(best);
    for (int i = 0; i < vocab; i += 8) {
        auto equal = _mm256_cmp_ps(_mm256_loadu_ps(x+i), target, _CMP_EQ_OQ);
        unsigned mask = unsigned(_mm256_movemask_ps(equal));
        if (mask) return i + __builtin_ctz(mask);
    }
#else
    for (int i = 0; i < vocab; ++i) best = std::max(best, x[i]);
    for (int i = 0; i < vocab; ++i) if (x[i] == best) return i;
#endif
    check(false, "no valid output token");
    return -1;
}

static bool trace(ggml_tensor * t, bool ask, void * data) {
    if (ask && t->buffer && t->op != GGML_OP_NONE) {
        auto & seen = *static_cast<std::set<std::string> *>(data);
        std::string s = std::string(ggml_backend_buffer_name(t->buffer)) + " " + ggml_op_name(t->op) + " ";
        for (int i = 0; i < 3 && t->src[i]; ++i) s += std::string(ggml_type_name(t->src[i]->type)) + " ";
        if (t->op == GGML_OP_GET_ROWS && t->src[0] && t->src[0]->buffer) {
            s += std::string(" source=") + t->src[0]->name + " buffer=" + ggml_backend_buffer_name(t->src[0]->buffer);
        }
        if (seen.insert(s).second) std::fprintf(stderr, "op-buffer %s\n", s.c_str());
    }
    return false;
}

static const float * process(llama_context * ctx, llama_batch_ext * batch,
                             const llama_token * tokens, int count, int start) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        int slot = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos pos = start + i;
        check(slot >= 0 && llama_batch_ext_set_pos(batch, slot, &pos) &&
              llama_batch_ext_set_output_logits(batch, slot, i == count-1), "batch setup failed");
    }
    check(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "llama_process failed");
    const float * ptr = llama_get_logits_ith(ctx, -1); // public accessor synchronizes and materializes all 8192 entries
    check(ptr != nullptr, "missing host logits");
    return ptr;
}

struct sample { double us; int token; };
int main(int argc, char ** argv) {
    check(argc >= 2 && argc <= 9, "llama_bench model.gguf [reps workers warmups decode prompt chunk context]");
    const int reps = argc>2 ? number(argv[2],1,10000) : 4;
    const int workers = argc>3 ? number(argv[3],1,256) : 8;
    const int warms = argc>4 ? number(argv[4],0,1000) : 2;
    const int dec = argc>5 ? number(argv[5],0,1000000) : 32;
    const int prompt = argc>6 ? number(argv[6],1,1000000) : 128;
    const int chunk = argc>7 ? number(argv[7],1,1000000) : 64;
    const int context = argc>8 ? number(argv[8],1,1000000) : 9216;
    check(prompt+dec<=context && chunk<=context, "prompt or chunk exceeds context");
    const char * want = setting("VELRI_LLAMA_DEVICE", "Vulkan0");
    std::string kv = setting("VELRI_LLAMA_KV","f32");
    std::string math = setting("VELRI_LLAMA_MATH",kv == "f32" ? "strict" : "native");
    std::string fa = setting("VELRI_LLAMA_FA","auto");
    check((kv=="f32" || kv=="f16") && (math=="strict" || math=="native"), "invalid KV or math setting");
    check(fa=="auto" || fa=="off" || fa=="on", "invalid FA setting");
    if (!std::strncmp(want,"Vulkan",6) && math=="strict") {
        for (auto flag : {"GGML_VK_DISABLE_F16","GGML_VK_DISABLE_COOPMAT","GGML_VK_DISABLE_COOPMAT2"}) {
            check(!setenv(flag,"1",1), "set Vulkan precision flags");
        }
    }
    if (!std::strncmp(want,"CUDA",4) && math=="strict") {
        check(!setenv("NVIDIA_TF32_OVERRIDE","0",1), "set CUDA TF32 override");
    }
    const int history = number(setting("BENCH_HISTORY", "0"), 0, context - prompt - dec);
    std::fprintf(stderr, "append history=%d timed=%d capacity=%d\n", history, prompt, context);
    std::vector<llama_token> ids(history + prompt), forced;
    for (int i=0;i<history+prompt;++i) ids[i] = (37*i+11)%vocab;
    if (const char * path=std::getenv("VELRI_FORCE_TOKENS")) {
        std::ifstream f(path);
        check(f.is_open(), "open forced IDs");
        int id;
        while (f>>id) { check(id>=0 && id<vocab,"forced ID outside vocab"); forced.push_back(id); }
        check(f.eof() && int(forced.size())==dec, "expected one forced input per decode");
    }
    const char * snapshot = std::getenv("VELRI_SPEED_LOGITS");
    std::vector<float> golden(snapshot ? size_t(dec+1)*vocab : 0);
    std::vector<sample> records(size_t(reps)*(dec+1));
    ggml_backend_load_all();
    llama_backend_init();
    ggml_backend_dev_t devices[] = { ggml_backend_dev_by_name(want), nullptr };
    check(devices[0] != nullptr && (ggml_backend_dev_type(devices[0])==GGML_BACKEND_DEVICE_TYPE_GPU ||
          ggml_backend_dev_type(devices[0])==GGML_BACKEND_DEVICE_TYPE_IGPU), "requested GPU unavailable");
    llama_model_params mp = llama_model_default_params();
    ggml_backend_buffer_type_t gpu_buffer = ggml_backend_dev_buffer_type(devices[0]);
    check(gpu_buffer && !ggml_backend_buft_is_host(gpu_buffer), "missing device-local weight buffer");
    const llama_model_tensor_buft_override weights[] = {{".*", gpu_buffer}, {nullptr, nullptr}};
    mp.tensor_buft_overrides = weights;
    mp.devices=devices;
    mp.n_gpu_layers=-1;
    mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu=0;
    auto * model=llama_model_load_from_file(argv[1],mp);
    check(model && llama_vocab_n_tokens(llama_model_get_vocab(model))==vocab, "load model with expected vocabulary");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx=context;
    cp.n_batch=chunk;
    cp.n_ubatch=chunk;
    cp.n_seq_max=1;
    cp.n_threads=cp.n_threads_batch=workers;
    cp.type_k=cp.type_v= kv=="f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
    cp.flash_attn_type=fa=="auto" ? LLAMA_FLASH_ATTN_TYPE_AUTO : fa=="off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv=true;
    cp.op_offload=true;
    cp.no_perf=true;
    std::set<std::string> traced;
    if (std::getenv("VELRI_LLAMA_TRACE")) { cp.cb_eval=trace; cp.cb_eval_user_data=&traced; }
    auto * ctx=llama_init_from_model(model,cp);
    check(ctx && int(llama_n_ctx(ctx))==context && int(llama_n_batch(ctx))==chunk && int(llama_n_ubatch(ctx))==chunk, "init matching context");
    auto * batch=llama_batch_ext_init(ctx);
    check(batch!=nullptr,"create batch");
    std::fprintf(stderr,"matched device=%s (%s) math=%s K=V=%s FA(request)=%s ctx=%u batch=%u ubatch=%u steps=%d full_host_logits=%d forced=%d\n",
                 want,ggml_backend_dev_description(devices[0]),math.c_str(),kv.c_str(),fa.c_str(),llama_n_ctx(ctx),llama_n_batch(ctx),llama_n_ubatch(ctx),dec,vocab,!forced.empty());
    llama_log_set([](ggml_log_level level, const char * message, void *) {
        if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(message, stderr);
    }, nullptr);
    const int warm_n = std::getenv("BENCH_WARM_PROMPT") ? number(std::getenv("BENCH_WARM_PROMPT"),1,prompt) : prompt;
    std::fprintf(stderr, "warm_prompt=%d measured_prompt=%d\n", warm_n, prompt);
    for (int r=0;r<warms+reps;++r) {
        const int run_n = r < warms ? warm_n : prompt;
        if (r) llama_memory_clear(llama_get_memory(ctx),false);
        for (int start = 0; start < history; start += chunk) {
            (void) process(ctx, batch, ids.data() + start, std::min(chunk, history - start), start);
        }
        llama_token next=0;
        for (int step=0;step<=dec;++step) {
            const auto begin=std::chrono::steady_clock::now();
            const float * logits=nullptr;
            if (!step) {
                for (int start=0;start<run_n;start+=chunk) {
                    logits=process(ctx,batch,ids.data()+history+start,std::min(chunk,run_n-start),history+start);
                }
            } else {
                auto input=forced.empty()?next:forced[step-1];
                logits=process(ctx,batch,&input,1,history+run_n+step-1);
            }
            next=greedy(logits);
            const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
            if(r>=warms) {
                if(r>warms) check(records[step].token==next,"greedy trajectory differs across repetitions");
                records[size_t(r-warms)*(dec+1)+step]={us,next};
                if(snapshot && r+1==warms+reps) std::memcpy(golden.data()+size_t(step)*vocab,logits,vocab*sizeof(float));
            }
        }
    }
    std::puts("phase,rep,step,us,replays,token");
    for (int r=0;r<reps;++r) for(int step=0;step<=dec;++step) {
        const auto & o=records[size_t(r)*(dec+1)+step];
        std::printf("%s,%d,%d,%.3f,0,%d\n",step?"decode":"prefill",r,step,o.us,o.token);
    }
    if(snapshot) {
        FILE * f=std::fopen(snapshot,"wb");
        check(f && std::fwrite(golden.data(),sizeof(float),golden.size(),f)==golden.size() && !std::fclose(f),"write logits");
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
