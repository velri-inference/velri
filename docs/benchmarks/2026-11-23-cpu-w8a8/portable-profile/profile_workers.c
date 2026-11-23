/* Linux profiling shim for the exported Slang W8A8 entries. */
#define _GNU_SOURCE
#include <inttypes.h>
#include <sched.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    alignas(64) uint64_t begin, end, cpu_begin, cpu_end;
    int first_cpu, last_cpu;
} worker_span;

typedef struct {
    uint64_t begin, end;
} graph_span;

static worker_span *workers;
static graph_span *graphs;
static size_t width, count, current, capacity;

static uint64_t clock_ns(clockid_t id) {
    struct timespec t;
    if (clock_gettime(id, &t)) abort();
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

/* Only the timed section of the existing graph driver enables profiling. */
void profile_init(size_t channels, size_t samples, size_t iterations) {
    const char *n = getenv("VELRI_PROFILE_WORKERS");
    if (!n || !getenv("VELRI_PROFILE_TRACES")) abort();
    count = strtoul(n, NULL, 10);
    if (!count || channels % count) abort();
    width = channels / count;
    capacity = samples * iterations;
    workers = aligned_alloc(alignof(worker_span), capacity * count * sizeof(*workers));
    graphs = calloc(capacity, sizeof(*graphs));
    if (!workers || !graphs) abort();
    for (size_t i = 0; i < capacity * count; ++i) workers[i] = (worker_span){0};
}

void profile_graph_begin(void) {
    if (current >= capacity) abort();
    graphs[current].begin = clock_ns(CLOCK_MONOTONIC);
}

void profile_graph_end(void) {
    graphs[current].end = clock_ns(CLOCK_MONOTONIC);
    ++current;
}

static worker_span *profile_enter(uint32_t col0, uint32_t col1) {
    if (!workers) return NULL;
    if (col1 - col0 != width || col0 % width || col1 / width > count) abort();
    worker_span *w = &workers[current * count + col0 / width];
    w->first_cpu = sched_getcpu();
    w->begin = clock_ns(CLOCK_MONOTONIC);
    w->cpu_begin = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    return w;
}

static void profile_leave(worker_span *w) {
    if (w) {
        w->cpu_end = clock_ns(CLOCK_THREAD_CPUTIME_ID);
        w->end = clock_ns(CLOCK_MONOTONIC);
        w->last_cpu = sched_getcpu();
    }
}

void profile_save(void) {
    if (current != capacity) abort();
    FILE *out = fopen(getenv("VELRI_PROFILE_TRACES"), "w");
    if (!out) abort();
    fputs("call,worker,first_cpu,last_cpu,graph_ns,kernel_start_ns,kernel_end_ns,thread_cpu_ns\n", out);
    for (size_t call = 0; call < current; ++call) {
        graph_span g = graphs[call];
        for (size_t i = 0; i < count; ++i) {
            worker_span w = workers[call * count + i];
            if (!(g.begin <= w.begin && w.begin < w.end && w.end <= g.end)) abort();
            fprintf(out, "%zu,%zu,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                    call, i, w.first_cpu, w.last_cpu, g.end - g.begin,
                    w.begin - g.begin, w.end - g.begin, w.cpu_end - w.cpu_begin);
        }
    }
    if (fclose(out)) abort();
    free(workers); workers = NULL;
    free(graphs);
}

#define PREFIX const int8_t *x, uint32_t xs, const int8_t *w
#define SUFFIX const float *bias, const float *xscl, const float *wscl, uint32_t flags, \
               float *y, uint32_t ys, uint32_t r0, uint32_t r1, uint32_t c0, uint32_t c1, uint32_t k
#define PASS_SUFFIX bias, xscl, wscl, flags, y, ys, r0, r1, c0, c1, k
#define WRAP_VARIANT(v) \
    extern void __real_velri_slang_##v##_linear_w8a8_f32(PREFIX, uint32_t ws, SUFFIX); \
    void __wrap_velri_slang_##v##_linear_w8a8_f32(PREFIX, uint32_t ws, SUFFIX) { \
        worker_span *span = profile_enter(c0,c1); \
        __real_velri_slang_##v##_linear_w8a8_f32(x,xs,w,ws,PASS_SUFFIX); \
        profile_leave(span); \
    } \
    extern void __real_velri_slang_##v##_linear_w8a8_f32_packed(PREFIX, SUFFIX); \
    void __wrap_velri_slang_##v##_linear_w8a8_f32_packed(PREFIX, SUFFIX) { \
        worker_span *span = profile_enter(c0,c1); \
        __real_velri_slang_##v##_linear_w8a8_f32_packed(x,xs,w,PASS_SUFFIX); \
        profile_leave(span); \
    }
WRAP_VARIANT(base)
#ifdef PROFILE_AVX2
WRAP_VARIANT(avx2)
#endif
#ifdef PROFILE_AVX512
WRAP_VARIANT(avx512)
#endif
