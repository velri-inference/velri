#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void velri_slang_dot_pack_w8a8(const int8_t *, uint32_t, uint32_t, uint32_t, int8_t *);
extern void velri_slang_dot_linear_w8a8_f32(const int8_t *, uint32_t, const int8_t *, uint32_t,
    const float *, const float *, const float *, uint32_t, float *, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern void velri_slang_dot_linear_w8a8_f32_packed(const int8_t *, uint32_t, const int8_t *,
    const float *, const float *, const float *, uint32_t, float *, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

enum { K = 4096, ROUNDS = 5, MAX_WORKERS = 16, MAX_RING = 8 };
typedef struct {
    const int8_t *x, *w;
    float *y, *xs, *ws, *ref;
} panel;
typedef struct {
    uint32_t cpu;
    panel ring[MAX_RING];
    uint64_t begin[ROUNDS], end[ROUNDS];
} worker;
static worker workers[MAX_WORKERS];
static uint32_t nworkers, rows, cols, ring_size, packed;
static uint64_t calls;
static pthread_barrier_t barrier;

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts)) abort();
    return UINT64_C(1000000000)*ts.tv_sec + ts.tv_nsec;
}
static void *allocate(size_t bytes) {
    void *data;
    if (posix_memalign(&data, 64, bytes)) abort();
    return data;
}
static void fill(int8_t *dest, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        dest[i] = (int32_t)(seed & 255u) - 128;
    }
}
static void init_panel(panel *p, uint32_t id) {
    int8_t *x = allocate(rows * K), *raw = allocate(cols * K);
    fill(x, rows * K, 47 + id * 953);
    fill(raw, cols * K, 23 + id * 331);
    p->x = x;
    p->w = raw;
    p->xs = allocate(rows * sizeof(float));
    p->ws = allocate(cols * sizeof(float));
    p->y = allocate(rows * cols * sizeof(float));
    p->ref = allocate(rows * cols * sizeof(float));
    for (uint32_t r = 0; r < rows; ++r) p->xs[r] = .01f;
    for (uint32_t c = 0; c < cols; ++c) p->ws[c] = .02f;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < K; ++k)
                sum += (int)x[r*K+k] * (int)raw[c*K+k];
            p->ref[r*cols+c] = (float)sum * (p->xs[r] * p->ws[c]);
        }
    }
    if (packed) {
        int8_t *encoded = allocate(cols * K);
        velri_slang_dot_pack_w8a8(raw, K, cols, K, encoded);
        p->w = encoded;
        free(raw);
    }
}
static inline void run_panel(panel *p) {
    if (packed)
        velri_slang_dot_linear_w8a8_f32_packed(p->x, K, p->w, NULL, p->xs, p->ws, 6,
                                               p->y, cols, 0, rows, 0, cols, K);
    else
        velri_slang_dot_linear_w8a8_f32(p->x, K, p->w, K, NULL, p->xs, p->ws, 6,
                                        p->y, cols, 0, rows, 0, cols, K);
}
static void check_panel(const panel *p) {
    if (memcmp(p->y, p->ref, rows * cols * sizeof(float))) {
        fprintf(stderr, "output mismatch\n");
        abort();
    }
}
static void run_batch(worker *w, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i)
        run_panel(&w->ring[i & (ring_size-1)]);
}
static void *worker_main(void *arg) {
    worker *w = arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(w->cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) abort();
    for (uint32_t j = 0; j < ring_size; ++j) {
        init_panel(&w->ring[j], w->cpu * MAX_RING + j);
        run_panel(&w->ring[j]);
        check_panel(&w->ring[j]);
    }
    pthread_barrier_wait(&barrier);
    uint64_t warm_until = now_ns() + 250000000;
    do { run_batch(w, 8); } while (now_ns() < warm_until);
    for (uint32_t i = 0; i < ROUNDS; ++i) {
        pthread_barrier_wait(&barrier);
        w->begin[i] = now_ns();
        run_batch(w, calls);
        w->end[i] = now_ns();
    }
    for (uint32_t j = 0; j < ring_size; ++j) check_panel(&w->ring[j]);
    return NULL;
}
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    nworkers = (uint32_t)atoi(argv[1]);
    rows = (uint32_t)atoi(argv[2]);
    cols = (uint32_t)atoi(argv[3]);
    ring_size = (uint32_t)atoi(argv[4]);
    packed = (uint32_t)atoi(argv[5]);
    if (!nworkers || nworkers > MAX_WORKERS || !rows || rows > 16 || !cols || cols > 80 ||
        !ring_size || ring_size > MAX_RING || (ring_size & (ring_size - 1)) || packed > 1) return 2;
    uint64_t ops = (uint64_t)2 * rows * cols * K;
    calls = (UINT64_C(16000000000)/ops + ring_size-1) / ring_size * ring_size;
    pthread_t ids[MAX_WORKERS];
    if (pthread_barrier_init(&barrier, NULL, nworkers)) abort();
    for (uint32_t w = 0; w < nworkers; ++w) {
        workers[w].cpu = w;
        if (pthread_create(&ids[w], NULL, worker_main, &workers[w])) abort();
    }
    for (uint32_t w = 0; w < nworkers; ++w)
        if (pthread_join(ids[w], NULL)) abort();
    for (uint32_t i = 0; i < ROUNDS; ++i) {
        uint64_t first = UINT64_MAX, last = 0;
        for (uint32_t w = 0; w < nworkers; ++w) {
            if (workers[w].begin[i] < first) first = workers[w].begin[i];
            if (workers[w].end[i] > last) last = workers[w].end[i];
        }
        uint64_t span = last - first;
        uint64_t bytes = ring_size * ((rows + cols) * (uint64_t)K + 4 * (rows*cols+rows+cols));
        printf("%u,%u,%u,%u,%u,%s,%zu,%zu,%u,%zu,%.9g\n", nworkers, rows, cols, K, ring_size,
               packed ? "packed" : "raw", (size_t)bytes, (size_t)calls, i, (size_t)span,
               (double)ops * nworkers * calls / (span * 1000.));
    }
    return 0;
}
