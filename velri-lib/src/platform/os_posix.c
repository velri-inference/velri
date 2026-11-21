#if defined(__linux__)
#define _GNU_SOURCE
#else
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#endif

#include "os.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if __has_include(<unistd.h>)
#include <unistd.h>
#endif
#if __has_include(<sys/mman.h>)
#include <sys/mman.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif
#if defined(__linux__) && __has_include(<sched.h>)
#include <sched.h>
#endif

/* ---- Aligned Memory ---- */

void *velri_os_aligned_alloc(size_t align, size_t bytes) {
    size_t eff_align = align < sizeof(void *) ? sizeof(void *) : align;
    if ((eff_align & (eff_align - 1U)) != 0U) {
        return nullptr;
    }
    size_t sz = bytes ? ((bytes + eff_align - 1U) & ~(eff_align - 1U)) : eff_align;
    void *p = nullptr;
    if (posix_memalign(&p, eff_align, sz) != 0) {
        return nullptr;
    }
    return p;
}

void velri_os_aligned_free(void *ptr) { free(ptr); }

/* ---- Guard-Page Memory ---- */

velri_os_guard_buf velri_os_guard_alloc(size_t bytes, size_t align) {
    velri_os_guard_buf gb = {0};
    if (align == 0U) {
        align = 1U;
    }
#if __has_include(<sys/mman.h>) && __has_include(<unistd.h>) && defined(MAP_ANONYMOUS)
    long ps = sysconf(_SC_PAGESIZE);
    size_t page = (ps > 0) ? (size_t)ps : 4096U;
    size_t need = ((bytes + align + page - 1U) / page) * page;
    if (need == 0U) {
        need = page;
    }
    size_t total = need + page;
    uint8_t *base =
        (uint8_t *)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        abort();
    }
    if (mprotect(base + need, page, PROT_NONE) != 0) {
        abort();
    }
    uintptr_t end_addr = (uintptr_t)(base + need);
    uintptr_t start_addr = (end_addr - bytes) & ~(uintptr_t)(align - 1U);
    gb.raw_map = base;
    gb.map_bytes = total;
    gb.ptr = (void *)start_addr;
#else
    gb.raw_map = malloc(bytes + align);
    if (!gb.raw_map) {
        abort();
    }
    uintptr_t u = ((uintptr_t)gb.raw_map + align - 1U) & ~(uintptr_t)(align - 1U);
    gb.ptr = (void *)u;
#endif
    return gb;
}

void velri_os_guard_free(velri_os_guard_buf *gb) {
    if (!gb || !gb->raw_map) {
        return;
    }
#if __has_include(<sys/mman.h>) && __has_include(<unistd.h>) && defined(MAP_ANONYMOUS)
    (void)munmap(gb->raw_map, gb->map_bytes);
#else
    free(gb->raw_map);
#endif
    gb->raw_map = nullptr;
    gb->ptr = nullptr;
    gb->map_bytes = 0U;
}

/* ---- Monotonic Time & Sleep ---- */

uint64_t velri_os_monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

void velri_os_sleep_ms(uint32_t ms) {
    if (ms == 0U) {
        return;
    }
    struct timespec ts = {
        .tv_sec = (time_t)(ms / 1000U),
        .tv_nsec = (long)((ms % 1000U) * 1000000U),
    };
    nanosleep(&ts, nullptr);
}

/* ---- Dynamic Shared Library Loading ---- */

velri_os_lib *velri_os_dlopen_vulkan(void) {
    static const char *const candidates[] = {
#if defined(__APPLE__)
        "libvulkan.1.dylib",
        "libvulkan.dylib",
        "libMoltenVK.dylib",
#else
        "libvulkan.so.1",
        "libvulkan.so",
#endif
        nullptr,
    };
    for (size_t i = 0U; candidates[i] != nullptr; ++i) {
        void *h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (h) {
            return (velri_os_lib *)h;
        }
    }
    return nullptr;
}

void *velri_os_dlsym(velri_os_lib *lib, const char *symbol) {
    if (!lib || !symbol) {
        return nullptr;
    }
    return dlsym((void *)lib, symbol);
}

void velri_os_dlclose(velri_os_lib *lib) {
    if (lib) {
        (void)dlclose((void *)lib);
    }
}

/* ---- Threads, Mutex, Condition Variable ---- */

struct velri_os_thread {
    pthread_t handle;
    velri_os_thread_fn fn;
    void *arg;
};

struct velri_os_mutex {
    pthread_mutex_t handle;
};

struct velri_os_cond {
    pthread_cond_t handle;
};

static void *posix_thread_trampoline(void *raw) {
    velri_os_thread *t = (velri_os_thread *)raw;
    t->fn(t->arg);
    return nullptr;
}

bool velri_os_thread_spawn(velri_os_thread **out_thread, velri_os_thread_fn fn, void *arg) {
    if (!out_thread || !fn) {
        return false;
    }
    *out_thread = nullptr;
    velri_os_thread *t = (velri_os_thread *)calloc(1U, sizeof(*t));
    if (!t) {
        return false;
    }
    t->fn = fn;
    t->arg = arg;
    if (pthread_create(&t->handle, nullptr, posix_thread_trampoline, t) != 0) {
        free(t);
        return false;
    }
    *out_thread = t;
    return true;
}

void velri_os_thread_join(velri_os_thread *thread) {
    if (!thread) {
        return;
    }
    (void)pthread_join(thread->handle, nullptr);
    free(thread);
}

bool velri_os_mutex_create(velri_os_mutex **out_mtx) {
    if (!out_mtx) {
        return false;
    }
    *out_mtx = nullptr;
    velri_os_mutex *m = (velri_os_mutex *)calloc(1U, sizeof(*m));
    if (!m) {
        return false;
    }
    if (pthread_mutex_init(&m->handle, nullptr) != 0) {
        free(m);
        return false;
    }
    *out_mtx = m;
    return true;
}

void velri_os_mutex_destroy(velri_os_mutex *mtx) {
    if (!mtx) {
        return;
    }
    pthread_mutex_destroy(&mtx->handle);
    free(mtx);
}

void velri_os_mutex_lock(velri_os_mutex *mtx) { pthread_mutex_lock(&mtx->handle); }

void velri_os_mutex_unlock(velri_os_mutex *mtx) { pthread_mutex_unlock(&mtx->handle); }

bool velri_os_cond_create(velri_os_cond **out_cv) {
    if (!out_cv) {
        return false;
    }
    *out_cv = nullptr;
    velri_os_cond *c = (velri_os_cond *)calloc(1U, sizeof(*c));
    if (!c) {
        return false;
    }
    if (pthread_cond_init(&c->handle, nullptr) != 0) {
        free(c);
        return false;
    }
    *out_cv = c;
    return true;
}

void velri_os_cond_destroy(velri_os_cond *cv) {
    if (!cv) {
        return;
    }
    pthread_cond_destroy(&cv->handle);
    free(cv);
}

void velri_os_cond_wait(velri_os_cond *cv, velri_os_mutex *mtx) {
    pthread_cond_wait(&cv->handle, &mtx->handle);
}

void velri_os_cond_signal(velri_os_cond *cv) { pthread_cond_signal(&cv->handle); }

/* ---- CPU & Cache Topology Hints ---- */

int velri_os_hardware_threads(void) {
#if defined(_SC_NPROCESSORS_ONLN)
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc > 0 && nproc < 65536) {
        return (int)nproc;
    }
#endif
    return 1;
}

#if defined(__linux__)
static uint8_t g_linux_l3_domain[1024];
#endif

void velri_os_detect_l3(int hardware_threads, int *out_num_domains, size_t *out_domain_bytes) {
    int num_domains = 1;
    size_t domain_bytes = 16ULL * 1024ULL * 1024ULL;

#if defined(__linux__)
    int seen_l3[256];
    int seen_cnt = 0;
    int limit = hardware_threads < 1024 ? hardware_threads : 1024;
    for (int cpu = 0; cpu < limit; ++cpu) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index3/id", cpu);
        FILE *f = fopen(path, "r");
        if (!f) {
            continue;
        }
        int id = -1;
        bool ok = (fscanf(f, "%d", &id) == 1 && id >= 0);
        fclose(f);
        if (!ok) {
            continue;
        }
        int dom_idx = -1;
        for (int i = 0; i < seen_cnt; ++i) {
            if (seen_l3[i] == id) {
                dom_idx = i;
                break;
            }
        }
        if (dom_idx < 0 && seen_cnt < 256) {
            dom_idx = seen_cnt;
            seen_l3[seen_cnt++] = id;
        }
        if (dom_idx >= 0) {
            g_linux_l3_domain[cpu] = (uint8_t)dom_idx;
        }
    }
    if (seen_cnt > 1) {
        num_domains = seen_cnt;
    }
    FILE *fsz = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (fsz) {
        unsigned long val = 0;
        char unit = '\0';
        if (fscanf(fsz, "%lu%c", &val, &unit) >= 1 && val > 0) {
            size_t mult = 1;
            if (unit == 'K' || unit == 'k') {
                mult = 1024ULL;
            } else if (unit == 'M' || unit == 'm') {
                mult = 1024ULL * 1024ULL;
            } else if (unit == 'G' || unit == 'g') {
                mult = 1024ULL * 1024ULL * 1024ULL;
            }
            if ((size_t)val * mult >= 1024ULL * 1024ULL) {
                domain_bytes = (size_t)val * mult;
            }
        }
        fclose(fsz);
    }
#else
    (void)hardware_threads;
#endif

    if (out_num_domains) {
        *out_num_domains = num_domains;
    }
    if (out_domain_bytes) {
        *out_domain_bytes = domain_bytes;
    }
}

int velri_os_current_l3_domain(void) {
#if defined(__linux__) && __has_include(<sched.h>)
    int cpu = sched_getcpu();
    if (cpu >= 0 && (size_t)cpu < sizeof(g_linux_l3_domain)) {
        return (int)g_linux_l3_domain[cpu];
    }
#endif
    return -1;
}

void velri_os_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

size_t velri_os_ctz64(uint64_t x) { return (size_t)__builtin_ctzll(x); }
