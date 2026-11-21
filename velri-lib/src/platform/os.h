#ifndef VELRI_PLATFORM_OS_H
#define VELRI_PLATFORM_OS_H

#include <stddef.h>
#include <stdint.h>

/*
 * Single OS abstraction boundary for Velri-lib (`velri_platform`).
 *
 * All OS-specific headers (<pthread.h>, <windows.h>, <dlfcn.h>, <sys/mman.h>,
 * <unistd.h>, <time.h>, <sched.h>) live strictly in `os_posix.c` and `os_win32.c`.
 * Core runtime, backends, tests, and tools include only this OS-agnostic header.
 */

/* ---- Aligned Memory ---- */
void *velri_os_aligned_alloc(size_t align, size_t bytes);
void velri_os_aligned_free(void *ptr);

/* ---- Guard-Page Memory (no-access page placed immediately after payload) ---- */
typedef struct {
    void *raw_map;
    size_t map_bytes;
    void *ptr;
} velri_os_guard_buf;

velri_os_guard_buf velri_os_guard_alloc(size_t bytes, size_t align);
void velri_os_guard_free(velri_os_guard_buf *gb);

/* ---- Monotonic Time & Sleep ---- */
uint64_t velri_os_monotonic_ns(void);
void velri_os_sleep_ms(uint32_t ms);

/* ---- Dynamic Shared Library Loading ---- */
typedef struct velri_os_lib velri_os_lib;
velri_os_lib *velri_os_dlopen_vulkan(void);
void *velri_os_dlsym(velri_os_lib *lib, const char *symbol);
void velri_os_dlclose(velri_os_lib *lib);

/* ---- Threads, Mutex, Condition Variable ---- */
typedef struct velri_os_thread velri_os_thread;
typedef struct velri_os_mutex velri_os_mutex;
typedef struct velri_os_cond velri_os_cond;

typedef void (*velri_os_thread_fn)(void *arg);

bool velri_os_thread_spawn(velri_os_thread **out_thread, velri_os_thread_fn fn, void *arg);
void velri_os_thread_join(velri_os_thread *thread);

bool velri_os_mutex_create(velri_os_mutex **out_mtx);
void velri_os_mutex_destroy(velri_os_mutex *mtx);
void velri_os_mutex_lock(velri_os_mutex *mtx);
void velri_os_mutex_unlock(velri_os_mutex *mtx);

bool velri_os_cond_create(velri_os_cond **out_cv);
void velri_os_cond_destroy(velri_os_cond *cv);
void velri_os_cond_wait(velri_os_cond *cv, velri_os_mutex *mtx);
void velri_os_cond_signal(velri_os_cond *cv);

/* ---- CPU & Cache Topology Hints ---- */
int velri_os_hardware_threads(void);
void velri_os_detect_l3(int hardware_threads, int *out_num_domains, size_t *out_domain_bytes);
int velri_os_current_l3_domain(void);
void velri_os_cpu_relax(void);
size_t velri_os_ctz64(uint64_t x);

#endif /* VELRI_PLATFORM_OS_H */
