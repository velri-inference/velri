/* Linux x86-64 CPU-clock IP sampling for the benchmark executable. Link with
 * -no-pie -Wl,--wrap=pthread_create. Only the warmed slot_run loop is sampled. */
#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

extern char __executable_start[], etext[];
static _Atomic unsigned *hits;
static _Atomic unsigned active, handlers, external_hits, overruns;
static struct timespec start_wall;
static struct rusage start_usage;

static void sample_ip(int signo, siginfo_t *info, void *context) {
    atomic_fetch_add(&handlers, 1);
    if (atomic_load(&active)) {
        uintptr_t ip = (uintptr_t)((ucontext_t *)context)->uc_mcontext.gregs[REG_RIP];
        if (ip >= (uintptr_t)__executable_start && ip < (uintptr_t)etext)
            atomic_fetch_add(&hits[ip - (uintptr_t)__executable_start], 1);
        else
            atomic_fetch_add(&external_hits, 1);
        if (info->si_overrun > 0) atomic_fetch_add(&overruns, (unsigned)info->si_overrun);
    }
    atomic_fetch_sub(&handlers, 1);
}

static timer_t start_timer(void) {
    const unsigned tid = (unsigned)syscall(SYS_gettid);
    struct sigevent event = {.sigev_notify = SIGEV_THREAD_ID, .sigev_signo = SIGPROF};
    event._sigev_un._tid = (pid_t)tid;
    const unsigned phase_ns = 250001 + (tid * 1511731u) % 750001;
    const struct itimerspec interval = {{0, 1000003}, {0, phase_ns}};
    timer_t timer;
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &event, &timer) ||
        timer_settime(timer, 0, &interval, NULL)) {
        perror("thread CPU timer");
        abort();
    }
    return timer;
}

extern int __real_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
struct thread_start { void *(*run)(void *); void *arg; };

static void *sample_thread(void *arg) {
    struct thread_start start = *(struct thread_start *)arg;
    free(arg);
    timer_t timer = start_timer();
    void *result = start.run(start.arg);
    timer_delete(timer);
    return result;
}

int __wrap_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                          void *(*run)(void *), void *arg) {
    struct thread_start *start = malloc(sizeof(*start));
    if (!start) abort();
    *start = (struct thread_start){run, arg};
    int status = __real_pthread_create(thread, attr, sample_thread, start);
    if (status) free(start);
    return status;
}

__attribute__((constructor)) static void init_sampler(void) {
    struct sigaction action = {.sa_sigaction = sample_ip, .sa_flags = SA_SIGINFO | SA_RESTART};
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGPROF, &action, NULL)) abort();
    start_timer();
}

void profile_start(void) {
    hits = calloc((uintptr_t)etext - (uintptr_t)__executable_start, sizeof(*hits));
    if (!hits) abort();
    clock_gettime(CLOCK_MONOTONIC, &start_wall);
    getrusage(RUSAGE_SELF, &start_usage);
    atomic_store(&active, 1);
}

static double seconds_between(struct timeval a, struct timeval b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_usec - a.tv_usec) * 1e-6;
}

void profile_stop(void) {
    atomic_store(&active, 0);
    while (atomic_load(&handlers)) {}
    struct rusage usage;
    struct timespec wall;
    getrusage(RUSAGE_SELF, &usage);
    clock_gettime(CLOCK_MONOTONIC, &wall);
    double seconds = (wall.tv_sec - start_wall.tv_sec) + (wall.tv_nsec - start_wall.tv_nsec) * 1e-9;
    FILE *out = fopen(getenv("VELRI_SAMPLE_FILE"), "w");
    if (!out) abort();
    unsigned samples = external_hits;
    for (uintptr_t ip = (uintptr_t)__executable_start; ip < (uintptr_t)etext; ++ip)
        samples += hits[ip - (uintptr_t)__executable_start];
    fprintf(out, "# sample=%u overruns=%u wall=%.8f user=%.8f system=%.8f\n",
            samples, (unsigned)overruns, seconds, seconds_between(start_usage.ru_utime, usage.ru_utime),
            seconds_between(start_usage.ru_stime, usage.ru_stime));
    for (uintptr_t ip = (uintptr_t)__executable_start; ip < (uintptr_t)etext; ++ip) {
        unsigned count = hits[ip - (uintptr_t)__executable_start];
        if (count) fprintf(out, "graph %lx %u\n", (unsigned long)ip, count);
    }
    if (external_hits) fprintf(out, "external 0 %u\n", (unsigned)external_hits);
    fclose(out);
}
