#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

static atomic_uint raw_calls, packed_calls;

#define OUTPUT_PARAMS                                                                              \
    const float *bias, const float *xs, const float *ws, uint32_t flags, float *y, uint32_t ys,    \
    uint32_t r0, uint32_t r1, uint32_t c0, uint32_t c1, uint32_t k
#define OUTPUT_ARGS bias, xs, ws, flags, y, ys, r0, r1, c0, c1, k

void __real_velri_slang_dot_linear_w8a8_f32(const int8_t *, uint32_t, const int8_t *, uint32_t,
                                            OUTPUT_PARAMS);
void __real_velri_slang_dot_linear_w8a8_f32_packed(const int8_t *, uint32_t, const int8_t *,
                                                   OUTPUT_PARAMS);

void __wrap_velri_slang_dot_linear_w8a8_f32(const int8_t *x, uint32_t stride, const int8_t *w,
                                            uint32_t wstride, OUTPUT_PARAMS) {
    atomic_fetch_add_explicit(&raw_calls, 1, memory_order_relaxed);
    __real_velri_slang_dot_linear_w8a8_f32(x, stride, w, wstride, OUTPUT_ARGS);
}

void __wrap_velri_slang_dot_linear_w8a8_f32_packed(const int8_t *x, uint32_t stride,
                                                   const int8_t *w, OUTPUT_PARAMS) {
    atomic_fetch_add_explicit(&packed_calls, 1, memory_order_relaxed);
    __real_velri_slang_dot_linear_w8a8_f32_packed(x, stride, w, OUTPUT_ARGS);
}

__attribute__((destructor)) static void report_calls(void) {
    fprintf(stderr, "actual dot: raw=%u packed=%u\n", atomic_load(&raw_calls),
            atomic_load(&packed_calls));
}
