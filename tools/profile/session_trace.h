#pragma once
#include "velri_lib/backend.h"
#include <stdint.h>

extern bool trace_active;
void trace_shape(char *out, size_t size, const velri_submit_item *items, uint32_t count);
void trace_sample(int pipe, const char *shape, double ns, uint32_t gx, uint32_t gy);
void trace_fail(const char *message);
#ifdef TRACE_VULKAN
void trace_gpu_collect(void);
#endif
