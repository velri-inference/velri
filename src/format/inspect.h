#ifndef VELRI_FORMAT_INSPECT_H
#define VELRI_FORMAT_INSPECT_H

#include "velri/velri.h"
#include <velri_lib/graph.h>

typedef struct {
    char *buf;
    size_t cap;
    size_t pos;
} velri_text_writer;

void velri_text_appendf(velri_text_writer *sb, const char *fmt, ...);
velri_program_device_mem velri_device_memory_from_plan(const velri_device_memory_report *s);
void velri_inspect_memory_summary(velri_text_writer *sb, size_t total, size_t staging,
                                  uint64_t transfers);
void velri_inspect_memory_device(velri_text_writer *sb, const velri_context *ctx,
                                 const velri_program_device_mem *dr);

#endif
