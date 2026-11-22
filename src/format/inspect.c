#include "format/inspect.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>

void velri_text_appendf(velri_text_writer *sb, const char *fmt, ...) {
    if (!sb || !sb->buf || sb->cap == 0U || sb->pos >= sb->cap) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(sb->buf + sb->pos, sb->cap - sb->pos, fmt, ap);
    va_end(ap);
    if (n > 0) {
        size_t avail = sb->cap - 1U - sb->pos;
        sb->pos += ((size_t)n < avail) ? (size_t)n : avail;
    }
}

velri_program_device_mem velri_device_memory_from_plan(const velri_device_memory_report *s) {
    return (velri_program_device_mem){
        .device_id = s->device_id,
        .is_uma = s->is_uma,
        .weight_bytes = s->weight_bytes,
        .state_bytes = s->state_bytes,
        .input_output_bytes = s->input_output_bytes,
        .scratch_packed_bytes = s->scratch_packed_bytes,
        .scratch_unpacked_bytes = s->scratch_unpacked_bytes,
        .allocator_overhead_bytes = s->allocator_overhead_bytes,
    };
}

void velri_inspect_memory_summary(velri_text_writer *sb, size_t total, size_t staging,
                                  uint64_t transfers) {
    velri_text_appendf(sb,
                       "Memory Plan: total=%zu B | ram_staging=%zu B | ram_transfers=%" PRIu64 "\n",
                       total, staging, transfers);
}

void velri_inspect_memory_device(velri_text_writer *sb, const velri_context *ctx,
                                 const velri_program_device_mem *dr) {
    velri_device_caps caps;
    bool ok = (velri_context_device_caps(ctx, dr->device_id, &caps) == VELRI_OK);
    velri_text_appendf(sb,
                       "  %s:%u | weights=%zu B | state=%zu B | io=%zu B | scratch=%zu B "
                       "(unpacked=%zu B) | ovh=%zu B\n",
                       ok ? caps.backend_name : "dev", ok ? caps.local_index : dr->device_id,
                       dr->weight_bytes, dr->state_bytes, dr->input_output_bytes,
                       dr->scratch_packed_bytes, dr->scratch_unpacked_bytes,
                       dr->allocator_overhead_bytes);
}
