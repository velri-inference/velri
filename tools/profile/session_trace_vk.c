/* Observes one benchmark device and one queue. Captures the exact recorded pipeline after
 * fusion and reads query pairs after public session_step has waited for the frame. */
#include "session_trace.h"
#include "vk_internal.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    char ops[256];
    int pipe;
    uint32_t gx, gy;
} dispatch;

typedef struct command_trace {
    VkCommandBuffer cmd;
    VkQueryPool pool;
    dispatch *events;
    uint64_t *ticks;
    uint32_t count, capacity;
    bool submitted;
    struct command_trace *next;
} command_trace;

static velri_backend_device *device;
static PFN_vkCmdDispatch real_dispatch;
static PFN_vkCmdBindPipeline real_bind;
static PFN_vkBeginCommandBuffer real_begin;
static PFN_vkQueueSubmit real_submit;
static PFN_vkDestroyDevice real_destroy;
static PFN_vkCreateQueryPool create_pool;
static PFN_vkDestroyQueryPool destroy_pool;
static PFN_vkCmdResetQueryPool reset_pool;
static PFN_vkCmdWriteTimestamp write_time;
static PFN_vkGetQueryPoolResults read_results;
static command_trace *commands;
static float ns_per_tick;
static uint64_t tick_mask;
static const velri_submit_item *active_items;
static uint32_t group_size, remaining;
static int pipeline;

static command_trace *find_command(VkCommandBuffer cmd) {
    for (command_trace *c = commands; c; c = c->next)
        if (c->cmd == cmd) return c;
    return NULL;
}

static VkResult VKAPI_CALL begin(VkCommandBuffer cmd, const VkCommandBufferBeginInfo *info) {
    command_trace *c = find_command(cmd);
    if (c) c->count = 0;
    return real_begin(cmd, info);
}

static void VKAPI_CALL bind(VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipeline p) {
    for (pipeline = 0; pipeline < PIPE_COUNT && device->pipes[pipeline] != p; ++pipeline) {}
    real_bind(cmd, bp, p);
}

static command_trace *reserve(VkCommandBuffer cmd) {
    command_trace *c = find_command(cmd);
    if (!c) {
        c = calloc(1, sizeof(*c));
        if (!c) trace_fail("allocating query batch");
        c->cmd = cmd;
        c->next = commands;
        commands = c;
    }
    if (!c->count) {
        if (remaining > (UINT32_MAX / 2U - 2U) / VK_MAX_DISPATCH_STAGES)
            trace_fail("command buffer exceeds timestamp index range");
        const uint32_t capacity = remaining * VK_MAX_DISPATCH_STAGES + 2U;
        if (capacity > c->capacity) {
            if (c->pool) destroy_pool(device->device, c->pool, NULL);
            c->capacity = capacity; /* Includes empty bracket and optional logits readback. */
            c->ticks = realloc(c->ticks, 2U * c->capacity * sizeof(*c->ticks));
            c->events = realloc(c->events, c->capacity * sizeof(*c->events));
            VkQueryPoolCreateInfo info = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2U * c->capacity};
            if (!c->ticks || !c->events || create_pool(device->device, &info, NULL, &c->pool) != VK_SUCCESS)
                trace_fail("allocating GPU timestamp queries");
        }
        reset_pool(cmd, c->pool, 0, 2U * c->capacity);
    }
    return c;
}

/* Drain shader writes at a transfer-stage boundary before the closing GPU timestamp.
 * The next dispatch follows this diagnostic barrier; one empty pair records its idle cost. */
static void fence_commands(VkCommandBuffer cmd) {
    VkMemoryBarrier memory = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
    device->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memory, 0, NULL, 0, NULL);
}

static dispatch *event_begin(command_trace *c) {
    if (c->count >= c->capacity) trace_fail("exceeded command query capacity");
    write_time(c->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c->pool, 2U * c->count);
    return &c->events[c->count];
}

static void event_end(command_trace *c) {
    fence_commands(c->cmd);
    write_time(c->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->pool, 2U * c->count + 1U);
    ++c->count;
}

static void VKAPI_CALL measure(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    if (!active_items) { real_dispatch(cmd, x, y, z); return; }
    command_trace *c = reserve(cmd);
    if (!c->count) {
        *event_begin(c) = (dispatch){.pipe = -2, .ops = "empty:0"};
        event_end(c);
    }
    if (pipeline >= PIPE_COUNT) trace_fail("unidentified GPU pipeline");
    dispatch *e = event_begin(c);
    *e = (dispatch){.pipe = pipeline, .gx = x, .gy = y};
    trace_shape(e->ops, sizeof(e->ops), active_items, group_size);
    real_dispatch(cmd, x, y, z);
    event_end(c);
}

void __real_velri_vk_record_readback(velri_backend_device *, VkCommandBuffer, VkBuffer,
                                     velri_io_range, VkBuffer);
void __wrap_velri_vk_record_readback(velri_backend_device *dev, VkCommandBuffer cmd, VkBuffer src,
                                     velri_io_range range, VkBuffer dst) {
    command_trace *c = find_command(cmd);
    if (c && c->count) {
        dispatch *event = event_begin(c);
        *event = (dispatch){.pipe = -3};
        snprintf(event->ops, sizeof(event->ops), "readback:%zu", range.bytes);
    }
    __real_velri_vk_record_readback(dev, cmd, src, range, dst);
    if (c && c->count) event_end(c);
}

static VkResult VKAPI_CALL submit(VkQueue queue, uint32_t count, const VkSubmitInfo *info, VkFence fence) {
    if (trace_active) {
        for (uint32_t s = 0; s < count; ++s) {
            for (uint32_t i = 0; i < info[s].commandBufferCount; ++i) {
                command_trace *c = find_command(info[s].pCommandBuffers[i]);
                if (!c || !c->count) continue;
                if (c->submitted) trace_fail("repeated command buffer in a single step");
                c->submitted = true;
            }
        }
    }
    return real_submit(queue, count, info, fence);
}

void trace_gpu_collect(void) {
    for (command_trace *c = commands; c; c = c->next) {
        if (!c->submitted) continue;
        uint32_t n = 2U * c->count;
        if (read_results(device->device, c->pool, 0, n, n * sizeof(*c->ticks), c->ticks,
                         sizeof(*c->ticks), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            trace_fail("unavailable GPU timestamps after completed step");
        for (uint32_t i = 0; i < c->count; ++i) {
            const dispatch *e = &c->events[i];
            uint64_t dt = (c->ticks[2*i+1] - c->ticks[2*i]) & tick_mask;
            trace_sample(e->pipe, e->ops, (double)dt * ns_per_tick, e->gx, e->gy);
        }
        c->submitted = false;
    }
}

static void VKAPI_CALL destroy(VkDevice vk, const VkAllocationCallbacks *callbacks) {
    while (commands) {
        command_trace *next = commands->next;
        destroy_pool(vk, commands->pool, NULL);
        free(commands->events);
        free(commands->ticks);
        free(commands);
        commands = next;
    }
    real_destroy(vk, callbacks);
    device = NULL;
}

static void install(velri_backend_device *dev) {
    device = dev;
    VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    dev->inst.vkGetPhysicalDeviceProperties2(dev->phys, &props);
    ns_per_tick = props.properties.limits.timestampPeriod;
    uint32_t n = 0;
    dev->inst.vkGetPhysicalDeviceQueueFamilyProperties(dev->phys, &n, NULL);
    VkQueueFamilyProperties *q = calloc(n, sizeof(*q));
    if (!q) trace_fail("allocating queue properties");
    dev->inst.vkGetPhysicalDeviceQueueFamilyProperties(dev->phys, &n, q);
    uint32_t bits = q[dev->queue_family].timestampValidBits;
    free(q);
    if (!bits || bits > 64) trace_fail("compute queue lacks timestamps");
    tick_mask = UINT64_MAX >> (64U - bits);
#define LOAD(dest, name) do { dest = (PFN_##name)dev->inst.vkGetDeviceProcAddr(dev->device, #name); \
    if (!dest) trace_fail("missing " #name); } while (0)
    LOAD(create_pool, vkCreateQueryPool);
    LOAD(destroy_pool, vkDestroyQueryPool);
    LOAD(reset_pool, vkCmdResetQueryPool);
    LOAD(write_time, vkCmdWriteTimestamp);
    LOAD(read_results, vkGetQueryPoolResults);
#undef LOAD
    real_dispatch = dev->vkCmdDispatch; dev->vkCmdDispatch = measure;
    real_bind = dev->vkCmdBindPipeline; dev->vkCmdBindPipeline = bind;
    real_begin = dev->vkBeginCommandBuffer; dev->vkBeginCommandBuffer = begin;
    real_submit = dev->vkQueueSubmit; dev->vkQueueSubmit = submit;
    real_destroy = dev->vkDestroyDevice; dev->vkDestroyDevice = destroy;
}

uint32_t __real_velri_vk_fuse_decode(velri_backend_device *, const velri_submit_item *,
                                     const vk_encoded_dispatch *, uint32_t, vk_encoded_dispatch *);
uint32_t __wrap_velri_vk_fuse_decode(velri_backend_device *dev, const velri_submit_item *items,
                                     const vk_encoded_dispatch *enc, uint32_t count,
                                     vk_encoded_dispatch *out) {
    if (!device) install(dev);
    if (device != dev) trace_fail("expected a single GPU device");
    uint32_t n = __real_velri_vk_fuse_decode(dev, items, enc, count, out);
    bool decode = items->params && items->params->has_seq_len && items->params->seq_len_override == 1;
    active_items = decode ? items : NULL;
    group_size = n ? n : 1;
    remaining = count;
    return n;
}
