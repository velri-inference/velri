#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES

#include "velri_lib/vulkan.h"
#include "velri_vk_shaders.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

/* Push-constant layouts matching `entry_vulkan.slang` compiled with `-fvk-use-c-layout`. */
typedef struct {
    uint64_t x, w, y;
    uint32_t rows, dim, x_stride, y_stride;
    float eps;
    uint32_t has_weight, unit_offset, pad;
} vk_rmsnorm_args;
static_assert(sizeof(vk_rmsnorm_args) == 56, "RmsNormArgs size mismatch");

typedef struct {
    uint64_t residual_inout, x, w, y;
    uint32_t rows, dim, r_stride, x_stride, y_stride;
    float eps;
    uint32_t has_weight, unit_offset;
} vk_fused_add_rmsnorm_args;
static_assert(sizeof(vk_fused_add_rmsnorm_args) == 64, "FusedAddRmsNormArgs size mismatch");

typedef struct {
    uint64_t x, w, b, y;
    uint32_t rows, dim, x_stride, y_stride;
    float eps;
    uint32_t has_weight, has_bias, pad;
} vk_layernorm_args;
static_assert(sizeof(vk_layernorm_args) == 64, "LayerNormArgs size mismatch");

typedef struct {
    uint64_t x, cos_tbl, sin_tbl, y;
    uint32_t n_tokens, n_heads, head_dim, rotary_dim, pos_offset;
    uint32_t tok_sx, head_sx, tok_sy, head_sy, mode;
} vk_rope_args;
static_assert(sizeof(vk_rope_args) == 72, "RopeArgs size mismatch");

typedef struct {
    uint64_t x, y;
    uint32_t count, pad;
} vk_activation_args;
static_assert(sizeof(vk_activation_args) == 24, "ActivationArgs size mismatch");

typedef struct {
    uint64_t gate, up, y;
    uint32_t count;
    float clamp_limit;
} vk_gated_mlp_args;
static_assert(sizeof(vk_gated_mlp_args) == 32, "GatedMlpArgs size mismatch");

typedef struct {
    uint64_t a, b, y;
    uint32_t count;
    float alpha, beta;
    uint32_t has_b, pad;
} vk_vec_axpby_args;
static_assert(sizeof(vk_vec_axpby_args) == 44 || sizeof(vk_vec_axpby_args) == 48, "VecAxpbyArgs size mismatch");

typedef struct {
    uint64_t a, b, y;
    uint32_t count, pad;
} vk_vec_mul_args;
static_assert(sizeof(vk_vec_mul_args) == 32, "VecMulArgs size mismatch");

typedef struct {
    uint64_t x, y;
    uint32_t rows, cols, x_stride, y_stride;
} vk_softmax_args;
static_assert(sizeof(vk_softmax_args) == 32, "SoftmaxArgs size mismatch");

typedef struct {
    uint64_t x, suh, y;
    uint32_t rows, cols, block_size, x_stride, y_stride, suh_mode;
    float inv_sqrt_b;
    uint32_t pad;
} vk_fwht_args;
static_assert(sizeof(vk_fwht_args) == 56, "FwhtArgs size mismatch");

typedef struct {
    uint64_t x, x_i8, scale_out;
    uint32_t rows, k, x_stride, xi8_stride;
} vk_quantize_row_args;
static_assert(sizeof(vk_quantize_row_args) == 40, "QuantizeRowArgs size mismatch");

typedef struct {
    uint64_t x, w, bias, x_scale, w_scale, y;
    uint32_t rows, m, k, x_stride, w_stride, y_stride, flags, split_k;
} vk_linear_args;
static_assert(sizeof(vk_linear_args) == 80, "GemvArgs/CoopGemmArgs size mismatch");

typedef struct {
    uint64_t partials, bias, x_scale, w_scale, y;
    uint32_t rows, m, part_stride_u4, y_stride, split_k, flags, is_int32, pad;
} vk_splitk_reduce_args;
static_assert(sizeof(vk_splitk_reduce_args) == 72, "SplitKReduceArgs size mismatch");

typedef struct {
    uint64_t src, dst;
    uint32_t count_vec4, pad;
} vk_ceiling_bw_args;
static_assert(sizeof(vk_ceiling_bw_args) == 24, "CeilingBwArgs size mismatch");

typedef struct {
    uint64_t sink;
    uint32_t iters;
    float seed;
} vk_ceiling_compute_args;
static_assert(sizeof(vk_ceiling_compute_args) == 16, "CeilingComputeArgs size mismatch");

enum {
    PIPE_RMSNORM_F32 = 0,
    PIPE_RMSNORM_BF16,
    PIPE_RMSNORM_F16,
    PIPE_FUSED_ADD_RMSNORM_F32,
    PIPE_LAYERNORM_F32,
    PIPE_ROPE_F32,
    PIPE_ROPE_BF16,
    PIPE_ACT_SILU_F32,
    PIPE_ACT_GELU_F32,
    PIPE_ACT_RELU2_F32,
    PIPE_GATED_SILU_F32,
    PIPE_GATED_GELU_F32,
    PIPE_VEC_AXPBY_F32,
    PIPE_VEC_MUL_F32,
    PIPE_SOFTMAX_F32,
    PIPE_FWHT_F32,
    PIPE_QUANTIZE_ROW_F32_I8,
    PIPE_GEMV_F32,
    PIPE_GEMV_F16,
    PIPE_GEMV_BF16,
    PIPE_GEMV_W8A8,
    PIPE_GEMM_F32,
    PIPE_SPLITK_REDUCE,
    PIPE_GEMM_COOPMAT_F16,
    PIPE_GEMM_COOPMAT_BF16,
    PIPE_GEMM_COOPMAT_W8A8,
    PIPE_GEMM_COOPMAT_F16_64,
    PIPE_GEMM_COOPMAT_BF16_64,
    PIPE_GEMM_COOPMAT_W8A8_64,
    PIPE_CEIL_BW,
    PIPE_CEIL_BW_READ,
    PIPE_CEIL_FMA_F32,
    PIPE_CEIL_COOPMAT_F16,
    PIPE_CEIL_COOPMAT_BF16,
    PIPE_CEIL_COOPMAT_I8,
    PIPE_COUNT
};

struct velri_vk_buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress bda;
    size_t size;
};

struct velri_vk_ctx {
    void *dl_handle;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    PFN_vkCreateInstance vkCreateInstance;
    PFN_vkDestroyInstance vkDestroyInstance;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
    PFN_vkCreateDevice vkCreateDevice;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;

    PFN_vkDestroyDevice vkDestroyDevice;
    PFN_vkGetDeviceQueue vkGetDeviceQueue;
    PFN_vkCreateCommandPool vkCreateCommandPool;
    PFN_vkDestroyCommandPool vkDestroyCommandPool;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
    PFN_vkResetCommandBuffer vkResetCommandBuffer;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
    PFN_vkEndCommandBuffer vkEndCommandBuffer;
    PFN_vkQueueSubmit vkQueueSubmit;
    PFN_vkQueueWaitIdle vkQueueWaitIdle;
    PFN_vkCreateFence vkCreateFence;
    PFN_vkDestroyFence vkDestroyFence;
    PFN_vkWaitForFences vkWaitForFences;
    PFN_vkResetFences vkResetFences;
    PFN_vkCreateQueryPool vkCreateQueryPool;
    PFN_vkDestroyQueryPool vkDestroyQueryPool;
    PFN_vkGetQueryPoolResults vkGetQueryPoolResults;
    PFN_vkCreateBuffer vkCreateBuffer;
    PFN_vkDestroyBuffer vkDestroyBuffer;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
    PFN_vkAllocateMemory vkAllocateMemory;
    PFN_vkFreeMemory vkFreeMemory;
    PFN_vkBindBufferMemory vkBindBufferMemory;
    PFN_vkMapMemory vkMapMemory;
    PFN_vkUnmapMemory vkUnmapMemory;
    PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress;
    PFN_vkCreateShaderModule vkCreateShaderModule;
    PFN_vkDestroyShaderModule vkDestroyShaderModule;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
    PFN_vkCreateComputePipelines vkCreateComputePipelines;
    PFN_vkDestroyPipeline vkDestroyPipeline;
    PFN_vkCmdBindPipeline vkCmdBindPipeline;
    PFN_vkCmdPushConstants vkCmdPushConstants;
    PFN_vkCmdDispatch vkCmdDispatch;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
    PFN_vkCmdFillBuffer vkCmdFillBuffer;
    PFN_vkCmdResetQueryPool vkCmdResetQueryPool;
    PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp;

    VkInstance instance;
    VkPhysicalDevice phys;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t queue_family;
    VkDevice device;
    VkQueue queue;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkQueryPool query_pool;
    VkPipelineLayout pipe_layout;
    VkPipeline pipes[PIPE_COUNT];

    /* Reusable host-visible staging buffer for upload/download. */
    VkBuffer stage_buf;
    VkDeviceMemory stage_mem;
    void *stage_map;
    size_t stage_cap;

    /* Reusable device-local partial buffer for Split-K GEMM reduction. */
    velri_vk_buf *splitk_buf;

    velri_vk_info info;
};

#define LOAD_INST(ctx, name)                                                                 \
    do {                                                                                     \
        PFN_vkVoidFunction _p = (ctx)->vkGetInstanceProcAddr((ctx)->instance, #name);        \
        memcpy(&(ctx)->name, &_p, sizeof((ctx)->name));                                      \
    } while (0)

#define LOAD_DEV(ctx, name)                                                                  \
    do {                                                                                     \
        PFN_vkVoidFunction _p = (ctx)->vkGetDeviceProcAddr((ctx)->device, #name);            \
        memcpy(&(ctx)->name, &_p, sizeof((ctx)->name));                                      \
    } while (0)

static bool has_ext(const VkExtensionProperties *exts, uint32_t count, const char *name) {
    for (uint32_t i = 0; i < count; ++i) {
        if (strcmp(exts[i].extensionName, name) == 0) return true;
    }
    return false;
}

static uint32_t find_mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t type_bits,
                              VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < mp->memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && ((mp->memoryTypes[i].propertyFlags & req) == req)) {
            return i;
        }
    }
    return UINT32_MAX;
}

static VkPipeline build_compute_pipe(velri_vk_ctx *ctx, const uint32_t *code, size_t code_bytes) {
    VkShaderModuleCreateInfo sm_ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_bytes,
        .pCode = code,
    };
    VkShaderModule sm = VK_NULL_HANDLE;
    if (ctx->vkCreateShaderModule(ctx->device, &sm_ci, nullptr, &sm) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    VkComputePipelineCreateInfo p_ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main",
        },
        .layout = ctx->pipe_layout,
    };
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult res = ctx->vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &p_ci, nullptr, &pipe);
    ctx->vkDestroyShaderModule(ctx->device, sm, nullptr);
    return (res == VK_SUCCESS) ? pipe : VK_NULL_HANDLE;
}

static velri_status ensure_staging(velri_vk_ctx *ctx, size_t bytes) {
    if (bytes <= ctx->stage_cap) return VELRI_OK;
    if (ctx->stage_map) {
        ctx->vkUnmapMemory(ctx->device, ctx->stage_mem);
        ctx->stage_map = nullptr;
    }
    if (ctx->stage_buf) {
        ctx->vkDestroyBuffer(ctx->device, ctx->stage_buf, nullptr);
        ctx->stage_buf = VK_NULL_HANDLE;
    }
    if (ctx->stage_mem) {
        ctx->vkFreeMemory(ctx->device, ctx->stage_mem, nullptr);
        ctx->stage_mem = VK_NULL_HANDLE;
    }
    ctx->stage_cap = 0;

    VkBufferCreateInfo b_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = (VkDeviceSize)bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (ctx->vkCreateBuffer(ctx->device, &b_ci, nullptr, &ctx->stage_buf) != VK_SUCCESS) {
        return VELRI_ERR_NOMEM;
    }
    VkMemoryRequirements mr;
    ctx->vkGetBufferMemoryRequirements(ctx->device, ctx->stage_buf, &mr);
    uint32_t mtype = find_mem_type(&ctx->mem_props, mr.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mtype == UINT32_MAX) return VELRI_ERR_UNSUPPORTED;

    VkMemoryAllocateInfo ma = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = mtype,
    };
    if (ctx->vkAllocateMemory(ctx->device, &ma, nullptr, &ctx->stage_mem) != VK_SUCCESS) {
        return VELRI_ERR_NOMEM;
    }
    if (ctx->vkBindBufferMemory(ctx->device, ctx->stage_buf, ctx->stage_mem, 0) != VK_SUCCESS) {
        return VELRI_ERR_NOMEM;
    }
    if (ctx->vkMapMemory(ctx->device, ctx->stage_mem, 0, mr.size, 0, &ctx->stage_map) != VK_SUCCESS) {
        return VELRI_ERR_NOMEM;
    }
    ctx->stage_cap = bytes;
    return VELRI_OK;
}

static velri_status submit_and_wait(velri_vk_ctx *ctx) {
    if (ctx->vkResetFences(ctx->device, 1, &ctx->fence) != VK_SUCCESS) return VELRI_ERR_INVALID;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd,
    };
    if (ctx->vkQueueSubmit(ctx->queue, 1, &si, ctx->fence) != VK_SUCCESS) return VELRI_ERR_INVALID;
    if (ctx->vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return VELRI_ERR_INVALID;
    }
    return VELRI_OK;
}

velri_status velri_vk_create(velri_vk_ctx **out_ctx) {
    if (!out_ctx) return VELRI_ERR_INVALID;
    *out_ctx = nullptr;

    velri_vk_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return VELRI_ERR_NOMEM;

    ctx->dl_handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!ctx->dl_handle) ctx->dl_handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!ctx->dl_handle) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_UNSUPPORTED;
    }
    void *sym = dlsym(ctx->dl_handle, "vkGetInstanceProcAddr");
    if (!sym) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_UNSUPPORTED;
    }
    memcpy(&ctx->vkGetInstanceProcAddr, &sym, sizeof(ctx->vkGetInstanceProcAddr));

    PFN_vkVoidFunction p_create_inst = ctx->vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    memcpy(&ctx->vkCreateInstance, &p_create_inst, sizeof(ctx->vkCreateInstance));
    if (!ctx->vkCreateInstance) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_UNSUPPORTED;
    }

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "velri-lib",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Velri",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo inst_ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (ctx->vkCreateInstance(&inst_ci, nullptr, &ctx->instance) != VK_SUCCESS) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_UNSUPPORTED;
    }

    LOAD_INST(ctx, vkDestroyInstance);
    LOAD_INST(ctx, vkEnumeratePhysicalDevices);
    LOAD_INST(ctx, vkGetPhysicalDeviceProperties2);
    LOAD_INST(ctx, vkGetPhysicalDeviceFeatures2);
    LOAD_INST(ctx, vkGetPhysicalDeviceMemoryProperties);
    LOAD_INST(ctx, vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_INST(ctx, vkEnumerateDeviceExtensionProperties);
    LOAD_INST(ctx, vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR);
    LOAD_INST(ctx, vkCreateDevice);
    LOAD_INST(ctx, vkGetDeviceProcAddr);

    uint32_t ndev = 0;
    ctx->vkEnumeratePhysicalDevices(ctx->instance, &ndev, nullptr);
    if (ndev == 0) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_UNSUPPORTED;
    }
    VkPhysicalDevice *devs = calloc(ndev, sizeof(*devs));
    if (!devs) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_NOMEM;
    }
    ctx->vkEnumeratePhysicalDevices(ctx->instance, &ndev, devs);

    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    uint32_t chosen_qf = UINT32_MAX;
    int chosen_score = -1;

    for (uint32_t i = 0; i < ndev; ++i) {
        VkPhysicalDeviceSubgroupProperties sg = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        };
        VkPhysicalDeviceProperties2 p2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &sg,
        };
        ctx->vkGetPhysicalDeviceProperties2(devs[i], &p2);
        if (p2.properties.apiVersion < VK_API_VERSION_1_3) continue;

        uint32_t nqf = 0;
        ctx->vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nqf, nullptr);
        VkQueueFamilyProperties *qfs = calloc(nqf, sizeof(*qfs));
        if (!qfs) continue;
        ctx->vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nqf, qfs);
        uint32_t qf = UINT32_MAX;
        for (uint32_t q = 0; q < nqf; ++q) {
            if ((qfs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfs[q].timestampValidBits > 0) {
                qf = q;
                if (!(qfs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
            }
        }
        free(qfs);
        if (qf == UINT32_MAX) continue;

        int score = (p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 10 : 1;
        if (score > chosen_score) {
            chosen_score = score;
            chosen = devs[i];
            chosen_qf = qf;
        }
    }
    free(devs);

    if (!chosen) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_UNSUPPORTED;
    }
    ctx->phys = chosen;
    ctx->queue_family = chosen_qf;

    VkPhysicalDeviceSubgroupProperties sg_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &sg_props,
    };
    ctx->vkGetPhysicalDeviceProperties2(ctx->phys, &props2);
    ctx->vkGetPhysicalDeviceMemoryProperties(ctx->phys, &ctx->mem_props);

    snprintf(ctx->info.device_name, sizeof(ctx->info.device_name), "%s", props2.properties.deviceName);
    ctx->info.api_version = props2.properties.apiVersion;
    ctx->info.subgroup_size = sg_props.subgroupSize ? sg_props.subgroupSize : 32u;
    ctx->info.max_shared_mem = props2.properties.limits.maxComputeSharedMemorySize;
    ctx->info.timestamp_period_ns = props2.properties.limits.timestampPeriod;

    uint32_t nexts = 0;
    ctx->vkEnumerateDeviceExtensionProperties(ctx->phys, nullptr, &nexts, nullptr);
    VkExtensionProperties *exts = calloc(nexts ? nexts : 1u, sizeof(*exts));
    if (!exts) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_NOMEM;
    }
    if (nexts > 0) ctx->vkEnumerateDeviceExtensionProperties(ctx->phys, nullptr, &nexts, exts);
    bool ext_coopmat = has_ext(exts, nexts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    bool ext_bf16 = has_ext(exts, nexts, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
    free(exts);

    VkPhysicalDeviceShaderBfloat16FeaturesKHR f_bf16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
    };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR f_cm = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13,
    };
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &f12,
    };
    void *tail = &f13;
    if (ext_coopmat) {
        f13.pNext = &f_cm;
        tail = &f_cm;
    }
    if (ext_bf16) {
        if (tail == &f13) f13.pNext = &f_bf16;
        else f_cm.pNext = &f_bf16;
    }
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f11,
    };
    ctx->vkGetPhysicalDeviceFeatures2(ctx->phys, &f2);

    ctx->info.has_bda = (f12.bufferDeviceAddress == VK_TRUE);
    ctx->info.has_f16_i8 = (f12.shaderFloat16 == VK_TRUE && f12.shaderInt8 == VK_TRUE &&
                            f2.features.shaderInt16 == VK_TRUE);
    ctx->info.has_int_dot = (f13.shaderIntegerDotProduct == VK_TRUE);
    ctx->info.has_coopmat = (ext_coopmat && f_cm.cooperativeMatrix == VK_TRUE);
    ctx->info.has_bf16 = (ext_bf16 && f_bf16.shaderBFloat16Type == VK_TRUE);

    if (!ctx->info.has_bda || !f12.vulkanMemoryModel || !f12.vulkanMemoryModelDeviceScope) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_UNSUPPORTED;
    }

    if (ctx->info.has_coopmat && ctx->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        uint32_t ncm = 0;
        ctx->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(ctx->phys, &ncm, nullptr);
        VkCooperativeMatrixPropertiesKHR *cm_props = calloc(ncm ? ncm : 1u, sizeof(*cm_props));
        if (cm_props && ncm > 0) {
            for (uint32_t i = 0; i < ncm; ++i) {
                cm_props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }
            ctx->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(ctx->phys, &ncm, cm_props);
            for (uint32_t i = 0; i < ncm; ++i) {
                const VkCooperativeMatrixPropertiesKHR *p = &cm_props[i];
                if (p->scope != VK_SCOPE_SUBGROUP_KHR || p->saturatingAccumulation) continue;
                if (p->MSize == 16 && p->NSize == 16 && p->KSize == 16 &&
                    p->AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    p->BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    p->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                    ctx->info.has_coopmat_f16 = true;
                }
                if (p->MSize == 16 && p->NSize == 16 && p->KSize == 16 &&
                    p->AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    p->BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    p->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    ext_bf16 && f_bf16.shaderBFloat16CooperativeMatrix == VK_TRUE) {
                    ctx->info.has_coopmat_bf16 = true;
                }
                if (p->MSize == 16 && p->NSize == 16 && p->KSize == 32 &&
                    p->AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                    p->BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                    p->CType == VK_COMPONENT_TYPE_SINT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_SINT32_KHR) {
                    ctx->info.has_coopmat_i8 = true;
                }
            }
        }
        free(cm_props);
    }

    const char *en_exts[4];
    uint32_t n_en_exts = 0;
    if (ctx->info.has_coopmat) en_exts[n_en_exts++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    if (ctx->info.has_bf16) en_exts[n_en_exts++] = VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME;

    VkPhysicalDeviceShaderBfloat16FeaturesKHR en_bf16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
        .shaderBFloat16Type = ctx->info.has_bf16 ? VK_TRUE : VK_FALSE,
        .shaderBFloat16CooperativeMatrix = ctx->info.has_coopmat_bf16 ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR en_cm = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = ctx->info.has_bf16 ? (void *)&en_bf16 : nullptr,
        .cooperativeMatrix = ctx->info.has_coopmat ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceVulkan13Features en13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = ctx->info.has_coopmat ? (void *)&en_cm : (ctx->info.has_bf16 ? (void *)&en_bf16 : nullptr),
        .shaderIntegerDotProduct = ctx->info.has_int_dot ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceVulkan12Features en12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &en13,
        .storageBuffer8BitAccess = f12.storageBuffer8BitAccess,
        .shaderFloat16 = f12.shaderFloat16,
        .shaderInt8 = f12.shaderInt8,
        .scalarBlockLayout = f12.scalarBlockLayout,
        .vulkanMemoryModel = VK_TRUE,
        .vulkanMemoryModelDeviceScope = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };
    VkPhysicalDeviceVulkan11Features en11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &en12,
        .storageBuffer16BitAccess = f11.storageBuffer16BitAccess,
    };
    VkPhysicalDeviceFeatures2 en2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &en11,
        .features = {
            .shaderInt16 = f2.features.shaderInt16,
        },
    };

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo q_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &qprio,
    };
    VkDeviceCreateInfo dev_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &en2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &q_ci,
        .enabledExtensionCount = n_en_exts,
        .ppEnabledExtensionNames = en_exts,
    };
    if (ctx->vkCreateDevice(ctx->phys, &dev_ci, nullptr, &ctx->device) != VK_SUCCESS) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_UNSUPPORTED;
    }

    LOAD_DEV(ctx, vkDestroyDevice);
    LOAD_DEV(ctx, vkGetDeviceQueue);
    LOAD_DEV(ctx, vkCreateCommandPool);
    LOAD_DEV(ctx, vkDestroyCommandPool);
    LOAD_DEV(ctx, vkAllocateCommandBuffers);
    LOAD_DEV(ctx, vkResetCommandBuffer);
    LOAD_DEV(ctx, vkBeginCommandBuffer);
    LOAD_DEV(ctx, vkEndCommandBuffer);
    LOAD_DEV(ctx, vkQueueSubmit);
    LOAD_DEV(ctx, vkQueueWaitIdle);
    LOAD_DEV(ctx, vkCreateFence);
    LOAD_DEV(ctx, vkDestroyFence);
    LOAD_DEV(ctx, vkWaitForFences);
    LOAD_DEV(ctx, vkResetFences);
    LOAD_DEV(ctx, vkCreateQueryPool);
    LOAD_DEV(ctx, vkDestroyQueryPool);
    LOAD_DEV(ctx, vkGetQueryPoolResults);
    LOAD_DEV(ctx, vkCreateBuffer);
    LOAD_DEV(ctx, vkDestroyBuffer);
    LOAD_DEV(ctx, vkGetBufferMemoryRequirements);
    LOAD_DEV(ctx, vkAllocateMemory);
    LOAD_DEV(ctx, vkFreeMemory);
    LOAD_DEV(ctx, vkBindBufferMemory);
    LOAD_DEV(ctx, vkMapMemory);
    LOAD_DEV(ctx, vkUnmapMemory);
    LOAD_DEV(ctx, vkGetBufferDeviceAddress);
    LOAD_DEV(ctx, vkCreateShaderModule);
    LOAD_DEV(ctx, vkDestroyShaderModule);
    LOAD_DEV(ctx, vkCreatePipelineLayout);
    LOAD_DEV(ctx, vkDestroyPipelineLayout);
    LOAD_DEV(ctx, vkCreateComputePipelines);
    LOAD_DEV(ctx, vkDestroyPipeline);
    LOAD_DEV(ctx, vkCmdBindPipeline);
    LOAD_DEV(ctx, vkCmdPushConstants);
    LOAD_DEV(ctx, vkCmdDispatch);
    LOAD_DEV(ctx, vkCmdPipelineBarrier);
    LOAD_DEV(ctx, vkCmdCopyBuffer);
    LOAD_DEV(ctx, vkCmdFillBuffer);
    LOAD_DEV(ctx, vkCmdResetQueryPool);
    LOAD_DEV(ctx, vkCmdWriteTimestamp);

    ctx->vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

    VkCommandPoolCreateInfo cp_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->queue_family,
    };
    if (ctx->vkCreateCommandPool(ctx->device, &cp_ci, nullptr, &ctx->cmd_pool) != VK_SUCCESS) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_NOMEM;
    }
    VkCommandBufferAllocateInfo cb_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (ctx->vkAllocateCommandBuffers(ctx->device, &cb_ai, &ctx->cmd) != VK_SUCCESS) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_NOMEM;
    }
    VkFenceCreateInfo fn_ci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (ctx->vkCreateFence(ctx->device, &fn_ci, nullptr, &ctx->fence) != VK_SUCCESS) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_NOMEM;
    }
    VkQueryPoolCreateInfo qp_ci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2,
    };
    if (ctx->vkCreateQueryPool(ctx->device, &qp_ci, nullptr, &ctx->query_pool) != VK_SUCCESS) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_NOMEM;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 128,
    };
    VkPipelineLayoutCreateInfo pl_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (ctx->vkCreatePipelineLayout(ctx->device, &pl_ci, nullptr, &ctx->pipe_layout) != VK_SUCCESS) {
        velri_vk_destroy(ctx);
        return VELRI_ERR_NOMEM;
    }

    /* Core F32 / BF16-emulated / ceiling pipelines (always available on Vulkan 1.3 + Int16). */
    ctx->pipes[PIPE_RMSNORM_F32] = build_compute_pipe(ctx, spv_vk_rmsnorm_f32, spv_vk_rmsnorm_f32_bytes);
    ctx->pipes[PIPE_RMSNORM_BF16] = build_compute_pipe(ctx, spv_vk_rmsnorm_bf16, spv_vk_rmsnorm_bf16_bytes);
    ctx->pipes[PIPE_FUSED_ADD_RMSNORM_F32] =
        build_compute_pipe(ctx, spv_vk_fused_add_rmsnorm_f32, spv_vk_fused_add_rmsnorm_f32_bytes);
    ctx->pipes[PIPE_LAYERNORM_F32] = build_compute_pipe(ctx, spv_vk_layernorm_f32, spv_vk_layernorm_f32_bytes);
    ctx->pipes[PIPE_ROPE_F32] = build_compute_pipe(ctx, spv_vk_rope_f32, spv_vk_rope_f32_bytes);
    ctx->pipes[PIPE_ROPE_BF16] = build_compute_pipe(ctx, spv_vk_rope_bf16, spv_vk_rope_bf16_bytes);
    ctx->pipes[PIPE_ACT_SILU_F32] =
        build_compute_pipe(ctx, spv_vk_activation_silu_f32, spv_vk_activation_silu_f32_bytes);
    ctx->pipes[PIPE_ACT_GELU_F32] =
        build_compute_pipe(ctx, spv_vk_activation_gelu_f32, spv_vk_activation_gelu_f32_bytes);
    ctx->pipes[PIPE_ACT_RELU2_F32] =
        build_compute_pipe(ctx, spv_vk_activation_relu2_f32, spv_vk_activation_relu2_f32_bytes);
    ctx->pipes[PIPE_GATED_SILU_F32] = build_compute_pipe(ctx, spv_vk_gated_silu_f32, spv_vk_gated_silu_f32_bytes);
    ctx->pipes[PIPE_GATED_GELU_F32] = build_compute_pipe(ctx, spv_vk_gated_gelu_f32, spv_vk_gated_gelu_f32_bytes);
    ctx->pipes[PIPE_VEC_AXPBY_F32] = build_compute_pipe(ctx, spv_vk_vec_axpby_f32, spv_vk_vec_axpby_f32_bytes);
    ctx->pipes[PIPE_VEC_MUL_F32] = build_compute_pipe(ctx, spv_vk_vec_mul_f32, spv_vk_vec_mul_f32_bytes);
    ctx->pipes[PIPE_SOFTMAX_F32] = build_compute_pipe(ctx, spv_vk_softmax_f32, spv_vk_softmax_f32_bytes);
    ctx->pipes[PIPE_GEMV_F32] = build_compute_pipe(ctx, spv_vk_gemv_f32, spv_vk_gemv_f32_bytes);
    ctx->pipes[PIPE_GEMV_BF16] = build_compute_pipe(ctx, spv_vk_gemv_bf16, spv_vk_gemv_bf16_bytes);
    ctx->pipes[PIPE_GEMM_F32] = build_compute_pipe(ctx, spv_vk_gemm_f32, spv_vk_gemm_f32_bytes);
    ctx->pipes[PIPE_SPLITK_REDUCE] = build_compute_pipe(ctx, spv_vk_splitk_reduce, spv_vk_splitk_reduce_bytes);
    ctx->pipes[PIPE_CEIL_BW] = build_compute_pipe(ctx, spv_vk_ceiling_bw_f32x4, spv_vk_ceiling_bw_f32x4_bytes);
    ctx->pipes[PIPE_CEIL_BW_READ] =
        build_compute_pipe(ctx, spv_vk_ceiling_bw_read_f32x4, spv_vk_ceiling_bw_read_f32x4_bytes);
    ctx->pipes[PIPE_CEIL_FMA_F32] = build_compute_pipe(ctx, spv_vk_ceiling_fma_f32, spv_vk_ceiling_fma_f32_bytes);

    if (ctx->info.has_f16_i8) {
        ctx->pipes[PIPE_RMSNORM_F16] = build_compute_pipe(ctx, spv_vk_rmsnorm_f16, spv_vk_rmsnorm_f16_bytes);
        ctx->pipes[PIPE_FWHT_F32] = build_compute_pipe(ctx, spv_vk_fwht_f32, spv_vk_fwht_f32_bytes);
        ctx->pipes[PIPE_QUANTIZE_ROW_F32_I8] =
            build_compute_pipe(ctx, spv_vk_quantize_row_f32_i8, spv_vk_quantize_row_f32_i8_bytes);
        ctx->pipes[PIPE_GEMV_F16] = build_compute_pipe(ctx, spv_vk_gemv_f16, spv_vk_gemv_f16_bytes);
    }
    if (ctx->info.has_f16_i8 && ctx->info.has_int_dot) {
        ctx->pipes[PIPE_GEMV_W8A8] = build_compute_pipe(ctx, spv_vk_gemv_w8a8, spv_vk_gemv_w8a8_bytes);
    }
    if (ctx->info.has_coopmat_f16) {
        ctx->pipes[PIPE_GEMM_COOPMAT_F16] =
            build_compute_pipe(ctx, spv_vk_gemm_coopmat_f16, spv_vk_gemm_coopmat_f16_bytes);
        ctx->pipes[PIPE_GEMM_COOPMAT_F16_64] =
            build_compute_pipe(ctx, spv_vk_gemm_coopmat_f16_64, spv_vk_gemm_coopmat_f16_64_bytes);
        ctx->pipes[PIPE_CEIL_COOPMAT_F16] =
            build_compute_pipe(ctx, spv_vk_ceiling_coopmat_f16, spv_vk_ceiling_coopmat_f16_bytes);
    }
    if (ctx->info.has_coopmat_bf16) {
        ctx->pipes[PIPE_GEMM_COOPMAT_BF16] =
            build_compute_pipe(ctx, spv_vk_gemm_coopmat_bf16, spv_vk_gemm_coopmat_bf16_bytes);
        ctx->pipes[PIPE_GEMM_COOPMAT_BF16_64] =
            build_compute_pipe(ctx, spv_vk_gemm_coopmat_bf16_64, spv_vk_gemm_coopmat_bf16_64_bytes);
        ctx->pipes[PIPE_CEIL_COOPMAT_BF16] =
            build_compute_pipe(ctx, spv_vk_ceiling_coopmat_bf16, spv_vk_ceiling_coopmat_bf16_bytes);
    }
    if (ctx->info.has_coopmat_i8) {
        ctx->pipes[PIPE_GEMM_COOPMAT_W8A8] =
            build_compute_pipe(ctx, spv_vk_gemm_coopmat_w8a8, spv_vk_gemm_coopmat_w8a8_bytes);
        ctx->pipes[PIPE_GEMM_COOPMAT_W8A8_64] =
            build_compute_pipe(ctx, spv_vk_gemm_coopmat_w8a8_64, spv_vk_gemm_coopmat_w8a8_64_bytes);
        ctx->pipes[PIPE_CEIL_COOPMAT_I8] =
            build_compute_pipe(ctx, spv_vk_ceiling_coopmat_i8, spv_vk_ceiling_coopmat_i8_bytes);
    }

    *out_ctx = ctx;
    return VELRI_OK;
}

void velri_vk_destroy(velri_vk_ctx *ctx) {
    if (!ctx) return;
    if (ctx->device) {
        if (ctx->queue) ctx->vkQueueWaitIdle(ctx->queue);
        if (ctx->splitk_buf) {
            velri_vk_buf_free(ctx, ctx->splitk_buf);
            ctx->splitk_buf = nullptr;
        }
        if (ctx->stage_map) ctx->vkUnmapMemory(ctx->device, ctx->stage_mem);
        if (ctx->stage_buf) ctx->vkDestroyBuffer(ctx->device, ctx->stage_buf, nullptr);
        if (ctx->stage_mem) ctx->vkFreeMemory(ctx->device, ctx->stage_mem, nullptr);
        for (int i = 0; i < PIPE_COUNT; ++i) {
            if (ctx->pipes[i]) ctx->vkDestroyPipeline(ctx->device, ctx->pipes[i], nullptr);
        }
        if (ctx->pipe_layout) ctx->vkDestroyPipelineLayout(ctx->device, ctx->pipe_layout, nullptr);
        if (ctx->query_pool) ctx->vkDestroyQueryPool(ctx->device, ctx->query_pool, nullptr);
        if (ctx->fence) ctx->vkDestroyFence(ctx->device, ctx->fence, nullptr);
        if (ctx->cmd_pool) ctx->vkDestroyCommandPool(ctx->device, ctx->cmd_pool, nullptr);
        ctx->vkDestroyDevice(ctx->device, nullptr);
    }
    if (ctx->instance) ctx->vkDestroyInstance(ctx->instance, nullptr);
    if (ctx->dl_handle) dlclose(ctx->dl_handle);
    free(ctx);
}

const velri_vk_info *velri_vk_get_info(const velri_vk_ctx *ctx) {
    return ctx ? &ctx->info : nullptr;
}

velri_status velri_vk_buf_alloc(velri_vk_ctx *ctx, size_t bytes, velri_vk_buf **out_buf) {
    if (!ctx || !out_buf) return VELRI_ERR_INVALID;
    *out_buf = nullptr;
    size_t alloc_bytes = bytes ? bytes : 16u;
    alloc_bytes = (alloc_bytes + 15u) & ~(size_t)15u;

    velri_vk_buf *b = calloc(1, sizeof(*b));
    if (!b) return VELRI_ERR_NOMEM;

    VkBufferCreateInfo b_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = (VkDeviceSize)alloc_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (ctx->vkCreateBuffer(ctx->device, &b_ci, nullptr, &b->buffer) != VK_SUCCESS) {
        free(b);
        return VELRI_ERR_NOMEM;
    }

    VkMemoryRequirements mr;
    ctx->vkGetBufferMemoryRequirements(ctx->device, b->buffer, &mr);
    uint32_t mtype = find_mem_type(&ctx->mem_props, mr.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mtype == UINT32_MAX) {
        ctx->vkDestroyBuffer(ctx->device, b->buffer, nullptr);
        free(b);
        return VELRI_ERR_NOMEM;
    }

    VkMemoryAllocateFlagsInfo flags_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo ma = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &flags_info,
        .allocationSize = mr.size,
        .memoryTypeIndex = mtype,
    };
    if (ctx->vkAllocateMemory(ctx->device, &ma, nullptr, &b->memory) != VK_SUCCESS) {
        ctx->vkDestroyBuffer(ctx->device, b->buffer, nullptr);
        free(b);
        return VELRI_ERR_NOMEM;
    }
    if (ctx->vkBindBufferMemory(ctx->device, b->buffer, b->memory, 0) != VK_SUCCESS) {
        ctx->vkFreeMemory(ctx->device, b->memory, nullptr);
        ctx->vkDestroyBuffer(ctx->device, b->buffer, nullptr);
        free(b);
        return VELRI_ERR_NOMEM;
    }

    VkBufferDeviceAddressInfo bda_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = b->buffer,
    };
    b->bda = ctx->vkGetBufferDeviceAddress(ctx->device, &bda_info);
    b->size = alloc_bytes;
    *out_buf = b;
    return VELRI_OK;
}

void velri_vk_buf_free(velri_vk_ctx *ctx, velri_vk_buf *buf) {
    if (!ctx || !buf) return;
    if (buf->buffer) ctx->vkDestroyBuffer(ctx->device, buf->buffer, nullptr);
    if (buf->memory) ctx->vkFreeMemory(ctx->device, buf->memory, nullptr);
    free(buf);
}

velri_status velri_vk_buf_upload(velri_vk_ctx *ctx, velri_vk_buf *dst, const void *src, size_t bytes) {
    if (!ctx || !dst || (!src && bytes > 0) || bytes > dst->size) return VELRI_ERR_INVALID;
    if (bytes == 0) return VELRI_OK;
    velri_status st = ensure_staging(ctx, bytes);
    if (st != VELRI_OK) return st;
    memcpy(ctx->stage_map, src, bytes);

    ctx->vkResetCommandBuffer(ctx->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    ctx->vkBeginCommandBuffer(ctx->cmd, &bi);
    VkBufferCopy region = {.srcOffset = 0, .dstOffset = 0, .size = (VkDeviceSize)bytes};
    ctx->vkCmdCopyBuffer(ctx->cmd, ctx->stage_buf, dst->buffer, 1, &region);
    ctx->vkEndCommandBuffer(ctx->cmd);
    return submit_and_wait(ctx);
}

velri_status velri_vk_buf_download(velri_vk_ctx *ctx, void *dst, const velri_vk_buf *src, size_t bytes) {
    if (!ctx || !src || (!dst && bytes > 0) || bytes > src->size) return VELRI_ERR_INVALID;
    if (bytes == 0) return VELRI_OK;
    velri_status st = ensure_staging(ctx, bytes);
    if (st != VELRI_OK) return st;

    ctx->vkResetCommandBuffer(ctx->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    ctx->vkBeginCommandBuffer(ctx->cmd, &bi);
    VkBufferCopy region = {.srcOffset = 0, .dstOffset = 0, .size = (VkDeviceSize)bytes};
    ctx->vkCmdCopyBuffer(ctx->cmd, src->buffer, ctx->stage_buf, 1, &region);
    ctx->vkEndCommandBuffer(ctx->cmd);
    st = submit_and_wait(ctx);
    if (st != VELRI_OK) return st;
    memcpy(dst, ctx->stage_map, bytes);
    return VELRI_OK;
}

static velri_status dispatch_timed_warm(velri_vk_ctx *ctx, VkPipeline pipe, const void *pc_data,
                                        uint32_t pc_size, uint32_t gx, uint32_t gy, uint32_t gz,
                                        uint32_t warm_iters, uint32_t iters, double *out_avg_ms) {
    if (!pipe) return VELRI_ERR_UNSUPPORTED;
    if (iters == 0) iters = 1;

    VkMemoryBarrier mbar = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };

    if (warm_iters > 0) {
        ctx->vkResetCommandBuffer(ctx->cmd, 0);
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        ctx->vkBeginCommandBuffer(ctx->cmd, &bi);
        ctx->vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        ctx->vkCmdPushConstants(ctx->cmd, ctx->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                pc_size, pc_data);
        for (uint32_t w = 0; w < warm_iters; ++w) {
            ctx->vkCmdDispatch(ctx->cmd, gx, gy, gz);
            if (w + 1u < warm_iters) {
                ctx->vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mbar, 0,
                                          nullptr, 0, nullptr);
            }
        }
        ctx->vkEndCommandBuffer(ctx->cmd);
        velri_status wst = submit_and_wait(ctx);
        if (wst != VELRI_OK) return wst;
    }

    ctx->vkResetCommandBuffer(ctx->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    ctx->vkBeginCommandBuffer(ctx->cmd, &bi);
    ctx->vkCmdResetQueryPool(ctx->cmd, ctx->query_pool, 0, 2);
    ctx->vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->query_pool, 0);
    ctx->vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    ctx->vkCmdPushConstants(ctx->cmd, ctx->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            pc_size, pc_data);
    for (uint32_t i = 0; i < iters; ++i) {
        ctx->vkCmdDispatch(ctx->cmd, gx, gy, gz);
        if (i + 1u < iters) {
            ctx->vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mbar, 0,
                                      nullptr, 0, nullptr);
        }
    }
    ctx->vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->query_pool, 1);
    ctx->vkEndCommandBuffer(ctx->cmd);

    velri_status st = submit_and_wait(ctx);
    if (st != VELRI_OK) return st;

    if (out_avg_ms) {
        uint64_t ts[2] = {0, 0};
        if (ctx->vkGetQueryPoolResults(ctx->device, ctx->query_pool, 0, 2, sizeof(ts), ts,
                                       sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) !=
            VK_SUCCESS) {
            return VELRI_ERR_INVALID;
        }
        double ns = (double)(ts[1] - ts[0]) * (double)ctx->info.timestamp_period_ns;
        *out_avg_ms = (ns / 1.0e6) / (double)iters;
    }
    return VELRI_OK;
}

static velri_status dispatch_timed(velri_vk_ctx *ctx, VkPipeline pipe, const void *pc_data,
                                   uint32_t pc_size, uint32_t gx, uint32_t gy, uint32_t gz,
                                   uint32_t iters, double *out_avg_ms) {
    uint32_t warm = (iters <= 1u) ? 0u : ((iters >= 10u) ? 8u : 2u);
    return dispatch_timed_warm(ctx, pipe, pc_data, pc_size, gx, gy, gz, warm, iters, out_avg_ms);
}

static inline uint32_t div_ceil_u32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static velri_status ensure_splitk_buf(velri_vk_ctx *ctx, size_t bytes) {
    if (ctx->splitk_buf && ctx->splitk_buf->size >= bytes) return VELRI_OK;
    if (ctx->splitk_buf) {
        velri_vk_buf_free(ctx, ctx->splitk_buf);
        ctx->splitk_buf = nullptr;
    }
    return velri_vk_buf_alloc(ctx, bytes, &ctx->splitk_buf);
}

static velri_status dispatch_splitk_timed(
    velri_vk_ctx *ctx, VkPipeline gemm_pipe, const vk_linear_args *gemm_args,
    uint32_t gx, uint32_t gy, uint32_t gz,
    const vk_splitk_reduce_args *red_args, uint32_t red_gx,
    uint32_t iters, double *out_avg_ms)
{
    VkPipeline red_pipe = ctx->pipes[PIPE_SPLITK_REDUCE];
    if (!gemm_pipe || !red_pipe) return VELRI_ERR_UNSUPPORTED;
    if (iters == 0) iters = 1;
    uint32_t warm_iters = (iters <= 1u) ? 0u : ((iters >= 10u) ? 8u : 2u);

    VkMemoryBarrier mbar = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };

    for (int phase = 0; phase < 2; ++phase) {
        uint32_t n_run = (phase == 0) ? warm_iters : iters;
        if (n_run == 0) continue;

        ctx->vkResetCommandBuffer(ctx->cmd, 0);
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        ctx->vkBeginCommandBuffer(ctx->cmd, &bi);
        if (phase == 1) {
            ctx->vkCmdResetQueryPool(ctx->cmd, ctx->query_pool, 0, 2);
            ctx->vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->query_pool, 0);
        }
        for (uint32_t i = 0; i < n_run; ++i) {
            ctx->vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gemm_pipe);
            ctx->vkCmdPushConstants(ctx->cmd, ctx->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    sizeof(*gemm_args), gemm_args);
            ctx->vkCmdDispatch(ctx->cmd, gx, gy, gz);
            ctx->vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mbar, 0,
                                      nullptr, 0, nullptr);
            ctx->vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, red_pipe);
            ctx->vkCmdPushConstants(ctx->cmd, ctx->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    sizeof(*red_args), red_args);
            ctx->vkCmdDispatch(ctx->cmd, red_gx, 1u, 1u);
            if (i + 1u < n_run) {
                ctx->vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mbar, 0,
                                          nullptr, 0, nullptr);
            }
        }
        if (phase == 1) {
            ctx->vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->query_pool, 1);
        }
        ctx->vkEndCommandBuffer(ctx->cmd);
        velri_status st = submit_and_wait(ctx);
        if (st != VELRI_OK) return st;
    }

    if (out_avg_ms) {
        uint64_t ts[2] = {0, 0};
        if (ctx->vkGetQueryPoolResults(ctx->device, ctx->query_pool, 0, 2, sizeof(ts), ts,
                                       sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) !=
            VK_SUCCESS) {
            return VELRI_ERR_INVALID;
        }
        double ns = (double)(ts[1] - ts[0]) * (double)ctx->info.timestamp_period_ns;
        *out_avg_ms = (ns / 1.0e6) / (double)iters;
    }
    return VELRI_OK;
}

static uint32_t choose_split_k(uint32_t rows, uint32_t m, uint32_t k, uint32_t tile_m, uint32_t tile_n,
                               uint32_t target_wgs, uint32_t bk_elems) {
    uint32_t wgs = div_ceil_u32(m, tile_n) * div_ceil_u32(rows, tile_m);
    if (rows < 16u || wgs == 0u || wgs >= target_wgs) return 1u;
    uint32_t k_tiles = div_ceil_u32(k, bk_elems);
    if (k_tiles < 16u) return 1u;
    uint32_t sk = target_wgs / wgs;
    if (sk > 8u) sk = 8u;
    if (sk > k_tiles / 4u) sk = k_tiles / 4u;
    if (sk < 2u) return 1u;
    uint32_t part_stride = (m + 3u) & ~3u;
    uint64_t part_bytes = (uint64_t)sk * (uint64_t)rows * (uint64_t)part_stride * 4u;
    if (part_bytes > 32u * 1024u * 1024u) return 1u;
    return sk;
}

velri_status velri_vk_linear(
    velri_vk_ctx *ctx, const velri_linear_desc *desc,
    uint32_t rows, uint32_t m, uint32_t k,
    const velri_vk_buf *x, uint32_t x_stride,
    const velri_vk_buf *w, uint32_t w_stride,
    const velri_vk_buf *bias, const velri_vk_buf *x_scale, const velri_vk_buf *w_scale,
    velri_vk_buf *y, uint32_t y_stride, uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !desc || !x || !w || !y || m == 0 || k == 0) return VELRI_ERR_INVALID;
    if (rows == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }

    uint32_t flags = (bias ? 1u : 0u) | (x_scale ? 2u : 0u) | (w_scale ? 4u : 0u);
    vk_linear_args args = {
        .x = x->bda,
        .w = w->bda,
        .bias = bias ? bias->bda : 0u,
        .x_scale = x_scale ? x_scale->bda : 0u,
        .w_scale = w_scale ? w_scale->bda : 0u,
        .y = y->bda,
        .rows = rows,
        .m = m,
        .k = k,
        .x_stride = x_stride,
        .w_stride = w_stride,
        .y_stride = y_stride,
        .flags = flags,
        .split_k = 1u,
    };

    uint32_t warps_per_wg = 128u / ctx->info.subgroup_size;
    VkPipeline pipe = VK_NULL_HANDLE;
    uint32_t gx = 1u, gy = 1u;
    uint32_t split_k = 1u;
    uint32_t is_int32 = 0u;

    if (desc->x_dtype == VELRI_DTYPE_F32 && desc->w_dtype == VELRI_DTYPE_F32 &&
        desc->y_dtype == VELRI_DTYPE_F32) {
        if (rows <= 2u) {
            pipe = ctx->pipes[PIPE_GEMV_F32];
            gx = div_ceil_u32(m, warps_per_wg);
            gy = rows;
        } else {
            pipe = ctx->pipes[PIPE_GEMM_F32];
            gx = div_ceil_u32(m, 128u);
            gy = div_ceil_u32(rows, 128u);
            split_k = choose_split_k(rows, m, k, 128u, 128u, 140u, 8u);
        }
    } else if (desc->x_dtype == VELRI_DTYPE_F16 && desc->w_dtype == VELRI_DTYPE_F16 &&
               desc->y_dtype == VELRI_DTYPE_F32) {
        if (rows <= 2u || !ctx->pipes[PIPE_GEMM_COOPMAT_F16]) {
            pipe = ctx->pipes[PIPE_GEMV_F16];
            gx = div_ceil_u32(m, warps_per_wg);
            gy = rows;
        } else if ((uint64_t)div_ceil_u32(m, 128u) * (uint64_t)div_ceil_u32(rows, 128u) < 140u) {
            pipe = ctx->pipes[PIPE_GEMM_COOPMAT_F16_64];
            gx = div_ceil_u32(m, 64u);
            gy = div_ceil_u32(rows, 64u);
        } else {
            pipe = ctx->pipes[PIPE_GEMM_COOPMAT_F16];
            gx = div_ceil_u32(m, 128u);
            gy = div_ceil_u32(rows, 128u);
        }
    } else if (desc->x_dtype == VELRI_DTYPE_BF16 && desc->w_dtype == VELRI_DTYPE_BF16 &&
               desc->y_dtype == VELRI_DTYPE_F32) {
        if (rows <= 2u || !ctx->pipes[PIPE_GEMM_COOPMAT_BF16]) {
            pipe = ctx->pipes[PIPE_GEMV_BF16];
            gx = div_ceil_u32(m, warps_per_wg);
            gy = rows;
        } else if ((uint64_t)div_ceil_u32(m, 128u) * (uint64_t)div_ceil_u32(rows, 128u) < 140u) {
            pipe = ctx->pipes[PIPE_GEMM_COOPMAT_BF16_64];
            gx = div_ceil_u32(m, 64u);
            gy = div_ceil_u32(rows, 64u);
        } else {
            pipe = ctx->pipes[PIPE_GEMM_COOPMAT_BF16];
            gx = div_ceil_u32(m, 128u);
            gy = div_ceil_u32(rows, 128u);
        }
    } else if (desc->x_dtype == VELRI_DTYPE_I8 && desc->w_dtype == VELRI_DTYPE_I8 &&
               desc->y_dtype == VELRI_DTYPE_F32) {
        if (rows <= 2u || !ctx->pipes[PIPE_GEMM_COOPMAT_W8A8]) {
            pipe = ctx->pipes[PIPE_GEMV_W8A8];
            gx = div_ceil_u32(m, warps_per_wg);
            gy = rows;
        } else if ((uint64_t)div_ceil_u32(m, 128u) * (uint64_t)div_ceil_u32(rows, 128u) < 140u) {
            pipe = ctx->pipes[PIPE_GEMM_COOPMAT_W8A8_64];
            gx = div_ceil_u32(m, 64u);
            gy = div_ceil_u32(rows, 64u);
        } else {
            pipe = ctx->pipes[PIPE_GEMM_COOPMAT_W8A8];
            gx = div_ceil_u32(m, 128u);
            gy = div_ceil_u32(rows, 128u);
        }
    } else {
        return VELRI_ERR_UNSUPPORTED;
    }

    if (split_k > 1u && ctx->pipes[PIPE_SPLITK_REDUCE] != VK_NULL_HANDLE) {
        uint32_t part_stride = (m + 3u) & ~3u;
        size_t part_bytes = (size_t)split_k * (size_t)rows * (size_t)part_stride * sizeof(float);
        velri_status st = ensure_splitk_buf(ctx, part_bytes);
        if (st != VELRI_OK) return st;

        args.y = ctx->splitk_buf->bda;
        args.y_stride = part_stride;
        args.split_k = split_k;

        vk_splitk_reduce_args rargs = {
            .partials = ctx->splitk_buf->bda,
            .bias = bias ? bias->bda : 0u,
            .x_scale = x_scale ? x_scale->bda : 0u,
            .w_scale = w_scale ? w_scale->bda : 0u,
            .y = y->bda,
            .rows = rows,
            .m = m,
            .part_stride_u4 = part_stride >> 2u,
            .y_stride = y_stride,
            .split_k = split_k,
            .flags = flags,
            .is_int32 = is_int32,
            .pad = 0u,
        };
        uint32_t red_gx = div_ceil_u32(rows * (part_stride >> 2u), 256u);
        return dispatch_splitk_timed(ctx, pipe, &args, gx, gy, split_k, &rargs, red_gx, iters,
                                     out_avg_ms);
    }

    return dispatch_timed(ctx, pipe, &args, sizeof(args), gx, gy, 1u, iters, out_avg_ms);
}

velri_status velri_vk_rmsnorm(
    velri_vk_ctx *ctx, velri_dtype io_dtype,
    uint32_t rows, uint32_t dim, float eps, bool unit_offset,
    const velri_vk_buf *x, uint32_t x_stride, const velri_vk_buf *weight,
    velri_vk_buf *y, uint32_t y_stride, uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !x || !y || dim == 0) return VELRI_ERR_INVALID;
    if (rows == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    VkPipeline pipe = VK_NULL_HANDLE;
    if (io_dtype == VELRI_DTYPE_F32) pipe = ctx->pipes[PIPE_RMSNORM_F32];
    else if (io_dtype == VELRI_DTYPE_BF16) pipe = ctx->pipes[PIPE_RMSNORM_BF16];
    else if (io_dtype == VELRI_DTYPE_F16) pipe = ctx->pipes[PIPE_RMSNORM_F16];
    else return VELRI_ERR_UNSUPPORTED;

    vk_rmsnorm_args args = {
        .x = x->bda,
        .w = weight ? weight->bda : 0u,
        .y = y->bda,
        .rows = rows,
        .dim = dim,
        .x_stride = x_stride,
        .y_stride = y_stride,
        .eps = eps,
        .has_weight = weight ? 1u : 0u,
        .unit_offset = unit_offset ? 1u : 0u,
        .pad = 0u,
    };
    return dispatch_timed(ctx, pipe, &args, sizeof(args), rows, 1u, 1u, iters, out_avg_ms);
}

velri_status velri_vk_fused_add_rmsnorm_f32(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t dim, float eps, bool unit_offset,
    velri_vk_buf *residual_inout, uint32_t r_stride,
    const velri_vk_buf *x, uint32_t x_stride,
    const velri_vk_buf *weight,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !residual_inout || !x || !y || dim == 0) return VELRI_ERR_INVALID;
    if (rows == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    vk_fused_add_rmsnorm_args args = {
        .residual_inout = residual_inout->bda,
        .x = x->bda,
        .w = weight ? weight->bda : 0u,
        .y = y->bda,
        .rows = rows,
        .dim = dim,
        .r_stride = r_stride,
        .x_stride = x_stride,
        .y_stride = y_stride,
        .eps = eps,
        .has_weight = weight ? 1u : 0u,
        .unit_offset = unit_offset ? 1u : 0u,
    };
    return dispatch_timed(ctx, ctx->pipes[PIPE_FUSED_ADD_RMSNORM_F32], &args, sizeof(args),
                          rows, 1u, 1u, iters, out_avg_ms);
}

velri_status velri_vk_layernorm_f32(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t dim, float eps,
    const velri_vk_buf *x, uint32_t x_stride,
    const velri_vk_buf *weight, const velri_vk_buf *bias,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !x || !y || dim == 0) return VELRI_ERR_INVALID;
    if (rows == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    vk_layernorm_args args = {
        .x = x->bda,
        .w = weight ? weight->bda : 0u,
        .b = bias ? bias->bda : 0u,
        .y = y->bda,
        .rows = rows,
        .dim = dim,
        .x_stride = x_stride,
        .y_stride = y_stride,
        .eps = eps,
        .has_weight = weight ? 1u : 0u,
        .has_bias = bias ? 1u : 0u,
        .pad = 0u,
    };
    return dispatch_timed(ctx, ctx->pipes[PIPE_LAYERNORM_F32], &args, sizeof(args),
                          rows, 1u, 1u, iters, out_avg_ms);
}

velri_status velri_vk_rope(
    velri_vk_ctx *ctx, velri_dtype dtype,
    uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim, uint32_t rotary_dim,
    uint32_t pos_offset,
    const velri_vk_buf *x, uint32_t tok_sx, uint32_t head_sx,
    const velri_vk_buf *cos_tbl, const velri_vk_buf *sin_tbl,
    velri_vk_buf *y, uint32_t tok_sy, uint32_t head_sy,
    velri_rope_mode mode, uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !x || !cos_tbl || !sin_tbl || !y) return VELRI_ERR_INVALID;
    if (n_tokens == 0 || n_heads == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    VkPipeline pipe = VK_NULL_HANDLE;
    if (dtype == VELRI_DTYPE_F32) pipe = ctx->pipes[PIPE_ROPE_F32];
    else if (dtype == VELRI_DTYPE_BF16) pipe = ctx->pipes[PIPE_ROPE_BF16];
    else return VELRI_ERR_UNSUPPORTED;

    uint32_t v_elems = (dtype == VELRI_DTYPE_F32) ? 4u : 8u;
    uint32_t half_rot = rotary_dim >> 1u;
    bool flat_fast = (mode == VELRI_ROPE_HALF) && (rotary_dim == head_dim) &&
                     (half_rot > 0u) && ((half_rot & (v_elems - 1u)) == 0u) &&
                     ((tok_sx & (v_elems - 1u)) == 0u) && ((head_sx & (v_elems - 1u)) == 0u) &&
                     ((tok_sy & (v_elems - 1u)) == 0u) && ((head_sy & (v_elems - 1u)) == 0u) &&
                     ((half_rot & 3u) == 0u) && ((x->bda & 15u) == 0u) && ((y->bda & 15u) == 0u) &&
                     ((cos_tbl->bda & 15u) == 0u) && ((sin_tbl->bda & 15u) == 0u);
    vk_rope_args args = {
        .x = x->bda,
        .cos_tbl = cos_tbl->bda,
        .sin_tbl = sin_tbl->bda,
        .y = y->bda,
        .n_tokens = n_tokens,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .rotary_dim = rotary_dim,
        .pos_offset = pos_offset,
        .tok_sx = tok_sx,
        .head_sx = head_sx,
        .tok_sy = tok_sy,
        .head_sy = head_sy,
        .mode = (uint32_t)mode | (flat_fast ? 2u : 0u),
    };
    if (flat_fast) {
        uint64_t total_vec = (uint64_t)n_tokens * (uint64_t)n_heads * (uint64_t)(half_rot / v_elems);
        uint32_t gx = (uint32_t)((total_vec + 255u) / 256u);
        return dispatch_timed(ctx, pipe, &args, sizeof(args), gx, 1u, 1u, iters, out_avg_ms);
    }
    /* General path: 1 warp per head, 8 heads per 256-thread WG. */
    return dispatch_timed(ctx, pipe, &args, sizeof(args), div_ceil_u32(n_heads, 8u), n_tokens, 1u,
                          iters, out_avg_ms);
}

velri_status velri_vk_activation_f32(
    velri_vk_ctx *ctx, velri_act_kind kind, uint32_t count,
    const velri_vk_buf *x, velri_vk_buf *y,
    uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !x || !y) return VELRI_ERR_INVALID;
    if (count == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    VkPipeline pipe = VK_NULL_HANDLE;
    if (kind == VELRI_ACT_SILU) pipe = ctx->pipes[PIPE_ACT_SILU_F32];
    else if (kind == VELRI_ACT_GELU_TANH) pipe = ctx->pipes[PIPE_ACT_GELU_F32];
    else if (kind == VELRI_ACT_RELU2) pipe = ctx->pipes[PIPE_ACT_RELU2_F32];
    else return VELRI_ERR_UNSUPPORTED;

    vk_activation_args args = {.x = x->bda, .y = y->bda, .count = count, .pad = 0u};
    uint32_t n_vec = div_ceil_u32(count, 4u);
    return dispatch_timed(ctx, pipe, &args, sizeof(args), div_ceil_u32(n_vec, 256u), 1u, 1u,
                          iters, out_avg_ms);
}

velri_status velri_vk_gated_mlp_f32(
    velri_vk_ctx *ctx, velri_act_kind act, uint32_t count, float clamp_limit,
    const velri_vk_buf *gate, const velri_vk_buf *up, velri_vk_buf *y,
    uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !gate || !up || !y) return VELRI_ERR_INVALID;
    if (count == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    VkPipeline pipe = VK_NULL_HANDLE;
    if (act == VELRI_ACT_SILU) pipe = ctx->pipes[PIPE_GATED_SILU_F32];
    else if (act == VELRI_ACT_GELU_TANH) pipe = ctx->pipes[PIPE_GATED_GELU_F32];
    else return VELRI_ERR_UNSUPPORTED;

    vk_gated_mlp_args args = {
        .gate = gate->bda,
        .up = up->bda,
        .y = y->bda,
        .count = count,
        .clamp_limit = clamp_limit,
    };
    uint32_t n_vec = div_ceil_u32(count, 4u);
    return dispatch_timed(ctx, pipe, &args, sizeof(args), div_ceil_u32(n_vec, 256u), 1u, 1u,
                          iters, out_avg_ms);
}

velri_status velri_vk_vec_axpby_f32(
    velri_vk_ctx *ctx, uint32_t count, float alpha, const velri_vk_buf *a,
    float beta, const velri_vk_buf *b, velri_vk_buf *y,
    uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !a || !y || (beta != 0.0f && !b)) return VELRI_ERR_INVALID;
    if (count == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    vk_vec_axpby_args args = {
        .a = a->bda,
        .b = b ? b->bda : 0u,
        .y = y->bda,
        .count = count,
        .alpha = alpha,
        .beta = beta,
        .has_b = (b && beta != 0.0f) ? 1u : 0u,
        .pad = 0u,
    };
    uint32_t n_vec = div_ceil_u32(count, 4u);
    return dispatch_timed(ctx, ctx->pipes[PIPE_VEC_AXPBY_F32], &args, sizeof(args),
                          div_ceil_u32(n_vec, 256u), 1u, 1u, iters, out_avg_ms);
}

velri_status velri_vk_vec_mul_f32(
    velri_vk_ctx *ctx, uint32_t count, const velri_vk_buf *a,
    const velri_vk_buf *b, velri_vk_buf *y,
    uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !a || !b || !y) return VELRI_ERR_INVALID;
    if (count == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    vk_vec_mul_args args = {.a = a->bda, .b = b->bda, .y = y->bda, .count = count, .pad = 0u};
    uint32_t n_vec = div_ceil_u32(count, 4u);
    return dispatch_timed(ctx, ctx->pipes[PIPE_VEC_MUL_F32], &args, sizeof(args),
                          div_ceil_u32(n_vec, 256u), 1u, 1u, iters, out_avg_ms);
}

velri_status velri_vk_softmax_f32(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t cols,
    const velri_vk_buf *x, uint32_t x_stride,
    velri_vk_buf *y, uint32_t y_stride, uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !x || !y || cols == 0) return VELRI_ERR_INVALID;
    if (rows == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    vk_softmax_args args = {
        .x = x->bda,
        .y = y->bda,
        .rows = rows,
        .cols = cols,
        .x_stride = x_stride,
        .y_stride = y_stride,
    };
    return dispatch_timed(ctx, ctx->pipes[PIPE_SOFTMAX_F32], &args, sizeof(args), rows, 1u, 1u,
                          iters, out_avg_ms);
}

velri_status velri_vk_fwht_f32(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t cols, uint32_t block_size,
    const velri_vk_buf *x, uint32_t x_stride,
    velri_dtype suh_dtype, const velri_vk_buf *suh,
    velri_vk_buf *y, uint32_t y_stride,
    uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !x || !y || cols == 0) return VELRI_ERR_INVALID;
    if (block_size != 16u && block_size != 32u && block_size != 64u &&
        block_size != 128u && block_size != 256u) {
        return VELRI_ERR_INVALID;
    }
    if ((cols % block_size) != 0u) return VELRI_ERR_INVALID;
    if (rows == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    uint32_t suh_mode = 0u;
    if (suh) {
        if (suh_dtype == VELRI_DTYPE_F32) suh_mode = 1u;
        else if (suh_dtype == VELRI_DTYPE_F16) suh_mode = 2u;
        else return VELRI_ERR_UNSUPPORTED;
    }
    vk_fwht_args args = {
        .x = x->bda,
        .suh = suh ? suh->bda : 0u,
        .y = y->bda,
        .rows = rows,
        .cols = cols,
        .block_size = block_size,
        .x_stride = x_stride,
        .y_stride = y_stride,
        .suh_mode = suh_mode,
        .inv_sqrt_b = 1.0f / sqrtf((float)block_size),
        .pad = 0u,
    };
    return dispatch_timed(ctx, ctx->pipes[PIPE_FWHT_F32], &args, sizeof(args),
                          div_ceil_u32(cols, 1024u), rows, 1u, iters, out_avg_ms);
}

velri_status velri_vk_quantize_row_f32_i8(
    velri_vk_ctx *ctx, uint32_t rows, uint32_t k,
    const velri_vk_buf *x, uint32_t x_stride,
    velri_vk_buf *x_i8, uint32_t xi8_stride,
    velri_vk_buf *scale_out,
    uint32_t iters, double *out_avg_ms)
{
    if (!ctx || !x || !x_i8 || !scale_out || k == 0) return VELRI_ERR_INVALID;
    if (rows == 0) {
        if (out_avg_ms) *out_avg_ms = 0.0;
        return VELRI_OK;
    }
    vk_quantize_row_args args = {
        .x = x->bda,
        .x_i8 = x_i8->bda,
        .scale_out = scale_out->bda,
        .rows = rows,
        .k = k,
        .x_stride = x_stride,
        .xi8_stride = xi8_stride,
    };
    return dispatch_timed(ctx, ctx->pipes[PIPE_QUANTIZE_ROW_F32_I8], &args, sizeof(args),
                          rows, 1u, 1u, iters, out_avg_ms);
}

velri_status velri_vk_measure_bw_gbps(velri_vk_ctx *ctx, size_t copy_bytes, uint32_t iters,
                                      double *out_gbps)
{
    if (!ctx || !out_gbps || copy_bytes < 16u) return VELRI_ERR_INVALID;
    uint32_t count_vec4 = (uint32_t)(copy_bytes / 16u);
    size_t exact_bytes = (size_t)count_vec4 * 16u;

    velri_vk_buf *src = nullptr;
    velri_vk_buf *dst = nullptr;
    velri_status st = velri_vk_buf_alloc(ctx, exact_bytes, &src);
    if (st != VELRI_OK) return st;
    st = velri_vk_buf_alloc(ctx, exact_bytes, &dst);
    if (st != VELRI_OK) {
        velri_vk_buf_free(ctx, src);
        return st;
    }

    /* Touch both buffers on GPU so physical pages are committed and non-zero. */
    ctx->vkResetCommandBuffer(ctx->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    ctx->vkBeginCommandBuffer(ctx->cmd, &bi);
    ctx->vkCmdFillBuffer(ctx->cmd, src->buffer, 0, (VkDeviceSize)exact_bytes, 0x3F800000u);
    ctx->vkCmdFillBuffer(ctx->cmd, dst->buffer, 0, (VkDeviceSize)exact_bytes, 0u);
    ctx->vkEndCommandBuffer(ctx->cmd);
    st = submit_and_wait(ctx);

    if (st == VELRI_OK) {
        vk_ceiling_bw_args args = {
            .src = src->bda,
            .dst = dst->bda,
            .count_vec4 = count_vec4,
            .pad = 0u,
        };
        double avg_ms = 0.0;
        st = dispatch_timed_warm(ctx, ctx->pipes[PIPE_CEIL_BW], &args, sizeof(args),
                                 div_ceil_u32(count_vec4, 1024u), 1u, 1u, 120u,
                                 iters ? iters : 20u, &avg_ms);
        if (st == VELRI_OK && avg_ms > 0.0) {
            *out_gbps = (2.0 * (double)exact_bytes) / (avg_ms * 1.0e-3) / 1.0e9;
        }
    }

    velri_vk_buf_free(ctx, dst);
    velri_vk_buf_free(ctx, src);
    return st;
}

velri_status velri_vk_measure_bw_read_gbps(velri_vk_ctx *ctx, size_t read_bytes, uint32_t iters,
                                           double *out_gbps)
{
    if (!ctx || !out_gbps || read_bytes < 16u) return VELRI_ERR_INVALID;
    uint32_t count_vec4 = (uint32_t)(read_bytes / 16u);
    size_t exact_bytes = (size_t)count_vec4 * 16u;

    velri_vk_buf *src = nullptr;
    velri_vk_buf *sink = nullptr;
    velri_status st = velri_vk_buf_alloc(ctx, exact_bytes, &src);
    if (st != VELRI_OK) return st;
    st = velri_vk_buf_alloc(ctx, 2048u * 8u * sizeof(float), &sink);
    if (st != VELRI_OK) {
        velri_vk_buf_free(ctx, src);
        return st;
    }

    ctx->vkResetCommandBuffer(ctx->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    ctx->vkBeginCommandBuffer(ctx->cmd, &bi);
    ctx->vkCmdFillBuffer(ctx->cmd, src->buffer, 0, (VkDeviceSize)exact_bytes, 0x3F800000u);
    ctx->vkEndCommandBuffer(ctx->cmd);
    st = submit_and_wait(ctx);

    if (st == VELRI_OK) {
        vk_ceiling_bw_args args = {
            .src = src->bda,
            .dst = sink->bda,
            .count_vec4 = count_vec4,
            .pad = 0u,
        };
        double best_gbps = 0.0;
        for (int attempt = 0; attempt < 2 && st == VELRI_OK; ++attempt) {
            double avg_ms = 0.0;
            st = dispatch_timed_warm(ctx, ctx->pipes[PIPE_CEIL_BW_READ], &args, sizeof(args),
                                     2048u, 1u, 1u, 160u, iters ? iters : 20u, &avg_ms);
            if (st == VELRI_OK && avg_ms > 0.0) {
                double gbps = (double)exact_bytes / (avg_ms * 1.0e-3) / 1.0e9;
                if (gbps > best_gbps) best_gbps = gbps;
            }
        }
        if (st == VELRI_OK && best_gbps > 0.0) {
            *out_gbps = best_gbps;
        }
    }

    velri_vk_buf_free(ctx, sink);
    velri_vk_buf_free(ctx, src);
    return st;
}

velri_status velri_vk_measure_fma_tflops(velri_vk_ctx *ctx, uint32_t workgroups, uint32_t loop_iters,
                                         double *out_tflops)
{
    if (!ctx || !out_tflops || workgroups == 0 || loop_iters == 0) return VELRI_ERR_INVALID;
    velri_vk_buf *sink = nullptr;
    velri_status st = velri_vk_buf_alloc(ctx, 4096u * sizeof(float), &sink);
    if (st != VELRI_OK) return st;

    vk_ceiling_compute_args args = {
        .sink = sink->bda,
        .iters = loop_iters,
        .seed = 0.5f,
    };
    double avg_ms = 0.0;
    st = dispatch_timed(ctx, ctx->pipes[PIPE_CEIL_FMA_F32], &args, sizeof(args),
                        workgroups, 1u, 1u, 5u, &avg_ms);
    if (st == VELRI_OK && avg_ms > 0.0) {
        /* 256 threads/WG * loop_iters * 16 unroll * 4 vec4 * 4 lanes * 2 FLOPs/FMA */
        double flops = (double)workgroups * 256.0 * (double)loop_iters * 512.0;
        *out_tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;
    }
    velri_vk_buf_free(ctx, sink);
    return st;
}

velri_status velri_vk_measure_coopmat_peak(velri_vk_ctx *ctx, velri_dtype dtype,
                                           uint32_t workgroups, uint32_t loop_iters,
                                           double *out_tops)
{
    if (!ctx || !out_tops || workgroups == 0 || loop_iters == 0) return VELRI_ERR_INVALID;
    VkPipeline pipe = VK_NULL_HANDLE;
    double k_step = 16.0;
    if (dtype == VELRI_DTYPE_F16) {
        pipe = ctx->pipes[PIPE_CEIL_COOPMAT_F16];
        k_step = 16.0;
    } else if (dtype == VELRI_DTYPE_BF16) {
        pipe = ctx->pipes[PIPE_CEIL_COOPMAT_BF16];
        k_step = 16.0;
    } else if (dtype == VELRI_DTYPE_I8) {
        pipe = ctx->pipes[PIPE_CEIL_COOPMAT_I8];
        k_step = 32.0;
    } else {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (!pipe) return VELRI_ERR_UNSUPPORTED;

    /* 16 warp slots * 8 tiles * 256 elements * 4 bytes = 128 KiB sink ring. */
    velri_vk_buf *sink = nullptr;
    velri_status st = velri_vk_buf_alloc(ctx, 16u * 8u * 256u * sizeof(float), &sink);
    if (st != VELRI_OK) return st;

    vk_ceiling_compute_args args = {
        .sink = sink->bda,
        .iters = loop_iters,
        .seed = 1.0f,
    };
    double avg_ms = 0.0;
    st = dispatch_timed(ctx, pipe, &args, sizeof(args), workgroups, 1u, 1u, 5u, &avg_ms);
    if (st == VELRI_OK && avg_ms > 0.0) {
        double warps_per_wg = 256.0 / (double)ctx->info.subgroup_size;
        double ops_per_mma = 2.0 * 16.0 * 16.0 * k_step;
        double total_ops = (double)workgroups * warps_per_wg * (double)loop_iters * 8.0 * ops_per_mma;
        *out_tops = total_ops / (avg_ms * 1.0e-3) / 1.0e12;
    }
    velri_vk_buf_free(ctx, sink);
    return st;
}
