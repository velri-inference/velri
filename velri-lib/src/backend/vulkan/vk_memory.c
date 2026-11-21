#include "vk_internal.h"

#include <stdlib.h>
#include <string.h>

/*
 * Vulkan Device-Local Buffer Allocation, Bounded Staging Transfers, and Buffer Zeroing.
 */

static uint32_t find_mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t type_bits,
                              VkMemoryPropertyFlags req) {
    for (uint32_t i = 0U; i < mp->memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) && ((mp->memoryTypes[i].propertyFlags & req) == req)) {
            return i;
        }
    }
    return UINT32_MAX;
}

velri_status velri_vk_alloc_buf_locked(velri_backend_device *dev, size_t bytes, size_t align,
                                       vk_buf **out_buf) {
    *out_buf = nullptr;
    size_t eff_align = 0U, alloc_bytes = 0U;
    velri_status st =
        velri_calc_alloc_bytes(bytes, align, VK_MIN_BUF_ALIGN_BYTES, &eff_align, &alloc_bytes);
    if (st != VELRI_OK) {
        return st;
    }
    (void)eff_align;

    vk_buf *b = (vk_buf *)calloc(1U, sizeof(*b));
    if (!b) {
        return VELRI_ERR_NOMEM;
    }
    VkBufferCreateInfo b_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = (VkDeviceSize)alloc_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (dev->vkCreateBuffer(dev->device, &b_ci, nullptr, &b->buffer) != VK_SUCCESS) {
        free(b);
        return VELRI_ERR_NOMEM;
    }

    VkMemoryRequirements mr;
    dev->vkGetBufferMemoryRequirements(dev->device, b->buffer, &mr);
    uint32_t mtype = find_mem_type(&dev->mem_props, mr.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mtype == UINT32_MAX) {
        dev->vkDestroyBuffer(dev->device, b->buffer, nullptr);
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
    if (dev->vkAllocateMemory(dev->device, &ma, nullptr, &b->memory) != VK_SUCCESS) {
        dev->vkDestroyBuffer(dev->device, b->buffer, nullptr);
        free(b);
        return VELRI_ERR_NOMEM;
    }
    if (dev->vkBindBufferMemory(dev->device, b->buffer, b->memory, 0) != VK_SUCCESS) {
        dev->vkFreeMemory(dev->device, b->memory, nullptr);
        dev->vkDestroyBuffer(dev->device, b->buffer, nullptr);
        free(b);
        return VELRI_ERR_NOMEM;
    }

    VkBufferDeviceAddressInfo bda_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = b->buffer,
    };
    b->bda = dev->vkGetBufferDeviceAddress(dev->device, &bda_info);
    b->size = alloc_bytes;
    *out_buf = b;
    return VELRI_OK;
}

void velri_vk_free_buf_locked(velri_backend_device *dev, vk_buf *b) {
    if (!b) return;
    if (b->buffer) dev->vkDestroyBuffer(dev->device, b->buffer, nullptr);
    if (b->memory) dev->vkFreeMemory(dev->device, b->memory, nullptr);
    free(b);
}

velri_status velri_vk_init_staging_buffer(velri_backend_device *dev) {
    static const size_t k_sizes[] = {
        VK_STAGE_MAX_BYTES, 4U * 1024U * 1024U, 1U * 1024U * 1024U, 256U * 1024U
    };
    for (size_t i = 0U; i < sizeof(k_sizes) / sizeof(k_sizes[0]); ++i) {
        size_t sz = k_sizes[i];
        VkBufferCreateInfo b_ci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = (VkDeviceSize)sz,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (dev->vkCreateBuffer(dev->device, &b_ci, nullptr, &dev->stage_buf) != VK_SUCCESS) {
            continue;
        }
        VkMemoryRequirements mr;
        dev->vkGetBufferMemoryRequirements(dev->device, dev->stage_buf, &mr);
        uint32_t mtype = find_mem_type(&dev->mem_props, mr.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (mtype == UINT32_MAX) {
            dev->vkDestroyBuffer(dev->device, dev->stage_buf, nullptr);
            dev->stage_buf = VK_NULL_HANDLE;
            return VELRI_ERR_UNSUPPORTED;
        }
        VkMemoryAllocateInfo ma = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mr.size,
            .memoryTypeIndex = mtype,
        };
        if (dev->vkAllocateMemory(dev->device, &ma, nullptr, &dev->stage_mem) == VK_SUCCESS &&
            dev->vkBindBufferMemory(dev->device, dev->stage_buf, dev->stage_mem, 0) == VK_SUCCESS &&
            dev->vkMapMemory(dev->device, dev->stage_mem, 0, mr.size, 0, &dev->stage_map) ==
                VK_SUCCESS) {
            dev->stage_cap = sz;
            return VELRI_OK;
        }
        if (dev->stage_mem) {
            dev->vkFreeMemory(dev->device, dev->stage_mem, nullptr);
            dev->stage_mem = VK_NULL_HANDLE;
        }
        dev->vkDestroyBuffer(dev->device, dev->stage_buf, nullptr);
        dev->stage_buf = VK_NULL_HANDLE;
    }
    return VELRI_ERR_NOMEM;
}

static velri_status xfer_begin_locked(velri_backend_device *dev) {
    if (dev->vkResetCommandBuffer(dev->xfer_cmd, 0U) != VK_SUCCESS) {
        return VELRI_ERR_EXEC;
    }
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    return (dev->vkBeginCommandBuffer(dev->xfer_cmd, &bi) == VK_SUCCESS) ? VELRI_OK
                                                                         : VELRI_ERR_EXEC;
}

static velri_status xfer_end_submit_locked(velri_backend_device *dev) {
    if (dev->vkEndCommandBuffer(dev->xfer_cmd) != VK_SUCCESS ||
        dev->vkResetFences(dev->device, 1U, &dev->xfer_fence) != VK_SUCCESS) {
        return VELRI_ERR_EXEC;
    }
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1U,
        .pCommandBuffers = &dev->xfer_cmd,
    };
    if (dev->vkQueueSubmit(dev->queue, 1U, &si, dev->xfer_fence) != VK_SUCCESS ||
        dev->vkWaitForFences(dev->device, 1U, &dev->xfer_fence, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS) {
        return VELRI_ERR_EXEC;
    }
    return VELRI_OK;
}

/* ---- 3. Memory Backend Contract ---- */

velri_status velri_vk_alloc(velri_backend_device *dev, size_t bytes, size_t align,
                            void **out_mem_handle) {
    if (!dev || !out_mem_handle) {
        return VELRI_ERR_INVALID;
    }
    vk_buf *buf = nullptr;
    velri_os_mutex_lock(dev->mtx);
    velri_status st = velri_vk_alloc_buf_locked(dev, bytes, align, &buf);
    velri_os_mutex_unlock(dev->mtx);
    if (st != VELRI_OK) {
        return st;
    }
    *out_mem_handle = buf;
    return VELRI_OK;
}

void velri_vk_free(velri_backend_device *dev, void *mem_handle) {
    if (!dev || !mem_handle) {
        return;
    }
    velri_os_mutex_lock(dev->mtx);
    velri_vk_free_buf_locked(dev, (vk_buf *)mem_handle);
    velri_os_mutex_unlock(dev->mtx);
}

static velri_status vk_copy_ram_impl(velri_backend_device *dev,
                                     const velri_buffer_binding *dev_buf, void *dst_ram,
                                     const void *src_ram, size_t bytes, bool to_device) {
    if (!dev || !dev_buf || !dev_buf->mem_handle ||
        (bytes > 0U && (to_device ? !src_ram : !dst_ram))) {
        return VELRI_ERR_INVALID;
    }
    const vk_buf *vb = (const vk_buf *)dev_buf->mem_handle;
    if (dev_buf->byte_offset > vb->size || bytes > vb->size - dev_buf->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes == 0U) {
        return VELRI_OK;
    }

    velri_os_mutex_lock(dev->mtx);
    velri_status st = VELRI_OK;
    size_t done = 0U;
    while (done < bytes) {
        size_t chunk = bytes - done;
        if (chunk > dev->stage_cap) {
            chunk = dev->stage_cap;
        }
        if (to_device) {
            memcpy(dev->stage_map, (const uint8_t *)src_ram + done, chunk);
        }
        st = xfer_begin_locked(dev);
        if (st != VELRI_OK) {
            break;
        }
        VkBufferCopy region = {
            .srcOffset = to_device ? 0U : (VkDeviceSize)(dev_buf->byte_offset + done),
            .dstOffset = to_device ? (VkDeviceSize)(dev_buf->byte_offset + done) : 0U,
            .size = (VkDeviceSize)chunk,
        };
        dev->vkCmdCopyBuffer(dev->xfer_cmd, to_device ? dev->stage_buf : vb->buffer,
                             to_device ? vb->buffer : dev->stage_buf, 1U, &region);
        st = xfer_end_submit_locked(dev);
        if (st != VELRI_OK) {
            break;
        }
        if (!to_device) {
            memcpy((uint8_t *)dst_ram + done, dev->stage_map, chunk);
        }
        done += chunk;
    }
    velri_os_mutex_unlock(dev->mtx);
    return st;
}

velri_status velri_vk_copy_from_ram(velri_backend_device *dev, const velri_buffer_binding *dst,
                                    const void *src_ram, size_t bytes) {
    return vk_copy_ram_impl(dev, dst, nullptr, src_ram, bytes, true);
}

velri_status velri_vk_copy_to_ram(velri_backend_device *dev, void *dst_ram,
                                  const velri_buffer_binding *src, size_t bytes) {
    return vk_copy_ram_impl(dev, src, dst_ram, nullptr, bytes, false);
}

velri_status velri_vk_zero_mem(velri_backend_device *dev, void *mem_handle, size_t bytes) {
    if (!dev || !mem_handle) {
        return VELRI_ERR_INVALID;
    }
    const vk_buf *vb = (const vk_buf *)mem_handle;
    if (bytes > vb->size) {
        return VELRI_ERR_INVALID;
    }
    if (bytes == 0U) {
        return VELRI_OK;
    }
    /* `vk_buf` size is always a multiple of 16 bytes; round fill size up to 4 bytes safely. */
    size_t fill_sz = 0U;
    if (velri_align_up_overflow(bytes, 4U, &fill_sz)) {
        return VELRI_ERR_OVERFLOW;
    }
    if (fill_sz > vb->size) {
        fill_sz = vb->size;
    }

    velri_os_mutex_lock(dev->mtx);
    velri_status st = xfer_begin_locked(dev);
    if (st == VELRI_OK) {
        dev->vkCmdFillBuffer(dev->xfer_cmd, vb->buffer, 0U, (VkDeviceSize)fill_sz, 0U);
        st = xfer_end_submit_locked(dev);
    }
    velri_os_mutex_unlock(dev->mtx);
    return st;
}
