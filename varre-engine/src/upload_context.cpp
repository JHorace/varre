/**
 * @file upload_context.cpp
 * @brief Buffer upload primitives with staging and transfer-queue support.
 */
#include "varre/engine/upload_context.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include "varre/engine/engine.hpp"

namespace varre::engine {
  namespace {
    /**
     * @brief Append queue resources only when queue family is not present yet.
     * @param queue_resources Destination queue resource list.
     * @param family_index Queue family index.
     * @param queue Queue handle.
     * @param device Logical device.
     */
    void append_unique_queue_resources(
        std::vector<UploadContext::QueueResources> *queue_resources,
        const std::uint32_t family_index,
        const vk::Queue queue,
        const vk::raii::Device &device
    ) {
      const bool exists = std::ranges::any_of(
          *queue_resources,
          [&](const UploadContext::QueueResources &resource) { return resource.family_index == family_index; }
      );
      if (exists || queue == VK_NULL_HANDLE) {
        return;
      }

      const vk::CommandPoolCreateInfo command_pool_info = vk::CommandPoolCreateInfo{}
          .setFlags(vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
          .setQueueFamilyIndex(family_index);
      queue_resources->push_back(
          UploadContext::QueueResources{
            .family_index = family_index,
            .queue = queue,
            .command_pool = vk::raii::CommandPool(device, command_pool_info),
          }
      );
    }

    /**
     * @brief Find a memory type matching required bits and properties.
     * @param memory_type_bits Vulkan memory type bitmask.
     * @param required_properties Desired memory properties.
     * @param memory_properties Physical-device memory properties.
     * @return Matching memory type index.
     */
    std::uint32_t find_memory_type_index(
        const std::uint32_t memory_type_bits,
        const vk::MemoryPropertyFlags required_properties,
        const vk::PhysicalDeviceMemoryProperties &memory_properties
    ) {
      for (std::uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        const bool bit_matches = (memory_type_bits & (1U << index)) != 0U;
        const bool property_matches = (memory_properties.memoryTypes[index].propertyFlags & required_properties) == required_properties;
        if (bit_matches && property_matches) {
          return index;
        }
      }
      throw std::runtime_error(
          fmt::format("Unable to find memory type with required properties: {:#x}", static_cast<std::uint32_t>(required_properties))
      );
    }

    /**
     * @brief Allocate one primary command buffer from a queue resource pool.
     * @param device Logical device.
     * @param resource Queue resources containing command pool.
     * @return RAII command buffer list containing one command buffer.
     */
    std::vector<vk::raii::CommandBuffer> allocate_one_time_command_buffer(
        const vk::raii::Device &device,
        const UploadContext::QueueResources &resource
    ) {
      const vk::CommandBufferAllocateInfo allocate_info = vk::CommandBufferAllocateInfo{}
          .setCommandPool(*resource.command_pool)
          .setLevel(vk::CommandBufferLevel::ePrimary)
          .setCommandBufferCount(1U);
      return device.allocateCommandBuffers(allocate_info);
    }
  } // namespace

  UploadContext::UploadContext(
      const EngineContext *engine,
      const vk::raii::Device *device,
      const vk::raii::PhysicalDevice *physical_device,
      std::vector<QueueResources> &&queue_resources,
      const std::uint32_t submission_queue_family_index,
      const vk::Queue submission_queue
  )
    : engine_(engine),
      device_(device),
      physical_device_(physical_device),
      queue_resources_(std::move(queue_resources)),
      submission_queue_family_index_(submission_queue_family_index),
      submission_queue_(submission_queue) {
  }

  UploadContext UploadContext::create(const EngineContext &engine, const UploadContextCreateInfo &info) {
    const vk::raii::Device &device = engine.device();
    const QueueTopology &topology = engine.queue_topology();

    std::vector<QueueResources> queue_resources;
    queue_resources.reserve(3U);
    append_unique_queue_resources(&queue_resources, topology.families.graphics, topology.graphics_queue, device);
    if (topology.families.async_compute.has_value() && topology.async_compute_queue.has_value()) {
      append_unique_queue_resources(&queue_resources, *topology.families.async_compute, *topology.async_compute_queue, device);
    }
    if (topology.families.transfer.has_value() && topology.transfer_queue.has_value()) {
      append_unique_queue_resources(&queue_resources, *topology.families.transfer, *topology.transfer_queue, device);
    }

    std::uint32_t submission_family_index = topology.families.graphics;
    vk::Queue submission_queue = topology.graphics_queue;
    if (info.prefer_transfer_queue && topology.families.transfer.has_value() && topology.transfer_queue.has_value()) {
      submission_family_index = *topology.families.transfer;
      submission_queue = *topology.transfer_queue;
    }

    return UploadContext{
        &engine,
        &device,
        &engine.physical_device_raii(),
        std::move(queue_resources),
        submission_family_index,
        submission_queue,
    };
  }

  const UploadContext::QueueResources *UploadContext::queue_resources_for_family(const std::uint32_t family_index) const {
    const auto it = std::ranges::find_if(
        queue_resources_,
        [&](const QueueResources &resource) { return resource.family_index == family_index; }
    );
    if (it == queue_resources_.end()) {
      return nullptr;
    }
    return &(*it);
  }

  UploadContext::QueueResources *UploadContext::queue_resources_for_family(const std::uint32_t family_index) {
    const auto it = std::ranges::find_if(
        queue_resources_,
        [&](const QueueResources &resource) { return resource.family_index == family_index; }
    );
    if (it == queue_resources_.end()) {
      return nullptr;
    }
    return &(*it);
  }

  void UploadContext::upload_buffer(const UploadBufferRequest &request) {
    if (device_ == nullptr || physical_device_ == nullptr || engine_ == nullptr) {
      throw std::runtime_error("UploadContext is not initialized.");
    }
    if (request.destination_buffer == VK_NULL_HANDLE) {
      throw std::runtime_error("UploadBufferRequest.destination_buffer must be valid.");
    }
    if (request.source_data.empty()) {
      return;
    }

    const QueueTopology &topology = engine_->queue_topology();
    const std::uint32_t destination_family_index = request.destination_queue_family_index.value_or(topology.families.graphics);
    QueueResources *submission_resources = queue_resources_for_family(submission_queue_family_index_);
    QueueResources *destination_resources = queue_resources_for_family(destination_family_index);
    if (submission_resources == nullptr) {
      throw std::runtime_error("Upload submission queue resources are unavailable.");
    }
    if (destination_resources == nullptr) {
      throw std::runtime_error(
          fmt::format(
              "Destination queue family {} is not tracked by UploadContext.",
              destination_family_index
          )
      );
    }

    const vk::DeviceSize upload_size = static_cast<vk::DeviceSize>(request.source_data.size_bytes());
    const vk::BufferCreateInfo staging_buffer_info = vk::BufferCreateInfo{}
        .setSize(upload_size)
        .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setQueueFamilyIndices(submission_queue_family_index_);

    vk::raii::Buffer staging_buffer(*device_, staging_buffer_info);
    const vk::MemoryRequirements memory_requirements = staging_buffer.getMemoryRequirements();
    const vk::PhysicalDeviceMemoryProperties memory_properties = physical_device_->getMemoryProperties();
    const std::uint32_t memory_type_index = find_memory_type_index(
        memory_requirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        memory_properties
    );
    const vk::MemoryAllocateInfo allocate_info = vk::MemoryAllocateInfo{}
        .setAllocationSize(memory_requirements.size)
        .setMemoryTypeIndex(memory_type_index);
    vk::raii::DeviceMemory staging_memory(*device_, allocate_info);
    staging_buffer.bindMemory(*staging_memory, 0U);

    void *mapped = staging_memory.mapMemory(0U, upload_size);
    std::memcpy(mapped, request.source_data.data(), request.source_data.size_bytes());
    staging_memory.unmapMemory();

    std::vector<vk::raii::CommandBuffer> transfer_command_buffers = allocate_one_time_command_buffer(*device_, *submission_resources);
    vk::raii::CommandBuffer &transfer_command_buffer = transfer_command_buffers.front();
    const vk::CommandBufferBeginInfo begin_info = vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    transfer_command_buffer.begin(begin_info);
    const vk::BufferCopy copy_region = vk::BufferCopy{}
        .setSrcOffset(0U)
        .setDstOffset(request.destination_offset)
        .setSize(upload_size);
    transfer_command_buffer.copyBuffer(*staging_buffer, request.destination_buffer, copy_region);

    const bool needs_queue_ownership_transfer =
        request.perform_queue_family_ownership_transfer && submission_queue_family_index_ != destination_family_index;
    if (needs_queue_ownership_transfer) {
      const vk::BufferMemoryBarrier2 release_barrier = vk::BufferMemoryBarrier2{}
          .setSrcStageMask(vk::PipelineStageFlagBits2::eCopy)
          .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
          .setDstStageMask(vk::PipelineStageFlagBits2::eNone)
          .setDstAccessMask(vk::AccessFlagBits2::eNone)
          .setSrcQueueFamilyIndex(submission_queue_family_index_)
          .setDstQueueFamilyIndex(destination_family_index)
          .setBuffer(request.destination_buffer)
          .setOffset(request.destination_offset)
          .setSize(upload_size);
      const vk::DependencyInfo release_dependency = vk::DependencyInfo{}.setBufferMemoryBarriers(release_barrier);
      transfer_command_buffer.pipelineBarrier2(release_dependency);
    } else {
      const vk::BufferMemoryBarrier2 visibility_barrier = vk::BufferMemoryBarrier2{}
          .setSrcStageMask(vk::PipelineStageFlagBits2::eCopy)
          .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
          .setDstStageMask(request.destination_stage_mask)
          .setDstAccessMask(request.destination_access_mask)
          .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
          .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
          .setBuffer(request.destination_buffer)
          .setOffset(request.destination_offset)
          .setSize(upload_size);
      const vk::DependencyInfo visibility_dependency = vk::DependencyInfo{}.setBufferMemoryBarriers(visibility_barrier);
      transfer_command_buffer.pipelineBarrier2(visibility_dependency);
    }

    transfer_command_buffer.end();

    vk::raii::Fence transfer_fence(*device_, vk::FenceCreateInfo{});
    const vk::SubmitInfo transfer_submit = vk::SubmitInfo{}.setCommandBuffers(*transfer_command_buffer);
    submission_resources->queue.submit(transfer_submit, *transfer_fence);

    const std::array transfer_fences{*transfer_fence};
    const vk::Result transfer_wait = device_->waitForFences(transfer_fences, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
    if (transfer_wait != vk::Result::eSuccess) {
      throw std::runtime_error("Upload transfer submission did not complete successfully.");
    }

    if (needs_queue_ownership_transfer) {
      std::vector<vk::raii::CommandBuffer> acquire_command_buffers = allocate_one_time_command_buffer(*device_, *destination_resources);
      vk::raii::CommandBuffer &acquire_command_buffer = acquire_command_buffers.front();
      acquire_command_buffer.begin(begin_info);
      const vk::BufferMemoryBarrier2 acquire_barrier = vk::BufferMemoryBarrier2{}
          .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
          .setSrcAccessMask(vk::AccessFlagBits2::eNone)
          .setDstStageMask(request.destination_stage_mask)
          .setDstAccessMask(request.destination_access_mask)
          .setSrcQueueFamilyIndex(submission_queue_family_index_)
          .setDstQueueFamilyIndex(destination_family_index)
          .setBuffer(request.destination_buffer)
          .setOffset(request.destination_offset)
          .setSize(upload_size);
      const vk::DependencyInfo acquire_dependency = vk::DependencyInfo{}.setBufferMemoryBarriers(acquire_barrier);
      acquire_command_buffer.pipelineBarrier2(acquire_dependency);
      acquire_command_buffer.end();

      vk::raii::Fence acquire_fence(*device_, vk::FenceCreateInfo{});
      const vk::SubmitInfo acquire_submit = vk::SubmitInfo{}.setCommandBuffers(*acquire_command_buffer);
      destination_resources->queue.submit(acquire_submit, *acquire_fence);

      const std::array acquire_fences{*acquire_fence};
      const vk::Result acquire_wait = device_->waitForFences(acquire_fences, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
      if (acquire_wait != vk::Result::eSuccess) {
        throw std::runtime_error("Upload ownership-acquire submission did not complete successfully.");
      }
    }
  }

  void UploadContext::wait_idle() const {
    if (device_ != nullptr) {
      device_->waitIdle();
    }
  }

  std::uint32_t UploadContext::submission_queue_family_index() const noexcept { return submission_queue_family_index_; }

  vk::Queue UploadContext::submission_queue() const noexcept { return submission_queue_; }
} // namespace varre::engine
