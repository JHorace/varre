/**
 * @file upload_context.cpp
 * @brief Buffer upload primitives with staging and transfer-queue support.
 */
#include "varre/engine/sync/upload_context.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <ranges>
#include <utility>

#include <fmt/format.h>

#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/errors.hpp"

namespace varre::engine {
namespace detail {
/**
 * @brief Append queue resources only when queue family is not present yet.
 * @param queue_resources Destination queue resource list.
 * @param family_index Queue family index.
 * @param queue Queue handle.
 * @param device Logical device.
 */
void append_unique_queue_resources(std::vector<UploadContext::QueueResources> *queue_resources, const std::uint32_t family_index, const vk::Queue queue,
                                   const vk::raii::Device &device) {
  const bool exists =
    std::ranges::any_of(*queue_resources, [&](const UploadContext::QueueResources &resource) { return resource.family_index == family_index; });
  if (exists || queue == VK_NULL_HANDLE) {
    return;
  }

  const vk::CommandPoolCreateInfo command_pool_info =
    vk::CommandPoolCreateInfo{}
      .setFlags(vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
      .setQueueFamilyIndex(family_index);
  queue_resources->push_back(UploadContext::QueueResources{
    .family_index = family_index,
    .queue = queue,
    .command_pool = vk::raii::CommandPool(device, command_pool_info),
  });
}

/**
 * @brief Find a memory type matching required bits and properties.
 * @param memory_type_bits Vulkan memory type bitmask.
 * @param required_properties Desired memory properties.
 * @param memory_properties Physical-device memory properties.
 * @return Matching memory type index.
 */
std::uint32_t find_memory_type_index(const std::uint32_t memory_type_bits, const vk::MemoryPropertyFlags required_properties,
                                     const vk::PhysicalDeviceMemoryProperties &memory_properties) {
  for (std::uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
    const bool bit_matches = (memory_type_bits & (1U << index)) != 0U;
    const bool property_matches = (memory_properties.memoryTypes[index].propertyFlags & required_properties) == required_properties;
    if (bit_matches && property_matches) {
      return index;
    }
  }
  throw make_engine_error(EngineErrorCode::kMissingRequirement,
                          fmt::format("Unable to find memory type with required properties: {:#x}", static_cast<std::uint32_t>(required_properties)));
}

/**
 * @brief Allocate one primary command buffer from a queue resource pool.
 * @param device Logical device.
 * @param resource Queue resources containing command pool.
 * @return RAII command buffer list containing one command buffer.
 */
std::vector<vk::raii::CommandBuffer> allocate_one_time_command_buffer(const vk::raii::Device &device, const UploadContext::QueueResources &resource) {
  const vk::CommandBufferAllocateInfo allocate_info =
    vk::CommandBufferAllocateInfo{}.setCommandPool(*resource.command_pool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1U);
  return device.allocateCommandBuffers(allocate_info);
}

/**
 * @brief Ensure semaphore wait stage mask is valid for submit2 semaphore waits.
 * @param stage_mask Requested stage mask.
 * @return Valid stage mask.
 */
[[nodiscard]] vk::PipelineStageFlags2 sanitize_wait_stage_mask(const vk::PipelineStageFlags2 stage_mask) {
  if (stage_mask == vk::PipelineStageFlagBits2::eNone) {
    return vk::PipelineStageFlagBits2::eAllCommands;
  }
  return stage_mask;
}
} // namespace detail

UploadContext::UploadContext(const EngineContext *engine, const vk::raii::Device *device, const vk::raii::PhysicalDevice *physical_device,
                             std::vector<QueueResources> &&queue_resources, const std::uint32_t submission_queue_family_index, const vk::Queue submission_queue,
                             vk::raii::Semaphore &&timeline_semaphore)
    : engine_(engine), device_(device), physical_device_(physical_device), queue_resources_(std::move(queue_resources)),
      submission_queue_family_index_(submission_queue_family_index), submission_queue_(submission_queue), timeline_semaphore_(std::move(timeline_semaphore)) {}

UploadContext::~UploadContext() {
  if (device_ == nullptr || timeline_semaphore_ == nullptr || in_flight_uploads_.empty()) {
    return;
  }

  const std::uint64_t wait_value = in_flight_uploads_.back().completion_value;
  if (wait_value > 0U) {
    try {
      const vk::SemaphoreWaitInfo wait_info = vk::SemaphoreWaitInfo{}.setSemaphores(*timeline_semaphore_).setValues(wait_value);
      static_cast<void>(device_->waitSemaphores(wait_info, std::numeric_limits<std::uint64_t>::max()));
    } catch (...) {
      try {
        device_->waitIdle();
      } catch (...) {
      }
    }
  }

  in_flight_uploads_.clear();
}

UploadContext UploadContext::create(const EngineContext &engine, const UploadContextCreateInfo &info) {
  const vk::raii::Device &device = engine.device();
  const DeviceQueueTopology &topology = engine.device_queue_topology();

  std::vector<QueueResources> queue_resources;
  queue_resources.reserve(3U);
  detail::append_unique_queue_resources(&queue_resources, topology.families.graphics, topology.graphics_queue, device);
  if (topology.families.async_compute.has_value() && topology.async_compute_queue.has_value()) {
    detail::append_unique_queue_resources(&queue_resources, *topology.families.async_compute, *topology.async_compute_queue, device);
  }
  if (topology.families.transfer.has_value() && topology.transfer_queue.has_value()) {
    detail::append_unique_queue_resources(&queue_resources, *topology.families.transfer, *topology.transfer_queue, device);
  }

  std::uint32_t submission_family_index = topology.families.graphics;
  vk::Queue submission_queue = topology.graphics_queue;
  if (info.prefer_transfer_queue && topology.families.transfer.has_value() && topology.transfer_queue.has_value()) {
    submission_family_index = *topology.families.transfer;
    submission_queue = *topology.transfer_queue;
  }

  const vk::SemaphoreTypeCreateInfo semaphore_type_info = vk::SemaphoreTypeCreateInfo{}.setSemaphoreType(vk::SemaphoreType::eTimeline).setInitialValue(0U);
  const vk::SemaphoreCreateInfo semaphore_create_info = vk::SemaphoreCreateInfo{}.setPNext(&semaphore_type_info);
  vk::raii::Semaphore timeline_semaphore(device, semaphore_create_info);

  return UploadContext{
    &engine, &device, &engine.physical_device_raii(), std::move(queue_resources), submission_family_index, submission_queue, std::move(timeline_semaphore),
  };
}

const UploadContext::QueueResources *UploadContext::queue_resources_for_family(const std::uint32_t family_index) const {
  const auto it = std::ranges::find_if(queue_resources_, [&](const QueueResources &resource) { return resource.family_index == family_index; });
  if (it == queue_resources_.end()) {
    return nullptr;
  }
  return &(*it);
}

UploadContext::QueueResources *UploadContext::queue_resources_for_family(const std::uint32_t family_index) {
  const auto it = std::ranges::find_if(queue_resources_, [&](const QueueResources &resource) { return resource.family_index == family_index; });
  if (it == queue_resources_.end()) {
    return nullptr;
  }
  return &(*it);
}

UploadDependencyToken UploadContext::upload_buffer_async(const UploadBufferRequest &request) {
  if (device_ == nullptr || physical_device_ == nullptr || engine_ == nullptr || timeline_semaphore_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "UploadContext is not initialized.");
  }
  if (request.destination_buffer == VK_NULL_HANDLE) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "UploadBufferRequest.destination_buffer must be valid.");
  }

  const vk::PipelineStageFlags2 wait_stage_mask = detail::sanitize_wait_stage_mask(request.destination_stage_mask);
  if (request.source_data.empty()) {
    return UploadDependencyToken{
      .semaphore = *timeline_semaphore_,
      .value = 0U,
      .stage_mask = wait_stage_mask,
    };
  }

  collect_completed_uploads();

  const DeviceQueueTopology &topology = engine_->device_queue_topology();
  const std::uint32_t destination_family_index = request.destination_queue_family_index.value_or(topology.families.graphics);
  QueueResources *submission_resources = queue_resources_for_family(submission_queue_family_index_);
  QueueResources *destination_resources = queue_resources_for_family(destination_family_index);
  if (submission_resources == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "Upload submission queue resources are unavailable.");
  }
  if (destination_resources == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Destination queue family {} is not tracked by UploadContext.", destination_family_index));
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
  const std::uint32_t memory_type_index = detail::find_memory_type_index(
    memory_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, memory_properties);
  const vk::MemoryAllocateInfo allocate_info = vk::MemoryAllocateInfo{}.setAllocationSize(memory_requirements.size).setMemoryTypeIndex(memory_type_index);
  vk::raii::DeviceMemory staging_memory(*device_, allocate_info);
  staging_buffer.bindMemory(*staging_memory, 0U);

  void *mapped = staging_memory.mapMemory(0U, upload_size);
  std::memcpy(mapped, request.source_data.data(), request.source_data.size_bytes());
  staging_memory.unmapMemory();

  std::vector<vk::raii::CommandBuffer> transfer_command_buffers = detail::allocate_one_time_command_buffer(*device_, *submission_resources);
  vk::raii::CommandBuffer &transfer_command_buffer = transfer_command_buffers.front();
  const vk::CommandBufferBeginInfo begin_info = vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  transfer_command_buffer.begin(begin_info);
  const vk::BufferCopy copy_region = vk::BufferCopy{}.setSrcOffset(0U).setDstOffset(request.destination_offset).setSize(upload_size);
  transfer_command_buffer.copyBuffer(*staging_buffer, request.destination_buffer, copy_region);

  const bool needs_queue_ownership_transfer = request.perform_queue_family_ownership_transfer && submission_queue_family_index_ != destination_family_index;
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
                                                          .setDstStageMask(wait_stage_mask)
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

  std::vector<vk::raii::CommandBuffer> acquire_command_buffers;
  if (needs_queue_ownership_transfer) {
    acquire_command_buffers = detail::allocate_one_time_command_buffer(*device_, *destination_resources);
    vk::raii::CommandBuffer &acquire_command_buffer = acquire_command_buffers.front();
    acquire_command_buffer.begin(begin_info);
    const vk::BufferMemoryBarrier2 acquire_barrier = vk::BufferMemoryBarrier2{}
                                                       .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                                                       .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                                                       .setDstStageMask(wait_stage_mask)
                                                       .setDstAccessMask(request.destination_access_mask)
                                                       .setSrcQueueFamilyIndex(submission_queue_family_index_)
                                                       .setDstQueueFamilyIndex(destination_family_index)
                                                       .setBuffer(request.destination_buffer)
                                                       .setOffset(request.destination_offset)
                                                       .setSize(upload_size);
    const vk::DependencyInfo acquire_dependency = vk::DependencyInfo{}.setBufferMemoryBarriers(acquire_barrier);
    acquire_command_buffer.pipelineBarrier2(acquire_dependency);
    acquire_command_buffer.end();
  }

  const std::uint64_t transfer_complete_value = ++next_timeline_value_;
  const std::uint64_t completion_value = needs_queue_ownership_transfer ? ++next_timeline_value_ : transfer_complete_value;

  const vk::CommandBufferSubmitInfo transfer_command_buffer_submit_info = vk::CommandBufferSubmitInfo{}.setCommandBuffer(*transfer_command_buffer);
  const vk::SemaphoreSubmitInfo transfer_signal_info =
    vk::SemaphoreSubmitInfo{}.setSemaphore(*timeline_semaphore_).setValue(transfer_complete_value).setStageMask(vk::PipelineStageFlagBits2::eAllCommands);
  const vk::SubmitInfo2 transfer_submit_info =
    vk::SubmitInfo2{}.setCommandBufferInfos(transfer_command_buffer_submit_info).setSignalSemaphoreInfos(transfer_signal_info);

  bool transfer_submitted = false;
  try {
    submission_resources->queue.submit2(transfer_submit_info, vk::Fence{});
    transfer_submitted = true;

    if (needs_queue_ownership_transfer) {
      const vk::CommandBufferSubmitInfo acquire_command_buffer_submit_info = vk::CommandBufferSubmitInfo{}.setCommandBuffer(*acquire_command_buffers.front());
      const vk::SemaphoreSubmitInfo acquire_wait_info =
        vk::SemaphoreSubmitInfo{}.setSemaphore(*timeline_semaphore_).setValue(transfer_complete_value).setStageMask(wait_stage_mask);
      const vk::SemaphoreSubmitInfo acquire_signal_info =
        vk::SemaphoreSubmitInfo{}.setSemaphore(*timeline_semaphore_).setValue(completion_value).setStageMask(vk::PipelineStageFlagBits2::eAllCommands);
      const vk::SubmitInfo2 acquire_submit_info = vk::SubmitInfo2{}
                                                    .setWaitSemaphoreInfos(acquire_wait_info)
                                                    .setCommandBufferInfos(acquire_command_buffer_submit_info)
                                                    .setSignalSemaphoreInfos(acquire_signal_info);
      destination_resources->queue.submit2(acquire_submit_info, vk::Fence{});
    }
  } catch (...) {
    if (transfer_submitted) {
      try {
        submission_resources->queue.waitIdle();
      } catch (...) {
      }
    }
    throw;
  }

  InFlightUpload in_flight{};
  in_flight.completion_value = completion_value;
  in_flight.staging_buffer = std::move(staging_buffer);
  in_flight.staging_memory = std::move(staging_memory);
  in_flight.submission_command_buffers = std::move(transfer_command_buffers);
  in_flight.destination_command_buffers = std::move(acquire_command_buffers);
  in_flight_uploads_.push_back(std::move(in_flight));

  return UploadDependencyToken{
    .semaphore = *timeline_semaphore_,
    .value = completion_value,
    .stage_mask = wait_stage_mask,
  };
}

void UploadContext::upload_buffer(const UploadBufferRequest &request) {
  const UploadDependencyToken token = upload_buffer_async(request);
  wait(token);
}

void UploadContext::wait(const UploadDependencyToken &token) {
  if (device_ == nullptr || timeline_semaphore_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "UploadContext is not initialized.");
  }
  if (token.semaphore == VK_NULL_HANDLE) {
    return;
  }

  const vk::SemaphoreWaitInfo wait_info = vk::SemaphoreWaitInfo{}.setSemaphores(token.semaphore).setValues(token.value);
  const vk::Result wait_result = device_->waitSemaphores(wait_info, std::numeric_limits<std::uint64_t>::max());
  if (wait_result != vk::Result::eSuccess) {
    throw make_vulkan_result_error(wait_result, "UploadContext timeline wait failed");
  }

  if (token.semaphore == *timeline_semaphore_) {
    collect_completed_uploads();
  }
}

void UploadContext::collect_completed_uploads() {
  if (device_ == nullptr || timeline_semaphore_ == nullptr || in_flight_uploads_.empty()) {
    return;
  }

  const std::uint64_t completed_value = timeline_semaphore_.getCounterValue();
  std::erase_if(in_flight_uploads_, [&](const InFlightUpload &in_flight) { return in_flight.completion_value <= completed_value; });
}

void UploadContext::wait_idle() const {
  if (device_ != nullptr) {
    device_->waitIdle();
  }
}

std::uint32_t UploadContext::submission_queue_family_index() const noexcept { return submission_queue_family_index_; }

vk::Queue UploadContext::submission_queue() const noexcept { return submission_queue_; }

PassExternalWait make_upload_dependency_wait(const PassPhaseId phase_id, const UploadDependencyToken &token) {
  if (token.semaphore == VK_NULL_HANDLE) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "Upload dependency token semaphore must be valid.");
  }

  return PassExternalWait{
    .phase_id = phase_id,
    .semaphore = token.semaphore,
    .value = token.value,
    .stage_mask = detail::sanitize_wait_stage_mask(token.stage_mask),
  };
}

void append_upload_dependency_wait(PassExecutionInfo *execution_info, const PassPhaseId phase_id, const UploadDependencyToken &token) {
  if (execution_info == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "append_upload_dependency_wait requires a non-null execution_info.");
  }
  if (token.semaphore == VK_NULL_HANDLE) {
    return;
  }

  execution_info->waits.push_back(make_upload_dependency_wait(phase_id, token));
}
} // namespace varre::engine
