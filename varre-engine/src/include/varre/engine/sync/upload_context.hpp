/**
 * @file upload_context.hpp
 * @brief Buffer upload primitives with staging and transfer-queue support.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/engine/render/pass_mode.hpp"

namespace varre::engine {

class EngineContext;

/**
 * @brief Upload-context creation options.
 */
struct UploadContextCreateInfo {
  /** @brief Prefer transfer queue when available; otherwise use graphics queue. */
  bool prefer_transfer_queue = true;
};

/**
 * @brief One immediate buffer upload request.
 */
struct UploadBufferRequest {
  /** @brief Destination buffer handle to receive uploaded data. */
  vk::Buffer destination_buffer = VK_NULL_HANDLE;
  /** @brief Destination byte offset into @ref destination_buffer. */
  vk::DeviceSize destination_offset = 0U;
  /** @brief Source bytes copied into an internally created staging buffer. */
  std::span<const std::byte> source_data;
  /**
   * @brief Destination queue family owning the resource after upload.
   *
   * When unset, graphics queue family is used.
   */
  std::optional<std::uint32_t> destination_queue_family_index;
  /**
   * @brief Perform queue-family ownership transfer when upload and destination families differ.
   *
   * Disable only when resources are created with compatible concurrent sharing.
   */
  bool perform_queue_family_ownership_transfer = true;
  /** @brief Destination pipeline stage used for visibility in ownership-acquire barrier. */
  vk::PipelineStageFlags2 destination_stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
  /** @brief Destination access mask used for visibility in ownership-acquire barrier. */
  vk::AccessFlags2 destination_access_mask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
};

/**
 * @brief Dependency token produced by asynchronous upload submissions.
 */
struct UploadDependencyToken {
  /** @brief Timeline semaphore signaled by upload completion. */
  vk::Semaphore semaphore = VK_NULL_HANDLE;
  /** @brief Timeline value signaled at completion. */
  std::uint64_t value = 0U;
  /** @brief Recommended stage mask for waiting before resource usage. */
  vk::PipelineStageFlags2 stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
};

/**
 * @brief Upload path that stages host data and submits copy work to transfer/graphics queues.
 */
class UploadContext {
public:
  /**
   * @brief Queue resources tracked by upload context.
   */
  struct QueueResources {
    std::uint32_t family_index = 0U;
    vk::Queue queue = VK_NULL_HANDLE;
    vk::raii::CommandPool command_pool{nullptr};
  };

  /**
   * @brief Create an upload context for one engine instance.
   * @param engine Initialized engine context.
   * @param info Upload-context creation options.
   * @return Initialized upload context.
   */
  [[nodiscard]] static UploadContext create(const EngineContext &engine, const UploadContextCreateInfo &info = {});

  /**
   * @brief Destroy upload context after draining outstanding GPU work.
   */
  ~UploadContext();

  /**
   * @brief Move-construct upload context.
   * @param other Context being moved from.
   */
  UploadContext(UploadContext &&other) noexcept = default;

  /**
   * @brief Move-assign upload context.
   * @param other Context being moved from.
   * @return `*this`.
   */
  UploadContext &operator=(UploadContext &&other) noexcept = default;

  UploadContext(const UploadContext &) = delete;
  UploadContext &operator=(const UploadContext &) = delete;

  /**
   * @brief Upload bytes asynchronously into a destination buffer.
   * @param request Upload request.
   * @return Completion token that can be consumed by pass-mode waits.
   */
  [[nodiscard]] UploadDependencyToken upload_buffer_async(const UploadBufferRequest &request);

  /**
   * @brief Upload bytes into a destination buffer using an internal staging allocation.
   * @param request Upload request.
   */
  void upload_buffer(const UploadBufferRequest &request);

  /**
   * @brief Wait until one upload dependency token is complete.
   * @param token Upload completion token.
   */
  void wait(const UploadDependencyToken &token);

  /**
   * @brief Wait until all device queues are idle.
   */
  void wait_idle() const;

  /**
   * @brief Queue family index used for copy submissions.
   * @return Upload submission queue family index.
   */
  [[nodiscard]] std::uint32_t submission_queue_family_index() const noexcept;

  /**
   * @brief Queue handle used for copy submissions.
   * @return Upload submission queue.
   */
  [[nodiscard]] vk::Queue submission_queue() const noexcept;

private:
  /**
   * @brief Internal constructor from prebuilt queue resources.
   */
  UploadContext(const EngineContext *engine, const vk::raii::Device *device, const vk::raii::PhysicalDevice *physical_device,
                std::vector<QueueResources> &&queue_resources, std::uint32_t submission_queue_family_index, vk::Queue submission_queue,
                vk::raii::Semaphore &&timeline_semaphore);

  /**
   * @brief In-flight upload resources retained until timeline completion.
   */
  struct InFlightUpload {
    std::uint64_t completion_value = 0U;
    vk::raii::Buffer staging_buffer{nullptr};
    vk::raii::DeviceMemory staging_memory{nullptr};
    std::vector<vk::raii::CommandBuffer> submission_command_buffers;
    std::vector<vk::raii::CommandBuffer> destination_command_buffers;
  };

  /**
   * @brief Discard completed in-flight upload resources.
   */
  void collect_completed_uploads();

  /**
   * @brief Find queue resources for one queue family.
   * @param family_index Queue family index.
   * @return Pointer to queue resources.
   */
  [[nodiscard]] const QueueResources *queue_resources_for_family(std::uint32_t family_index) const;

  /**
   * @brief Find queue resources for one queue family.
   * @param family_index Queue family index.
   * @return Pointer to queue resources.
   */
  [[nodiscard]] QueueResources *queue_resources_for_family(std::uint32_t family_index);

  const EngineContext *engine_ = nullptr;
  const vk::raii::Device *device_ = nullptr;
  const vk::raii::PhysicalDevice *physical_device_ = nullptr;
  std::vector<QueueResources> queue_resources_;
  std::uint32_t submission_queue_family_index_ = 0U;
  vk::Queue submission_queue_ = VK_NULL_HANDLE;
  vk::raii::Semaphore timeline_semaphore_{nullptr};
  std::uint64_t next_timeline_value_ = 0U;
  std::vector<InFlightUpload> in_flight_uploads_;
};

/**
 * @brief Convert one upload dependency token into a pass-execution wait edge.
 * @param phase_id Target phase identifier.
 * @param token Upload dependency token.
 * @return External wait edge for pass execution.
 */
[[nodiscard]] PassExternalWait make_upload_dependency_wait(PassPhaseId phase_id, const UploadDependencyToken &token);

/**
 * @brief Append one upload dependency wait edge to pass execution info.
 * @param execution_info Destination pass execution info.
 * @param phase_id Target phase identifier.
 * @param token Upload dependency token.
 */
void append_upload_dependency_wait(PassExecutionInfo *execution_info, PassPhaseId phase_id, const UploadDependencyToken &token);

} // namespace varre::engine

namespace varre::engine::sync {
using ::varre::engine::append_upload_dependency_wait;
using ::varre::engine::make_upload_dependency_wait;
using ::varre::engine::UploadBufferRequest;
using ::varre::engine::UploadContext;
using ::varre::engine::UploadContextCreateInfo;
using ::varre::engine::UploadDependencyToken;
} // namespace varre::engine::sync
