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
   * @brief Upload bytes into a destination buffer using an internal staging allocation.
   * @param request Upload request.
   */
  void upload_buffer(const UploadBufferRequest &request);

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
  UploadContext(
      const EngineContext *engine,
      const vk::raii::Device *device,
      const vk::raii::PhysicalDevice *physical_device,
      std::vector<QueueResources> &&queue_resources,
      std::uint32_t submission_queue_family_index,
      vk::Queue submission_queue
  );

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
};

} // namespace varre::engine
