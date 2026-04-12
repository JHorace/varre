/**
 * @file frame_loop.hpp
 * @brief Per-frame synchronization and acquire/submit/present primitives.
 */
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace varre::engine {

class EngineContext;
class SwapchainContext;

/**
 * @brief Synchronization objects used by one frame slot.
 */
struct FrameSyncPrimitives {
  /** @brief Signaled by image acquisition and waited in graphics submit. */
  vk::raii::Semaphore image_available{nullptr};
  /** @brief Signaled by graphics submit and waited in present. */
  vk::raii::Semaphore render_finished{nullptr};
  /** @brief Signals completion of GPU work submitted for this frame slot. */
  vk::raii::Fence in_flight{nullptr};
};

/**
 * @brief Result classification for swapchain image acquisition.
 */
enum class FrameAcquireStatus {
  /** @brief Frame acquisition succeeded with an optimal swapchain state. */
  kSuccess,
  /** @brief Acquisition succeeded, but the swapchain is suboptimal. */
  kSuboptimal,
  /** @brief Acquisition failed because the swapchain is out of date. */
  kOutOfDate,
};

/**
 * @brief Result of acquiring the next frame image.
 */
struct AcquiredFrame {
  /** @brief Acquisition status. */
  FrameAcquireStatus status = FrameAcquireStatus::kSuccess;
  /** @brief Frame slot index used for synchronization. */
  std::uint32_t frame_index = 0U;
  /** @brief Swapchain image index to render into when acquisition succeeded. */
  std::uint32_t image_index = 0U;
};

/**
 * @brief Result classification for presentation.
 */
enum class FramePresentStatus {
  /** @brief Presentation succeeded with an optimal swapchain state. */
  kSuccess,
  /** @brief Presentation succeeded, but the swapchain is suboptimal. */
  kSuboptimal,
  /** @brief Presentation failed because the swapchain is out of date. */
  kOutOfDate,
};

/**
 * @brief Optional creation overrides for frame-loop state.
 */
struct FrameLoopCreateInfo {
  /**
   * @brief Frame-slot count.
   *
   * A value of `0` uses `SwapchainContext::max_frames_in_flight()`.
   */
  std::uint32_t frame_count = 0U;
};

/**
 * @brief Minimal frame loop that manages synchronization and present flow.
 */
class FrameLoop {
public:
  /**
   * @brief Create frame synchronization state for a swapchain.
   * @param engine Initialized engine context.
   * @param swapchain Swapchain context this loop operates on.
   * @param info Optional creation overrides.
   * @return Initialized frame loop.
   * @throws std::runtime_error on Vulkan allocation failures.
   */
  [[nodiscard]] static FrameLoop create(const EngineContext &engine, const SwapchainContext &swapchain, const FrameLoopCreateInfo &info = {});

  /**
   * @brief Move-construct the frame loop.
   * @param other Instance being moved from.
   */
  FrameLoop(FrameLoop &&other) noexcept = default;

  /**
   * @brief Move-assign the frame loop.
   * @param other Instance being moved from.
   * @return `*this`.
   */
  FrameLoop &operator=(FrameLoop &&other) noexcept = default;

  FrameLoop(const FrameLoop &) = delete;
  FrameLoop &operator=(const FrameLoop &) = delete;

  /**
   * @brief Acquire the next swapchain image for the current frame slot.
   * @param swapchain Active swapchain context.
   * @param timeout_ns Fence/acquire timeout in nanoseconds.
   * @return Acquisition metadata and status.
   */
  [[nodiscard]] AcquiredFrame acquire_next_image(
      const SwapchainContext &swapchain,
      std::uint64_t timeout_ns = std::numeric_limits<std::uint64_t>::max()
  );

  /**
   * @brief Submit one command buffer on the graphics queue for the current frame.
   * @param command_buffer Recorded command buffer to submit.
   * @param wait_stage_mask Pipeline stage to wait on `image_available`.
   */
  void submit_graphics(vk::CommandBuffer command_buffer, vk::PipelineStageFlags wait_stage_mask = vk::PipelineStageFlagBits::eColorAttachmentOutput);

  /**
   * @brief Present one swapchain image for the current frame.
   * @param swapchain Active swapchain context.
   * @param image_index Image index returned by @ref acquire_next_image.
   * @return Presentation status.
   */
  [[nodiscard]] FramePresentStatus present(const SwapchainContext &swapchain, std::uint32_t image_index);

  /**
   * @brief Reset image-tracking state after swapchain recreation.
   * @param swapchain New swapchain context.
   */
  void reset_for_swapchain(const SwapchainContext &swapchain);

  /**
   * @brief Block until the device is idle.
   */
  void wait_idle() const;

  /**
   * @brief Current frame slot index.
   * @return Frame slot index in `[0, frame_count)`.
   */
  [[nodiscard]] std::uint32_t current_frame_index() const noexcept;

  /**
   * @brief Number of frame slots.
   * @return Frame slot count.
   */
  [[nodiscard]] std::uint32_t frame_count() const noexcept;

  /**
   * @brief Synchronization primitives for the current frame slot.
   * @return Immutable frame-sync object.
   */
  [[nodiscard]] const FrameSyncPrimitives &current_sync() const noexcept;

private:
  /**
   * @brief Internal constructor from pre-built synchronization resources.
   */
  FrameLoop(
      const vk::raii::Device *device,
      vk::Queue graphics_queue,
      vk::Queue present_queue,
      std::vector<FrameSyncPrimitives> &&frames,
      std::vector<vk::Fence> &&image_in_flight_fences
  );

  const vk::raii::Device *device_ = nullptr;
  vk::Queue graphics_queue_ = VK_NULL_HANDLE;
  vk::Queue present_queue_ = VK_NULL_HANDLE;
  std::vector<FrameSyncPrimitives> frames_;
  std::vector<vk::Fence> image_in_flight_fences_;
  std::uint32_t current_frame_index_ = 0U;
  bool frame_acquired_ = false;
};

} // namespace varre::engine
