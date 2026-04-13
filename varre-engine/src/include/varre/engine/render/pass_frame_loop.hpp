/**
 * @file pass_frame_loop.hpp
 * @brief Pass-mode frame orchestration over acquire/execute/present.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include <vulkan/vulkan_raii.hpp>

#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/swapchain.hpp"
#include "varre/engine/render/pass_mode.hpp"
#include "varre/engine/sync/frame_loop.hpp"

namespace varre::engine {

/**
 * @brief Per-frame context passed to pass-graph builders.
 */
struct PassFrameContext {
  /** @brief Frame slot index used by synchronization primitives. */
  std::uint32_t frame_index = 0U;
  /** @brief Swapchain image index acquired for this frame. */
  std::uint32_t image_index = 0U;
  /** @brief Stable pass resource ID for the acquired swapchain image. */
  PassResourceId swapchain_resource_id = 0U;
  /** @brief Swapchain extent for this frame. */
  vk::Extent2D extent{};
  /** @brief Swapchain image format for this frame. */
  vk::Format image_format = vk::Format::eUndefined;
  /** @brief Swapchain image handle for @ref image_index. */
  vk::Image image = VK_NULL_HANDLE;
  /** @brief Swapchain image view handle for @ref image_index. */
  vk::ImageView image_view = VK_NULL_HANDLE;
};

/**
 * @brief Callback used to build one frame-local pass graph.
 */
using PassGraphBuildCallback = std::function<void(const PassFrameContext &context, PassGraph *graph)>;

/**
 * @brief Optional callback invoked after successful swapchain recreation.
 */
using PassSwapchainRecreatedCallback = std::function<void(const SwapchainContext &swapchain)>;

/**
 * @brief Frame execution outcome classification.
 */
enum class PassFrameRunStatus : std::uint8_t {
  /** @brief Frame was rendered and presented without swapchain recreation. */
  kPresented = 0U,
  /** @brief Frame was rendered/presented and swapchain was recreated after present. */
  kPresentedAndSwapchainRecreated,
  /** @brief Swapchain was recreated before rendering due to acquire out-of-date. */
  kSwapchainRecreatedBeforeRender,
  /** @brief Swapchain requires recreation but recoverable recreate attempt was deferred. */
  kSwapchainRecreateDeferred,
};

/**
 * @brief Result of one pass-frame execution.
 */
struct PassFrameRunResult {
  /** @brief High-level frame execution status. */
  PassFrameRunStatus status = PassFrameRunStatus::kPresented;
  /** @brief Acquire status returned by @ref FrameLoop::acquire_next_image. */
  FrameAcquireStatus acquire_status = FrameAcquireStatus::kSuccess;
  /** @brief Present status when presentation happened. */
  std::optional<FramePresentStatus> present_status;
  /** @brief Structured engine error code for recoverable frame status signals. */
  std::optional<EngineErrorCode> error_code;
  /** @brief Frame slot index used during execution. */
  std::uint32_t frame_index = 0U;
  /** @brief Swapchain image index used during execution when acquired. */
  std::uint32_t image_index = 0U;
  /** @brief True when pass graph execution occurred for this frame. */
  bool executed_graph = false;
  /** @brief True when swapchain recreation succeeded during this call. */
  bool swapchain_recreated = false;
};

/**
 * @brief One frame execution request for pass-mode orchestration.
 */
struct PassFrameRunInfo {
  /**
   * @brief Acquire timeout in nanoseconds.
   *
   * Defaults to `uint64_max`.
   */
  std::uint64_t acquire_timeout_ns = std::numeric_limits<std::uint64_t>::max();
  /**
   * @brief Stage mask used for external wait on frame `image_available`.
   */
  vk::PipelineStageFlags2 image_available_wait_stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
  /**
   * @brief Stage mask used when signaling frame `render_finished`.
   */
  vk::PipelineStageFlags2 render_finished_signal_stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
  /**
   * @brief Optional swapchain recreation preferences.
   *
   * When unset, `SwapchainContext::recreate()` is used.
   */
  std::optional<SwapchainCreateInfo> recreate_info;
  /**
   * @brief Optional callback invoked after successful swapchain recreation.
   */
  PassSwapchainRecreatedCallback on_swapchain_recreated;
};

/**
 * @brief Pass-mode frame orchestrator using FrameLoop + PassExecutor.
 */
class PassFrameLoop {
public:
  /**
   * @brief Create pass-frame orchestration runtime.
   * @param engine Initialized engine context.
   * @param swapchain Active swapchain context.
   * @param frame_loop_create_info Frame-loop creation options.
   * @param pass_executor_create_info Pass-executor creation options.
   * @return Initialized pass-frame loop.
   */
  [[nodiscard]] static PassFrameLoop create(const EngineContext &engine, const SwapchainContext &swapchain,
                                            const FrameLoopCreateInfo &frame_loop_create_info = {},
                                            const PassExecutorCreateInfo &pass_executor_create_info = {});

  /**
   * @brief Move-construct pass-frame loop.
   * @param other Instance being moved from.
   */
  PassFrameLoop(PassFrameLoop &&other) noexcept = default;

  /**
   * @brief Move-assign pass-frame loop.
   * @param other Instance being moved from.
   * @return `*this`.
   */
  PassFrameLoop &operator=(PassFrameLoop &&other) noexcept = default;

  PassFrameLoop(const PassFrameLoop &) = delete;
  PassFrameLoop &operator=(const PassFrameLoop &) = delete;

  /**
   * @brief Execute one frame: acquire -> build/execute pass graph -> present.
   * @param swapchain Swapchain to acquire/present/recreate.
   * @param build_graph Callback that builds pass graph for the acquired frame.
   * @param run_info Optional per-frame execution options.
   * @return Frame execution result.
   */
  [[nodiscard]] PassFrameRunResult run_frame(SwapchainContext *swapchain, const PassGraphBuildCallback &build_graph, const PassFrameRunInfo &run_info = {});

  /**
   * @brief Access underlying frame loop.
   * @return Mutable frame-loop reference.
   */
  [[nodiscard]] FrameLoop &frame_loop() noexcept;

  /**
   * @brief Access underlying frame loop.
   * @return Immutable frame-loop reference.
   */
  [[nodiscard]] const FrameLoop &frame_loop() const noexcept;

  /**
   * @brief Access underlying pass executor.
   * @return Mutable pass-executor reference.
   */
  [[nodiscard]] PassExecutor &pass_executor() noexcept;

  /**
   * @brief Access underlying pass executor.
   * @return Immutable pass-executor reference.
   */
  [[nodiscard]] const PassExecutor &pass_executor() const noexcept;

  /**
   * @brief Wait for all engine/device work used by this loop to go idle.
   */
  void wait_idle() const;

private:
  /**
   * @brief Attempt swapchain recreation and optionally invoke callback.
   * @param swapchain Swapchain instance to recreate.
   * @param run_info Per-frame execution options.
   * @return `true` when recreation succeeded.
   */
  [[nodiscard]] bool try_recreate_swapchain(SwapchainContext *swapchain, const PassFrameRunInfo &run_info);

  /**
   * @brief Internal constructor.
   */
  PassFrameLoop(FrameLoop &&frame_loop, PassExecutor &&pass_executor);

  FrameLoop frame_loop_;
  PassExecutor pass_executor_;
};

} // namespace varre::engine

namespace varre::engine::render {
using ::varre::engine::PassFrameContext;
using ::varre::engine::PassFrameLoop;
using ::varre::engine::PassFrameRunInfo;
using ::varre::engine::PassFrameRunResult;
using ::varre::engine::PassFrameRunStatus;
using ::varre::engine::PassGraphBuildCallback;
using ::varre::engine::PassSwapchainRecreatedCallback;
} // namespace varre::engine::render
