/**
 * @file swapchain.hpp
 * @brief Swapchain creation and image-view ownership primitives.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace varre::engine {

class EngineContext;
class SurfaceContext;

/**
 * @brief Configuration for swapchain creation.
 */
struct SwapchainCreateInfo {
  /** @brief Preferred framebuffer extent when the surface does not force one. */
  vk::Extent2D preferred_extent{1280U, 720U};
  /** @brief Preferred present mode; falls back to FIFO if unavailable. */
  vk::PresentModeKHR preferred_present_mode = vk::PresentModeKHR::eMailbox;
  /** @brief Preferred swapchain image format. */
  vk::Format preferred_format = vk::Format::eB8G8R8A8Srgb;
  /** @brief Preferred color space associated with @ref preferred_format. */
  vk::ColorSpaceKHR preferred_color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
  /** @brief Usage flags requested for swapchain images. */
  vk::ImageUsageFlags image_usage = vk::ImageUsageFlagBits::eColorAttachment;
  /** @brief Preferred composite-alpha mode. */
  vk::CompositeAlphaFlagBitsKHR composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
  /** @brief Requested upper bound for frame resources created in the frame loop. */
  std::uint32_t max_frames_in_flight = 2U;
};

/**
 * @brief Queue topology resolved for one surface-backed swapchain.
 */
struct SurfaceQueueTopology {
  /** @brief Graphics queue family index. */
  std::uint32_t graphics_family_index = 0U;
  /** @brief Graphics queue handle. */
  vk::Queue graphics_queue = VK_NULL_HANDLE;
  /** @brief Optional async compute queue family index. */
  std::optional<std::uint32_t> async_compute_family_index;
  /** @brief Optional async compute queue handle. */
  std::optional<vk::Queue> async_compute_queue;
  /** @brief Optional transfer queue family index. */
  std::optional<std::uint32_t> transfer_family_index;
  /** @brief Optional transfer queue handle. */
  std::optional<vk::Queue> transfer_queue;
  /** @brief Presentation queue family index. */
  std::uint32_t present_family_index = 0U;
  /** @brief Presentation queue handle. */
  vk::Queue present_queue = VK_NULL_HANDLE;
  /** @brief Indicates whether async compute is dedicated. */
  bool has_dedicated_async_compute = false;
  /** @brief Indicates whether transfer is dedicated. */
  bool has_dedicated_transfer = false;
  /** @brief True when graphics and present use the same family. */
  bool graphics_and_present_share_family = true;
  /** @brief Sharing mode required for swapchain images. */
  vk::SharingMode image_sharing_mode = vk::SharingMode::eExclusive;
  /** @brief Queue-family indices used when sharing mode is concurrent. */
  std::vector<std::uint32_t> image_sharing_family_indices;
};

/**
 * @brief Owning swapchain primitive with image-view cache.
 */
class SwapchainContext {
public:
  /**
   * @brief Create a swapchain and associated image views for one surface.
   * @param engine Initialized engine context.
   * @param surface Platform-owned surface context.
   * @param info Swapchain configuration.
   * @return Initialized swapchain context.
   * @throws std::runtime_error on capability or creation failures.
   */
  [[nodiscard]] static SwapchainContext create(const EngineContext &engine, const SurfaceContext &surface, const SwapchainCreateInfo &info = {});

  /**
   * @brief Recreate the swapchain for the same engine/surface pair.
   * @param info Requested swapchain preferences for the recreated object.
   * @return Recreated swapchain context.
   */
  [[nodiscard]] SwapchainContext recreate(const SwapchainCreateInfo &info) const;

  /**
   * @brief Recreate the swapchain using the previous creation preferences.
   * @return Recreated swapchain context.
   */
  [[nodiscard]] SwapchainContext recreate() const;

  /**
   * @brief Move-construct the swapchain context.
   * @param other Instance being moved from.
   */
  SwapchainContext(SwapchainContext &&other) noexcept = default;

  /**
   * @brief Move-assign the swapchain context.
   * @param other Instance being moved from.
   * @return `*this`.
   */
  SwapchainContext &operator=(SwapchainContext &&other) noexcept = default;

  SwapchainContext(const SwapchainContext &) = delete;
  SwapchainContext &operator=(const SwapchainContext &) = delete;

  /**
   * @brief Surface used for swapchain creation.
   * @return Non-owning Vulkan surface handle.
   */
  [[nodiscard]] vk::SurfaceKHR surface() const noexcept;

  /**
   * @brief Underlying Vulkan swapchain object.
   * @return Immutable RAII swapchain.
   */
  [[nodiscard]] const vk::raii::SwapchainKHR &swapchain() const noexcept;

  /**
   * @brief Swapchain image format.
   * @return Selected image format.
   */
  [[nodiscard]] vk::Format image_format() const noexcept;

  /**
   * @brief Swapchain color space.
   * @return Selected color space.
   */
  [[nodiscard]] vk::ColorSpaceKHR color_space() const noexcept;

  /**
   * @brief Swapchain extent.
   * @return Width/height in pixels.
   */
  [[nodiscard]] vk::Extent2D extent() const noexcept;

  /**
   * @brief Swapchain present mode.
   * @return Selected present mode.
   */
  [[nodiscard]] vk::PresentModeKHR present_mode() const noexcept;

  /**
   * @brief Number of swapchain images.
   * @return Image count.
   */
  [[nodiscard]] std::uint32_t image_count() const noexcept;

  /**
   * @brief Maximum number of frame resources expected to be used concurrently.
   * @return Max frames-in-flight value.
   */
  [[nodiscard]] std::uint32_t max_frames_in_flight() const noexcept;

  /**
   * @brief Queue-family index selected for presentation.
   * @return Present queue family index.
   */
  [[nodiscard]] std::uint32_t present_queue_family_index() const noexcept;

  /**
   * @brief Queue handle selected for presentation.
   * @return Present queue.
   */
  [[nodiscard]] vk::Queue present_queue() const noexcept;

  /**
   * @brief Surface-resolved queue topology used by this swapchain.
   * @return Immutable queue topology.
   */
  [[nodiscard]] const SurfaceQueueTopology &queue_topology() const noexcept;

  /**
   * @brief Swapchain image handles.
   * @return Span over non-owning image handles.
   */
  [[nodiscard]] std::span<const vk::Image> images() const noexcept;

  /**
   * @brief Cached image views for swapchain images.
   * @return Span over immutable RAII image-view objects.
   */
  [[nodiscard]] std::span<const vk::raii::ImageView> image_views() const noexcept;

private:
  /**
   * @brief Internal swapchain creation entry point shared by create/recreate.
   * @param engine Engine context that owns device resources.
   * @param surface_context Surface context associated with this swapchain.
   * @param info Creation preferences.
   * @param old_swapchain Previous swapchain handle for Vulkan recreation path.
   * @return Newly created swapchain context.
   */
  [[nodiscard]] static SwapchainContext create_internal(
      const EngineContext &engine,
      const SurfaceContext &surface_context,
      const SwapchainCreateInfo &info,
      vk::SwapchainKHR old_swapchain
  );

  /**
   * @brief Internal constructor from fully initialized objects.
   */
  SwapchainContext(
      const EngineContext *engine,
      const SurfaceContext *surface_context,
      vk::SurfaceKHR surface,
      vk::raii::SwapchainKHR &&swapchain,
      std::vector<vk::Image> &&images,
      std::vector<vk::raii::ImageView> &&image_views,
      vk::Format image_format,
      vk::ColorSpaceKHR color_space,
      vk::Extent2D extent,
      vk::PresentModeKHR present_mode,
      std::uint32_t max_frames_in_flight,
      SurfaceQueueTopology queue_topology,
      SwapchainCreateInfo create_info
  );

  const EngineContext *engine_ = nullptr;
  const SurfaceContext *surface_context_ = nullptr;
  vk::SurfaceKHR surface_ = VK_NULL_HANDLE;
  vk::raii::SwapchainKHR swapchain_{nullptr};
  std::vector<vk::Image> images_;
  std::vector<vk::raii::ImageView> image_views_;
  vk::Format image_format_ = vk::Format::eUndefined;
  vk::ColorSpaceKHR color_space_ = vk::ColorSpaceKHR::eSrgbNonlinear;
  vk::Extent2D extent_{};
  vk::PresentModeKHR present_mode_ = vk::PresentModeKHR::eFifo;
  std::uint32_t max_frames_in_flight_ = 1U;
  SurfaceQueueTopology queue_topology_{};
  SwapchainCreateInfo create_info_{};
};

} // namespace varre::engine
