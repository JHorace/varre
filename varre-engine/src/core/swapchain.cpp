/**
 * @file swapchain.cpp
 * @brief Swapchain primitive implementation.
 */
#include "varre/engine/core/swapchain.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <ranges>
#include <vector>

#include <fmt/format.h>

#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/surface.hpp"

namespace varre::engine {
namespace detail {
/**
 * @brief Queue candidate that can potentially present to a surface.
 */
struct QueueCandidate {
  std::uint32_t family_index = 0U;
  vk::Queue queue = VK_NULL_HANDLE;
};

/**
 * @brief Append queue candidate only if its family index is not present yet.
 * @param candidates Candidate list.
 * @param family_index Queue family index.
 * @param queue Queue handle.
 */
void append_unique_queue_candidate(std::vector<QueueCandidate> *candidates, const std::uint32_t family_index, const vk::Queue queue) {
  const bool exists = std::ranges::any_of(*candidates, [&](const QueueCandidate &candidate) { return candidate.family_index == family_index; });
  if (!exists && queue != VK_NULL_HANDLE) {
    candidates->push_back(QueueCandidate{.family_index = family_index, .queue = queue});
  }
}

/**
 * @brief Append queue-family index only if it has not been added yet.
 * @param indices Destination family-index list.
 * @param family_index Queue family index to append.
 */
void append_unique_family_index(std::vector<std::uint32_t> *indices, const std::uint32_t family_index) {
  const bool exists = std::ranges::any_of(*indices, [&](const std::uint32_t value) { return value == family_index; });
  if (!exists) {
    indices->push_back(family_index);
  }
}

/**
 * @brief Select present queue family among the queues that are already created.
 * @param engine Initialized engine context.
 * @param surface Target window surface.
 * @return Present-capable queue candidate.
 */
QueueCandidate select_present_queue(const EngineContext &engine, const vk::SurfaceKHR surface) {
  const DeviceQueueTopology &base_topology = engine.device_queue_topology();
  std::vector<QueueCandidate> candidates;
  candidates.reserve(3U);

  append_unique_queue_candidate(&candidates, base_topology.families.graphics, base_topology.graphics_queue);
  if (base_topology.families.async_compute.has_value() && base_topology.async_compute_queue.has_value()) {
    append_unique_queue_candidate(&candidates, *base_topology.families.async_compute, *base_topology.async_compute_queue);
  }
  if (base_topology.families.transfer.has_value() && base_topology.transfer_queue.has_value()) {
    append_unique_queue_candidate(&candidates, *base_topology.families.transfer, *base_topology.transfer_queue);
  }

  const vk::raii::PhysicalDevice &physical_device = engine.physical_device_raii();
  for (const QueueCandidate &candidate : candidates) {
    if (physical_device.getSurfaceSupportKHR(candidate.family_index, surface)) {
      return candidate;
    }
  }

  throw make_engine_error(EngineErrorCode::kSurfaceUnsupported,
                          "No present-capable queue exists among the queue families created in EngineContext. Recreate EngineContext with a compatible queue "
                          "configuration.");
}

/**
 * @brief Build surface-resolved queue topology from the engine queue topology.
 * @param engine Initialized engine context.
 * @param surface Target window surface.
 * @return Queue topology including present and sharing details.
 */
SwapchainQueueTopology resolve_surface_queue_topology(const EngineContext &engine, const vk::SurfaceKHR surface) {
  const DeviceQueueTopology &base_topology = engine.device_queue_topology();
  const QueueCandidate present_candidate = select_present_queue(engine, surface);

  SwapchainQueueTopology queue_topology{
    .graphics_family_index = base_topology.families.graphics,
    .graphics_queue = base_topology.graphics_queue,
    .async_compute_family_index = base_topology.families.async_compute,
    .async_compute_queue = base_topology.async_compute_queue,
    .transfer_family_index = base_topology.families.transfer,
    .transfer_queue = base_topology.transfer_queue,
    .present_family_index = present_candidate.family_index,
    .present_queue = present_candidate.queue,
    .has_dedicated_async_compute = base_topology.has_dedicated_async_compute,
    .has_dedicated_transfer = base_topology.has_dedicated_transfer,
  };

  queue_topology.graphics_and_present_share_family = queue_topology.graphics_family_index == queue_topology.present_family_index;
  queue_topology.image_sharing_mode = queue_topology.graphics_and_present_share_family ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent;

  if (queue_topology.image_sharing_mode == vk::SharingMode::eConcurrent) {
    append_unique_family_index(&queue_topology.image_sharing_family_indices, queue_topology.graphics_family_index);
    append_unique_family_index(&queue_topology.image_sharing_family_indices, queue_topology.present_family_index);
  }
  return queue_topology;
}

/**
 * @brief Pick an image count based on surface capabilities.
 * @param capabilities Surface capabilities.
 * @return Selected image count.
 */
std::uint32_t select_image_count(const vk::SurfaceCapabilitiesKHR &capabilities) {
  std::uint32_t image_count = capabilities.minImageCount + 1U;
  if (capabilities.maxImageCount > 0U && image_count > capabilities.maxImageCount) {
    image_count = capabilities.maxImageCount;
  }
  return image_count;
}

/**
 * @brief Clamp preferred extent to surface limits when extent is not fixed.
 * @param capabilities Surface capabilities.
 * @param preferred Preferred extent from caller.
 * @return Selected swapchain extent.
 */
vk::Extent2D select_extent(const vk::SurfaceCapabilitiesKHR &capabilities, const vk::Extent2D preferred) {
  if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
    return capabilities.currentExtent;
  }

  const std::uint32_t width = std::clamp(preferred.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
  const std::uint32_t height = std::clamp(preferred.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
  if (width == 0U || height == 0U) {
    throw make_engine_error(EngineErrorCode::kSurfaceUnsupported, "Swapchain extent resolved to zero; surface is likely minimized.");
  }
  return vk::Extent2D{width, height};
}

/**
 * @brief Choose a surface format with preferred fallback to the first supported one.
 * @param available_formats Formats reported by Vulkan.
 * @param preferred_format Preferred pixel format.
 * @param preferred_color_space Preferred color space.
 * @return Selected format record.
 */
vk::SurfaceFormatKHR select_surface_format(const std::vector<vk::SurfaceFormatKHR> &available_formats, const vk::Format preferred_format,
                                           const vk::ColorSpaceKHR preferred_color_space) {
  if (available_formats.empty()) {
    throw make_engine_error(EngineErrorCode::kSurfaceUnsupported, "Surface reports no supported swapchain image formats.");
  }

  const auto it = std::ranges::find_if(
    available_formats, [&](const vk::SurfaceFormatKHR format) { return format.format == preferred_format && format.colorSpace == preferred_color_space; });
  if (it != available_formats.end()) {
    return *it;
  }
  return available_formats.front();
}

/**
 * @brief Choose a present mode, falling back to FIFO when needed.
 * @param available_modes Modes reported by Vulkan.
 * @param preferred_mode Preferred mode.
 * @return Selected present mode.
 */
vk::PresentModeKHR select_present_mode(const std::vector<vk::PresentModeKHR> &available_modes, const vk::PresentModeKHR preferred_mode) {
  if (std::ranges::find(available_modes, preferred_mode) != available_modes.end()) {
    return preferred_mode;
  }
  if (std::ranges::find(available_modes, vk::PresentModeKHR::eFifo) != available_modes.end()) {
    return vk::PresentModeKHR::eFifo;
  }
  if (available_modes.empty()) {
    throw make_engine_error(EngineErrorCode::kSurfaceUnsupported, "Surface reports no supported present modes.");
  }
  return available_modes.front();
}

/**
 * @brief Pick a supported composite alpha mode.
 * @param capabilities Surface capabilities.
 * @param preferred Preferred composite alpha mode.
 * @return Selected composite alpha mode.
 */
vk::CompositeAlphaFlagBitsKHR select_composite_alpha(const vk::SurfaceCapabilitiesKHR &capabilities, const vk::CompositeAlphaFlagBitsKHR preferred) {
  const vk::CompositeAlphaFlagsKHR supported = capabilities.supportedCompositeAlpha;
  if (static_cast<bool>(supported & preferred)) {
    return preferred;
  }

  constexpr std::array fallback_order{
    vk::CompositeAlphaFlagBitsKHR::eOpaque,
    vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
    vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
    vk::CompositeAlphaFlagBitsKHR::eInherit,
  };
  for (const vk::CompositeAlphaFlagBitsKHR candidate : fallback_order) {
    if (static_cast<bool>(supported & candidate)) {
      return candidate;
    }
  }

  throw make_engine_error(EngineErrorCode::kSurfaceUnsupported, "Surface reports no supported composite alpha mode.");
}

/**
 * @brief Create color image views for all swapchain images.
 * @param device Logical device.
 * @param images Swapchain image handles.
 * @param format Swapchain image format.
 * @return Created image views.
 */
std::vector<vk::raii::ImageView> create_swapchain_image_views(const vk::raii::Device &device, const std::vector<vk::Image> &images, const vk::Format format) {
  std::vector<vk::raii::ImageView> views;
  views.reserve(images.size());
  for (const vk::Image image : images) {
    const vk::ImageViewCreateInfo view_create_info = vk::ImageViewCreateInfo{}
                                                       .setImage(image)
                                                       .setViewType(vk::ImageViewType::e2D)
                                                       .setFormat(format)
                                                       .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                              .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                                              .setBaseMipLevel(0U)
                                                                              .setLevelCount(1U)
                                                                              .setBaseArrayLayer(0U)
                                                                              .setLayerCount(1U));
    views.emplace_back(device, view_create_info);
  }
  return views;
}

} // namespace detail

SwapchainContext SwapchainContext::create_internal(const EngineContext &engine, const SurfaceContext &surface_context, const SwapchainCreateInfo &info,
                                                   const vk::SwapchainKHR old_swapchain) {
  const vk::SurfaceKHR surface = surface_context.handle();

  const vk::raii::PhysicalDevice &physical_device = engine.physical_device_raii();
  const vk::SurfaceCapabilitiesKHR capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);
  const std::vector<vk::SurfaceFormatKHR> available_formats = physical_device.getSurfaceFormatsKHR(surface);
  const std::vector<vk::PresentModeKHR> available_present_modes = physical_device.getSurfacePresentModesKHR(surface);
  const SwapchainQueueTopology queue_topology = detail::resolve_surface_queue_topology(engine, surface);

  const vk::SurfaceFormatKHR surface_format = detail::select_surface_format(available_formats, info.preferred_format, info.preferred_color_space);
  const vk::PresentModeKHR present_mode = detail::select_present_mode(available_present_modes, info.preferred_present_mode);
  const vk::Extent2D extent = detail::select_extent(capabilities, info.preferred_extent);
  const std::uint32_t image_count = detail::select_image_count(capabilities);
  const vk::CompositeAlphaFlagBitsKHR composite_alpha = detail::select_composite_alpha(capabilities, info.composite_alpha);

  if ((capabilities.supportedUsageFlags & info.image_usage) != info.image_usage) {
    throw make_engine_error(EngineErrorCode::kSurfaceUnsupported,
                            fmt::format("Surface does not support requested swapchain image usage flags: {:#x}", static_cast<std::uint32_t>(info.image_usage)));
  }

  vk::SwapchainCreateInfoKHR create_info = vk::SwapchainCreateInfoKHR{}
                                             .setSurface(surface)
                                             .setMinImageCount(image_count)
                                             .setImageFormat(surface_format.format)
                                             .setImageColorSpace(surface_format.colorSpace)
                                             .setImageExtent(extent)
                                             .setImageArrayLayers(1U)
                                             .setImageUsage(info.image_usage)
                                             .setPreTransform(capabilities.currentTransform)
                                             .setCompositeAlpha(composite_alpha)
                                             .setPresentMode(present_mode)
                                             .setClipped(VK_TRUE)
                                             .setOldSwapchain(old_swapchain);

  if (queue_topology.image_sharing_mode == vk::SharingMode::eConcurrent) {
    create_info = create_info.setImageSharingMode(queue_topology.image_sharing_mode).setQueueFamilyIndices(queue_topology.image_sharing_family_indices);
  } else {
    create_info = create_info.setImageSharingMode(vk::SharingMode::eExclusive);
  }

  vk::raii::SwapchainKHR swapchain(engine.device(), create_info);
  std::vector<vk::Image> images = swapchain.getImages();
  if (images.empty()) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "Vulkan created a swapchain with zero images.");
  }

  std::vector<vk::raii::ImageView> image_views = detail::create_swapchain_image_views(engine.device(), images, surface_format.format);
  const std::uint32_t max_frames_in_flight = std::clamp(info.max_frames_in_flight, 1U, static_cast<std::uint32_t>(images.size()));

  return SwapchainContext{
    &engine,
    &surface_context,
    surface,
    std::move(swapchain),
    std::move(images),
    std::move(image_views),
    surface_format.format,
    surface_format.colorSpace,
    extent,
    present_mode,
    max_frames_in_flight,
    queue_topology,
    info,
  };
}

SwapchainContext::SwapchainContext(const EngineContext *engine, const SurfaceContext *surface_context, const vk::SurfaceKHR surface,
                                   vk::raii::SwapchainKHR &&swapchain, std::vector<vk::Image> &&images, std::vector<vk::raii::ImageView> &&image_views,
                                   const vk::Format image_format, const vk::ColorSpaceKHR color_space, const vk::Extent2D extent,
                                   const vk::PresentModeKHR present_mode, const std::uint32_t max_frames_in_flight, SwapchainQueueTopology queue_topology,
                                   SwapchainCreateInfo create_info)
    : engine_(engine), surface_context_(surface_context), surface_(surface), swapchain_(std::move(swapchain)), images_(std::move(images)),
      image_views_(std::move(image_views)), image_format_(image_format), color_space_(color_space), extent_(extent), present_mode_(present_mode),
      max_frames_in_flight_(max_frames_in_flight), swapchain_queue_topology_(std::move(queue_topology)), create_info_(std::move(create_info)) {}

SwapchainContext SwapchainContext::create(const EngineContext &engine, const SurfaceContext &surface_context, const SwapchainCreateInfo &info) {
  return create_internal(engine, surface_context, info, vk::SwapchainKHR{});
}

vk::SurfaceKHR SwapchainContext::surface() const noexcept { return surface_; }

SwapchainContext SwapchainContext::recreate(const SwapchainCreateInfo &info) const {
  if (engine_ == nullptr || surface_context_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "SwapchainContext::recreate requires a swapchain created through SwapchainContext::create.");
  }
  return create_internal(*engine_, *surface_context_, info, *swapchain_);
}

SwapchainContext SwapchainContext::recreate() const { return recreate(create_info_); }

const vk::raii::SwapchainKHR &SwapchainContext::swapchain() const noexcept { return swapchain_; }

vk::Format SwapchainContext::image_format() const noexcept { return image_format_; }

vk::ColorSpaceKHR SwapchainContext::color_space() const noexcept { return color_space_; }

vk::Extent2D SwapchainContext::extent() const noexcept { return extent_; }

vk::PresentModeKHR SwapchainContext::present_mode() const noexcept { return present_mode_; }

std::uint32_t SwapchainContext::image_count() const noexcept { return static_cast<std::uint32_t>(images_.size()); }

std::uint32_t SwapchainContext::max_frames_in_flight() const noexcept { return max_frames_in_flight_; }

std::uint32_t SwapchainContext::present_queue_family_index() const noexcept { return swapchain_queue_topology_.present_family_index; }

vk::Queue SwapchainContext::present_queue() const noexcept { return swapchain_queue_topology_.present_queue; }

const SwapchainQueueTopology &SwapchainContext::swapchain_queue_topology() const noexcept { return swapchain_queue_topology_; }

const SwapchainQueueTopology &SwapchainContext::queue_topology() const noexcept { return swapchain_queue_topology(); }

std::span<const vk::Image> SwapchainContext::images() const noexcept { return images_; }

std::span<const vk::raii::ImageView> SwapchainContext::image_views() const noexcept { return image_views_; }
} // namespace varre::engine
