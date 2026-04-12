/**
 * @file surface.cpp
 * @brief Platform-facing surface ownership boundary implementation.
 */
#include "varre/engine/core/surface.hpp"

#include <stdexcept>
#include <utility>

namespace varre::engine {
SurfaceContext::SurfaceContext(vk::raii::SurfaceKHR &&surface) : surface_(std::move(surface)) {}

SurfaceContext SurfaceContext::adopt(const vk::raii::Instance &instance, const vk::SurfaceKHR surface) {
  if (surface == VK_NULL_HANDLE) {
    throw std::runtime_error("SurfaceContext::adopt requires a valid VkSurfaceKHR.");
  }
  return SurfaceContext{vk::raii::SurfaceKHR(instance, surface)};
}

vk::SurfaceKHR SurfaceContext::handle() const noexcept { return *surface_; }

const vk::raii::SurfaceKHR &SurfaceContext::surface() const noexcept { return surface_; }
} // namespace varre::engine
