/**
 * @file surface.hpp
 * @brief Platform-facing surface ownership boundary.
 */
#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace varre::engine {

/**
 * @brief Owning Vulkan surface wrapper intended to be created by platform code.
 *
 * The platform layer (SDL/GLFW/etc) creates a raw `VkSurfaceKHR`, then adopts
 * it into this RAII type. Engine modules consume the resulting abstraction
 * without taking ownership of platform-windowing concerns.
 */
class SurfaceContext {
public:
  /**
   * @brief Adopt an externally created Vulkan surface handle.
   * @param instance Vulkan instance that owns the surface.
   * @param surface Raw surface handle produced by the platform layer.
   * @return RAII-managed surface context.
   * @throws std::runtime_error when @p surface is null.
   */
  [[nodiscard]] static SurfaceContext adopt(const vk::raii::Instance &instance, vk::SurfaceKHR surface);

  /**
   * @brief Move-construct the surface context.
   * @param other Instance being moved from.
   */
  SurfaceContext(SurfaceContext &&other) noexcept = default;

  /**
   * @brief Move-assign the surface context.
   * @param other Instance being moved from.
   * @return `*this`.
   */
  SurfaceContext &operator=(SurfaceContext &&other) noexcept = default;

  SurfaceContext(const SurfaceContext &) = delete;
  SurfaceContext &operator=(const SurfaceContext &) = delete;

  /**
   * @brief Borrow the raw Vulkan surface handle.
   * @return Non-owning surface handle.
   */
  [[nodiscard]] vk::SurfaceKHR handle() const noexcept;

  /**
   * @brief Access the underlying RAII surface object.
   * @return Immutable RAII surface.
   */
  [[nodiscard]] const vk::raii::SurfaceKHR &surface() const noexcept;

private:
  /**
   * @brief Internal constructor from an already adopted surface.
   * @param surface RAII surface object.
   */
  explicit SurfaceContext(vk::raii::SurfaceKHR &&surface);

  vk::raii::SurfaceKHR surface_{nullptr};
};

} // namespace varre::engine
