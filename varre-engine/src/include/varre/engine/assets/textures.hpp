/**
 * @file textures.hpp
 * @brief Texture upload API for decoding image files and creating sampled Vulkan textures.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/engine/sync/upload_context.hpp"

struct VmaAllocator_T;
struct VmaAllocation_T;
using VmaAllocator = VmaAllocator_T *;
using VmaAllocation = VmaAllocation_T *;

namespace varre::engine {

class EngineContext;

/**
 * @brief GPU image allocation owned by a VMA allocator.
 */
class GpuImage final {
public:
  /**
   * @brief Construct an empty GPU image handle.
   */
  GpuImage() = default;

  /**
   * @brief Destroy the VMA image allocation when valid.
   */
  ~GpuImage();

  /**
   * @brief Move-construct the GPU image.
   * @param other Image being moved from.
   */
  GpuImage(GpuImage &&other) noexcept;

  /**
   * @brief Move-assign the GPU image.
   * @param other Image being moved from.
   * @return `*this`.
   */
  GpuImage &operator=(GpuImage &&other) noexcept;

  GpuImage(const GpuImage &) = delete;
  GpuImage &operator=(const GpuImage &) = delete;

  /**
   * @brief Whether this object owns a valid image allocation.
   * @return `true` when image and allocation are valid.
   */
  [[nodiscard]] bool valid() const noexcept;

  /**
   * @brief Vulkan image handle.
   * @return Non-owning Vulkan image handle.
   */
  [[nodiscard]] vk::Image image() const noexcept;

  /**
   * @brief Image extent.
   * @return Width/height/depth extent.
   */
  [[nodiscard]] vk::Extent3D extent() const noexcept;

private:
  /**
   * @brief Construct from already-created VMA image allocation.
   */
  GpuImage(VmaAllocator allocator, VkImage image, VmaAllocation allocation, vk::Extent3D extent) noexcept;

  /**
   * @brief Release currently owned image allocation and reset to empty.
   */
  void reset() noexcept;

  VmaAllocator allocator_ = nullptr;
  VkImage image_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = nullptr;
  vk::Extent3D extent_{};

  friend class TextureUploadService;
};

/**
 * @brief One uploaded sampled texture resource.
 */
struct GpuTexture {
  /** @brief Path used to decode this texture when loaded from file. */
  std::string source_path;
  /** @brief Device-local texture image. */
  GpuImage image{};
  /** @brief Vulkan image view for shader sampling. */
  vk::raii::ImageView image_view{nullptr};
  /** @brief Vulkan sampler used for shader sampling. */
  vk::raii::Sampler sampler{nullptr};
  /** @brief Texture format. */
  vk::Format format = vk::Format::eR8G8B8A8Srgb;
};

/**
 * @brief Texture-upload service creation options.
 */
struct TextureUploadCreateInfo {
  /** @brief Prefer transfer queue for uploads when available. */
  bool prefer_transfer_queue = true;
  /** @brief Enable queue-family ownership transfer when upload queue differs from graphics queue. */
  bool perform_queue_family_ownership_transfer = true;
};

/**
 * @brief Texture upload request options.
 */
struct TextureUploadRequest {
  /** @brief Texture image format for decoded RGBA8 data. */
  vk::Format format = vk::Format::eR8G8B8A8Srgb;
  /** @brief Additional Vulkan usage flags added to the texture image. */
  vk::ImageUsageFlags additional_image_usage{};
  /** @brief Minification filter for the sampler. */
  vk::Filter min_filter = vk::Filter::eLinear;
  /** @brief Magnification filter for the sampler. */
  vk::Filter mag_filter = vk::Filter::eLinear;
  /** @brief Sampler U address mode. */
  vk::SamplerAddressMode address_mode_u = vk::SamplerAddressMode::eRepeat;
  /** @brief Sampler V address mode. */
  vk::SamplerAddressMode address_mode_v = vk::SamplerAddressMode::eRepeat;
  /** @brief Sampler W address mode. */
  vk::SamplerAddressMode address_mode_w = vk::SamplerAddressMode::eRepeat;
};

/**
 * @brief Service that decodes and uploads textures as sampled Vulkan images.
 */
class TextureUploadService {
public:
  /**
   * @brief Create a texture-upload service bound to one engine device.
   * @param engine Initialized engine context.
   * @param info Texture-upload creation options.
   * @return Initialized texture-upload service.
   */
  [[nodiscard]] static TextureUploadService create(const EngineContext &engine, const TextureUploadCreateInfo &info = {});

  /**
   * @brief Destroy cached textures and allocator resources.
   */
  ~TextureUploadService();

  /**
   * @brief Move-construct the service.
   * @param other Service being moved from.
   */
  TextureUploadService(TextureUploadService &&other) noexcept;

  /**
   * @brief Move-assign the service.
   * @param other Service being moved from.
   * @return `*this`.
   */
  TextureUploadService &operator=(TextureUploadService &&other) noexcept;

  TextureUploadService(const TextureUploadService &) = delete;
  TextureUploadService &operator=(const TextureUploadService &) = delete;

  /**
   * @brief Upload a texture from an image file path without caching.
   * @param path Image file path.
   * @param request Upload and sampler options.
   * @return Newly uploaded texture.
   */
  [[nodiscard]] GpuTexture upload_from_file(std::string_view path, const TextureUploadRequest &request = {});

  /**
   * @brief Resolve a cached texture by path or upload it from file.
   * @param path Image file path.
   * @return Immutable cached texture reference.
   */
  [[nodiscard]] const GpuTexture &get_or_upload_from_file(std::string_view path);

  /**
   * @brief Upload one RGBA8 texture payload.
   * @param source_path Optional source label used in diagnostics.
   * @param rgba_pixels RGBA8 byte payload.
   * @param width Texture width in texels.
   * @param height Texture height in texels.
   * @param request Upload and sampler options.
   * @return Newly uploaded texture.
   */
  [[nodiscard]] GpuTexture upload_rgba8(std::string_view source_path, std::span<const std::byte> rgba_pixels, std::uint32_t width,
                                        std::uint32_t height, const TextureUploadRequest &request = {});

  /**
   * @brief Destroy all cached textures after waiting for upload queues to go idle.
   */
  void clear();

  /**
   * @brief Wait until all upload/device queues are idle.
   */
  void wait_idle() const;

  /**
   * @brief Number of cached textures.
   * @return Cache size.
   */
  [[nodiscard]] std::size_t size() const noexcept;

  /**
   * @brief Create one device-local image using VMA.
   * @param width Texture width in texels.
   * @param height Texture height in texels.
   * @param format Texture format.
   * @param additional_usage Additional usage flags.
   * @return Allocated GPU image.
   */
  [[nodiscard]] GpuImage create_device_local_image(std::uint32_t width, std::uint32_t height, vk::Format format,
                                                   vk::ImageUsageFlags additional_usage = {}) const;

  /**
   * @brief Build a sampled image view for a texture image.
   * @param image Texture image handle.
   * @param format Texture format.
   * @param aspect_mask Image aspect mask (default is eColor).
   * @return Texture image view.
   */
  [[nodiscard]] vk::raii::ImageView create_image_view(vk::Image image, vk::Format format,
                                                      vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eColor) const;

  /**
   * @brief Build a sampler for a texture upload request.
   * @param request Texture upload request.
   * @return Texture sampler.
   */
  [[nodiscard]] vk::raii::Sampler create_sampler(const TextureUploadRequest &request) const;

  /**
   * @brief Find cached texture index by path.
   * @param path Cache lookup path.
   * @return Cached index when present; `cached_paths_.size()` otherwise.
   */
  [[nodiscard]] std::size_t find_cached_index(std::string_view path) const noexcept;

private:
  /**
   * @brief Internal constructor from prebuilt resources.
   */
  TextureUploadService(const EngineContext *engine, const vk::raii::Device *device, UploadContext &&upload_context, VmaAllocator allocator,
                       std::uint32_t submission_family_index, std::uint32_t graphics_family_index, vk::Queue submission_queue, vk::Queue graphics_queue,
                       vk::raii::CommandPool &&submission_command_pool, vk::raii::CommandPool &&graphics_command_pool,
                       bool perform_queue_family_ownership_transfer);

  /**
   * @brief Destroy allocator resources owned by this service.
   */
  void release_allocator() noexcept;

  /**
   * @brief Decode one image file to RGBA8 bytes.
   * @param path Image file path.
   * @param out_width Output texture width.
   * @param out_height Output texture height.
   * @return RGBA8 byte payload.
   */
  [[nodiscard]] std::vector<std::byte> decode_file_rgba8(std::string_view path, std::uint32_t *out_width, std::uint32_t *out_height) const;

  /**
   * @brief Upload one staging payload into a texture image and transition to shader-read layout.
   * @param image Destination GPU image.
   * @param payload Staging byte payload.
   */
  void upload_image_payload(const GpuImage &image, std::span<const std::byte> payload);

  const EngineContext *engine_ = nullptr;
  const vk::raii::Device *device_ = nullptr;
  UploadContext upload_context_;
  VmaAllocator allocator_ = nullptr;
  std::uint32_t submission_family_index_ = 0U;
  std::uint32_t graphics_family_index_ = 0U;
  vk::Queue submission_queue_ = VK_NULL_HANDLE;
  vk::Queue graphics_queue_ = VK_NULL_HANDLE;
  vk::raii::CommandPool submission_command_pool_{nullptr};
  vk::raii::CommandPool graphics_command_pool_{nullptr};
  bool perform_queue_family_ownership_transfer_ = true;
  std::vector<std::string> cached_paths_;
  std::vector<GpuTexture> cached_textures_;
};

} // namespace varre::engine

namespace varre::engine::asset {
using ::varre::engine::GpuImage;
using ::varre::engine::GpuTexture;
using ::varre::engine::TextureUploadCreateInfo;
using ::varre::engine::TextureUploadRequest;
using ::varre::engine::TextureUploadService;
} // namespace varre::engine::asset
