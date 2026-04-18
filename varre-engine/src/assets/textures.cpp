/**
 * @file textures.cpp
 * @brief Texture upload implementation using stb_image, VMA, and Vulkan queue submissions.
 */
#include "varre/engine/assets/textures.hpp"

#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/errors.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <vk_mem_alloc.h>

namespace varre::engine {
namespace {
namespace detail {
/**
 * @brief Throw structured engine errors on Vulkan Memory Allocator failure.
 * @param result VMA result code.
 * @param context Error context string.
 */
void throw_on_vma_error(const VkResult result, const std::string_view context) {
  if (result == VK_SUCCESS) {
    return;
  }
  throw make_vulkan_result_error(static_cast<vk::Result>(result), context);
}

/**
 * @brief Throw when requested texture dimensions are invalid.
 * @param width Texture width.
 * @param height Texture height.
 * @param label Source label used for diagnostics.
 */
void validate_texture_dimensions(const std::uint32_t width, const std::uint32_t height, const std::string_view label) {
  if (width == 0U || height == 0U) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Texture '{}' has invalid dimensions {}x{}.", label, width, height));
  }
}

/**
 * @brief Checked multiplication of two 32-bit values.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @param context Error context string.
 * @return Product as `std::size_t`.
 */
[[nodiscard]] std::size_t checked_mul_u32(const std::uint32_t lhs, const std::uint32_t rhs, const std::string_view context) {
  const std::size_t product = static_cast<std::size_t>(lhs) * static_cast<std::size_t>(rhs);
  if (lhs != 0U && (product / static_cast<std::size_t>(lhs)) != static_cast<std::size_t>(rhs)) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Texture size overflow while computing {}.", context));
  }
  return product;
}

/**
 * @brief Allocate one primary command buffer from a command pool.
 * @param device Logical device.
 * @param command_pool Command pool handle.
 * @return RAII command buffer list with one buffer.
 */
[[nodiscard]] std::vector<vk::raii::CommandBuffer> allocate_one_time_command_buffer(const vk::raii::Device &device,
                                                                                    const vk::raii::CommandPool &command_pool) {
  const vk::CommandBufferAllocateInfo allocate_info =
    vk::CommandBufferAllocateInfo{}.setCommandPool(*command_pool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1U);
  return device.allocateCommandBuffers(allocate_info);
}

/**
 * @brief Submit one command buffer and block until completion.
 * @param device Logical device.
 * @param queue Submission queue.
 * @param command_buffer Command buffer to submit.
 * @param wait_context Wait error context.
 */
void submit_and_wait(const vk::raii::Device &device, const vk::Queue queue, const vk::CommandBuffer command_buffer,
                     const std::string_view wait_context) {
  vk::raii::Fence fence(device, vk::FenceCreateInfo{});
  const vk::SubmitInfo submit_info = vk::SubmitInfo{}.setCommandBuffers(command_buffer);
  queue.submit(submit_info, *fence);
  const std::array fences{*fence};
  const vk::Result wait_result = device.waitForFences(fences, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
  if (wait_result != vk::Result::eSuccess) {
    throw make_vulkan_result_error(wait_result, wait_context);
  }
}

/**
 * @brief Temporary staging-buffer allocation.
 */
struct StagingBuffer {
  StagingBuffer() = default;

  VmaAllocator allocator = nullptr;
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;

  /**
   * @brief Destroy staging-buffer allocation when valid.
   */
  ~StagingBuffer() {
    if (allocator != nullptr && buffer != VK_NULL_HANDLE && allocation != nullptr) {
      vmaDestroyBuffer(allocator, buffer, allocation);
    }
  }

  StagingBuffer(const StagingBuffer &) = delete;
  StagingBuffer &operator=(const StagingBuffer &) = delete;
  StagingBuffer(StagingBuffer &&) = delete;
  StagingBuffer &operator=(StagingBuffer &&) = delete;
};
} // namespace detail
} // namespace

GpuImage::GpuImage(const VmaAllocator allocator, const VkImage image, const VmaAllocation allocation, const vk::Extent3D extent) noexcept
    : allocator_(allocator), image_(image), allocation_(allocation), extent_(extent) {}

GpuImage::~GpuImage() { reset(); }

GpuImage::GpuImage(GpuImage &&other) noexcept
    : allocator_(other.allocator_), image_(other.image_), allocation_(other.allocation_), extent_(other.extent_) {
  other.allocator_ = nullptr;
  other.image_ = VK_NULL_HANDLE;
  other.allocation_ = nullptr;
  other.extent_ = vk::Extent3D{};
}

GpuImage &GpuImage::operator=(GpuImage &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  allocator_ = other.allocator_;
  image_ = other.image_;
  allocation_ = other.allocation_;
  extent_ = other.extent_;
  other.allocator_ = nullptr;
  other.image_ = VK_NULL_HANDLE;
  other.allocation_ = nullptr;
  other.extent_ = vk::Extent3D{};
  return *this;
}

bool GpuImage::valid() const noexcept { return image_ != VK_NULL_HANDLE && allocation_ != nullptr; }

vk::Image GpuImage::image() const noexcept { return vk::Image{image_}; }

vk::Extent3D GpuImage::extent() const noexcept { return extent_; }

void GpuImage::reset() noexcept {
  if (allocator_ != nullptr && image_ != VK_NULL_HANDLE && allocation_ != nullptr) {
    vmaDestroyImage(allocator_, image_, allocation_);
  }
  allocator_ = nullptr;
  image_ = VK_NULL_HANDLE;
  allocation_ = nullptr;
  extent_ = vk::Extent3D{};
}

TextureUploadService::TextureUploadService(const EngineContext *engine, const vk::raii::Device *device, UploadContext &&upload_context,
                                           const VmaAllocator allocator, const std::uint32_t submission_family_index,
                                           const std::uint32_t graphics_family_index, const vk::Queue submission_queue,
                                           const vk::Queue graphics_queue, vk::raii::CommandPool &&submission_command_pool,
                                           vk::raii::CommandPool &&graphics_command_pool, const bool perform_queue_family_ownership_transfer)
    : engine_(engine), device_(device), upload_context_(std::move(upload_context)), allocator_(allocator),
      submission_family_index_(submission_family_index), graphics_family_index_(graphics_family_index), submission_queue_(submission_queue),
      graphics_queue_(graphics_queue), submission_command_pool_(std::move(submission_command_pool)),
      graphics_command_pool_(std::move(graphics_command_pool)), perform_queue_family_ownership_transfer_(perform_queue_family_ownership_transfer) {}

TextureUploadService::~TextureUploadService() {
  if (allocator_ != nullptr) {
    try {
      wait_idle();
    } catch (...) {
    }
  }
  cached_textures_.clear();
  cached_paths_.clear();
  release_allocator();
}

TextureUploadService::TextureUploadService(TextureUploadService &&other) noexcept
    : engine_(other.engine_), device_(other.device_), upload_context_(std::move(other.upload_context_)), allocator_(other.allocator_),
      submission_family_index_(other.submission_family_index_), graphics_family_index_(other.graphics_family_index_),
      submission_queue_(other.submission_queue_), graphics_queue_(other.graphics_queue_),
      submission_command_pool_(std::move(other.submission_command_pool_)), graphics_command_pool_(std::move(other.graphics_command_pool_)),
      perform_queue_family_ownership_transfer_(other.perform_queue_family_ownership_transfer_), cached_paths_(std::move(other.cached_paths_)),
      cached_textures_(std::move(other.cached_textures_)) {
  other.engine_ = nullptr;
  other.device_ = nullptr;
  other.allocator_ = nullptr;
  other.submission_family_index_ = 0U;
  other.graphics_family_index_ = 0U;
  other.submission_queue_ = VK_NULL_HANDLE;
  other.graphics_queue_ = VK_NULL_HANDLE;
}

TextureUploadService &TextureUploadService::operator=(TextureUploadService &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  cached_textures_.clear();
  cached_paths_.clear();
  release_allocator();

  engine_ = other.engine_;
  device_ = other.device_;
  upload_context_ = std::move(other.upload_context_);
  allocator_ = other.allocator_;
  submission_family_index_ = other.submission_family_index_;
  graphics_family_index_ = other.graphics_family_index_;
  submission_queue_ = other.submission_queue_;
  graphics_queue_ = other.graphics_queue_;
  submission_command_pool_ = std::move(other.submission_command_pool_);
  graphics_command_pool_ = std::move(other.graphics_command_pool_);
  perform_queue_family_ownership_transfer_ = other.perform_queue_family_ownership_transfer_;
  cached_paths_ = std::move(other.cached_paths_);
  cached_textures_ = std::move(other.cached_textures_);

  other.engine_ = nullptr;
  other.device_ = nullptr;
  other.allocator_ = nullptr;
  other.submission_family_index_ = 0U;
  other.graphics_family_index_ = 0U;
  other.submission_queue_ = VK_NULL_HANDLE;
  other.graphics_queue_ = VK_NULL_HANDLE;
  return *this;
}

TextureUploadService TextureUploadService::create(const EngineContext &engine, const TextureUploadCreateInfo &info) {
  UploadContext upload_context = UploadContext::create(engine, UploadContextCreateInfo{
                                                                 .prefer_transfer_queue = info.prefer_transfer_queue,
                                                               });
  const vk::raii::Device &device = engine.device();
  const DeviceQueueTopology &topology = engine.device_queue_topology();
  const std::uint32_t submission_family_index = upload_context.submission_queue_family_index();
  const std::uint32_t graphics_family_index = topology.families.graphics;
  const vk::Queue submission_queue = upload_context.submission_queue();
  const vk::Queue graphics_queue = topology.graphics_queue;

  if (!info.perform_queue_family_ownership_transfer && submission_family_index != graphics_family_index) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            "TextureUploadService requires queue-family ownership transfer when upload and graphics families differ.");
  }

  const vk::CommandPoolCreateInfo command_pool_info = vk::CommandPoolCreateInfo{}
                                                        .setFlags(vk::CommandPoolCreateFlagBits::eTransient |
                                                                  vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                                                        .setQueueFamilyIndex(submission_family_index);
  vk::raii::CommandPool submission_command_pool(device, command_pool_info);

  vk::raii::CommandPool graphics_command_pool{nullptr};
  if (submission_family_index != graphics_family_index) {
    const vk::CommandPoolCreateInfo graphics_pool_info = vk::CommandPoolCreateInfo{}
                                                           .setFlags(vk::CommandPoolCreateFlagBits::eTransient |
                                                                     vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                                                           .setQueueFamilyIndex(graphics_family_index);
    graphics_command_pool = vk::raii::CommandPool(device, graphics_pool_info);
  }

  VmaAllocatorCreateInfo allocator_create_info{};
  allocator_create_info.instance = static_cast<VkInstance>(*engine.instance());
  allocator_create_info.physicalDevice = static_cast<VkPhysicalDevice>(engine.physical_device());
  allocator_create_info.device = static_cast<VkDevice>(*engine.device());
  allocator_create_info.vulkanApiVersion = engine.device_profile().api_version;

  VmaAllocator allocator = nullptr;
  const VkResult allocator_result = vmaCreateAllocator(&allocator_create_info, &allocator);
  detail::throw_on_vma_error(allocator_result, "Failed to create VMA allocator for TextureUploadService");

  return TextureUploadService{
    &engine,
    &device,
    std::move(upload_context),
    allocator,
    submission_family_index,
    graphics_family_index,
    submission_queue,
    graphics_queue,
    std::move(submission_command_pool),
    std::move(graphics_command_pool),
    info.perform_queue_family_ownership_transfer,
  };
}

GpuTexture TextureUploadService::upload_from_file(const std::string_view path, const TextureUploadRequest &request) {
  std::uint32_t width = 0U;
  std::uint32_t height = 0U;
  const std::vector<std::byte> rgba_pixels = decode_file_rgba8(path, &width, &height);
  return upload_rgba8(path, rgba_pixels, width, height, request);
}

const GpuTexture &TextureUploadService::get_or_upload_from_file(const std::string_view path) {
  const std::size_t existing_index = find_cached_index(path);
  if (existing_index != cached_paths_.size()) {
    return cached_textures_[existing_index];
  }
  cached_paths_.push_back(std::string{path});
  cached_textures_.push_back(upload_from_file(path));
  return cached_textures_.back();
}

GpuTexture TextureUploadService::upload_rgba8(const std::string_view source_path, const std::span<const std::byte> rgba_pixels,
                                              const std::uint32_t width, const std::uint32_t height, const TextureUploadRequest &request) {
  if (engine_ == nullptr || device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "TextureUploadService is not initialized.");
  }
  if (allocator_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "TextureUploadService allocator is not initialized.");
  }
  detail::validate_texture_dimensions(width, height, source_path);

  const std::size_t expected_bytes = detail::checked_mul_u32(width, height, "width*height") * 4U;
  if (rgba_pixels.size_bytes() != expected_bytes) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Texture '{}' has {} bytes but {} were expected for RGBA8 {}x{}.", source_path,
                                        rgba_pixels.size_bytes(), expected_bytes, width, height));
  }

  GpuImage image = create_device_local_image(width, height, request.format, request.additional_image_usage);
  upload_image_payload(image, rgba_pixels);

  GpuTexture texture{};
  texture.source_path = std::string{source_path};
  texture.image = std::move(image);
  texture.image_view = create_image_view(texture.image.image(), request.format);
  texture.sampler = create_sampler(request);
  texture.format = request.format;
  return texture;
}

void TextureUploadService::clear() {
  if (allocator_ == nullptr) {
    cached_textures_.clear();
    cached_paths_.clear();
    return;
  }
  wait_idle();
  cached_textures_.clear();
  cached_paths_.clear();
}

void TextureUploadService::wait_idle() const { upload_context_.wait_idle(); }

std::size_t TextureUploadService::size() const noexcept { return cached_textures_.size(); }

void TextureUploadService::release_allocator() noexcept {
  if (allocator_ != nullptr) {
    vmaDestroyAllocator(allocator_);
  }
  allocator_ = nullptr;
}

std::vector<std::byte> TextureUploadService::decode_file_rgba8(const std::string_view path, std::uint32_t *out_width, std::uint32_t *out_height) const {
  if (out_width == nullptr || out_height == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "decode_file_rgba8 requires non-null output dimensions.");
  }
  const std::string path_string{path};
  int decoded_width = 0;
  int decoded_height = 0;
  int decoded_channels = 0;
  std::unique_ptr<stbi_uc, void (*)(void *)> pixels{stbi_load(path_string.c_str(), &decoded_width, &decoded_height, &decoded_channels, STBI_rgb_alpha),
                                                     stbi_image_free};
  if (!pixels) {
    const char *reason = stbi_failure_reason();
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Failed to decode texture '{}': {}", path, reason != nullptr ? reason : "unknown stb_image failure"));
  }

  if (decoded_width <= 0 || decoded_height <= 0) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Texture '{}' decoded to invalid dimensions {}x{}.", path, decoded_width, decoded_height));
  }

  *out_width = static_cast<std::uint32_t>(decoded_width);
  *out_height = static_cast<std::uint32_t>(decoded_height);
  detail::validate_texture_dimensions(*out_width, *out_height, path);

  const std::size_t byte_count = detail::checked_mul_u32(*out_width, *out_height, "decoded width*height") * 4U;
  std::vector<std::byte> bytes(byte_count);
  std::memcpy(bytes.data(), pixels.get(), byte_count);
  return bytes;
}

GpuImage TextureUploadService::create_device_local_image(const std::uint32_t width, const std::uint32_t height, const vk::Format format,
                                                         const vk::ImageUsageFlags additional_usage) const {
  VkImageCreateInfo image_create_info{};
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.imageType = VK_IMAGE_TYPE_2D;
  image_create_info.format = static_cast<VkFormat>(format);
  image_create_info.extent = VkExtent3D{width, height, 1U};
  image_create_info.mipLevels = 1U;
  image_create_info.arrayLayers = 1U;
  image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage =
    static_cast<VkImageUsageFlags>(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | additional_usage);
  image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo allocation_create_info{};
  allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  const VkResult create_result = vmaCreateImage(allocator_, &image_create_info, &allocation_create_info, &image, &allocation, nullptr);
  detail::throw_on_vma_error(create_result, "Failed to create texture image allocation");

  return GpuImage{
    allocator_,
    image,
    allocation,
    vk::Extent3D{width, height, 1U},
  };
}

void TextureUploadService::upload_image_payload(const GpuImage &image, const std::span<const std::byte> payload) {
  if (!image.valid()) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "upload_image_payload requires a valid destination image.");
  }
  if (payload.empty()) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "upload_image_payload requires a non-empty payload.");
  }
  if (device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "TextureUploadService device is not initialized.");
  }

  detail::StagingBuffer staging{};
  staging.allocator = allocator_;

  VkBufferCreateInfo staging_buffer_create_info{};
  staging_buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  staging_buffer_create_info.size = static_cast<VkDeviceSize>(payload.size_bytes());
  staging_buffer_create_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  staging_buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo staging_alloc_create_info{};
  staging_alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  staging_alloc_create_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VmaAllocationInfo staging_alloc_info{};
  const VkResult staging_result = vmaCreateBuffer(allocator_, &staging_buffer_create_info, &staging_alloc_create_info, &staging.buffer,
                                                  &staging.allocation, &staging_alloc_info);
  detail::throw_on_vma_error(staging_result, "Failed to create texture staging buffer");
  if (staging_alloc_info.pMappedData == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "Texture staging allocation returned a null mapped pointer.");
  }
  std::memcpy(staging_alloc_info.pMappedData, payload.data(), payload.size_bytes());

  std::vector<vk::raii::CommandBuffer> submission_buffers = detail::allocate_one_time_command_buffer(*device_, submission_command_pool_);
  vk::raii::CommandBuffer &submission_cmd = submission_buffers.front();
  const vk::CommandBufferBeginInfo begin_info = vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  submission_cmd.begin(begin_info);

  const vk::ImageSubresourceRange subresource_range = vk::ImageSubresourceRange{}
                                                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                        .setBaseMipLevel(0U)
                                                        .setLevelCount(1U)
                                                        .setBaseArrayLayer(0U)
                                                        .setLayerCount(1U);

  const vk::ImageMemoryBarrier2 to_transfer = vk::ImageMemoryBarrier2{}
                                                .setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
                                                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                                                .setDstStageMask(vk::PipelineStageFlagBits2::eCopy)
                                                .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
                                                .setOldLayout(vk::ImageLayout::eUndefined)
                                                .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
                                                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                                .setImage(image.image())
                                                .setSubresourceRange(subresource_range);
  submission_cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(to_transfer));

  const vk::BufferImageCopy copy_region = vk::BufferImageCopy{}
                                            .setBufferOffset(0U)
                                            .setBufferRowLength(0U)
                                            .setBufferImageHeight(0U)
                                            .setImageSubresource(vk::ImageSubresourceLayers{}
                                                                   .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                                   .setMipLevel(0U)
                                                                   .setBaseArrayLayer(0U)
                                                                   .setLayerCount(1U))
                                            .setImageOffset(vk::Offset3D{0, 0, 0})
                                            .setImageExtent(image.extent());
  submission_cmd.copyBufferToImage(vk::Buffer{staging.buffer}, image.image(), vk::ImageLayout::eTransferDstOptimal, copy_region);

  const bool needs_ownership_transfer =
    perform_queue_family_ownership_transfer_ && submission_family_index_ != graphics_family_index_;
  if (needs_ownership_transfer) {
    const vk::ImageMemoryBarrier2 release_barrier = vk::ImageMemoryBarrier2{}
                                                      .setSrcStageMask(vk::PipelineStageFlagBits2::eCopy)
                                                      .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
                                                      .setDstStageMask(vk::PipelineStageFlagBits2::eNone)
                                                      .setDstAccessMask(vk::AccessFlagBits2::eNone)
                                                      .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
                                                      .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                                                      .setSrcQueueFamilyIndex(submission_family_index_)
                                                      .setDstQueueFamilyIndex(graphics_family_index_)
                                                      .setImage(image.image())
                                                      .setSubresourceRange(subresource_range);
    submission_cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(release_barrier));
  } else {
    const vk::ImageMemoryBarrier2 to_shader_read = vk::ImageMemoryBarrier2{}
                                                     .setSrcStageMask(vk::PipelineStageFlagBits2::eCopy)
                                                     .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
                                                     .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                                                     .setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
                                                     .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
                                                     .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                                                     .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                                     .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                                     .setImage(image.image())
                                                     .setSubresourceRange(subresource_range);
    submission_cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(to_shader_read));
  }

  submission_cmd.end();
  detail::submit_and_wait(*device_, submission_queue_, *submission_cmd, "Texture upload submit on submission queue did not complete successfully");

  if (needs_ownership_transfer) {
    std::vector<vk::raii::CommandBuffer> acquire_buffers = detail::allocate_one_time_command_buffer(*device_, graphics_command_pool_);
    vk::raii::CommandBuffer &acquire_cmd = acquire_buffers.front();
    acquire_cmd.begin(begin_info);
    const vk::ImageMemoryBarrier2 acquire_barrier = vk::ImageMemoryBarrier2{}
                                                      .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                                                      .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                                                      .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                                                      .setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
                                                      .setOldLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                                                      .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                                                      .setSrcQueueFamilyIndex(submission_family_index_)
                                                      .setDstQueueFamilyIndex(graphics_family_index_)
                                                      .setImage(image.image())
                                                      .setSubresourceRange(subresource_range);
    acquire_cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(acquire_barrier));
    acquire_cmd.end();
    detail::submit_and_wait(*device_, graphics_queue_, *acquire_cmd, "Texture upload ownership acquire did not complete successfully");
  }
}

vk::raii::ImageView TextureUploadService::create_image_view(const vk::Image image, const vk::Format format,
                                                           const vk::ImageAspectFlags aspect_mask) const {
  if (device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "TextureUploadService device is not initialized.");
  }
  const vk::ImageViewCreateInfo create_info = vk::ImageViewCreateInfo{}
                                                .setImage(image)
                                                .setViewType(vk::ImageViewType::e2D)
                                                .setFormat(format)
                                                .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                       .setAspectMask(aspect_mask)
                                                                       .setBaseMipLevel(0U)
                                                                       .setLevelCount(1U)
                                                                       .setBaseArrayLayer(0U)
                                                                       .setLayerCount(1U));
  return vk::raii::ImageView{*device_, create_info};
}

vk::raii::Sampler TextureUploadService::create_sampler(const TextureUploadRequest &request) const {
  if (device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "TextureUploadService device is not initialized.");
  }
  const vk::SamplerCreateInfo create_info = vk::SamplerCreateInfo{}
                                              .setMagFilter(request.mag_filter)
                                              .setMinFilter(request.min_filter)
                                              .setMipmapMode(vk::SamplerMipmapMode::eLinear)
                                              .setAddressModeU(request.address_mode_u)
                                              .setAddressModeV(request.address_mode_v)
                                              .setAddressModeW(request.address_mode_w)
                                              .setMipLodBias(0.0F)
                                              .setAnisotropyEnable(VK_FALSE)
                                              .setMaxAnisotropy(1.0F)
                                              .setCompareEnable(VK_FALSE)
                                              .setCompareOp(vk::CompareOp::eAlways)
                                              .setMinLod(0.0F)
                                              .setMaxLod(0.0F)
                                              .setBorderColor(vk::BorderColor::eFloatOpaqueWhite)
                                              .setUnnormalizedCoordinates(VK_FALSE);
  return vk::raii::Sampler{*device_, create_info};
}

std::size_t TextureUploadService::find_cached_index(const std::string_view path) const noexcept {
  const auto it = std::ranges::find(cached_paths_, path);
  if (it == cached_paths_.end()) {
    return cached_paths_.size();
  }
  return static_cast<std::size_t>(std::distance(cached_paths_.begin(), it));
}
} // namespace varre::engine
