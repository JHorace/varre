/**
 * @file models.hpp
 * @brief Model asset upload API for creating VMA-backed GPU mesh buffers.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/assets/models.hpp"
#include "varre/engine/sync/upload_context.hpp"

struct VmaAllocator_T;
struct VmaAllocation_T;
using VmaAllocator = VmaAllocator_T *;
using VmaAllocation = VmaAllocation_T *;

namespace varre::engine {

class EngineContext;

/**
 * @brief GPU buffer allocation owned by a VMA allocator.
 */
class GpuBuffer final {
public:
  /**
   * @brief Construct an empty GPU buffer handle.
   */
  GpuBuffer() = default;

  /**
   * @brief Destroy the VMA allocation and buffer when valid.
   */
  ~GpuBuffer();

  /**
   * @brief Move-construct the GPU buffer.
   * @param other Buffer being moved from.
   */
  GpuBuffer(GpuBuffer &&other) noexcept;

  /**
   * @brief Move-assign the GPU buffer.
   * @param other Buffer being moved from.
   * @return `*this`.
   */
  GpuBuffer &operator=(GpuBuffer &&other) noexcept;

  GpuBuffer(const GpuBuffer &) = delete;
  GpuBuffer &operator=(const GpuBuffer &) = delete;

  /**
   * @brief Whether this object owns a valid GPU buffer allocation.
   * @return `true` when both buffer and allocation are valid.
   */
  [[nodiscard]] bool valid() const noexcept;

  /**
   * @brief Vulkan buffer handle.
   * @return Non-owning Vulkan buffer handle.
   */
  [[nodiscard]] vk::Buffer buffer() const noexcept;

  /**
   * @brief Allocated byte size.
   * @return Buffer size in bytes.
   */
  [[nodiscard]] vk::DeviceSize size_bytes() const noexcept;

private:
  /**
   * @brief Construct from already-created VMA allocation state.
   */
  GpuBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, vk::DeviceSize size_bytes) noexcept;

  /**
   * @brief Release currently owned allocation and reset to empty.
   */
  void reset() noexcept;

  VmaAllocator allocator_ = nullptr;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = nullptr;
  vk::DeviceSize size_bytes_ = 0U;

  friend class ModelUploadService;
};

/**
 * @brief One uploaded mesh model stored in GPU vertex/index buffers.
 */
struct GpuMesh {
  /** @brief Source model asset identifier. */
  varre::assets::ModelId model_id = static_cast<varre::assets::ModelId>(0U);
  /** @brief Device-local vertex buffer. */
  GpuBuffer vertex_buffer{};
  /** @brief Device-local index buffer. */
  GpuBuffer index_buffer{};
  /** @brief Vertex count in @ref vertex_buffer. */
  std::uint32_t vertex_count = 0U;
  /** @brief Index count in @ref index_buffer. */
  std::uint32_t index_count = 0U;
  /** @brief Index type for draw indexed commands. */
  vk::IndexType index_type = vk::IndexType::eUint32;
};

/**
 * @brief Model-upload service creation options.
 */
struct ModelUploadCreateInfo {
  /** @brief Prefer transfer queue for uploads when available. */
  bool prefer_transfer_queue = true;
};

/**
 * @brief Service that uploads generated model assets into GPU buffers.
 */
class ModelUploadService {
public:
  /**
   * @brief Create a model-upload service bound to one engine device.
   * @param engine Initialized engine context.
   * @param info Model-upload creation options.
   * @return Initialized model-upload service.
   */
  [[nodiscard]] static ModelUploadService create(const EngineContext &engine, const ModelUploadCreateInfo &info = {});

  /**
   * @brief Destroy all cached meshes and allocator resources.
   */
  ~ModelUploadService();

  /**
   * @brief Move-construct the model-upload service.
   * @param other Service being moved from.
   */
  ModelUploadService(ModelUploadService &&other) noexcept;

  /**
   * @brief Move-assign the model-upload service.
   * @param other Service being moved from.
   * @return `*this`.
   */
  ModelUploadService &operator=(ModelUploadService &&other) noexcept;

  ModelUploadService(const ModelUploadService &) = delete;
  ModelUploadService &operator=(const ModelUploadService &) = delete;

  /**
   * @brief Resolve one model asset and return cached or newly uploaded GPU mesh buffers.
   * @param model_id Model asset identifier.
   * @return Immutable uploaded mesh reference.
   */
  [[nodiscard]] const GpuMesh &get_or_upload(varre::assets::ModelId model_id);

  /**
   * @brief Upload one already decoded model asset into GPU buffers.
   * @param model Decoded model asset.
   * @return Newly uploaded GPU mesh.
   */
  [[nodiscard]] GpuMesh upload(const varre::assets::ModelAsset &model);

  /**
   * @brief Upload one already decoded model asset and cache the result by model ID.
   * @param model Decoded model asset.
   * @return Immutable uploaded mesh reference.
   */
  [[nodiscard]] const GpuMesh &upload_and_cache(const varre::assets::ModelAsset &model);

  /**
   * @brief Destroy all cached GPU meshes after waiting for upload queues to go idle.
   */
  void clear();

  /**
   * @brief Wait until all upload/device queues are idle.
   */
  void wait_idle() const;

  /**
   * @brief Number of cached model meshes.
   * @return Cache size.
   */
  [[nodiscard]] std::size_t size() const noexcept;

private:
  /**
   * @brief Internal constructor from prebuilt upload context and allocator.
   */
  ModelUploadService(const EngineContext *engine, UploadContext &&upload_context, VmaAllocator allocator);

  /**
   * @brief Destroy allocator resources owned by this service.
   */
  void release_allocator() noexcept;

  /**
   * @brief Create one device-local Vulkan buffer using VMA.
   * @param size_bytes Requested buffer size in bytes.
   * @param usage Vulkan usage flags.
   * @return Allocated GPU buffer.
   */
  [[nodiscard]] GpuBuffer create_device_local_buffer(vk::DeviceSize size_bytes, vk::BufferUsageFlags usage) const;

  /**
   * @brief Upload raw bytes into one GPU buffer with a destination visibility barrier.
   * @param destination Destination GPU buffer.
   * @param data Byte range to upload.
   * @param destination_access_mask Destination access mask.
   */
  void upload_bytes_to_buffer(const GpuBuffer &destination, std::span<const std::byte> data, vk::AccessFlags2 destination_access_mask);

  /**
   * @brief Find the cached model index for one model identifier.
   * @param model_id Model identifier.
   * @return Cached index when present.
   */
  [[nodiscard]] std::size_t find_cached_index(varre::assets::ModelId model_id) const noexcept;

  const EngineContext *engine_ = nullptr;
  UploadContext upload_context_;
  VmaAllocator allocator_ = nullptr;
  std::vector<varre::assets::ModelId> cached_ids_;
  std::vector<GpuMesh> cached_meshes_;
};

} // namespace varre::engine

namespace varre::engine::asset {
using ::varre::engine::GpuBuffer;
using ::varre::engine::GpuMesh;
using ::varre::engine::ModelUploadCreateInfo;
using ::varre::engine::ModelUploadService;
} // namespace varre::engine::asset
