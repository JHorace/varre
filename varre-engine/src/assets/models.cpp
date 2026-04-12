/**
 * @file models.cpp
 * @brief Model asset upload implementation backed by VMA and UploadContext.
 */
#include "varre/engine/assets/models.hpp"

#include <exception>
#include <limits>
#include <ranges>
#include <string_view>
#include <utility>

#include <fmt/format.h>

#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/errors.hpp"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace varre::engine {
namespace detail {
/**
 * @brief Convert Vulkan Memory Allocator result into structured engine errors.
 * @param result VMA result.
 * @param context Error context message.
 */
void throw_on_vma_error(const VkResult result, const std::string_view context) {
  if (result == VK_SUCCESS) {
    return;
  }
  throw make_vulkan_result_error(static_cast<vk::Result>(result), context);
}

/**
 * @brief Checked narrowing cast to `std::uint32_t`.
 * @param value Input size value.
 * @param context Error context message.
 * @return Narrowed 32-bit unsigned value.
 */
[[nodiscard]] std::uint32_t checked_u32(const std::size_t value, const std::string_view context) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("{} exceeds uint32_t range ({}).", context, static_cast<unsigned long long>(value)));
  }
  return static_cast<std::uint32_t>(value);
}
} // namespace detail

GpuBuffer::GpuBuffer(const VmaAllocator allocator, const VkBuffer buffer, const VmaAllocation allocation, const vk::DeviceSize size_bytes) noexcept
    : allocator_(allocator), buffer_(buffer), allocation_(allocation), size_bytes_(size_bytes) {}

GpuBuffer::~GpuBuffer() { reset(); }

GpuBuffer::GpuBuffer(GpuBuffer &&other) noexcept
    : allocator_(other.allocator_), buffer_(other.buffer_), allocation_(other.allocation_), size_bytes_(other.size_bytes_) {
  other.allocator_ = nullptr;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = nullptr;
  other.size_bytes_ = 0U;
}

GpuBuffer &GpuBuffer::operator=(GpuBuffer &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  allocator_ = other.allocator_;
  buffer_ = other.buffer_;
  allocation_ = other.allocation_;
  size_bytes_ = other.size_bytes_;
  other.allocator_ = nullptr;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = nullptr;
  other.size_bytes_ = 0U;
  return *this;
}

bool GpuBuffer::valid() const noexcept { return buffer_ != VK_NULL_HANDLE && allocation_ != nullptr; }

vk::Buffer GpuBuffer::buffer() const noexcept { return vk::Buffer{buffer_}; }

vk::DeviceSize GpuBuffer::size_bytes() const noexcept { return size_bytes_; }

void GpuBuffer::reset() noexcept {
  if (allocator_ != nullptr && buffer_ != VK_NULL_HANDLE && allocation_ != nullptr) {
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
  }
  allocator_ = nullptr;
  buffer_ = VK_NULL_HANDLE;
  allocation_ = nullptr;
  size_bytes_ = 0U;
}

ModelUploadService::ModelUploadService(const EngineContext *engine, UploadContext &&upload_context, const VmaAllocator allocator)
    : engine_(engine), upload_context_(std::move(upload_context)), allocator_(allocator) {}

ModelUploadService::~ModelUploadService() {
  if (allocator_ != nullptr) {
    try {
      wait_idle();
    } catch (...) {
    }
  }
  cached_meshes_.clear();
  cached_ids_.clear();
  release_allocator();
}

ModelUploadService::ModelUploadService(ModelUploadService &&other) noexcept
    : engine_(other.engine_), upload_context_(std::move(other.upload_context_)), allocator_(other.allocator_),
      cached_ids_(std::move(other.cached_ids_)), cached_meshes_(std::move(other.cached_meshes_)) {
  other.engine_ = nullptr;
  other.allocator_ = nullptr;
}

ModelUploadService &ModelUploadService::operator=(ModelUploadService &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  cached_meshes_.clear();
  cached_ids_.clear();
  release_allocator();
  engine_ = other.engine_;
  upload_context_ = std::move(other.upload_context_);
  allocator_ = other.allocator_;
  cached_ids_ = std::move(other.cached_ids_);
  cached_meshes_ = std::move(other.cached_meshes_);
  other.engine_ = nullptr;
  other.allocator_ = nullptr;
  return *this;
}

ModelUploadService ModelUploadService::create(const EngineContext &engine, const ModelUploadCreateInfo &info) {
  UploadContext upload_context = UploadContext::create(engine, UploadContextCreateInfo{
                                                         .prefer_transfer_queue = info.prefer_transfer_queue,
                                                       });

  VmaAllocatorCreateInfo allocator_create_info{};
  allocator_create_info.instance = static_cast<VkInstance>(*engine.instance());
  allocator_create_info.physicalDevice = static_cast<VkPhysicalDevice>(engine.physical_device());
  allocator_create_info.device = static_cast<VkDevice>(*engine.device());
  allocator_create_info.vulkanApiVersion = engine.device_profile().api_version;

  VmaAllocator allocator = nullptr;
  const VkResult allocator_result = vmaCreateAllocator(&allocator_create_info, &allocator);
  detail::throw_on_vma_error(allocator_result, "Failed to create VMA allocator for ModelUploadService");

  return ModelUploadService{
    &engine,
    std::move(upload_context),
    allocator,
  };
}

const GpuMesh &ModelUploadService::get_or_upload(const varre::assets::ModelId model_id) {
  const std::size_t existing_index = find_cached_index(model_id);
  if (existing_index != cached_ids_.size()) {
    return cached_meshes_[existing_index];
  }

  varre::assets::ModelAsset model{};
  try {
    model = varre::assets::load_model(model_id);
  } catch (const EngineError &) {
    throw;
  } catch (const std::exception &error) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Failed to load model '{}' (id={}): {}", varre::assets::model_name(model_id),
                                        static_cast<std::uint32_t>(model_id), error.what()));
  }
  return upload_and_cache(model);
}

GpuMesh ModelUploadService::upload(const varre::assets::ModelAsset &model) {
  if (engine_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "ModelUploadService is not initialized.");
  }
  if (allocator_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "ModelUploadService allocator is not initialized.");
  }
  if (model.vertices.empty()) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Model '{}' has no vertices and cannot be uploaded.", varre::assets::model_name(model.id)));
  }

  const std::span<const varre::assets::Vertex> vertex_span{model.vertices.data(), model.vertices.size()};
  const std::span<const std::byte> vertex_bytes = std::as_bytes(vertex_span);
  GpuBuffer vertex_buffer = create_device_local_buffer(
    static_cast<vk::DeviceSize>(vertex_bytes.size_bytes()),
    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer);

  upload_bytes_to_buffer(vertex_buffer, vertex_bytes, vk::AccessFlagBits2::eVertexAttributeRead);

  GpuBuffer index_buffer{};
  std::uint32_t index_count = 0U;
  if (!model.indices.empty()) {
    const std::span<const std::uint32_t> index_span{model.indices.data(), model.indices.size()};
    const std::span<const std::byte> index_bytes = std::as_bytes(index_span);
    index_buffer = create_device_local_buffer(
      static_cast<vk::DeviceSize>(index_bytes.size_bytes()),
      vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer);
    upload_bytes_to_buffer(index_buffer, index_bytes, vk::AccessFlagBits2::eIndexRead);
    index_count = detail::checked_u32(model.indices.size(), "Model index count");
  }

  GpuMesh mesh{};
  mesh.model_id = model.id;
  mesh.vertex_buffer = std::move(vertex_buffer);
  mesh.index_buffer = std::move(index_buffer);
  mesh.vertex_count = detail::checked_u32(model.vertices.size(), "Model vertex count");
  mesh.index_count = index_count;
  mesh.index_type = vk::IndexType::eUint32;
  return mesh;
}

const GpuMesh &ModelUploadService::upload_and_cache(const varre::assets::ModelAsset &model) {
  const std::size_t existing_index = find_cached_index(model.id);
  if (existing_index != cached_ids_.size()) {
    return cached_meshes_[existing_index];
  }
  cached_ids_.push_back(model.id);
  cached_meshes_.push_back(upload(model));
  return cached_meshes_.back();
}

void ModelUploadService::clear() {
  if (allocator_ == nullptr) {
    cached_meshes_.clear();
    cached_ids_.clear();
    return;
  }
  wait_idle();
  cached_meshes_.clear();
  cached_ids_.clear();
}

void ModelUploadService::wait_idle() const { upload_context_.wait_idle(); }

std::size_t ModelUploadService::size() const noexcept { return cached_meshes_.size(); }

void ModelUploadService::release_allocator() noexcept {
  if (allocator_ != nullptr) {
    vmaDestroyAllocator(allocator_);
  }
  allocator_ = nullptr;
}

GpuBuffer ModelUploadService::create_device_local_buffer(const vk::DeviceSize size_bytes, const vk::BufferUsageFlags usage) const {
  if (allocator_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "ModelUploadService allocator is not initialized.");
  }
  if (size_bytes == 0U) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "create_device_local_buffer requires a non-zero size.");
  }

  VkBufferCreateInfo buffer_create_info{};
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.size = static_cast<VkDeviceSize>(size_bytes);
  buffer_create_info.usage = static_cast<VkBufferUsageFlags>(usage);
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocation_create_info{};
  allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  const VkResult create_result = vmaCreateBuffer(allocator_, &buffer_create_info, &allocation_create_info, &buffer, &allocation, nullptr);
  detail::throw_on_vma_error(create_result, "Failed to create a model GPU buffer");

  return GpuBuffer{
    allocator_,
    buffer,
    allocation,
    size_bytes,
  };
}

void ModelUploadService::upload_bytes_to_buffer(const GpuBuffer &destination, const std::span<const std::byte> data,
                                                const vk::AccessFlags2 destination_access_mask) {
  if (!destination.valid()) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "upload_bytes_to_buffer requires a valid destination buffer.");
  }
  if (data.empty()) {
    return;
  }

  try {
    upload_context_.upload_buffer(UploadBufferRequest{
      .destination_buffer = destination.buffer(),
      .destination_offset = 0U,
      .source_data = data,
      .perform_queue_family_ownership_transfer = true,
      .destination_stage_mask = vk::PipelineStageFlagBits2::eVertexInput,
      .destination_access_mask = destination_access_mask,
    });
  } catch (const EngineError &) {
    throw;
  } catch (const std::exception &error) {
    throw make_engine_error(EngineErrorCode::kInvalidState, fmt::format("Model buffer upload failed: {}", error.what()));
  }
}

std::size_t ModelUploadService::find_cached_index(const varre::assets::ModelId model_id) const noexcept {
  const auto it = std::ranges::find(cached_ids_, model_id);
  if (it == cached_ids_.end()) {
    return cached_ids_.size();
  }
  return static_cast<std::size_t>(std::distance(cached_ids_.begin(), it));
}
} // namespace varre::engine
