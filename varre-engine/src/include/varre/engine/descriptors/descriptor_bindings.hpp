/**
 * @file descriptor_bindings.hpp
 * @brief Descriptor binding bridge that validates resources and emits descriptor writes.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/engine/descriptors/material_descriptors.hpp"

namespace varre::engine {

class EngineContext;

/**
 * @brief One buffer descriptor element used for descriptor writes.
 */
struct BufferDescriptorBindingResource {
  /** @brief Buffer handle bound to the descriptor slot. */
  vk::Buffer buffer = VK_NULL_HANDLE;
  /** @brief Byte offset into @ref buffer. */
  vk::DeviceSize offset = 0U;
  /** @brief Byte range used for the descriptor slot. */
  vk::DeviceSize range = VK_WHOLE_SIZE;
};

/**
 * @brief One combined-image-sampler descriptor element used for descriptor writes.
 */
struct CombinedImageSamplerDescriptorBindingResource {
  /** @brief Sampled image view handle. */
  vk::ImageView image_view = VK_NULL_HANDLE;
  /** @brief Sampler handle. */
  vk::Sampler sampler = VK_NULL_HANDLE;
  /** @brief Image layout used for sampling. */
  vk::ImageLayout image_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
};

/**
 * @brief One descriptor binding resource payload identified by descriptor set and binding index.
 */
struct DescriptorBindingResource {
  /** @brief Descriptor set index. */
  std::uint32_t set = 0U;
  /** @brief Descriptor binding index inside @ref set. */
  std::uint32_t binding = 0U;
  /** @brief Buffer resources for buffer descriptor types. */
  std::vector<BufferDescriptorBindingResource> buffers;
  /** @brief Combined image sampler resources for sampled image descriptor types. */
  std::vector<CombinedImageSamplerDescriptorBindingResource> combined_image_samplers;
};

/**
 * @brief Immutable descriptor-write batch with owned backing storage.
 *
 * The info vectors keep backing memory alive for `vk::WriteDescriptorSet` pointers.
 */
struct DescriptorWriteBatch {
  /** @brief Backing storage for all buffer descriptor infos referenced by @ref writes. */
  std::vector<vk::DescriptorBufferInfo> buffer_infos;
  /** @brief Backing storage for all image descriptor infos referenced by @ref writes. */
  std::vector<vk::DescriptorImageInfo> image_infos;
  /** @brief Descriptor write operations ready for `vkUpdateDescriptorSets`. */
  std::vector<vk::WriteDescriptorSet> writes;

  /**
   * @brief Number of descriptor write operations.
   * @return Write count.
   */
  [[nodiscard]] std::size_t size() const noexcept;

  /**
   * @brief Whether this batch has no writes.
   * @return `true` when @ref writes is empty.
   */
  [[nodiscard]] bool empty() const noexcept;
};

/**
 * @brief Descriptor binding bridge for material descriptor layouts.
 *
 * This bridge validates required bindings from `MaterialDescriptorLayout` and supports:
 * - `eUniformBuffer`
 * - `eUniformBufferDynamic`
 * - `eStorageBuffer`
 * - `eStorageBufferDynamic`
 * - `eCombinedImageSampler`
 */
class MaterialDescriptorBindingBridge {
public:
  /**
   * @brief Create a descriptor binding bridge for one engine device.
   * @param engine Initialized engine context.
   * @return Initialized bridge.
   */
  [[nodiscard]] static MaterialDescriptorBindingBridge create(const EngineContext &engine);

  /**
   * @brief Build a validated descriptor write batch for one material layout.
   * @param layout Material descriptor layout returned by resolver.
   * @param descriptor_sets Descriptor sets indexed by set number (`descriptor_sets[set]`).
   * @param resources Binding resources keyed by set+binding.
   * @return Descriptor write batch.
   */
  [[nodiscard]] DescriptorWriteBatch build_write_batch(const MaterialDescriptorLayout &layout, std::span<const vk::DescriptorSet> descriptor_sets,
                                                       std::span<const DescriptorBindingResource> resources) const;

  /**
   * @brief Apply one descriptor write batch to the underlying Vulkan device.
   * @param batch Descriptor write batch.
   */
  void apply_write_batch(const DescriptorWriteBatch &batch) const;

  /**
   * @brief Build and apply one descriptor write batch in one call.
   * @param layout Material descriptor layout returned by resolver.
   * @param descriptor_sets Descriptor sets indexed by set number (`descriptor_sets[set]`).
   * @param resources Binding resources keyed by set+binding.
   */
  void write(const MaterialDescriptorLayout &layout, std::span<const vk::DescriptorSet> descriptor_sets,
             std::span<const DescriptorBindingResource> resources) const;

private:
  /**
   * @brief Internal constructor from device pointer.
   */
  explicit MaterialDescriptorBindingBridge(const vk::raii::Device *device);

  const vk::raii::Device *device_ = nullptr;
};

} // namespace varre::engine

namespace varre::engine::descriptor {
using ::varre::engine::BufferDescriptorBindingResource;
using ::varre::engine::CombinedImageSamplerDescriptorBindingResource;
using ::varre::engine::DescriptorBindingResource;
using ::varre::engine::DescriptorWriteBatch;
using ::varre::engine::MaterialDescriptorBindingBridge;
} // namespace varre::engine::descriptor
