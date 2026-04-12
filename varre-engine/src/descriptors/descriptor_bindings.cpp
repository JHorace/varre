/**
 * @file descriptor_bindings.cpp
 * @brief Descriptor binding bridge implementation.
 */
#include "varre/engine/descriptors/descriptor_bindings.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/errors.hpp"

namespace varre::engine {
namespace detail {
/**
 * @brief Whether descriptor type is supported by the binding bridge.
 * @param descriptor_type Vulkan descriptor type.
 * @return `true` when descriptor type is supported.
 */
[[nodiscard]] bool is_supported_descriptor_type(const vk::DescriptorType descriptor_type) {
  switch (descriptor_type) {
  case vk::DescriptorType::eUniformBuffer:
  case vk::DescriptorType::eUniformBufferDynamic:
  case vk::DescriptorType::eStorageBuffer:
  case vk::DescriptorType::eStorageBufferDynamic:
  case vk::DescriptorType::eCombinedImageSampler:
    return true;
  default:
    return false;
  }
}

/**
 * @brief Whether descriptor type expects buffer descriptor infos.
 * @param descriptor_type Vulkan descriptor type.
 * @return `true` when descriptor type expects `vk::DescriptorBufferInfo`.
 */
[[nodiscard]] bool is_buffer_descriptor_type(const vk::DescriptorType descriptor_type) {
  switch (descriptor_type) {
  case vk::DescriptorType::eUniformBuffer:
  case vk::DescriptorType::eUniformBufferDynamic:
  case vk::DescriptorType::eStorageBuffer:
  case vk::DescriptorType::eStorageBufferDynamic:
    return true;
  default:
    return false;
  }
}

/**
 * @brief Find binding specification by set+binding in one material descriptor layout.
 * @param layout Material descriptor layout.
 * @param set Descriptor set index.
 * @param binding Descriptor binding index.
 * @return Pointer to binding specification when found.
 */
[[nodiscard]] const DescriptorBindingSpec *find_binding_spec(const MaterialDescriptorLayout &layout, const std::uint32_t set, const std::uint32_t binding) {
  for (const DescriptorSetLayoutSpec &set_spec : layout.descriptor_set_layout_specs) {
    if (set_spec.set != set) {
      continue;
    }
    for (const DescriptorBindingSpec &binding_spec : set_spec.bindings) {
      if (binding_spec.binding == binding) {
        return &binding_spec;
      }
    }
    return nullptr;
  }
  return nullptr;
}

/**
 * @brief Validate one buffer descriptor resource list.
 * @param resource Binding resource payload.
 * @param expected_count Expected descriptor count.
 * @param set Descriptor set index.
 * @param binding Descriptor binding index.
 */
void validate_buffer_resource_payload(const DescriptorBindingResource &resource, const std::uint32_t expected_count, const std::uint32_t set,
                                      const std::uint32_t binding) {
  if (!resource.combined_image_samplers.empty()) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Descriptor set {} binding {} expects buffer descriptors, but image descriptors were provided.", set, binding));
  }
  if (resource.buffers.size() != static_cast<std::size_t>(expected_count)) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Descriptor set {} binding {} expects {} buffer descriptor(s), but {} were provided.", set, binding, expected_count,
                                        resource.buffers.size()));
  }
  for (std::size_t index = 0; index < resource.buffers.size(); ++index) {
    const BufferDescriptorBindingResource &buffer_resource = resource.buffers[index];
    if (buffer_resource.buffer == VK_NULL_HANDLE) {
      throw make_engine_error(
        EngineErrorCode::kInvalidArgument,
        fmt::format("Descriptor set {} binding {} buffer element {} has an invalid buffer handle.", set, binding, index));
    }
    if (buffer_resource.range == 0U) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument,
                              fmt::format("Descriptor set {} binding {} buffer element {} has zero range.", set, binding, index));
    }
  }
}

/**
 * @brief Validate one image descriptor resource list.
 * @param resource Binding resource payload.
 * @param expected_count Expected descriptor count.
 * @param set Descriptor set index.
 * @param binding Descriptor binding index.
 */
void validate_image_resource_payload(const DescriptorBindingResource &resource, const std::uint32_t expected_count, const std::uint32_t set,
                                     const std::uint32_t binding) {
  if (!resource.buffers.empty()) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Descriptor set {} binding {} expects image descriptors, but buffer descriptors were provided.", set, binding));
  }
  if (resource.combined_image_samplers.size() != static_cast<std::size_t>(expected_count)) {
    throw make_engine_error(
      EngineErrorCode::kInvalidArgument,
      fmt::format("Descriptor set {} binding {} expects {} combined image sampler descriptor(s), but {} were provided.", set, binding, expected_count,
                  resource.combined_image_samplers.size()));
  }
  for (std::size_t index = 0; index < resource.combined_image_samplers.size(); ++index) {
    const CombinedImageSamplerDescriptorBindingResource &image_resource = resource.combined_image_samplers[index];
    if (image_resource.image_view == VK_NULL_HANDLE) {
      throw make_engine_error(
        EngineErrorCode::kInvalidArgument,
        fmt::format("Descriptor set {} binding {} image element {} has an invalid image-view handle.", set, binding, index));
    }
    if (image_resource.sampler == VK_NULL_HANDLE) {
      throw make_engine_error(
        EngineErrorCode::kInvalidArgument,
        fmt::format("Descriptor set {} binding {} image element {} has an invalid sampler handle.", set, binding, index));
    }
  }
}
} // namespace detail

std::size_t DescriptorWriteBatch::size() const noexcept { return writes.size(); }

bool DescriptorWriteBatch::empty() const noexcept { return writes.empty(); }

MaterialDescriptorBindingBridge::MaterialDescriptorBindingBridge(const vk::raii::Device *device) : device_(device) {}

MaterialDescriptorBindingBridge MaterialDescriptorBindingBridge::create(const EngineContext &engine) {
  return MaterialDescriptorBindingBridge{&engine.device()};
}

DescriptorWriteBatch MaterialDescriptorBindingBridge::build_write_batch(const MaterialDescriptorLayout &layout,
                                                                        const std::span<const vk::DescriptorSet> descriptor_sets,
                                                                        const std::span<const DescriptorBindingResource> resources) const {
  if (device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "MaterialDescriptorBindingBridge is not initialized.");
  }

  std::map<std::pair<std::uint32_t, std::uint32_t>, const DescriptorBindingResource *> resources_by_binding;
  for (const DescriptorBindingResource &resource : resources) {
    const auto key = std::make_pair(resource.set, resource.binding);
    auto [it, inserted] = resources_by_binding.try_emplace(key, &resource);
    if (!inserted) {
      throw make_engine_error(
        EngineErrorCode::kInvalidArgument,
        fmt::format("Descriptor resource list contains duplicate entries for set {} binding {}.", resource.set, resource.binding));
    }
  }

  for (const auto &[key, resource] : resources_by_binding) {
    static_cast<void>(resource);
    const DescriptorBindingSpec *binding_spec = detail::find_binding_spec(layout, key.first, key.second);
    if (binding_spec == nullptr) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument,
                              fmt::format("Descriptor resource provided for unknown set {} binding {}.", key.first, key.second));
    }
  }

  struct PendingWrite {
    vk::DescriptorSet dst_set = VK_NULL_HANDLE;
    std::uint32_t dst_binding = 0U;
    vk::DescriptorType descriptor_type = vk::DescriptorType::eSampler;
    std::uint32_t descriptor_count = 0U;
    std::size_t first_info_index = 0U;
    bool uses_buffer_infos = false;
  };

  DescriptorWriteBatch batch{};
  std::vector<PendingWrite> pending_writes;

  for (const DescriptorSetLayoutSpec &set_spec : layout.descriptor_set_layout_specs) {
    if (set_spec.bindings.empty()) {
      continue;
    }
    if (set_spec.set >= descriptor_sets.size()) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument,
                              fmt::format("Descriptor set {} is required by layout, but descriptor_sets contains only {} set(s).", set_spec.set,
                                          descriptor_sets.size()));
    }

    const vk::DescriptorSet dst_set = descriptor_sets[set_spec.set];
    if (dst_set == VK_NULL_HANDLE) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument,
                              fmt::format("Descriptor set {} handle is invalid for material descriptor write.", set_spec.set));
    }

    for (const DescriptorBindingSpec &binding_spec : set_spec.bindings) {
      if (!detail::is_supported_descriptor_type(binding_spec.descriptor_type)) {
        throw make_engine_error(
          EngineErrorCode::kMissingRequirement,
          fmt::format("Descriptor set {} binding {} uses unsupported descriptor type {} in MaterialDescriptorBindingBridge.", set_spec.set,
                      binding_spec.binding, vk::to_string(binding_spec.descriptor_type)));
      }

      const auto resource_it = resources_by_binding.find(std::make_pair(set_spec.set, binding_spec.binding));
      if (resource_it == resources_by_binding.end()) {
        throw make_engine_error(
          EngineErrorCode::kInvalidArgument,
          fmt::format("Descriptor set {} binding {} is required but no resource payload was provided.", set_spec.set, binding_spec.binding));
      }

      const DescriptorBindingResource &resource = *resource_it->second;
      if (detail::is_buffer_descriptor_type(binding_spec.descriptor_type)) {
        detail::validate_buffer_resource_payload(resource, binding_spec.descriptor_count, set_spec.set, binding_spec.binding);
        const std::size_t first_index = batch.buffer_infos.size();
        for (const BufferDescriptorBindingResource &buffer_resource : resource.buffers) {
          batch.buffer_infos.push_back(vk::DescriptorBufferInfo{}
                                         .setBuffer(buffer_resource.buffer)
                                         .setOffset(buffer_resource.offset)
                                         .setRange(buffer_resource.range));
        }
        pending_writes.push_back(PendingWrite{
          .dst_set = dst_set,
          .dst_binding = binding_spec.binding,
          .descriptor_type = binding_spec.descriptor_type,
          .descriptor_count = binding_spec.descriptor_count,
          .first_info_index = first_index,
          .uses_buffer_infos = true,
        });
      } else {
        detail::validate_image_resource_payload(resource, binding_spec.descriptor_count, set_spec.set, binding_spec.binding);
        const std::size_t first_index = batch.image_infos.size();
        for (const CombinedImageSamplerDescriptorBindingResource &image_resource : resource.combined_image_samplers) {
          batch.image_infos.push_back(vk::DescriptorImageInfo{}
                                        .setSampler(image_resource.sampler)
                                        .setImageView(image_resource.image_view)
                                        .setImageLayout(image_resource.image_layout));
        }
        pending_writes.push_back(PendingWrite{
          .dst_set = dst_set,
          .dst_binding = binding_spec.binding,
          .descriptor_type = binding_spec.descriptor_type,
          .descriptor_count = binding_spec.descriptor_count,
          .first_info_index = first_index,
          .uses_buffer_infos = false,
        });
      }
    }
  }

  batch.writes.reserve(pending_writes.size());
  for (const PendingWrite &pending : pending_writes) {
    vk::WriteDescriptorSet write = vk::WriteDescriptorSet{}
                                     .setDstSet(pending.dst_set)
                                     .setDstBinding(pending.dst_binding)
                                     .setDstArrayElement(0U)
                                     .setDescriptorType(pending.descriptor_type)
                                     .setDescriptorCount(pending.descriptor_count);
    if (pending.uses_buffer_infos) {
      write.setPBufferInfo(batch.buffer_infos.data() + pending.first_info_index);
    } else {
      write.setPImageInfo(batch.image_infos.data() + pending.first_info_index);
    }
    batch.writes.push_back(write);
  }

  return batch;
}

void MaterialDescriptorBindingBridge::apply_write_batch(const DescriptorWriteBatch &batch) const {
  if (device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "MaterialDescriptorBindingBridge is not initialized.");
  }
  if (batch.writes.empty()) {
    return;
  }
  device_->updateDescriptorSets(batch.writes, {});
}

void MaterialDescriptorBindingBridge::write(const MaterialDescriptorLayout &layout, const std::span<const vk::DescriptorSet> descriptor_sets,
                                            const std::span<const DescriptorBindingResource> resources) const {
  const DescriptorWriteBatch batch = build_write_batch(layout, descriptor_sets, resources);
  apply_write_batch(batch);
}
} // namespace varre::engine
