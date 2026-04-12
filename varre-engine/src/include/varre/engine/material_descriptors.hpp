/**
 * @file material_descriptors.hpp
 * @brief Material/pipeline descriptor interface built on descriptor caches.
 */
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/assets/shaders.hpp"
#include "varre/engine/descriptors.hpp"

namespace varre::engine {

class EngineContext;

/**
 * @brief Descriptor-interface request using already resolved shader asset views.
 */
struct MaterialDescriptorRequest {
  /** @brief Shader asset views used to build descriptor set layouts. */
  std::span<const varre::assets::ShaderAssetView> shaders;
  /** @brief Push-constant ranges for pipeline layout creation. */
  std::vector<vk::PushConstantRange> push_constant_ranges;
  /** @brief Flags used for all descriptor set layout creations. */
  vk::DescriptorSetLayoutCreateFlags descriptor_set_layout_create_flags{};
  /** @brief Flags used for pipeline layout creation. */
  vk::PipelineLayoutCreateFlags pipeline_layout_create_flags{};
};

/**
 * @brief Descriptor-interface request using generated shader identifiers.
 */
struct MaterialDescriptorShaderIdRequest {
  /** @brief Shader IDs resolved through `varre::assets::get_shader`. */
  std::span<const varre::assets::ShaderId> shader_ids;
  /** @brief Push-constant ranges for pipeline layout creation. */
  std::vector<vk::PushConstantRange> push_constant_ranges;
  /** @brief Flags used for all descriptor set layout creations. */
  vk::DescriptorSetLayoutCreateFlags descriptor_set_layout_create_flags{};
  /** @brief Flags used for pipeline layout creation. */
  vk::PipelineLayoutCreateFlags pipeline_layout_create_flags{};
};

/**
 * @brief Material/pipeline descriptor result with cache-backed Vulkan handles.
 */
struct MaterialDescriptorLayout {
  /** @brief Descriptor set-layout specs in pipeline-set-index order. */
  std::vector<DescriptorSetLayoutSpec> descriptor_set_layout_specs;
  /** @brief Descriptor set layouts in pipeline-set-index order. */
  std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
  /** @brief Cached pipeline layout handle. */
  vk::PipelineLayout pipeline_layout = VK_NULL_HANDLE;
};

/**
 * @brief High-level interface for material and pipeline descriptor layout lookup.
 */
class MaterialPipelineDescriptorInterface {
public:
  /**
   * @brief Create a material/pipeline descriptor interface instance.
   * @param engine Initialized engine context.
   * @return Initialized interface.
   */
  [[nodiscard]] static MaterialPipelineDescriptorInterface create(const EngineContext &engine);

  /**
   * @brief Move-construct the interface.
   * @param other Interface being moved from.
   */
  MaterialPipelineDescriptorInterface(MaterialPipelineDescriptorInterface &&other) noexcept = default;

  /**
   * @brief Move-assign the interface.
   * @param other Interface being moved from.
   * @return `*this`.
   */
  MaterialPipelineDescriptorInterface &operator=(MaterialPipelineDescriptorInterface &&other) noexcept = default;

  MaterialPipelineDescriptorInterface(const MaterialPipelineDescriptorInterface &) = delete;
  MaterialPipelineDescriptorInterface &operator=(const MaterialPipelineDescriptorInterface &) = delete;

  /**
   * @brief Get or create descriptor + pipeline layout state from shader asset views.
   * @param request Descriptor request.
   * @return Material descriptor layout state.
   */
  [[nodiscard]] MaterialDescriptorLayout get_or_create(const MaterialDescriptorRequest &request);

  /**
   * @brief Get or create descriptor + pipeline layout state from shader IDs.
   * @param request Descriptor request by shader ID.
   * @return Material descriptor layout state.
   */
  [[nodiscard]] MaterialDescriptorLayout get_or_create(const MaterialDescriptorShaderIdRequest &request);

  /**
   * @brief Clear all cached descriptor set layouts and pipeline layouts.
   */
  void clear();

  /**
   * @brief Number of cached descriptor set layouts.
   * @return Descriptor set-layout cache size.
   */
  [[nodiscard]] std::size_t descriptor_set_layout_cache_size() const noexcept;

  /**
   * @brief Number of cached pipeline layouts.
   * @return Pipeline-layout cache size.
   */
  [[nodiscard]] std::size_t pipeline_layout_cache_size() const noexcept;

private:
  /**
   * @brief Internal constructor.
   */
  MaterialPipelineDescriptorInterface(
      DescriptorSetLayoutCache &&descriptor_set_layout_cache,
      PipelineLayoutCache &&pipeline_layout_cache
  );

  DescriptorSetLayoutCache descriptor_set_layout_cache_;
  PipelineLayoutCache pipeline_layout_cache_;
};

} // namespace varre::engine
