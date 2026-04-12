/**
 * @file shaders.hpp
 * @brief Shader-module cache and pipeline-stage helpers for generated shader assets.
 */
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/assets/shaders.hpp"

namespace varre::engine {

class EngineContext;

/**
 * @brief Convert a generated shader stage into a Vulkan shader-stage flag bit.
 * @param stage Generated shader stage value.
 * @return Vulkan shader-stage flag bit.
 */
[[nodiscard]] vk::ShaderStageFlagBits to_vk_shader_stage(varre::assets::ShaderStage stage);

/**
 * @brief Build one pipeline shader-stage create-info from generated shader metadata.
 * @param shader Generated shader metadata.
 * @param shader_module Vulkan shader-module handle.
 * @param create_flags Optional stage create flags.
 * @return Initialized Vulkan stage create-info.
 */
[[nodiscard]] vk::PipelineShaderStageCreateInfo make_pipeline_shader_stage_create_info(const varre::assets::ShaderAssetView &shader,
                                                                                       vk::ShaderModule shader_module,
                                                                                       vk::PipelineShaderStageCreateFlags create_flags = {});

/**
 * @brief Cache of Vulkan shader modules backed by generated shader assets.
 */
class ShaderModuleCache {
public:
  /**
   * @brief Create a shader-module cache bound to one engine device.
   * @param engine Initialized engine context.
   * @return Initialized shader-module cache.
   */
  [[nodiscard]] static ShaderModuleCache create(const EngineContext &engine);

  /**
   * @brief Move-construct the cache.
   * @param other Cache being moved from.
   */
  ShaderModuleCache(ShaderModuleCache &&other) noexcept = default;

  /**
   * @brief Move-assign the cache.
   * @param other Cache being moved from.
   * @return `*this`.
   */
  ShaderModuleCache &operator=(ShaderModuleCache &&other) noexcept = default;

  ShaderModuleCache(const ShaderModuleCache &) = delete;
  ShaderModuleCache &operator=(const ShaderModuleCache &) = delete;

  /**
   * @brief Resolve a generated shader by ID and return a cached or newly created shader module.
   * @param shader_id Generated shader identifier.
   * @return Vulkan shader-module handle.
   */
  [[nodiscard]] vk::ShaderModule get_or_create(varre::assets::ShaderId shader_id);

  /**
   * @brief Return a cached or newly created shader module from explicit shader metadata.
   * @param shader Generated shader metadata.
   * @return Vulkan shader-module handle.
   */
  [[nodiscard]] vk::ShaderModule get_or_create(const varre::assets::ShaderAssetView &shader);

  /**
   * @brief Resolve several generated shader IDs and return shader-module handles.
   * @param shader_ids Generated shader identifier list.
   * @return Shader-module handles in the same order as @p shader_ids.
   */
  [[nodiscard]] std::vector<vk::ShaderModule> get_or_create_all(std::span<const varre::assets::ShaderId> shader_ids);

  /**
   * @brief Build graphics/compute pipeline stage infos from generated shader IDs.
   * @param shader_ids Generated shader identifier list.
   * @return Pipeline stage create-info entries in the same order as @p shader_ids.
   */
  [[nodiscard]] std::vector<vk::PipelineShaderStageCreateInfo> build_pipeline_shader_stages(std::span<const varre::assets::ShaderId> shader_ids);

  /**
   * @brief Build graphics/compute pipeline stage infos from generated shader views.
   * @param shaders Generated shader metadata list.
   * @return Pipeline stage create-info entries in the same order as @p shaders.
   */
  [[nodiscard]] std::vector<vk::PipelineShaderStageCreateInfo> build_pipeline_shader_stages(
    std::span<const varre::assets::ShaderAssetView> shaders);

  /**
   * @brief Remove all cached shader modules.
   */
  void clear();

  /**
   * @brief Number of cached shader modules.
   * @return Cache size.
   */
  [[nodiscard]] std::size_t size() const noexcept;

private:
  /**
   * @brief Internal constructor from a Vulkan device pointer.
   */
  explicit ShaderModuleCache(const vk::raii::Device *device);

  const vk::raii::Device *device_ = nullptr;
  std::vector<varre::assets::ShaderId> ids_;
  std::vector<vk::raii::ShaderModule> modules_;
};

} // namespace varre::engine

namespace varre::engine::asset {
using ::varre::engine::make_pipeline_shader_stage_create_info;
using ::varre::engine::ShaderModuleCache;
using ::varre::engine::to_vk_shader_stage;
} // namespace varre::engine::asset
