/**
 * @file shader_objects.hpp
 * @brief Shader-object helpers for generated shader assets.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/assets/shaders.hpp"
#include "varre/engine/descriptors/material_descriptors.hpp"
#include "varre/engine/render/pass_mode.hpp"

namespace varre::engine {

class EngineContext;

/**
 * @brief Convert a generated shader stage into a Vulkan shader-stage flag bit.
 * @param stage Generated shader stage value.
 * @return Vulkan shader-stage flag bit.
 */
[[nodiscard]] vk::ShaderStageFlagBits to_vk_shader_stage(varre::assets::ShaderStage stage);

/**
 * @brief Shader-object creation request using generated shader identifiers.
 */
struct ShaderObjectCreateRequestById {
  /** @brief Shader IDs resolved through `varre::assets::get_shader`. */
  std::span<const varre::assets::ShaderId> shader_ids;
  /** @brief Push-constant ranges used for shader-object interface compatibility. */
  std::vector<vk::PushConstantRange> push_constant_ranges;
  /** @brief Vulkan shader-object create flags applied to each shader in the request. */
  vk::ShaderCreateFlagsEXT shader_create_flags{};
};

/**
 * @brief One shader object bound to a stage and generated shader identifier.
 */
struct ShaderObjectBinding {
  /** @brief Shader identifier backing this object. */
  varre::assets::ShaderId shader_id = static_cast<varre::assets::ShaderId>(0U);
  /** @brief Shader stage this object is created for. */
  vk::ShaderStageFlagBits stage = vk::ShaderStageFlagBits::eVertex;
  /** @brief Vulkan shader-object handle used with `vkCmdBindShadersEXT`. */
  vk::ShaderEXT shader = VK_NULL_HANDLE;
  /** @brief Shader entry point used for object creation. */
  const char *entry_point = nullptr;
};

/**
 * @brief Shader-object set and descriptor interface resolved for one request.
 */
struct ShaderObjectSet {
  /** @brief Descriptor interface used while creating this shader-object set. */
  MaterialDescriptorLayout descriptor_layout;
  /** @brief Stage bindings consumed by pass-mode encoders. */
  std::vector<ShaderObjectBinding> bindings;
};

/**
 * @brief Cache for Vulkan shader objects backed by generated shader assets.
 */
class ShaderObjectCache {
public:
  /**
   * @brief Create a shader-object cache bound to one engine device.
   * @param engine Initialized engine context.
   * @return Initialized shader-object cache.
   */
  [[nodiscard]] static ShaderObjectCache create(const EngineContext &engine);

  /**
   * @brief Move-construct the cache.
   * @param other Cache being moved from.
   */
  ShaderObjectCache(ShaderObjectCache &&other) noexcept = default;

  /**
   * @brief Move-assign the cache.
   * @param other Cache being moved from.
   * @return `*this`.
   */
  ShaderObjectCache &operator=(ShaderObjectCache &&other) noexcept = default;

  ShaderObjectCache(const ShaderObjectCache &) = delete;
  ShaderObjectCache &operator=(const ShaderObjectCache &) = delete;

  /**
   * @brief Resolve generated shaders and return a shader-object set.
   * @param request Shader-object creation request.
   * @return Shader-object set with resolved descriptor interface and stage bindings.
   */
  [[nodiscard]] ShaderObjectSet get_or_create(const ShaderObjectCreateRequestById &request);

  /**
   * @brief Remove all cached shader objects and descriptor interface state.
   */
  void clear();

  /**
   * @brief Number of cached shader objects.
   * @return Cache size.
   */
  [[nodiscard]] std::size_t size() const noexcept;

private:
  /**
   * @brief Internal cache key representation.
   */
  struct Key {
    varre::assets::ShaderId shader_id = static_cast<varre::assets::ShaderId>(0U);
    std::uint64_t pipeline_layout_handle = 0U;
    std::uint32_t create_flags = 0U;

    [[nodiscard]] bool operator==(const Key &) const = default;
  };

  /**
   * @brief Internal constructor from initialized dependencies.
   */
  ShaderObjectCache(const vk::raii::Device *device, MaterialDescriptorLayoutResolver &&descriptor_layout_resolver);

  /**
   * @brief Resolve one cached or newly created shader object.
   * @param shader Generated shader metadata.
   * @param descriptor_layout Descriptor interface for this shader set.
   * @param push_constant_ranges Push-constant ranges for shader-object creation.
   * @param create_flags Vulkan shader-create flags.
   * @return Shader-object handle.
   */
  [[nodiscard]] vk::ShaderEXT get_or_create_shader_object(const varre::assets::ShaderAssetView &shader, const MaterialDescriptorLayout &descriptor_layout,
                                                          std::span<const vk::PushConstantRange> push_constant_ranges, vk::ShaderCreateFlagsEXT create_flags);

  const vk::raii::Device *device_ = nullptr;
  MaterialDescriptorLayoutResolver descriptor_layout_resolver_;
  std::vector<Key> keys_;
  std::vector<vk::raii::ShaderEXT> shader_objects_;
};

/**
 * @brief Convert shader-object bindings into pass-mode shader bindings.
 * @param bindings Shader-object bindings.
 * @return Pass-mode shader bindings in the same order as @p bindings.
 */
[[nodiscard]] std::vector<PassShaderBinding> make_pass_shader_bindings(std::span<const ShaderObjectBinding> bindings);

/**
 * @brief Convert one shader-object set into pass-mode shader bindings.
 * @param shader_set Shader-object set.
 * @return Pass-mode shader bindings in the same order as `shader_set.bindings`.
 */
[[nodiscard]] std::vector<PassShaderBinding> make_pass_shader_bindings(const ShaderObjectSet &shader_set);

/**
 * @brief Bind one shader-object binding list through a pass command encoder.
 * @param encoder Destination encoder.
 * @param bindings Shader-object bindings.
 */
void bind_shader_set(const PassCommandEncoder &encoder, std::span<const ShaderObjectBinding> bindings);

/**
 * @brief Bind one cached shader-object set through a pass command encoder.
 * @param encoder Destination encoder.
 * @param shader_set Shader-object set.
 */
void bind_shader_set(const PassCommandEncoder &encoder, const ShaderObjectSet &shader_set);

} // namespace varre::engine

namespace varre::engine::asset {
using ::varre::engine::bind_shader_set;
using ::varre::engine::make_pass_shader_bindings;
using ::varre::engine::ShaderObjectBinding;
using ::varre::engine::ShaderObjectCache;
using ::varre::engine::ShaderObjectCreateRequestById;
using ::varre::engine::ShaderObjectSet;
using ::varre::engine::to_vk_shader_stage;
} // namespace varre::engine::asset

