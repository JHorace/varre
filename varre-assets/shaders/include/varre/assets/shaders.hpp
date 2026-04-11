/**
 * @file shaders.hpp
 * @brief Public shader-asset API for the varre assets module.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace varre::assets {

/**
 * @brief Logical shader stage for a compiled shader module.
 */
enum class ShaderStage : std::uint8_t {
  /** @brief Placeholder for uninitialized or unknown stage. */
  kUnspecified = 0,
};

/**
 * @brief Stable identifier for a generated shader asset.
 */
enum class ShaderId : std::uint32_t {
  /** @brief Placeholder value used before generated IDs exist. */
  kUnimplemented = 0,
};

/**
 * @brief Lightweight descriptor layout entry used by generated shader metadata.
 */
struct DescriptorSetLayoutBinding {
  /** @brief Descriptor set index. */
  std::uint32_t set = 0;
  /** @brief Binding index within the descriptor set. */
  std::uint32_t binding = 0;
  /** @brief Backend descriptor type value (mirrors Vulkan numeric values). */
  std::uint32_t descriptor_type = 0;
  /** @brief Number of descriptors in this binding. */
  std::uint32_t descriptor_count = 0;
  /** @brief Backend stage-flag mask (mirrors Vulkan numeric values). */
  std::uint32_t stage_flags = 0;
};

/**
 * @brief Immutable view over one embedded shader asset.
 */
struct ShaderAssetView {
  /** @brief Identifier for this shader entry. */
  ShaderId id = ShaderId::kUnimplemented;
  /** @brief Pointer to embedded SPIR-V data. */
  const std::byte *data = nullptr;
  /** @brief Size of @ref data in bytes. */
  std::size_t size = 0;
  /** @brief Shader stage for this module. */
  ShaderStage stage = ShaderStage::kUnspecified;
  /** @brief Null-terminated shader entry-point name. */
  const char *entry_point = nullptr;
  /** @brief Pointer to descriptor binding metadata for this shader. */
  const DescriptorSetLayoutBinding *descriptor_set_layout_bindings = nullptr;
  /** @brief Number of entries in @ref descriptor_set_layout_bindings. */
  std::size_t descriptor_set_layout_binding_count = 0;
};

/**
 * @brief Resolve shader metadata for a specific shader id.
 * @param id Shader identifier to resolve.
 * @return Pointer to immutable shader metadata, or `nullptr` when unavailable.
 */
[[nodiscard]] const ShaderAssetView *get_shader(ShaderId id);

} // namespace varre::assets
