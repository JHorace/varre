#pragma once

#include <cstddef>
#include <cstdint>

namespace varre::assets {

enum class ShaderStage : std::uint8_t {
  kUnspecified = 0,
};

enum class ShaderId : std::uint32_t {
  kUnimplemented = 0,
};

struct DescriptorSetLayoutBinding {
  std::uint32_t set = 0;
  std::uint32_t binding = 0;
  std::uint32_t descriptor_type = 0;
  std::uint32_t descriptor_count = 0;
  std::uint32_t stage_flags = 0;
};

struct ShaderAssetView {
  ShaderId id = ShaderId::kUnimplemented;
  const std::byte* data = nullptr;
  std::size_t size = 0;
  ShaderStage stage = ShaderStage::kUnspecified;
  const char* entry_point = nullptr;
  const DescriptorSetLayoutBinding* descriptor_set_layout_bindings = nullptr;
  std::size_t descriptor_set_layout_binding_count = 0;
};

[[nodiscard]] const ShaderAssetView* get_shader(ShaderId id);

}  // namespace varre::assets

