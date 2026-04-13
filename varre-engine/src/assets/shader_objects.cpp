/**
 * @file shader_objects.cpp
 * @brief Shader-object helper implementation.
 */
#include "varre/engine/assets/shader_objects.hpp"

#include <cstdint>
#include <ranges>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/errors.hpp"

namespace varre::engine {
namespace detail {
/**
 * @brief Validate generated shader metadata before Vulkan object creation.
 * @param shader Generated shader metadata.
 */
void validate_shader_asset_view(const varre::assets::ShaderAssetView &shader) {
  if (shader.data == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Shader '{}' has no embedded bytecode.", varre::assets::shader_name(shader.id)));
  }
  if (shader.size == 0U) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Shader '{}' has empty embedded bytecode.", varre::assets::shader_name(shader.id)));
  }
  if ((shader.size % sizeof(std::uint32_t)) != 0U) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Shader '{}' bytecode size ({}) is not word-aligned.", varre::assets::shader_name(shader.id), shader.size));
  }
  if (shader.entry_point == nullptr || shader.entry_point[0] == '\0') {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Shader '{}' has an invalid entry point.", varre::assets::shader_name(shader.id)));
  }
}

/**
 * @brief Validate that each stage appears at most once in one shader-view list.
 * @param shaders Shader views to validate.
 */
void validate_unique_shader_view_stages(const std::span<const varre::assets::ShaderAssetView> shaders) {
  for (std::size_t index = 0; index < shaders.size(); ++index) {
    const vk::ShaderStageFlagBits lhs_stage = to_vk_shader_stage(shaders[index].stage);
    for (std::size_t next = index + 1U; next < shaders.size(); ++next) {
      const vk::ShaderStageFlagBits rhs_stage = to_vk_shader_stage(shaders[next].stage);
      if (lhs_stage == rhs_stage) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                fmt::format("Duplicate shader stage '{}' in shader-object request.", vk::to_string(lhs_stage)));
      }
    }
  }
}

/**
 * @brief Stable cache token derived from a pipeline-layout handle.
 * @param pipeline_layout Pipeline-layout handle.
 * @return Integer token used in shader-object cache keys.
 */
[[nodiscard]] std::uint64_t pipeline_layout_handle_token(const vk::PipelineLayout pipeline_layout) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(static_cast<VkPipelineLayout>(pipeline_layout)));
}

/**
 * @brief Validate app-provided next-stage masks against one request's stage set.
 * @param shader_name Diagnostic shader name.
 * @param stage Shader stage being created.
 * @param next_stage App-provided `nextStage` mask.
 * @param present_stages Stage set present in the request.
 */
void validate_next_stage_mask(const std::string_view shader_name, const vk::ShaderStageFlagBits stage, const vk::ShaderStageFlags next_stage,
                              const vk::ShaderStageFlags present_stages) {
  if (next_stage == vk::ShaderStageFlags{}) {
    return;
  }

  const vk::ShaderStageFlags stages_without_self = present_stages & ~vk::ShaderStageFlags{stage};
  const vk::ShaderStageFlags invalid_bits = next_stage & ~stages_without_self;
  if (invalid_bits != vk::ShaderStageFlags{}) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Shader '{}' provides next_stage={} that references stage bits not present in the request (or itself).",
                                        shader_name, vk::to_string(next_stage)));
  }
}
} // namespace detail

vk::ShaderStageFlagBits to_vk_shader_stage(const varre::assets::ShaderStage stage) {
  switch (stage) {
  case varre::assets::ShaderStage::kVertex:
    return vk::ShaderStageFlagBits::eVertex;
  case varre::assets::ShaderStage::kTessellationControl:
    return vk::ShaderStageFlagBits::eTessellationControl;
  case varre::assets::ShaderStage::kTessellationEvaluation:
    return vk::ShaderStageFlagBits::eTessellationEvaluation;
  case varre::assets::ShaderStage::kGeometry:
    return vk::ShaderStageFlagBits::eGeometry;
  case varre::assets::ShaderStage::kFragment:
    return vk::ShaderStageFlagBits::eFragment;
  case varre::assets::ShaderStage::kCompute:
    return vk::ShaderStageFlagBits::eCompute;
  case varre::assets::ShaderStage::kTask:
    return vk::ShaderStageFlagBits::eTaskEXT;
  case varre::assets::ShaderStage::kMesh:
    return vk::ShaderStageFlagBits::eMeshEXT;
  case varre::assets::ShaderStage::kRaygen:
    return vk::ShaderStageFlagBits::eRaygenKHR;
  case varre::assets::ShaderStage::kUnspecified:
  default:
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "Generated shader stage is unspecified.");
  }
}

ShaderObjectCache::ShaderObjectCache(const vk::raii::Device *device, MaterialDescriptorLayoutResolver &&descriptor_layout_resolver)
    : device_(device), descriptor_layout_resolver_(std::move(descriptor_layout_resolver)) {}

ShaderObjectCache ShaderObjectCache::create(const EngineContext &engine) {
  return ShaderObjectCache{
    &engine.device(),
    MaterialDescriptorLayoutResolver::create(engine),
  };
}

vk::ShaderEXT ShaderObjectCache::get_or_create_shader_object(const varre::assets::ShaderAssetView &shader, const MaterialDescriptorLayout &descriptor_layout,
                                                             const std::span<const vk::PushConstantRange> push_constant_ranges,
                                                             const vk::ShaderCreateFlagsEXT create_flags, const vk::ShaderStageFlags next_stage) {
  if (device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "ShaderObjectCache is not initialized.");
  }
  detail::validate_shader_asset_view(shader);
  const vk::ShaderStageFlagBits stage = to_vk_shader_stage(shader.stage);

  const Key key{
    .shader_id = shader.id,
    .pipeline_layout_handle = detail::pipeline_layout_handle_token(descriptor_layout.pipeline_layout),
    .create_flags = static_cast<std::uint32_t>(create_flags),
    .next_stage_mask = static_cast<std::uint32_t>(next_stage),
  };

  for (std::size_t index = 0; index < keys_.size(); ++index) {
    if (keys_[index] == key) {
      return *shader_objects_[index];
    }
  }

  const vk::ShaderCreateInfoEXT create_info = vk::ShaderCreateInfoEXT{}
                                                .setFlags(create_flags)
                                                .setStage(stage)
                                                .setNextStage(next_stage)
                                                .setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
                                                .setCodeSize(shader.size)
                                                .setPCode(static_cast<const void *>(shader.data))
                                                .setPName(shader.entry_point)
                                                .setSetLayouts(descriptor_layout.descriptor_set_layouts)
                                                .setPushConstantRanges(push_constant_ranges);
  shader_objects_.emplace_back(*device_, create_info);
  keys_.push_back(key);
  return *shader_objects_.back();
}

ShaderObjectSet ShaderObjectCache::get_or_create(const ShaderObjectCreateRequestById &request) {
  if (device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "ShaderObjectCache is not initialized.");
  }
  if (request.shaders.empty()) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "ShaderObjectCreateRequestById::shaders must not be empty.");
  }

  std::vector<varre::assets::ShaderAssetView> shaders;
  shaders.reserve(request.shaders.size());
  std::vector<vk::ShaderStageFlags> next_stages;
  next_stages.reserve(request.shaders.size());
  vk::ShaderStageFlags present_stages{};
  for (const ShaderObjectCreateEntryById &entry : request.shaders) {
    const varre::assets::ShaderAssetView *shader = varre::assets::get_shader(entry.shader_id);
    if (shader == nullptr) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument,
                              fmt::format("Shader asset lookup failed for ShaderId value {}.", static_cast<std::uint32_t>(entry.shader_id)));
    }
    detail::validate_shader_asset_view(*shader);
    const vk::ShaderStageFlagBits stage = to_vk_shader_stage(shader->stage);
    present_stages |= stage;
    shaders.push_back(*shader);
    next_stages.push_back(entry.next_stage);
  }
  detail::validate_unique_shader_view_stages(shaders);

  for (std::size_t index = 0; index < shaders.size(); ++index) {
    detail::validate_next_stage_mask(varre::assets::shader_name(shaders[index].id), to_vk_shader_stage(shaders[index].stage), next_stages[index], present_stages);
  }

  const MaterialDescriptorLayout descriptor_layout = descriptor_layout_resolver_.get_or_create(MaterialDescriptorRequest{
    .shaders = shaders,
    .push_constant_ranges = request.push_constant_ranges,
    .descriptor_set_layout_create_flags = {},
    .pipeline_layout_create_flags = {},
  });

  std::vector<ShaderObjectBinding> bindings;
  bindings.reserve(shaders.size());
  for (std::size_t index = 0; index < shaders.size(); ++index) {
    const varre::assets::ShaderAssetView &shader = shaders[index];
    const vk::ShaderStageFlags next_stage = next_stages[index];
    const vk::ShaderEXT shader_object =
      get_or_create_shader_object(shader, descriptor_layout, request.push_constant_ranges, request.shader_create_flags, next_stage);
    bindings.push_back(ShaderObjectBinding{
      .shader_id = shader.id,
      .stage = to_vk_shader_stage(shader.stage),
      .shader = shader_object,
      .entry_point = shader.entry_point,
    });
  }

  return ShaderObjectSet{
    .descriptor_layout = descriptor_layout,
    .bindings = std::move(bindings),
  };
}

void ShaderObjectCache::clear() {
  shader_objects_.clear();
  keys_.clear();
  descriptor_layout_resolver_.clear();
}

std::size_t ShaderObjectCache::size() const noexcept { return shader_objects_.size(); }

std::vector<PassShaderBinding> make_pass_shader_bindings(const std::span<const ShaderObjectBinding> bindings) {
  std::vector<PassShaderBinding> pass_bindings;
  pass_bindings.reserve(bindings.size());
  for (const ShaderObjectBinding &binding : bindings) {
    pass_bindings.push_back(PassShaderBinding{
      .stage = binding.stage,
      .shader = binding.shader,
    });
  }
  return pass_bindings;
}

std::vector<PassShaderBinding> make_pass_shader_bindings(const ShaderObjectSet &shader_set) { return make_pass_shader_bindings(shader_set.bindings); }

void bind_shader_set(const PassCommandEncoder &encoder, const std::span<const ShaderObjectBinding> bindings) {
  encoder.bind_shaders(make_pass_shader_bindings(bindings));
}

void bind_shader_set(const PassCommandEncoder &encoder, const ShaderObjectSet &shader_set) { bind_shader_set(encoder, shader_set.bindings); }

} // namespace varre::engine
