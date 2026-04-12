/**
 * @file shaders.cpp
 * @brief Shader-module cache and pipeline-stage helper implementation.
 */
#include "varre/engine/assets/shaders.hpp"

#include <cstdint>
#include <cstring>
#include <ranges>
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
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Shader '{}' has no embedded bytecode.", varre::assets::shader_name(shader.id)));
  }
  if (shader.size == 0U) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Shader '{}' has empty embedded bytecode.", varre::assets::shader_name(shader.id)));
  }
  if ((shader.size % sizeof(std::uint32_t)) != 0U) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Shader '{}' bytecode size ({}) is not word-aligned.",
                                                                            varre::assets::shader_name(shader.id), shader.size));
  }
  if (shader.entry_point == nullptr || shader.entry_point[0] == '\0') {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Shader '{}' has an invalid entry point.", varre::assets::shader_name(shader.id)));
  }
}

/**
 * @brief Copy bytecode into a word-aligned container required by Vulkan shader-module creation.
 * @param shader Generated shader metadata.
 * @return Bytecode words copied from generated shader bytes.
 */
std::vector<std::uint32_t> copy_spirv_words(const varre::assets::ShaderAssetView &shader) {
  std::vector<std::uint32_t> words(shader.size / sizeof(std::uint32_t));
  std::memcpy(words.data(), shader.data, shader.size);
  return words;
}

/**
 * @brief Validate that each stage appears at most once in one pipeline-stage list.
 * @param stages Pipeline stages to validate.
 */
void validate_unique_stages(const std::span<const vk::PipelineShaderStageCreateInfo> stages) {
  for (std::size_t index = 0; index < stages.size(); ++index) {
    for (std::size_t next = index + 1U; next < stages.size(); ++next) {
      if (stages[index].stage == stages[next].stage) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                fmt::format("Duplicate shader stage '{}' in pipeline stage list.", vk::to_string(stages[index].stage)));
      }
    }
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

vk::PipelineShaderStageCreateInfo make_pipeline_shader_stage_create_info(const varre::assets::ShaderAssetView &shader,
                                                                         const vk::ShaderModule shader_module,
                                                                         const vk::PipelineShaderStageCreateFlags create_flags) {
  detail::validate_shader_asset_view(shader);
  return vk::PipelineShaderStageCreateInfo{}
    .setFlags(create_flags)
    .setStage(to_vk_shader_stage(shader.stage))
    .setModule(shader_module)
    .setPName(shader.entry_point);
}

ShaderModuleCache::ShaderModuleCache(const vk::raii::Device *device) : device_(device) {}

ShaderModuleCache ShaderModuleCache::create(const EngineContext &engine) { return ShaderModuleCache{&engine.device()}; }

vk::ShaderModule ShaderModuleCache::get_or_create(const varre::assets::ShaderId shader_id) {
  const varre::assets::ShaderAssetView *shader = varre::assets::get_shader(shader_id);
  if (shader == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("Shader asset lookup failed for ShaderId value {}.", static_cast<std::uint32_t>(shader_id)));
  }
  return get_or_create(*shader);
}

vk::ShaderModule ShaderModuleCache::get_or_create(const varre::assets::ShaderAssetView &shader) {
  if (device_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "ShaderModuleCache is not initialized.");
  }
  detail::validate_shader_asset_view(shader);
  static_cast<void>(to_vk_shader_stage(shader.stage));

  const auto existing = std::ranges::find(ids_, shader.id);
  if (existing != ids_.end()) {
    const std::size_t existing_index = static_cast<std::size_t>(std::distance(ids_.begin(), existing));
    return *modules_[existing_index];
  }

  const std::vector<std::uint32_t> code_words = detail::copy_spirv_words(shader);
  const vk::ShaderModuleCreateInfo create_info = vk::ShaderModuleCreateInfo{}.setCodeSize(shader.size).setPCode(code_words.data());
  modules_.emplace_back(*device_, create_info);
  ids_.push_back(shader.id);
  return *modules_.back();
}

std::vector<vk::ShaderModule> ShaderModuleCache::get_or_create_all(const std::span<const varre::assets::ShaderId> shader_ids) {
  std::vector<vk::ShaderModule> modules;
  modules.reserve(shader_ids.size());
  for (const varre::assets::ShaderId shader_id : shader_ids) {
    modules.push_back(get_or_create(shader_id));
  }
  return modules;
}

std::vector<vk::PipelineShaderStageCreateInfo> ShaderModuleCache::build_pipeline_shader_stages(
  const std::span<const varre::assets::ShaderId> shader_ids) {
  std::vector<varre::assets::ShaderAssetView> shaders;
  shaders.reserve(shader_ids.size());
  for (const varre::assets::ShaderId shader_id : shader_ids) {
    const varre::assets::ShaderAssetView *shader = varre::assets::get_shader(shader_id);
    if (shader == nullptr) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument,
                              fmt::format("Shader asset lookup failed for ShaderId value {}.", static_cast<std::uint32_t>(shader_id)));
    }
    shaders.push_back(*shader);
  }
  return build_pipeline_shader_stages(shaders);
}

std::vector<vk::PipelineShaderStageCreateInfo> ShaderModuleCache::build_pipeline_shader_stages(
  const std::span<const varre::assets::ShaderAssetView> shaders) {
  std::vector<vk::PipelineShaderStageCreateInfo> stages;
  stages.reserve(shaders.size());
  for (const varre::assets::ShaderAssetView &shader : shaders) {
    const vk::ShaderModule module = get_or_create(shader);
    stages.push_back(make_pipeline_shader_stage_create_info(shader, module));
  }
  detail::validate_unique_stages(stages);
  return stages;
}

void ShaderModuleCache::clear() {
  modules_.clear();
  ids_.clear();
}

std::size_t ShaderModuleCache::size() const noexcept { return modules_.size(); }
} // namespace varre::engine
