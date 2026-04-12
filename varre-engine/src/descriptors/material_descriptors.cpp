/**
 * @file material_descriptors.cpp
 * @brief Material/pipeline descriptor interface implementation.
 */
#include "varre/engine/descriptors/material_descriptors.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

namespace varre::engine {
namespace detail {
/**
 * @brief Expand sparse set-layout specs into contiguous set index order.
 * @param sparse_specs Sparse specs containing only sets discovered in reflection.
 * @param create_flags Descriptor set-layout create flags for synthesized empty sets.
 * @return Dense set-layout specs indexed from `0..max_set`.
 */
std::vector<DescriptorSetLayoutSpec> densify_set_layout_specs(std::vector<DescriptorSetLayoutSpec> sparse_specs,
                                                              const vk::DescriptorSetLayoutCreateFlags create_flags) {
  if (sparse_specs.empty()) {
    return sparse_specs;
  }

  std::ranges::sort(sparse_specs, [](const DescriptorSetLayoutSpec &lhs, const DescriptorSetLayoutSpec &rhs) { return lhs.set < rhs.set; });

  const std::uint32_t max_set_index = sparse_specs.back().set;
  std::vector<DescriptorSetLayoutSpec> dense_specs;
  dense_specs.reserve(static_cast<std::size_t>(max_set_index) + 1U);

  std::size_t sparse_index = 0U;
  for (std::uint32_t set_index = 0U; set_index <= max_set_index; ++set_index) {
    if (sparse_index < sparse_specs.size() && sparse_specs[sparse_index].set == set_index) {
      dense_specs.push_back(std::move(sparse_specs[sparse_index]));
      ++sparse_index;
    } else {
      dense_specs.push_back(DescriptorSetLayoutSpec{
        .set = set_index,
        .bindings = {},
        .create_flags = create_flags,
      });
    }
  }
  return dense_specs;
}
} // namespace detail

MaterialDescriptorLayoutResolver::MaterialDescriptorLayoutResolver(DescriptorSetLayoutCache &&descriptor_set_layout_cache,
                                                                   PipelineLayoutCache &&pipeline_layout_cache)
    : descriptor_set_layout_cache_(std::move(descriptor_set_layout_cache)), pipeline_layout_cache_(std::move(pipeline_layout_cache)) {}

MaterialDescriptorLayoutResolver MaterialDescriptorLayoutResolver::create(const EngineContext &engine) {
  return MaterialDescriptorLayoutResolver{
    DescriptorSetLayoutCache::create(engine),
    PipelineLayoutCache::create(engine),
  };
}

MaterialDescriptorLayout MaterialDescriptorLayoutResolver::get_or_create(const MaterialDescriptorRequest &request) {
  const std::vector<ReflectedDescriptorBinding> reflected_bindings = reflect_descriptor_bindings(request.shaders);
  std::vector<DescriptorSetLayoutSpec> sparse_specs = build_descriptor_set_layout_specs(reflected_bindings, request.descriptor_set_layout_create_flags);
  std::vector<DescriptorSetLayoutSpec> dense_specs = detail::densify_set_layout_specs(std::move(sparse_specs), request.descriptor_set_layout_create_flags);
  const std::vector<vk::DescriptorSetLayout> set_layouts = descriptor_set_layout_cache_.get_or_create_all(dense_specs);

  const vk::PipelineLayout pipeline_layout = pipeline_layout_cache_.get_or_create(PipelineLayoutSpec{
    .set_layouts = set_layouts,
    .push_constant_ranges = request.push_constant_ranges,
    .create_flags = request.pipeline_layout_create_flags,
  });

  return MaterialDescriptorLayout{
    .descriptor_set_layout_specs = std::move(dense_specs),
    .descriptor_set_layouts = set_layouts,
    .pipeline_layout = pipeline_layout,
  };
}

MaterialDescriptorLayout MaterialDescriptorLayoutResolver::get_or_create(const MaterialDescriptorRequestById &request) {
  std::vector<varre::assets::ShaderAssetView> shader_views;
  shader_views.reserve(request.shader_ids.size());
  for (const varre::assets::ShaderId shader_id : request.shader_ids) {
    const varre::assets::ShaderAssetView *shader = varre::assets::get_shader(shader_id);
    if (shader == nullptr) {
      throw std::runtime_error(fmt::format("Shader asset lookup failed for ShaderId value {}.", static_cast<std::uint32_t>(shader_id)));
    }
    shader_views.push_back(*shader);
  }

  return get_or_create(MaterialDescriptorRequest{
    .shaders = shader_views,
    .push_constant_ranges = request.push_constant_ranges,
    .descriptor_set_layout_create_flags = request.descriptor_set_layout_create_flags,
    .pipeline_layout_create_flags = request.pipeline_layout_create_flags,
  });
}

void MaterialDescriptorLayoutResolver::clear() {
  pipeline_layout_cache_.clear();
  descriptor_set_layout_cache_.clear();
}

std::size_t MaterialDescriptorLayoutResolver::descriptor_set_layout_cache_size() const noexcept { return descriptor_set_layout_cache_.size(); }

std::size_t MaterialDescriptorLayoutResolver::pipeline_layout_cache_size() const noexcept { return pipeline_layout_cache_.size(); }
} // namespace varre::engine
