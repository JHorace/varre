/**
 * @file descriptors.cpp
 * @brief Descriptor reflection adapters and layout-cache primitives.
 */
#include "varre/engine/descriptors/descriptors.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include "varre/engine/core/engine.hpp"

namespace varre::engine {
namespace detail {
/**
 * @brief Canonical key representation for one descriptor binding.
 */
struct BindingKey {
  std::uint32_t binding = 0U;
  std::uint32_t descriptor_type = 0U;
  std::uint32_t descriptor_count = 0U;
  std::uint32_t stage_flags = 0U;

  [[nodiscard]] bool operator==(const BindingKey &) const = default;
};

/**
 * @brief Validate that descriptor type/count match for merged bindings.
 * @param lhs Existing merged binding.
 * @param rhs Incoming binding to merge.
 */
void validate_binding_compatibility(const ReflectedDescriptorBinding &lhs, const ReflectedDescriptorBinding &rhs) {
  if (lhs.descriptor_type != rhs.descriptor_type || lhs.descriptor_count != rhs.descriptor_count) {
    throw std::runtime_error(fmt::format("Incompatible descriptor binding reflection at set {} binding {}", lhs.set, lhs.binding));
  }
}

/**
 * @brief Sort descriptor binding specs by binding index.
 * @param bindings Binding list.
 */
void sort_binding_specs(std::vector<DescriptorBindingSpec> *bindings) {
  std::ranges::sort(*bindings, [](const DescriptorBindingSpec &lhs, const DescriptorBindingSpec &rhs) { return lhs.binding < rhs.binding; });
}

/**
 * @brief Build canonical descriptor binding word list for one set-layout spec.
 * @param spec Input set-layout specification.
 * @return Canonical binding word list.
 */
std::vector<std::uint32_t> canonical_binding_words(const DescriptorSetLayoutSpec &spec) {
  std::vector<BindingKey> keys;
  keys.reserve(spec.bindings.size());
  for (const DescriptorBindingSpec &binding : spec.bindings) {
    keys.push_back(BindingKey{
      .binding = binding.binding,
      .descriptor_type = static_cast<std::uint32_t>(binding.descriptor_type),
      .descriptor_count = binding.descriptor_count,
      .stage_flags = static_cast<std::uint32_t>(binding.stage_flags),
    });
  }
  std::ranges::sort(keys, [](const BindingKey &lhs, const BindingKey &rhs) { return lhs.binding < rhs.binding; });

  std::vector<std::uint32_t> words;
  words.reserve(keys.size() * 4U);
  for (const BindingKey &key : keys) {
    words.push_back(key.binding);
    words.push_back(key.descriptor_type);
    words.push_back(key.descriptor_count);
    words.push_back(key.stage_flags);
  }
  return words;
}

/**
 * @brief Build canonical push-constant range word list.
 * @param ranges Push-constant ranges.
 * @return Canonical push-constant word list.
 */
std::vector<std::uint32_t> canonical_push_constant_words(const std::vector<vk::PushConstantRange> &ranges) {
  std::vector<vk::PushConstantRange> ordered = ranges;
  std::ranges::sort(ordered, [](const vk::PushConstantRange &lhs, const vk::PushConstantRange &rhs) {
    if (lhs.offset != rhs.offset) {
      return lhs.offset < rhs.offset;
    }
    if (lhs.size != rhs.size) {
      return lhs.size < rhs.size;
    }
    return static_cast<std::uint32_t>(lhs.stageFlags) < static_cast<std::uint32_t>(rhs.stageFlags);
  });

  std::vector<std::uint32_t> words;
  words.reserve(ordered.size() * 3U);
  for (const vk::PushConstantRange &range : ordered) {
    words.push_back(static_cast<std::uint32_t>(range.stageFlags));
    words.push_back(range.offset);
    words.push_back(range.size);
  }
  return words;
}
} // namespace detail

std::vector<ReflectedDescriptorBinding> reflect_descriptor_bindings(const varre::assets::ShaderAssetView &shader) {
  std::vector<ReflectedDescriptorBinding> bindings;
  bindings.reserve(shader.descriptor_set_layout_binding_count);
  for (std::size_t index = 0; index < shader.descriptor_set_layout_binding_count; ++index) {
    const varre::assets::DescriptorSetLayoutBinding source = shader.descriptor_set_layout_bindings[index];
    bindings.push_back(ReflectedDescriptorBinding{
      .set = source.set,
      .binding = source.binding,
      .descriptor_type = static_cast<vk::DescriptorType>(source.descriptor_type),
      .descriptor_count = source.descriptor_count,
      .stage_flags = vk::ShaderStageFlags{source.stage_flags},
    });
  }
  return bindings;
}

std::vector<ReflectedDescriptorBinding> reflect_descriptor_bindings(std::span<const varre::assets::ShaderAssetView> shaders) {
  std::map<std::pair<std::uint32_t, std::uint32_t>, ReflectedDescriptorBinding> merged;
  for (const varre::assets::ShaderAssetView &shader : shaders) {
    std::vector<ReflectedDescriptorBinding> bindings = reflect_descriptor_bindings(shader);
    for (const ReflectedDescriptorBinding &binding : bindings) {
      const auto key = std::make_pair(binding.set, binding.binding);
      const auto it = merged.find(key);
      if (it == merged.end()) {
        merged.emplace(key, binding);
        continue;
      }
      detail::validate_binding_compatibility(it->second, binding);
      it->second.stage_flags |= binding.stage_flags;
    }
  }

  std::vector<ReflectedDescriptorBinding> out;
  out.reserve(merged.size());
  for (const auto &[key, binding] : merged) {
    static_cast<void>(key);
    out.push_back(binding);
  }
  return out;
}

std::vector<DescriptorSetLayoutSpec> build_descriptor_set_layout_specs(std::span<const ReflectedDescriptorBinding> bindings,
                                                                       const vk::DescriptorSetLayoutCreateFlags create_flags) {
  std::map<std::uint32_t, DescriptorSetLayoutSpec> by_set;
  for (const ReflectedDescriptorBinding &binding : bindings) {
    auto [it, inserted] = by_set.try_emplace(binding.set, DescriptorSetLayoutSpec{
                                                            .set = binding.set,
                                                            .bindings = {},
                                                            .create_flags = create_flags,
                                                          });
    DescriptorSetLayoutSpec &spec = it->second;
    spec.bindings.push_back(DescriptorBindingSpec{
      .binding = binding.binding,
      .descriptor_type = binding.descriptor_type,
      .descriptor_count = binding.descriptor_count,
      .stage_flags = binding.stage_flags,
    });
    if (!inserted) {
      spec.create_flags = create_flags;
    }
  }

  std::vector<DescriptorSetLayoutSpec> out;
  out.reserve(by_set.size());
  for (auto &[set, spec] : by_set) {
    static_cast<void>(set);
    detail::sort_binding_specs(&spec.bindings);
    out.push_back(std::move(spec));
  }
  return out;
}

DescriptorSetLayoutCache::DescriptorSetLayoutCache(const vk::raii::Device *device) : device_(device) {}

DescriptorSetLayoutCache DescriptorSetLayoutCache::create(const EngineContext &engine) { return DescriptorSetLayoutCache{&engine.device()}; }

vk::DescriptorSetLayout DescriptorSetLayoutCache::get_or_create(const DescriptorSetLayoutSpec &spec) {
  if (device_ == nullptr) {
    throw std::runtime_error("DescriptorSetLayoutCache is not initialized.");
  }
  const Key key{
    .create_flags = static_cast<std::uint32_t>(spec.create_flags),
    .binding_words = detail::canonical_binding_words(spec),
  };

  for (std::size_t index = 0; index < keys_.size(); ++index) {
    if (keys_[index] == key) {
      return *layouts_[index];
    }
  }

  if ((key.binding_words.size() % 4U) != 0U) {
    throw std::runtime_error("DescriptorSetLayoutCache internal key encoding is corrupted.");
  }
  std::vector<vk::DescriptorSetLayoutBinding> bindings;
  bindings.reserve(key.binding_words.size() / 4U);
  for (std::size_t offset = 0; offset < key.binding_words.size(); offset += 4U) {
    bindings.push_back(vk::DescriptorSetLayoutBinding{}
                         .setBinding(key.binding_words[offset + 0U])
                         .setDescriptorType(static_cast<vk::DescriptorType>(key.binding_words[offset + 1U]))
                         .setDescriptorCount(key.binding_words[offset + 2U])
                         .setStageFlags(vk::ShaderStageFlags{key.binding_words[offset + 3U]})
                         .setPImmutableSamplers(nullptr));
  }
  const vk::DescriptorSetLayoutCreateInfo create_info =
    vk::DescriptorSetLayoutCreateInfo{}.setFlags(static_cast<vk::DescriptorSetLayoutCreateFlags>(key.create_flags)).setBindings(bindings);
  layouts_.emplace_back(*device_, create_info);
  keys_.push_back(key);
  return *layouts_.back();
}

std::vector<vk::DescriptorSetLayout> DescriptorSetLayoutCache::get_or_create_all(std::span<const DescriptorSetLayoutSpec> specs) {
  std::vector<vk::DescriptorSetLayout> layouts;
  layouts.reserve(specs.size());
  for (const DescriptorSetLayoutSpec &spec : specs) {
    layouts.push_back(get_or_create(spec));
  }
  return layouts;
}

void DescriptorSetLayoutCache::clear() {
  layouts_.clear();
  keys_.clear();
}

std::size_t DescriptorSetLayoutCache::size() const noexcept { return layouts_.size(); }

PipelineLayoutCache::PipelineLayoutCache(const vk::raii::Device *device) : device_(device) {}

PipelineLayoutCache PipelineLayoutCache::create(const EngineContext &engine) { return PipelineLayoutCache{&engine.device()}; }

vk::PipelineLayout PipelineLayoutCache::get_or_create(const PipelineLayoutSpec &spec) {
  if (device_ == nullptr) {
    throw std::runtime_error("PipelineLayoutCache is not initialized.");
  }

  Key key{
    .create_flags = static_cast<std::uint32_t>(spec.create_flags),
    .set_layout_handles = {},
    .push_constant_words = detail::canonical_push_constant_words(spec.push_constant_ranges),
  };
  key.set_layout_handles.reserve(spec.set_layouts.size());
  for (const vk::DescriptorSetLayout layout : spec.set_layouts) {
    key.set_layout_handles.push_back(reinterpret_cast<std::uint64_t>(static_cast<VkDescriptorSetLayout>(layout)));
  }

  for (std::size_t index = 0; index < keys_.size(); ++index) {
    if (keys_[index] == key) {
      return *layouts_[index];
    }
  }

  if ((key.push_constant_words.size() % 3U) != 0U) {
    throw std::runtime_error("PipelineLayoutCache internal key encoding is corrupted.");
  }
  std::vector<vk::PushConstantRange> push_constant_ranges;
  push_constant_ranges.reserve(key.push_constant_words.size() / 3U);
  for (std::size_t offset = 0; offset < key.push_constant_words.size(); offset += 3U) {
    push_constant_ranges.push_back(vk::PushConstantRange{}
                                     .setStageFlags(vk::ShaderStageFlags{key.push_constant_words[offset + 0U]})
                                     .setOffset(key.push_constant_words[offset + 1U])
                                     .setSize(key.push_constant_words[offset + 2U]));
  }

  const vk::PipelineLayoutCreateInfo create_info = vk::PipelineLayoutCreateInfo{}
                                                     .setFlags(static_cast<vk::PipelineLayoutCreateFlags>(key.create_flags))
                                                     .setSetLayouts(spec.set_layouts)
                                                     .setPushConstantRanges(push_constant_ranges);
  layouts_.emplace_back(*device_, create_info);
  keys_.push_back(std::move(key));
  return *layouts_.back();
}

void PipelineLayoutCache::clear() {
  layouts_.clear();
  keys_.clear();
}

std::size_t PipelineLayoutCache::size() const noexcept { return layouts_.size(); }
} // namespace varre::engine
