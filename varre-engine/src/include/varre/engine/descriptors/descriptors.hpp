/**
 * @file descriptors.hpp
 * @brief Descriptor reflection adapters and layout-cache primitives.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/assets/shaders.hpp"

namespace varre::engine {

class EngineContext;

/**
 * @brief Reflection-friendly descriptor binding record.
 */
struct ReflectedDescriptorBinding {
  /** @brief Descriptor set index. */
  std::uint32_t set = 0U;
  /** @brief Binding index in descriptor set. */
  std::uint32_t binding = 0U;
  /** @brief Vulkan descriptor type. */
  vk::DescriptorType descriptor_type = vk::DescriptorType::eSampler;
  /** @brief Descriptor array count. */
  std::uint32_t descriptor_count = 1U;
  /** @brief Shader stages using this binding. */
  vk::ShaderStageFlags stage_flags{};
};

/**
 * @brief Descriptor binding specification for one set-layout entry.
 */
struct DescriptorBindingSpec {
  /** @brief Binding index. */
  std::uint32_t binding = 0U;
  /** @brief Vulkan descriptor type. */
  vk::DescriptorType descriptor_type = vk::DescriptorType::eSampler;
  /** @brief Descriptor array count. */
  std::uint32_t descriptor_count = 1U;
  /** @brief Shader stages using this binding. */
  vk::ShaderStageFlags stage_flags{};
};

/**
 * @brief Descriptor set-layout specification.
 */
struct DescriptorSetLayoutSpec {
  /** @brief Descriptor set index used by pipeline layout ordering. */
  std::uint32_t set = 0U;
  /** @brief Bindings in this descriptor set. */
  std::vector<DescriptorBindingSpec> bindings;
  /** @brief Vulkan descriptor set-layout create flags. */
  vk::DescriptorSetLayoutCreateFlags create_flags{};
};

/**
 * @brief Pipeline-layout specification.
 */
struct PipelineLayoutSpec {
  /** @brief Descriptor set layouts ordered by descriptor set index. */
  std::vector<vk::DescriptorSetLayout> set_layouts;
  /** @brief Push-constant ranges visible to this pipeline layout. */
  std::vector<vk::PushConstantRange> push_constant_ranges;
  /** @brief Vulkan pipeline-layout create flags. */
  vk::PipelineLayoutCreateFlags create_flags{};
};

/**
 * @brief Convert one generated shader asset into reflection binding records.
 * @param shader Generated shader asset.
 * @return Reflection bindings for this shader.
 */
[[nodiscard]] std::vector<ReflectedDescriptorBinding> reflect_descriptor_bindings(const varre::assets::ShaderAssetView &shader);

/**
 * @brief Merge descriptor bindings from multiple generated shader assets.
 * @param shaders Shader list.
 * @return Merged reflection bindings with combined stage visibility.
 */
[[nodiscard]] std::vector<ReflectedDescriptorBinding> reflect_descriptor_bindings(std::span<const varre::assets::ShaderAssetView> shaders);

/**
 * @brief Build descriptor set-layout specs from reflection bindings.
 * @param bindings Reflection bindings.
 * @param create_flags Descriptor set-layout create flags applied to every spec.
 * @return Set-layout specs grouped by descriptor set index.
 */
[[nodiscard]] std::vector<DescriptorSetLayoutSpec> build_descriptor_set_layout_specs(std::span<const ReflectedDescriptorBinding> bindings,
                                                                                     vk::DescriptorSetLayoutCreateFlags create_flags = {});

/**
 * @brief Cache for descriptor set layouts keyed by canonicalized binding schemas.
 */
class DescriptorSetLayoutCache {
public:
  /**
   * @brief Create a descriptor set-layout cache.
   * @param engine Initialized engine context.
   * @return Initialized cache.
   */
  [[nodiscard]] static DescriptorSetLayoutCache create(const EngineContext &engine);

  /**
   * @brief Move-construct the cache.
   * @param other Cache being moved from.
   */
  DescriptorSetLayoutCache(DescriptorSetLayoutCache &&other) noexcept = default;

  /**
   * @brief Move-assign the cache.
   * @param other Cache being moved from.
   * @return `*this`.
   */
  DescriptorSetLayoutCache &operator=(DescriptorSetLayoutCache &&other) noexcept = default;

  DescriptorSetLayoutCache(const DescriptorSetLayoutCache &) = delete;
  DescriptorSetLayoutCache &operator=(const DescriptorSetLayoutCache &) = delete;

  /**
   * @brief Fetch or create one descriptor set layout from schema.
   * @param spec Descriptor set-layout specification.
   * @return Descriptor set-layout handle.
   */
  [[nodiscard]] vk::DescriptorSetLayout get_or_create(const DescriptorSetLayoutSpec &spec);

  /**
   * @brief Fetch or create descriptor set layouts for all specs.
   * @param specs Descriptor set-layout specs.
   * @return Descriptor set-layout handles ordered by `specs`.
   */
  [[nodiscard]] std::vector<vk::DescriptorSetLayout> get_or_create_all(std::span<const DescriptorSetLayoutSpec> specs);

  /**
   * @brief Remove all cached descriptor set layouts.
   */
  void clear();

  /**
   * @brief Number of cached descriptor set layouts.
   * @return Cache size.
   */
  [[nodiscard]] std::size_t size() const noexcept;

private:
  /**
   * @brief Internal cache key representation.
   */
  struct Key {
    std::uint32_t create_flags = 0U;
    std::vector<std::uint32_t> binding_words;

    [[nodiscard]] bool operator==(const Key &) const = default;
  };

  /**
   * @brief Internal constructor.
   */
  explicit DescriptorSetLayoutCache(const vk::raii::Device *device);

  const vk::raii::Device *device_ = nullptr;
  std::vector<vk::raii::DescriptorSetLayout> layouts_;
  std::vector<Key> keys_;
};

/**
 * @brief Cache for pipeline layouts keyed by set-layout handles and push constants.
 */
class PipelineLayoutCache {
public:
  /**
   * @brief Create a pipeline-layout cache.
   * @param engine Initialized engine context.
   * @return Initialized cache.
   */
  [[nodiscard]] static PipelineLayoutCache create(const EngineContext &engine);

  /**
   * @brief Move-construct the cache.
   * @param other Cache being moved from.
   */
  PipelineLayoutCache(PipelineLayoutCache &&other) noexcept = default;

  /**
   * @brief Move-assign the cache.
   * @param other Cache being moved from.
   * @return `*this`.
   */
  PipelineLayoutCache &operator=(PipelineLayoutCache &&other) noexcept = default;

  PipelineLayoutCache(const PipelineLayoutCache &) = delete;
  PipelineLayoutCache &operator=(const PipelineLayoutCache &) = delete;

  /**
   * @brief Fetch or create one pipeline layout from schema.
   * @param spec Pipeline-layout specification.
   * @return Pipeline-layout handle.
   */
  [[nodiscard]] vk::PipelineLayout get_or_create(const PipelineLayoutSpec &spec);

  /**
   * @brief Remove all cached pipeline layouts.
   */
  void clear();

  /**
   * @brief Number of cached pipeline layouts.
   * @return Cache size.
   */
  [[nodiscard]] std::size_t size() const noexcept;

private:
  /**
   * @brief Internal cache key representation.
   */
  struct Key {
    std::uint32_t create_flags = 0U;
    std::vector<std::uint64_t> set_layout_handles;
    std::vector<std::uint32_t> push_constant_words;

    [[nodiscard]] bool operator==(const Key &) const = default;
  };

  /**
   * @brief Internal constructor.
   */
  explicit PipelineLayoutCache(const vk::raii::Device *device);

  const vk::raii::Device *device_ = nullptr;
  std::vector<vk::raii::PipelineLayout> layouts_;
  std::vector<Key> keys_;
};

} // namespace varre::engine

namespace varre::engine::descriptor {
using ::varre::engine::DescriptorBindingSpec;
using ::varre::engine::DescriptorSetLayoutCache;
using ::varre::engine::DescriptorSetLayoutSpec;
using ::varre::engine::PipelineLayoutCache;
using ::varre::engine::PipelineLayoutSpec;
using ::varre::engine::ReflectedDescriptorBinding;
} // namespace varre::engine::descriptor
