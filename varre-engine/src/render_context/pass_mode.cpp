/**
 * @file pass_mode.cpp
 * @brief Queue-aware pass graph executor using shader objects + dynamic rendering.
 */
#include "varre/engine/render/pass_mode.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/errors.hpp"

namespace varre::engine {
namespace detail {
/**
 * @brief Whether one extension is present in resolved device profile.
 * @param profile Resolved engine device profile.
 * @param extension_name Extension name to search.
 * @return `true` when extension is enabled on logical-device creation.
 */
[[nodiscard]] bool has_enabled_extension(const DeviceProfile &profile, const std::string_view extension_name) {
  return std::ranges::any_of(profile.enabled_extensions, [&](const std::string &enabled) { return enabled == extension_name; });
}

/**
 * @brief Validate engine profile requirements needed by pass-mode execution.
 * @param profile Resolved engine device profile.
 */
void validate_pass_mode_device_profile(const DeviceProfile &profile) {
  if (profile.api_version < VK_API_VERSION_1_3) {
    throw make_engine_error(EngineErrorCode::kMissingRequirement, "PassExecutor requires Vulkan API version 1.3 or newer.");
  }

  if (!has_enabled_extension(profile, VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) {
    throw make_engine_error(EngineErrorCode::kMissingRequirement,
                            fmt::format("PassExecutor requires '{}' to be enabled on EngineContext device creation.", VK_EXT_SHADER_OBJECT_EXTENSION_NAME));
  }
}

/**
 * @brief Append dependency edge only when not present yet.
 * @param dependencies Destination dependency list.
 * @param phase_index Dependency phase index.
 */
void append_unique_dependency(std::vector<std::size_t> *dependencies, const std::size_t phase_index) {
  if (!std::ranges::contains(*dependencies, phase_index)) {
    dependencies->push_back(phase_index);
  }
}

/**
 * @brief Stage mask used for queue-submit waits/signals from one phase kind.
 * @param phase_kind Phase kind.
 * @return Stage mask for submit semaphores.
 */
[[nodiscard]] vk::PipelineStageFlags2 submit_stage_mask(const PassPhaseKind phase_kind) {
  switch (phase_kind) {
  case PassPhaseKind::kGraphics:
    return vk::PipelineStageFlagBits2::eAllGraphics;
  case PassPhaseKind::kCompute:
    return vk::PipelineStageFlagBits2::eComputeShader;
  case PassPhaseKind::kTransfer:
    return vk::PipelineStageFlagBits2::eTransfer;
  default:
    return vk::PipelineStageFlagBits2::eAllCommands;
  }
}

/**
 * @brief Emit a dependency barrier only when barrier lists are non-empty.
 * @param command_buffer Destination command buffer.
 * @param buffer_barriers Buffer barriers.
 * @param image_barriers Image barriers.
 */
void emit_barriers(const vk::CommandBuffer command_buffer, std::span<const vk::BufferMemoryBarrier2> buffer_barriers,
                   std::span<const vk::ImageMemoryBarrier2> image_barriers) {
  if (buffer_barriers.empty() && image_barriers.empty()) {
    return;
  }
  vk::DependencyInfo dependency_info{};
  if (!buffer_barriers.empty()) {
    dependency_info = dependency_info.setBufferMemoryBarriers(buffer_barriers);
  }
  if (!image_barriers.empty()) {
    dependency_info = dependency_info.setImageMemoryBarriers(image_barriers);
  }
  command_buffer.pipelineBarrier2(dependency_info);
}

/**
 * @brief Validate one pass phase ID against graph size.
 * @param phase_id Phase identifier.
 * @param phase_count Graph phase count.
 * @param context Context string for diagnostics.
 */
void validate_phase_id(const PassPhaseId phase_id, const std::size_t phase_count, const std::string_view context) {
  if (phase_id >= phase_count) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument,
                            fmt::format("{} references invalid phase id {} (phase_count={}).", context, phase_id, phase_count));
  }
}
} // namespace detail

PassCommandEncoder::PassCommandEncoder(const vk::CommandBuffer command_buffer, const PassPhaseKind phase_kind, PFN_vkCmdBindShadersEXT cmd_bind_shaders_ext)
    : command_buffer_(command_buffer), phase_kind_(phase_kind), cmd_bind_shaders_ext_(cmd_bind_shaders_ext) {}

vk::CommandBuffer PassCommandEncoder::command_buffer() const noexcept { return command_buffer_; }

PassPhaseKind PassCommandEncoder::phase_kind() const noexcept { return phase_kind_; }

void PassCommandEncoder::require_phase_kind(const PassPhaseKind expected, const std::string_view operation) const {
  if (phase_kind_ != expected) {
    throw make_engine_error(EngineErrorCode::kInvalidState,
                            fmt::format("PassCommandEncoder::{} is only valid for phase kind {}.", operation, static_cast<int>(expected)));
  }
}

void PassCommandEncoder::bind_shaders(const std::span<const PassShaderBinding> bindings) const {
  if (cmd_bind_shaders_ext_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "PassCommandEncoder cannot bind shader objects because vkCmdBindShadersEXT is unavailable.");
  }
  if (bindings.empty()) {
    return;
  }

  std::vector<VkShaderStageFlagBits> stages;
  stages.reserve(bindings.size());
  std::vector<VkShaderEXT> shaders;
  shaders.reserve(bindings.size());
  for (const PassShaderBinding &binding : bindings) {
    if (binding.shader == VK_NULL_HANDLE) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument, "PassCommandEncoder::bind_shaders received VK_NULL_HANDLE shader.");
    }
    stages.push_back(static_cast<VkShaderStageFlagBits>(binding.stage));
    shaders.push_back(static_cast<VkShaderEXT>(binding.shader));
  }
  cmd_bind_shaders_ext_(static_cast<VkCommandBuffer>(command_buffer_), static_cast<std::uint32_t>(bindings.size()), stages.data(), shaders.data());
}

void PassCommandEncoder::bind_descriptor_sets(const vk::PipelineBindPoint bind_point, const vk::PipelineLayout layout, const std::uint32_t first_set,
                                              const std::span<const vk::DescriptorSet> descriptor_sets,
                                              const std::span<const std::uint32_t> dynamic_offsets) const {
  command_buffer_.bindDescriptorSets(bind_point, layout, first_set, descriptor_sets, dynamic_offsets);
}

void PassCommandEncoder::push_constants(const vk::PipelineLayout layout, const vk::ShaderStageFlags stage_flags, const std::uint32_t offset,
                                        const std::span<const std::byte> data) const {
  if (data.empty()) {
    return;
  }
  command_buffer_.pushConstants(layout, stage_flags, offset, static_cast<std::uint32_t>(data.size_bytes()), data.data());
}

void PassCommandEncoder::set_viewports(const std::span<const vk::Viewport> viewports) const { command_buffer_.setViewportWithCount(viewports); }

void PassCommandEncoder::set_scissors(const std::span<const vk::Rect2D> scissors) const { command_buffer_.setScissorWithCount(scissors); }

void PassCommandEncoder::set_primitive_topology(const vk::PrimitiveTopology topology) const { command_buffer_.setPrimitiveTopology(topology); }

void PassCommandEncoder::set_cull_mode(const vk::CullModeFlags cull_mode) const { command_buffer_.setCullMode(cull_mode); }

void PassCommandEncoder::set_front_face(const vk::FrontFace front_face) const { command_buffer_.setFrontFace(front_face); }

void PassCommandEncoder::set_depth_test_enable(const bool enabled) const { command_buffer_.setDepthTestEnable(enabled); }

void PassCommandEncoder::set_depth_write_enable(const bool enabled) const { command_buffer_.setDepthWriteEnable(enabled); }

void PassCommandEncoder::set_depth_compare_op(const vk::CompareOp compare_op) const { command_buffer_.setDepthCompareOp(compare_op); }

void PassCommandEncoder::set_rasterizer_discard_enable(const bool enabled) const { command_buffer_.setRasterizerDiscardEnable(enabled); }

void PassCommandEncoder::bind_vertex_buffers(const std::uint32_t first_binding, const std::span<const vk::Buffer> buffers,
                                             const std::span<const vk::DeviceSize> offsets) const {
  if (buffers.size() != offsets.size()) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "PassCommandEncoder::bind_vertex_buffers requires equal buffer/offset counts.");
  }
  command_buffer_.bindVertexBuffers(first_binding, buffers, offsets);
}

void PassCommandEncoder::bind_index_buffer(const vk::Buffer buffer, const vk::DeviceSize offset, const vk::IndexType index_type) const {
  command_buffer_.bindIndexBuffer(buffer, offset, index_type);
}

void PassCommandEncoder::draw(const std::uint32_t vertex_count, const std::uint32_t instance_count, const std::uint32_t first_vertex,
                              const std::uint32_t first_instance) const {
  require_phase_kind(PassPhaseKind::kGraphics, "draw");
  command_buffer_.draw(vertex_count, instance_count, first_vertex, first_instance);
}

void PassCommandEncoder::draw_indexed(const std::uint32_t index_count, const std::uint32_t instance_count, const std::uint32_t first_index,
                                      const std::int32_t vertex_offset, const std::uint32_t first_instance) const {
  require_phase_kind(PassPhaseKind::kGraphics, "draw_indexed");
  command_buffer_.drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
}

void PassCommandEncoder::dispatch(const std::uint32_t group_count_x, const std::uint32_t group_count_y, const std::uint32_t group_count_z) const {
  require_phase_kind(PassPhaseKind::kCompute, "dispatch");
  command_buffer_.dispatch(group_count_x, group_count_y, group_count_z);
}

PassPhaseId PassGraph::add_phase(PassPhaseDesc description, PassRecordCallback record) {
  const PassPhaseId phase_id = static_cast<PassPhaseId>(phases_.size());
  phases_.push_back(PassPhase{
    .description = std::move(description),
    .record = std::move(record),
  });
  return phase_id;
}

const PassPhase &PassGraph::phase(const PassPhaseId phase_id) const {
  detail::validate_phase_id(phase_id, phases_.size(), "PassGraph::phase");
  return phases_[phase_id];
}

std::span<const PassPhase> PassGraph::phases() const noexcept { return phases_; }

void PassGraph::clear() noexcept { phases_.clear(); }

bool PassGraph::empty() const noexcept { return phases_.empty(); }

std::size_t PassGraph::size() const noexcept { return phases_.size(); }

PassExecutor::PassExecutor(const EngineContext *engine, const vk::raii::Device *device, std::vector<QueueRuntime> queue_runtimes,
                           const std::size_t graphics_queue_runtime_index, const std::size_t async_compute_queue_runtime_index,
                           const std::size_t transfer_queue_runtime_index, vk::raii::Semaphore timeline_semaphore, const PassExecutorCreateInfo create_info,
                           PFN_vkCmdBindShadersEXT cmd_bind_shaders_ext)
    : engine_(engine), device_(device), queue_runtimes_(std::move(queue_runtimes)), graphics_queue_runtime_index_(graphics_queue_runtime_index),
      async_compute_queue_runtime_index_(async_compute_queue_runtime_index), transfer_queue_runtime_index_(transfer_queue_runtime_index),
      timeline_semaphore_(std::move(timeline_semaphore)), create_info_(create_info), cmd_bind_shaders_ext_(cmd_bind_shaders_ext) {}

PassExecutor PassExecutor::create(const EngineContext &engine, const PassExecutorCreateInfo &info) {
  detail::validate_pass_mode_device_profile(engine.device_profile());

  const PFN_vkCmdBindShadersEXT cmd_bind_shaders_ext =
    reinterpret_cast<PFN_vkCmdBindShadersEXT>(vkGetDeviceProcAddr(static_cast<VkDevice>(*engine.device()), "vkCmdBindShadersEXT"));
  if (cmd_bind_shaders_ext == nullptr) {
    throw make_engine_error(
      EngineErrorCode::kMissingRequirement,
      "vkCmdBindShadersEXT could not be loaded even though VK_EXT_shader_object was enabled. Ensure extension feature chains are enabled.");
  }

  const DeviceQueueTopology &topology = engine.device_queue_topology();
  const bool has_async_compute = topology.families.async_compute.has_value() && topology.async_compute_queue.has_value();
  const bool has_transfer = topology.families.transfer.has_value() && topology.transfer_queue.has_value();

  if (!has_async_compute && !info.allow_async_compute_fallback_to_graphics) {
    throw make_engine_error(EngineErrorCode::kMissingRequirement, "PassExecutor requires async compute queue, but none is available.");
  }
  if (!has_transfer && !info.allow_transfer_fallback_to_graphics) {
    throw make_engine_error(EngineErrorCode::kMissingRequirement, "PassExecutor requires transfer queue, but none is available.");
  }

  std::vector<QueueRuntime> queue_runtimes;
  queue_runtimes.reserve(3U);

  const auto register_queue_runtime = [&](const PassQueueKind queue_kind, const std::uint32_t family_index, const vk::Queue queue) -> std::size_t {
    for (std::size_t index = 0; index < queue_runtimes.size(); ++index) {
      if (queue_runtimes[index].family_index == family_index) {
        return index;
      }
    }

    const vk::CommandPoolCreateInfo pool_create_info =
      vk::CommandPoolCreateInfo{}
        .setFlags(vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        .setQueueFamilyIndex(family_index);
    queue_runtimes.push_back(QueueRuntime{
      .queue_kind = queue_kind,
      .family_index = family_index,
      .queue = queue,
      .command_pool = vk::raii::CommandPool(engine.device(), pool_create_info),
    });
    return queue_runtimes.size() - 1U;
  };

  const std::size_t graphics_queue_runtime_index = register_queue_runtime(PassQueueKind::kGraphics, topology.families.graphics, topology.graphics_queue);

  const std::uint32_t async_family_index = has_async_compute ? *topology.families.async_compute : topology.families.graphics;
  const vk::Queue async_queue = has_async_compute ? *topology.async_compute_queue : topology.graphics_queue;
  const std::size_t async_compute_queue_runtime_index = register_queue_runtime(PassQueueKind::kAsyncCompute, async_family_index, async_queue);

  const std::uint32_t transfer_family_index = has_transfer ? *topology.families.transfer : topology.families.graphics;
  const vk::Queue transfer_queue = has_transfer ? *topology.transfer_queue : topology.graphics_queue;
  const std::size_t transfer_queue_runtime_index = register_queue_runtime(PassQueueKind::kTransfer, transfer_family_index, transfer_queue);

  const vk::SemaphoreTypeCreateInfo semaphore_type_info = vk::SemaphoreTypeCreateInfo{}.setSemaphoreType(vk::SemaphoreType::eTimeline).setInitialValue(0U);
  const vk::SemaphoreCreateInfo semaphore_create_info = vk::SemaphoreCreateInfo{}.setPNext(&semaphore_type_info);
  vk::raii::Semaphore timeline_semaphore(engine.device(), semaphore_create_info);

  return PassExecutor{
    &engine,
    &engine.device(),
    std::move(queue_runtimes),
    graphics_queue_runtime_index,
    async_compute_queue_runtime_index,
    transfer_queue_runtime_index,
    std::move(timeline_semaphore),
    info,
    cmd_bind_shaders_ext,
  };
}

std::size_t PassExecutor::queue_runtime_index_for(const PassQueueKind queue_kind) const {
  switch (queue_kind) {
  case PassQueueKind::kGraphics:
    return graphics_queue_runtime_index_;
  case PassQueueKind::kAsyncCompute:
    return async_compute_queue_runtime_index_;
  case PassQueueKind::kTransfer:
    return transfer_queue_runtime_index_;
  case PassQueueKind::kAuto:
  default:
    return graphics_queue_runtime_index_;
  }
}

void PassExecutor::execute(const PassGraph &graph, const PassExecutionInfo &execution_info) {
  if (device_ == nullptr || engine_ == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "PassExecutor is not initialized.");
  }

  const std::span<const PassPhase> phases = graph.phases();
  if (phases.empty()) {
    return;
  }

  struct ResolvedPhase {
    const PassPhase *phase = nullptr;
    std::size_t queue_runtime_index = 0U;
    std::vector<std::size_t> dependencies;
    std::vector<vk::BufferMemoryBarrier2> pre_buffer_barriers;
    std::vector<vk::ImageMemoryBarrier2> pre_image_barriers;
    std::vector<vk::BufferMemoryBarrier2> post_buffer_barriers;
    std::vector<vk::ImageMemoryBarrier2> post_image_barriers;
  };

  std::vector<ResolvedPhase> resolved_phases(phases.size());

  struct BufferUsageState {
    std::size_t phase_index = 0U;
    std::uint32_t queue_family_index = 0U;
    vk::Buffer buffer = VK_NULL_HANDLE;
    vk::DeviceSize offset = 0U;
    vk::DeviceSize size = VK_WHOLE_SIZE;
    vk::PipelineStageFlags2 stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
    vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eMemoryRead;
    bool writes = false;
  };

  struct ImageUsageState {
    std::size_t phase_index = 0U;
    std::uint32_t queue_family_index = 0U;
    vk::Image image = VK_NULL_HANDLE;
    vk::ImageSubresourceRange subresource_range = vk::ImageSubresourceRange{};
    vk::ImageLayout layout = vk::ImageLayout::eGeneral;
    vk::PipelineStageFlags2 stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
    vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eMemoryRead;
    bool writes = false;
  };

  std::unordered_map<PassResourceId, BufferUsageState> buffer_usage_states;
  std::unordered_map<PassResourceId, ImageUsageState> image_usage_states;

  const auto resolve_queue_runtime_index = [&](const PassPhaseDesc &phase_desc) -> std::size_t {
    switch (phase_desc.kind) {
    case PassPhaseKind::kGraphics:
      if (phase_desc.queue != PassQueueKind::kAuto && phase_desc.queue != PassQueueKind::kGraphics) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Phase '{}' is graphics but requested unsupported queue kind {}.",
                                                                               phase_desc.name, static_cast<int>(phase_desc.queue)));
      }
      return queue_runtime_index_for(PassQueueKind::kGraphics);
    case PassPhaseKind::kCompute:
      if (phase_desc.queue == PassQueueKind::kTransfer) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Phase '{}' is compute but requested transfer queue.", phase_desc.name));
      }
      if (phase_desc.queue == PassQueueKind::kGraphics) {
        return queue_runtime_index_for(PassQueueKind::kGraphics);
      }
      if (phase_desc.queue == PassQueueKind::kAsyncCompute || phase_desc.queue == PassQueueKind::kAuto) {
        return queue_runtime_index_for(PassQueueKind::kAsyncCompute);
      }
      return queue_runtime_index_for(PassQueueKind::kGraphics);
    case PassPhaseKind::kTransfer:
      if (phase_desc.queue == PassQueueKind::kAsyncCompute) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Phase '{}' is transfer but requested async compute queue.", phase_desc.name));
      }
      if (phase_desc.queue == PassQueueKind::kGraphics) {
        return queue_runtime_index_for(PassQueueKind::kGraphics);
      }
      if (phase_desc.queue == PassQueueKind::kTransfer || phase_desc.queue == PassQueueKind::kAuto) {
        return queue_runtime_index_for(PassQueueKind::kTransfer);
      }
      return queue_runtime_index_for(PassQueueKind::kGraphics);
    default:
      return queue_runtime_index_for(PassQueueKind::kGraphics);
    }
  };

  for (std::size_t phase_index = 0; phase_index < phases.size(); ++phase_index) {
    const PassPhase &phase = phases[phase_index];
    if (!phase.record) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Phase '{}' does not provide a recording callback.", phase.description.name));
    }
    if (phase.description.kind == PassPhaseKind::kGraphics) {
      if (!phase.description.graphics_rendering.has_value()) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                fmt::format("Graphics phase '{}' must provide PassGraphicsRenderingInfo.", phase.description.name));
      }
      if (phase.description.graphics_rendering->layer_count == 0U) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                fmt::format("Graphics phase '{}' has layer_count=0, which is invalid.", phase.description.name));
      }
    } else if (phase.description.graphics_rendering.has_value()) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument,
                              fmt::format("Non-graphics phase '{}' cannot provide graphics_rendering metadata.", phase.description.name));
    }

    resolved_phases[phase_index].phase = &phase;
    resolved_phases[phase_index].queue_runtime_index = resolve_queue_runtime_index(phase.description);

    const std::uint32_t current_queue_family = queue_runtimes_[resolved_phases[phase_index].queue_runtime_index].family_index;

    for (const PassPhaseId dependency_phase_id : phase.description.explicit_dependencies) {
      detail::validate_phase_id(dependency_phase_id, phases.size(), "PassPhaseDesc.explicit_dependencies");
      if (dependency_phase_id >= phase_index) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument, fmt::format("Phase '{}' dependency on phase id {} must refer to an earlier phase.",
                                                                               phase.description.name, dependency_phase_id));
      }
      detail::append_unique_dependency(&resolved_phases[phase_index].dependencies, dependency_phase_id);
    }

    for (const PassBufferAccess &buffer_access : phase.description.buffer_accesses) {
      if (buffer_access.buffer == VK_NULL_HANDLE) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                fmt::format("Phase '{}' contains PassBufferAccess with VK_NULL_HANDLE.", phase.description.name));
      }
      const auto state_it = buffer_usage_states.find(buffer_access.resource_id);
      if (state_it != buffer_usage_states.end()) {
        const BufferUsageState &previous_state = state_it->second;
        if (previous_state.buffer != buffer_access.buffer || previous_state.offset != buffer_access.offset || previous_state.size != buffer_access.size) {
          throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                  fmt::format("Phase '{}' reused buffer resource id {} with a different handle/range. Use separate resource IDs.",
                                              phase.description.name, buffer_access.resource_id));
        }

        const bool queue_changed = previous_state.queue_family_index != current_queue_family;
        const bool needs_memory_sync = previous_state.writes || buffer_access.writes;
        if (queue_changed || needs_memory_sync) {
          detail::append_unique_dependency(&resolved_phases[phase_index].dependencies, previous_state.phase_index);
        }

        if (queue_changed) {
          resolved_phases[previous_state.phase_index].post_buffer_barriers.push_back(vk::BufferMemoryBarrier2{}
                                                                                       .setSrcStageMask(previous_state.stage_mask)
                                                                                       .setSrcAccessMask(previous_state.access_mask)
                                                                                       .setDstStageMask(vk::PipelineStageFlagBits2::eNone)
                                                                                       .setDstAccessMask(vk::AccessFlagBits2::eNone)
                                                                                       .setSrcQueueFamilyIndex(previous_state.queue_family_index)
                                                                                       .setDstQueueFamilyIndex(current_queue_family)
                                                                                       .setBuffer(buffer_access.buffer)
                                                                                       .setOffset(buffer_access.offset)
                                                                                       .setSize(buffer_access.size));
          resolved_phases[phase_index].pre_buffer_barriers.push_back(vk::BufferMemoryBarrier2{}
                                                                       .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                                                                       .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                                                                       .setDstStageMask(buffer_access.stage_mask)
                                                                       .setDstAccessMask(buffer_access.access_mask)
                                                                       .setSrcQueueFamilyIndex(previous_state.queue_family_index)
                                                                       .setDstQueueFamilyIndex(current_queue_family)
                                                                       .setBuffer(buffer_access.buffer)
                                                                       .setOffset(buffer_access.offset)
                                                                       .setSize(buffer_access.size));
        } else if (needs_memory_sync) {
          resolved_phases[phase_index].pre_buffer_barriers.push_back(vk::BufferMemoryBarrier2{}
                                                                       .setSrcStageMask(previous_state.stage_mask)
                                                                       .setSrcAccessMask(previous_state.access_mask)
                                                                       .setDstStageMask(buffer_access.stage_mask)
                                                                       .setDstAccessMask(buffer_access.access_mask)
                                                                       .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                                                       .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                                                       .setBuffer(buffer_access.buffer)
                                                                       .setOffset(buffer_access.offset)
                                                                       .setSize(buffer_access.size));
        }
      }

      buffer_usage_states[buffer_access.resource_id] = BufferUsageState{
        .phase_index = phase_index,
        .queue_family_index = current_queue_family,
        .buffer = buffer_access.buffer,
        .offset = buffer_access.offset,
        .size = buffer_access.size,
        .stage_mask = buffer_access.stage_mask,
        .access_mask = buffer_access.access_mask,
        .writes = buffer_access.writes,
      };
    }

    for (const PassImageAccess &image_access : phase.description.image_accesses) {
      if (image_access.image == VK_NULL_HANDLE) {
        throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                fmt::format("Phase '{}' contains PassImageAccess with VK_NULL_HANDLE.", phase.description.name));
      }
      const auto state_it = image_usage_states.find(image_access.resource_id);
      if (state_it != image_usage_states.end()) {
        const ImageUsageState &previous_state = state_it->second;
        if (previous_state.image != image_access.image || previous_state.subresource_range != image_access.subresource_range) {
          throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                  fmt::format("Phase '{}' reused image resource id {} with a different handle/range. Use separate resource IDs.",
                                              phase.description.name, image_access.resource_id));
        }

        const bool queue_changed = previous_state.queue_family_index != current_queue_family;
        const bool layout_changed = previous_state.layout != image_access.layout;
        const bool needs_sync = previous_state.writes || image_access.writes || layout_changed;
        if (queue_changed || needs_sync) {
          detail::append_unique_dependency(&resolved_phases[phase_index].dependencies, previous_state.phase_index);
        }

        if (queue_changed) {
          const vk::ImageLayout transfer_layout = layout_changed ? image_access.layout : previous_state.layout;
          resolved_phases[previous_state.phase_index].post_image_barriers.push_back(vk::ImageMemoryBarrier2{}
                                                                                      .setSrcStageMask(previous_state.stage_mask)
                                                                                      .setSrcAccessMask(previous_state.access_mask)
                                                                                      .setDstStageMask(vk::PipelineStageFlagBits2::eNone)
                                                                                      .setDstAccessMask(vk::AccessFlagBits2::eNone)
                                                                                      .setOldLayout(previous_state.layout)
                                                                                      .setNewLayout(transfer_layout)
                                                                                      .setSrcQueueFamilyIndex(previous_state.queue_family_index)
                                                                                      .setDstQueueFamilyIndex(current_queue_family)
                                                                                      .setImage(image_access.image)
                                                                                      .setSubresourceRange(image_access.subresource_range));
          resolved_phases[phase_index].pre_image_barriers.push_back(vk::ImageMemoryBarrier2{}
                                                                      .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                                                                      .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                                                                      .setDstStageMask(image_access.stage_mask)
                                                                      .setDstAccessMask(image_access.access_mask)
                                                                      .setOldLayout(transfer_layout)
                                                                      .setNewLayout(image_access.layout)
                                                                      .setSrcQueueFamilyIndex(previous_state.queue_family_index)
                                                                      .setDstQueueFamilyIndex(current_queue_family)
                                                                      .setImage(image_access.image)
                                                                      .setSubresourceRange(image_access.subresource_range));
        } else if (needs_sync) {
          resolved_phases[phase_index].pre_image_barriers.push_back(vk::ImageMemoryBarrier2{}
                                                                      .setSrcStageMask(previous_state.stage_mask)
                                                                      .setSrcAccessMask(previous_state.access_mask)
                                                                      .setDstStageMask(image_access.stage_mask)
                                                                      .setDstAccessMask(image_access.access_mask)
                                                                      .setOldLayout(previous_state.layout)
                                                                      .setNewLayout(image_access.layout)
                                                                      .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                                                      .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                                                      .setImage(image_access.image)
                                                                      .setSubresourceRange(image_access.subresource_range));
        }
      }

      image_usage_states[image_access.resource_id] = ImageUsageState{
        .phase_index = phase_index,
        .queue_family_index = current_queue_family,
        .image = image_access.image,
        .subresource_range = image_access.subresource_range,
        .layout = image_access.layout,
        .stage_mask = image_access.stage_mask,
        .access_mask = image_access.access_mask,
        .writes = image_access.writes,
      };
    }
  }

  for (ResolvedPhase &phase : resolved_phases) {
    std::ranges::sort(phase.dependencies);
    phase.dependencies.erase(std::unique(phase.dependencies.begin(), phase.dependencies.end()), phase.dependencies.end());
  }

  for (const PassExternalWait &wait : execution_info.waits) {
    detail::validate_phase_id(wait.phase_id, phases.size(), "PassExecutionInfo.waits");
    if (wait.semaphore == VK_NULL_HANDLE) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument, "PassExecutionInfo.waits cannot contain VK_NULL_HANDLE semaphore.");
    }
  }
  for (const PassExternalSignal &signal : execution_info.signals) {
    detail::validate_phase_id(signal.phase_id, phases.size(), "PassExecutionInfo.signals");
    if (signal.semaphore == VK_NULL_HANDLE) {
      throw make_engine_error(EngineErrorCode::kInvalidArgument, "PassExecutionInfo.signals cannot contain VK_NULL_HANDLE semaphore.");
    }
  }

  for (QueueRuntime &queue_runtime : queue_runtimes_) {
    queue_runtime.command_pool.reset(vk::CommandPoolResetFlags{});
  }

  std::vector<vk::raii::CommandBuffer> command_buffers;
  command_buffers.reserve(phases.size());
  std::vector<vk::CommandBuffer> command_buffer_handles(phases.size(), VK_NULL_HANDLE);

  for (std::size_t phase_index = 0; phase_index < resolved_phases.size(); ++phase_index) {
    const ResolvedPhase &resolved_phase = resolved_phases[phase_index];
    const PassPhase &phase = *resolved_phase.phase;
    QueueRuntime &queue_runtime = queue_runtimes_[resolved_phase.queue_runtime_index];

    const vk::CommandBufferAllocateInfo allocate_info =
      vk::CommandBufferAllocateInfo{}.setCommandPool(*queue_runtime.command_pool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1U);
    std::vector<vk::raii::CommandBuffer> allocated_command_buffers = device_->allocateCommandBuffers(allocate_info);
    command_buffers.push_back(std::move(allocated_command_buffers.front()));
    command_buffer_handles[phase_index] = *command_buffers.back();

    const vk::CommandBufferBeginInfo begin_info = vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    command_buffer_handles[phase_index].begin(begin_info);

    detail::emit_barriers(command_buffer_handles[phase_index], resolved_phase.pre_buffer_barriers, resolved_phase.pre_image_barriers);

    if (phase.description.kind == PassPhaseKind::kGraphics) {
      const PassGraphicsRenderingInfo &rendering = *phase.description.graphics_rendering;
      std::vector<vk::RenderingAttachmentInfo> color_attachments;
      color_attachments.reserve(rendering.color_attachments.size());
      for (const PassColorAttachmentDesc &color_attachment : rendering.color_attachments) {
        if (color_attachment.image_view == VK_NULL_HANDLE) {
          throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                  fmt::format("Graphics phase '{}' contains color attachment with VK_NULL_HANDLE image_view.", phase.description.name));
        }
        vk::RenderingAttachmentInfo attachment_info = vk::RenderingAttachmentInfo{}
                                                        .setImageView(color_attachment.image_view)
                                                        .setImageLayout(color_attachment.image_layout)
                                                        .setLoadOp(color_attachment.load_op)
                                                        .setStoreOp(color_attachment.store_op)
                                                        .setResolveMode(color_attachment.resolve_mode)
                                                        .setResolveImageView(color_attachment.resolve_image_view)
                                                        .setResolveImageLayout(color_attachment.resolve_image_layout);
        if (color_attachment.clear_value.has_value()) {
          attachment_info = attachment_info.setClearValue(*color_attachment.clear_value);
        }
        color_attachments.push_back(attachment_info);
      }

      std::optional<vk::RenderingAttachmentInfo> depth_attachment_info;
      if (rendering.depth_attachment.has_value()) {
        const PassDepthAttachmentDesc &depth_attachment = *rendering.depth_attachment;
        if (depth_attachment.image_view == VK_NULL_HANDLE) {
          throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                  fmt::format("Graphics phase '{}' contains depth attachment with VK_NULL_HANDLE image_view.", phase.description.name));
        }
        vk::RenderingAttachmentInfo attachment_info = vk::RenderingAttachmentInfo{}
                                                        .setImageView(depth_attachment.image_view)
                                                        .setImageLayout(depth_attachment.image_layout)
                                                        .setLoadOp(depth_attachment.load_op)
                                                        .setStoreOp(depth_attachment.store_op);
        if (depth_attachment.clear_value.has_value()) {
          attachment_info = attachment_info.setClearValue(*depth_attachment.clear_value);
        }
        depth_attachment_info = attachment_info;
      }

      std::optional<vk::RenderingAttachmentInfo> stencil_attachment_info;
      if (rendering.stencil_attachment.has_value()) {
        const PassDepthAttachmentDesc &stencil_attachment = *rendering.stencil_attachment;
        if (stencil_attachment.image_view == VK_NULL_HANDLE) {
          throw make_engine_error(EngineErrorCode::kInvalidArgument,
                                  fmt::format("Graphics phase '{}' contains stencil attachment with VK_NULL_HANDLE image_view.", phase.description.name));
        }
        vk::RenderingAttachmentInfo attachment_info = vk::RenderingAttachmentInfo{}
                                                        .setImageView(stencil_attachment.image_view)
                                                        .setImageLayout(stencil_attachment.image_layout)
                                                        .setLoadOp(stencil_attachment.load_op)
                                                        .setStoreOp(stencil_attachment.store_op);
        if (stencil_attachment.clear_value.has_value()) {
          attachment_info = attachment_info.setClearValue(*stencil_attachment.clear_value);
        }
        stencil_attachment_info = attachment_info;
      }

      vk::RenderingInfo rendering_info = vk::RenderingInfo{}
                                           .setRenderArea(rendering.render_area)
                                           .setLayerCount(rendering.layer_count)
                                           .setViewMask(rendering.view_mask)
                                           .setColorAttachments(color_attachments);
      if (depth_attachment_info.has_value()) {
        rendering_info = rendering_info.setPDepthAttachment(&(*depth_attachment_info));
      }
      if (stencil_attachment_info.has_value()) {
        rendering_info = rendering_info.setPStencilAttachment(&(*stencil_attachment_info));
      }

      command_buffer_handles[phase_index].beginRendering(rendering_info);
      PassCommandEncoder encoder{command_buffer_handles[phase_index], phase.description.kind, cmd_bind_shaders_ext_};
      phase.record(encoder);
      command_buffer_handles[phase_index].endRendering();
    } else {
      PassCommandEncoder encoder{command_buffer_handles[phase_index], phase.description.kind, cmd_bind_shaders_ext_};
      phase.record(encoder);
    }

    detail::emit_barriers(command_buffer_handles[phase_index], resolved_phase.post_buffer_barriers, resolved_phase.post_image_barriers);
    command_buffer_handles[phase_index].end();
  }

  std::vector<std::vector<PassExternalWait>> waits_by_phase(phases.size());
  for (const PassExternalWait &wait : execution_info.waits) {
    waits_by_phase[wait.phase_id].push_back(wait);
  }
  std::vector<std::vector<PassExternalSignal>> signals_by_phase(phases.size());
  for (const PassExternalSignal &signal : execution_info.signals) {
    signals_by_phase[signal.phase_id].push_back(signal);
  }

  std::vector<std::uint64_t> phase_timeline_values(phases.size(), 0U);
  std::uint64_t timeline_value_cursor = completed_timeline_value_;
  for (std::size_t phase_index = 0; phase_index < resolved_phases.size(); ++phase_index) {
    const ResolvedPhase &phase = resolved_phases[phase_index];
    const QueueRuntime &queue_runtime = queue_runtimes_[phase.queue_runtime_index];

    std::vector<vk::SemaphoreSubmitInfo> wait_infos;
    wait_infos.reserve(phase.dependencies.size() + waits_by_phase[phase_index].size());
    for (const std::size_t dependency_index : phase.dependencies) {
      const bool cross_queue = resolved_phases[dependency_index].queue_runtime_index != phase.queue_runtime_index;
      if (!cross_queue) {
        continue;
      }
      wait_infos.push_back(vk::SemaphoreSubmitInfo{}
                             .setSemaphore(*timeline_semaphore_)
                             .setValue(phase_timeline_values[dependency_index])
                             .setStageMask(detail::submit_stage_mask(phase.phase->description.kind)));
    }
    for (const PassExternalWait &wait : waits_by_phase[phase_index]) {
      wait_infos.push_back(vk::SemaphoreSubmitInfo{}.setSemaphore(wait.semaphore).setValue(0U).setStageMask(wait.stage_mask));
    }

    std::vector<vk::SemaphoreSubmitInfo> signal_infos;
    signal_infos.reserve(1U + signals_by_phase[phase_index].size());
    phase_timeline_values[phase_index] = ++timeline_value_cursor;
    signal_infos.push_back(vk::SemaphoreSubmitInfo{}
                             .setSemaphore(*timeline_semaphore_)
                             .setValue(phase_timeline_values[phase_index])
                             .setStageMask(detail::submit_stage_mask(phase.phase->description.kind)));
    for (const PassExternalSignal &signal : signals_by_phase[phase_index]) {
      signal_infos.push_back(vk::SemaphoreSubmitInfo{}.setSemaphore(signal.semaphore).setValue(0U).setStageMask(signal.stage_mask));
    }

    const vk::CommandBufferSubmitInfo command_buffer_submit_info = vk::CommandBufferSubmitInfo{}.setCommandBuffer(command_buffer_handles[phase_index]);
    const vk::SubmitInfo2 submit_info =
      vk::SubmitInfo2{}.setWaitSemaphoreInfos(wait_infos).setCommandBufferInfos(command_buffer_submit_info).setSignalSemaphoreInfos(signal_infos);
    queue_runtime.queue.submit2(submit_info, vk::Fence{});
  }

  const vk::SemaphoreWaitInfo wait_info = vk::SemaphoreWaitInfo{}.setSemaphores(*timeline_semaphore_).setValues(timeline_value_cursor);
  const vk::Result wait_result = device_->waitSemaphores(wait_info, std::numeric_limits<std::uint64_t>::max());
  if (wait_result != vk::Result::eSuccess) {
    throw make_vulkan_result_error(wait_result, "PassExecutor timeline wait failed");
  }
  completed_timeline_value_ = timeline_value_cursor;
}

void PassExecutor::wait_idle() const {
  if (device_ != nullptr) {
    device_->waitIdle();
  }
}

} // namespace varre::engine
