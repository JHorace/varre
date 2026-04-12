/**
 * @file pass_mode.hpp
 * @brief Queue-aware pass graph and shader-object command encoding APIs.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace varre::engine {

class EngineContext;

/**
 * @brief Stable identifier for one declared pass resource.
 */
using PassResourceId = std::uint64_t;

/**
 * @brief Stable identifier for one registered phase in a pass graph.
 */
using PassPhaseId = std::uint32_t;

/**
 * @brief Phase execution type.
 */
enum class PassPhaseKind : std::uint8_t {
  /** @brief Graphics phase using dynamic rendering. */
  kGraphics = 0U,
  /** @brief Compute-only phase. */
  kCompute,
  /** @brief Transfer-only phase. */
  kTransfer,
};

/**
 * @brief Queue preference for one phase.
 */
enum class PassQueueKind : std::uint8_t {
  /**
   * @brief Auto-select queue from phase kind and available topology.
   *
   * - Graphics phases use graphics queue.
   * - Compute phases prefer async-compute then fall back per executor policy.
   * - Transfer phases prefer transfer then fall back per executor policy.
   */
  kAuto = 0U,
  /** @brief Force graphics queue selection. */
  kGraphics,
  /** @brief Force asynchronous compute queue selection. */
  kAsyncCompute,
  /** @brief Force transfer queue selection. */
  kTransfer,
};

/**
 * @brief Buffer usage declaration for one phase.
 */
struct PassBufferAccess {
  /** @brief Logical resource identifier used by dependency analysis. */
  PassResourceId resource_id = 0U;
  /** @brief Vulkan buffer handle used for generated barriers. */
  vk::Buffer buffer = VK_NULL_HANDLE;
  /** @brief Buffer subrange offset used for generated barriers. */
  vk::DeviceSize offset = 0U;
  /** @brief Buffer subrange size used for generated barriers. */
  vk::DeviceSize size = VK_WHOLE_SIZE;
  /** @brief Pipeline stage mask where this phase accesses the buffer. */
  vk::PipelineStageFlags2 stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
  /** @brief Access mask for this phase usage. */
  vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eMemoryRead;
  /** @brief True when this phase writes the buffer. */
  bool writes = false;
};

/**
 * @brief Image usage declaration for one phase.
 */
struct PassImageAccess {
  /** @brief Logical resource identifier used by dependency analysis. */
  PassResourceId resource_id = 0U;
  /** @brief Vulkan image handle used for generated barriers. */
  vk::Image image = VK_NULL_HANDLE;
  /** @brief Image subresource range used for generated barriers. */
  vk::ImageSubresourceRange subresource_range =
    vk::ImageSubresourceRange{}.setAspectMask(vk::ImageAspectFlagBits::eColor).setBaseMipLevel(0U).setLevelCount(1U).setBaseArrayLayer(0U).setLayerCount(1U);
  /** @brief Expected image layout during this phase usage. */
  vk::ImageLayout layout = vk::ImageLayout::eGeneral;
  /** @brief Pipeline stage mask where this phase accesses the image. */
  vk::PipelineStageFlags2 stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
  /** @brief Access mask for this phase usage. */
  vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eMemoryRead;
  /** @brief True when this phase writes the image. */
  bool writes = false;
};

/**
 * @brief Dynamic rendering color attachment declaration.
 */
struct PassColorAttachmentDesc {
  /** @brief Target image view for dynamic rendering. */
  vk::ImageView image_view = VK_NULL_HANDLE;
  /** @brief Layout used for @ref image_view during rendering. */
  vk::ImageLayout image_layout = vk::ImageLayout::eColorAttachmentOptimal;
  /** @brief Attachment load operation. */
  vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eLoad;
  /** @brief Attachment store operation. */
  vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore;
  /** @brief Optional clear value used when `load_op == eClear`. */
  std::optional<vk::ClearValue> clear_value;
  /** @brief Optional resolve image view. */
  vk::ImageView resolve_image_view = VK_NULL_HANDLE;
  /** @brief Layout used for @ref resolve_image_view when present. */
  vk::ImageLayout resolve_image_layout = vk::ImageLayout::eColorAttachmentOptimal;
  /** @brief Resolve mode for this attachment. */
  vk::ResolveModeFlagBits resolve_mode = vk::ResolveModeFlagBits::eNone;
};

/**
 * @brief Dynamic rendering depth/stencil attachment declaration.
 */
struct PassDepthAttachmentDesc {
  /** @brief Target image view for depth/stencil attachment. */
  vk::ImageView image_view = VK_NULL_HANDLE;
  /** @brief Layout used for @ref image_view during rendering. */
  vk::ImageLayout image_layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
  /** @brief Attachment load operation. */
  vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eLoad;
  /** @brief Attachment store operation. */
  vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore;
  /** @brief Optional clear value used when `load_op == eClear`. */
  std::optional<vk::ClearValue> clear_value;
};

/**
 * @brief Dynamic rendering configuration for graphics phases.
 */
struct PassGraphicsRenderingInfo {
  /** @brief Render area used by `vkCmdBeginRendering`. */
  vk::Rect2D render_area{};
  /** @brief Layer count for rendering. */
  std::uint32_t layer_count = 1U;
  /** @brief View mask for multiview rendering. */
  std::uint32_t view_mask = 0U;
  /** @brief Color attachments used by this graphics phase. */
  std::vector<PassColorAttachmentDesc> color_attachments;
  /** @brief Optional depth attachment. */
  std::optional<PassDepthAttachmentDesc> depth_attachment;
  /** @brief Optional stencil attachment. */
  std::optional<PassDepthAttachmentDesc> stencil_attachment;
};

/**
 * @brief Declarative phase description used by the pass graph scheduler.
 */
struct PassPhaseDesc {
  /** @brief Diagnostic name for this phase. */
  std::string name;
  /** @brief Phase execution kind. */
  PassPhaseKind kind = PassPhaseKind::kGraphics;
  /** @brief Queue preference for this phase. */
  PassQueueKind queue = PassQueueKind::kAuto;
  /** @brief Explicit dependency edges to earlier phase IDs. */
  std::vector<PassPhaseId> explicit_dependencies;
  /** @brief Declared buffer accesses for dependency + barrier generation. */
  std::vector<PassBufferAccess> buffer_accesses;
  /** @brief Declared image accesses for dependency + barrier generation. */
  std::vector<PassImageAccess> image_accesses;
  /**
   * @brief Dynamic rendering metadata.
   *
   * Required for graphics phases and ignored for compute/transfer phases.
   */
  std::optional<PassGraphicsRenderingInfo> graphics_rendering;
};

/**
 * @brief One shader-object stage binding command.
 */
struct PassShaderBinding {
  /** @brief Shader stage to bind. */
  vk::ShaderStageFlagBits stage = vk::ShaderStageFlagBits::eVertex;
  /** @brief Vulkan shader object handle. */
  vk::ShaderEXT shader = VK_NULL_HANDLE;
};

/**
 * @brief Command encoder presented to one phase callback.
 */
class PassCommandEncoder {
public:
  /**
   * @brief Access wrapped command buffer.
   * @return Command buffer handle.
   */
  [[nodiscard]] vk::CommandBuffer command_buffer() const noexcept;

  /**
   * @brief Access phase kind this encoder belongs to.
   * @return Phase kind.
   */
  [[nodiscard]] PassPhaseKind phase_kind() const noexcept;

  /**
   * @brief Bind shader objects for the current phase.
   * @param bindings Stage-to-shader bindings.
   */
  void bind_shaders(std::span<const PassShaderBinding> bindings) const;

  /**
   * @brief Bind descriptor sets.
   * @param bind_point Vulkan bind point.
   * @param layout Pipeline layout used for descriptor interfaces.
   * @param first_set First descriptor set index.
   * @param descriptor_sets Descriptor sets to bind.
   * @param dynamic_offsets Optional dynamic offsets.
   */
  void bind_descriptor_sets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout, std::uint32_t first_set,
                            std::span<const vk::DescriptorSet> descriptor_sets, std::span<const std::uint32_t> dynamic_offsets = {}) const;

  /**
   * @brief Push constants for shader-object compatible layouts.
   * @param layout Pipeline layout handle.
   * @param stage_flags Shader stages receiving the constants.
   * @param offset Byte offset into push constant range.
   * @param data Raw bytes to push.
   */
  void push_constants(vk::PipelineLayout layout, vk::ShaderStageFlags stage_flags, std::uint32_t offset, std::span<const std::byte> data) const;

  /**
   * @brief Set one or more dynamic viewports.
   * @param viewports Viewport list.
   */
  void set_viewports(std::span<const vk::Viewport> viewports) const;

  /**
   * @brief Set one or more dynamic scissors.
   * @param scissors Scissor list.
   */
  void set_scissors(std::span<const vk::Rect2D> scissors) const;

  /**
   * @brief Set dynamic primitive topology.
   * @param topology Primitive topology.
   */
  void set_primitive_topology(vk::PrimitiveTopology topology) const;

  /**
   * @brief Set dynamic cull mode.
   * @param cull_mode Cull mode.
   */
  void set_cull_mode(vk::CullModeFlags cull_mode) const;

  /**
   * @brief Set dynamic front-face winding.
   * @param front_face Front face.
   */
  void set_front_face(vk::FrontFace front_face) const;

  /**
   * @brief Enable or disable depth test dynamically.
   * @param enabled Depth test state.
   */
  void set_depth_test_enable(bool enabled) const;

  /**
   * @brief Enable or disable depth writes dynamically.
   * @param enabled Depth write state.
   */
  void set_depth_write_enable(bool enabled) const;

  /**
   * @brief Set dynamic depth compare operation.
   * @param compare_op Compare operation.
   */
  void set_depth_compare_op(vk::CompareOp compare_op) const;

  /**
   * @brief Enable or disable rasterizer discard dynamically.
   * @param enabled Rasterizer discard state.
   */
  void set_rasterizer_discard_enable(bool enabled) const;

  /**
   * @brief Bind vertex buffers.
   * @param first_binding First vertex binding slot.
   * @param buffers Vertex buffer handles.
   * @param offsets Byte offsets for @p buffers.
   */
  void bind_vertex_buffers(std::uint32_t first_binding, std::span<const vk::Buffer> buffers, std::span<const vk::DeviceSize> offsets) const;

  /**
   * @brief Bind index buffer.
   * @param buffer Index buffer handle.
   * @param offset Byte offset into index buffer.
   * @param index_type Index element type.
   */
  void bind_index_buffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType index_type) const;

  /**
   * @brief Record `vkCmdDraw`.
   * @param vertex_count Vertex count.
   * @param instance_count Instance count.
   * @param first_vertex First vertex index.
   * @param first_instance First instance index.
   */
  void draw(std::uint32_t vertex_count, std::uint32_t instance_count = 1U, std::uint32_t first_vertex = 0U, std::uint32_t first_instance = 0U) const;

  /**
   * @brief Record `vkCmdDrawIndexed`.
   * @param index_count Index count.
   * @param instance_count Instance count.
   * @param first_index First index.
   * @param vertex_offset Added to vertex index.
   * @param first_instance First instance index.
   */
  void draw_indexed(std::uint32_t index_count, std::uint32_t instance_count = 1U, std::uint32_t first_index = 0U, std::int32_t vertex_offset = 0,
                    std::uint32_t first_instance = 0U) const;

  /**
   * @brief Record `vkCmdDispatch`.
   * @param group_count_x Workgroup count X.
   * @param group_count_y Workgroup count Y.
   * @param group_count_z Workgroup count Z.
   */
  void dispatch(std::uint32_t group_count_x, std::uint32_t group_count_y, std::uint32_t group_count_z) const;

private:
  friend class PassExecutor;

  /**
   * @brief Internal constructor.
   */
  PassCommandEncoder(vk::CommandBuffer command_buffer, PassPhaseKind phase_kind, PFN_vkCmdBindShadersEXT cmd_bind_shaders_ext);

  /**
   * @brief Validate phase kind for one command.
   * @param expected Expected phase kind.
   * @param operation Operation name for diagnostics.
   */
  void require_phase_kind(PassPhaseKind expected, std::string_view operation) const;

  vk::CommandBuffer command_buffer_ = VK_NULL_HANDLE;
  PassPhaseKind phase_kind_ = PassPhaseKind::kGraphics;
  PFN_vkCmdBindShadersEXT cmd_bind_shaders_ext_ = nullptr;
};

/**
 * @brief Callback recorded for one phase.
 */
using PassRecordCallback = std::function<void(PassCommandEncoder &encoder)>;

/**
 * @brief One graph phase with declaration + recording callback.
 */
struct PassPhase {
  /** @brief Declarative description for scheduling and barriers. */
  PassPhaseDesc description;
  /** @brief Recording callback for this phase. */
  PassRecordCallback record;
};

/**
 * @brief Pass graph made of queue-aware graphics/compute/transfer phases.
 */
class PassGraph {
public:
  /**
   * @brief Add one phase to the graph.
   * @param description Phase declaration.
   * @param record Phase recording callback.
   * @return Stable phase identifier.
   */
  [[nodiscard]] PassPhaseId add_phase(PassPhaseDesc description, PassRecordCallback record);

  /**
   * @brief Access one phase by identifier.
   * @param phase_id Phase identifier.
   * @return Immutable phase reference.
   */
  [[nodiscard]] const PassPhase &phase(PassPhaseId phase_id) const;

  /**
   * @brief Access all phases in insertion order.
   * @return Phase span.
   */
  [[nodiscard]] std::span<const PassPhase> phases() const noexcept;

  /**
   * @brief Clear all phases from the graph.
   */
  void clear() noexcept;

  /**
   * @brief Whether graph contains no phases.
   * @return `true` when graph is empty.
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Number of phases.
   * @return Phase count.
   */
  [[nodiscard]] std::size_t size() const noexcept;

private:
  std::vector<PassPhase> phases_;
};

/**
 * @brief External wait edge injected before one phase submit.
 */
struct PassExternalWait {
  /** @brief Target phase identifier. */
  PassPhaseId phase_id = 0U;
  /** @brief Semaphore to wait on. */
  vk::Semaphore semaphore = VK_NULL_HANDLE;
  /** @brief Stage mask used for the wait edge. */
  vk::PipelineStageFlags2 stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
};

/**
 * @brief External signal edge injected after one phase submit.
 */
struct PassExternalSignal {
  /** @brief Target phase identifier. */
  PassPhaseId phase_id = 0U;
  /** @brief Semaphore to signal. */
  vk::Semaphore semaphore = VK_NULL_HANDLE;
  /** @brief Stage mask used for the signal edge. */
  vk::PipelineStageFlags2 stage_mask = vk::PipelineStageFlagBits2::eAllCommands;
};

/**
 * @brief External synchronization hooks for one graph execution.
 */
struct PassExecutionInfo {
  /** @brief Additional waits merged into submit for selected phases. */
  std::vector<PassExternalWait> waits;
  /** @brief Additional signals merged into submit for selected phases. */
  std::vector<PassExternalSignal> signals;
};

/**
 * @brief Pass-executor creation options.
 */
struct PassExecutorCreateInfo {
  /** @brief Allow compute phases to fall back to graphics queue when async compute is unavailable. */
  bool allow_async_compute_fallback_to_graphics = true;
  /** @brief Allow transfer phases to fall back to graphics queue when transfer queue is unavailable. */
  bool allow_transfer_fallback_to_graphics = true;
};

/**
 * @brief Queue-aware pass executor using dynamic rendering + shader objects.
 */
class PassExecutor {
public:
  /**
   * @brief Create pass executor resources.
   * @param engine Initialized engine context.
   * @param info Executor creation options.
   * @return Initialized pass executor.
   */
  [[nodiscard]] static PassExecutor create(const EngineContext &engine, const PassExecutorCreateInfo &info = {});

  /**
   * @brief Move-construct the executor.
   * @param other Executor being moved from.
   */
  PassExecutor(PassExecutor &&other) noexcept = default;

  /**
   * @brief Move-assign the executor.
   * @param other Executor being moved from.
   * @return `*this`.
   */
  PassExecutor &operator=(PassExecutor &&other) noexcept = default;

  PassExecutor(const PassExecutor &) = delete;
  PassExecutor &operator=(const PassExecutor &) = delete;

  /**
   * @brief Record and execute a pass graph.
   * @param graph Pass graph.
   * @param execution_info Optional external synchronization hooks.
   */
  void execute(const PassGraph &graph, const PassExecutionInfo &execution_info = {});

  /**
   * @brief Block until executor-owned submissions are complete.
   */
  void wait_idle() const;

private:
  /**
   * @brief Queue runtime resources.
   */
  struct QueueRuntime {
    PassQueueKind queue_kind = PassQueueKind::kGraphics;
    std::uint32_t family_index = 0U;
    vk::Queue queue = VK_NULL_HANDLE;
    vk::raii::CommandPool command_pool{nullptr};
  };

  /**
   * @brief Internal constructor from initialized resources.
   */
  PassExecutor(const EngineContext *engine, const vk::raii::Device *device, std::vector<QueueRuntime> queue_runtimes, std::size_t graphics_queue_runtime_index,
               std::size_t async_compute_queue_runtime_index, std::size_t transfer_queue_runtime_index, vk::raii::Semaphore timeline_semaphore,
               PassExecutorCreateInfo create_info, PFN_vkCmdBindShadersEXT cmd_bind_shaders_ext);

  /**
   * @brief Resolve queue runtime index from a queue kind.
   * @param queue_kind Queue kind.
   * @return Queue runtime index.
   */
  [[nodiscard]] std::size_t queue_runtime_index_for(PassQueueKind queue_kind) const;

  const EngineContext *engine_ = nullptr;
  const vk::raii::Device *device_ = nullptr;
  std::vector<QueueRuntime> queue_runtimes_;
  std::size_t graphics_queue_runtime_index_ = 0U;
  std::size_t async_compute_queue_runtime_index_ = 0U;
  std::size_t transfer_queue_runtime_index_ = 0U;
  vk::raii::Semaphore timeline_semaphore_{nullptr};
  std::uint64_t completed_timeline_value_ = 0U;
  PassExecutorCreateInfo create_info_{};
  PFN_vkCmdBindShadersEXT cmd_bind_shaders_ext_ = nullptr;
};

} // namespace varre::engine

namespace varre::engine::render {
using ::varre::engine::PassBufferAccess;
using ::varre::engine::PassColorAttachmentDesc;
using ::varre::engine::PassCommandEncoder;
using ::varre::engine::PassDepthAttachmentDesc;
using ::varre::engine::PassExecutionInfo;
using ::varre::engine::PassExecutor;
using ::varre::engine::PassExecutorCreateInfo;
using ::varre::engine::PassExternalSignal;
using ::varre::engine::PassExternalWait;
using ::varre::engine::PassGraph;
using ::varre::engine::PassGraphicsRenderingInfo;
using ::varre::engine::PassImageAccess;
using ::varre::engine::PassPhase;
using ::varre::engine::PassPhaseDesc;
using ::varre::engine::PassPhaseId;
using ::varre::engine::PassPhaseKind;
using ::varre::engine::PassQueueKind;
using ::varre::engine::PassRecordCallback;
using ::varre::engine::PassResourceId;
using ::varre::engine::PassShaderBinding;
} // namespace varre::engine::render
