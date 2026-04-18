/**
 * @file pass_frame_loop.cpp
 * @brief Pass-mode frame orchestration implementation.
 */
#include "varre/engine/render/pass_frame_loop.hpp"

#include <string>
#include <string_view>
#include <utility>

#include <fmt/format.h>

#include "varre/engine/core/errors.hpp"

namespace varre::engine {
namespace detail {
constexpr PassResourceId kInternalSwapchainPresentResourceIdBase = 0xFFFF'0000'0000'0000ULL;
constexpr std::string_view kInternalPresentTransitionPhaseName = "__varre_present_transition";

void validate_swapchain_resource_id_usage(const PassGraph &graph, const PassFrameContext &frame_context) {
  const auto validate_swapchain_resource = [&](const PassPhaseDesc &phase_desc, const PassResourceId resource_id, const vk::Image image,
                                               const std::string_view usage_label) {
    if (image != frame_context.image) {
      return;
    }
    if (resource_id != frame_context.swapchain_resource_id) {
      throw make_engine_error(
        EngineErrorCode::kInvalidArgument,
        fmt::format("Phase '{}' uses swapchain image in {} with resource id {} but frame-context swapchain_resource_id is {}.",
                    phase_desc.name, usage_label, resource_id, frame_context.swapchain_resource_id));
    }
  };

  for (const PassPhase &phase : graph.phases()) {
    for (const PassImageAccess &image_access : phase.description.image_accesses) {
      validate_swapchain_resource(phase.description, image_access.resource_id, image_access.image, "PassImageAccess");
    }

    if (!phase.description.graphics_rendering.has_value()) {
      continue;
    }

    const PassGraphicsRenderingInfo &rendering = *phase.description.graphics_rendering;
    for (const PassColorAttachmentDesc &color_attachment : rendering.color_attachments) {
      validate_swapchain_resource(phase.description, color_attachment.resource_id, color_attachment.image, "color attachment");
      if (color_attachment.resolve_image_view != VK_NULL_HANDLE && color_attachment.resolve_mode != vk::ResolveModeFlagBits::eNone) {
        validate_swapchain_resource(phase.description, color_attachment.resolve_resource_id, color_attachment.resolve_image, "resolve attachment");
      }
    }

    if (rendering.depth_attachment.has_value()) {
      const PassDepthAttachmentDesc &depth_attachment = *rendering.depth_attachment;
      validate_swapchain_resource(phase.description, depth_attachment.resource_id, depth_attachment.image, "depth attachment");
    }
    if (rendering.stencil_attachment.has_value()) {
      const PassDepthAttachmentDesc &stencil_attachment = *rendering.stencil_attachment;
      validate_swapchain_resource(phase.description, stencil_attachment.resource_id, stencil_attachment.image, "stencil attachment");
    }
  }
}

void append_present_transition_phase(PassGraph *graph, const PassFrameContext &frame_context) {
  if (graph == nullptr || graph->empty()) {
    return;
  }

  std::vector<PassPhaseId> dependencies;
  dependencies.reserve(graph->size());
  for (PassPhaseId phase_id = 0U; phase_id < static_cast<PassPhaseId>(graph->size()); ++phase_id) {
    dependencies.push_back(phase_id);
  }

  static_cast<void>(graph->add_phase(
    PassPhaseDesc{
      .name = std::string{kInternalPresentTransitionPhaseName},
      .kind = PassPhaseKind::kTransfer,
      .queue = PassQueueKind::kGraphics,
      .explicit_dependencies = std::move(dependencies),
      .buffer_accesses = {},
      .image_accesses =
        {
          PassImageAccess{
            .resource_id = frame_context.swapchain_resource_id,
            .image = frame_context.image,
            .subresource_range =
              vk::ImageSubresourceRange{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0U)
                .setLevelCount(1U)
                .setBaseArrayLayer(0U)
                .setLayerCount(1U),
            .stage_mask = vk::PipelineStageFlagBits2::eAllCommands,
            .access_mask = vk::AccessFlagBits2::eMemoryRead,
            .writes = false,
          },
        },
      .graphics_rendering = std::nullopt,
    },
    [](PassCommandEncoder & /*encoder*/) {}));
}
} // namespace detail

PassFrameLoop::PassFrameLoop(FrameLoop &&frame_loop, PassExecutor &&pass_executor)
    : frame_loop_(std::move(frame_loop)), pass_executor_(std::move(pass_executor)) {}

PassFrameLoop PassFrameLoop::create(const EngineContext &engine, const SwapchainContext &swapchain, const FrameLoopCreateInfo &frame_loop_create_info,
                                    const PassExecutorCreateInfo &pass_executor_create_info) {
  return PassFrameLoop{
    FrameLoop::create(engine, swapchain, frame_loop_create_info),
    PassExecutor::create(engine, pass_executor_create_info),
  };
}

bool PassFrameLoop::try_recreate_swapchain(SwapchainContext *swapchain, const PassFrameRunInfo &run_info) {
  const bool recreated =
    run_info.recreate_info.has_value() ? frame_loop_.try_recreate_swapchain(swapchain, *run_info.recreate_info) : frame_loop_.try_recreate_swapchain(swapchain);
  if (recreated) {
    pass_executor_.reset_tracked_image_states();
  }
  if (recreated && run_info.on_swapchain_recreated) {
    run_info.on_swapchain_recreated(*swapchain);
  }
  return recreated;
}

PassFrameRunResult PassFrameLoop::run_frame(SwapchainContext *swapchain, const PassGraphBuildCallback &build_graph, const PassFrameRunInfo &run_info) {
  if (swapchain == nullptr) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "PassFrameLoop::run_frame requires a valid SwapchainContext pointer.");
  }
  if (!build_graph) {
    throw make_engine_error(EngineErrorCode::kInvalidArgument, "PassFrameLoop::run_frame requires a valid pass-graph builder callback.");
  }

  const AcquiredFrame acquired = frame_loop_.acquire_next_image(*swapchain, run_info.acquire_timeout_ns);
  PassFrameRunResult result{
    .status = PassFrameRunStatus::kPresented,
    .acquire_status = acquired.status,
    .present_status = std::nullopt,
    .error_code = acquired.error_code,
    .frame_index = acquired.frame_index,
    .image_index = acquired.image_index,
    .executed_graph = false,
    .swapchain_recreated = false,
  };

  if (acquired.status == FrameAcquireStatus::kOutOfDate) {
    result.swapchain_recreated = try_recreate_swapchain(swapchain, run_info);
    result.status = result.swapchain_recreated ? PassFrameRunStatus::kSwapchainRecreatedBeforeRender : PassFrameRunStatus::kSwapchainRecreateDeferred;
    return result;
  }
  if (acquired.image_index >= swapchain->image_count()) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "PassFrameLoop acquire returned an out-of-range swapchain image index.");
  }

  PassGraph graph;
  const std::span<const vk::Image> images = swapchain->images();
  const std::span<const vk::raii::ImageView> image_views = swapchain->image_views();
  if (acquired.image_index >= images.size() || acquired.image_index >= image_views.size()) {
    throw make_engine_error(EngineErrorCode::kInvalidState, "PassFrameLoop could not resolve acquired swapchain image/image_view.");
  }

  const PassFrameContext frame_context{
    .frame_index = acquired.frame_index,
    .image_index = acquired.image_index,
    .swapchain_resource_id = detail::kInternalSwapchainPresentResourceIdBase + static_cast<PassResourceId>(acquired.image_index),
    .extent = swapchain->extent(),
    .image_format = swapchain->image_format(),
    .image = images[acquired.image_index],
    .image_view = *image_views[acquired.image_index],
  };
  build_graph(frame_context, &graph);
  detail::validate_swapchain_resource_id_usage(graph, frame_context);
  detail::append_present_transition_phase(&graph, frame_context);

  if (graph.empty()) {
    frame_loop_.submit_graphics_batch(GraphicsSubmitBatch{
      .waits = {},
      .command_buffers = {},
      .signals = {},
      .wait_for_swapchain_image = true,
      .swapchain_image_wait_stage = vk::PipelineStageFlagBits::eAllCommands,
      .signal_render_finished = true,
    });
  } else {
    const FrameSyncPrimitives &sync = frame_loop_.current_sync();
    PassExecutionInfo execution_info{
      .waits =
        {
          PassExternalWait{
            .phase_id = 0U,
            .semaphore = *sync.image_available,
            .stage_mask = run_info.image_available_wait_stage_mask,
          },
        },
      .signals =
        {
          PassExternalSignal{
            .phase_id = static_cast<PassPhaseId>(graph.size() - 1U),
            .semaphore = sync.render_finished,
            .stage_mask = run_info.render_finished_signal_stage_mask,
          },
        },
      .signal_fence = *sync.in_flight,
    };
    pass_executor_.execute(graph, execution_info);
    frame_loop_.notify_external_submit(true);
    result.executed_graph = true;
  }

  const PresentedFrame presented = frame_loop_.present(*swapchain, acquired.image_index);
  result.present_status = presented.status;
  if (!result.error_code.has_value()) {
    result.error_code = presented.error_code;
  }

  if (frame_loop_.swapchain_recreation_required()) {
    result.swapchain_recreated = try_recreate_swapchain(swapchain, run_info);
    result.status = result.swapchain_recreated ? PassFrameRunStatus::kPresentedAndSwapchainRecreated : PassFrameRunStatus::kSwapchainRecreateDeferred;
    return result;
  }

  result.status = PassFrameRunStatus::kPresented;
  return result;
}

FrameLoop &PassFrameLoop::frame_loop() noexcept { return frame_loop_; }

const FrameLoop &PassFrameLoop::frame_loop() const noexcept { return frame_loop_; }

PassExecutor &PassFrameLoop::pass_executor() noexcept { return pass_executor_; }

const PassExecutor &PassFrameLoop::pass_executor() const noexcept { return pass_executor_; }

void PassFrameLoop::wait_idle() const {
  frame_loop_.wait_idle();
  pass_executor_.wait_idle();
}

} // namespace varre::engine
