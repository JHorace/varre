/**
 * @file main.cpp
 * @brief SDL3 + pass-mode triangle app bootstrap target.
 */
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

#include <SDL3/SDL.h>

#include "varre/app/core.hpp"
#include "varre/engine/errors.hpp"
#include "varre/engine/pass_frame_loop.hpp"
#include "varre/engine/pass_mode.hpp"

namespace {

[[nodiscard]] vk::ImageSubresourceRange swapchain_color_subresource_range() {
  return vk::ImageSubresourceRange{}
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setBaseMipLevel(0U)
    .setLevelCount(1U)
    .setBaseArrayLayer(0U)
    .setLayerCount(1U);
}

[[nodiscard]] vk::ClearValue animated_clear_color(const float t_seconds) {
  const float red = 0.1F + 0.05F * (1.0F + std::sin(t_seconds * 0.8F));
  const float green = 0.1F + 0.05F * (1.0F + std::sin(t_seconds * 1.3F + 1.0F));
  const float blue = 0.2F + 0.08F * (1.0F + std::sin(t_seconds * 0.6F + 2.0F));

  vk::ClearValue clear_value{};
  clear_value.color = vk::ClearColorValue{std::array<float, 4>{red, green, blue, 1.0F}};
  return clear_value;
}

void build_triangle_frame_graph(const varre::engine::PassFrameContext &frame_context, varre::engine::PassGraph *graph, const float t_seconds) {
  if (graph == nullptr) {
    throw std::invalid_argument("build_triangle_frame_graph requires a valid graph pointer.");
  }

  const vk::ImageSubresourceRange subresource_range = swapchain_color_subresource_range();

  static_cast<void>(graph->add_phase(
    varre::engine::PassPhaseDesc{
      .name = "triangle_clear_pass",
      .kind = varre::engine::PassPhaseKind::kGraphics,
      .queue = varre::engine::PassQueueKind::kGraphics,
      .explicit_dependencies = {},
      .buffer_accesses = {},
      .image_accesses = {},
      .graphics_rendering =
        varre::engine::PassGraphicsRenderingInfo{
          .render_area = vk::Rect2D{vk::Offset2D{0, 0}, frame_context.extent},
          .layer_count = 1U,
          .view_mask = 0U,
          .color_attachments =
            {
              varre::engine::PassColorAttachmentDesc{
                .resource_id = frame_context.swapchain_resource_id,
                .image = frame_context.image,
                .subresource_range = subresource_range,
                .image_view = frame_context.image_view,
                .load_op = vk::AttachmentLoadOp::eClear,
                .store_op = vk::AttachmentStoreOp::eStore,
                .clear_value = animated_clear_color(t_seconds),
              },
            },
          .depth_attachment = std::nullopt,
          .stencil_attachment = std::nullopt,
        },
    },
    [](varre::engine::PassCommandEncoder & /*encoder*/) {
      // Triangle draw submission is intentionally deferred to a later app step.
    }));
}

} // namespace

int main() {
  try {
    varre::app::AppCoreCreateInfo create_info{};
    create_info.window.title = "varre-app triangle";
    create_info.window.width = 1280;
    create_info.window.height = 720;

    create_info.engine.application_name = "varre-app-triangle";
    create_info.engine.enable_validation = true;

    varre::app::AppCore app = varre::app::AppCore::create(create_info);
    const auto start_time = std::chrono::steady_clock::now();

    bool running = true;
    while (running) {
      bool recreate_requested = false;
      SDL_Event event{};
      while (app.poll_event(&event)) {
        const varre::app::AppEventSignals signals = app.classify_event(event);
        if (signals.close_requested) {
          running = false;
          break;
        }
        recreate_requested = recreate_requested || signals.swapchain_recreate_requested;
      }
      if (!running) {
        break;
      }

      if (recreate_requested) {
        try {
          static_cast<void>(app.pass_frame_loop().frame_loop().try_recreate_swapchain(&app.swapchain(), app.swapchain_recreate_info_from_window()));
        } catch (const varre::engine::EngineError &error) {
          std::cerr << "Swapchain recreate failed: " << error.what() << '\n';
        }
      }

      const auto now = std::chrono::steady_clock::now();
      const float t_seconds = std::chrono::duration<float>(now - start_time).count();

      varre::engine::PassFrameRunInfo run_info{};
      run_info.recreate_info = app.swapchain_recreate_info_from_window();

      const varre::engine::PassFrameRunResult result = app.pass_frame_loop().run_frame(
        &app.swapchain(), [t_seconds](const varre::engine::PassFrameContext &context, varre::engine::PassGraph *graph) {
          build_triangle_frame_graph(context, graph, t_seconds);
        },
        run_info);

      if (result.status == varre::engine::PassFrameRunStatus::kSwapchainRecreateDeferred) {
        SDL_Delay(10U);
      }
    }

    app.wait_idle();
    return 0;
  } catch (const varre::engine::EngineError &error) {
    std::cerr << "Engine error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "Fatal error: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "Fatal error: unknown exception.\n";
    return 1;
  }
}
