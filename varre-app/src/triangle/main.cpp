/**
 * @file main.cpp
 * @brief SDL3 + pass-mode triangle app bootstrap target.
 */
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include <SDL3/SDL.h>

#include "varre/app/core.hpp"
#include "varre/assets/shaders.hpp"
#include "varre/engine/assets/shaders.hpp"
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

[[nodiscard]] vk::ClearValue white_clear_color() {
  vk::ClearValue clear_value{};
  clear_value.color = vk::ClearColorValue{std::array<float, 4>{1.0F, 1.0F, 1.0F, 1.0F}};
  return clear_value;
}

void record_triangle_draw(varre::engine::PassCommandEncoder &encoder, const vk::Extent2D extent,
                          const std::span<const varre::engine::PassShaderBinding> shader_bindings) {
  encoder.bind_shaders(shader_bindings);

  const vk::Viewport viewport{
    0.0F,
    0.0F,
    static_cast<float>(extent.width),
    static_cast<float>(extent.height),
    0.0F,
    1.0F,
  };
  const vk::Rect2D scissor{vk::Offset2D{0, 0}, extent};
  encoder.set_viewports(std::span<const vk::Viewport>{&viewport, 1U});
  encoder.set_scissors(std::span<const vk::Rect2D>{&scissor, 1U});

  encoder.set_primitive_topology(vk::PrimitiveTopology::eTriangleList);
  encoder.set_rasterizer_discard_enable(false);
  encoder.set_cull_mode(vk::CullModeFlagBits::eNone);
  encoder.set_front_face(vk::FrontFace::eCounterClockwise);
  encoder.set_primitive_restart_enable(false);
  encoder.set_line_width(1.0F);

  encoder.set_depth_test_enable(false);
  encoder.set_depth_write_enable(false);
  encoder.set_depth_compare_op(vk::CompareOp::eAlways);
  encoder.set_depth_bounds_test_enable(false);
  encoder.set_depth_bias_enable(false);
  encoder.set_stencil_test_enable(false);

  const std::array<vk::Bool32, 1> color_blend_enables{VK_FALSE};
  const std::array<vk::ColorComponentFlags, 1> color_write_masks{
    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  const std::array<vk::ColorBlendEquationEXT, 1> blend_equations{
    vk::ColorBlendEquationEXT{}
      .setSrcColorBlendFactor(vk::BlendFactor::eOne)
      .setDstColorBlendFactor(vk::BlendFactor::eZero)
      .setColorBlendOp(vk::BlendOp::eAdd)
      .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
      .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
      .setAlphaBlendOp(vk::BlendOp::eAdd),
  };
  encoder.set_logic_op_enable(false);
  encoder.set_color_blend_enable(0U, color_blend_enables);
  encoder.set_color_blend_equation(0U, blend_equations);
  encoder.set_color_write_mask(0U, color_write_masks);

  encoder.set_rasterization_samples(vk::SampleCountFlagBits::e1);
  const std::array<vk::SampleMask, 1> sample_mask{0xFFFF'FFFFU};
  encoder.set_sample_mask(vk::SampleCountFlagBits::e1, sample_mask);
  encoder.set_alpha_to_coverage_enable(false);
  encoder.set_alpha_to_one_enable(false);

  encoder.draw(3U, 1U, 0U, 0U);
}

void build_triangle_frame_graph(const varre::engine::PassFrameContext &frame_context, varre::engine::PassGraph *graph,
                                const std::span<const varre::engine::PassShaderBinding> shader_bindings) {
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
                .clear_value = white_clear_color(),
              },
            },
          .depth_attachment = std::nullopt,
          .stencil_attachment = std::nullopt,
        },
    },
    [shader_bindings, extent = frame_context.extent](varre::engine::PassCommandEncoder &encoder) {
      record_triangle_draw(encoder, extent, shader_bindings);
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
    varre::engine::ShaderObjectCache shader_cache = varre::engine::ShaderObjectCache::create(app.engine());
    const std::array<varre::assets::ShaderId, 2> triangle_shader_ids{
      varre::assets::ShaderId::SHADER_TRIANGLE_VERTEX_VERTMAIN,
      varre::assets::ShaderId::SHADER_TRIANGLE_FRAGMENT_FRAGMAIN,
    };
    const varre::engine::ShaderObjectSet triangle_shader_set = shader_cache.get_or_create(varre::engine::ShaderObjectCreateRequestById{
      .shader_ids = std::span<const varre::assets::ShaderId>{triangle_shader_ids},
      .push_constant_ranges = {},
      .shader_create_flags = {},
    });
    const std::vector<varre::engine::PassShaderBinding> triangle_shader_bindings = varre::engine::make_pass_shader_bindings(triangle_shader_set);

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

      varre::engine::PassFrameRunInfo run_info{};
      run_info.recreate_info = app.swapchain_recreate_info_from_window();

      const varre::engine::PassFrameRunResult result = app.pass_frame_loop().run_frame(
        &app.swapchain(),
        [&triangle_shader_bindings](const varre::engine::PassFrameContext &context, varre::engine::PassGraph *graph) {
          build_triangle_frame_graph(context, graph, std::span<const varre::engine::PassShaderBinding>{triangle_shader_bindings});
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
