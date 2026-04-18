/**
 * @file main.cpp
 * @brief SDL3 + pass-mode mesh_simple app bootstrap target.
 */
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include <optional>

#include <SDL3/SDL.h>

#include "varre/app/core.hpp"
#include "varre/assets/models.hpp"
#include "varre/assets/shaders.hpp"
#include "varre/engine/assets/models.hpp"
#include "varre/engine/assets/shader_objects.hpp"
#include "varre/engine/assets/textures.hpp"
#include "varre/engine/errors.hpp"
#include "varre/engine/pass_frame_loop.hpp"
#include "varre/engine/pass_mode.hpp"

namespace {

constexpr varre::engine::PassResourceId kMeshVertexResourceId = 100U;
constexpr varre::engine::PassResourceId kMeshIndexResourceId = 101U;
constexpr varre::engine::PassResourceId kMeshDepthResourceId = 102U;

struct GpuDepthBuffer {
  varre::engine::GpuImage image;
  vk::raii::ImageView view = nullptr;

  static GpuDepthBuffer create(const varre::engine::EngineContext &engine, const varre::engine::TextureUploadService &texture_service,
                               const std::uint32_t width, const std::uint32_t height) {
    varre::engine::GpuImage depth_image =
      texture_service.create_device_local_image(width, height, vk::Format::eD32Sfloat, vk::ImageUsageFlagBits::eDepthStencilAttachment);

    vk::raii::ImageView view = texture_service.create_image_view(depth_image.image(), vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth);
    return GpuDepthBuffer{
      .image = std::move(depth_image),
      .view = std::move(view),
    };
  }
};

[[nodiscard]] vk::ImageSubresourceRange swapchain_color_subresource_range() {
  return vk::ImageSubresourceRange{}
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setBaseMipLevel(0U)
    .setLevelCount(1U)
    .setBaseArrayLayer(0U)
    .setLayerCount(1U);
}

[[nodiscard]] vk::ClearValue dark_clear_color() {
  vk::ClearValue clear_value{};
  clear_value.color = vk::ClearColorValue{std::array<float, 4>{0.10F, 0.10F, 0.10F, 1.0F}};
  return clear_value;
}

void fit_model_to_view_in_place(varre::assets::ModelAsset *model, float width, float height) {
  if (model == nullptr || model->vertices.empty()) {
    throw std::invalid_argument("fit_model_to_view_in_place requires a non-empty model.");
  }

  const float aspect_ratio = width / height;

  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  float max_z = std::numeric_limits<float>::lowest();

  for (const varre::assets::Vertex &vertex : model->vertices) {
    min_x = std::min(min_x, vertex.px);
    min_y = std::min(min_y, vertex.py);
    min_z = std::min(min_z, vertex.pz);
    max_x = std::max(max_x, vertex.px);
    max_y = std::max(max_y, vertex.py);
    max_z = std::max(max_z, vertex.pz);
  }

  const float center_x = (min_x + max_x) * 0.5F;
  const float center_y = (min_y + max_y) * 0.5F;
  const float center_z = (min_z + max_z) * 0.5F;

  const float extent_x = max_x - min_x;
  const float extent_y = max_y - min_y;
  const float extent_z = max_z - min_z;
  const float max_extent = std::max({extent_x, extent_y, extent_z});

  // Scale the largest dimension to fit comfortably within the NDC range [-1, 1].
  // Using 0.8 to leave some margin and fit Z into [0, 1].
  const float scale = (max_extent > 0.0F) ? (0.8F / max_extent) : 1.0F;

  for (varre::assets::Vertex &vertex : model->vertices) {
    vertex.px = (vertex.px - center_x) * scale / aspect_ratio;
    vertex.py = -(vertex.py - center_y) * scale;
    vertex.pz = (vertex.pz - center_z) * scale + 0.5F;

    vertex.cx = 1.0F;
    vertex.cy = 0.0F;
    vertex.cz = 0.0F;

    vertex.ny = -vertex.ny;
  }
}

void record_mesh_draw(varre::engine::PassCommandEncoder &encoder, const vk::Extent2D extent,
                      const std::span<const varre::engine::PassShaderBinding> shader_bindings, const varre::engine::GpuMesh &mesh) {
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

  const vk::VertexInputBindingDescription2EXT binding_description =
    vk::VertexInputBindingDescription2EXT{}.setBinding(0U).setStride(sizeof(varre::assets::Vertex)).setInputRate(vk::VertexInputRate::eVertex).setDivisor(1U);
  const std::array<vk::VertexInputAttributeDescription2EXT, 3> attribute_descriptions{
    vk::VertexInputAttributeDescription2EXT{}
      .setLocation(0U)
      .setBinding(0U)
      .setFormat(vk::Format::eR32G32B32Sfloat)
      .setOffset(static_cast<std::uint32_t>(offsetof(varre::assets::Vertex, px))),
    vk::VertexInputAttributeDescription2EXT{}
      .setLocation(1U)
      .setBinding(0U)
      .setFormat(vk::Format::eR32G32B32Sfloat)
      .setOffset(static_cast<std::uint32_t>(offsetof(varre::assets::Vertex, cx))),
    vk::VertexInputAttributeDescription2EXT{}
      .setLocation(2U)
      .setBinding(0U)
      .setFormat(vk::Format::eR32G32B32Sfloat)
      .setOffset(static_cast<std::uint32_t>(offsetof(varre::assets::Vertex, nx))),
  };
  encoder.set_vertex_input(std::span<const vk::VertexInputBindingDescription2EXT>{&binding_description, 1U}, attribute_descriptions);

  encoder.set_primitive_topology(vk::PrimitiveTopology::eTriangleList);
  encoder.set_polygon_mode(vk::PolygonMode::eFill);
  encoder.set_rasterizer_discard_enable(false);
  encoder.set_cull_mode(vk::CullModeFlagBits::eBack);
  encoder.set_front_face(vk::FrontFace::eClockwise);
  encoder.set_primitive_restart_enable(false);
  encoder.set_line_width(1.0F);

  encoder.set_depth_test_enable(true);
  encoder.set_depth_write_enable(true);
  encoder.set_depth_compare_op(vk::CompareOp::eLessOrEqual);
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

  const std::array<vk::Buffer, 1> vertex_buffers{mesh.vertex_buffer.buffer()};
  const std::array<vk::DeviceSize, 1> vertex_offsets{0U};
  encoder.bind_vertex_buffers(0U, vertex_buffers, vertex_offsets);

  if (mesh.index_count > 0U) {
    encoder.bind_index_buffer(mesh.index_buffer.buffer(), 0U, mesh.index_type);
    encoder.draw_indexed(mesh.index_count, 1U, 0U, 0, 0U);
  } else {
    encoder.draw(mesh.vertex_count, 1U, 0U, 0U);
  }
}

void build_mesh_frame_graph(const varre::engine::PassFrameContext &frame_context, varre::engine::PassGraph *graph,
                            const std::span<const varre::engine::PassShaderBinding> shader_bindings, const varre::engine::GpuMesh &mesh,
                            const GpuDepthBuffer &depth_buffer) {
  if (graph == nullptr) {
    throw std::invalid_argument("build_mesh_frame_graph requires a valid graph pointer.");
  }

  std::vector<varre::engine::PassBufferAccess> buffer_accesses;
  buffer_accesses.reserve(2U);
  buffer_accesses.push_back(varre::engine::PassBufferAccess{
    .resource_id = kMeshVertexResourceId,
    .buffer = mesh.vertex_buffer.buffer(),
    .offset = 0U,
    .size = mesh.vertex_buffer.size_bytes(),
    .stage_mask = vk::PipelineStageFlagBits2::eVertexInput,
    .access_mask = vk::AccessFlagBits2::eVertexAttributeRead,
    .writes = false,
  });
  if (mesh.index_count > 0U) {
    buffer_accesses.push_back(varre::engine::PassBufferAccess{
      .resource_id = kMeshIndexResourceId,
      .buffer = mesh.index_buffer.buffer(),
      .offset = 0U,
      .size = mesh.index_buffer.size_bytes(),
      .stage_mask = vk::PipelineStageFlagBits2::eVertexInput,
      .access_mask = vk::AccessFlagBits2::eIndexRead,
      .writes = false,
    });
  }

  const vk::ImageSubresourceRange subresource_range = swapchain_color_subresource_range();
  const vk::ImageSubresourceRange depth_subresource_range =
    vk::ImageSubresourceRange{}.setAspectMask(vk::ImageAspectFlagBits::eDepth).setBaseMipLevel(0U).setLevelCount(1U).setBaseArrayLayer(0U).setLayerCount(1U);

  static_cast<void>(graph->add_phase(
    varre::engine::PassPhaseDesc{
      .name = "mesh_simple_pass",
      .kind = varre::engine::PassPhaseKind::kGraphics,
      .queue = varre::engine::PassQueueKind::kGraphics,
      .explicit_dependencies = {},
      .buffer_accesses = std::move(buffer_accesses),
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
                .clear_value = dark_clear_color(),
              },
            },
          .depth_attachment =
            varre::engine::PassDepthAttachmentDesc{
              .resource_id = kMeshDepthResourceId,
              .image = depth_buffer.image.image(),
              .subresource_range = depth_subresource_range,
              .image_view = *depth_buffer.view,
              .load_op = vk::AttachmentLoadOp::eClear,
              .store_op = vk::AttachmentStoreOp::eDontCare,
              .clear_value = vk::ClearValue{vk::ClearDepthStencilValue{1.0F, 0U}},
            },
          .stencil_attachment = std::nullopt,
        },
    },
    [shader_bindings, extent = frame_context.extent, &mesh](varre::engine::PassCommandEncoder &encoder) {
      record_mesh_draw(encoder, extent, shader_bindings, mesh);
    }));
}

} // namespace

int main() {
  try {
    varre::app::AppCoreCreateInfo create_info{};
    create_info.window.title = "varre-app mesh_simple";
    create_info.window.width = 1280;
    create_info.window.height = 720;

    create_info.engine.application_name = "varre-app-mesh_simple";
    create_info.engine.enable_validation = true;

    varre::app::AppCore app = varre::app::AppCore::create(create_info);
    varre::engine::ShaderObjectCache shader_cache = varre::engine::ShaderObjectCache::create(app.engine());
    varre::engine::ModelUploadService model_upload = varre::engine::ModelUploadService::create(app.engine());
    varre::engine::TextureUploadService texture_service = varre::engine::TextureUploadService::create(app.engine());

    varre::assets::ModelId mesh_id = varre::assets::ModelId::UTAH_TEAPOT;
    varre::assets::ModelAsset model = varre::assets::load_model(mesh_id);
    fit_model_to_view_in_place(&model, static_cast<float>(create_info.window.width), static_cast<float>(create_info.window.height));
    const varre::engine::GpuMesh &mesh = model_upload.upload_and_cache(model);

    GpuDepthBuffer depth_buffer = GpuDepthBuffer::create(app.engine(), texture_service, app.swapchain().extent().width, app.swapchain().extent().height);

    const std::array<varre::engine::ShaderObjectCreateEntryById, 2> mesh_shader_entries{
      varre::engine::ShaderObjectCreateEntryById{
        .shader_id = varre::assets::ShaderId::MESH_SIMPLE_VERTEX_VERTEXMAIN,
        .next_stage = vk::ShaderStageFlagBits::eFragment,
      },
      varre::engine::ShaderObjectCreateEntryById{
        .shader_id = varre::assets::ShaderId::MESH_SIMPLE_FRAGMENT_FRAGMENTMAIN,
        .next_stage = {},
      },
    };
    const varre::engine::ShaderObjectSet mesh_shader_set = shader_cache.get_or_create(varre::engine::ShaderObjectCreateRequestById{
      .shaders = mesh_shader_entries,
      .push_constant_ranges = {},
      .shader_create_flags = {},
    });
    const std::vector<varre::engine::PassShaderBinding> mesh_shader_bindings = varre::engine::make_pass_shader_bindings(mesh_shader_set);

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
          depth_buffer = GpuDepthBuffer::create(app.engine(), texture_service, app.swapchain().extent().width, app.swapchain().extent().height);
        } catch (const varre::engine::EngineError &error) {
          std::cerr << "Swapchain recreate failed: " << error.what() << '\n';
        }
      }

      varre::engine::PassFrameRunInfo run_info{};
      run_info.recreate_info = app.swapchain_recreate_info_from_window();

      const varre::engine::PassFrameRunResult result = app.pass_frame_loop().run_frame(
        &app.swapchain(),
        [&mesh_shader_bindings, &mesh, &depth_buffer](const varre::engine::PassFrameContext &context, varre::engine::PassGraph *graph) {
          build_mesh_frame_graph(context, graph, std::span<const varre::engine::PassShaderBinding>{mesh_shader_bindings}, mesh, depth_buffer);
        },
        run_info);

      if (result.status == varre::engine::PassFrameRunStatus::kSwapchainRecreateDeferred) {
        SDL_Delay(10U);
      }
    }

    model_upload.wait_idle();
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
