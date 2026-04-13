#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "varre/assets/models.hpp"
#include "varre/assets/shaders.hpp"
#include "varre/engine/assets/models.hpp"
#include "varre/engine/assets/shader_objects.hpp"
#include "varre/engine/core/engine.hpp"
#include "varre/engine/core/errors.hpp"
#include "varre/engine/core/surface.hpp"
#include "varre/engine/core/swapchain.hpp"
#include "varre/engine/render/pass_mode.hpp"
#include "varre/engine/sync/frame_loop.hpp"

namespace {

/**
 * @brief Result wrapper for best-effort engine creation in runtime tests.
 */
struct EngineAttempt {
  std::optional<varre::engine::EngineContext> engine;
  std::string reason;
};

/**
 * @brief Try creating an engine and capture a skip reason on environment failures.
 * @param info Engine init configuration.
 * @return Engine instance on success, otherwise skip reason.
 */
[[nodiscard]] EngineAttempt try_create_engine(const varre::engine::EngineInitInfo &info = {}) {
  try {
    return EngineAttempt{
      .engine = varre::engine::EngineContext::create(info),
      .reason = {},
    };
  } catch (const varre::engine::EngineError &error) {
    return EngineAttempt{
      .engine = std::nullopt,
      .reason = std::string(error.what()),
    };
  } catch (const std::exception &error) {
    return EngineAttempt{
      .engine = std::nullopt,
      .reason = std::string(error.what()),
    };
  }
}

/**
 * @brief Find one compatible memory type index for allocation requirements.
 * @param memory_properties Physical-device memory properties.
 * @param memory_type_bits Vulkan memory type bit mask.
 * @return Memory type index.
 */
[[nodiscard]] std::uint32_t find_memory_type_index(const vk::PhysicalDeviceMemoryProperties &memory_properties, const std::uint32_t memory_type_bits) {
  for (std::uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
    if ((memory_type_bits & (1U << index)) != 0U) {
      return index;
    }
  }
  throw std::runtime_error("No compatible Vulkan memory type found.");
}

/**
 * @brief Lightweight RAII buffer allocation used by pass-mode tests.
 */
struct TestBufferAllocation {
  vk::raii::Buffer buffer{nullptr};
  vk::raii::DeviceMemory memory{nullptr};
};

/**
 * @brief Create and bind one test buffer.
 * @param engine Initialized engine context.
 * @param size_bytes Buffer size.
 * @param usage Buffer usage flags.
 * @return Created Vulkan buffer + memory allocation.
 */
[[nodiscard]] TestBufferAllocation create_test_buffer(const varre::engine::EngineContext &engine, const vk::DeviceSize size_bytes,
                                                      const vk::BufferUsageFlags usage) {
  const vk::BufferCreateInfo buffer_create_info = vk::BufferCreateInfo{}.setSize(size_bytes).setUsage(usage).setSharingMode(vk::SharingMode::eExclusive);
  vk::raii::Buffer buffer(engine.device(), buffer_create_info);
  const vk::MemoryRequirements requirements = buffer.getMemoryRequirements();

  const vk::PhysicalDeviceMemoryProperties memory_properties = engine.physical_device_raii().getMemoryProperties();
  const std::uint32_t memory_type_index = find_memory_type_index(memory_properties, requirements.memoryTypeBits);
  const vk::MemoryAllocateInfo allocate_info = vk::MemoryAllocateInfo{}.setAllocationSize(requirements.size).setMemoryTypeIndex(memory_type_index);
  vk::raii::DeviceMemory memory(engine.device(), allocate_info);
  buffer.bindMemory(*memory, 0U);

  return TestBufferAllocation{
    .buffer = std::move(buffer),
    .memory = std::move(memory),
  };
}

/**
 * @brief Create one headless Vulkan surface when VK_EXT_headless_surface is available.
 * @param engine Initialized engine context.
 * @param out_reason Skip reason on failure.
 * @return Surface context when created.
 */
[[nodiscard]] std::optional<varre::engine::SurfaceContext> try_create_headless_surface(const varre::engine::EngineContext &engine, std::string *out_reason) {
  const auto create_headless_surface =
    reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(vkGetInstanceProcAddr(static_cast<VkInstance>(*engine.instance()), "vkCreateHeadlessSurfaceEXT"));
  if (create_headless_surface == nullptr) {
    *out_reason = "vkCreateHeadlessSurfaceEXT is unavailable.";
    return std::nullopt;
  }

  const VkHeadlessSurfaceCreateInfoEXT create_info{
    .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
    .pNext = nullptr,
    .flags = 0U,
  };

  VkSurfaceKHR surface = VK_NULL_HANDLE;
  const VkResult result = create_headless_surface(static_cast<VkInstance>(*engine.instance()), &create_info, nullptr, &surface);
  if (result != VK_SUCCESS) {
    *out_reason = "vkCreateHeadlessSurfaceEXT failed.";
    return std::nullopt;
  }

  try {
    return varre::engine::SurfaceContext::adopt(engine.instance(), surface);
  } catch (...) {
    vkDestroySurfaceKHR(static_cast<VkInstance>(*engine.instance()), surface, nullptr);
    throw;
  }
}

} // namespace

TEST_CASE("engine init enforces Vulkan 1.3+ runtime baseline", "[engine][core]") {
  varre::engine::EngineInitInfo init_info{};
  init_info.api_version = VK_API_VERSION_1_2;

  try {
    static_cast<void>(varre::engine::EngineContext::create(init_info));
    FAIL("EngineContext::create should reject api_version < 1.3.");
  } catch (const varre::engine::EngineError &error) {
    REQUIRE(error.code() == varre::engine::EngineErrorCode::kMissingRequirement);
  }
}

TEST_CASE("engine selects queues and resolves feature/extension profile", "[engine][core]") {
  varre::engine::EngineInitInfo init_info{};
  init_info.device_profile.optional_extensions.push_back("VK_VARRE_nonexistent_optional_extension");
  const EngineAttempt attempt = try_create_engine(init_info);
  if (!attempt.engine.has_value()) {
    SKIP(attempt.reason);
  }

  const varre::engine::EngineContext &engine = *attempt.engine;
  const varre::engine::DeviceProfile &profile = engine.device_profile();

  REQUIRE(profile.api_version >= VK_API_VERSION_1_3);
  REQUIRE(std::ranges::find(profile.enabled_extensions, std::string{VK_EXT_SHADER_OBJECT_EXTENSION_NAME}) != profile.enabled_extensions.end());
  REQUIRE(std::ranges::find(profile.missing_optional_extensions, std::string{"VK_VARRE_nonexistent_optional_extension"}) !=
          profile.missing_optional_extensions.end());

  const varre::engine::DeviceQueueTopology &topology = engine.device_queue_topology();
  REQUIRE(topology.graphics_queue != VK_NULL_HANDLE);
  REQUIRE(topology.families.async_compute.has_value());
  REQUIRE(topology.families.transfer.has_value());
}

TEST_CASE("engine rejects impossible required device extension sets", "[engine][core]") {
  varre::engine::EngineInitInfo init_info{};
  init_info.device_profile.required_extensions.push_back("VK_VARRE_nonexistent_required_extension");

  try {
    static_cast<void>(varre::engine::EngineContext::create(init_info));
    FAIL("EngineContext::create should fail when required device extensions are unavailable.");
  } catch (const varre::engine::EngineError &error) {
    REQUIRE(error.code() == varre::engine::EngineErrorCode::kNoSuitableDevice);
  } catch (const std::exception &error) {
    SKIP(error.what());
  } catch (...) {
    SKIP("EngineContext::create failed with a non-standard exception in this environment.");
  }
}

TEST_CASE("pass executor builds implicit dependencies for shared buffer resources", "[engine][pass][barriers]") {
  const EngineAttempt attempt = try_create_engine();
  if (!attempt.engine.has_value()) {
    SKIP(attempt.reason);
  }
  const varre::engine::EngineContext &engine = *attempt.engine;

  varre::engine::PassExecutor executor = varre::engine::PassExecutor::create(engine);
  TestBufferAllocation buffer =
    create_test_buffer(engine, 256U, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc);

  varre::engine::PassGraph graph;
  static_cast<void>(graph.add_phase(
    varre::engine::PassPhaseDesc{
      .name = "upload_like_transfer",
      .kind = varre::engine::PassPhaseKind::kTransfer,
      .queue = varre::engine::PassQueueKind::kTransfer,
      .explicit_dependencies = {},
      .buffer_accesses =
        {
          varre::engine::PassBufferAccess{
            .resource_id = 1U,
            .buffer = *buffer.buffer,
            .offset = 0U,
            .size = 256U,
            .stage_mask = vk::PipelineStageFlagBits2::eTransfer,
            .access_mask = vk::AccessFlagBits2::eTransferWrite,
            .writes = true,
          },
        },
      .image_accesses = {},
      .graphics_rendering = std::nullopt,
    },
    [](varre::engine::PassCommandEncoder & /*encoder*/) {}));

  static_cast<void>(graph.add_phase(
    varre::engine::PassPhaseDesc{
      .name = "compute_consume",
      .kind = varre::engine::PassPhaseKind::kCompute,
      .queue = varre::engine::PassQueueKind::kAsyncCompute,
      .explicit_dependencies = {},
      .buffer_accesses =
        {
          varre::engine::PassBufferAccess{
            .resource_id = 1U,
            .buffer = *buffer.buffer,
            .offset = 0U,
            .size = 256U,
            .stage_mask = vk::PipelineStageFlagBits2::eComputeShader,
            .access_mask = vk::AccessFlagBits2::eShaderRead,
            .writes = false,
          },
        },
      .image_accesses = {},
      .graphics_rendering = std::nullopt,
    },
    [](varre::engine::PassCommandEncoder & /*encoder*/) {}));

  REQUIRE_NOTHROW(executor.execute(graph));
}

TEST_CASE("pass executor validates malformed phase resource declarations", "[engine][pass][validation]") {
  const EngineAttempt attempt = try_create_engine();
  if (!attempt.engine.has_value()) {
    SKIP(attempt.reason);
  }
  const varre::engine::EngineContext &engine = *attempt.engine;

  varre::engine::PassExecutor executor = varre::engine::PassExecutor::create(engine);
  TestBufferAllocation buffer = create_test_buffer(engine, 256U, vk::BufferUsageFlagBits::eStorageBuffer);

  varre::engine::PassGraph graph;
  static_cast<void>(graph.add_phase(
    varre::engine::PassPhaseDesc{
      .name = "invalid_stage_mask",
      .kind = varre::engine::PassPhaseKind::kCompute,
      .queue = varre::engine::PassQueueKind::kAuto,
      .explicit_dependencies = {},
      .buffer_accesses =
        {
          varre::engine::PassBufferAccess{
            .resource_id = 1U,
            .buffer = *buffer.buffer,
            .offset = 0U,
            .size = 64U,
            .stage_mask = vk::PipelineStageFlags2{},
            .access_mask = vk::AccessFlagBits2::eShaderRead,
            .writes = false,
          },
        },
      .image_accesses = {},
      .graphics_rendering = std::nullopt,
    },
    [](varre::engine::PassCommandEncoder & /*encoder*/) {}));

  try {
    executor.execute(graph);
    FAIL("PassExecutor::execute should reject empty stage_mask in resource access.");
  } catch (const varre::engine::EngineError &error) {
    REQUIRE(error.code() == varre::engine::EngineErrorCode::kInvalidArgument);
  }
}

TEST_CASE("pass executor cross-queue timeline synchronization path executes", "[engine][pass][timeline]") {
  const EngineAttempt attempt = try_create_engine();
  if (!attempt.engine.has_value()) {
    SKIP(attempt.reason);
  }
  const varre::engine::EngineContext &engine = *attempt.engine;
  const varre::engine::QueueFamilyIndices &families = engine.device_queue_topology().families;
  if (!families.transfer.has_value() || !families.async_compute.has_value() || (*families.transfer == *families.async_compute)) {
    SKIP("Cross-queue transfer/compute topology is not available on this device.");
  }

  varre::engine::PassExecutor executor = varre::engine::PassExecutor::create(engine);
  TestBufferAllocation buffer =
    create_test_buffer(engine, 256U, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc);

  varre::engine::PassGraph graph;
  static_cast<void>(graph.add_phase(
    varre::engine::PassPhaseDesc{
      .name = "xfer",
      .kind = varre::engine::PassPhaseKind::kTransfer,
      .queue = varre::engine::PassQueueKind::kTransfer,
      .explicit_dependencies = {},
      .buffer_accesses =
        {
          varre::engine::PassBufferAccess{
            .resource_id = 7U,
            .buffer = *buffer.buffer,
            .offset = 0U,
            .size = 128U,
            .stage_mask = vk::PipelineStageFlagBits2::eTransfer,
            .access_mask = vk::AccessFlagBits2::eTransferWrite,
            .writes = true,
          },
        },
      .image_accesses = {},
      .graphics_rendering = std::nullopt,
    },
    [](varre::engine::PassCommandEncoder & /*encoder*/) {}));

  static_cast<void>(graph.add_phase(
    varre::engine::PassPhaseDesc{
      .name = "compute",
      .kind = varre::engine::PassPhaseKind::kCompute,
      .queue = varre::engine::PassQueueKind::kAsyncCompute,
      .explicit_dependencies = {},
      .buffer_accesses =
        {
          varre::engine::PassBufferAccess{
            .resource_id = 7U,
            .buffer = *buffer.buffer,
            .offset = 0U,
            .size = 128U,
            .stage_mask = vk::PipelineStageFlagBits2::eComputeShader,
            .access_mask = vk::AccessFlagBits2::eShaderRead,
            .writes = false,
          },
        },
      .image_accesses = {},
      .graphics_rendering = std::nullopt,
    },
    [](varre::engine::PassCommandEncoder & /*encoder*/) {}));

  REQUIRE_NOTHROW(executor.execute(graph));
}

TEST_CASE("shader and model services integrate with generated asset lookup", "[engine][assets][integration]") {
  const EngineAttempt attempt = try_create_engine();
  if (!attempt.engine.has_value()) {
    SKIP(attempt.reason);
  }
  const varre::engine::EngineContext &engine = *attempt.engine;

  std::size_t shader_count = 0U;
  const varre::assets::ShaderId *shader_ids = varre::assets::all_shader_ids(&shader_count);
  REQUIRE(shader_ids != nullptr);
  REQUIRE(shader_count > 0U);

  varre::engine::ShaderObjectCache shader_cache = varre::engine::ShaderObjectCache::create(engine);
  const varre::engine::ShaderObjectSet shader_set = shader_cache.get_or_create(varre::engine::ShaderObjectCreateRequestById{
    .shader_ids = std::span<const varre::assets::ShaderId>{shader_ids, 1U},
    .push_constant_ranges = {},
    .shader_create_flags = {},
  });
  REQUIRE(shader_set.bindings.size() == 1U);
  REQUIRE(shader_set.bindings.front().shader != VK_NULL_HANDLE);
  REQUIRE(shader_set.bindings.front().entry_point != nullptr);

  const std::vector<varre::engine::PassShaderBinding> pass_bindings = varre::engine::make_pass_shader_bindings(shader_set);
  REQUIRE(pass_bindings.size() == shader_set.bindings.size());
  REQUIRE(pass_bindings.front().shader != VK_NULL_HANDLE);

  std::size_t model_count = 0U;
  const varre::assets::ModelId *model_ids = varre::assets::all_model_ids(&model_count);
  REQUIRE(model_ids != nullptr);
  REQUIRE(model_count > 0U);

  varre::engine::ModelUploadService model_upload = varre::engine::ModelUploadService::create(engine);
  std::vector<varre::engine::UploadDependencyToken> upload_dependencies;
  const varre::engine::GpuMesh &mesh = model_upload.get_or_upload_with_dependencies(model_ids[0], &upload_dependencies);
  REQUIRE(mesh.vertex_count > 0U);
  REQUIRE_FALSE(upload_dependencies.empty());

  varre::engine::PassExecutionInfo execution_info{};
  model_upload.append_upload_dependency_waits(&execution_info, 0U, model_ids[0]);
  REQUIRE(execution_info.waits.size() == upload_dependencies.size());
  for (const varre::engine::PassExternalWait &wait : execution_info.waits) {
    REQUIRE(wait.semaphore != VK_NULL_HANDLE);
  }
}

TEST_CASE("smoke init and swapchain recreate path through frame loop", "[engine][swapchain][smoke]") {
  varre::engine::EngineInitInfo init_info{};
  init_info.required_instance_extensions.push_back(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
  init_info.required_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  const EngineAttempt attempt = try_create_engine(init_info);
  if (!attempt.engine.has_value()) {
    SKIP(attempt.reason);
  }
  const varre::engine::EngineContext &engine = *attempt.engine;

  std::string surface_reason;
  std::optional<varre::engine::SurfaceContext> surface = try_create_headless_surface(engine, &surface_reason);
  if (!surface.has_value()) {
    SKIP(surface_reason);
  }

  varre::engine::SwapchainCreateInfo swapchain_info{};
  swapchain_info.preferred_extent = vk::Extent2D{64U, 64U};
  swapchain_info.max_frames_in_flight = 2U;

  std::optional<varre::engine::SwapchainContext> swapchain;
  try {
    swapchain = varre::engine::SwapchainContext::create(engine, *surface, swapchain_info);
  } catch (const varre::engine::EngineError &error) {
    SKIP(error.what());
  }
  REQUIRE(swapchain.has_value());
  REQUIRE(swapchain->image_count() > 0U);

  varre::engine::FrameLoop frame_loop = varre::engine::FrameLoop::create(engine, *swapchain);
  bool recreated = false;
  REQUIRE_NOTHROW(recreated = frame_loop.try_recreate_swapchain(&(*swapchain)));
  REQUIRE((recreated || frame_loop.swapchain_recreation_required()));
  REQUIRE(swapchain->image_count() > 0U);
}
