/**
 * @file engine.hpp
 * @brief Core Vulkan engine initialization interfaces.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "varre/engine/core/errors.hpp"

namespace varre::engine {

/**
 * @brief Queue-family indices chosen for the logical device.
 */
struct QueueFamilyIndices {
  /** @brief Queue family index used for graphics work (always present). */
  std::uint32_t graphics = 0;
  /** @brief Optional queue family index used for asynchronous compute work. */
  std::optional<std::uint32_t> async_compute;
  /** @brief Optional queue family index used for transfer work. */
  std::optional<std::uint32_t> transfer;
};

/**
 * @brief Resolved queue topology for the created logical device.
 */
struct DeviceQueueTopology {
  /** @brief Queue-family indices selected during physical-device selection. */
  QueueFamilyIndices families;
  /** @brief Graphics queue handle. */
  vk::Queue graphics_queue = VK_NULL_HANDLE;
  /** @brief Optional asynchronous compute queue handle. */
  std::optional<vk::Queue> async_compute_queue;
  /** @brief Optional transfer queue handle. */
  std::optional<vk::Queue> transfer_queue;
  /** @brief True when async compute queue is non-graphics dedicated. */
  bool has_dedicated_async_compute = false;
  /** @brief True when transfer queue excludes graphics+compute flags. */
  bool has_dedicated_transfer = false;
};

/**
 * @brief Backward-compatible alias for @ref DeviceQueueTopology.
 */
using QueueTopology = DeviceQueueTopology;

/**
 * @brief Requested device-profile capabilities before physical-device selection.
 */
struct DeviceProfileRequest {
  /** @brief Required Vulkan device extension names. */
  std::vector<std::string> required_extensions;
  /** @brief Optional Vulkan device extension names enabled when available. */
  std::vector<std::string> optional_extensions;
};

/**
 * @brief Resolved device-profile capabilities after physical-device selection.
 */
struct DeviceProfile {
  /** @brief Vulkan device name from selected physical-device properties. */
  std::string device_name;
  /** @brief Selected physical-device type. */
  vk::PhysicalDeviceType device_type = vk::PhysicalDeviceType::eOther;
  /** @brief Selected physical-device vendor identifier. */
  std::uint32_t vendor_id = 0U;
  /** @brief Selected physical-device device identifier. */
  std::uint32_t device_id = 0U;
  /** @brief Vulkan API version supported by the selected physical device. */
  std::uint32_t api_version = VK_API_VERSION_1_0;
  /** @brief Device extensions enabled on logical-device creation. */
  std::vector<std::string> enabled_extensions;
  /** @brief Optional extensions requested but unavailable on the selected device. */
  std::vector<std::string> missing_optional_extensions;
};

/**
 * @brief Configuration for Vulkan core initialization.
 */
struct EngineInitInfo {
  /** @brief Application name passed to Vulkan instance creation. */
  std::string application_name = "varre-engine";
  /** @brief Application version passed to Vulkan instance creation. */
  std::uint32_t application_version = VK_MAKE_API_VERSION(0, 0, 1, 0);
  /** @brief Engine version passed to Vulkan instance creation. */
  std::uint32_t engine_version = VK_MAKE_API_VERSION(0, 0, 1, 0);
  /** @brief Vulkan API version requested for instance creation (must be >= 1.3). */
  std::uint32_t api_version = VK_API_VERSION_1_3;
  /** @brief Enable Vulkan validation layers and debug messenger. */
  bool enable_validation = false;
  /** @brief Prefer discrete GPUs during physical-device selection. */
  bool prefer_discrete_gpu = true;
  /** @brief Require a non-graphics compute queue family. */
  bool require_async_compute = false;
  /** @brief Require a dedicated transfer queue family. */
  bool require_dedicated_transfer = false;
  /** @brief Required Vulkan instance extension names. */
  std::vector<std::string> required_instance_extensions;
  /** @brief Required Vulkan instance layer names. */
  std::vector<std::string> required_instance_layers;
  /**
   * @brief Required Vulkan device extension names.
   *
   * This legacy field is merged into `device_profile.required_extensions`.
   */
  std::vector<std::string> required_device_extensions;
  /** @brief Device profile request used during physical-device negotiation. */
  DeviceProfileRequest device_profile;
};

/**
 * @brief Fully initialized Vulkan core context using `vk::raii` ownership.
 */
class EngineContext {
public:
  /**
   * @brief Create a fully initialized Vulkan core context.
   * @param info Initialization configuration.
   * @return Initialized engine context.
   * @throws EngineError on initialization failure.
   */
  [[nodiscard]] static EngineContext create(const EngineInitInfo &info = {});

  /**
   * @brief Move-construct the context.
   * @param other Context being moved from.
   */
  EngineContext(EngineContext &&other) noexcept = default;

  /**
   * @brief Move-assign the context.
   * @param other Context being moved from.
   * @return `*this`.
   */
  EngineContext &operator=(EngineContext &&other) noexcept = default;

  EngineContext(const EngineContext &) = delete;
  EngineContext &operator=(const EngineContext &) = delete;

  /**
   * @brief Access the Vulkan context object.
   * @return Immutable Vulkan RAII context.
   */
  [[nodiscard]] const vk::raii::Context &context() const noexcept;

  /**
   * @brief Access the Vulkan instance.
   * @return Immutable Vulkan RAII instance.
   */
  [[nodiscard]] const vk::raii::Instance &instance() const noexcept;

  /**
   * @brief Access the selected physical-device handle.
   * @return Vulkan physical-device handle.
   */
  [[nodiscard]] vk::PhysicalDevice physical_device() const noexcept;

  /**
   * @brief Access the selected physical-device RAII wrapper.
   * @return Immutable Vulkan RAII physical device.
   */
  [[nodiscard]] const vk::raii::PhysicalDevice &physical_device_raii() const noexcept;

  /**
   * @brief Access the logical device.
   * @return Immutable Vulkan RAII device.
   */
  [[nodiscard]] const vk::raii::Device &device() const noexcept;

  /**
   * @brief Access selected queue-family indices.
   * @return Immutable queue-family index set.
   */
  [[nodiscard]] const QueueFamilyIndices &queue_family_indices() const noexcept;

  /**
   * @brief Access resolved queue topology for the logical device.
   * @return Immutable queue topology.
   */
  [[nodiscard]] const DeviceQueueTopology &device_queue_topology() const noexcept;

  /**
   * @brief Access resolved queue topology for the logical device.
   * @return Immutable queue topology.
   *
   * Compatibility alias for @ref device_queue_topology.
   */
  [[nodiscard]] const DeviceQueueTopology &queue_topology() const noexcept;

  /**
   * @brief Access graphics queue handle.
   * @return Graphics queue handle.
   */
  [[nodiscard]] vk::Queue graphics_queue() const noexcept;

  /**
   * @brief Access asynchronous compute queue handle when available.
   * @return Optional compute queue handle.
   */
  [[nodiscard]] std::optional<vk::Queue> async_compute_queue() const noexcept;

  /**
   * @brief Access transfer queue handle when available.
   * @return Optional transfer queue handle.
   */
  [[nodiscard]] std::optional<vk::Queue> transfer_queue() const noexcept;

  /**
   * @brief Whether validation was enabled during initialization.
   * @return `true` when validation was enabled.
   */
  [[nodiscard]] bool validation_enabled() const noexcept;

  /**
   * @brief Access the resolved device profile used for logical-device creation.
   * @return Immutable negotiated device profile.
   */
  [[nodiscard]] const DeviceProfile &device_profile() const noexcept;

private:
  /**
   * @brief Internal constructor from already-initialized RAII objects.
   */
  EngineContext(vk::raii::Context &&context, vk::raii::Instance &&instance, vk::raii::DebugUtilsMessengerEXT &&debug_messenger,
                vk::raii::PhysicalDevice &&physical_device, vk::raii::Device &&device, QueueFamilyIndices queue_family_indices, vk::Queue graphics_queue,
                std::optional<vk::Queue> async_compute_queue, std::optional<vk::Queue> transfer_queue, bool validation_enabled,
                DeviceQueueTopology device_queue_topology, DeviceProfile device_profile);

  vk::raii::Context context_;
  vk::raii::Instance instance_{nullptr};
  vk::raii::DebugUtilsMessengerEXT debug_messenger_{nullptr};
  vk::raii::PhysicalDevice physical_device_{nullptr};
  vk::raii::Device device_{nullptr};
  QueueFamilyIndices queue_family_indices_{};
  vk::Queue graphics_queue_{VK_NULL_HANDLE};
  std::optional<vk::Queue> async_compute_queue_;
  std::optional<vk::Queue> transfer_queue_;
  bool validation_enabled_ = false;
  DeviceQueueTopology device_queue_topology_{};
  DeviceProfile device_profile_{};
};

} // namespace varre::engine

namespace varre::engine::core {
using ::varre::engine::DeviceProfile;
using ::varre::engine::DeviceProfileRequest;
using ::varre::engine::DeviceQueueTopology;
using ::varre::engine::EngineContext;
using ::varre::engine::EngineInitInfo;
using ::varre::engine::QueueFamilyIndices;
using ::varre::engine::QueueTopology;
} // namespace varre::engine::core
