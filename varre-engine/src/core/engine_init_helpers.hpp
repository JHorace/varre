/**
 * @file engine_init_helpers.hpp
 * @brief Internal helpers for engine instance/device initialization.
 */
#pragma once

#include "varre/engine/core/engine.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace varre::engine::detail {

/**
 * @brief Default validation-layer name used when validation is enabled.
 */
inline constexpr std::string_view kValidationLayerName = "VK_LAYER_KHRONOS_validation";

/**
 * @brief Information about selected queue families and queue topology quality.
 */
struct QueueSelection {
  QueueFamilyIndices indices;
  bool has_dedicated_async_compute = false;
  bool has_dedicated_transfer = false;
};

/**
 * @brief Selected physical-device outcome, including negotiated profile.
 */
struct DeviceSelection {
  std::size_t index = 0U;
  QueueSelection queues{};
  std::vector<std::string> enabled_extensions;
  std::vector<std::string> missing_optional_extensions;
};

void append_unique(std::vector<std::string> *names, std::string_view name);
std::vector<const char *> to_c_string_ptrs(const std::vector<std::string> &names);
std::vector<std::string> enumerate_instance_extension_names(const vk::raii::Context &context);
std::vector<std::string> enumerate_instance_layer_names(const vk::raii::Context &context);
std::vector<std::string> enumerate_device_extension_names(const vk::raii::PhysicalDevice &physical_device);
void validate_required_names(const std::vector<std::string> &available, const std::vector<std::string> &required, std::string_view category);
DeviceProfileRequest build_device_profile_request(const EngineInitInfo &info);
std::pair<std::vector<std::string>, std::vector<std::string>> resolve_device_extensions(const std::vector<std::string> &available,
                                                                                        const DeviceProfileRequest &request);
std::optional<QueueSelection> select_queue_families(const vk::raii::PhysicalDevice &physical_device);
DeviceSelection select_physical_device(const vk::raii::PhysicalDevices &physical_devices, const EngineInitInfo &info,
                                       const DeviceProfileRequest &profile_request);
std::vector<vk::DeviceQueueCreateInfo> build_queue_create_infos(const QueueFamilyIndices &indices);

VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                     const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *user_data);

} // namespace varre::engine::detail
