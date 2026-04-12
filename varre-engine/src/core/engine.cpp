/**
 * @file engine.cpp
 * @brief Vulkan core initialization implementation using `vk::raii`.
 */
#include "varre/engine/core/engine.hpp"

#include <utility>
#include <vector>

#include "engine_init_helpers.hpp"

namespace varre::engine {

EngineContext::EngineContext(vk::raii::Context &&context, vk::raii::Instance &&instance, vk::raii::DebugUtilsMessengerEXT &&debug_messenger,
                             vk::raii::PhysicalDevice &&physical_device, vk::raii::Device &&device, const QueueFamilyIndices queue_family_indices,
                             const vk::Queue graphics_queue, const std::optional<vk::Queue> async_compute_queue, const std::optional<vk::Queue> transfer_queue,
                             const bool validation_enabled, DeviceQueueTopology device_queue_topology, DeviceProfile device_profile)
    : context_(std::move(context)), instance_(std::move(instance)), debug_messenger_(std::move(debug_messenger)), physical_device_(std::move(physical_device)),
      device_(std::move(device)), queue_family_indices_(queue_family_indices), graphics_queue_(graphics_queue), async_compute_queue_(async_compute_queue),
      transfer_queue_(transfer_queue), validation_enabled_(validation_enabled), device_queue_topology_(std::move(device_queue_topology)),
      device_profile_(std::move(device_profile)) {}

EngineContext EngineContext::create(const EngineInitInfo &info) {
  if (info.api_version < VK_API_VERSION_1_3) {
    throw make_engine_error(EngineErrorCode::kMissingRequirement, "EngineContext requires Vulkan API version 1.3 or newer.");
  }

  vk::raii::Context context;

  std::vector<std::string> required_layers = info.required_instance_layers;
  std::vector<std::string> required_instance_extensions = info.required_instance_extensions;
  if (info.enable_validation) {
    detail::append_unique(&required_layers, detail::kValidationLayerName);
    detail::append_unique(&required_instance_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  detail::validate_required_names(detail::enumerate_instance_layer_names(context), required_layers, "instance layer");
  detail::validate_required_names(detail::enumerate_instance_extension_names(context), required_instance_extensions, "instance extension");

  const std::vector<const char *> instance_layer_ptrs = detail::to_c_string_ptrs(required_layers);
  const std::vector<const char *> instance_extension_ptrs = detail::to_c_string_ptrs(required_instance_extensions);

  const vk::ApplicationInfo application_info = vk::ApplicationInfo{}
                                                 .setPApplicationName(info.application_name.c_str())
                                                 .setApplicationVersion(info.application_version)
                                                 .setPEngineName("varre-engine")
                                                 .setEngineVersion(info.engine_version)
                                                 .setApiVersion(info.api_version);

  const vk::InstanceCreateInfo instance_create_info = vk::InstanceCreateInfo{}
                                                        .setPApplicationInfo(&application_info)
                                                        .setEnabledLayerCount(static_cast<std::uint32_t>(instance_layer_ptrs.size()))
                                                        .setPpEnabledLayerNames(instance_layer_ptrs.data())
                                                        .setEnabledExtensionCount(static_cast<std::uint32_t>(instance_extension_ptrs.size()))
                                                        .setPpEnabledExtensionNames(instance_extension_ptrs.data());

  vk::raii::Instance instance(context, instance_create_info);

  vk::raii::DebugUtilsMessengerEXT debug_messenger{nullptr};
  if (info.enable_validation) {
    const vk::DebugUtilsMessengerCreateInfoEXT debug_create_info =
      vk::DebugUtilsMessengerCreateInfoEXT{}
        .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
        .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        .setPfnUserCallback(vk::PFN_DebugUtilsMessengerCallbackEXT(&detail::vulkan_debug_callback));
    debug_messenger = vk::raii::DebugUtilsMessengerEXT(instance, debug_create_info);
  }

  vk::raii::PhysicalDevices physical_devices(instance);
  if (physical_devices.empty()) {
    throw make_engine_error(EngineErrorCode::kNoSuitableDevice, "No Vulkan physical devices are available.");
  }

  const DeviceProfileRequest device_profile_request = detail::build_device_profile_request(info);
  const detail::DeviceSelection selected_device = detail::select_physical_device(physical_devices, info, device_profile_request);
  vk::raii::PhysicalDevice physical_device = std::move(physical_devices[selected_device.index]);

  const std::vector<const char *> enabled_device_extension_ptrs = detail::to_c_string_ptrs(selected_device.enabled_extensions);
  const std::vector<vk::DeviceQueueCreateInfo> queue_create_infos = detail::build_queue_create_infos(selected_device.queues.indices);

  vk::DeviceCreateInfo device_create_info = vk::DeviceCreateInfo{}
                                              .setQueueCreateInfos(queue_create_infos)
                                              .setEnabledExtensionCount(static_cast<std::uint32_t>(enabled_device_extension_ptrs.size()))
                                              .setPpEnabledExtensionNames(enabled_device_extension_ptrs.data());

  std::optional<vk::PhysicalDeviceFeatures2> device_features2;
  std::optional<vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR> unified_image_layouts_features;
  std::optional<vk::PhysicalDeviceVulkan12Features> vulkan_12_features;
  std::optional<vk::PhysicalDeviceVulkan13Features> vulkan_13_features;
  std::optional<vk::PhysicalDeviceShaderObjectFeaturesEXT> shader_object_features;
  unified_image_layouts_features = vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR{}.setUnifiedImageLayouts(VK_TRUE);
  shader_object_features = vk::PhysicalDeviceShaderObjectFeaturesEXT{}.setPNext(&(*unified_image_layouts_features)).setShaderObject(VK_TRUE);
  vulkan_13_features =
    vk::PhysicalDeviceVulkan13Features{}.setPNext(&(*shader_object_features)).setDynamicRendering(VK_TRUE).setSynchronization2(VK_TRUE);
  vulkan_12_features = vk::PhysicalDeviceVulkan12Features{}.setPNext(&(*vulkan_13_features)).setTimelineSemaphore(VK_TRUE);
  device_features2 = vk::PhysicalDeviceFeatures2{}.setPNext(&(*vulkan_12_features));
  device_create_info = device_create_info.setPNext(&(*device_features2));

  vk::raii::Device device(physical_device, device_create_info);

  const vk::Queue graphics_queue = device.getQueue(selected_device.queues.indices.graphics, 0U);
  std::optional<vk::Queue> async_compute_queue;
  if (selected_device.queues.indices.async_compute.has_value()) {
    async_compute_queue = device.getQueue(*selected_device.queues.indices.async_compute, 0U);
  }

  std::optional<vk::Queue> transfer_queue;
  if (selected_device.queues.indices.transfer.has_value()) {
    transfer_queue = device.getQueue(*selected_device.queues.indices.transfer, 0U);
  }

  DeviceQueueTopology device_queue_topology{
    .families = selected_device.queues.indices,
    .graphics_queue = graphics_queue,
    .async_compute_queue = async_compute_queue,
    .transfer_queue = transfer_queue,
    .has_dedicated_async_compute = selected_device.queues.has_dedicated_async_compute,
    .has_dedicated_transfer = selected_device.queues.has_dedicated_transfer,
  };

  const vk::PhysicalDeviceProperties selected_properties = physical_device.getProperties();
  DeviceProfile resolved_device_profile{
    .device_name = selected_properties.deviceName,
    .device_type = selected_properties.deviceType,
    .vendor_id = selected_properties.vendorID,
    .device_id = selected_properties.deviceID,
    .api_version = selected_properties.apiVersion,
    .enabled_extensions = selected_device.enabled_extensions,
    .missing_optional_extensions = selected_device.missing_optional_extensions,
  };

  return EngineContext{
    std::move(context),
    std::move(instance),
    std::move(debug_messenger),
    std::move(physical_device),
    std::move(device),
    selected_device.queues.indices,
    graphics_queue,
    async_compute_queue,
    transfer_queue,
    info.enable_validation,
    std::move(device_queue_topology),
    std::move(resolved_device_profile),
  };
}

const vk::raii::Context &EngineContext::context() const noexcept { return context_; }

const vk::raii::Instance &EngineContext::instance() const noexcept { return instance_; }

vk::PhysicalDevice EngineContext::physical_device() const noexcept { return *physical_device_; }

const vk::raii::PhysicalDevice &EngineContext::physical_device_raii() const noexcept { return physical_device_; }

const vk::raii::Device &EngineContext::device() const noexcept { return device_; }

const QueueFamilyIndices &EngineContext::queue_family_indices() const noexcept { return queue_family_indices_; }

const DeviceQueueTopology &EngineContext::device_queue_topology() const noexcept { return device_queue_topology_; }

const DeviceQueueTopology &EngineContext::queue_topology() const noexcept { return device_queue_topology(); }

vk::Queue EngineContext::graphics_queue() const noexcept { return graphics_queue_; }

std::optional<vk::Queue> EngineContext::async_compute_queue() const noexcept { return async_compute_queue_; }

std::optional<vk::Queue> EngineContext::transfer_queue() const noexcept { return transfer_queue_; }

bool EngineContext::validation_enabled() const noexcept { return validation_enabled_; }

const DeviceProfile &EngineContext::device_profile() const noexcept { return device_profile_; }

} // namespace varre::engine
