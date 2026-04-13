/**
 * @file engine_init_helpers.cpp
 * @brief Internal helper implementation for engine initialization.
 */
#include "engine_init_helpers.hpp"

#include <algorithm>
#include <exception>
#include <ranges>
#include <set>
#include <unordered_set>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

namespace varre::engine::detail {

namespace {
/**
 * @brief Validate required pass-mode features on one physical device.
 * @param physical_device Physical device candidate.
 * @param missing_features Human-readable missing feature names.
 * @return `true` when all required features are supported.
 */
[[nodiscard]] bool supports_pass_mode_features(const vk::raii::PhysicalDevice &physical_device, std::vector<std::string> *missing_features) {
  missing_features->clear();

  const auto feature_chain =
    physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features,
                                 vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceShaderObjectFeaturesEXT,
                                 vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR>();
  const vk::PhysicalDeviceVulkan11Features &vulkan_11_features = feature_chain.get<vk::PhysicalDeviceVulkan11Features>();
  const vk::PhysicalDeviceVulkan12Features &vulkan_12_features = feature_chain.get<vk::PhysicalDeviceVulkan12Features>();
  const vk::PhysicalDeviceVulkan13Features &vulkan_13_features = feature_chain.get<vk::PhysicalDeviceVulkan13Features>();
  const vk::PhysicalDeviceShaderObjectFeaturesEXT &shader_object_features = feature_chain.get<vk::PhysicalDeviceShaderObjectFeaturesEXT>();
  const vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR &unified_image_layouts_features =
    feature_chain.get<vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR>();

  if (!vulkan_11_features.shaderDrawParameters) {
    missing_features->push_back("shaderDrawParameters");
  }
  if (!vulkan_12_features.timelineSemaphore) {
    missing_features->push_back("timelineSemaphore");
  }
  if (!unified_image_layouts_features.unifiedImageLayouts) {
    missing_features->push_back("unifiedImageLayouts");
  }
  if (!vulkan_13_features.dynamicRendering) {
    missing_features->push_back("dynamicRendering");
  }
  if (!vulkan_13_features.synchronization2) {
    missing_features->push_back("synchronization2");
  }
  if (!shader_object_features.shaderObject) {
    missing_features->push_back("shaderObject");
  }
  return missing_features->empty();
}
} // namespace

void append_unique(std::vector<std::string> *names, const std::string_view name) {
  const bool exists = std::ranges::any_of(*names, [&](const std::string &existing) { return existing == name; });
  if (!exists) {
    names->emplace_back(name);
  }
}

std::vector<const char *> to_c_string_ptrs(const std::vector<std::string> &names) {
  std::vector<const char *> out;
  out.reserve(names.size());
  for (const std::string &name : names) {
    out.push_back(name.c_str());
  }
  return out;
}

std::vector<std::string> enumerate_instance_extension_names(const vk::raii::Context &context) {
  const std::vector<vk::ExtensionProperties> properties = context.enumerateInstanceExtensionProperties();
  std::vector<std::string> names;
  names.reserve(properties.size());
  for (const vk::ExtensionProperties &property : properties) {
    names.emplace_back(property.extensionName);
  }
  return names;
}

std::vector<std::string> enumerate_instance_layer_names(const vk::raii::Context &context) {
  const std::vector<vk::LayerProperties> properties = context.enumerateInstanceLayerProperties();
  std::vector<std::string> names;
  names.reserve(properties.size());
  for (const vk::LayerProperties &property : properties) {
    names.emplace_back(property.layerName);
  }
  return names;
}

std::vector<std::string> enumerate_device_extension_names(const vk::raii::PhysicalDevice &physical_device) {
  const std::vector<vk::ExtensionProperties> properties = physical_device.enumerateDeviceExtensionProperties();
  std::vector<std::string> names;
  names.reserve(properties.size());
  for (const vk::ExtensionProperties &property : properties) {
    names.emplace_back(property.extensionName);
  }
  return names;
}

void validate_required_names(const std::vector<std::string> &available, const std::vector<std::string> &required, const std::string_view category) {
  std::unordered_set<std::string> available_set(available.begin(), available.end());
  std::vector<std::string> missing;
  missing.reserve(required.size());
  for (const std::string &name : required) {
    if (!available_set.contains(name)) {
      missing.push_back(name);
    }
  }

  if (!missing.empty()) {
    throw make_engine_error(EngineErrorCode::kMissingRequirement, fmt::format("Missing required {}(s): {}", category, fmt::join(missing, ", ")));
  }
}

DeviceProfileRequest build_device_profile_request(const EngineInitInfo &info) {
  DeviceProfileRequest request = info.device_profile;
  for (const std::string &extension : info.required_device_extensions) {
    append_unique(&request.required_extensions, extension);
  }
  append_unique(&request.required_extensions, VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
  append_unique(&request.required_extensions, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
  return request;
}

std::pair<std::vector<std::string>, std::vector<std::string>> resolve_device_extensions(const std::vector<std::string> &available,
                                                                                        const DeviceProfileRequest &request) {
  validate_required_names(available, request.required_extensions, "device extension");

  std::unordered_set<std::string> available_set(available.begin(), available.end());
  std::vector<std::string> enabled_extensions;
  enabled_extensions.reserve(request.required_extensions.size() + request.optional_extensions.size());
  for (const std::string &name : request.required_extensions) {
    append_unique(&enabled_extensions, name);
  }

  std::vector<std::string> missing_optional_extensions;
  missing_optional_extensions.reserve(request.optional_extensions.size());
  for (const std::string &name : request.optional_extensions) {
    if (available_set.contains(name)) {
      append_unique(&enabled_extensions, name);
    } else {
      append_unique(&missing_optional_extensions, name);
    }
  }

  return {enabled_extensions, missing_optional_extensions};
}

std::optional<QueueSelection> select_queue_families(const vk::raii::PhysicalDevice &physical_device) {
  const std::vector<vk::QueueFamilyProperties> queue_properties = physical_device.getQueueFamilyProperties();

  std::optional<std::uint32_t> graphics_family;
  std::optional<std::uint32_t> dedicated_compute_family;
  std::optional<std::uint32_t> fallback_compute_family;
  std::optional<std::uint32_t> dedicated_transfer_family;
  std::optional<std::uint32_t> fallback_transfer_family;

  for (std::uint32_t family_index = 0; family_index < queue_properties.size(); ++family_index) {
    const vk::QueueFlags flags = queue_properties[family_index].queueFlags;

    if (!graphics_family.has_value() && static_cast<bool>(flags & vk::QueueFlagBits::eGraphics)) {
      graphics_family = family_index;
    }

    if (static_cast<bool>(flags & vk::QueueFlagBits::eCompute)) {
      if (!static_cast<bool>(flags & vk::QueueFlagBits::eGraphics)) {
        if (!dedicated_compute_family.has_value()) {
          dedicated_compute_family = family_index;
        }
      } else if (!fallback_compute_family.has_value()) {
        fallback_compute_family = family_index;
      }
    }

    if (static_cast<bool>(flags & vk::QueueFlagBits::eTransfer)) {
      const bool has_graphics = static_cast<bool>(flags & vk::QueueFlagBits::eGraphics);
      const bool has_compute = static_cast<bool>(flags & vk::QueueFlagBits::eCompute);
      if (!has_graphics && !has_compute) {
        if (!dedicated_transfer_family.has_value()) {
          dedicated_transfer_family = family_index;
        }
      } else if (!fallback_transfer_family.has_value()) {
        fallback_transfer_family = family_index;
      }
    }
  }

  if (!graphics_family.has_value()) {
    return std::nullopt;
  }

  std::optional<std::uint32_t> async_compute_family = dedicated_compute_family;
  if (!async_compute_family.has_value()) {
    async_compute_family = fallback_compute_family;
  }
  if (!async_compute_family.has_value()) {
    async_compute_family = graphics_family;
  }

  std::optional<std::uint32_t> transfer_family = dedicated_transfer_family;
  if (!transfer_family.has_value()) {
    transfer_family = fallback_transfer_family;
  }
  if (!transfer_family.has_value()) {
    transfer_family = async_compute_family;
  }
  if (!transfer_family.has_value()) {
    transfer_family = graphics_family;
  }

  return QueueSelection{
    .indices =
      QueueFamilyIndices{
        .graphics = *graphics_family,
        .async_compute = async_compute_family,
        .transfer = transfer_family,
      },
    .has_dedicated_async_compute = dedicated_compute_family.has_value(),
    .has_dedicated_transfer = dedicated_transfer_family.has_value(),
  };
}

DeviceSelection select_physical_device(const vk::raii::PhysicalDevices &physical_devices, const EngineInitInfo &info,
                                       const DeviceProfileRequest &profile_request) {
  struct Candidate {
    DeviceSelection selection{};
    int score = 0;
  };

  std::optional<Candidate> best;
  std::vector<std::string> rejection_reasons;

  for (std::size_t index = 0; index < physical_devices.size(); ++index) {
    const vk::raii::PhysicalDevice &physical_device = physical_devices[index];
    const std::optional<QueueSelection> selected_queues = select_queue_families(physical_device);
    if (!selected_queues.has_value()) {
      rejection_reasons.push_back(fmt::format("device[{}]: missing graphics queue", index));
      continue;
    }
    if (info.require_async_compute && !selected_queues->has_dedicated_async_compute) {
      rejection_reasons.push_back(fmt::format("device[{}]: dedicated async compute queue not available", index));
      continue;
    }
    if (info.require_dedicated_transfer && !selected_queues->has_dedicated_transfer) {
      rejection_reasons.push_back(fmt::format("device[{}]: dedicated transfer queue not available", index));
      continue;
    }

    std::vector<std::string> missing_pass_mode_features;
    if (!supports_pass_mode_features(physical_device, &missing_pass_mode_features)) {
      rejection_reasons.push_back(fmt::format("device[{}]: missing required pass-mode feature(s): {}", index, fmt::join(missing_pass_mode_features, ", ")));
      continue;
    }

    const std::vector<std::string> available_device_extensions = enumerate_device_extension_names(physical_device);
    std::vector<std::string> enabled_extensions;
    std::vector<std::string> missing_optional_extensions;
    try {
      std::tie(enabled_extensions, missing_optional_extensions) = resolve_device_extensions(available_device_extensions, profile_request);
    } catch (const std::exception &ex) {
      rejection_reasons.push_back(fmt::format("device[{}]: {}", index, ex.what()));
      continue;
    }

    const vk::PhysicalDeviceProperties properties = physical_device.getProperties();
    int score = 0;
    switch (properties.deviceType) {
    case vk::PhysicalDeviceType::eDiscreteGpu:
      score += info.prefer_discrete_gpu ? 1000 : 100;
      break;
    case vk::PhysicalDeviceType::eIntegratedGpu:
      score += info.prefer_discrete_gpu ? 500 : 1000;
      break;
    case vk::PhysicalDeviceType::eVirtualGpu:
      score += 250;
      break;
    case vk::PhysicalDeviceType::eCpu:
      score += 100;
      break;
    default:
      score += 50;
      break;
    }

    if (selected_queues->has_dedicated_async_compute) {
      score += 50;
    }
    if (selected_queues->has_dedicated_transfer) {
      score += 50;
    }
    const int optional_extension_bonus = static_cast<int>(profile_request.optional_extensions.size()) - static_cast<int>(missing_optional_extensions.size());
    score += optional_extension_bonus * 5;

    const Candidate candidate{
      .selection =
        DeviceSelection{
          .index = index,
          .queues = *selected_queues,
          .enabled_extensions = enabled_extensions,
          .missing_optional_extensions = missing_optional_extensions,
        },
      .score = score,
    };
    if (!best.has_value() || candidate.score > best->score) {
      best = candidate;
    }
  }

  if (!best.has_value()) {
    std::string details = "<no candidate devices>";
    if (!rejection_reasons.empty()) {
      details = fmt::format("{}", fmt::join(rejection_reasons, " | "));
    }
    throw make_engine_error(EngineErrorCode::kNoSuitableDevice, fmt::format("Unable to select a suitable Vulkan physical device: {}", details));
  }
  return std::move(best->selection);
}

std::vector<vk::DeviceQueueCreateInfo> build_queue_create_infos(const QueueFamilyIndices &indices) {
  const float queue_priority = 1.0F;
  std::set<std::uint32_t> unique_families{indices.graphics};
  if (indices.async_compute.has_value()) {
    unique_families.insert(*indices.async_compute);
  }
  if (indices.transfer.has_value()) {
    unique_families.insert(*indices.transfer);
  }

  std::vector<vk::DeviceQueueCreateInfo> create_infos;
  create_infos.reserve(unique_families.size());
  for (const std::uint32_t family_index : unique_families) {
    create_infos.push_back(vk::DeviceQueueCreateInfo{}.setQueueFamilyIndex(family_index).setQueueCount(1U).setPQueuePriorities(&queue_priority));
  }
  return create_infos;
}

VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_callback(const VkDebugUtilsMessageSeverityFlagBitsEXT severity, const VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                     const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void * /*user_data*/
) {
  const char *message = (callback_data != nullptr && callback_data->pMessage != nullptr) ? callback_data->pMessage : "<no message>";
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0U) {
    spdlog::error("Vulkan validation [{:#x}]: {}", static_cast<std::uint32_t>(message_type), message);
  } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0U) {
    spdlog::warn("Vulkan validation [{:#x}]: {}", static_cast<std::uint32_t>(message_type), message);
  } else {
    spdlog::info("Vulkan validation [{:#x}]: {}", static_cast<std::uint32_t>(message_type), message);
  }
  return VK_FALSE;
}

} // namespace varre::engine::detail
