/**
 * @file errors.hpp
 * @brief Structured engine error categories and exception type.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <vulkan/vulkan_raii.hpp>

namespace varre::engine {

/**
 * @brief Canonical engine error codes used across runtime components.
 */
enum class EngineErrorCode : std::uint8_t {
  /** @brief Unknown or uncategorized failure. */
  kUnknown = 0U,
  /** @brief Required runtime capability or dependency is missing. */
  kMissingRequirement,
  /** @brief No suitable physical device was found for requested constraints. */
  kNoSuitableDevice,
  /** @brief Surface or presentation capabilities do not satisfy requirements. */
  kSurfaceUnsupported,
  /** @brief API misuse or invalid object state for the attempted operation. */
  kInvalidState,
  /** @brief Invalid caller input (e.g. null pointer, out-of-range index). */
  kInvalidArgument,
  /** @brief Operation timed out while waiting for GPU synchronization. */
  kTimeout,
  /** @brief Swapchain is out of date and must be recreated. */
  kSwapchainOutOfDate,
  /** @brief Swapchain is suboptimal and should be recreated soon. */
  kSwapchainSuboptimal,
  /** @brief Unexpected or unsupported Vulkan result. */
  kVulkanResult,
};

/**
 * @brief Convert an engine error code to a stable display token.
 * @param code Error code.
 * @return String token for logging/diagnostics.
 */
[[nodiscard]] constexpr std::string_view engine_error_code_name(const EngineErrorCode code) noexcept {
  switch (code) {
  case EngineErrorCode::kMissingRequirement:
    return "missing_requirement";
  case EngineErrorCode::kNoSuitableDevice:
    return "no_suitable_device";
  case EngineErrorCode::kSurfaceUnsupported:
    return "surface_unsupported";
  case EngineErrorCode::kInvalidState:
    return "invalid_state";
  case EngineErrorCode::kInvalidArgument:
    return "invalid_argument";
  case EngineErrorCode::kTimeout:
    return "timeout";
  case EngineErrorCode::kSwapchainOutOfDate:
    return "swapchain_out_of_date";
  case EngineErrorCode::kSwapchainSuboptimal:
    return "swapchain_suboptimal";
  case EngineErrorCode::kVulkanResult:
    return "vulkan_result";
  case EngineErrorCode::kUnknown:
  default:
    return "unknown";
  }
}

/**
 * @brief Whether an error code represents a recoverable runtime condition.
 * @param code Error code.
 * @return `true` when caller can typically recover and continue running.
 */
[[nodiscard]] constexpr bool is_recoverable_engine_error(const EngineErrorCode code) noexcept {
  switch (code) {
  case EngineErrorCode::kSwapchainOutOfDate:
  case EngineErrorCode::kSwapchainSuboptimal:
    return true;
  case EngineErrorCode::kUnknown:
  case EngineErrorCode::kMissingRequirement:
  case EngineErrorCode::kNoSuitableDevice:
  case EngineErrorCode::kSurfaceUnsupported:
  case EngineErrorCode::kInvalidState:
  case EngineErrorCode::kInvalidArgument:
  case EngineErrorCode::kTimeout:
  case EngineErrorCode::kVulkanResult:
  default:
    return false;
  }
}

/**
 * @brief Engine exception carrying a structured error code.
 */
class EngineError final : public std::runtime_error {
public:
  /**
   * @brief Construct an engine error.
   * @param code Structured error code.
   * @param message Human-readable error message.
   * @param vk_result Optional Vulkan result when error originates from a Vulkan call.
   */
  EngineError(EngineErrorCode code, std::string message, std::optional<vk::Result> vk_result = std::nullopt)
      : std::runtime_error(std::move(message)), code_(code), vk_result_(vk_result) {}

  /**
   * @brief Access structured error code.
   * @return Error code.
   */
  [[nodiscard]] EngineErrorCode code() const noexcept { return code_; }

  /**
   * @brief Whether this error is recoverable by normal runtime control-flow.
   * @return `true` when error is considered recoverable.
   */
  [[nodiscard]] bool recoverable() const noexcept { return is_recoverable_engine_error(code_); }

  /**
   * @brief Access optional Vulkan result payload.
   * @return Vulkan result when provided.
   */
  [[nodiscard]] const std::optional<vk::Result> &vk_result() const noexcept { return vk_result_; }

private:
  EngineErrorCode code_ = EngineErrorCode::kUnknown;
  std::optional<vk::Result> vk_result_;
};

/**
 * @brief Build an engine error from an explicit code and message.
 * @param code Structured error code.
 * @param message Human-readable diagnostic message.
 * @return Engine error instance.
 */
[[nodiscard]] inline EngineError make_engine_error(const EngineErrorCode code, const std::string_view message) {
  return EngineError{code, std::string{message}};
}

/**
 * @brief Build a Vulkan-result error with code `kVulkanResult`.
 * @param result Vulkan result value.
 * @param context Human-readable operation context.
 * @return Engine error instance with Vulkan result payload.
 */
[[nodiscard]] inline EngineError make_vulkan_result_error(const vk::Result result, const std::string_view context) {
  std::string message{context};
  message.append(" (VkResult=");
  message.append(std::to_string(static_cast<std::int32_t>(result)));
  message.push_back(')');
  return EngineError{EngineErrorCode::kVulkanResult, std::move(message), result};
}

} // namespace varre::engine
