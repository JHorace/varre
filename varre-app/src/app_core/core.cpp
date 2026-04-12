/**
 * @file core.cpp
 * @brief SDL3-backed app bootstrap/runtime implementation.
 */
#include "varre/app/core.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

namespace varre::app {
namespace {

[[nodiscard]] std::runtime_error sdl_error(const char *context) {
  return std::runtime_error(std::string(context) + ": " + SDL_GetError());
}

template <typename TStringContainer>
void append_unique_string(TStringContainer *container, const std::string &value) {
  if (container == nullptr) {
    throw std::invalid_argument("append_unique_string requires a valid container pointer.");
  }
  if (std::ranges::find(*container, value) == container->end()) {
    container->push_back(value);
  }
}

[[nodiscard]] bool is_window_event_type(const std::uint32_t event_type) {
  return event_type >= static_cast<std::uint32_t>(SDL_EVENT_WINDOW_FIRST) && event_type <= static_cast<std::uint32_t>(SDL_EVENT_WINDOW_LAST);
}

} // namespace

AppCore::SdlVideoSubsystemGuard::SdlVideoSubsystemGuard(const bool initialized) noexcept : initialized_(initialized) {}

AppCore::SdlVideoSubsystemGuard AppCore::SdlVideoSubsystemGuard::create() {
  if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
    throw sdl_error("SDL_InitSubSystem(SDL_INIT_VIDEO) failed");
  }
  return SdlVideoSubsystemGuard{true};
}

AppCore::SdlVideoSubsystemGuard::SdlVideoSubsystemGuard(SdlVideoSubsystemGuard &&other) noexcept : initialized_(other.initialized_) {
  other.initialized_ = false;
}

AppCore::SdlVideoSubsystemGuard &AppCore::SdlVideoSubsystemGuard::operator=(SdlVideoSubsystemGuard &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (initialized_) {
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
  }
  initialized_ = other.initialized_;
  other.initialized_ = false;
  return *this;
}

AppCore::SdlVideoSubsystemGuard::~SdlVideoSubsystemGuard() {
  if (initialized_) {
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
  }
}

void AppCore::WindowDeleter::operator()(SDL_Window *window) const noexcept {
  if (window != nullptr) {
    SDL_DestroyWindow(window);
  }
}

void AppCore::update_swapchain_extent_from_window(SDL_Window *window, const bool sync_enabled, engine::SwapchainCreateInfo *swapchain_create_info) {
  if (!sync_enabled || window == nullptr || swapchain_create_info == nullptr) {
    return;
  }

  int width = 0;
  int height = 0;
  if (!SDL_GetWindowSizeInPixels(window, &width, &height)) {
    throw sdl_error("SDL_GetWindowSizeInPixels failed");
  }
  if (width <= 0 || height <= 0) {
    return;
  }

  const auto width_u32 = static_cast<std::uint64_t>(width);
  const auto height_u32 = static_cast<std::uint64_t>(height);
  if (width_u32 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) ||
      height_u32 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error("Window pixel size exceeds uint32_t range required by Vulkan extent.");
  }
  swapchain_create_info->preferred_extent = vk::Extent2D{
    static_cast<std::uint32_t>(width),
    static_cast<std::uint32_t>(height),
  };
}

std::vector<std::string> AppCore::query_required_instance_extensions(SDL_Window *window) {
  if (window == nullptr) {
    throw std::invalid_argument("query_required_instance_extensions requires a valid SDL window.");
  }
  if ((SDL_GetWindowFlags(window) & SDL_WINDOW_VULKAN) == 0U) {
    throw std::invalid_argument("query_required_instance_extensions requires a window created with SDL_WINDOW_VULKAN.");
  }

  std::uint32_t extension_count = 0U;
  const char *const *extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
  if (extensions == nullptr) {
    throw sdl_error("SDL_Vulkan_GetInstanceExtensions failed");
  }

  std::vector<std::string> required_extensions;
  required_extensions.reserve(extension_count);
  for (std::uint32_t i = 0U; i < extension_count; ++i) {
    if (extensions[i] == nullptr) {
      continue;
    }
    append_unique_string(&required_extensions, std::string{extensions[i]});
  }

  if (required_extensions.empty()) {
    throw std::runtime_error("SDL_Vulkan_GetInstanceExtensions returned no required extensions.");
  }

  return required_extensions;
}

void AppCore::append_required_instance_extensions(engine::EngineInitInfo *init_info, SDL_Window *window) {
  if (init_info == nullptr) {
    throw std::invalid_argument("append_required_instance_extensions requires a valid EngineInitInfo pointer.");
  }
  const std::vector<std::string> required_extensions = query_required_instance_extensions(window);
  for (const std::string &extension : required_extensions) {
    append_unique_string(&init_info->required_instance_extensions, extension);
  }
}

engine::SurfaceContext AppCore::create_surface_for_window(const engine::EngineContext &engine, SDL_Window *window) {
  if (window == nullptr) {
    throw std::invalid_argument("create_surface_for_window requires a valid SDL window.");
  }

  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(*engine.instance()), nullptr, &surface)) {
    throw sdl_error("SDL_Vulkan_CreateSurface failed");
  }

  try {
    return engine::SurfaceContext::adopt(engine.instance(), vk::SurfaceKHR{surface});
  } catch (...) {
    SDL_Vulkan_DestroySurface(static_cast<VkInstance>(*engine.instance()), surface, nullptr);
    throw;
  }
}

AppCore AppCore::create(const AppCoreCreateInfo &create_info) {
  SdlVideoSubsystemGuard sdl_video = SdlVideoSubsystemGuard::create();

  SDL_WindowFlags window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN);
  if (create_info.window.resizable) {
    window_flags = static_cast<SDL_WindowFlags>(window_flags | SDL_WINDOW_RESIZABLE);
  }
  if (create_info.window.high_pixel_density) {
    window_flags = static_cast<SDL_WindowFlags>(window_flags | SDL_WINDOW_HIGH_PIXEL_DENSITY);
  }

  WindowPtr window{SDL_CreateWindow(create_info.window.title.c_str(), create_info.window.width, create_info.window.height, window_flags)};
  if (window == nullptr) {
    throw sdl_error("SDL_CreateWindow failed");
  }

  engine::EngineInitInfo engine_init_info = create_info.engine;
  append_required_instance_extensions(&engine_init_info, window.get());
  append_unique_string(&engine_init_info.device_profile.required_extensions, std::string{VK_KHR_SWAPCHAIN_EXTENSION_NAME});

  engine::EngineContext engine = engine::EngineContext::create(engine_init_info);
  engine::SurfaceContext surface = create_surface_for_window(engine, window.get());

  engine::SwapchainCreateInfo swapchain_create_info = create_info.swapchain;
  update_swapchain_extent_from_window(window.get(), create_info.sync_swapchain_extent_with_window, &swapchain_create_info);

  const SDL_WindowID window_id = SDL_GetWindowID(window.get());
  if (window_id == 0U) {
    throw sdl_error("SDL_GetWindowID failed");
  }

  return AppCore{
    std::move(sdl_video),
    std::move(window),
    window_id,
    create_info.sync_swapchain_extent_with_window,
    std::move(engine),
    std::move(surface),
    swapchain_create_info,
    create_info.frame_loop,
    create_info.pass_executor,
  };
}

AppCore::AppCore(SdlVideoSubsystemGuard &&sdl_video, WindowPtr &&window, const SDL_WindowID window_id, const bool sync_swapchain_extent_with_window,
                 engine::EngineContext &&engine, engine::SurfaceContext &&surface, engine::SwapchainCreateInfo swapchain_create_info,
                 const engine::FrameLoopCreateInfo &frame_loop_create_info, const engine::PassExecutorCreateInfo &pass_executor_create_info)
    : sdl_video_(std::move(sdl_video)), window_(std::move(window)), window_id_(window_id),
      sync_swapchain_extent_with_window_(sync_swapchain_extent_with_window), engine_(std::move(engine)), surface_(std::move(surface)),
      swapchain_(engine::SwapchainContext::create(engine_, surface_, swapchain_create_info)),
      pass_frame_loop_(engine::PassFrameLoop::create(engine_, swapchain_, frame_loop_create_info, pass_executor_create_info)),
      swapchain_create_info_(swapchain_create_info) {}

SDL_Window *AppCore::window() const noexcept { return window_.get(); }

SDL_WindowID AppCore::window_id() const noexcept { return window_id_; }

engine::EngineContext &AppCore::engine() noexcept { return engine_; }

const engine::EngineContext &AppCore::engine() const noexcept { return engine_; }

engine::SurfaceContext &AppCore::surface() noexcept { return surface_; }

const engine::SurfaceContext &AppCore::surface() const noexcept { return surface_; }

engine::SwapchainContext &AppCore::swapchain() noexcept { return swapchain_; }

const engine::SwapchainContext &AppCore::swapchain() const noexcept { return swapchain_; }

engine::PassFrameLoop &AppCore::pass_frame_loop() noexcept { return pass_frame_loop_; }

const engine::PassFrameLoop &AppCore::pass_frame_loop() const noexcept { return pass_frame_loop_; }

bool AppCore::poll_event(SDL_Event *event) const {
  if (event == nullptr) {
    throw std::invalid_argument("AppCore::poll_event requires a valid SDL_Event pointer.");
  }
  return SDL_PollEvent(event);
}

AppEventSignals AppCore::classify_event(const SDL_Event &event) const {
  AppEventSignals signals{};

  if (event.type == static_cast<std::uint32_t>(SDL_EVENT_QUIT)) {
    signals.close_requested = true;
    return signals;
  }
  if (!is_window_event_type(event.type) || event.window.windowID != window_id_) {
    return signals;
  }

  switch (event.type) {
  case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
  case SDL_EVENT_WINDOW_DESTROYED:
    signals.close_requested = true;
    break;
  default:
    break;
  }

  switch (event.type) {
  case SDL_EVENT_WINDOW_RESIZED:
  case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
  case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
  case SDL_EVENT_WINDOW_SAFE_AREA_CHANGED:
  case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
  case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
  case SDL_EVENT_WINDOW_RESTORED:
    signals.swapchain_recreate_requested = true;
    break;
  default:
    break;
  }

  return signals;
}

engine::SwapchainCreateInfo AppCore::swapchain_recreate_info_from_window() const {
  engine::SwapchainCreateInfo recreate_info = swapchain_create_info_;
  update_swapchain_extent_from_window(window_.get(), sync_swapchain_extent_with_window_, &recreate_info);
  return recreate_info;
}

void AppCore::wait_idle() const { pass_frame_loop_.wait_idle(); }

} // namespace varre::app
