/**
 * @file core.hpp
 * @brief SDL3-backed app bootstrap and runtime ownership for pass-mode apps.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <SDL3/SDL.h>

#include "varre/engine/engine.hpp"
#include "varre/engine/pass_frame_loop.hpp"
#include "varre/engine/surface.hpp"
#include "varre/engine/swapchain.hpp"

namespace varre::app {

/**
 * @brief SDL window creation parameters.
 */
struct SdlWindowCreateInfo {
  /** @brief Window title. */
  std::string title = "varre";
  /** @brief Initial window width in logical coordinates. */
  int width = 1280;
  /** @brief Initial window height in logical coordinates. */
  int height = 720;
  /** @brief Create the window as resizable. */
  bool resizable = true;
  /** @brief Request high-pixel-density back buffer where supported. */
  bool high_pixel_density = true;
};

/**
 * @brief Bootstrap inputs for one app runtime.
 */
struct AppCoreCreateInfo {
  /** @brief SDL window creation info. */
  SdlWindowCreateInfo window;
  /** @brief Engine initialization inputs. */
  engine::EngineInitInfo engine;
  /** @brief Initial swapchain creation info. */
  engine::SwapchainCreateInfo swapchain;
  /** @brief Frame-loop creation options. */
  engine::FrameLoopCreateInfo frame_loop;
  /** @brief Pass-executor creation options. */
  engine::PassExecutorCreateInfo pass_executor;
  /**
   * @brief Keep swapchain preferred extent synchronized with current window pixel size.
   *
   * When true, @ref swapchain_recreate_info_from_window will refresh extent values.
   */
  bool sync_swapchain_extent_with_window = true;
};

/**
 * @brief High-level app signals extracted from one SDL event.
 */
struct AppEventSignals {
  /** @brief True when app shutdown was requested. */
  bool close_requested = false;
  /** @brief True when swapchain recreation is recommended. */
  bool swapchain_recreate_requested = false;
};

/**
 * @brief SDL/engine bootstrap runtime for app targets.
 */
class AppCore {
public:
  /**
   * @brief Create one SDL window + Vulkan runtime stack.
   * @param create_info Bootstrap inputs.
   * @return Initialized app runtime.
   */
  [[nodiscard]] static AppCore create(const AppCoreCreateInfo &create_info = {});

  /**
   * @brief Query SDL-required Vulkan instance extensions for one Vulkan-capable window.
   * @param window SDL window created with `SDL_WINDOW_VULKAN`.
   * @return Required extension names.
   */
  [[nodiscard]] static std::vector<std::string> query_required_instance_extensions(SDL_Window *window);

  /**
   * @brief Merge SDL-required instance extensions into engine init info.
   * @param init_info Destination engine init info.
   * @param window SDL window created with `SDL_WINDOW_VULKAN`.
   */
  static void append_required_instance_extensions(engine::EngineInitInfo *init_info, SDL_Window *window);

  /**
   * @brief Create and adopt a Vulkan surface for one SDL window.
   * @param engine Initialized engine context.
   * @param window SDL window created with `SDL_WINDOW_VULKAN`.
   * @return Adopted surface context.
   */
  [[nodiscard]] static engine::SurfaceContext create_surface_for_window(const engine::EngineContext &engine, SDL_Window *window);

  AppCore(AppCore &&other) noexcept = delete;
  AppCore &operator=(AppCore &&other) noexcept = delete;
  AppCore(const AppCore &) = delete;
  AppCore &operator=(const AppCore &) = delete;

  /**
   * @brief Access the owned SDL window.
   * @return SDL window pointer.
   */
  [[nodiscard]] SDL_Window *window() const noexcept;

  /**
   * @brief Access SDL window identifier.
   * @return Window ID.
   */
  [[nodiscard]] SDL_WindowID window_id() const noexcept;

  /**
   * @brief Access initialized engine context.
   * @return Mutable engine context.
   */
  [[nodiscard]] engine::EngineContext &engine() noexcept;

  /**
   * @brief Access initialized engine context.
   * @return Immutable engine context.
   */
  [[nodiscard]] const engine::EngineContext &engine() const noexcept;

  /**
   * @brief Access adopted surface context.
   * @return Mutable surface context.
   */
  [[nodiscard]] engine::SurfaceContext &surface() noexcept;

  /**
   * @brief Access adopted surface context.
   * @return Immutable surface context.
   */
  [[nodiscard]] const engine::SurfaceContext &surface() const noexcept;

  /**
   * @brief Access current swapchain context.
   * @return Mutable swapchain context.
   */
  [[nodiscard]] engine::SwapchainContext &swapchain() noexcept;

  /**
   * @brief Access current swapchain context.
   * @return Immutable swapchain context.
   */
  [[nodiscard]] const engine::SwapchainContext &swapchain() const noexcept;

  /**
   * @brief Access pass-frame runtime.
   * @return Mutable pass-frame loop.
   */
  [[nodiscard]] engine::PassFrameLoop &pass_frame_loop() noexcept;

  /**
   * @brief Access pass-frame runtime.
   * @return Immutable pass-frame loop.
   */
  [[nodiscard]] const engine::PassFrameLoop &pass_frame_loop() const noexcept;

  /**
   * @brief Poll one SDL event.
   * @param event Destination event storage.
   * @return True when an event was dequeued.
   */
  [[nodiscard]] bool poll_event(SDL_Event *event) const;

  /**
   * @brief Classify one SDL event into app-level control signals.
   * @param event SDL event value.
   * @return Extracted app signals.
   */
  [[nodiscard]] AppEventSignals classify_event(const SDL_Event &event) const;

  /**
   * @brief Build swapchain recreate info synchronized to the current window size.
   * @return Swapchain create info suitable for recreation.
   */
  [[nodiscard]] engine::SwapchainCreateInfo swapchain_recreate_info_from_window() const;

  /**
   * @brief Wait until runtime-owned GPU work is complete.
   */
  void wait_idle() const;

private:
  /**
   * @brief RAII guard for SDL video subsystem lifetime.
   */
  class SdlVideoSubsystemGuard {
  public:
    /**
     * @brief Initialize SDL video subsystem.
     * @return Initialized guard.
     */
    [[nodiscard]] static SdlVideoSubsystemGuard create();

    SdlVideoSubsystemGuard(SdlVideoSubsystemGuard &&other) noexcept;
    SdlVideoSubsystemGuard &operator=(SdlVideoSubsystemGuard &&other) noexcept;

    SdlVideoSubsystemGuard(const SdlVideoSubsystemGuard &) = delete;
    SdlVideoSubsystemGuard &operator=(const SdlVideoSubsystemGuard &) = delete;

    ~SdlVideoSubsystemGuard();

  private:
    explicit SdlVideoSubsystemGuard(bool initialized) noexcept;

    bool initialized_ = false;
  };

  /**
   * @brief Deleter for owned SDL window handles.
   */
  struct WindowDeleter {
    void operator()(SDL_Window *window) const noexcept;
  };

  using WindowPtr = std::unique_ptr<SDL_Window, WindowDeleter>;

  /**
   * @brief Synchronize swapchain preferred extent from current window pixel size.
   * @param window SDL window.
   * @param sync_enabled Whether synchronization is enabled.
   * @param swapchain_create_info Destination recreate info.
   */
  static void update_swapchain_extent_from_window(SDL_Window *window, bool sync_enabled, engine::SwapchainCreateInfo *swapchain_create_info);

  /**
   * @brief Internal constructor from initialized resources.
   */
  AppCore(SdlVideoSubsystemGuard &&sdl_video, WindowPtr &&window, SDL_WindowID window_id, bool sync_swapchain_extent_with_window,
          engine::EngineContext &&engine, engine::SurfaceContext &&surface, engine::SwapchainCreateInfo swapchain_create_info,
          const engine::FrameLoopCreateInfo &frame_loop_create_info, const engine::PassExecutorCreateInfo &pass_executor_create_info);

  SdlVideoSubsystemGuard sdl_video_;
  WindowPtr window_{nullptr, WindowDeleter{}};
  SDL_WindowID window_id_ = 0U;
  bool sync_swapchain_extent_with_window_ = true;
  engine::EngineContext engine_;
  engine::SurfaceContext surface_;
  engine::SwapchainContext swapchain_;
  engine::PassFrameLoop pass_frame_loop_;
  engine::SwapchainCreateInfo swapchain_create_info_{};
};

} // namespace varre::app
