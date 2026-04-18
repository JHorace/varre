include(FetchContent)

# Workspace subtree matching the Rust crate `varre-engine`.
# Dependencies are resolved here so later engine targets can link a single
# interface target.

find_package(Vulkan REQUIRED)

find_package(glm CONFIG QUIET)
if(NOT TARGET glm::glm)
  FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(glm)
endif()
if(NOT TARGET glm::glm)
  message(FATAL_ERROR "Unable to resolve glm target.")
endif()

if(NOT TARGET varre_stb_image)
  add_library(varre_stb_image INTERFACE)
endif()
find_path(VARRE_STB_INCLUDE_DIR stb_image.h)
if(VARRE_STB_INCLUDE_DIR)
  target_include_directories(varre_stb_image INTERFACE "${VARRE_STB_INCLUDE_DIR}")
else()
  FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(stb)
  target_include_directories(varre_stb_image INTERFACE "${stb_SOURCE_DIR}")
endif()

find_package(VulkanMemoryAllocator CONFIG QUIET)
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
  set(VMA_BUILD_SAMPLES OFF CACHE BOOL "Disable VMA sample builds." FORCE)
  set(VMA_BUILD_DOCUMENTATION OFF CACHE BOOL "Disable VMA docs." FORCE)
  set(VMA_ENABLE_INSTALL OFF CACHE BOOL "Disable VMA install rules." FORCE)
  FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.2.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(VulkanMemoryAllocator)
endif()
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
  message(FATAL_ERROR "Expected GPUOpen::VulkanMemoryAllocator target was not found.")
endif()

find_package(fmt CONFIG QUIET)
if(NOT TARGET fmt::fmt)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.0.2
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(fmt)
endif()
if(NOT TARGET fmt::fmt)
  message(FATAL_ERROR "Unable to resolve fmt target.")
endif()

find_package(spdlog CONFIG QUIET)
if(NOT TARGET spdlog::spdlog)
  set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "Disable spdlog examples." FORCE)
  set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Disable spdlog tests." FORCE)
  set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Disable spdlog benchmarks." FORCE)
  set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt target." FORCE)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(spdlog)
endif()
if(NOT TARGET spdlog::spdlog)
  message(FATAL_ERROR "Unable to resolve spdlog target.")
endif()

if(NOT TARGET varre_assets)
  message(FATAL_ERROR "varre_engine requires varre_assets target to be available.")
endif()

if(NOT TARGET varre_engine_dependencies)
  add_library(varre_engine_dependencies INTERFACE)
endif()
add_library(varre::engine_dependencies ALIAS varre_engine_dependencies)

target_compile_features(varre_engine_dependencies INTERFACE cxx_std_23)
target_link_libraries(
  varre_engine_dependencies
  INTERFACE
    Vulkan::Vulkan
    glm::glm
    varre_stb_image
    GPUOpen::VulkanMemoryAllocator
    fmt::fmt
    spdlog::spdlog
    varre_assets
)
if(TARGET Vulkan::Headers)
  target_link_libraries(varre_engine_dependencies INTERFACE Vulkan::Headers)
endif()

if(UNIX AND NOT APPLE)
  # Enable Linux-specific surface support.
  # This allows use of Wayland, XCB, and X11 Vulkan types and extensions.
  target_compile_definitions(varre_engine_dependencies INTERFACE
    VK_USE_PLATFORM_WAYLAND_KHR
    VK_USE_PLATFORM_XCB_KHR
    VK_USE_PLATFORM_XLIB_KHR
  )
endif()

if(WIN32)
  # Enable Win32 surface support on Windows.
  target_compile_definitions(varre_engine_dependencies INTERFACE VK_USE_PLATFORM_WIN32_KHR)
endif()

if(BUILD_TESTING)
  find_package(Catch2 3 CONFIG QUIET)
  if(NOT Catch2_FOUND)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.6.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
  endif()
  if(NOT TARGET Catch2::Catch2WithMain)
    message(FATAL_ERROR "Unable to resolve Catch2::Catch2WithMain target.")
  endif()

  if(NOT TARGET varre_engine_test_dependencies)
    add_library(varre_engine_test_dependencies INTERFACE)
  endif()
  add_library(varre::engine_test_dependencies ALIAS varre_engine_test_dependencies)

  target_compile_features(varre_engine_test_dependencies INTERFACE cxx_std_23)
  target_link_libraries(
    varre_engine_test_dependencies
    INTERFACE
      varre_engine_dependencies
      Catch2::Catch2WithMain
  )
endif()

add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
