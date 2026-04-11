cmake_minimum_required(VERSION 3.25)

project(varre VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectOptions)

if(VARRE_BUILD_ASSETS)
  add_subdirectory(varre-assets)
endif()

if(VARRE_BUILD_ENGINE)
  add_subdirectory(varre-engine)
endif()

if(VARRE_BUILD_APPS)
  add_subdirectory(varre-app)
endif()

