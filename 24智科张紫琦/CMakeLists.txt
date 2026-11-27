cmake_minimum_required(VERSION 3.16)
project(rm_autoaim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 统一 Release 优化级别为 -O2
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(OpenCV REQUIRED)

add_executable(rm_autoaim
  src/app/main.cpp
  src/detect/color_detector.cpp
  src/geometry/pnp.cpp
  src/viz/overlay.cpp
)

target_include_directories(rm_autoaim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(rm_autoaim PRIVATE ${OpenCV_LIBRARY_DIRS})
target_link_libraries(rm_autoaim PRIVATE ${OpenCV_LIBS})

target_compile_options(rm_autoaim PRIVATE -Wall -Wextra)
