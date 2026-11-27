cmake_minimum_required(VERSION 3.16)
project(rm_autoaim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

# Ubuntu 22.04 ships OpenCV 4.5.4 as libopencv-dev. Requiring 4.5.4 is enough:
# solvePnPGeneric / SOLVEPNP_IPPE / solvePnPRefineLM all exist by then.
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio calib3d video)

message(STATUS "OpenCV ${OpenCV_VERSION} at ${OpenCV_DIR}")

add_executable(autoaim
  src/main.cpp
  src/config.cpp
  src/types.cpp
  src/armor_corners.cpp
  src/light_bar_detector.cpp
  src/armor_matcher.cpp
  src/pnp_solver.cpp
  src/tracker.cpp
  src/anti_gyro.cpp
  src/visualizer.cpp
)

target_include_directories(autoaim PRIVATE include)
target_link_libraries(autoaim PRIVATE ${OpenCV_LIBS})

target_compile_options(autoaim PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)

# -march=native is deliberately NOT enabled: it makes the binary crash with an
# illegal instruction on a different CPU, and a demo clip will be rendered on a
# different machine than the one that built it.
option(AUTOAIM_ENABLE_MARCH_NATIVE "build with -march=native" OFF)
if(AUTOAIM_ENABLE_MARCH_NATIVE)
  target_compile_options(autoaim PRIVATE -march=native)
endif()

install(TARGETS autoaim RUNTIME DESTINATION bin)
