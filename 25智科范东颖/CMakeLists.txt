cmake_minimum_required(VERSION 3.16)
project(rm_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
  set(OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "Directory of OpenCVConfig.cmake")
endif()
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
if(OpenCV_VERSION VERSION_LESS "4.8")
  message(WARNING "OpenCV ${OpenCV_VERSION} < 4.8: dnn may fail to load the ONNX models; "
                  "configure with -DOpenCV_DIR=<opencv>/lib/cmake/opencv4")
endif()

option(USE_OPENVINO "Use OpenVINO for YOLO inference (faster on Intel CPU)" OFF)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(vision
    src/main.cpp
    src/utils.cpp
    src/detector.cpp
    src/pnp.cpp
    src/tracker.cpp
    src/number_classifier.cpp
    src/aimer.cpp
)

target_link_libraries(vision ${OpenCV_LIBS})

if(USE_OPENVINO)
  if(NOT OpenVINO_DIR)
    set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake" CACHE PATH "OpenVINO cmake dir")
  endif()
  find_package(OpenVINO REQUIRED)
  target_compile_definitions(vision PRIVATE HAVE_OPENVINO)
  target_link_libraries(vision openvino::runtime)
  message(STATUS "YOLO backend: OpenVINO (${OpenVINO_DIR})")
else()
  message(STATUS "YOLO backend: OpenCV dnn (use -DUSE_OPENVINO=ON for OpenVINO)")
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/model ${CMAKE_CURRENT_SOURCE_DIR}/config
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
