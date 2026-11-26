cmake_minimum_required(VERSION 3.16)
project(rm_autoaim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/utf-8 /W3)
else()
  add_compile_options(-Wall -Wextra -O2)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui calib3d)

file(GLOB_RECURSE RM_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(rm_autoaim ${RM_SOURCES})
target_include_directories(rm_autoaim PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(rm_autoaim PRIVATE ${OpenCV_LIBS})

message(STATUS "OpenCV ${OpenCV_VERSION} (${OpenCV_DIR})")
