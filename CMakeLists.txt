cmake_minimum_required(VERSION 3.16)
project(rm_autoaim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d videoio dnn)
find_package(yaml-cpp REQUIRED)

add_executable(rm_autoaim
    src/main.cpp
    src/calibration.cpp
    src/detector.cpp
    src/digit_recognizer.cpp
    src/tracker.cpp
    src/aimer.cpp
    src/utils.cpp
)

target_include_directories(rm_autoaim PRIVATE include ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR})
if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(rm_autoaim PRIVATE ${OpenCV_LIBS} yaml-cpp::yaml-cpp)
else()
    target_link_libraries(rm_autoaim PRIVATE ${OpenCV_LIBS} yaml-cpp)
endif()
