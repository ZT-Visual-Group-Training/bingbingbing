cmake_minimum_required(VERSION 3.16)
project(aegis_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 直接复用主工程的核心源码（不在 ROS2 里重复实现算法）
set(AEGIS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(AEGIS_SRC
  ${AEGIS_ROOT}/src/config.cpp
  ${AEGIS_ROOT}/src/adaptive_detector.cpp
  ${AEGIS_ROOT}/src/armor_tracker.cpp
  ${AEGIS_ROOT}/src/pnp_solver.cpp
  ${AEGIS_ROOT}/src/digit_recognizer.cpp
  ${AEGIS_ROOT}/src/pipeline.cpp
  ${AEGIS_ROOT}/src/hud.cpp
)

add_library(aegis_core STATIC ${AEGIS_SRC})
target_include_directories(aegis_core PUBLIC
  ${AEGIS_ROOT}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(aegis_core PUBLIC ${OpenCV_LIBS})

add_executable(aegis_node src/aegis_node.cpp)
ament_target_dependencies(aegis_node rclcpp sensor_msgs geometry_msgs std_msgs cv_bridge)
target_link_libraries(aegis_node aegis_core ${OpenCV_LIBS})

install(TARGETS aegis_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()

