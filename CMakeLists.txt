cmake_minimum_required(VERSION 3.16)
project(AegisVision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
message(STATUS "AegisVision: OpenCV ${OpenCV_VERSION} @ ${OpenCV_DIR}")

# ---------------------------------------------------------------------------
# 核心库：所有算法都在这里，方便 ROS2 节点 / 单元测试复用
# ---------------------------------------------------------------------------
add_library(aegis_core STATIC
    src/config.cpp
    src/adaptive_detector.cpp
    src/armor_tracker.cpp
    src/pnp_solver.cpp
    src/pipeline.cpp
    src/hud.cpp
    src/digit_recognizer.cpp
)

target_include_directories(aegis_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(aegis_core PUBLIC ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(aegis_core PRIVATE /utf-8 /W3 /O2)
endif()

# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
add_executable(aegis src/main.cpp)
target_link_libraries(aegis PRIVATE aegis_core)
if(MSVC)
    target_compile_options(aegis PRIVATE /utf-8)
endif()

# ---------------------------------------------------------------------------
# 开发期勘察工具（默认不编译，-DAEGIS_BUILD_TOOLS=ON 时才生成）
# ---------------------------------------------------------------------------
option(AEGIS_BUILD_TOOLS "Build development probe tools" OFF)
if(AEGIS_BUILD_TOOLS)
    foreach(tool probe probe2 probe3)
        add_executable(${tool} src/${tool}.cpp)
        target_include_directories(${tool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${tool} PRIVATE ${OpenCV_LIBS})
        if(MSVC)
            target_compile_options(${tool} PRIVATE /utf-8)
        endif()
    endforeach()

    add_executable(probe_digit src/probe_digit.cpp)
    target_link_libraries(probe_digit PRIVATE aegis_core)
    if(MSVC)
        target_compile_options(probe_digit PRIVATE /utf-8)
    endif()

    add_executable(sweep src/sweep.cpp)
    target_link_libraries(sweep PRIVATE aegis_core)
    if(MSVC)
        target_compile_options(sweep PRIVATE /utf-8)
    endif()

    add_executable(vframe src/vframe.cpp)
    target_link_libraries(vframe PRIVATE ${OpenCV_LIBS})
endif()

# ---------------------------------------------------------------------------
# Windows：把 OpenCV 运行库复制到 exe 目录，双击即可运行
# ---------------------------------------------------------------------------
if(WIN32)
    file(GLOB OPENCV_RUNTIME_DLLS "${OpenCV_DIR}/x64/*/bin/opencv_*.dll")
    list(FILTER OPENCV_RUNTIME_DLLS EXCLUDE REGEX "d\\.dll$")
    foreach(target_name aegis)
        if(OPENCV_RUNTIME_DLLS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_RUNTIME_DLLS}
                        "$<TARGET_FILE_DIR:${target_name}>"
                COMMENT "Copying OpenCV DLLs")
        endif()
    endforeach()
endif()


