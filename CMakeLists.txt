cmake_minimum_required(VERSION 3.10)
project(demo5)

set(CMAKE_CXX_STANDARD 17)

# 本机 vcpkg 的 OpenCV 只有 Release 版 DLL，Debug 配置会因缺
# libopencv_calib3d4d.dll 无法运行。这里强制 Release，保证 CLion 里
# 直接点运行就能跑。迁到 Ubuntu 后（apt 版 OpenCV）此设置无害，可保留。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)

# 检测器实现（检测算法本体，和主程序、运行模式无关）
# TemporalFilter.cpp 时序多数投票滤波器：默认关闭，只在 --temporal 时生效
add_library(armor_core ArmorDetector.cpp ImuReader.cpp TemporalFilter.cpp)
target_include_directories(armor_core PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(armor_core PRIVATE ${OpenCV_LIBS})

# 统一主程序：一个 main.cpp + --mode 切换 image / video / hsv 三种功能。
# 以前的 main_video.cpp / hsv_tuner.cpp 已合并进 main.cpp，备份在 _legacy/unified_20261123/。
add_executable(armor_vision main.cpp)
target_include_directories(armor_vision PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(armor_vision PRIVATE armor_core ${OpenCV_LIBS})

# 检测流程可视化工具（独立程序，学习/答辩用，不影响 armor_vision）。
# 运行后生成 pipeline_truth.png：颜色阈值->形状筛选->余光被淘汰的完整流程图。
# 在 CLion 里选 debug_pipeline 这个 target，直接点运行即可。
add_executable(debug_pipeline debug_pipeline.cpp)
target_include_directories(debug_pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(debug_pipeline PRIVATE ${OpenCV_LIBS})

# HSV 调参专用程序：零参数启动即进 HSV 调试器，直接弹窗拖滑块。
# 编译宏让 main 默认模式变成 hsv，绕开"CLion 运行配置参数丢失/不会填"的问题。
# 在 CLion 里选 hsv_tuner 这个 target，点运行 ▶ 即可，什么都不用填。
add_executable(hsv_tuner main.cpp)
target_compile_definitions(hsv_tuner PRIVATE HSV_TUNER_STANDALONE)
target_include_directories(hsv_tuner PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hsv_tuner PRIVATE armor_core ${OpenCV_LIBS})

# IMU 模块体检程序：验证 CSV 读取 / 四元数转姿态角 / 手眼外参坐标转换。
# 在 CLion 里选 imu_check target 运行即可，不检测装甲板、秒出结果。
# （答辩演示用：保留）
add_executable(imu_check imu_check.cpp)
target_include_directories(imu_check PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(imu_check PRIVATE armor_core ${OpenCV_LIBS})

# 说明：性能剖析/降分辨率A-B/时序A-B 等临时诊断工具（profile_detect、
# bench_detect、ab_scale、ab_temporal、diag_temporal）已在交付整理时移除，
# 只保留核心检测 + 答辩演示所需目标。完整工具链见开发工程 D:/demo5/。
