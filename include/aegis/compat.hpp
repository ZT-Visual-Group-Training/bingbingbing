// compat.hpp —— OpenCV 4.x / 5.x 兼容层
// OpenCV 5.0 把 findContours / contourArea / minAreaRect / fitLine 等 2D 几何函数
// 从 imgproc 拆到了新的 geometry 模块；这里按“头文件是否存在”自动适配，
// 保证同一份源码既能在 Ubuntu 22.04 + OpenCV 4.5.4 上编译，也能在本机 OpenCV 5.0 上编译。
#pragma once

#include <opencv2/opencv.hpp>

#if defined(__has_include)
#if __has_include(<opencv2/geometry/2d.hpp>)
#include <opencv2/geometry/2d.hpp>
#endif
#if __has_include(<opencv2/geometry/3d.hpp>)
#include <opencv2/geometry/3d.hpp>
#endif
#endif
