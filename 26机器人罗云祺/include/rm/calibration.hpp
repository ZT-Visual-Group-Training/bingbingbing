#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace rm
{

// 相机内外参
struct Calibration
{
  cv::Mat K;   // 相机内参矩阵
  cv::Mat D;   // 畸变系数
  cv::Matx33d R_camera2gimbal = cv::Matx33d::eye();
  cv::Vec3d t_camera2gimbal{0, 0, 0};
  // 图像尺寸
  int width = 0;
  int height = 0;

  static Calibration load(const std::string & path);
};

}
