#pragma once

#include <opencv2/core.hpp>

#include "rm/types.hpp"

namespace rm
{

// 匀速模型跟踪 + 预测
class Tracker
{
public:
  void reset();
  void update(const Pose & pose, double t);  // 喂一个有效的观测，预测和更新

  // 预测 t 时刻的装甲板中心（相机系）
  bool predict(double t, cv::Vec3d & xyz) const;

private:
  bool has_ = false;
  double t_ = 0.0;
  cv::Vec3d p_{0, 0, 0};  // m
  cv::Vec3d v_{0, 0, 0};  // m/s
};

}
