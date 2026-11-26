#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

#include "rm/calibration.hpp"
#include "rm/config.hpp"
#include "rm/types.hpp"

namespace rm
{

class Solver
{
public:
  Solver(const Calibration & calib, const Config & cfg);

  // 输入装甲板四角点（TL,TR,BR,BL），输出装甲板中心在相机系下的位姿
  Pose solve(const Armor & armor) const;

  // 四角点投影回像素（画坐标轴和自检重投影误差）
  std::vector<cv::Point2f> project(const Pose & pose, ArmorSize size) const;

  // 对4个角点做去畸变（快且不糊图）
  std::vector<cv::Point2f> undistortCorners(const std::array<cv::Point2f, 4> & corners) const;

  // 返回4个3D角点
  std::array<cv::Point3f, 4> objectPoints(ArmorSize size) const;

  const Calibration & calibration() const { return calib_; }

private:
  Calibration calib_;
  double small_width_mm_ = 135.0;
  double small_height_mm_ = 56.0;
  double big_width_mm_ = 230.0;
  double big_height_mm_ = 56.0;
};

}
