#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "rm/config.hpp"
#include "rm/types.hpp"

namespace rm
{

// 调试
struct DetectorDebug
{
  cv::Mat binary;                   // 二值图
  std::vector<Lightbar> lightbars;  // 通过几何筛选的灯条
};

// 传统视觉检测：灰度阈值 -> 轮廓 -> 灯条 -> 配对 -> 装甲板
class Detector
{
public:
  explicit Detector(const Config & cfg);

  std::vector<Armor> detect(const cv::Mat & bgr);

  const DetectorDebug & debug() const { return debug_; }

private:
  bool checkLightbar(const Lightbar & bar) const;
  bool checkArmor(const Lightbar & left, const Lightbar & right) const;
  Color colorOf(const cv::Mat & bgr, const std::vector<cv::Point> & contour) const;

  // 防止外部 Config 销毁
  Config cfg_;
  // 每帧更新的调试状态
  DetectorDebug debug_;

  double threshold_ = 150.0;
  double max_angle_error_ = 45.0 * CV_PI / 180.0;
  double min_bar_ratio_ = 1.5;
  double max_bar_ratio_ = 20.0;
  double min_bar_len_ = 8.0;
  double min_armor_ratio_ = 1.0;
  double max_armor_ratio_ = 5.0;
  double max_side_ratio_ = 1.5;
  double max_rect_error_ = 25.0 * CV_PI / 180.0;
  double min_armor_width_ = 60.0;  // 像素；滤掉远处灯带/反光凑出来的“小装甲板”
  int enemy_color_ = 0;  // 0 红 / 1 蓝 / -1 不过滤
};

}
