#pragma once

// 自瞄系统公共数据结构

#include <array>

#include <opencv2/core.hpp>

namespace rm {

enum class EnemyColor { kRed = 0, kBlue = 1 };

struct LightBar {
  cv::Rect rect;       // 灯条轴对齐外接矩形（像素）
  cv::Point2f center;  // 矩形中心
};

struct Armor {
  LightBar left;                          // 左侧灯条
  LightBar right;                         // 右侧灯条
  std::array<cv::Point2f, 4> corners{};   // P1 左上, P2 右上, P3 右下, P4 左下
  cv::Rect rect;                          // 四角外接矩形（绘图/裁剪用）
  cv::Point2f center;                     // 对角交点（PnP 无效时的 2D 中心）
  EnemyColor color = EnemyColor::kRed;
};

// 单块装甲板的 PnP 结果（相机坐标系，单位米）
struct ArmorPose {
  bool valid = false;
  cv::Vec3d rvec{0.0, 0.0, 0.0};
  cv::Vec3d tvec{0.0, 0.0, 0.0};
  double reproj_error_px = 0.0;  // 四角平均回投误差
  double distance_m = 0.0;       // ||tvec||
  double yaw_deg = 0.0;          // 板面法线相对相机光轴的偏航
  double pitch_deg = 0.0;        // 板面法线俯仰
  cv::Point2f projected_center;  // 三维中心投影回图像
};

struct ArmorObservation {
  Armor armor;
  ArmorPose pose;
};

}  
