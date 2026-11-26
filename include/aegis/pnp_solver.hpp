// =============================================================================
// pnp_solver.hpp —— 特色三（上）：单目 PnP 三维定位
// =============================================================================
#pragma once

#include "aegis/config.hpp"
#include "aegis/types.hpp"

namespace aegis {

class PnPSolver {
 public:
  PnPSolver(const CameraConfig& camera, const ArmorConfig& armor);

  // 输入一块检测到的装甲板（4 个角点 + 大小类型），输出相机系位姿。
  Pose solve(const Armor& armor) const;

  // 把相机系坐标转到云台系：p_gimbal = R_cg * p_cam + t_cg。
  cv::Vec3d cameraToGimbal(const cv::Vec3d& pCam) const;

  // 相机系 -> 世界系（用 IMU 四元数），用于三维轨迹可视化。
  cv::Vec3d cameraToWorld(const cv::Vec3d& pCam, const cv::Vec4d& qGimbal) const;

 private:
  CameraConfig camera_;
  ArmorConfig armor_;
};

}  // namespace aegis
