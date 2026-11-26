// =============================================================================
// pnp_solver.cpp —— 用 4 个角点解算装甲板在相机系下的 6DoF 位姿
//
// 数学：已知相机内参 K、畸变 D、装甲板物理尺寸 (W,H,0) 和它在图像上的 4 个角点，
//      求 R,t 使得 project(K,D,R,t,P3d) ≈ p2d。
//      这是典型的 PnP（Perspective-n-Point）问题。
//      4 个共面点用 IPPE 有闭式解，再用 LM 迭代最小化重投影误差。
// =============================================================================
#include "aegis/pnp_solver.hpp"

#include <cmath>

namespace aegis {
namespace {

constexpr double kPi = 3.14159265358979323846;

double toDeg(double rad) { return rad * 180.0 / kPi; }

// 四元数 (w,x,y,z) -> 旋转矩阵。IMU 给的是云台在世界系下的姿态。
cv::Matx33d quatToRotation(const cv::Vec4d& qIn) {
  cv::Vec4d q = qIn;
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1e-9) return cv::Matx33d::eye();
  q /= n;
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];
  return cv::Matx33d(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y));
}

}  // namespace

PnPSolver::PnPSolver(const CameraConfig& camera, const ArmorConfig& armor)
    : camera_(camera), armor_(armor) {}

Pose PnPSolver::solve(const Armor& armor) const {
  Pose pose;
  if (armor.width <= 0.0F || armor.height <= 0.0F) return pose;

  // 物理尺寸：按检测阶段推断出的小/大装甲板选择宽度，高度固定 55mm。
  const double w = (armor.size == ArmorSize::Large) ? armor_.largeWidthM : armor_.smallWidthM;
  const double h = armor_.heightM;

  // 装甲板自身坐标系：原点在板中心，x 向右，y 向下，z 垂直板面向外。
  // 顺序必须和图像角点一致：左上、右上、右下、左下。
  const std::vector<cv::Point3f> objectPoints = {
      cv::Point3f(static_cast<float>(-w / 2.0), static_cast<float>(-h / 2.0), 0.0F),
      cv::Point3f(static_cast<float>(w / 2.0), static_cast<float>(-h / 2.0), 0.0F),
      cv::Point3f(static_cast<float>(w / 2.0), static_cast<float>(h / 2.0), 0.0F),
      cv::Point3f(static_cast<float>(-w / 2.0), static_cast<float>(h / 2.0), 0.0F),
  };
  std::vector<cv::Point2f> imagePoints(armor.corners.begin(), armor.corners.end());

  cv::Vec3d rvec;
  cv::Vec3d tvec;
  // SOLVEPNP_IPPE：针对“共面 4 点”的解析解，比迭代法快且不会陷入局部极小。
  const bool ok = cv::solvePnP(objectPoints, imagePoints, camera_.cameraMatrix, camera_.distCoeffs,
                               rvec, tvec, false, cv::SOLVEPNP_IPPE);
  if (!ok || tvec[2] <= 0.0) return pose;  // 目标必须在相机前方

  // 用 LM 再抛光一次，压低重投影误差（实测从 0.6px 降到 0.2px 量级）。
  try {
    cv::solvePnPRefineLM(objectPoints, imagePoints, camera_.cameraMatrix, camera_.distCoeffs, rvec,
                         tvec, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20,
                                                1e-6));
  } catch (const cv::Exception&) {
    // 极端情况下 LM 不收敛，保留 IPPE 的结果即可。
  }

  // 重投影误差：把 3D 点重新投回图像，和检测到的角点求平均欧氏距离。
  std::vector<cv::Point2f> reprojected;
  cv::projectPoints(objectPoints, rvec, tvec, camera_.cameraMatrix, camera_.distCoeffs, reprojected);
  double err = 0.0;
  for (std::size_t i = 0; i < reprojected.size(); ++i) {
    err += cv::norm(reprojected[i] - imagePoints[i]);
  }
  err /= static_cast<double>(reprojected.size());

  pose.valid = true;
  pose.tvec = tvec;
  pose.rvec = rvec;
  pose.distance = cv::norm(tvec);
  pose.yaw = toDeg(std::atan2(tvec[0], tvec[2]));
  pose.pitch = toDeg(std::atan2(-tvec[1], std::sqrt(tvec[0] * tvec[0] + tvec[2] * tvec[2])));
  pose.reprojError = err;
  pose.gimbal = cameraToGimbal(tvec);
  return pose;
}

cv::Vec3d PnPSolver::cameraToGimbal(const cv::Vec3d& pCam) const {
  return camera_.R_camera2gimbal * pCam + camera_.t_camera2gimbal;
}

cv::Vec3d PnPSolver::cameraToWorld(const cv::Vec3d& pCam, const cv::Vec4d& qGimbal) const {
  const cv::Vec3d pGimbal = cameraToGimbal(pCam);
  return quatToRotation(qGimbal) * pGimbal;
}

}  // namespace aegis

