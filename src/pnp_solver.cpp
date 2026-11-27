#include "autoaim/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>

// ===========================================================================
// 本文件：「位姿估计」——从 4 个像素点反推装甲板在相机坐标系下的位置
//
// 输入：Armor 的四个角点（像素）+ 相机内参 + 装甲板真实尺寸
// 输出：Pose { rvec, tvec, reproj_err, ok }，tvec 就是题目要的三维位置（米）
//
// 数学本质：
//   每个 3D↔2D 点对给出 2 个方程：
//       u = fx·(X/Z) + cx
//       v = fy·(Y/Z) + cy
//   未知数是 6 个（旋转 3 + 平移 3）。所以：
//       3 个点 → 6 个方程，刚好，但 P3P 最多有 4 组解（不唯一）
//       4 个点 → 8 个方程，超定 → IPPE 给出 2 组解，再消歧
//       6 个点 → 12 个方程，严重超定，可以最小二乘抗噪
//
// 为什么 4 点共面还有 2 组解：
//   平面目标存在"绕平面内某轴翻转 180°"的镜像解 —— 就像透过窗户看一个相框，
//   也可能是反过来放的相框，两者的投影几乎一样。称为平面位姿歧义。
//
// 三步实现，每一步都有明确目的：
//
//   ① undistortPoints（只对 4 个点，不做整图校正）
//      整图 undistort 要几十毫秒；只校 4 个点数学上完全等价，代价可忽略。
//      注意 P=K 的含义：把归一化坐标再用 K 投影回"理想像素坐标"，
//      所以后面的 solvePnP 必须传空畸变系数，否则会重复校正。
//
//   ② solvePnPGeneric + SOLVEPNP_IPPE（返回所有解，不只一个）
//      IPPE 是共面点的解析解，无需初值、稳定、快。它返回 2 组解，
//      我们逐组算重投影误差，取误差小的，并且要求 tvec.z > 0（目标在相机前方）。
//
//   ③ solvePnPRefineLM（Levenberg–Marquardt 精化）
//      在②的解附近最小化重投影误差，把角点噪声的影响压到最小。
//
// 重投影误差是"质量指标"，不只是消歧手段：
//   把解出的 R,t 加上已知 3D 点再投影回图像，和检测到的像素点比距离。
//   通俗说：用你估的位置把门框"画回去"，看能不能盖住照片里的门框。
//   实测正常帧 <1.2 px；>max_reproj_err_px（2.5）就认为该帧不可信、丢弃，
//   宁可漏检也不输出一个错误的坐标 —— 自瞄里错误的坐标比没有坐标更危险。
//
// 为什么"中心点"绝对不能喂给 PnP：
//   只用 1 个中心点 + 已知尺寸，无法反解 6 个自由度（信息不足）。
//   板中心只用来画十字和给跟踪器，PnP 的输入永远是那 4 个角点。
// ===========================================================================

namespace autoaim {

PnpSolver::PnpSolver(const CameraConfig& cam, const DetectorConfig& cfg)
    : cam_(cam), cfg_(cfg) {
  const double hw = cfg_.armor_width_m * 0.5;
  const double hh = cfg_.armor_height_m * 0.5;
  // Object frame: origin at plate centre, X right along 135 mm, Y DOWN along
  // 56 mm (same direction as the image v axis), Z out of the plate.
  // Order MUST match Armor::corners: P1 TL, P2 TR, P3 BR, P4 BL.
  obj_ = {
      cv::Point3f(static_cast<float>(-hw), static_cast<float>(-hh), 0.f),  // P1 TL
      cv::Point3f(static_cast<float>(+hw), static_cast<float>(-hh), 0.f),  // P2 TR
      cv::Point3f(static_cast<float>(+hw), static_cast<float>(+hh), 0.f),  // P3 BR
      cv::Point3f(static_cast<float>(-hw), static_cast<float>(+hh), 0.f),  // P4 BL
  };
  K_ = cam.cameraMatrix();
  D_ = cam.distCoeffs();
}

Pose PnpSolver::solve(const Armor& armor) const {
  Pose pose;
  std::vector<cv::Point2f> img_pts(armor.corners.begin(), armor.corners.end());

  // Undistort only the four corners: cheaper than undistorting the frame, and it
  // keeps the overlay drawn on the original (distorted) image.
  std::vector<cv::Point2f> und;
  cv::undistortPoints(img_pts, und, K_, D_, cv::noArray(), K_);

  std::vector<cv::Mat> rvecs, tvecs;
  cv::Mat reproj_err;
  int nsol = 0;
  try {
    nsol = cv::solvePnPGeneric(obj_, und, K_, cv::Mat(), rvecs, tvecs, false,
                               cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(),
                               reproj_err);
  } catch (const cv::Exception&) {
    return pose;
  }
  if (nsol <= 0) return pose;

  // IPPE also returns the mirror solution. Keep the smaller reprojection error
  // among solutions that actually sit in front of the camera.
  double best_err = 1e18;
  cv::Mat best_r, best_t;
  for (int i = 0; i < nsol; ++i) {
    if (tvecs[i].at<double>(2) <= 0.0) continue;
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj_, rvecs[i], tvecs[i], K_, cv::Mat(), proj);
    double acc = 0.0;
    for (size_t k = 0; k < proj.size(); ++k) acc += cv::norm(proj[k] - und[k]);
    const double err = acc / static_cast<double>(proj.size());
    if (err < best_err) {
      best_err = err;
      best_r = rvecs[i];
      best_t = tvecs[i];
    }
  }
  if (best_r.empty()) return pose;

  cv::Mat rvec = best_r.clone(), tvec = best_t.clone();
  if (cfg_.refine_iterations > 0) {
    cv::solvePnPRefineLM(obj_, und, K_, cv::Mat(), rvec, tvec,
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                          cfg_.refine_iterations, 1e-6));
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj_, rvec, tvec, K_, cv::Mat(), proj);
    double acc = 0.0;
    for (size_t k = 0; k < proj.size(); ++k) acc += cv::norm(proj[k] - und[k]);
    best_err = acc / static_cast<double>(proj.size());
  }

  pose.rvec = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
  pose.tvec = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
  pose.reproj_err_px = best_err;
  pose.ok = (best_err <= cfg_.max_reproj_err_px) && (pose.z() > 0.0);
  return pose;
}

}  // namespace autoaim
