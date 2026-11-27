#include "geometry/pnp.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace rm {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfWidthM = 0.0675;   // 135 mm / 2
constexpr double kHalfHeightM = 0.028;   // 56 mm / 2
constexpr double kRadToDeg = 180.0 / kPi;

}  // namespace

ArmorPose solveArmorPose(const Armor& armor, const cv::Mat& camera_matrix,
                         const cv::Mat& distort_coeffs, double max_reproj_error_px) {
  ArmorPose pose;
  if (camera_matrix.empty() || armor.rect.width <= 0 || armor.rect.height <= 0) return pose;

  // P1 左上、P2 右上、P3 右下、P4 左下 对应板面系的 (+/-x, +/-y)，y 轴向上
  const std::vector<cv::Point3f> object_points = {
      {-kHalfWidthM, +kHalfHeightM, 0.0f},
      {+kHalfWidthM, +kHalfHeightM, 0.0f},
      {+kHalfWidthM, -kHalfHeightM, 0.0f},
      {-kHalfWidthM, -kHalfHeightM, 0.0f},
  };
  const std::vector<cv::Point2f> image_points(armor.corners.begin(), armor.corners.end());

  cv::Mat rvec;
  cv::Mat tvec;
  // 平面目标 IPPE 有两个解（镜像歧义）。竖直安装的装甲板在图像里横边应接近水平，
  // 因此选“板面 x 轴在图像平面内倾角最小”的解。
  std::vector<cv::Mat> rvec_candidates;
  std::vector<cv::Mat> tvec_candidates;
  const int solutions =
      cv::solvePnPGeneric(object_points, image_points, camera_matrix, distort_coeffs,
                          rvec_candidates, tvec_candidates, false, cv::SOLVEPNP_IPPE);
  if (solutions <= 0) return pose;

  int best = -1;
  double best_roll_deg = 1e9;
  for (int i = 0; i < solutions; ++i) {
    if (tvec_candidates[i].at<double>(2) <= 0.0) continue;
    cv::Mat rotation_i;
    cv::Rodrigues(rvec_candidates[i], rotation_i);
    const double roll_deg =
        std::atan2(rotation_i.at<double>(1, 0), rotation_i.at<double>(0, 0)) * kRadToDeg;
    if (std::abs(roll_deg) < best_roll_deg) {
      best_roll_deg = std::abs(roll_deg);
      best = i;
    }
  }
  if (best < 0) return pose;
  rvec = rvec_candidates[best].clone();
  tvec = tvec_candidates[best].clone();

  cv::solvePnPRefineLM(object_points, image_points, camera_matrix, distort_coeffs, rvec, tvec);

  std::vector<cv::Point2f> reprojected;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, distort_coeffs, reprojected);
  double error_sum = 0.0;
  for (std::size_t i = 0; i < reprojected.size(); ++i) {
    error_sum += cv::norm(reprojected[i] - image_points[i]);
  }
  pose.reproj_error_px = error_sum / static_cast<double>(reprojected.size());
  if (pose.reproj_error_px > max_reproj_error_px) return pose;

  const std::vector<cv::Point3f> center_point = {{0.0f, 0.0f, 0.0f}};
  std::vector<cv::Point2f> center_projected;
  cv::projectPoints(center_point, rvec, tvec, camera_matrix, distort_coeffs, center_projected);

  pose.rvec = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
  pose.tvec = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
  pose.distance_m = cv::norm(pose.tvec);
  pose.projected_center = center_projected.front();
  if (pose.tvec[2] <= 0.0) return pose;

  // 板面法线 = 旋转矩阵第三列；统一翻到朝向相机的一侧（相机 z 轴向前）
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  cv::Vec3d normal(rotation.at<double>(0, 2), rotation.at<double>(1, 2),
                   rotation.at<double>(2, 2));
  if (normal[2] > 0.0) normal = -normal;
  pose.yaw_deg = std::atan2(normal[0], -normal[2]) * kRadToDeg;
  pose.pitch_deg = std::asin(std::clamp(normal[1], -1.0, 1.0)) * kRadToDeg;
  pose.valid = true;

  return pose;
}

}  // namespace rm
