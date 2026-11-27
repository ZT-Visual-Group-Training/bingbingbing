#include "autoaim/types.hpp"

#include <opencv2/calib3d.hpp>

namespace autoaim {

void Pose::yawPitchDeg(double* yaw_deg, double* pitch_deg) const {
  cv::Matx33d R;
  cv::Rodrigues(cv::Mat(rvec), R);
  // Column 2 of R is the plate normal expressed in the camera frame.
  const double nx = R(0, 2), ny = R(1, 2), nz = R(2, 2);
  if (yaw_deg) *yaw_deg = std::atan2(nx, nz) * 180.0 / CV_PI;
  if (pitch_deg) {
    const double s = std::max(-1.0, std::min(1.0, -ny));
    *pitch_deg = std::asin(s) * 180.0 / CV_PI;
  }
}

int FrameResult::primaryIndex() const {
  if (armors.empty()) return -1;
  // 时序层已经选好了主目标就直接用它 —— 这是"跨帧一致性"覆盖"单帧最高分"的地方
  if (primary_override == -2) return -1;  // 时序层判定：本帧没有可输出的主目标
  if (primary_override >= 0 && static_cast<size_t>(primary_override) < armors.size()) {
    return primary_override;
  }
  // ArmourMatcher already sorts by ascending score, so the first valid pose wins.
  for (size_t i = 0; i < armors.size(); ++i) {
    if (i < poses.size() && poses[i].ok) return static_cast<int>(i);
  }
  return 0;
}

const Pose* FrameResult::primaryPose() const {
  const int i = primaryIndex();
  if (i < 0 || static_cast<size_t>(i) >= poses.size()) return nullptr;
  return &poses[i];
}

}  // namespace autoaim
