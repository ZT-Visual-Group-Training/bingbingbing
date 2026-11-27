// Stage 3: 2D corners -> 3D pose.
#pragma once

#include "autoaim/config.hpp"
#include "autoaim/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace autoaim {

class PnpSolver {
 public:
  PnpSolver(const CameraConfig& cam, const DetectorConfig& cfg);

  /// Solve the plate pose. The four corners are undistorted first and PnP runs
  /// with no distortion model, so the result is consistent with projectPoints on
  /// the undistorted image.
  Pose solve(const Armor& armor) const;

  /// Object points, exposed for the report / tests: (+-67.5, +-28, 0) mm with
  /// Y pointing DOWN to match the image v axis.
  const std::vector<cv::Point3f>& objectPoints() const { return obj_; }

 private:
  CameraConfig cam_;
  DetectorConfig cfg_;
  std::vector<cv::Point3f> obj_;
  cv::Mat K_;
  cv::Mat D_;
};

}  // namespace autoaim
