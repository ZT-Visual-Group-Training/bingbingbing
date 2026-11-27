// Stage 4 (optional): constant-velocity Kalman filter on the plate position.
#pragma once

#include "autoaim/config.hpp"

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>

namespace autoaim {

/// Why a Kalman filter at 166 fps? Smoothing buys little when the frame interval
/// is 6 ms. The real value is latency compensation: the pose corresponds to the
/// exposure instant, and this extrapolates it to the moment of firing.
class ArmorTracker {
 public:
  explicit ArmorTracker(const DetectorConfig& cfg);

  void reset();
  bool initialized() const { return initialized_; }

  /// Feed a measurement (camera-frame position, metres) and get the filtered state.
  cv::Vec3d update(const cv::Vec3d& measurement);

  /// Advance and return the predicted position without a measurement (coasting).
  cv::Vec3d predictOnly();

  cv::Vec3d position() const { return pos_; }
  cv::Vec3d velocity() const { return vel_; }

  /// Search window around the predicted position, clamped to the image.
  cv::Rect roiAround(int pad_px, const cv::Size& frame_size, double fx, double cx,
                     double cy) const;

 private:
  void initFilter();

  DetectorConfig cfg_;
  cv::KalmanFilter kf_;
  bool initialized_ = false;
  cv::Vec3d pos_{0, 0, 0};
  cv::Vec3d vel_{0, 0, 0};
};

}  // namespace autoaim
