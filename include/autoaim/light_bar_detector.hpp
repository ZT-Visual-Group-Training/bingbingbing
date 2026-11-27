// Stage 1: find light bars in a BGR frame.
#pragma once

#include "autoaim/config.hpp"
#include "autoaim/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace autoaim {

class LightBarDetector {
 public:
  explicit LightBarDetector(const DetectorConfig& cfg) : cfg_(cfg) {}

  /// Detect light bars. When `roi` is non-empty only that region is searched and
  /// the returned centres are translated back to full-frame coordinates.
  std::vector<LightBar> detect(const cv::Mat& bgr, const cv::Rect& roi = cv::Rect());

  /// Intermediate images, for the debug view and for the report figures.
  const cv::Mat& diffImage() const { return diff_; }
  const cv::Mat& binaryImage() const { return binary_; }

  /// The ROI used by the last detect() call. Corner sub-pixel refinement runs on
  /// diffImage(), so callers must shift full-frame corners by -lastRoi().tl first.
  const cv::Rect& lastRoi() const { return last_roi_; }

  const DetectorConfig& config() const { return cfg_; }

 private:
  /// Chrominance operator: the channel whose difference separates the enemy
  /// colour from neutral (white/grey) light.
  void computeDiff(const cv::Mat& bgr);

  DetectorConfig cfg_;
  cv::Mat diff_;
  cv::Mat binary_;
  cv::Rect last_roi_{0, 0, 0, 0};
};

}  // namespace autoaim
