// =============================================================================
// pipeline.hpp —— 把检测、数字识别、跟踪、PnP 串成一条流水线
// 每个阶段单独计时，性能面板用的是真实测量值，不是估算。
// =============================================================================
#pragma once

#include "aegis/adaptive_detector.hpp"
#include "aegis/armor_tracker.hpp"
#include "aegis/config.hpp"
#include "aegis/digit_recognizer.hpp"
#include "aegis/pnp_solver.hpp"

#include <string>

namespace aegis {

class Pipeline {
 public:
  explicit Pipeline(const AppConfig& config);

  FrameResult process(const cv::Mat& frame, int frameIndex, double timestampMs);

  const AdaptiveDetector& detector() const { return detector_; }
  const ArmorTracker& tracker() const { return tracker_; }
  const AppConfig& config() const { return config_; }

  // 最近一帧的检测耗时（供 HUD 画曲线）。
  double lastDetectMs() const { return lastDetectMs_; }

 private:
  AppConfig config_;
  AdaptiveDetector detector_;
  ArmorTracker tracker_;
  PnPSolver pnp_;
  DigitRecognizer digits_;
  cv::Rect2f lastPrimaryBox_;
  double lastDetectMs_ = 0.0;
};

}  // namespace aegis
