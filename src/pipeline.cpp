// =============================================================================
// pipeline.cpp —— 检测 -> 跟踪 -> PnP 的主流程
// =============================================================================
#include "aegis/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace aegis {
namespace {

using Clock = std::chrono::steady_clock;

double msSince(const Clock::time_point& t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

cv::Rect expandBox(const cv::Rect2f& box, float scale, const cv::Size& size) {
  const cv::Point2f c(box.x + box.width * 0.5F, box.y + box.height * 0.5F);
  const int w = static_cast<int>(box.width * scale);
  const int h = static_cast<int>(box.height * scale);
  int x = static_cast<int>(c.x) - w / 2;
  int y = static_cast<int>(c.y) - h / 2;
  x = std::max(0, x);
  y = std::max(0, y);
  const int x2 = std::min(size.width, x + w);
  const int y2 = std::min(size.height, y + h);
  return cv::Rect(x, y, std::max(0, x2 - x), std::max(0, y2 - y));
}

}  // namespace

Pipeline::Pipeline(const AppConfig& config)
    : config_(config),
      detector_(config.camera, config.detector),
      tracker_(config.tracker),
      pnp_(config.camera, config.armor) {
  lastPrimaryBox_ = cv::Rect2f();
}

FrameResult Pipeline::process(const cv::Mat& frame, int frameIndex, double timestampMs) {
  FrameResult result;
  const auto tAll = Clock::now();

  // 前 30 帧先只跑严格档，避免片头噪声生成假轨迹；
  // 之后允许“严格档为空时放宽几何”，用于大偏航目标重捕。
  const bool allowRelaxed = frameIndex > 30;

  // ---- 1. 检测：优先在上帧主目标附近搜索（ROI），失败则回退全图 ----
  cv::Rect roi;
  bool roiUsed = false;
  if (config_.enableRoi && lastPrimaryBox_.area() > 1.0F) {
    roi = expandBox(lastPrimaryBox_, 2.4F, frame.size());
    roiUsed = roi.area() > 400;
  }

  auto t0 = Clock::now();
  std::vector<Armor> armors = detector_.detect(frame, roi, -1, allowRelaxed);
  // ROI 内没找到目标时，本帧立刻回退全图再找一次，避免“一丢就再也找不回来”。
  if (armors.empty() && roiUsed) {
    armors = detector_.detect(frame, cv::Rect(), -1, allowRelaxed);
    roiUsed = false;
  }
  const ChannelStats stats = detector_.stats();
  result.ambient = stats.ambientV;
  result.thresholdLevel = detector_.lastLevel();
  result.usedRoi = roiUsed;
  result.relaxedGeometry = detector_.lastRelaxedGeometry();
  result.sThreshold = static_cast<float>(stats.sThreshold);
  result.vThreshold = static_cast<float>(stats.vThreshold);
  result.redBars = stats.redBars;
  result.blueBars = stats.blueBars;

  // ---- 2. 数字识别（可选）：给每块装甲板填 number ----
  for (auto& armor : armors) {
    if (config_.enableDigits) {
      float digitConf = 0.0F;
      armor.number = digits_.recognize(frame, armor, digitConf);
      armor.numberConfidence = digitConf;
    }
  }
  result.timing.detect = msSince(t0);
  lastDetectMs_ = result.timing.detect;

  // ---- 3. 跟踪 ----
  t0 = Clock::now();
  std::vector<Track> tracks = tracker_.update(armors, timestampMs);
  result.timing.track = msSince(t0);

  // ---- 4. 主目标 PnP ----
  t0 = Clock::now();
  const int primary = tracker_.primaryId();
  result.primaryTrack = primary;
  if (primary >= 0) {
    const Track* track = tracker_.find(primary);
    if (track != nullptr) {
      lastPrimaryBox_ = track->box;
      // 找离主轨迹最近的检测框，用它做 PnP（而不是用跟踪框，避免误差累积）。
      const Armor* best = nullptr;
      float bestDist = 1e9F;
      for (const auto& armor : armors) {
        const float dx = armor.center.x - (track->box.x + track->box.width * 0.5F);
        const float dy = armor.center.y - (track->box.y + track->box.height * 0.5F);
        const float d = dx * dx + dy * dy;
        if (d < bestDist) {
          bestDist = d;
          best = &armor;
        }
      }
      if (best != nullptr && bestDist < 90.0F * 90.0F) {
        const Pose pose = pnp_.solve(*best);
        tracker_.updatePose(primary, pose);
        // HUD 上的数字缩略图：把装甲板四点校正成 160×64 的小图。
        if (pose.valid) result.armorChip = DigitRecognizer::warpArmor(frame, *best, cv::Size(160, 64));
      }
    }
  }
  result.timing.pose = msSince(t0);

  result.armors = std::move(armors);
  result.tracks = std::move(tracks);
  result.timing.total = msSince(tAll);
  return result;
}

}  // namespace aegis
