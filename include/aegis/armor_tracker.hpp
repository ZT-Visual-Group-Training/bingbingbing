// =============================================================================
// armor_tracker.hpp —— 特色二：多帧目标稳定追踪
// 检测器给的是“每帧独立”的结果：会抖、会丢、会误检。
// 跟踪器负责把帧与帧之间的信息接起来：
//   1. 卡尔曼滤波平滑中心/尺寸，并预测下一帧位置；
//   2. 匈牙利算法做全局最优关联；
//   3. 轨迹生命周期（tentative -> confirmed -> lost -> removed）过滤单帧误检；
//   4. 多帧数字投票，避免某一帧数字识别错了就输出错。
// =============================================================================
#pragma once

#include "aegis/config.hpp"
#include "aegis/types.hpp"

#include <vector>

namespace aegis {

class ArmorTracker {
 public:
  explicit ArmorTracker(const TrackerConfig& params);

  // 输入本帧检测，返回更新后的全部轨迹（含 tentative）。
  std::vector<Track> update(const std::vector<Armor>& detections, double nowMs);

  // 管线算完 PnP 后回填位姿，下一帧 HUD 就能用。
  void updatePose(int trackId, const Pose& pose);

  int primaryId() const { return primaryId_; }
  const Track* find(int id) const;

 private:
  struct Impl {
    Track track;
    cv::KalmanFilter kf;
    bool kfReady = false;
    cv::Rect2f predicted;
  };

  void initKalman(Impl& impl, const Armor& det) const;
  cv::Rect2f stateToBox(const cv::Mat& state) const;

  TrackerConfig params_;
  std::vector<Impl> tracks_;
  int nextId_ = 1;
  int nextDisplayId_ = 1;
  int primaryId_ = -1;
};

}  // namespace aegis
