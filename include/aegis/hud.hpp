// =============================================================================
// hud.hpp —— 特色四：检测性能分析面板 + 三维可视化
// =============================================================================
#pragma once

#include "aegis/config.hpp"
#include "aegis/types.hpp"

#include <deque>
#include <string>

namespace aegis {

// HUD 需要的历史统计（由 main 每帧更新后传进来）。
struct HudState {
  int frameIndex = 0;
  double fps = 0.0;
  double avgFps = 0.0;
  int processedFrames = 0;
  int detectedFrames = 0;
  int totalFrames = 0;
  int detectedAfterLock = 0;  // 首次锁定之后：检测成功帧数
  int framesAfterLock = 0;    // 首次锁定之后：总帧数
  double lostTimeMs = 0.0;
  double trackedTimeMs = 0.0;
  double avgLatencyMs = 0.0;
  bool baseline = false;
  bool intro = true;
  std::deque<float> detectHistory;   // 最近 N 帧检测耗时
  std::deque<float> fpsHistory;      // 最近 N 帧 FPS
  std::deque<float> successHistory;  // 0/1，检测成败，用于底部时间条
  std::deque<cv::Vec3d> poseTrail;   // 三维轨迹（相机系，最多 120 点）
};

class Hud {
 public:
  explicit Hud(const AppConfig& config);

  // 主绘制入口：检测框、角点、三维面板、性能面板、底部时间条。
  void draw(cv::Mat& img, const FrameResult& result, HudState& state);

  // 片头/片尾的标题卡（用于演示视频）。
  void drawIntro(cv::Mat& img, double progress) const;
  void drawOutro(cv::Mat& img, double progress, const HudState& state) const;

 private:
  AppConfig config_;
  cv::Point2f imageCenter_;
};

}  // namespace aegis
