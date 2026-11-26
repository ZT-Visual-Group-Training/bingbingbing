// =============================================================================
// adaptive_detector.hpp —— 特色一：红蓝双通道 + 亮度自适应装甲板检测
// =============================================================================
#pragma once

#include "aegis/config.hpp"
#include "aegis/types.hpp"

namespace aegis {

// 每帧的颜色统计，既用于自适应阈值，也用于 HUD 展示。
struct ChannelStats {
  double ambientV = 0.0;   // V 通道均值：整体环境亮度
  double p999V = 0.0;      // V 的 99.9% 分位：高亮光源（灯条）的亮度水平
  double brightS = 0.0;    // 高亮像素的平均饱和度：过曝时饱和度会下降
  double sThreshold = 0.0; // 本帧实际使用的 S 阈值
  double vThreshold = 0.0; // 本帧实际使用的 V 阈值
  int redBars = 0;         // 红色候选灯条数
  int blueBars = 0;        // 蓝色候选灯条数
};

class AdaptiveDetector {
 public:
  AdaptiveDetector(const CameraConfig& camera, const DetectorConfig& params);

  // roi 为空表示全图搜索；forcedLevel=-1 表示自动选择严格/放宽阈值。
  // allowRelaxed=false 时禁用“放宽档几何”，只在跟踪已锁定时使用，避免假目标抢主目标。
  std::vector<Armor> detect(const cv::Mat& bgr, const cv::Rect& roi, int forcedLevel = -1,
                            bool allowRelaxed = true);

  const ChannelStats& stats() const { return stats_; }
  int lastLevel() const { return level_; }
  bool lastRelaxedGeometry() const { return geometryRelaxed_; }

 private:
  // 由整帧统计量推导出 S/V 阈值，这一步就是“自适应”的核心。
  void updateAdaptiveThresholds(const cv::Mat& hsv);
  // 生成某个颜色的二值掩码（红跨 0 度，需要两段 hue 合并）。
  cv::Mat makeMask(const cv::Mat& hsv, Color color, int level) const;
  // 从掩码中提取灯条候选。maxAngle<=0 时使用 params_.maxBarAngle。
  std::vector<LightBar> findBars(cv::Mat mask, Color color, double maxAngle = -1.0) const;
  // 尝试把两条灯条配成一块装甲板；relaxed=true 时使用放宽档几何门限。
  bool buildArmor(const LightBar& a, const LightBar& b, Armor& armor, bool relaxed = false) const;
  // 面板亮度校验：真装甲板的面板比灯条暗，窗户/灯管夹出来的假目标不满足。
  bool plateLooksLikeArmor(const cv::Mat& hsv, const Armor& armor) const;
  // 亚像素角点细化：在灰度图上把 4 个角点吸附到真实边缘。
  void refineCorners(const cv::Mat& gray, Armor& armor) const;

  CameraConfig camera_;
  DetectorConfig params_;
  ChannelStats stats_;
  int level_ = 0;
  bool geometryRelaxed_ = false;
};

}  // namespace aegis

