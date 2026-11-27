#include "autoaim/tracker.hpp"

#include <algorithm>
#include <cmath>

namespace autoaim {

ArmorTracker::ArmorTracker(const DetectorConfig& cfg) : cfg_(cfg), kf_(6, 3, 0) {
  initFilter();
}

// ===========================================================================
// 卡尔曼滤波参数怎么调（答辩会问）
//
// 状态 [x y z vx vy vz]（相机系位置 + 速度），观测 [x y z]（PnP 输出）。
// 帧间 dt 取 1（每帧更新一次），所以速度的单位是"米/帧"。
//
// 两个噪声参数是一对跷跷板：
//
//   tracker_process_noise  Q（过程噪声，默认 1e-3）
//     含义：我有多不相信"匀速"这个假设。
//     Q 调大 → 滤波器认为目标可能随时变向 → 更贴观测 → **响应快、但抖**
//     Q 调小 → 更相信匀速 → 平滑、**滞后**
//     现象判断：目标急停急转时坐标"追不上"（滞后）→ 调大 Q；
//               静止时坐标来回小幅跳（抖）→ 调小 Q。
//
//   tracker_measure_noise  R（测量噪声，默认 1e-2）
//     含义：我有多不相信 PnP 的测量值。
//     R 调大 → 更信预测 → 更平滑、**滞后加重**
//     R 调小 → 更信观测 → 等于没滤波
//     现象判断：遮挡导致单帧跳变时坐标被带飞 → 调大 R；
//               检测很稳但滤波后明显迟钝 → 调小 R。
//
// 本工程实测：166 fps 下帧间隔只有 6 ms，匀速假设几乎总成立，
// 所以平滑的收益很小（噪声已经被高帧率平均掉了）。卡尔曼在这里的真正价值
// 是**延迟补偿**：把位姿外推到"击发时刻"，而不是停在曝光时刻。
// 这就是为什么它默认关闭 —— 不开也不影响精度，开了是为了预测。
//
// 另注：ROI 搜索窗用的也是这里的预测位置（见 main.cpp），所以即使不开
// --tracker，只要开 --roi，滤波器仍会在后台推进以提供预测点。
// ===========================================================================

void ArmorTracker::initFilter() {
  // State [x y z vx vy vz], measurement [x y z], dt = 1 frame.
  kf_.transitionMatrix = cv::Mat::eye(6, 6, CV_32F);
  for (int i = 0; i < 3; ++i) kf_.transitionMatrix.at<float>(i, i + 3) = 1.f;

  cv::setIdentity(kf_.measurementMatrix, cv::Scalar(1));
  cv::setIdentity(kf_.processNoiseCov, cv::Scalar(static_cast<float>(cfg_.tracker_process_noise)));
  cv::setIdentity(kf_.measurementNoiseCov,
                  cv::Scalar(static_cast<float>(cfg_.tracker_measure_noise)));
  cv::setIdentity(kf_.errorCovPost, cv::Scalar(1));
  kf_.statePost.setTo(0);
}

void ArmorTracker::reset() {
  initialized_ = false;
  pos_ = cv::Vec3d(0, 0, 0);
  vel_ = cv::Vec3d(0, 0, 0);
  initFilter();
}

cv::Vec3d ArmorTracker::update(const cv::Vec3d& measurement) {
  cv::Mat z(3, 1, CV_32F);
  for (int i = 0; i < 3; ++i) z.at<float>(i) = static_cast<float>(measurement[i]);

  if (!initialized_) {
    // Seed position, zero velocity.
    kf_.statePost.setTo(0.f);
    for (int i = 0; i < 3; ++i) kf_.statePost.at<float>(i) = z.at<float>(i);
    initialized_ = true;
    pos_ = measurement;
    vel_ = cv::Vec3d(0, 0, 0);
    return pos_;
  }

  kf_.predict();
  const cv::Mat x = kf_.correct(z);
  for (int i = 0; i < 3; ++i) {
    pos_[i] = x.at<float>(i);
    vel_[i] = x.at<float>(i + 3);
  }
  return pos_;
}

cv::Vec3d ArmorTracker::predictOnly() {
  if (!initialized_) return pos_;
  const cv::Mat x = kf_.predict();
  for (int i = 0; i < 3; ++i) {
    pos_[i] = x.at<float>(i);
    vel_[i] = x.at<float>(i + 3);
  }
  return pos_;
}

cv::Rect ArmorTracker::roiAround(int pad_px, const cv::Size& frame_size, double fx, double cx,
                                 double cy) const {
  if (!initialized_ || pos_[2] <= 0.1) return cv::Rect();
  // Project the tracked centre back to pixels, then pad.
  const int px = static_cast<int>(std::lround(fx * pos_[0] / pos_[2] + cx));
  const int py = static_cast<int>(std::lround(fx * pos_[1] / pos_[2] + cy));
  const int rad = std::max(pad_px, 16);
  const cv::Rect r(px - rad, py - rad, 2 * rad, 2 * rad);
  return r & cv::Rect(0, 0, frame_size.width, frame_size.height);
}

}  // namespace autoaim
