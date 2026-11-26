// =============================================================================
// armor_tracker.cpp —— 卡尔曼 + 匈牙利 + 生命周期
// =============================================================================
#include "aegis/armor_tracker.hpp"

#include "aegis/hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aegis {
namespace {

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float w = std::max(0.0F, x2 - x1);
  const float h = std::max(0.0F, y2 - y1);
  const float inter = w * h;
  const float uni = a.area() + b.area() - inter;
  return uni > 1e-6F ? inter / uni : 0.0F;
}

cv::Rect2f armorBox(const Armor& a) {
  return {a.center.x - a.width * 0.5F, a.center.y - a.height * 0.5F, a.width, a.height};
}

}  // namespace

ArmorTracker::ArmorTracker(const TrackerConfig& params) : params_(params) {}

void ArmorTracker::initKalman(Impl& impl, const Armor& det) const {
  // 状态：cx, cy, w, h, vx, vy, vw, vh
  impl.kf.init(8, 4, 0, CV_32F);
  cv::setIdentity(impl.kf.transitionMatrix);
  for (int i = 0; i < 4; ++i) impl.kf.transitionMatrix.at<float>(i, i + 4) = 1.0F;

  impl.kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
  cv::setIdentity(impl.kf.measurementMatrix);

  // 过程噪声：匀速模型，位置噪声小、速度噪声稍大（允许目标突然加速）。
  cv::setIdentity(impl.kf.processNoiseCov, cv::Scalar(params_.processNoise));
  // 观测噪声：检测框本身有 1~3px 抖动，这里按像素方差建模。
  cv::setIdentity(impl.kf.measurementNoiseCov, cv::Scalar(params_.measurementNoise));
  cv::setIdentity(impl.kf.errorCovPost, cv::Scalar(80.0));

  impl.kf.statePost.at<float>(0) = det.center.x;
  impl.kf.statePost.at<float>(1) = det.center.y;
  impl.kf.statePost.at<float>(2) = det.width;
  impl.kf.statePost.at<float>(3) = det.height;
  for (int i = 4; i < 8; ++i) impl.kf.statePost.at<float>(i) = 0.0F;
  impl.kfReady = true;
}

cv::Rect2f ArmorTracker::stateToBox(const cv::Mat& s) const {
  const float cx = s.at<float>(0);
  const float cy = s.at<float>(1);
  const float w = std::max(2.0F, s.at<float>(2));
  const float h = std::max(2.0F, s.at<float>(3));
  return {cx - w * 0.5F, cy - h * 0.5F, w, h};
}

std::vector<Track> ArmorTracker::update(const std::vector<Armor>& detections, double nowMs) {
  // ---- 1. 先让所有轨迹“预测”到当前帧 ----
  for (auto& impl : tracks_) {
    if (!impl.kfReady) continue;
    const cv::Mat pred = impl.kf.predict();
    impl.predicted = stateToBox(pred);
  }

  // ---- 2. 构造代价矩阵 ----
  const int n = static_cast<int>(tracks_.size());
  const int m = static_cast<int>(detections.size());
  const double kInf = 1e6;
  std::vector<std::vector<double>> cost(n, std::vector<double>(m, kInf));
  for (int i = 0; i < n; ++i) {
    const auto& tr = tracks_[i];
    for (int j = 0; j < m; ++j) {
      const auto& det = detections[j];
      const cv::Rect2f db = armorBox(det);
      const float ov = iou(tr.predicted, db);
      const float dx = tr.predicted.x + tr.predicted.width * 0.5F - det.center.x;
      const float dy = tr.predicted.y + tr.predicted.height * 0.5F - det.center.y;
      const float dist = std::sqrt(dx * dx + dy * dy);

      // 波门：距离太远直接认为不可能匹配（防止目标交叉时“抢 ID”）。
      if (ov < 0.02F && dist > params_.gateDistance) continue;

      double c = 1.0 - ov;
      c += 0.35 * std::min(1.0, static_cast<double>(dist) / std::max(1.0F, params_.gateDistance));
      // 尺寸一致性：宽高比变化大的候选要扣分。
      const float ratioA = det.width / std::max(1.0F, det.height);
      const float ratioB = tr.predicted.width / std::max(1.0F, tr.predicted.height);
      c += 0.25 * std::min(1.0F, std::abs(ratioA - ratioB) / 2.0F);
      if (det.color != tr.track.color) c += 0.6;
      if (tr.track.number >= 0 && det.number >= 0 && tr.track.number != det.number) c += 0.5;
      cost[i][j] = c;
    }
  }

  const std::vector<int> assign = hungarian(cost);
  std::vector<bool> detUsed(m, false);
  std::vector<int> matchedDet(n, -1);

  // ---- 3. 用检测更新匹配上的轨迹 ----
  for (int i = 0; i < n; ++i) {
    const int j = assign[i];
    if (j < 0 || j >= m || cost[i][j] > kInf * 0.5) continue;
    auto& impl = tracks_[i];
    const Armor& det = detections[j];
    detUsed[j] = true;
    matchedDet[i] = j;

    const cv::Mat z = (cv::Mat_<float>(4, 1) << det.center.x, det.center.y, det.width, det.height);
    const cv::Mat corrected = impl.kf.correct(z);
    impl.track.box = stateToBox(corrected);
    impl.track.velocity = {impl.kf.statePost.at<float>(4), impl.kf.statePost.at<float>(5)};
    impl.track.hits++;
    impl.track.misses = 0;
    impl.track.age++;
    impl.track.score = det.confidence;
    impl.track.color = det.color;
    // 确认的瞬间分配一个对外编号：内部 id 会因误检增长，展示用 id 从 1 连续编号。
    if (!impl.track.confirmed && impl.track.hits >= params_.minHitsToConfirm) {
      impl.track.confirmed = true;
      impl.track.displayId = nextDisplayId_++;
    }
    impl.track.lastUpdateMs = nowMs;

    // 多帧数字投票：同一块板子连续多帧识别到同一个数字才采信。
    if (det.number >= 0 && det.number < 10) {
      impl.track.numberConfidence = det.numberConfidence;
      impl.track.digitVotes[det.number]++;
      int best = 0;
      for (int d = 1; d < 10; ++d) {
        if (impl.track.digitVotes[d] > impl.track.digitVotes[best]) best = d;
      }
      if (impl.track.digitVotes[best] >= 2) impl.track.number = best;
    }

    impl.track.trail.push_back({impl.track.box.x + impl.track.box.width * 0.5F,
                                impl.track.box.y + impl.track.box.height * 0.5F});
    while (static_cast<int>(impl.track.trail.size()) > params_.maxTrail) impl.track.trail.pop_front();
  }

  // ---- 4. 未匹配的轨迹：预测外推（coast），连续丢失太多就删除 ----
  for (int i = 0; i < n; ++i) {
    if (matchedDet[i] >= 0) continue;
    auto& impl = tracks_[i];
    impl.track.misses++;
    impl.track.age++;
    impl.track.score *= 0.88F;
    impl.track.box = impl.predicted;
    impl.track.trail.push_back({impl.predicted.x + impl.predicted.width * 0.5F,
                                impl.predicted.y + impl.predicted.height * 0.5F});
    while (static_cast<int>(impl.track.trail.size()) > params_.maxTrail) impl.track.trail.pop_front();
  }

  const int maxMisses = params_.maxMisses;
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [maxMisses](const Impl& impl) {
                                 const int limit = impl.track.confirmed ? maxMisses : 2;
                                 return impl.track.misses > limit;
                               }),
                tracks_.end());

  // ---- 5. 新检测 -> 新轨迹 ----
  for (int j = 0; j < m; ++j) {
    if (detUsed[j]) continue;
    Impl impl;
    impl.track.id = nextId_++;
    impl.track.box = armorBox(detections[j]);
    impl.track.color = detections[j].color;
    impl.track.score = detections[j].confidence;
    impl.track.hits = 1;
    impl.track.age = 1;
    impl.track.number = detections[j].number;
    impl.track.lastUpdateMs = nowMs;
    impl.track.trail.push_back(detections[j].center);
    initKalman(impl, detections[j]);
    tracks_.push_back(std::move(impl));
  }

  // ---- 6. 选主目标：已确认、最近有更新、离画面中心最近 ----
  // 先只在“最近 2 帧有更新”的轨迹里挑；一个都没有时，保留旧主目标最多 8 帧，
  // 让 HUD 进入 COASTING 而不是瞬间丢失（自瞄目标记忆）。
  const cv::Point2f imageCenter(720.0F, 540.0F);
  double bestMetric = -1e9;
  int freshBest = -1;
  for (const auto& impl : tracks_) {
    if (!impl.track.confirmed || impl.track.misses > 2) continue;
    const float dx = impl.track.box.x + impl.track.box.width * 0.5F - imageCenter.x;
    const float dy = impl.track.box.y + impl.track.box.height * 0.5F - imageCenter.y;
    const double centerBias = 1.0 - std::min(1.0, std::sqrt(dx * dx + dy * dy) / 900.0);
    const double metric = impl.track.score + 0.35 * centerBias + 0.02 * impl.track.hits;
    if (metric > bestMetric) {
      bestMetric = metric;
      freshBest = impl.track.id;
    }
  }

  if (freshBest >= 0) {
    primaryId_ = freshBest;
  } else if (primaryId_ >= 0) {
    bool keep = false;
    for (const auto& impl : tracks_) {
      if (impl.track.id == primaryId_ && impl.track.confirmed && impl.track.misses <= 8) {
        keep = true;
        break;
      }
    }
    if (!keep) primaryId_ = -1;
  }

  std::vector<Track> out;
  out.reserve(tracks_.size());
  for (const auto& impl : tracks_) out.push_back(impl.track);
  return out;
}

void ArmorTracker::updatePose(int trackId, const Pose& pose) {
  for (auto& impl : tracks_) {
    if (impl.track.id == trackId) {
      impl.track.pose = pose;
      if (pose.valid) {
        impl.track.zHistory.push_back(static_cast<float>(pose.tvec[2]));
        while (impl.track.zHistory.size() > 90) impl.track.zHistory.pop_front();
      }
      return;
    }
  }
}

const Track* ArmorTracker::find(int id) const {
  for (const auto& impl : tracks_) {
    if (impl.track.id == id) return &impl.track;
  }
  return nullptr;
}

}  // namespace aegis

