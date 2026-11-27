#include "detect/color_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace rm {

namespace {

struct PairCandidate {
  std::size_t first = 0;
  std::size_t second = 0;
  float score = 0.0f;
};

// 两条对角线 c0-c2 与 c1-c3 的交点；透视下它是四边形中心投影的精确位置
cv::Point2f diagonalIntersection(const std::array<cv::Point2f, 4>& corners) {
  const cv::Point2f p = corners[0];
  const cv::Point2f r = corners[2] - corners[0];
  const cv::Point2f q = corners[1];
  const cv::Point2f s = corners[3] - corners[1];
  const float denom = r.x * s.y - r.y * s.x;
  if (std::abs(denom) < 1e-6f) {
    return (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
  }
  const float t = ((q - p).x * s.y - (q - p).y * s.x) / denom;
  return p + t * r;
}

}  // namespace

ColorDetector::ColorDetector() : ColorDetector(Params{}) {}

ColorDetector::ColorDetector(Params params) : params_(params) {}

std::vector<Armor> ColorDetector::detect(const cv::Mat& bgr) {
  last_bar_count_ = 0;
  std::vector<Armor> armors;
  if (bgr.empty() || bgr.type() != CV_8UC3) return armors;

  // 1. 颜色分割：红 = R - max(G,B)，蓝 = B - max(G,R)
  cv::Mat channels[3];
  cv::split(bgr, channels);
  cv::Mat color_score;
  if (params_.color == EnemyColor::kRed) {
    cv::subtract(channels[2], cv::max(channels[1], channels[0]), color_score);
  } else {
    cv::subtract(channels[0], cv::max(channels[1], channels[2]), color_score);
  }

  cv::Mat mask;
  cv::threshold(color_score, mask, params_.threshold, 255, cv::THRESH_BINARY);
  if (params_.close_size > 1) {
    // 灯条中心过曝为白色，颜色掩码只剩外圈光晕；闭运算把同一灯条的缺口连起来
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(params_.close_size, params_.close_size));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  }

  // 2. 灯条候选：竖直细长的外接矩形
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  const int max_bar_height = bgr.rows / 2;
  std::vector<LightBar> bars;
  bars.reserve(contours.size());
  for (const auto& contour : contours) {
    const cv::Rect rect = cv::boundingRect(contour);
    if (rect.width < 3 || rect.height < params_.min_bar_height) continue;
    if (rect.height > max_bar_height || rect.y < 4) continue;

    const float aspect = static_cast<float>(rect.height) / static_cast<float>(rect.width);
    if (aspect < params_.min_bar_aspect || aspect > params_.max_bar_aspect) continue;
    if (cv::contourArea(contour) < params_.min_bar_area) continue;

    bars.push_back({rect, cv::Point2f(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f)});
  }
  last_bar_count_ = static_cast<int>(bars.size());
  if (bars.size() < 2) return armors;

  // 3. 配对：用 135mm / 56mm 约束“同一块小装甲板”的宽度，超出即不组队
  const float expected_ratio = params_.armor_width_mm / params_.armor_height_mm;
  const float ratio_max = expected_ratio * params_.max_width_ratio;
  const float ratio_min = expected_ratio * params_.min_width_ratio;

  std::vector<PairCandidate> candidates;
  for (std::size_t i = 0; i < bars.size(); ++i) {
    for (std::size_t j = i + 1; j < bars.size(); ++j) {
      const LightBar* left = &bars[i];
      const LightBar* right = &bars[j];
      if (left->center.x > right->center.x) std::swap(left, right);

      const float h_left = static_cast<float>(left->rect.height);
      const float h_right = static_cast<float>(right->rect.height);
      const float h_min = std::min(h_left, h_right);
      const float h_max = std::max(h_left, h_right);
      if (h_min <= 0.0f || h_max / h_min > params_.max_height_ratio) continue;

      const float dy = std::abs(left->center.y - right->center.y);
      if (dy > params_.max_center_dy_ratio * h_min) continue;

      const cv::Rect box = left->rect | right->rect;
      // 用较高灯条作高度参考：转动/遮挡时灯条下段可能断亮，平均高度会偏小
      const float ratio = static_cast<float>(box.width) / h_max;
      if (ratio < ratio_min || ratio > ratio_max) continue;

      const int gap = right->rect.x - (left->rect.x + left->rect.width);
      const int widest = std::max(left->rect.width, right->rect.width);
      if (static_cast<float>(gap) < -params_.max_overlap_ratio * static_cast<float>(widest)) {
        continue;
      }

      const float score = std::abs(ratio - expected_ratio) / expected_ratio +
                          0.5f * (h_max - h_min) / h_max +
                          0.5f * dy / h_min;
      candidates.push_back({i, j, score});
    }
  }

  // 4. 每个灯条只属于一块装甲板：按配对分数贪心取最优
  std::sort(candidates.begin(), candidates.end(),
            [](const PairCandidate& a, const PairCandidate& b) { return a.score < b.score; });

  std::vector<bool> used(bars.size(), false);
  for (const auto& candidate : candidates) {
    if (used[candidate.first] || used[candidate.second]) continue;

    const LightBar* left = &bars[candidate.first];
    const LightBar* right = &bars[candidate.second];
    if (left->center.x > right->center.x) std::swap(left, right);

    // 大框四角取两条灯条各自的外沿端点：
    // 左灯条左上外角、右灯条右上外角、右灯条右下外角、左灯条左下外角。
    // 两条灯条的上下端点允许不同高度，从而保留透视梯形信息供 PnP 解 pitch/roll。
    //
    // 但灯条下段可能断亮（比另一条明显短）。若两条灯条只有一端对齐、另一端差很多，
    // 说明短的那端是断亮而非透视收缩，用较长灯条把它补齐，避免四角不成矩形。
    float left_top = static_cast<float>(left->rect.y);
    float left_bottom = static_cast<float>(left->rect.y + left->rect.height);
    float right_top = static_cast<float>(right->rect.y);
    float right_bottom = static_cast<float>(right->rect.y + right->rect.height);

    const float height_left = left_bottom - left_top;
    const float height_right = right_bottom - right_top;
    const float height_max = std::max(height_left, height_right);
    const float height_min = std::min(height_left, height_right);
    if (height_max > 0.0f && height_min / height_max < 0.85f) {
      const float tolerance = std::max(2.0f, 0.06f * height_max);
      const float dy_top = std::abs(left_top - right_top);
      const float dy_bottom = std::abs(left_bottom - right_bottom);
      if (dy_top <= dy_bottom && dy_top <= tolerance) {
        if (height_left < height_right) {
          left_bottom = right_bottom;
        } else {
          right_bottom = left_bottom;
        }
      } else if (dy_bottom < dy_top && dy_bottom <= tolerance) {
        if (height_left < height_right) {
          left_top = right_top;
        } else {
          right_top = left_top;
        }
      }
    }

    Armor armor;
    armor.left = *left;
    armor.right = *right;
    armor.corners = {
        cv::Point2f(static_cast<float>(left->rect.x), left_top),
        cv::Point2f(static_cast<float>(right->rect.x + right->rect.width), right_top),
        cv::Point2f(static_cast<float>(right->rect.x + right->rect.width), right_bottom),
        cv::Point2f(static_cast<float>(left->rect.x), left_bottom)};

    const std::vector<cv::Point> corner_points = {
        cv::Point(cvRound(armor.corners[0].x), cvRound(armor.corners[0].y)),
        cv::Point(cvRound(armor.corners[1].x), cvRound(armor.corners[1].y)),
        cv::Point(cvRound(armor.corners[2].x), cvRound(armor.corners[2].y)),
        cv::Point(cvRound(armor.corners[3].x), cvRound(armor.corners[3].y))};
    armor.rect = cv::boundingRect(corner_points);
    armor.center = diagonalIntersection(armor.corners);
    armor.color = params_.color;
    armors.push_back(armor);

    used[candidate.first] = true;
    used[candidate.second] = true;
  }

  std::sort(armors.begin(), armors.end(), [](const Armor& a, const Armor& b) {
    return a.rect.x < b.rect.x;
  });
  return armors;
}

}  // namespace rm
