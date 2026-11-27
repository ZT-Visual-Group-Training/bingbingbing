#include "viz/overlay.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace rm {

namespace {

const cv::Scalar kBarColor(0, 255, 255);    // 黄：灯条外接矩形
const cv::Scalar kArmorColor(0, 255, 0);    // 绿：P1..P4 装甲板四边形
const cv::Scalar kCenterColor(0, 0, 255);   // 红：三维中心投影
const cv::Scalar kTextOutline(0, 0, 0);

}  // namespace

void drawArmors(cv::Mat& image, const std::vector<ArmorObservation>& observations) {
  if (image.empty()) return;

  const double scale = std::max(0.5, image.cols / 1440.0);
  const int bar_thickness = std::max(1, cvRound(scale));
  const int armor_thickness = std::max(2, cvRound(2.0 * scale));
  const int dot_radius = std::max(3, cvRound(4.0 * scale));

  for (const ArmorObservation& observation : observations) {
    const Armor& armor = observation.armor;
    const cv::Rect& r = armor.rect;

    cv::rectangle(image, armor.left.rect, kBarColor, bar_thickness, cv::LINE_AA);
    cv::rectangle(image, armor.right.rect, kBarColor, bar_thickness, cv::LINE_AA);

    // 大框：P1 左上 -> P2 右上 -> P3 右下 -> P4 左下（顺时针）
    std::vector<cv::Point> quad;
    quad.reserve(armor.corners.size());
    for (const auto& corner : armor.corners) {
      quad.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }
    cv::polylines(image, quad, true, kArmorColor, armor_thickness, cv::LINE_AA);
    for (const auto& corner : quad) {
      cv::circle(image, corner, std::max(2, cvRound(2.0 * scale)), kArmorColor, cv::FILLED,
                 cv::LINE_AA);
    }

    // 击打点：PnP 三维中心投影
    const cv::Point center(cvRound(observation.pose.projected_center.x),
                           cvRound(observation.pose.projected_center.y));
    cv::circle(image, center, dot_radius + 1, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
    cv::circle(image, center, dot_radius, kCenterColor, cv::FILLED, cv::LINE_AA);

    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double label_scale = 0.75 * scale;
    const int label_thickness = std::max(1, cvRound(2.0 * scale));
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize("P1", font, label_scale, label_thickness, &baseline);
    const int gap = std::max(2, cvRound(4.0 * scale));

    auto put_label = [&](const std::string& text, cv::Point org, const cv::Scalar& color,
                         double font_scale, const cv::Size& size) {
      org.x = std::clamp(org.x, 1, std::max(1, image.cols - size.width - 1));
      org.y = std::clamp(org.y, size.height + 1, image.rows - 1);
      cv::putText(image, text, org, font, font_scale, kTextOutline, label_thickness + 2,
                  cv::LINE_AA);
      cv::putText(image, text, org, font, font_scale, color, label_thickness, cv::LINE_AA);
    };

    const bool labels_above = r.y - gap >= text_size.height + 1;
    const bool labels_below = r.y + r.height + text_size.height + gap <= image.rows - 1;
    const int top_y = labels_above ? r.y - gap : r.y + text_size.height + gap;
    const int bottom_y =
        labels_below ? r.y + r.height + text_size.height + gap : r.y + r.height - gap;

    put_label("P1", cv::Point(quad[0].x + gap, top_y), kArmorColor, label_scale, text_size);
    put_label("P2", cv::Point(quad[1].x - text_size.width - gap, top_y), kArmorColor, label_scale,
              text_size);
    put_label("P3", cv::Point(quad[2].x - text_size.width - gap, bottom_y), kArmorColor,
              label_scale, text_size);
    put_label("P4", cv::Point(quad[3].x + gap, bottom_y), kArmorColor, label_scale, text_size);

    char depth_text[32];
    std::snprintf(depth_text, sizeof(depth_text), "z=%.2fm", observation.pose.distance_m);
    const double depth_scale = 0.55 * scale;
    int depth_baseline = 0;
    const cv::Size depth_size =
        cv::getTextSize(depth_text, font, depth_scale, label_thickness, &depth_baseline);
    const int depth_y =
        labels_below ? std::min(image.rows - 1, bottom_y + depth_size.height + gap) : bottom_y;
    put_label(depth_text, cv::Point(quad[3].x + gap, depth_y), kBarColor, depth_scale, depth_size);
  }
}

void drawHud(cv::Mat& image, const std::vector<std::string>& lines, double scale) {
  if (image.empty() || lines.empty()) return;

  const int font = cv::FONT_HERSHEY_SIMPLEX;
  const double font_scale = 0.62 * scale;
  const int thickness = std::max(1, static_cast<int>(std::lround(1.6 * scale)));
  const int line_h = static_cast<int>(std::lround(30 * scale));
  const int pad = static_cast<int>(std::lround(12 * scale));

  int max_w = 0;
  for (const auto& line : lines) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(line, font, font_scale, thickness, &baseline);
    max_w = std::max(max_w, size.width);
  }

  const cv::Rect panel(0, 0, std::min(image.cols, max_w + 2 * pad),
                       std::min(image.rows, line_h * static_cast<int>(lines.size()) + 2 * pad));
  cv::Mat roi = image(panel);
  cv::Mat dark = roi * 0.35;  // 压暗背景，保证白字可读
  dark.copyTo(roi);

  for (std::size_t i = 0; i < lines.size(); ++i) {
    const cv::Point org(pad, pad + line_h * static_cast<int>(i) +
                                 static_cast<int>(std::lround(0.72 * line_h)));
    cv::putText(image, lines[i], org, font, font_scale, kTextOutline, thickness + 2, cv::LINE_AA);
    cv::putText(image, lines[i], org, font, font_scale, cv::Scalar(255, 255, 255), thickness,
                cv::LINE_AA);
  }
}

}  // namespace rm
