#include "rm/visualizer.hpp"

#include <cstdio>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace rm
{

Visualizer::Visualizer(const Config & cfg) { (void)cfg; }

void Visualizer::draw(
  cv::Mat & bgr, const std::vector<Armor> & armors, const std::vector<Pose> & poses,
  double frame_ms, int frame_index, double timestamp_s) const
{
  // HUD
  char hud[160];
  // %04d：帧号补零；%.3f：秒保留三位；%.2f：毫秒两位
  std::snprintf(
    hud, sizeof(hud), "frame %04d   t=%.3fs   %.2f ms", frame_index, timestamp_s, frame_ms);
  // 铺黑底
    cv::rectangle(bgr, cv::Rect(10, 10, 470, 46), cv::Scalar(0, 0, 0), -1);  // -1 = 填充
  cv::putText(bgr, hud, cv::Point(20, 42), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);  // 绿色，线宽 2

  if (armors.empty()) {
    cv::putText(
      bgr, "no armor", cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);  // 红色
    return;
  }

  // 逐块装甲板
  for (size_t i = 0; i < armors.size(); ++i) {
    const Armor & a = armors[i];

    // BGR
    cv::Scalar color(0, 255, 0);
    if (a.color == Color::Red) color = cv::Scalar(0, 0, 255);
    if (a.color == Color::Blue) color = cv::Scalar(255, 0, 0);

    // 顺序 TL→TR→BR→BL
    for (int k = 0; k < 4; ++k) {
      cv::line(bgr, a.corners[static_cast<size_t>(k)], a.corners[static_cast<size_t>((k + 1) % 4)], color, 2);
    }
    // 中心点：黄色实心圆，半径 4
    cv::circle(bgr, a.center, 4, cv::Scalar(0, 255, 255), -1);

    char info[192];
    // number 未识别时显示 "?"
    const char * name = (a.number > 0) ? std::to_string(a.number).c_str() : "?";
    if (i < poses.size() && poses[i].valid) {
      const cv::Vec3d & p = poses[i].tvec;
      // 单位从米换成毫米显示，距离用 norm(tvec)
      std::snprintf(
        info, sizeof(info), "%s %s id=%s  (%.0f, %.0f, %.0f) mm  %.2f m  e=%.1fpx",
        colorName(a.color), sizeName(a.size), name, p[0] * 1000.0, p[1] * 1000.0, p[2] * 1000.0,
        cv::norm(p), poses[i].reproj_error);
    } else {
      std::snprintf(info, sizeof(info), "%s %s id=%s", colorName(a.color), sizeName(a.size), name);
    }

    // 左上角上方 10 px, 贴边时可能出界
    cv::putText(
      bgr, info, cv::Point(static_cast<int>(a.corners[0].x), static_cast<int>(a.corners[0].y) - 10),
      cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
  }
}

}
