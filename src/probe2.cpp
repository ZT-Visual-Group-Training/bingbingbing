// probe2.cpp —— 灯条候选勘察（开发期）
#include "aegis/compat.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "video.avi";
  const int frameIdx = argc > 2 ? std::atoi(argv[2]) : 800;
  cv::VideoCapture cap(path);
  cv::Mat frame;
  cap.set(cv::CAP_PROP_POS_FRAMES, frameIdx);
  cap.read(frame);
  if (frame.empty()) return 1;

  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  std::printf("meanV=%.1f p99V=%.1f maxV=%.1f\n", cv::mean(ch[2])[0], 0.0,
              static_cast<double>(*std::max_element(ch[2].begin<uchar>(), ch[2].end<uchar>())));

  cv::Mat m1, m2, red;
  cv::inRange(hsv, cv::Scalar(0, 140, 100), cv::Scalar(12, 255, 255), m1);
  cv::inRange(hsv, cv::Scalar(165, 140, 100), cv::Scalar(180, 255, 255), m2);
  cv::bitwise_or(m1, m2, red);
  cv::morphologyEx(red, red, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5)));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(red.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::printf("contours=%zu\n", contours.size());

  struct C { cv::RotatedRect rr; float area; };
  std::vector<C> cs;
  for (auto& c : contours) {
    if (c.size() < 5) continue;
    const double a = cv::contourArea(c);
    if (a < 15) continue;
    cs.push_back({cv::minAreaRect(c), static_cast<float>(a)});
  }
  std::sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.area > b.area; });

  cv::Mat vis = frame.clone();
  for (size_t i = 0; i < cs.size() && i < 40; ++i) {
    const auto& c = cs[i];
    const float w = std::min(c.rr.size.width, c.rr.size.height);
    const float h = std::max(c.rr.size.width, c.rr.size.height);
    cv::Point2f pts[4];
    c.rr.points(pts);
    for (int k = 0; k < 4; ++k) cv::line(vis, pts[k], pts[(k + 1) % 4], cv::Scalar(0, 255, 0), 1);
    cv::putText(vis, cv::format("%zu %.0fx%.0f", i, w, h), c.rr.center,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    std::printf("#%zu c=(%.0f,%.0f) w=%.1f h=%.1f area=%.0f ratio=%.2f\n", i, c.rr.center.x,
                c.rr.center.y, w, h, c.area, h / std::max(1.0f, w));
  }
  cv::imwrite("probe_frame.png", vis);
  cv::imwrite("probe_mask.png", red);
  return 0;
}

