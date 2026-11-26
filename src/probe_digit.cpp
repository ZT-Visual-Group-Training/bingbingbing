// probe_digit.cpp —— 观察数字 ROI 的标准图（开发期）
#include "aegis/compat.hpp"
#include "aegis/digit_recognizer.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "video.avi";
  const int idx = argc > 2 ? std::atoi(argv[2]) : 800;
  cv::VideoCapture cap(path);
  cap.set(cv::CAP_PROP_POS_FRAMES, idx);
  cv::Mat frame;
  cap.read(frame);
  if (frame.empty()) return 1;

  aegis::Armor armor;
  armor.corners = {cv::Point2f(280, 648), cv::Point2f(416, 648), cv::Point2f(416, 703),
                   cv::Point2f(280, 703)};
  cv::Mat warp = aegis::DigitRecognizer::warpArmor(frame, armor, cv::Size(160, 64));
  cv::Mat big;
  cv::resize(warp, big, cv::Size(), 5, 5, cv::INTER_NEAREST);
  cv::imwrite("warp.png", big);

  cv::Mat inner = warp(cv::Rect(20, 2, 120, 60));
  cv::Mat gray;
  cv::cvtColor(inner, gray, cv::COLOR_BGR2GRAY);
  cv::Mat eq;
  cv::normalize(gray, eq, 0, 255, cv::NORM_MINMAX);
  cv::Mat bin;
  const double otsu = cv::threshold(eq, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::Mat binBig;
  cv::resize(bin, binBig, cv::Size(), 6, 6, cv::INTER_NEAREST);
  cv::imwrite("warp_bin.png", binBig);
  cv::Mat eqBig;
  cv::resize(eq, eqBig, cv::Size(), 6, 6, cv::INTER_NEAREST);
  cv::imwrite("warp_eq.png", eqBig);

  double mn, mx;
  cv::minMaxLoc(gray, &mn, &mx);
  std::printf("inner gray range %.0f..%.0f otsu=%.0f\n", mn, mx, otsu);
  return 0;
}

