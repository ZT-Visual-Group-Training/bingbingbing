// probe3.cpp —— 装甲板局部放大与颜色统计（开发期）
#include "aegis/compat.hpp"

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

  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::Mat crop = frame(cv::Rect(150, 600, 500, 180));
  cv::Mat big;
  cv::resize(crop, big, cv::Size(), 3, 3, cv::INTER_NEAREST);
  cv::imwrite("probe_armor_zoom.png", big);

  // 统计两个灯条内的 HSV 分布
  for (int y = 640; y < 710; ++y) {
    for (int x = 270; x < 430; ++x) {
      const cv::Vec3b hsvp = hsv.at<cv::Vec3b>(y, x);
      const cv::Vec3b bgr = frame.at<cv::Vec3b>(y, x);
      if (hsvp[1] > 150 && hsvp[2] > 120) {
        if ((x > 270 && x < 292) || (x > 408 && x < 428)) {
          std::printf("bar px (%d,%d) H=%d S=%d V=%d BGR=%d,%d,%d\n", x, y, hsvp[0], hsvp[1],
                      hsvp[2], bgr[0], bgr[1], bgr[2]);
        }
      }
    }
  }
  // 数字区域统计（灰度/HSV）
  cv::Mat numberRoi = hsv(cv::Rect(300, 645, 90, 65));
  cv::Scalar m = cv::mean(numberRoi);
  std::printf("number ROI mean H=%.1f S=%.1f V=%.1f\n", m[0], m[1], m[2]);
  double mn, mx;
  cv::minMaxLoc(numberRoi, &mn, &mx);
  std::printf("number ROI V range %.1f..%.1f\n", mn, mx);

  // 整幅图的 V 直方图分位数
  cv::Mat v;
  cv::extractChannel(hsv, v, 2);
  cv::Mat hist;
  int histSize = 256;
  float range[] = {0, 256};
  const float* ranges[] = {range};
  cv::calcHist(&v, 1, nullptr, cv::noArray(), hist, 1, &histSize, ranges);
  long total = v.total();
  long acc = 0;
  for (int i = 0; i < 256; ++i) {
    acc += static_cast<long>(hist.at<float>(i));
    if (acc > total * 0.99) {
      std::printf("p99 V = %d\n", i);
      break;
    }
  }
  return 0;
}
