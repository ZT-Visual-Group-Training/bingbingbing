// probe.cpp —— 数据勘察工具（不属于最终系统，仅用于开发期分析）
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "video.avi";
  cv::VideoCapture cap(path);
  if (!cap.isOpened()) {
    std::printf("cannot open %s\n", path.c_str());
    return 1;
  }
  const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  const double fps = cap.get(cv::CAP_PROP_FPS);
  const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  std::printf("video=%s frames=%d fps=%.3f size=%dx%d\n", path.c_str(), total, fps, w, h);

  const int cols = 5, rows = 4;
  const int tw = 360, th = 270;
  cv::Mat sheet(rows * th, cols * tw, CV_8UC3, cv::Scalar(20, 20, 20));

  cv::Mat frame, hsv;
  for (int i = 0; i < cols * rows; ++i) {
    const int idx = total > 0 ? i * total / (cols * rows) : i * 50;
    cap.set(cv::CAP_PROP_POS_FRAMES, idx);
    if (!cap.read(frame) || frame.empty()) break;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    const double mv = cv::mean(ch[2])[0];
    const double sv = cv::mean(ch[1])[0];

    // 红色像素：H 跨 0 度，需要两段
    cv::Mat m1, m2, red;
    cv::inRange(hsv, cv::Scalar(0, 120, 90), cv::Scalar(10, 255, 255), m1);
    cv::inRange(hsv, cv::Scalar(170, 120, 90), cv::Scalar(180, 255, 255), m2);
    cv::bitwise_or(m1, m2, red);
    const int redPx = cv::countNonZero(red);

    // 蓝色
    cv::Mat blue;
    cv::inRange(hsv, cv::Scalar(100, 120, 90), cv::Scalar(130, 255, 255), blue);
    const int bluePx = cv::countNonZero(blue);

    std::printf("f%04d meanV=%.1f meanS=%.1f redPx=%d bluePx=%d\n", idx, mv, sv, redPx, bluePx);

    cv::Mat small;
    cv::resize(frame, small, cv::Size(tw, th));
    cv::putText(small, cv::format("f%d V%.0f R%d", idx, mv, redPx), cv::Point(8, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    small.copyTo(sheet(cv::Rect((i % cols) * tw, (i / cols) * th, tw, th)));
  }
  cv::imwrite("probe_sheet.jpg", sheet);
  return 0;
}

