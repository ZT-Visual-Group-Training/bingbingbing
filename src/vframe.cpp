// vframe.cpp —— 从视频里抓一帧存成 PNG（开发/验收用）
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("用法: vframe <video> <index> <out.png>\n");
    return 1;
  }
  cv::VideoCapture cap(argv[1]);
  if (!cap.isOpened()) return 1;
  cap.set(cv::CAP_PROP_POS_FRAMES, std::atoi(argv[2]));
  cv::Mat frame;
  cap.read(frame);
  if (frame.empty()) return 1;
  cv::imwrite(argv[3], frame);
  std::printf("saved %s %dx%d\n", argv[3], frame.cols, frame.rows);
  return 0;
}
