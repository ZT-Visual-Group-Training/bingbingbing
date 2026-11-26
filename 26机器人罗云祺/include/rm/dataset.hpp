#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>

#include "rm/types.hpp"

namespace rm
{

struct FrameMeta
{
  int index = 0;
  int64_t monotonic_ns = 0;
  double elapsed_s = 0.0;
};

// 打开视频和帧表
class Dataset
{
public:
  bool open(const std::string & video_path, const std::string & frames_csv, std::string * error = nullptr);
  // 读下一帧
  bool next(Frame & frame);

  size_t size() const { return meta_.size(); }
  double fps() const { return fps_; }
  cv::Size imageSize() const { return size_; }
  const std::vector<FrameMeta> & meta() const { return meta_; }

private:
  cv::VideoCapture cap_;
  std::vector<FrameMeta> meta_;  //整张 frames.csv 预先读进内存，按顺序
  size_t cursor_ = 0;
  double fps_ = 0.0;
  cv::Size size_{0, 0};
};

}
