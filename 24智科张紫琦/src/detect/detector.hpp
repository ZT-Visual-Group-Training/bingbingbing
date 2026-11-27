#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "common/types.hpp"

namespace rm {

// 装甲板检测器接口：输入 BGR 图像，输出本帧检出的装甲板（像素域）。
class Detector {
 public:
  virtual ~Detector() = default;
  virtual std::vector<Armor> detect(const cv::Mat& bgr) = 0;
};

}  // namespace rm
