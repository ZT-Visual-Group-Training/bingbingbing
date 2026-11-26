#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "rm/config.hpp"
#include "rm/types.hpp"

namespace rm
{

class Visualizer
{
public:
  explicit Visualizer(const Config & cfg);

  // 接口 / 作画(HUD、四角、中心、位姿文字)
  void draw(
    cv::Mat & bgr, const std::vector<Armor> & armors, const std::vector<Pose> & poses,  // poses[i] 对应 armors[i]，顺序必须一致
    double frame_ms, int frame_index, double timestamp_s) const;

  // 预留开关
    private:
  bool show_pose_ = true;
};
