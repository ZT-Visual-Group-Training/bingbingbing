#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "common/types.hpp"

namespace rm {

// 画装甲板：两条灯条外接框 + P1..P4 四边形 + 三维中心投影点 + 距离
void drawArmors(cv::Mat& image, const std::vector<ArmorObservation>& observations);

// 左上角半透明信息板
void drawHud(cv::Mat& image, const std::vector<std::string>& lines, double scale);

}  // namespace rm
