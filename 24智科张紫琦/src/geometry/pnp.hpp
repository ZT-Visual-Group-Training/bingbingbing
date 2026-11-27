#pragma once

#include <opencv2/core.hpp>

#include "common/types.hpp"

namespace rm {

// 小装甲板外沿 135 mm x 56 mm，物点 (±67.5, ±28, 0) mm（板面系，y 向上）。
// 用 P1..P4 四角做 SOLVEPNP_IPPE + LM 精化；四角平均回投误差超过阈值则判为无效。
ArmorPose solveArmorPose(const Armor& armor, const cv::Mat& camera_matrix,
                         const cv::Mat& distort_coeffs, double max_reproj_error_px = 8.0);

}  // namespace rm
