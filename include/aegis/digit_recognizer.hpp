// =============================================================================
// digit_recognizer.hpp —— 装甲板数字识别（0~9）
// 思路：把装甲板四点透视变换到固定尺寸 -> 提取数字二值图 -> 归一化 ->
//      与内置模板做多尺度归一化互相关（NCC），取最大响应。
// 不依赖任何深度学习运行时，模型就是几十个 20×20 的模板，启动 0ms。
// =============================================================================
#pragma once

#include "aegis/config.hpp"
#include "aegis/types.hpp"

#include <array>
#include <vector>

namespace aegis {

class DigitRecognizer {
 public:
  DigitRecognizer();

  // 返回识别结果：0~9，-1 表示无法识别；confidence 输出置信度。
  int recognize(const cv::Mat& bgr, const Armor& armor, float& confidence) const;

  // 把装甲板区域透视变换成标准图（供调试和 HUD 缩略图使用）。
  static cv::Mat warpArmor(const cv::Mat& bgr, const Armor& armor, const cv::Size& size);

 private:
  // 每个数字保存多个模板（不同字体/粗细/缩放）。
  // mutable：识别过程中会把高置信度的实拍样本加入模板库（在线自举）。
  mutable std::array<std::vector<cv::Mat>, 10> templates_;
};

}  // namespace aegis
