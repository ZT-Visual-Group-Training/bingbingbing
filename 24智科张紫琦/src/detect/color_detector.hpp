#pragma once

#include "detect/detector.hpp"

namespace rm {

// 传统颜色检测器：
//   通道相减得到红色/蓝色灯条掩码 -> 轮廓 -> 灯条几何筛选 -> 两两配对 -> 装甲板外接框。
// 本数据中灯条中心过曝为白色，红色只剩外圈光晕，因此掩码呈“空心环”，
// 不能用“轮廓面积/外接矩形面积”过滤，改为依赖外接矩形几何与配对约束。
class ColorDetector : public Detector {
 public:
  struct Params {
    EnemyColor color = EnemyColor::kRed;
    int threshold = 50;       // 通道差值阈值（R-max(G,B) 或 B-max(G,R)）
    int close_size = 5;       // 闭运算核，用于把断开的灯条光晕连成一条

    float min_bar_aspect = 1.5f;   // 灯条高/宽下限（竖直细长）
    float max_bar_aspect = 20.0f;  // 灯条高/宽上限
    int min_bar_height = 26;       // 灯条最小长度（px，1440x1080 下的有效射程保护）
    double min_bar_area = 25.0;    // 轮廓最小面积（px）

    // 小装甲板 135 mm x 56 mm，其中高度取灯条长度
    float armor_width_mm = 135.0f;
    float armor_height_mm = 56.0f;
    float max_width_ratio = 1.22f;  // 允许的装甲板宽/灯条高 相对 135/56 的上限倍数
    float min_width_ratio = 0.60f;  // 下限（大角度偏转时投影变窄）

    float max_height_ratio = 1.6f;  // 两灯条高度比上限（转车时灯条下段可能断亮变短）
    float max_center_dy_ratio = 0.4f;  // 两灯条中心竖直偏差 / 较矮灯条高度
    float max_overlap_ratio = 0.35f;   // 允许两灯条水平重叠 / 较宽灯条宽度
  };

  ColorDetector();
  explicit ColorDetector(Params params);

  std::vector<Armor> detect(const cv::Mat& bgr) override;

  // 本帧通过灯条几何筛选的灯条数量（仅用于显示/调试）
  int lastBarCount() const { return last_bar_count_; }

 private:
  Params params_;
  int last_bar_count_ = 0;
};

}  // namespace rm
