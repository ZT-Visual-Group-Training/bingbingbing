// Overlay: corners, centre, 3D position, timing.
#pragma once

#include "autoaim/types.hpp"

#include <opencv2/core.hpp>

namespace autoaim {

struct DrawOptions {
  bool show_bars = true;      // thin yellow boxes around every light bar
  bool show_secondary = true; // other armour candidates, thin
  bool show_corner_ids = true;
  // 是否绘制未通过时序确认的候选。默认 false：只出现一两帧的候选如果照画，
  // 画面上就会一闪而过一个青框，看起来像误检 —— 其实它已经被时序确认挡住了。
  bool show_tentative = false;
  // 是否绘制卡尔曼预测点（延迟补偿后的预瞄位置）
  bool show_prediction = true;
  // 是否把反陀螺推出的四块装甲投影回画面画圈（不同颜色）
  bool show_gyro = true;
  // Outward inflation of the DRAWN quad, pixels at 1440 px width. The PnP corners
  // sit on the thresholded core of each bar, which is a few pixels inside the
  // visible glow; drawing them exactly makes the box look like it cuts the bars.
  // Raise this to make the box visibly enclose the bars (visual only - solvePnP
  // keeps the real corners, so distance is unaffected).
  float inflate_px = 5.0f;
  // 左右两条边按"四边形宽度的比例"向内收，让框贴着灯条内侧显示。
  // 这是纯显示量：solvePnP 仍然用外缘角点（物理上 135 mm 是外缘间距），
  // 所以框好看不影响测距精度。0 = 不外收（框住外缘）。
  float inset_ratio = 0.09f;
  // 显示用的"上下加高"比例：画出的四边形高度 = 角点高度 x (1 + ratio)。
  // 同样只影响画面 —— PnP 用的是真实板高 armor_height_m，所以"框包住数字"
  // 这件事不会牺牲测距精度。0 = 不加高。
  float extra_height_ratio = 0.9f;
};

void drawOverlay(cv::Mat& bgr, const FrameResult& result, const DrawOptions& opt);

}  // namespace autoaim
