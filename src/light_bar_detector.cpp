#include "autoaim/light_bar_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

// ===========================================================================
// 本文件：装甲板的「灯条检测」——整条流水线的第一步
//
// 输入：一帧 BGR 图（1440x1080）
// 输出：std::vector<LightBar>，每根灯条带 中心 / 长度 / 宽度 / 轴方向 / 亮度
//
// 为什么这一步最关键：
//   后面所有环节（配对、角点、PnP）都以「灯条准不准」为前提。
//   灯条少了 → 配不成对 → 漏检；灯条碎了 → 长度量不准 → 距离偏。
//   所以本文件大半代码都在解决「怎么把灯条从背景里干净地抠出来」。
//
// 七个子步骤，每一步对应一类失败模式：
//   ① 通道差 R−B     : 让白色背景灯"消失"，只留橙红色 LED
//   ② 亮度门限       : 只在足够亮的区域找，压掉暗部噪点
//   ③ 二值化         : 灰度 → 黑白
//   ④ 开运算 3x3     : 去掉 JPEG / CMOS 的孤立噪点
//   ⑤ 闭运算 3x17    : 补竖向断裂（数字遮挡 / JPEG 块 / LED 灯珠间隙）
//   ⑥ findContours   : 取连通块
//   ⑦ 几何筛 + PCA轴 : 只留"像灯条"的，并算出它的真实方向
//
// 三个关键取舍（面试常问）：
//
//  [1] 为什么用 R−B，不用灰度、不用 HSV？
//      这段素材是 2 ms 曝光、整体极暗，灯条中心已经过曝、绿通道被抬高。
//      实测同一帧：R−G 只剩 348 px 且碎成 5 块；R−B 有 1882 px、两块完整
//      （523 / 496 px）。白灯管 R≈G≈B，任何通道差都≈0，于是顺带免掉了
//      背景建模。HSV 不行是因为 H 是比值（分母 max−min），暗处和过曝处都
//      不稳定，而且红色跨 H 的 0/179 边界，要拆成两个区间。
//
//  [2] 为什么闭运算的核是"竖长条"3×17，而不是方形？
//      灯条是竖直细条，它的断裂必然是竖向的。竖核能补竖向缺口，但不会把
//      左右相邻的两根灯条粘成一块 —— 而 17×17 的方核会。核高 17 是扫出来的：
//      9 不够（灯条还是断的）、17 刚好、21 无变化。
//
//  [3] 为什么用 fitLine(PCA) 取轴向，而不是 minAreaRect 的 angle？
//      细长且略微破碎的轮廓，minAreaRect 的角度会被噪声带偏几度；而配对要求
//      两根灯条的角度差 ≤10°，几度的误差就足以把真配对误杀。PCA 拟合的是
//      整体点云的主方向，稳得多。
// ===========================================================================

namespace autoaim {
namespace {
constexpr double kRad2Deg = 180.0 / CV_PI;
}

void LightBarDetector::computeDiff(const cv::Mat& bgr) {
  std::vector<cv::Mat> ch;
  cv::split(bgr, ch);  // B, G, R

  const bool red = (cfg_.enemy_color != "blue");
  const cv::Mat& g = ch[1];
  const cv::Mat& prim = red ? ch[2] : ch[0];   // R for red, B for blue
  const cv::Mat& other = red ? ch[0] : ch[2];  // B for red, R for blue

  if (cfg_.chroma_mode == "rg") {
    cv::subtract(prim, g, diff_);
  } else if (cfg_.chroma_mode == "cr") {
    // Cr = 0.5R - 0.4187G - 0.0813B + 128 (BT.601); Cb analogous for blue.
    cv::Mat tmp;
    if (red) {
      cv::addWeighted(ch[2], 0.5, g, -0.4187, 128.0, tmp);
      cv::addWeighted(tmp, 1.0, ch[0], -0.0813, 0.0, diff_);
    } else {
      cv::addWeighted(ch[0], 0.5, g, -0.3313, 128.0, tmp);
      cv::addWeighted(tmp, 1.0, ch[2], -0.1687, 0.0, diff_);
    }
  } else {
    // Default "rb". Measured on the assessment footage this is the only operator
    // that keeps the bars whole: R-G leaves 348 px split into 5 fragments,
    // R-B leaves 1882 px in two clean 523/496 px bars.
    cv::subtract(prim, other, diff_);
  }

  if (cfg_.use_brightness_gate) {
    cv::Mat vmax;
    cv::max(ch[0], ch[1], vmax);
    cv::max(vmax, ch[2], vmax);
    cv::Mat mask;
    cv::threshold(vmax, mask, cfg_.brightness_min, 255, cv::THRESH_BINARY);
    cv::bitwise_and(diff_, mask, diff_);
  }
}

std::vector<LightBar> LightBarDetector::detect(const cv::Mat& bgr, const cv::Rect& roi_in) {
  std::vector<LightBar> bars;

  cv::Rect roi = roi_in;
  if (roi.width > 4 && roi.height > 4) {
    roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
  } else {
    roi = cv::Rect(0, 0, bgr.cols, bgr.rows);
  }
  if (roi.width <= 4 || roi.height <= 4) return bars;
  last_roi_ = roi;

  const cv::Mat view = bgr(roi);
  computeDiff(view);

  cv::threshold(diff_, binary_, cfg_.diff_threshold, 255, cv::THRESH_BINARY);

  if (cfg_.do_open && cfg_.morph_open_kernel > 1) {
    const cv::Mat k = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(cfg_.morph_open_kernel, cfg_.morph_open_kernel));
    cv::morphologyEx(binary_, binary_, cv::MORPH_OPEN, k);
  }
  if (cfg_.do_close && cfg_.morph_close_w > 1 && cfg_.morph_close_h > 1) {
    const cv::Mat k = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(cfg_.morph_close_w, cfg_.morph_close_h));
    cv::morphologyEx(binary_, binary_, cv::MORPH_CLOSE, k);
  }

  std::vector<std::vector<cv::Point>> contours;
  // CHAIN_APPROX_SIMPLE, not NONE: NONE returns every boundary pixel, and the
  // pixel-grid staircase then biases the PCA axis fit, which shifts the measured
  // length/width and angle until valid bars fall out of the geometry filter.
  // Measured on clip A frames 0..199: SIMPLE 85.0% hit / 81.0% pose,
  // NONE 75.0% / 72.5%. It is also several times faster to fit and project.
  cv::findContours(binary_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  const int dom_channel = (cfg_.enemy_color != "blue") ? 2 : 0;
  const cv::Mat dom = view;

  for (const auto& c : contours) {
    // A valid closed contour needs 3 points; anything smaller has zero area and
    // is dropped by the area test below anyway.
    if (c.size() < 3) continue;
    const double area = cv::contourArea(c);
    if (area < cfg_.min_area || area > cfg_.max_area) continue;

    const cv::RotatedRect rr = cv::minAreaRect(c);
    const double l0 = std::max(rr.size.width, rr.size.height);
    const double w0 = std::min(rr.size.width, rr.size.height);
    if (l0 < cfg_.min_lightbar_length || w0 < 1e-6) continue;

    // Axis from a PCA line fit rather than the rect angle: for a thin, slightly
    // ragged blob the rect angle can be skewed by several degrees, which then
    // fails the pairing angle check. fitLine is far more stable.
    std::vector<cv::Point2f> ptsf(c.size());
    for (size_t i = 0; i < c.size(); ++i) ptsf[i] = cv::Point2f(c[i]);
    cv::Vec4f line;
    cv::fitLine(ptsf, line, cv::DIST_L2, 0, 0.01, 0.01);
    cv::Point2f u(line[0], line[1]);
    const float nu = std::sqrt(u.x * u.x + u.y * u.y);
    if (nu < 1e-9f) continue;
    u /= nu;
    if (u.y > 0.f) u = -u;  // long axis points "up" (image y grows downwards)
    const cv::Point2f v(-u.y, u.x);

    double tmin = 1e18, tmax = -1e18, smin = 1e18, smax = -1e18;
    for (const auto& p : ptsf) {
      const cv::Point2f rel = p - rr.center;
      const double t = rel.x * u.x + rel.y * u.y;
      const double s = rel.x * v.x + rel.y * v.y;
      tmin = std::min(tmin, t);
      tmax = std::max(tmax, t);
      smin = std::min(smin, s);
      smax = std::max(smax, s);
    }
    const double length = tmax - tmin;
    const double width = smax - smin;
    if (length < cfg_.min_lightbar_length || width < 1e-6) continue;

    const double ratio = length / width;
    if (ratio < cfg_.min_lightbar_ratio || ratio > cfg_.max_lightbar_ratio) continue;

    const double fill = area / std::max(length * width, 1e-9);
    if (fill < cfg_.min_fill_ratio) continue;

    double ang = std::atan2(u.y, u.x) * kRad2Deg;
    if (ang < -90.0) ang += 180.0;
    else if (ang > 90.0) ang -= 180.0;
    if (std::fabs(90.0 - std::fabs(ang)) > cfg_.max_tilt_deg) continue;

    LightBar bar;
    bar.center = rr.center + u * static_cast<float>(0.5 * (tmin + tmax)) +
                 v * static_cast<float>(0.5 * (smin + smax));
    bar.length = static_cast<float>(length);
    bar.width = static_cast<float>(width);
    bar.axis = u;
    bar.half = v;
    bar.angle_deg = static_cast<float>(ang);

    // LED-saturation gate sampled along the CENTRE LINE. Sampling the contour
    // boundary would read the dim transition pixels, not the bright core.
    const int ns = 7;
    double acc = 0.0;
    for (int k = 0; k < ns; ++k) {
      const double t = (-0.45 + 0.9 * k / (ns - 1)) * bar.length;
      const int y = static_cast<int>(std::lround(bar.center.y + u.y * t));
      const int x = static_cast<int>(std::lround(bar.center.x + u.x * t));
      const int cy = std::max(0, std::min(dom.rows - 1, y));
      const int cx = std::max(0, std::min(dom.cols - 1, x));
      acc += dom.ptr<uchar>(cy)[cx * dom.channels() + dom_channel];
    }
    bar.mean_val = static_cast<float>(acc / ns);
    if (bar.mean_val < cfg_.min_bar_mean) continue;

    // Keep the raw rect (ROI-local here, shifted below like everything else) so
    // armorCornersFromLightBars() can consume the documented RotatedRect input.
    bar.rect = rr;

    // back to full-frame coordinates
    bar.center += cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
    bar.rect.center += cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
    bars.push_back(bar);
  }

  return bars;
}

}  // namespace autoaim
