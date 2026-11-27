#include "autoaim/visualizer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace autoaim {
namespace {

// ---------------------------------------------------------------- text helpers
//
// Every label is drawn twice: a thick black pass first, then the colour on top.
// That is what makes text readable over both the near-black scene and the blown
// out ceiling lights, and it is why the numbers stop "hiding" behind thin lines.

void textOutlined(cv::Mat& img, const std::string& s, cv::Point org, double scale,
                  const cv::Scalar& color, int thickness = 2) {
  cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0),
              thickness + 2, cv::LINE_AA);
  cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness,
              cv::LINE_AA);
}

/// Font scale so the overlay looks the same on any resolution.
double uiScale(const cv::Mat& img) {
  return std::max(0.6, img.cols / 1440.0);
}

/// cv::getTextSize RETURNS the size; its 5th argument is an output baseline.
/// Ignoring the return value and using the baseline as the height makes the line
/// pitch smaller than the glyphs, so multi-line panels overlap themselves.
cv::Size textSize(const std::string& s, double scale, int thickness = 2) {
  int baseline = 0;
  return cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
}

void quadCorners(const LightBar& b, std::vector<cv::Point>& out) {
  const cv::Point2f u = b.axis * (0.5f * b.length);
  const cv::Point2f v = b.half * (0.5f * b.width);
  out.clear();
  out.emplace_back(cv::Point(b.center - u - v));
  out.emplace_back(cv::Point(b.center + u - v));
  out.emplace_back(cv::Point(b.center + u + v));
  out.emplace_back(cv::Point(b.center - u + v));
}

void drawCross(cv::Mat& img, cv::Point p, int half, const cv::Scalar& color,
               int thickness) {
  cv::line(img, p + cv::Point(-half, 0), p + cv::Point(half, 0), color, thickness,
           cv::LINE_AA);
  cv::line(img, p + cv::Point(0, -half), p + cv::Point(0, half), color, thickness,
           cv::LINE_AA);
}

/// Darkened panel anchored at a fixed spot, so the readout can never collide with
/// the moving corner labels.
void drawPanel(cv::Mat& img, cv::Point top_left, const std::vector<std::string>& lines,
               double scale) {
  if (lines.empty()) return;
  const int thick = 2;
  cv::Size sz = textSize(lines.front(), scale, thick);
  int text_w = sz.width;
  int text_h = sz.height;
  for (const auto& s : lines) {
    const cv::Size t = textSize(s, scale, thick);
    text_w = std::max(text_w, t.width);
    text_h = std::max(text_h, t.height);
  }
  const int line_h = static_cast<int>(std::lround(text_h * 1.65));
  const int pad = static_cast<int>(std::lround(text_h * 0.7));

  cv::Rect box(top_left.x - pad, top_left.y - text_h - pad / 2, text_w + 2 * pad,
               line_h * static_cast<int>(lines.size()) + pad);
  box &= cv::Rect(0, 0, img.cols, img.rows);
  if (box.width <= 2 || box.height <= 2) return;

  cv::Mat roi = img(box);
  cv::Mat shade(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
  cv::addWeighted(roi, 0.28, shade, 0.72, 0.0, roi);
  cv::rectangle(img, box, cv::Scalar(70, 70, 70), 1);

  cv::Point org(top_left.x, top_left.y);
  for (const auto& s : lines) {
    textOutlined(img, s, org, scale, cv::Scalar(0, 255, 0), thick);
    org.y += line_h;
  }
}

}  // namespace

void drawOverlay(cv::Mat& bgr, const FrameResult& result, const DrawOptions& opt) {
  const double s = uiScale(bgr);

  const cv::Scalar kBarColor(0, 255, 255);
  const cv::Scalar kPrimary(0, 255, 0);
  const cv::Scalar kSecondary(110, 150, 0);
  const cv::Scalar kLabel(0, 255, 255);
  const cv::Scalar kLabelDim(200, 200, 120);
  const cv::Scalar kHud(0, 255, 255);

  if (opt.show_bars) {
    std::vector<cv::Point> q;
    for (const auto& b : result.bars) {
      quadCorners(b, q);
      cv::polylines(bgr, q, true, kBarColor, 1);
    }
  }

  const int primary = result.primaryIndex();
  for (size_t i = 0; i < result.armors.size(); ++i) {
    const bool is_primary = (static_cast<int>(i) == primary);
    if (!is_primary && !opt.show_secondary) continue;

    const Armor& a = result.armors[i];
    // Every candidate gets the full marking (quad, four corners, centre), not
    // just the best one. The requirement is "mark the four corners and the centre
    // of the armour plate"; drawing a bare line for the 2nd candidate looks like a
    // stray artefact and, strictly, leaves that plate unmarked.
    // 只有"已确认"的目标才画亮绿。未确认的（刚出现、或背景偶然装配一次）
    // 只画暗色细框 —— 画面诚实：没确认的东西不该看起来像一个目标。
    const bool emphasize = is_primary && result.target_confirmed;

    // 未通过时序确认的候选默认不画。它们的共同特征是只出现一两帧，照画的话
    // 画面上就会一闪而过一个青框 —— 既不是目标也不是误检，是我们多画了。
    // 调试时用 draw_tentative=1 打开。
    if (!emphasize) {
      if (!opt.show_tentative) continue;
      if (!is_primary && !opt.show_secondary) continue;
    }
    const cv::Scalar col = emphasize ? kPrimary : kSecondary;

    // The PnP corners sit on the THRESHOLDED CORE of each bar, which is a couple
    // of pixels inside the visible glow, so an exact quad looks like it cuts into
    // the bars. Inflate the DRAWN quad (and its markers) so it visibly encloses
    // the plate. Visual only: solvePnP still gets `a.corners`.
    const float margin = opt.inflate_px * static_cast<float>(s);
    std::array<cv::Point2f, 4> shown = a.corners;
    for (auto& c : shown) {
      cv::Point2f d = c - a.center;
      const float n = std::sqrt(d.x * d.x + d.y * d.y);
      if (n > 1e-3f) c += d * (margin / n);
    }

    // 上下加高：让绿框视觉上包住数字区域。
    // 纯显示量 —— PnP 用 a.corners 和 armor_height_m，所以测距不受影响。
    if (opt.extra_height_ratio > 0.f) {
      cv::Point2f up = shown[0] - shown[3];             // P1 TL - P4 BL
      const float hlen = std::sqrt(up.x * up.x + up.y * up.y);
      if (hlen > 6.f) {
        up *= (1.f / hlen);
        const float d = 0.5f * std::min(opt.extra_height_ratio, 2.0f) * hlen;
        shown[0] += up * d;  // P1
        shown[1] += up * d;  // P2
        shown[2] -= up * d;  // P3
        shown[3] -= up * d;  // P4
      }
    }

    // 左右向内收：让绿框视觉上贴着两根灯条的内侧。
    // 注意这只是"画在哪里"的问题 —— PnP 用的仍是 a.corners（灯条外缘），
    // 所以 135 mm 的物理定义没有被破坏，测距精度不受影响。
    if (opt.inset_ratio > 0.f) {
      const cv::Point2f left_mid = 0.5f * (shown[0] + shown[3]);   // P1 TL + P4 BL
      const cv::Point2f right_mid = 0.5f * (shown[1] + shown[2]);  // P2 TR + P3 BR
      cv::Point2f hdir = right_mid - left_mid;
      const float wlen = std::sqrt(hdir.x * hdir.x + hdir.y * hdir.y);
      if (wlen > 8.f) {
        hdir *= (1.f / wlen);
        // 单侧最多收 35%，避免收过头把四边形翻过来
        const float d = std::min(opt.inset_ratio, 0.35f) * wlen;
        shown[0] += hdir * d;  // P1 左上
        shown[3] += hdir * d;  // P4 左下
        shown[1] -= hdir * d;  // P2 右上
        shown[2] -= hdir * d;  // P3 右下
      }
    }

    std::vector<cv::Point> poly;
    for (const auto& c : shown) poly.emplace_back(cv::Point(c));
    cv::polylines(bgr, poly, true, col, emphasize ? 3 : 2);

    cv::Point centre(0, 0);
    for (const auto& c : shown) centre += cv::Point(c);
    centre = cv::Point(centre.x / 4, centre.y / 4);

    const float diag = std::max(cv::norm(shown[1] - shown[0]),
                                cv::norm(shown[3] - shown[0]));
    const int ring = static_cast<int>(std::lround(std::min(12.0, std::max(4.0, diag * 0.055))));

    drawCross(bgr, centre, emphasize ? static_cast<int>(std::lround(9 * s))
                                      : static_cast<int>(std::lround(6 * s)),
              col, 2);


    const double label_scale = (emphasize ? 0.85 : 0.65) * s;
    const int label_thick = 2;
    const bool show_labels = opt.show_corner_ids && diag > 45.0;

    static const char* kNames[4] = {"P1", "P2", "P3", "P4"};
    for (size_t k = 0; k < shown.size(); ++k) {
      const cv::Point p(shown[k]);
      cv::circle(bgr, p, ring, col, 2, cv::LINE_AA);
      if (!show_labels) continue;

      cv::Point2f dir = shown[k] - cv::Point2f(centre);
      const float n = std::sqrt(dir.x * dir.x + dir.y * dir.y);
      if (n > 1e-3f) dir *= (1.f / n); else dir = cv::Point2f(0.f, -1.f);

      const cv::Size tsz = textSize(kNames[k], label_scale, label_thick);
      const float off = static_cast<float>(ring) + tsz.height * 0.9f +
                        std::max(6.0f, diag * 0.05f);
      const cv::Point anchor(p.x + static_cast<int>(std::lround(dir.x * off)),
                             p.y + static_cast<int>(std::lround(dir.y * off)));
      const cv::Point org(anchor.x - tsz.width / 2, anchor.y + tsz.height / 2);
      textOutlined(bgr, kNames[k], org, label_scale, emphasize ? kLabel : kLabelDim,
                   label_thick);
    }
  }

  // ------------------------------------------------- 卡尔曼预测点（每帧只画一次）
  //
  // 放在"逐装甲板循环"之外，因为最关键的情况恰恰是"这一帧检测失败、
  // 但目标其实还在"——那时没有任何装甲板可画，只能靠预测点维持目标。
  //
  // 物理含义：位姿对应曝光那一刻，而击发还有 predict_latency_ms 的延迟，
  // 这个橙色点就是"到真正开火时，装甲板预计在哪里"。
  if (opt.show_prediction && result.has_prediction) {
    const cv::Point pp(cv::Point(result.predicted_px));
    const cv::Scalar kPredCol(0, 165, 255);  // BGR 橙
    const int r = static_cast<int>(std::lround(7 * s));
    cv::circle(bgr, pp, r, kPredCol, 2, cv::LINE_AA);
    cv::circle(bgr, pp, 2, kPredCol, -1, cv::LINE_AA);
    // 十字准星，表示"预瞄点"
    cv::line(bgr, cv::Point(pp.x - r - 4, pp.y), cv::Point(pp.x + r + 4, pp.y), kPredCol, 1,
             cv::LINE_AA);
    cv::line(bgr, cv::Point(pp.x, pp.y - r - 4), cv::Point(pp.x, pp.y + r + 4), kPredCol, 1,
             cv::LINE_AA);
    if (primary >= 0 && static_cast<size_t>(primary) < result.armors.size()) {
      cv::line(bgr, cv::Point(result.armors[primary].center), pp, kPredCol, 1, cv::LINE_AA);
    }
    // textOutlined 的签名是 (img, text, org, scale, colour, thickness)，
    // 没有"描边色"参数，轮廓色是内部固定的黑色。
    textOutlined(bgr, "PRED",
                 pp + cv::Point(static_cast<int>(12 * s), static_cast<int>(-12 * s)), 0.6 * s,
                 kPredCol, 2);
  }

  // ---------------------------------------------- 反陀螺：四块装甲的可视化
  //
  // 把陀螺模块推出的四块装甲（含被遮挡的三块）的 3D 坐标投影回画面，
  // 用四种颜色画空心圆。实心小点表示"当前实测那块"（= 相位 0）。
  //
  // 颜色约定：
  //   青   plate 0（当前实测）
  //   黄   plate 1
  //   紫   plate 2
  //   蓝   plate 3
  //   白粗 推荐交给云台的那块（best）
  //
  // 画出的圆不在真实装甲位置上（被遮挡的板本来就不在画面里），
  // 它们表示"按陀螺模型推断，那块板此刻在图像上会投影到哪"。
  if (opt.show_gyro && result.gyro_valid) {
    static const cv::Scalar kGyroCol[4] = {
        cv::Scalar(255, 255, 0),    // 青 BGR(255,255,0)
        cv::Scalar(0, 220, 255),    // 黄
        cv::Scalar(255, 0, 220),    // 紫
        cv::Scalar(255, 120, 0)};   // 蓝
    for (int k = 0; k < 4; ++k) {
      if (!result.gyro_px_ok[static_cast<size_t>(k)]) continue;
      const cv::Point pp(cv::Point(result.gyro_px[static_cast<size_t>(k)]));
      const cv::Scalar col = kGyroCol[k];
      const bool is_best = (k == result.gyro_best);
      const int r = static_cast<int>(std::lround((is_best ? 22 : 14) * s));
      cv::circle(bgr, pp, r, col, is_best ? 3 : 2, cv::LINE_AA);
      if (k == 0) cv::circle(bgr, pp, 3, col, -1, cv::LINE_AA);  // 实测块画实心
      if (is_best) {
        char gb[32];
        std::snprintf(gb, sizeof(gb), "GYRO BEST");
        textOutlined(bgr, gb, pp + cv::Point(r + 4, -r), 0.55 * s, cv::Scalar(255, 255, 255), 2);
      }
    }
    // 旋转中心（灰白小方框）：判断拟合出的圆心是否落在目标身上
    if (result.gyro_centre_ok) {
      const cv::Point cc(cv::Point(result.gyro_centre_px));
      cv::rectangle(bgr, cc - cv::Point(6, 6), cc + cv::Point(6, 6),
                    cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
      cv::line(bgr, cc - cv::Point(9, 0), cc + cv::Point(9, 0), cv::Scalar(200, 200, 200), 1);
      cv::line(bgr, cc - cv::Point(0, 9), cc + cv::Point(0, 9), cv::Scalar(200, 200, 200), 1);
    }
    // 提前量提示：如果最优板不是当前这块，说明在"瞄还没转过来的那块"
    if (!result.gyro_best_is_current) {
      char nb[96];
      std::snprintf(nb, sizeof(nb), "NEXT +%.0fms  yaw %+.0f deg",
                    result.gyro_predict_ahead_s * 1000.0, result.gyro_best_yaw_at_fire);
      textOutlined(bgr, nb, cv::Point(static_cast<int>(14 * s), static_cast<int>(108 * s)),
                   0.75 * s, cv::Scalar(255, 255, 255), 2);
    }
  }

  // ---------------------------------------------------------------- HUD
  char hud[256];
  const double ms = result.timing.total_ms;
  const double fps = (ms > 1e-6) ? 1000.0 / ms : 0.0;
  std::snprintf(hud, sizeof(hud), "%.2f ms   %.1f FPS   armors: %zu", ms, fps,
                result.armors.size());
  textOutlined(bgr, hud, cv::Point(static_cast<int>(14 * s), static_cast<int>(38 * s)),
               1.0 * s, kHud, 2);

  std::snprintf(hud, sizeof(hud), "detect %.2f  match %.2f  pnp %.2f ms",
                result.timing.detect_ms, result.timing.match_ms, result.timing.pnp_ms);
  textOutlined(bgr, hud, cv::Point(static_cast<int>(14 * s), static_cast<int>(72 * s)),
               0.72 * s, kHud, 2);

  // ---------------------------------------------------------------- readout
  std::vector<std::string> panel;
  const Pose* p = result.primaryPose();
  if (p == nullptr || primary < 0) {
    // 这一帧没检出装甲板。如果滤波器还在续推，就把预测位置报出来 ——
    // 这正是"短暂遮挡不甩丢"要展示的状态：没有实测，但有预测。
    if (result.has_prediction) {
      panel.emplace_back("no armor  (coasting)");
      char ph[128];
      std::snprintf(ph, sizeof(ph), "PRED %+.3f %+.3f %+.3f m", result.predicted_tvec[0],
                    result.predicted_tvec[1], result.predicted_tvec[2]);
      panel.emplace_back(ph);
    } else {
      panel.emplace_back("no armor");
    }
    drawPanel(bgr, cv::Point(static_cast<int>(14 * s), static_cast<int>(120 * s)), panel,
              0.85 * s);
    return;
  }

  if (p->ok) {
    double yaw = 0, pitch = 0;
    p->yawPitchDeg(&yaw, &pitch);
    std::snprintf(hud, sizeof(hud), "X %+.3f   Y %+.3f   Z %.3f m", p->x(), p->y(), p->z());
    panel.emplace_back(hud);
    std::snprintf(hud, sizeof(hud), "range %.3f m", p->range());
    panel.emplace_back(hud);
    // Yaw only. Pitch is deliberately not shown: with four coplanar points a
    // plate that tilts back by theta shrinks its apparent height by cos(theta),
    // which is indistinguishable from moving further away. The height is only
    // ~68 px here, so 1 px of corner error maps to about 10 deg of pitch - the
    // estimate is noise (measured: std 16-25 deg, sign flips on adjacent
    // frames). Yaw is observable because the width contracts as cos(yaw) and the
    // width is ~178 px, giving 5x more leverage.
    std::snprintf(hud, sizeof(hud), "yaw %+.1f deg", yaw);
    panel.emplace_back(hud);
    std::snprintf(hud, sizeof(hud), "reproj err %.2f px", p->reproj_err_px);
    panel.emplace_back(hud);
    // 卡尔曼预测（延迟补偿）：把位姿按速度外推到击发时刻
    if (result.has_prediction) {
      std::snprintf(hud, sizeof(hud), "pred %+.3f %+.3f %+.3f m  (+%.0f ms)",
                    result.predicted_tvec[0], result.predicted_tvec[1],
                    result.predicted_tvec[2], result.predict_latency_ms);
      panel.emplace_back(hud);
    }
    // 时序状态：0 无 / 1 确认 / 2 续帧(丢失容忍期内, 用预测位置) / 3 次候选
    static const char* kTrackText[4] = {"none", "confirmed", "predicted", "tentative"};
    const int ts = (result.track_state >= 0 && result.track_state < 4) ? result.track_state : 0;
    std::snprintf(hud, sizeof(hud), "track %s (%d)", kTrackText[ts], result.confirm_hits);
    panel.emplace_back(hud);
    drawPanel(bgr, cv::Point(static_cast<int>(14 * s), static_cast<int>(120 * s)), panel,
              0.85 * s);
  } else {
    std::snprintf(hud, sizeof(hud), "pose rejected (err %.2f px)", p->reproj_err_px);
    panel.emplace_back(hud);
    drawPanel(bgr, cv::Point(static_cast<int>(14 * s), static_cast<int>(120 * s)), panel,
              0.85 * s);
    // the panel text colour is green; flag the rejection near the plate too
    cv::Point c(0, 0);
    for (const auto& q : result.armors[primary].corners) c += cv::Point(q);
    c = cv::Point(c.x / 4, c.y / 4);
    textOutlined(bgr, "rejected", c + cv::Point(-static_cast<int>(40 * s),
                                               static_cast<int>(22 * s)),
                 0.8 * s, cv::Scalar(0, 140, 255), 2);
  }
}

}  // namespace autoaim
