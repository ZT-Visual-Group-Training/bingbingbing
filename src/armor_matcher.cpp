#include "autoaim/armor_matcher.hpp"

#include "autoaim/armor_corners.hpp"

#include <algorithm>
#include <cmath>

// ===========================================================================
// 本文件：装甲板的「灯条配对」——把两根灯条认成一块装甲板
//
// 输入：std::vector<LightBar>（可能 0~15 根，有真有假）
// 输出：std::vector<Armor>，按 score 升序（越小越像真的），第一个就是最佳目标
//
// 为什么要配对：
//   单独一根灯条无法确定位置 —— 它可能是小装甲板的左条、大装甲板的右条，
//   或者哨兵上的一根。只有成对的灯条才能确定「这是一块 135 mm 宽的板」。
//
// 五个硬约束，每个都有物理理由，也都是实测调出来的：
//
//   [1] 长度比 ≤ 1.25
//       同一块板的两根灯条在 3D 里一样长；但透视下近的一侧看起来更长。
//       实测真配对 ≤1.10，跨板误配 1.27~1.52 → 阈值取中间的 1.25。
//
//   [2] 中心距 / 平均长度 ∈ [1.2, 4.2]，理想值 2.2
//       (135 − 灯条宽约12) / 56 = 2.20。为什么用区间不用等式：
//       透视不是相似变换，斜着看时这个比值会变，所以只能给区间 + 打分。
//       分母用"平均长度"是因为两侧缩放不同，取平均更稳。
//
//   [3] 两灯条角度差 ≤ 10°
//       同一块板上的两根灯条在 3D 里平行。实测真配对 ≤3°，跨板误配 11~18°。
//
//   [4] 连线方向 ≈ 垂直于灯条长轴（±25°）
//       同一块板的两根灯条是"并排"的，所以它们的连线应当沿着灯条的短轴方向。
//       用这个而不是比较 |dy|：装甲板在画面里会倾斜，|dy| 会随倾角变化，
//       而"连线垂直于灯条"是旋转不变量。
//
//   [5] 四边形宽高比 ∈ [1.4, 3.2]
//       135:56 = 2.41。太宽（>3.2）说明把两块板各一根灯条配到了一起。
//       注意：这个比值用「灯条四点」算，必须在「板高外扩」之前算，
//       否则板子比灯条高时宽高比会掉到 1.3 被误杀。
//
// 打分（不是"满足就选"，而是"选最好的"）：
//   score = |长度比−1| + |中心距比−2.2| + 角度差/10 + 横向偏差/25
//   实测真装甲板 0.30~0.62，背景灯具凑成的假配对 1.14~1.72，
//   所以 max_armor_score = 1.0 就把背景误检挡掉了。
// ===========================================================================

namespace autoaim {
namespace {
constexpr double kRad2Deg = 180.0 / CV_PI;
}

float ArmorMatcher::angleDiffDeg(float a, float b) {
  float d = std::fabs(a - b);
  while (d >= 180.f) d -= 180.f;
  return std::min(d, 180.f - d);
}

Armor ArmorMatcher::build(const LightBar& left, const LightBar& right, float score,
                          float center_ratio, float length_ratio,
                          float angle_diff) const {
  auto outer_pair = [](const LightBar& bar, const LightBar& other,
                       cv::Point2f* top, cv::Point2f* bottom) {
    cv::Point2f v = bar.half;
    const cv::Point2f to_other = other.center - bar.center;
    if (to_other.dot(v) > 0.f) v = -v;  // short-axis sign pointing away from the pair
    const cv::Point2f mid = bar.center + v * (0.5f * bar.width);
    cv::Point2f a = mid + bar.axis * (0.5f * bar.length);
    cv::Point2f b = mid - bar.axis * (0.5f * bar.length);
    if (a.y > b.y) std::swap(a, b);
    *top = a;
    *bottom = b;
  };

  Armor armor;

  // ---- REPLACEMENT POINT -------------------------------------------------
  // Old code took the four plate corners straight off the two bar end points.
  // Now they are derived by extending each bar's centre axis, offsetting to its
  // outer edge, and intersecting with the perpendicular top/bottom boundary
  // lines, so a corner is constrained by whole lines instead of one noisy tip.
  std::vector<cv::Point2f> corners;
  if (armorCornersFromLightBars(left.rect, right.rect, &corners,
                                static_cast<float>(cfg_.plate_edge_offset))) {
    armor.corners = {corners[0], corners[1], corners[2], corners[3]};
  } else {
    // Degenerate pair (nearly parallel lines / zero-size bar): keep the old
    // endpoint construction as the fallback so behaviour never gets worse.
    cv::Point2f lt, lb, rt, rb;
    outer_pair(left, right, &lt, &lb);
    outer_pair(right, left, &rt, &rb);
    armor.corners = {lt, rt, rb, lb};
  }
  // ------------------------------------------------------------------------

  // ---- aspect gate, computed on the BAR quad -----------------------------
  // This check exists to reject two bars that are not one plate, so it must see
  // the light-bar geometry (135:56 = 2.41). It is deliberately evaluated BEFORE
  // the plate-height extension below: if the plate is taller than its bars, the
  // final quad's aspect is legitimately smaller than 2.4 and must not be rejected.
  {
    const float w0 = cv::norm(armor.corners[1] - armor.corners[0]);  // P2 - P1
    const float h0 = cv::norm(armor.corners[3] - armor.corners[0]);  // P4 - P1
    armor.aspect = w0 / std::max(h0, 1e-6f);
    armor.width_px = w0;   // 供"最小可信尺寸"门限使用
  }

  // If the plate is taller than its light bars, push the top/bottom edges out
  // symmetrically so the quad encloses the whole plate (the number sits on the
  // plate, not just between the bar ends). The px<->mm scale comes from the bar
  // itself, and PnP uses armor_height_m as the object-point height, so extending
  // here keeps the distance consistent instead of breaking it.
  const float mean_bar_px = 0.5f * (left.length + right.length);
  if (cfg_.bar_length_m > 1e-6 && mean_bar_px > 1.f) {
    const float extra_m = static_cast<float>(cfg_.armor_height_m - cfg_.bar_length_m);
    if (extra_m > 0.f) {
      const float px_per_m = mean_bar_px / static_cast<float>(cfg_.bar_length_m);
      const float d_px = 0.5f * extra_m * px_per_m;

      cv::Point2f up = armor.corners[0] - armor.corners[3];  // P1(TL) - P4(BL)
      const float nu = std::sqrt(up.x * up.x + up.y * up.y);
      if (nu > 1e-6f) {
        up *= (d_px / nu);
        armor.corners[0] += up;  // TL
        armor.corners[1] += up;  // TR
        armor.corners[2] -= up;  // BR
        armor.corners[3] -= up;  // BL
      }
    }
  }

  armorCenterFromCorners(std::vector<cv::Point2f>(armor.corners.begin(),
                                                  armor.corners.end()),
                         &armor.center);

  armor.score = score;
  armor.center_ratio = center_ratio;
  armor.length_ratio = length_ratio;
  armor.angle_diff = angle_diff;
  return armor;
}

std::vector<Armor> ArmorMatcher::match(const std::vector<LightBar>& bars) const {
  std::vector<Armor> armors;
  const size_t n = bars.size();
  if (n < 2) return armors;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const LightBar& a = bars[i];
      const LightBar& b = bars[j];
      const LightBar& lb = (a.center.x <= b.center.x) ? a : b;
      const LightBar& rb = (a.center.x <= b.center.x) ? b : a;

      const float mean_len = 0.5f * (lb.length + rb.length);
      if (mean_len <= 1e-6f) continue;

      const float len_ratio =
          std::max(lb.length, rb.length) / std::max(std::min(lb.length, rb.length), 1e-6f);
      if (len_ratio > cfg_.max_length_ratio) continue;

      const cv::Point2f d = rb.center - lb.center;
      const float dist = std::sqrt(d.x * d.x + d.y * d.y);
      if (dist < 1e-6f) continue;
      const float center_ratio = dist / mean_len;
      if (center_ratio < cfg_.min_center_ratio || center_ratio > cfg_.max_center_ratio) {
        continue;
      }

      const float adiff = angleDiffDeg(lb.angle_deg, rb.angle_deg);
      if (adiff > cfg_.max_angle_diff_deg) continue;

      // 亮度一致性：同一块装甲板的两根灯条由同一路 LED 驱动，亮度必然接近。
      // 背景凑出来的两根"灯条"常常一根是灯管、一根是反光件，亮度差很大。
      // 这一步用物理约束挡掉"形态上像、物理上不像"的配对。
      {
        const float bmax = std::max(lb.mean_val, rb.mean_val);
        const float bmin = std::min(lb.mean_val, rb.mean_val);
        if (bmin > 1.f && bmax / bmin > cfg_.max_bar_brightness_ratio) continue;
      }

      // The join must run across the bars, i.e. along their short axis.
      cv::Point2f vavg = lb.half + rb.half;
      const float nv = std::sqrt(vavg.x * vavg.x + vavg.y * vavg.y);
      if (nv < 1e-6f) continue;
      vavg = cv::Point2f(vavg.x / nv, vavg.y / nv);
      const float cos_a =
          std::min(1.f, std::fabs((d.x * vavg.x + d.y * vavg.y) / dist));
      const float side_deg = std::acos(std::min(1.f, std::max(-1.f, cos_a))) *
                             static_cast<float>(kRad2Deg);
      if (side_deg > cfg_.max_side_angle_deg) continue;

      const float score = std::fabs(len_ratio - 1.f) +
                          std::fabs(center_ratio - static_cast<float>(cfg_.ideal_center_ratio)) +
                          adiff / static_cast<float>(cfg_.max_angle_diff_deg) +
                          side_deg / static_cast<float>(cfg_.max_side_angle_deg);

      Armor armor = build(lb, rb, score, center_ratio, len_ratio, adiff);
      if (armor.aspect < cfg_.min_quad_aspect || armor.aspect > cfg_.max_quad_aspect) {
        continue;
      }
      // Reject weak pairs outright. Measured: genuine plates score 0.30-0.62,
      // background light fixtures that happen to form a 135:56 quad score
      // 1.14-1.72. Without this the background pairs get drawn as if they were
      // targets.
      if (armor.score > cfg_.max_armor_score) {
        continue;
      }
      // 最小可信尺寸：真装甲板在 2 m 处约 160 px 宽，背景误检往往只有 25~40 px。
      // 物理上等价于"最小可信距离"：w_px = fx x 0.135 / d，45 px 约等于 7.0 m。
      // 比距离门限更便宜 —— 在 PnP 之前就用像素量把它挡掉。
      if (armor.width_px < cfg_.min_armor_width_px) {
        continue;
      }
      armors.push_back(armor);
    }
  }

  std::sort(armors.begin(), armors.end(),
            [](const Armor& x, const Armor& y) { return x.score < y.score; });
  return armors;
}

}  // namespace autoaim
