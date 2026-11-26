// =============================================================================
// hud.cpp —— 特色三/四的可视化实现
//
// 画面分三层：
//   1. 目标层：锁定括号、角点、中心、距离标签、轨迹尾巴；
//   2. 数据层：左侧性能/检测/跟踪面板，右侧三维定位面板，底部时间条；
//   3. 片头片尾：演示视频用的标题卡和总结卡。
//
// 所有数值都来自 FrameResult / HudState 的真实测量，没有“写死的好看数字”。
// =============================================================================
#include "aegis/hud.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace aegis {
namespace {

// ---- 配色（OpenCV 是 BGR 顺序）----
const cv::Scalar kCyan(255, 214, 10);    // 主色：青
const cv::Scalar kAmber(0, 176, 255);    // 警告 / 缓存
const cv::Scalar kRed(70, 70, 255);      // 丢失 / 危险
const cv::Scalar kGreen(120, 255, 120);  // 正常
const cv::Scalar kWhite(238, 242, 245);
const cv::Scalar kGray(150, 158, 165);
const cv::Scalar kDim(95, 102, 110);
const cv::Scalar kPanel(12, 10, 8);

constexpr int kPanelAlpha = 168;  // 0..255，面板不透明度

// 带透明度的矩形填充。
void alphaFill(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, double alpha) {
  const int x1 = std::clamp(r.x, 0, img.cols);
  const int y1 = std::clamp(r.y, 0, img.rows);
  const int x2 = std::clamp(r.x + r.width, 0, img.cols);
  const int y2 = std::clamp(r.y + r.height, 0, img.rows);
  if (x2 <= x1 || y2 <= y1) return;
  cv::Mat roi = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
  cv::Mat layer(roi.size(), img.type(), color);
  cv::addWeighted(layer, alpha, roi, 1.0 - alpha, 0.0, roi);
}

// 切角面板：左上角切一个 12px 的斜角，比圆角更有“军工界面”的感觉。
// 性能注意：这里只对面板自身大小做透明混合，绝不能对整帧建 mask——
// 之前图省事对 1440×1080 建 mask，一帧 HUD 要 40ms，直接拖垮演示帧率。
void panel(cv::Mat& img, const cv::Rect& r, const cv::Scalar& accent, bool header = true) {
  alphaFill(img, r, kPanel, 0.66);

  const int cut = 12;
  const std::vector<cv::Point> pts = {
      {r.x + cut, r.y},
      {r.x + r.width - 1, r.y},
      {r.x + r.width - 1, r.y + r.height - 1},
      {r.x, r.y + r.height - 1},
      {r.x, r.y + cut},
  };
  cv::polylines(img, pts, true, accent, 1, cv::LINE_AA);
  // 左上角斜切：用面板底色盖掉直角，视觉上是“切角”而不是圆角。
  cv::line(img, {r.x, r.y + cut}, {r.x + cut, r.y}, accent, 1, cv::LINE_AA);
  if (header) {
    cv::line(img, {r.x + 1, r.y + 30}, {r.x + r.width - 1, r.y + 30}, accent * 0.35, 1, cv::LINE_AA);
  }
}

// 普通文字（带 1px 阴影，暗背景上更清楚）。
void text(cv::Mat& img, const std::string& s, const cv::Point& p, double scale,
          const cv::Scalar& color, int thickness = 1, int font = cv::FONT_HERSHEY_DUPLEX) {
  cv::putText(img, s, p + cv::Point(1, 1), font, scale, cv::Scalar(0, 0, 0), thickness + 1,
              cv::LINE_AA);
  cv::putText(img, s, p, font, scale, color, thickness, cv::LINE_AA);
}

// 数字使用更“工程感”的字体。
void num(cv::Mat& img, const std::string& s, const cv::Point& p, double scale,
         const cv::Scalar& color, int thickness = 1) {
  text(img, s, p, scale, color, thickness, cv::FONT_HERSHEY_DUPLEX);
}

// 带字距的标题，例如 A E G I S   V I S I O N。
void spacedText(cv::Mat& img, const std::string& s, const cv::Point& p, double scale,
                const cv::Scalar& color, int thickness, int gap) {
  int x = p.x;
  int baseline = 0;
  for (char c : s) {
    const std::string one(1, c);
    text(img, one, {x, p.y}, scale, color, thickness);
    const cv::Size sz = cv::getTextSize(one, cv::FONT_HERSHEY_DUPLEX, scale, thickness, &baseline);
    x += sz.width + gap;
  }
}

std::string fixed(double v, int decimals) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
  return buf;
}

// 横向进度条。
void bar(cv::Mat& img, const cv::Rect& r, float ratio, const cv::Scalar& color) {
  cv::rectangle(img, r, kDim, 1);
  const int w = static_cast<int>(std::clamp(ratio, 0.0F, 1.0F) * (r.width - 2));
  if (w > 0) cv::rectangle(img, cv::Rect(r.x + 1, r.y + 1, w, r.height - 2), color, cv::FILLED);
}

// 折线图：把一组数值画进 rect。
void sparkline(cv::Mat& img, const cv::Rect& r, const std::deque<float>& data, double vmax,
               const cv::Scalar& color) {
  if (data.size() < 2) return;
  const double step = static_cast<double>(r.width) / std::max<std::size_t>(1, data.size() - 1);
  cv::Point prev;
  for (std::size_t i = 0; i < data.size(); ++i) {
    const double v = std::clamp(data[i] / vmax, 0.0, 1.0);
    const cv::Point p(r.x + static_cast<int>(i * step),
                      r.y + r.height - static_cast<int>(v * r.height));
    if (i > 0) cv::line(img, prev, p, color, 1, cv::LINE_AA);
    prev = p;
  }
}

// 目标锁定括号。
void brackets(cv::Mat& img, const cv::Rect2f& box, const cv::Scalar& color, bool locked) {
  const int x1 = static_cast<int>(box.x);
  const int y1 = static_cast<int>(box.y);
  const int x2 = static_cast<int>(box.x + box.width);
  const int y2 = static_cast<int>(box.y + box.height);
  const int len = std::clamp(std::min(x2 - x1, y2 - y1) / 4, 8, 26);
  const int t = locked ? 2 : 1;
  cv::line(img, {x1, y1}, {x1 + len, y1}, color, t, cv::LINE_AA);
  cv::line(img, {x1, y1}, {x1, y1 + len}, color, t, cv::LINE_AA);
  cv::line(img, {x2, y1}, {x2 - len, y1}, color, t, cv::LINE_AA);
  cv::line(img, {x2, y1}, {x2, y1 + len}, color, t, cv::LINE_AA);
  cv::line(img, {x1, y2}, {x1 + len, y2}, color, t, cv::LINE_AA);
  cv::line(img, {x1, y2}, {x1, y2 - len}, color, t, cv::LINE_AA);
  cv::line(img, {x2, y2}, {x2 - len, y2}, color, t, cv::LINE_AA);
  cv::line(img, {x2, y2}, {x2, y2 - len}, color, t, cv::LINE_AA);
}

// ---- 三维视角：手工实现一个 look-at + 透视投影 ----
struct View3D {
  cv::Vec3d eye;
  cv::Matx33d rot;  // 世界 -> 相机
  double focal = 1.0;
  cv::Point2f center;
};

View3D makeView(const cv::Rect& panel) {
  View3D v;
  v.eye = cv::Vec3d(1.35, -1.9, -2.9);
  const cv::Vec3d target(0.0, 0.15, 2.5);
  const cv::Vec3d up(0.0, -1.0, 0.0);  // 图像 y 向下，所以“上”是 -y

  const cv::Vec3d f = cv::normalize(target - v.eye);
  const cv::Vec3d r = cv::normalize(f.cross(up));
  const cv::Vec3d u = r.cross(f);
  v.rot = cv::Matx33d(r[0], r[1], r[2], u[0], u[1], u[2], f[0], f[1], f[2]);
  v.focal = panel.width / (2.0 * std::tan(24.0 * CV_PI / 180.0));
  v.center = {panel.width * 0.5F, panel.height * 0.56F};
  return v;
}

// 世界点 -> 面板局部坐标；返回 false 表示在相机后面。
bool project(const View3D& v, const cv::Vec3d& p, cv::Point& out) {
  const cv::Vec3d d = p - v.eye;
  const cv::Vec3d c(v.rot(0, 0) * d[0] + v.rot(0, 1) * d[1] + v.rot(0, 2) * d[2],
                    v.rot(1, 0) * d[0] + v.rot(1, 1) * d[1] + v.rot(1, 2) * d[2],
                    v.rot(2, 0) * d[0] + v.rot(2, 1) * d[1] + v.rot(2, 2) * d[2]);
  if (c[2] < 0.2) return false;
  out.x = static_cast<int>(v.center.x + v.focal * c[0] / c[2]);
  out.y = static_cast<int>(v.center.y - v.focal * c[1] / c[2]);
  return true;
}

void line3d(cv::Mat& img, const View3D& v, const cv::Vec3d& a, const cv::Vec3d& b,
            const cv::Scalar& color, int thickness = 1) {
  cv::Point pa;
  cv::Point pb;
  if (project(v, a, pa) && project(v, b, pb)) cv::line(img, pa, pb, color, thickness, cv::LINE_AA);
}

}  // namespace

Hud::Hud(const AppConfig& config) : config_(config) {
  imageCenter_ = {config_.camera.width * 0.5F, config_.camera.height * 0.5F};
}

// =============================================================================
// 主绘制
// =============================================================================
void Hud::draw(cv::Mat& img, const FrameResult& result, HudState& state) {
  const int W = img.cols;
  const int H = img.rows;

  // ---- 找主目标轨迹 ----
  const Track* primary = nullptr;
  for (const auto& t : result.tracks) {
    if (t.id == result.primaryTrack) primary = &t;
  }
  const bool locked = primary != nullptr && primary->confirmed && primary->misses == 0;
  const bool coasting = primary != nullptr && primary->confirmed && primary->misses > 0;

  // ===========================================================================
  // 1. 目标层：括号、角点、轨迹、距离标签
  // ===========================================================================
  for (const auto& t : result.tracks) {
    if (!t.confirmed) continue;
    const cv::Scalar color = (t.id == result.primaryTrack) ? (coasting ? kAmber : kCyan) : kGreen;
    brackets(img, t.box, color, t.id == result.primaryTrack);

    // 轨迹尾巴：越旧越暗。
    if (t.trail.size() > 2) {
      for (std::size_t i = 1; i < t.trail.size(); ++i) {
        const float k = static_cast<float>(i) / static_cast<float>(t.trail.size());
        cv::line(img, t.trail[i - 1], t.trail[i], color * (0.15 + 0.55 * k), 1, cv::LINE_AA);
      }
    }

    if (t.id == result.primaryTrack) {
      const cv::Point p(static_cast<int>(t.box.x) - 2, static_cast<int>(t.box.y) - 12);
      const std::string label = cv::format("TRK-%02d   NUM %s", std::max(1, t.displayId),
                                           t.number >= 0 ? std::to_string(t.number).c_str() : "-");
      int baseline = 0;
      const cv::Size sz =
          cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.55, 1, &baseline);
      alphaFill(img, cv::Rect(p.x - 4, p.y - sz.height - 3, sz.width + 10, sz.height + 8), kPanel, 0.72);
      text(img, label, p, 0.55, color, 1);
    }
  }

  // 主目标的角点/中心
  if (primary != nullptr && locked) {
    for (const auto& armor : result.armors) {
      const float dx = armor.center.x - (primary->box.x + primary->box.width * 0.5F);
      const float dy = armor.center.y - (primary->box.y + primary->box.height * 0.5F);
      if (dx * dx + dy * dy > 40.0F * 40.0F) continue;
      for (int i = 0; i < 4; ++i) {
        cv::circle(img, armor.corners[i], 3, kCyan, cv::FILLED, cv::LINE_AA);
        cv::line(img, armor.corners[i], armor.corners[(i + 1) % 4], kCyan * 0.8, 1, cv::LINE_AA);
      }
      cv::drawMarker(img, armor.center, kCyan, cv::MARKER_CROSS, 12, 1, cv::LINE_AA);
    }
  }

  // ===========================================================================
  // 2. 左侧数据面板
  // ===========================================================================
  const int lx = 24;
  const int lw = 296;

  // --- 性能 ---
  {
    const cv::Rect r(lx, 76, lw, 214);
    panel(img, r, kCyan);
    text(img, "PERFORMANCE", {r.x + 14, r.y + 21}, 0.52, kCyan, 1);

    num(img, fixed(state.fps, 1), {r.x + 14, r.y + 82}, 2.1, kWhite, 3);
    text(img, "ALGO FPS", {r.x + 222, r.y + 58}, 0.36, kGray, 1);
    text(img, "(EXCL. RENDER)", {r.x + 222, r.y + 76}, 0.3, kDim, 1);

    sparkline(img, cv::Rect(r.x + 14, r.y + 96, r.width - 28, 40), state.fpsHistory, 240.0, kCyan);

    const double budget = 1000.0 / 60.0;  // 16.7ms 一帧的预算
    const char* names[] = {"DET", "TRK", "POSE"};
    const double vals[] = {result.timing.detect, result.timing.track, result.timing.pose};
    for (int i = 0; i < 3; ++i) {
      const int y = r.y + 148 + i * 20;
      text(img, names[i], {r.x + 14, y}, 0.4, kGray, 1);
      bar(img, cv::Rect(r.x + 64, y - 9, 150, 7), static_cast<float>(vals[i] / budget),
          i == 0 ? kCyan : (i == 1 ? kAmber : kGreen));
      num(img, fixed(vals[i], 2) + " ms", {r.x + 220, y}, 0.4, kWhite, 1);
    }
    num(img, "TOTAL " + fixed(result.timing.total, 2) + " ms", {r.x + 14, r.y + 206}, 0.42, kGray);
  }

  // --- 检测 ---
  {
    const cv::Rect r(lx, 302, lw, 200);
    panel(img, r, kCyan);
    text(img, "ADAPTIVE DETECTION", {r.x + 14, r.y + 21}, 0.5, kCyan, 1);
    const bool relaxed = result.thresholdLevel == 1;
    const std::string modeText =
        (relaxed ? "THRESHOLD RELAXED" : "THRESHOLD STRICT") +
        std::string(result.relaxedGeometry ? " / GEOM RELAXED" : "");
    text(img, modeText, {r.x + 14, r.y + 46}, 0.38,
         (relaxed || result.relaxedGeometry) ? kAmber : kGreen, 1);

    text(img, "AMBIENT V", {r.x + 14, r.y + 72}, 0.4, kGray);
    num(img, fixed(result.ambient, 1), {r.x + 130, r.y + 72}, 0.45, kWhite);
    bar(img, cv::Rect(r.x + 200, r.y + 63, 80, 7), static_cast<float>(result.ambient / 255.0), kCyan);

    text(img, "S / V THR", {r.x + 14, r.y + 94}, 0.4, kGray);
    num(img, fixed(result.sThreshold, 0) + " / " + fixed(result.vThreshold, 0), {r.x + 130, r.y + 94},
        0.45, kWhite);

    text(img, "BARS  R/B", {r.x + 14, r.y + 116}, 0.4, kGray);
    num(img, std::to_string(result.redBars) + " / " + std::to_string(result.blueBars),
        {r.x + 130, r.y + 116}, 0.45, kWhite);

    text(img, "ROI", {r.x + 14, r.y + 138}, 0.4, kGray);
    num(img, result.usedRoi ? "ON  (TRACKING)" : "OFF (FULL FRAME)", {r.x + 130, r.y + 138}, 0.4,
        result.usedRoi ? kGreen : kAmber);

    text(img, result.ambient < 45 ? "LOW-LIGHT MODE" : "NORMAL MODE", {r.x + 14, r.y + 168}, 0.42,
         result.ambient < 45 ? kAmber : kGreen, 1);
    cv::circle(img, {r.x + r.width - 24, r.y + 163}, 5, result.ambient < 45 ? kAmber : kGreen,
               cv::FILLED, cv::LINE_AA);
  }

  // --- 跟踪 ---
  {
    const cv::Rect r(lx, 514, lw, 186);
    panel(img, r, kCyan);
    text(img, "TRACKING", {r.x + 14, r.y + 21}, 0.52, kCyan, 1);

    const std::string stateText = locked ? "LOCKED" : (coasting ? "COASTING" : "SEARCHING");
    const cv::Scalar stateColor = locked ? kGreen : (coasting ? kAmber : kRed);
    text(img, stateText, {r.x + 14, r.y + 50}, 0.55, stateColor, 1);

    if (primary != nullptr) {
      text(img, "ID", {r.x + 14, r.y + 78}, 0.4, kGray);
      num(img, cv::format("%02d", std::max(1, primary->displayId)), {r.x + 70, r.y + 78}, 0.45,
          kWhite);
      text(img, "NUM", {r.x + 120, r.y + 78}, 0.4, kGray);
      num(img, primary->number >= 0 ? std::to_string(primary->number) : "-", {r.x + 180, r.y + 78},
          0.45, kWhite);
      text(img, "HITS", {r.x + 14, r.y + 100}, 0.4, kGray);
      num(img, std::to_string(primary->hits), {r.x + 70, r.y + 100}, 0.45, kWhite);
      text(img, "VEL", {r.x + 120, r.y + 100}, 0.4, kGray);
      num(img, fixed(cv::norm(primary->velocity), 1) + " px/f", {r.x + 180, r.y + 100}, 0.4, kWhite);

      double jitter = 0.0;
      if (primary->zHistory.size() > 2) {
        double mean = 0.0;
        for (float z : primary->zHistory) mean += z;
        mean /= primary->zHistory.size();
        for (float z : primary->zHistory) jitter += (z - mean) * (z - mean);
        jitter = std::sqrt(jitter / primary->zHistory.size());
      }
      text(img, "Z JITTER", {r.x + 14, r.y + 122}, 0.4, kGray);
      num(img, fixed(jitter * 1000.0, 0) + " mm", {r.x + 100, r.y + 122}, 0.45, kWhite);
      text(img, "CONF", {r.x + 14, r.y + 146}, 0.4, kGray);
      bar(img, cv::Rect(r.x + 70, r.y + 137, 120, 7), primary->score, kCyan);
      num(img, fixed(primary->score * 100, 0) + "%", {r.x + 198, r.y + 146}, 0.4, kWhite);
    } else {
      text(img, "NO TARGET", {r.x + 14, r.y + 80}, 0.6, kRed, 1);
      text(img, "WAITING FOR DETECTION", {r.x + 14, r.y + 108}, 0.4, kGray);
    }
  }

  // --- 目标遥测 ---
  {
    const cv::Rect r(lx, 712, lw, 344);
    panel(img, r, kCyan);
    text(img, "TARGET TELEMETRY", {r.x + 14, r.y + 21}, 0.5, kCyan, 1);

    if (!result.armorChip.empty()) {
      const cv::Rect chip(r.x + 14, r.y + 38, 130, 52);
      cv::Mat small;
      cv::resize(result.armorChip, small, chip.size());
      small.copyTo(img(chip));
      cv::rectangle(img, chip, kCyan * 0.7, 1);
      if (primary != nullptr && primary->number >= 0) {
        num(img, std::to_string(primary->number), {r.x + 160, r.y + 82}, 2.2, kCyan, 3);
        text(img, "DIGIT", {r.x + 160, r.y + 38}, 0.4, kGray);
      }
    } else {
      text(img, "NO IMAGE", {r.x + 14, r.y + 70}, 0.5, kDim);
    }

    if (primary != nullptr && primary->pose.valid) {
      const Pose& p = primary->pose;
      const char* labels[] = {"X", "Y", "Z"};
      for (int i = 0; i < 3; ++i) {
        text(img, labels[i], {r.x + 14, r.y + 118 + i * 24}, 0.45, kGray);
        num(img, fixed(p.tvec[i], 2) + " m", {r.x + 46, r.y + 118 + i * 24}, 0.55, kWhite);
      }
      text(img, "DIST", {r.x + 160, r.y + 118}, 0.4, kGray);
      num(img, fixed(p.distance, 2) + " m", {r.x + 206, r.y + 118}, 0.5, kCyan, 2);

      text(img, "YAW", {r.x + 160, r.y + 142}, 0.4, kGray);
      num(img, fixed(p.yaw, 1) + " deg", {r.x + 206, r.y + 142}, 0.45, kWhite);
      text(img, "PITCH", {r.x + 160, r.y + 166}, 0.4, kGray);
      num(img, fixed(p.pitch, 1) + " deg", {r.x + 206, r.y + 166}, 0.45, kWhite);

      text(img, "REPROJ", {r.x + 14, r.y + 206}, 0.4, kGray);
      num(img, fixed(p.reprojError, 2) + " px", {r.x + 90, r.y + 206}, 0.45,
          p.reprojError < 1.5 ? kGreen : kAmber);
      text(img, "CONF", {r.x + 160, r.y + 206}, 0.4, kGray);
      num(img, fixed(primary->score * 100, 0) + "%", {r.x + 206, r.y + 206}, 0.45, kWhite);

      bar(img, cv::Rect(r.x + 14, r.y + 224, 266, 8), static_cast<float>(p.distance / 8.0), kCyan);
      for (int i = 1; i <= 4; ++i) {
        const int x = r.x + 14 + static_cast<int>(266 * (i * 2.0 / 8.0));
        cv::line(img, {x, r.y + 224}, {x, r.y + 232}, kDim, 1);
        num(img, std::to_string(i * 2), {x - 4, r.y + 246}, 0.32, kDim);
      }
      text(img, "RANGE (m)", {r.x + 14, r.y + 268}, 0.38, kGray);
      text(img, "CAMERA FRAME / SOLVEPNP IPPE + LM", {r.x + 14, r.y + 300}, 0.34, kDim);
    } else {
      text(img, "POSE UNAVAILABLE", {r.x + 14, r.y + 120}, 0.5, kDim);
    }
  }

  // ===========================================================================
  // 3. 右侧三维定位面板
  // ===========================================================================
  {
    const cv::Rect r(W - 520, 76, 496, 386);
    panel(img, r, kCyan);
    text(img, "3D LOCALIZATION", {r.x + 14, r.y + 21}, 0.5, kCyan, 1);
    text(img, "CAMERA FRAME / PnP", {r.x + 220, r.y + 21}, 0.34, kDim);

    cv::Mat view(r.height - 30, r.width, CV_8UC3, cv::Scalar(10, 14, 16));
    const View3D v = makeView(cv::Rect(0, 0, r.width, r.height - 30));

    const double groundY = 0.42;
    for (int z = 1; z <= 8; ++z) {
      line3d(view, v, cv::Vec3d(-3.2, groundY, z), cv::Vec3d(3.2, groundY, z),
             cv::Scalar(28, 42, 46));
    }
    for (int x = -3; x <= 3; ++x) {
      line3d(view, v, cv::Vec3d(x, groundY, 1), cv::Vec3d(x, groundY, 8), cv::Scalar(28, 42, 46));
    }

    const double fovHalf = std::tan(32.0 * CV_PI / 180.0);
    for (int s = -1; s <= 1; s += 2) {
      line3d(view, v, cv::Vec3d(0, 0, 0), cv::Vec3d(s * fovHalf * 6.0, 0, 6.0),
             cv::Scalar(70, 95, 100));
    }

    for (int ring = 2; ring <= 6; ring += 2) {
      std::vector<cv::Point> pts;
      for (int a = 0; a <= 360; a += 12) {
        const double rad = a * CV_PI / 180.0;
        cv::Point p;
        if (project(v, cv::Vec3d(ring * std::cos(rad), groundY, ring * std::sin(rad)), p)) {
          pts.push_back(p);
        }
      }
      if (pts.size() > 2) {
        cv::polylines(view, pts, false, cv::Scalar(45, 70, 74), 1, cv::LINE_AA);
        cv::Point lbl;
        if (project(v, cv::Vec3d(0, groundY, ring), lbl)) {
          text(view, std::to_string(ring) + "m", lbl + cv::Point(4, 12), 0.3, kDim);
        }
      }
    }

    cv::Point camPt;
    if (project(v, cv::Vec3d(0, 0, 0), camPt)) {
      cv::circle(view, camPt, 4, kCyan, cv::FILLED, cv::LINE_AA);
      text(view, "CAM", camPt + cv::Point(7, 4), 0.32, kCyan);
    }

    if (primary != nullptr && primary->pose.valid) {
      const Pose& pose = primary->pose;
      cv::Matx33d R;
      cv::Rodrigues(pose.rvec, R);
      const double w = 0.135;
      const double h = 0.055;
      std::vector<cv::Vec3d> corners;
      for (int i = 0; i < 4; ++i) {
        const double sx = (i == 0 || i == 3) ? -w / 2 : w / 2;
        const double sy = (i < 2) ? -h / 2 : h / 2;
        corners.push_back(cv::Vec3d(R * cv::Vec3d(sx, sy, 0.0)) + pose.tvec);
      }
      std::vector<cv::Point> quad;
      bool ok = true;
      for (const auto& c : corners) {
        cv::Point p;
        if (!project(v, c, p)) {
          ok = false;
          break;
        }
        quad.push_back(p);
      }
      if (ok) {
        cv::Mat overlay = view.clone();
        cv::fillConvexPoly(overlay, quad, kCyan, cv::LINE_AA);
        cv::addWeighted(overlay, 0.30, view, 0.70, 0.0, view);
        cv::polylines(view, quad, true, kCyan, 1, cv::LINE_AA);
      }
      line3d(view, v, cv::Vec3d(0, 0, 0), pose.tvec, kCyan * 0.55, 1);
      line3d(view, v, pose.tvec, cv::Vec3d(pose.tvec[0], groundY, pose.tvec[2]),
             cv::Scalar(60, 90, 95));
      cv::Point tp;
      if (project(v, pose.tvec, tp)) {
        // 装甲板只有 135mm，在 2~5m 的三维视图里非常小，
        // 额外画一个锁定环 + 距离标签，肉眼才跟得上。
        cv::circle(view, tp, 14, kCyan, 1, cv::LINE_AA);
        cv::circle(view, tp, 22, kCyan * 0.5, 1, cv::LINE_AA);
        cv::drawMarker(view, tp, kCyan, cv::MARKER_CROSS, 10, 1, cv::LINE_AA);
        text(view, fixed(pose.distance, 2) + " m", tp + cv::Point(26, -16), 0.42, kCyan, 1);
        text(view, "TARGET", tp + cv::Point(26, 2), 0.32, kGray, 1);
      }

      if (state.poseTrail.size() > 2) {
        for (std::size_t i = 1; i < state.poseTrail.size(); ++i) {
          const float k = static_cast<float>(i) / state.poseTrail.size();
          line3d(view, v, state.poseTrail[i - 1], state.poseTrail[i], kCyan * (0.15 + 0.55 * k), 1);
        }
      }
    }

    line3d(view, v, cv::Vec3d(0, 0, 0), cv::Vec3d(0.6, 0, 0), cv::Scalar(70, 70, 230), 2);
    line3d(view, v, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0.6, 0), cv::Scalar(90, 220, 90), 2);
    line3d(view, v, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0.6), cv::Scalar(240, 180, 60), 2);

    view.copyTo(img(cv::Rect(r.x, r.y + 30, r.width, r.height - 30)));
    cv::rectangle(img, r, kCyan * 0.5, 1);
  }

  // ===========================================================================
  // 4. 右下流水线 + 底部时间轴
  // ===========================================================================
  {
    const cv::Rect r(W - 320, 476, 296, 210);
    panel(img, r, kCyan);
    text(img, "PIPELINE", {r.x + 14, r.y + 21}, 0.5, kCyan, 1);
    const double vals[] = {result.timing.detect, result.timing.track, result.timing.pose};
    const char* names[] = {"DETECT", "TRACK", "POSE"};
    const cv::Scalar colors[] = {kCyan, kAmber, kGreen};
    for (int i = 0; i < 3; ++i) {
      const int y = r.y + 52 + i * 30;
      text(img, names[i], {r.x + 14, y}, 0.4, kGray);
      const int w = static_cast<int>(std::clamp(vals[i] / 15.0, 0.0, 1.0) * 150);
      cv::rectangle(img, cv::Rect(r.x + 84, y - 9, w, 8), colors[i], cv::FILLED);
      num(img, fixed(vals[i], 2), {r.x + 244, y}, 0.42, kWhite);
    }
    text(img, "FRAME BUDGET 16.7 ms", {r.x + 14, r.y + 156}, 0.34, kDim);
    text(img, config_.enableDigits ? "DIGIT RECOGNITION ON" : "DIGIT RECOGNITION OFF",
         {r.x + 14, r.y + 180}, 0.34, config_.enableDigits ? kGreen : kDim);
  }

  {
    const cv::Rect r(lx + lw + 16, H - 50, W - 2 * (lx + lw) - 32, 36);
    alphaFill(img, r, kPanel, 0.55);
    text(img, "DETECTION TIMELINE", {r.x + 10, r.y + 12}, 0.3, kDim);
    const int n = static_cast<int>(state.successHistory.size());
    const int w = r.width - 20;
    for (int i = 0; i < n; ++i) {
      const int x = r.x + 10 + static_cast<int>(static_cast<double>(i) / std::max(1, n - 1) * w);
      const bool ok = state.successHistory[static_cast<std::size_t>(i)] > 0.5F;
      cv::line(img, {x, r.y + 18}, {x, r.y + 32}, ok ? kGreen : kRed, 1);
    }
    cv::rectangle(img, r, kDim, 1);
  }

  // ===========================================================================
  // 5. 顶栏
  // ===========================================================================
  {
    const cv::Rect barRect(0, 0, W, 56);
    alphaFill(img, barRect, kPanel, 0.72);
    cv::line(img, {0, 55}, {W, 55}, kCyan * 0.6, 1);

    spacedText(img, "AEGIS VISION", {24, 36}, 0.62, kWhite, 2, 3);
    text(img, "ADAPTIVE ARMOR DETECTION / 3D LOCALIZATION", {250, 36}, 0.36, kGray);

    const std::string mode = config_.baselineMode ? "BASELINE / FIXED" : "ADAPTIVE / ONLINE";
    const cv::Scalar modeColor = config_.baselineMode ? kAmber : kGreen;
    text(img, mode, {W - 340, 33}, 0.4, modeColor, 1);
    cv::circle(img, {W - 356, 28}, 4, modeColor, cv::FILLED);
    text(img, "FRAME " + std::to_string(state.frameIndex), {W - 180, 33}, 0.4, kGray);
    if (primary != nullptr) {
      text(img, cv::format("ID %02d", std::max(1, primary->displayId)), {W - 90, 33}, 0.4, kCyan);
    }
  }
}

// =============================================================================
// 片头
// =============================================================================
void Hud::drawIntro(cv::Mat& img, double progress) const {
  cv::Mat dark;
  img.convertTo(dark, img.type(), 0.22);
  dark.copyTo(img);
  const int W = img.cols;
  const int H = img.rows;

  const double p = std::clamp(progress, 0.0, 1.0);
  for (int y = 0; y < H; y += 4) cv::line(img, {0, y}, {W, y}, cv::Scalar(0, 0, 0), 1);

  const int cx = W / 2;
  const int baseY = H / 2 - 40;
  spacedText(img, "AEGIS VISION", {cx - 300, baseY}, 1.5, kWhite, 3, 8);
  text(img, "ADAPTIVE ARMOR DETECTION & 3D LOCALIZATION", {cx - 300, baseY + 44}, 0.65, kCyan);
  cv::line(img, {cx - 300, baseY + 66}, {cx - 300 + static_cast<int>(600 * p), baseY + 66}, kCyan, 2);

  const char* lines[] = {"01   RED / BLUE ADAPTIVE DETECTION", "02   MULTI-FRAME STABLE TRACKING",
                         "03   PnP 6-DOF LOCALIZATION", "04   REAL-TIME PERFORMANCE PANEL"};
  for (int i = 0; i < 4; ++i) {
    if (p * 4.0 < i + 0.15) continue;
    const int y = baseY + 120 + i * 38;
    text(img, lines[i], {cx - 300, y}, 0.52, i == 0 ? kCyan : kGray, 1);
  }
  text(img, "ROBOMASTER 2027  /  VISUAL GROUP", {cx - 300, H - 80}, 0.45, kDim);
}

// =============================================================================
// 片尾
// =============================================================================
void Hud::drawOutro(cv::Mat& img, double progress, const HudState& state) const {
  cv::Mat dark;
  img.convertTo(dark, img.type(), 0.18);
  dark.copyTo(img);
  const int W = img.cols;
  const int H = img.rows;
  const double p = std::clamp(progress, 0.0, 1.0);
  const int cx = W / 2;

  spacedText(img, "MISSION SUMMARY", {cx - 290, 250}, 0.85, kWhite, 2, 4);
  cv::line(img, {cx - 290, 274}, {cx - 290 + static_cast<int>(580 * p), 274}, kCyan, 2);

  const double rate = state.framesAfterLock > 0
                          ? 100.0 * state.detectedAfterLock / state.framesAfterLock
                          : (state.totalFrames > 0
                                 ? 100.0 * state.detectedFrames / state.totalFrames
                                 : 0.0);
  const char* labels[] = {"FRAMES", "ALGO FPS", "DETECTION RATE", "AVG LATENCY"};
  const std::string values[] = {std::to_string(state.totalFrames), fixed(state.avgFps, 1),
                                fixed(rate, 2) + " %", fixed(state.avgLatencyMs, 2) + " ms"};
  for (int i = 0; i < 4; ++i) {
    const int y = 350 + i * 62;
    text(img, labels[i], {cx - 290, y}, 0.5, kGray);
    num(img, values[i], {cx + 60, y}, 0.8, kCyan, 2);
  }
  text(img, "AEGIS VISION  /  ROBOMASTER 2027", {cx - 290, H - 140}, 0.5, kDim);
}

}  // namespace aegis
