#include "autoaim/armor_corners.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>

// ===========================================================================
// 本文件：「四角点外推」——用直线求交代替直接取灯条端点
//
// 问题：
//   最朴素的做法是把两根灯条的端点直接当装甲板的四个角。这四个数是四次
//   独立测量，任何一次抖动（灯条末端被遮挡、JPEG 把边缘压断、少亮一颗灯珠）
//   都会原封不动地进入 PnP，直接变成位姿误差。
//
// 做法（直线约束，让每个角由两条线共同决定）：
//
//        TL ─────────── TR        ← 上边界线：过两灯条"外缘顶点"的连线，
//        │              │            方向强制垂直于两根灯条方向的平均值
//        │   数字区域    │            （板子是矩形，上下边必然垂直于灯条）
//        │              │
//        BL ─────────── BR        ← 下边界线：同理
//        ↑              ↑
//     左灯条外缘线    右灯条外缘线
//     （过灯条中心 + 外法线×半宽，方向 = 该灯条自身轴向）
//
//   四个角点 = 左右两条"外缘线" × 上下两条"边界线"的交点。
//   于是每个角点被两条完整的直线约束，而不是被一个噪声点决定。
//
// 两个刻意的设计选择：
//
//   [1] 左右两条边线各用自己的灯条方向，不做"强制平行"
//       如果强行让两边平行，四边形就成了平行四边形，而真实透视投影下
//       矩形会呈现为一般四边形（近大远小，两边会收敛）。
//       强加平行会引入系统偏差，尤其在板子偏航时。所以保留各自方向。
//
//   [2] 上下边界线的方向用两根灯条方向的"平均值"的垂线
//       上下边必须垂直于灯条，这是矩形这一物理事实；用平均值可以把
//       两根灯条各自的几度角度噪声平均掉一半。
//
// edge_offset 的作用（显示风格可调，不影响算法）：
//   +1 = 边线落在灯条"外缘"（板宽 = 外缘到外缘，物理上 135 mm 的标准定义）
//    0 = 落在灯条中心线
//   -1 = 落在灯条"内缘"（板宽 = 两灯条之间的空隙）—— 视觉上更像"夹住数字"
//
// 亚像素精化 refineArmorCornersSubPix()：
//   在色差图上用 cornerSubPix 把角点细化到亚像素。这些角点正好落在灯条的
//   高对比边界上，是 cornerSubPix 最擅长的情况。实测重投影误差从约 1.0 px
//   降到 0.44 px。如果某个角点移动超过 max_shift_px（说明锁到了错误的边），
//   整个精化结果被丢弃，保留原角点 —— 宁可不动，不能动错。
// ===========================================================================

namespace autoaim {
namespace {

struct Line2D {
  cv::Point2f p;  // a point on the line
  cv::Point2f d;  // unit direction
};

/// Geometry of one light bar, derived from its RotatedRect.
struct BarAxes {
  cv::Point2f c;    // centre
  cv::Point2f u;    // long-axis unit vector, pointing "up" (u.y <= 0)
  cv::Point2f v;    // short-axis unit vector (perpendicular to u)
  float half_len;   // half the long side
  float half_wid;   // half the short side
};

/// Extract the long/short axes from a RotatedRect WITHOUT trusting its `angle`.
///
/// OpenCV's RotatedRect angle convention changed between 4.x releases (and whether
/// size.width follows the angle or the angle follows size.width is a classic
/// source of silent 90-degree errors). Rebuilding the axes from the four corner
/// points is convention independent: whichever edge is longer IS the long axis.
bool axesFromRect(const cv::RotatedRect& r, BarAxes* out) {
  if (out == nullptr) return false;
  cv::Point2f pts[4];
  r.points(pts);

  const cv::Point2f e0 = pts[1] - pts[0];
  const cv::Point2f e1 = pts[2] - pts[1];
  const float l0 = std::sqrt(e0.x * e0.x + e0.y * e0.y);
  const float l1 = std::sqrt(e1.x * e1.x + e1.y * e1.y);
  if (l0 < 1e-6f || l1 < 1e-6f) return false;

  cv::Point2f u;
  float len = 0.f, wid = 0.f;
  if (l0 >= l1) {
    u = e0 / l0;
    len = l0;
    wid = l1;
  } else {
    u = e1 / l1;
    len = l1;
    wid = l0;
  }
  if (u.y > 0.f) u = -u;  // image y grows downwards; make u point "up"

  out->c = r.center;
  out->u = u;
  out->v = cv::Point2f(-u.y, u.x);
  out->half_len = 0.5f * len;
  out->half_wid = 0.5f * wid;
  return true;
}

/// Intersection of two lines. Returns false when they are (nearly) parallel.
bool intersectLines(const Line2D& a, const Line2D& b, cv::Point2f* out) {
  const double den = static_cast<double>(a.d.x) * b.d.y - static_cast<double>(a.d.y) * b.d.x;
  if (std::fabs(den) < 1e-4) return false;  // < ~0.006 deg apart
  const double px = static_cast<double>(b.p.x) - a.p.x;
  const double py = static_cast<double>(b.p.y) - a.p.y;
  const double t = (px * b.d.y - py * b.d.x) / den;
  if (!std::isfinite(t)) return false;
  *out = cv::Point2f(static_cast<float>(a.p.x + a.d.x * t),
                     static_cast<float>(a.p.y + a.d.y * t));
  return true;
}

/// True when the four points form a convex, clockwise, non-self-intersecting quad.
bool isConvexClockwise(const std::vector<cv::Point2f>& p) {
  if (p.size() != 4) return false;
  double sign = 0.0;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f a = p[i];
    const cv::Point2f b = p[(i + 1) % 4];
    const cv::Point2f c = p[(i + 2) % 4];
    const double cross = static_cast<double>(b.x - a.x) * (c.y - b.y) -
                         static_cast<double>(b.y - a.y) * (c.x - b.x);
    if (std::fabs(cross) < 1e-6) return false;  // collinear
    if (i == 0) {
      sign = cross > 0 ? 1.0 : -1.0;
    } else if ((cross > 0 ? 1.0 : -1.0) != sign) {
      return false;  // sign flip => non-convex / self-intersecting
    }
  }
  return true;
}

}  // namespace

bool armorCornersFromLightBars(const cv::RotatedRect& left, const cv::RotatedRect& right,
                               std::vector<cv::Point2f>* corners, float edge_offset) {
  if (corners == nullptr) return false;
  corners->clear();

  BarAxes L, R;
  if (!axesFromRect(left, &L) || !axesFromRect(right, &R)) return false;
  if (L.half_wid < 0.5f || R.half_wid < 0.5f) return false;

  // ---- 1. outer normal of each bar (pointing away from the partner) ---------
  cv::Point2f nL = L.v;
  cv::Point2f nR = R.v;
  if ((R.c - L.c).dot(nL) > 0.f) nL = -nL;
  if ((L.c - R.c).dot(nR) > 0.f) nR = -nR;

  // ---- 2. outer edge line of each bar: through (centre + n*half_width) ------
  // Direction is the bar's own long axis, so perspective is preserved and the
  // two edges are free to converge as the real projection does.
  // edge_offset scales the sideways offset of each edge line: +1 outer side,
  // 0 bar centre, -1 inner side. This is what makes the box hug the inside of the
  // bars instead of the outside.
  const Line2D edgeL{L.c + nL * (L.half_wid * edge_offset), L.u};
  const Line2D edgeR{R.c + nR * (R.half_wid * edge_offset), R.u};

  // ---- 3. the plate's outer top / bottom points on those edges --------------
  const cv::Point2f lt = edgeL.p + L.u * L.half_len;
  const cv::Point2f rt = edgeR.p + R.u * R.half_len;
  const cv::Point2f lb = edgeL.p - L.u * L.half_len;
  const cv::Point2f rb = edgeR.p - R.u * R.half_len;

  // ---- 4. top / bottom boundary lines --------------------------------------
  // A plate is a rectangle, so its top and bottom edges are perpendicular to the
  // light bars. Using the MEAN of the two bar directions halves the angle noise,
  // while each outer edge keeps its own direction above, so no parallel
  // constraint is imposed on the left/right sides.
  cv::Point2f u_mean = L.u + R.u;
  const float un = std::sqrt(u_mean.x * u_mean.x + u_mean.y * u_mean.y);
  if (un < 0.2f) return false;  // bars pointing opposite ways: nonsense pair
  u_mean /= un;
  const cv::Point2f v_mean(-u_mean.y, u_mean.x);  // perpendicular

  const cv::Point2f top_mid = 0.5f * (lt + rt);
  const cv::Point2f bot_mid = 0.5f * (lb + rb);
  const Line2D topLine{top_mid, v_mean};
  const Line2D botLine{bot_mid, v_mean};

  // ---- 5. four corners = intersections ------------------------------------
  cv::Point2f TL, TR, BR, BL;
  if (!intersectLines(edgeL, topLine, &TL)) return false;
  if (!intersectLines(edgeR, topLine, &TR)) return false;
  if (!intersectLines(edgeR, botLine, &BR)) return false;
  if (!intersectLines(edgeL, botLine, &BL)) return false;

  std::vector<cv::Point2f> quad{TL, TR, BR, BL};

  // A corner that flew far away from its own bar means the lines were nearly
  // parallel: treat as a failed construction rather than returning garbage.
  const float span = static_cast<float>(cv::norm(L.c - R.c)) + L.half_len + R.half_len;
  const float lim = 1.5f * span;
  for (const auto& q : quad) {
    if (cv::norm(q - L.c) > lim && cv::norm(q - R.c) > lim) return false;
  }
  if (!isConvexClockwise(quad)) return false;

  *corners = quad;
  return true;
}

bool refineArmorCornersSubPix(const cv::Mat& image, std::vector<cv::Point2f>* corners,
                              int win, int max_iters, double eps, double max_shift_px) {
  if (corners == nullptr || corners->size() != 4) return false;
  if (image.empty()) return false;

  cv::Mat gray;
  if (image.channels() > 1) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }
  if (gray.depth() != CV_8U) {
    cv::Mat tmp;
    gray.convertTo(tmp, CV_8U);
    gray = tmp;
  }

  // cornerSubPix requires the whole window inside the image; skip if any corner
  // is too close to the border rather than letting OpenCV throw.
  const int margin = std::max(win, 2);
  for (const auto& p : *corners) {
    if (p.x < margin || p.y < margin || p.x > gray.cols - 1 - margin ||
        p.y > gray.rows - 1 - margin) {
      return false;
    }
  }

  const std::vector<cv::Point2f> before = *corners;
  std::vector<cv::Point2f> after = before;
  cv::cornerSubPix(gray, after, cv::Size(win, win), cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                    max_iters, eps));

  for (size_t i = 0; i < after.size(); ++i) {
    if (cv::norm(after[i] - before[i]) > static_cast<float>(max_shift_px)) {
      return false;  // locked onto the wrong edge; keep the original corners
    }
    if (!std::isfinite(after[i].x) || !std::isfinite(after[i].y)) return false;
  }
  *corners = after;
  return true;
}

void armorCenterFromCorners(const std::vector<cv::Point2f>& corners,
                            cv::Point2f* center) {
  if (center == nullptr) return;
  if (corners.empty()) {
    *center = cv::Point2f(0.f, 0.f);
    return;
  }
  cv::Point2f sum(0.f, 0.f);
  for (const auto& c : corners) sum += c;
  *center = sum * (1.f / static_cast<float>(corners.size()));
}

}  // namespace autoaim
