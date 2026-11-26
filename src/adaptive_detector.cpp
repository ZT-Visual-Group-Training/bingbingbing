// =============================================================================
// adaptive_detector.cpp —— 特色一实现
//
// 传统装甲板检测的经典流程是：
//   颜色阈值 -> 形态学 -> 找灯条 -> 两两配对 -> 角点
// 这套流程在“实验室白天”很好用，但一放到低照度/强光/金属反光里就会崩：
//   * 固定 V 阈值：暗了漏检，亮了满屏噪点；
//   * 固定 S 阈值：灯条过曝后中心变白、饱和度掉下去，反而检不到；
//   * 环境里的红色尾灯、蓝色窗户会混进候选。
//
// 本文件的改进点（答辩时按这四条讲）：
//   1. 红蓝双通道同时建模，红色跨 0° 用两段 hue 合并；
//   2. 用整帧 V 直方图的分位数 + 高亮像素饱和度，反推本帧阈值（自适应）；
//   3. 严格/放宽两级阈值 + 跟踪反馈（跟踪丢失时自动放宽）；
//   4. 灯条几何 + 装甲板宽高比 + 填充率 + 共线性的多约束配对，
//      再用 NMS 去掉同一块板子被重复配对出来的候选。
// =============================================================================
#include "aegis/adaptive_detector.hpp"

#include <algorithm>`n#include <cstdio>`n#include <cstdlib>
#include <cmath>
#include <limits>

namespace aegis {
namespace {

constexpr double kPi = 3.14159265358979323846;

double toDeg(double rad) { return rad * 180.0 / kPi; }

// 把角度规范到 [-90, 90]，方便比较灯条倾角。
double normalizeAngle(double deg) {
  while (deg > 90.0) deg -= 180.0;
  while (deg < -90.0) deg += 180.0;
  return deg;
}

// 计算两个矩形的 IoU（交并比）。NMS 和跟踪器都会用。
float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float w = std::max(0.0F, x2 - x1);
  const float h = std::max(0.0F, y2 - y1);
  const float inter = w * h;
  const float uni = a.area() + b.area() - inter;
  return uni > 1e-6F ? inter / uni : 0.0F;
}

// 取旋转矩形的长边端点。OpenCV 的 RotatedRect 角度语义在 4.x/5.x 有差异，
// 直接比较 4 条边长度最稳，不依赖 angle 的定义。
void longAxisEndpoints(const cv::RotatedRect& rr, cv::Point2f& top, cv::Point2f& bottom,
                       float& length, float& width) {
  cv::Point2f p[4];
  rr.points(p);
  float bestLen = -1.0F;
  cv::Point2f bestA, bestB;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f a = p[i];
    const cv::Point2f b = p[(i + 1) % 4];
    const float len = static_cast<float>(cv::norm(b - a));
    if (len > bestLen) {
      bestLen = len;
      bestA = a;
      bestB = b;
    }
  }
  // 让 top 的 y 更小（图像坐标 y 向下），保证角点顺序统一。
  if (bestA.y > bestB.y) std::swap(bestA, bestB);
  top = bestA;
  bottom = bestB;
  length = bestLen;
  width = static_cast<float>(rr.size.area() / std::max(1.0F, bestLen));
}

// 点到直线的有符号距离，用于“两灯条是否共线/等高”的软约束。
float signedDy(const cv::Point2f& a, const cv::Point2f& b) { return b.y - a.y; }

// 取灯条“朝外”的那条长边的两个端点。
// 为什么要专门取外沿：角点如果取在灯条内侧，四点透视变换会把装甲板压窄，
// 数字会整体偏向一边甚至被裁掉（调试时踩过这个坑）。
void outerEdgeEndpoints(const LightBar& bar, const cv::Point2f& otherCenter, cv::Point2f& top,
                        cv::Point2f& bottom) {
  cv::Point2f p[4];
  bar.rect.points(p);
  const float e01 = static_cast<float>(cv::norm(p[1] - p[0]));
  const float e12 = static_cast<float>(cv::norm(p[2] - p[1]));

  // 找出两条“长边”，取离另一根灯条更远的那条（也就是装甲板外沿）。
  int a0 = 0;
  int a1 = 1;
  int b0 = 2;
  int b1 = 3;
  if (e12 > e01) {
    a0 = 1;
    a1 = 2;
    b0 = 3;
    b1 = 0;
  }
  const cv::Point2f ma = 0.5F * (p[a0] + p[a1]);
  const cv::Point2f mb = 0.5F * (p[b0] + p[b1]);
  const int i0 = (cv::norm(ma - otherCenter) >= cv::norm(mb - otherCenter)) ? a0 : b0;
  const int i1 = (i0 == a0) ? a1 : b1;

  top = (p[i0].y <= p[i1].y) ? p[i0] : p[i1];
  bottom = (p[i0].y <= p[i1].y) ? p[i1] : p[i0];
}

// 四个角点的外接矩形（浮点版，避免 boundingRect 把点取整后 IoU 抖动）。
cv::Rect2f bboxOf(const std::array<cv::Point2f, 4>& pts) {
  float minX = pts[0].x;
  float minY = pts[0].y;
  float maxX = pts[0].x;
  float maxY = pts[0].y;
  for (const auto& p : pts) {
    minX = std::min(minX, p.x);
    minY = std::min(minY, p.y);
    maxX = std::max(maxX, p.x);
    maxY = std::max(maxY, p.y);
  }
  return {minX, minY, maxX - minX, maxY - minY};
}

}  // namespace

AdaptiveDetector::AdaptiveDetector(const CameraConfig& camera, const DetectorConfig& params)
    : camera_(camera), params_(params) {}

void AdaptiveDetector::updateAdaptiveThresholds(const cv::Mat& hsv) {
  // 传入的是整帧降采样后的 HSV（约 180×135），不是 ROI——这样即使跟踪框
  // 停在暗处，测到的也是整幅画面的环境亮度，阈值不会来回跳。
  cv::Mat v;
  cv::extractChannel(hsv, v, 2);  // HSV 的第 2 通道就是 V
  const cv::Mat& small = v;
  const double meanV = cv::mean(small)[0];

  int hist[256] = {0};
  for (int y = 0; y < small.rows; ++y) {
    const uchar* row = small.ptr<uchar>(y);
    for (int x = 0; x < small.cols; ++x) ++hist[row[x]];
  }
  const long total = static_cast<long>(small.total());
  long acc = 0;
  int p99 = 255;
  int p999 = 255;
  for (int i = 0; i < 256; ++i) {
    acc += hist[i];
    if (acc >= total * 0.99 && p99 == 255) p99 = i;
    if (acc >= total * 0.999) {
      p999 = i;
      break;
    }
  }

  // ---- 2. 高亮像素的平均饱和度：判断灯条是否过曝 ----
  cv::Mat brightMask;
  const int brightThr = std::max(120, p99);
  cv::inRange(v, brightThr, 255, brightMask);
  double brightS = 190.0;
  if (cv::countNonZero(brightMask) > 40) {
    brightS = cv::mean(hsv, brightMask)[1];
  }

  // ---- 3. 由统计量反推阈值 ----
  // 环境越暗，阈值越低；这是“自适应”最直观的一层。
  const double k = std::clamp(meanV / 90.0, 0.45, 1.40);
  double vBase = params_.vMinBase * (0.70 + 0.30 * k);
  double sBase = params_.sMinBase * (0.80 + 0.20 * k);

  // 过曝补偿：灯条中心被打成纯白时 S 会掉到 100 以下，
  // 此时如果还坚持 S>=150 就会“灯条断成两截”，所以主动降 S。
  const double overexposed = std::clamp((175.0 - brightS) / 100.0, 0.0, 0.65);
  sBase *= (1.0 - 0.38 * overexposed);

  // 记录到 stats_，HUD 上会显示这些数字，答辩时可以指着讲。
  stats_.ambientV = meanV;
  stats_.p999V = p999;
  stats_.brightS = brightS;

  // 基线模式：不做任何自适应，固定使用配置里的阈值。
  // 这是消融实验的对照组，用来量化“自适应”到底带来了多少提升。
  if (params_.fixedThreshold) {
    stats_.sThreshold = params_.sMinBase;
    stats_.vThreshold = params_.vMinBase;
    return;
  }

  // 严格档：宁可漏检，也不把背景算进来。
  if (level_ == 0) {
    stats_.sThreshold = std::clamp(sBase * 1.08, 85.0, 225.0);
    stats_.vThreshold = std::clamp(vBase * 1.10, 70.0, 205.0);
  } else {
    // 放宽档：跟踪已经丢失时，用更低的门限把目标“捞”回来，
    // 代价是候选变多，后面靠几何配对和跟踪器兜底。
    stats_.sThreshold = std::clamp(sBase * 0.74, 70.0, 210.0);
    stats_.vThreshold = std::clamp(vBase * 0.70, 55.0, 190.0);
  }
}

cv::Mat AdaptiveDetector::makeMask(const cv::Mat& hsv, Color color, int level) const {
  // level 只影响 stats_ 里已经算好的阈值，这里不再重复计算。
  const double s = stats_.sThreshold;
  const double v = stats_.vThreshold;
  cv::Mat mask;
  if (color == Color::Red) {
    // 红色 hue 在 0° 附近跨 wrap-around，必须两段取并集，否则会漏掉偏橙的灯条。
    cv::Mat low;
    cv::Mat high;
    cv::inRange(hsv, cv::Scalar(0, s, v), cv::Scalar(15, 255, 255), low);
    cv::inRange(hsv, cv::Scalar(160, s, v), cv::Scalar(180, 255, 255), high);
    cv::bitwise_or(low, high, mask);
  } else {
    // 蓝色集中在 100~135°，不跨边界，一段就够。
    cv::inRange(hsv, cv::Scalar(95, s, v), cv::Scalar(140, 255, 255), mask);
  }

  // ---- 过曝核心补偿（本项目的关键改动之一）----
  // 高亮度 LED 在 2ms 曝光下中心会直接打到 255（纯白），HSV 里 S 骤降，
  // 只靠颜色阈值会把灯条挖成“空心”。这里额外取一路“高亮低饱和”掩码，
  // 但只在颜色掩码附近生效，避免把窗户、灯管等白色高光也算进来。
  // 基线对比时把 coreCompensation 关掉，就能量出这一路到底值多少。
  if (params_.coreCompensation) {
    cv::Mat core;
    cv::inRange(hsv, cv::Scalar(0, 18, 215), cv::Scalar(180, 255, 255), core);
    cv::Mat ring;
    cv::dilate(mask, ring, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
    cv::bitwise_and(core, ring, core);
    cv::bitwise_or(mask, core, mask);
  }

  // 形态学：先竖向闭运算把灯条内部的暗缝连起来，再开运算去掉孤立噪点。
  const cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 7));
  const cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, closeKernel);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, openKernel);
  (void)level;
  return mask;
}

std::vector<LightBar> AdaptiveDetector::findBars(cv::Mat mask, Color color,
                                                double maxAngle) const {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<LightBar> bars;
  bars.reserve(contours.size());
  for (const auto& contour : contours) {
    if (contour.size() < 5) continue;  // 至少 5 个点才能拟合旋转矩形
    const double area = cv::contourArea(contour);
    if (area < 12.0) continue;

    LightBar bar;
    bar.rect = cv::minAreaRect(contour);
    float length = 0.0F;
    float width = 0.0F;
    longAxisEndpoints(bar.rect, bar.top, bar.bottom, length, width);
    if (length < 8.0F || width < 1.0F) continue;

    const double ratio = length / std::max(1.0F, width);
    const double fill = area / std::max(1.0F, length * width);

    // 几何过滤：细长、够长、填充率高、别太歪。
    if (length < params_.minBarLength || length > params_.maxBarLength) continue;
    if (ratio < params_.minBarRatio || ratio > params_.maxBarRatio) continue;
    if (fill < params_.minFill) continue;

    const cv::Point2f dir = bar.top - bar.bottom;
    bar.length = length;
    bar.width = width;
    bar.fill = static_cast<float>(fill);
    bar.angle = static_cast<float>(normalizeAngle(toDeg(std::atan2(dir.x, -dir.y))));
    const double angleLimit = (maxAngle > 0.0) ? maxAngle : params_.maxBarAngle;
    if (std::abs(bar.angle) > angleLimit) continue;
    bar.color = color;
    bars.push_back(bar);
  }
  return bars;
}

bool AdaptiveDetector::buildArmor(const LightBar& a, const LightBar& b, Armor& armor,
                                  bool relaxed) const {
  // 放宽档只动“几何容差”，不动置信度公式里的其他项。
  const double maxDyRatio = relaxed ? params_.maxDyRatioRelaxed : params_.maxDyRatio;
  const double maxAngleDiff = relaxed ? params_.maxAngleDiffRelaxed : params_.maxAngleDiff;
  const double minArmorRatio = relaxed ? params_.minArmorRatioRelaxed : params_.minArmorRatio;
  const double angleNorm = relaxed ? params_.maxBarAngleRelaxed : params_.maxBarAngle;

  // ---- 1. 先按 x 排序，确定左右灯条 ----
  const LightBar& left = (a.rect.center.x <= b.rect.center.x) ? a : b;
  const LightBar& right = (&left == &a) ? b : a;
  const float h1 = left.length;
  const float h2 = right.length;
  const float avgH = 0.5F * (h1 + h2);
  if (avgH < static_cast<float>(params_.minArmorHeight)) return false;

  // ---- 2. 高度一致性：左右灯条应在同一高度、长度接近 ----
  const float hRatio = std::max(h1, h2) / std::max(1.0F, std::min(h1, h2));
  if (hRatio > static_cast<float>(params_.maxHeightRatio)) return false;

  const float dy = std::abs(signedDy(left.top, right.top));
  if (dy > avgH * static_cast<float>(maxDyRatio)) return false;

  // ---- 3. 平行性：两根灯条都该是竖直的，夹角不能太大 ----
  if (std::abs(left.angle - right.angle) > maxAngleDiff) return false;

  // ---- 4. 间距/宽高比：小装甲 135:55≈2.45，大装甲 225:55≈4.09 ----
  const float gap = right.rect.center.x - left.rect.center.x;
  const float outerW = gap + 0.5F * (left.width + right.width);
  if (outerW <= 0.0F) return false;
  const float ratio = outerW / avgH;
  if (ratio < minArmorRatio || ratio > params_.maxArmorRatio) return false;

  const double smallRatio = 0.135 / 0.055;
  const double largeRatio = 0.225 / 0.055;
  const double errSmall = std::abs(ratio - smallRatio) / smallRatio;
  const double errLarge = std::abs(ratio - largeRatio) / largeRatio;
  const bool isLarge = errLarge < errSmall;
  const double ratioErr = std::min(errSmall, errLarge);

  // 透视会让投影宽度变窄，但不能宽过理论值太多，否则多半是配错。
  const double expectedRatio = isLarge ? largeRatio : smallRatio;
  if (ratio > expectedRatio * params_.maxArmorWidthRatio) return false;

  // ---- 5. 组合置信度 ----
  const double angleErr = (std::abs(left.angle) + std::abs(right.angle)) * 0.5 /
                          std::max(1.0, angleNorm);
  const double dyErr = dy / std::max(1.0F, avgH);
  const double heightErr = std::abs(h1 - h2) / std::max(1.0F, avgH);
  const double fillScore = 0.5 * (left.fill + right.fill);
  const double score = 1.0 - std::clamp(0.45 * ratioErr * 3.0 + 0.20 * angleErr +
                                            0.20 * dyErr + 0.15 * heightErr,
                                        0.0, 1.0);
  armor.confidence = static_cast<float>(0.55 * score + 0.45 * std::clamp(fillScore, 0.0, 1.0));
  if (armor.confidence < 0.35F) return false;

  // ---- 6. 组装角点：左灯条外沿上、右灯条外沿上、右灯条外沿下、左灯条外沿下 ----
  cv::Point2f lt;
  cv::Point2f lb;
  cv::Point2f rt;
  cv::Point2f rb;
  outerEdgeEndpoints(left, right.rect.center, lt, lb);
  outerEdgeEndpoints(right, left.rect.center, rt, rb);
  armor.corners = {lt, rt, rb, lb};
  armor.leftBar = left;
  armor.rightBar = right;
  armor.leftBar.top = lt;
  armor.leftBar.bottom = lb;
  armor.rightBar.top = rt;
  armor.rightBar.bottom = rb;
  armor.center = 0.25F * (armor.corners[0] + armor.corners[1] + armor.corners[2] + armor.corners[3]);
  armor.width = outerW;
  armor.height = avgH;
  armor.color = left.color;
  armor.size = isLarge ? ArmorSize::Large : ArmorSize::Small;

  // 用投影宽度/理论宽度的比值粗估偏航角，只用于 HUD 显示。
  const double cosYaw = std::clamp(ratio / expectedRatio, 0.0, 1.0);
  armor.yaw = static_cast<float>(toDeg(std::acos(cosYaw)));

  // 角点顺序检查：如果四边形自交（上边和下边交叉），直接丢弃。
  const cv::Point2f e1 = armor.corners[1] - armor.corners[0];
  const cv::Point2f e2 = armor.corners[2] - armor.corners[1];
  if (e1.x * e2.y - e1.y * e2.x <= 0.0F) return false;
  return true;
}

bool AdaptiveDetector::plateLooksLikeArmor(const cv::Mat& hsv, const Armor& armor) const {
  // 取两条灯条之间、上下各缩进 20% 的“面板区域”。
  const float x1f = armor.leftBar.rect.center.x + armor.leftBar.width * 0.5F + 1.0F;
  const float x2f = armor.rightBar.rect.center.x - armor.rightBar.width * 0.5F - 1.0F;
  if (x2f - x1f < 4.0F) return false;
  const float topY = std::min(armor.leftBar.top.y, armor.rightBar.top.y) + 0.2F * armor.height;
  const float botY = std::max(armor.leftBar.bottom.y, armor.rightBar.bottom.y) - 0.2F * armor.height;
  if (botY - topY < 4.0F) return false;

  const int x = std::clamp(static_cast<int>(x1f), 0, hsv.cols - 1);
  const int y = std::clamp(static_cast<int>(topY), 0, hsv.rows - 1);
  const int w = std::clamp(static_cast<int>(x2f - x1f), 1, hsv.cols - x);
  const int h = std::clamp(static_cast<int>(botY - topY), 1, hsv.rows - y);
  const double plateV = cv::mean(hsv(cv::Rect(x, y, w, h)))[2];

  // 灯条亮度：沿灯条轴线取 3 个采样点，避免把背景算进来。
  double barV = 0.0;
  int samples = 0;
  for (const LightBar* bar : {&armor.leftBar, &armor.rightBar}) {
    for (float t : {-0.3F, 0.0F, 0.3F}) {
      const cv::Point2f p = bar->top * (0.5F + t) + bar->bottom * (0.5F - t);
      const int px = std::clamp(static_cast<int>(p.x), 0, hsv.cols - 1);
      const int py = std::clamp(static_cast<int>(p.y), 0, hsv.rows - 1);
      barV += hsv.at<cv::Vec3b>(py, px)[2];
      ++samples;
    }
  }
  barV /= std::max(1, samples);

  // 真板子：面板明显比灯条暗；假目标（两块亮条 + 亮背景）会接近甚至超过灯条亮度。
  if (std::getenv("AEGIS_DEBUG") != nullptr) {
    std::fprintf(stderr, "  armor c=(%.0f,%.0f) w=%.0f h=%.0f conf=%.2f plateV=%.0f barV=%.0f\n",
                 armor.center.x, armor.center.y, armor.width, armor.height, armor.confidence,
                 plateV, barV);
  }
  if (plateV > 0.70 * barV && plateV > 85.0) return false;
  return true;
}

void AdaptiveDetector::refineCorners(const cv::Mat& gray, Armor& armor) const {
  // cornerSubPix 要求初始点及其窗口都在图像内，贴边的角点直接跳过；
  // 贴边目标本来也会被画面裁掉一部分，不做亚像素细化影响不大。
  std::vector<cv::Point2f> pts;
  std::vector<int> index;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f& p = armor.corners[i];
    if (p.x > 6.0F && p.y > 6.0F && p.x < gray.cols - 7.0F && p.y < gray.rows - 7.0F) {
      pts.push_back(p);
      index.push_back(i);
    }
  }
  if (pts.empty()) return;

  std::vector<cv::Point2f> refined = pts;
  cv::cornerSubPix(gray, refined, cv::Size(5, 5), cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.05));

  // 只在“细化后没有跑远”时采纳，防止角点被旁边的反光吸走。
  for (std::size_t k = 0; k < refined.size(); ++k) {
    if (cv::norm(refined[k] - pts[k]) < 4.0) armor.corners[index[k]] = refined[k];
  }
  armor.center = 0.25F * (armor.corners[0] + armor.corners[1] + armor.corners[2] + armor.corners[3]);
}

std::vector<Armor> AdaptiveDetector::detect(const cv::Mat& bgr, const cv::Rect& roi,
                                            int forcedLevel, bool allowRelaxed) {
  std::vector<Armor> result;
  if (bgr.empty()) return result;

  // ROI 必须夹到图像范围内，否则 OpenCV 会抛异常。
  cv::Rect search = roi.empty() ? cv::Rect(0, 0, bgr.cols, bgr.rows) : roi;
  const int sx = std::max(0, search.x);
  const int sy = std::max(0, search.y);
  const int ex = std::min(bgr.cols, search.x + search.width);
  const int ey = std::min(bgr.rows, search.y + search.height);
  search = cv::Rect(sx, sy, std::max(0, ex - sx), std::max(0, ey - sy));
  if (search.width < 24 || search.height < 24) search = cv::Rect(0, 0, bgr.cols, bgr.rows);
  const cv::Mat view = bgr(search);

  cv::Mat hsv;
  cv::cvtColor(view, hsv, cv::COLOR_BGR2HSV);
  cv::Mat gray;
  cv::cvtColor(view, gray, cv::COLOR_BGR2GRAY);

  // 环境统计固定用整帧的 1/8 缩略图：便宜（约 0.1ms），且与 ROI 无关。
  cv::Mat statsSmall;
  cv::Mat statsHsv;
  cv::resize(bgr, statsSmall, cv::Size(), 0.125, 0.125, cv::INTER_NEAREST);
  cv::cvtColor(statsSmall, statsHsv, cv::COLOR_BGR2HSV);

  // ---- 自适应阈值：先按当前档位算一次 ----
  level_ = forcedLevel >= 0 ? forcedLevel : 0;
  updateAdaptiveThresholds(statsHsv);

  // ---- 双通道：敌人颜色决定“谁算装甲板”，另一路只做统计给 HUD ----
  const Color enemy = params_.enemyColor;
  const Color other = (enemy == Color::Red) ? Color::Blue : Color::Red;

  cv::Mat enemyMask = makeMask(hsv, enemy, level_);
  std::vector<LightBar> enemyBars = findBars(enemyMask, enemy);

  if (std::getenv("AEGIS_DEBUG") != nullptr) {
    std::fprintf(stderr,
                 "[det] amb=%.1f p999=%.0f bS=%.0f S=%.0f V=%.0f lvl=%d bars=%zu\n",
                 stats_.ambientV, stats_.p999V, stats_.brightS, stats_.sThreshold,
                 stats_.vThreshold, level_, enemyBars.size());
    cv::imwrite("debug_mask.png", enemyMask);
    for (const auto& b : enemyBars) {
      std::fprintf(stderr, "  bar (%.0f,%.0f) L=%.1f W=%.1f ang=%.1f fill=%.2f\n", b.rect.center.x,
                   b.rect.center.y, b.length, b.width, b.angle, b.fill);
    }
  }

  // 严格档没凑够两根灯条 -> 自动切放宽档重来一次（只有一帧多花 ~1ms）。
  // 基线模式固定单档，不享受这个回退机制。
  if (!params_.fixedThreshold && enemyBars.size() < 2 && forcedLevel < 0 && level_ == 0) {
    level_ = 1;
    updateAdaptiveThresholds(statsHsv);
    enemyMask = makeMask(hsv, enemy, level_);
    enemyBars = findBars(enemyMask, enemy);
  }

  // 另一路颜色同样跑一遍，仅用于 HUD 展示“双通道都在工作”。
  const cv::Mat otherMask = makeMask(hsv, other, level_);
  const std::vector<LightBar> otherBars = findBars(otherMask, other);
  stats_.redBars = (enemy == Color::Red) ? static_cast<int>(enemyBars.size())
                                         : static_cast<int>(otherBars.size());
  stats_.blueBars = (enemy == Color::Blue) ? static_cast<int>(enemyBars.size())
                                           : static_cast<int>(otherBars.size());

  // ---- 两两配对（严格档）----
  auto pairUp = [&](const std::vector<LightBar>& bars, bool relaxed) {
    for (std::size_t i = 0; i < bars.size(); ++i) {
      for (std::size_t j = i + 1; j < bars.size(); ++j) {
        Armor armor;
        if (!buildArmor(bars[i], bars[j], armor, relaxed)) continue;
        // 装甲板面板一定比灯条暗：窗户/灯管这类“两块亮条夹一块亮背景”的
        // 假目标会在这里被挡掉（低照度场景下尤其有效）。
        if (!plateLooksLikeArmor(hsv, armor)) continue;
        armor.thresholdLevel = level_;
        armor.ambient = static_cast<float>(stats_.ambientV);
        result.push_back(armor);
      }
    }
  };

  geometryRelaxed_ = false;
  pairUp(enemyBars, false);

  // 严格几何一块板都没配出来 -> 允许更大的灯条倾角、更大的 dy 和更小的宽高比，
  // 救“大偏航”目标（60° 偏航时 135mm 板投影宽度只剩一半，宽高比会掉到 1.2）。
  // 只在严格档完全为空时启用，正常视角下不会放宽，避免引入假目标。
  if (allowRelaxed && result.empty() && enemyBars.size() >= 2 && !params_.fixedThreshold) {
    enemyBars = findBars(enemyMask, enemy, params_.maxBarAngleRelaxed);
    pairUp(enemyBars, true);
    // 放宽档只接受“大且像”的候选：小反光条不配在丢失时把主目标抢走。
    result.erase(std::remove_if(result.begin(), result.end(),
                                [this](const Armor& a) {
                                  return a.height < 22.0F || a.confidence < 0.50F;
                                }),
                 result.end());
    geometryRelaxed_ = !result.empty();
  }

  // ---- NMS：同一块装甲板可能被多个灯条组合命中，只保留分最高的 ----
  std::sort(result.begin(), result.end(),
            [](const Armor& x, const Armor& y) { return x.confidence > y.confidence; });
  std::vector<Armor> kept;
  for (const auto& cand : result) {
    const cv::Rect2f box = bboxOf(cand.corners);
    bool suppressed = false;
    for (const auto& k : kept) {
      if (iou(box, bboxOf(k.corners)) > 0.35F) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push_back(cand);
    if (static_cast<int>(kept.size()) >= params_.maxArmors) break;
  }

  // ---- 亚像素角点 + 坐标偏移回全图 ----
  for (auto& armor : kept) {
    refineCorners(gray, armor);
    for (auto& p : armor.corners) p += cv::Point2f(static_cast<float>(search.x), static_cast<float>(search.y));
    armor.center += cv::Point2f(static_cast<float>(search.x), static_cast<float>(search.y));
    armor.leftBar.rect.center += cv::Point2f(static_cast<float>(search.x), static_cast<float>(search.y));
    armor.rightBar.rect.center += cv::Point2f(static_cast<float>(search.x), static_cast<float>(search.y));

    armor.leftBar.top += cv::Point2f(static_cast<float>(search.x), static_cast<float>(search.y));
    armor.leftBar.bottom += cv::Point2f(static_cast<float>(search.x), static_cast<float>(search.y));
    armor.rightBar.top += cv::Point2f(static_cast<float>(search.x), static_cast<float>(search.y));
    armor.rightBar.bottom += cv::Point2f(static_cast<float>(search.x), static_cast<float>(search.y));
  }
  return kept;
}

}  // namespace aegis



