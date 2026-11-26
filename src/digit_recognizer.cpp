// =============================================================================
// digit_recognizer.cpp —— 装甲板数字识别（0~9）
//
// 为什么不用 YOLO/CNN：
//   考核视频里数字只占 40×50 像素，训练一个能泛化的检测网络成本很高；
//   而数字是“高对比度、固定位置、白字黑底”的强先验目标，用归一化模板匹配
//   反而更快更稳（单帧 <0.1ms，零模型文件）。
//
// 流程：
//   四点透视变换 -> 裁掉两侧灯条 -> 灰度拉伸 -> Otsu 二值化 ->
//   找最大连通域 -> 按外接框归一化到 40×40 -> 与 0~9 的多字体模板做 NCC
//
// 额外做了一个“在线模板自举”：连续高置信度识别到某个数字时，把实拍样本
//   加入该数字的模板库，解决仿真字体和实车字体不一致的问题。
// =============================================================================
#include "aegis/digit_recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace aegis {
namespace {

constexpr int kPatch = 40;  // 归一化后的数字图尺寸

// 把任意大小的数字二值图缩放到 40×40，保持长宽比并居中。
cv::Mat normalizePatch(const cv::Mat& bin) {
  cv::Mat out(kPatch, kPatch, CV_32F, cv::Scalar(0));
  if (bin.empty()) return out;

  std::vector<cv::Point> pts;
  cv::findNonZero(bin, pts);
  if (pts.empty()) return out;
  const cv::Rect box = cv::boundingRect(pts);
  if (box.width < 3 || box.height < 6) return out;

  const cv::Mat crop = bin(box);
  const float scale = 34.0F / static_cast<float>(std::max(box.width, box.height));
  const int w = std::max(1, static_cast<int>(std::lround(box.width * scale)));
  const int h = std::max(1, static_cast<int>(std::lround(box.height * scale)));
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);

  const int ox = (kPatch - w) / 2;
  const int oy = (kPatch - h) / 2;
  resized.convertTo(out(cv::Rect(ox, oy, w, h)), CV_32F, 1.0 / 255.0);
  return out;
}

// 两个同尺寸 40×40 图的归一化互相关（NCC）。
// 这里手写而不是用 cv::matchTemplate：同尺寸模板时 OpenCV 会走 DFT，
// 30 个模板要 1.5ms；手写逐像素点积只要 0.05ms，整帧延迟直接降下来。
double ncc(const cv::Mat& a, const cv::Mat& b) {
  const int n = a.rows * a.cols;
  const float* pa = a.ptr<float>();
  const float* pb = b.ptr<float>();
  double ma = 0.0;
  double mb = 0.0;
  for (int i = 0; i < n; ++i) {
    ma += pa[i];
    mb += pb[i];
  }
  ma /= n;
  mb /= n;
  double va = 0.0;
  double vb = 0.0;
  double cov = 0.0;
  for (int i = 0; i < n; ++i) {
    const double da = pa[i] - ma;
    const double db = pb[i] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  if (va < 1e-9 || vb < 1e-9) return 0.0;
  return cov / std::sqrt(va * vb);
}

// 生成一个数字的“字体模板”：白字黑底，再做同样的归一化。
cv::Mat renderDigit(int digit, int font, double scale, int thickness) {
  cv::Mat canvas(96, 96, CV_8U, cv::Scalar(0));
  const std::string text = std::to_string(digit);
  int baseline = 0;
  const cv::Size sz = cv::getTextSize(text, font, scale, thickness, &baseline);
  const cv::Point org((96 - sz.width) / 2, (96 + sz.height) / 2);
  cv::putText(canvas, text, org, font, scale, cv::Scalar(255), thickness, cv::LINE_AA);
  return normalizePatch(canvas);
}

}  // namespace

DigitRecognizer::DigitRecognizer() {
  // 三种 Hershey 字体的笔画粗细/圆角不同，覆盖实车数字的主要形态。
  const int fonts[] = {cv::FONT_HERSHEY_SIMPLEX, cv::FONT_HERSHEY_DUPLEX,
                       cv::FONT_HERSHEY_TRIPLEX};
  for (int d = 0; d < 10; ++d) {
    for (int f : fonts) {
      templates_[d].push_back(renderDigit(d, f, 2.2, 7));
      templates_[d].push_back(renderDigit(d, f, 2.2, 10));
    }
  }
}

cv::Mat DigitRecognizer::warpArmor(const cv::Mat& bgr, const Armor& armor, const cv::Size& size) {
  const std::vector<cv::Point2f> src(armor.corners.begin(), armor.corners.end());
  const std::vector<cv::Point2f> dst = {
      {0.0F, 0.0F},
      {static_cast<float>(size.width - 1), 0.0F},
      {static_cast<float>(size.width - 1), static_cast<float>(size.height - 1)},
      {0.0F, static_cast<float>(size.height - 1)},
  };
  const cv::Mat H = cv::getPerspectiveTransform(src, dst);
  cv::Mat out;
  cv::warpPerspective(bgr, out, H, size, cv::INTER_LINEAR);
  return out;
}

int DigitRecognizer::recognize(const cv::Mat& bgr, const Armor& armor, float& confidence) const {
  confidence = 0.0F;
  if (bgr.empty() || armor.width < 25.0F || armor.height < 12.0F) return -1;

  // 数字往往比灯条高不少（考核用的板子数字印在灯条上下的面板上），
  // 透视变换前把四角向外扩，避免把数字上下裁掉。
  Armor padded = armor;
  const cv::Point2f c = armor.center;
  for (auto& p : padded.corners) {
    p = c + cv::Point2f((p.x - c.x) * 1.06F, (p.y - c.y) * 1.80F);
  }

  const cv::Mat warp = warpArmor(bgr, padded, cv::Size(160, 64));  // 裁掉左右灯条（各留 22px），只保留面板中间的“数字区”。
  const cv::Mat inner = warp(cv::Rect(22, 0, 116, 64));

  cv::Mat gray;
  cv::cvtColor(inner, gray, cv::COLOR_BGR2GRAY);

  // 低照度下数字和面板只差 20~30 个灰度级，先做百分位拉伸再二值化。
  cv::Mat stretched;
  cv::normalize(gray, stretched, 0, 255, cv::NORM_MINMAX);
  cv::GaussianBlur(stretched, stretched, cv::Size(3, 3), 0);

  cv::Mat bin;
  cv::threshold(stretched, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::morphologyEx(bin, bin, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

  if (std::getenv("AEGIS_DEBUG") != nullptr) {
    static int dbg = 0;
    if (dbg < 8) {
      cv::imwrite(cv::format("dbg_warp_%d.png", dbg), warp);
      cv::Mat b;
      cv::resize(bin, b, cv::Size(), 4, 4, cv::INTER_NEAREST);
      cv::imwrite(cv::format("dbg_bin_%d.png", dbg), b);
      ++dbg;
    }
  }

  // 取最大连通域：数字一定是面板里最大的一块亮区。
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return -1;
  auto best = std::max_element(
      contours.begin(), contours.end(),
      [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
  if (cv::contourArea(*best) < 60.0) return -1;

  const cv::Rect box = cv::boundingRect(*best);
  // 数字的几何约束：
  //   * 太宽/太高 -> 整块面板反光；
  //   * 太窄太高 -> 灯条边缘/竖向反光（最常见的“假 1”）；
  //   * 贴到左右边界 -> 不是面板中间的数字。
  const float aspect = static_cast<float>(box.height) / std::max(1, box.width);
  if (box.width < 20 || box.width > 110 || box.height < 14 || box.height > 64) return -1;
  if (aspect < 0.45F || aspect > 3.2F) return -1;
  if (box.x < 2 || box.x + box.width > inner.cols - 2) return -1;

  const cv::Mat patch = normalizePatch(bin(box));
  if (cv::countNonZero(patch > 0.05F) < 30) return -1;

  // ---- 与模板做归一化互相关 ----
  int bestDigit = -1;
  double bestScore = -1.0;
  double secondScore = -1.0;
  for (int d = 0; d < 10; ++d) {
    double digitBest = -1.0;
    for (const auto& t : templates_[d]) digitBest = std::max(digitBest, ncc(patch, t));
    if (digitBest > bestScore) {
      secondScore = bestScore;
      bestScore = digitBest;
      bestDigit = d;
    } else if (digitBest > secondScore) {
      secondScore = digitBest;
    }
  }
  if (bestDigit < 0 || bestScore < 0.30) return -1;

  if (std::getenv("AEGIS_DEBUG") != nullptr) {
    std::fprintf(stderr, "[digit] best=%d score=%.3f second=%.3f box=(%d,%d,%d,%d)\n", bestDigit,
                 bestScore, secondScore, box.x, box.y, box.width, box.height);
  }

  // 置信度同时看“绝对相似度”和“与第二名的差距”，避免 3/8 这种形近字摇摆。
  const double margin = std::max(0.0, bestScore - secondScore);
  confidence =
      static_cast<float>(std::clamp(0.65 * std::max(0.0, bestScore) + 2.0 * margin, 0.0, 1.0));

  // ---- 在线模板自举 ----
  if (confidence > 0.75F && templates_[bestDigit].size() < 10) {
    templates_[bestDigit].push_back(patch.clone());
  }
  return bestDigit;
}

}  // namespace aegis

