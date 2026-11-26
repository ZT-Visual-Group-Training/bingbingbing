// =============================================================================
// sweep.cpp —— 消融实验：亮度 × 运动模糊 的检测成功率网格
//
// 做法：从真实视频里挑 40 帧“目标清晰可见”的画面作为基准（GT），
//      对每帧做 6 档亮度 × 3 档高斯模糊的扰动，分别喂给
//        A) 固定阈值检测器（基线）
//        B) 自适应阈值检测器（本方案）
//      统计“仍能找回同一块装甲板”的比例，输出 CSV + 热力图。
//
// 这个实验回答答辩时最容易被问的问题：
//   “你的自适应到底有用吗？拿数据说话。”
// =============================================================================
#include "aegis/adaptive_detector.hpp"
#include "aegis/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Cell {
  int total = 0;
  int adaptiveOk = 0;
  int fixedOk = 0;
};

bool matches(const std::vector<aegis::Armor>& armors, const cv::Point2f& gt, float tolerance) {
  for (const auto& a : armors) {
    if (cv::norm(a.center - gt) < tolerance) return true;
  }
  return false;
}

cv::Scalar rateColor(double rate) {
  // 0 -> 红，0.5 -> 黄，1 -> 绿
  const int r = static_cast<int>(255 * (1.0 - rate));
  const int g = static_cast<int>(255 * rate);
  return cv::Scalar(60, g, r);  // BGR
}

}  // namespace

int main(int argc, char** argv) {
  const std::string video = argc > 1 ? argv[1] : "video.avi";
  const std::string configPath = argc > 2 ? argv[2] : "config/aegis.yaml";
  const std::string outCsv = argc > 3 ? argv[3] : "results/sweep.csv";
  const std::string outPng = argc > 4 ? argv[4] : "results/sweep.png";

  aegis::AppConfig cfg = aegis::loadConfig(configPath);
  aegis::AdaptiveDetector adaptive(cfg.camera, cfg.detector);

  aegis::DetectorConfig fixedCfg = cfg.detector;
  fixedCfg.fixedThreshold = true;
  aegis::AdaptiveDetector fixed(cfg.camera, fixedCfg);

  cv::VideoCapture cap(video);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "无法打开 %s\n", video.c_str());
    return 1;
  }
  const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

  const int offsets[] = {-40, -20, 0, 30, 60, 100};
  const double blurs[] = {0.0, 1.5, 3.0};
  const int nScale = 6;
  const int nBlur = 3;
  std::vector<std::vector<Cell>> cells(nScale, std::vector<Cell>(nBlur));

  const int samples = 40;
  int usedFrames = 0;
  cv::Mat frame;
  for (int s = 0; s < samples; ++s) {
    const int idx = 150 + static_cast<int>((total - 200.0) * s / (samples - 1));
    cap.set(cv::CAP_PROP_POS_FRAMES, idx);
    if (!cap.read(frame) || frame.empty()) continue;

    // ---- 用原图建立 GT：只挑“高置信度、尺寸够大”的帧 ----
    const std::vector<aegis::Armor> gtArmors = adaptive.detect(frame, cv::Rect());
    const aegis::Armor* gt = nullptr;
    for (const auto& a : gtArmors) {
      if (a.height < 26.0F || a.confidence < 0.65F) continue;
      if (gt == nullptr || a.confidence > gt->confidence) gt = &a;
    }
    if (gt == nullptr) continue;
    const cv::Point2f gtCenter = gt->center;
    const float tolerance = std::max(14.0F, gt->width * 0.22F);
    ++usedFrames;

    for (int i = 0; i < nScale; ++i) {
      for (int j = 0; j < nBlur; ++j) {
        cv::Mat variant;
        frame.convertTo(variant, frame.type(), 1.0, offsets[i]);
        if (blurs[j] > 0.0) {
          cv::GaussianBlur(variant, variant, cv::Size(0, 0), blurs[j]);
        }
        const auto aOut = adaptive.detect(variant, cv::Rect());
        const auto fOut = fixed.detect(variant, cv::Rect());
        auto& cell = cells[i][j];
        ++cell.total;
        if (matches(aOut, gtCenter, tolerance)) ++cell.adaptiveOk;
        if (matches(fOut, gtCenter, tolerance)) ++cell.fixedOk;
      }
    }
  }

  std::ofstream csv(outCsv);
  csv << "brightness_offset,blur_sigma,frames,adaptive_ok,adaptive_rate,fixed_ok,fixed_rate\n";
  for (int i = 0; i < nScale; ++i) {
    for (int j = 0; j < nBlur; ++j) {
      const auto& c = cells[i][j];
      const double ar = c.total > 0 ? static_cast<double>(c.adaptiveOk) / c.total : 0.0;
      const double fr = c.total > 0 ? static_cast<double>(c.fixedOk) / c.total : 0.0;
      csv << offsets[i] << ',' << blurs[j] << ',' << c.total << ',' << c.adaptiveOk << ',' << ar
          << ',' << c.fixedOk << ',' << fr << '\n';
    }
  }

  // ---- 画两张并排的热力图 ----
  const int cellW = 200;
  const int cellH = 130;
  const int left = 220;
  const int top = 160;
  const int gapX = 120;
  const int gridW = nScale * cellW;
  const int w = left * 2 + gridW * 2 + gapX;
  const int h = top + nBlur * cellH + 170;
  cv::Mat img(h, w, CV_8UC3, cv::Scalar(18, 16, 14));

  auto drawGrid = [&](int ox, const char* title, bool adaptiveMode) {
    cv::putText(img, title, {ox, 58}, cv::FONT_HERSHEY_DUPLEX, 1.25,
                adaptiveMode ? cv::Scalar(255, 214, 10) : cv::Scalar(150, 160, 170), 2);
    for (int j = 0; j < nBlur; ++j) {
      for (int i = 0; i < nScale; ++i) {
        const auto& c = cells[i][j];
        const double rate = c.total > 0
                                ? static_cast<double>(adaptiveMode ? c.adaptiveOk : c.fixedOk) / c.total
                                : 0.0;
        const cv::Rect r(ox + i * cellW, top + j * cellH, cellW - 4, cellH - 4);
        cv::rectangle(img, r, rateColor(rate), cv::FILLED);
        cv::putText(img, cv::format("%.0f%%", rate * 100), {r.x + 22, r.y + 54},
                    cv::FONT_HERSHEY_DUPLEX, 1.45, cv::Scalar(10, 10, 10), 3);
        if (j == 0) {
          cv::putText(img, cv::format("%+d", offsets[i]), {r.x + 40, top - 12},
                      cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(190, 195, 200), 2);
        }
        if (i == 0) {
          cv::putText(img, cv::format("b%.1f", blurs[j]), {ox - 78, r.y + 54},
                      cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(190, 195, 200), 2);
        }
      }
    }
  };

  drawGrid(left, "FIXED THRESHOLD (BASELINE)", false);
  drawGrid(left + gridW + gapX, "ADAPTIVE (OURS)", true);
  cv::putText(img, "AMBIENT OFFSET  /  BLUR SIGMA", {left, h - 38}, cv::FONT_HERSHEY_DUPLEX, 0.85,
              cv::Scalar(150, 158, 165), 1);
  cv::imwrite(outPng, img);
  std::printf("GT frames=%d  cells=%dx%d  -> %s / %s\n", usedFrames, nScale, nBlur, outCsv.c_str(),
              outPng.c_str());
  return 0;
}



