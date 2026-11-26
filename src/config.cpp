// =============================================================================
// config.cpp —— 用 OpenCV FileStorage 读 YAML
// 不引入 yaml-cpp，是为了让工程依赖只剩 OpenCV（考核要求允许的方式）。
// 注意：OpenCV 的 YAML 解析比标准 YAML 严格，几个实际踩过的坑都写在注释里。
// =============================================================================
#include "aegis/config.hpp"

#include <iostream>
#include <vector>

namespace aegis {
namespace {

// 小工具：节点存在才读取，避免“少写一个字段就崩溃”。
template <typename T>
void readIf(const cv::FileNode& node, const char* key, T& dst) {
  const cv::FileNode n = node[key];
  if (!n.empty()) n >> dst;
}

// 读一维数组：YAML 的 [a, b, c] 必须用 vector 接收。
// 直接 `n >> cv::Mat` 会让 OpenCV 去读 rows/cols 子节点，触发 isMap 断言。
bool readDoubleSeq(const cv::FileNode& node, const char* key, std::vector<double>& out) {
  const cv::FileNode n = node[key];
  if (n.empty() || !n.isSeq()) return false;
  out.clear();
  for (const auto& v : n) {
    double d = 0.0;
    v >> d;
    out.push_back(d);
  }
  return true;
}

// 读 3x3 矩阵：YAML 里写成 9 个数的一行。
bool readMat33(const cv::FileNode& node, const char* key, cv::Matx33d& dst) {
  std::vector<double> v;
  if (!readDoubleSeq(node, key, v) || v.size() != 9) return false;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) dst(r, c) = v[static_cast<std::size_t>(r * 3 + c)];
  }
  return true;
}

// 读三维向量（YAML 里是 [x, y, z]）。
bool readVec3(const cv::FileNode& node, const char* key, cv::Vec3d& dst) {
  std::vector<double> v;
  if (!readDoubleSeq(node, key, v) || v.size() != 3) return false;
  dst = cv::Vec3d(v[0], v[1], v[2]);
  return true;
}

}  // namespace

AppConfig loadConfig(const std::string& path) {
  AppConfig cfg;
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "[config] 无法打开 " << path << "，使用内置默认参数\n";
    return cfg;
  }

  const cv::FileNode cam = fs["camera"];
  if (!cam.empty()) {
    readIf(cam, "width", cfg.camera.width);
    readIf(cam, "height", cfg.camera.height);
    readMat33(cam, "camera_matrix", cfg.camera.cameraMatrix);

    std::vector<double> dist;
    if (readDoubleSeq(cam, "distort_coeffs", dist) && dist.size() == 5) {
      cfg.camera.distCoeffs = cv::Mat(1, 5, CV_64F, dist.data()).clone();
    }
    readMat33(cam, "R_camera2gimbal", cfg.camera.R_camera2gimbal);
    readVec3(cam, "t_camera2gimbal", cfg.camera.t_camera2gimbal);
  }

  const cv::FileNode armor = fs["armor"];
  if (!armor.empty()) {
    readIf(armor, "small_width_m", cfg.armor.smallWidthM);
    readIf(armor, "large_width_m", cfg.armor.largeWidthM);
    readIf(armor, "height_m", cfg.armor.heightM);
    std::string color = "red";
    readIf(armor, "enemy_color", color);
    cfg.detector.enemyColor = (color == "blue") ? Color::Blue : Color::Red;
  }

  const cv::FileNode det = fs["detector"];
  if (!det.empty()) {
    readIf(det, "s_min_base", cfg.detector.sMinBase);
    readIf(det, "v_min_base", cfg.detector.vMinBase);
    readIf(det, "min_bar_length", cfg.detector.minBarLength);
    readIf(det, "max_bar_length", cfg.detector.maxBarLength);
    readIf(det, "min_bar_ratio", cfg.detector.minBarRatio);
    readIf(det, "max_bar_ratio", cfg.detector.maxBarRatio);
    readIf(det, "min_fill", cfg.detector.minFill);
    readIf(det, "max_bar_angle", cfg.detector.maxBarAngle);
    readIf(det, "max_height_ratio", cfg.detector.maxHeightRatio);
    readIf(det, "max_dy_ratio", cfg.detector.maxDyRatio);
    readIf(det, "max_angle_diff", cfg.detector.maxAngleDiff);
    readIf(det, "min_armor_ratio", cfg.detector.minArmorRatio);
    readIf(det, "max_armor_ratio", cfg.detector.maxArmorRatio);
    readIf(det, "min_armor_ratio_relaxed", cfg.detector.minArmorRatioRelaxed);
    readIf(det, "max_bar_angle_relaxed", cfg.detector.maxBarAngleRelaxed);
    readIf(det, "max_angle_diff_relaxed", cfg.detector.maxAngleDiffRelaxed);
    readIf(det, "max_dy_ratio_relaxed", cfg.detector.maxDyRatioRelaxed);
    readIf(det, "min_armor_height", cfg.detector.minArmorHeight);
    readIf(det, "max_armors", cfg.detector.maxArmors);
  }

  const cv::FileNode trk = fs["tracker"];
  if (!trk.empty()) {
    readIf(trk, "min_hits", cfg.tracker.minHitsToConfirm);
    readIf(trk, "max_misses", cfg.tracker.maxMisses);
    readIf(trk, "gate_distance", cfg.tracker.gateDistance);
    readIf(trk, "process_noise", cfg.tracker.processNoise);
    readIf(trk, "measurement_noise", cfg.tracker.measurementNoise);
  }

  const cv::FileNode out = fs["output"];
  if (!out.empty()) {
    readIf(out, "fps", cfg.outputFps);
    readIf(out, "digits", cfg.enableDigits);
    readIf(out, "roi", cfg.enableRoi);
    readIf(out, "hud", cfg.drawHud);
  }
  return cfg;
}

}  // namespace aegis
