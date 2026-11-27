#pragma once

// 读取 calibration/ 下的三个标定文件：
//   intrinsics.yaml  -> 相机内参 K 与畸变系数 D
//   handeye.yaml     -> 相机->云台外参 (R, t) 与 云台->IMU 安装旋转
//   calibration.yaml -> 合并后的标定，用作回退与一致性校验
// 所有矩阵统一为 CV_64F：K/R 为 3x3，t 为 3x1，t 的单位为米。
//
// 这批 yaml 没有 %YAML 指令头、且含 OpenCV FileStorage 解析不了的字段
// （如 calibrated_at: 20261030_185655），所以这里用一个只覆盖用到的语法子集的
// 轻量解析器：`key: 标量`、`key: [a, b, ...]`、以及 `key:` 后续 `- 元素` 的块序列。

#include <opencv2/core.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace rm {

struct Calibration {
  cv::Mat camera_matrix;     // 3x3
  cv::Mat distort_coeffs;    // 1xN
  cv::Mat R_camera2gimbal;   // 3x3，无手眼时为空
  cv::Mat t_camera2gimbal;   // 3x1，单位米，无手眼时为空
  cv::Mat R_gimbal2imubody;  // 3x3
  int image_width = 0;
  int image_height = 0;

  bool hasHandeye() const { return !R_camera2gimbal.empty() && !t_camera2gimbal.empty(); }

  // calib_dir 为数据目录下的 calibration/
  static Calibration load(const std::filesystem::path& calib_dir);
};

namespace detail {

inline std::string yamlTrim(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

inline std::string stripComment(const std::string& line) {
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '#' && (i == 0 || std::isspace(static_cast<unsigned char>(line[i - 1])))) {
      return line.substr(0, i);
    }
  }
  return line;
}

inline std::string unquote(const std::string& s) {
  if (s.size() >= 2 &&
      ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

inline std::vector<std::string> splitInline(const std::string& value) {
  std::vector<std::string> out;
  const std::string v = yamlTrim(value);
  if (v.size() >= 2 && v.front() == '[' && v.back() == ']') {
    const std::string body = v.substr(1, v.size() - 2);
    std::string cur;
    for (char c : body) {
      if (c == ',') {
        const std::string item = yamlTrim(cur);
        if (!item.empty()) out.push_back(unquote(item));
        cur.clear();
      } else {
        cur.push_back(c);
      }
    }
    const std::string item = yamlTrim(cur);
    if (!item.empty()) out.push_back(unquote(item));
  } else if (!v.empty()) {
    out.push_back(unquote(v));
  }
  return out;
}

struct YamlMap {
  std::map<std::string, std::vector<std::string>> kv;

  bool has(const std::string& key) const { return kv.count(key) != 0; }

  const std::vector<std::string>* find(const std::string& key) const {
    const auto it = kv.find(key);
    return it == kv.end() ? nullptr : &it->second;
  }
};

inline YamlMap parseYaml(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("无法打开标定文件: " + path.string());

  YamlMap out;
  std::string key;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const std::string s = yamlTrim(stripComment(line));
    if (s.empty() || s == "---" || s == "...") continue;

    if (s[0] == '-') {
      if (key.empty()) {
        throw std::runtime_error(path.string() + ": 第 " + std::to_string(line_no) +
                                 " 行序列元素没有所属 key");
      }
      const std::string item = yamlTrim(s.substr(1));
      if (!item.empty()) out.kv[key].push_back(unquote(item));
      continue;
    }

    const auto colon = s.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error(path.string() + ": 第 " + std::to_string(line_no) +
                               " 行无法解析: '" + s + "'");
    }
    key = yamlTrim(s.substr(0, colon));
    const std::string value = yamlTrim(s.substr(colon + 1));
    if (value.empty()) {
      out.kv[key];  // 占位，后续可能是块序列
    } else {
      out.kv[key] = splitInline(value);
    }
  }
  return out;
}

inline double toDouble(const std::string& s, const std::string& file, const std::string& key) {
  try {
    return std::stod(s);
  } catch (const std::exception&) {
    throw std::runtime_error(file + ": 字段 " + key + " 含非数值元素 '" + s + "'");
  }
}

inline cv::Mat readMat(const YamlMap& yaml, const std::string& file, const std::string& key,
                       int rows, int cols) {
  const auto* values = yaml.find(key);
  if (!values) return {};

  const int total = rows * cols;
  if (static_cast<int>(values->size()) != total) {
    throw std::runtime_error(file + ": 字段 " + key + " 期望 " + std::to_string(total) +
                             " 个数值，实际 " + std::to_string(values->size()));
  }

  cv::Mat m(rows, cols, CV_64F);
  for (int r = 0; r < rows; ++r) {
    double* row = m.ptr<double>(r);
    for (int c = 0; c < cols; ++c) {
      row[c] = toDouble((*values)[static_cast<std::size_t>(r * cols + c)], file, key);
    }
  }
  return m;
}

inline int readInt(const YamlMap& yaml, const std::string& file, const std::string& key,
                   int fallback) {
  const auto* values = yaml.find(key);
  if (!values || values->empty()) return fallback;
  try {
    return std::stoi((*values)[0]);
  } catch (const std::exception&) {
    throw std::runtime_error(file + ": 字段 " + key + " 不是整数");
  }
}

inline double maxAbsDiff(const cv::Mat& a, const cv::Mat& b) {
  double m = 0.0;
  cv::minMaxLoc(cv::abs(a - b), nullptr, &m);
  return m;
}

inline void checkSame(const cv::Mat& a, const cv::Mat& b, const std::string& what) {
  if (a.empty() || b.empty()) return;
  if (a.size() != b.size() || a.type() != b.type()) {
    throw std::runtime_error("标定文件不一致: " + what + " 尺寸/类型不同");
  }
  const double scale = std::max(1.0, cv::norm(a, cv::NORM_INF));
  if (maxAbsDiff(a, b) > 1e-6 * scale) {
    throw std::runtime_error("标定文件不一致: " + what +
                             " (intrinsics/handeye 与 calibration.yaml 对不上)");
  }
}

inline void fillIfEmpty(cv::Mat& dst, const cv::Mat& src) {
  if (dst.empty() && !src.empty()) dst = src;
}

}  // namespace detail

inline Calibration Calibration::load(const std::filesystem::path& calib_dir) {
  if (!std::filesystem::is_directory(calib_dir)) {
    throw std::runtime_error("标定目录不存在: " + calib_dir.string());
  }

  const auto p_intr = calib_dir / "intrinsics.yaml";
  const auto p_he = calib_dir / "handeye.yaml";
  const auto p_cal = calib_dir / "calibration.yaml";

  if (!std::filesystem::exists(p_intr)) {
    throw std::runtime_error("缺少标定文件: " + p_intr.string());
  }

  Calibration c;

  {
    const auto yaml = detail::parseYaml(p_intr);
    c.camera_matrix = detail::readMat(yaml, p_intr.string(), "camera_matrix", 3, 3);
    c.distort_coeffs = detail::readMat(yaml, p_intr.string(), "distort_coeffs", 1, 5);
    c.image_width = detail::readInt(yaml, p_intr.string(), "image_width", c.image_width);
    c.image_height = detail::readInt(yaml, p_intr.string(), "image_height", c.image_height);
    if (c.camera_matrix.empty()) throw std::runtime_error(p_intr.string() + ": 缺少 camera_matrix");
    if (c.distort_coeffs.empty()) throw std::runtime_error(p_intr.string() + ": 缺少 distort_coeffs");
  }

  if (std::filesystem::exists(p_he)) {
    const auto yaml = detail::parseYaml(p_he);
    c.R_camera2gimbal = detail::readMat(yaml, p_he.string(), "R_camera2gimbal", 3, 3);
    c.t_camera2gimbal = detail::readMat(yaml, p_he.string(), "t_camera2gimbal", 3, 1);
    c.R_gimbal2imubody = detail::readMat(yaml, p_he.string(), "R_gimbal2imubody", 3, 3);
    if (c.image_width == 0) {
      c.image_width = detail::readInt(yaml, p_he.string(), "image_width", 0);
    }
    if (c.image_height == 0) {
      c.image_height = detail::readInt(yaml, p_he.string(), "image_height", 0);
    }
  }

  if (std::filesystem::exists(p_cal)) {
    const auto yaml = detail::parseYaml(p_cal);
    const cv::Mat K = detail::readMat(yaml, p_cal.string(), "camera_matrix", 3, 3);
    const cv::Mat D = detail::readMat(yaml, p_cal.string(), "distort_coeffs", 1, 5);
    const cv::Mat R = detail::readMat(yaml, p_cal.string(), "R_camera2gimbal", 3, 3);
    const cv::Mat t = detail::readMat(yaml, p_cal.string(), "t_camera2gimbal", 3, 1);
    const cv::Mat Rgi = detail::readMat(yaml, p_cal.string(), "R_gimbal2imubody", 3, 3);

    detail::checkSame(c.camera_matrix, K, "camera_matrix");
    detail::checkSame(c.distort_coeffs, D, "distort_coeffs");
    detail::checkSame(c.R_camera2gimbal, R, "R_camera2gimbal");
    detail::checkSame(c.t_camera2gimbal, t, "t_camera2gimbal");
    detail::checkSame(c.R_gimbal2imubody, Rgi, "R_gimbal2imubody");

    detail::fillIfEmpty(c.camera_matrix, K);
    detail::fillIfEmpty(c.distort_coeffs, D);
    detail::fillIfEmpty(c.R_camera2gimbal, R);
    detail::fillIfEmpty(c.t_camera2gimbal, t);
    detail::fillIfEmpty(c.R_gimbal2imubody, Rgi);
    if (c.image_width == 0) {
      c.image_width = detail::readInt(yaml, p_cal.string(), "image_width", 0);
    }
    if (c.image_height == 0) {
      c.image_height = detail::readInt(yaml, p_cal.string(), "image_height", 0);
    }
  }

  if (c.R_gimbal2imubody.empty()) c.R_gimbal2imubody = cv::Mat::eye(3, 3, CV_64F);

  return c;
}

}  // namespace rm
