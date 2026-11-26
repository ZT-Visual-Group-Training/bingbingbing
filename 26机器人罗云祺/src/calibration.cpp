#include "rm/calibration.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rm
{

namespace
{

// 去掉首尾空白（空格、制表、换行）
std::string trim(const std::string & s)
{
  size_t b = 0;
  size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// 把一段文本里的所有数字抠出来
void collectNumbers(const std::string & text, std::vector<double> & out)
{
  const char * p = text.c_str();
  while (*p != '\0') {
    if (std::isdigit(static_cast<unsigned char>(*p)) || *p == '-' || *p == '+' || *p == '.') {
      char * end = nullptr;
      const double v = std::strtod(p, &end);
      if (end != p) {
        out.push_back(v);
        p = end;
        continue;
      }
    }
    ++p;
  }
}

// 数据集里的标定文件是标准 YAML：
//   camera_matrix: [a, b, c, ...]        <- 行内
//   distort_coeffs:
//   - a                                   <- 块状
//   - b
// 这里只做“找到 key 后把数字收集起来”，对这两种写法都成立。
std::vector<double> findList(const std::vector<std::string> & lines, const std::string & key)
{
  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string t = trim(lines[i]);
    if (t.rfind(key + ":", 0) != 0) continue;  // 必须从行首匹配 "key:"

    std::vector<double> out;
    collectNumbers(t.substr(t.find(':') + 1), out);
    if (!out.empty()) return out;

    for (size_t j = i + 1; j < lines.size(); ++j) {
      const std::string u = trim(lines[j]);
      if (u.empty()) continue;
      if (u[0] != '-') break;  // 遇到下一个 key 就停
      collectNumbers(u, out);
    }
    return out;
  }
  return {};
}

// 读单个标量
double findScalar(const std::vector<std::string> & lines, const std::string & key, double fallback)
{
  for (const auto & line : lines) {
    const std::string t = trim(line);
    if (t.rfind(key + ":", 0) != 0) continue;
    std::vector<double> out;
    collectNumbers(t.substr(t.find(':') + 1), out);
    if (!out.empty()) return out.front();
    return fallback;
  }
  return fallback;
}

}

Calibration Calibration::load(const std::string & path)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open calibration: " + path);

  // 1. 按行读，顺手去掉 # 注释
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    lines.push_back(line);
  }

  // 2. 按 key 抠出四组数
  const auto k = findList(lines, "camera_matrix");
  const auto d = findList(lines, "distort_coeffs");
  const auto r = findList(lines, "R_camera2gimbal");
  const auto t = findList(lines, "t_camera2gimbal");

  // 3. 校验，数量不对直接抛异常，避免后面拿错数据算半天
  if (k.size() != 9) throw std::runtime_error("calibration: camera_matrix must have 9 values");
  if (d.size() < 4) throw std::runtime_error("calibration: distort_coeffs must have >= 4 values");
  if (r.size() != 9) throw std::runtime_error("calibration: R_camera2gimbal must have 9 values");
  if (t.size() != 3) throw std::runtime_error("calibration: t_camera2gimbal must have 3 values");

  Calibration calib;
  // 4. K 是行优先展平的 3x3
  calib.K = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) calib.K.at<double>(i / 3, i % 3) = k[static_cast<size_t>(i)];

  // 5. D 是 1x5：k1 k2 p1 p2 k3
  calib.D = cv::Mat(1, static_cast<int>(d.size()), CV_64F);
  for (size_t i = 0; i < d.size(); ++i) calib.D.at<double>(0, static_cast<int>(i)) = d[i];

  // 6. R 是 3x3，t 是 3x1(m)
  for (int i = 0; i < 9; ++i) {
    calib.R_camera2gimbal(static_cast<int>(i / 3), static_cast<int>(i % 3)) =
      r[static_cast<size_t>(i)];
  }
  calib.t_camera2gimbal = cv::Vec3d(t[0], t[1], t[2]);

  // 7. 图像尺寸
  calib.width = static_cast<int>(findScalar(lines, "image_width", 0));
  calib.height = static_cast<int>(findScalar(lines, "image_height", 0));
  return calib;
}

}
