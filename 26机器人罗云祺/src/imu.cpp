#include "rm/imu.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace rm
{

namespace
{

std::vector<std::string> split(const std::string & line, char sep)
{
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r') {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

}

bool Imu::load(const std::string & csv_path, std::string * error)
{
  samples_.clear();

  std::ifstream in(csv_path);
  if (!in) {
    if (error) *error = "cannot open " + csv_path;
    return false;
  }

  std::string line;
  std::getline(in, line);  // header

  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto tok = split(line, ',');
    if (tok.size() < 7) continue;
    try {
      // // 列顺序: frame_index,monotonic_ns,elapsed_s,qw,qx,qy,qz,...
      ImuSample s;
      // 第 0 列是 frame_index
      s.ns = std::stoll(tok[1]);
      s.qw = std::stod(tok[3]);
      s.qx = std::stod(tok[4]);
      s.qy = std::stod(tok[5]);
      s.qz = std::stod(tok[6]);

      // 归一化: 浮点误差会让模长偏离 1，不归一后面算旋转矩阵会缩放
      const double n = std::sqrt(s.qw * s.qw + s.qx * s.qx + s.qy * s.qy + s.qz * s.qz);
      if (n < 1e-9) continue;
      s.qw /= n;
      s.qx /= n;
      s.qy /= n;
      s.qz /= n;
      samples_.push_back(s);
    } catch (...) {
      continue;
    }
  }

  // 防御性排序
  std::sort(samples_.begin(), samples_.end(), [](const ImuSample & a, const ImuSample & b) {
    return a.ns < b.ns;
  });
  return !samples_.empty();
}

bool Imu::quatAt(int64_t ns, std::array<double, 4> & q) const
{
  if (samples_.empty()) return false;
  // 超出范围：取首/尾样本
  if (ns <= samples_.front().ns) {
    q = {samples_.front().qw, samples_.front().qx, samples_.front().qy, samples_.front().qz};
    return true;
  }
  if (ns >= samples_.back().ns) {
    q = {samples_.back().qw, samples_.back().qx, samples_.back().qy, samples_.back().qz};
    return true;
  }

  // 二分找第一个 >= ns 的样本
  const auto it = std::lower_bound(
    samples_.begin(), samples_.end(), ns,
    [](const ImuSample & s, int64_t v) { return s.ns < v; });

  const ImuSample & b = *it;  // ns 之后的样本
  const ImuSample & a = *(it - 1);  // ns 之前的样本
  const double span = static_cast<double>(b.ns - a.ns);
  const double t = span > 0.0 ? static_cast<double>(ns - a.ns) / span : 0.0;

  double w0 = a.qw, x0 = a.qx, y0 = a.qy, z0 = a.qz;
  double w1 = b.qw, x1 = b.qx, y1 = b.qy, z1 = b.qz;

  // 四元数 q 和 -q 表示同一个旋转，点积为负说明走了“长弧"
  double dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1;
  if (dot < 0.0) {
    w1 = -w1;
    x1 = -x1;
    y1 = -y1;
    z1 = -z1;
    dot = -dot;
  }

  double w, x, y, z;
  if (dot > 0.9995) {
    // 线性插值再归一化
    w = w0 + t * (w1 - w0);
    x = x0 + t * (x1 - x0);
    y = y0 + t * (y1 - y0);
    z = z0 + t * (z1 - z0);
  } else {
    // slerp：球面线性插值
    const double theta_0 = std::acos(std::min(1.0, dot));  // 两个四元数的夹角
    const double sin_theta_0 = std::sin(theta_0);
    const double s0 = std::sin((1.0 - t) * theta_0) / sin_theta_0;
    const double s1 = std::sin(t * theta_0) / sin_theta_0;
    w = s0 * w0 + s1 * w1;
    x = s0 * x0 + s1 * x1;
    y = s0 * y0 + s1 * y1;
    z = s0 * z0 + s1 * z1;
  }

  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  q = {w / n, x / n, y / n, z / n};
  return true;
}

}
