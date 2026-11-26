#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rm
{

// 时间戳 + 四元数
struct ImuSample
{
  int64_t ns = 0;
  double qw = 1.0, qx = 0.0, qy = 0.0, qz = 0.0;
};

// 逐帧四元数,只加载并打印
class Imu
{
public:
  bool load(const std::string & csv_path, std::string * error = nullptr);
  bool quatAt(int64_t ns, std::array<double, 4> & qwxyz) const;  // 球面线性插值：保证插出来的仍是单位四元数
  size_t size() const { return samples_.size(); }

private:
  std::vector<ImuSample> samples_;
};

}
