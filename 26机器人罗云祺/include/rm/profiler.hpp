#pragma once

#include <string>
#include <vector>

namespace rm
{

// 处理耗时统计
class Profiler
{
public:
  void add(double ms) { samples_.push_back(ms); }  // 一帧耗时
  size_t count() const { return samples_.size(); }  // 已统计的帧数
  double mean() const;  // 算术平均
  double p95() const;  // 95 分位
  double max() const;  // 最慢一帧
  void print(const std::string & name) const;

private:
  std::vector<double> samples_;
};

}
