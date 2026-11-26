#include "rm/profiler.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace rm
{

double Profiler::mean() const
{
  if (samples_.empty()) return 0.0;
  double sum = 0.0;
  for (double v : samples_) sum += v;
  return sum / static_cast<double>(samples_.size());
}

double Profiler::p95() const
{
  if (samples_.empty()) return 0.0;
  // 拷贝一份再排序, samples_ 是按帧顺序存的，排序会打乱时间序列
  std::vector<double> sorted = samples_;
  std::sort(sorted.begin(), sorted.end());
  // 95% 的帧比它快，剩下 5% 比它慢
  size_t idx = static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1));
  return sorted[idx];
}

double Profiler::max() const
{
  if (samples_.empty()) return 0.0;
  return *std::max_element(samples_.begin(), samples_.end());
}

void Profiler::print(const std::string & name) const
{
  // %zu 是 size_t；%.3f 保留三位小数
  std::printf(
    "[%s] n=%zu  mean=%.3f ms  p95=%.3f ms  max=%.3f ms\n", name.c_str(), samples_.size(), mean(),
    p95(), max());
}

}
