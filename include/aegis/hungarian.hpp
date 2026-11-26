// =============================================================================
// hungarian.hpp —— 匈牙利算法（最小代价二分图匹配）
// 跟踪器的“检测-轨迹关联”本质上是一个指派问题：
//   n 条轨迹、m 个检测，如何一一配对使总代价最小。
// 贪心最近邻在目标交叉时会抢错，匈牙利能得到全局最优解。
// =============================================================================
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace aegis {

// 输入 cost[i][j]（i=轨迹，j=检测），返回 assign[i] = 匹配到的检测下标，-1 表示未匹配。
// 这里用经典的 O(n^3) 势函数实现（Jonker-Volgenant 的简化版）。
inline std::vector<int> hungarian(const std::vector<std::vector<double>>& cost) {
  const int n = static_cast<int>(cost.size());
  if (n == 0) return {};
  const int m = static_cast<int>(cost[0].size());
  if (m == 0) return std::vector<int>(n, -1);

  const int dim = std::max(n, m);
  const double kInf = std::numeric_limits<double>::max() / 4.0;

  // 扩展成方阵；不存在的边给一个很大的代价，保证不会被优先选中。
  std::vector<std::vector<double>> a(dim + 1, std::vector<double>(dim + 1, 0.0));
  for (int i = 1; i <= dim; ++i) {
    for (int j = 1; j <= dim; ++j) {
      if (i <= n && j <= m) {
        a[i][j] = cost[i - 1][j - 1];
      } else {
        a[i][j] = kInf;
      }
    }
  }

  std::vector<double> u(dim + 1, 0.0);
  std::vector<double> v(dim + 1, 0.0);
  std::vector<int> p(dim + 1, 0);
  std::vector<int> way(dim + 1, 0);

  for (int i = 1; i <= dim; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(dim + 1, kInf);
    std::vector<char> used(dim + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = kInf;
      int j1 = 0;
      for (int j = 1; j <= dim; ++j) {
        if (used[j]) continue;
        const double cur = a[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= dim; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assign(n, -1);
  for (int j = 1; j <= dim; ++j) {
    if (p[j] >= 1 && p[j] <= n && j <= m) assign[p[j] - 1] = j - 1;
  }
  return assign;
}

}  // namespace aegis
