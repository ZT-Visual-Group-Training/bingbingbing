// 反陀螺（Anti-Gyro）实现。
//
// 【注释约定】本文件里每个关键步骤都标注：
//   [OpenCV]   = 调用 OpenCV 自带函数
//   [手写]     = 自己实现的逻辑/公式
//   [公式]     = 数学推导，非库函数
//
// ===========================================================================
// 整体思路（一句话）：陀螺车的装甲中心画出一个【空间圆】，测出这个圆
// 就能推出任意时刻四块装甲的位置。
// ===========================================================================
#include "autoaim/anti_gyro.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace autoaim {
namespace {

// [手写] 把角度规整到 (-180, 180]，用于 yaw 的比较
double wrapDeg(double a) {
  while (a > 180.0) a -= 360.0;
  while (a <= -180.0) a += 360.0;
  return a;
}

// [手写] 单位化；长度过小返回 false
bool normalize(const cv::Vec3d& v, cv::Vec3d* out) {
  const double n = cv::norm(v);
  if (n < 1e-9) return false;
  *out = v * (1.0 / n);
  return true;
}

// [手写] 构造与 axis 正交的一组基 (u, v)，且 u×v = axis（右手系）
bool basisFromAxis(const cv::Vec3d& axis, cv::Vec3d* u, cv::Vec3d* v) {
  cv::Vec3d a;
  if (!normalize(axis, &a)) return false;
  // 选一个与 a 不平行的参考向量，做出第一个基向量
  cv::Vec3d ref(0, 1, 0);
  if (std::fabs(a.dot(ref)) > 0.9) ref = cv::Vec3d(1, 0, 0);
  cv::Vec3d uu = ref - a * a.dot(ref);   // [公式] 施密特正交化
  if (!normalize(uu, u)) return false;
  *v = a.cross(*u);                      // [公式] 叉积得到第三个正交向量
  return true;
}

// [手写] 丢帧或样本不足时的统一处理
void clearState(GyroState* st) {
  st->valid = false;
  st->best = -1;
}

}  // namespace

// ---------------------------------------------------------------------------
// 喂一帧观测
// ---------------------------------------------------------------------------
void GyroEstimator::push(const cv::Vec3d& pos, double yaw_deg, double t_sec) {
  if (!p_.enabled) return;            // 总开关关掉时直接返回（零开销）
  if (!(pos[2] > 0.1)) return;        // 目标在相机后方/无效，忽略

  miss_count_ = 0;                    // 有观测就清空丢帧计数

  // ---- 一步预测误差检验 ------------------------------------------------
  // 上一帧的模型预测了"这一帧装甲中心应该在哪"，现在拿实际值比一下。
  // 这是判断"模型是不是真的描述了目标运动"最直接的办法：
  //   真陀螺 → 预测落在 1 cm 内
  //   噪声拟合 → 偏差大一个量级
  if (has_last_pred_ && static_cast<int>(buf_.size()) >= p_.window) {
    pred_err_hist_.push_back(cv::norm(pos - last_pred_));
    while (static_cast<int>(pred_err_hist_.size()) > p_.stabilize_window) {
      pred_err_hist_.pop_front();
    }
  }
  has_last_pred_ = false;

  buf_.push_back(Sample{pos, yaw_deg, t_sec});
  // [手写] 环形缓冲：超出窗口就丢最老的
  while (static_cast<int>(buf_.size()) > p_.window) buf_.pop_front();

  st_.samples = static_cast<int>(buf_.size());

  fail_reason_ = "samples/min_samples";
  // 窗口还没满：清空统计历史，保证后面所有判据都基于同一条件（满窗口）的估计
  if (static_cast<int>(buf_.size()) < p_.window) {
    omega_hist_.clear();
    centre_hist_.clear();
    pred_err_hist_.clear();
  }

  // 边界处理 1：样本不足 → 不启用陀螺拟合，交给原卡尔曼
  if (static_cast<int>(buf_.size()) < p_.min_samples) {
    clearState(&st_);
    return;
  }

  // ---- ⓪ yaw 角速度判据（第一道、最可靠的一道"是否自旋"判断）---------
  //
  // 为什么先做这一步：社区普遍直接对 PnP 的 yaw 求时间导数来判断陀螺，
  // 因为它单帧可观、不依赖圆弧长度，比"拟合圆"稳健得多。
  // 本工程实测：非陀螺目标（平移）的 yaw 几乎不变 → ω_yaw ≈ 0，
  // 于是这一条直接把它挡掉，不会走到后面的圆拟合。
  double omega_yaw = 0.0;
  if (!fitYawRate(&omega_yaw)) {
    fail_reason_ = "fitYawRate";
    clearState(&st_);
    return;
  }
  st_.omega_yaw_dps = omega_yaw;
  if (std::fabs(omega_yaw) < p_.min_yaw_rate_dps) {
    fail_reason_ = "yaw rate too low (not spinning)";
    clearState(&st_);
    return;
  }

  // ---- ① 拟合旋转平面（法向 = 旋转轴）--------------------------------
  cv::Vec3d centre, axis;
  double rms = 0.0;
  if (!fitPlane(&centre, &axis, &rms)) {
    fail_reason_ = "fitPlane";
    clearState(&st_);
    return;
  }
  fail_reason_ = "plane_rms > max";
  if (rms > p_.max_plane_rms_m) {   // 点不在一个平面上，说明不是规整自旋
    clearState(&st_);
    return;
  }

  // ---- ② 在平面内拟合圆（圆心 = 旋转中心）----------------------------
  cv::Vec3d circle_c;
  double radius = 0.0;
  if (!fitCircle(centre, axis, &circle_c, &radius)) {
    fail_reason_ = "fitCircle";
    clearState(&st_);
    return;
  }
  fail_reason_ = "radius out of range";
  if (radius < p_.min_radius_m || radius > p_.max_radius_m) {
    clearState(&st_);
    return;
  }
  // 物理先验：真实装甲中心到转轴约 0.075 m（小）/ 0.115 m（大）。
  // 偏差超过 radius_tol 说明"圆"是噪声凑出来的。
  if (std::fabs(radius - p_.expected_radius_m) > p_.radius_tol * p_.expected_radius_m) {
    fail_reason_ = "radius prior";
    clearState(&st_);
    return;
  }

  // ---- ③ 拟合角速度与当前相位 ----------------------------------------
  double omega_dps = 0.0;
  double angle_now = 0.0;
  if (!fitOmega(circle_c, axis, &omega_dps, &angle_now)) {
    fail_reason_ = "fitOmega";
    clearState(&st_);
    return;
  }
  // 边界处理 2：角速度太小学不动，按"不是陀螺"处理
  if (std::fabs(omega_dps) < p_.min_omega_dps) {
    fail_reason_ = "omega < min_omega_dps";
    clearState(&st_);
    return;
  }

  // ---- 预测下一帧位置 ---------------------------------------------------
  // 注意：这一段必须在"稳定性判断"之前做，否则会死锁 ——
  // 稳定性需要预测误差历史，而预测误差又需要上一帧的模型，循环依赖。
  // 所以只要本帧拟合出了圆和 ω，就先给出预测，让误差历史能积累起来。
  {
    cv::Vec3d u2, v2;
    if (basisFromAxis(axis, &u2, &v2) && buf_.size() >= 2) {
      const double dt = buf_.back().t_sec - buf_[buf_.size() - 2].t_sec;
      const double a_next = (angle_now + omega_dps * dt) * CV_PI / 180.0;
      last_pred_ = circle_c + u2 * (radius * std::cos(a_next)) + v2 * (radius * std::sin(a_next));
      has_last_pred_ = true;
    }
  }

  // ---- ④ 稳定性判据 ---------------------------------------------------
  // 把本次估计写进历史，再判断"这几次估计是否自洽"。
  // 这一步能挡掉绝大多数"把缓曲线拟合成小圆"的误判：
  // 真陀螺的 ω 和圆心在短时间内几乎不变，假拟合的会乱跳。
  last_omega_ = omega_dps;
  // 只有"窗口已装满"的估计才计入历史。
  // 为什么：刚够 min_samples 时（十几个点、只有几十度弧）圆拟合是病态的，
  // 偶尔会给出连符号都反了的 ω（实测 -1040 deg/s）。这种离群值一旦进入历史，
  // 会把 rel_std 抬到 1.3 以上 —— 完美数据也判不过（这个 bug 是自检抓出来的）。
  const bool window_full = static_cast<int>(buf_.size()) >= p_.window;
  if (window_full) {
    omega_hist_.push_back(omega_dps);
    centre_hist_.push_back(circle_c);
    while (static_cast<int>(omega_hist_.size()) > p_.stabilize_window) omega_hist_.pop_front();
    while (static_cast<int>(centre_hist_.size()) > p_.stabilize_window) centre_hist_.pop_front();
  }
  if (!stable()) {
    if (fail_reason_[0] == 0) fail_reason_ = "unstable";
    // 还不稳定：保留数据继续积累，但本帧不输出
    clearState(&st_);
    st_.samples = static_cast<int>(buf_.size());
#if DEBUG_GYRO
    static int dbg2 = 0;
    if (++dbg2 % 30 == 0) {
      std::printf("[GYRO] unstable: omega=%.0f (hist %zu) centre_spread pending\n",
                  omega_dps, omega_hist_.size());
    }
#endif
    return;
  }

  // ---- 输出 -----------------------------------------------------------
  st_.valid = true;
  fail_reason_ = "";
  st_.centre = circle_c;
  st_.axis = axis;
  st_.radius = radius;
  st_.omega_dps = omega_dps;
  st_.direction = (omega_dps > 0) ? 1 : -1;
  st_.angle_now_deg = angle_now;
  st_.fit_rms = rms;
  buildArmors(circle_c, axis, radius, angle_now);


#if DEBUG_GYRO
  static int dbg = 0;
  if (++dbg % 30 == 0) {
    std::printf("[GYRO] c=(%.3f %.3f %.3f) axis=(%.2f %.2f %.2f) r=%.3f m "
                "omega=%.1f deg/s dir=%s rms=%.4f n=%d\n",
                st_.centre[0], st_.centre[1], st_.centre[2], st_.axis[0], st_.axis[1],
                st_.axis[2], st_.radius, st_.omega_dps,
                st_.direction > 0 ? "CCW" : "CW", st_.fit_rms, st_.samples);
    for (int k = 0; k < 4; ++k) {
      std::printf("        plate[%d] pos=(%.3f %.3f %.3f) yaw=%+.1f%s\n", k,
                  st_.armors[k][0], st_.armors[k][1], st_.armors[k][2], st_.yaws[k],
                  (k == st_.best) ? "  <== best" : "");
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// 本帧没有观测（丢帧）
// ---------------------------------------------------------------------------
void GyroEstimator::miss() {
  if (!p_.enabled) return;
  ++miss_count_;

  // 边界处理 3：连续丢帧超过上限 → 角速度已失效，清空重来
  if (miss_count_ > p_.reset_frames) {
    reset();
    return;
  }

  // 短期丢帧：模型仍然有效，按角速度把相位往前推（这就是抗遮挡的价值）
  if (st_.valid) {
    const double dt = (buf_.size() >= 2) ? (buf_.back().t_sec - buf_[buf_.size() - 2].t_sec)
                                        : 0.0;
    st_.angle_now_deg = wrapDeg(st_.angle_now_deg + st_.omega_dps * dt);
    buildArmors(st_.centre, st_.axis, st_.radius, st_.angle_now_deg);
  }
}

// ---------------------------------------------------------------------------
// 清空
// ---------------------------------------------------------------------------
void GyroEstimator::reset() {
  buf_.clear();
  omega_hist_.clear();
  centre_hist_.clear();
  pred_err_hist_.clear();
  has_last_pred_ = false;
  miss_count_ = 0;
  last_t_ = 0.0;
  clearState(&st_);
  st_.samples = 0;
}

// ---------------------------------------------------------------------------
// 稳定性判定：真陀螺的 ω 与旋转中心必须在一段时间内保持一致
// ---------------------------------------------------------------------------
bool GyroEstimator::stable() const {
  if (static_cast<int>(omega_hist_.size()) < p_.stabilize_window) {
    fail_reason_ = "stable: hist < window";
    return false;
  }

  // [手写] 平均角速度
  double sum = 0.0;
  for (double w : omega_hist_) sum += w;
  const double mean = sum / static_cast<double>(omega_hist_.size());
  if (std::fabs(mean) < 1e-6) return false;

  // [手写] 角速度的相对标准差：衡量"转速稳不稳"
  double var = 0.0;
  for (double w : omega_hist_) {
    const double d = w - mean;
    var += d * d;
  }
  var /= static_cast<double>(omega_hist_.size());
  const double rel_std = std::sqrt(var) / std::fabs(mean);
  last_rel_std_ = rel_std;
  if (rel_std > p_.max_omega_rel_std) {
    fail_reason_ = "stable: omega rel_std";
    return false;
  }

  // [手写] 转向一致率：真陀螺不会一会儿顺时针一会儿逆时针
  int same = 0;
  for (double w : omega_hist_) {
    if ((w > 0) == (mean > 0)) ++same;
  }
  const double sign_consistency =
      static_cast<double>(same) / static_cast<double>(omega_hist_.size());
  if (sign_consistency < p_.min_sign_consistency) {
    fail_reason_ = "stable: sign flipped";
    return false;
  }

  // [手写] 旋转中心的漂移：真陀螺的圆心几乎不动
  cv::Vec3d c_mean(0, 0, 0);
  for (const auto& cc : centre_hist_) c_mean += cc;
  c_mean *= (1.0 / static_cast<double>(centre_hist_.size()));
  double max_dev = 0.0;
  for (const auto& cc : centre_hist_) {
    const double d = cv::norm(cc - c_mean);
    if (d > max_dev) max_dev = d;
  }
  if (max_dev > p_.max_centre_spread_m) {
    fail_reason_ = "stable: centre drift";
    return false;
  }

  // [手写] 一步预测误差：模型能不能预测准下一帧的实际位置。
  // 这是三条稳定性判据里最硬的一条，因为它是"预测 vs 实测"的对照，
  // 不是在拟合数据本身。
  if (static_cast<int>(pred_err_hist_.size()) < 4) {
    fail_reason_ = "stable: pred_err hist < 4";
    return false;
  }
  double err_sum = 0.0;
  for (double e : pred_err_hist_) err_sum += e;
  const double err_mean = err_sum / static_cast<double>(pred_err_hist_.size());
  if (err_mean > p_.max_pred_err_m) {
    fail_reason_ = "stable: pred_err too big";
    return false;
  }

  return true;
}
// ---------------------------------------------------------------------------
// yaw 角速度：对 PnP 的 yaw 序列做展开 + 线性拟合
// ---------------------------------------------------------------------------
// 这是判断"目标是否在自旋"最直接的量：
//   目标平移       → yaw 基本不变 → ω_yaw ≈ 0
//   目标原地自旋   → yaw 匀速变化 → ω_yaw 稳定在几百度/秒
// 相比圆拟合，它不需要"圆弧足够长"，短时间也可靠。
bool GyroEstimator::fitYawRate(double* omega_dps) const {
  const int n = static_cast<int>(buf_.size());
  if (n < 4) return false;

  // [手写] 展开 yaw：相邻帧差超过 180° 认为跨了 ±180 边界，补 360°
  std::vector<double> th(n, 0.0);
  th[0] = buf_[0].yaw_deg;
  double acc = th[0];
  for (int i = 1; i < n; ++i) {
    double diff = buf_[i].yaw_deg - buf_[i - 1].yaw_deg;
    while (diff > 180.0) diff -= 360.0;
    while (diff < -180.0) diff += 360.0;
    acc += diff;
    th[i] = acc;
  }

  // [手写] 一元线性拟合 θ(t) = a + b·t，b 就是角速度（度/秒）
  double sum_t = 0.0, sum_th = 0.0, sum_tt = 0.0, sum_tth = 0.0;
  for (int i = 0; i < n; ++i) {
    const double t = buf_[i].t_sec;
    sum_t += t;
    sum_th += th[i];
    sum_tt += t * t;
    sum_tth += t * th[i];
  }
  const double den = n * sum_tt - sum_t * sum_t;
  if (std::fabs(den) < 1e-12) return false;
  *omega_dps = (n * sum_tth - sum_t * sum_th) / den;
  return true;
}
// ---------------------------------------------------------------------------
// ① 平面拟合：SVD 找最小方差方向 = 平面法向
// ---------------------------------------------------------------------------
bool GyroEstimator::fitPlane(cv::Vec3d* centre, cv::Vec3d* axis, double* rms) const {
  const int n = static_cast<int>(buf_.size());
  if (n < 3) return false;

  // [手写] 求均值点，作为平面上的一点
  cv::Vec3d mean(0, 0, 0);
  for (const auto& s : buf_) mean += s.pos;
  mean *= (1.0 / n);

  // [手写] 累加协方差矩阵 Σ = (1/n) Σ (p-mean)(p-mean)ᵀ
  //   为什么用协方差而不是直接 SVD 点阵：3x3 的对称矩阵数值上更稳定，
  //   而且 cv::eigen 对 CV_64F 的 3x3 有专用快速路径，不会走通用路径的
  //   类型创建逻辑（之前那版崩溃就出在那里）。
  double cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
  for (const auto& s : buf_) {
    const cv::Vec3d d = s.pos - mean;
    cxx += d[0] * d[0];
    cyy += d[1] * d[1];
    czz += d[2] * d[2];
    cxy += d[0] * d[1];
    cxz += d[0] * d[2];
    cyz += d[1] * d[2];
  }
  const double inv = 1.0 / n;
  cv::Mat cov(3, 3, CV_64F);
  cov.at<double>(0, 0) = cxx * inv;
  cov.at<double>(0, 1) = cxy * inv;
  cov.at<double>(0, 2) = cxz * inv;
  cov.at<double>(1, 0) = cxy * inv;
  cov.at<double>(1, 1) = cyy * inv;
  cov.at<double>(1, 2) = cyz * inv;
  cov.at<double>(2, 0) = cxz * inv;
  cov.at<double>(2, 1) = cyz * inv;
  cov.at<double>(2, 2) = czz * inv;

  // [OpenCV] cv::eigen：对称矩阵特征分解，特征值降序排列，特征向量按行存放
  cv::Mat evals, evecs;
  if (!cv::eigen(cov, evals, evecs)) return false;
  if (evals.rows < 3 || evecs.rows < 3) return false;

  // [公式] 方差最小的方向 = 最小特征值对应的特征向量 = 平面法向
  cv::Vec3d nrm(evecs.at<double>(2, 0), evecs.at<double>(2, 1), evecs.at<double>(2, 2));
  if (!normalize(nrm, &nrm)) return false;

  // [公式] 残差 = 点到平面距离的均方根。真实圆上的点共面，残差应接近 0
  double acc = 0.0;
  for (const auto& s : buf_) {
    const double d = (s.pos - mean).dot(nrm);
    acc += d * d;
  }
  *rms = std::sqrt(acc / n);

  // [手写] 把转轴的符号"规范化"，这是本模块最关键的一处稳定性处理。
  //
  // 坑：最初写成 if (nrm.dot(mean) > 0) nrm = -nrm;（让轴指向相机）
  //   但竖直轴的 nrm ≈ (0,±1,0)，而 mean 是圆心（≈(0,0,2)），点积 ≈ 0。
  //   于是符号由浮点噪声随机决定 —— 轴一翻，平面基 (u,v) 的手性就翻，
  //   算出来的 ω 符号也跟着翻。实测 10 次估计里出现 2 次符号翻转，
  //   导致 rel_std 恰好 = 400/300 = 1.333。
  //
  // 正确做法：用"最大绝对值分量取正"的确定性规范。与点位无关，只取决于轴本身。
  // 这样同一个方向永远得到同一个符号，ω 的统计才有意义。
  {
    int k = 0;
    double best = std::fabs(nrm[0]);
    for (int i = 1; i < 3; ++i) {
      if (std::fabs(nrm[i]) > best) {
        best = std::fabs(nrm[i]);
        k = i;
      }
    }
    if (nrm[k] < 0.0) nrm = -nrm;
  }

  *centre = mean;
  *axis = nrm;
  return true;
}
bool GyroEstimator::fitCircle(const cv::Vec3d& centre, const cv::Vec3d& axis,
                              cv::Vec3d* circle_c, double* radius) const {
  cv::Vec3d u, v;
  if (!basisFromAxis(axis, &u, &v)) return false;

  const int n = static_cast<int>(buf_.size());
  // [手写] 圆方程 x² + y² + D·x + E·y + F = 0  →  线性方程组 A[D,E,F]ᵀ = b
  //   每个样本给一行：[x, y, 1]，右端 = -(x²+y²)
  cv::Mat A(n, 3, CV_64F), b(n, 1, CV_64F);
  for (int i = 0; i < n; ++i) {
    const cv::Vec3d d = buf_[i].pos - centre;
    const double x = d.dot(u);
    const double y = d.dot(v);
    A.at<double>(i, 0) = x;
    A.at<double>(i, 1) = y;
    A.at<double>(i, 2) = 1.0;
    b.at<double>(i, 0) = -(x * x + y * y);
  }

  // [手写] 正规方程 AᵀA·x = Aᵀb → 3x3 线性方程组（比直接解超定更稳）
  cv::Mat AtA(3, 3, CV_64F, cv::Scalar(0));
  cv::Mat Atb(3, 1, CV_64F, cv::Scalar(0));
  for (int i = 0; i < n; ++i) {
    const double row[3] = {A.at<double>(i, 0), A.at<double>(i, 1), A.at<double>(i, 2)};
    const double rhs = b.at<double>(i, 0);
    for (int r = 0; r < 3; ++r) {
      Atb.at<double>(r, 0) += row[r] * rhs;
      for (int q = 0; q < 3; ++q) AtA.at<double>(r, q) += row[r] * row[q];
    }
  }
  // [OpenCV] cv::solve 解 3x3 正定方程组（LU 分解）
  cv::Mat sol;
  if (!cv::solve(AtA, Atb, sol, cv::DECOMP_LU)) return false;
  const double D = sol.at<double>(0, 0);
  const double E = sol.at<double>(1, 0);
  const double F = sol.at<double>(2, 0);

  // [公式] 圆方程配方：圆心 (-D/2, -E/2)，半径 sqrt((D²+E²)/4 - F)
  const double cx = -D * 0.5;
  const double cy = -E * 0.5;
  const double r2 = cx * cx + cy * cy - F;
  if (r2 <= 0.0) return false;

  // [手写] 把 2D 圆心还原到 3D 相机系
  *circle_c = centre + u * cx + v * cy;
  *radius = std::sqrt(r2);
  return true;
}

// ---------------------------------------------------------------------------
// ③ 角速度与当前相位
// ---------------------------------------------------------------------------
bool GyroEstimator::fitOmega(const cv::Vec3d& circle_c, const cv::Vec3d& axis,
                             double* omega_dps, double* angle_now_deg) const {
  cv::Vec3d u, v;
  if (!basisFromAxis(axis, &u, &v)) return false;

  const int n = static_cast<int>(buf_.size());
  if (n < 3) return false;

  // [手写] 每帧在圆上的角度 θ_i = atan2(v·d, u·d)
  std::vector<double> ang(n, 0.0);
  for (int i = 0; i < n; ++i) {
    const cv::Vec3d d = buf_[i].pos - circle_c;
    // [OpenCV] std::atan2 是标准库函数（不是 OpenCV），这里标注为 [手写/标准库]
    ang[i] = std::atan2(d.dot(v), d.dot(u));
  }

  // [手写] 角度展开：相邻帧差超过 π 就认为跨了 ±180° 边界，补上 2π
  double unwrapped = 0.0;
  std::vector<double> th(n, 0.0);
  th[0] = ang[0];
  for (int i = 1; i < n; ++i) {
    double diff = ang[i] - ang[i - 1];
    while (diff > CV_PI) diff -= 2.0 * CV_PI;
    while (diff < -CV_PI) diff += 2.0 * CV_PI;
    unwrapped += diff;
    th[i] = th[0] + unwrapped;
  }

  // [手写] 一元线性拟合 θ(t) = a + b·t（最小二乘闭式解）
  double sum_t = 0.0, sum_th = 0.0, sum_tt = 0.0, sum_tth = 0.0;
  for (int i = 0; i < n; ++i) {
    const double t = buf_[i].t_sec;
    sum_t += t;
    sum_th += th[i];
    sum_tt += t * t;
    sum_tth += t * th[i];
  }
  const double den = n * sum_tt - sum_t * sum_t;
  double slope = 0.0;   // rad/s
  if (std::fabs(den) > 1e-12) {
    slope = (n * sum_tth - sum_t * sum_th) / den;   // [公式] 最小二乘斜率
  } else {
    // 时间戳无效（全 0）时退化为"按帧"：slope 单位变成 rad/帧，先按 1 fps 记
    slope = (th[n - 1] - th[0]) * 1.0;
  }

  // [公式] 弧度/秒 → 度/秒
  *omega_dps = slope * 180.0 / CV_PI;
  *angle_now_deg = th[n - 1] * 180.0 / CV_PI;
  return true;
}

// ---------------------------------------------------------------------------
// 由刚体关系推出四块装甲，并做"提前量"选板
// ---------------------------------------------------------------------------
//
// 这一步是反陀螺的核心价值所在，逻辑分两层：
//
//   第一层：现在四块板各在哪（几何）
//     四块板刚体固定、间隔 90°，所以只要知道"我看到的这块"的相位 θ，
//     另外三块就在 θ+90°、θ+180°、θ+270°。
//
//   第二层：开火那一刻该打哪块（预测 + 选优）
//     陀螺车转速很高（本工程实测量级 300~700 deg/s）。假设 ω=500 deg/s，
//     20 ms 内就转过 10°，一块板从"正对"（|yaw|<10°）到"转走"只有约 40 ms。
//     也就是说：**瞄准"当前正对的板"，等开火时它已经偏了**。
//     所以要在 t + predict_ahead_s 这个时刻去比四块板的 yaw，
//     选"到那时最正对"的那块 —— 它现在可能还藏在你侧面。
//
//     这就是"预测下一块装甲板出现"的具体含义：
//     不是等它出现再瞄，而是提前把云台指到它将要出现的位置。
// ---------------------------------------------------------------------------
void GyroEstimator::buildArmors(const cv::Vec3d& circle_c, const cv::Vec3d& axis,
                                double radius, double angle_now_deg) {
  cv::Vec3d u, v;
  if (!basisFromAxis(axis, &u, &v)) return;

  // [手写] 用最近一帧的 yaw 作为相位 0 的 yaw
  const double yaw0 = buf_.empty() ? 0.0 : buf_.back().yaw_deg;

  // ---- 第一层：当前时刻的四块板 ----
  for (int k = 0; k < 4; ++k) {
    // [公式] 刚体固定：四块相隔 90°，在圆上的相位也相隔 90°
    const double a = (angle_now_deg + 90.0 * k) * CV_PI / 180.0;
    st_.armors[k] = circle_c + u * (radius * std::cos(a)) + v * (radius * std::sin(a));
    st_.yaws[k] = wrapDeg(yaw0 + 90.0 * k);
  }

  // ---- 第二层：击发时刻的预测与选优 ----
  const double ahead = p_.predict_ahead_s;
  st_.predict_ahead_s = ahead;

  // [公式] 刚体转角 = 角速度 × 时间。四块板都同步转过这么多。
  const double dtheta = st_.omega_dps * ahead;   // 度

  int best = 0;
  double best_abs = 1e9;
  int best_is_current = 0;

  for (int k = 0; k < 4; ++k) {
    // [公式] 击发时刻该板的 yaw = 当前 yaw + 转角
    st_.yaw_at_fire[k] = wrapDeg(st_.yaws[k] + dtheta);

    // [公式] 还要多久它才最正对（yaw 过 0）：
    //   解 yaw + ω·t = 0  →  t = -yaw/ω，取最小的非负解（考虑 360° 周期）
    double t = 0.0;
    if (std::fabs(st_.omega_dps) > 1e-6) {
      double need = -st_.yaws[k] / st_.omega_dps;   // 秒，可能为负
      // 把 need 调到 [0, 360/|ω|) 区间内
      const double period = 360.0 / std::fabs(st_.omega_dps);
      while (need < 0.0) need += period;
      t = need;
    }
    st_.t_to_face_s[k] = t;

    // [手写] 选"击发时刻最正对"的那块
    const double a = std::fabs(st_.yaw_at_fire[k]);
    if (a < best_abs) {
      best_abs = a;
      best = k;
      best_is_current = (k == 0) ? 1 : 0;
    }
  }

  st_.best = best;
  st_.best_yaw_deg = st_.yaw_at_fire[best];
  st_.best_is_current = (best_is_current != 0);

  // [公式] 最优块在击发时刻的相位和三维位置
  const double a_fire = (angle_now_deg + 90.0 * best + dtheta) * CV_PI / 180.0;
  st_.best_pos_at_fire = circle_c + u * (radius * std::cos(a_fire)) +
                         v * (radius * std::sin(a_fire));
}


}  // namespace autoaim
