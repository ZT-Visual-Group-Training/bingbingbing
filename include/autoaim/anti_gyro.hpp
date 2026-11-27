// 反陀螺（Anti-Gyro）：对付"原地自旋"的敌方机器人。
//
// ===========================================================================
// 为什么需要单独的模块（而不是继续用卡尔曼）
//
// 卡尔曼做的是「单块装甲的匀速运动」模型：状态是 [x y z vx vy vz]，
// 它假设目标是"沿直线匀速走"。
//
// 但陀螺车的行为完全不一样：
//   - 它原地自旋，装甲板的**中心点**沿一个【空间圆】运动
//   - 圆周运动的速度方向一直在变，所以"匀速直线"模型一直在跟错
//   - 四块装甲刚体固定（互相 90°），同一时刻只有一块朝向你
//   - 你看到的那块一旦转走，另一块马上出现 —— 位置会"跳"过去
//
// 所以本层的模型是：
//   【四块装甲（刚体，间隔 90°）+ 绕某轴以角速度 ω 匀速旋转】
// 观测量（来自卡尔曼平滑后的单块装甲 3D 坐标）只是这个刚体上的一个点。
//
// 数学上要估 4 个量：
//   1. 旋转中心 c（3D 点，相机系）
//   2. 旋转轴 n（单位向量，理想情况是竖直的）
//   3. 角速度 ω（rad/s，符号表示转向）
//   4. 当前相位 θ（当前这块装甲在圆上的角度）
//
// ===========================================================================
// 算法（三步，全部离线可解释）
//
//   ① 拟合旋转平面：把最近 N 帧的平滑位置收进环形缓冲，去均值后做 SVD，
//      最小奇异值对应的右奇异向量就是平面法向 = 旋转轴 n。
//      → 这一步用 OpenCV 的 cv::SVD（矩阵分解），几何推导是手写的。
//
//   ② 拟合圆：把点投影到平面基 (u, v) 上，做代数最小二乘圆拟合
//      （解 3 元线性方程 x²+y² + D x + E y + F = 0），得到圆心和半径。
//      → 线性方程组用 OpenCV 的 cv::solve，构建方程是手写的。
//
//   ③ 拟合角速度：对每帧算 atan2 角度，展开（去 2π 跳变）后做一元线性拟合，
//      斜率 = 每帧角增量，乘 fps 得到 ω。符号就是转向。
//      → 角度展开和线性拟合都是手写逻辑。
//
// ===========================================================================
// 边界处理（三条，必须有）
//
//   1. 样本不够（< min_samples）：不输出陀螺结果，main 继续用卡尔曼。
//   2. 拟合质量差（半径太小 / 平面残差太大 / ω 太小）：判为"不是陀螺"。
//   3. 连续丢帧超过 reset_frames：清空缓冲，等下次重新积累。
//
// ===========================================================================
// 与主循环的接入顺序
//
//   检测 → 配对 → 角点 → PnP → 卡尔曼(平滑 + 预测) → 【反陀螺】 → 选装甲 → 云台
//                                              ↑ 本模块
//   注意：本模块**只读**卡尔曼的输出，不修改它；卡尔曼仍然是唯一的"当前块"来源。
// ===========================================================================
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <deque>
#include <vector>

// ---- 调试开关 -------------------------------------------------------------
// 打开后每 N 帧打印一次：旋转中心、轴、角速度、转向、四块装甲坐标与最优选择。
// 关闭后所有打印代码被预处理器剔除，运行时零开销。
#ifndef DEBUG_GYRO
#define DEBUG_GYRO 0
#endif

namespace autoaim {

/// 反陀螺的一整套输出（main 拿去投影、绘制、选目标）。
struct GyroState {
  bool valid = false;                 // 本帧是否成功输出陀螺模型
  cv::Vec3d centre{0, 0, 0};          // 旋转中心（相机坐标系，米）
  cv::Vec3d axis{0, 1, 0};            // 旋转轴单位向量（相机系，理想是竖直）
  double radius = 0.0;                // 旋转半径（米），应约等于 ±90° 相位间距
  double omega_dps = 0.0;             // 角速度（度/秒），由圆拟合得到
  // yaw 角速度（度/秒）：直接对 PnP 输出的 yaw 求时间导数。
  // 这是社区的主流判据（比圆拟合稳健得多），也是判断"是否在自旋"的主依据：
  // 目标不转时 ω_yaw ≈ 0；真陀螺时它稳定在 300~700 度/秒。
  double omega_yaw_dps = 0.0;
  int direction = 0;                  // +1 逆时针 / -1 顺时针 / 0 未知
  double angle_now_deg = 0.0;         // 当前这块装甲在圆上的相位角
  std::array<cv::Vec3d, 4> armors;    // 四块装甲中心（当前时刻，0 = 实测那块）
  std::array<double, 4> yaws{0, 0, 0, 0};  // 四块装甲当前时刻的 yaw（度）

  // ---- 提前量：击发时刻的预测（反陀螺真正的价值）------------------------
  // 打陀螺车不能瞄"现在正对的那块"——它转得很快，等开火时已经转走了。
  // 要瞄的是"击发时刻转到我正前方"的那块，哪怕它现在还没出现。
  double predict_ahead_s = 0.0;       // 向前预测了多久（秒）
  std::array<double, 4> yaw_at_fire{0, 0, 0, 0};   // 击发时刻各板的预测 yaw
  std::array<double, 4> t_to_face_s{-1, -1, -1, -1};  // 还需多久该板最正对（秒）
  int best = -1;                      // 击发时刻最正对的那块 → 交给云台
  double best_yaw_deg = 0.0;          // 该块在击发时刻的 yaw（越小越正对）
  cv::Vec3d best_pos_at_fire{0, 0, 0};  // 该块在击发时刻的三维位置
  bool best_is_current = false;       // true = 就是现在看到的那块（无需提前）
  int samples = 0;                    // 参与拟合的样本数（调试用）
  double fit_rms = 0.0;               // 平面拟合残差（米），调试用
};

/// 反陀螺估计器。
///
/// 用法（主循环里，卡尔曼 update 之后）：
/// ```cpp
/// GyroEstimator gyro(cfg.gyro);
/// ...
/// const cv::Vec3d smoothed = tracker.position();     // 卡尔曼平滑后的位置
/// gyro.push(smoothed, yaw_deg, now_seconds);         // 喂一帧观测
/// const GyroState& st = gyro.state();
/// if (st.valid) { /* 用 st.armors / st.best */ }
/// ```
class GyroEstimator {
 public:
  /// 参数（全部来自 config/default.yaml 的 gyro 段）
  struct Params {
    bool enabled = true;      // 总开关（关掉后 push() 直接返回，不占算力）
    int window = 45;          // 环形缓冲长度（帧）。166 fps 下 45 帧 ≈ 0.27 秒
    int min_samples = 12;     // 少于这个样本数不做拟合（0.07 秒，约 12° 转角）
    double min_omega_dps = 150.0;  // 圆拟合角速度下限
    // yaw 角速度下限（度/秒）。社区常用 200~300：低于它认为目标没有自旋。
    // 这是第一道、也是最可靠的一道"是否陀螺"判据。
    double min_yaw_rate_dps = 200.0;
    double max_plane_rms_m = 0.05; // 平面拟合残差上限（5 cm），超了判为不成立
    double min_radius_m = 0.03;    // 旋转半径下限（装甲中心到转轴的距离约 0.075 m）
    double max_radius_m = 0.60;    // 半径上限，超了说明拟合跑飞
    int reset_frames = 30;    // 连续多少帧没有新观测就清空（≈0.18 秒）
    // 提前量：向前预测多久来挑"击发时刻该瞄哪块"。
    // 典型取 50~120 ms（含总延迟 + 一点点瞄准余量），决定"瞄现在还是瞄下一块"。
    double predict_ahead_s = 0.08;
    // 认为"已经正对"的 yaw 门限（度）。击发时刻 |yaw| 小于它就认为可直接打。
    double face_on_deg = 25.0;

    // ---- 稳定性判据（区分"真陀螺"和"拟合噪声"，最关键的一条）--------
    // 卡尔曼平滑后的轨迹非常光滑，任意一段缓曲线都能被"小半径圆"拟合出
    // 0.3 mm 的残差，所以单帧"拟合得好"不能证明它在转。
    // 真陀螺车的 ω 和旋转中心必须在时间上稳定，于是要求：
    //   ω 的相对标准差 < max_omega_rel_std
    //   旋转中心在窗口内的最大偏移 < max_centre_spread_m
    //   ω 的符号一致率 >= min_sign_consistency
    // 实测：非陀螺数据上 ω 在 -574 ~ +550 乱跳、中心跳 0.2 m，三条全不过。
    int stabilize_window = 10;
    double max_omega_rel_std = 0.25;
    double max_centre_spread_m = 0.08;
    double min_sign_consistency = 0.8;
    // 最硬的一条：让模型预测"下一帧装甲中心会在哪"，再和实测比。
    // 真陀螺的模型能预测到 1 cm 以内；拟合噪声的预测误差会大一个量级。
    // 这一条同时挡住"参数缓慢漂移但短时间内自洽"的假拟合。
    double max_pred_err_m = 0.020;   // 一步预测误差上限（2 cm）
    // 半径的物理先验：装甲中心到转轴的距离约为 0.075 m（小装甲）/ 0.115 m（大装甲）。
    // 拟合出的半径偏离太多，说明"圆"是噪声凑出来的，不是真的刚体旋转。
    double expected_radius_m = 0.075;
    double radius_tol = 0.45;
  };

  GyroEstimator() = default;
  explicit GyroEstimator(const Params& p) : p_(p) {}

  void configure(const Params& p) { p_ = p; }
  const Params& params() const { return p_; }

  /// 喂一帧观测（卡尔曼平滑后的装甲中心 + 该帧解出的 yaw）。
  /// @param pos    相机系位置（米）
  /// @param yaw_deg 该装甲的 yaw（度），用来给另外三块推 yaw
  /// @param t_sec  时间戳（秒），用于算角速度；传 0 则退化为"按帧数算"
  void push(const cv::Vec3d& pos, double yaw_deg, double t_sec);

  /// 通知本帧没有观测（丢帧）。内部累计 miss 计数，超限自动 reset。
  void miss();

  /// 当前估计结果（push/miss 之后调用）。
  const GyroState& state() const { return st_; }

  /// 清空所有历史（切换目标、长时丢失、人工重置时调用）。
  void reset();

  /// 本帧是否可以输出陀螺结果
  bool ready() const { return st_.valid; }
  /// 上一次被判无效的原因（自检/调试用）。空串表示没有失败。
  const char* failReason() const { return fail_reason_; }
  /// 调试：最近一次拟合的 ω 与其历史统计（不依赖是否 valid）
  double lastOmega() const { return last_omega_; }
  double lastOmegaRelStd() const { return last_rel_std_; }
  int omegaHistorySize() const { return static_cast<int>(omega_hist_.size()); }

 private:
  // ---- 内部拟合流程 ------------------------------------------------------
  bool fitPlane(cv::Vec3d* centre, cv::Vec3d* axis, double* rms) const;
  bool fitCircle(const cv::Vec3d& centre, const cv::Vec3d& axis, cv::Vec3d* circle_c,
                 double* radius) const;
  bool fitOmega(const cv::Vec3d& circle_c, const cv::Vec3d& axis, double* omega_dps,
                double* angle_now_deg) const;
  void buildArmors(const cv::Vec3d& circle_c, const cv::Vec3d& axis, double radius,
                   double angle_now_deg);
  /// 由 yaw 序列求角速度（度/秒）。社区主流判据：单帧可观、比圆拟合稳健。
  bool fitYawRate(double* omega_dps) const;

  Params p_;
  GyroState st_;

  // 环形缓冲：观测（位置 + yaw + 时间戳）
  struct Sample {
    cv::Vec3d pos;
    double yaw_deg = 0.0;
    double t_sec = 0.0;
  };
  std::deque<Sample> buf_;
  // 最近若干次估计结果，用于判断参数在时间上是否稳定（真陀螺的必要条件）
  std::deque<double> omega_hist_;
  std::deque<cv::Vec3d> centre_hist_;
  // 一步预测误差历史：模型预测下一帧位置 vs 实际位置的偏差
  std::deque<double> pred_err_hist_;
  cv::Vec3d last_pred_{0, 0, 0};   // 上一帧算出的"下一帧位置预测"
  bool has_last_pred_ = false;
  bool stable() const;   // 稳定性判定
  int miss_count_ = 0;   // 连续丢帧计数
  mutable const char* fail_reason_ = "";   // 最近一次判无效的原因
  mutable double last_omega_ = 0.0;     // 最近一次拟合出的 ω（度/秒）
  mutable double last_rel_std_ = 0.0;   // 最近一次 ω 历史相对标准差
  double last_t_ = 0.0;  // 上一帧时间戳，用来算 dt
};

}  // namespace autoaim
