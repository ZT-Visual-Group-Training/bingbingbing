// Tunable parameters, loaded from a YAML file so nothing is hard-coded.
//
// Every field carries the physical reason for its default value: the examiner
// will ask "why 85?" and the answer has to be a measurement, not a guess.
#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace autoaim {

/// Camera intrinsics + distortion, in the OpenCV convention.
struct CameraConfig {
  double fx = 2334.3100677892139;
  double fy = 2335.8999103578203;
  double cx = 722.62133349286796;
  double cy = 534.62083305548731;
  // k1 k2 p1 p2 k3
  double k1 = -0.0800088054505347;
  double k2 = 0.0856750344533676;
  double p1 = 0.002790303946222533;
  double p2 = -0.0007315334877910064;
  double k3 = 0.0;
  int width = 1440;
  int height = 1080;

  // Hand-eye extrinsics from calibration/handeye.yaml. Parsed and reported so the
  // supplied calibration is fully consumed, but the task asks for the plate
  // position in the CAMERA frame, so R/t are not applied by default.
  bool has_camera2gimbal = false;
  double R_camera2gimbal[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double t_camera2gimbal[3] = {0, 0, 0};
  bool has_gimbal2imubody = false;
  double R_gimbal2imubody[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  cv::Mat cameraMatrix() const;
  cv::Mat distCoeffs() const;
};

/// Detection / matching / PnP thresholds.
struct DetectorConfig {
  // ---- colour -------------------------------------------------------------
  // "red" | "blue"
  std::string enemy_color = "red";
  // "rb" (R-B / B-R), "rg" (R-G / B-G) or "cr" (YCrCb Cr / Cb).
  // Measured on the assessment footage, only "rb" keeps the bars whole:
  //   R-G -> 348 px, bars split into 5 fragments (G lifted by overexposure)
  //   R-B -> 1882 px, bars intact (523 / 496 px)
  std::string chroma_mode = "rb";
  int diff_threshold = 85;
  bool use_brightness_gate = true;
  int brightness_min = 80;

  // ---- morphology ---------------------------------------------------------
  int morph_open_kernel = 3;
  int morph_close_w = 3;
  // Light bars are vertical strips, so a break is a *vertical* gap (digit
  // overlap, JPEG block, LED gap). A tall kernel bridges it without merging two
  // horizontally adjacent bars, which a large square kernel would do.
  // Swept on clip A: 17 -> 85%, 19 -> 87%, 21 -> 91.5%, 23 -> 92% (saturates),
  // measured separately on a far segment (frames 0-199) and a near one (800-999).
  int morph_close_h = 23;
  bool do_open = true;
  bool do_close = true;

  // ---- light-bar filter ---------------------------------------------------
  double min_area = 8.0;             // px^2, ~16 m away at 8x1.5 px
  double max_area = 6000.0;          // px^2, ~0.7 m away
  double min_lightbar_ratio = 1.5;   // length / width
  double max_lightbar_ratio = 20.0;
  double min_lightbar_length = 8.0;  // px, 8 px <-> 16.3 m at fx=2334
  double min_fill_ratio = 0.45;      // contourArea / rectArea
  double max_tilt_deg = 50.0;        // bar vs image vertical
  double min_bar_mean = 100.0;       // dominant-channel mean along the centre line

  // ---- pairing ------------------------------------------------------------
  // Tightened from measurements: true pairs had angle diff <= 3 deg and length
  // ratio <= 1.10; cross-plate false pairs sat at 11-18 deg and 1.27-1.52.
  double max_length_ratio = 1.25;
  double min_center_ratio = 1.2;
  double max_center_ratio = 4.2;
  double max_angle_diff_deg = 10.0;
  // The join must run across the bars (along their short axis). Rotation
  // invariant, unlike comparing raw |dy|.
  double max_side_angle_deg = 25.0;
  double ideal_center_ratio = 2.2;  // (135 - ~12 mm) / 56 mm
  double min_quad_aspect = 1.4;
  double max_quad_aspect = 3.2;
  // Reject candidates whose pair score is too poor. Measured on this footage:
  // genuine plates scored 0.30-0.62, background light fixtures that happen to
  // form a 135:56 quad scored 1.14-1.72.
  // 两根灯条的亮度比上限。实测这个特征没有区分度（真目标 p95=1.47，
  // 背景 p95=1.11，背景反而更一致），收紧只会误杀真目标，故默认关闭。
  double max_bar_brightness_ratio = 2.5;
  // 装甲板的最小像素宽度。真板在 2 m 处约 160 px，背景误检只有 25~40 px。
  // 物理上等价于"最小可信距离"：w = fx*0.135/d，45 px 约 7.0 m。
  double min_armor_width_px = 45.0;
  double max_armor_score = 1.0;
  // Outward inflation (px at 1440 width) of the DRAWN plate quad so it visibly
  // encloses the bars. Visual only: solvePnP keeps the real corners.
  double draw_inflate_px = 5.0;
  // 左右两条边向内收的比例（相对四边形宽度）。纯显示量：让框贴着灯条内侧，
  // PnP 仍用外缘角点，所以不影响测距。0.09 ≈ 一个灯条宽度占板宽的比例。
  double draw_inset_ratio = 0.09;
  // 显示用"上下加高"比例（画出的高度 = 角点高度 x (1+ratio)），让框包住数字。
  // 纯显示量：PnP 用 armor_height_m，所以不加高也不影响测距。0 = 不加高。
  double draw_extra_height_ratio = 0.9;
  // 是否绘制未通过时序确认的候选：0 = 不画（默认，画面最干净），1 = 画成暗青
  int draw_tentative = 0;

  // ---- physical armour ----------------------------------------------------
  double armor_width_m = 0.135;
  // Where the plate's left/right edges sit relative to each light bar:
  //   +1 = bar OUTER side (plate width = outer-to-outer)
  //    0 = bar centre line
  //   -1 = bar INNER side (plate width = the gap between the two bars)
  double plate_edge_offset = -1.0;
  // Light-bar length. The detector measures the bar, so this is the px<->mm scale
  // reference used when converting a plate-height difference into pixels.
  double bar_length_m = 0.056;
  // PLATE height. Set this larger than bar_length_m when the plate extends above
  // and below its light bars (the bars are vertical strips inset from the plate
  // edges): the four corners are then extended symmetrically beyond the bar ends.
  // The PnP object points use this same value, so the distance stays consistent.
  // Equal to bar_length_m reproduces the plain "height = bar length" model.
  double armor_height_m = 0.056;

  // ---- PnP ----------------------------------------------------------------
  double max_reproj_err_px = 2.5;
  int refine_iterations = 20;

  // ---- ROI reuse ----------------------------------------------------------
  bool enable_roi = false;
  int roi_pad_px = 200;             // 20 ms x 1 m/s = 12 px, so 10x margin
  int roi_lost_frames = 3;          // fall back to the full frame after this many misses

  // ---- temporal confirmation ----------------------------------------------
  // A candidate only counts as a real target after being seen confirm_hits
  // frames in a row within confirm_gate_m of the previous position. Random
  // background pairs flicker, so they never reach the count. After confirmation,
  // up to confirm_max_misses frames may be missed (prediction holds the target)
  // before it is dropped - that stops a brief occlusion from toggling the target.
  bool enable_confirm = true;
  int confirm_hits = 3;
  int confirm_max_misses = 5;
  double confirm_gate_m = 0.6;
  // 关联判据 1：中心像素距离。166 fps 下帧间隔 6 ms，目标移动不了几个像素，
  // 所以"上一帧中心"和"这一帧候选中心"必须很近；背景误检每帧乱跳，靠这条分开。
  double confirm_gate_px = 60.0;
  // 关联判据 2：四角宽度比。宽度 = fx*0.135/d，所以宽度比等价于距离比；
  // 6 ms 内距离不会大幅变化，比值必须接近 1。
  double confirm_width_ratio = 1.4;

  // ---- 反陀螺（Anti-Gyro）-----------------------------------------------
  // 敌方机器人原地自旋时，装甲中心沿空间圆运动。本段参数控制"圆拟合"的
  // 采样窗口与有效性门限，详见 include/autoaim/anti_gyro.hpp 的注释。
  bool enable_gyro = true;
  int gyro_window = 45;            // 环形缓冲帧数（166 fps 下约 0.27 秒）
  int gyro_min_samples = 12;       // 少于这个样本数不做拟合，回退卡尔曼
  double gyro_min_omega_dps = 150; // 低于此角速度判为"不是陀螺"
  double gyro_max_plane_rms_m = 0.05;  // 平面拟合残差上限（米）
  int gyro_reset_frames = 30;      // 连续丢帧多少后清空陀螺状态
  int draw_gyro = 0;               // 是否把四块装甲投影回画面画圈（默认关，避免误判时闪烁）
  // yaw 角速度下限：社区判"是否自旋"的主判据
  double gyro_min_yaw_rate_dps = 200.0;
  // 提前量（秒）：向前预测多久来挑"击发时刻该瞄哪块"。
  // 陀螺车 500 deg/s 时，80 ms 就转过 40° —— 这才是"瞄下一块"的关键。
  double gyro_predict_ahead_s = 0.08;
  // 认为"已正对"的 yaw 门限（度），用于判断是否还需要提前瞄
  double gyro_face_on_deg = 25.0;
  // 稳定性判据：真陀螺的 ω 和旋转中心必须在一段时间内保持一致，
  // 否则就是"把缓曲线拟合成小圆"的误判（实测非陀螺数据上 ω 会乱跳 1000 deg/s）
  int gyro_stabilize_window = 10;      // 用最近多少次估计判断
  double gyro_max_omega_rel_std = 0.25;    // ω 相对标准差上限
  double gyro_max_centre_spread_m = 0.08;  // 圆心最大漂移（米）
  // 一步预测误差上限：模型预测的下一帧位置 vs 实测。真陀螺 <1 cm，噪声拟合更大
  double gyro_max_pred_err_m = 0.020;
  // 旋转半径的物理先验：装甲中心到转轴的距离（小装甲约 0.075 m，大装甲约 0.115 m）
  double gyro_expected_radius_m = 0.075;
  double gyro_radius_tol = 0.45;   // 允许 ±45% 偏差

  // ---- tracking -----------------------------------------------------------
  bool enable_tracker = true;
  // 延迟补偿时间（秒）：从曝光到真正击发之间的总延迟，用来把位姿外推到击发时刻。
  // 典型值 20~40 ms（曝光 + 传输 + 解算 + 开火指令）。填 0 就退化成"不做预测"。
  double predict_latency_s = 0.020;
  // 是否在画面上绘制预测点与预测坐标
  int draw_prediction = 1;
  double tracker_process_noise = 1e-3;
  double tracker_measure_noise = 1e-2;
};

struct RuntimeConfig {
  std::string video_path;
  std::string config_path;
  std::string calib_path;      // optional: the team's calibration.yaml
  std::string out_video;       // empty = do not write
  std::string out_csv;         // empty = do not write
  bool headless = false;       // no imshow
  int max_frames = 0;          // 0 = all
  bool verbose = false;
};

/// Wrapper so the config reads like a single object.
struct Config {
  CameraConfig camera;
  DetectorConfig detector;
  RuntimeConfig runtime;

  /// Load from a YAML file (OpenCV FileStorage). Throws on unreadable file.
  static Config load(const std::string& path);
  /// Read just intrinsics from an existing calibration file (best effort).
  bool loadCalibration(const std::string& path);
  /// Print every field, for the run log / reproducibility.
  void dump() const;
};

}  // namespace autoaim


