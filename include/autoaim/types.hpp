// Core value types shared by the whole pipeline.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <array>
#include <vector>

namespace autoaim {

/// One light bar (a glowing vertical strip on an armour plate).
struct LightBar {
  cv::Point2f center;            // bar centre, full-frame pixels
  float length = 0.f;            // extent along the long axis, px
  float width = 0.f;             // extent along the short axis, px
  cv::Point2f axis{0.f, -1.f};   // unit vector along the long axis, pointing "up"
  cv::Point2f half{1.f, 0.f};    // unit vector along the short axis
  float angle_deg = 0.f;         // long-axis direction normalised to [-90, 90)
  float mean_val = 0.f;          // mean dominant-channel value along the centre line
  // Original minAreaRect of this bar. Kept so armourCornersFromLightBars() can be
  // fed the documented RotatedRect inputs without re-fitting anything.
  cv::RotatedRect rect;

  float ratio() const { return length / std::max(width, 1e-6f); }
};

/// An armour plate candidate: two bars plus the four plate corners.
struct Armor {
  // Corner order is fixed, clockwise from the top-left, and is what the overlay
  // labels P1..P4. It also defines the PnP object-point order, so the two must
  // always be changed together:
  //   0 / P1 = top-left      1 / P2 = top-right
  //   2 / P3 = bottom-right  3 / P4 = bottom-left
  std::array<cv::Point2f, 4> corners{};
  // Mean of the four corners. Overlay + tracker only: it is NEVER passed to PnP,
  // because handing the solver the centre instead of the corners would throw away
  // the perspective information and make the pose unobservable.
  cv::Point2f center{0.f, 0.f};
  int left_bar = -1;
  int right_bar = -1;
  float score = 0.f;
  float center_ratio = 0.f;  // centre distance / mean bar length
  float length_ratio = 0.f;  // longer bar / shorter bar
  float angle_diff = 0.f;    // degrees between the two bar axes
  float aspect = 0.f;        // corner-quad width / height, px
  // 四边形在图像上的像素宽度（板高外扩之前的值）。用来做"最小可信尺寸"判断：
  // 真实装甲板在 2 m 处约 160 px，背景误检往往只有 25~40 px，差一个数量级。
  float width_px = 0.f;
  bool has_pose = false;
};

/// Result of the PnP solve.
struct Pose {
  cv::Vec3d rvec{0, 0, 0};
  cv::Vec3d tvec{0, 0, 0};
  double reproj_err_px = 1e9;
  bool ok = false;

  double x() const { return tvec[0]; }
  double y() const { return tvec[1]; }
  double z() const { return tvec[2]; }
  double range() const { return cv::norm(tvec); }

  /// Plate normal expressed as yaw/pitch in the camera frame, degrees.
  void yawPitchDeg(double* yaw_deg, double* pitch_deg) const;
};

/// Per-frame timing breakdown, in milliseconds.
struct FrameTiming {
  double decode_ms = 0.0;
  double detect_ms = 0.0;   // colour split + threshold + morphology + contours
  double match_ms = 0.0;    // pairing + corner construction
  double pnp_ms = 0.0;
  double draw_ms = 0.0;
  double total_ms = 0.0;
};

/// Everything the pipeline produced for one frame.
struct FrameResult {
  std::vector<LightBar> bars;
  std::vector<Armor> armors;
  std::vector<Pose> poses;
  FrameTiming timing;
  // Temporal-confirmation state of the reported target (see temporal_filter.hpp).
  // A detection not seen for `confirm_hits` consecutive frames stays "tentative":
  // it is still drawn, but its coordinates are not treated as a fireable target.
  bool target_confirmed = false;
  int confirm_hits = 0;
  // 时序层选定的主目标下标；-1 表示"按原来的 score 顺序取"。
  // primaryIndex() 会优先返回它，于是绘制和 CSV 都自动跟随时序判定。
  int primary_override = -1;
  // 卡尔曼预测（延迟补偿）：把当前位姿按速度外推到"击发时刻"的位置。
  // predicted_px 是它在图像上的投影，用来在画面上画预测点。
  bool has_prediction = false;
  cv::Vec3d predicted_tvec{0, 0, 0};
  cv::Point2f predicted_px{0.f, 0.f};
  double predict_latency_ms = 0.0;
  // ---- 反陀螺（Anti-Gyro）的输出 ----------------------------------------
  // 四块装甲的 3D 坐标投影回画面后的像素位置，用于调试可视化。
  bool gyro_valid = false;              // 本帧陀螺模型是否成立
  cv::Vec3d gyro_centre{0, 0, 0};       // 旋转中心（相机系）
  double gyro_omega_dps = 0.0;
  double gyro_omega_yaw_dps = 0.0;     // yaw 角速度（社区主判据）          // 角速度（度/秒），符号=转向
  int gyro_direction = 0;               // +1 逆时针 / -1 顺时针
  int gyro_samples = 0;                 // 参与拟合的样本数
  int gyro_best = -1;                   // 推荐交给云台的那块
  std::array<cv::Vec3d, 4> gyro_armors; // 四块装甲中心（相机系，米）
  std::array<cv::Point2f, 4> gyro_px;   // 四块装甲的投影像素
  std::array<bool, 4> gyro_px_ok{{false, false, false, false}};
  // 提前量：击发时刻该瞄哪块（"下一块"预测）
  double gyro_best_yaw_at_fire = 0.0;   // 那块在击发时刻的 yaw（度）
  bool gyro_best_is_current = false;    // true=就是现在这块，false=在瞄还没出现的那块
  double gyro_t_to_face_s = -1.0;       // 还需多久它最正对（秒）
  double gyro_predict_ahead_s = 0.0;    // 本次向前预测了多久
  cv::Vec3d gyro_best_pos_at_fire{0, 0, 0};
  cv::Point2f gyro_centre_px{0.f, 0.f};
  bool gyro_centre_ok = false;
  // 时序状态码（0 无 / 1 确认 / 2 续帧 / 3 次候选），写进 CSV 便于复盘
  int track_state = 0;

  int primaryIndex() const;  // best armour, or -1
  const Pose* primaryPose() const;
};

}  // namespace autoaim
