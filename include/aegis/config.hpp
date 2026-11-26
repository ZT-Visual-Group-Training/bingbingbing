// =============================================================================
// config.hpp —— 参数装载
// 设计原则：所有“会调的数字”都放进 YAML，代码里不写死魔法数；
//          这样答辩现场换一段视频/换一台相机，只改配置不改代码。
// =============================================================================
#pragma once

#include "aegis/types.hpp"

#include <string>

namespace aegis {

struct CameraConfig {
  int width = 1440;
  int height = 1080;
  cv::Matx33d cameraMatrix = cv::Matx33d::eye();
  cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
  cv::Matx33d R_camera2gimbal = cv::Matx33d::eye();
  cv::Vec3d t_camera2gimbal{0.0, 0.0, 0.0};
};

struct ArmorConfig {
  double smallWidthM = 0.135;  // RM 小装甲板外沿宽 135 mm
  double largeWidthM = 0.225;  // RM 大装甲板外沿宽 225 mm
  double heightM = 0.055;      // 灯条长 55 mm，也就是装甲板高
};

struct DetectorConfig {
  Color enemyColor = Color::Red;

  // 自适应阈值的基础值；实际阈值 = 基础值 × 环境系数（见 detector.cpp）
  double sMinBase = 150.0;
  double vMinBase = 110.0;

  bool fixedThreshold = false;  // true = 关闭自适应（用于基线对比实验）
  bool coreCompensation = true; // true = 补过曝灯条的白色核心（本方案的关键改动）

  // 灯条几何门限
  double minBarLength = 10.0;   // 太短的灯条在远距离没有 PnP 价值
  double maxBarLength = 520.0;  // 太长的多半是灯管/反光
  double minBarRatio = 1.45;    // 长边/短边
  double maxBarRatio = 26.0;
  double minFill = 0.52;        // 轮廓填充率
  double maxBarAngle = 35.0;    // 灯条相对竖直方向的最大倾角

  // 装甲板配对门限
  double maxHeightRatio = 1.55;   // 左右灯条长度比
  double maxDyRatio = 0.45;       // 两灯条中心 y 差 / 平均长度
  double maxAngleDiff = 12.0;     // 两灯条夹角
  double minArmorRatio = 1.55;    // 装甲板宽/高
  double maxArmorRatio = 5.20;
  double maxArmorWidthRatio = 1.30;  // 投影宽度 / 理论宽度（透视畸变上限）

  // 放宽档几何门限：只在“严格档一块板都没配出来”时启用，用来救大偏航目标
  // （偏航 60° 时 135mm 板投影宽度只剩一半，宽高比会掉到 1.2 左右）。
  double minArmorRatioRelaxed = 0.70;
  double maxBarAngleRelaxed = 50.0;
  double maxAngleDiffRelaxed = 20.0;
  double maxDyRatioRelaxed = 0.90;

  double minArmorHeight = 14.0;  // 太小的“装甲板”没有 PnP 价值，直接丢弃
  int maxArmors = 4;  // 一帧最多输出几块装甲板
};

struct TrackerConfig {
  int minHitsToConfirm = 3;  // 连续命中 3 帧才“确认”，抑制单帧误检
  int maxMisses = 12;        // 连续丢失 12 帧后删除轨迹（165fps 下约 72 ms）
  float gateDistance = 90.0F;  // 数据关联的最大中心距离（像素）
  float processNoise = 0.6F;   // 卡尔曼过程噪声
  float measurementNoise = 2.2F;  // 卡尔曼观测噪声
  int maxTrail = 60;         // 轨迹尾巴长度
};

struct AppConfig {
  CameraConfig camera;
  ArmorConfig armor;
  DetectorConfig detector;
  TrackerConfig tracker;

  std::string videoPath;
  std::string imuPath;         // frame_imu.csv，可选
  std::string outputPath;      // 输出视频
  std::string csvPath;         // 输出逐帧结果
  std::string baselinePath;    // 基线对比视频（可选）

  int outputFps = 60;          // 演示视频帧率
  bool showWindow = false;     // 是否实时预览（服务器上跑就关掉）
  bool enableDigits = true;    // 是否启用数字识别
  bool enableRoi = true;       // 是否启用跟踪 ROI 加速
  bool drawHud = true;         // 是否绘制 HUD
  bool baselineMode = false;   // 基线模式：固定阈值 + 不开跟踪
  int maxFrames = -1;          // -1 表示整段视频
};

// 从 YAML 读取配置；文件不存在或字段缺失时使用上面的默认值（并打印提示）。
AppConfig loadConfig(const std::string& path);

}  // namespace aegis
