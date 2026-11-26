// =============================================================================
// types.hpp —— AegisVision 的公共数据结构
// 说明：整个工程只依赖 OpenCV，所有模块之间通过这里定义的结构体通信，
//      避免“检测器直接画图、跟踪器直接读像素”的耦合写法。
// =============================================================================
#pragma once

#include "aegis/compat.hpp"

#include <array>
#include <deque>
#include <string>
#include <vector>

namespace aegis {

// 敌方颜色。RM 规则里红蓝是对抗阵营，识别时必须只认其中一种，否则会把队友当敌人。
enum class Color : int { Red = 0, Blue = 1 };

// 装甲板尺寸类型：小装甲板 135×55 mm，大装甲板 225×55 mm（RM 标准）。
enum class ArmorSize : int { Small = 0, Large = 1 };

// 一条“灯条”的几何描述。
// 灯条 = 装甲板两侧的发光条，是传统视觉里最稳定的特征：
// 亮度高、饱和度高、竖直细长，比整块装甲板更容易在暗光/运动模糊下存活。
struct LightBar {
  cv::RotatedRect rect;      // 最小外接旋转矩形（比正矩形更贴合倾斜的灯条）
  cv::Point2f top;           // 灯条长轴的上端点（用于拼装甲板角点）
  cv::Point2f bottom;        // 灯条长轴的下端点
  float length = 0.0F;       // 长边像素长度
  float width = 0.0F;        // 短边像素宽度
  float angle = 0.0F;        // 长边相对竖直方向的角度（度，右倾为正）
  float fill = 0.0F;         // 轮廓面积 / 外接矩形面积，用来排除“十字反光”
  Color color = Color::Red;  // 来自哪个颜色通道
};

// 一块装甲板的检测结果。
struct Armor {
  std::array<cv::Point2f, 4> corners{};  // 顺序固定：左上、右上、右下、左下（图像坐标）
  cv::Point2f center{};                  // 四角对角线交点
  float width = 0.0F;                    // 像素宽度（两灯条外沿距离）
  float height = 0.0F;                   // 像素高度（灯条长度）
  Color color = Color::Red;
  ArmorSize size = ArmorSize::Small;     // 由宽高比推断大小装甲板
  int number = -1;                       // 识别出的数字 0..9，-1 表示未识别
  float numberConfidence = 0.0F;         // 数字置信度 0..1
  float confidence = 0.0F;               // 整块装甲板的几何置信度 0..1
  float yaw = 0.0F;                      // 由宽高比粗估的偏航角（度），仅用于显示
  int thresholdLevel = 0;                // 0=严格阈值 1=放宽阈值（用于展示自适应过程）
  float ambient = 0.0F;                  // 当帧环境亮度（V 均值），仅用于调试面板
  LightBar leftBar;                      // 左灯条
  LightBar rightBar;                     // 右灯条
};

// 三维位姿（相机坐标系，单位米；z 向前，x 向右，y 向下）。
struct Pose {
  bool valid = false;
  cv::Vec3d tvec{};          // 平移：装甲板中心在相机系下的坐标
  cv::Vec3d rvec{};          // 旋转向量：装甲板坐标系 -> 相机坐标系
  double distance = 0.0;     // 直线距离 ||tvec||
  double yaw = 0.0;          // 水平角 atan2(x, z)，度
  double pitch = 0.0;        // 俯仰角 atan2(-y, sqrt(x²+z²))，度
  double reprojError = 0.0;  // 四个角点的平均重投影误差（像素）
  cv::Vec3d gimbal{};        // 转到云台坐标系后的位置（用于和 IMU 对齐）
};

// 跟踪器内部的一条轨迹。
struct Track {
  int id = -1;                       // 轨迹 ID，画面上显示 T-07 这种编号
  cv::Rect2f box;                    // 当前（滤波后）图像框
  cv::Point2f velocity;              // 图像速度 px/frame
  int hits = 0;                      // 累计命中次数
  int misses = 0;                    // 连续丢失次数
  int age = 0;                       // 存活帧数
  bool confirmed = false;            // 是否已“确认”（连续命中 >= 阈值）
  float score = 0.0F;                // 最近一次检测置信度
  Color color = Color::Red;
  int number = -1;                   // 数字投票结果（多帧投票，抗单帧误识别）
  float numberConfidence = 0.0F;     // 最近一次数字识别置信度
  int displayId = -1;                // 对外显示的编号（确认后从 1 开始，避免内部 ID 乱跳）
  std::array<int, 10> digitVotes{};  // 每个数字的票数
  Pose pose;                         // 最近一次 PnP 结果
  std::deque<cv::Point2f> trail;     // 轨迹尾巴（用于画渐隐轨迹）
  std::deque<float> zHistory;        // 深度历史（用于稳定性指标）
  double lastUpdateMs = 0.0;         // 最近一次被检测更新的时间戳
};

// 单帧耗时拆解，性能面板直接用这些数字。
struct FrameTiming {
  double detect = 0.0;
  double track = 0.0;
  double pose = 0.0;
  double draw = 0.0;
  double total = 0.0;
};

// 单帧的完整输出（检测 + 跟踪 + 位姿 + 计时）。
struct FrameResult {
  std::vector<Armor> armors;      // 本帧所有候选装甲板
  std::vector<Track> tracks;      // 本帧所有轨迹（含未确认）
  int primaryTrack = -1;          // 主目标轨迹 ID（离画面中心最近/最稳定）
  FrameTiming timing;             // 各阶段耗时
  double ambient = 0.0;           // 环境亮度
  int thresholdLevel = 0;         // 本帧使用的阈值档位
  bool usedRoi = false;           // 本帧是否走了 ROI 加速
  bool relaxedGeometry = false;   // 本帧是否启用了放宽档几何（大偏航目标）
  float sThreshold = 0.0F;        // 本帧实际 S 阈值（HUD 展示用）
  float vThreshold = 0.0F;        // 本帧实际 V 阈值
  int redBars = 0;                // 红色候选灯条数
  int blueBars = 0;               // 蓝色候选灯条数
  cv::Mat armorChip;              // 主目标透视校正后的小图（HUD 上显示数字用）
};

}  // namespace aegis
