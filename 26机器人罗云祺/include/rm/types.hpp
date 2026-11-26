#pragma once

#include <array>
#include <cstdint>

#include <opencv2/core.hpp>

namespace rm
{

// 强类型枚举
enum class Color { Red, Blue, Unknown };
enum class ArmorSize { Small, Big };

inline const char * colorName(Color c)
{
  switch (c) {
    case Color::Red:
      return "red";
    case Color::Blue:
      return "blue";
    default:
      return "unknown";
  }
}

inline const char * sizeName(ArmorSize s) { return (s == ArmorSize::Big) ? "big" : "small"; }

struct Lightbar
{
  // 去重
  int id = -1;

  cv::RotatedRect rect;
  cv::Point2f top{};
  cv::Point2f bottom{};
  cv::Point2f center{};

  float angle = 0.F;        // 灯条方向
  float angle_error = 0.F;  // 与竖直方向的夹角绝对值
  float length = 0.F;       // 上下端点距离
  float width = 0.F;        // 短边
  float ratio = 0.F;        // length / width

  Color color = Color::Unknown;
};

struct Armor
{
  std::array<cv::Point2f, 4> corners{};
  // 四角点顺序 TL, TR, BR, BL
  cv::Point2f center{};
  cv::Rect2f box{};

  Color color = Color::Unknown;
  ArmorSize size = ArmorSize::Small;
  int number = -1;       // -1 表示没识别
  float confidence = 0.F;
};

// 单帧 PnP 结果
struct Pose
{
  bool valid = false;
  cv::Vec3d rvec{0, 0, 0};  // 旋转向量
  cv::Vec3d tvec{0, 0, 0};  // 平移向量
  double reproj_error = 0.0;  // 四角点平均重投影误差
};

struct Frame
{
  int index = 0;
  int64_t timestamp_ns = 0;
  double elapsed_s = 0.0;
  cv::Mat image;
};

}
