#include "rm/tracker.hpp"

namespace rm
{

// 清空状态
  void Tracker::reset()
{
  has_ = false;
  t_ = 0.0;
  p_ = cv::Vec3d(0, 0, 0);
  v_ = cv::Vec3d(0, 0, 0);
}

// 喂一个观测：只有有效位姿才更新
void Tracker::update(const Pose & pose, double t)
{
  if (!pose.valid) return;  // 没解出来就不动状态

  // 先用最简单的两点差分占位，之后换成卡尔曼/EKF。
  if (has_ && t > t_ + 1e-6) {  // 防重复时间戳/除零
    const cv::Vec3d v = (pose.tvec - p_) * (1.0 / (t - t_));
    v_ = 0.5 * v_ + 0.5 * v;  // 一阶低通，新速度只占一半，压住单帧抖动
  }
  p_ = pose.tvec;  // 更新位置
  t_ = t;  // 更新时间
  has_ = true;  // 有历史了
}

// 预测 t 时刻的位置：匀速模型 p = p0 + v * Δt
bool Tracker::predict(double t, cv::Vec3d & xyz) const
{
  if (!has_) return false;  // 没见过目标，没法预测
  xyz = p_ + v_ * (t - t_);
  return true;
}

}
