// ============================================================
// ImuReader.h —— IMU 数据读取 + 坐标转换模块（独立模块，可单独理解）
//
// 这个模块解决什么问题？
//   ArmorDetector 用 PnP 算出的是"装甲板在【相机坐标系】下的位置" tvec。
//   但相机是装在云台上的，而且往往是歪着装/侧着装——这段考核视频实测
//   roll≈102°、pitch≈86°、yaw≈74°，几乎是侧装。所以这个相机系坐标
//   对机器人"转云台瞄准"没有直接意义，必须换算到云台/车体坐标系。
//
//   官方数据包给了两样关键东西：
//     ① 每一帧视频对应的 IMU 姿态（四元数 qw/qx/qy/qz）—— "相机当时怎么摆"
//     ② 手眼标定外参（R_camera2gimbal / t_camera2gimbal / R_gimbal2imubody）
//        —— "相机与云台/IMU 之间差了多远、转了多少"
//
//   本模块提供两个能力：
//     A. 四元数 → 人能看懂的姿态角（roll/pitch/yaw，单位：度）
//     B. 相机系坐标 → IMU body 系坐标（这才是云台控制真正要用的量）
//
// 为什么说它是"独立模块"？
//   它不依赖 ArmorDetector，也不改任何现有检测代码。main.cpp 需要时才用，
//   不用时整套检测行为完全不变。这就是"低耦合"的好处：加功能不碰老代码。
// ============================================================
#ifndef IMU_READER_H
#define IMU_READER_H

#include <opencv2/opencv.hpp>   // cv::Mat / cv::Vec3d / cv::FileStorage
#include <cstddef>              // std::size_t
#include <string>
#include <vector>

// ------------------------------------------------------------
// ① 一帧 IMU 数据：把 CSV 的一行打包成一个结构
//    CSV 表头（imu_frame_imu.csv）：
//    frame_index,monotonic_ns,elapsed_s,qw,qx,qy,qz,left_imu_ns,right_imu_ns,imu_gap_ms,status
// ------------------------------------------------------------
struct ImuPose {
    int    frameIndex = -1;    // 对应视频帧号（0 基，与 CSV 第一行对齐）
    double elapsedS   = 0.0;   // 相对采集开始的秒数
    double qw = 1, qx = 0, qy = 0, qz = 0;   // 四元数（使用前会归一化）
    double roll = 0, pitch = 0, yaw = 0;      // 由四元数解出的欧拉角（度）
    std::string status;         // 数据来源标记：interpolated / measured ...
    bool valid = false;         // 这一行是否成功解析
};

// ------------------------------------------------------------
// ② 手眼外参：描述"相机 / 云台 / IMU body"三个坐标系之间的关系
//    数据源：calibration_handeye.yaml
//
//    坐标转换链（本模块的核心公式）：
//      1) 相机系点 p_cam  →  云台系点 p_gimbal
//         p_gimbal = R_camera2gimbal * p_cam + t_camera2gimbal
//      2) 云台系点 p_gimbal  →  IMU body 系点 p_imu
//         p_imu = R_gimbal2imubody * p_gimbal
//    合起来：p_imu = R_gimbal2imubody * (R_camera2gimbal * p_cam + t_camera2gimbal)
// ------------------------------------------------------------
struct HandEyeExtrinsics {
    cv::Mat  R_camera2gimbal;   // 3x3，相机系 → 云台系 的旋转
    cv::Mat  R_gimbal2imubody;  // 3x3，云台系 → IMU body 系 的旋转
    cv::Vec3d t_camera2gimbal;  // 平移向量（米），相机原点在云台系下的位置
    bool valid = false;         // 三个外参是否都成功读入

    // 从 handeye.yaml 读取；成功返回 true
    bool load(const std::string& yamlPath);

    // 核心：把 PnP 给出的相机系坐标 pCam 转成 IMU body 系坐标
    // 没加载外参时原样返回 pCam，保证"接不接都不影响检测结果"
    cv::Vec3d cameraToImuBody(const cv::Vec3d& pCam) const;
};

// ------------------------------------------------------------
// ③ 逐帧 IMU 读取器：一次读入整个 CSV，之后按帧号 O(1) 查询
// ------------------------------------------------------------
class ImuReader {
public:
    // 读取 CSV；表头自动跳过，坏行自动忽略；成功返回 true
    bool load(const std::string& csvPath);

    // 按视频帧号取姿态；帧号越界返回 nullptr
    const ImuPose* poseAt(int frameIndex) const;

    // 诊断用接口
    std::size_t size()     const { return poses_.size(); }
    bool        empty()    const { return poses_.empty(); }
    int         minFrame() const;   // 数据覆盖的最小帧号
    int         maxFrame() const;   // 数据覆盖的最大帧号

private:
    std::vector<ImuPose> poses_;
};

// ------------------------------------------------------------
// ④ 两个数学小工具（公开出来，方便你单独写小程序验证）
// ------------------------------------------------------------

// 四元数 → 3x3 旋转矩阵（CV_64F）。内部先归一化，防止数值漂移。
cv::Mat quatToRotation(double w, double x, double y, double z);

// 四元数 → ZYX 欧拉角 roll/pitch/yaw（单位：度）。
// ZYX 是机器人最常用的约定：先绕 Z(yaw)，再绕 Y(pitch)，最后绕 X(roll)。
cv::Vec3d quatToEulerDeg(double w, double x, double y, double z);

#endif // IMU_READER_H
