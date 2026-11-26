// ============================================================
// imu_check.cpp —— IMU 模块的"体检程序"
//   不检测装甲板，只验证三件事：
//     ① CSV 能不能读、读了多少帧
//     ② 四元数转姿态角对不对（和之前 Python 算的 roll≈102/pitch≈86/yaw≈74 对比）
//     ③ 手眼外参矩阵对不对（正交性、行列式≈1）+ 一个坐标转换样例
//   跑法：编译后 imu_check.exe [imu_frame_imu.csv] [calibration_handeye.yaml]
// ============================================================
#include "ImuReader.h"
#include <iostream>

int main(int argc, char** argv) {
    std::string csv = argc > 1 ? argv[1] : "imu_data/imu_frame_imu.csv";
    std::string yaml = argc > 2 ? argv[2] : "imu_data/calibration_handeye.yaml";

    ImuReader imu;
    if (!imu.load(csv)) {
        std::cerr << "读取 IMU CSV 失败: " << csv << '\n';
        return 1;
    }
    std::cout << "IMU 帧数: " << imu.size()
              << "  帧范围: " << imu.minFrame() << " ~ " << imu.maxFrame() << '\n';

    // 打印几个代表帧的姿态角
    for (int f : {0, 500, 1000, 1769}) {
        const ImuPose* p = imu.poseAt(f);
        if (!p) { std::cout << "帧 " << f << " 无数据\n"; continue; }
        std::cout << "帧 " << f << ": roll=" << p->roll << " pitch=" << p->pitch
                  << " yaw=" << p->yaw << " status=" << p->status << '\n';
    }

    HandEyeExtrinsics ext;
    if (!ext.load(yaml)) {
        std::cerr << "读取手眼外参失败: " << yaml << '\n';
        return 2;
    }
    std::cout << "\nR_camera2gimbal = \n" << ext.R_camera2gimbal << '\n';
    std::cout << "R_gimbal2imubody = \n" << ext.R_gimbal2imubody << '\n';
    std::cout << "t_camera2gimbal = " << ext.t_camera2gimbal << '\n';

    // 正交性检查：R * R^T 应该≈单位阵
    cv::Mat I1 = ext.R_camera2gimbal * ext.R_camera2gimbal.t();
    cv::Mat I2 = ext.R_gimbal2imubody * ext.R_gimbal2imubody.t();
    cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);
    std::cout << "R_c2g 正交误差 = " << cv::norm(I1, eye) << '\n';
    std::cout << "R_g2i 正交误差 = " << cv::norm(I2, eye) << '\n';

    // 坐标转换样例：假设 PnP 给出相机系点 (0, 0, 2) 米（正前方2米）
    cv::Vec3d pCam(0, 0, 2);
    cv::Vec3d pImu = ext.cameraToImuBody(pCam);
    std::cout << "\n坐标转换样例: 相机系 (0,0,2)m -> IMU body 系 ("
              << pImu[0] << ", " << pImu[1] << ", " << pImu[2] << ")m\n";
    std::cout << "长度保持(旋转不改距离): "
              << cv::norm(pImu) << "m (应≈ " << cv::norm(pCam) << "m + 平移影响)\n";
    return 0;
}
