#pragma once
#include <opencv2/core.hpp>

struct AimResult
{
    bool ok = false;
    double yawCam = 0.0;
    double pitchCam = 0.0;
    double yawGimbal = 0.0;
    double pitchGimbal = 0.0;
    cv::Point3d predPos;
    double tFlight = 0.0;
    double leadTime = 0.0;
};

void setAimParams(double bulletSpeed, double latencySec, double gravity);

void setAimExtrinsics(const cv::Mat& R_camera2gimbal, const cv::Mat& t_camera2gimbal);

AimResult solveAim(const cv::Point3d& pos, const cv::Point3d& vel, bool hasVelocity);
