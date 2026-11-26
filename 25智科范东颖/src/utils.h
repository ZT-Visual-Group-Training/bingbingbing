#pragma once
#include <opencv2/opencv.hpp>
#include <string>

struct CameraParams {
    int image_width = 1440;
    int image_height = 1080;
    cv::Mat camera_matrix;
    cv::Mat distort_coeffs;
    cv::Mat R_camera2gimbal;
    cv::Mat t_camera2gimbal;
};

struct ArmorParams {
    std::string enemy_color = "red";
    float armor_width_mm  = 135.0f;
    float armor_height_mm = 55.0f;
};

struct AimParams {
    double bullet_speed = 20.0;
    double latency_ms   = 30.0;
    double gravity      = 9.8;
};

bool loadCameraParams(const std::string& path, CameraParams& p);
bool loadArmorParams(const std::string& path, ArmorParams& p);
bool loadAimParams(const std::string& path, AimParams& p);
