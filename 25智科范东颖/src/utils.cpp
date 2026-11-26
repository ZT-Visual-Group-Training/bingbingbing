#include "utils.h"

#include <iostream>
#include <vector>

static bool readNumberSeq(const cv::FileNode& node, std::vector<double>& out)
{
    if (node.empty() || !node.isSeq())
        return false;
    out.clear();
    for (auto it = node.begin(); it != node.end(); ++it)
        out.push_back((double)(*it));
    return !out.empty();
}

bool loadCameraParams(const std::string& path, CameraParams& p)
{
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[ERROR] cannot open camera config: " << path << std::endl;
            return false;
        }

        p.image_width  = (int)fs["image_width"];
        p.image_height = (int)fs["image_height"];

        std::vector<double> k, d, r, t;
        if (!readNumberSeq(fs["camera_matrix"], k) || k.size() != 9) {
            std::cerr << "[ERROR] camera_matrix must have 9 values: " << path << std::endl;
            return false;
        }
        if (!readNumberSeq(fs["distort_coeffs"], d)) {
            std::cerr << "[ERROR] distort_coeffs missing: " << path << std::endl;
            return false;
        }
        p.camera_matrix  = cv::Mat(3, 3, CV_64F, k.data()).clone();
        p.distort_coeffs = cv::Mat(1, (int)d.size(), CV_64F, d.data()).clone();

        if (readNumberSeq(fs["R_camera2gimbal"], r) && r.size() == 9)
            p.R_camera2gimbal = cv::Mat(3, 3, CV_64F, r.data()).clone();
        if (readNumberSeq(fs["t_camera2gimbal"], t) && t.size() == 3)
            p.t_camera2gimbal = cv::Mat(3, 1, CV_64F, t.data()).clone();

        fs.release();
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] read camera config failed " << path << ": " << e.what() << std::endl;
        return false;
    }
}

bool loadArmorParams(const std::string& path, ArmorParams& p)
{
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[ERROR] cannot open armor config: " << path << std::endl;
            return false;
        }

        cv::FileNode color = fs["enemy_color"];
        if (color.isString())
            p.enemy_color = (std::string)color;

        if (!fs["armor_width_mm"].empty())
            p.armor_width_mm = (float)fs["armor_width_mm"];
        if (!fs["armor_height_mm"].empty())
            p.armor_height_mm = (float)fs["armor_height_mm"];

        fs.release();
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] read armor config failed " << path << ": " << e.what() << std::endl;
        return false;
    }
}

bool loadAimParams(const std::string& path, AimParams& p)
{
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[WARN] cannot open aim config: " << path << std::endl;
            return false;
        }
        if (!fs["bullet_speed"].empty()) p.bullet_speed = (double)fs["bullet_speed"];
        if (!fs["latency_ms"].empty())   p.latency_ms   = (double)fs["latency_ms"];
        if (!fs["gravity"].empty())      p.gravity      = (double)fs["gravity"];
        fs.release();
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[WARN] read aim config failed " << path << ": " << e.what() << std::endl;
        return false;
    }
}
