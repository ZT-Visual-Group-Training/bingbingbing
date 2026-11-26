#include "solver.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

Pose PoseSolver::solve(const Armor & armor) const
{
    Pose pose;
    pose.valid = false;

    // 四角点去畸变
    const std::vector<cv::Point2f> img = undistortCorners(armor.corners);

    // 3D 点
    const auto obj_a = objectPoints(armor.size);
    const std::vector<cv::Point3f> obj(obj_a.begin(), obj_a.end());

    // IPPE：平面目标会返回两组解
    std::vector<cv::Mat> rvecs, tvecs;

    const int n = cv::solvePnPGeneric(
        obj, img_pts, calib_.K, cv::Mat(), 
        rvecs, tvecs, false, cv::SOLVEPNP_IPPE
    );
    
    // 如果解算失败，直接返回
    if (n <= 0) {
        return pose;
    }

    // 遍历两组解
    double best_err = 1e18;
    int best = -1;

    for (int i = 0; i < n; ++i) {
        // 获取平移向量的 x/y/z 值
        const double x = tvecs[i].at<double>(0);
        const double y = tvecs[i].at<double>(1);
        const double z = tvecs[i].at<double>(2);

        if (z <= 0.0) continue;
        cv::Mat R;
        cv::Rodrigues(rvecs[i], R);

        const double nx = R.at<double>(0, 2);
        const double ny = R.at<double>(1, 2);
        const double nz = R.at<double>(2, 2);

        // 法向与 tvec 同向
        if (nx * x + ny * y + nz * z <= 0.0) continue; 

        // 把 3D 点投影回 2D
        std::vector<cv::Point2f> proj;
        cv::projectPoints(obj, rvecs[i], tvecs[i], calib_.K, cv::Mat(), proj);

        // 算一下重投影误差
        double e = 0.0;
        for (int k = 0; k < 4; ++k) {
            e += cv::norm(proj[k] - img_pts[k]); // 求两点距离
        }

        // 如果误差比之前的都小，更新 best
        if (e < best_err) {
            best_err = e;
            best = i;
        }
    }

    // 如果连一个合理的解都没有，直接返回
    if (best < 0) {
        std::cout << "[Solver] 没找到合理的正深度的解" << std::endl;
        return pose;
    }

    // LM 精化
    cv::Mat rvec = rvecs[best].clone(); // clone() 防污染
    cv::Mat tvec = tvecs[best].clone();
    cv::solvePnPRefineLM(obj, img_pts, calib_.K, cv::Mat(), rvec, tvec);

    // 重新算误差
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, rvec, tvec, calib_.K, cv::Mat(), proj);
    
    double e = 0.0;
    for (int k = 0; k < 4; ++k) {
        e += cv::norm(proj[k] - img_pts[k]);
    }

    // 填充结果
    pose.valid = true;
    pose.rvec = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    pose.tvec = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    pose.reproj_error = e / 4.0; // 4 个点的平均误差

    return pose;
}
