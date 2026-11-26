#pragma once
#include <opencv2/core.hpp>

void setPoseParams(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                   float width_mm, float height_mm);

bool getPose(const cv::Point2f pts[4], cv::Mat& tvec, cv::Mat& rvec);
