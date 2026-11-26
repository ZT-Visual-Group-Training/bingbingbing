#pragma once
#include <opencv2/core.hpp>

void trackerInit(const cv::Point3d& pos);

cv::Point3d trackerPredict();

void trackerUpdate(const cv::Point3d& meas);

cv::Point3d trackerVelocity();
