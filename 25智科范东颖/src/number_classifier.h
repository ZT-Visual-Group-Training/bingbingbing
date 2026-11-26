#pragma once
#include <opencv2/core.hpp>
#include <string>

bool loadDigitModel();

std::string classifyNumber(const cv::Mat& src, const cv::Point2f pts[4], double& confidence);
