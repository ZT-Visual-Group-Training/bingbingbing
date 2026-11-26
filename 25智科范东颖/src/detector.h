#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>

struct LightBar
{
    cv::Point2f center;
    cv::Point2f p1, p2;
    float length;
    float angle;
};

struct Armor
{
    cv::Point2f pts[4];
    cv::Point2f center;
    std::string number;
};

void findColor(const cv::Mat& img, cv::Mat& mask);

void setEnemyColor(const std::string& color);

void findLightBars(const cv::Mat& mask, std::vector<LightBar>& bars);

void getArmors(const cv::Mat& img, const cv::Mat& mask, std::vector<Armor>& armors);

bool loadYolo();

void detectByYolo(const cv::Mat& img, std::vector<Armor>& armors);
