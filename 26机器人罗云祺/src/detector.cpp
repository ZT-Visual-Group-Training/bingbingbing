#include "detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

// 构造函数
Detector::Detector(const Config & cfg) : cfg_(cfg)
{
  threshold_      = cfg.getDouble("detector.gray_threshold", 150.0);
  min_area_       = cfg.getDouble("detector.min_lightbar_area", 100.0);
  min_ratio_      = cfg.getDouble("detector.min_lightbar_ratio", 1.5);
  max_ratio_      = cfg.getDouble("detector.max_lightbar_ratio", 20.0);
  max_angle_      = cfg.getDouble("detector.max_angle_error_deg", 45.0);
  max_height_diff_ = cfg.getDouble("detector.max_height_diff_px", 40.0);
  max_y_diff_     = cfg.getDouble("detector.max_y_diff_px", 40.0);
  min_armor_width_ = cfg.getDouble("detector.min_armor_width_px", 60.0);
  max_armor_width_ = cfg.getDouble("detector.max_armor_width_px", 800.0);
  max_angle_diff_  = cfg.getDouble("detector.max_rectangular_error_deg", 25.0);

  std::cout << "[Detector] 初始化成功" << std::endl;
}

// 判断颜色
Color Detector::colorOf(const cv::Mat & bgr, const std::vector<cv::Point> & contour) const {

    cv::Rect rect = cv::boundingRect(contour);
    cv::Mat roi = bgr(rect);
    
    // 计算BGR通道的均值
    cv::Scalar mean_color = cv::mean(roi);
    float b = mean_color[0];
    float g = mean_color[1];
    float r = mean_color[2];

    if (r > b && r > g) {
        return Color::RED;
    } else if (b > r && b > g) {
        return Color::BLUE;
    }
    return Color::UNKNOWN; 
}

// 单灯条过滤条件
bool Detector::checkLightbar(const Lightbar & bar) const {
    // 面积过滤
    if (bar.rect.size.area() <  min_area) return false;

    // 长宽比过滤
    if (bar.ratio <  min_ratio || bar.ratio >  max_ratio) return false;

    // 角度过滤
    if (std::abs(bar.angle) >  max_angle) return false;

    return true;
}

// 双灯条配对成装甲板
bool Detector::checkArmor(const Lightbar & left, const Lightbar & right) const {
    if (left.center.x >= right.center.x) return false;

    float height_diff = std::abs(left.length - right.length);
    if (height_diff >  max_height_diff) return false;

    float y_diff = std::abs(left.center.y - right.center.y);
    if (y_diff >  max_y_diff) return false;

    // 两灯条间距要在合理范围内
    float distance = cv::norm(left.center - right.center);
    if (distance <  min_armor_width || distance >  max_armor_width) return false;

    // 角度差不能太大
    float angle_diff = std::abs(left.angle - right.angle);
    if (angle_diff >  max_angle_diff) return false;

    // 颜色必须一致
    if (left.color != right.color) return false;

    return true;
}

// 核心检测函数
std::vector<Armor> Detector::detect(const cv::Mat & bgr) {
    std::vector<Armor> armors;
    std::vector<Lightbar> lightbars;

    // ---------------- 二值化 + 轮廓 ----------------
    cv::Mat gray, binary;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    // 使用固定的阈值或者配置阈值
    cv::threshold(gray, binary,  threshold, 255, cv::THRESH_BINARY);
    
    debug_.binary = binary;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // ---------------- 轮廓变灯条 ----------------
    for (size_t i = 0; i < contours.size(); i++) {
        // 拟合最小外接矩形
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        
        // 获取四个顶点
        cv::Point2f pts[4];
        rect.points(pts);

        std::vector<cv::Point2f> sorted_pts(pts, pts + 4);
        std::sort(sorted_pts.begin(), sorted_pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });

        cv::Point2f left_top, left_bottom, right_top, right_bottom;
        if (sorted_pts[0].y < sorted_pts[1].y) {
            left_top = sorted_pts[0]; left_bottom = sorted_pts[1];
        } else {
            left_top = sorted_pts[1]; left_bottom = sorted_pts[0];
        }
        if (sorted_pts[2].y < sorted_pts[3].y) {
            right_top = sorted_pts[2]; right_bottom = sorted_pts[3];
        } else {
            right_top = sorted_pts[3]; right_bottom = sorted_pts[2];
        }

        // 计算上下短边中点
        cv::Point2f top_center = (left_top + right_top) / 2.0;
        cv::Point2f bottom_center = (left_bottom + right_bottom) / 2.0;
        cv::Point2f center = (top_center + bottom_center) / 2.0;

        float bar_len = cv::norm(top_center - bottom_center);  // 长边
        float bar_w   = cv::norm(left_top - right_top);        // 短边
        float ratio   = bar_len / std::max(1.0f, bar_w);
        
        // 计算倾斜角
        const double dy = bottom_center.y - top_center.y;
        const double angle_error = std::acos(std::clamp(dy / bar_len, -1.0, 1.0)) * 180.0 / CV_PI;
        if (angle_error >  max_angle) return false;

        Lightbar bar;
        bar.rect = rect;
        bar.top = top_center;
        bar.bottom = bottom_center;
        bar.center = center;
        bar.length = length;
        bar.width = width;
        bar.ratio = ratio;
        bar.angle = angle;
        bar.color = colorOf(bgr, contours[i]);

        // 过滤灯条
        if (checkLightbar(bar)) {
            lightbars.push_back(bar);
        }
    }

    debug_.lightbars = lightbars;

    // ---------------- 配对 ----------------
    std::sort(lightbars.begin(), lightbars.end(), [](const Lightbar& a, const Lightbar& b) {
        return a.center.x < b.center.x;
    });

    std::vector<Armor> raw_armors;
    // 双重循环进行两两配对
    for (size_t i = 0; i < lightbars.size(); i++) {
        for (size_t j = i + 1; j < lightbars.size(); j++) {
            if (lightbars[j].center.x - lightbars[i].center.x >  max_armor_width) {
                break;
            }

            if (checkArmor(lightbars[i], lightbars[j])) {
                Armor armor;
                armor.left = lightbars[i];
                armor.right = lightbars[j];
                armor.center = (armor.left.center + armor.right.center) / 2.0;
                armor.color = armor.left.color;
                // armor.type = (armor.right.center.x - armor.left.center.x) > 100 ? ArmorType::LARGE : ArmorType::SMALL; 
                
                raw_armors.push_back(armor);
            }
        }
    }

    // ---------------- 去重 ----------------
    // 如果两块装甲板共用了同一根灯条，保留面积大的那块
    for (size_t i = 0; i < raw_armors.size(); i++) {
        bool is_duplicate = false;
        for (size_t j = 0; j < armors.size(); j++) {
          
            bool same_left = cv::norm(raw_armors[i].left.center - armors[j].left.center) < 5.0;
            bool same_right = cv::norm(raw_armors[i].right.center - armors[j].right.center) < 5.0;

            if (same_left || same_right) {
                is_duplicate = true;
                // 算面积
                float area_i = raw_armors[i].left.rect.size.area() + raw_armors[i].right.rect.size.area();
                float area_j = armors[j].left.rect.size.area() + armors[j].right.rect.size.area();
                
                // 如果新的面积大，替换掉旧的
                if (area_i > area_j) {
                    armors[j] = raw_armors[i]; 
                }
                break;
            }
        }
        if (!is_duplicate) {
            armors.push_back(raw_armors[i]);
        }
    }

    return armors;
}
