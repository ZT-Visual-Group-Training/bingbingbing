#include "auto_aim.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace armor {
namespace {

// 兼容采集工具导出的 YAML 头部、BOM 和换行格式。
std::string normalizeYaml(std::string text) {
    if(text.compare(0, 3, "\xEF\xBB\xBF") == 0) { text.erase(0, 3); }
    std::istringstream lines(text);
    std::ostringstream result;
    result << "%YAML:1.0\n---\n";
    std::string line;
    while (std::getline(lines, line)) {
        if(!line.empty() && line.back() == '\r') { line.pop_back(); }
        if(line.rfind("%YAML", 0) == 0 || line == "---") { continue; }
        if(line.rfind("- ", 0) == 0) { result << "  "; }
        const std::size_t colon = line.find(':');
        if(colon != std::string::npos) {
            const std::size_t value = line.find_first_not_of(" \t", colon + 1);
            if(value != std::string::npos && std::isdigit(static_cast<unsigned char>(line[value])) &&
                line.find('_', value) != std::string::npos && line[value] != '"' && line[value] != '\'') {
                const std::size_t end = line.find_first_of(" \t#", value);
                const std::string scalar = line.substr(value, end == std::string::npos ? end : end - value);
                if(scalar.find_first_not_of("0123456789_") == std::string::npos && scalar.find('_') != std::string::npos) {
                    line.insert(value, "\"");
                    line.insert(value + scalar.size() + 1, "\"");
                }
            }
        }
        result << line << '\n';
    }
    return result.str();
}

// 将标定数组统一读成单通道 double 矩阵，并检查内容是否合法。
cv::Mat readNumericArray(const cv::FileNode& node, const std::string& name) {
    if(node.empty()) { throw std::runtime_error("Missing calibration key: " + name); }
    cv::Mat result;
    if(node.isSeq()) {
        std::vector<double> values;
        for (const cv::FileNode& value : node) {
            if(!value.isInt() && !value.isReal()) {
                throw std::runtime_error(name + " must be a flat numeric sequence");
            }
            values.push_back(static_cast<double>(value));
        }
        result = cv::Mat(values).clone();
    } else if(node.isMap()) {
        node >> result; 
    } else {
        throw std::runtime_error("Invalid calibration array: " + name);
    }
    if(result.empty() || result.channels() != 1) {
        throw std::runtime_error("Empty or multichannel calibration array: " + name);
    }
    result.convertTo(result, CV_64F);
    if(!cv::checkRange(result)) { throw std::runtime_error(name + " contains nonfinite values"); }
    return result;
}

// 位姿向量在绘制和写出前需要排除非有限值。
bool finite(const cv::Vec3d& value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

// 将文字限制在图像范围内，并加描边以适应不同背景。
void label(cv::Mat& frame, const std::string& text, cv::Point origin,
           const cv::Scalar& color, double scale = 0.65) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    origin.x = std::clamp(origin.x, 0, std::max(0, frame.cols - size.width - 2));
    origin.y = std::clamp(origin.y, size.height + 2, frame.rows - baseline - 2);
    cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

} 


// 读取相机内参、畸变系数和标定分辨率，并检查矩阵格式。
CameraParameters loadCalibration(const std::string& path) {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    if(!file) { throw std::runtime_error("Cannot read calibration: " + path); }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    cv::FileStorage storage(normalizeYaml(text), cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if(!storage.isOpened()) { throw std::runtime_error("Cannot parse calibration YAML"); }
    CameraParameters output;
    cv::Mat matrix = readNumericArray(storage["camera_matrix"], "camera_matrix");
    if(matrix.total() != 9) { throw std::runtime_error("camera_matrix requires 9 numbers"); }
    output.camera_matrix = matrix.reshape(1, 3).clone();
    const cv::Mat& k = output.camera_matrix;
    if(k.at<double>(0, 0) <= 0 || k.at<double>(1, 1) <= 0 ||
        std::abs(k.at<double>(0, 1)) > 1e-9 || std::abs(k.at<double>(1, 0)) > 1e-9 ||
        std::abs(k.at<double>(2, 0)) > 1e-9 || std::abs(k.at<double>(2, 1)) > 1e-9 ||
        std::abs(k.at<double>(2, 2) - 1.0) > 1e-9) {
        throw std::runtime_error("Invalid pinhole camera_matrix");
    }
    cv::Mat distortion = readNumericArray(storage["distort_coeffs"], "distort_coeffs");
    const std::size_t n = distortion.total();
    if(n != 4 && n != 5 && n != 8 && n != 12 && n != 14) {
        throw std::runtime_error("distort_coeffs requires 4, 5, 8, 12 or 14 numbers");
    }
    output.distort_coeffs = distortion.reshape(1, static_cast<int>(n)).clone();
    if(!storage["image_width"].empty()) { storage["image_width"] >> output.image_size.width; }
    if(!storage["image_height"].empty()) { storage["image_height"] >> output.image_size.height; }
    if(output.image_size.width <= 0 || output.image_size.height <= 0) {
        throw std::runtime_error("Invalid calibration image dimensions");
    }
    return output;
}

// 保存检测参数，供每帧的检测阶段共用。
ArmorDetector::ArmorDetector(DetectorConfig config) : config_(std::move(config)) {}


// 1. 按颜色提取灯条区域；可选补回过曝造成的亮芯断裂。
std::vector<ColorMask> ArmorDetector::segment(const cv::Mat& bgr) const {
    if(bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("Detector requires a nonempty CV_8UC3 BGR frame");
    }
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    std::vector<ColorMask> output;
    if(config_.color != ColorMode::Blue) {
        cv::Mat low, high, red;
        
        
        cv::inRange(hsv, cv::Scalar(0, 101, 101), cv::Scalar(10, 255, 255), low);
        cv::inRange(hsv, cv::Scalar(170, 101, 101), cv::Scalar(180, 255, 255), high);
        cv::bitwise_or(low, high, red);
        output.push_back({Color::Red, red});
    }
    if(config_.color != ColorMode::Red) {
        cv::Mat blue;
        cv::inRange(hsv, cv::Scalar(100, 101, 101), cv::Scalar(130, 255, 255), blue);
        output.push_back({Color::Blue, blue});
    }
    
    if(config_.recover_bright_cores) {
        // 高亮连通域只有与原颜色区域足够重合时才补入掩码，减少白色背景干扰。
        cv::Mat bright, labels, stats, centroids;
        cv::inRange(hsv, cv::Scalar(0, 0, 160), cv::Scalar(180, 255, 255), bright);
        const int count = cv::connectedComponentsWithStats(bright, labels, stats, centroids, 8, CV_32S);
        for (ColorMask& colored : output) {
            std::vector<int> support(static_cast<std::size_t>(count), 0);
            for (int y = 0; y < labels.rows; ++y) {
                const int* ids = labels.ptr<int>(y);
                const unsigned char* seed = colored.mask.ptr<unsigned char>(y);
                for (int x = 0; x < labels.cols; ++x) {
                    if(seed[x] != 0 && ids[x] != 0) { ++support[ids[x]]; }
                }
            }
            std::vector<bool> accepted(static_cast<std::size_t>(count), false);
            for (int id = 1; id < count; ++id) {
                const int width = stats.at<int>(id, cv::CC_STAT_WIDTH);
                const int height = stats.at<int>(id, cv::CC_STAT_HEIGHT);
                const int area = stats.at<int>(id, cv::CC_STAT_AREA);
                accepted[id] = height >= 8 && height > 2 * width && area <= 5000 &&
                               support[id] >= 6 && support[id] >= 0.05 * area;
            }
            for (int y = 0; y < labels.rows; ++y) {
                const int* ids = labels.ptr<int>(y);
                unsigned char* mask = colored.mask.ptr<unsigned char>(y);
                for (int x = 0; x < labels.cols; ++x) {
                    if(accepted[ids[x]]) { mask[x] = 255; }
                }
            }
        }
    }
    return output;
}


// 2. 去除小噪点，并连接灯条区域中的小间隙。
std::vector<ColorMask> ArmorDetector::morph(const std::vector<ColorMask>& masks) const {
    std::vector<ColorMask> output;
    const cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, config_.open_kernel);
    
    const cv::Size close_size = config_.recover_bright_cores ? cv::Size(3, 3) : config_.close_kernel;
    const cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, close_size);
    for (const ColorMask& colored : masks) {
        cv::Mat cleaned;
        // 开运算去除散点，闭运算填补局部断口。
        cv::morphologyEx(colored.mask, cleaned, cv::MORPH_OPEN, open_kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, close_kernel);
        output.push_back({colored.color, cleaned});
    }
    return output;
}


// 3. 提取外轮廓，并按面积过滤明显噪声。
std::vector<ContourCandidate> ArmorDetector::contours(const std::vector<ColorMask>& masks) const {
    std::vector<ContourCandidate> output;
    for (const ColorMask& colored : masks) {
        std::vector<std::vector<cv::Point>> found;
        cv::findContours(colored.mask, found, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const std::vector<cv::Point>& contour : found) {
            const double area = cv::contourArea(contour);
            if(area > config_.min_contour_area) { output.push_back({colored.color, contour, area}); }
        }
    }
    return output;
}


// 4. 用旋转矩形拟合轮廓，按长宽比、倾角和填充度筛选灯条。
std::vector<LightBar> ArmorDetector::lights(const std::vector<ContourCandidate>& candidates) const {
    std::vector<LightBar> output;
    for (const ContourCandidate& contour : candidates) {
        const cv::RotatedRect rect = cv::minAreaRect(contour.points);
        cv::Point2f vertices[4];
        rect.points(vertices);
        
        cv::Point2f edge = vertices[1] - vertices[0];
        cv::Point2f other = vertices[2] - vertices[1];
        if(cv::norm(edge) < cv::norm(other)) { std::swap(edge, other); }
        const double length = cv::norm(edge), width = cv::norm(other);
        if(width <= 1e-6 || length / width <= config_.min_light_aspect) { continue; }
        if(edge.y < 0) { edge = -edge; }
        const double tilt = std::atan2(edge.x, edge.y) * 180.0 / CV_PI;
        if(std::abs(tilt) >= config_.max_tilt_deg ||
            contour.area / (length * width) <= config_.min_fill_ratio) { continue; }
        
        output.push_back({contour.color, rect.center, rect.center - 0.5F * edge,
                          rect.center + 0.5F * edge, length, width, tilt});
    }
    std::sort(output.begin(), output.end(), [](const LightBar& a, const LightBar& b) {
        return a.center.x < b.center.x;
    });
    return output;
}


// 5. 根据灯条几何关系筛选候选配对，再评分并避免重复使用灯条。
std::vector<LightPair> ArmorDetector::pair(const std::vector<LightBar>& lights, const cv::Mat& bgr) const {
    std::vector<LightPair> candidates;
    for (std::size_t i = 0; i < lights.size(); ++i) {
        for (std::size_t j = i + 1; j < lights.size(); ++j) {
            const LightBar& left = lights[i];
            const LightBar& right = lights[j];
            if(left.color != right.color) { continue; }
            const double angle_diff = std::abs(left.tilt_deg - right.tilt_deg);
            const double height_diff = std::abs(left.length - right.length) / std::max(left.length, right.length);
            const double mean_length = (left.length + right.length) * 0.5;
            const double dx_ratio = (right.center.x - left.center.x) / mean_length;
            const double dy_ratio = std::abs(right.center.y - left.center.y) / mean_length;
            const double center_angle = std::atan2(right.center.y - left.center.y,
                                                   right.center.x - left.center.x) * 180.0 / CV_PI;
            const double skew = std::abs(center_angle + 0.5 * (left.tilt_deg + right.tilt_deg));
            if(angle_diff >= config_.max_angle_diff_deg || height_diff >= config_.max_height_diff_ratio ||
                dx_ratio <= config_.min_pair_distance_ratio || dx_ratio >= config_.max_pair_distance_ratio ||
                dy_ratio >= config_.max_vertical_offset_ratio || skew >= config_.max_pair_skew_deg) { continue; }
            
            // 两根候选灯条之间若还有同色灯条，就不跨过它们配对。
            bool intervening = false;
            for (std::size_t k = i + 1; k < j; ++k) {
                const LightBar& middle = lights[k];
                if(middle.color == left.color &&
                    middle.center.x > left.center.x && middle.center.x < right.center.x &&
                    std::abs(middle.center.y - 0.5F * (left.center.y + right.center.y)) < 0.5 * mean_length &&
                    middle.length > 0.5 * mean_length) { intervening = true; break; }
            }
            if(intervening) { continue; }
            
            if(config_.check_pair_content) {
                // 透视展开两根灯条之间的区域，用亮度和灰度变化排除空白区域。
                const cv::Point2f source[4] = {left.top, right.top, right.bottom, left.bottom};
                const cv::Point2f destination[4] = {{0, 0}, {63, 0}, {63, 31}, {0, 31}};
                cv::Mat patch, gray;
                cv::warpPerspective(bgr, patch, cv::getPerspectiveTransform(source, destination), cv::Size(64, 32));
                cv::cvtColor(patch(cv::Rect(16, 6, 32, 20)), gray, cv::COLOR_BGR2GRAY);
                cv::Mat visible;
                cv::threshold(gray, visible, 30, 255, cv::THRESH_BINARY);
                cv::Scalar mean, deviation;
                cv::meanStdDev(gray, mean, deviation);
                if(cv::countNonZero(visible) < 0.05 * gray.total() || deviation[0] < 6.0) { continue; }
            }
            const cv::Point2f quad[4] = {left.top, right.top, right.bottom, left.bottom};
            const std::vector<cv::Point2f> polygon(quad, quad + 4);

            
            // 评分模块 1/5：灯条倾角越接近，得分越高。
            const double angle_score = std::clamp(1.0 - angle_diff / 12.0, 0.0, 1.0);

            // 评分模块 2/5：高度差在 10% 内满分，达到 40% 时为零分。
            const double height_ratio = std::abs(left.length - right.length) / mean_length;
            const double height_score = std::clamp(1.0 - std::max(0.0, height_ratio - 0.1) / 0.3, 0.0, 1.0);

            // 评分模块 3/5：灯条水平间距越符合装甲板比例，得分越高。
            double horizontal_score = 0.0;
            if(dx_ratio >= 1.4 && dx_ratio <= 2.8) {
                horizontal_score = 1.0;
            } else if(dx_ratio > 1.0 && dx_ratio < 1.4) {
                horizontal_score = (dx_ratio - 1.0) / 0.4;
            } else if(dx_ratio > 2.8 && dx_ratio < 3.6) {
                horizontal_score = (3.6 - dx_ratio) / 0.8;
            }

            // 评分模块 4/5：两根灯条中心的纵向偏移越小，得分越高。
            const double vertical_score = std::clamp(1.0 - std::max(0.0, dy_ratio - 0.15) / 0.35, 0.0, 1.0);

            // 评分模块 5/5：候选四边形越凸且面积越合理，得分越高。
            const double ideal_area = (right.center.x - left.center.x) * mean_length;
            const double shape_ratio = ideal_area > 0.0 ?
                std::min(1.0, std::abs(cv::contourArea(polygon)) / ideal_area) : 0.0;
            const double shape_score = cv::isContourConvex(polygon) ?
                std::clamp((shape_ratio - 0.5) / 0.35, 0.0, 1.0) : 0.0;

            
            const double score = 0.2 * (angle_score + height_score + horizontal_score +
                                        vertical_score + shape_score);
            if(score < 0.6) { continue; }
            ArmorSize size = dx_ratio > config_.large_armor_ratio ? ArmorSize::Large : ArmorSize::Small;
            if(config_.size == SizeMode::Small) { size = ArmorSize::Small; }
            if(config_.size == SizeMode::Large) { size = ArmorSize::Large; }
            candidates.push_back({i, j, size, score});
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const LightPair& a, const LightPair& b) {
        return a.score > b.score;
    });
    // 优先保留高分配对，每根灯条只分配给一个装甲板。
    std::vector<LightPair> output;
    std::vector<bool> used(lights.size(), false);
    for (const LightPair& candidate : candidates) {
        if(!used[candidate.left] && !used[candidate.right]) {
            output.push_back(candidate);
            used[candidate.left] = used[candidate.right] = true;
        }
    }
    return output;
}


// 6. 按统一顺序整理装甲板四角点，并过滤无效四边形。
std::vector<ArmorDetection> ArmorDetector::corners(const std::vector<LightBar>& lights,
                                                    const std::vector<LightPair>& pairs) const {
    std::vector<ArmorDetection> output;
    for (const LightPair& pair : pairs) {
        const LightBar& left = lights[pair.left];
        const LightBar& right = lights[pair.right];
        ArmorDetection armor;
        armor.color = left.color;
        armor.size = pair.size;
        armor.corners = {left.top, right.top, right.bottom, left.bottom};
        armor.center = (left.top + right.top + right.bottom + left.bottom) * 0.25F;
        armor.light_length = (left.length + right.length) * 0.5;
        const std::vector<cv::Point2f> polygon(armor.corners.begin(), armor.corners.end());
        if(cv::isContourConvex(polygon) && cv::contourArea(polygon) > 1.0) { output.push_back(armor); }
    }
    return output;
}


// 返回最终检测结果；中间处理步骤由 diagnose 顺序调用。
std::vector<ArmorDetection> ArmorDetector::detect(const cv::Mat& bgr) const {
    const DetectionDiagnostics diagnostics = diagnose(bgr);
    return corners(diagnostics.lights, diagnostics.pairs);
}


// 按 1–6 的顺序执行分割、形态学处理、轮廓筛选、灯条筛选、配对和角点生成。
DetectionDiagnostics ArmorDetector::diagnose(const cv::Mat& bgr) const {
    auto segmented = segment(bgr);
    auto cleaned = morph(segmented);
    auto found_contours = contours(cleaned);
    auto found_lights = lights(found_contours);
    auto found_pairs = pair(found_lights, bgr);
    return {std::move(found_contours), std::move(found_lights), std::move(found_pairs)};
}


// 建立装甲板自身坐标系下的四角点，长度单位为米。
std::vector<cv::Point3d> modelPoints(ArmorSize size) {
    const double half_width = size == ArmorSize::Small ? 0.0675 : 0.115;
    constexpr double half_height = 0.028;
    return {{-half_width, -half_height, 0.0}, {half_width, -half_height, 0.0},
            {half_width, half_height, 0.0}, {-half_width, half_height, 0.0}};
}


// 用四角点执行 PnP，再检查深度和重投影误差。
PoseOutput estimatePose(const ArmorDetection& armor, const CameraParameters& camera) {
    PoseOutput output;
    const std::vector<cv::Point3d> object_points = modelPoints(armor.size);
    std::vector<cv::Point2d> image_points;
    for (const cv::Point2f& point : armor.corners) {
        if(!std::isfinite(point.x) || !std::isfinite(point.y)) { return output; }
        image_points.emplace_back(point.x, point.y);
    }
    const std::vector<cv::Point2f> polygon(armor.corners.begin(), armor.corners.end());
    if(!cv::isContourConvex(polygon) || cv::contourArea(polygon) < 1.0) { return output; }
    
    // IPPE 求平面目标初值，LM 根据重投影误差进一步细化。
    if(!cv::solvePnP(object_points, image_points, camera.camera_matrix, camera.distort_coeffs,
                      output.rvec, output.tvec, false, cv::SOLVEPNP_IPPE)) { return output; }
    
    cv::solvePnPRefineLM(object_points, image_points, camera.camera_matrix, camera.distort_coeffs,
                         output.rvec, output.tvec, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 1e-6));
    // 拒绝非有限解、相机后方的目标，以及任一角点深度为负的解。
    if(!finite(output.rvec) || !finite(output.tvec) || output.tvec[2] <= 0.0) { return output; }
    cv::Matx33d rotation;
    cv::Rodrigues(output.rvec, rotation);
    for (const cv::Point3d& point : object_points) {
        const cv::Vec3d camera_point = rotation * cv::Vec3d(point.x, point.y, point.z) + output.tvec;
        if(camera_point[2] <= 0.0) { return output; }
    }
    std::vector<cv::Point2d> projected;
    cv::projectPoints(object_points, output.rvec, output.tvec,
                      camera.camera_matrix, camera.distort_coeffs, projected);
    double squared_error = 0.0;
    for (std::size_t i = 0; i < projected.size(); ++i) {
        const cv::Point2d error = projected[i] - image_points[i];
        squared_error += error.dot(error);
    }
    output.reprojection_error_px = std::sqrt(squared_error / 4.0);
    // 重投影误差超过与目标图像尺寸相关的阈值时，位姿不标记为有效。
    const double error_limit = armor.light_length > 0.0 ?
        std::min(5.0, std::max(1.0, 0.06 * armor.light_length)) : 5.0;
    output.valid = std::isfinite(output.reprojection_error_px) && output.reprojection_error_px <= error_limit;
    return output;
}


// 绘制检测框、角点、中心点和 PnP 位置，返回标注后的帧。
cv::Mat drawTargets(const cv::Mat& bgr, const std::vector<TargetResult>& targets) {
    cv::Mat output = bgr.clone();
    for (const TargetResult& target : targets) {
        const cv::Scalar color = target.armor.color == Color::Red ? cv::Scalar(70, 70, 255) : cv::Scalar(255, 170, 30);
        std::vector<cv::Point> polygon;
        for (const cv::Point2f& point : target.armor.corners) { polygon.emplace_back(cvRound(point.x), cvRound(point.y)); }
        cv::polylines(output, std::vector<std::vector<cv::Point>>{polygon}, true, color, 2, cv::LINE_AA);
        const std::array<std::string, 4> corner_names{"LT", "RT", "RB", "LB"};
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            cv::circle(output, polygon[i], 4, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
            label(output, corner_names[i], polygon[i] + cv::Point(5, -5), cv::Scalar(0, 255, 255), 0.4);
        }
        const cv::Point center(cvRound(target.armor.center.x), cvRound(target.armor.center.y));
        cv::drawMarker(output, center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
        std::ostringstream info;
        info << colorName(target.armor.color) << ' ' << sizeName(target.armor.size);
        label(output, info.str(), polygon[0] + cv::Point(0, -40), color);
        std::ostringstream position;
        if(target.pose.valid) {
            const cv::Vec3d& p = target.pose.tvec;
            position << std::fixed << std::setprecision(2) << "Pos: (" << p[0] << ", " << p[1] << ", " << p[2] << ") m";
        } else {
            position << "Pos: unavailable (PnP rejected)";
        }
        label(output, position.str(), polygon[0] + cv::Point(0, -18), cv::Scalar(255, 255, 255));
    }
    return output;
}


// 在视频帧上显示单帧处理耗时和对应帧率。
void drawTiming(cv::Mat& frame, double processing_ms, double processing_fps) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << "Process: " << processing_ms
         << " ms   " << processing_fps << " FPS";
    label(frame, text.str(), cv::Point(20, 35), cv::Scalar(0, 255, 0), 0.8);
}


// 打开 MJPG 视频输出；编码器不可用时直接报错。
AnnotatedVideoWriter::AnnotatedVideoWriter(const std::string& path, cv::Size size, double fps) : size_(size) {
    if(!writer_.open(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size_, true)) {
        throw std::runtime_error("Cannot open MJPG output: " + path);
    }
}


// 写入前检查帧尺寸和类型与视频设置一致。
void AnnotatedVideoWriter::write(const cv::Mat& frame) {
    if(frame.size() != size_ || frame.type() != CV_8UC3) { throw std::runtime_error("Invalid output frame"); }
    writer_.write(frame);
}
void AnnotatedVideoWriter::close() { writer_.release(); }


// 解析路径和检测选项；没有指定 CSV 时使用视频输出的同名路径。
Options parseOptions(int argc, char** argv) {
    Options output;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if(!seen.insert(key).second) { throw std::runtime_error("Duplicate argument: " + key); }
        if(key == "--help" || key == "-h") { output.help = true; continue; }
        if(key == "--show") { output.show = true; continue; }
        if(key != "--video" && key != "--calibration" && key != "--output" && key != "--csv" &&
            key != "--color" && key != "--armor-size" && key != "--max-frames" &&
            key != "--light-recovery" && key != "--pair-content" && key != "--threads") {
            throw std::runtime_error("Unknown argument: " + key);
        }
        if(i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
            throw std::runtime_error("Missing value for " + key);
        }
        const std::string value = argv[++i];
        if(key == "--video") { output.video = value; }
        else if(key == "--calibration") { output.calibration = value; }
        else if(key == "--output") { output.output = value; }
        else if(key == "--csv") { output.csv = value; }
        else if(key == "--color") {
            if(value == "red") { output.detector.color = ColorMode::Red; }
            else if(value == "blue") { output.detector.color = ColorMode::Blue; }
            else if(value == "both") { output.detector.color = ColorMode::Both; }
            else { throw std::runtime_error("--color must be red, blue or both"); }
        } else if(key == "--armor-size") {
            if(value == "small") { output.detector.size = SizeMode::Small; }
            else if(value == "large") { output.detector.size = SizeMode::Large; }
            else if(value == "auto") { output.detector.size = SizeMode::Auto; }
            else { throw std::runtime_error("--armor-size must be small, large or auto"); }
        } else if(key == "--light-recovery") {
            if(value != "on" && value != "off") { throw std::runtime_error("--light-recovery must be on or off"); }
            output.detector.recover_bright_cores = value == "on";
        } else if(key == "--pair-content") {
            if(value != "on" && value != "off") { throw std::runtime_error("--pair-content must be on or off"); }
            output.detector.check_pair_content = value == "on";
        } else if(key == "--threads") {
            if(value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
                throw std::runtime_error("--threads must be an integer in [0,256]");
            }
            const unsigned long long n = std::stoull(value);
            if(n > 256) { throw std::runtime_error("--threads must be in [0,256]"); }
            output.threads = static_cast<int>(n);
        } else if(key == "--max-frames") {
            if(value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
                throw std::runtime_error("--max-frames must be a nonnegative integer");
            }
            output.max_frames = std::stoull(value);
        }
    }
    if(!output.help && (output.video.empty() || output.calibration.empty() || output.output.empty())) {
        throw std::runtime_error("Required: --video, --calibration, --output");
    }
    if(!output.help && output.csv.empty()) {
        std::filesystem::path csv_path = std::filesystem::u8path(output.output);
        csv_path.replace_extension(".csv");
        output.csv = csv_path.u8string();
    }
    return output;
}

// 返回命令行参数帮助文本。
std::string usage() {
    return "Usage: auto_aim --video INPUT.avi --calibration CAMERA.yaml --output OUTPUT.avi\n"
           "                [--color red|blue|both] [--armor-size auto|small|large]\n"
           "                [--csv RESULTS.csv] [--max-frames N] [--show]\n"
           "                [--light-recovery on|off] [--pair-content on|off] [--threads N]\n"
           "Default: both colors, auto size, all frames, no GUI; CSV uses output stem.\n";
}
const char* colorName(Color color) { return color == Color::Red ? "red" : "blue"; }
const char* sizeName(ArmorSize size) { return size == ArmorSize::Small ? "small" : "large"; }

} 


namespace {

// 汇总处理帧数、检测数量和算法耗时。
struct RunStatistics {
    std::uint64_t frames = 0, detected_frames = 0, valid_poses = 0;
    double total_processing_ms = 0.0;
};


// 防止输出覆盖输入或已有文件，并按需创建父目录。
void prepareOutput(const std::string& name, const armor::Options& options) {
    const std::filesystem::path path = std::filesystem::u8path(name);
    const auto absolute = std::filesystem::weakly_canonical(path);
    if(absolute == std::filesystem::weakly_canonical(std::filesystem::u8path(options.video)) ||
        absolute == std::filesystem::weakly_canonical(std::filesystem::u8path(options.calibration))) {
        throw std::runtime_error("Output must not overwrite an input file");
    }
    if(std::filesystem::exists(path)) { throw std::runtime_error("Output already exists; choose a new path: " + name); }
    if(path.has_parent_path()) { std::filesystem::create_directories(path.parent_path()); }
}


void writeCsv(std::ofstream& csv, std::uint64_t frame, double timestamp,
              const armor::TargetResult* target, double elapsed) {
    // 字段顺序与 CSV 表头一致；没有目标时也保留该帧记录。
    csv << frame << ',' << timestamp << ',';
    if(target == nullptr) {
        csv << "none,none,0,";
        for (int i = 0; i < 17; ++i) { csv << ','; }
    } else {
        csv << armor::colorName(target->armor.color) << ',' << armor::sizeName(target->armor.size)
            << ',' << (target->pose.valid ? 1 : 0) << ',';
        for (const cv::Point2f& p : target->armor.corners) { csv << p.x << ',' << p.y << ','; }
        csv << target->armor.center.x << ',' << target->armor.center.y << ',';
        if(target->pose.valid) {
            for (int i = 0; i < 3; ++i) { csv << target->pose.tvec[i] << ','; }
            for (int i = 0; i < 3; ++i) { csv << target->pose.rvec[i] << ','; }
            csv << target->pose.reprojection_error_px << ',';
        } else {
            for (int i = 0; i < 7; ++i) { csv << ','; }
        }
    }
    csv << elapsed << ',' << (elapsed > 0.0 ? 1000.0 / elapsed : 0.0) << '\n';
}

} 


int main(int argc, char** argv) {
    try {
        // 1. 读取运行参数和相机标定，检查输出路径。
        const armor::Options options = armor::parseOptions(argc, argv);
        if(options.help) { std::cout << armor::usage(); return 0; }
        if(options.threads > 0) { cv::setNumThreads(options.threads); }
        const armor::CameraParameters camera = armor::loadCalibration(options.calibration);
        if(std::filesystem::u8path(options.output).extension() != ".avi") {
            throw std::runtime_error("--output must have an .avi extension for MJPG AVI");
        }
        if(std::filesystem::weakly_canonical(std::filesystem::u8path(options.output)) ==
            std::filesystem::weakly_canonical(std::filesystem::u8path(options.csv))) {
            throw std::runtime_error("Video output and CSV output must use different paths");
        }

        // 2. 打开视频并检查分辨率，创建视频和 CSV 输出。
        cv::VideoCapture capture(options.video);
        if(!capture.isOpened()) { throw std::runtime_error("Cannot open video: " + options.video); }
        cv::Mat frame;
        if(!capture.read(frame) || frame.empty()) { throw std::runtime_error("Video is empty or undecodable"); }
        const cv::Size required_size(1440, 1080);
        if(frame.size() != required_size || frame.size() != camera.image_size) {
            throw std::runtime_error("Video and calibration must both use 1440x1080; do not resize without scaling K");
        }
        double input_fps = capture.get(cv::CAP_PROP_FPS);
        if(!std::isfinite(input_fps) || input_fps <= 0.0) {
            input_fps = 165.0;
            std::cerr << "Warning: input FPS unavailable; using 165 FPS for timestamps.\n";
        }
        const double expected_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
        prepareOutput(options.output, options);
        prepareOutput(options.csv, options);
        armor::AnnotatedVideoWriter writer(options.output, required_size, 165.0);
        std::ofstream csv(std::filesystem::u8path(options.csv));
        if(!csv) { throw std::runtime_error("Cannot open CSV: " + options.csv); }
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << "frame,timestamp_s,color,size,pnp_valid,lt_x,lt_y,rt_x,rt_y,rb_x,rb_y,lb_x,lb_y,"
               "center_x,center_y,x_m,y_m,z_m,"
               "rvec_x,rvec_y,rvec_z,reprojection_rms_px,processing_ms,processing_fps\n";
        csv << std::fixed << std::setprecision(6);
        armor::ArmorDetector detector(options.detector);
        RunStatistics stats;
        bool stopped_early = false;
        std::cout << "OpenCV " << CV_VERSION << "; input: " << input_fps
                  << " FPS; output: MJPG 1440x1080 at 165 FPS\n";
        if(options.show) { cv::namedWindow("RoboMaster auto aim", cv::WINDOW_NORMAL); }
        const auto run_start = std::chrono::steady_clock::now();

        do {
            // 3. 检测当前帧中的装甲板。
            const auto start = std::chrono::steady_clock::now();
            const std::vector<armor::ArmorDetection> detections = detector.detect(frame);
            const double timestamp = static_cast<double>(stats.frames) / input_fps;
            std::vector<armor::TargetResult> targets;
            cv::Mat visualization = frame;

            // 4. 对检测结果进行 PnP 解算并绘制标注。
            if(!detections.empty()) {
                ++stats.detected_frames;
                for (const armor::ArmorDetection& detection : detections) {
                    const armor::PoseOutput pose = armor::estimatePose(detection, camera);
                    targets.push_back({detection, pose});
                    if(pose.valid) { ++stats.valid_poses; }
                }
                visualization = armor::drawTargets(frame, targets);
            }

            // 5. 记录当前帧的算法处理耗时。
            const auto end = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            stats.total_processing_ms += elapsed;
            armor::drawTiming(visualization, elapsed, elapsed > 0.0 ? 1000.0 / elapsed : 0.0);

            // 6. 写入标注视频和逐帧结果；需要时显示预览。
            writer.write(visualization);
            if(targets.empty()) { writeCsv(csv, stats.frames, timestamp, nullptr, elapsed); }
            else {
                for (const armor::TargetResult& target : targets) { writeCsv(csv, stats.frames, timestamp, &target, elapsed); }
            }
            ++stats.frames;
            if(stats.frames % 100 == 0) { std::cout << "Processed " << stats.frames << " frames\n"; }
            if(options.show) {
                cv::imshow("RoboMaster auto aim", visualization);
                const int key = cv::waitKey(1);
                if(key == 27 || key == 'q') { stopped_early = true; break; }
            }
            if(options.max_frames > 0 && stats.frames >= options.max_frames) { stopped_early = true; break; }
        } while (capture.read(frame) && !frame.empty());

        // 7. 关闭输入输出，检查解码情况并汇总运行数据。
        writer.close();
        csv.close();
        capture.release();
        if(options.show) { cv::destroyAllWindows(); }
        if(!stopped_early && std::isfinite(expected_frames) && expected_frames > 0 &&
            static_cast<double>(stats.frames) + 0.5 < expected_frames) {
            throw std::runtime_error("Video decoding ended before advertised frame count; output is partial");
        }
        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
        const double average_ms = stats.total_processing_ms / static_cast<double>(stats.frames);
        std::cout << std::fixed << std::setprecision(2)
                  << "Frames: " << stats.frames << ", detected frames: " << stats.detected_frames
                  << ", valid poses: " << stats.valid_poses << '\n'
                  << "Mean processing: " << average_ms << " ms (" << 1000.0 / average_ms << " FPS)\n"
                  << "Throughput including video/CSV I/O: " << static_cast<double>(stats.frames) / total_seconds << " FPS\n"
                  << "Video: " << options.output << "\nCSV: " << options.csv << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n' << armor::usage();
        return 1;
    }
}
