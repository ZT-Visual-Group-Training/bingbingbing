// ============================================================
// ImuReader.cpp —— IMU 模块的"怎么做"部分
//   对应 ImuReader.h 里声明的每一个函数，一个一个实现。
// ============================================================
#include "ImuReader.h"

#include <algorithm>   // std::min / std::max（夹住 asin 的输入）
#include <cmath>       // std::sqrt / std::atan2 / std::asin
#include <fstream>     // std::ifstream 读 CSV
#include <sstream>     // std::stringstream 按逗号切字符串

// ---------- 小工具：把一行 CSV 按逗号切开 ----------
static std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty() && item.back() == '\r')   // 去掉 Windows 换行残留的 \r
            item.pop_back();
        out.push_back(item);
    }
    return out;
}

// ============================================================
// 数学工具 1：四元数 → 3x3 旋转矩阵
//   四元数 (w,x,y,z) 是"用 4 个数表示三维旋转"的写法，比旋转矩阵省空间、
//   不会遇到万向节死锁。但要用来乘坐标，得先转成 3x3 矩阵。
//   公式是标准的，记不住没关系——会用即可。
// ============================================================
cv::Mat quatToRotation(double w, double x, double y, double z) {
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    if (n < 1e-12) return cv::Mat::eye(3, 3, CV_64F);   // 防御：零四元数返回单位阵
    w /= n; x /= n; y /= n; z /= n;                      // 归一化

    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y));
    return R;
}

// ============================================================
// 数学工具 2：四元数 → ZYX 欧拉角（roll/pitch/yaw，度）
//   roll  = 绕 X 轴转（左右翻滚）
//   pitch = 绕 Y 轴转（抬头低头）
//   yaw   = 绕 Z 轴转（左右转头）
//   asin 的输入必须夹在 [-1,1]，否则浮点误差会让它出 NaN。
// ============================================================
cv::Vec3d quatToEulerDeg(double w, double x, double y, double z) {
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    if (n < 1e-12) return cv::Vec3d(0, 0, 0);
    w /= n; x /= n; y /= n; z /= n;

    double roll  = std::atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y));
    double sinp  = 2*(w*y - z*x);
    sinp = std::max(-1.0, std::min(1.0, sinp));   // 关键：防止 asin 出 NaN
    double pitch = std::asin(sinp);
    double yaw   = std::atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z));

    const double k = 180.0 / CV_PI;               // 弧度 → 度
    return cv::Vec3d(roll*k, pitch*k, yaw*k);
}

// ============================================================
// ImuReader::load —— 读整个 frame_imu.csv
// ============================================================
bool ImuReader::load(const std::string& csvPath) {
    std::ifstream in(csvPath);
    if (!in.is_open()) return false;
    poses_.clear();

    std::string line;
    if (!std::getline(in, line)) return false;   // 第一行是表头，直接跳过

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f = splitCsv(line);
        // 需要的列：0=frame_index 2=elapsed_s 3~6=qw/qx/qy/qz 10=status
        if (f.size() < 7) continue;

        ImuPose p;
        try {
            p.frameIndex = std::stoi(f[0]);
            p.elapsedS   = std::stod(f[2]);
            p.qw = std::stod(f[3]);
            p.qx = std::stod(f[4]);
            p.qy = std::stod(f[5]);
            p.qz = std::stod(f[6]);
        } catch (...) {
            continue;   // 某一行坏了就跳过，不影响整体
        }
        if (f.size() >= 11) p.status = f[10];

        cv::Vec3d e = quatToEulerDeg(p.qw, p.qx, p.qy, p.qz);
        p.roll = e[0]; p.pitch = e[1]; p.yaw = e[2];
        p.valid = true;
        poses_.push_back(p);
    }
    return !poses_.empty();
}

// 按帧号取姿态：正常数据帧号连续，直接按下标取最快；不连续则回退线性查找
const ImuPose* ImuReader::poseAt(int frameIndex) const {
    if (frameIndex < 0) return nullptr;
    if (!poses_.empty() && frameIndex < static_cast<int>(poses_.size())
        && poses_[static_cast<std::size_t>(frameIndex)].frameIndex == frameIndex) {
        return &poses_[static_cast<std::size_t>(frameIndex)];
    }
    for (const auto& p : poses_)
        if (p.frameIndex == frameIndex) return &p;
    return nullptr;
}

int ImuReader::minFrame() const {
    return poses_.empty() ? -1 : poses_.front().frameIndex;
}

int ImuReader::maxFrame() const {
    return poses_.empty() ? -1 : poses_.back().frameIndex;
}

// ============================================================
// 外参读取：官方 handeye.yaml 含中文注释，OpenCV 的 FileStorage 读不了，
// 所以这里用最朴素、最稳的办法——逐行文本解析：
//   找 "R_camera2gimbal: [a,b,c,...]" 这样的行，抠出 [ ] 里的数字。
// 好处：不挑编码、不挑 YAML 解析器、换 Ubuntu 也一样能读。
// ============================================================
namespace {
// 在文本 line 里找形如 "key: [n1,n2,n3]" 的数组，解析出 count 个 double。
static bool parseArrayLine(const std::string& line, const std::string& key,
                           std::vector<double>& out, std::size_t count) {
    std::size_t kp = line.find(key);
    if (kp == std::string::npos) return false;
    std::size_t lb = line.find('[', kp);
    std::size_t rb = line.find(']', lb == std::string::npos ? kp : lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;

    std::string body = line.substr(lb + 1, rb - lb - 1);
    std::vector<std::string> parts = splitCsv(body);   // 用逗号切（数字里没逗号）
    if (parts.size() < count) return false;

    out.clear();
    try {
        for (std::size_t i = 0; i < count; ++i) out.push_back(std::stod(parts[i]));
    } catch (...) {
        return false;
    }
    return true;
}
}  // namespace

bool HandEyeExtrinsics::load(const std::string& yamlPath) {
    std::ifstream in(yamlPath);
    if (!in.is_open()) return false;

    std::vector<double> rc, rg, t;
    std::string line;
    while (std::getline(in, line)) {
        if (rc.empty())  parseArrayLine(line, "R_camera2gimbal",  rc, 9);
        if (rg.empty())  parseArrayLine(line, "R_gimbal2imubody", rg, 9);
        if (t.empty())   parseArrayLine(line, "t_camera2gimbal",  t,  3);
    }
    if (rc.size() != 9 || rg.size() != 9 || t.size() != 3) return false;

    R_camera2gimbal  = cv::Mat(rc, true).reshape(1, 3).clone();   // 9 个数 → 3x3
    R_gimbal2imubody = cv::Mat(rg, true).reshape(1, 3).clone();
    t_camera2gimbal  = cv::Vec3d(t[0], t[1], t[2]);
    valid = true;
    return true;
}

// 核心坐标转换：相机系 → 云台系 → IMU body 系
cv::Vec3d HandEyeExtrinsics::cameraToImuBody(const cv::Vec3d& pCam) const {
    if (!valid) return pCam;   // 没外参就原样返回，绝不改检测本身
    cv::Mat pc = (cv::Mat_<double>(3, 1) << pCam[0], pCam[1], pCam[2]);
    cv::Mat tmat = cv::Mat(t_camera2gimbal);                // Vec3d 显式转 Mat，避免运算符歧义
    cv::Mat pg = R_camera2gimbal * pc + tmat;               // 相机系 → 云台系
    cv::Mat pi = R_gimbal2imubody * pg;                     // 云台系 → IMU body 系
    return cv::Vec3d(pi.at<double>(0), pi.at<double>(1), pi.at<double>(2));
}
