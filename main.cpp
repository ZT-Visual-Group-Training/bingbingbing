// ============================================================================
// main.cpp —— 装甲板视觉系统【统一入口】
//
// 以前有三个 main（main.cpp / main_video.cpp / hsv_tuner.cpp），功能分散、代码重复。
// 现在合并成这一个文件：用一个 --mode 参数切换三种功能，公共代码只写一次。
//
// 三种模式：
//   armor_vision                          默认 = video 视频检测（考核主用）
//   armor_vision --mode image             单张图片检测（自测/看单帧）
//   armor_vision --mode video --video x.avi  视频逐帧检测
//   armor_vision --mode hsv                HSV 阈值调试器（拖滑块找颜色阈值）
//   armor_vision --help                   查看全部用法
//
// 文件结构（从上到下）：
//   ① 公共工具        loadCalib —— 三种模式共用的相机参数读取
//   ② 模式一：image    makeSyntheticScene + runImage
//   ③ 模式二：video    makeFrame + makeTestVideo + runVideo
//   ④ 模式三：hsv      HSV 全局变量 + buildMask + runHsv
//   ⑤ main             解析 --mode，分发到对应模式
// ============================================================================

#include <opencv2/opencv.hpp>   // OpenCV 全家桶
#include <chrono>               // 计时：std::chrono
#include <cmath>                // std::sin / std::fabs
#include <filesystem>           // std::filesystem::exists：判断默认视频在不在
#include <fstream>              // std::ofstream：按 s 时把数值写进 txt 文件
#include <iostream>             // std::cout / std::cerr
#include <sstream>              // std::ostringstream：拼字符串后同时输出到屏幕和文件
#include <string>               // std::string
#include <vector>               // std::vector

#include "ArmorDetector.h"
#include "TemporalFilter.h"      // 检测器：ArmorDetector / CameraCalib / ArmorParam
#include "ImuReader.h"          // IMU 模块：读取姿态四元数 + 坐标转换（独立模块）

// ============================================================================
// IMU 画面叠加：只负责"把 IMU 信息画到结果图上"，不改检测本身。
// 这样即使不接 IMU，原来的检测画面也一模一样。
//   - 左上：姿态角 roll/pitch/yaw（度）+ 数据来源 status
//   - 每块装甲板下方：相机系坐标 tvec(C) + 转成 IMU body 系后的坐标 tvec(I)
// ============================================================================
static void drawImuOverlay(cv::Mat& out,
                           const ImuPose* pose,
                           const std::vector<Armor>& armors,
                           const HandEyeExtrinsics& ext) {
    if (!pose) return;   // 原地画，不再 clone 整张图（省一次 1440x1080 全图拷贝）

    // 左上姿态角
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1)
        << "IMU  R:" << pose->roll << " P:" << pose->pitch
        << " Y:" << pose->yaw << " [" << pose->status << "]";
    cv::putText(out, oss.str(), cv::Point(12, 54),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    // 每块装甲板：相机系(C) → IMU body 系(I) 坐标。
    // 用和 drawResult 相同的 labelShiftY 错开，保证同一块板的三行文字上下排开、不串到邻板。
    for (size_t i = 0; i < armors.size(); ++i) {
        cv::Vec3d pI = ext.cameraToImuBody(armors[i].tvec);
        std::ostringstream s;
        s << std::fixed << std::setprecision(2)
          << "C(" << armors[i].tvec[0] << "," << armors[i].tvec[1] << "," << armors[i].tvec[2] << ")"
          << " -> I(" << pI[0] << "," << pI[1] << "," << pI[2] << ")";
        int shift = labelShiftY(i);
        cv::Point org(cvRound(armors[i].center.x - 88),
                      cvRound(armors[i].center.y + 44 + shift));
        cv::putText(out, s.str(), org,
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
}

// ============================================================================
// ① 公共工具：读相机标定参数
//    三种模式都要用，所以只写这一份。读不到文件就返回 false，调用方用默认值。
//    期望 YAML 里有 camera_matrix(3x3) 和 distortion_coefficients(5个)。
// ============================================================================
static bool loadCalib(const std::string& path, CameraCalib& calib) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    cv::Mat K, D;
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> D;
    fs.release();
    if (K.empty()) return false;
    calib.fx = K.at<double>(0, 0);
    calib.fy = K.at<double>(1, 1);
    calib.cx = K.at<double>(0, 2);
    calib.cy = K.at<double>(1, 2);
    if (!D.empty() && D.rows >= 5) {
        calib.k1 = D.at<double>(0); calib.k2 = D.at<double>(1);
        calib.p1 = D.at<double>(2); calib.p2 = D.at<double>(3);
        calib.k3 = D.at<double>(4);
    }
    return true;
}

// 小工具：按路径载入相机参数并打印提示（三种模式通用，省去重复代码）
static CameraCalib prepareCalib(const std::string& calibPath) {
    CameraCalib calib;
    if (!calibPath.empty() && loadCalib(calibPath, calib)) {
        std::cout << "[armor] 已载入相机参数: " << calibPath << '\n';
    } else {
        std::cout << "[armor] 使用内置相机参数（已写入官方标定值 1440x1080）\n";
    }
    return calib;
}

// ============================================================================
// ② 模式一：单张图片检测
// ============================================================================

// 没给图片时造一张合成图：左边红板、右边蓝板，方便没数据时自测
static cv::Mat makeSyntheticScene() {
    cv::Mat img(360, 640, CV_8UC3, cv::Scalar(35, 35, 35));
    cv::rectangle(img, cv::Rect(60, 100, 180, 160), cv::Scalar(20, 20, 20), cv::FILLED);
    cv::rectangle(img, cv::Rect(90,  130, 16, 100), cv::Scalar(0, 0, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(200, 130, 16, 100), cv::Scalar(0, 0, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(380, 100, 180, 160), cv::Scalar(20, 20, 20), cv::FILLED);
    cv::rectangle(img, cv::Rect(410, 130, 16, 100), cv::Scalar(255, 0, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(520, 130, 16, 100), cv::Scalar(255, 0, 0), cv::FILLED);
    cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
    return img;
}

// image 模式主函数：读一张图 → 检测 → 保存 → 打印数据
static int runImage(int argc, char** argv) {
    std::string imagePath, outPath = "armor_result.png", calibPath;
    bool show = true, showLights = true;   // 默认画灯条识别点，--no-lights 可关
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--image"  && i + 1 < argc) imagePath = argv[++i];
        else if (arg == "--out"    && i + 1 < argc) outPath   = argv[++i];
        else if (arg == "--camera" && i + 1 < argc) calibPath = argv[++i];
        else if (arg == "--no-show")               show = false;
        else if (arg == "--no-lights")             showLights = false;
        else if (arg.rfind("--", 0) != 0)           imagePath = arg;  // 位置参数当图片
        else { std::cerr << "[image] 未知参数: " << arg << '\n'; return 1; }
    }

    cv::Mat bgr = imagePath.empty() ? cv::Mat() : cv::imread(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        bgr = makeSyntheticScene();
        std::cout << "[image] 未提供有效图片，使用内置合成图（红板+蓝板）\n";
    }
    CameraCalib calib = prepareCalib(calibPath);

    ArmorDetector detector;
    std::vector<LightBar> lights;   // 顺带取出灯条，用于画识别点
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Armor> armors = detector.detect(bgr, calib, showLights ? &lights : nullptr);
    auto t1 = std::chrono::steady_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cv::Mat result = ArmorDetector::drawResult(bgr, armors, elapsedMs,
                                               showLights ? lights : std::vector<LightBar>{});
    if (!cv::imwrite(outPath, result)) {
        std::cerr << "[image] 结果保存失败: " << outPath << '\n';
        return 2;
    }
    std::cout << "[image] 图像尺寸: " << bgr.cols << "x" << bgr.rows << '\n';
    std::cout << "[image] 检测到装甲板: " << armors.size() << " 块\n";
    for (size_t i = 0; i < armors.size(); ++i) {
        const Armor& a = armors[i];
        std::cout << "  装甲板#" << i
                  << " 颜色=" << (a.color == ArmorColor::Red ? "红" : "蓝")
                  << " 中心(" << (int)a.center.x << "," << (int)a.center.y << ")"
                  << " 距离=" << a.distance << "m\n";
    }
    std::cout << "[image] 处理耗时: " << elapsedMs << " ms\n";
    std::cout << "[image] 结果已保存: " << outPath << '\n';
    if (show) {
        cv::imshow("armor result", result);
        std::cout << "[image] 按任意键关闭窗口...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}

// ============================================================================
// ③ 模式二：视频逐帧检测（考核主用）
// ============================================================================

// 造一帧动画：phase 变化 → 红板左右移动、蓝板缩放
static cv::Mat makeFrame(double phase) {
    cv::Mat img(360, 640, CV_8UC3, cv::Scalar(35, 35, 35));
    int rx = 60 + static_cast<int>(50 * std::sin(phase));
    cv::rectangle(img, cv::Rect(rx, 100, 180, 160), cv::Scalar(20, 20, 20), cv::FILLED);
    cv::rectangle(img, cv::Rect(rx + 30, 130, 16, 100), cv::Scalar(0, 0, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(rx + 140, 130, 16, 100), cv::Scalar(0, 0, 255), cv::FILLED);

    double s = 1.0 + 0.35 * std::sin(phase * 0.7 + 1.0);
    int bw = static_cast<int>(180 * s), bh = static_cast<int>(160 * s);
    int bx = 380 - (bw - 180) / 2, by = 100 - (bh - 160) / 2;
    int lw = static_cast<int>(16 * s), lh = static_cast<int>(100 * s);
    int ly = by + (bh - lh) / 2;
    cv::rectangle(img, cv::Rect(bx, by, bw, bh), cv::Scalar(20, 20, 20), cv::FILLED);
    cv::rectangle(img, cv::Rect(bx + 30, ly, lw, lh), cv::Scalar(255, 0, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(bx + bw - 30 - lw, ly, lw, lh), cv::Scalar(255, 0, 0), cv::FILLED);
    cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
    return img;
}

static bool makeTestVideo(const std::string& path) {
    cv::VideoWriter writer(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           30.0, cv::Size(640, 360));
    if (!writer.isOpened()) {
        std::cerr << "[video] 无法创建测试视频: " << path << '\n';
        return false;
    }
    for (int i = 0; i < 90; ++i) writer.write(makeFrame(i * 0.12));
    writer.release();
    std::cout << "[video] 已生成测试视频: " << path << "（90 帧 / 30fps / 3 秒）\n";
    return true;
}

// video 模式主函数：打开视频 → 逐帧 读/检测/画/写 → 收尾
static int runVideo(int argc, char** argv) {
    std::string videoPath, outPath = "armor_result.avi", calibPath;
    std::string imuPath, handeyePath;                 // IMU 姿态 CSV + 手眼外参 YAML
    bool show = true, onlyMakeTest = false, showLights = true;   // 默认画灯条识别点
    bool useImu = true;                               // 默认尽量自动加载 IMU
    bool noWrite = false;                             // --no-write：只看实时画面时不写盘，减少卡顿
    bool useTemporal = false;                         // --temporal：时序多数投票修断灯条歪框（默认关）
    bool fastMode = false;                            // --fast：极速模式，detectScale=0.75（默认关，保精度）
    int dumpFrame = -1, maxFrames = -1;               // maxFrames>0 时只处理前 N 帧（快速自测用）
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--video"      && i + 1 < argc) videoPath = argv[++i];
        else if (arg == "--out"        && i + 1 < argc) outPath   = argv[++i];
        else if (arg == "--camera"     && i + 1 < argc) calibPath = argv[++i];
        else if (arg == "--imu"        && i + 1 < argc) imuPath     = argv[++i];
        else if (arg == "--handeye"    && i + 1 < argc) handeyePath = argv[++i];
        else if (arg == "--max-frames" && i + 1 < argc) maxFrames = std::stoi(argv[++i]);
        else if (arg == "--no-imu")                   useImu = false;
        else if (arg == "--no-write")                 noWrite = true;
        else if (arg == "--temporal")                 useTemporal = true;
        else if (arg == "--fast")                     fastMode = true;
        else if (arg == "--make-test")                onlyMakeTest = true;
        else if (arg == "--no-show")                  show = false;
        else if (arg == "--no-lights")                showLights = false;
        else if (arg == "--dump-frame" && i + 1 < argc) dumpFrame = std::stoi(argv[++i]);
        else if (arg.rfind("--", 0) != 0)              videoPath = arg;
        else { std::cerr << "[video] 未知参数: " << arg << '\n'; return 1; }
    }

    if (onlyMakeTest)
        return makeTestVideo(videoPath.empty() ? "test_scene.avi" : videoPath) ? 0 : 1;
    if (videoPath.empty()) {
        // 默认用真实考核视频；只有它不存在时才退回合成测试视频（方便换环境自测）
        const char* realVideo = "C:/carrie/video.avi";
        if (std::filesystem::exists(realVideo)) {
            videoPath = realVideo;
        } else {
            std::cerr << "[video] 默认视频不存在: " << realVideo << "，改用合成测试视频\n";
            videoPath = "test_scene.avi";
            if (!makeTestVideo(videoPath)) return 1;
        }
    }

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[video] 打不开视频: " << videoPath << '\n';
        return 1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (fps <= 0 || fps > 120) fps = 30;

    cv::VideoWriter writer;
    if (!noWrite) {
        writer.open(outPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "[video] 无法创建输出视频: " << outPath << '\n';
            return 2;
        }
    }
    CameraCalib calib = prepareCalib(calibPath);

    // ---- IMU 接入（可选，找不到数据就跳过，绝不影响检测）----
    // 默认路径：exe 同目录的 imu_data/ 文件夹（已随项目附带官方数据）。
    // 迁移到 Ubuntu 时把 imu_data/ 整个拷过去即可，路径自动认。
    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    if (exeDir.empty()) exeDir = ".";
    // CLion 里 exe 常在项目下的 cmake-build-debug/，数据在项目根目录，
    // 所以依次找：exe同目录/imu_data → exe父目录/imu_data，找到哪个用哪个。
    auto findData = [&](const std::string& fileName) -> std::string {
        std::filesystem::path p1 = std::filesystem::path(exeDir) / "imu_data" / fileName;
        std::filesystem::path p2 = std::filesystem::path(exeDir).parent_path() / "imu_data" / fileName;
        if (std::filesystem::exists(p1)) return p1.string();
        if (std::filesystem::exists(p2)) return p2.string();
        return p1.string();   // 都没找到就返回第一个，让加载失败打印清晰路径
    };
    if (imuPath.empty())     imuPath     = findData("imu_frame_imu.csv");
    if (handeyePath.empty()) handeyePath = findData("calibration_handeye.yaml");

    ImuReader imu;
    HandEyeExtrinsics ext;
    bool imuReady = false;
    if (useImu) {
        bool okPose = imu.load(imuPath);
        bool okExt  = ext.load(handeyePath);
        imuReady = okPose && okExt;
        if (imuReady) {
            std::cout << "[imu] 已载入 " << imu.size() << " 帧姿态（帧 "
                      << imu.minFrame() << "~" << imu.maxFrame() << "）+ 手眼外参\n";
            const ImuPose* p0 = imu.poseAt(0);
            if (p0) std::cout << std::fixed << std::setprecision(2)
                              << "[imu] 第0帧姿态 roll=" << p0->roll
                              << " pitch=" << p0->pitch << " yaw=" << p0->yaw
                              << "  status=" << p0->status << '\n';
        } else {
            std::cout << "[imu] 未找到 IMU 数据/外参，本轮跳过 IMU（检测照常）\n"
                      << "      期望: " << imuPath << '\n'
                      << "      期望: " << handeyePath << '\n';
        }
    } else {
        std::cout << "[imu] --no-imu 已指定，跳过 IMU\n";
    }
    std::cout << std::flush;   // 立刻刷出加载信息，避免被后来的强杀丢掉

    // 【--fast 极速模式】detectScale=0.75 + 关高斯模糊：
    //   实测 12.4→约7.7ms（约1.6倍）、中心点误差<1px、框数0差异；
    //   代价是测距偏差最大约0.2~0.4m。默认关闭以保证测距精度。
    ArmorParam detParam;
    if (fastMode) { detParam.detectScale = 0.75f; detParam.preBlur = false; }
    ArmorDetector detector(detParam);
    if (fastMode) std::cout << "[fast] 极速模式：scale=0.75 + 关模糊（测距偏差最大约0.4m）\n";
    // 时序滤波器：默认关闭；--temporal 打开后启用 enabled=true
    TemporalParam tparam;
    tparam.enabled = useTemporal;
    TemporalFilter temporal(tparam);
    if (useTemporal)
        std::cout << "[temporal] 时序多数投票已启用（修断灯条歪框，默认关闭）\n";
    int frameNo = 0;
    double sumMs = 0.0;
    std::cout << "[video] 输入视频: " << videoPath
              << " | " << width << "x" << height
              << " | " << fps << "fps | 共 " << total << " 帧\n";

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;
        std::vector<LightBar> lights;   // 顺带取出灯条，用于画识别点
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Armor> armors = detector.detect(frame, calib, showLights ? &lights : nullptr);
        if (useTemporal) temporal.apply(armors, detector, calib);   // 时序修正（默认跳过）
        auto t1 = std::chrono::steady_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cv::Mat result = ArmorDetector::drawResult(frame, armors, elapsedMs,
                                                   showLights ? lights : std::vector<LightBar>{});
        // IMU 叠加：用当前帧号取姿态，把相机系坐标换算到 IMU body 系后画到图上。
        // 检测框/角点/耗时都是原来的 drawResult 画的，这里只是多叠一层文字。
        if (imuReady) {
            const ImuPose* pose = imu.poseAt(frameNo);
            drawImuOverlay(result, pose, armors, ext);
        }
        std::string info = "Frame " + std::to_string(frameNo + 1) + "/" + std::to_string(total);
        cv::putText(result, info, cv::Point(width - 130, 22), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        if (!noWrite) writer.write(result);

        if (frameNo == dumpFrame) {
            cv::imwrite("frame_dump.png", result);
            std::cout << "[video] 已导出第 " << frameNo << " 帧 → frame_dump.png\n";
        }
        if (show) {
            cv::imshow("armor video", result);
            if (cv::waitKey(1) == 27) break;
        }
        sumMs += elapsedMs;
        if ((frameNo + 1) % 30 == 0)
            std::cout << "[video] 已处理 " << (frameNo + 1) << " / " << total << " 帧\n";
        ++frameNo;
        if (maxFrames > 0 && frameNo >= maxFrames) break;   // 只处理前 N 帧后提前结束
    }

    cap.release();
    writer.release();
    if (show) cv::destroyAllWindows();
    double avgMs = frameNo > 0 ? sumMs / frameNo : 0.0;
    std::cout << "[video] 处理完成，共 " << frameNo << " 帧\n";
    std::cout << "[video] 平均每帧耗时: " << avgMs << " ms（理论上限 "
              << (avgMs > 0 ? 1000.0 / avgMs : 0) << " fps）\n";
    std::cout << "[video] " << (noWrite ? "未写盘（--no-write）" : "结果视频已保存: " + outPath) << '\n';
    return 0;
}

// ============================================================================
// ④ 模式三：HSV 阈值调试器（拖滑块找颜色阈值）
//    这些全局变量专门给 OpenCV 滑块用：createTrackbar 把变量地址交给它，
//    拖滑块时它直接改这些变量，主循环读变量就能拿到最新值。
// ============================================================================
// 【2026-11-24 修复】滑块默认值直接取自 ArmorParam，保证调试器和实际检测用同一套初值。
// 以前这里写死 redHighH1=10，而实际代码已改成 30，导致"调出来的值抄回去对不上"。
static ArmorParam g_tunerParam;                // 用 ArmorParam 默认构造，作为唯一真值来源
static int g_lowH1 = g_tunerParam.redLowH1, g_highH1 = g_tunerParam.redHighH1;
static int g_lowH2 = g_tunerParam.redLowH2, g_highH2 = g_tunerParam.redHighH2;
static int g_blueLowH = g_tunerParam.blueLowH, g_blueHighH = g_tunerParam.blueHighH;
static int g_minS = g_tunerParam.minS, g_minV = g_tunerParam.minV;
static int g_color = 0;                        // 0=红 1=蓝
static int g_openK = 1;                        // 形态学开关：1=开(和实际一致) 0=关

static void onTrack(int /*value*/, void* /*userData*/) {}  // 空回调，变量已被滑块直接改

// 用当前阈值生成掩膜：白色=被选中的像素（灯条），黑色=背景
static cv::Mat buildMask(const cv::Mat& hsv) {
    cv::Mat mask;
    if (g_color == 0) {
        cv::Mat p1, p2;
        cv::inRange(hsv, cv::Scalar(g_lowH1, g_minS, g_minV),
                         cv::Scalar(g_highH1, 255, 255), p1);
        cv::inRange(hsv, cv::Scalar(g_lowH2, g_minS, g_minV),
                         cv::Scalar(g_highH2, 255, 255), p2);
        cv::bitwise_or(p1, p2, mask);
    } else {
        cv::inRange(hsv, cv::Scalar(g_blueLowH, g_minS, g_minV),
                         cv::Scalar(g_blueHighH, 255, 255), mask);
    }
    // 【2026-11-24 修复】形态学与实际 makeColorMask 完全一致：
    // 3x3 开运算去噪 + 3x3 闭运算迭代2次填孔。g_openK=1 做，=0 关（看原始掩膜对比）。
    if (g_openK > 0) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    }
    return mask;
}

// 把单通道显示成伪彩，方便看 H/S/V 各通道
static cv::Mat channelToColor(const cv::Mat& ch, int colorMap) {
    cv::Mat out;
    cv::applyColorMap(ch, out, colorMap);
    return out;
}

// hsv 模式主函数：读视频/图片 → 建滑块 → 循环更新掩膜 → 按键逐帧切换 → s 打印数值
static int runHsv(int argc, char** argv) {
    // exe 所在目录：按 s 时把数值写到这里，打印完整路径给用户，绕开控制台缓冲问题
    std::string argv0Dir = std::filesystem::path(argv[0]).parent_path().string();
    if (argv0Dir.empty()) argv0Dir = ".";

    std::string media = "C:/carrie/video.avi";
    int startFrame = 900;
    bool useImage = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--image" && i + 1 < argc) { media = argv[++i]; useImage = true; }
        else if (a == "--video" && i + 1 < argc) { media = argv[++i]; useImage = false; }
        else if (a == "--frame" && i + 1 < argc) { startFrame = std::stoi(argv[++i]); }
        else if (a.rfind("--", 0) != 0)          { media = a; }
        else { std::cerr << "[hsv] 未知参数: " << a << '\n'; return 1; }
    }

    // cap 不再读完就 release，而是保持打开，供逐帧读取
    cv::VideoCapture cap;
    cv::Mat frame;                 // 当前画面
    int frameNo = startFrame;      // 当前帧号（显示用）
    int totalFrames = 0;           // 总帧数；<=0 表示未知（AVI 无索引时常见）

    if (useImage) {
        frame = cv::imread(media);
        frameNo = 0;
    } else {
        cap.open(media);
        if (!cap.isOpened()) {
            std::cerr << "[hsv] 打不开视频: " << media << '\n';
            return 1;
        }
        totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        // 【2026-11-24 修复】只有 totalFrames>0 才用它裁剪帧号，
        // 否则无索引 AVI 返回 0，旧代码会把 frameNo 设成 -1。
        if (totalFrames > 0 && frameNo >= totalFrames) frameNo = totalFrames - 1;
        if (frameNo > 0) cap.set(cv::CAP_PROP_POS_FRAMES, frameNo);
        if (!cap.read(frame) || frame.empty()) {
            // 随机 seek 失败时兜底：从头顺序读到目标帧
            cap.release();
            cap.open(media);
            for (int i = 0; i < frameNo && cap.read(frame); ++i) {}
        }
    }
    if (frame.empty()) {
        std::cerr << "[hsv] 读不到画面: " << media << '\n';
        return 1;
    }
    std::cout << "[hsv] 载入画面: " << frame.cols << "x" << frame.rows;
    if (useImage) std::cout << " (image)\n";
    else          std::cout << " (frame " << frameNo << ")\n";

    cv::Mat hsv;
    auto refreshHsv = [&]() { cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); };
    refreshHsv();

    // 下一帧：顺序 read，最可靠（不依赖 AVI 索引）
    auto nextFrame = [&]() -> bool {
        if (useImage) return false;
        cv::Mat next;
        if (cap.read(next) && !next.empty()) { frame = next; ++frameNo; return true; }
        return false;
    };
    // 上一帧：先试随机 seek，失败则从头顺序读（AVI 无索引时的兜底，稍慢但准）
    auto prevFrame = [&]() -> bool {
        if (useImage || frameNo <= 0) return false;
        int target = frameNo - 1;
        cap.set(cv::CAP_PROP_POS_FRAMES, target);
        cv::Mat prev;
        bool ok = cap.read(prev);
        double pos = cap.get(cv::CAP_PROP_POS_FRAMES);
        if (ok && !prev.empty() && (pos < 0 || std::fabs(pos - target) <= 1.0)) {
            frame = prev; frameNo = target; return true;
        }
        cap.release();
        cap.open(media);
        for (int i = 0; i < target; ++i) {
            if (!cap.read(frame) || frame.empty()) return false;
        }
        frameNo = target;
        return true;
    };

    cv::namedWindow("HSV Tuner", cv::WINDOW_NORMAL);
    cv::namedWindow("Effect: original | mask | overlay", cv::WINDOW_NORMAL);
    cv::namedWindow("HSV channels: H | S | V", cv::WINDOW_NORMAL);
    cv::createTrackbar("color:0Red 1Blue",  "HSV Tuner", &g_color,     1, onTrack);
    cv::createTrackbar("Red H low1",        "HSV Tuner", &g_lowH1,   180, onTrack);
    cv::createTrackbar("Red H high1",       "HSV Tuner", &g_highH1,  180, onTrack);
    cv::createTrackbar("Red H low2",        "HSV Tuner", &g_lowH2,   180, onTrack);
    cv::createTrackbar("Red H high2",       "HSV Tuner", &g_highH2,  180, onTrack);
    cv::createTrackbar("Blue H low",        "HSV Tuner", &g_blueLowH,180, onTrack);
    cv::createTrackbar("Blue H high",       "HSV Tuner", &g_blueHighH,180, onTrack);
    cv::createTrackbar("S min (saturation)","HSV Tuner", &g_minS,    255, onTrack);
    cv::createTrackbar("V min (brightness)","HSV Tuner", &g_minV,    255, onTrack);
    cv::createTrackbar("morph: 1on 0off",   "HSV Tuner", &g_openK,     1, onTrack);

    std::cout << "\n===== HSV 调参用法 =====\n"
              << "  拖滑块     : 实时改变阈值，看掩膜变化\n"
              << "  d / n      : 下一帧     a / p : 上一帧\n"
              << "  空格       : 自动播放 / 暂停（约30帧每秒）\n"
              << "  s          : 打印数值（抄回 ArmorDetector.h 的 ArmorParam）\n"
              << "  r          : 恢复默认值\n"
              << "  ESC / q    : 退出\n"
              << "  橙色灯条抓不到 → 重点调大 Red H high1\n"
              << "========================\n\n" << std::flush;

    bool playing = false;   // 空格键控制的自动播放状态
    while (true) {
        cv::Mat mask = buildMask(hsv);
        int white = cv::countNonZero(mask);
        double pct = 100.0 * white / (mask.rows * mask.cols);

        cv::Mat maskBgr;
        cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
        cv::Mat overlay = frame.clone();
        cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
        cv::Mat roi;
        overlay.copyTo(roi, mask);
        cv::addWeighted(roi, 0.5, green, 0.5, 0.0, roi);
        roi.copyTo(overlay, mask);

        std::string tag = (g_color == 0 ? "RED" : "BLUE");
        cv::putText(maskBgr, "white: " + std::to_string(white) + " (" +
                    std::to_string(pct).substr(0, 5) + "%)",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 255), 2);
        cv::putText(overlay, tag, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX,
                    1.2, cv::Scalar(0, 255, 255), 3);

        // 当前帧号标在画面上，翻帧时一眼知道看到第几帧
        std::string frameTag = useImage ? "IMAGE"
                             : ("frame " + std::to_string(frameNo)
                                + (totalFrames > 0
                                   ? "/" + std::to_string(totalFrames) : ""));
        cv::putText(maskBgr, frameTag, cv::Point(10, 62), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(overlay, frameTag, cv::Point(10, 72), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 255, 255), 2);

        cv::Mat compare, compare2;
        cv::hconcat(frame, maskBgr, compare);
        cv::hconcat(compare, overlay, compare2);
        cv::resize(compare2, compare2, cv::Size(), 0.5, 0.5);

        cv::Mat ch[3];
        cv::split(hsv, ch);
        cv::Mat hC = channelToColor(ch[0], cv::COLORMAP_HSV);
        cv::Mat sC = channelToColor(ch[1], cv::COLORMAP_VIRIDIS);
        cv::Mat vC = channelToColor(ch[2], cv::COLORMAP_BONE);
        cv::Mat hsvView, hsvView2;
        cv::hconcat(hC, sC, hsvView);
        cv::hconcat(hsvView, vC, hsvView2);
        cv::resize(hsvView2, hsvView2, cv::Size(), 0.5, 0.5);

        cv::imshow("Effect: original | mask | overlay", compare2);
        cv::imshow("HSV channels: H | S | V", hsvView2);
        cv::imshow("HSV Tuner", mask);

        int key = cv::waitKey(30) & 0xFF;
        if (key == 27 || key == 'q') break;
        else if (key == 'd' || key == 'n') {
            if (!nextFrame()) std::cout << "[hsv] 已经是最后一帧\n";
            else refreshHsv();
        } else if (key == 'a' || key == 'p') {
            if (!prevFrame()) std::cout << "[hsv] 已经是第一帧\n";
            else refreshHsv();
        } else if (key == ' ') {
            playing = !playing;
            std::cout << (playing ? "[hsv] 开始自动播放\n" : "[hsv] 暂停\n");
        } else if (key == 'r') {
            // 【2026-11-24 修复】从 ArmorParam 恢复，不再写死旧值
            g_lowH1 = g_tunerParam.redLowH1;  g_highH1 = g_tunerParam.redHighH1;
            g_lowH2 = g_tunerParam.redLowH2;  g_highH2 = g_tunerParam.redHighH2;
            g_blueLowH = g_tunerParam.blueLowH; g_blueHighH = g_tunerParam.blueHighH;
            g_minS = g_tunerParam.minS; g_minV = g_tunerParam.minV;
            g_color = 0; g_openK = 1;
        } else if (key == 's') {
            // 先拼成一个字符串，然后同时：① 打印控制台 ② 写入 txt 文件
            // 【2026-11-24 修复】旧代码只 cout，CLion 里 stdout 是全缓冲，
            // '\n' 不会立即刷新，数值其实打印了但卡在缓冲区看不到。
            // 现在用 std::endl（强制刷新）+ 落盘到文件，双保险。
            std::ostringstream oss;
            oss << "========== COPY INTO ArmorParam ==========\n"
                << "    int redLowH1  = " << g_lowH1   << ";\n"
                << "    int redHighH1 = " << g_highH1  << ";\n"
                << "    int redLowH2  = " << g_lowH2   << ";\n"
                << "    int redHighH2 = " << g_highH2  << ";\n"
                << "    int blueLowH  = " << g_blueLowH  << ";\n"
                << "    int blueHighH = " << g_blueHighH << ";\n"
                << "    int minS      = " << g_minS << ";\n"
                << "    int minV      = " << g_minV << ";\n"
                << "==========================================\n"
                << "  当前帧: " << frameNo << "\n";

            std::cout << "\n" << oss.str() << std::flush;

            // 写到 exe 同目录的 hsv_values.txt，打开这个文件抄数值也行
            std::string txtPath = (std::filesystem::path(argv0Dir) / "hsv_values.txt").string();
            std::ofstream ofs(txtPath, std::ios::trunc);
            if (ofs) {
                ofs << oss.str();
                ofs.close();
                std::cout << "[hsv] 数值已保存到: " << txtPath << std::endl;
            } else {
                std::cerr << "[hsv] 写文件失败（不影响控制台输出）: " << txtPath << std::endl;
            }

            // 画面上也给个即时反馈，确认"按 s 生效了"
            cv::putText(compare2, "VALUES PRINTED -> hsv_values.txt",
                        cv::Point(10, compare2.rows - 15), cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(0, 255, 0), 2);
        }

        // 自动播放：每轮循环读一帧；到末尾自动停
        if (playing && !useImage) {
            if (nextFrame()) refreshHsv();
            else { playing = false; std::cout << "[hsv] 到视频末尾，停止播放\n"; }
        }
    }
    if (cap.isOpened()) cap.release();
    cv::destroyAllWindows();
    return 0;
}

// ============================================================================
// ⑤ 主入口：打印用法 + 解析 --mode + 分发
// ============================================================================
static void printUsage(const char* prog) {
    std::cout <<
        "装甲板视觉系统 —— 统一入口\n"
        "用法: " << prog << " [--mode image|video|hsv] [选项]\n"
        "\n"
        "模式（不写 --mode 时默认 video）:\n"
        "  image   单张图片检测，结果存 png\n"
        "  video   视频逐帧检测，结果存 avi（考核主用，默认）\n"
        "  hsv     HSV 阈值调试器，拖滑块找颜色阈值\n"
        "\n"
        "image 选项:\n"
        "  --image <图>     输入图片（不给则用内置合成图）\n"
        "  --out <文件>     输出图片，默认 armor_result.png\n"
        "  --camera <yaml>  相机标定文件\n"
        "  --no-show        不弹窗口（批量跑时用）\n"
        "\n"
        "video 选项:\n"
        "  --video <视频>   输入视频（默认 C:/carrie/video.avi，不在才用合成视频）\n"
        "  --out <文件>     输出视频，默认 armor_result.avi\n"
        "  --camera <yaml>  相机标定文件\n"
        "  --make-test      只生成测试视频，不检测\n"
        "  --dump-frame N   把第 N 帧另存为 frame_dump.png\n"
        "  --no-show        不弹窗口\n"
        "  --no-write       只弹窗口看实时效果、不写结果视频（少一次大图编码，明显减少卡顿）\n"
        "  --max-frames N   只处理前 N 帧（快速自测用）\n"
        "  --temporal       开启时序多数投票修断灯条歪框（默认关闭）\n"
        "  --fast           极速模式 scale=0.75+关模糊（约1.6倍速，测距偏差最大约0.4m，默认关闭）\n"
        "  --imu <csv>      IMU 姿态文件（默认 exe 同目录 imu_data/imu_frame_imu.csv）\n"
        "  --handeye <yaml> 手眼外参（默认 exe 同目录 imu_data/calibration_handeye.yaml）\n"
        "  --no-imu         关闭 IMU（默认自动加载，缺文件则跳过不影响检测）\n"
        "\n"
        "hsv 选项:\n"
        "  --video <视频>   要调试的视频，默认 C:/carrie/video.avi\n"
        "  --image <图>     要调试的图片（无逐帧功能）\n"
        "  --frame N        起始帧号，默认 900\n"
        "  窗口内按键: d/a 下一帧/上一帧, 空格 播放/暂停, s 打印数值, r 恢复默认\n"
        "\n"
        "示例:\n"
        "  " << prog << "                                   " << "视频检测（默认）\n"
        "  " << prog << " --mode image --image armor.png      " << "图片检测\n"
        "  " << prog << " --mode video --video video.avi       " << "考核视频检测\n"
        "  " << prog << " --mode hsv --frame 900               " << "调 HSV 阈值\n"
        "  " << prog << " --help                               " << "看这段说明\n";
}

int main(int argc, char** argv) {
    // 先扫一遍参数：处理 --help / --mode，并把这两种参数从"给模式用的参数"里摘掉
    // 【2026-11-24】HSV_TUNER_STANDALONE 宏：给独立 hsv_tuner 程序用，
    // 让它零参数启动就直接进 HSV 调试器（CLion 里选它点运行即弹窗，不用填参数）。
#ifdef HSV_TUNER_STANDALONE
    std::string mode = "hsv";     // 独立调参程序：默认 hsv
#else
    std::string mode = "video";   // 统一入口默认视频模式
#endif
    std::vector<char*> rest;
    rest.push_back(argv[0]);      // 保留程序名（模式函数按 argc/argv 习惯用）
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }

    int restArgc = static_cast<int>(rest.size());
    char** restArgv = rest.data();

    if (mode == "image") return runImage(restArgc, restArgv);
    if (mode == "video") return runVideo(restArgc, restArgv);
    if (mode == "hsv")   return runHsv(restArgc, restArgv);

    std::cerr << "未知模式: " << mode << "（可选 image / video / hsv），用 --help 看用法\n";
    return 1;
}
