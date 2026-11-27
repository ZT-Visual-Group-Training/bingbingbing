// 自瞄系统：数据 IO + 传统颜色灯条检测 + 打框可视化
//   按 video/frames.csv 的行顺序读取 video.avi（时间基准只用 csv，不用 AVI 的 fps），
//   读取 calibration/ 下的三个标定 yaml，检测红/蓝灯条并组出装甲板，
//   输出带框、中心点和文字叠加的 mp4，并打印首末 elapsed_s 与检测统计。

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "detect/color_detector.hpp"
#include "geometry/pnp.hpp"
#include "io/calib.hpp"
#include "io/dataset.hpp"
#include "viz/overlay.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path data;
  fs::path out;
  double out_fps = 0.0;  // 0 表示使用 frames.csv 推算的标称帧率
  int max_frames = -1;
  bool show = false;
  rm::EnemyColor enemy = rm::EnemyColor::kRed;
  int threshold = 50;
};

void printUsage(const char* exe) {
  std::cout << "用法: " << exe << " <数据目录> [选项]\n"
            << "  --out <path.mp4>   输出视频路径（默认 out/<数据名>_overlay.mp4）\n"
            << "  --fps <f>          输出视频帧率（默认按 frames.csv 推算）\n"
            << "  --max-frames <n>   只处理前 n 帧\n"
            << "  --enemy <red|blue> 目标颜色（默认 red）\n"
            << "  --threshold <n>    颜色通道差值阈值（默认 50）\n"
            << "  --show             同步窗口显示（需要显示环境）\n";
}

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto needValue = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " 缺少参数值");
      return argv[++i];
    };

    if (a == "-h" || a == "--help") {
      printUsage(argv[0]);
      std::exit(0);
    } else if (a == "--out") {
      args.out = needValue("--out");
    } else if (a == "--fps") {
      args.out_fps = std::stod(needValue("--fps"));
    } else if (a == "--max-frames") {
      args.max_frames = std::stoi(needValue("--max-frames"));
    } else if (a == "--enemy") {
      const std::string v = needValue("--enemy");
      if (v == "red") {
        args.enemy = rm::EnemyColor::kRed;
      } else if (v == "blue") {
        args.enemy = rm::EnemyColor::kBlue;
      } else {
        throw std::runtime_error("--enemy 只支持 red / blue");
      }
    } else if (a == "--threshold") {
      args.threshold = std::stoi(needValue("--threshold"));
    } else if (a == "--show") {
      args.show = true;
    } else if (!a.empty() && a[0] == '-') {
      throw std::runtime_error("未知选项: " + a);
    } else if (args.data.empty()) {
      args.data = a;
    } else {
      throw std::runtime_error("多余参数: " + a);
    }
  }
  if (args.data.empty()) {
    printUsage(argv[0]);
    throw std::runtime_error("必须指定数据目录");
  }
  return args;
}

std::string fmt(double v, int precision) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision) << v;
  return os.str();
}

std::string vecText(const cv::Mat& m) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(4) << '[';
  for (int i = 0; i < static_cast<int>(m.total()); ++i) {
    if (i) os << ", ";
    os << m.at<double>(static_cast<int>(i / m.cols), i % m.cols);
  }
  os << ']';
  return os.str();
}

std::string colorName(rm::EnemyColor color) {
  return color == rm::EnemyColor::kRed ? "red" : "blue";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);

    rm::Dataset ds = rm::Dataset::open(args.data);
    const rm::Calibration calib = rm::Calibration::load(args.data / "calibration");

    if (ds.frameSize().width > 0 && calib.image_width > 0 &&
        (ds.frameSize().width != calib.image_width || ds.frameSize().height != calib.image_height)) {
      std::cerr << "[warn] 视频分辨率 " << ds.frameSize().width << 'x' << ds.frameSize().height
                << " 与标定分辨率 " << calib.image_width << 'x' << calib.image_height
                << " 不一致\n";
    }

    fs::path out = args.out;
    if (out.empty()) out = fs::path("out") / (ds.name() + "_overlay.mp4");
    if (out.has_parent_path()) fs::create_directories(out.parent_path());

    const double out_fps = args.out_fps > 0.0 ? args.out_fps : ds.nominalFps();
    if (out_fps <= 0.0) throw std::runtime_error("无法确定输出帧率，请用 --fps 指定");

    const auto& stamps = ds.stamps();
    const auto& first = stamps.front();
    const auto& last = stamps.back();

    std::cout << "dataset   : " << ds.root().string() << "\n"
              << "frames    : csv=" << ds.size() << "  video=" << ds.videoFrameCount() << "\n"
              << "time      : first elapsed_s=" << fmt(first.elapsed_s, 6)
              << "  last elapsed_s=" << fmt(last.elapsed_s, 6)
              << "  span=" << fmt(last.elapsed_s - first.elapsed_s, 6) << " s\n"
              << "nominal   : " << fmt(ds.nominalFps(), 3) << " fps (来自 frames.csv)\n"
              << "image     : " << ds.frameSize().width << 'x' << ds.frameSize().height << "\n"
              << "calib K   : fx=" << fmt(calib.camera_matrix.at<double>(0, 0), 3)
              << " fy=" << fmt(calib.camera_matrix.at<double>(1, 1), 3)
              << " cx=" << fmt(calib.camera_matrix.at<double>(0, 2), 3)
              << " cy=" << fmt(calib.camera_matrix.at<double>(1, 2), 3) << "\n"
              << "calib D   : " << vecText(calib.distort_coeffs) << "\n"
              << "handeye   : " << (calib.hasHandeye() ? "yes" : "no");
    if (calib.hasHandeye()) std::cout << "  t_camera2gimbal(m)=" << vecText(calib.t_camera2gimbal);
    std::cout << "\n";

    if (ds.videoFrameCount() != 0 && ds.videoFrameCount() != ds.size()) {
      std::cerr << "[warn] video 容器帧数(" << ds.videoFrameCount()
                << ") != frames.csv 行数(" << ds.size() << ")，以 csv 为准\n";
    }

    rm::ColorDetector::Params detector_params;
    detector_params.color = args.enemy;
    detector_params.threshold = args.threshold;
    rm::ColorDetector detector(detector_params);

    std::cout << "detector  : color(" << colorName(args.enemy) << ")  threshold="
              << args.threshold << "\n"
              << "pnp       : IPPE + LM, 装甲板 135x56 mm, 回投误差上限 8 px\n";

    const int limit =
        args.max_frames > 0 ? std::min<int>(args.max_frames, static_cast<int>(ds.size()))
                            : static_cast<int>(ds.size());

    cv::VideoWriter writer;
    cv::Mat frame;
    rm::FrameStamp stamp;
    int written = 0;
    double prev_elapsed = first.elapsed_s;
    int64_t prev_mono = first.monotonic_ns;
    int frames_with_target = 0;
    long long total_armors = 0;
    double detect_ms_sum = 0.0;
    double detect_ms_max = 0.0;
    double pnp_ms_sum = 0.0;
    double pnp_ms_max = 0.0;
    double reproj_sum = 0.0;
    long long reproj_count = 0;
    double z_min = 0.0;
    double z_max = 0.0;
    double z_sum = 0.0;

    while (written < limit && ds.next(frame, stamp)) {
      if (!writer.isOpened()) {
        // 本数据约 166 fps，mp4v 的 timebase 上限 65535 放不下，改用 H.264
        writer.open(out.string(), cv::VideoWriter::fourcc('a', 'v', 'c', '1'), out_fps,
                    frame.size());
        if (!writer.isOpened()) {
          throw std::runtime_error("无法创建输出视频: " + out.string() +
                                   "（可用 --fps 降低输出帧率后重试）");
        }
      }

      const double dt_ms = (stamp.monotonic_ns - prev_mono) / 1e6;
      const double inst_fps = dt_ms > 0.0 ? 1000.0 / dt_ms : 0.0;
      prev_elapsed = stamp.elapsed_s;
      prev_mono = stamp.monotonic_ns;

      const auto t0 = std::chrono::steady_clock::now();
      const std::vector<rm::Armor> armors = detector.detect(frame);
      const double detect_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
              .count();
      detect_ms_sum += detect_ms;
      detect_ms_max = std::max(detect_ms_max, detect_ms);

      const auto t_pnp = std::chrono::steady_clock::now();
      std::vector<rm::ArmorObservation> observations;
      observations.reserve(armors.size());
      for (const auto& armor : armors) {
        rm::ArmorPose pose =
            rm::solveArmorPose(armor, calib.camera_matrix, calib.distort_coeffs, 8.0);
        if (pose.valid) observations.push_back({armor, pose});
      }
      const double pnp_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pnp)
              .count();
      pnp_ms_sum += pnp_ms;
      pnp_ms_max = std::max(pnp_ms_max, pnp_ms);

      const rm::ArmorObservation* nearest = nullptr;
      for (const auto& observation : observations) {
        if (nearest == nullptr || observation.pose.distance_m < nearest->pose.distance_m) {
          nearest = &observation;
        }
      }

      if (!observations.empty()) {
        ++frames_with_target;
        total_armors += static_cast<long long>(observations.size());
        for (const auto& observation : observations) {
          const rm::ArmorPose& pose = observation.pose;
          reproj_sum += pose.reproj_error_px;
          ++reproj_count;
          const double z = pose.tvec[2];
          if (reproj_count == 1 || z < z_min) z_min = z;
          if (reproj_count == 1 || z > z_max) z_max = z;
          z_sum += z;
        }
      }

      const double s = std::max(0.5, frame.cols / 1440.0);
      std::string pose_line = "pose: none";
      if (nearest != nullptr) {
        const rm::ArmorPose& pose = nearest->pose;
        pose_line = "nearest x=" + fmt(pose.tvec[0], 2) + " y=" + fmt(pose.tvec[1], 2) +
                    " z=" + fmt(pose.tvec[2], 2) + " m  yaw=" + fmt(pose.yaw_deg, 1) +
                    " pitch=" + fmt(pose.pitch_deg, 1) + " deg  err=" +
                    fmt(pose.reproj_error_px, 2) + " px";
      }
      const std::vector<std::string> lines = {
          "RM AUTO-AIM  step3 - armor pose  [" + ds.name() + "]",
          "frame " + std::to_string(written) + " / " + std::to_string(ds.size()) +
              "    cam_id " + std::to_string(stamp.camera_frame_id),
          "t = " + fmt(stamp.elapsed_s, 6) + " s    dt = " + fmt(dt_ms, 2) +
              " ms    inst " + fmt(inst_fps, 1) + " fps",
          "armors " + std::to_string(observations.size()) + "    bars " +
              std::to_string(detector.lastBarCount()) + "    detect " + fmt(detect_ms, 2) +
              " ms    pnp " + fmt(pnp_ms, 2) + " ms",
          pose_line,
      };
      rm::drawHud(frame, lines, s);
      rm::drawArmors(frame, observations);

      writer.write(frame);
      ++written;

      if (args.show) {
        cv::imshow("rm_autoaim", frame);
        if (cv::waitKey(1) == 27) break;
      }
      if (written % 240 == 0) {
        std::cerr << "\r处理中 " << written << " / " << limit << std::flush;
      }
    }
    if (written >= 240) std::cerr << "\n";

    writer.release();
    if (args.show) cv::destroyAllWindows();

    if (written == 0) throw std::runtime_error("没有写出任何帧");

    const double target_frame_pct =
        written > 0 ? 100.0 * static_cast<double>(frames_with_target) / written : 0.0;
    const double armors_per_target_frame =
        frames_with_target > 0 ? static_cast<double>(total_armors) / frames_with_target : 0.0;

    std::cout << "output    : " << out.string() << "  (" << written << " 帧 @ " << fmt(out_fps, 2)
              << " fps)\n"
              << "armors    : 检出帧 " << frames_with_target << " / " << written << " ("
              << fmt(target_frame_pct, 1) << "%)，装甲板共 " << total_armors << " 个（检出帧均 "
              << fmt(armors_per_target_frame, 2) << "）\n"
              << "detect    : avg " << fmt(detect_ms_sum / written, 2) << " ms，max "
              << fmt(detect_ms_max, 2) << " ms\n"
              << "pnp       : avg " << fmt(pnp_ms_sum / written, 2) << " ms，max "
              << fmt(pnp_ms_max, 2) << " ms";
    if (reproj_count > 0) {
      std::cout << "，回投 avg " << fmt(reproj_sum / static_cast<double>(reproj_count), 3)
                << " px\n"
                << "pose z    : " << fmt(z_min, 2) << " ~ " << fmt(z_max, 2) << " m，mean "
                << fmt(z_sum / static_cast<double>(reproj_count), 2) << " m\n";
    } else {
      std::cout << "\n";
    }
    std::cout << "last frame: elapsed_s=" << fmt(prev_elapsed, 6) << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
}
