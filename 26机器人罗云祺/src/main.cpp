#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "rm/calibration.hpp"
#include "rm/config.hpp"
#include "rm/dataset.hpp"
#include "rm/detector.hpp"
#include "rm/imu.hpp"
#include "rm/profiler.hpp"
#include "rm/solver.hpp"
#include "rm/tracker.hpp"
#include "rm/visualizer.hpp"

namespace fs = std::filesystem;

namespace
{

struct Args
{
  std::string config = "config/autoaim.yaml";
  std::string dataset;
  std::string out;
  int max_frames = -1;  // -1 表示全跑
  int dump_frame = -1;  // -1 表示不存单帧
  bool headless = false;
};

// 打印用法
void usage(const char * exe)
{
  std::cout << "usage: " << exe << " [options]\n"
            << "  --config PATH     配置文件（默认 config/autoaim.yaml)\n"
            << "  --dataset PATH    覆盖 dataset_root\n"
            << "  --out PATH        覆盖 output.dir\n"
            << "  --max-frames N    只处理前 N 帧（调试用）\n"
            << "  --dump N          把第 N 帧的标注图存成 output/frame_N.jpg\n"
            << "  --headless        不开窗口\n";
}

Args parseArgs(int argc, char ** argv)
{
  Args args;
  // argv[0] 是程序名
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    const auto need = [&](const char * name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };

    if (a == "--config") {
      args.config = need("--config");
    } else if (a == "--dataset") {
      args.dataset = need("--dataset");
    } else if (a == "--out") {
      args.out = need("--out");
    } else if (a == "--max-frames") {
      args.max_frames = std::stoi(need("--max-frames"));
    } else if (a == "--dump") {
      args.dump_frame = std::stoi(need("--dump"));
    } else if (a == "--headless") {
      args.headless = true;  // 开关
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      std::exit(0);  // 打印完直接退出
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  return args;
}

std::string joinPath(const std::string & a, const std::string & b)
{
  if (a.empty()) return b;  // "" + "x" -> "x"
  if (b.empty()) return a;  // "x" + "" -> "x"
  const char last = a.back();
  if (last == '/' || last == '\\') return a + b;  // "data/" + "x" -> "data/x"
  return a + "/" + b;  // "data" + "x" -> "data/x"
}

}

int main(int argc, char ** argv)
{
  try {
    const Args args = parseArgs(argc, argv);

    // 读配置
    rm::Config cfg = rm::Config::load(args.config);
    if (!args.dataset.empty()) cfg.set("dataset_root", args.dataset);  // --dataset 覆盖
    if (!args.out.empty()) cfg.set("output.dir", args.out);  // --out 覆盖

    const std::string root = cfg.getString("dataset_root", "data");  // 数据根目录
    // 配置里 display=1 且命令行没加 --headless，才开窗口
    const bool display = cfg.getBool("output.display", true) && !args.headless;

    // ---------- 标定 ----------
    const auto calib_path = joinPath(root, cfg.getString("calibration", "calibration/calibration.yaml"));
    const rm::Calibration calib = rm::Calibration::load(calib_path);
    // 打印 fx/fy/cx/cy 和畸变, K 是 3x3 的 double 矩阵
    std::cout << "[calib] " << calib_path << "\n"
              << "        fx=" << calib.K.at<double>(0, 0) << " fy=" << calib.K.at<double>(1, 1)
              << " cx=" << calib.K.at<double>(0, 2) << " cy=" << calib.K.at<double>(1, 2)
              << "  [" << calib.width << "x" << calib.height << "]\n"
              << "        D=" << calib.D << "\n";

    // ---------- 数据 ----------
    rm::Dataset dataset;
    std::string err;  // 接错误信息
    if (!dataset.open(
          joinPath(root, cfg.getString("video", "video/video.avi")),
          joinPath(root, cfg.getString("frames_csv", "video/frames.csv")), &err)) {
      throw std::runtime_error(err);
    }
    std::cout << "[data ] " << dataset.size() << " frames, " << dataset.fps() << " fps, "
              << dataset.imageSize() << "\n";

    // IMU 读进来验证链路
    rm::Imu imu;
    if (cfg.has("frame_imu_csv")) {
      if (!imu.load(joinPath(root, cfg.getString("frame_imu_csv")), &err)) {
        std::cout << "[imu  ] skip: " << err << "\n";
      } else {
        // 取第一帧的时间戳，查该时刻的四元数
        std::array<double, 4> q{};
        const bool ok = imu.quatAt(dataset.meta().front().monotonic_ns, q);
        std::cout << "[imu  ] " << imu.size() << " samples"
                  << (ok ? ", q0 = [" + std::to_string(q[0]) + ", " + std::to_string(q[1]) + ", " +
                             std::to_string(q[2]) + ", " + std::to_string(q[3]) + "]"
                         : "")
                  << "\n";
      }
    }

    // ---------- 输出 ----------
    const std::string out_dir = cfg.getString("output.dir", "output");
    fs::create_directories(out_dir);  // 不存在就创建，已存在不报错

    const bool save_video = cfg.getBool("output.save_video", true);
    const bool save_csv = cfg.getBool("output.save_csv", true);
    const double out_fps = cfg.getDouble("output.video_fps", 30.0);
    const int stride = std::max(1, static_cast<int>(std::lround(dataset.fps() / std::max(1.0, out_fps))));

    cv::VideoWriter writer;
    if (save_video) {
      writer.open(
        joinPath(out_dir, "demo.mp4"), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), out_fps,
        dataset.imageSize());
      if (!writer.isOpened()) std::cerr << "[warn ] cannot open demo.mp4 for writing\n";
    }

    std::ofstream csv;
    if (save_csv) {
      csv.open(joinPath(out_dir, "result.csv"));
      csv << "frame,timestamp_ns,elapsed_s,valid,color,number,x_m,y_m,z_m,distance_m,reproj_px,"
             "frame_ms\n";
    }

    // ---------- 主循环 ----------
    rm::Detector detector(cfg);  // 读 detector.* 参数
    rm::Solver solver(calib, cfg);  // 读 armor.* 尺寸
    rm::Tracker tracker;
    rm::Visualizer visualizer(cfg);
    rm::Profiler profiler;  // 统计每帧 detect+solve 耗时

    const auto wall_start = std::chrono::steady_clock::now();  // 总墙钟时间
    int processed = 0;  // 实际处理帧数

    rm::Frame frame;
    while (dataset.next(frame)) {
      if (args.max_frames > 0 && processed >= args.max_frames) break;  // --max-frames 限制

      const auto t0 = std::chrono::steady_clock::now();
      const auto armors = detector.detect(frame.image);  // 2D 检测

      std::vector<rm::Pose> poses;
      poses.reserve(armors.size());
      for (const auto & armor : armors) poses.push_back(solver.solve(armor));

      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      profiler.add(ms);

      cv::Mat canvas = frame.image.clone();
      visualizer.draw(canvas, armors, poses, ms, frame.index, frame.elapsed_s);

      if (args.dump_frame >= 0 && frame.index == args.dump_frame) {
        cv::imwrite(joinPath(out_dir, "frame_" + std::to_string(frame.index) + ".jpg"), canvas);
      }

      if (writer.isOpened() && frame.index % stride == 0) writer.write(canvas);

      if (csv.is_open()) {
        const auto row = [&](const rm::Armor * a, const rm::Pose * p) {
          csv << frame.index << ',' << frame.timestamp_ns << ',' << frame.elapsed_s << ','
              << ((p != nullptr && p->valid) ? 1 : 0) << ','
              << (a != nullptr ? rm::colorName(a->color) : "") << ','
              << (a != nullptr ? std::to_string(a->number) : "") << ',';
          if (p != nullptr && p->valid) {
            csv << p->tvec[0] << ',' << p->tvec[1] << ',' << p->tvec[2] << ',' << cv::norm(p->tvec)
                << ',' << p->reproj_error << ',';
          } else {
            csv << ",,,,,";
          }
          csv << ms << '\n';
        };

        if (armors.empty()) {
          row(nullptr, nullptr);
        } else {
          for (size_t i = 0; i < armors.size(); ++i) row(&armors[i], &poses[i]);
        }
      }

      if (display) {
        cv::Mat show;
        cv::resize(canvas, show, cv::Size(), 0.5, 0.5);
        cv::imshow("rm_autoaim", show);
        const int key = cv::waitKey(1);
        if (key == 27) break;
      }

      ++processed;
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

    // 收尾
    if (writer.isOpened()) writer.release();
    if (csv.is_open()) csv.close();
    if (display) cv::destroyAllWindows();

    // 打印总帧数、墙钟时间、平均帧率和detect+solve 的 mean/p95/max
    std::cout << "[done ] frames=" << processed << "  wall=" << wall_s << "s  avg="
              << (wall_s > 0.0 ? processed / wall_s : 0.0) << " fps\n";
    profiler.print("detect+solve");
    std::cout << "[out  ] " << out_dir << "\n";
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
}
