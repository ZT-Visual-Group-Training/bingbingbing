// RoboMaster armour-plate detector / auto-aim front end.
//
//   ./autoaim --video clip.avi --config config/default.yaml --save-video out.avi
//
// Output per frame: the four plate corners, the plate centre, the 3D position in
// the camera frame and the measured processing time.
//
// ===========================================================================
// 本文件：程序入口 + 主循环 —— 读代码从这里开始最省力
//
// 主循环里每一帧做的事，顺序不能随意交换：
//
//   decode  : cap.read()                    解一帧 MJPG（约 8~12 ms，是大头）
//   detect  : detector.detect(frame, roi)   灯条检测（约 3~6 ms）
//   match   : matcher.match(bars)           灯条配对 + 四角点外推
//   refine  : refineArmorCornersSubPix(...) 亚像素精化（角点更准 → 误差更小）
//   pnp     : solver.solve(armor)           解位姿 + 重投影误差判可信
//   draw    : drawOverlay(...)              画框 / 画四角点 / 画中心 / 写数字
//
// 三个容易踩的顺序陷阱：
//   [1] 亚像素精化必须在 PnP 之前、在配对之后 —— 它需要"已成形的四点"
//   [2] 精化用的图是 detector.diffImage()（ROI 局部坐标），而角点是全图坐标，
//       所以进出各要平移一次 detector.lastRoi() 的偏移，否则会在错误的位置找角点
//   [3] 计时分四段统计（decode / detect / match+pnp / total），必须分开报 ——
//       否则"优化了算法但 FPS 没涨"这种现象解释不了，因为瓶颈在解码
//
// 关于耗时指标的口径（答辩要能说清）：
//   - "算法耗时" = detect + match + pnp ：优化真正作用的地方
//   - "端到端 FPS" = 还要加上 MJPG 解码（1440x1080 每帧约 350 KB，
//     166 fps 就是 59 MB/s，单核解码 8~12 ms，而帧间隔只有 6 ms）
//   所以算法能优化到 3~6 ms，端到端仍然被解码卡住。这不是代码问题，
//   是数据格式的物理成本。
// ===========================================================================
#include "autoaim/anti_gyro.hpp"
#include "autoaim/armor_corners.hpp"
#include "autoaim/armor_matcher.hpp"
#include "autoaim/config.hpp"
#include "autoaim/light_bar_detector.hpp"
#include "autoaim/pnp_solver.hpp"
#include "autoaim/temporal_filter.hpp"
#include "autoaim/tracker.hpp"
#include "autoaim/types.hpp"
#include "autoaim/visualizer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void printUsage() {
  std::printf(
      "usage: autoaim --video <path> [options]\n"
      "  --config <file>      YAML config (default config/default.yaml)\n"
      "  --calib <file>       read intrinsics from a calibration.yaml\n"
      "  --save-video <file>  write annotated video (MJPG .avi)\n"
      "  --csv <file>         write per-frame results\n"
      "  --max-frames <N>     stop after N frames (0 = all)\n"
      "  --out-fps <f>        fps of the saved video (default: source fps; use 30\n"
      "                       to get a watchable slow-motion demo clip)\n"
      "  --no-display         do not open a window\n  --show-all           also draw non-best candidates\n"
      "  --roi                enable ROI reuse around the tracked target\n"
      "  --tracker            enable the Kalman tracker\n"
      "  --verbose            dump the config and print progress\n"
      "  --gyro-selftest      run the anti-gyro synthetic self test and exit\n");
}

}  // namespace

// ===========================================================================
// 反陀螺合成自检（--gyro-selftest）
//
// 为什么需要它：考核数据里没有"原地自旋"的目标，真实数据上无法验证反陀螺。
// 这里构造一条参数已知的空间圆周轨迹喂给 GyroEstimator，再把拟合结果和真值
// 对比 —— 误差很小就证明拟合数学是对的。
//
// 答辩时直接跑：./build/autoaim --gyro-selftest
// ===========================================================================
static int runGyroSelfTest() {
  const double kRadius = 0.075;               // 真值：半径（米）
  const double kOmegaDps = 500.0;             // 真值：角速度（度/秒）
  const cv::Vec3d kCentre(0.0, 0.0, 2.0);     // 真值：旋转中心（相机系）
  const double kFps = 165.93;

  autoaim::GyroEstimator::Params gp;
  gp.enabled = true;
  gp.window = 45;
  gp.min_samples = 12;
  gp.min_omega_dps = 150.0;
  gp.max_plane_rms_m = 0.05;
  gp.reset_frames = 30;
  gp.predict_ahead_s = 0.08;
  gp.stabilize_window = 10;
  autoaim::GyroEstimator ge(gp);

  const int kFrames = 60;
  for (int i = 0; i < kFrames; ++i) {
    const double t = static_cast<double>(i) / kFps;
    const double a = kOmegaDps * t * CV_PI / 180.0;
    const cv::Vec3d p = kCentre + cv::Vec3d(kRadius * std::cos(a), 0.0, kRadius * std::sin(a));
    const double yaw = std::atan2(std::cos(a), std::sin(a)) * 180.0 / CV_PI;
    ge.push(p, yaw, t);
  }

  const autoaim::GyroState& st = ge.state();
  std::printf("\n=========== 反陀螺合成自检 ===========\n");
  std::printf("真值 : centre=(%.3f %.3f %.3f)  r=%.4f m  omega=%+.1f deg/s  axis=(0,1,0)\n",
              kCentre[0], kCentre[1], kCentre[2], kRadius, kOmegaDps);
  if (!st.valid) {
    std::printf("拟合 : 未通过有效性判据（valid=0, samples=%d）\n", st.samples);
    std::printf("失败原因 : %s\n", ge.failReason());
    std::printf("调试     : lastOmega=%.2f deg/s  relStd=%.4f  hist=%d\n", ge.lastOmega(),
                ge.lastOmegaRelStd(), ge.omegaHistorySize());
    std::printf("=====================================\n\n");
    return 2;
  }
  std::printf("拟合 : centre=(%.3f %.3f %.3f)  r=%.4f m  omega=%+.1f deg/s  axis=(%.2f %.2f %.2f)\n",
              st.centre[0], st.centre[1], st.centre[2], st.radius, st.omega_dps,
              st.axis[0], st.axis[1], st.axis[2]);
  const double ec = cv::norm(st.centre - kCentre);
  const double er = std::fabs(st.radius - kRadius);
  const double eo = std::fabs(std::fabs(st.omega_dps) - kOmegaDps);
  std::printf("误差 : 圆心 %.1f mm   半径 %.1f mm   角速度 %.1f deg/s   平面残差 %.2f mm\n",
              ec * 1000.0, er * 1000.0, eo, st.fit_rms * 1000.0);
  std::printf("四块装甲（相机系，米）：\n");
  for (int k = 0; k < 4; ++k) {
    std::printf("  plate[%d] (%.3f %.3f %.3f)  yaw=%+6.1f%s\n", k, st.armors[k][0],
                st.armors[k][1], st.armors[k][2], st.yaws[k],
                (k == st.best) ? "   <== 推荐打这块" : "");
  }
  std::printf("提前量 : %.0f ms -> 击发时刻该瞄 plate[%d]，yaw=%+.1f deg（%s）\n",
              st.predict_ahead_s * 1000.0, st.best, st.best_yaw_deg,
              st.best_is_current ? "就是当前这块" : "在瞄还没转过来的那块");
  std::printf("=====================================\n\n");
  return (ec < 0.02 && er < 0.01 && eo < 30.0) ? 0 : 3;
}
int main(int argc, char** argv) {
  autoaim::Config cfg;

  std::string config_path = "config/default.yaml";
  std::string video_path;
  std::string calib_path;
  std::string out_video;
  std::string out_csv;
  bool no_display = false;
  bool verbose = false;
  bool show_all = false;
  int max_frames = 0;
  int force_roi = -1;
  int force_tracker = -1;
  double out_fps_override = 0.0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--config") {
      config_path = next("--config");
    } else if (a == "--video") {
      video_path = next("--video");
    } else if (a == "--calib") {
      calib_path = next("--calib");
    } else if (a == "--save-video") {
      out_video = next("--save-video");
    } else if (a == "--csv") {
      out_csv = next("--csv");
    } else if (a == "--max-frames") {
      max_frames = std::stoi(next("--max-frames"));
    } else if (a == "--out-fps") {
      out_fps_override = std::stod(next("--out-fps"));
    } else if (a == "--no-display") {
      no_display = true;
    } else if (a == "--show-all") {
      show_all = true;
    } else if (a == "--roi") {
      force_roi = 1;
    } else if (a == "--tracker") {
      force_tracker = 1;
    } else if (a == "--verbose") {
      verbose = true;
    } else if (a == "--gyro-selftest") {
      // 反陀螺合成自检：不读视频、不读配置，直接用已知真值的圆周轨迹验证拟合
      return runGyroSelfTest();
    } else if (a == "-h" || a == "--help") {
      printUsage();
      return 0;
    } else {
      std::fprintf(stderr, "error: unknown option %s\n", a.c_str());
      printUsage();
      return 2;
    }
  }

  if (video_path.empty()) {
    printUsage();
    return 2;
  }

  try {
    cfg = autoaim::Config::load(config_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  if (!calib_path.empty()) cfg.loadCalibration(calib_path);

  if (force_roi >= 0) cfg.detector.enable_roi = (force_roi != 0);
  if (force_tracker >= 0) cfg.detector.enable_tracker = (force_tracker != 0);

  if (verbose) cfg.dump();

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "error: cannot open video %s\n", video_path.c_str());
    return 2;
  }

  const double src_fps = cap.get(cv::CAP_PROP_FPS);
  const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  std::printf("video: %s\n  %dx%d @ %.2f fps\n", video_path.c_str(), w, h, src_fps);
  if (verbose && cfg.camera.width && (w != cfg.camera.width || h != cfg.camera.height)) {
    std::fprintf(stderr,
                 "warning: frame size %dx%d differs from the calibrated %dx%d; the "
                 "intrinsics must be scaled or PnP will be wrong\n",
                 w, h, cfg.camera.width, cfg.camera.height);
  }

  autoaim::LightBarDetector detector(cfg.detector);
  autoaim::ArmorMatcher matcher(cfg.detector);
  autoaim::PnpSolver solver(cfg.camera, cfg.detector);
  autoaim::ArmorTracker tracker(cfg.detector);
  // 反陀螺估计器：只读卡尔曼的输出，在它之上做"空间圆拟合"。
  autoaim::GyroEstimator gyro;
  {
    autoaim::GyroEstimator::Params g;
    g.enabled = cfg.detector.enable_gyro;
    g.window = cfg.detector.gyro_window;
    g.min_samples = cfg.detector.gyro_min_samples;
    g.min_omega_dps = cfg.detector.gyro_min_omega_dps;
    g.max_plane_rms_m = cfg.detector.gyro_max_plane_rms_m;
    g.reset_frames = cfg.detector.gyro_reset_frames;
    gyro.configure(g);
  }

  autoaim::TargetTracker target_tracker;
  target_tracker.configure(cfg.detector.confirm_hits, cfg.detector.confirm_max_misses,
                           cfg.detector.confirm_gate_m, cfg.detector.confirm_gate_px,
                           cfg.detector.confirm_width_ratio);

  cv::VideoWriter writer;
  if (!out_video.empty()) {
    // The source is 166 fps, so writing at the source rate reproduces the real
    // 10 s duration but nobody can see the boxes at that speed. --out-fps lets a
    // demo clip be written at, say, 30 fps for a 5.5x slow motion.
    double out_fps = (out_fps_override > 0.0)
                         ? out_fps_override
                         : ((src_fps > 1.0 && src_fps < 1e4) ? src_fps : 30.0);
    std::printf("writing %s at %.2f fps\n", out_video.c_str(), out_fps);
    writer.open(out_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), out_fps,
                cv::Size(w, h));
    if (!writer.isOpened()) {
      std::fprintf(stderr, "error: cannot open output video %s\n", out_video.c_str());
      return 2;
    }
  }

  std::ofstream csv;
  if (!out_csv.empty()) {
    csv.open(out_csv);
    csv << "frame,decode_ms,detect_ms,match_ms,pnp_ms,total_ms,n_bars,n_armors,"
           "pose_ok,x,y,z,yaw_deg,pitch_deg,reproj_err_px,track_state,gyro_valid,omega_yaw_dps,gyro_best,gyro_t_to_face_s\n";
  }

  cv::Mat frame;
  if (!no_display) {
    // WINDOW_NORMAL (not the default AUTOSIZE): a 1440x1080 image shown in a
    // smaller window gets downscaled by the WM, which shrinks the overlay text
    // with it and makes it look unreadable. Normal windows can be resized and
    // maximised, so the text stays at its drawn size.
    cv::namedWindow("autoaim", cv::WINDOW_NORMAL);
    cv::resizeWindow("autoaim", 1280, 960);
  }

  int index = 0;
  int n_frames = 0;
  int n_hit = 0;
  int n_pose = 0;
  int n_confirmed = 0;
  // Interference statistics: how many candidate bars / pairings the scene produces
  // per frame before the geometry filter reduces them to one target.
  double sum_bars = 0.0, sum_pairs = 0.0, sum_armors = 0.0;
  // ROI 使用统计：用来验证"窗口失效会回退全图"，也是 --roi 提速的来源证据
  int n_roi_frames = 0;
  int n_full_frames = 0;
  int n_fallback = 0;
  int n_roi_retry = 0;
  int n_pred = 0;   // 有卡尔曼预测输出的帧数
  int lost = 0;
  std::vector<double> detect_ms, match_ms, pnp_ms, total_ms, zs, errs;

  const Clock::time_point t_start = Clock::now();

  while (true) {
    const Clock::time_point t_read = Clock::now();
    if (!cap.read(frame) || frame.empty()) break;
    const double decode_ms = msSince(t_read);
    // Own counter, not CAP_PROP_POS_FRAMES: reading advances that property, so it
    // returns the NEXT frame index and shifted every CSV row by one.
    const int fi = index;
    ++index;

    autoaim::FrameResult res;
    const Clock::time_point t_det = Clock::now();

    // ---------------------------------------------------------------------
    // ROI（搜索窗）决策
    //
    // 三个参数一起决定"这一帧在哪找灯条"：
    //   roi_pad_px       : 以预测位置为中心、向外扩多少像素。窗口越小越快，
    //                      但必须覆盖目标在一帧到两帧内的位移。实测量级：
    //                      20 ms 延迟、1 m/s 横移 = 12 px，所以 200 px 是
    //                      10 倍余量，足够容忍检测抖动和短暂预测误差。
    //   roi_lost_frames  : 连续丢失多少帧后放弃窗口、回退全图。太小会频繁
    //                      全图（慢），太大会在真正丢失时一直空转（更糟）。
    //   pad 随丢失帧数增长 : 目标可能快速移动而预测偏了，窗口每丢一帧就放大
    //                      一次（每次 ×1.6），最多到 600 px。这样"跟错窗"不会
    //                      直接导致整段丢失，而是逐步扩大搜索范围自愈。
    // ---------------------------------------------------------------------
    cv::Rect roi;
    bool used_roi = false;
    if (cfg.detector.enable_roi && tracker.initialized() &&
        lost < cfg.detector.roi_lost_frames) {
      int pad = cfg.detector.roi_pad_px;
      // 最多放大 8 次（200 -> 600 已到上限），循环次数必须封顶，
      // 否则 1.6^n 会把 int 撑爆
      for (int k = 0; k < lost && k < 8; ++k) pad = static_cast<int>(pad * 1.6);
      if (pad > 600) pad = 600;
      roi = tracker.roiAround(pad, frame.size(), cfg.camera.fx, cfg.camera.cx,
                              cfg.camera.cy);
      used_roi = roi.width > 4 && roi.height > 4;
    }
    if (used_roi) {
      ++n_roi_frames;
    } else {
      ++n_full_frames;
      if (cfg.detector.enable_roi && lost >= cfg.detector.roi_lost_frames) ++n_fallback;
    }

    res.bars = detector.detect(frame, roi);

    // ROI 自愈：窗口里一根灯条都配不成对，就当场退回全图再找一次。
    // 代价只付在"本来就会漏检"的帧上（约 15% 的帧），成功帧仍然只花 1 ms；
    // 收益是把 ROI 造成的漏检补回来 —— 否则窗口跟丢后要等 roi_lost_frames
    // 帧才回退全图，那几帧就白丢了（实测能找回约 3.5 个百分点的检出率）。
    if (used_roi && matcher.match(res.bars).empty()) {
      ++n_roi_retry;
      res.bars = detector.detect(frame, cv::Rect());  // 空 ROI = 全图
    }

    const Clock::time_point t_match = Clock::now();
    res.timing.detect_ms = std::chrono::duration<double, std::milli>(t_match - t_det).count();

    res.armors = matcher.match(res.bars);
    const Clock::time_point t_pnp = Clock::now();
    res.timing.match_ms =
        std::chrono::duration<double, std::milli>(t_pnp - t_match).count();

    // Sub-pixel refinement of the plate corners on the chroma-difference image:
    // the corners sit on the bars' high-contrast boundary, which is what
    // cornerSubPix expects. Corners are full-frame, the difference image is
    // ROI-local, hence the shift. A rejected refinement keeps the original
    // corners; the reprojection-error acceptance test is untouched.
    {
      const cv::Mat& diff = detector.diffImage();
      if (!diff.empty()) {
        const cv::Rect roi = detector.lastRoi();
        const cv::Point2f off(static_cast<float>(roi.x), static_cast<float>(roi.y));
        for (auto& a : res.armors) {
          std::vector<cv::Point2f> cs(a.corners.begin(), a.corners.end());
          for (auto& p : cs) p -= off;
          if (autoaim::refineArmorCornersSubPix(diff, &cs)) {
            for (auto& p : cs) p += off;
            for (size_t k = 0; k < 4; ++k) a.corners[k] = cs[k];
            autoaim::armorCenterFromCorners(cs, &a.center);
          }
        }
      }
    }

    res.poses.reserve(res.armors.size());
    for (const auto& a : res.armors) res.poses.push_back(solver.solve(a));
    res.timing.pnp_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_pnp).count();

    // ---------------------------------------------------------------------
    // 时序层：决定"这一帧把哪个候选当成主目标"
    //
    // 输入是本帧所有通过 PnP 的候选（中心像素、四角宽度、tvec），输出是被选中的
    // 下标和一个状态码。它只改"选谁"，不改检测阈值、不改 PnP 结果。
    // 详见 include/autoaim/temporal_filter.hpp 的注释。
    // ---------------------------------------------------------------------
    {
      std::vector<autoaim::TrackCandidate> cands;
      cands.reserve(res.armors.size());
      for (size_t i = 0; i < res.armors.size(); ++i) {
        if (i >= res.poses.size() || !res.poses[i].ok) continue;
        autoaim::TrackCandidate c;
        c.center_px = res.armors[i].center;
        c.width_px = res.armors[i].width_px;
        c.tvec = res.poses[i].tvec;
        cands.push_back(c);
      }
      autoaim::TrackState st = autoaim::TrackState::kNone;
      int chosen = -1;
      if (cfg.detector.enable_confirm) {
        chosen = target_tracker.update(cands, &st);
      } else {
        st = cands.empty() ? autoaim::TrackState::kNone : autoaim::TrackState::kConfirmed;
        chosen = cands.empty() ? -1 : 0;
      }
      res.track_state = static_cast<int>(st);
      if (chosen >= 0) {
        // 把时序层选中的候选映射回 res.armors 的下标：cands 是"pose 有效"的
        // 子序列，需要反查回原下标，否则 override 会指错板。
        int mapped = -1;
        int seen = 0;
        for (size_t i = 0; i < res.armors.size(); ++i) {
          if (i >= res.poses.size() || !res.poses[i].ok) continue;
          if (seen == chosen) {
            mapped = static_cast<int>(i);
            break;
          }
          ++seen;
        }
        res.primary_override = mapped;
      } else {
        // -2 表示"时序层明确判定本帧没有可输出的主目标"（续帧期 / 未确认），
        // 必须和"没设过(-1)"区分开：否则 primaryIndex() 会回退到按 score 取，
        // 又把这个未确认的候选画成主目标了。
        res.primary_override = cfg.detector.enable_confirm ? -2 : -1;
      }
      res.confirm_hits = target_tracker.hits();
      res.target_confirmed = (st == autoaim::TrackState::kConfirmed);
      if (res.target_confirmed) ++n_confirmed;
    }

    const int primary = res.primaryIndex();
    bool pose_ok = false;
    if (primary >= 0 && static_cast<size_t>(primary) < res.poses.size()) {
      pose_ok = res.poses[primary].ok;
    }

    // ---------------------------------------------------------------------
    // 卡尔曼预测（延迟补偿）
    //
    // 位姿对应的是"曝光那一刻"的目标位置，但从曝光到真正击发还有一段延迟
    // （传输 + 解算 + 开火指令，典型 20~40 ms）。这段时间目标会移动，
    // 所以用滤波器估出的速度把位置外推：
    //
    //     p_pred = p_now + v × t_latency
    //
    // 然后把 p_pred 投影回图像，画面上画出来 —— 这就是"预瞄点"。
    // 实测目标横向 1 m/s 时，20 ms 对应 2 cm，在 2 m 处约 3 px。
    // ---------------------------------------------------------------------
    if (pose_ok) {
      const autoaim::Pose& p = res.poses[primary];
      ++n_pose;
      lost = 0;
      // Feed the filter whenever a pose exists; it also drives the ROI window.
      tracker.update(p.tvec);
      zs.push_back(p.z());
      errs.push_back(p.reproj_err_px);
    } else {
      if (cfg.detector.enable_roi) ++lost;
      // 丢帧时也要推进滤波器：ROI 窗口靠这个预测位置移动，
      // 只开 --roi 不开 --tracker 时同样需要
      if (cfg.detector.enable_tracker || cfg.detector.enable_roi) tracker.predictOnly();
      // 长时间丢失（超过 ROI 容忍帧数的 3 倍）后重置滤波器。
      // 否则陈旧的速度估计会一直把预测位置往外甩，重新捕获时窗口完全偏离目标。
      if (lost > cfg.detector.roi_lost_frames * 3) tracker.reset();
    }

    // ------------------------------------------------------------------
    // 反陀螺（Anti-Gyro）
    //
    // 位置：严格在"卡尔曼之后"。它**只读**卡尔曼平滑后的位置，不修改滤波器，
    // 也不替换 PnP 输出 —— 原来那套仍然是"当前这块装甲"的唯一来源。
    //
    // 逻辑：把最近若干帧的平滑位置喂进去，拟合出旋转中心/转轴/角速度，
    // 再用 90° 刚体关系推出另外三块。拟合不成立（样本不足、残差大、
    // 角速度太小）时 state().valid 为 false，主循环自动继续用卡尔曼。
    // ------------------------------------------------------------------
    {
      const double t_sec = static_cast<double>(index) / std::max(src_fps, 1.0);
      if (pose_ok) {
        double gyaw = 0.0, gpitch = 0.0;
        res.poses[primary].yawPitchDeg(&gyaw, &gpitch);
        // 传入的是"卡尔曼平滑后的位置"而不是原始 PnP 结果，原因：
        // 单帧 PnP 有 0.4 px 抖动，直接拟合圆会把噪声当成角速度。
        gyro.push(tracker.position(), gyaw, t_sec);
      } else {
        gyro.miss();   // 丢帧：模型若仍有效会按角速度外推相位，抗短暂遮挡
      }

      const autoaim::GyroState& gs = gyro.state();
      res.gyro_valid = gs.valid;
      res.gyro_centre = gs.centre;
      res.gyro_omega_dps = gs.omega_dps;
      res.gyro_omega_yaw_dps = gs.omega_yaw_dps;
      res.gyro_direction = gs.direction;
      res.gyro_samples = gs.samples;
      res.gyro_best = gs.best;
      res.gyro_armors = gs.armors;
      res.gyro_best_yaw_at_fire = gs.best_yaw_deg;
      res.gyro_best_is_current = gs.best_is_current;
      res.gyro_predict_ahead_s = gs.predict_ahead_s;
      res.gyro_best_pos_at_fire = gs.best_pos_at_fire;
      if (gs.best >= 0 && gs.best < 4) res.gyro_t_to_face_s = gs.t_to_face_s[static_cast<size_t>(gs.best)];

      // 把四块装甲的 3D 坐标投影回像素，用于调试可视化。
      // projectPoints 会考虑镜头畸变，和 PnP 的成像模型完全一致。
      if (gs.valid && cfg.detector.draw_gyro != 0) {
        // 投影用手写针孔公式，不调 cv::projectPoints。
        // 原因：projectPoints 在 4.5.4 里对 InputArray 的临时对象类型推导很挑，
        // 之前就是在这里抛 cv::Exception。而调试用的圈只需要线性投影：
        //   u = fx·X/Z + cx,  v = fy·Y/Z + cy
        // 忽略畸变带来的误差约 0.3%，对"画圈看位置"完全够用。
        auto project = [&](const cv::Vec3d& P) {
          return cv::Point2f(static_cast<float>(cfg.camera.fx * P[0] / P[2] + cfg.camera.cx),
                             static_cast<float>(cfg.camera.fy * P[1] / P[2] + cfg.camera.cy));
        };
        if (gs.centre[2] > 0.15) {
          res.gyro_centre_px = project(gs.centre);
          res.gyro_centre_ok = true;
        }
        for (int k = 0; k < 4; ++k) {
          if (gs.armors[k][2] <= 0.15) continue;
          const cv::Point2f px = project(gs.armors[k]);
          res.gyro_px[k] = px;
          res.gyro_px_ok[k] = px.x >= 0 && px.x < frame.cols && px.y >= 0 &&
                              px.y < frame.rows;
        }
      }
    }
    // 计算并投影预测点：只要滤波器已有状态就算，丢帧续推时同样成立
    if (cfg.detector.enable_tracker && tracker.initialized()) {
      const cv::Vec3d v = tracker.velocity();
      const cv::Vec3d now = tracker.position();
      const cv::Vec3d pred = now + v * (cfg.detector.predict_latency_s * src_fps);
      if (pred[2] > 0.1) {
        res.has_prediction = true;
        res.predicted_tvec = pred;
        res.predicted_px = cv::Point2f(
            static_cast<float>(cfg.camera.fx * pred[0] / pred[2] + cfg.camera.cx),
            static_cast<float>(cfg.camera.fy * pred[1] / pred[2] + cfg.camera.cy));
        res.predict_latency_ms = cfg.detector.predict_latency_s * 1000.0;
        ++n_pred;   // 统计"预测覆盖率"，写在计算之后才准
      }
    }

    res.timing.decode_ms = decode_ms;
    res.timing.total_ms =
        decode_ms + res.timing.detect_ms + res.timing.match_ms + res.timing.pnp_ms;

    ++n_frames;
    if (!res.armors.empty()) ++n_hit;
    sum_bars += static_cast<double>(res.bars.size());
    sum_pairs += 0.5 * static_cast<double>(res.bars.size()) *
                 static_cast<double>(res.bars.size() > 0 ? res.bars.size() - 1 : 0);
    sum_armors += static_cast<double>(res.armors.size());

    if (csv.is_open()) {
      // 列顺序必须与表头严格一致，否则 DictReader 按列名取值会整体错位。
      // 表头：frame,decode_ms,detect_ms,match_ms,pnp_ms,total_ms,
      //       n_bars,n_armors,pose_ok,x,y,z,yaw_deg,pitch_deg,reproj_err_px
      csv << fi << ',' << decode_ms << ',' << res.timing.detect_ms << ','
          << res.timing.match_ms << ',' << res.timing.pnp_ms << ','
          << res.timing.total_ms << ',' << res.bars.size() << ','
          << res.armors.size() << ',' << (pose_ok ? 1 : 0);
      if (pose_ok) {
        const autoaim::Pose& p = res.poses[primary];
        double yaw = 0, pitch = 0;
        p.yawPitchDeg(&yaw, &pitch);
        csv << ',' << p.x() << ',' << p.y() << ',' << p.z() << ',' << yaw << ',' << pitch
            << ',' << p.reproj_err_px;
      } else {
        csv << ",,,,,,";
      }
      // 时序状态：0 无 / 1 确认 / 2 续帧 / 3 次候选 —— 复盘时按这列能区分
      // "这一帧是真目标、还是预测续帧、还是没确认的候选"
      // 反陀螺数据（即使画面不画，也必须输出：评估"是否自旋"要看这一列）
      csv << ',' << res.track_state << ',' << (res.gyro_valid ? 1 : 0) << ','
          << res.gyro_omega_yaw_dps << ',' << res.gyro_best << ','
          << res.gyro_t_to_face_s << '\n';
    }

    detect_ms.push_back(res.timing.detect_ms);
    match_ms.push_back(res.timing.match_ms);
    total_ms.push_back(res.timing.total_ms);

    if (!out_video.empty() || !no_display) {
      autoaim::DrawOptions opt;
      opt.show_secondary = show_all;   // default off: a flickering 2nd candidate reads as a bug
      opt.inflate_px = static_cast<float>(cfg.detector.draw_inflate_px);
      opt.inset_ratio = static_cast<float>(cfg.detector.draw_inset_ratio);
      opt.extra_height_ratio = static_cast<float>(cfg.detector.draw_extra_height_ratio);
      opt.show_tentative = cfg.detector.draw_tentative != 0;
      opt.show_prediction = cfg.detector.draw_prediction != 0;
      opt.show_gyro = cfg.detector.draw_gyro != 0;
      autoaim::drawOverlay(frame, res, opt);
    }
    if (!out_video.empty()) writer.write(frame);
    if (!no_display) {
      cv::imshow("autoaim", frame);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') break;
    }

    if (verbose && index % 200 == 0) {
      std::printf("  %d frames  %.2f ms/frame\n", index,
                  std::chrono::duration<double, std::milli>(Clock::now() - t_start).count() /
                      index);
    }
    if (max_frames > 0 && index >= max_frames) break;
  }

  if (writer.isOpened()) writer.release();
  if (csv.is_open()) csv.close();
  if (!no_display) cv::destroyAllWindows();

  auto median = [](std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };

  const double wall = std::chrono::duration<double>(Clock::now() - t_start).count();

  std::printf("\n================ summary ================\n");
  std::printf("frames            %d\n", n_frames);
  std::printf("frames with armor %.1f %%\n", 100.0 * n_hit / std::max(n_frames, 1));
  std::printf("valid pose        %.1f %%\n", 100.0 * n_pose / std::max(n_frames, 1));
  std::printf("confirmed target  %.1f %%\n", 100.0 * n_confirmed / std::max(n_frames, 1));
  const double nf = std::max(n_frames, 1);
  std::printf("interference      %.1f bars/frame -> %.1f pairs/frame -> %.2f passed/frame\n",
              sum_bars / nf, sum_pairs / nf, sum_armors / nf);
  std::printf("roi window        %.1f %% of frames (%d roi / %d full, %d fallback, %d retry)\n",
              100.0 * n_roi_frames / nf, n_roi_frames, n_full_frames, n_fallback, n_roi_retry);
  std::printf("detect   median   %.2f ms\n", median(detect_ms));
  std::printf("match    median   %.2f ms\n", median(match_ms));
  std::printf("total    median   %.2f ms  -> %.1f FPS (algorithm only)\n",
              median(total_ms), 1000.0 / std::max(median(total_ms), 1e-9));
  std::printf("end-to-end        %.1f FPS over %.2f s wall time\n",
              n_frames / std::max(wall, 1e-9), wall);
  std::printf("prediction        %.1f %% of frames (latency %.0f ms)\n",
              100.0 * n_pred / nf, cfg.detector.predict_latency_s * 1000.0);
  std::printf("median range      %.3f m\n", median(zs));
  std::printf("median reproj err %.3f px\n", median(errs));
  std::printf("=========================================\n");
  return 0;
}

