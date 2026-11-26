// =============================================================================
// main.cpp —— 命令行入口
//
// 用法示例：
//   aegis --config config/aegis.yaml --video data/video.avi --out demo.mp4
//   aegis --config config/aegis.yaml --video data/video.avi --csv result.csv
//   aegis --baseline ...      # 固定阈值基线（消融实验用）
//   aegis --no-roi ...        # 关闭 ROI 加速（测全图性能）
// =============================================================================
#include "aegis/hud.hpp"
#include "aegis/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

struct Args {
  std::string config = "config/aegis.yaml";
  std::string video;
  std::string out;
  std::string csv;
  std::string dump;
  int snapshot = -1;
  int maxFrames = -1;
  int startFrame = 0;
  bool show = false;
  bool quiet = false;
  bool noDigits = false;
  bool noRoi = false;
  bool baseline = false;
  bool fixed = false;
  bool noIntro = false;
  int introSeconds = 3;
  int outroSeconds = 3;
};

Args parseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (k == "--config") a.config = next();
    else if (k == "--video") a.video = next();
    else if (k == "--out") a.out = next();
    else if (k == "--csv") a.csv = next();
    else if (k == "--dump") a.dump = next();
    else if (k == "--snapshot") a.snapshot = std::atoi(next().c_str());
    else if (k == "--max-frames") a.maxFrames = std::atoi(next().c_str());
    else if (k == "--start") a.startFrame = std::atoi(next().c_str());
    else if (k == "--show") a.show = true;
    else if (k == "--no-digits") a.noDigits = true;
    else if (k == "--no-roi") a.noRoi = true;
    else if (k == "--baseline") a.baseline = true;
    else if (k == "--fixed") a.fixed = true;
    else if (k == "--no-intro") a.noIntro = true;
    else if (k == "--quiet") a.quiet = true;
    else if (k == "--help" || k == "-h") {
      std::printf("用法: aegis --config <yaml> --video <avi> [--out <mp4>] [--csv <csv>]\n"
                  "            [--max-frames N] [--start N] [--baseline] [--no-roi] [--no-digits]\n"
                  "            [--show] [--dump <png> --snapshot N] [--no-intro]\n");
      std::exit(0);
    }
  }
  return a;
}

// 演示视频的抽帧器：把 165fps 的源视频映射到 60fps 输出。
struct FramePacer {
  double step = 1.0;
  double next = 0.0;
  int written = 0;
  bool due(int frameIndex) {
    if (frameIndex + 1e-9 >= next) {
      ++written;
      next = written * step;
      return true;
    }
    return false;
  }
};

}  // namespace

int main(int argc, char** argv) {
  const Args args = parseArgs(argc, argv);
  aegis::AppConfig cfg = aegis::loadConfig(args.config);
  if (args.noDigits) cfg.enableDigits = false;
  if (args.noRoi) cfg.enableRoi = false;
  if (args.fixed) cfg.detector.fixedThreshold = true;
  if (args.baseline) {
    cfg.baselineMode = true;
    cfg.enableDigits = false;
    cfg.enableRoi = false;
    cfg.detector.fixedThreshold = true;
    cfg.detector.coreCompensation = false;
  }

  const std::string videoPath = args.video.empty() ? cfg.videoPath : args.video;
  if (videoPath.empty()) {
    std::fprintf(stderr, "没有指定视频：--video 或 config 里的 video_path\n");
    return 1;
  }

  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "无法打开视频 %s\n", videoPath.c_str());
    return 1;
  }
  const double srcFps = cap.get(cv::CAP_PROP_FPS) > 1.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
  const int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  if (!args.quiet) {
    std::printf("视频 %s  %dx%d  %.2f fps  %d 帧\n", videoPath.c_str(), width, height, srcFps,
                totalFrames);
  }

  cv::VideoWriter writer;
  if (!args.out.empty()) {
    writer.open(args.out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cfg.outputFps,
                cv::Size(width, height));
    if (!writer.isOpened()) {
      std::fprintf(stderr, "无法创建输出视频 %s\n", args.out.c_str());
      return 1;
    }
  }

  std::ofstream csv;
  if (!args.csv.empty()) {
    csv.open(args.csv);
    csv << "frame,time_s,detections,primary,display_id,number,number_conf,threshold_level,roi,"
           "raw_x,raw_y,trk_x,trk_y,x,y,z,distance,yaw,pitch,reproj,detect_ms,track_ms,pose_ms,"
           "total_ms\n";
  }

  aegis::Pipeline pipeline(cfg);
  aegis::Hud hud(cfg);
  aegis::HudState state;
  state.baseline = cfg.baselineMode;

  FramePacer pacer;
  pacer.step = srcFps / std::max(1, cfg.outputFps);

  cv::Mat frame;
  cv::Mat firstFrame;
  cv::Mat lastFrame;
  int frameIdx = 0;
  int processed = 0;
  double emaFps = 0.0;
  double sumPipelineMs = 0.0;
  const auto tStart = std::chrono::steady_clock::now();

  auto writeIntro = [&](const cv::Mat& background) {
    if (writer.isOpened() && !args.noIntro) {
      const int n = args.introSeconds * cfg.outputFps;
      for (int i = 0; i < n; ++i) {
        cv::Mat card = background.clone();
        hud.drawIntro(card, static_cast<double>(i) / std::max(1, n - 1));
        writer.write(card);
      }
    }
  };

  while (cap.read(frame)) {
    if (frameIdx < args.startFrame) {
      ++frameIdx;
      continue;
    }
    if (args.maxFrames > 0 && frameIdx >= args.startFrame + args.maxFrames) break;
    if (frame.empty()) break;

    if (processed == 0) {
      firstFrame = frame.clone();
      writeIntro(firstFrame);
    }

    const double timeMs = static_cast<double>(frameIdx) * 1000.0 / srcFps;
    const auto t0 = std::chrono::steady_clock::now();
    aegis::FrameResult result = pipeline.process(frame, frameIdx, timeMs);
    const double wallMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sumPipelineMs += wallMs;

    // 指数滑动平均，HUD 上不会因为单帧抖动乱跳。
    const double instFps = 1000.0 / std::max(0.5, wallMs);
    emaFps = (emaFps <= 0.0) ? instFps : emaFps * 0.92 + instFps * 0.08;

    // ---- 统计 ----
    ++processed;
    ++state.frameIndex;
    state.fps = emaFps;
    state.processedFrames = processed;
    state.totalFrames = processed;
    const bool hasTarget = result.primaryTrack >= 0;
    if (hasTarget) {
      ++state.detectedFrames;
      state.trackedTimeMs += wallMs;
    } else {
      state.lostTimeMs += wallMs;
    }
    // 第一次锁定之后的统计：考核视频前 150 帧目标被画面裁掉，算进去会虚低。
    static bool lockedOnce = false;
    if (hasTarget) lockedOnce = true;
    if (lockedOnce) {
      ++state.framesAfterLock;
      if (hasTarget) ++state.detectedAfterLock;
    }
    state.detectHistory.push_back(static_cast<float>(result.timing.detect));
    if (state.detectHistory.size() > 120) state.detectHistory.pop_front();
    state.fpsHistory.push_back(static_cast<float>(emaFps));
    if (state.fpsHistory.size() > 120) state.fpsHistory.pop_front();
    state.successHistory.push_back(hasTarget ? 1.0F : 0.0F);
    if (state.successHistory.size() > 240) state.successHistory.pop_front();

    // 三维轨迹
    for (const auto& t : result.tracks) {
      if (t.id == result.primaryTrack && t.pose.valid) {
        state.poseTrail.push_back(t.pose.tvec);
        if (state.poseTrail.size() > 120) state.poseTrail.pop_front();
      }
    }

    // ---- CSV ----
    if (csv.is_open()) {
      const aegis::Track* primary = nullptr;
      for (const auto& t : result.tracks) {
        if (t.id == result.primaryTrack) primary = &t;
      }
      csv << frameIdx << ',' << cv::format("%.4f", timeMs / 1000.0) << ',' << result.armors.size()
          << ',' << result.primaryTrack << ',' << (primary ? primary->displayId : -1) << ','
          << (primary ? primary->number : -1) << ','
          << cv::format("%.3f", primary ? primary->numberConfidence : 0.0) << ','
          << result.thresholdLevel << ',' << (result.usedRoi ? 1 : 0);

      // 原始检测中心 vs 跟踪滤波后中心：用来量化“多帧追踪到底把抖动压了多少”。
      float rawX = 0.0F;
      float rawY = 0.0F;
      float trkX = 0.0F;
      float trkY = 0.0F;
      if (primary != nullptr) {
        trkX = primary->box.x + primary->box.width * 0.5F;
        trkY = primary->box.y + primary->box.height * 0.5F;
        float bestD = 1e9F;
        for (const auto& a : result.armors) {
          const float dx = a.center.x - trkX;
          const float dy = a.center.y - trkY;
          const float d = dx * dx + dy * dy;
          if (d < bestD) {
            bestD = d;
            rawX = a.center.x;
            rawY = a.center.y;
          }
        }
        if (bestD > 60.0F * 60.0F) {
          rawX = trkX;
          rawY = trkY;
        }
      }
      csv << ',' << rawX << ',' << rawY << ',' << trkX << ',' << trkY;
      if (primary != nullptr && primary->pose.valid) {
        const auto& p = primary->pose;
        csv << ',' << p.tvec[0] << ',' << p.tvec[1] << ',' << p.tvec[2] << ',' << p.distance << ','
            << p.yaw << ',' << p.pitch << ',' << p.reprojError;
      } else {
        csv << ",,,,,,,";
      }
      csv << ',' << result.timing.detect << ',' << result.timing.track << ',' << result.timing.pose
          << ',' << result.timing.total << '\n';
    }

    // ---- 写视频 ----
    const bool due = pacer.due(processed - 1);
    if (writer.isOpened() && due) {
      cv::Mat vis = frame.clone();
      if (cfg.drawHud) hud.draw(vis, result, state);
      writer.write(vis);
      lastFrame = vis.clone();
    }

    if (args.show) {
      cv::Mat vis = frame.clone();
      if (cfg.drawHud) hud.draw(vis, result, state);
      cv::imshow("AegisVision", vis);
      if (cv::waitKey(1) == 27) break;
    }
    if (!args.dump.empty() && (args.snapshot < 0 || frameIdx == args.snapshot)) {
      cv::Mat vis = frame.clone();
      if (cfg.drawHud) hud.draw(vis, result, state);
      cv::imwrite(args.dump, vis);
    }
    ++frameIdx;
  }

  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
  state.avgFps = processed / std::max(1e-6, sumPipelineMs / 1000.0);
  state.totalFrames = processed;
  state.avgLatencyMs = sumPipelineMs / std::max(1, processed);

  if (writer.isOpened() && !lastFrame.empty()) {
    const int n = args.outroSeconds * cfg.outputFps;
    // 片尾用干净的首帧做底，避免直播画面上的 HUD 面板透过来显得乱。
    const cv::Mat background = firstFrame.empty() ? lastFrame : firstFrame;
    for (int i = 0; i < n; ++i) {
      cv::Mat card = background.clone();
      hud.drawOutro(card, static_cast<double>(i) / std::max(1, n - 1), state);
      writer.write(card);
    }
  }

  if (!args.quiet) {
    std::printf("处理 %d 帧，耗时 %.2fs，平均 %.1f fps，写出 %d 帧\n", processed, elapsed,
                state.avgFps, pacer.written);
  }
  return 0;
}
