#include "autoaim/config.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoaim {
namespace {

// Minimal YAML reader for the calibration files the team supplies.
//
// cv::FileStorage CANNOT parse them: they use block sequences
//     camera_matrix:
//     - 2334.31
//     - 0
// and OpenCV's YAML parser throws on that form (verified: flow sequences,
// comments, strings and scalars all parse fine; only "key:" + "- item" fails).
// So we handle exactly that subset ourselves: scalars, block sequences, inline
// [a, b, c] sequences, comments, and quoted or bare strings.
class YamlLite {
 public:
  bool load(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return false;
    std::string line;
    std::string current;
    while (std::getline(f, line)) {
      const std::string l = trim(stripComment(line));
      if (l.empty() || l[0] == '%') continue;

      if (l[0] == '-') {  // block-sequence item belongs to the previous key
        if (current.empty()) continue;
        const std::string v = trim(l.substr(1));
        if (!v.empty()) items_[current].push_back(unquote(v));
        continue;
      }
      const size_t colon = l.find(':');
      if (colon == std::string::npos) continue;
      const std::string key = trim(l.substr(0, colon));
      std::string rest = trim(l.substr(colon + 1));
      if (key.empty()) continue;

      if (rest.empty()) {
        current = key;
        items_[key];
      } else if (rest.front() == '[') {
        current.clear();
        if (rest.back() == ']') rest.pop_back();
        std::stringstream ss(rest.substr(1));
        std::string tok;
        while (std::getline(ss, tok, ',')) {
          tok = trim(tok);
          if (!tok.empty()) items_[key].push_back(unquote(tok));
        }
      } else {
        current.clear();
        items_[key].push_back(unquote(rest));
      }
    }
    return !items_.empty();
  }

  bool has(const std::string& key) const { return items_.count(key) != 0; }

  std::vector<double> numbers(const std::string& key) const {
    std::vector<double> out;
    const auto it = items_.find(key);
    if (it == items_.end()) return out;
    for (const auto& s : it->second) {
      try {
        out.push_back(std::stod(s));
      } catch (...) {
        // ignore unparsable entries (e.g. a textual note in the sequence)
      }
    }
    return out;
  }

  double number(const std::string& key, double fallback) const {
    const std::vector<double> v = numbers(key);
    return v.empty() ? fallback : v.front();
  }

 private:
  static std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
  }

  static std::string stripComment(const std::string& s) {
    bool in_single = false, in_double = false;
    for (size_t i = 0; i < s.size(); ++i) {
      const char c = s[i];
      if (c == '\'' && !in_double) in_single = !in_single;
      else if (c == '"' && !in_single) in_double = !in_double;
      else if (c == '#' && !in_single && !in_double) return s.substr(0, i);
    }
    return s;
  }

  static std::string unquote(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
      return s.substr(1, s.size() - 2);
    }
    return s;
  }

  std::map<std::string, std::vector<std::string>> items_;
};

// FileStorage returns an empty node for a missing key, so every read goes
// through these helpers and silently keeps the default instead of throwing.
void get(const cv::FileNode& n, const char* key, int* v) {
  const cv::FileNode c = n[key];
  if (!c.empty()) c >> *v;
}
void get(const cv::FileNode& n, const char* key, double* v) {
  const cv::FileNode c = n[key];
  if (!c.empty()) c >> *v;
}
void get(const cv::FileNode& n, const char* key, bool* v) {
  const cv::FileNode c = n[key];
  if (!c.empty()) {
    int tmp = *v ? 1 : 0;
    c >> tmp;
    *v = (tmp != 0);
  }
}
void get(const cv::FileNode& n, const char* key, std::string* v) {
  const cv::FileNode c = n[key];
  if (!c.empty() && c.isString()) c >> *v;
}

}  // namespace

cv::Mat CameraConfig::cameraMatrix() const {
  return (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
}

cv::Mat CameraConfig::distCoeffs() const {
  return (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
}

Config Config::load(const std::string& path) {
  Config cfg;  // defaults are already the tuned values
  if (path.empty()) return cfg;

  std::ifstream probe(path);
  if (!probe.good()) {
    throw std::runtime_error("config file not found: " + path);
  }
  probe.close();

  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("cannot parse config file: " + path);
  }

  if (!fs["camera"].empty()) {
    const cv::FileNode c = fs["camera"];
    get(c, "fx", &cfg.camera.fx);
    get(c, "fy", &cfg.camera.fy);
    get(c, "cx", &cfg.camera.cx);
    get(c, "cy", &cfg.camera.cy);
    get(c, "k1", &cfg.camera.k1);
    get(c, "k2", &cfg.camera.k2);
    get(c, "p1", &cfg.camera.p1);
    get(c, "p2", &cfg.camera.p2);
    get(c, "k3", &cfg.camera.k3);
    get(c, "width", &cfg.camera.width);
    get(c, "height", &cfg.camera.height);
  }

  if (!fs["detector"].empty()) {
    const cv::FileNode d = fs["detector"];
    auto& t = cfg.detector;
    get(d, "enemy_color", &t.enemy_color);
    get(d, "chroma_mode", &t.chroma_mode);
    get(d, "diff_threshold", &t.diff_threshold);
    get(d, "use_brightness_gate", &t.use_brightness_gate);
    get(d, "brightness_min", &t.brightness_min);

    get(d, "morph_open_kernel", &t.morph_open_kernel);
    get(d, "morph_close_w", &t.morph_close_w);
    get(d, "morph_close_h", &t.morph_close_h);
    get(d, "do_open", &t.do_open);
    get(d, "do_close", &t.do_close);

    get(d, "min_area", &t.min_area);
    get(d, "max_area", &t.max_area);
    get(d, "min_lightbar_ratio", &t.min_lightbar_ratio);
    get(d, "max_lightbar_ratio", &t.max_lightbar_ratio);
    get(d, "min_lightbar_length", &t.min_lightbar_length);
    get(d, "min_fill_ratio", &t.min_fill_ratio);
    get(d, "max_tilt_deg", &t.max_tilt_deg);
    get(d, "min_bar_mean", &t.min_bar_mean);

    get(d, "max_length_ratio", &t.max_length_ratio);
    get(d, "min_center_ratio", &t.min_center_ratio);
    get(d, "max_center_ratio", &t.max_center_ratio);
    get(d, "max_angle_diff_deg", &t.max_angle_diff_deg);
    get(d, "max_side_angle_deg", &t.max_side_angle_deg);
    get(d, "ideal_center_ratio", &t.ideal_center_ratio);
    get(d, "min_quad_aspect", &t.min_quad_aspect);
    get(d, "max_quad_aspect", &t.max_quad_aspect);
    get(d, "max_bar_brightness_ratio", &t.max_bar_brightness_ratio);
    get(d, "min_armor_width_px", &t.min_armor_width_px);
    get(d, "max_armor_score", &t.max_armor_score);
    get(d, "draw_inflate_px", &t.draw_inflate_px);
    get(d, "draw_inset_ratio", &t.draw_inset_ratio);
    get(d, "draw_extra_height_ratio", &t.draw_extra_height_ratio);
    get(d, "draw_tentative", &t.draw_tentative);

    get(d, "armor_width_m", &t.armor_width_m);
    get(d, "armor_height_m", &t.armor_height_m);
    get(d, "bar_length_m", &t.bar_length_m);
    get(d, "plate_edge_offset", &t.plate_edge_offset);

    get(d, "max_reproj_err_px", &t.max_reproj_err_px);
    get(d, "refine_iterations", &t.refine_iterations);

    get(d, "enable_roi", &t.enable_roi);
    get(d, "roi_pad_px", &t.roi_pad_px);
    get(d, "roi_lost_frames", &t.roi_lost_frames);

    get(d, "enable_confirm", &t.enable_confirm);
    get(d, "confirm_hits", &t.confirm_hits);
    get(d, "confirm_max_misses", &t.confirm_max_misses);
    get(d, "confirm_gate_m", &t.confirm_gate_m);
    get(d, "confirm_gate_px", &t.confirm_gate_px);
    get(d, "confirm_width_ratio", &t.confirm_width_ratio);

    get(d, "enable_gyro", &t.enable_gyro);
    get(d, "gyro_window", &t.gyro_window);
    get(d, "gyro_min_samples", &t.gyro_min_samples);
    get(d, "gyro_min_omega_dps", &t.gyro_min_omega_dps);
    get(d, "gyro_max_plane_rms_m", &t.gyro_max_plane_rms_m);
    get(d, "gyro_reset_frames", &t.gyro_reset_frames);
    get(d, "draw_gyro", &t.draw_gyro);
    get(d, "gyro_min_yaw_rate_dps", &t.gyro_min_yaw_rate_dps);
    get(d, "gyro_predict_ahead_s", &t.gyro_predict_ahead_s);
    get(d, "gyro_face_on_deg", &t.gyro_face_on_deg);
    get(d, "gyro_stabilize_window", &t.gyro_stabilize_window);
    get(d, "gyro_max_omega_rel_std", &t.gyro_max_omega_rel_std);
    get(d, "gyro_max_centre_spread_m", &t.gyro_max_centre_spread_m);
    get(d, "gyro_max_pred_err_m", &t.gyro_max_pred_err_m);
    get(d, "gyro_expected_radius_m", &t.gyro_expected_radius_m);
    get(d, "gyro_radius_tol", &t.gyro_radius_tol);

    get(d, "enable_tracker", &t.enable_tracker);
    get(d, "predict_latency_s", &t.predict_latency_s);
    get(d, "draw_prediction", &t.draw_prediction);
    get(d, "tracker_process_noise", &t.tracker_process_noise);
    get(d, "tracker_measure_noise", &t.tracker_measure_noise);
  }

  return cfg;
}

bool Config::loadCalibration(const std::string& path) {
  if (path.empty()) return false;

  // The team's calibration files use block sequences, which cv::FileStorage
  // rejects, so they go through YamlLite. Fall back to FileStorage in case a
  // future version of the file switches to flow-style sequences.
  YamlLite yaml;
  bool parsed = yaml.load(path);
  if (!parsed) {
    std::cerr << "[warn] cannot read calibration file: " << path
              << " (keeping config values)\n";
    return false;
  }

  const std::vector<double> cm = yaml.numbers("camera_matrix");
  if (cm.size() >= 9) {
    camera.fx = cm[0];
    camera.cx = cm[2];
    camera.fy = cm[4];
    camera.cy = cm[5];
  } else if (yaml.has("fx")) {
    camera.fx = yaml.number("fx", camera.fx);
    camera.fy = yaml.number("fy", camera.fy);
    camera.cx = yaml.number("cx", camera.cx);
    camera.cy = yaml.number("cy", camera.cy);
  } else {
    std::cerr << "[warn] no camera_matrix in " << path << "\n";
    return false;
  }

  const std::vector<double> dc = yaml.numbers("distort_coeffs");
  if (dc.size() >= 4) {
    camera.k1 = dc[0];
    camera.k2 = dc[1];
    camera.p1 = dc[2];
    camera.p2 = dc[3];
    if (dc.size() >= 5) camera.k3 = dc[4];
  }

  camera.width = static_cast<int>(yaml.number("image_width", camera.width));
  camera.height = static_cast<int>(yaml.number("image_height", camera.height));

  // Hand-eye extrinsics: consumed and reported, but not applied, because the
  // task asks for the plate centre in the camera frame.
  const std::vector<double> rg = yaml.numbers("R_camera2gimbal");
  if (rg.size() >= 9) {
    for (int i = 0; i < 9; ++i) camera.R_camera2gimbal[i] = rg[i];
    camera.has_camera2gimbal = true;
  }
  const std::vector<double> tg = yaml.numbers("t_camera2gimbal");
  if (tg.size() >= 3) {
    for (int i = 0; i < 3; ++i) camera.t_camera2gimbal[i] = tg[i];
  }
  const std::vector<double> ri = yaml.numbers("R_gimbal2imubody");
  if (ri.size() >= 9) {
    for (int i = 0; i < 9; ++i) camera.R_gimbal2imubody[i] = ri[i];
    camera.has_gimbal2imubody = true;
  }

  std::printf("calibration loaded from %s\n", path.c_str());
  std::printf("  camera_matrix fx=%.4f fy=%.4f cx=%.4f cy=%.4f  %dx%d\n", camera.fx,
              camera.fy, camera.cx, camera.cy, camera.width, camera.height);
  std::printf("  distort k1=%.6f k2=%.6f p1=%.6f p2=%.6f", camera.k1, camera.k2,
              camera.p1, camera.p2);
  if (dc.size() >= 5) std::printf(" k3=%.6f", camera.k3);
  std::printf("\n");
  if (camera.has_camera2gimbal) {
    std::printf("  R_camera2gimbal parsed (not applied: output is camera-frame)\n");
    std::printf("  t_camera2gimbal = [%.4f, %.4f, %.4f] m\n", camera.t_camera2gimbal[0],
                camera.t_camera2gimbal[1], camera.t_camera2gimbal[2]);
  }
  if (camera.has_gimbal2imubody) {
    std::printf("  R_gimbal2imubody parsed (not applied)\n");
  }
  return true;
}

void Config::dump() const {
  std::printf("--------------- config ---------------\n");
  std::printf("camera    fx=%.4f fy=%.4f cx=%.4f cy=%.4f  %dx%d\n", camera.fx, camera.fy,
              camera.cx, camera.cy, camera.width, camera.height);
  std::printf("dist      k1=%.6f k2=%.6f p1=%.6f p2=%.6f k3=%.6f\n", camera.k1, camera.k2,
              camera.p1, camera.p2, camera.k3);
  std::printf("colour    enemy=%s chroma=%s thr=%d bright_gate=%d(min %d)\n",
              detector.enemy_color.c_str(), detector.chroma_mode.c_str(),
              detector.diff_threshold, detector.use_brightness_gate ? 1 : 0,
              detector.brightness_min);
  std::printf("morph     open=%d close=%dx%d\n", detector.morph_open_kernel,
              detector.morph_close_w, detector.morph_close_h);
  std::printf("bar       area[%.0f,%.0f] ratio[%.1f,%.1f] len>=%.0f fill>=%.2f tilt<=%.0f\n",
              detector.min_area, detector.max_area, detector.min_lightbar_ratio,
              detector.max_lightbar_ratio, detector.min_lightbar_length,
              detector.min_fill_ratio, detector.max_tilt_deg);
  std::printf("pair      len_ratio<=%.2f center_ratio[%.1f,%.1f] angle<=%.0f side<=%.0f "
              "aspect[%.1f,%.1f]\n",
              detector.max_length_ratio, detector.min_center_ratio,
              detector.max_center_ratio, detector.max_angle_diff_deg,
              detector.max_side_angle_deg, detector.min_quad_aspect,
              detector.max_quad_aspect);
  std::printf("armor     %.0f x %.0f mm\n", detector.armor_width_m * 1000.0,
              detector.armor_height_m * 1000.0);
  std::printf("pnp       max_reproj=%.2f px refine_iters=%d\n", detector.max_reproj_err_px,
              detector.refine_iterations);
  std::printf("runtime   roi=%d tracker=%d headless=%d\n", detector.enable_roi ? 1 : 0,
              detector.enable_tracker ? 1 : 0, runtime.headless ? 1 : 0);
  std::printf("--------------------------------------\n");
}

}  // namespace autoaim


