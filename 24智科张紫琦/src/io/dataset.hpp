#pragma once

// 数据集读取：以 video/frames.csv 的行顺序为准逐帧读取 video.avi。
// 时间基准全部来自 frames.csv 的 elapsed_s，不使用 AVI 容器自带的 fps/时长。

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rm {

// frames.csv 中的一行
struct FrameStamp {
  std::int64_t frame_index = -1;
  std::int64_t monotonic_ns = 0;
  std::int64_t unix_ns = 0;
  double elapsed_s = 0.0;  // 相对录制起点的秒数
  std::int64_t camera_frame_id = -1;
  std::int64_t device_timestamp_ticks = -1;
};

class Dataset {
 public:
  Dataset() = default;
  Dataset(Dataset&&) noexcept = default;
  Dataset& operator=(Dataset&&) noexcept = default;
  Dataset(const Dataset&) = delete;
  Dataset& operator=(const Dataset&) = delete;

  // root 为数据目录，如 data/2026-11-13_18-25-25_462305
  static Dataset open(const std::filesystem::path& root);

  const std::filesystem::path& root() const { return root_; }
  const std::string& name() const { return name_; }
  const std::vector<FrameStamp>& stamps() const { return stamps_; }
  std::size_t size() const { return stamps_.size(); }

  // 由 frames.csv 首末时间戳估算的平均帧率
  double nominalFps() const { return nominal_fps_; }
  // 容器内帧数（仅用于一致性检查，不参与时间计算）
  std::size_t videoFrameCount() const { return video_frames_; }
  cv::Size frameSize() const { return frame_size_; }

  // 顺序读取下一帧；帧数与 csv 行数一致。读到末尾返回 false。
  bool next(cv::Mat& frame, FrameStamp& stamp);

 private:
  std::filesystem::path root_;
  std::string name_;
  std::vector<FrameStamp> stamps_;
  std::unique_ptr<cv::VideoCapture> cap_;
  std::size_t cursor_ = 0;
  double nominal_fps_ = 0.0;
  std::size_t video_frames_ = 0;
  cv::Size frame_size_{};
};

namespace detail {

inline std::string trim(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

inline std::vector<std::string> splitCsv(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',') {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(trim(cur));
  return out;
}

inline std::size_t columnOf(const std::vector<std::string>& header, const char* name) {
  for (std::size_t i = 0; i < header.size(); ++i) {
    if (header[i] == name) return i;
  }
  throw std::runtime_error(std::string("frames.csv 缺少列: ") + name);
}

inline double toDouble(const std::string& s, const char* what, std::size_t line) {
  try {
    return std::stod(s);
  } catch (const std::exception&) {
    throw std::runtime_error("frames.csv 第 " + std::to_string(line) + " 行 " + what +
                             " 解析失败: '" + s + "'");
  }
}

inline std::int64_t toInt(const std::string& s, const char* what, std::size_t line) {
  try {
    return std::stoll(s);
  } catch (const std::exception&) {
    throw std::runtime_error("frames.csv 第 " + std::to_string(line) + " 行 " + what +
                             " 解析失败: '" + s + "'");
  }
}

inline std::vector<FrameStamp> readFramesCsv(const std::filesystem::path& csv) {
  std::ifstream in(csv);
  if (!in) throw std::runtime_error("无法打开 " + csv.string());

  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error(csv.string() + " 为空");

  const auto header = splitCsv(line);
  const std::size_t c_frame = columnOf(header, "frame_index");
  const std::size_t c_mono = columnOf(header, "monotonic_ns");
  const std::size_t c_unix = columnOf(header, "unix_ns");
  const std::size_t c_elapsed = columnOf(header, "elapsed_s");
  const std::size_t c_cam = columnOf(header, "camera_frame_id");
  const std::size_t c_dev = columnOf(header, "device_timestamp_ticks");

  std::vector<FrameStamp> stamps;
  std::size_t line_no = 1;
  while (std::getline(in, line)) {
    ++line_no;
    if (trim(line).empty()) continue;
    const auto cols = splitCsv(line);
    if (cols.size() < header.size()) {
      throw std::runtime_error("frames.csv 第 " + std::to_string(line_no) + " 行列数不足");
    }
    FrameStamp s;
    s.frame_index = toInt(cols[c_frame], "frame_index", line_no);
    s.monotonic_ns = toInt(cols[c_mono], "monotonic_ns", line_no);
    s.unix_ns = toInt(cols[c_unix], "unix_ns", line_no);
    s.elapsed_s = toDouble(cols[c_elapsed], "elapsed_s", line_no);
    s.camera_frame_id = toInt(cols[c_cam], "camera_frame_id", line_no);
    s.device_timestamp_ticks = toInt(cols[c_dev], "device_timestamp_ticks", line_no);
    stamps.push_back(s);
  }
  return stamps;
}

}  // namespace detail

inline Dataset Dataset::open(const std::filesystem::path& root) {
  if (!std::filesystem::is_directory(root)) {
    throw std::runtime_error("数据目录不存在: " + root.string());
  }

  Dataset ds;
  ds.root_ = root;
  ds.name_ = root.filename().string();

  const auto csv = root / "video" / "frames.csv";
  const auto avi = root / "video" / "video.avi";
  if (!std::filesystem::exists(csv)) throw std::runtime_error("缺少文件: " + csv.string());
  if (!std::filesystem::exists(avi)) throw std::runtime_error("缺少文件: " + avi.string());

  ds.stamps_ = detail::readFramesCsv(csv);
  if (ds.stamps_.empty()) throw std::runtime_error(csv.string() + " 没有数据行");

  ds.cap_ = std::make_unique<cv::VideoCapture>(avi.string(), cv::CAP_FFMPEG);
  if (!ds.cap_->isOpened()) throw std::runtime_error("无法打开视频: " + avi.string());

  ds.video_frames_ = static_cast<std::size_t>(ds.cap_->get(cv::CAP_PROP_FRAME_COUNT));
  ds.frame_size_ = cv::Size(static_cast<int>(ds.cap_->get(cv::CAP_PROP_FRAME_WIDTH)),
                            static_cast<int>(ds.cap_->get(cv::CAP_PROP_FRAME_HEIGHT)));

  const auto& first = ds.stamps_.front();
  const auto& last = ds.stamps_.back();
  const double span = last.elapsed_s - first.elapsed_s;
  if (ds.stamps_.size() > 1 && span > 0.0) {
    ds.nominal_fps_ = static_cast<double>(ds.stamps_.size() - 1) / span;
  }
  return ds;
}

inline bool Dataset::next(cv::Mat& frame, FrameStamp& stamp) {
  if (cursor_ >= stamps_.size()) return false;
  if (!cap_ || !cap_->isOpened()) throw std::runtime_error("视频未打开");

  if (!cap_->read(frame) || frame.empty()) {
    throw std::runtime_error("按 frames.csv 第 " + std::to_string(cursor_) +
                             " 行取帧失败（视频解码提前结束）");
  }
  stamp = stamps_[cursor_];
  if (frame_size_.width <= 0) frame_size_ = frame.size();
  ++cursor_;
  return true;
}

}  // namespace rm
