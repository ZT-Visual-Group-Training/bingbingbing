#include "rm/dataset.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace rm
{

namespace
{

// 按分隔符切字符串
std::vector<std::string> split(const std::string & line, char sep)
{
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r') {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

}

bool Dataset::open(
  const std::string & video_path, const std::string & frames_csv, std::string * error)
{
  meta_.clear();
  cursor_ = 0;

  // 把整张 frames.csv 读进内存
  std::ifstream in(frames_csv);
  if (!in) {
    if (error) *error = "cannot open " + frames_csv;
    return false;
  }

  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto tok = split(line, ',');
    if (tok.size() < 4) continue;  // 至少要有 index / ns / unix / elapsed 四列
    try {
      FrameMeta m;
      m.index = std::stoi(tok[0]);  // 帧号
      m.monotonic_ns = std::stoll(tok[1]);  // 单调时钟纳秒（时间戳）
      m.elapsed_s = std::stod(tok[3]);  // 相对开始的秒数
      meta_.push_back(m);
    } catch (...) {
      continue;  // 某一行格式坏了就跳过
    }
  }

  if (meta_.empty()) {
    if (error) *error = "no rows in " + frames_csv;
    return false;
  }

  // 打开视频
  if (!cap_.open(video_path)) {
    if (error) *error = "cannot open video " + video_path;
    return false;
  }

  // 帧率、分辨率从视频容器读，用来算抽帧 stride
  fps_ = cap_.get(cv::CAP_PROP_FPS);
  size_ = cv::Size(
    static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
    static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
  return true;
}

bool Dataset::next(Frame & frame)
{
  if (cursor_ >= meta_.size()) return false;  // 帧表读完

  cv::Mat image;
  if (!cap_.read(image) || image.empty()) return false;  // 视频读完/解码失败

  const FrameMeta & m = meta_[cursor_];
  frame.index = m.index;
  frame.timestamp_ns = m.monotonic_ns;
  frame.elapsed_s = m.elapsed_s;
  frame.image = image;  // 浅拷贝，共享像素
  ++cursor_;
  return true;
}

}
