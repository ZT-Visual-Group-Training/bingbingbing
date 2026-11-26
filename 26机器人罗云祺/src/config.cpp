#include "rm/config.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rm
{

namespace
{

// 去掉首尾空白
std::string trim(const std::string & s)
{
  size_t b = 0;
  size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// 把 "[1, 2, 3]" 或 "1,2,3" 变成 ["1","2","3"]
std::vector<std::string> tokens(const std::string & value)
{
  std::string t = value;
  for (char & c : t) {
    if (c == '[' || c == ']' || c == ',') c = ' ';
  }
  std::istringstream ss(t);
  std::vector<std::string> out;
  std::string tok;
  while (ss >> tok) out.push_back(tok);
  return out;
}

}

// 读整个文件，存成 map<string,string>
Config Config::load(const std::string & path)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);

  Config cfg;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
        // 去掉 # 后面的注释
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    line = trim(line);
    if (line.empty()) continue;  // 空行/纯注释行

    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;

    const auto key = trim(line.substr(0, colon));
    const auto value = trim(line.substr(colon + 1));
    if (key.empty()) continue;
    // 按字符串存
    cfg.kv_[key] = value;
  }
  return cfg;
}

bool Config::has(const std::string & key) const { return kv_.find(key) != kv_.end(); }

std::string Config::getString(const std::string & key, const std::string & fallback) const
{
  const auto it = kv_.find(key);
  return it == kv_.end() ? fallback : it->second;
}

// 转 double：找不到、为空、格式错都返回 fallback
double Config::getDouble(const std::string & key, double fallback) const
{
  const auto it = kv_.find(key);
  if (it == kv_.end() || it->second.empty()) return fallback;
  try {
    return std::stod(it->second);
  } catch (...) {
    return fallback;
  }
}

int Config::getInt(const std::string & key, int fallback) const
{
  return static_cast<int>(getDouble(key, static_cast<double>(fallback)));
}

bool Config::getBool(const std::string & key, bool fallback) const
{
  const auto it = kv_.find(key);
  if (it == kv_.end()) return fallback;
  const std::string v = it->second;
  if (v == "1" || v == "true" || v == "True" || v == "yes") return true;
  if (v == "0" || v == "false" || v == "False" || v == "no") return false;
  return fallback;
}

std::vector<double> Config::getDoubles(const std::string & key) const
{
  std::vector<double> out;
  const auto it = kv_.find(key);
  if (it == kv_.end()) return out;
  for (const auto & tok : tokens(it->second)) {
    try {
      out.push_back(std::stod(tok));
    } catch (...) {
      // 某个 token 不是数字就跳过
    }
  }
  return out;
}

std::vector<int> Config::getInts(const std::string & key) const
{
  std::vector<int> out;
  for (double v : getDoubles(key)) out.push_back(static_cast<int>(v));
  return out;
}

// 命令行覆盖：main 里 --dataset / --out 走这里
void Config::set(const std::string & key, const std::string & value) { kv_[key] = value; }

}
