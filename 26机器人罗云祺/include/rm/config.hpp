#pragma once

#include <map>
#include <string>
#include <vector>

namespace rm
{

class Config
{
public:
  static Config load(const std::string & path);

  bool has(const std::string & key) const;
  // 四个带默认值的取值函数
  std::string getString(const std::string & key, const std::string & fallback = "") const;
  double getDouble(const std::string & key, double fallback = 0.0) const;
  int getInt(const std::string & key, int fallback = 0) const;
  bool getBool(const std::string & key, bool fallback = false) const;
  // 分隔符
  std::vector<double> getDoubles(const std::string & key) const;
  std::vector<int> getInts(const std::string & key) const;

  void set(const std::string & key, const std::string & value);

  const std::map<std::string, std::string> & raw() const { return kv_; }

private:
  std::map<std::string, std::string> kv_;
};

}
