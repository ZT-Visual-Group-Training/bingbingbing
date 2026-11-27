// Stage 2: pair light bars into armour plates.
#pragma once

#include "autoaim/config.hpp"
#include "autoaim/types.hpp"

#include <vector>

namespace autoaim {

class ArmorMatcher {
 public:
  explicit ArmorMatcher(const DetectorConfig& cfg) : cfg_(cfg) {}

  /// Pair bars into armour candidates, sorted by ascending score (best first).
  std::vector<Armor> match(const std::vector<LightBar>& bars) const;

 private:
  /// Corner order is TL, BL, BR, TR. The corners are the OUTER vertical edges of
  /// the two bars, because the 135 mm spans the outer faces of the bars and the
  /// 56 mm is the bar length (the task statement says "height = light bar
  /// length").
  Armor build(const LightBar& left, const LightBar& right, float score,
              float center_ratio, float length_ratio, float angle_diff) const;

  static float angleDiffDeg(float a, float b);

  DetectorConfig cfg_;
};

}  // namespace autoaim
