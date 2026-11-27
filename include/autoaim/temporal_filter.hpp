// 跨帧时序一致性确认：决定"这一帧该把哪个候选当成主目标"。
//
// ===========================================================================
// 为什么需要它
//
// 检测、配对、PnP 都是**逐帧独立**的：每帧只看这张图，不知道上一帧发生了什么。
// 于是两类问题无法用单帧信息解决：
//
//   1) 背景灯具偶尔凑成一个 135:56 的四边形，通过了所有几何约束 → 这一帧
//      它可能拿到最高分，直接变成"主目标"，画面上闪一个框。
//      —— 单帧里它和真装甲板长得一模一样，只能靠"它下一帧还在不在"来区分。
//
//   2) 真目标被短暂遮挡 1~3 帧 → 单帧看就是"消失了"，主目标被清空，
//      云台会甩回原位再重新找。实际上目标一直在，只是这几帧看不见。
//
// 所以加一层：维护"当前主目标"的历史状态，用**帧间一致性**而不是"单帧分数"
// 来决定谁是主目标。
//
// ===========================================================================
// 三个判据（缺一不可，各自的物理含义）
//
//   [1] 中心像素距离  gate_px
//       目标在 6 ms（166 fps）内移动不了几个像素，所以"上一帧的中心"和
//       "这一帧候选的中心"应当很近。背景误检每帧位置乱跳，靠这条分开。
//
//   [2] 四角宽度比  width_ratio
//       宽度(px) = fx × 0.135 / 距离，所以宽度比等价于"距离比"。
//       6 ms 内目标距离不会大幅变化，宽度比必须接近 1。这条同时挡住了
//       "把远处小目标和近处大目标配成同一个"的错误关联。
//
//   [3] 三维距离  gate_m（3D 冗余判据）
//       PnP 已经给出 tvec，直接在米制空间比一次。像素判据在近距离时不够灵敏
//       （目标大、同样的米数只移动几像素），米制判据补上这一点。
//
// ===========================================================================
// 状态机与对外语义（CSV 里 track_state 列就用这个编码）
//
//   0 NONE       本帧没有任何候选
//   1 CONFIRMED  连续 confirm_hits 帧关联上 → 这是主目标，坐标可以输出
//   2 PREDICTED  已确认目标本帧丢失，但在 confirm_max_misses 帧容忍期内 →
//                不甩丢，用 tracker 的预测位置续帧（坐标标记为低可信）
//   3 TENTATIVE  有候选但还没累计够帧数 → 只记录不输出，不参与主目标选择
//
// 快速切换通道：
//   如果当前主目标关联失败、而另一个候选连续出现 confirm_hits 帧，
//   立即切换过去。否则旧目标丢失后要等 max_misses 帧才允许换，
//   真·新目标会被无谓地抑制一段时间（"锁死旧目标"）。
//
// 与卡尔曼的分工（别混为一谈）：
//   本文件 = 判断"这个目标是不是真的、是不是同一个"（离散状态机）
//   tracker = 判断"这个真目标现在在哪、往哪走"（连续位置估计）
//
// 与检测/配对/PnP 的边界：
//   本层**只决定选哪个候选当主目标**，不修改任何检测阈值、不改 PnP 结果。
//   PnP 算出来什么样就是什么样，本层只是"从一堆候选里挑一个"。
// ===========================================================================
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace autoaim {

/// 一帧里的一个候选，喂给时序层做关联。
struct TrackCandidate {
  cv::Point2f center_px{0.f, 0.f};  // 装甲板中心的像素位置（四角平均）
  float width_px = 0.f;             // 四角四边形的像素宽度（灯条外缘到外缘）
  cv::Vec3d tvec{0, 0, 0};          // PnP 输出的相机系位置（米）
};

enum class TrackState : int {
  kNone = 0,       // 本帧无候选
  kConfirmed = 1,  // 已确认主目标
  kPredicted = 2,  // 丢失但容忍期内，用预测续帧
  kTentative = 3,  // 有候选但未确认
};

class TargetTracker {
 public:
  void configure(int confirm_hits, int max_misses, double gate_m, double gate_px,
                 double width_ratio) {
    confirm_hits_ = std::max(1, confirm_hits);
    max_misses_ = std::max(0, max_misses);
    gate_m_ = std::max(0.05, gate_m);
    gate_px_ = std::max(4.0, gate_px);
    width_ratio_ = std::max(1.01, width_ratio);
  }

  /// 每帧调用一次。
  /// @param cands 本帧所有通过 PnP 的候选
  /// @param state OUT: 本帧的时序状态
  /// @return 被选为主目标的候选下标；-1 表示本帧没有可输出的主目标
  int update(const std::vector<TrackCandidate>& cands, TrackState* state) {
    int best = -1;
    float best_dist = 1e18f;

    // ---- 1. 与历史目标关联（三个判据同时满足才算同一个目标）----
    if (has_target_) {
      for (size_t i = 0; i < cands.size(); ++i) {
        const float dpx = cv::norm(cands[i].center_px - last_center_px_);
        if (dpx > static_cast<float>(gate_px_)) continue;

        if (last_width_px_ > 1.f && cands[i].width_px > 1.f) {
          const float wr = std::max(cands[i].width_px, last_width_px_) /
                           std::min(cands[i].width_px, last_width_px_);
          if (wr > static_cast<float>(width_ratio_)) continue;
        }
        if (cv::norm(cands[i].tvec - last_tvec_) > gate_m_) continue;

        if (dpx < best_dist) {
          best_dist = dpx;
          best = static_cast<int>(i);
        }
      }
    }

    // ---- 2. 关联成功：累计命中，刷新状态 ----
    if (best >= 0) {
      ++hits_;
      if (hits_ > confirm_hits_) hits_ = confirm_hits_;
      misses_ = 0;
      challenger_ = 0;
      has_target_ = true;
      remember(cands[best]);
      if (state) *state = confirmed() ? TrackState::kConfirmed : TrackState::kTentative;
      return best;
    }

    // ---- 3. 关联失败 ----
    //
    // 没有历史目标时（相机刚启动、或上一个目标已确认丢失），本帧第一个候选
    // 直接"播种"成新的跟踪目标，命中计数从 1 开始。少了这一步，新目标永远
    // 进不了累计流程，状态会一直停在 tentative —— 这是第一版重写时漏掉的。
    if (!has_target_) {
      if (cands.empty()) {
        if (state) *state = TrackState::kNone;
        return -1;
      }
      has_target_ = true;
      hits_ = 1;
      misses_ = 0;
      challenger_ = 0;
      remember(cands.front());
      if (state) *state = confirmed() ? TrackState::kConfirmed : TrackState::kTentative;
      return 0;
    }

    ++misses_;

    // 快速切换通道：别的候选连续 confirm_hits 帧出现 → 立刻换目标，
    // 不能因为旧目标还在容忍期内就把新目标一直压着。
    if (!cands.empty()) {
      ++challenger_;
      if (challenger_ >= confirm_hits_) {
        hits_ = confirm_hits_;
        misses_ = 0;
        challenger_ = 0;
        remember(cands.front());
        if (state) *state = TrackState::kConfirmed;
        return 0;
      }
    } else {
      challenger_ = 0;
    }

    // ---- 4. 容忍期内：不甩丢，交给 tracker 预测续帧 ----
    if (misses_ <= max_misses_) {
      if (state) *state = TrackState::kPredicted;
      return -1;
    }

    // ---- 5. 真丢失：清空 ----
    reset();
    if (state) *state = cands.empty() ? TrackState::kNone : TrackState::kTentative;
    return -1;
  }

  bool confirmed() const { return has_target_ && hits_ >= confirm_hits_; }
  int hits() const { return hits_; }
  int misses() const { return misses_; }
  bool hasTarget() const { return has_target_; }

  void reset() {
    has_target_ = false;
    hits_ = 0;
    misses_ = 0;
    challenger_ = 0;
    last_center_px_ = cv::Point2f(0.f, 0.f);
    last_width_px_ = 0.f;
    last_tvec_ = cv::Vec3d(0, 0, 0);
  }

 private:
  void remember(const TrackCandidate& c) {
    last_center_px_ = c.center_px;
    last_width_px_ = c.width_px;
    last_tvec_ = c.tvec;
  }

  int confirm_hits_ = 3;
  int max_misses_ = 5;
  double gate_m_ = 0.6;      // 6 ms 内走不到 0.6 m（= 100 m/s）
  double gate_px_ = 60.0;    // 2 m 处 0.6 m ≈ 90 px，取 60 留余量
  double width_ratio_ = 1.4; // 距离变化 40% 以内视为同一目标

  bool has_target_ = false;
  int hits_ = 0;
  int misses_ = 0;
  int challenger_ = 0;
  cv::Point2f last_center_px_{0.f, 0.f};
  float last_width_px_ = 0.f;
  cv::Vec3d last_tvec_{0, 0, 0};
};

}  // namespace autoaim
