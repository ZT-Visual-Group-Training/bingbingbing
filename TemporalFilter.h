// ============================================================
// TemporalFilter.h —— 时序多数投票滤波器（2026-11-26）
//
// 解决什么问题：
//   视频里某根灯条中间有暗缝、断成两截时，只检测到短的那一截，
//   配对画框取"两根平均高度"，短截把框压歪。
//
// 为什么用"时序"而不是"同一帧两根比高度"：
//   上一轮已验证：单帧看"两根差多少"无法区分两种情况——
//     ① 灯条断了（应该修）   ② 车斜着把灯条拍短了（不能修）
//   它们在单帧里特征一样。但放到"时间轴"上就完全不同：
//     ① 断灯条：这根灯条前几帧都正常，突然这一帧变短；
//     ② 斜视角板：这根灯条前几帧就一直短，不是突然事件。
//   所以用"这根灯条相对它自己的历史，是不是突然变短"来判定。
//
// 关键设计：历史只用来"判定是否断裂"，不提供修正数值。
//   修正值取"当前帧两根里较长的那根"——因为断掉的那根只会变短、
//   不会变长，长的那根就是可信的真实高度；且它是当前帧实测值，
//   不受车忽远忽近带来的透视缩放影响。
//
// 小白提示：这是独立模块，默认关闭(enabled=false)。
//   它不改动 ArmorDetector 的检测逻辑，只在 detect() 之后做后处理。
// ============================================================
#ifndef TEMPORAL_FILTER_H
#define TEMPORAL_FILTER_H

#include "ArmorDetector.h"
#include <vector>
#include <deque>

// 滤波器的可调参数（都给保守默认值，需 A/B 验证后才考虑默认开启）
struct TemporalParam {
    bool  enabled          = false;  // 总开关：默认关闭，行为=原来的检测器
    int   historyLen       = 7;      // 每块板保留最近多少帧的高度比历史
    int   minHistory       = 3;      // 至少积累几帧历史才允许修正（前几帧只看不修）
    float matchDistMax     = 60.f;   // 跨帧关联：中心点位移超过这个像素数视为不同板
    float healthyRatioMin  = 0.75f;  // 【突然断裂判据】历史比低于此值视为斜视角板
    float dropRatio        = 0.80f;  // 【突然断裂判据】当前比 < 历史中位×此值 → 突然变短
    // 【持续断裂判据】灯条整段视频都一长一短(非突发)时用这个：
    //   历史比稳定落在 [persistLow, persistHigh] 且波动很小 → 判为持续断裂 → 取较长根修框高
    //   下界挡住极端斜视角板(ratio≈0.34)，上界挡住正常对称板(>0.78)
    float persistLow       = 0.54f;
    float persistHigh      = 0.78f;
    float persistHMax      = 55.f;   // 较长灯条像素高度上限：真断裂的是远处小板(≈42px)，
                                    // 近处大板(≈89px)一长一短多是斜视角真实透视，不能修
    float stabilityMax     = 0.08f;  // 历史 ratio 标准差上限：越大说明越不稳定，不修
    bool  usePersistent    = true;   // 启用持续断裂判据（这是修第157帧实际需要的）
    bool  useSudden        = false;  // 突然断裂判据（本视频无突发断裂，默认关）
    int   maxMiss          = 5;      // 连续多少帧没匹配到就删除该轨迹（防止串板）
    bool  revertIfInvalid  = true;   // 修正后若 PnP 失效/距离跳变过大则回退，绝不引入新坏框
};

// 时序滤波器：持有跨帧状态，逐帧调用 apply()
class TemporalFilter {
public:
    explicit TemporalFilter(const TemporalParam& p = TemporalParam{}) : p_(p) {}

    // 对一帧的装甲板结果做时序修正（原地修改 armors）。
    // det/calib 用于修正四角点后重新解算 PnP，保证距离和框一致。
    void apply(std::vector<Armor>& armors, const ArmorDetector& det,
               const CameraCalib& calib);

    const TemporalParam& param() const { return p_; }
    void reset() { tracks_.clear(); }   // 换视频/跳帧时清空历史

private:
    struct Track {
        cv::Point2f lastCenter;       // 上一帧这块板的中心（用于关联下一帧）
        std::deque<float> ratios;     // 修正后采用的高度比历史 min/max
        std::deque<float> heights;    // 修正后采用的框高历史
        int missed = 0;               // 连续未匹配帧数
    };
    std::vector<Track> tracks_;
    TemporalParam p_;
};

#endif // TEMPORAL_FILTER_H
