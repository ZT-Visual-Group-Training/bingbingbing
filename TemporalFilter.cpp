// ============================================================
// TemporalFilter.cpp —— 时序多数投票滤波器的实现
// ============================================================
#include "TemporalFilter.h"
#include <algorithm>
#include <cmath>

namespace {

// 求 deque 的中位数（多数投票用中位数而非均值：抗单帧异常）
float medianOf(const std::deque<float>& q) {
    if (q.empty()) return 0.f;
    std::vector<float> v(q.begin(), q.end());
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * 0.5f;
}

// 求标准差：衡量历史是否"稳定"。持续断裂的历史 ratio 波动很小；
// 忽远忽近/角度乱变的板波动大，不应修。
float stdevOf(const std::deque<float>& q) {
    if (q.size() < 2) return 0.f;
    double sum = 0.0;
    for (float x : q) sum += x;
    double mean = sum / q.size();
    double var = 0.0;
    for (float x : q) var += (x - mean) * (x - mean);
    return (float)std::sqrt(var / q.size());
}

// 用给定的新框高 H，围绕板中心 u(中轴)/v(高方向) 重建四角点。
// 与 ArmorDetector::matchLights 里规整四角点的构造方式完全一致，
// 保证 PnP 的四角顺序语义不变（左上→右上→右下→左下）。
void rebuildCornersWithHeight(Armor& a, float newH) {
    if (a.corners.size() != 4 || newH <= 1.f) return;
    cv::Point2f topMid    = (a.corners[0] + a.corners[1]) * 0.5f;
    cv::Point2f bottomMid = (a.corners[3] + a.corners[2]) * 0.5f;
    cv::Point2f center    = a.center;
    cv::Point2f u = topMid - bottomMid;              // 高方向（下→上）
    float hCur = cv::norm(u);
    if (hCur < 1.f) return;
    u *= (1.f / hCur);
    // 水平方向：用上边两点确定
    cv::Point2f wdir = a.corners[1] - a.corners[0];
    float wCur = cv::norm(wdir);
    if (wCur < 1.f) return;
    wdir *= (1.f / wCur);
    cv::Point2f halfV = u * (newH * 0.5f);
    cv::Point2f halfW = wdir * (wCur * 0.5f);
    // 保持中心不变，宽度不变，只把高度换成 newH
    a.corners[0] = center - halfW + halfV;
    a.corners[1] = center + halfW + halfV;
    a.corners[2] = center + halfW - halfV;
    a.corners[3] = center - halfW - halfV;
}

}  // namespace

void TemporalFilter::apply(std::vector<Armor>& armors, const ArmorDetector& det,
                           const CameraCalib& calib) {
    if (!p_.enabled) return;                 // 默认关闭：完全无副作用
    if (armors.empty()) {                    // 本帧无板：只让已有轨迹老化
        for (auto& t : tracks_) ++t.missed;
        return;
    }

    const size_t n = armors.size();
    std::vector<int> match(n, -1);           // 每块板匹配到的轨迹下标
    std::vector<bool> trackUsed(tracks_.size(), false);

    // ---- 第一步：跨帧关联，用中心点最近邻（贪心，一条轨迹只能用一次）----
    // 先把"距离最近"的候选排出来，避免先到先得把近的抢走。
    struct Cand { int a, t; float d; };
    std::vector<Cand> cands;
    for (size_t i = 0; i < n; ++i) {
        for (size_t t = 0; t < tracks_.size(); ++t) {
            float d = cv::norm(armors[i].center - tracks_[t].lastCenter);
            if (d <= p_.matchDistMax) cands.push_back({(int)i, (int)t, d});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& x, const Cand& y) { return x.d < y.d; });
    for (const auto& c : cands) {
        if (match[c.a] != -1 || trackUsed[c.t]) continue;
        match[c.a] = c.t;
        trackUsed[c.t] = true;
    }

    // ---- 第二步：逐块板判定 + 修正 ----
    for (size_t i = 0; i < n; ++i) {
        Armor& a = armors[i];
        float h1 = a.barH1, h2 = a.barH2;

        // 没记录灯条高度（旧数据/未填）→ 不处理，只更新轨迹位置
        if (h1 <= 1.f || h2 <= 1.f) {
            if (match[i] >= 0) { tracks_[match[i]].lastCenter = a.center;
                                 tracks_[match[i]].missed = 0; }
            continue;
        }
        float hMin = std::min(h1, h2), hMax = std::max(h1, h2);
        float ratio = hMin / hMax;           // 当前帧两根高度比（越接近1越对称）

        int ti = match[i];
        if (ti < 0) {                        // 没匹配到历史 → 新建轨迹，本帧不修
            TemporalFilter::Track nt;
            nt.lastCenter = a.center;
            nt.ratios.push_back(ratio);
            nt.heights.push_back(hMax);
            nt.missed = 0;
            tracks_.push_back(nt);
            continue;
        }

        Track& tr = tracks_[ti];
        tr.lastCenter = a.center;
        tr.missed = 0;

        bool fixed = false;
        float usedRatio = ratio, usedH = hMax;

        if ((int)tr.ratios.size() >= p_.minHistory) {
            float medRatio = medianOf(tr.ratios);
            float sd = stdevOf(tr.ratios);

            // 判据A（突然断裂）：历史一直对称(健康) + 当前突然变得不对称。
            bool sudden = p_.useSudden &&
                          (medRatio >= p_.healthyRatioMin && ratio < medRatio * p_.dropRatio);

            // 判据B（持续断裂）：历史 ratio 长期稳定落在中段区间。
            // 灯条整段视频都一长一短（真断裂/半遮挡）时，ratio 恒定且波动小；
            // 极端斜视角板 ratio≈0.34 落在下界之外，不会被误伤。
            bool persistent = p_.usePersistent &&
                              (medRatio >= p_.persistLow && medRatio <= p_.persistHigh &&
                               sd <= p_.stabilityMax && ratio <= p_.persistHigh &&
                               hMax <= p_.persistHMax);

            if (sudden || persistent) {
                // 修正值 = 当前帧较长的那根（断掉只会变短，长的可信）
                float newH = hMax;
                Armor backup = a;            // 备份，用于回退
                rebuildCornersWithHeight(a, newH);
                det.solveArmorPnP(a, calib); // 重算距离，保证和框一致
                bool accept = a.valid;
                // 安全检查：修正不能让距离跳变过大，也不能丢框
                if (p_.revertIfInvalid && backup.valid && accept) {
                    double d0 = backup.distance, d1 = a.distance;
                    if (d0 > 0.01 && std::fabs(d1 - d0) / d0 > 0.25) accept = false;
                }
                if (accept) {
                    fixed = true;
                    // 突然断裂：写入1.0(视作已修复)，让后续帧稳定保持修好的状态；
                    // 持续断裂：保留原始 ratio，否则历史变对称后下一帧就认不出断裂了。
                    usedRatio = sudden ? 1.f : ratio;
                    usedH = newH;
                } else {
                    a = backup;              // 回退：绝不让修正引入坏框
                }
            }
        }
        (void)fixed;

        // 写入"修正后"的状态到历史：让后续帧继续认为这块板是健康的，
        // 这样断裂持续多帧时每帧都能修，而不是修一帧后历史被污染。
        tr.ratios.push_back(usedRatio);
        tr.heights.push_back(usedH);
        if ((int)tr.ratios.size() > p_.historyLen) {
            tr.ratios.pop_front();
            tr.heights.pop_front();
        }
    }

    // ---- 第三步：老化，删除长期没匹配到的轨迹（防止板消失后串到新板）----
    // 注意：第二步可能 push_back 新轨迹，所以不能用旧的 trackUsed（会越界）。
    // 判断依据：一条轨迹若出现在 match[] 里说明本帧匹配到了（missed 已在第二步清零）；
    // 否则本帧没匹配到，missed++。新建的轨迹在第二步已设 missed=0，这里会++成1，无妨。
    for (size_t t = 0; t < tracks_.size();) {
        bool wasMatched = (std::find(match.begin(), match.end(), (int)t) != match.end());
        if (!wasMatched) ++tracks_[t].missed;
        if (tracks_[t].missed > p_.maxMiss)
            tracks_.erase(tracks_.begin() + t);
        else
            ++t;
    }
}
