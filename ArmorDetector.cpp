// ============================================================
// ArmorDetector.cpp —— 真正干活的代码（所有函数的实现）
//
// 类外定义成员函数时，函数名前要加 "ArmorDetector::"，
// 意思是"这个函数属于 ArmorDetector 类"。
// 下面按照检测的先后顺序排列，读代码就像读一条流水线。
// ============================================================
#include "ArmorDetector.h"
#include <algorithm>   // std::sort / std::swap / std::min / std::max
#include <cmath>       // std::fabs
#include <sstream>     // std::ostringstream，拼字符串用
#include <iomanip>     // std::setprecision，控制小数位

// ---------- 相机参数 → OpenCV 需要的矩阵 ----------
// solvePnP 要的是 3x3 内参矩阵和畸变向量，这里把结构里的数字拼起来
cv::Mat CameraCalib::cameraMatrix() const {
    // 形如 [ fx  0  cx ]
    //      [  0  fy cy ]
    //      [  0   0   1 ]
    return (cv::Mat_<double>(3, 3) <<
            fx,  0, cx,
             0, fy, cy,
             0,  0,  1);
}
cv::Mat CameraCalib::distCoeffs() const {
    return (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
}

// ---------- 第 1 步：生成颜色掩膜 ----------
// 【性能优化 2026-11-25】把原来一个函数拆成两段：
//   段A bgr→HSV（GaussianBlur + 颜色空间转换）——最耗时，红蓝只需各做一次
//   段B HSV→掩膜（inRange + 形态学）——便宜，红蓝各做各的
// detectLights 里会先把 bgr 转成一次 HSV，然后红蓝共用，省掉重复的段A。

// 段B：从已转好的 HSV 图生成某一种颜色的掩膜（注意：掩膜阈值不随缩放变，颜色阈值是H/S/V绝对值）
cv::Mat ArmorDetector::makeColorMaskFromHsv(const cv::Mat& hsv, ArmorColor color) const {
    cv::Mat mask;
    if (color == ArmorColor::Red) {
        // 红色跨 0/180 边界 → 两段 inRange 再或起来
        cv::Mat lower, upper;
        cv::inRange(hsv,
                    cv::Scalar(param_.redLowH1,  param_.minS, param_.minV),
                    cv::Scalar(param_.redHighH1, 255, 255), lower);
        cv::inRange(hsv,
                    cv::Scalar(param_.redLowH2,  param_.minS, param_.minV),
                    cv::Scalar(param_.redHighH2, 255, 255), upper);
        cv::bitwise_or(lower, upper, mask);
    } else {
        // 蓝色只跨一段
        cv::inRange(hsv,
                    cv::Scalar(param_.blueLowH,  param_.minS, param_.minV),
                    cv::Scalar(param_.blueHighH, 255, 255), mask);
    }

    // 形态学清理：开运算去掉孤立噪点，闭运算填补小孔
    // 【性能优化 2026-11-26·无损】闭运算原来是 3x3 核迭代 2 次 = 膨胀2次+腐蚀2次，
    // 对矩形核等价于单次 5x5 核闭运算（膨胀的 Minkowski 和性质）。
    // A/B 实测 300 帧掩膜逐像素 0 差异、轮廓数 0 差异；省一次全图遍历。
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelClose);
    return mask;
}

// 段A+B：对外保留原来的单图接口（image 模式/剖析工具用），行为完全等价
cv::Mat ArmorDetector::makeColorMask(const cv::Mat& bgr, ArmorColor color) const {
    CV_Assert(!bgr.empty());   // 进场先查图在不在，空的就当场报错

    cv::Mat src = bgr;
    if (src.channels() == 4)      cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);
    else if (src.channels() == 1) cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);

    cv::Mat blurred, hsv;
    cv::GaussianBlur(src, blurred, cv::Size(3, 3), 0);
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    return makeColorMaskFromHsv(hsv, color);
}

// ---------- 小工具：修正旋转矩形 ----------
// OpenCV 的 fitEllipse 返回的宽高谁长谁短不固定，这里统一成 height=长边
cv::RotatedRect ArmorDetector::adjustRec(const cv::RotatedRect& rec) {
    cv::RotatedRect r = rec;
    if (r.size.width > r.size.height) {
        std::swap(r.size.width, r.size.height);
        r.angle += 90.f;
    }
    while (r.angle >= 90.f)  r.angle -= 180.f;   // 角度归一到 [-90, 90)
    while (r.angle <  -90.f) r.angle += 180.f;
    return r;
}

// ---------- 灯条断裂修复：合并被暗缝切断的上下两截轮廓 ----------
// 官方视频里灯条中间偶有暗缝，颜色分割后一根灯条变成上下两个轮廓。
// 判据（全部满足才合并，宁缺毋滥）：
//   ① 两截 bbox 中心 x 差 <= breakMaxDx（同一竖直列）
//   ② 两截 bbox 在 y 上有正间隙，且 <= breakMaxGap（中间是窄暗缝，不是两根并列灯条）
//   ③ 两截都细长竖直（各自宽/高 <= breakMaxAspect）
//   ④ 两截拟合倾角差 <= breakMaxAngleDiff（同一根灯条倾角必然一致）
//   ⑤ 合并后总高度 <= breakMaxMergedH（防止把两根独立灯条首尾连起来）
void ArmorDetector::mergeBrokenContours(std::vector<std::vector<cv::Point>>& contours,
                                        const ArmorParam& P) {
    if (!P.mergeBrokenLight || contours.size() < 2) return;

    struct Piece {
        cv::Rect br;
        cv::Point2f center;
        float angle;   // adjustRec 归一后的倾角，接近 0=竖直
        bool usable;
    };
    std::vector<Piece> pc(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        pc[i].usable = false;
        if (contours[i].size() < 5) continue;
        float area = (float)cv::contourArea(contours[i]);
        if (area < P.lightAreaMin) continue;               // 太小的碎片不参与合并
        pc[i].br = cv::boundingRect(contours[i]);
        float w = (float)pc[i].br.width, h = (float)pc[i].br.height;
        if (h < 1.f || w / h > P.breakMaxAspect) continue; // 必须细长竖直
        if (h > P.breakMaxPieceH) continue;                // 单截不能是完整灯条（否则是另一块板的）
        cv::RotatedRect rr = adjustRec(cv::fitEllipse(contours[i]));
        pc[i].angle  = rr.angle;
        pc[i].center = rr.center;
        pc[i].usable = true;
    }

    std::vector<bool> consumed(contours.size(), false);  // 已被合并掉的段
    std::vector<std::vector<cv::Point>> merged;
    merged.reserve(contours.size());

    for (size_t i = 0; i < contours.size(); ++i) {
        if (consumed[i]) continue;
        int partner = -1;
        if (pc[i].usable) {
            float bestGap = 1e9f;
            for (size_t j = 0; j < contours.size(); ++j) {
                if (j == i || consumed[j] || !pc[j].usable) continue;
                float dx = std::fabs(pc[i].center.x - pc[j].center.x);
                if (dx > P.breakMaxDx) continue;

                // y 方向间隙：要求一上一下、中间有正间隙
                int aTop = pc[i].br.y, aBot = pc[i].br.y + pc[i].br.height;
                int bTop = pc[j].br.y, bBot = pc[j].br.y + pc[j].br.height;
                int gap;
                if (aBot <= bTop)      gap = bTop - aBot;   // i 在上
                else if (bBot <= aTop) gap = aTop - bBot;   // j 在上
                else                   continue;            // y 区间重叠 → 不是上下断裂
                if (gap < P.breakMinGap || gap > P.breakMaxGap) continue;

                float dAng = std::fabs(pc[i].angle - pc[j].angle);
                if (dAng > 90.f) dAng = 180.f - dAng;       // 角度环绕
                if (dAng > P.breakMaxAngleDiff) continue;

                int top = std::min(aTop, bTop), bot = std::max(aBot, bBot);
                if (bot - top > P.breakMaxMergedH) continue;

                if (gap < bestGap) { bestGap = (float)gap; partner = (int)j; }
            }
        }
        if (partner >= 0) {
            // 两段点集拼起来
            std::vector<cv::Point> combined = contours[i];
            combined.insert(combined.end(), contours[partner].begin(), contours[partner].end());
            merged.push_back(std::move(combined));
            consumed[partner] = true;
        } else {
            merged.push_back(std::move(contours[i]));
        }
    }
    contours.swap(merged);
}

// ---------- 第 2 步：检测灯条（红蓝都找）----------
std::vector<LightBar> ArmorDetector::detectLights(const cv::Mat& bgr,
                                                  const ArmorParam* paramOverride) const {
    std::vector<LightBar> lights;
    // 降分辨率时用缩放过阈值的参数副本；平时用自身 param_，行为不变
    const ArmorParam& P = paramOverride ? *paramOverride : param_;

    // 【性能优化 2026-11-25】HSV 只转一次，红蓝共用。
    // 原来是 makeColorMask 里各转一次（GaussianBlur+BGR2HSV 做了两遍），
    // 这两个全图操作占了整条流水线的 86%。转一次后红蓝各自只做便宜的 inRange+形态学。
    CV_Assert(!bgr.empty());
    cv::Mat src = bgr;
    if (src.channels() == 4)      cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);
    else if (src.channels() == 1) cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
    cv::Mat hsv;
    if (P.preBlur) {
        cv::Mat blurred;
        cv::GaussianBlur(src, blurred, cv::Size(3, 3), 0);
        cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    } else {
        // 【--fast 极速模式】跳过高斯模糊：实测省约2ms，框数/中心点几乎不变，
        // 代价是个别帧测距偏差可达0.23m（默认模式仍走上面的模糊分支保精度）。
        cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    }

    // 对红、蓝各做一遍掩膜→找轮廓→筛选，代码相同，用循环跑两次
    const ArmorColor colors[2] = { ArmorColor::Red, ArmorColor::Blue };
    for (ArmorColor color : colors) {
        cv::Mat mask = makeColorMaskFromHsv(hsv, color);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 【2026-11-25】先把被暗缝切断的灯条段合并，再做后面的面积/角度/宽高比筛选。
        // 修复前：两截各过不了高度关或配错对；修复后：拼成一截完整灯条再拟合。
        mergeBrokenContours(contours, P);

        for (size_t i = 0; i < contours.size(); ++i) {
            // 关 1：面积 + 点数
            float area = static_cast<float>(cv::contourArea(contours[i]));
            if (area < P.lightAreaMin ||
                static_cast<int>(contours[i].size()) <= P.minPoints) continue;

            // 椭圆拟合 → 旋转矩形，再修正
            cv::RotatedRect rec = adjustRec(cv::fitEllipse(contours[i]));

            // 关 1.5【新增·防背景误检】：灯条像素高度下限。
            // 远处灯管/反光只有几像素高，真板灯条在画面里一定够高。
            // 放在椭圆拟合之后，因为此时才知道长边(height)是多少。
            if (rec.size.height < P.lightHeightMin) continue;

            // 关 2：角度（灯条应接近竖直）
            if (std::fabs(rec.angle) > P.lightAngleMax) continue;

            // 关 3：长宽比 + 填充率
            float aspect = rec.size.width / rec.size.height;
            float crown  = area / rec.size.area();
            if (aspect > P.aspectRatioMax || crown < P.crownMin) continue;

            // 过关：放大一点，打包成 LightBar
            rec.size.height *= P.expandScale;
            rec.size.width  *= P.expandScale;
            LightBar bar;
            bar.rect   = rec;
            bar.center = rec.center;
            bar.height = rec.size.height;
            bar.angle  = rec.angle;
            bar.color  = color;
            lights.push_back(bar);
        }
    }
    return lights;
}

// ---------- 小工具：取灯条某一侧的两个角点，按"上、下"排好 ----------
// 【2026-11-24 修复三角形描边 bug】
// 旧算法：直接按 x 排序取最左/最右两点。对竖直灯条没问题，但灯条倾斜时
//   （如 -14°），最右两点可能 y 几乎相同（如 (842,743) 和 (850,741)），
//   导致四角点右边塌缩成一条 7px 短边，框画成三角形楔子（左高89 右高7）。
// 新算法：先按灯条【长轴方向投影】把 4 点分成"长轴两端"两组（各2点），
//   每组再按 x 取极值（左灯条取更左、右灯条取更右），最后按 y 定上/下。
//   这样无论灯条倾斜多少度，取的都是真正的"上端外侧点 + 下端外侧点"。
// leftSide=true 取左侧点（给左灯条用），false 取右侧点（给右灯条用）
std::vector<cv::Point2f> ArmorDetector::sidePoints(const LightBar& bar, bool leftSide) {
    cv::Point2f pts[4];
    bar.rect.points(pts);
    std::vector<cv::Point2f> v(pts, pts + 4);

    // 旋转矩形的两条轴方向（OpenCV: angle 是宽边相对 x 轴的夹角）
    double ang = bar.rect.angle * CV_PI / 180.0;
    cv::Point2f axisW((float)std::cos(ang), (float)std::sin(ang));    // 宽边方向
    cv::Point2f axisH((float)-std::sin(ang), (float)std::cos(ang));   // 高边方向
    // 长轴 = size 较大的那条边（灯条是细长条，长轴沿灯条长度方向）
    bool widthIsLong = bar.rect.size.width >= bar.rect.size.height;
    const cv::Point2f& longAxis = widthIsLong ? axisW : axisH;

    // 4 点投影到长轴，按投影值排序，分成两端两组
    std::vector<std::pair<float, cv::Point2f>> proj;
    proj.reserve(4);
    for (const auto& p : v) {
        proj.emplace_back(p.x * longAxis.x + p.y * longAxis.y, p);
    }
    std::sort(proj.begin(), proj.end(),
              [](const std::pair<float, cv::Point2f>& a,
                 const std::pair<float, cv::Point2f>& b) { return a.first < b.first; });

    std::vector<cv::Point2f> endA = {proj[0].second, proj[1].second};  // 长轴一端
    std::vector<cv::Point2f> endB = {proj[2].second, proj[3].second};  // 长轴另一端
    // 每组内按 x 取外侧极值：左灯条取更左(x小)，右灯条取更右(x大)
    auto pickOuter = [leftSide](const cv::Point2f& a, const cv::Point2f& b) {
        return leftSide ? (a.x < b.x ? a : b) : (a.x > b.x ? a : b);
    };
    cv::Point2f pA = pickOuter(endA[0], endA[1]);
    cv::Point2f pB = pickOuter(endB[0], endB[1]);
    // 按 y 定上/下：side[0]=上，side[1]=下
    cv::Point2f top = (pA.y < pB.y) ? pA : pB;
    cv::Point2f bot = (pA.y < pB.y) ? pB : pA;
    return {top, bot};
}

// ---------- 第 3 步：灯条配对 → 求四角点 ----------
// 【2026-11-23 重写】从"按 x 相邻贪心"改成"全局最佳匹配"：
// 旧算法只看相邻两根，谁在左边谁先配，会把数字板的左灯条抢给旁边被遮挡的板。
// 新算法：枚举所有灯条对 → 过硬性条件初筛 → 按"匹配分数"排序（高度差最重要）
//         → 从最像的一对开始配，一根灯条只能用一次。
std::vector<Armor> ArmorDetector::matchLights(const std::vector<LightBar>& lights) const {
    std::vector<Armor> armors;
    if (lights.size() < 2) return armors;

    // 一个候选配对：记录两根的索引 + 分数（分数越小越像同一块板）
    struct Pair { int i, j; float score; };
    std::vector<Pair> candidates;

    // 第一步：枚举所有灯条对 (i<j)，过硬性条件，算分数
    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            const LightBar& A = lights[i];
            const LightBar& B = lights[j];

            if (A.color != B.color) continue;                       // 颜色要一样
            float hMin = std::min(A.height, B.height);
            float hMax = std::max(A.height, B.height);
            float heightDiff = (hMax - hMin) / hMax;                // 高度差比例
            if (heightDiff > param_.pairHeightRatioMax) continue;   // 硬性：高度要接近
            if (std::fabs(A.center.y - B.center.y) > param_.pairYDiffMax) continue;   // 硬性：y 接近
            if (std::fabs(A.angle - B.angle) > param_.pairAngleDiffMax) continue;     // 硬性：倾角接近

            // 左灯条 = x 小的，右灯条 = x 大的
            const LightBar& L = (A.center.x < B.center.x) ? A : B;
            const LightBar& R = (A.center.x < B.center.x) ? B : A;
            float dist = R.center.x - L.center.x;
            if (dist < param_.pairDistMin) continue;                 // 硬性：不能重叠
            if (dist / hMax > param_.pairDistMaxRatio) continue;     // 硬性：不能离太远

            // 匹配分数（越小越像同一块板），三项都归一化，避免单位干扰：
            //   ① 高度差比例；② y 差/灯条高；③ 间距与"预期板宽"的偏差。
            // 第③项是关键：装甲板宽约是灯条高的 2.2 倍。过曝会把灯条撑长，
            // 但两根灯条的间距不会变，所以间距项能纠正过曝造成的误配。
            float avgH = (A.height + B.height) * 0.5f;
            float yDiffNorm = std::fabs(A.center.y - B.center.y) / hMax;
            float expectW = 2.2f * avgH;              // 预期板宽 = 2.2 × 灯条高
            float widthDiff = std::fabs(dist - expectW) / expectW;
            // 【2026-11-23 收口】新增第④项角度扣分。硬门槛放宽到 25° 后，
            // 角度不再"一刀切"，而是转成连续扣分：越平行越加分，越歪越扣分。
            // 这样真板（高度/间距都最匹配）依然胜出，歪扭的错配会被扣分压下去。
            float angleDiffNorm = std::fabs(A.angle - B.angle) / param_.pairAngleDiffMax;
            float score = 2.0f * heightDiff           // ① 高度差（比例）
                        + 1.0f * yDiffNorm            // ② y 差（归一化）
                        + 2.0f * widthDiff            // ③ 间距偏差（关键）
                        + 0.5f * angleDiffNorm;       // ④ 角度差（新，软扣分）
            candidates.push_back({(int)i, (int)j, score});
        }
    }

    // 第二步：按分数从优到劣排序（最像同一块板的排最前）
    std::sort(candidates.begin(), candidates.end(),
              [](const Pair& a, const Pair& b) { return a.score < b.score; });

    // 第三步：从最优对开始贪心选取，一根灯条只能用一次
    std::vector<bool> used(lights.size(), false);
    for (const Pair& c : candidates) {
        if (used[c.i] || used[c.j]) continue;   // 任一根已被用掉，跳过

        const LightBar& L = (lights[c.i].center.x < lights[c.j].center.x)
                          ? lights[c.i] : lights[c.j];
        const LightBar& R = (lights[c.i].center.x < lights[c.j].center.x)
                          ? lights[c.j] : lights[c.i];

        // 【2026-11-24 规整四角点】远板/斜视角下两根灯条常被拍得一长一短、
        // 中心带 y 差，若直接用 sidePoints 的外侧端点，上/下边会歪斜（"歪框"）。
        // 这里改用两灯条【中心连线】做中轴、统一高度重建规整四边形：
        // 框的位置仍由两灯条真实中心决定、宽度仍是真实间距（距离/检出不受影响），
        // 只是把高度对齐，避免一边高一边低把框拉歪。
        cv::Point2f C1 = L.center, C2 = R.center;
        cv::Point2f u = C2 - C1;                 // 中轴：从左到右
        float midW = cv::norm(u);
        if (midW < 1.f) continue;
        u *= (1.f / midW);
        cv::Point2f v(-u.y, u.x);                // 垂直于中轴 = 框的高方向
        float H = (L.height + R.height) * 0.5f;  // 统一高度取两根平均
        cv::Point2f halfV = v * (H * 0.5f);
        Armor a;
        a.color = L.color;
        // 【2026-11-26】记录配对的两根灯条高度，供时序滤波器判断是否断裂。
        // 只写入、不读取，对现有检测结果零影响（滤波器默认关闭时完全无感知）。
        a.barH1 = L.height;
        a.barH2 = R.height;
        // 顺时针顺序，这个顺序要和下面 objectPoints 严格对应！
        a.corners.push_back(C1 - halfV);   // 0 左上
        a.corners.push_back(C2 - halfV);   // 1 右上
        a.corners.push_back(C2 + halfV);   // 2 右下
        a.corners.push_back(C1 + halfV);   // 3 左下

        a.center = cv::Point2f(0.f, 0.f);
        for (const auto& p : a.corners) a.center += p;
        a.center *= 0.25f;

        // 【2026-11-23 收口】宽高比上下限双向检查：
        //   超上限 → 细长错框（远处/过曝只剩灯条尖，宽高比 6~16）
        //   低于下限 → 窄框（只罩住一根灯条，帧 121/901 那种 D=0.06m 的假框）
        // 真板约 2.4，斜视角压到 1.5 左右，所以下限取 1.15 留足余量。
        float boxW = cv::norm(a.corners[0] - a.corners[1]);  // 上边宽
        float boxH = cv::norm(a.corners[0] - a.corners[3]);  // 左边高
        if (boxH < 1.f) continue;
        float aspect = boxW / boxH;
        if (aspect < param_.armorAspectMin || aspect > param_.armorAspectMax) continue;

        used[c.i] = used[c.j] = true;
        armors.push_back(a);
    }
    return armors;
}

// ---------- 第 4 步：PnP 测距（本项目最核心的一段）----------
// 已知：装甲板 4 个角点在"真实世界"里的坐标 + 它们在"照片"里的像素坐标 + 相机参数
// 求：装甲板相对相机在哪里（tvec）、朝哪个方向（rvec）
void ArmorDetector::solveArmorPnP(Armor& armor, const CameraCalib& calib) const {
    if (armor.corners.size() != 4) return;   // 角点不全，放弃

    // 【世界点】装甲板四角在板自身坐标系里的真实坐标（单位：米，z=0 因为是平面）
    // 顺序必须和 corners 一一对应：左上→右上→右下→左下
    // 尺寸由 useSmallArmor 决定：默认小装甲板 135×56mm（高度=灯条长度）
    float w = param_.armorWidth()  * 0.5f;  // 半宽（当前选中尺寸）
    float h = param_.armorHeight() * 0.5f;  // 半高 = 灯条长度的一半
    std::vector<cv::Point3f> objectPoints = {
        cv::Point3f(-w,  h, 0),   // 0 左上
        cv::Point3f( w,  h, 0),   // 1 右上
        cv::Point3f( w, -h, 0),   // 2 右下
        cv::Point3f(-w, -h, 0)    // 3 左下
    };

    // 【图像点】就是检测到的四角点（像素坐标），直接用 armor.corners

    // 一行调用 OpenCV 的 PnP 求解器（迭代法，最通用）
    cv::solvePnP(objectPoints, armor.corners,
                 calib.cameraMatrix(), calib.distCoeffs(),
                 armor.rvec, armor.tvec, false, cv::SOLVEPNP_ITERATIVE);

    // tvec = (x, y, z)，单位米，z 基本就是"正前方距离"
    armor.distance = cv::norm(armor.tvec);   // 直线距离 = 向量长度

    // 【2026-11-23 新增·防离谱距离】合理性校验。
    // 框一旦退化（过窄/角点共线），PnP 会算出 0.06m 或 6.2m 这种不可能的值。
    // 以前无条件 valid=true 全部画出来，现在超区间的判为无效，
    // 主程序只画 valid 的框，假距离就不会再出现在画面上。
    if (armor.distance < param_.validDistMin || armor.distance > param_.validDistMax) {
        armor.valid = false;
        return;
    }
    // 再挡一类：PnP 解出 NaN/无穷（数值发散时会出现）
    if (!std::isfinite(armor.distance)) {
        armor.valid = false;
        return;
    }

    // 【2026-11-23 新增·防碎片假框】宽度自洽校验。
    // 真板的像素宽度应接近 "fx * 板宽 / 距离"（透视倾斜会略小，实测 0.85~1.0）。
    // 碎片误配（把数字笔画/一小段灯条当整板）实际宽度只有期望的一半，
    // 说明它和算出的距离对不上，是假框，判无效。
    float actualW = cv::norm(armor.corners[0] - armor.corners[1]);
    double z = armor.tvec[2];
    if (z > 0.01) {
        double expectW = calib.fx * param_.armorWidth() / z;
        if (expectW > 0 && actualW / expectW < param_.widthConsistencyMin) {
            armor.valid = false;
            return;
        }
    }
    armor.valid = true;
}

// ---------- 一步到位：把整条流水线串起来 ----------
std::vector<Armor> ArmorDetector::detect(const cv::Mat& bgr, const CameraCalib& calib,
                                         std::vector<LightBar>* outLights) const {
    // 【性能优化 2026-11-25】两步预处理，都把坐标还原回原图，保证测距/框位置不变：
    //   ① ROI 裁剪：只扫描画面下方（装甲板都在下半部），省掉上方像素操作
    //   ② 可选缩放：在更小的图上做掩膜/形态学/轮廓（默认关闭，见参数说明）
    float roiY0 = param_.detectRoiYMinRatio * bgr.rows;   // 裁剪起始 y
    cv::Mat roiMat = (roiY0 > 0.5f && roiY0 < bgr.rows)
                   ? bgr(cv::Rect(0, static_cast<int>(roiY0), bgr.cols,
                                  bgr.rows - static_cast<int>(roiY0)))
                   : bgr;
    const cv::Mat* base = &roiMat;

    float sc = param_.detectScale;
    cv::Mat work;
    ArmorParam scaledP;                       // 缩放过绝对像素阈值的参数副本
    const ArmorParam* lightParam = nullptr;
    if (sc > 0.f && sc < 1.f) {
        cv::resize(*base, work, cv::Size(), sc, sc, cv::INTER_AREA);
        base = &work;
        scaledP = param_.scaledFor(sc);
        lightParam = &scaledP;
    }

    std::vector<LightBar> lights = detectLights(*base, lightParam);   // 1+2 找灯条（在 ROI/缩放图上）

    // 坐标还原回原图：先 ÷sc 缩放还原，再 +roiY0 补回裁剪偏移
    if (base != &roiMat || &roiMat != &bgr) {
        float inv = (sc > 0.f && sc < 1.f) ? 1.f / sc : 1.f;
        for (auto& L : lights) {
            L.center.x *= inv;  L.center.y = L.center.y * inv + roiY0;
            L.rect.center.x *= inv;  L.rect.center.y = L.rect.center.y * inv + roiY0;
            L.rect.size.width  *= inv;
            L.rect.size.height *= inv;
            L.height *= inv;
        }
    }

    // 【2026-11-24】若调用方想要灯条（画识别点用），这里把它带出去（已是原图坐标）。
    if (outLights) *outLights = lights;
    std::vector<Armor> armors = matchLights(lights);    // 3 配对+四角点（原图坐标）
    for (auto& a : armors) solveArmorPnP(a, calib);     // 4 逐块 PnP 测距

    // 【2026-11-23 收口】出口处两道过滤，后面的画图/计数/输出看到的都是干净结果：
    //   ① PnP 无效（距离离谱/NaN）的板丢掉；
    //   ② ROI：装甲板中心 y 必须落在画面合理区间，挡掉背景灯箱误检。
    //      （真板在画面下半部 y≈0.6~0.75，背景灯箱在 y≈0.27）
    int imgH = bgr.rows;
    float yMin = param_.armorCenterYMinRatio * imgH;
    float yMax = param_.armorCenterYMaxRatio * imgH;
    std::vector<Armor> validArmors;
    for (const auto& a : armors) {
        if (!a.valid) continue;
        if (a.center.y < yMin || a.center.y > yMax) continue;
        validArmors.push_back(a);
    }
    return validArmors;
}

// ---------- 第 5 步：把结果画到图上 ----------
namespace {
// 带黑色底板的文字：即使两块板的标注靠得近，也不会糊成一团看不清。
// org 是文字左下角（基线点）。
static void putLabel(cv::Mat& img, const std::string& txt, cv::Point org,
                     double scale, cv::Scalar color, int thick = 1) {
    int base = 0;
    cv::Size sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &base);
    cv::rectangle(img, org + cv::Point(-2, -sz.height - 2),
                  org + cv::Point(sz.width + 2, base + 2),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, txt, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thick, cv::LINE_AA);
}
}  // namespace

cv::Mat ArmorDetector::drawResult(const cv::Mat& bgr,
                                  const std::vector<Armor>& armors,
                                  double elapsedMs,
                                  const std::vector<LightBar>& lights) {
    cv::Mat out = bgr.clone();

    // 【2026-11-24 新增】画灯条识别点：每根灯条画椭圆框 + 中心点 + 编号 L0/L1...
    // 只有调用方传了 lights 才画（默认空则跳过，行为与以前完全一致）。
    // 这样能看到"检测阶段到底识别到了哪些灯条"，而不只是最终装甲板。
    for (size_t i = 0; i < lights.size(); ++i) {
        const LightBar& L = lights[i];
        cv::Scalar lc = (L.color == ArmorColor::Red)
                      ? cv::Scalar(0, 255, 255)      // 红灯条用青色框
                      : cv::Scalar(255, 255, 0);     // 蓝灯条用蓝色框
        cv::Point2f p[4];
        L.rect.points(p);
        for (int k = 0; k < 4; ++k)
            cv::line(out, p[k], p[(k + 1) % 4], lc, 1);
        cv::circle(out, L.center, 4, cv::Scalar(0, 0, 255), -1);   // 灯条中心点
        cv::putText(out, "L" + std::to_string(i), L.center + cv::Point2f(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, lc, 1);
    }

    for (size_t i = 0; i < armors.size(); ++i) {
        const Armor& a = armors[i];
        cv::Scalar color = (a.color == ArmorColor::Red)
                         ? cv::Scalar(0, 255, 0)      // 红板用绿框
                         : cv::Scalar(0, 255, 255);   // 蓝板用黄框

        // 画四边形框：把 4 个角点依次连线
        for (int k = 0; k < 4; ++k) {
            cv::line(out, a.corners[k], a.corners[(k + 1) % 4], color, 2);
        }
        // 画 4 个角点（小圆点）
        for (const auto& p : a.corners) {
            cv::circle(out, p, 4, cv::Scalar(0, 0, 255), -1);
        }
        // 画中心点
        cv::circle(out, a.center, 5, cv::Scalar(255, 0, 255), -1);

        // 拼三维坐标文字：X/Y/Z（米）+ 距离
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "X:" << a.tvec[0] << " Y:" << a.tvec[1]
            << " Z:" << a.tvec[2] << " D:" << a.distance << "m";
        // 相邻编号的板上下错开，避免两块板的文字叠在一起
        int shift = labelShiftY(i);
        cv::Point txtOrg(cvRound(a.center.x - 88), cvRound(a.center.y - 18 + shift));
        putLabel(out, oss.str(), txtOrg, 0.5, cv::Scalar(255, 255, 255));
        putLabel(out, "Armor#" + std::to_string(i),
                 cv::Point(cvRound(a.center.x - 30), cvRound(a.center.y + 26 + shift)),
                 0.5, color);
    }
    // 左上角显示处理耗时
    std::ostringstream perf;
    perf << std::fixed << std::setprecision(1) << elapsedMs << " ms";
    cv::putText(out, "Time: " + perf.str(), cv::Point(12, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    return out;
}
