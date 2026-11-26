// ============================================================
// ArmorDetector.h —— 装甲板检测器的"说明书"（只放声明/类型定义）
//
// 这是整套框架的核心头文件。它定义了 4 样东西：
//   ① CameraCalib  相机标定参数（焦距、主点、畸变）—— PnP 要用
//   ② ArmorParam   所有可调阈值 + 装甲板真实尺寸 —— 调参用
//   ③ LightBar / Armor  两个数据结构（灯条 / 装甲板结果）
//   ④ ArmorDetector 类 —— 对外提供检测能力
//
// 小白提示：.h 文件负责"有哪些东西"，具体"怎么做"在 ArmorDetector.cpp
// ============================================================
#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H

#include <opencv2/opencv.hpp>   // cv::Mat / cv::RotatedRect / cv::solvePnP 等都在这里
#include <vector>

// ---------- 颜色枚举：给红、蓝各一个名字，代码里不用裸写 0/1 ----------
enum class ArmorColor { Red = 0, Blue = 1 };

// ------------------------------------------------------------
// ① 相机标定参数（Camera Calibration）
//
// 什么是"标定参数"？相机把三维世界拍成二维照片时会"压缩+扭曲"，
// 这组数字描述"怎么压、怎么扭"，PnP 靠它把二维反算回三维。
//
// 【2026-11-23 已接入官方真实内参】来自官方数据包的
// calibration/calibration.yaml（分辨率 1440x1080）。写进默认值后，
// 直接运行即生效，不必每次 --camera。需要覆盖时仍可 --camera xxx.yaml。
// 注意：用内置合成图(640x360)自测时这组内参不匹配，距离数字会不准，
//       要看准确距离请用真实考核视频。
// ------------------------------------------------------------
struct CameraCalib {
    // 真实内参（官方标定值）
    double fx = 2334.310067789214, fy = 2335.8999103578203;  // 焦距（像素）
    double cx = 722.621333492868,  cy = 534.6208330554873;   // 主点（光心像素位置）
    // 畸变系数 k1,k2,p1,p2,k3（来自官方 distort_coeffs）
    double k1 = -0.08000880545053471, k2 = 0.08567503445336759,
           p1 = 0.002790303946222533, p2 = -0.0007315334877910064, k3 = 0;

    cv::Mat cameraMatrix() const;    // 由上面数字拼成 3x3 内参矩阵（实现在 .cpp）
    cv::Mat distCoeffs() const;      // 由畸变系数拼成 5x1 向量
};

// ------------------------------------------------------------
// ② 检测参数：把所有可调阈值集中在一起，调参只改这里
// ------------------------------------------------------------
struct ArmorParam {
    // --- 颜色阈值（HSV 空间，H 范围 0~180，S/V 范围 0~255）---
    // 红色跨 0/180 边界，要取两段并集
    // 【2026-11-23 按真实考核视频校准】灯条是橙黄/橙红色，实测 H≈12~18，
    // 原来的 high1=10 只覆盖正红，把灯条全漏掉（全视频检出率仅 5%）。
    // 放宽到 30 覆盖橙黄灯条，实测全视频检出率升到约 95%。
    // 【2026-11-24 用户在 HSV 调试器逐帧调校后抄回】：redHighH1 30->32, minV 80->182
    int redLowH1 = 0, redHighH1 = 32;
    int redLowH2 = 170, redHighH2 = 180;
    // 蓝色在 H≈120 附近，取一段
    int blueLowH = 100, blueHighH = 130;
    int minS = 100;                  // 饱和度下限：颜色要够鲜艳
    int minV = 182;                  // 亮度下限：用户在第900帧调到182（只留很亮的像素）

    // --- 灯条筛选阈值（你之前 demo5 学过的那套）---
    float lightAreaMin = 30.f;       // 轮廓面积下限，太小是噪点
    int   minPoints     = 5;         // 点数下限，fitEllipse 至少要 5 个点
    float lightAngleMax = 40.f;      // 灯条偏离竖直的最大角度（度）
    float aspectRatioMax = 0.5f;     // 宽/高 上限，灯条应细长
    float crownMin = 0.35f;          // 填充率下限：轮廓面积/外接矩形面积
    float expandScale = 1.1f;        // 灯条矩形放大系数（+10%，框住光晕）

    // 【2026-11-23 新增·漏洞收口】灯条像素高度下限。
    // 背景里的场馆灯管、远处反光，只有几个像素高，面积/点数也可能碰巧过关，
    // 但真装甲板的灯条在画面里一定有相当高度。低于此高度一律不当灯条，
    // 从源头掐掉"背景误检"（帧 501/1051/1301 右上角那种假框）。
    // 注意：这是"像素"高度，跟装甲板物理尺寸(米)无关。
    float lightHeightMin = 20.f;

    // --- ★新增★ 灯条配对阈值：判断两根灯条是不是同一块板 ---
    // 【2026-11-23 校准】放宽以识别斜视角装甲板：车斜着时两根灯条在画面里
    // 一端高一端低（y 差可达 80px），暗灯条椭圆拟合偏短（高度差约 63%）。
    // 原来的 0.6/40 会把这种真板漏掉；放宽后靠下面的宽高比闸门兜底防错框。
    float pairHeightRatioMax = 0.7f; // 两灯条高度差比例上限（差太多不是一对）
    float pairYDiffMax = 90.f;       // 两灯条中心 y 差上限（像素）斜视角实测达83px
    // 【2026-11-23 收口】15→25：真板在斜视角下两灯条朝相反方向倾斜，
    // 角度差实测达 17~19°（帧900/750），旧的 15° 硬门槛把真板一刀切掉，
    // 算法只能退而配错相邻的碎片灯条。放宽后靠评分里的角度扣分保持择优能力。
    float pairAngleDiffMax = 25.f;   // 两灯条倾角差上限（度）
    float pairDistMin = 10.f;        // 两灯条中心最小间距（像素）
    // 【2026-11-23 收口】8→4：真板间距约 2.2 倍灯条高，
    // 8 倍太松，会让"车身左侧灯条 + 右侧灯条"这种横跨整车的误配通过（帧 751）。
    float pairDistMaxRatio = 4.f;    // 间距/灯条高度 的上限（离太远不成对）
    // 真装甲板宽高比：135mm / 56mm ≈ 2.4，斜视角会被压到 1.5 左右。
    // 上限挡"细长错框"(宽高比 6~16)，下限挡"窄框"(只罩住一根灯条，帧 121/901)。
    float armorAspectMin = 1.15f;    // 框宽高比下限：比这还窄就不是完整板
    float armorAspectMax = 3.2f;     // 框宽高比上限

    // --- ★装甲板真实物理尺寸（单位：米）---
    // 【2026-11-23 按考核数据更新】
    // 小装甲板：宽 135mm × 高 56mm；其中"高 56mm"取的是灯条长度。
    // 原因：PnP 的四角点，上边=两灯条顶端、下边=两灯条底端，
    //       世界坐标 y 方向(armorHeight)对应的正是灯条长度，不是整板含边框高度。
    float smallArmorWidth  = 0.135f;  // 小板宽（两灯条外侧之间）
    float smallArmorHeight = 0.056f;  // 小板高 = 灯条长度 56mm

    // 大装甲板（旧数据，保留备用）：425mm × 245mm
    float bigArmorWidth    = 0.425f;
    float bigArmorHeight   = 0.245f;

    // 当前用哪套：true=小装甲板（考核默认），false=大装甲板
    bool useSmallArmor = true;

    // 兼容旧接口：armorWidth/armorHeight 仍可被外部读取，值=当前选中的尺寸
    float armorWidth()  const { return useSmallArmor ? smallArmorWidth  : bigArmorWidth; }
    float armorHeight() const { return useSmallArmor ? smallArmorHeight : bigArmorHeight; }

    // 【性能优化】返回一份"把绝对像素阈值按比例缩放"的参数副本，供降分辨率检测用。
    // 比值类阈值（宽高比、填充率、归一化距离）不随分辨率变，保持原值。
    ArmorParam scaledFor(float sc) const {
        ArmorParam p = *this;
        p.lightAreaMin    = lightAreaMin * sc * sc;   // 面积按平方缩
        p.lightHeightMin  = lightHeightMin * sc;      // 线性像素长度
        p.pairYDiffMax    = pairYDiffMax * sc;        // 像素距离
        p.pairDistMin     = pairDistMin  * sc;
        p.minPoints       = std::max(5, static_cast<int>(minPoints * sc)); // fitEllipse 至少5点
        return p;
    }

    // 【2026-11-23 新增·漏洞收口】PnP 距离合理性区间（米）。
    // PnP 在框退化(过窄/点共线)时会算出 0.06m、6.2m 这种一眼假的距离，
    // 以前算完就画，现在超出区间的直接判为无效、不画框。
    // 考核场景里小车一般离相机 0.3~8 米，按这个范围兜底。
    float validDistMin = 0.3f;
    float validDistMax = 8.0f;

    // 【2026-11-23 新增·防背景误检】装甲板中心在画面中的纵向位置区间（比例 0~1）。
    // 考核视频里小车和装甲板都在画面下半部（y≈0.6~0.75），
    // 而背景灯箱在右上角（y≈0.27）。限制这个区间可干净挡掉背景假框。
    // 用比例而非像素，换分辨率也不用改。设 0 表示不限制下界。
    float armorCenterYMinRatio = 0.45f;  // 中心 y 至少在画面 45% 高度处
    float armorCenterYMaxRatio = 1.0f;   // 上界 1.0 = 不限制

    // 【2026-11-23 新增·防碎片假框】PnP 宽度自洽比值下限。
    // 原理：已知板真实宽 0.135m，针孔模型下"期望像素宽 = fx * 0.135 / 距离"。
    // 真板实测"实际宽/期望宽"≈0.85~1.0；碎片假框（把一小段灯条当整板）只有 0.5。
    // 低于此比值说明框和距离不自洽，是误配，判无效。
    // 【2026-11-24 调整 0.6->0.5】修复 sidePoints 后四角点还原了斜视角真板的
    //   真实形状，斜视角板的比值会降到 0.55 左右（正对镜头仍是 0.85~1.0）。
    //   0.6 会把斜视角真板（如帧120）误杀；实测新算法下阈值 0.5 通过 98% 候选，
    //   碎片假框仍有 armorAspectMin / ROI / 左右高度对称等多道关卡拦截。
    float widthConsistencyMin = 0.5f;

    // 【性能优化 2026-11-25】检测缩放比例：1.0=原分辨率；0.5=半分辨率检测（约4倍加速）。
    // 注意：A/B 实测降分辨率会系统性丢失远处小板（灯条缩到6px、点数不足），
    // 所以默认 1.0 保持检测完整性；0.5 仅作"极限速度模式"备选，不默认启用。
    float detectScale = 1.0f;

    // 【性能优化 2026-11-26】检测前是否做高斯模糊降噪。
    //   true(默认)=保留原行为，测距精度最佳；
    //   false=去掉，省约2ms。1240帧严格A/B：框数0差异、中心点平均误差0.14px，
    //          但个别帧PnP测距偏差可达0.23m，故只在 --fast 极速模式启用。
    bool preBlur = true;

    // 【性能优化 2026-11-26】检测 ROI：只扫描画面下方这个比例以上的区域。
    // 装甲板中心实测都在 y≈0.6~0.75，出口 armorCenterYMinRatio=0.45 已先滤掉更靠上的板，
    // 所以从 45% 高度开始扫描可省掉 45% 像素操作，且检测结果完全不变。
    // A/B 验证（第二段 500 帧）：ROI 0.35→0.50 框数 0 差异、中心点最大误差 0.01px；
    // 取 0.45 留足余量，耗时 13.5→11.8ms/帧（约快 13%）。
    // 设 0.0 表示扫描整幅图（回到原始行为）。
    float detectRoiYMinRatio = 0.45f;

    // --- ★2026-11-25 新增：灯条断裂修复（官方视频里灯条中间有暗缝，断成两截）---
    // 原理：过曝/拍摄伪影让一根灯条中间出现一小段暗区，颜色分割后变成上下两个轮廓。
    // 若直接当两根灯条，配对会错。这里在拟合椭圆之前，把"同一竖直列、中间隔一条
    // 很窄暗缝、倾角一致、都细长"的两个轮廓点集合并成一根再拟合。
    bool  mergeBrokenLight   = true;    // 总开关；设 false 可回到修复前行为（做 A/B 用）
    float breakMaxDx         = 10.f;    // 两截中心 x 差上限（像素），太大说明不是同一列
    float breakMinGap        = 1.f;     // 两截 bbox 的最小 y 间隙（0=重叠，不算断裂）
    float breakMaxGap        = 14.f;    // 两截 bbox 的最大 y 间隙（暗缝宽度）
    float breakMaxAngleDiff  = 6.f;     // 两截倾角差上限（度），一根灯条倾角必然一致
    float breakMaxAspect     = 0.75f;   // 两截各自的宽/高上限，确保细长竖直
    float breakMaxPieceH     = 48.f;    // 单截高度上限：断裂后的每截都比完整灯条短
    float breakMaxMergedH    = 75.f;    // 合并后总高度上限（正常灯条约65px），挡住两块板的灯条相连
};

// ------------------------------------------------------------
// ③ 两个数据结构：把一组相关数据打包
// ------------------------------------------------------------

// 一根灯条
struct LightBar {
    cv::RotatedRect rect;            // 拟合出的旋转矩形（位置+宽高+角度）
    cv::Point2f center;              // 中心点（其实 rect.center 也有，存一份方便用）
    float height = 0.f;              // 灯条长度（矩形的长边）
    float angle  = 0.f;              // 偏离竖直的角度
    ArmorColor color = ArmorColor::Red;
};

// 一块装甲板（检测的最终结果）
struct Armor {
    std::vector<cv::Point2f> corners; // 四个角点（顺时针：左上→右上→右下→左下）
    cv::Point2f center;               // 中心点（四角平均）
    ArmorColor color = ArmorColor::Red;
    cv::Vec3d tvec = cv::Vec3d(0,0,0);// PnP 解出的三维位置（米），z 约等于"距离"
    cv::Vec3d rvec = cv::Vec3d(0,0,0);// 装甲板的朝向（旋转向量）
    double distance = 0.0;            // 到相机的直线距离（米）= tvec 的长度
    bool valid = false;               // PnP 是否解算成功

    // 【2026-11-26 时序多数投票用】配对的两根灯条高度（画框时 H=(h1+h2)/2）。
    // 现有代码没有任何地方读这两个字段，默认 0，完全向后兼容；
    // 只有打开的时序滤波器才会用到，用来判断"这根灯条是不是突然变短了"。
    float barH1 = 0.f;
    float barH2 = 0.f;
};

// 标注文字的上下错开量：画面里相邻编号的两块板，一个往上、一个往下，
// 这样即使两块板挨得近，各自的坐标文字也不会叠在一起看不清。
// drawResult（三维坐标/Armor#）和 main.cpp 的 IMU 叠加都调用它，保持一致。
inline int labelShiftY(size_t idx) {
    return (idx % 2 == 0) ? -42 : 46;
}

// ------------------------------------------------------------
// ④ 检测器类：把 5 个相关函数打包成一个"工具箱"
// ------------------------------------------------------------
class ArmorDetector {
public:
    explicit ArmorDetector(const ArmorParam& p = ArmorParam{}) : param_(p) {}

    // 生成指定颜色的掩膜（红 or 蓝），白色=目标区域，黑色=背景
    cv::Mat makeColorMask(const cv::Mat& bgr, ArmorColor color) const;

    // 检测所有灯条（红+蓝都找），返回灯条列表
    // 【性能优化】可选 paramOverride：降分辨率检测时传入缩放过阈值的参数副本；
    // 不传（默认 nullptr）则用自身 param_，行为与以前完全一致。
    std::vector<LightBar> detectLights(const cv::Mat& bgr,
                                       const ArmorParam* paramOverride = nullptr) const;

    // 把灯条两两配对成装甲板，并求出四角点（还不算三维）
    std::vector<Armor> matchLights(const std::vector<LightBar>& lights) const;

    // 对一块装甲板做 PnP，把结果写回 armor（tvec/rvec/distance）
    void solveArmorPnP(Armor& armor, const CameraCalib& calib) const;

    // 一步到位：检测灯条→配对→PnP，返回所有装甲板
    // 【2026-11-24】可选参数 outLights：传入指针可顺带把检测到的灯条取出来画点。
    // 不传（默认 nullptr）则行为完全不变，不影响任何现有调用。
    std::vector<Armor> detect(const cv::Mat& bgr, const CameraCalib& calib,
                              std::vector<LightBar>* outLights = nullptr) const;

    // 画图：四角点、中心点、三维坐标、耗时，返回画好的图
    // 【2026-11-24】可选参数 lights：传入灯条列表则额外画出灯条椭圆框+中心点+编号，
    // 方便看到"识别的点"。不传（默认空）则和以前一样只画装甲板，行为不变。
    static cv::Mat drawResult(const cv::Mat& bgr,
                              const std::vector<Armor>& armors,
                              double elapsedMs,
                              const std::vector<LightBar>& lights = {});

    const ArmorParam& param() const { return param_; }   // 只读参数入口

    // 【调试用】暴露断裂合并函数，供诊断工具验证（正常 Detection 不需直接调用）
    static void mergeBrokenContoursForTest(std::vector<std::vector<cv::Point>>& c,
                                           const ArmorParam& p) { mergeBrokenContours(c, p); }

private:
    // 【性能优化 2026-11-25】从"已经转好的 HSV 图"直接生成掩膜。
    // 红蓝检测共用同一份 HSV，避免对同一帧重复做 GaussianBlur+BGR2HSV（占原耗时86%的大头）。
    // 对外接口 makeColorMask(bgr,color) 保持不变，内部改为"转一次HSV + 调这个函数"。
    cv::Mat makeColorMaskFromHsv(const cv::Mat& hsv, ArmorColor color) const;

    // 小工具：修正 OpenCV 旋转矩形，让 height 一定代表"长边"
    static cv::RotatedRect adjustRec(const cv::RotatedRect& rec);
    // 【2026-11-25】把同一根灯条被暗缝切断的两个轮廓合并成一个（原地修改 contours）
    static void mergeBrokenContours(std::vector<std::vector<cv::Point>>& contours,
                                    const ArmorParam& P);
    // 小工具：从一根灯条里取"左侧或右侧"的两个角点，按上、下排好
    static std::vector<cv::Point2f> sidePoints(const LightBar& bar, bool leftSide);

    ArmorParam param_;
};

#endif // ARMOR_DETECTOR_H
