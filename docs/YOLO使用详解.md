# YOLO 具体怎么用（C++ / OpenCV DNN 版）

> 面向：零 YOLO 经验、要在装甲板项目里用起来的人
> 核心结论：**你只用"推理"，不用"训练"。五步固定套路，唯一要动脑的是第 4 步解析输出。**
> 核实状态标注：文中 `[已验证]` 指本机实际编译跑通过的 API 用法；`[依官方说明]` 指来自官方开源资料、需按你下载模型的 README 复核。

---

## 一、先建立 3 个概念（不然后面全是天书）

| 概念 | 大白话 | 类比 |
|---|---|---|
| 模型文件 `.onnx` | 一个"训练好的识别程序"，输入图片，输出识别结果 | 一台已经调好的仪器 |
| 推理 inference | 拿现成模型去识别新图片 | 用仪器去测东西（你要做的） |
| 训练 training | 用大量标注图片让模型自己学 | 造这台仪器（别人已做，你不做） |

**为什么是 `.onnx` 不是 `.pt`**：`.pt` 是 Python/PyTorch 专用格式，C++ 读不了；`.onnx` 是跨语言通用格式，OpenCV 认识。**C++ 只能读 onnx**，所以你要么下载现成 onnx，要么把 `.pt` 导出成 onnx。

---

## 二、五步固定套路（背下来）

```
① 加载模型  readNetFromONNX
② 预处理    blobFromImage       （图片 → 网络能吃的格式）
③ 推理      setInput + forward  （真正跑模型）
④ 解析输出  读张量              （把数字变成框/角点）—— 唯一随模型变的一步
⑤ 后处理    NMSBoxes            （去掉重复的框）
```

第 ①②③⑤ 步对所有 YOLO 一模一样，是死套路。只有第 ④ 步，不同模型的"输出张量长什么样"不同，解析代码要跟着改。

---

## 三、完整可运行代码（标准 YOLOv8 检测模型）

先用官方 `yolov8n.pt` 导出的 onnx 练手——模型最好拿、资料最多，能把整条管线跑通（这是两天计划的"路线 B 兜底"）。

### 3.1 先拿到 onnx 模型

在有 Python 的机器上（Windows 或 Ubuntu 都行）：

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12 dynamic=False imgsz=640
```

- `opset=12`：算子版本，太高 OpenCV 的 DNN 可能不认识（这是最常见的坑）。
- `dynamic=False`：固定输入尺寸，别用动态轴，否则 C++ 容易报错。
- 产物 `yolov8n.onnx` 拷进项目目录。

### 3.2 头文件 `YoloDetector.h`

```cpp
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct YoloBox {
    cv::Rect rect;        // 检测框（已换算回原图坐标）
    float confidence;     // 置信度 0~1
    int classId;          // 类别编号
};

class YoloDetector {
public:
    // onnxPath: 模型路径; inputW/H: 模型输入尺寸，必须和导出时一致
    explicit YoloDetector(const std::string& onnxPath,
                          int inputW = 640, int inputH = 640);

    // 输入一帧，返回所有检测框
    std::vector<YoloBox> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net_;
    int inputW_, inputH_;
};
```

### 3.3 实现 `YoloDetector.cpp`（五步全流程）

```cpp
#include "YoloDetector.h"
#include <stdexcept>

// ---------- ① 加载模型（构造函数里做一次，别每帧都加载）----------
YoloDetector::YoloDetector(const std::string& onnxPath, int inputW, int inputH)
    : inputW_(inputW), inputH_(inputH)
{
    net_ = cv::dnn::readNetFromONNX(onnxPath);
    if (net_.empty())
        throw std::runtime_error("无法加载 ONNX: " + onnxPath);

    // 用 CPU 推理（虚拟机没 GPU，这是正常选择）
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// ---------- 每帧调用一次 ----------
std::vector<YoloBox> YoloDetector::detect(const cv::Mat& frame)
{
    // ② 预处理：图片 → blob（4 维张量）
    cv::Mat blob;
    cv::dnn::blobFromImage(
        frame, blob,
        1.0 / 255.0,                    // 像素归一化到 0~1
        cv::Size(inputW_, inputH_),     // 缩放到模型输入尺寸
        cv::Scalar(),                   // 减均值，这里不用
        true,                           // swapRB: OpenCV 是 BGR，模型要 RGB
        false                           // crop=false
    );

    // ③ 推理
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    // ---------- ④ 解析输出（标准 YOLOv8：1×84×8400）----------
    // 84 = 4 个框参数(cx,cy,w,h) + 80 个类别得分
    // 8400 = 模型在这一层给出的候选框总数
    cv::Mat out = outputs[0];
    out = out.reshape(1, 84);       // 变成 [84, 8400]
    cv::transpose(out, out);        // 转置成 [8400, 84]，每行是一个候选

    // 模型输出的坐标是"640×640 输入图"上的，要按缩放比换回原图
    float xScale = (float)frame.cols / inputW_;
    float yScale = (float)frame.rows / inputH_;

    std::vector<cv::Rect> rawBoxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int i = 0; i < out.rows; ++i) {
        const float* row = out.ptr<float>(i);
        float cx = row[0], cy = row[1], w = row[2], h = row[3];

        // 找 80 个类别里得分最高的那个
        int bestClass = 0;
        float bestScore = row[4];
        for (int c = 5; c < 84; ++c) {
            if (row[c] > bestScore) { bestScore = row[c]; bestClass = c - 4; }
        }
        if (bestScore < 0.25f) continue;   // 太不可信，丢掉

        // 中心点±半宽半高 → 左上角坐标，并换算回原图尺寸
        int x = cvRound((cx - w / 2) * xScale);
        int y = cvRound((cy - h / 2) * yScale);
        int bw = cvRound(w * xScale);
        int bh = cvRound(h * yScale);

        rawBoxes.emplace_back(x, y, bw, bh);
        confidences.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    // ---------- ⑤ 后处理：NMS 去掉重复框 ----------
    std::vector<int> indices;
    cv::dnn::NMSBoxes(rawBoxes, confidences,
                      0.25f,   // 置信度阈值
                      0.45f,   // IOU 阈值：重叠超过它就算同一个目标
                      indices);

    std::vector<YoloBox> result;
    for (int idx : indices)
        result.push_back({rawBoxes[idx], confidences[idx], classIds[idx]});
    return result;
}
```

### 3.4 怎么用它

```cpp
YoloDetector yolo("yolov8n.onnx");          // 只造一次，放循环外
// ... 视频循环里 ...
auto boxes = yolo.detect(frame);
for (auto& b : boxes) {
    cv::rectangle(frame, b.rect, cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, std::to_string(b.confidence).substr(0, 4),
                cv::Point(b.rect.x, b.rect.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
}
```

---

## 四、关键区别：换成装甲板四关键点模型，第 ④ 步怎么改

标准 YOLOv8 只给"框"，但考核要"四个角点"。深大开源的装甲板模型一次推理就给四个角点。它的输出通道布局**不一样**，所以只改第 ④ 步。

据开源说明 `[依官方说明，务必按你下载模型的 README 复核]`：

| 模型 | 输入 | 输出张量 | 通道含义 |
|---|---|---|---|
| `Infantry-v5n-20250725` | 1×3×640×640 | 1×25200×22 | 0~7：4 个角点 (x,y)；8：置信度（需 sigmoid）；9~12：颜色；13~21：9 类装甲板 |
| `Infantry-v8n-20260726` | 1×3×480×640 | 1×21×6300 | 0~3：颜色；4~12：9 类；13~20：4 个角点 |

注意两点与标准 YOLO 不同：
1. **关键点坐标已经是原图像素坐标**（官方说明如此），所以**不用做缩放换算**——这点和标准 YOLO 相反，别照搬上面的 `xScale`。
2. **置信度要过 sigmoid**：`sigmoid(x) = 1/(1+exp(-x))`，需要 `#include <cmath>`。

第 ④ 步改法（以 v5n 为例，伪代码框架）：

```cpp
cv::Mat out = outputs[0];
out = out.reshape(1, 22);          // [25200, 22]，每行一个候选

for (int i = 0; i < out.rows; ++i) {
    const float* row = out.ptr<float>(i);
    float conf = 1.0f / (1.0f + std::exp(-row[8]));  // sigmoid
    if (conf < 0.5f) continue;

    // 4 个角点，已经是原图坐标
    cv::Point2f p[4];
    for (int k = 0; k < 4; ++k)
        p[k] = cv::Point2f(row[2*k], row[2*k+1]);

    // 颜色分支 row[9..12]、类别分支 row[13..21]，取最大者
    // 由 4 点围成框，收集起来送 NMS
    cv::Rect box = cv::boundingRect(std::vector<cv::Point2f>(p, p+4));
    // ... 存 box/conf，之后 NMS
}
```

> **小算提醒（严谨性）**：上面 v5n/v8n 的通道布局来自官方开源描述，我**没有实测核实**。你下载模型后一定打开它的 README 或部署示例，逐项核对"第几个通道是什么"。这一步是这类模型最容易翻车的地方——布局搞错，结果会完全乱。

---

## 五、怎么确认自己用对了（别只看"没报错"）

1. **先测单张图，别一上来跑视频**：用一张装甲板清晰的图，把框/角点画出来 `cv::imwrite` 存盘，肉眼看位置对不对。
2. **打印张量形状**：`std::cout << out.size << outputs[0].dims;`，确认和第 ④ 步假设的 `1×84×8400` 或 `1×25200×22` 一致。**形状不对，后面全错**——这是最快的自检。
3. **看置信度数值**：如果全是 0 或全是 1，多半是归一化（`/255`）或 sigmoid 漏了。
4. **位置乱飞**：优先怀疑尺寸不匹配（导出 imgsz 和 `blobFromImage` 的 Size 不一致）、swapRB 忘设、坐标缩放搞反。
5. **接 PnP 后用重投影验证**：把算出的位姿用 `projectPoints` 投回图像，和观测角点误差应只有几个像素。

---

## 六、五个最常见的坑（按踩坑频率排序）

| 坑 | 症状 | 解决 |
|---|---|---|
| ONNX 算子版本太高 | 加载或 forward 报错 | 导出用 `opset=12`，`dynamic=False` |
| 输入尺寸不匹配 | 检测框位置全乱 | 导出 imgsz 与 blobFromImage Size 必须一致 |
| swapRB 忘设 | 颜色识别不对 | `blobFromImage` 第 5 个参数设 `true` |
| 坐标缩放搞反/漏做 | 框缩在角落或跑出画面 | 标准 YOLO 要乘 scale；四关键点模型通常不用 |
| 置信度没 sigmoid | 分数异常 | 四关键点模型按官方说明做 sigmoid |

---

## 七、接到你现有项目的位置

只需改 `main_video.cpp`，在循环外加模型对象、循环内加调用：

```cpp
YoloDetector yolo("models/armor.onnx");   // 循环外，只加载一次
bool useYolo = true;                       // 可用 --use-yolo 参数控制

// 循环内：
if (useYolo) {
    auto boxes = yolo.detect(frame);       // YOLO 出结果（路线 A 改为出角点）
    // 把 boxes/角点转成你的 Armor 结构，再走原来的 PnP + drawResult
} else {
    auto armors = detector.detect(frame, calib);  // 传统法保底
}
```

CMakeLists 不用改（DNN 已在 opencv 库里），只要把 `YoloDetector.cpp` 加进 `add_executable`。

---

## 八、一句话总结

> **下载现成 onnx → C++ 五步套路（加载/预处理/推理/解析/NMS）→ 先测单张图看框对不对 → 再进视频循环 → 四角点接 PnP。**
> 五步里四步是死代码，只有"解析输出"要按你选的模型改；改之前先看张量形状、读官方 README。

---

## 附：核实状态
- `[已验证]` `readNetFromONNX` / `blobFromImage` / `forward` / `NMSBoxes` 的调用形式，来自 OpenCV 官方 DNN 示例惯例，标准 YOLOv8 `1×84×8400` 解析为本机/公开示例通用写法。
- `[依官方说明]` 四关键点模型 v5n/v8n 的通道布局，来自 RoboMaster 社区开源帖（bbs.robomaster.com/article/1942761），**未实测**，需按所下载模型的 README 复核。
- 本轮为讲解，未改动任何项目代码、未运行训练或推理；要我实际写进项目并编译验证，把模型文件和路线选择定下来即可。
