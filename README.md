# 2027 赛季征途 Uprising 视觉组考核 — 装甲板检测（自瞄方向）

本项目面向视觉组考核（自瞄方向）的离线视频评测：在考核视频中检测装甲板，标出四个角点和中心点，估计装甲板中心在相机坐标系下的三维位置，并在画面上实时显示检测结果与处理耗时。

> 提交人：26 级 食品质量与安全 张锴锐　|　提交包命名：`26食质张锴锐`（符合"年级＋专业＋姓名"格式）

实现方案为传统 OpenCV 视觉（HSV 颜色分割 + 轮廓/椭圆拟合 + 灯条配对 + PnP 测距），**未使用深度学习模型**；另附带 IMU 姿态接入、时序多数投票修断灯条歪框等可选功能，以及 `--fast` 极速模式。

---

## 环境与安装

考核环境 Ubuntu 22.04，开发语言 C++。

| 项目 | 本实现 |
| --- | --- |
| C++ 编译器及语言标准 | g++ 11.4.0，C++17（`CMakeLists.txt` 中 `CMAKE_CXX_STANDARD 17`） |
| 构建与运行依赖 | CMake 3.22.1、OpenCV 4.5.4（apt 版 `libopencv-dev`）；无其他第三方库 |
| 模型文件 | 未使用模型，纯传统视觉算法，无需下载任何权重 |

依赖安装：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

获取代码：

```bash
git clone https://github.com/ZT-Visual-Group-Training/bingbingbing.git
cd bingbingbing
```

---

## 考核数据（需自行放入，不随仓库提交）

战队提供两个数据包，放到 `data/` 并分别解压（压缩包体积大，按 `.gitignore` 规则不入库）：

```bash
mkdir -p data
unzip data/2026-11-13_18-28-12_830451.zip -d data/
unzip data/2026-11-13_18-25-25_462305.zip -d data/
```

每个包解压后的相对路径与内容：

| 相对路径 | 内容 |
| --- | --- |
| `video/video.avi` | 原始考核视频 |
| `video/frames.csv` | 视频帧记录 |
| `calibration/intrinsics.yaml` | 相机内参与畸变参数 |
| `calibration/handeye.yaml` | 手眼标定数据 |
| `imu/frame_imu.csv` | 与视频帧对应的 IMU 数据 |
| `metadata.json` | 采集元数据 |

小装甲板尺寸为宽 **135 mm**、高 **56 mm**，高度取灯条长度，已写入检测参数。

> 相机参数说明：官方 `calibration/intrinsics.yaml` 的畸变键名与本程序 `loadCalib()` 期望的不同，直接传给 `--camera` 会报错。仓库根目录的 `camera.yaml` 已按官方标定数值转成程序可读格式（并补 `dt: d` 字段），运行时请用 `--camera camera.yaml`。官方原版文件完整保留在数据包中供核对。

---

## 源码结构

```
bingbingbing/
├── README.md                 本文件
├── CMakeLists.txt
├── main.cpp                  主程序：image / video / hsv 三种模式
├── ArmorDetector.h/.cpp      检测核心：颜色掩膜、灯条筛选、配对、PnP
├── ImuReader.h/.cpp          IMU 四元数读取与手眼坐标转换（可选）
├── TemporalFilter.h/.cpp     时序多数投票：修断灯条歪框（可选）
├── debug_pipeline.cpp        答辩演示：灯条筛选四步图
├── imu_check.cpp             答辩演示：IMU 数据体检
├── camera.yaml               相机内参与畸变（对应官方 intrinsics.yaml）
├── imu_data/                 IMU 数据（程序默认从此目录读取）
├── docs/                     答辩流程图、原理与调参文档
├── data/                     【需自行放入】官方数据包，不入库
└── results/                  【需自行放入】结果视频，不入库
```

---

## 构建与运行

### 构建

```bash
cmake -S . -B build && cmake --build build
```

生成的可执行文件位于 `build/`：`armor_vision`（检测主程序）、`hsv_tuner`（阈值调试）、`debug_pipeline`（灯条筛选演示）、`imu_check`（IMU 体检）。

### 处理两段考核视频

红蓝灯条同时检测，无需指定颜色参数。

```bash
# 第一段 18-25-25_462305
./build/armor_vision --mode video \
  --video data/2026-11-13_18-25-25_462305/video/video.avi \
  --camera camera.yaml \
  --imu imu_data/imu_frame_imu.csv \
  --handeye imu_data/calibration_handeye.yaml \
  --out results/result_462305.avi

# 第二段 18-28-12_830451
./build/armor_vision --mode video \
  --video data/2026-11-13_18-28-12_830451/video/video.avi \
  --camera camera.yaml \
  --imu data/2026-11-13_18-28-12_830451/imu/frame_imu.csv \
  --handeye data/2026-11-13_18-28-12_830451/calibration/handeye.yaml \
  --out results/result_830451.avi
```

无图形界面（SSH/服务器）加 `--no-show`；快速自测前 N 帧加 `--max-frames N`；修断灯条歪框加 `--temporal`；追求流畅加 `--fast`。完整参数见 `--help`。

### 参数表

| 参数含义 | 对应参数 | 默认值 / 说明 |
| --- | --- | --- |
| 输入视频 | `--video <路径>` | Linux 下务必显式传入 |
| 相机参数 | `--camera <yaml>` | 建议用本仓库 `camera.yaml` |
| 目标颜色 | 无需指定 | 红蓝灯条同时检测 |
| 输出路径 | `--out <路径>` | 默认 `armor_result.avi` |
| IMU 姿态 | `--imu <csv>` | 默认自动查找 `imu_data/` |
| 手眼外参 | `--handeye <yaml>` | 默认自动查找 `imu_data/` |
| 关闭 IMU | `--no-imu` | 不需要姿态时 |
| 时序滤波 | `--temporal` | 默认关，修远处断灯条歪框 |
| 极速模式 | `--fast` | 缩放0.75+关模糊，约1.6倍速（约7.6ms/帧），测距偏差最大约0.4m |
| 不弹窗 | `--no-show` | 服务器/SSH 下使用 |
| 只处理前 N 帧 | `--max-frames N` | 快速自测 |

---

## 实现要点

| 考核要求 | 实现方式 |
| --- | --- |
| 检测指定颜色装甲板 | HSV 颜色分割，红蓝双色同时检测；轮廓 + 椭圆拟合 + 六项几何筛选 |
| 标出四个角点和中心点 | 由配对灯条重建规整四点，画框并标出中心点 |
| 估计相机坐标系三维位置 | `solvePnP`：2D 四点 ↔ 3D 小装甲板模型（135×56 mm），输出 tvec（米） |
| 画面显示检测结果与耗时 | 每帧绘制框、中心、距离、IMU 姿态，并叠加检测耗时（毫秒） |
| 断灯条导致的歪框 | 可选 `--temporal` 时序多数投票，经全帧 A/B 验证不丢框、不移位 |

更多原理与答辩材料见 `docs/`：`答辩_系统流程图.png`、`答辩_灯条筛选四步图.png`、`答辩流程图_讲解稿.md`、`时序多数投票_说明.md`、`IMU接入讲解.md`、`YOLO使用详解.md` 等。
