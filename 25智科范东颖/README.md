# 2027 赛季 Uprising 视觉组考核 —— 自瞄方向
04252222   智能科学与技术   范东颖
Ubuntu 22.04 + C++17 + OpenCV 的 RoboMaster 装甲板检测与自瞄程序。

- 检测红/蓝装甲板，标出四个角点与中心点；
- PnP 估计装甲板中心在相机坐标系下的位置；
- 画面显示检测结果、三维位置与每帧处理耗时；
- 数字识别、卡尔曼跟踪、自瞄解算（yaw/pitch、提前量、重力下坠、云台系转换）；
- 传统检测在目标旋转、灯条被遮挡/断开时容易失效，此时用 YOLO 补检，保证连续检出。

## 环境要求

| 项目 | 要求 |
|---|---|
| 系统 | Ubuntu 22.04 |
| 语言/标准 | C++17 |
| 构建 | CMake >= 3.16、g++ |
| OpenCV | >= 4.8（YOLO 补检用 OpenCV dnn 加载 ONNX 需要；4.5.4 只能跑传统检测 + 数字识别） |
| OpenVINO | 2024.6（本次提交启用），用于加速 YOLO 推理 |

依赖安装：

```bash
sudo apt update && sudo apt install -y build-essential cmake libopencv-dev
```

> 注意：Ubuntu 22.04 `apt` 默认安装的是 OpenCV 4.5.4，可以完成传统检测与数字识别，
> 但若未启用 OpenVINO、YOLO 改走 OpenCV dnn，则可能因 OpenCV 版本过低而加载失败，并自动退回纯传统检测（日志会打印 `[WARN] YOLO model load failed`）；
> 需要完整功能请安装 4.8+（例如源码编译），或启用 OpenVINO。

OpenCV 4.13.0 源码编译安装（本次提交的构建方式，安装到 `$HOME/opencv-4.13.0`）：

```bash
sudo apt install -y build-essential cmake git libgtk2.0-dev pkg-config
git clone --branch 4.13.0 --depth 1 https://github.com/opencv/opencv.git ~/opencv-src
cmake -S ~/opencv-src -B ~/opencv-src/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/opencv-4.13.0
cmake --build ~/opencv-src/build -j$(nproc)
cmake --install ~/opencv-src/build
```

本次提交使用的构建环境：Ubuntu 22.04 + g++ 11（-O3）+ 源码编译的 OpenCV 4.13.0
（`cmake -DOpenCV_DIR=~/opencv-4.13.0/lib/cmake/opencv4 -DUSE_OPENVINO=ON -DOpenVINO_DIR=/opt/intel/openvino_2024.6.0/runtime/cmake`），YOLO 推理使用 OpenVINO 2024.6。

（可选）OpenVINO（本次提交使用 2024.6）：从 https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/
下载 ubuntu22 的 tgz，解压到 `/opt/intel`，然后 `source /opt/intel/openvino_2024.6.0/setupvars.sh`。

## 编译

```bash
# 默认：YOLO 走 OpenCV dnn，不需要 OpenVINO
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# OpenVINO 加速 YOLO（本次提交使用此构建）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENVINO=ON \
      -DOpenVINO_DIR=/opt/intel/openvino_2024.6.0/runtime/cmake
cmake --build build -j$(nproc)
```

产物为 `build/vision`，`config/`、`model/` 会被复制到 `build/`，可在项目根目录或 `build/` 内运行。

## 运行

```bash
./build/vision <video> [out_video] [out_csv] [options]
```

```bash
./build/vision data/2026-11-13_18-25-25_462305/video/video.avi output/result_1.mp4 output/result_1.csv --no-gui
./build/vision data/2026-11-13_18-28-12_830451/video/video.avi output/result_2.mp4 output/result_2.csv --no-gui
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `<video>` | 必填 | 输入视频 |
| `[out_video]` | `output/result.mp4` | 输出视频（mp4 不可用时回退 MJPG AVI） |
| `[out_csv]` | `output/result.csv` | 逐帧结果 CSV |
| `--config-dir <DIR>` | `config` | 配置目录 |
| `--max-frames <N>` | 全部 | 只处理前 N 帧 |
| `--no-gui` | 关 | 不弹窗，仅写文件 |

### 配置文件参数

程序启动时从 `--config-dir`（默认 `config/`）读取三个 YAML 文件（OpenCV `FileStorage` 格式，首行 `%YAML:1.0` 必需）；修改后重新运行即可生效，无需重新编译。

**`config/camera.yaml` —— 相机内参与相机→云台外参**（数值与考核提供的标定一致）

| 字段 | 类型/单位 | 当前值 | 含义 |
|---|---|---|---|
| `image_width` / `image_height` | int / px | 1440 / 1080 | 标定对应的图像分辨率；启动时会与输入视频帧尺寸比对，不一致直接报错，避免内参错配导致测距错误 |
| `camera_matrix` | 3×3 行主序 / px | fx=2334.31, fy=2335.90, cx=722.62, cy=534.62 | 相机内参矩阵 K，按 `[fx,0,cx; 0,fy,cy; 0,0,1]` 排列，用于 `solvePnP` 与角点反投影 |
| `distort_coeffs` | 5 个数 | k1=-0.0800, k2=0.0857, p1=0.00279, p2=-0.00073, k3=0 | 畸变系数，顺序 `k1,k2,p1,p2,k3`，供 `solvePnP` / `projectPoints` 使用 |
| `R_camera2gimbal` | 3×3 行主序 | 见文件 | 相机系→云台系的旋转矩阵，用于把目标位置从相机系转换到云台系（输出 yaw/pitch） |
| `t_camera2gimbal` | 3 维 / m | (0.1539, -0.0437, 0.0171) | 相机系→云台系的平移向量，单位米 |

**`config/armor.yaml` —— 装甲板目标参数**

| 字段 | 类型/单位 | 当前值 | 含义 |
|---|---|---|---|
| `armor_width_mm` | float / mm | 135 | PnP 物体点中两块灯条之间的距离（RM 小装甲板 135 mm）；决定测距尺度，改错会使距离整体偏大或偏小 |
| `armor_height_mm` | float / mm | 56 | PnP 物体点中的灯条长度（约 56 mm），不是装甲板整体高度（125 mm） |
| `enemy_color` | 字符串 | `"both"` | 检测的装甲板颜色：`"red"` 只检红、`"blue"` 只检蓝、`"both"` 两者都检（本提交使用 both） |

**`config/aim.yaml` —— 自瞄 / 弹道参数**

| 字段 | 类型/单位 | 当前值 | 含义 |
|---|---|---|---|
| `bullet_speed` | float / m·s⁻¹ | 20.0 | 弹丸初速，用于计算飞行时间 `t_f = 距离 / 速度`（RM 裁判系统默认 20 m/s） |
| `latency_ms` | float / ms | 30.0 | 固定系统延迟（曝光 + 处理 + 通信），计入提前量：`lead_time = latency + t_f` |
| `gravity` | float / m·s² | 9.8 | 重力加速度，用于下坠补偿 `drop = 0.5·g·t_f²` |

> 模型路径在代码中固定为相对路径 `model/...`，因此需要在项目根目录（或 CMake 已复制 `model/` 的 `build/` 目录）下运行。

### 结果 CSV 字段（22 列）

| 列 | 含义 |
|---|---|
| `frame` | 输入视频帧号（从 0 起），与输出视频帧号一一对应 |
| `time_s` | 该帧时间戳 = frame / 视频 fps（秒） |
| `detected` | 本帧主目标是否成功检出并解算（0/1，当前实现与 `is_primary` 同值） |
| `num_armors` | 本帧检出的装甲板数量 |
| `number` | 主目标编号（传统路径为数字如 `3`；YOLO 补检为 `R3`/`B3` 形式） |
| `is_primary` | 本行是否为主目标（离画面中心最近）且 PnP 成功 |
| `x` `y` `z` | 主目标中心在相机系下坐标，m（X 右、Y 下、Z 前） |
| `kf_x` `kf_y` `kf_z` | 卡尔曼滤波后的位置，m |
| `yaw_cam` `pitch_cam` | 相机系解算角，deg（已加提前量与下坠补偿） |
| `yaw_gimbal` `pitch_gimbal` | 转换到云台系后的解算角，deg |
| `pred_x` `pred_y` `pred_z` | 按提前量预测的命中点，m |
| `t_flight` `lead_time` | 弹丸飞行时间 / 总提前时间（延迟 + 飞行），s |
| `ms` | 本帧处理耗时，ms |

## 目录结构

```
.
├── CMakeLists.txt
├── README.md
├── .gitignore
├── src/        # main / detector / number_classifier / pnp / tracker / aimer / utils
├── config/     # camera.yaml / armor.yaml / aim.yaml
├── model/      # armor_pose416.onnx / mlp.onnx / label.txt
├── data/       # README.md（原始数据不在仓库中）
└── output/     # 结果视频与 CSV
```

## 方法

1. **颜色分割**：BGR→HSV，取红/蓝掩膜，形态学去噪；
2. **传统检测**：找灯条（轮廓 + `minAreaRect`/`fitLine`）→ 两两配对成装甲板 → 数字识别后仅保留 1~5 号装甲板（过滤 outpost / guard / base / negative 等非自瞄目标）；
3. **YOLO 补检**：当传统检测本帧没有给出结果时（目标旋转、灯条被遮挡或断开导致端点/配对失效），用 YOLO（14 类 + 4 关键点）检测并补上，使序列保持连续；
4. **数字识别**：透视变换取 32×28，中间 20×28 灰度+OTSU，MLP 输出类别与置信度；
5. **PnP**：四角点 + 装甲板尺寸 135×56 mm → `solvePnP` → 相机系 (x,y,z)；
6. **卡尔曼**：6 维状态（位置+速度）对主目标做平滑与预测；
7. **自瞄解算**：相机系下 `yaw=atan2(x,z)`、`pitch=-atan2(y,z)`（相机 Y 轴向下，取负使抬升为正）；云台系下 `yaw=atan2(-y,x)`、`pitch=atan2(z,√(x²+y²))`（云台 Z 轴向上）；迭代求飞行时间做提前量 `P=pos+vel·t`；下坠补偿 `0.5·g·t²` 折算成角度后叠加到 pitch；相机→云台的转换用标定外参。
8. **输出**：画面叠加四角点/中心点/数字/三维位置/yaw·pitch/预测点/准星，写视频与逐帧 CSV。

坐标系：相机 X 右、Y 下、Z 前；云台 X 前、Y 左、Z 上；yaw 右正、pitch 抬正。

### 强光 / 逆光 / 运动模糊下的检测增强

- **强光 / 逆光 / 过曝**：原方案用固定 HSV 阈值（饱和度 `S >= 110`、亮度 `V >= 180`）判断红/蓝灯条。强光下装甲板被白光冲淡、饱和度下降，逆光或欠曝时亮度不足，都会掉出固定阈值导致整条灯条丢失。为此增加一条与绝对亮度无关的分割路径：用 `R - max(G,B)`、`B - max(R,G)` 衡量像素的红/蓝相对优势，光照整体变亮或变暗时三个通道同加或同乘，该差值基本保持，因此强光、逆光、欠曝条件下仍能分离出装甲板；阈值由 Otsu 按每帧的差值分布自动确定，并设下限 40 避免画面中没有目标时过分割。该结果与原 HSV 掩膜取并集：正常光照下保持原结果，仅在原方法漏检处补充候选，补充出的候选仍须通过灯条几何约束与数字识别置信度确认。
- **运动模糊**：灯条边缘模糊会造成二值掩膜断裂、灯条长度与倾角计算偏差。开运算保持 3×3 矩形核去噪不变，闭运算改为 5×5 椭圆核，用于合并断裂的灯条，使灯条检测与后续配对更稳定。
- **兜底**：若过曝严重到颜色信息完全丢失（R、G、B 均饱和且接近），颜色方法无法恢复，由 YOLO 补检兜底。

上述增强默认启用，不需要额外参数。

## 结果与性能

两段考核数据（均为 165.93 fps），`output/` 下的视频与 CSV 均由最终提交代码在
Ubuntu 22.04（g++ -O3 + OpenCV 4.13.0，启用 OpenVINO）上跑出：

| 数据 | 帧数 | 检出 | 其中 YOLO 补检 |
|---|---|---|---|
| 视频1 `2026-11-13_18-25-25` | 1770 | 1770/1770 | 93 帧（5.3%） |
| 视频2 `2026-11-13_18-28-12` | 1240 | 1240/1240 | 133 帧（10.7%） |

YOLO 补检帧集中在目标旋转/灯条断开的片段
（视频1：163、167~194、544、878~908、1257~1258、1593~1622；视频2：0~132），
这两段数据最终都达到 100% 连续检出。

处理耗时（统计自 `output/result_1.csv`、`output/result_2.csv` 的 `ms` 列）：

| 数据 | 平均 | P50 | P95 | 最大 |
|---|---|---|---|---|
| 视频1 | 9.45 ms | 9.34 ms | 14.36 ms | 29.46 ms |
| 视频2 | 8.77 ms | 8.23 ms | 14.82 ms | 28.62 ms |

耗时尖峰集中在触发 YOLO 补检的帧（视频1/视频2 平均约 16.3/15.3 ms），传统检测帧平均约 8~9 ms。

`output/result_*.mp4` 为原生分辨率 1440×1080、165.93 fps、完整 1770/1240 帧
（H.264 编码，系统自带播放器即可播放），视频第 i 帧与对应 CSV 的第 i 行一一对应。

优化说明（开发期测量，用来说明优化前后的变化；不同构建环境分别列出，不作跨环境比较）：

| 优化项 | 优化前 | 优化后 | 测量方式 |
|---|---|---|---|
| YOLO 模型输入尺寸 640×640 → 416×416（导出时选择） | 57.09 ms/次 | 24.54 ms/次（0.43×） | 同一台机器、同一后端（OpenCV dnn），各 30 次推理平均 |
| YOLO 推理后端 OpenCV dnn → OpenVINO（可选开关 `-DUSE_OPENVINO=ON`，本次提交启用） | YOLO 触发帧 34.1 / 32.8 ms（视频1 / 视频2） | 9.7 / 9.8 ms（最小 8.2，最大 21.2 / 42.9） | 取同一批 YOLO 触发帧（93 / 133 帧），分别在两种构建下统计耗时 |

说明：本次提交的结果文件使用 **OpenVINO** 后端（`-DUSE_OPENVINO=ON`）；上表“OpenCV dnn”一行为未启用
OpenVINO 的对照构建测量（同一批 YOLO 补检帧约 34 ms，切换到 OpenVINO 后约 9.7 ms），因此结果 CSV 的
P95（11.61 / 15.97 ms）已是被 OpenVINO 压低后的值。若目标机没有 OpenVINO，可退回 dnn 构建，功能不变，
但 YOLO 补检帧会明显变慢。

## 参考来源、AI 使用与本人完成内容

### 第三方资源

`model/` 下的 3 个文件（`armor_pose416.onnx`、`mlp.onnx`、`label.txt`）**已随仓库提交**，无需额外下载即可复现；如需重新获取，来源、许可证与 MD5 见本节。

- **数字识别模型** `model/mlp.onnx`、`model/label.txt`：来自
  [chenjunnn/rm_auto_aim](https://github.com/chenjunnn/rm_auto_aim) 的 `armor_detector/model/`
  （MIT，Copyright (c) 2022 ChenJun）。
  网络结构为 `Flatten → Gemm(560→120) → ReLU → Gemm(120→84) → ReLU → Gemm(84→9)`，
  与该仓库数字识别输入（20×28 = 560）及 9 类标签一一对应；`label.txt` 内容一致，仅换行符由 CRLF 改为 LF。
- **灯条/装甲板数据结构**：参考上述仓库的 `Light{top, bottom, length, width, tilt_angle}`，
  简化为 `LightBar{center, p1, p2, length, angle}`。
- **YOLO 检测模型** `model/armor_pose416.onnx`：来自
  [huo0136/-YOLO26-Pose](https://github.com/huo0136/-YOLO26-Pose) 的 `weights/best416.onnx`，
  MD5 完全一致（详见下方「YOLO 模型」）。
- **YOLO 类别与关键点方案**：参考
  [zRzRzRzRzRzRz/YOLO-of-RoboMaster-Keypoints-Detection-2023](https://github.com/zRzRzRzRzRzRz/YOLO-of-RoboMaster-Keypoints-Detection-2023)
  （Apache-2.0）的 14 类（B1~B5、BO、BS、R1~R5、RO、RS）与 4 关键点定义。
- **OpenVINO 推理接口用法**：参考
  [TongjiSuperPower/sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)（MIT）。
- **检测/跟踪总体流程、自瞄解算思路**：参考常规 RoboMaster 自瞄方案（含 rm_auto_aim）。

### YOLO 模型（第三方，非本人训练）

`model/armor_pose416.onnx` 来自 GitHub 仓库
[huo0136/-YOLO26-Pose](https://github.com/huo0136/-YOLO26-Pose) 的 `weights/best416.onnx`。


- 训练者/提供者：仓库作者 huo0136（仓库 commit：2026-11-07 "Add YOLO26-Pose training results"）；
- 训练配置（仓库 README）：YOLO26n-Pose、数据集 `armor_pose.yaml`、imgsz 640、epochs 100、
  batch 16、Ultralytics 8.4.172、RTX 3080 Ti；
- 类别/关键点：14 类（B1~B5、BO、BS、R1~R5、RO、RS）+ 4 关键点；
- 许可证：该仓库未提供 LICENSE 文件；模型由 ultralytics（AGPL-3.0）训练，按 AGPL-3.0 条款使用；
- 另：`armor_pose_best.pt`、`armor_pose_last.pt` 同样来自该仓库（与 `weights/best.pt`、`weights/last.pt`
  MD5 一致），仅用于导出 640 输入的早期 ONNX，未包含在提交仓库中。

### AI 辅助范围

- **AI 辅助实现/修改**：`src/tracker.*`（卡尔曼滤波）、`src/aimer.*`（自瞄解算）、`src/utils.*`（配置读取）、
  `src/detector.cpp` 中 YOLO 解码/NMS/OpenVINO 分支、`src/number_classifier.*` 的接入部分、
  OpenVINO 后端接入。
- **其余部分**（传统检测、PnP、主流程、测试与分析）见下。

### 本人完成的内容

- 传统检测：HSV 颜色分割、灯条轮廓检测（`minAreaRect` / `fitLine`）、灯条配对与装甲板四角点提取、
  阈值与几何约束调参；
- PnP 测距：四角点与装甲板尺寸（135×56 mm）的 `solvePnP` 调用、结果验证与相机参数核对；
- 主流程：逐帧处理、结果可视化（四角点/中心点/数字/位置/角度/耗时）与视频、CSV 输出；
- 数字识别接入与检测结果过滤逻辑；
- 在 Ubuntu 22.04 上完成编译、两段考核数据的运行、参数调整与问题排查；
- 结果分析（检出率、耗时统计、坐标稳定性）。

