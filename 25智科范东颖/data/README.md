# 考核数据说明

将战队提供的两个 ZIP 放入本目录（`data/`），分别解压（保留压缩包内的目录结构）：

```bash
unzip data/2026-11-13_18-28-12_830451.zip -d data/
unzip data/2026-11-13_18-25-25_462305.zip -d data/
```

解压后每个数据包内包含：`video/video.avi`（原始视频）、`video/frames.csv`（帧记录）、
`calibration/intrinsics.yaml`（相机内参与畸变）、`calibration/handeye.yaml`（手眼标定）、
`imu/frame_imu.csv`（与视频帧对应的 IMU 数据）、`metadata.json`（采集元数据）。

使用时把数据放到本目录（或任意路径均可）：

```
data/
├── 2026-11-13_18-25-25_462305/video/video.avi
└── 2026-11-13_18-28-12_830451/video/video.avi
```

在项目根目录运行：

```bash
./build/vision data/2026-11-13_18-25-25_462305/video/video.avi output/result_1.mp4 output/result_1.csv --no-gui
./build/vision data/2026-11-13_18-28-12_830451/video/video.avi output/result_2.mp4 output/result_2.csv --no-gui
```

参数说明、CSV 字段含义见根目录 `README.md`。
