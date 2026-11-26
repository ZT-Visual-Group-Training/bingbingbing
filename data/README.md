# data/ —— 考核数据放置说明

仓库**不包含**原始考核视频（两段合计约 1.1 GB）。请把战队提供的两个 ZIP 放到这里再解压：

```bash
# 在仓库根目录执行
unzip 2026-11-13_18-28-12_830451.zip -d data/
unzip 2026-11-13_18-25-25_462305.zip -d data/
```

解压后应得到：

```
data/
├── 2026-11-13_18-28-12_830451/
│   ├── video/video.avi
│   ├── video/frames.csv
│   ├── calibration/intrinsics.yaml
│   ├── calibration/handeye.yaml
│   ├── imu/frame_imu.csv
│   └── metadata.json
└── 2026-11-13_18-25-25_462305/
    └── ...（同上）
```

`data/` 下除本文件外都被 `.gitignore` 忽略，不会误提交大文件。
