# IMU 接入讲解（小白版）

> 这份文档讲清楚三件事：① IMU 数据是什么 ② 怎么算姿态角 ③ 怎么把测距结果换算到机器人坐标系。
> 配合代码 `ImuReader.h/.cpp` 和体检程序 `imu_check.cpp` 一起看。

---

## 一、为什么装甲板项目要接 IMU？

我们的程序用 PnP 算出装甲板的位置 `tvec`，但这个坐标是**相机坐标系**下的：

- 原点在相机光心
- Z 轴沿着镜头看出去的方向

问题在于：相机是**装在云台上**的，而且可能是歪着装/侧着装。官方数据实测这段视频：

```
roll≈102°   pitch≈86°   yaw≈75°
```

roll 102° 意味着相机几乎是**侧躺着**装的（正常横装 roll≈0°）。
所以"相机系 Z=1.67m 前方"这句话，对机器人想"转云台瞄准"来说没法直接用。

**IMU + 手眼标定**就是来解决这个的：
- IMU 告诉你"相机当时怎么摆"（姿态）
- 手眼标定告诉你"相机和云台之间差了多远、转了多少"（外参）

两者结合，就能把 PnP 结果换算到**云台/车体坐标系**，这才是控制要用的量。

> 补充：这段视频相机几乎不动（roll 全程只变 0.75°）。所以这里 IMU 的价值是**坐标转换**，
> 不是"运动补偿"。真实比赛相机随云台快速转动时，IMU 还能做运动模糊补偿和姿态预测。

---

## 二、IMU 数据长什么样？

官方给了 `imu_frame_imu.csv`，**每一行对应视频的一帧**（共 1770 行，帧号 0~1769）：

```
frame_index,monotonic_ns,elapsed_s,qw,qx,qy,qz,left_imu_ns,right_imu_ns,imu_gap_ms,status
0,1445687071724,0.000252,0.68710,0.19089,0.68598,-0.14454,...,interpolated
1,...,0.68711,0.19090,0.68596,-0.14454,...,interpolated
```

我们真正用到的是：
- `frame_index`：帧号，用来和视频第 N 帧对齐
- `qw,qx,qy,qz`：**四元数**，表示这一帧的姿态

### 什么是四元数？

四元数是用 4 个数 `(w,x,y,z)` 表示三维旋转的方法。
你只需要记住两点：

1. 它比"欧拉角(roll/pitch/yaw)"更适合计算——不会遇到万向节死锁、便于插值。
2. 人看不懂四元数，所以我们要把它**转成欧拉角**（roll/pitch/yaw，单位度）才直观。

这个转换在 `ImuReader.cpp` 的 `quatToEulerDeg()` 里，公式是固定的数学式子，会用即可，不用背。

---

## 三、手眼外参：三个坐标系的关系

`calibration_handeye.yaml` 里给了三组数：

```yaml
R_camera2gimbal:  [9个数字]   # 相机系 → 云台系 的旋转（3x3矩阵）
t_camera2gimbal:  [3个数字]   # 平移量（米），相机原点在云台系下的位置
R_gimbal2imubody: [9个数字]   # 云台系 → IMU body 系 的旋转
```

名字怎么读？`R_camera2gimbal` = "camera to gimbal" = 从相机系转到云台系。
`R` 是旋转矩阵（3x3），`t` 是平移向量（3个数）。

### 坐标转换公式（核心）

```
第1步：相机系 → 云台系
    p_gimbal = R_camera2gimbal × p_cam + t_camera2gimbal

第2步：云台系 → IMU body 系
    p_imu = R_gimbal2imubody × p_gimbal

合起来就是：
    p_imu = R_gimbal2imubody × (R_camera2gimbal × p_cam + t_camera2gimbal)
```

代码里对应 `HandEyeExtrinsics::cameraToImuBody()`，就这几行：

```cpp
cv::Mat pc = (cv::Mat_<double>(3,1) << pCam[0], pCam[1], pCam[2]);
cv::Mat tmat = cv::Mat(t_camera2gimbal);
cv::Mat pg = R_camera2gimbal * pc + tmat;   // 相机 → 云台
cv::Mat pi = R_gimbal2imubody * pg;         // 云台 → IMU body
return cv::Vec3d(pi.at<double>(0), ...);
```

> 小坑：官方 yaml 里有中文注释，OpenCV 的 `FileStorage` 读不了会崩溃。
> 所以 `ImuReader.cpp` 用**纯文本逐行解析**（找 `R_camera2gimbal: [...]` 抠出数字），
> 不挑编码，换 Ubuntu 也一样能读。

---

## 四、代码怎么组织的？（为什么不动检测代码）

IMU 是**独立模块**，和装甲板检测完全解耦：

| 文件 | 作用 |
|---|---|
| `ImuReader.h` | 声明：`ImuPose`（一帧姿态）、`ImuReader`（读 CSV）、`HandEyeExtrinsics`（外参+转换） |
| `ImuReader.cpp` | 实现：读文件、四元数转欧拉角、坐标转换 |
| `main.cpp` | 视频循环里按帧取姿态，用 `drawImuOverlay()` 把信息叠到画面上 |
| `imu_check.cpp` | 体检程序：验证一切是否正确 |

关键点：`drawImuOverlay()` 只是往结果图上**多画几行文字**，
`ArmorDetector::drawResult()` 一行都没改。所以：

- 关掉 IMU（`--no-imu`）→ 画面和以前一模一样
- 没有数据文件 → 自动跳过，检测照常

这就是"加功能不影响老功能"的稳妥做法。

---

## 五、怎么验证接对了？

### 跑体检程序

CLion 里选 `imu_check` target 点运行，或命令行：

```bash
./imu_check.exe imu_data/imu_frame_imu.csv imu_data/calibration_handeye.yaml
```

正确输出应该类似：

```
IMU 帧数: 1770  帧范围: 0 ~ 1769
帧 0: roll=102.34 pitch=86.24 yaw=74.88
帧 500: roll=102.41 pitch=86.22 yaw=74.48
R_c2g 正交误差 = 2.9e-16        ← 接近0说明是合法旋转矩阵
R_g2i 正交误差 = 0
坐标转换样例: 相机系 (0,0,2)m -> IMU body (-0.10,-2.15,-0.04)m
```

三个判断标准：
1. **帧数=1770、范围 0~1769** → CSV 读全了、和视频对齐
2. **姿态角稳定**（roll 约 102°）→ 四元数转换对了
3. **正交误差≈0** → 外参矩阵合法；转换后长度量级仍在 2m 左右 → 公式用对了

### 看结果视频画面

跑视频时画面会显示：

- 左上：`IMU  R:102.2 P:86.2 Y:74.9 [interpolated]`
- 每块装甲板下方：`C(-0.15,-0.11,1.67) -> I(-0.20,-1.82,0.11)`
  - `C(...)` = 相机系坐标（原来的 tvec）
  - `I(...)` = 换算到 IMU body 系的坐标

---

## 六、怎么自己编译运行？

CLion 里：
1. Tools → CMake → Reload CMake Project（让新 target 出现）
2. 编译前先关掉正在运行的窗口（避免 `Permission denied`）
3. Build Project
4. 选 `armor_vision` 运行 → 默认读视频+自动加载 IMU
5. 选 `imu_check` 运行 → 秒出验证结果

命令行：

```bash
# 完整视频（带 IMU，默认自动找 imu_data/）
./armor_vision.exe --mode video --video C:/carrie/video.avi --no-show

# 只跑前30帧快速看效果
./armor_vision.exe --mode video --max-frames 30 --no-show --dump-frame 10

# 明确指定数据路径（迁移目录后用）
./armor_vision.exe --mode video --imu D:/xxx/imu_frame_imu.csv --handeye D:/xxx/calibration_handeye.yaml

# 不要 IMU
./armor_vision.exe --mode video --no-imu
```

---

## 七、迁移到 Ubuntu 22.04 注意什么？

1. 把整个 `imu_data/` 文件夹一起拷过去（程序默认在 exe 或项目根目录找它）
2. 路径查找逻辑会依次找 `exe同目录/imu_data` → `exe父目录/imu_data`，CLion 的 build 子目录也能自动找到
3. 纯文本解析不依赖 OpenCV 版本，Ubuntu 上 apt 装的 OpenCV 4.5.4 同样能读
4. 如果路径不同，用 `--imu` / `--handeye` 显式指定即可
