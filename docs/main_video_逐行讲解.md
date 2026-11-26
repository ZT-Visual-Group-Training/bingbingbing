# main_video.cpp 逐行讲解（小白版）

> 阅读方法：先看"这一行是什么"，再看"为什么这么写"，最后看"它有什么用"。
> 行号对应 `main_video.cpp` 的实际行号。注释行（以 `//` 开头）是给人看的说明，编译器会跳过。

---

## 第 ① 块：文件头 + 工具箱（第 1–21 行）

### 第 1–14 行：文件头注释
```
// main_video.cpp —— 视频版主程序
// 和图片版区别只有两点：① VideoCapture 循环读帧 ② VideoWriter 写视频
```
- **是什么**：一大段以 `//` 开头的文字。
- **为什么**：C++ 里 `//` 后面的内容编译器完全忽略，专门写给人看。这里告诉你整个文件的定位（视频版）和用法示例。
- **有什么用**：你以后忘记这文件干嘛的，看开头 14 行就够了，不用往下翻。

### 第 15–19 行：引入标准/第三方工具箱
```cpp
#include <opencv2/opencv.hpp>   // 第15行：OpenCV 全家桶
#include <chrono>               // 第16行：计时
#include <cmath>                // 第17行：数学函数 sin
#include <iostream>             // 第18行：cout/cerr 打印
#include <string>               // 第19行：字符串
```
- **是什么**：`#include` 叫"包含头文件"，等于把别人写好的工具说明书拿进来。
- **为什么用尖括号 `<>`**：这些都是**系统/第三方库**，编译器去系统路径找。
- **逐行作用**：
  - 第15行：OpenCV 主头文件，`cv::Mat`、`cv::rectangle`、`VideoCapture` 全在里面，包含它等于一次引进整个 OpenCV。
  - 第16行：`<chrono>` 提供高精度时钟，用来测"处理一帧花了多少毫秒"。
  - 第17行：`<cmath>` 提供 `std::sin`，第31/37行算摆动要用。
  - 第18行：`<iostream>` 提供 `std::cout`（打印到屏幕）和 `std::cerr`（打印错误）。
  - 第19行：`<string>` 提供 `std::string`，存文件路径和拼字符串要用。

### 第 21 行：引入自己写的检测器
```cpp
#include "ArmorDetector.h"
```
- **是什么**：引入**你自己项目里**的头文件。
- **为什么用引号 `""`**：告诉编译器"先去我当前文件夹找"——这正是你之前 demo5 报错的原因（当时用了尖括号找不到）。
- **有什么用**：这行之后你才能用 `ArmorDetector`、`CameraCalib`、`Armor` 这些类型。检测算法都在这个头文件对应的 `.cpp` 里，这里只是把它"接进来"。

---

## 第 ② 块：makeFrame —— 制造一帧会动的测试图（第 27–49 行）

### 为什么需要这个函数
手上没有考核视频时，得自己造一段，否则程序没东西可处理。这个函数造"一帧"，参数 `phase`（相位）变化就能让装甲板动起来，连起来就是动画。

### 第 27 行：函数定义
```cpp
static cv::Mat makeFrame(double phase) {
```
- `static`：只在**本文件内可见**的内部工具，不暴露给别的文件。
- `cv::Mat`：返回值类型 —— 返回一张图（OpenCV 里图片就是 `cv::Mat`）。
- `double phase`：**输入参数**，一个不断变化的数。它变大时，板子的位置和大小跟着变，这就是"动画"的秘密。
- 类比：像一台印花机器，你给它不同的 `phase`（编号），它就印出略有不同的画面。

### 第 28 行：造一张空白画布
```cpp
cv::Mat img(360, 640, CV_8UC3, cv::Scalar(35, 35, 35));
```
- `360` = 高（行数），`640` = 宽（列数）—— 注意 **OpenCV 先高后宽**，和平时说的 640×360 顺序相反。
- `CV_8UC3`：每个像素用 8 位（0–255）、3 个通道（B 蓝、G 绿、R 红），即彩色图。
- `cv::Scalar(35,35,35)`：给三个通道都填 35，是很深的灰色 —— 这就是背景色。
- 有什么用：一张 640×360 的深灰"画布"，后面往上画装甲板。

### 第 30–34 行：画左边的红色装甲板
```cpp
// 第31行
int rx = 60 + static_cast<int>(50 * std::sin(phase));
```
- `std::sin(phase)`：正弦函数，结果在 **-1 到 +1** 之间来回摆动（phase 不断增大时，它像波浪一样上下）。
- `50 * sin(...)`：范围变成 **-50 到 +50**。
- `+ 60`：整体平移，范围变成 **10 到 110** —— 这就是红板左上角的 x 坐标。
- `static_cast<int>(...)`：强制转成整数，因为像素坐标必须是整数。
- **作用**：phase 变化时，rx 在 10~110 之间左右移动 → 红板左右摆动（模拟目标横向移动）。

```cpp
// 第32行：画黑色底板
cv::rectangle(img, cv::Rect(rx, 100, 180, 160), cv::Scalar(20, 20, 20), cv::FILLED);
```
- `cv::Rect(rx, 100, 180, 160)`：矩形的 (左上角x, 左上角y, 宽, 高) = (rx, 100, 180, 160)。
- `cv::Scalar(20,20,20)`：接近黑色，模拟装甲板的黑色底板。
- `cv::FILLED`：填满整个矩形（不是只画边框）。

```cpp
// 第33、34行：画两根红色灯条
cv::rectangle(img, cv::Rect(rx + 30, 130, 16, 100), cv::Scalar(0, 0, 255), cv::FILLED);
cv::rectangle(img, cv::Rect(rx + 140, 130, 16, 100), cv::Scalar(0, 0, 255), cv::FILLED);
```
- `cv::Scalar(0,0,255)`：B=0,G=0,R=255 → **红色**（注意 OpenCV 是 BGR 顺序，红在最后一位）。
- 两根灯条宽 16、高 100，竖直放置，一根在 `rx+30`，一根在 `rx+140`，中间隔着黑底。
- **为什么两根**：装甲板的典型特征就是"两根竖灯条 + 中间黑底"，你的检测算法正是靠配对两根灯条来找装甲板的。

### 第 36–45 行：画右边的蓝色装甲板（会缩放）
```cpp
// 第37行
double s = 1.0 + 0.35 * std::sin(phase * 0.7 + 1.0);
```
- 又一个正弦，范围 **0.65 到 1.35** —— 这是一个**缩放系数 s**。
- `phase * 0.7 + 1.0`：让蓝板的变化节奏和红板不同步（0.7 是频率，1.0 是相位偏移），看起来更自然。

```cpp
// 第38行
int bw = static_cast<int>(180 * s), bh = static_cast<int>(160 * s);
```
- 黑底的宽和高都乘以 s：s 变大板子变大，s 变小板子变小。

```cpp
// 第39、40行
int bx = 380 - (bw - 180) / 2;
int by = 100 - (bh - 160) / 2;
```
- **为什么要减**：板子放大后左上角位置要往左上挪一点，才能让**板子中心固定不动**、只是整体变大变小。否则放大时会往右下"跑"。

```cpp
// 第41、42行
int lw = static_cast<int>(16 * s), lh = static_cast<int>(100 * s);
int ly = by + (bh - lh) / 2;
```
- 灯条宽高也按 s 缩放；`ly` 让灯条在黑底里**垂直居中**。

```cpp
// 第43–45行：画出来（和红板同理）
cv::rectangle(img, cv::Rect(bx, by, bw, bh), cv::Scalar(20, 20, 20), cv::FILLED);  // 黑底
cv::rectangle(img, cv::Rect(bx + 30, ly, lw, lh), cv::Scalar(255, 0, 0), cv::FILLED);  // 左蓝条
cv::rectangle(img, cv::Rect(bx + bw - 30 - lw, ly, lw, lh), cv::Scalar(255, 0, 0), cv::FILLED);  // 右蓝条
```
- `cv::Scalar(255,0,0)`：B=255 → **蓝色**。
- 第二根蓝条的位置用 `bx + bw - 30 - lw`，意思是"从板子右边缘往左数 30"，这样板子缩放时右灯条始终贴着右边。
- **为什么缩放**：模拟装甲板"由远及近/由近及远"——近了看起来大，远了看起来小。这样 PnP 算出的距离会随帧变化，能验证测距是否真的工作。

### 第 47–49 行：模糊 + 返回
```cpp
cv::GaussianBlur(img, img, cv::Size(5, 5), 0);   // 第47行
return img;                                       // 第48行
}                                                 // 第49行
```
- `GaussianBlur`：高斯模糊，用一个 5×5 的小窗口把图像"抹匀"一点。
- **为什么**：真实摄像头拍的画面边缘没那么锐利（镜头失焦、运动模糊），加一点模糊更接近真实场景，也能测试算法抗模糊能力。
- `return img;` 把这一帧返回给调用者；`}` 结束函数。

---

## 第 ③ 块：makeTestVideo —— 把很多帧拼成视频（第 52–66 行）

### 第 52 行
```cpp
static bool makeTestVideo(const std::string& path) {
```
- 返回 `bool`（true 成功 / false 失败），调用者能据此判断是否继续。
- `const std::string& path`：`const` 表示函数里不修改这个字符串；`&` 引用表示不拷贝、直接传进来（省内存）。

### 第 54–55 行：创建视频写入器
```cpp
cv::VideoWriter writer(path, cv::VideoWriter::fourcc('M','J','P','G'),
                       30.0, cv::Size(640, 360));
```
- `VideoWriter`：视频"写入器"，相当于视频版的 `imwrite`。
- `fourcc('M','J','P','G')`：指定编码格式为 MJPG。**为什么选它**：兼容性最好，Windows 和 Ubuntu 都能写、都能播。
- `30.0`：每秒 30 帧（30 fps）。
- `cv::Size(640,360)`：画面尺寸，必须和每帧图一致（makeFrame 造的就是 640×360）。
- **为什么这四个参数必须在写第一帧前定好**：视频文件开头要写入这些"元信息"，后面每帧都得遵守。

### 第 56–59 行：检查是否创建成功
```cpp
if (!writer.isOpened()) {
    std::cerr << "[armor] 无法创建测试视频: " << path << '\n';
    return false;
}
```
- `isOpened()`：检查写入器是否真的可用（路径无效、没装编码器都会失败）。
- `std::cerr`：错误输出流，专门打印错误信息。
- `return false`：失败就提前退出，避免后面白白写 90 帧。
- **这是"防御式编程"**：假设可能失败，先检查，早发现早报错。

### 第 60–62 行：循环写 90 帧
```cpp
for (int i = 0; i < 90; ++i) {
    writer.write(makeFrame(i * 0.12));
}
```
- 循环 90 次，每次：
  - `i * 0.12`：把整数 i 变成变化的 phase（i=0→0，i=89→10.68）。
  - `makeFrame(...)`：造一帧这个 phase 对应的图。
  - `writer.write(...)`：把这一帧追加写进视频。
- 90 帧 ÷ 30fps = **3 秒**视频。
- **为什么 phase 每次加 0.12**：步长适中 —— 太大动画跳得快，太小几乎看不出动。

### 第 63–66 行：收尾
```cpp
writer.release();     // 63行
std::cout << ...;     // 64行：打印成功信息
return true;          // 65行
}                     // 66行
```
- `release()`：**必须调用**。写入器内部有缓冲区，不释放的话数据可能没全部刷进文件，导致视频损坏或打不开。
- `return true` 告诉调用者"生成成功了"。

---

## 第 ④ 块：loadCalib —— 读取相机参数（第 69–87 行）

### 背景：为什么需要相机参数
PnP 测距要靠相机的"焦距、主点、畸变"这些数字（内参）。考核会给一个 YAML 配置文件，这个函数负责把它读出来。

### 第 69 行
```cpp
static bool loadCalib(const std::string& path, CameraCalib& calib) {
```
- 两个参数：`path` 只读（const）；`calib` **不带 const** —— 因为函数要把读到的参数**写进 calib**（这叫"输出参数"，靠引用传回结果）。

### 第 70–76 行：打开文件 + 读矩阵
```cpp
cv::FileStorage fs(path, cv::FileStorage::READ);  // 70行：以只读方式打开配置文件
if (!fs.isOpened()) return false;                  // 71行：打不开就失败
cv::Mat K, D;                                      // 72行：K存内参矩阵，D存畸变系数
fs["camera_matrix"] >> K;                          // 73行：按键名读取 3x3 内参矩阵
fs["distortion_coefficients"] >> D;                // 74行：读取畸变系数向量
fs.release();                                      // 75行：关闭文件
if (K.empty()) return false;                       // 76行：内参没读到就失败
```
- `FileStorage` 是 OpenCV 读写 YAML/XML 配置文件的工具。
- `fs["键名"] >> 变量`：按文件中的键名取值，**键名必须和 YAML 里完全一致**。
- `K.empty()`：检查是否为空，空说明文件里没有这个字段。

### 第 77–85 行：拆解矩阵到结构体
```cpp
calib.fx = K.at<double>(0, 0);   // 77行：焦距 x
calib.fy = K.at<double>(1, 1);   // 78行：焦距 y
calib.cx = K.at<double>(0, 2);   // 79行：主点 x（光心位置）
calib.cy = K.at<double>(1, 2);   // 80行：主点 y
```
- 内参是一个 3×3 矩阵，标准形状是：
  ```
  [ fx   0   cx ]
  [  0  fy   cy ]
  [  0   0    1 ]
  ```
  `K.at<double>(行, 列)` 就是取第几行第几列，所以 fx 在 (0,0)、cx 在 (0,2)。

```cpp
if (!D.empty() && D.rows >= 5) {           // 81行：至少有5个畸变系数才读
    calib.k1 = D.at<double>(0); calib.k2 = D.at<double>(1);  // 82行
    calib.p1 = D.at<double>(2); calib.p2 = D.at<double>(3);  // 83行
    calib.k3 = D.at<double>(4);                               // 84行
}
```
- `D.rows >= 5`：**边界检查**，不够 5 个就去读第 5 个会越界崩溃。这是防呆。
- k1,k2,k3 是径向畸变，p1,p2 是切向畸变 —— 描述镜头把直线"掰弯"的程度。

### 第 86–87 行
```cpp
return true;   // 成功
}              // 结束函数
```

---

## 第 ⑤ 块：main 准备段 —— 搭好舞台（第 89–152 行）

### 第 89 行：程序入口
```cpp
int main(int argc, char** argv) {
```
- 所有 C++ 程序从这里开始执行。
- `argc`：命令行参数个数；`argv`：参数内容数组。比如运行 `./armor_video --video a.avi`，则 argc=3，argv[0] 是程序名，argv[1]="--video"，argv[2]="a.avi"。

### 第 90–92 行：准备配置变量
```cpp
std::string videoPath, outPath = "armor_result.avi", calibPath;
bool show = true, onlyMakeTest = false;
int dumpFrame = -1;
```
- `videoPath`：输入视频路径（默认空）；`outPath`：输出视频路径（默认 armor_result.avi）；`calibPath`：相机参数路径。
- `show`：要不要弹窗口显示，默认 true；`onlyMakeTest`：是否只生成测试视频，默认 false。
- `dumpFrame = -1`：**哨兵值**。-1 表示"不导出任何单帧"（因为帧号从 0 开始，不会是负数，用 -1 当"没设置"的标志很安全）。

### 第 95–108 行：解析命令行参数
```cpp
for (int i = 1; i < argc; ++i) {          // 95行：从1开始，跳过程序名
    std::string arg = argv[i];            // 96行：取出当前参数
    if (arg == "--video" && i + 1 < argc) videoPath = argv[++i];   // 97行
    else if (arg == "--out" && i + 1 < argc) outPath = argv[++i];   // 98行
    else if (arg == "--camera" && i + 1 < argc) calibPath = argv[++i]; // 99行
    else if (arg == "--make-test") onlyMakeTest = true;             // 100行
    else if (arg == "--no-show") show = false;                      // 101行
    else if (arg == "--dump-frame" && i + 1 < argc) dumpFrame = std::stoi(argv[++i]); // 102行
    else if (arg.rfind("--", 0) != 0) videoPath = arg;              // 103行
    else { /* 未知参数，报错 */ }                                    // 104–107行
}
```
- `i + 1 < argc`：**防越界检查** —— 确认后面还有值可取，否则 `--video` 后面没路径会崩。
- `argv[++i]`：`++i` 先把 i 加 1（跳到参数值），再取它。比如 `--video a.avi` 遇到 `--video` 时 i 指向它，`++i` 后 i 指向 a.avi。
- 第102行 `std::stoi`：把字符串转成整数（"45" → 45）。
- 第103行 `arg.rfind("--", 0) != 0`：`rfind("--",0)` 检查是否**以 "--" 开头**。不以 "--" 开头 → 认为是直接给的视频路径（简写 `--video` 的偷懒写法）。
- 第104–107行：其它不认识的参数 → 打印错误并 `return 1` 结束程序。

### 第 111–117 行：没数据就自动造
```cpp
if (onlyMakeTest) {
    return makeTestVideo(videoPath.empty() ? "test_scene.avi" : videoPath) ? 0 : 1;
}
if (videoPath.empty()) {
    videoPath = "test_scene.avi";
    if (!makeTestVideo(videoPath)) return 1;
}
```
- 第112行：如果只要生成测试视频，调用 `makeTestVideo` 后立即结束（成功返回 0，失败返回 1）。里面用了两层三元运算符 `? :`。
- 第114–117行：如果没给视频路径，就自动造一个 `test_scene.avi` 来处理。**好处**：你手上没考核数据时，程序也能立刻跑起来演示。

### 第 120–129 行：打开视频 + 读取信息
```cpp
cv::VideoCapture cap(videoPath);              // 120行：视频"读取器"，相当于视频版 imread
if (!cap.isOpened()) { ...; return 1; }       // 121–124行：打不开就报错退出
double fps = cap.get(cv::CAP_PROP_FPS);       // 125行：帧率（每秒几帧）
int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));   // 126行：宽
int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));  // 127行：高
int total  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));   // 128行：总帧数
if (fps <= 0 || fps > 120) fps = 30;          // 129行：帧率异常时的兜底
```
- `VideoCapture`：视频读取器。构造时传文件路径就打开该文件；传 `0` 则打开摄像头。
- `cap.get(属性)`：读取视频的元信息。`CAP_PROP_FPS` 帧率、`FRAME_WIDTH/HEIGHT` 尺寸、`FRAME_COUNT` 总帧数。
- 第129行：**为什么兜底** —— 有些视频格式的帧率读出来是 0 或乱值，不处理的话后面写视频会出错，所以强行设为 30。

### 第 132–137 行：准备输出视频
```cpp
cv::VideoWriter writer(outPath, cv::VideoWriter::fourcc('M','J','P','G'),
                       fps, cv::Size(width, height));  // 132–133行
if (!writer.isOpened()) { ...; return 2; }              // 134–137行
```
- **关键**：输出的 `fps` 和 `cv::Size(width,height)` 都取自输入视频 —— 保证输出视频和输入一样大、一样快。若不一致，结果视频可能播放异常。

### 第 140–144 行：相机参数
```cpp
CameraCalib calib;                                    // 140行：用默认值创建对象
if (!calibPath.empty() && loadCalib(calibPath, calib)) // 141行：尝试从文件读
    std::cout << "[armor] 已载入相机参数: " ...;        // 142行：成功提示
else
    std::cout << "[armor] 使用内置占位相机参数...";      // 144行：警告用默认值
```
- `CameraCalib calib;`：创建一个参数对象，默认是占位值（只匹配合成图）。
- **为什么要警告**：占位参数不是真实相机的，测出的距离不准。考核时必须用 `--camera` 传老师给的真实参数。

### 第 146–152 行：初始化 + 打印信息
```cpp
ArmorDetector detector;    // 146行：造一个检测器对象（核心工具）
int frameNo = 0;           // 147行：帧计数器，从0开始
double sumMs = 0.0;        // 148行：累计耗时，最后算平均
// 150–152行：打印输入视频的路径、尺寸、帧率、总帧数
```
- `ArmorDetector detector;`：这才是真正干活的"检测器"，`detect()` 方法就在它身上。
- `frameNo` 用来数处理到第几帧；`sumMs` 累计每帧耗时，最后求平均值。

---

## 第 ⑥ 块：核心循环 + 收尾（第 155–207 行）—— 全文件最重要

### 第 155 行：无限循环
```cpp
while (true) {
```
- 条件恒为 true，会一直转，靠循环体内的 `break` 跳出。
- **为什么用 while(true)**：视频帧数事先不确定（读到结尾才知道），用 while + break 最自然。

### 第 156–158 行：读一帧
```cpp
cv::Mat frame;                                              // 156行：空的图，每帧复用
if (!cap.read(frame) || frame.empty()) break;              // 158行
```
- `cap.read(frame)`：读下一帧放进 `frame`，成功返回 true，读到结尾返回 false。
- `|| frame.empty()`：双重保险 —— 即使 read 返回 true 但图是空的，也退出。
- `break`：跳出 while 循环 → 视频处理结束。
- **这就是视频处理的"心跳"**：每转一圈，处理一帧。

### 第 161–164 行：计时（考核要求显示耗时）
```cpp
auto t0 = std::chrono::steady_clock::now();       // 161行：记开始时刻
std::vector<Armor> armors = detector.detect(frame, calib);  // 162行：核心检测！
auto t1 = std::chrono::steady_clock::now();       // 163行：记结束时刻
double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();  // 164行
```
- `auto`：让编译器自动推断变量类型（这里推断出时钟时间点类型），省得写又长又难记的类型名。
- 第162行 `detector.detect(frame, calib)`：**整套算法就这一行** —— 输入这一帧和相机参数，输出检测到的装甲板列表 `vector<Armor>`。图片版、视频版这行完全一样。
- `steady_clock`：稳定时钟，不受系统时间调整影响，专门用来测时间差。
- `duration<double, std::milli>(t1-t0).count()`：把时间差换算成**毫秒**数字。
- **类比**：t0 是摁下秒表，t1 是按下停止，elapsedMs 是读数。夹在中间的就是要测的检测耗时。

### 第 167 行：画结果
```cpp
cv::Mat result = ArmorDetector::drawResult(frame, armors, elapsedMs);
```
- 把检测结果（四角点、中心点、三维坐标）和耗时**画到图上**，返回画好的新图 `result`。
- 因为是 `static` 静态函数，用类名 `ArmorDetector::` 直接调用，不需要对象。

### 第 170–172 行：在画面上写帧号
```cpp
std::string info = "Frame " + std::to_string(frameNo + 1) + "/" + std::to_string(total);  // 170行
cv::putText(result, info, cv::Point(width - 130, 22), cv::FONT_HERSHEY_SIMPLEX,
            0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);                                // 171–172行
```
- 第170行：拼出 `"Frame 45/90"` 这样的字符串。`std::to_string` 把数字转字符串。`frameNo + 1` 因为帧号从 0 开始，显示时给人看要从 1 开始。
- 第171–172行 `putText`：把文字画到画面上。
  - `cv::Point(width - 130, 22)`：**右上角**坐标（x = 宽度-130，y=22）。为什么放右上角？左上角已经被耗时文字占了（在 drawResult 里画的），避免重叠。
  - `0.6`：字号；`cv::Scalar(255,255,255)`：白色；`1`：线宽；`cv::LINE_AA`：抗锯齿（字边缘更平滑）。

### 第 175 行：写入视频
```cpp
writer.write(result);
```
- 把画好的这一帧追加到输出视频。**每帧都写一次**，90 帧写完就是完整视频。

### 第 178–181 行：可选导出单帧
```cpp
if (frameNo == dumpFrame) {
    cv::imwrite("frame_dump.png", result);
    std::cout << "[armor] 已导出第 " << frameNo << " 帧 → frame_dump.png\n";
}
```
- 如果这一帧的编号正好等于你用 `--dump-frame` 指定的编号，就另存为一张 png 图片。
- **有什么用**：想在视频里挑某一帧仔细看检测效果时特别方便（虚拟机里没播放器也能看图）。
- 因为 `dumpFrame` 默认是 -1，正常运行时这个 if 永远不成立，不影响性能。

### 第 185–188 行：实时显示
```cpp
if (show) {
    cv::imshow("armor video", result);       // 186行：在窗口显示这一帧
    if (cv::waitKey(1) == 27) break;         // 187行
}
```
- `imshow`：显示这一帧。
- `waitKey(1)`：**只等 1 毫秒**就继续 —— 这样窗口才能快速刷新，视频才连贯。
  - ⚠️ **对比**：图片版用 `waitKey(0)`（无限等按键）。视频版若写成 0，会卡在第一帧等你按键，视频"不动"。这是最经典的坑。
- `== 27`：27 是 Esc 键的编码，按 Esc 可提前退出（break 跳出循环）。

### 第 190–193 行：统计与进度
```cpp
sumMs += elapsedMs;                                    // 190行：累加耗时
if ((frameNo + 1) % 30 == 0)                           // 191行：每30帧
    std::cout << "[armor] 已处理 " << (frameNo + 1) << " / " << total << " 帧\n";  // 192行
++frameNo;                                             // 193行：帧号+1
```
- 第190行：把这一帧的耗时累加到 `sumMs`，最后算平均。
- 第191行 `% 30 == 0`：`%` 是取余数。帧号是 30 的倍数时（每处理 30 帧）才打印一次进度，避免刷屏。
- 第193行 `++frameNo`：计数器加 1，准备处理下一帧。
- 第194行 `}`：while 循环结束。

### 第 197–199 行：释放资源
```cpp
cap.release();                    // 197行：关闭视频读取器
writer.release();                 // 198行：关闭视频写入器（必须！否则文件不完整）
if (show) cv::destroyAllWindows();// 199行：关闭显示窗口
```
- `release()` 释放占用的资源和缓冲区。**不 release 输出视频可能打不开**，这是新手常见问题。

### 第 201–206 行：打印统计 + 结束
```cpp
double avgMs = frameNo > 0 ? sumMs / frameNo : 0.0;  // 201行：平均耗时（防除零）
std::cout << "[armor] 处理完成，共 " << frameNo << " 帧\n";            // 202行
std::cout << "[armor] 平均每帧耗时: " << avgMs << " ms（理论上限 "
          << (avgMs > 0 ? 1000.0 / avgMs : 0) << " fps）\n";           // 203–204行
std::cout << "[armor] 结果视频已保存: " << outPath << '\n';            // 205行
return 0;                                                              // 206行
}                                                                      // 207行
```
- 第201行：`frameNo > 0 ? ... : 0.0` —— 三元运算符，防止一帧都没处理时除以 0。
- 第203–204行：`1000.0 / avgMs` 把"每帧多少毫秒"换算成"每秒多少帧"（1 秒=1000 毫秒）。比如每帧 5ms → 理论 200fps。
- `return 0`：告诉操作系统程序正常结束（非 0 表示异常）。

---

## 全文件核心概念速记

| 概念 | 白话解释 |
|---|---|
| `cv::Mat` | 一张图片，视频的每一帧都是它 |
| `VideoCapture` | 视频读取器，一帧一帧往外取 |
| `VideoWriter` | 视频写入器，一帧一帧拼成视频 |
| `cap.read()` | 取一帧，读到结尾返回 false |
| `while(true) + break` | 无限循环，读到视频结尾就跳出 |
| `waitKey(1)` | 视频显示等 1ms；图片版才是 `waitKey(0)` |
| `release()` | 收尾释放，不写视频文件会损坏 |
| `chrono` | 高精度计时，夹住 detect() 测耗时 |
| `static`（函数前） | 内部工具，只在本文件可见 |
| `const &` | 只读引用，不拷贝、不修改 |
| 引用传参（`CameraCalib&`） | 让函数能把结果写回调用者的变量 |
| `auto` | 让编译器自动推断类型 |
| `std::stoi` | 字符串转整数 |
| `rfind` | 检查字符串前缀/后缀 |
| 三元 `? :` | 简写的 if-else |
| BGR | OpenCV 颜色顺序是蓝绿红，红=(0,0,255)，蓝=(255,0,0) |
