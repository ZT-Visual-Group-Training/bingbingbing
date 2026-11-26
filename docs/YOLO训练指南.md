# YOLO 训练指南（装甲板数字识别专用）

> 面向：零训练经验、已跑通手写装甲板检测的人
> 核实状态：`[官方]` = 来自 Ultralytics 官方文档（docs.ultralytics.com）已核实；`[经验]` = 工程建议，非官方硬性要求。
> 重要前提：**训练在 Python 做，推理在 C++ 做**。你之前的《YOLO使用详解.md》讲的是 C++ 推理（用现成模型），
> 本指南讲的是前一步——怎么把模型"训出来"。

---

## 零、先想清楚：你要训练"检测"还是"分类"？

| 路线 | 输入 | 输出 | 适合你吗 |
|---|---|---|---|
| **检测 detect** | 整张图 | 数字框 + 类别 + 置信度 | 想让 YOLO 一手包办"找板+认数字" |
| **分类 classify** | 已裁好的数字小图 | 是几号 | ⭐ 更省事：你已有流水线能框板，只需认数字 |

**建议**：先用你现有的 `detect()` 框出装甲板 → 透视拉正 → 裁出数字区，
然后只训练一个**五分类模型**认 1~5。标注工作量比检测小得多（不用画框，只需给小图打类别标签）。
本指南下面按**检测**路线写（更通用），分类路线在第 6 节给出改法。

---

## 一、装环境（一次性的）

```bash
pip install ultralytics
```

验证装好：

```bash
yolo version
```

能打印版本号即可。`[经验]` 建议 Python 3.9~3.11，太新（3.13）可能有依赖问题。

---

## 二、准备数据集（最费人工的一步）

### 2.1 采集图片

从你的考核视频抽帧，每类数字至少 **100~200 张** `[经验]`，五个数字共 500~1000 张起。
覆盖不同距离、角度、光照——数据越全，模型越稳。

### 2.2 标注

用 LabelImg 或 Roboflow 等工具，给每张图里的数字画框，标类别（1~5）。
导出为 YOLO 格式：一张图对应一个同名 `.txt`，每行一个目标：

```
<class_id>  <x_center>  <y_center>  <width>  <height>
```

`[官方]` 关键规则：
- 坐标是**归一化的 0~1**：`x_center = 像素x / 图宽`，`width = 框宽 / 图宽`，y 同理除以图高
- 类别编号**从 0 开始**：数字 1→类别 0，数字 2→类别 1，……数字 5→类别 4

### 2.3 目录结构 `[官方]`

```
armor_dataset/
├── images/
│   ├── train/     # 训练图
│   └── val/       # 验证图
└── labels/
    ├── train/     # 对应标签 txt
    └── val/
```

按 8:2 划分训练和验证集。

---

## 三、写 data.yaml `[官方]`

在 `armor_dataset/` 下新建 `data.yaml`：

```yaml
path: armor_dataset          # 数据集根目录
train: images/train          # 相对 path
val: images/val

names:                       # 类别名，编号从 0 开始
  0: '1'
  1: '2'
  2: '3'
  3: '4'
  4: '5'
```

`[官方]` `names` 的键数就是类别数，新版可省略 `nc` 字段（旧教程写的 `nc: 5` 保留也无妨）。

---

## 四、训练 `[官方]`

### Python 方式（推荐）

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")       # 加载预训练权重，收敛更快（强烈建议）
results = model.train(
    data="armor_dataset/data.yaml",
    epochs=100,                  # 训练轮数，不够就加到 150~300
    imgsz=640,                   # 输入尺寸，必须和导出时一致
)
```

### 命令行方式

```bash
yolo detect train data=armor_dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

`[官方]` 训练会输出指标图表和权重到 `runs/detect/train/`，其中 `weights/best.pt` 是效果最好的模型。

### 中断后继续 `[官方]`

```python
model = YOLO("runs/detect/train/weights/last.pt")
results = model.train(resume=True)
```

`[官方]` 至少跑完 1 个 epoch 才能 resume（默认每轮结束存一次 checkpoint）。

---

## 五、导出 ONNX 给 C++ 用 `[官方 + 经验]`

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12 dynamic=False imgsz=640
```

`[经验]` 三个参数是为 OpenCV DNN 兼容：
- `opset=12`：算子版本太高 OpenCV 不认识（最常见的坑）
- `dynamic=False`：固定输入尺寸，别用动态轴
- `imgsz=640`：和训练时一致

产物 `best.onnx` 拷进 D:/demo5，就能用《YOLO使用详解.md》里的 C++ 代码加载推理。

---

## 六、分类路线（更省事，推荐先试）

如果按第零节的"分类"路线，改动只有两处：

1. 数据集结构改为按类别分文件夹，无需画框：

```
armor_cls/
├── train/
│   ├── 1/  ├── 2/  ├── 3/  ├── 4/  ├── 5/
└── val/
    ├── 1/  ├── 2/  ├── 3/  ├── 4/  ├── 5/
```

2. 训练命令换成分类：

```python
from ultralytics import YOLO
model = YOLO("yolov8n-cls.pt")
results = model.train(data="armor_cls", epochs=100, imgsz=64)
```

`[经验]` 分类模型的 `imgsz` 可以很小（数字小图 64×64 就够），训练快、模型也小。

---

## 七、常见坑

| 现象 | 原因 | 解法 |
|---|---|---|
| 训练报找不到类别 | yaml 路径写错 / 标签类别超范围 | 检查 `path` 相对关系、类别编号 0~4 |
| loss 不降、准确率低 | 数据太少或标注质量差 | 每类补到 100+ 张，检查框有没有画歪 |
| C++ 加载 onnx 报错 | opset 太高 / 动态轴 | 按第五节加 `opset=12 dynamic=False` |
| 导出后类别对不上 | 导出和训练的 names 不一致 | 用同一个 best.pt 导出，别手动改 yaml |
| 通用模型认不出数字 | yolov8n.pt 是 COCO 80 类，没有装甲板数字 | 必须按本指南用自己的数据训练 |

---

## 八、诚实说明局限

- 训练效果上限由**数据质量**决定。合成图（像 `digit_demo.cpp` 那样）能凑数启动流程，
  但和真实视频有域差距，别指望纯合成数据训出的模型在实拍上好用。
- 你当前环境（截至 2026-11-25）**没有装 ultralytics、也没有任何 .onnx/.pt 模型**，
  本指南是"照着做"的操作说明，**尚未在本机实际跑过训练**。跑之前请按第一节装环境。
