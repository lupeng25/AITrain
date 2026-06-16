# AITrain Studio 用户使用手册

本文面向 AITrain Studio 的终端用户，说明如何在图形界面中完成数据集制作、导入、校验、训练、评估、部署验证和交付证据汇总。本文不介绍源码结构、Worker 协议或 Python trainer JSON 细节。

## 1. 快速闭环

推荐按下面顺序使用：

```text
启动并注册
  -> 创建或打开项目
  -> 环境自检
  -> 制作或导入数据集
  -> 必要时执行数据集格式转换
  -> 校验数据集
  -> 生成质量报告并修复问题
  -> 划分数据集或创建数据快照
  -> 启动训练
  -> 在“任务与产物”查看 checkpoint、ONNX、报告和预览图
  -> 评估模型并在“模型库 > 评估报告”查看结果
  -> 注册到“模型库”
  -> 在“部署验证”完成模型导出、导出后验证和推理验证
  -> 环境 > 交付证据和诊断包
```

重要边界：

- 主流程优先使用官方 YOLO / PaddleOCR 后端，训练由 Worker 启动独立 Python 进程执行。
- 生产训练只使用官方后端；旧的 tiny detector、Python mock、小型 OCR CTC 和 C++ 分割/OCR 训练 scaffold 已从产品训练路径中物理删除。
- TensorRT 需要兼容的 NVIDIA RTX / SM 75+ 环境；不支持的 GPU 会显示为 `hardware-blocked`。
- YOLO12 分割 `.pt` 预训练权重当前依赖 Ultralytics 上游是否提供 `yolo12*-seg.pt`；2026-06-14 记录的 Ultralytics 8.3.171 环境无法解析 `yolo12n-seg.pt`，应视为上游权重 blocker，不是数据集或 AITrain C++ runtime 问题。
- YOLO26 当前是独立兼容阶段，不属于 P1 主矩阵。2026-06-14/15 共享 Ultralytics 8.3.171 生命周期环境下 20 个 YOLO26 行全部失败，属于模型配置/官方权重不可用或包代码不兼容 blocker；随后隔离 YOLO26 targeted full 在 2026-06-15 通过 20/20 训练、官方 ONNX、AITrain C++ ONNX 推理和 TensorRT 验证。YOLO26 不支持导出或转换为 NCNN。
- OCR 的公开数据或生成数据 smoke 只能证明流程和产物可用，不能替代客户业务数据上的精度验收；当前 OCR 产品路线仅覆盖 PaddleOCR Det / Rec / System 官方工具链。

## 2. 启动、授权和项目

首次启动 `AITrainStudio.exe` 时，如果没有有效注册码，会先显示注册窗口。

1. 点击“复制机器码”。
2. 将机器码发给授权方。
3. 收到 `AITRAIN1...` 开头的离线注册码后粘贴到注册窗口。
4. 点击“验证并启动”进入主界面。

注册码绑定当前机器。换机器后需要使用新机器码重新签发注册码。主程序只内置公钥；私钥只应保存在授权方的注册码生成器环境中。

进入主界面后，先在“项目”页创建或打开项目。项目目录用于集中保存：

- 数据集索引和校验记录
- Worker 任务历史
- 训练报告、评估报告和导出记录
- 模型版本和交付产物

## 3. 环境自检

在“环境”页点击“执行环境自检”。自检结果会分组显示：

| Profile | 检查内容 | 常见处理 |
|---|---|---|
| YOLO | Python、Ultralytics、Torch、ONNX、ONNX Runtime | 安装或切换到 YOLO 专用 Python 环境 |
| OCR | PaddlePaddle、PaddleOCR、PaddleOCR 源码 checkout、官方脚本可用性 | 使用隔离 OCR Python 环境，避免 Torch / Paddle DLL 冲突 |
| TensorRT | NVIDIA 驱动、CUDA、cuDNN、TensorRT、GPU compute capability | 使用 RTX / SM 75+ 机器，旧 GPU 保持 `hardware-blocked` |

如果环境自检失败，先按“环境”页的修复建议处理，再启动训练、导出或推理。不要把缺少依赖的训练失败当作数据集或模型问题。

如果训练参数使用 `device=0` 或其他 GPU 设备号，YOLO Python 环境必须安装 CUDA 版 PyTorch。CPU-only 环境只能使用 `device=cpu`，否则会在训练前失败。现场可通过训练参数 `pythonExecutable` 或环境变量 `AITRAIN_PYTHON_EXECUTABLE` 指向已验证的 CUDA YOLO Python。

## 4. 制作数据集

### 4.1 标注工具

“数据集”页使用 X-AnyLabeling 作为外部标注工具。程序会从以下位置检测：

- 环境变量 `AITRAIN_XANYLABELING_EXE`
- 程序目录
- `tools/x-anylabeling`
- `.deps/annotation-tools/X-AnyLabeling`
- `PATH`

在“数据集”页选择数据集目录后，可以点击“启动 X-AnyLabeling”打开标注工具。标注完成后回到 AITrain Studio，点击“标注后刷新 / 重新校验”。

### 4.2 YOLO 检测数据集

YOLO 检测用于目标框训练。推荐目录：

```text
dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

标签文件与图片同名，扩展名为 `.txt`。每行格式为：

```text
class_id center_x center_y width height
```

坐标使用 0 到 1 的归一化值。X-AnyLabeling 中应导出 YOLO bbox 标签。

### 4.3 YOLO 分割数据集

YOLO 分割用于多边形 mask 训练。目录结构与 YOLO 检测一致。每行标签格式为：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

坐标同样使用 0 到 1 的归一化值。X-AnyLabeling 中应导出 YOLO polygon 标签。

### 4.4 PaddleOCR Rec 数据集

OCR Rec 用于文字识别。推荐目录：

```text
dataset/
  dict.txt
  rec_gt.txt
  images/
    sample.png
```

`rec_gt.txt` 每行包含图片相对路径和文字标签，中间用 Tab 分隔：

```text
images/sample.png<TAB>label
```

`dict.txt` 是字符字典。训练前应确认标签中出现的字符都包含在字典中。

### 4.5 PaddleOCR Det 数据集

OCR Det 用于文字检测。推荐目录：

```text
dataset/
  det_gt.txt
  images/
    sample.png
```

`det_gt.txt` 每行包含图片相对路径和 JSON 数组，中间用 Tab 分隔：

```text
images/sample.png<TAB>[{"transcription":"text","points":[[1,1],[30,1],[30,20],[1,20]]}]
```

`transcription` 为 `###` 或 `*` 时会作为 PaddleOCR 忽略文本保留。

## 5. 导入、校验、划分和快照

在“数据集”页完成以下操作：

1. 点击“选择数据集”，选择数据集根目录。
2. 检查“格式”是否自动识别正确；必要时手动选择。
3. 如果源数据集不是目标训练格式，先执行“数据集格式转换”。
4. 点击“校验数据集”。
5. 如果校验失败，查看问题表格和校验 JSON。
6. 点击“生成质量报告”，查看缺失标签、孤立标签、非法框、多边形错误、OCR 标签错误、重复样本等问题。
7. 需要修复时点击“打开问题清单”或“X-AnyLabeling 修复”。
8. 修复后点击“标注后刷新 / 重新校验”。

### 5.1 数据集格式转换

“数据集格式转换”用于把已有 COCO、Pascal VOC 或 YOLO 标注转换为当前训练流程可用的布局。当前 GUI 暴露的是已实现的 COCO / Pascal VOC / YOLO 检测 / YOLO 分割转换矩阵；不在下拉框或报告中标记为 supported 的组合不能当作已实现能力。

转换步骤：

1. 在“数据集”页选择源格式和目标格式。
2. 选择源路径。COCO 通常选择标注 JSON，VOC 可选择 XML 文件或 XML 目录，YOLO 选择包含 `data.yaml` 或标签目录的根目录。
3. 选择输出目录。建议使用新的空目录，避免覆盖已有数据集。
4. 按需要设置是否复制图片。复制图片更利于交付和迁移；引用图片更快但依赖原始路径。
5. 点击“开始转换”，在进度、日志和任务列表中查看执行状态。
6. 转换完成后查看输出路径和 conversion report。
7. 需要训练时，手动选择或导入转换后的输出目录，再重新运行“校验数据集”。

转换任务由 Worker 执行，支持进度、日志和取消。转换结果不会自动注册为新数据集，也不会自动替换当前选中的数据集；这是为了避免误把中间产物当作已验收数据集。详细矩阵和报告字段见 `docs/dataset-conversion.md`。

### 5.2 数据集划分和快照

只有通过当前格式校验的数据集才能进入主训练流程。数据集划分支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec。

划分时可以设置：

- 输出目录，留空时默认写入项目的 `datasets/normalized`
- train / val / test 比例
- 随机 seed

数据快照用于复现实验。可以在“数据集”页手动点击“创建数据快照”；如果启动训练时没有可用快照，程序会自动排队创建快照，再继续训练。

## 6. 训练模型

进入“训练实验”页前，先确保已经打开项目，并且在“数据集”页选择了通过校验的数据集。

训练步骤：

1. 检查顶部“当前数据集”和“摘要”。
2. 选择任务类型。
3. 选择训练后端。
4. 选择或输入模型预设。
5. 设置 Epochs、Batch Size、Image Size。
6. 如需继续训练，在“高级 / 诊断后端”中选择 Resume checkpoint。
7. 点击“启动训练”。
8. 在“训练监控”查看进度和指标曲线。
9. 在“训练日志”和“任务与产物”查看日志、checkpoint、报告、ONNX 和预览图。

推荐后端：

| 数据/任务 | 推荐后端 | 推荐模型预设 | 说明 |
|---|---|---|---|
| YOLO 检测 | `ultralytics_yolo_detect` | 默认 `yolov8n.yaml`；可选 YOLOv5u 标准 P5 检测、YOLOv8 / YOLO11 / YOLO12 `n/s/m/l/x` `.yaml`、`.pt`，以及 YOLOv8 P2/P6 `.yaml`；YOLO26 检测按独立 targeted matrix 证据放行训练/ONNX/TensorRT，NCNN 不作为可选目标 | 官方 Ultralytics 检测训练、ONNX 导出和 `val()` 评估；推理、benchmark、部署验证走 AITrain C++ runtime |
| YOLO 分割 | `ultralytics_yolo_segment` | 默认 `yolov8n-seg.yaml`；可选 YOLOv8 / YOLO11 `n/s/m/l/x` `-seg.yaml`、`-seg.pt`，YOLO12 `n/s/m/l/x` `-seg.yaml`；YOLO12 `-seg.pt` 按上游权重 blocker 处理，YOLO26 分割按独立 targeted matrix 证据放行训练/ONNX/TensorRT，NCNN 不作为可选目标 | 官方 Ultralytics 分割训练、ONNX 导出和 `val()` 评估；mask 后处理、推理、benchmark、部署验证走 AITrain C++ runtime |
| 语义分割 Mask PNG | `smp_semantic_segmentation` | 默认 `smp_unet_resnet34`；可选 `smp_unetplusplus_resnet34`、`smp_fpn_resnet34`、`smp_deeplabv3plus_resnet50`、`smp_segformer_mit_b0` | SMP 专用语义分割，要求 `classes.txt` 和单通道 PNG class-id mask；训练导出 `best.pt`、`best.onnx` 和 SMP 报告；推理、overlay、benchmark 和部署验证只承诺 ONNX Runtime；SMP 不需要 NCNN/TensorRT 导出 |
| PaddleOCR Det | `paddleocr_det_official` | 默认 `PP-OCRv5_mobile_det`；可选 `PP-OCRv4_mobile_det`、`PP-OCRv5_server_det`、`PP-OCRv6_tiny/small/medium_det` | 官方 PaddleOCR 检测工具链，建议使用隔离 OCR 环境；PP-OCRv5/v6 preset 需要 PaddleOCR 源码 checkout |
| PaddleOCR Rec | `paddleocr_rec_official` | 默认 `PP-OCRv5_mobile_rec`；可选 `PP-OCRv4_mobile_rec`、`PP-OCRv5_server_rec`、`en_PP-OCRv5_mobile_rec`、`PP-OCRv6_tiny/small/medium_rec` | 官方 PaddleOCR Rec adapter，可运行 train/export/predict；`paddleocr_rec` 仅作为数据集格式保留 |

官方后端依赖第三方包和许可条款。商业分发前需要单独审查 Ultralytics、PaddleOCR、PaddlePaddle、Torch 等依赖的许可证。

旧的 `tiny_linear_detector`、小型 `paddleocr_rec` CTC trainer、`python_mock` 和 C++ 分割/OCR 训练 scaffold 已物理删除，不会出现在用户训练后端列表中，也不会作为主验收 passed 依据。`paddleocr_rec` 仅作为数据集格式保留。

YOLO、SMP 与 OCR 的产品边界不同：YOLO 的训练、ONNX 导出和检测/实例分割评估来自官方 Ultralytics；后续单图推理、benchmark 和部署验证默认使用 AITrain C++ ONNX Runtime / NCNN runtime，TensorRT 当前用于 engine 导出和部署验证状态记录。SMP 是专用语义分割路线，不复用 YOLO instance segmentation 的 task/backend/export 语义；SMP 只承诺 ONNX Runtime 推理、overlay、benchmark 和部署验证，NCNN/TensorRT 导出不属于 SMP 能力范围，也不是 SMP 验收要求。OCR 则只接受 PaddleOCR 官方 Det / Rec / System 报告作为当前产品证据。PP-OCRv5/PP-OCRv6 支持只增加 Det / Rec / System 官方链路，不表示已经覆盖 PP-StructureV3、PP-ChatOCR、PaddleOCR-VL、文档方向分类、图像矫正、文本行方向分类或 PaddleOCR C++ 本地部署。PP-OCRv6 tiny 的语言覆盖按 PaddleOCR 官方限制处理，客户域生产声明仍需客户数据验收。

YOLO 模型预设下拉是完整产品化入口，但仍允许手动输入官方 Ultralytics 可解析的模型名。训练页会做任务匹配预检：检测后端不能选择 `-seg` 模型，分割后端必须选择 `-seg` 模型。YOLOv5 支持按 Ultralytics YOLOv5u 检测路线处理，预设包含 `yolov5n/s/m/l/x.yaml` 和 `yolov5nu/su/mu/lu/xu.pt`；不承诺兼容原始 `ultralytics/yolov5` 仓库旧权重，也不把 YOLOv5 segmentation 或 P6 纳入当前产品矩阵。YOLO12 分割 `.yaml` 是当前可验证路线；YOLO12 分割 `.pt` 只有在安装的 Ultralytics 能解析官方 `yolo12*-seg.pt` 权重时才能运行，当前记录为 `blocked_missing_official_weight`。YOLO26 作为独立兼容阶段跟踪 detection 和 instance segmentation 标准 `n/s/m/l/x` `.yaml`、`.pt` 预设；共享 Ultralytics 8.3.171 环境下仍应记录为 `blocked_model_unavailable` / `blocked_ultralytics_incompatible`，隔离 targeted full 证据仅支持训练、官方 ONNX、AITrain C++ ONNX 推理和 TensorRT。YOLO26 NCNN 不作为客户可用部署目标，“部署验证 > 模型导出”会移除该选项，Worker 会拒绝 `format=ncnn`。YOLO26 不支持 semantic segmentation、classification、pose、OBB、tracking 或 YOLOE-26。`.pt` 权重不随 AITrain 包分发，首次使用时可能由官方 Ultralytics 包下载到用户环境。

训练页“验证与导出”区域支持 YOLO 官方导出参数：`dynamic`、`half`、`int8 TensorRT` 和 `end2end`。默认仍导出 ONNX；勾选 `dynamic` 或 `half` 会传给官方 ONNX export；勾选 `int8 TensorRT` 会在 ONNX 之外额外尝试官方 TensorRT INT8 engine export，并使用本次训练的 `data.yaml` 做 calibration。`end2end=auto` 会优先读取已加载 Ultralytics 模型配置中的默认值；没有默认值时按 YOLO26 detection=true、其他模型=false 处理；需要时可显式选择 `true` 或 `false`。`end2end` 只有在当前 Ultralytics 版本和目标格式支持时才传给官方导出；不支持时必须显示 failed 或 blocked。NCNN 使用传统 YOLO 解码，拒绝 `end2end=true`，非 YOLO26 模型会生成 `end2end=false` 的中间 ONNX；YOLO26 会直接拒绝 NCNN。不支持的组合会明确失败，不会静默降级。

## 7. 任务与产物

“任务与产物”页是训练、校验、划分、导出、推理和评估的统一历史入口。

常用操作：

- 选择历史任务查看 artifacts、metrics、exports。
- 预览 JSON、YAML、TXT、CSV、LOG、图片 overlay。
- 选中 checkpoint、ONNX、engine 或官方导出目录后点击“用作导出输入”，会跳到“部署验证 > 模型导出”。
- 选中 ONNX、NCNN `.param` 或 AITrain export sidecar 后点击“用作推理模型”，会跳到“部署验证 > 推理验证”。TensorRT engine 当前用于部署验证状态记录，不作为单图推理输入。
- 选中训练产物后注册为模型版本。
- 对训练任务执行“复现实验”，复用原请求、数据快照、seed、后端和模型预设。

如果任务失败，先查看任务详情中的错误摘要和下一步建议，再检查“环境”页和数据集质量报告。

## 8. 模型库和评估报告

训练完成后，可以对模型进行评估，并在“模型库 > 评估报告”查看结果。

当前评估能力：

- YOLO 检测：通过 Ultralytics 官方 `val()` 输出 precision、recall、mAP50、mAP50-95、per-class maps、官方 confusion/PR/F1/P/R plots 和 predictions JSON（取决于官方版本和参数）。
- YOLO 分割：通过 Ultralytics 官方 `val()` 输出 box/mask precision、recall、maskMap50、mask mAP50-95、per-class mask maps、官方 confusion/PR/F1/P/R plots 和 predictions JSON（取决于官方版本和参数）。
- 语义分割：通过 SMP evaluator 输出 mIoU、meanDice、pixelAccuracy、per-class IoU/Dice、confusion matrix、低质量样本和 overlay。
- OCR Rec：通过 PaddleOCR 官方 Rec/System 报告和客户域 OCR 验收查看；AITrain 不用 C++ OCR ONNX 后处理生成当前产品评估证据。

评估依赖模型格式、数据集格式和可用 Python 环境。YOLO 检测和实例分割评估完全使用 Ultralytics 官方 `YOLO(...).val()`；AITrain 只保留 `evaluation_report.json` 外壳和任务产物记录，不再计算本地 AP/mAP、mask IoU、TP/FP/FN、错误样本或本地 overlay。SMP 语义分割评估使用 SMP/ONNX evaluator 计算像素级指标和 overlay。OCR 评估通过官方 PaddleOCR 报告查看。

“模型库 > 模型版本”用于管理已注册的模型版本。建议注册时关联：

- 来源任务
- 数据集快照
- checkpoint
- ONNX 或 engine
- 评估报告
- benchmark 或交付报告

模型库中的模型可以继续进入“部署验证 > 模型导出”或“部署验证 > 推理验证”。

## 9. 部署验证 > 模型导出

在“部署验证 > 模型导出”可以从训练产物生成部署格式。推荐从“任务与产物”选中模型产物后点击“用作导出输入”，避免手动填错路径。

导出格式：

| 格式 | 输入 | 输出 | 说明 |
|---|---|---|---|
| ONNX | checkpoint、`.pt`、已有 ONNX、AITrain export sidecar | `.onnx` 和 sidecar report | `.pt` 输入走官方 Ultralytics export；已有 ONNX 或指向 ONNX 的 sidecar 继续走 AITrain copy / report 路径 |
| NCNN | ONNX 或 `.pt` | `.param` 和 `.bin` | `.pt` 输入会先生成静态 FP32 官方 ONNX，再走 `onnx2ncnn`；`dynamic`、`half`、`int8` 会被拒绝 |
| TensorRT | ONNX 或 `.pt` | `.engine` / `.plan` | `.pt` 输入走官方 Ultralytics TensorRT export；INT8 需要 GPU/TensorRT 和 calibration data；旧 GPU 会 `hardware-blocked` |

输出路径留空时，已打开项目会默认写入项目的 `models/exported`；未打开项目时通常写入输入模型同目录。

“官方参数”区域会随模型导出请求传递 `format`、`dynamic`、`half`、`int8`、`imgsz`、`batch`、`device`，以及 TensorRT INT8 所需的 calibration `data.yaml`。ONNX 不支持 `int8=true`；NCNN 不支持 `dynamic/half/int8`；无 TensorRT、GPU 环境或 calibration data 时，TensorRT INT8 会明确失败或阻塞。

导出后建议在同一 tab 填写“验证图片”，点击“验证导出产物”：

- ONNX：必须能通过 ONNX Runtime 对样本图完成推理，才视为 `passed`。
- TensorRT：兼容硬件和 runtime 上可推理为 `passed`；旧 GPU 或 runtime 不满足时显示 `hardware-blocked`。
- NCNN：已支持 YOLO 检测/分割的 runtime 部署验证；无 NCNN SDK/runtime 时会明确失败，缺少样本图时会返回 `blocked`。

NCNN 当前本机验证边界：检测模型已经通过 Hyuto YOLOv8 ONNX -> NCNN runtime smoke；分割模型已经通过 nihui 预转换 YOLOv8n-seg pnnx/DFL NCNN artifact + AITrain sidecar 的 runtime smoke。部分 YOLOv8-seg ONNX 经 `onnx2ncnn` 后仍可能包含 NCNN 不支持的 `Shape` layer，此时会生成失败报告，不应标记为通过。

## 10. 部署验证 > 推理验证

在“部署验证 > 推理验证”执行单张图片验证：

1. 选择模型路径。可以手动选择 ONNX、NCNN `.param` 或 AITrain export sidecar，也可以从“任务与产物”点击“用作推理模型”带入。
2. 选择验证图片。
3. 选择输出目录；留空时写入模型同目录的 `inference`。
4. 点击“开始推理”。
5. 查看结果摘要和 overlay 预览。
6. 在“任务与产物”中查看完整 prediction JSON、overlay 和耗时信息。

当前本地推理验证支持：

- YOLO 检测：基于官方 Ultralytics ONNX / NCNN 产物，由 AITrain C++ runtime 输出类别、置信度、NMS、检测框 overlay。TensorRT engine 当前请走“部署验证 > 模型导出”的“验证导出产物”。
- YOLO 分割：基于官方 Ultralytics ONNX / NCNN 产物，由 AITrain C++ runtime 输出检测框、mask、mask area、半透明 overlay。TensorRT engine 当前请走“部署验证 > 模型导出”的“验证导出产物”。

OCR 路线只依赖 PaddleOCR 官方实现。Det / Rec / System 的推理、评估和可视化结果应从官方 PaddleOCR 任务产物与报告中查看，不通过 AITrain C++ OCR ONNX 后处理作为产品路径。

## 11. 数据集复核、环境页交付证据和诊断包

### 11.1 数据集 > 质量与复核

“数据集 > 质量与复核”用于把问题样本重新送回标注和数据集校验闭环。可以加载：

- 数据质量报告中的 `problem_samples.json`
- 评估报告中的 `error_samples.json`
- 低置信样本清单
- `rework_sample_set.json`

加载后可按来源、问题类型、类别、split、评估错误、OCR edit distance / CER、低置信信息筛选。点击“生成复核清单”会写出 X-AnyLabeling 可用的本地图片列表和 `rework_sample_set.json`。v1 不内嵌标注器，也不实现多人协作；标注完成后回到“数据集”页刷新、重新校验并创建快照。

### 11.2 环境 > 交付证据

“环境”页中的“交付证据”分区汇总以下状态：

- 本机 RC
- clean Windows
- TensorRT
- 客户域 OCR
- 包体完整性
- 部署验证
- 诊断包

可以导入外部 JSON / Markdown 验收结果，状态会显示为 `passed`、`blocked`、`failed`、`hardware-blocked` 或 `not_run`。真实验收脚本仍以 `tools\local-rc-closeout.ps1`、`tools\release-freeze-handoff.ps1`、`tools\customer-ocr-validation.ps1` 为准；GUI 负责调度 Worker 或展示结果，不替代 clean Windows、package-root TensorRT 或客户域 OCR 的真实返回证据。

### 11.3 客户域 OCR 验收

在“环境 > 交付证据”中填写客户域 Det 数据集、Rec 数据集、System 图片，以及 Det/Rec/System 官方报告。默认门槛为 Rec accuracy >= `0.70`、CER <= `0.30`，且必须不是 public/generated/smoke 数据。Total-Text、generated smoke 和 `.deps` 示例只能证明流程可跑，不能证明客户域 OCR 生产精度。

### 11.4 一键诊断包

“生成诊断包”会收集 Worker self-check、环境 profile、GPU/驱动、最近任务日志、失败请求、artifact index、插件状态和授权摘要。诊断包是只读证据，不会修改用户的全局 Python、CUDA 或驱动环境。

## 12. 系统设置 > 插件

“系统设置 > 插件”用于查看内置插件和本地 marketplace 插件。v1 是本地/离线优先机制，不是联网插件商店，也不代表插件发布者签名已经被强制校验。

常用操作：

1. 点击“扫描插件”刷新当前插件列表。
2. 查看插件类型、版本、来源、启用状态和加载错误。
3. 从本地插件包安装 marketplace 插件。
4. 启用或禁用 marketplace 插件。
5. 需要清理时执行卸载，并重新扫描确认状态。

边界说明：

- 插件不得绕过现有模型、数据集、导出、推理、验证等接口边界。
- Windows 可能锁定正在使用的 Qt plugin DLL；如果禁用或卸载返回 `disable-failed`，先关闭正在使用该插件的任务和窗口，再重启程序或重新扫描。
- marketplace v1 的禁用/卸载以状态安全为优先，不会强制删除被系统锁定的 DLL。
- 安装第三方插件前需要单独审查来源、许可证和二进制风险。

开发和包格式说明见 `docs/plugin-marketplace.md` 与 `docs/plugin-package-format.md`。

## 13. 常见问题

### 启动后只看到注册窗口

当前机器没有有效注册码。复制机器码给授权方，使用该机器码签发注册码后再验证。其他机器的注册码不能复用。

### 提示应用未配置授权公钥

这是构建配置问题，不是用户输入问题。需要确认主程序编译时已设置正确的 `AITRAIN_LICENSE_PUBLIC_KEY`。

### 数据集无法训练

先在“数据集”页运行校验。未通过当前格式校验的数据集不能启动训练。常见原因包括图片缺失、标签路径错误、YOLO 坐标越界、OCR label 文件格式错误、字典缺失字符。

### 官方训练后端启动失败

先运行“环境”页自检。YOLO 后端需要 Python、Ultralytics、Torch、ONNX、ONNX Runtime；OCR 官方后端建议使用隔离 PaddleOCR 环境和 PaddleOCR 源码 checkout。

### TensorRT 显示 hardware-blocked

当前机器 GPU 或 runtime 不满足 TensorRT engine build 要求。使用 ONNX Runtime / NCNN 继续“部署验证 > 推理验证”，或换到 RTX / SM 75+ 机器执行 TensorRT 导出和部署验证。

### OCR smoke 通过但业务图片效果不好

smoke 只证明流程、依赖和产物可用。OCR 业务可用性必须使用客户域数据重新训练、评估和验收。不要用生成数据或公开 smoke 结果声明客户域生产就绪。

### NCNN 导出失败

确认已安装 NCNN 工具，并配置 `AITRAIN_NCNN_ONNX2NCNN` 或 `AITRAIN_NCNN_ROOT`。若要执行部署验证，还需要用 `AITRAIN_NCNN_ROOT` 配置 NCNN SDK/runtime 并提供样本图；外部 `.param/.bin` 模型必须提供 AITrain sidecar，或显式传入 `modelFamily`、`classNames`、`inputBlob`、`outputBlobs` 和 `decoder`。

如果 YOLOv8-seg ONNX 转出的 NCNN `.param` 包含 `Shape` 等 unsupported layer，当前属于转换兼容性问题。处理方式是使用静态/兼容导出的 ONNX、pnnx/nihui 风格的预转换 NCNN artifact，或提供已验证的 sidecar/config 后走 `--ncnn-param-smoke` 验证现有 `.param/.bin`；不要把该失败当作 runtime 通过。

部署验证失败报告会给出 `failureCategory` 和下一步建议：`sdk_missing` 表示未启用 NCNN SDK/runtime，`sample_missing` 表示缺少样本图，`sidecar_missing` 表示外部模型缺 AITrain sidecar 或显式 blob/decoder 配置，`unsupported_layer` 表示 `.param` 中存在当前 NCNN runtime 无法加载的层，`runtime_failed` 表示加载、输出提取或后处理失败。

## 14. 建议的最小试用流程

如果只是第一次试用：

1. 在“项目”页创建一个测试项目。
2. 使用示例脚本生成最小数据集：

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```

3. 在“数据集”页导入 `.deps\examples-smoke\yolo_detect`。
4. 校验数据集，必要时划分或创建快照。
5. 在“训练实验”页选择 `ultralytics_yolo_detect` 和 `yolov8n.yaml`，运行少量 epoch。
6. 在“任务与产物”查看 `best.pt`、`best.onnx` 和训练报告。
7. 将 `best.onnx` 用作推理模型，进入“部署验证 > 推理验证”选择一张图片运行验证。

该流程用于确认安装、环境和闭环是否正常，不代表训练精度。

## 授权私钥安全说明

正式私钥文件必须保存在授权方本机或受控密钥目录，不能放进项目仓库、客户交付包、日志、诊断包或证据目录。仓库内只允许保留 `tools/aitrain-license-private-key.example.json` 这类无敏感内容模板。如果旧私钥曾进入源码或对外分发，应视为已泄漏：生成新的 key pair，用新公钥重新构建 `AITRAIN_LICENSE_PUBLIC_KEY`，旧私钥不再用于任何客户注册码。
