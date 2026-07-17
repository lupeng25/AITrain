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
  -> 将训练产物登记为 V2 模型包，再在“部署验证”完成部署验证和推理验证
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
| Anomalib | Python、anomalib、torch、torchvision、lightning、timm、Pillow、numpy、opencv | 安装 `python_trainers\requirements-anomaly.txt`，EfficientAD 另需 imagenetDir，训练 batchSize 固定为 1 |
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
- `.deps/tools/annotation-tools/X-AnyLabeling`
- `.deps/annotation-tools/X-AnyLabeling`
- `PATH`

在“数据集”页仍可点击“启动 X-AnyLabeling”直接打开普通外部标注工具。受控修复闭环使用 V2 Artifact：先从 Data Quality 任务取得 Repair ArtifactId，再点击“准备修复会话”，输入该 ArtifactId 和一个新的空工作目录。Worker 校验已提交 Repair Artifact、复制不可变 Snapshot，并提交 Session Artifact；GUI 只显示 Session ArtifactId 和 Evidence ArtifactId，不读取 Artifact 裸路径。会话准备成功后，GUI 以独立工作目录启动本地 X-AnyLabeling。标注完成后点击“同步标注会话”，确认 Session ArtifactId 和同一工作目录；Worker 会重新校验基线、文件集合、编辑白名单与哈希。只有合法变更才登记新的 Dataset Version/Snapshot；无变化、冲突、越界或取消都不会生成正式新版本。任务、Artifact 和新版本状态在 V2“任务与产物”及项目汇总中刷新。

AITrain Studio 不内嵌 X-AnyLabeling GUI / PyQt 进程，也不打包 X-AnyLabeling Server。X-AnyLabeling 保持本地外部依赖；如需随产品分发，需要单独完成第三方许可证和包体评审。

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

### 4.4 YOLO OBB 旋转框数据集

YOLO OBB 用于旋转框检测。目录结构与 YOLO 检测一致，`data.yaml` 建议包含 `task: obb`。每行标签格式固定为 9 列：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

四个角点坐标使用 0 到 1 的归一化值。AITrain 会校验列数、类别范围、坐标范围、有限数值、非零面积四边形，并对疑似非矩形标注给出 warning。正好 4 点的 YOLO segmentation polygon 会标为 ambiguous，避免自动误判为 OBB；这类数据应手选 `yolo_obb` 格式。

### 4.5 异常检测 Folder 数据集

异常检测用于工业良品/异常样本的一类或少样本缺陷检测。推荐目录：

```text
dataset/
  train/
    good/
  test/
    good/
    anomaly/
  masks/
    test/
      anomaly/
```

`train/good` 必须存在。`val/good`、`val/anomaly`、`test/good`、`test/anomaly` 可按数据情况提供。像素级 mask 可放在 `masks/val/anomaly/<stem>.png` 或 `masks/test/anomaly/<stem>.png`。MVTec 风格 `test/<defect_type>` 和 `ground_truth/<defect_type>/<stem>_mask.png` 也可以识别。

只有 good 样本时可以训练 PatchCore / EfficientAD，但评估会显示为 `limited`。EfficientAD 在 Anomalib 2.5 下使用 `modelSize=small|medium`，训练 batchSize 固定为 1；旧的 `s/m` 输入只作为兼容值归一化。MVTec 官方 `tar.xz` 不需要手工提前解压给质量矩阵脚本，脚本可从 `.deps/datasets/downloads/mvtec_ad/mvtec_anomaly_detection.tar.xz` 物化分类目录，也可以直接使用已预物化的 `.deps/datasets/materialized/mvtec-ad/<category>`。Anomalib v1 的推理/benchmark/部署验证运行时是 `anomalib_python`，不会生成 AITrain C++ ONNX/TensorRT/NCNN anomaly runtime。

### 4.6 PaddleOCR Rec 数据集

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

### 4.7 PaddleOCR Det 数据集

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

“数据集格式转换”用于把已有 COCO、Pascal VOC 或 YOLO 标注转换为当前训练流程可用的布局。当前 GUI 暴露的是已实现的 COCO / Pascal VOC / YOLO 检测 / YOLO 分割原生转换矩阵，以及通过本地 X-AnyLabeling CLI 执行的 YOLO Detection / YOLO Segmentation / YOLO OBB 与 XLABEL 互转。不在下拉框或报告中标记为 supported 的组合不能当作已实现能力。

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

只有通过当前格式校验的数据集才能进入主训练流程。数据集划分支持 YOLO 检测、YOLO 分割、YOLO OBB、语义分割 Mask PNG、PaddleOCR Det 和 PaddleOCR Rec。

划分时可以设置：

- 输出目录，留空时默认写入项目的 `datasets/normalized`
- train / val / test 比例
- 随机 seed

数据快照用于冻结训练输入。可以在“数据集”页手动点击“创建数据快照”；如果启动训练时没有可用快照，程序会先要求创建快照，再继续训练。

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
| YOLO OBB 旋转框 | `ultralytics_yolo_obb` | 默认 `yolo11n-obb.pt`；可选 YOLO11 `n/s/m/l/x` 的 `-obb.pt` 和 `-obb.yaml`；YOLO26 OBB 不进入 v1 默认 preset，可手动输入官方可解析模型名尝试 | 官方 Ultralytics OBB 训练、ONNX 导出和 `val()` 评估；旋转四边形推理、overlay、benchmark、部署验证走 AITrain C++ ONNX Runtime；NCNN 不纳入 OBB v1 |
| 语义分割 Mask PNG | `smp_semantic_segmentation` | 默认 `smp_unet_resnet34`；可选 `smp_unetplusplus_resnet34`、`smp_fpn_resnet34`、`smp_deeplabv3plus_resnet50`、`smp_segformer_mit_b0` | SMP 专用语义分割，要求 `classes.txt` 和单通道 PNG class-id mask；训练导出 `best.pt`、`best.onnx` 和 SMP 报告；推理、overlay、benchmark 和部署验证只承诺 ONNX Runtime；SMP 不需要 NCNN/TensorRT 导出 |
| PaddleOCR Det | `paddleocr_det_official` | 默认 `PP-OCRv5_mobile_det`；可选 `PP-OCRv4_mobile_det`、`PP-OCRv5_server_det`、`PP-OCRv6_tiny/small/medium_det` | 官方 PaddleOCR 检测工具链，建议使用隔离 OCR 环境；PP-OCRv5/v6 preset 需要 PaddleOCR 源码 checkout |
| PaddleOCR Rec | `paddleocr_rec_official` | 默认 `PP-OCRv5_mobile_rec`；可选 `PP-OCRv4_mobile_rec`、`PP-OCRv5_server_rec`、`en_PP-OCRv5_mobile_rec`、`PP-OCRv6_tiny/small/medium_rec` | 官方 PaddleOCR Rec adapter，可运行 train/export/predict；`paddleocr_rec` 仅作为数据集格式保留 |

官方后端依赖第三方包和许可条款。商业分发前需要单独审查 Ultralytics、PaddleOCR、PaddlePaddle、Torch 等依赖的许可证。

旧的 `tiny_linear_detector`、小型 `paddleocr_rec` CTC trainer、`python_mock` 和 C++ 分割/OCR 训练 scaffold 已物理删除，不会出现在用户训练后端列表中，也不会作为主验收 passed 依据。`paddleocr_rec` 仅作为数据集格式保留。

YOLO、SMP 与 OCR 的产品边界不同：YOLO 的训练、ONNX 导出和检测/实例分割/OBB 评估来自官方 Ultralytics；后续单图推理、benchmark 和部署验证默认使用 AITrain C++ ONNX Runtime，NCNN 只覆盖支持的 YOLO 检测/分割，不覆盖 OBB v1，TensorRT 当前用于 engine 导出和部署验证状态记录。SMP 是专用语义分割路线，不复用 YOLO instance segmentation 的 task/backend/export 语义；SMP 只承诺 ONNX Runtime 推理、overlay、benchmark 和部署验证，NCNN/TensorRT 导出不属于 SMP 能力范围，也不是 SMP 验收要求。OCR 则只接受 PaddleOCR 官方 Det / Rec / System 报告作为当前产品证据。PP-OCRv5/PP-OCRv6 支持只增加 Det / Rec / System 官方链路，不表示已经覆盖 PP-StructureV3、PP-ChatOCR、PaddleOCR-VL、文档方向分类、图像矫正、文本行方向分类或 PaddleOCR C++ 本地部署。PP-OCRv6 tiny 的语言覆盖按 PaddleOCR 官方限制处理，客户域生产声明仍需客户数据验收。

YOLO 模型预设下拉是完整产品化入口，但仍允许手动输入官方 Ultralytics 可解析的模型名。训练页会做任务匹配预检：检测后端不能选择 `-seg` 或 `-obb` 模型，分割后端必须选择 `-seg` 模型，OBB 后端默认选择 `-obb` 模型。YOLOv5 支持按 Ultralytics YOLOv5u 检测路线处理，预设包含 `yolov5n/s/m/l/x.yaml` 和 `yolov5nu/su/mu/lu/xu.pt`；不承诺兼容原始 `ultralytics/yolov5` 仓库旧权重，也不把 YOLOv5 segmentation 或 P6 纳入当前产品矩阵。YOLO12 分割 `.yaml` 是当前可验证路线；YOLO12 分割 `.pt` 只有在安装的 Ultralytics 能解析官方 `yolo12*-seg.pt` 权重时才能运行，当前记录为 `blocked_missing_official_weight`。YOLO26 作为独立兼容阶段跟踪 detection 和 instance segmentation 标准 `n/s/m/l/x` `.yaml`、`.pt` 预设；共享 Ultralytics 8.3.171 环境下仍应记录为 `blocked_model_unavailable` / `blocked_ultralytics_incompatible`，隔离 targeted full 证据仅支持训练、官方 ONNX、AITrain C++ ONNX 推理和 TensorRT。YOLO26 NCNN 不作为客户可用部署目标，V2 Manifest 与 Runtime 能力矩阵会拒绝 NCNN 路由。YOLO26 不支持 semantic segmentation、classification、pose、OBB、tracking 或 YOLOE-26。`.pt` 权重不随 AITrain 包分发，首次使用时可能由官方 Ultralytics 包下载到用户环境。

训练页“验证与导出”区域支持 YOLO 官方导出参数：`dynamic`、`half`、`int8 TensorRT` 和 `end2end`。默认仍导出 ONNX；勾选 `dynamic` 或 `half` 会传给官方 ONNX export；勾选 `int8 TensorRT` 会在 ONNX 之外额外尝试官方 TensorRT INT8 engine export，并使用本次训练的 `data.yaml` 做 calibration。`end2end=auto` 会优先读取已加载 Ultralytics 模型配置中的默认值；没有默认值时按 YOLO26 detection=true、其他模型=false 处理；需要时可显式选择 `true` 或 `false`。`end2end` 只有在当前 Ultralytics 版本和目标格式支持时才传给官方导出；不支持时必须显示 failed 或 blocked。NCNN 使用传统 YOLO 解码，拒绝 `end2end=true`，非 YOLO26 模型会生成 `end2end=false` 的中间 ONNX；YOLO26 会直接拒绝 NCNN。不支持的组合会明确失败，不会静默降级。

## 7. 任务与产物

“任务与产物”页是训练、校验、划分、导出、推理和评估的统一历史入口。

常用操作：

- 选择历史任务查看 artifacts、metrics、exports。
- 预览 JSON、YAML、TXT、CSV、LOG、图片 overlay。
- 任务产物页不再把 checkpoint、ONNX 或 engine 裸路径直接送入导出、评估、benchmark 或推理；模型必须先登记为带 Manifest 和哈希的 V2 模型包。
- 选中 ONNX、NCNN `.param` 或 AITrain export sidecar 后点击“用作推理模型”，会跳到“部署验证 > 推理验证”。TensorRT engine 当前用于部署验证状态记录，不作为单图推理输入。
- 选中训练产物后注册为模型版本。
- 历史训练任务只读展示原始请求、数据快照、seed、后端和模型预设；如需再次训练，请新建训练任务并显式选择输入。

如果任务失败，先查看任务详情中的错误摘要和下一步建议，再检查“环境”页和数据集质量报告。

## 8. 模型库和评估报告

训练完成后，可以对模型进行评估，并在“模型库 > 评估报告”查看结果。

当前评估能力：

- YOLO 检测：通过 Ultralytics 官方 `val()` 输出 precision、recall、mAP50、mAP50-95、per-class maps、官方 confusion/PR/F1/P/R plots 和 predictions JSON（取决于官方版本和参数）。
- YOLO 分割：通过 Ultralytics 官方 `val()` 输出 box/mask precision、recall、maskMap50、mask mAP50-95、per-class mask maps、官方 confusion/PR/F1/P/R plots 和 predictions JSON（取决于官方版本和参数）。
- YOLO OBB：通过 Ultralytics 官方 `val()` 输出 OBB box precision、recall、mAP50、mAP50-95、per-class maps 和官方 plots/predictions（取决于官方版本和参数）。
- 语义分割：通过 SMP evaluator 输出 mIoU、meanDice、pixelAccuracy、per-class IoU/Dice、confusion matrix、低质量样本和 overlay。
- OCR Rec：通过 PaddleOCR 官方 Rec/System 报告和客户域 OCR 验收查看；AITrain 不用 C++ OCR ONNX 后处理生成当前产品评估证据。

评估依赖模型格式、数据集格式和可用 Python 环境。YOLO 检测、实例分割和 OBB 评估完全使用 Ultralytics 官方 `YOLO(...).val()`；AITrain 只保留 `evaluation_report.json` 外壳和任务产物记录，不再计算本地 AP/mAP、旋转 AP、mask IoU、TP/FP/FN、错误样本或本地 overlay。SMP 语义分割评估使用 SMP/ONNX evaluator 计算像素级指标和 overlay。OCR 评估通过官方 PaddleOCR 报告查看。

“模型库 > 模型版本”用于管理已注册的模型版本。建议注册时关联：

- 来源任务
- 数据集快照
- checkpoint
- ONNX 或 engine
- 评估报告
- benchmark 或交付报告

模型库中的已验证 V2 模型包可以继续进入“部署验证”或“推理验证”；旧模型版本记录只用于迁移期审计。

## 9. 部署验证

“部署验证”只接受已经登记且校验通过的 V2 模型包，不接受 checkpoint、ONNX、NCNN param 或 TensorRT engine 裸路径。

1. 在“模型库”导入模型文件和用户确认的 Manifest 草稿，等待系统计算 SHA-256 并登记 `ModelPackageId`。
2. 在“部署验证”选择已验证模型包。
3. 选择一张验证图片。
4. 点击“开始部署验证”。
5. 在状态区和“任务与产物”中查看报告、预测、overlay 与精确 runtime 状态。

模型导出已从独立 GUI 裸路径命令移除。训练产生的首次导出由八步训练 Workflow 的 `Export` 步骤负责；外部模型必须先通过 V2 导入流程形成模型包。部署验证会依次校验 Manifest、Artifact Store 边界、入口文件与 SHA-256，再由声明的 runtime route 和能力矩阵决定是否可执行。

当前边界：

- ONNX Runtime 是现有 GUI 产品推理与部署验证主路径。
- NCNN 必须具有显式模型合同、param/bin 完整性和受支持 decoder；不能靠文件后缀猜测模型。
- TensorRT 会精确区分 SDK、依赖、硬件与 decoder 状态；decoder 未实现时不得声明真实推理成功。
- OBB v1 只承诺 ONNX Runtime；SMP 只承诺 ONNX Runtime；Anomaly 使用 Worker 管理的 Anomalib Python 包；OCR 验收使用 PaddleOCR 官方 Det/Rec/System 报告。

## 10. 部署验证 > 推理验证

单图推理同样只接受 V2 模型包：

1. 选择已验证的 `ModelPackageId`。
2. 选择验证图片。
3. 点击“开始推理”。
4. 查看结果摘要和 overlay 预览。
5. 在“任务与产物”中复查已提交 prediction JSON、overlay、耗时与 Workflow Step。

推理输出先写入 V2 runtime staging；Worker 成功后才由 Workspace 校验候选文件并原子提交 Artifact，失败或取消会清理暂存目录。任务产物的裸路径不能直接作为推理模型。

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

“生成诊断包”会收集 Worker self-check、环境 profile、GPU/驱动、最近任务日志、失败请求、artifact index、内置能力状态和授权摘要。诊断包是只读证据，不会修改用户的全局 Python、CUDA 或驱动环境。

## 12. 系统设置 > 内置能力

“系统设置 > 内置能力”用于查看编译期能力注册表、任务类型、数据集格式和官方后端边界。能力不可在运行时下载或替换。

常用操作：

1. 点击“刷新能力摘要”查看当前注册表。
2. 查看能力支持的任务、数据集格式、后端和导出边界。
3. 在训练页选择内置能力与官方后端。

边界说明：

- 能力注册表不得绕过现有模型、数据集、导出、推理、验证等接口边界。
- 新增能力必须同时更新注册表、Worker 兼容性校验、GUI 选择器和验收测试。

Worker 可通过 `aitrain_worker.exe --builtin-capabilities` 输出机器可读的能力矩阵。

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

正式私钥文件必须保存在授权方本机或受控密钥目录，不能放进项目仓库、客户交付包、日志、诊断包或证据目录。注册码生成器只保存 `.aitrainkey` 受保护文件：私钥由 Windows DPAPI 绑定到当前用户，并将文件 ACL 收紧为仅当前用户；界面默认不显示或导出明文私钥。受保护文件不能复制给另一个 Windows 用户直接使用。仓库内只允许保留无敏感内容的说明模板。如果旧私钥曾进入源码或对外分发，应视为已泄漏：生成新的 key pair，用新公钥重新构建 `AITRAIN_LICENSE_PUBLIC_KEY`，旧私钥不再用于任何客户注册码。

主程序会保存由 DPAPI 保护的最近可信 UTC，并允许 5 分钟系统校时容差。系统时间明显早于可信时间时，授权校验会报告时钟回拨；可信时间文件损坏或来自其他 Windows 用户时也会拒绝验证。纯离线授权无法抵御拥有管理员权限并可完整替换程序、用户配置和系统状态的攻击者；该机制用于提高普通文件篡改和简单调钟的成本，不应作为硬件安全模块或在线授权服务的替代品。
