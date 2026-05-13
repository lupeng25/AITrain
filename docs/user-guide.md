# AITrain Studio 用户使用手册

本文面向 AITrain Studio 的终端用户，说明如何在图形界面中完成数据集制作、导入、校验、训练、评估、导出和推理验证。本文不介绍源码结构、Worker 协议或 Python trainer JSON 细节。

## 1. 快速闭环

推荐按下面顺序使用：

```text
启动并注册
  -> 创建或打开项目
  -> 环境自检
  -> 制作或导入数据集
  -> 校验数据集
  -> 生成质量报告并修复问题
  -> 划分数据集或创建数据快照
  -> 启动训练
  -> 在“任务与产物”查看 checkpoint、ONNX、报告和预览图
  -> 评估模型并注册到“模型库”
  -> 导出模型
  -> 推理验证
```

重要边界：

- 主流程优先使用官方 YOLO / PaddleOCR 后端，训练由 Worker 启动独立 Python 进程执行。
- `tiny_linear_detector` 和 `python_mock` 只用于诊断、演示或协议测试，不是真实 YOLO/OCR 训练能力。
- TensorRT 需要兼容的 NVIDIA RTX / SM 75+ 环境；不支持的 GPU 会显示为 `hardware-blocked`。
- OCR 的公开数据或生成数据 smoke 只能证明流程和产物可用，不能替代客户业务数据上的精度验收。

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
3. 点击“校验数据集”。
4. 如果校验失败，查看问题表格和校验 JSON。
5. 点击“生成质量报告”，查看缺失标签、孤立标签、非法框、多边形错误、OCR 标签错误、重复样本等问题。
6. 需要修复时点击“打开问题清单”或“X-AnyLabeling 修复”。
7. 修复后点击“标注后刷新 / 重新校验”。

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
| YOLO 检测 | `ultralytics_yolo_detect` | `yolov8n.yaml`、`yolo11n.yaml`、`yolo12n.yaml` | 官方 Ultralytics 检测训练和 ONNX 导出 |
| YOLO 分割 | `ultralytics_yolo_segment` | `yolov8n-seg.yaml`、`yolo11n-seg.yaml`、`yolo12n-seg.yaml` | 官方 Ultralytics 分割训练和 mask ONNX 后处理 |
| PaddleOCR Det | `paddleocr_det_official` | `PP-OCRv4_mobile_det` | 官方 PaddleOCR 检测工具链，建议使用隔离 OCR 环境 |
| PaddleOCR Rec | `paddleocr_rec` | `paddle_ctc_smoke` | 小型 PaddlePaddle CTC 识别训练，可导出 ONNX；不是完整 PP-OCRv4 官方链路 |
| PaddleOCR Rec 官方 | `paddleocr_rec_official` | `PP-OCRv4_mobile_rec` | 官方 PaddleOCR Rec adapter，可运行 train/export/predict |
| PaddleOCR System | `paddleocr_system_official` | `PP-OCRv4_det_rec_system` | 官方 `predict_system.py` 端到端推理链路 |
| 诊断 | `tiny_linear_detector` | `diagnostic` | C++ scaffold，只用于诊断或演示 |
| 协议测试 | `python_mock` | `diagnostic` | Worker/Python 协议 fixture，不是真训练 |

官方后端依赖第三方包和许可条款。商业分发前需要单独审查 Ultralytics、PaddleOCR、PaddlePaddle、Torch 等依赖的许可证。

## 7. 任务与产物

“任务与产物”页是训练、校验、划分、导出、推理和评估的统一历史入口。

常用操作：

- 选择历史任务查看 artifacts、metrics、exports。
- 预览 JSON、YAML、TXT、CSV、LOG、图片 overlay。
- 选中 checkpoint、ONNX、engine 或官方导出目录后点击“用作导出输入”。
- 选中 ONNX、AITrain export 或 TensorRT engine 后点击“用作推理模型”。
- 选中训练产物后注册为模型版本。
- 对训练任务执行“复现实验”，复用原请求、数据快照、seed、后端和模型预设。

如果任务失败，先查看任务详情中的错误摘要和下一步建议，再检查“环境”页和数据集质量报告。

## 8. 评估报告和模型库

训练完成后，可以对模型进行评估，并在“评估报告”页查看结果。

当前评估能力：

- YOLO 检测：precision、recall、AP50、mAP50、per-class metrics、confusion matrix、error samples、overlay。
- YOLO 分割：maskIoU、maskMap50、per-class metrics、confusion matrix、error samples、overlay。
- OCR Rec：accuracy、editDistance、CER、WER、error samples、overlay。

评估依赖模型格式、数据集格式和可用 runtime。非 ONNX 或官方工具链产物可能需要先导出，或通过官方报告查看。

“模型库”用于管理已注册的模型版本。建议注册时关联：

- 来源任务
- 数据集快照
- checkpoint
- ONNX 或 engine
- 评估报告
- benchmark 或交付报告

模型库中的模型可以继续进入导出或推理验证页面。

## 9. 模型导出

在“模型导出”页可以从训练产物生成部署格式。推荐从“任务与产物”选中模型产物后点击“用作导出输入”，避免手动填错路径。

导出格式：

| 格式 | 输入 | 输出 | 说明 |
|---|---|---|---|
| ONNX | checkpoint、已有 ONNX、AITrain export | `.onnx` 和 sidecar report | 主交付格式，可继续推理验证 |
| NCNN | ONNX 或可生成 ONNX 的输入 | `.param` 和 `.bin` | 依赖本机 `onnx2ncnn`；当前 GUI 不运行 NCNN 推理 |
| TensorRT | ONNX | `.engine` / `.plan` | 需要 RTX / SM 75+ 和 TensorRT runtime；旧 GPU 会 `hardware-blocked` |
| tiny detector JSON | tiny detector checkpoint | 诊断 JSON | 仅用于 scaffold 诊断，不是主交付格式 |

输出路径留空时，已打开项目会默认写入项目的 `models/exported`；未打开项目时通常写入输入模型同目录。

## 10. 推理验证

在“推理验证”页执行单张图片验证：

1. 选择模型路径。可以手动选择 ONNX、AITrain export、TensorRT engine，也可以从“任务与产物”点击“用作推理模型”带入。
2. 选择验证图片。
3. 选择输出目录；留空时写入模型同目录的 `inference`。
4. 点击“开始推理”。
5. 查看结果摘要和 overlay 预览。
6. 在“任务与产物”中查看完整 prediction JSON、overlay 和耗时信息。

当前 C++ ONNX Runtime 推理支持：

- YOLO 检测：类别、置信度、NMS、检测框 overlay。
- YOLO 分割：检测框、mask、mask area、半透明 overlay。
- OCR Rec：CTC greedy decode、文本和置信度摘要。
- OCR Det：DB-style probability map v1 后处理，输出文字区域 polygon 和 overlay。

PaddleOCR System 的端到端结果仍以官方工具链任务产物为主，不等同于完整 C++ PaddleOCR System runtime。

## 11. 常见问题

### 启动后只看到注册窗口

当前机器没有有效注册码。复制机器码给授权方，使用该机器码签发注册码后再验证。其他机器的注册码不能复用。

### 提示应用未配置授权公钥

这是构建配置问题，不是用户输入问题。需要确认主程序编译时已设置正确的 `AITRAIN_LICENSE_PUBLIC_KEY`。

### 数据集无法训练

先在“数据集”页运行校验。未通过当前格式校验的数据集不能启动训练。常见原因包括图片缺失、标签路径错误、YOLO 坐标越界、OCR label 文件格式错误、字典缺失字符。

### 官方训练后端启动失败

先运行“环境”页自检。YOLO 后端需要 Python、Ultralytics、Torch、ONNX、ONNX Runtime；OCR 官方后端建议使用隔离 PaddleOCR 环境和 PaddleOCR 源码 checkout。

### TensorRT 显示 hardware-blocked

当前机器 GPU 或 runtime 不满足 TensorRT engine build 要求。使用 ONNX Runtime 继续验证，或换到 RTX / SM 75+ 机器执行 TensorRT 导出和推理。

### OCR smoke 通过但业务图片效果不好

smoke 只证明流程、依赖和产物可用。OCR 业务可用性必须使用客户域数据重新训练、评估和验收。不要用生成数据或公开 smoke 结果声明客户域生产就绪。

### NCNN 导出失败

确认已安装 NCNN 工具，并配置 `AITRAIN_NCNN_ONNX2NCNN` 或 `AITRAIN_NCNN_ROOT`。NCNN 导出只生成部署产物，当前 AITrain Studio 推理页不运行 NCNN 模型。

## 12. 建议的最小试用流程

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
7. 将 `best.onnx` 用作推理模型，进入“推理验证”页选择一张图片运行验证。

该流程用于确认安装、环境和闭环是否正常，不代表训练精度。
