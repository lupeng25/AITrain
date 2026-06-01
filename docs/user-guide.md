# AITrain Studio 用户使用手册

本文面向 AITrain Studio 的终端用户，说明如何在图形界面中完成数据集制作、导入、校验、训练、评估、导出和推理验证。本文不介绍源码结构、Worker 协议或 Python trainer JSON 细节。

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
  -> 评估模型并注册到“模型库”
  -> 导出模型
  -> 导出后验证
  -> 推理验证
  -> 交付验收和诊断包
```

重要边界：

- 主流程优先使用官方 YOLO / PaddleOCR 后端，训练由 Worker 启动独立 Python 进程执行。
- 生产训练只使用官方后端；旧的 tiny detector、Python mock、小型 OCR CTC 和 C++ 分割/OCR 训练 scaffold 已从产品训练路径中物理删除。
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
| YOLO 检测 | `ultralytics_yolo_detect` | `yolov8n.yaml`、`yolo11n.yaml`、`yolo12n.yaml` | 官方 Ultralytics 检测训练和 ONNX 导出 |
| YOLO 分割 | `ultralytics_yolo_segment` | `yolov8n-seg.yaml`、`yolo11n-seg.yaml`、`yolo12n-seg.yaml` | 官方 Ultralytics 分割训练和 mask ONNX 后处理 |
| PaddleOCR Det | `paddleocr_det_official` | `PP-OCRv4_mobile_det` | 官方 PaddleOCR 检测工具链，建议使用隔离 OCR 环境 |
| PaddleOCR Rec | `paddleocr_rec_official` | `PP-OCRv4_mobile_rec` | 官方 PaddleOCR Rec adapter，可运行 train/export/predict；`paddleocr_rec` 仅作为数据集格式保留 |

官方后端依赖第三方包和许可条款。商业分发前需要单独审查 Ultralytics、PaddleOCR、PaddlePaddle、Torch 等依赖的许可证。

旧的 `tiny_linear_detector`、小型 `paddleocr_rec` CTC trainer、`python_mock` 和 C++ 分割/OCR 训练 scaffold 已物理删除，不会出现在用户训练后端列表中，也不会作为主验收 passed 依据。`paddleocr_rec` 仅作为数据集格式保留。

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

- YOLO 检测：precision、recall、AP50、mAP50、mAP50-95、per-class metrics、confusion matrix、error samples、overlay。
- YOLO 分割：maskIoU、maskMap50、mask mAP50-95、per-class metrics、confusion matrix、error samples、overlay。
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
| NCNN | ONNX 或可生成 ONNX 的输入 | `.param` 和 `.bin` | 导出依赖本机 `onnx2ncnn`；部署验证在配置 NCNN SDK/runtime 且提供样本图时运行 YOLO 检测/分割推理 |
| TensorRT | ONNX | `.engine` / `.plan` | 需要 RTX / SM 75+ 和 TensorRT runtime；旧 GPU 会 `hardware-blocked` |

输出路径留空时，已打开项目会默认写入项目的 `models/exported`；未打开项目时通常写入输入模型同目录。

导出后建议在同页填写“验证图片”，点击“验证导出产物”：

- ONNX：必须能通过 ONNX Runtime 对样本图完成推理，才视为 `passed`。
- TensorRT：兼容硬件和 runtime 上可推理为 `passed`；旧 GPU 或 runtime 不满足时显示 `hardware-blocked`。
- NCNN：已支持 YOLO 检测/分割的 runtime 部署验证；无 NCNN SDK/runtime 时会明确失败，缺少样本图时会返回 `blocked`。

NCNN 当前本机验证边界：检测模型已经通过 Hyuto YOLOv8 ONNX -> NCNN runtime smoke；分割模型已经通过 nihui 预转换 YOLOv8n-seg pnnx/DFL NCNN artifact + AITrain sidecar 的 runtime smoke。部分 YOLOv8-seg ONNX 经 `onnx2ncnn` 后仍可能包含 NCNN 不支持的 `Shape` layer，此时会生成失败报告，不应标记为通过。

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

## 11. 样本复核、交付验收和诊断包

### 11.1 样本复核

“样本复核”页用于把问题样本重新送回标注和数据集校验闭环。可以加载：

- 数据质量报告中的 `problem_samples.json`
- 评估报告中的 `error_samples.json`
- 低置信样本清单
- `rework_sample_set.json`

加载后可按来源、问题类型、类别、split、评估错误、OCR edit distance / CER、低置信信息筛选。点击“生成复核清单”会写出 X-AnyLabeling 可用的本地图片列表和 `rework_sample_set.json`。v1 不内嵌标注器，也不实现多人协作；标注完成后回到“数据集”页刷新、重新校验并创建快照。

### 11.2 交付验收

“交付验收”页汇总以下状态：

- 本机 RC
- clean Windows
- TensorRT
- 客户域 OCR
- 包体完整性
- 部署验证
- 诊断包

可以导入外部 JSON / Markdown 验收结果，状态会显示为 `passed`、`blocked`、`failed`、`hardware-blocked` 或 `not_run`。真实验收脚本仍以 `tools\local-rc-closeout.ps1`、`tools\release-freeze-handoff.ps1`、`tools\customer-ocr-validation.ps1` 为准；GUI 负责调度 Worker 或展示结果。

### 11.3 客户域 OCR 验收

在“交付验收”页填写客户域 Det 数据集、Rec 数据集、System 图片、Det/Rec/System 官方报告，以及可选 Det ONNX evidence。默认门槛为 Rec accuracy >= `0.70`、CER <= `0.30`，且必须不是 public/generated/smoke 数据。Total-Text、generated smoke 和 `.deps` 示例只能证明流程可跑，不能证明客户域 OCR 生产精度。

### 11.4 一键诊断包

“生成诊断包”会收集 Worker self-check、环境 profile、GPU/驱动、最近任务日志、失败请求、artifact index、插件状态和授权摘要。诊断包是只读证据，不会修改用户的全局 Python、CUDA 或驱动环境。

## 12. 插件 marketplace

“插件”页用于查看内置插件和本地 marketplace 插件。v1 是本地/离线优先机制，不是联网插件商店，也不代表插件发布者签名已经被强制校验。

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

当前机器 GPU 或 runtime 不满足 TensorRT engine build 要求。使用 ONNX Runtime 继续验证，或换到 RTX / SM 75+ 机器执行 TensorRT 导出和推理。

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
7. 将 `best.onnx` 用作推理模型，进入“推理验证”页选择一张图片运行验证。

该流程用于确认安装、环境和闭环是否正常，不代表训练精度。

## 授权私钥安全说明

正式私钥文件必须保存在授权方本机或受控密钥目录，不能放进项目仓库、客户交付包、日志、诊断包或证据目录。仓库内只允许保留 `tools/aitrain-license-private-key.example.json` 这类无敏感内容模板。如果旧私钥曾进入源码或对外分发，应视为已泄漏：生成新的 key pair，用新公钥重新构建 `AITRAIN_LICENSE_PUBLIC_KEY`，旧私钥不再用于任何客户注册码。
