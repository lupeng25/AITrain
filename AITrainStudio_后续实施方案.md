# AITrain Studio 后续实施方案

生成日期：2026-05-01
项目目录：`C:\Users\73200\Desktop\code\AITrain`
权威状态入口：`docs/harness/current-status.md`

本文用于规划 AITrain Studio 在阶段 7 之后的中长期路线。阶段状态以 `docs/harness/current-status.md` 为准；本文提供目标、交付、验收标准和风险控制。

## 1. 当前结论

AITrain Studio 已具备可运行的平台骨架：Qt Widgets GUI、独立 Worker、JSON Lines over `QLocalSocket`、SQLite 项目元数据、插件接口、数据集校验、tiny detector scaffold、ONNX Runtime 推理链路、分割 scaffold、OCR scaffold、Windows 打包和 TensorRT runtime 接入。

当前关键事实：

- 阶段 1-2 已作为平台和数据系统初版完成。
- 阶段 3-6 是可执行 scaffold / baseline，不是真正 YOLO/OCR 官方训练。
- 阶段 7 代码完成，本机自检、打包 smoke、ZIP 生成路径已具备；TensorRT 真 engine 验收因 GTX 1060 / SM 61 被 TensorRT 10 拒绝而挂起，需 RTX / SM 75+ 机器补验收。
- 阶段 8 已完成 Python Trainer Adapter，真实训练优先通过 Worker 启动独立 Python 子进程实现。
- 阶段 9 已接入 Ultralytics YOLO detection 官方训练，并完成本机 CPU 小数据 smoke。
- 阶段 10 已完成真实 YOLO detection ONNX Runtime 推理和 ONNX 复制导出；TensorRT 转换仍需外部 RTX / SM 75+ 验收。
- 阶段 11 已接入 Ultralytics YOLO segmentation 官方训练，并完成本机 CPU 小数据 smoke；C++ ONNX Runtime 已补 YOLOv8-seg mask 后处理和 overlay。
- 阶段 12 已接入 PaddlePaddle OCR Rec CTC 训练，并完成本机 CPU 小数据 smoke；C++ ONNX Runtime 已补 CTC greedy decode。它兼容 PaddleOCR-style Rec 数据，但还不是完整 PP-OCRv4 官方训练配置和导出链路。
- 阶段 13 的本机产品化目标是文档、依赖、样例数据生成、硬件兼容矩阵、打包 smoke 和外部 RTX 验收清单。

## 2. 实施原则

- 训练逻辑必须进入 `src/core`、插件或 `aitrain_worker`，不得放入 `MainWindow`。
- 长任务必须通过 Worker 执行，GUI 只负责交互、状态展示和任务调度。
- SQLite 元数据必须通过 `ProjectRepository` 管理。
- 模型、数据集、导出、推理、验证能力继续走插件接口。
- Qt 继续保持 5.12+ 兼容，除非明确升级。
- scaffold 必须明确标注，不把 tiny detector、分割 baseline、OCR baseline 描述成真实 YOLO/OCR 训练。
- 真实训练优先复用官方 Python 实现或成熟开源实现，AITrain Studio 负责编排、产物归一化、推理、转换、部署和打包。
- Python 训练进程必须使用统一 JSON request 和 JSON Lines 输出协议。
- C++ 侧保留 ONNX Runtime、TensorRT、overlay、prediction JSON、打包和环境自检能力。
- 每个阶段必须有最小数据集验收和 `.\tools\harness-check.ps1` 验证。

## 3. 阶段路线总览

| 阶段 | 状态 | 目标 | 验收标准 |
|---|---|---|---|
| 阶段 1 | 已完成为平台 scaffold | 平台稳定化 | Worker、任务状态、SQLite、插件接口、harness 可用 |
| 阶段 2 | 已完成为初版系统 | 数据集系统 | YOLO 检测、YOLO 分割、PaddleOCR Rec 校验可用 |
| 阶段 3 | scaffold 完成 | YOLO 检测训练入口 | tiny detector 可训练、保存 checkpoint、输出指标 |
| 阶段 4 | tiny detector 路径完成 | ONNX 导出与推理 | tiny ONNX 导出、ONNX Runtime 推理、overlay 输出 |
| 阶段 5 | scaffold 完成 | YOLO 分割训练入口 | 分割数据读取、mask baseline、preview、指标输出 |
| 阶段 6 | scaffold 完成 | OCR Rec 训练入口 | OCR 数据读取、字典、baseline checkpoint、preview |
| 阶段 7 | 代码完成，外部 GPU 验收待补 | TensorRT 与 Windows 打包 | 本机自检和打包通过；TensorRT 真验收待 RTX / SM 75+ |
| 阶段 8 | 已完成 | Python Trainer Adapter 与环境管理 | Worker 可启动 Python 训练进程并接收 JSON Lines 事件 |
| 阶段 9 | 已完成本机 CPU smoke | 官方 YOLO 检测训练接入 | Ultralytics detection 训练、ONNX 导出、report 产物可用 |
| 阶段 10 | ONNX Runtime 已完成，TensorRT 外部 GPU 待验收 | 真实检测 ONNX 推理与转换 | YOLO decoder、NMS、类别映射、overlay、ONNX sidecar 可用 |
| 阶段 11 | 已完成本机 CPU smoke | 官方 YOLO 分割训练接入 | Ultralytics segmentation 训练、ONNX 导出、mask 指标可用 |
| 阶段 12 | 已完成本机 CPU smoke | PaddlePaddle OCR Rec 训练接入 | PaddlePaddle CTC Rec 训练、checkpoint、ONNX、dict、accuracy、C++ decode 可用 |
| 阶段 13 | 本机产品化完成，外部验收待补 | 产品化与验收 | 样例数据、文档、兼容矩阵、打包 smoke 已完成；外部 GPU / 干净机器验收待补 |

## 4. 已完成阶段复盘

### 4.1 阶段 1：平台稳定化

已完成：

- Qt Widgets 工作台骨架。
- `aitrain_worker` 独立进程。
- JSON Lines over `QLocalSocket` 通信。
- 任务、指标、artifact、export、环境记录的 SQLite 持久化。
- 插件接口和内置插件骨架。
- harness 上下文和检查脚本。

保留约束：

- 任务状态迁移不能散落在 GUI、Worker、插件多处重复实现。
- Worker 失败必须给出明确错误消息。
- GUI 不直接承载训练计算。

### 4.2 阶段 2：数据集系统

已完成：

- YOLO detection 数据校验。
- YOLO segmentation 数据读取与基础校验。
- PaddleOCR Rec 标签和字典读取。
- detection 数据集划分初版。

后续增强：

- 大数据集校验进度和取消。
- 错误样本跳转和修复建议。
- segmentation 和 OCR 的完整划分策略。

### 4.3 阶段 3-6：C++ scaffold

阶段 3-6 提供可执行 baseline，用于 GUI、Worker、artifact、report、测试和最小闭环验证。

明确限制：

- tiny detector 不是真实 YOLO。
- C++ segmentation baseline 没有真实 mask head / mask loss。
- C++ OCR baseline 没有真实 CRNN / CTC loss。
- 这些 scaffold 继续保留为 demo、fallback 和测试后端，但不得作为真实训练能力对外宣称。

### 4.4 阶段 7：TensorRT 与 Windows 打包

已完成：

- 安装目录布局。
- CMake install / CPack ZIP。
- `tools/package-smoke.ps1`。
- `aitrain_worker --self-check`。
- CUDA、cuDNN、TensorRT、ONNX Runtime DLL 发现。
- TensorRT SDK-backed ONNX 到 engine 导出和 engine 推理代码路径。
- 本机 NVIDIA 驱动已升级到 582.28，CUDA 13 runtime 自检通过。

当前挂起项：

- 本机 GTX 1060 是 SM 61，TensorRT 10 engine build 报错：`Target GPU SM 61 is not supported by this TensorRT release`。
- 真 TensorRT 验收必须在 RTX / SM 75+ 机器或云 GPU 上执行。

外部验收命令：

```powershell
.\aitrain_worker.exe --self-check
.\aitrain_worker.exe --tensorrt-smoke <work-dir>
.\tools\package-smoke.ps1
```

## 5. 官方 Python 后端策略

真实训练不再优先自研 C++ / LibTorch 核心，而是优先接入成熟 Python 后端。AITrain Studio 的核心价值是把这些训练能力产品化、流程化、可视化、可诊断和可部署。

| 任务 | 首选实现 | 状态 | 约束 |
|---|---|---|---|
| YOLO detection | Ultralytics YOLO Python API | 已接入 | 需处理 AGPL-3.0 / Enterprise license |
| YOLO segmentation | Ultralytics YOLO segment Python API | 已接入 | 同 detection，商业闭源分发前必须确认许可 |
| OCR Rec | PaddlePaddle / PaddleOCR-style Rec | 已接入小型 CTC trainer | 不是完整 PP-OCRv4 官方 config；后续补官方配置和导出 |

接入规则：

- 官方 Python 包不进入 GUI 进程。
- Worker 启动 Python 子进程，并通过 JSON request / JSON Lines 协议交换状态。
- 官方训练输出必须归一化为 checkpoint、model、dict/class mapping、training report、preview 或 validation result。
- C++ 侧继续负责项目状态、SQLite、取消、失败上报、ONNX Runtime、TensorRT、overlay、prediction JSON 和 packaging。

## 6. 阶段 8：Python Trainer Adapter 与环境管理

目标：建立统一 Python 训练桥接层，让 C++ Worker 能稳定启动、监控、取消和接收 Python 训练任务。

当前状态：

- Worker 通过 `QProcess` 启动 Python 子进程。
- Worker 写出 `python_trainer_request.json`。
- Python trainer 通过 stdout JSON Lines 返回 `log`、`progress`、`metric`、`artifact`、`completed`、`failed`。
- Worker 可以传播 Python trainer 失败。
- 环境自检会报告 Python、Ultralytics、PaddleOCR、PaddlePaddle 可用性。
- `python_mock` trainer 仅用于协议和 Worker 链路验证，不是真实训练。

验收状态：已完成，`.\tools\harness-check.ps1` 通过。

## 7. 阶段 9：官方 YOLO 检测训练接入

目标：接入 Ultralytics YOLO detection，完成真实目标检测训练、验证、导出和产物归一化。

当前状态：

- 已完成 `python_trainers/detection/ultralytics_trainer.py`。
- Worker 将 `trainingBackend=ultralytics_yolo_detect` 和 `trainingBackend=ultralytics_yolo` 路由到该 trainer。
- trainer 会生成 AITrain 归一化 YOLO `data.yaml`，调用 `ultralytics.YOLO(...).train()`，并导出 ONNX。
- trainer 上报 `boxLoss`、`classLoss`、`dflLoss`、`precision`、`recall`、`mAP50`、`mAP50_95`、`loss`。
- trainer 归一化 artifact：`best.pt`、`last.pt`、`results.csv`、`args.yaml`、`aitrain_yolo_data.yaml`、`model.onnx`、`ultralytics_training_report.json`。
- 已用 fake official API package 做 Worker 端确定性测试，避免 CI 下载大模型。
- 已在 `.deps` Python 安装 Ultralytics 8.4.45、Torch 2.11.0 CPU、ONNX 1.21.0、ONNX Runtime 1.25.1。
- 已用最小 detection 数据集和 `yolov8n.yaml` 在 CPU 上跑通 1 epoch smoke，产出 `best.pt`、`best.onnx` 和 `ultralytics_training_report.json`。

验收状态：已完成本机 CPU smoke。

## 8. 阶段 10：真实检测 ONNX 推理与模型转换

目标：让官方 YOLO detection 训练产物可以由 AITrain Studio 的 C++ 推理和部署链路使用。

当前状态：

- C++ ONNX Runtime 可以识别并运行 Ultralytics YOLO detection ONNX。
- 已实现 letterbox 预处理、YOLOv8-style 输出 decode、置信度过滤、NMS、类别映射、坐标回原图、prediction JSON 和 overlay。
- `exportDetectionCheckpoint(..., format=onnx)` 可接受真实 YOLO ONNX 源文件，复制模型并写入 AITrain sidecar。
- TensorRT 转换入口可接受 ONNX 源文件；本机 GTX 1060 / SM 61 仍不能作为 TensorRT 10 真验收机器。
- 已用阶段 9 CPU smoke 产出的 `best.onnx` 做本机 ONNX Runtime decode 测试。

验收状态：ONNX Runtime 已完成；TensorRT external GPU pending。

## 9. 阶段 11：官方 YOLO 分割训练接入

目标：复用阶段 8 的 Python Trainer Adapter，接入 Ultralytics YOLO segmentation 训练、导出和指标。

当前状态：

- 已完成 `python_trainers/segmentation/ultralytics_trainer.py`。
- Worker 将 `trainingBackend=ultralytics_yolo_segment` 路由到该 trainer。
- trainer 复用 Ultralytics 官方 Python API，默认模型为 `yolov8n-seg.yaml`。
- 已解析 mask 指标：`maskPrecision`、`maskRecall`、`maskMap50`、`maskMap50_95`。
- 已用最小 polygon segmentation 数据集在 CPU 上跑通 1 epoch smoke，产出 `best.pt`、`best.onnx` 和 `ultralytics_training_report.json`。
- C++ ONNX Runtime 已支持 YOLOv8-seg 后处理：boxes、mask coefficients、prototype masks、mask resize/crop、NMS、prediction JSON 和 overlay。
- 当前 C++ segmentation 训练后端仍是 scaffold。

验收状态：已完成本机 CPU smoke。

后续增强：

- C++ 侧补真实 segmentation ONNX 后处理：boxes、mask coefficients、prototype masks、mask resize / crop / overlay。
- GUI 显示真实 mask overlay、预测 mask 和耗时。

## 10. 阶段 12：PaddlePaddle OCR Rec 训练接入

目标：接入真实 OCR Rec 训练能力，替换当前 OCR Rec scaffold 的真实训练缺口。

当前状态：

- 已安装 PaddlePaddle 3.3.1 和 PaddleOCR 3.5.0 到 `.deps` Python。
- 已完成 `python_trainers/ocr_rec/paddleocr_trainer.py`。
- Worker 将 `trainingBackend=paddleocr_rec` 路由到该 trainer。
- trainer 读取 PaddleOCR-style Rec 数据：
  - `dict.txt`
  - `rec_gt.txt`
  - `images/`
- trainer 使用 PaddlePaddle 搭建小型 CTC Rec 网络，支持 CPU 小数据训练。
- 已输出 `loss`、`accuracy`、`editDistance`。
- 已归一化 artifact：
  - `paddleocr_rec_ctc.pdparams`
  - `paddleocr_rec_ctc.onnx`
  - `dict.txt`
  - `paddleocr_rec_training_report.json`
- 已用最小 OCR Rec 数据集在 CPU 上跑通 1 epoch smoke。
- C++ ONNX Runtime 已支持当前 CTC ONNX 模型的 resize / grayscale / logits decode / CTC greedy decode / preview overlay。

明确边界：

- 当前实现是 PaddlePaddle CTC Rec trainer，兼容 PaddleOCR-style 数据。
- 当前实现不是完整 PP-OCRv4 官方训练配置。
- 当前还未完成 PaddleOCR 官方 inference model 导出和完整 PP-OCRv4 官方配置。

验收状态：已完成本机 CPU smoke。

后续增强：

- 接入 PaddleOCR 官方训练配置生成。
- 支持官方 inference model 导出。
- 后续补完整 PaddleOCR 官方 inference model 导出和更完整的 OCR 预处理策略。
- 支持 checkpoint resume。

## 11. 阶段 13：产品化与验收

目标：把平台从开发可用推进到可交付、可复现、可诊断。

本机已完成交付：

- Python 环境说明：
  - 推荐 venv。
  - requirements 文件。
  - 离线 wheelhouse 策略。
  - Python path 配置策略。
- 示例数据集：
  - YOLO detection 最小样例。
  - YOLO segmentation 最小样例。
  - PaddleOCR Rec 最小样例。
- 用户文档：
  - 安装说明。
  - Python 训练环境说明。
  - 第三方官方后端许可证说明。
  - 数据集格式说明。
  - 训练流程说明。
  - 导出和推理说明。
  - 常见环境错误说明。
- 硬件兼容矩阵：
  - CPU-only 可用能力。
  - GTX 1060 / SM 61 可用能力。
  - RTX / SM 75+ TensorRT 验收要求。
- 本机 package smoke。
- 外部 RTX / SM 75+ TensorRT engine 验收清单。

新增产物：

- `python_trainers/requirements-yolo.txt`
- `python_trainers/requirements-ocr.txt`
- `docs/training-backends.md`
- `docs/hardware-compatibility.md`
- `examples/create-minimal-datasets.py`
- `examples/README.md`
- README 和 Python trainer 协议文档更新
- package smoke 对 examples、docs、requirements 的安装布局检查

本机验收结果：

- `examples/create-minimal-datasets.py` 可生成 detection、segmentation、OCR Rec 最小数据集和 request JSON。
- 生成的三类 request 已直接跑通 CPU trainer smoke。
- `.\tools\harness-check.ps1` 通过。
- `.\tools\package-smoke.ps1 -SkipBuild` 通过。

外部机器待完成：

- 干净 Windows 机器完整安装验收。
- RTX / SM 75+ TensorRT smoke。
- 真实中等规模数据集训练验收。

验收标准：

- 干净 Windows 机器可启动 GUI。
- 插件可加载。
- Worker 自检可解释当前 C++ runtime 和 Python trainer 环境能力。
- 无 GPU 或不兼容 GPU 时提示清晰。
- 示例数据可跑通 detection、segmentation、OCR 的最小训练闭环。
- RTX / SM 75+ 机器上 TensorRT smoke 通过。

## 12. 推荐优先级

| 优先级 | 事项 | 原因 |
|---|---|---|
| P0 | 阶段 8 Python Trainer Adapter | 所有真实训练后端的共同入口 |
| P0 | 阶段 9 官方 YOLO 检测训练 | 检测是后续分割、导出、部署的基础 |
| P0 | 阶段 10 真实检测 ONNX 后处理 | 训练产物必须能被 C++ 推理、部署和验证 |
| P1 | 阶段 11 官方 YOLO 分割训练 | 复用检测和 Python adapter，扩展 mask 能力 |
| P1 | 阶段 12 PaddlePaddle OCR Rec 训练 | 补齐 OCR 产品线 |
| P1 | TensorRT GPU capability 自检增强 | 避免用户在 SM 61 等硬件上误判可用性 |
| P2 | 阶段 13 产品化验收 | 面向交付、文档和兼容矩阵 |

## 13. 风险与控制

| 风险 | 影响 | 控制方式 |
|---|---|---|
| Ultralytics 许可证不适合闭源商用 | 后续分发受限 | 后端可选化，文档明确 AGPL / Enterprise 许可 |
| Python 依赖体积大、安装复杂 | 用户环境失败率高 | venv、requirements、离线 wheelhouse、环境自检和清晰错误 |
| 官方包输出格式变化 | 指标和 artifact 解析失败 | adapter 层隔离，记录官方包版本，测试固定最小样例 |
| 本机 GPU 不支持 TensorRT 10 | 阶段 7 真 engine 无法本机验收 | 保留外部 GPU 验收待办，本机继续推进 Python 训练和 ONNX 推理 |
| PaddlePaddle Windows GPU 兼容性变化 | OCR 训练无法稳定运行 | 先支持 CPU 小样例验收，再补 CUDA 版本矩阵 |
| 数据集质量不可控 | 训练指标异常 | 训练前强制校验，错误定位到文件和行号 |
| GUI 与训练耦合 | 稳定性下降 | 训练只在 Worker 启动的 Python 子进程中实现 |
| scaffold 被误认为真实能力 | 产品判断失真 | 文档和 UI 明确标注 scaffold / baseline / official backend |

## 14. 下一步执行建议

阶段 10-13 的本机目标已完成。下一步是外部验收：

1. 在干净 Windows 机器安装包体并启动 GUI。
2. 运行 packaged Worker self-check 和 plugin smoke。
3. 在 RTX / SM 75+ 机器运行 TensorRT smoke。
4. 用非最小真实数据集分别跑 detection、segmentation、OCR Rec 的长一点训练。
5. 将外部验收结果写回 `docs/harness/current-status.md`。

阶段 7 / 10 保留待办：

- 找到 RTX / SM 75+ 机器后运行 `aitrain_worker.exe --self-check`。
- 运行 `aitrain_worker.exe --tensorrt-smoke <work-dir>`。
- 补记外部 GPU 验收结果，并把 `docs/harness/current-status.md` 中阶段 7/10 的 TensorRT 状态从 pending 更新为 accepted 或 done。

## 15. 阶段 14：官方 PaddleOCR Rec 适配器

目标：在保留现有 `paddleocr_rec` 小型 CTC 后端的前提下，新增官方 PaddleOCR / PP-OCRv4 Rec 链路适配层。

已完成：

- 新增 `python_trainers/ocr_rec/paddleocr_official_adapter.py`。
- Worker 新增后端路由：`paddleocr_rec_official`、`paddleocr_ppocrv4_rec`。
- adapter 可把 AITrain PaddleOCR-style Rec 数据转换为官方 PaddleOCR 训练需要的材料：`train_list.txt`、`val_list.txt`、`dict.txt`、`aitrain_ppocrv4_rec.yml`、`paddleocr_official_rec_report.json`、`run_official_train.ps1`、`run_official_export.ps1`。
- `prepareOnly=true` 已完成本机 smoke，用于验证配置生成、数据列表、报告和 Worker 事件链路。
- `runOfficial=true` 或 `prepareOnly=false` 时，可在提供 `paddleOcrRepoPath` 或 `AITRAIN_PADDLEOCR_REPO` 后调用官方 `tools/train.py` 与 `tools/export_model.py`。

明确边界：

- `prepareOnly=true` 只代表官方配置和命令准备完成，不代表已经训练出官方模型。
- 当前共享 `.deps` Python 同时装有 PaddlePaddle 和 PyTorch；官方 PaddleOCR 训练脚本经 `albumentations` 导入 PyTorch 后会触发 Windows DLL 冲突。完整官方长训练应在隔离 OCR Python 环境或干净机器上运行。
- 当前 `paddleocr_rec` 小型 CTC 后端仍保留，用于快速 smoke、C++ ONNX CTC decode 验证和不依赖官方仓库的本机闭环。

验收：

```powershell
.\.deps\python-3.13.13-embed-amd64\python.exe -m py_compile python_trainers\ocr_rec\paddleocr_official_adapter.py
.\.deps\python-3.13.13-embed-amd64\python.exe python_trainers\ocr_rec\paddleocr_official_adapter.py --request .deps\phase14-ocr-smoke\paddleocr_rec_official_request.json
```

## 16. 阶段 15：推理结果 UI 可读性增强

目标：不改变 Worker 推理入口、不把推理逻辑放入 GUI，只让 GUI 更清楚地区分 detection、segmentation 和 OCR recognition 的结果类型。

已完成：

- `inferenceResult` 显示 `taskType`。
- 读取 `inference_predictions.json` 后生成简短摘要：detection 显示目标数量和首个类别/置信度；segmentation 显示实例数量、首个类别/置信度和 mask area；OCR recognition 显示识别文本和置信度。
- overlay 显示链路保持不变。
- 若 JSON 缺失或格式异常，GUI 给出清晰提示，但不影响 artifact 记录。

## 17. 阶段 16：隔离 OCR 官方训练环境验收

目标：在不混用 YOLO / PyTorch 环境的前提下，验证官方 PaddleOCR PP-OCRv4 Rec 训练和导出链路可以真实跑通。

已完成：

- 新增 `tools/phase16-ocr-official-smoke.ps1`。
- 脚本会创建或复用 `.deps/python-3.13.13-ocr-amd64` 隔离 Python。
- 脚本会安装 PaddlePaddle 与官方 PaddleOCR 源码仓库 requirements。
- 脚本会检查官方 `tools/train.py` 可导入。
- 脚本会生成最小 OCR Rec 数据集，并通过 `paddleocr_rec_official` 后端运行 1 epoch CPU 官方训练。
- 脚本会调用官方 `tools/export_model.py` 生成 `official_inference`。
- adapter 已解析官方 stdout 指标并写入 report / Worker metric 事件：
  - `loss`
  - `ctcLoss`
  - `nrtrLoss`
  - `accuracy`
  - `normalizedEditDistance`

本机验收产物：

- `official_model/best_accuracy.pdparams`
- `official_inference/inference.yml`
- `paddleocr_official_rec_report.json`

验收命令：

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

明确边界：

- 阶段 16 证明官方 PaddleOCR 训练/导出链路和 AITrain Worker adapter 能闭环。
- 阶段 16 使用极小样例数据，只验证 wiring、artifact 和 report，不代表 OCR 精度。
- 后续若要让 C++ 直接消费官方 Paddle inference model，还需要继续实现 Paddle inference runtime 或 Paddle2ONNX 转换链路。

## 18. 阶段 17-21：交付验收基线

目标：不新增产品训练功能，只把当前本机 smoke 成果收敛为可交付、可复现、可外部验收的基线。

### 18.1 阶段 17：本机基线冻结

已完成：

- 审阅当前未提交改动，确认属于 Phase 16、推理增强、验收脚本和文档收口范围。
- 运行 `git diff --check`，仅存在 Git 换行提示，无空白错误。
- 运行 `.\tools\harness-check.ps1` 并通过。
- 运行 `.\tools\package-smoke.ps1 -SkipBuild` 并通过。
- 运行 `.\tools\phase16-ocr-official-smoke.ps1` 并通过，产出官方 PaddleOCR tiny smoke checkpoint、inference config 和 report。

边界：

- C++ tiny detector、segmentation baseline、OCR baseline 仍是 scaffold / baseline。
- Phase 7 / Phase 10 TensorRT 不得标记为完成，仍需 RTX / SM 75+ 真机验收。

### 18.2 阶段 18：验收包和一键验收脚本

新增产物：

- `docs/acceptance-runbook.md`：记录本机、包体、公开/生成数据、TensorRT 外部验收流程。
- `tools/acceptance-smoke.ps1`：统一入口，支持 `-LocalBaseline`、`-Package`、`-PublicDatasets`、`-TensorRT`。
- CMake install 和 `tools/package-smoke.ps1` 已补充验收脚本与 runbook 的包体检查。

本机验收：

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
.\tools\harness-check.ps1
```

包体目录也可直接运行：

```powershell
.\tools\acceptance-smoke.ps1 -Package
```

### 18.3 阶段 19：TensorRT 外部验收准备

已完成：

- `tools\acceptance-smoke.ps1 -TensorRT` 会先运行 Worker self-check。
- 脚本会检测 `nvidia-smi` 和 GPU compute capability。
- 当前 GTX 1060 / SM 61 上明确输出 `hardware-blocked`，不伪造通过。
- 在 RTX / SM 75+ 上，脚本会继续运行 `aitrain_worker.exe --tensorrt-smoke <work-dir>`。

外部验收命令：

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt
```

### 18.4 阶段 20：小规模真实数据训练、推理、转换验收

已完成：

- `tools\acceptance-smoke.ps1 -PublicDatasets` 会生成最小 detection、segmentation、OCR Rec 数据集。
- 脚本会尝试通过 Ultralytics 官方包 materialize COCO8 / COCO8-seg；当前本机 materialization 不可用时回退到生成数据集。
- YOLO detection 通过 `ultralytics_yolo_detect` adapter 跑通 1 epoch CPU smoke，产出 `best.pt`、ONNX 和 `ultralytics_training_report.json`。
- YOLO segmentation 通过 `ultralytics_yolo_segment` adapter 跑通 1 epoch CPU smoke，产出 `best.pt`、ONNX 和 mask 指标 report。
- OCR Rec 小型 PaddlePaddle CTC backend 跑通 1 epoch CPU smoke，产出 `.pdparams`、ONNX、`dict.txt` 和 report。
- C++ ONNX Runtime 相关 CTest 继续覆盖 detection、segmentation mask postprocess、OCR CTC decode 和 overlay 链路。
- 隔离官方 PaddleOCR Rec train/export smoke 再次通过。

边界：

- 生成数据集和 tiny 官方数据集只验证 wiring、artifact、report、转换和推理链路，不代表真实精度。
- 若 ICDAR2015 等公开数据集需要注册或交互下载，应记录为外部数据获取阻塞，不影响本阶段必过 smoke。

### 18.5 阶段 21：发布前收口

已完成：

- README 增加 Phase 17-21 验收入口。
- `docs/training-backends.md` 增加统一训练 smoke 和 PaddleOCR 官方隔离环境说明。
- `docs/hardware-compatibility.md` 改为使用 `tools\acceptance-smoke.ps1 -TensorRT` 作为外部 TensorRT 验收入口。
- `docs/harness/current-status.md` 增加 Phase 17-21 状态，并继续把 TensorRT 标记为 external pending / hardware-blocked。
- `docs/acceptance-runbook.md` 作为交付验收 runbook 保留。

最终本机验收命令：

```powershell
.\tools\harness-context.ps1
.\tools\harness-check.ps1
.\tools\package-smoke.ps1 -SkipBuild
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package
```

仍待外部完成：

- 在干净 Windows 机器上验证 ZIP 包启动、Worker self-check、plugin smoke 和 Python 环境提示。
- 在 RTX / SM 75+ 机器上运行真实 TensorRT smoke，并将结果回写 `docs/harness/current-status.md`。
