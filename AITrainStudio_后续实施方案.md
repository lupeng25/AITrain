# AITrain Studio 后续实施方案

生成日期：2026-04-30
项目目录：`C:\Users\73200\Desktop\code\AITrain`
权威状态入口：`docs/harness/current-status.md`

本文档用于规划 AITrain Studio 在阶段 7 之后的开发路线。阶段状态以 `docs/harness/current-status.md` 为准；本文档提供中长期路线、阶段交付、验收标准和风险控制。

## 1. 当前结论

AITrain Studio 已具备可运行的平台骨架：Qt Widgets GUI、独立 Worker、JSON Lines over `QLocalSocket`、SQLite 项目元数据、插件接口、数据集校验、tiny detector scaffold、ONNX Runtime 推理链路、分割 scaffold、OCR scaffold、Windows 打包和 TensorRT runtime 接入。

当前关键事实：

- 阶段 1-2 已作为平台和数据系统初版完成。
- 阶段 3-6 是可执行 scaffold / baseline，不是完整真实 YOLO/OCR 训练。
- 阶段 7 代码完成，本机打包、自检、package smoke、ZIP 生成已通过。
- 阶段 7 TensorRT 真 engine 验收因本机 GTX 1060 / SM 61 被 TensorRT 10 拒绝而挂起，需要 RTX / SM 75+ 机器或云 GPU 补验收。
- 当前最有价值的本机开发主线是阶段 8：Python Trainer Adapter。真实训练优先复用官方 Python 实现；C++ 侧继续负责 Worker 编排、项目管理、推理、模型转换、TensorRT 和打包。

## 2. 实施原则

- 训练逻辑必须进入 `src/core`、插件或 `aitrain_worker`，不得放入 `MainWindow`。
- 长任务必须通过 Worker 执行，GUI 只负责交互、状态展示和任务调度。
- SQLite 元数据必须通过 `ProjectRepository` 管理。
- 模型、数据集、导出、推理、验证能力继续走插件接口。
- Qt 继续保持 5.12+ 兼容，除非明确升级。
- scaffold 必须明确标注，不把 tiny detector、分割 baseline、OCR baseline 描述成真实 YOLO/OCR 训练。
- 真实训练优先通过 `aitrain_worker` 启动独立 Python 子进程实现，不把 Python 嵌入 GUI 进程。
- 若官方 Python 实现可用，优先适配官方实现；只有在许可、离线部署、产物格式或稳定性不满足要求时，才自研轻量训练实现。
- Python 训练进程必须使用统一 JSON request 和 JSON Lines 输出协议，Worker 负责转发日志、进度、指标、artifact、失败和取消。
- C++ 侧保留 ONNX Runtime、TensorRT、overlay、prediction JSON、打包和环境自检能力。
- 每个阶段必须有最小数据集验收和 `.\tools\harness-check.ps1` 验证。

## 3. 阶段路线总览

| 阶段 | 状态 | 目标 | 验收标准 |
|---|---|---|---|
| 阶段 1 | 已完成为平台 scaffold | 平台稳定化 | Worker、任务状态、SQLite、插件接口、harness 可用 |
| 阶段 2 | 已完成为初版系统 | 数据集系统 | YOLO 检测/分割、PaddleOCR Rec 校验可用 |
| 阶段 3 | scaffold 完成 | YOLO 检测训练入口 | tiny detector 可训练、保存 checkpoint、输出指标 |
| 阶段 4 | tiny detector 路径完成 | ONNX 导出与推理 | tiny ONNX 导出、ONNX Runtime 推理、overlay 输出 |
| 阶段 5 | scaffold 完成 | YOLO 分割训练入口 | 分割数据读取、mask baseline、preview、指标输出 |
| 阶段 6 | scaffold 完成 | OCR Rec 训练入口 | OCR 数据读取、字典、baseline checkpoint、preview |
| 阶段 7 | 代码完成，外部 GPU 验收待补 | TensorRT 与 Windows 打包 | 本机自检和打包通过，TensorRT 真验收待 RTX / SM 75+ |
| 阶段 8 | 已完成 | Python Trainer Adapter 与环境管理 | Worker 可启动 Python 训练进程，接收 JSON Lines 指标和 artifact |
| 阶段 9 | 已完成本机 CPU smoke | 官方 YOLO 检测训练接入 | Worker 已接入 Ultralytics YOLO detection trainer adapter，可导出 checkpoint / ONNX / report |
| 阶段 10 | 待开始 | 真实检测 ONNX 推理与模型转换 | C++ ONNX Runtime decoder、NMS、类别映射、overlay、TensorRT 转换链路可用 |
| 阶段 11 | 待开始 | 官方 YOLO 分割训练接入 | 官方 Python YOLO segmentation 可训练、导出、生成 mask 预览 |
| 阶段 12 | 待开始 | 官方 PaddleOCR Rec 训练接入 | 官方 PaddleOCR / PaddlePaddle 训练、导出、decode、accuracy 可用 |
| 阶段 13 | 待开始 | 产品化与验收 | 样例数据、文档、兼容矩阵、干净机器和外部 GPU 验收完成 |

## 4. 已完成阶段复盘

### 4.1 阶段 1：平台稳定化

已完成内容：

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

已完成内容：

- YOLO 检测数据校验。
- YOLO segmentation 数据读取与基础校验。
- PaddleOCR Rec 标签和字典读取。
- 检测数据集划分初版。

后续增强方向：

- 大数据集校验进度和取消。
- 错误样本跳转和修复建议。
- 分割和 OCR 的更完整划分策略。

### 4.3 阶段 3：YOLO 检测训练 scaffold

已完成内容：

- tiny linear detector 训练闭环。
- loss / mAP50-style 指标。
- checkpoint 保存和恢复。
- preview artifact。

明确限制：

- 这不是 LibTorch/CUDA YOLO。
- 这不是完整 detection head、bbox loss、objectness/classification loss。

### 4.4 阶段 4：ONNX 导出与推理 scaffold

已完成内容：

- tiny detector ONNX 导出。
- ONNX Runtime 推理。
- checkpoint / ONNX 一致性测试。
- Worker 推理、prediction JSON、overlay 输出。

明确限制：

- 完整 YOLO decoder、NMS、类别映射和真实模型后处理仍未完成。
- OCR 和分割的完整 ONNX 导出仍未完成。

### 4.5 阶段 5：YOLO 分割 scaffold

已完成内容：

- `SegmentationDataset` 和 `SegmentationDataLoader`。
- polygon-to-mask。
- letterbox 对齐 mask。
- 多 polygon / 多 class mask。
- overlay preview 和 mask preview artifact。
- Worker 侧 mask 指标和 scaffold checkpoint。

明确限制：

- 没有真实 mask head。
- 没有真实 mask loss。
- 没有 CUDA 训练。

### 4.6 阶段 6：OCR Rec scaffold

已完成内容：

- `OcrRecDataset`。
- 字符字典加载。
- label encode / decode。
- resize / pad batching。
- Worker 侧 `ctcLoss`、`accuracy`、`editDistance` baseline。
- scaffold checkpoint 和 preview artifact。

明确限制：

- 没有真实 CRNN。
- 没有真实 CTC loss。
- 没有真实 OCR ONNX 导出。

### 4.7 阶段 7：TensorRT 与 Windows 打包

已完成内容：

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

阶段 7 外部验收命令：

```powershell
.\aitrain_worker.exe --self-check
.\aitrain_worker.exe --tensorrt-smoke <work-dir>
.\tools\package-smoke.ps1
```

## 5. 官方 Python 实现选型原则

目标：真实训练不再优先自研 C++ / LibTorch 训练核心，而是优先接入成熟官方 Python 实现。AITrain Studio 负责把这些训练能力产品化、流程化、可视化和可部署。

官方实现优先级：

| 任务 | 首选实现 | 原因 | 约束 |
|---|---|---|---|
| YOLO 检测 | Ultralytics YOLO Python / CLI | 官方支持 detection 训练、Python API、CLI、导出 | 需要处理 AGPL-3.0 / Enterprise License 取舍 |
| YOLO 分割 | Ultralytics YOLO segment Python / CLI | 同一套数据、训练、导出和模型生态 | 同检测，商业闭源分发前必须确认许可 |
| PaddleOCR Rec | PaddleOCR / PaddlePaddle 官方训练脚本 | 官方支持 PP-OCR 系列训练、推理、部署和 ONNX / 高性能推理路径 | 依赖 PaddlePaddle 环境和 Windows GPU 兼容性 |

官方实现接入规则：

- 官方 Python 包不直接进入 GUI 进程。
- `aitrain_worker` 启动 Python 子进程，并通过 JSON request / JSON Lines 协议交换状态。
- 官方训练输出必须被归一化为 AITrain Studio artifact：
  - checkpoint。
  - best model。
  - ONNX 或官方可转换模型。
  - class / dict 映射。
  - training report。
  - preview / validation result。
- C++ 侧继续负责：
  - 项目和任务状态。
  - SQLite 记录。
  - Worker 取消和失败上报。
  - ONNX Runtime 推理。
  - TensorRT 转换和推理。
  - overlay、prediction JSON、packaging。
- 若官方实现因许可、安装体积、离线部署、Windows GPU 兼容性或导出格式不满足要求，再考虑维护自研 fallback。

许可和分发原则：

- Ultralytics 官方文档说明 YOLO 提供 AGPL-3.0 和 Enterprise 两种许可。若 AITrain Studio 未来闭源商用分发，应把 Ultralytics 后端标为可选组件，并在文档中要求用户自行确认 AGPL 或企业许可。
- PaddleOCR GitHub 仓库标注为 Apache-2.0 license，适合优先作为 OCR 官方后端，但仍需在打包前确认 PaddlePaddle、CUDA、cuDNN 等依赖许可。
- 产品里必须显示“后端来源”和“许可证提示”，不要把第三方官方训练能力伪装成本项目自研。

## 6. 阶段 8：Python Trainer Adapter 与环境管理

目标：建立统一 Python 训练桥接层，让 C++ Worker 能稳定启动、监控、取消和接收 Python 训练任务。阶段 8 不直接追求模型精度，重点是把训练子进程协议打稳。

当前进度：

- 已完成训练后端状态查询、checkpoint schema v2 元数据、模型族和 scaffold 标记。
- Worker 检测训练产物和完成消息会携带当前训练后端、模型族、scaffold 状态。
- 未实现的 `yolo_style_libtorch` 后端现在会明确报错，不会静默回退成 tiny detector。
- 默认可执行训练仍是 `tiny_linear_detector`，还不是真实 YOLO-style 训练。
- 已完成 Python Trainer Adapter：
  - Worker 通过 `QProcess` 启动 Python 子进程。
  - Worker 写出 `python_trainer_request.json`。
  - Python trainer 通过 stdout JSON Lines 返回 `log`、`progress`、`metric`、`artifact`、`completed`、`failed`。
  - Worker 可以传播 Python trainer 失败。
  - 环境自检会报告 Python、Ultralytics、PaddleOCR、PaddlePaddle 可用性。
  - `python_mock` trainer 仅用于协议和 Worker 链路验证，不是真实训练。

### 6.1 交付内容

- 新增 `python_trainers/` 或 `trainers/python/` 目录。
- 定义 Python trainer request JSON：
  - task id。
  - task type。
  - dataset path。
  - output path。
  - backend id。
  - epochs、batch size、image size、device、export options。
- 定义 Python trainer stdout JSON Lines 协议：
  - `log`。
  - `progress`。
  - `metric`。
  - `artifact`。
  - `completed`。
  - `failed`。
- Worker 使用 `QProcess` 启动 Python，不解析普通文本日志作为唯一状态来源。
- 支持取消：Worker 取消时终止 Python 子进程，并记录 canceled 状态。
- Python 环境自检：
  - Python 可执行文件。
  - venv 路径。
  - pip package 检测。
  - CUDA / CPU 训练能力。
  - Ultralytics 可用性。
  - PaddleOCR / PaddlePaddle 可用性。
- 保留 C++ tiny detector scaffold 作为 demo / fallback / test backend。

### 6.2 边界

- 不在 GUI 进程内嵌 Python。
- 不要求阶段 8 完成真实 YOLO 训练。
- 不删除现有 C++ scaffold。
- 不把官方包复制进源码仓库；依赖通过环境安装、离线 wheelhouse 或包体可选组件处理。

### 6.3 验收标准

- Worker 能启动一个最小 Python trainer mock，并收到 JSON Lines 日志、进度、metric、artifact、completed。
- Worker 能处理 Python trainer 返回 failed。
- Worker 能取消 Python trainer。
- 环境自检能明确报告 Python、Ultralytics、PaddleOCR / PaddlePaddle 的可用或缺失。
- `.\tools\harness-check.ps1` 通过。

## 7. 阶段 9：官方 YOLO 检测训练接入

目标：优先接入官方 YOLO Python 实现，完成真实目标检测训练、验证、导出和产物归一化。

当前状态：

- 已完成 `python_trainers/detection/ultralytics_trainer.py`。
- Worker 已将 `trainingBackend=ultralytics_yolo_detect` 和 `trainingBackend=ultralytics_yolo` 默认路由到该 trainer。
- trainer 会生成 AITrain 归一化 YOLO `data.yaml`，调用官方 `ultralytics.YOLO(...).train()`，并导出 ONNX。
- trainer 会上报 `boxLoss`、`classLoss`、`dflLoss`、`precision`、`recall`、`mAP50`、`mAP50_95`、`loss`。
- trainer 会归一化 artifact：`best.pt`、`last.pt`、`results.csv`、`args.yaml`、`aitrain_yolo_data.yaml`、`model.onnx`、`ultralytics_training_report.json`。
- 已用 fake official API package 做 Worker 端确定性测试，避免 CI 下载大模型。
- 已在 `.deps` Python 安装 Ultralytics 8.4.45、Torch 2.11.0 CPU、ONNX 1.21.0、ONNX Runtime 1.25.1。
- 已用最小 YOLO detection 数据集和 `yolov8n.yaml` 在 CPU 上跑通 1 epoch smoke，产出 `best.pt`、`best.onnx` 和 `ultralytics_training_report.json`。

### 7.1 交付内容

- `python_trainers/detection/` 后端。
- 支持 Ultralytics Python API 或 CLI：
  - `detect train`。
  - `detect val`。
  - `export format=onnx`。
- 将 AITrain Studio 检测数据集路径转换为官方 YOLO data yaml。
- 支持训练参数：
  - epochs。
  - batch。
  - imgsz。
  - device。
  - pretrained / from scratch。
  - workers。
  - project / run name。
- 解析训练输出并上报：
  - train loss。
  - box loss。
  - class loss。
  - dfl loss 或官方等价指标。
  - precision。
  - recall。
  - mAP50。
  - mAP50-95。
- 归一化 artifact：
  - best `.pt`。
  - last `.pt`。
  - exported `.onnx`。
  - results csv / json。
  - confusion matrix 或 preview。
- 写入 AITrain training report。

### 7.2 边界

- 官方 Ultralytics 后端必须显示许可证提示。
- 若用户未安装 Ultralytics，Worker 应给出安装指引，不自动静默下载大包。
- 若本机 GPU 不适合训练，允许 CPU 小数据集验收，但 UI 必须提示性能限制。
- 训练成功不等于 TensorRT 成功；TensorRT 仍由阶段 7 外部 GPU 验收和阶段 10 转换链路负责。

### 7.3 验收标准

- 最小 YOLO detection 数据集可通过官方 Python 后端训练 1 epoch。
- Worker 可以显示真实训练 loss 和 mAP 指标。
- 训练完成后能找到 best checkpoint 和 ONNX。
- checkpoint / report / class names 被写入 artifact。
- 失败时能定位到 Python 环境、官方包、数据集、CUDA 或训练参数。

## 8. 阶段 10：真实检测 ONNX 推理与模型转换

目标：让官方 YOLO 检测训练产物可以由 AITrain Studio 的 C++ 推理和部署链路使用。

### 8.1 交付内容

- 读取官方 YOLO ONNX sidecar。
- ONNX Runtime session shape 检查。
- YOLO detection decoder。
- NMS。
- 置信度过滤。
- 类别映射。
- bbox 坐标从 letterbox 空间映射回原图。
- prediction JSON。
- overlay 图像输出。
- Worker 推理流程接入真实 YOLO ONNX。
- TensorRT 转换入口：
  - ONNX -> TensorRT engine。
  - engine sidecar。
  - GPU capability 检查。

### 8.2 验收标准

- 同一张图片在官方 Python predict 和 C++ ONNX Runtime 推理中输出同类、同位置范围的 bbox。
- 输出 bbox 坐标不越界。
- NMS 后结果稳定。
- GUI 推理页能显示结果和耗时。
- TensorRT 转换在 RTX / SM 75+ 机器上可验收；GTX 1060 / SM 61 上给出清晰不兼容提示。

## 9. 阶段 11：官方 YOLO 分割训练接入

目标：复用阶段 8 的 Python Trainer Adapter，接入官方 YOLO segmentation 训练、导出和预览。

### 9.1 交付内容

- `python_trainers/segmentation/` 后端。
- 支持官方 YOLO segmentation train / val / export。
- 将 AITrain Studio segmentation dataset 转换或校验为官方可用格式。
- 上报：
  - box loss。
  - class loss。
  - mask / segment loss。
  - precision。
  - recall。
  - mask mAP。
- 归一化 artifact：
  - best `.pt`。
  - last `.pt`。
  - ONNX。
  - mask preview。
  - training report。
- C++ 侧补真实 segmentation ONNX 后处理：
  - boxes。
  - mask coefficients。
  - prototype masks。
  - mask resize / crop / overlay。

### 9.2 验收标准

- 最小 YOLO segmentation 数据集可训练 1 epoch。
- Worker 能显示真实 mask 相关指标。
- 训练产物可以导出 ONNX。
- C++ 推理能生成 mask overlay。
- 当前 C++ segmentation scaffold 仍保留为测试后端，不再被描述为真实训练。

## 10. 阶段 12：官方 PaddleOCR Rec 训练接入

目标：接入 PaddleOCR / PaddlePaddle 官方训练能力，替换当前 OCR Rec scaffold 的真实训练缺口。

### 10.1 交付内容

- `python_trainers/ocr_rec/` 后端。
- 支持 PaddleOCR Rec 官方训练配置生成。
- 支持字典、训练/验证 label 文件、图片路径映射。
- 支持训练参数：
  - epochs。
  - batch size。
  - image shape。
  - learning rate。
  - pretrained。
  - device。
- 上报：
  - loss。
  - accuracy。
  - edit distance 或官方等价指标。
- 归一化 artifact：
  - Paddle checkpoint。
  - inference model。
  - ONNX 或可转换模型。
  - dict。
  - preview JSON。
  - training report。
- C++ 侧补 OCR ONNX Runtime 推理：
  - resize / pad。
  - logits decode。
  - CTC greedy decode。
  - preview。

### 10.2 验收标准

- PaddleOCR Rec 格式小数据集可训练 1 epoch。
- Worker 能显示真实 accuracy / loss。
- checkpoint 可恢复或继续训练。
- 导出模型可以被 C++ 或官方推理路径验证。
- GUI 显示图片、真实文本、预测文本和耗时。

## 11. 阶段 13：产品化与验收

目标：把平台从开发可用推进到可交付、可复现、可诊断。

### 11.1 交付内容

- Python 环境管理：
  - 推荐 venv。
  - requirements 文件。
  - 离线 wheelhouse 方案。
  - Python path 配置页或环境变量策略。
- 示例数据集：
  - YOLO 检测最小样例。
  - YOLO 分割最小样例。
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
- 干净机器包体验证。
- 外部 RTX / SM 75+ TensorRT engine 验收。
- 崩溃、失败和日志收集策略。

### 11.2 验收标准

- 干净 Windows 机器可启动 GUI。
- 插件可加载。
- Worker 自检可解释当前 C++ runtime 和 Python trainer 环境能力。
- 无 GPU 或不兼容 GPU 时提示清晰。
- RTX / SM 75+ 机器上 TensorRT smoke 通过。
- 示例数据能跑通检测、分割、OCR 的训练、导出、推理最小闭环。

## 12. 推荐优先级

| 优先级 | 事项 | 原因 |
|---|---|---|
| P0 | 阶段 8 Python Trainer Adapter | 所有真实训练后端的共同入口 |
| P0 | 阶段 9 官方 YOLO 检测训练 | 先把目标检测真实训练跑通，是后续分割、导出、部署的基础 |
| P0 | 阶段 10 真实检测 ONNX 后处理 | 训练产物必须能被 C++ 推理、部署和验证 |
| P1 | TensorRT GPU capability 自检增强 | 避免用户在 SM 61 等不兼容硬件上误判 TensorRT 可用 |
| P1 | 阶段 11 官方 YOLO 分割训练 | 复用检测和 Python adapter，扩展 mask 能力 |
| P1 | 阶段 12 官方 PaddleOCR Rec 训练 | 补齐 OCR 产品线 |
| P2 | 阶段 13 产品化验收 | 面向交付、文档和兼容矩阵 |

## 13. 风险与控制

| 风险 | 影响 | 控制方式 |
|---|---|---|
| 官方 YOLO 许可证不适合闭源商用 | 后续分发受限 | 后端可选化，文档明确 AGPL / Enterprise 许可，必要时提供自研或替代后端 |
| Python 依赖体积大、安装复杂 | 用户环境失败率高 | venv、requirements、离线 wheelhouse、环境自检和清晰错误 |
| 官方包输出格式变化 | 解析指标和 artifact 失败 | 通过 adapter 层隔离，记录官方包版本，测试固定最小样例 |
| 本机 GPU 不支持 TensorRT 10 | 阶段 7 真 engine 无法本机验收 | 保留外部 GPU 验收待办，本机继续推进 Python 训练和 ONNX 推理 |
| PaddlePaddle Windows GPU 兼容性变化 | OCR 训练无法稳定运行 | 先支持 CPU 小样例验收，再补 CUDA 版本矩阵 |
| 数据集质量不可控 | 训练指标异常 | 训练前强制校验，错误定位到文件和行号 |
| GUI 与训练耦合 | 稳定性下降 | 训练只在 Worker 启动的 Python 子进程中实现 |
| scaffold 被误认为真实能力 | 产品判断失真 | 文档和 UI 明确标注 scaffold / baseline / official backend |

## 14. 下一步执行建议

阶段 9 已完成本机 CPU smoke。下一步进入阶段 10：真实检测 ONNX 推理与模型转换。

第一批任务建议：

1. 已完成：梳理当前 `DetectionTrainer` tiny detector 与真实训练接口的可替换边界。
2. 已完成：定义检测 checkpoint schema v2 元数据、训练后端状态和最小模型配置边界。
3. 已完成：新增 Python trainer request / JSON Lines 协议文档和 mock trainer。
4. 已完成：Worker 使用 `QProcess` 启动 mock Python trainer，并转发 log / progress / metric / artifact / completed / failed。
5. 已完成：增加 Python 环境自检，报告 Python、Ultralytics、PaddleOCR / PaddlePaddle 可用性。
6. 已完成：接入官方 YOLO detection trainer adapter，优先通过 Ultralytics Python API，产出 checkpoint、report、class mapping 相关 data yaml 和 ONNX。
7. 已完成：安装真实官方依赖后，用最小 YOLO detection 数据集跑 1 epoch smoke，并记录结果。
8. 下一步：阶段 10 接入真实 YOLO ONNX decoder、NMS、类别映射、overlay 和部署产物转换。
9. 每完成一个小步运行 `.\tools\harness-check.ps1`。

阶段 7 保留待办：

- 找到 RTX / SM 75+ 机器后，运行 `aitrain_worker.exe --self-check`。
- 运行 `aitrain_worker.exe --tensorrt-smoke <work-dir>`。
- 补记外部 GPU 验收结果，并把 `docs/harness/current-status.md` 中阶段 7 状态从 `hardware-blocked` 更新为 `accepted` 或 `done`。
