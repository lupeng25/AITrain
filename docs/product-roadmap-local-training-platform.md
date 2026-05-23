# AITrain Studio 后续功能规划与实现方案（本地交付版）

日期：2026-05-13

定位：Windows + Qt Widgets + Worker 的本地视觉训练平台。

本文是 Phase 39+ 的当前方向文档。长期历史阶段仍可参考 `AITrainStudio_后续实施方案.md`，但下一步实施优先以本文和 `docs/harness/current-status.md` 为准。当前权威状态仍以 `docs/harness/current-status.md` 为准。

## 1. 当前基线与判断

AITrain Studio 已完成 Worker、SQLite、任务记录、artifact 浏览、YOLO 检测/分割官方训练链路、PaddleOCR Det/Rec/System 官方工具链入口、ONNX Runtime 推理、RTX 4090 TensorRT 验收证据、打包、离线授权、CPU smoke、插件 marketplace、环境 profile、样本复核、部署验证、诊断包、客户域 OCR 验收向导和成熟 Qt Widgets workbench。

维护基线：UI 与 core 的第一层源文件拆分已经完成。`MainWindow` 和 `ProductWorkflow` 都已拆为 companion 文件；`ProductWorkflow.cpp` 只保留公共入口锚点，snapshot、quality、evaluation、benchmark、delivery、acceptance、pipeline 逻辑分别落在同目录 companion 文件中。这是行为保持型重构，不改变 Worker protocol、SQLite schema、插件接口、报告 JSON 或算法语义。

当前项目的主要缺口不是继续堆更多 demo 后端。Phase 39A/39B/39C/41/49 已经把已有 detection / segmentation / OCR 能力推进到本地可交付闭环：

- 数据可质检。
- 训练可追踪。
- 模型可评估。
- 产物可导出和 benchmark。
- 模型版本可注册。
- 结果可生成交付报告。
- scaffold、official backend、hardware-blocked 能力边界必须清楚。

当前优先级固定为：

1. 收集外部 clean Windows package acceptance 证据，除非本 lane 明确继续 defer。
2. 只在明确重开时做 package-root TensorRT rerun；旧 GPU 保持 `hardware-blocked`。
3. 用客户域数据执行 OCR 验收；public Total-Text / generated smoke 不能作为生产证明。
4. 保持 Phase 40 新算法后置，除非产品优先级重新确认。

Phase 40 的分类、姿态、OBB、异常检测等训练后端扩展后置；在现有闭环足够硬之前，不作为主线推进。

## 2. 架构约束

- GUI 只做交互、调度和展示。
- 长任务继续进入 `aitrain_worker`。
- 元数据通过 `ProjectRepository` 写入 SQLite。
- 训练、推理、评估、导出逻辑不进入 `MainWindow`。
- 官方训练优先通过 Worker 管理的 Python trainer subprocess。
- C++ tiny detector、segmentation baseline、OCR baseline 只能作为 scaffold / diagnostic backend。
- TensorRT 真机验收已有 RTX 4090 D 证据；当前/旧 GTX 1060 / SM 61 仍只记录为 `hardware-blocked`。
- PaddleOCR System 当前是官方 `predict_system.py` 工具链路径；C++ DB-style Det ONNX probability-map 后处理已有 v1 wiring，但不代表 PP-OCRv5 精度 parity。

## 3. Phase 39A：真实评估补齐

目标：让 detection / segmentation / OCR Rec 都能通过 `evaluateModel` 生成可信的质量判断报告。

### Detection

保持现有 AP50 评估路径，并补充本地 COCO-style mAP50-95：

- 继续支持 tiny detector、detection ONNX Runtime 和可用 TensorRT detection model。
- 输出 precision、recall、AP50、mAP50、`mAP50_95`、per-class metrics、confusion matrix、error samples、overlay artifacts。
- 后续只做报告结构统一，不重写已通过的核心逻辑。

### Segmentation

新增真实 segmentation evaluation：

- 读取 YOLO segmentation val / test / train split。
- 将 polygon ground truth rasterize 为 mask。
- 调用现有 YOLOv8-seg ONNX Runtime 后处理。
- 计算 mask IoU、mask precision、mask recall、mask AP50、`maskMap50_95` 和 per-class mask 指标。
- 输出 `evaluation_report.json`、`per_class_metrics.csv`、`error_samples.json`、overlay artifacts。
- 报告 `scaffold=false`，但 limitations 说明本地 mAP50-95 仍需要客户/目标域验收配合使用。

### OCR Rec

新增真实 OCR Rec evaluation：

- 读取 PaddleOCR Rec label file 和 dictionary。
- 调用现有 OCR ONNX Runtime greedy decode。
- 计算 accuracy、edit distance、CER、WER。
- 输出错误样本、错误字符统计、低置信或空预测样本。
- 输出 `evaluation_report.json`、`error_samples.json`、可读 preview / overlay artifacts。
- official PaddleOCR inference model 仍不直接走 C++ runtime；报告中明确限制。

## 4. Phase 39B：本地流水线真实执行

目标：让 `runLocalPipeline` 不只是生成计划或请求文件，而是能真实编排可执行闭环。

`train-evaluate-export-register` 模板必须按顺序执行：

1. validate dataset
2. create dataset snapshot
3. start training
4. evaluate model
5. export ONNX
6. register model version
7. generate delivery report

`export-infer-benchmark-report` 模板必须按顺序执行：

1. export or reuse model artifact
2. run inference smoke
3. benchmark model
4. generate delivery report

官方 Python backend 的流水线要求：

- 复用现有 Python trainer adapter，不在 GUI 进程内执行训练。
- 每个 step 记录 request、artifact、metrics、状态和失败原因。
- 缺少 Python / Ultralytics / PaddleOCR / PaddlePaddle 环境时，流水线必须明确失败并产出可读诊断。
- 不允许只写 `training_request.json` 就把 official backend step 视为完成。

保留的限制：

- TensorRT step 根据硬件记录 `passed` / `failed` / `hardware-blocked`；RTX 4090 D 已有通过证据，旧 GPU 仍应为 `hardware-blocked`。
- PaddleOCR System 继续作为 official tool inference，不声明为 C++ DB ONNX postprocess。
- C++ tiny/scaffold backend 只用于 diagnostic pipeline。

## 5. Phase 39C：benchmark、模型库与交付报告

目标：让训练产物能被真实判断“能不能部署、如何交付、来自哪里”。

### Benchmark

增强 `benchmarkModel`：

- 支持 ONNX Runtime detection / segmentation / OCR Rec。
- 对已有 TensorRT engine 可做 runtime benchmark；不在 GTX 1060 / SM 61 上伪造 engine build 成功。
- 输出 average latency、P50、P95、P99、throughput、runtime、model family、input shape、sample image、timedInference。
- 失败时写清 runtime 缺失、模型不支持或硬件受限原因。

### Model Registry

模型库增强：

- 注册模型版本时关联 source task、dataset snapshot、checkpoint、ONNX、evaluation report、benchmark report。
- 模型库展示最近一次 evaluation 和 benchmark 摘要。
- 模型版本状态继续使用 draft / validated / exported / accepted / archived 语义。
- scaffold / limitations 标记必须进入模型版本摘要，避免误交付。

### Delivery Report

增强 `generateDeliveryReport`：

- 生成 HTML、JSON、model card、artifact inventory。
- 报告包含项目、数据集、snapshot hash、训练 backend、模型 preset、epochs、环境摘要、评估指标、错误样本摘要、benchmark 摘要、导出产物、ONNX sidecar、模型版本信息。
- 报告必须明确列出 scaffold / official / hardware-blocked 限制。
- 报告必须作为 task artifact 记录，并能从任务与产物页预览或打开目录。

## 6. Phase 41 Lite：环境 profile 与修复向导

目标：降低 Python、CUDA、Paddle、Torch、TensorRT 环境失败率，让用户知道该修哪里。

环境页增加三个 profile：

- YOLO profile：Python executable、Ultralytics、Torch、ONNX、ONNX Runtime。
- OCR profile：isolated OCR Python、PaddlePaddle、PaddleOCR、PaddleOCR source checkout、official smoke script readiness。
- TensorRT profile：NVIDIA driver、CUDA、cuDNN、TensorRT DLLs、GPU compute capability、SM 75+ acceptance status。

实现原则：

- 只做诊断和建议命令，不自动修改用户全局环境。
- 可以引导用户使用 `.deps` 隔离环境。
- Torch / Paddle 冲突必须可见。
- TensorRT 在 GTX 1060 / SM 61 上应显示 `hardware-blocked`，不是失败或通过。

## 7. Phase 49 Lite：交付闭环工作台

目标：把本地 YOLO/OCR 能力变成“能验收、能诊断、能交付”的 GUI 闭环，而不新增算法后端。

已落地能力：

- `样本复核`：读取 `problem_samples.json`、`error_samples.json`、`rework_sample_set.json`、`evaluation_report.json`，按来源、问题类型、类别、split、OCR edit distance / CER、搜索文本过滤，并导出 X-AnyLabeling 复核清单。
- `交付验收`：汇总本机 RC、clean Windows、TensorRT、客户域 OCR、包体完整性、诊断包和部署验证状态，显示 `passed` / `blocked` / `failed` / `hardware-blocked`。
- 客户域 OCR 验收：通过 Worker/core 生成客户 OCR manifest 和 summary；public/generated/smoke 数据只能作为流程 evidence，不能作为生产 OCR 精度证明。
- 诊断包：收集 Worker self-check、环境 profile、GPU/runtime、最近任务日志、失败 request、artifact index、插件状态和授权摘要。
- 导出后验证：ONNX 要可推理；TensorRT 区分 `passed` / `failed` / `hardware-blocked`；NCNN 在配置 SDK/runtime 和样本图时执行 YOLO 检测/分割 runtime inference。2026-05-16 本机证据已覆盖 Hyuto YOLOv8 detection ONNX -> NCNN 和 nihui 预转换 YOLOv8n-seg pnnx/DFL NCNN；YOLOv8-seg ONNX 若残留 unsupported `Shape` layer，则记录 failed report。

保留限制：

- 不自动修改用户全局 Python/CUDA 环境。
- 不把 X-AnyLabeling 嵌入 GUI。
- 不新增 SQLite schema 或插件接口语义。
- 不用 Total-Text 或 generated smoke 宣称客户域 OCR production ready。

## 8. 暂缓事项

- 暂不优先新增图像分类、姿态、OBB、异常检测训练后端。
- 暂不把 clean Windows package acceptance 或 package-root TensorRT rerun 标记为通过，除非收到外部证据。
- NCNN runtime validation 覆盖 YOLO 检测/分割；外部 NCNN 模型需要 sidecar 或显式 blob/decoder 配置。当前不要把失败的 YOLOv8-seg ONNX -> `onnx2ncnn` `Shape` layer case 说成通过证据。
- 暂不把 X-AnyLabeling 嵌入 GUI。
- 暂不把 Python 训练嵌入 GUI 主进程。
- 暂不把 C++ scaffold 宣称为真实 YOLO/OCR 训练能力。
- 暂不优先做云平台、Kubernetes 调度、多人权限和 Web 控制台。

## 9. 验收与测试计划

每个实现阶段必须运行：

```powershell
.\tools\harness-check.ps1
git diff --check
```

文档-only 更新至少运行：

```powershell
git diff --check
.\tools\harness-context.ps1
```

建议 smoke：

```powershell
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild
```

手工 GUI 验收：

- 导入 generated detection / segmentation / OCR Rec 数据集。
- 运行质检、snapshot、训练、评估、benchmark、报告、样本复核、部署验证和诊断包。
- 在任务与产物页检查 JSON / CSV / image / HTML artifact。
- 在模型库检查 evaluation / benchmark / pipeline 记录。
- 在交付验收页确认客户 OCR gate、TensorRT `hardware-blocked` 和 NCNN SDK/runtime/sample-image 要求文案。
- 确认 scaffold 和 TensorRT hardware-blocked 文案清晰。

## 10. 参考平台

- [ClearML](https://github.com/clearml/clearml)：实验管理、数据管理、Pipeline、调度、模型与 artifact 追踪。
- [MLflow](https://github.com/mlflow/mlflow)：实验追踪、模型注册表、模型 lineage、版本和阶段管理。
- [FiftyOne](https://github.com/voxel51/fiftyone)：数据集可视化、数据清洗、模型分析、样本筛选。
- [CVAT](https://docs.cvat.ai/docs/getting_started/overview/) / [Label Studio](https://github.com/HumanSignal/label-studio)：标注、导入导出、QA、自动化标注闭环。
- [DVC](https://dvc.org/doc/user-guide/what-is-dvc)：数据版本、实验复现、pipeline 元数据。
- [Kubeflow Pipelines](https://github.com/kubeflow/pipelines)：端到端 ML 工作流编排。
- [Ultralytics](https://github.com/ultralytics/ultralytics) / [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：现有官方训练后端继续作为主要算法来源。
