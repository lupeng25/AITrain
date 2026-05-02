# AITrain Studio 后续功能规划与实现方案（本地交付版）

日期：2026-05-02

定位：Windows + Qt Widgets + Worker 的本地视觉训练平台。

## 1. 参考平台

- [ClearML](https://github.com/clearml/clearml)：实验管理、数据管理、Pipeline、调度、模型与 artifact 追踪。
- [MLflow](https://github.com/mlflow/mlflow)：实验追踪、模型注册表、模型 lineage、版本和阶段管理。
- [FiftyOne](https://github.com/voxel51/fiftyone)：数据集可视化、数据清洗、模型分析、样本筛选。
- [CVAT](https://docs.cvat.ai/docs/getting_started/overview/) / [Label Studio](https://github.com/HumanSignal/label-studio)：标注、导入导出、QA、自动化标注闭环。
- [DVC](https://dvc.org/doc/user-guide/what-is-dvc)：数据版本、实验复现、pipeline 元数据。
- [Kubeflow Pipelines](https://github.com/kubeflow/pipelines)：端到端 ML 工作流编排。
- [Ultralytics](https://github.com/ultralytics/ultralytics) / [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：现有官方训练后端继续作为主要算法来源。

## 2. 当前基线

AITrain Studio 已完成 Worker、SQLite、任务记录、artifact 浏览、YOLO 检测/分割、PaddleOCR Det/Rec/System 链路、ONNX Runtime 推理、打包、离线授权和 CPU smoke。

后续重点不是继续堆 demo 后端，而是补齐“数据 -> 实验 -> 评估 -> 模型注册 -> 导出部署 -> 复现报告”的产品闭环。

架构约束保持不变：

- GUI 只做交互、调度和展示。
- 长任务继续进入 `aitrain_worker`。
- 元数据通过 `ProjectRepository` 写入 SQLite。
- 训练、推理、评估、导出逻辑不进入 `MainWindow`。
- scaffold 后端必须继续标注为占位或诊断能力。

## 3. 功能实现方案

### 3.1 实验管理与对比

目标：把当前“任务历史”升级为可比较的训练实验系统。

实现：

- 新增 SQLite 表：`experiments`、`experiment_runs`。
- `experiments` 保存实验组：名称、任务类型、目标数据集、备注、标签。
- `experiment_runs` 关联已有 `tasks`：训练后端、模型预设、request JSON、环境摘要、最佳指标、主要 artifact。
- Worker 训练完成后继续发出 `metric` / `artifact`，GUI 或 Repository 将其归档到 run summary。
- GUI 在“训练实验”页增加实验列表、run 表格、指标曲线对比、参数差异视图。
- 支持按任务类型、后端、状态、数据集版本、指标排序筛选。

### 3.2 模型注册表

目标：把训练产物从普通 artifact 升级为可管理的模型版本。

实现：

- 新增 SQLite 表：`model_versions`。
- 字段包含：模型名、版本号、来源任务、来源实验 run、数据集 snapshot、checkpoint、ONNX、TensorRT engine、评估报告、状态、备注。
- 状态建议：`draft`、`validated`、`exported`、`accepted`、`archived`。
- 在任务产物页增加“注册为模型版本”动作。
- 在“模型交付”分组下新增“模型库”页面，展示模型版本、指标、导出格式和来源 lineage。
- 导出、推理页面优先从模型库选择模型，而不是只手动选路径。

### 3.3 模型评估与误差分析

目标：补齐正式训练平台最关键的质量判断能力。

实现：

- Worker 新增命令：`evaluateModel`。
- 输入：模型路径、任务类型、数据集路径或 dataset snapshot、阈值配置、输出目录。
- 检测任务输出：precision、recall、mAP、per-class 指标、confusion matrix、误检样本、漏检样本、低置信样本。
- 分割任务输出：mask IoU、mask mAP、per-class mask 指标、overlay 对比图。
- OCR Rec 输出：accuracy、edit distance、CER/WER、错误字符统计、低置信样本。
- 结果写入 `evaluation_reports` 表，并作为 task artifact 记录 JSON、CSV、PNG。
- GUI 增加评估报告页：总览指标、类别指标、错误样本列表、overlay 预览。

### 3.4 数据质量检查与数据闭环

目标：让平台不仅能训练，还能告诉用户数据哪里有问题。

实现：

- Worker 新增命令：`curateDataset`。
- 在现有 dataset validation 基础上扩展检查项：坏图、空图、无法读取图、label 越界、类别 id 越界、polygon 点数不足、空标注、重复图片、重复标签、类别分布、尺寸分布、train/val/test 分布差异。
- 输出 `dataset_quality_report.json`、`class_distribution.csv`、`problem_samples.json`。
- GUI 数据集页增加“质量报告”和“问题样本”区域。
- 支持将问题样本导出为 X-AnyLabeling 待修复清单。
- 标注后刷新仍走现有校验/重新入库流程。

### 3.5 数据集快照与复现实验

目标：保证“同一个实验能被重新跑出来”。

实现：

- 新增 SQLite 表：`dataset_snapshots`。
- Worker 新增命令：`createDatasetSnapshot`。
- 扫描数据集文件，生成 manifest：图片/标签相对路径、文件大小、mtime、hash、split 信息，以及 `data.yaml` / `dict.txt` / `rec_gt.txt` 等入口文件 hash。
- 训练 request 中记录 `datasetSnapshotId`。
- 实验 run 记录 Python 包版本、后端版本、模型预设、随机种子、训练参数。
- 增加“复现实验”按钮：复制原 request，默认使用同一 snapshot 和参数重新提交 Worker。

### 3.6 本地流水线模板

目标：一键执行常见闭环，减少用户手动串步骤。

实现：

- 新增 SQLite 表：`pipeline_runs`。
- Worker 新增命令：`runLocalPipeline`。
- v1 固定提供模板：
  - 数据集校验 -> 数据集快照 -> 训练 -> 评估 -> ONNX 导出 -> 模型注册。
  - 模型导出 -> 推理 smoke -> 性能基准 -> 交付报告。
- Pipeline 内部仍复用现有 Worker 命令，不重复实现训练逻辑。
- 每个 step 写入独立 task，`pipeline_runs` 保存 step task ids 和状态。
- GUI 新增“流水线”入口或放入“训练实验”页高级区域。

### 3.7 部署基准与模型优化

目标：让用户知道模型能不能真正部署。

实现：

- Worker 新增命令：`benchmarkModel`。
- 支持 ONNX Runtime CPU/GPU、TensorRT engine 路径。
- 输出：平均延迟、P50/P95/P99、吞吐、输入尺寸、batch、运行设备、runtime 版本。
- TensorRT 继续保持 RTX / SM 75+ 外部验收规则。
- 后续增加 INT8/PTQ：用户选择 calibration dataset，Worker 生成校准 manifest，TensorRT 路径使用校准数据构建 engine。
- GUI 模型库显示每个模型版本的 benchmark 结果。

### 3.8 训练后端扩展

目标：在现有 YOLO/PaddleOCR 基础上扩展任务类型，但不破坏平台边界。

实现顺序：

1. 图像分类：优先接 Ultralytics classify。
2. 姿态估计：优先接 Ultralytics pose。
3. 旋转框 OBB：优先接 Ultralytics OBB。
4. 异常检测：评估 Anomalib 或轻量本地后端。
5. OpenMMLab/MMYOLO：作为高级配置导入，不替代现有官方后端。

每个后端必须走 Python trainer adapter，并提供 generated dataset、1 epoch tiny smoke、artifact/report 归一化和 scaffold/official 能力说明。

### 3.9 环境管理增强

目标：降低 Python、CUDA、Paddle、Torch 环境失败率。

实现：

- 扩展 `environmentCheck` 输出：Python executable、pip 包版本、Torch/Paddle 冲突、CUDA/cuDNN/TensorRT/ONNX Runtime 路径、GPU compute capability。
- 增加环境 profile：YOLO 环境、OCR 隔离环境、TensorRT 部署环境。
- GUI 环境页显示 profile 状态、缺失依赖、建议命令。
- 不自动修改用户全局环境；只生成可复制的修复脚本或引导到 `.deps` 隔离环境。

### 3.10 报告导出与交付包

目标：让一次训练结果可以直接交付给客户或内部验收。

实现：

- 新增 report generator，优先输出 HTML，后续可导出 PDF。
- 报告内容：项目和数据集摘要、数据质量报告、训练参数和环境、指标曲线和最终指标、评估误差样本、模型版本和导出格式、benchmark 结果、scaffold/限制说明。
- GUI 在模型库和实验页提供“生成报告”。
- 报告作为 artifact 记录。

## 4. 推荐实施阶段

- Phase 34：实验管理与模型注册表。
- Phase 35：数据集快照和复现实验。
- Phase 36：模型评估与误差分析。
- Phase 37：数据质量检查和问题样本闭环。
- Phase 38：本地流水线模板（已完成本地执行闭环）。
- Phase 39：部署 benchmark 和 TensorRT 外部验收增强。
- Phase 40：图像分类/姿态/OBB 等训练后端扩展。
- Phase 41：环境 profile 和修复向导。
- Phase 42：HTML/PDF 报告与交付包增强。

## 5. 测试计划

- Repository 测试：新增表初始化、迁移、插入、查询、旧项目兼容。
- Worker 测试：`evaluateModel`、`benchmarkModel`、`curateDataset`、`createDatasetSnapshot`、`runLocalPipeline` 的成功和失败路径。
- Python adapter 测试：fake backend 保持确定性，真实后端用 tiny smoke。
- GUI 手工验收：数据集、训练实验、任务产物、模型库、评估报告、环境页完整走一遍。
- 标准命令：继续运行 `.\tools\harness-check.ps1`。
- 发布验收：继续运行 package smoke、CPU smoke；TensorRT 只在 RTX / SM 75+ 机器记录通过。

## 6. 明确暂缓

- 暂不优先做云平台、Kubernetes 调度、多人权限和 Web 控制台。
- 暂不把 X-AnyLabeling 嵌入 GUI。
- 暂不把 Python 训练嵌入主进程。
- 暂不把 C++ scaffold 宣称为真实 YOLO/OCR 训练能力。
