# AITrain Studio 后续功能规划与实现方案（本地交付版）

日期：2026-06-21

定位：Windows + Qt Widgets + Worker 的本地视觉训练平台。

> 2026-07-16 破坏性重构覆盖：本文早期 Phase/脚本描述仅作历史证据。当前实现以 SQLite schema 11、V2 Storage/Artifact/Workflow、Protocol V2 和官方适配器为准；旧 Repository、ProductWorkflow、裸路径 Worker 命令及独立 smoke 脚本均已删除。

本文是 Phase 39+ 的当前方向文档。当前权威状态仍以 `docs/harness/current-status.md` 为准。已删除或外部保留的历史路线笔记不作为下一步实施、阶段状态或验收口径来源。

## 1. 当前基线与判断

AITrain Studio 已完成 Worker、SQLite、任务记录、artifact 浏览、YOLO 检测/分割官方训练链路、PaddleOCR Det/Rec/System 官方工具链入口、ONNX Runtime 推理、RTX 4090 TensorRT 验收证据、打包、离线授权、CPU smoke、内置能力注册表、环境 profile、样本复核、部署验证、诊断包、客户域 OCR 验收向导和成熟 Qt Widgets workbench。

维护基线：UI 与 core 的第一层源文件拆分已经完成。`MainWindow` 和 V2 workflow service 已拆为职责明确的 companion 文件；旧 `ProductWorkflow` 实现和兼容门面已删除，snapshot、quality、evaluation、benchmark、delivery、acceptance、pipeline 逻辑由对应的 V2 workspace service 负责。这是当前破坏性架构，不应再描述为 V1/V2 兼容层。

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
4. 下一阶段工业模型扩展只覆盖异常检测/定位、OBB、专用语义分割；专用语义分割首版已选择 SMP 落地，OBB v1 已通过本地 smoke/质量矩阵证据，异常检测 v1 已选择 Anomalib PatchCore + EfficientAD 并通过本地 public MVTec 默认三类质量矩阵。

Phase 40 backlog 的优先级已重新确认：异常检测/定位、OBB、专用语义分割作为下一阶段开发方向；其中专用语义分割已先落地 SMP 首版闭环，并通过本机 RTX 4090D GPU realtest；OBB v1 走官方 Ultralytics OBB + AITrain ONNX Runtime 旋转框部署路径，并在 2026-06-17 通过 `tools\phase-obb-ultralytics-smoke.ps1` 与 `tools\phase-obb-dota-quality-matrix.ps1` 本地验证；异常检测 v1 走 `taskType=anomaly_detection`、`datasetFormat=anomaly_folder`、`trainingBackend=anomalib_patchcore|anomalib_efficientad`、`runtime=anomalib_python`，不引入 C++ ONNX/TensorRT/NCNN anomaly runtime，并在 2026-06-18 通过 public MVTec 默认三类矩阵 6/6。OBB 现有证据是 public DOTA/workflow evidence，异常检测现有证据是 public MVTec workflow/quality evidence，二者都不能替代客户域工业精度证明。图像分类、姿态/关键点、YOLO-World、YOLOE、3D/RGB-D、视频/时序以及云/多人协作不属于当前项目方向，除非新的范围决策明确加入。

下一阶段目标不是堆 demo 后端，而是把工业检测常见任务纳入现有闭环：

- 异常检测/定位：首版已选 Anomalib PatchCore + EfficientAD。PatchCore 是默认后端；EfficientAD 需要显式 `imagenetDir`、`AITRAIN_ANOMALIB_IMAGENET_DIR` 或 `.deps/anomalib/imagenette`，Anomalib 2.5 训练 batchSize 固定为 1，缺失外部数据时 blocked，不自动下载。v1 支持仅良品训练、异常分数、热力图、阈值、OK/NG、mask/overlay、评估和交付报告，但运行边界固定为 Worker-managed Python/Anomalib artifacts；当前已有 public MVTec 默认三类矩阵证据，不等同客户域工业精度。
- OBB：基于官方 Ultralytics OBB 能力，旋转框数据集校验、训练、ONNX 导出、官方 `val()` 评估、AITrain C++ ONNX Runtime 旋转框推理/overlay/benchmark/部署验证、GUI 入口和脚本闭环已形成本地证据；NCNN 不纳入 OBB v1，TensorRT engine export 只记录可选状态，不作为必过项，TensorRT runtime OBB inference 不宣称为通过能力。不要把普通 YOLO bbox 结果伪装成 OBB，也不要把 public DOTA/workflow evidence 说成客户域工业精度证明。
- 专用语义分割：面向像素级工业缺陷/区域分割，首版使用 `segmentation_models.pytorch`，与现有 YOLO 实例分割区分，输出 per-pixel mask、面积/类别像素统计、overlay、评估和部署限制；当前已有 RTX 4090D GPU realtest 证据，Mask2Former 等路线仍可作为后续扩展候选。

2026-06-14/15 全量模型生命周期运行中的新增边界：共享 Ultralytics 8.3.171 环境下 YOLO26 检测/实例分割 20 行全部失败，原因是官方模型配置/权重不可用或包代码不兼容。随后形成的隔离 YOLO26 matrix 结果仅作为历史证据保留；原脚本已删除且不可重跑。YOLO26 NCNN 历史尝试 20/20 failed，当前产品不提供 YOLO26 NCNN 导出/转换。

## 2. 架构约束

- GUI 只做交互、调度和展示。
- 长任务继续进入 `aitrain_worker`。
- 元数据通过 `ProjectStore` 写入 SQLite；GUI 只经 Query Service/Presenter 读取，业务写入经 `ProjectWorkspace`/Worker 收口。
- 训练、推理、评估、导出逻辑不进入 `MainWindow`。
- 官方训练优先通过 Worker 管理的 Python trainer subprocess。
- 旧的 C++ tiny detector、segmentation baseline、OCR baseline、small OCR CTC 和 shipped Python mock 已物理删除，不能作为产品训练 backend。
- TensorRT 真机验收已有 RTX 4090 D 证据；当前/旧 GTX 1060 / SM 61 仍只记录为 `hardware-blocked`。
- PaddleOCR System 当前是官方 `predict_system.py` 工具链路径；OCR 训练、导出、推理、评估和验收收口为 PaddleOCR 官方-only。历史 C++ Det ONNX wiring 不作为产品路线或验收要求。

## 3. Phase 39A：真实评估补齐

目标：让 YOLO detection / segmentation / OBB 和 OCR 都能生成可信的质量判断报告；YOLO detection/segmentation/OBB 指标由 Ultralytics 官方 `val()` 产生，AITrain 不再自行计算本地 AP/mAP、rotated AP 或 mask IoU。OCR 评估和验收收口为 PaddleOCR 官方 Det/Rec/System 报告和客户域验收证据。

### Detection

使用 Ultralytics 官方 `YOLO(...).val()` 作为唯一评估路径：

- 输出 AITrain wrapper `evaluation_report.json`、官方 metrics、官方 run dir、官方 plots/predictions。
- 不再输出本地 per-class CSV、error samples、confusion CSV 或本地 overlay。
- ONNX Runtime / TensorRT / NCNN 继续用于推理、benchmark 和部署验证，不作为评估指标来源。

### Segmentation

使用 Ultralytics 官方 `YOLO(...).val()` 作为唯一 segmentation evaluation：

- 输出 AITrain wrapper `evaluation_report.json`、官方 box/mask metrics、官方 run dir、官方 plots/predictions。
- AITrain 不再在 C++ 中执行 GT polygon -> mask、预测 mask IoU、本地 mask AP/mAP、error samples 或本地 overlay 评估。
- 报告 `scaffold=false`，但 limitations 说明指标来源为 Ultralytics official val，客户/目标域验收仍需配合使用。

### OCR

OCR 评估和验收使用 PaddleOCR 官方报告路径：

- Det / Rec / System 证据来自 PaddleOCR 官方 adapter、official report 和 `predict_system.py` 输出。
- AITrain 不再把历史 C++ OCR ONNX wiring 当作产品 OCR 评估、benchmark、部署验证或验收路线。
- 客户域 OCR production claim 必须使用客户/目标域数据、官方 Det/Rec/System 报告和 `customer-ocr-validation` 结果。
- Public Total-Text、generated smoke、`.deps` 示例和 tiny CPU smoke 只能证明工作流，不证明客户域 OCR 精度。

## 4. Phase 39B：历史本地流水线（已删除）

原 `runLocalPipeline` 同时维护另一套训练、评估、导出和报告编排，与 V2 Workflow 形成双事实源，现已从产品代码、协议、GUI 和测试中物理删除。

当前方向统一为：

1. 生产训练通过训练工作流 Profile 执行八步工作流。
2. 运行时交付迁移到 `ImportOrResolveModel → ValidateManifest → RunInferenceSmoke → Benchmark → DeploymentValidate → RenderDeliveryReport`。
3. 每个步骤只消费已提交 Artifact，并持久化状态、指标、失败和 Evidence。
4. 缺少 SDK、依赖、硬件或 decoder 时使用 Runtime V2 精确状态，不再使用笼统 `hardware-blocked` 字符串。
5. PaddleOCR System 保持官方 Det+Rec 组合工具链，不声明为 C++ OCR ONNX 路线。

## 5. Phase 39C：benchmark、模型库与交付报告

目标：让训练产物能被真实判断“能不能部署、如何交付、来自哪里”。

### Benchmark

将 benchmark 收口到 V2 运行时交付 Workflow 的 `Benchmark` 步骤；旧 `benchmarkModel` 裸路径 Worker 命令已删除：

- 支持 ONNX Runtime detection / segmentation；OCR 结果通过 PaddleOCR 官方报告和任务产物汇总。
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
- 报告必须作为 committed task artifact 记录，并能从任务与产物页按 ArtifactId 和包内相对路径受控预览；GUI 不暴露或打开 Artifact Store 物理目录。

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

目标：把本地 YOLO/OBB/SMP/anomaly/OCR 能力变成“能验收、能诊断、能交付”的 GUI 闭环。后续算法扩展已单独收敛到工业模型方向：OBB、专用语义分割和异常检测/定位。

已落地能力：

- `数据集 > 质量与复核`：输入已提交的复核 ArtifactId，由 Query Service 在通过 inventory/hash 复验的 Artifact 包内按相对成员名读取 `problem_samples.json`、`error_samples.json`、`rework_sample_set.json`、`evaluation_report.json`；按来源、问题类型、类别、split、OCR edit distance / CER、搜索文本过滤，并导出 X-AnyLabeling 复核清单。GUI 不接受任意本地 JSON 路径，也不打开 staging 文件。
- `环境 > 交付证据`：汇总本机 RC、clean Windows、TensorRT、客户域 OCR、包体完整性、诊断包和部署验证状态，显示 `passed` / `blocked` / `failed` / `hardware-blocked`。
- 客户域 OCR 验收：通过 Worker/core 生成客户 OCR manifest 和 summary；public/generated/smoke 数据只能作为流程 evidence，不能作为生产 OCR 精度证明。
- 诊断包：收集 Worker self-check、环境 profile、GPU/runtime、最近任务日志、失败 request、artifact index、内置能力状态和授权摘要。
- 导出后验证：ONNX 要可推理；TensorRT 区分 `passed` / `failed` / `hardware-blocked`；NCNN 在配置 SDK/runtime 和样本图时执行 YOLO 检测/分割 runtime inference。2026-05-16 本机证据已覆盖 Hyuto YOLOv8 detection ONNX -> NCNN 和 nihui 预转换 YOLOv8n-seg pnnx/DFL NCNN；YOLOv8-seg ONNX 若残留 unsupported `Shape` layer，则记录 failed report。

保留限制：

- 不自动修改用户全局 Python/CUDA 环境。
- 不把 X-AnyLabeling 嵌入 GUI。
- 不新增 SQLite schema 或内置能力注册表语义。
- 不用 Total-Text 或 generated smoke 宣称客户域 OCR production ready。

## 8. 下一阶段：工业模型能力扩展

下一阶段开发方向：

- 异常检测/定位：工业 anomaly detection / localization 工作流 v1 已接入 Worker-managed Python adapter，不在 GUI 进程内训练。首批能力覆盖 `anomaly_folder` 数据结构、正常样本训练、异常分数、热力图、阈值选择、OK/NG 分类、mask/overlay 产物、评估报告、benchmark 和交付限制；2026-06-18 默认 public MVTec 三类矩阵已通过 6/6，后续重点是目标域数据指标和更完整的客户域像素级评估证据。
- OBB：旋转框数据集、训练、评估、导出和结果展示闭环已通过本地 smoke/质量矩阵验证。首选官方 Ultralytics OBB 路径；报告必须记录官方来源、模型 preset、输入格式、评估指标、ONNX Runtime 部署证据、TensorRT 可选状态、NCNN v1 不支持边界和失败原因。客户域工业可用性仍需目标域数据验证。
- 专用语义分割：新增区别于 YOLO 实例分割的 semantic segmentation 路线，用于裂纹、污渍、涂层、焊缝、气孔等像素级区域检测。首批实现已选择 SMP，覆盖 Mask PNG 校验/划分、SMP 训练、评估、ONNX 导出、ONNX Runtime 推理/overlay、benchmark、GUI 入口和 package smoke；NCNN/TensorRT 导出不属于 SMP 能力范围，也不是 SMP 验收要求。

实现约束：

- 沿用 Worker JSON request / artifact / metric / report 模式；必要协议扩展必须同步测试。
- 训练、推理、评估、导出、benchmark 和报告逻辑继续放在 core/CapabilityRegistry/Worker 边界内，不进入 `MainWindow`。
- GUI 只添加入口、状态、日志、预览和复用现有任务/产物/模型库/部署验证页面。
- 每项新能力必须从 dataset validator、最小 smoke 数据、adapter request 示例、失败诊断、交付报告限制和 harness 测试一起落地。
- 未经真实 smoke / evaluation / deployment evidence，不得声明生产可用。

## 9. 暂缓事项

- 图像分类、姿态/关键点、YOLO-World、YOLOE、3D/RGB-D、视频/时序训练后端不属于当前项目方向。
- 暂不把 clean Windows package acceptance 或 package-root TensorRT rerun 标记为通过，除非收到外部证据。
- NCNN runtime validation 覆盖 YOLO 检测/分割；外部 NCNN 模型需要 sidecar 或显式 blob/decoder 配置。当前不要把失败的 YOLOv8-seg ONNX -> `onnx2ncnn` `Shape` layer case 说成通过证据。
- 暂不把 X-AnyLabeling 嵌入 GUI。
- 暂不把 Python 训练嵌入 GUI 主进程。
- 暂不重新引入 C++ scaffold 训练能力；真实 YOLO/OBB/OCR 训练只走官方后端，SMP 和 Anomalib 只走 Worker-managed 上游 Python 适配器。
- 云平台、Kubernetes 调度、多人权限和 Web 控制台不属于当前项目方向。

## 10. 验收与测试计划

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
- 运行质检、snapshot、训练、评估、benchmark、报告、`数据集 > 质量与复核`、`部署验证` 和诊断包。
- 在任务与产物页检查 JSON / CSV / image / HTML artifact。
- 在模型库检查已登记 Model Package；在任务与产物页检查评估、benchmark、workflow step、Metric 与 committed Artifact 记录。
- 在 `环境 > 交付证据` 确认客户 OCR gate、TensorRT `hardware-blocked` 和 NCNN SDK/runtime/sample-image 要求文案。
- 确认 scaffold 和 TensorRT hardware-blocked 文案清晰。

## 10. 参考平台

- [ClearML](https://github.com/clearml/clearml)：实验管理、数据管理、Pipeline、调度、模型与 artifact 追踪。
- [MLflow](https://github.com/mlflow/mlflow)：实验追踪、模型注册表、模型 lineage、版本和阶段管理。
- [FiftyOne](https://github.com/voxel51/fiftyone)：数据集可视化、数据清洗、模型分析、样本筛选。
- [CVAT](https://docs.cvat.ai/docs/getting_started/overview/) / [Label Studio](https://github.com/HumanSignal/label-studio)：标注、导入导出、QA、自动化标注闭环。
- [DVC](https://dvc.org/doc/user-guide/what-is-dvc)：数据版本、实验复现、pipeline 元数据。
- [Kubeflow Pipelines](https://github.com/kubeflow/pipelines)：端到端 ML 工作流编排。
- [Ultralytics](https://github.com/ultralytics/ultralytics) / [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：现有官方训练后端继续作为主要算法来源。
