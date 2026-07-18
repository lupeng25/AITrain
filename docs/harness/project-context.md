# Project Context

## 项目定位

AITrain Studio 是一个 Windows + NVIDIA GPU 本地视觉训练平台。当前技术方向：

- C++20
- Qt Widgets
- CMake
- SQLite
- 独立 Worker 进程
- 编译期内置能力注册表与 Worker 官方适配器

目标能力：

- 训练 YOLO 风格检测模型。
- 训练 YOLO 风格分割模型。
- 训练 YOLO OBB 旋转框检测模型。
- 训练 Anomalib PatchCore / EfficientAD 异常检测模型。
- 训练 PaddleOCR Det / Rec 官方模型，并通过 PaddleOCR System 做端到端 OCR 推理验收。
- 管理数据集。
- 转换模型格式并做部署验证。
- 做推理验证。
- 做数据集质量复核、评估、模型库、交付报告、部署验证、诊断包和客户域 OCR 验收。

训练实现方向：

- 真实模型训练优先通过 Worker 启动独立 Python 子进程实现。
- 若官方 Python 实现可用，优先适配官方实现，而不是自研训练框架。
- 检测、分割和 OBB 优先评估 Ultralytics YOLO；异常检测优先评估 Anomalib PatchCore / EfficientAD；OCR Det / Rec / System 优先评估 PaddleOCR / PaddlePaddle 官方工具链。
- 官方后端必须显式记录来源、版本和许可证约束。
- C++ 侧继续负责 GUI、Worker 编排、数据集校验、SQLite、ONNX Runtime/TensorRT 推理、打包和部署。
- YOLO 检测/分割/OBB 的产品边界是“官方 Ultralytics 训练/ONNX 导出/`val()` 评估 + AITrain C++ runtime 推理/benchmark/部署验证”，不是 OCR 那种端到端 official-only。OBB v1 的部署路径是 ONNX Runtime；NCNN 不纳入 OBB v1。
- 异常检测 v1 的产品边界是 Worker-managed Python/Anomalib artifact runtime：`anomaly_sidecar.json`、checkpoint、`inference_predictions.json`、heatmap、overlay 和 binary mask；不声明 AITrain C++ ONNX Runtime、TensorRT 或 NCNN 异常检测 runtime。
- 不把 Python 嵌入 `MainWindow` 或 GUI 进程。
- 旧的 C++ tiny/scaffold 训练实现已物理删除；生产训练只通过官方后端。

## 当前事实

阶段状态以 `docs/harness/current-status.md` 为准。不要只根据长期路线图末尾的历史“下一步建议”判断当前阶段。

当前样本复核和评估报告统一以已提交 ArtifactId 为入口，GUI 只读取通过 inventory/hash 复验的 Artifact 包内相对成员，不暴露任意本地路径或 staging 文件。数据集页不提供任意“打开数据目录”按钮；外部目录只作为显式导入/转换边界输入，标注修复通过 Session/Artifact 身份闭环。

截至 2026-06-18，官方 YOLO 检测/分割/OBB 训练与 ONNX 导出、Anomalib PatchCore/EfficientAD 异常检测 Python 路由、PaddleOCR Det/Rec/System 官方工具链、PP-OCRv4/v5/v6 official adapter preset、YOLO C++ ONNX Runtime 推理、RTX 4090D TensorRT 验收证据、本地插件 marketplace、环境 profile、数据集质量/快照、数据集格式转换 GUI、评估、benchmark、模型库、交付报告、样本复核、部署验证、诊断包和环境页交付证据 GUI 闭环都已落地到当前本地代码。当前主导航收敛为 9 个对象型工作区：`总览`、`项目`、`数据集`、`训练实验`、`任务与产物`、`模型库`、`部署验证`、`环境`、`系统设置`；样本复核、评估报告、模型导出、推理验证、插件和应用设置作为对应工作区 tab 呈现。2026-06-05 RTX 4090D follow-up 刷新记录 local baseline/package acceptance、GUI walkthrough、历史 Phase 47 Det ONNX+CTest、CPU training smoke、Phase 45 YOLO matrix、TensorRT 和 public OCR GPU workflow 均有通过证据。2026-06-17 OBB v1 通过 `tools\phase-obb-ultralytics-smoke.ps1` 和 `tools\phase-obb-dota-quality-matrix.ps1` 本地验证，覆盖官方 Ultralytics OBB 训练/ONNX/`val()`、AITrain ONNX Runtime 旋转框推理、benchmark 和部署验证；该证据是 public DOTA/workflow evidence，不是客户域工业精度证明。2026-06-18 异常检测 v1 默认 public MVTec 三类矩阵通过 6/6，覆盖 `bottle/hazelnut/leather` x `anomalib_patchcore/anomalib_efficientad`，summary 为 `.deps\anomaly-mvtec-quality-matrix\anomaly_mvtec_quality_matrix_summary.json`；该证据是 public MVTec workflow/quality evidence，不是客户域工业精度证明。准确阶段边界仍以 `docs/harness/current-status.md` 为准。

2026-06-14/15 全量模型生命周期验证暴露了当前软件影响项：Ultralytics 8.3.171 无法解析 YOLO12 分割 `.pt` 权重 `yolo12n-seg.pt`，这类行应记录为上游权重 blocker；共享 Ultralytics 8.3.171 环境下 YOLO26 20 行全部失败，原因是 `.yaml` 配置不存在、官方权重不可解析或 nano `.pt` 权重与包代码不兼容；进度页面可能把历史 `row_summary.json` 失败计入当前 failed 数；GPU YOLO 训练必须使用 CUDA PyTorch 环境。随后隔离 YOLO26 targeted matrix 在 2026-06-15 取得 `-Full -Epochs 100` 20/20 训练、官方 ONNX、AITrain C++ ONNX 推理和 TensorRT 通过证据。YOLO26 NCNN 历史尝试 20/20 failed，当前产品移除 YOLO26 NCNN 选项并拒绝 `format=ncnn`。详细边界见 `docs/harness/current-status.md` 和 `docs/yolo-model-support-matrix.md`。

当前架构覆写（2026-07-10）：上段历史记录中的“本地插件 marketplace”、插件 tab 和五个插件骨架已从源码与发布包删除；当前能力矩阵统一来自 `src/core/CapabilityRegistry`，任务请求使用 `capabilityId`，旧 `plugin_id` 仅在 SQLite 打开迁移时读取并随后移除。

已完成：

- `AITrainStudio.exe` Qt GUI。
- `aitrain_worker.exe` 独立任务进程。
- 控制面使用当前版本 JSONL over `QLocalSocket`；Python Adapter 使用认证 loopback 事件通道，stdout/stderr 只作为原始诊断日志。
- SQLite 项目、任务、指标存储。
- 内置能力注册表和官方后端描述。
- 数据集校验初版。
- 数据集转换 GUI 入口：已实现 COCO/VOC/YOLO 转换矩阵的 Worker 编排、表单预检、进度/日志/取消和结果展示；转换产物不自动登记为数据集。
- 官方 Ultralytics YOLO detection 训练、ONNX 导出和 `val()` 评估；ONNX Runtime 推理、overlay、benchmark 和部署验证由 AITrain C++ runtime 执行。
- YOLO 分割官方训练与数据闭环：
  - `SegmentationDataset`
  - `SegmentationDataLoader`
  - polygon-to-mask
  - letterbox 对齐 mask
  - 多 polygon / 多 class mask
  - overlay preview
  - mask preview artifact
  - Worker 端 `maskLoss`、`maskCoverage`、`maskIoU`、`segmentationMap50`
  - official backend artifacts
- OCR Det / Rec / System 官方训练、导出、推理与数据闭环：
  - `paddleocr_det_official` 支持 `PP-OCRv4_mobile_det`、`PP-OCRv5_mobile_det`、`PP-OCRv5_server_det`、`PP-OCRv6_tiny_det`、`PP-OCRv6_small_det`、`PP-OCRv6_medium_det`
  - `paddleocr_rec_official` 支持 `PP-OCRv4_mobile_rec`、`PP-OCRv5_mobile_rec`、`PP-OCRv5_server_rec`、`en_PP-OCRv5_mobile_rec`、`PP-OCRv6_tiny_rec`、`PP-OCRv6_small_rec`、`PP-OCRv6_medium_rec`
  - `paddleocr_system_official` 调用官方 `predict_system.py`，并从 Rec preset/report 推导 `rec_algorithm`
  - `OcrRecDataset`
  - 字符字典加载
  - label encode/decode
  - resize/pad batching
  - Worker 端 `ctcLoss`、`accuracy`、`editDistance`
  - official backend artifacts
  - preview artifact
- VSCode 构建、运行、调试配置。
- QtTest 基础覆盖。
- Worker-managed Python Trainer Adapter；协议测试使用临时 Python trainer fixture，仓库不再提供 shipped `python_mock`。
- 官方 Ultralytics YOLO detection / segmentation / OBB 训练、导出，以及 AITrain C++ ONNX Runtime 推理 smoke；OBB v1 在 2026-06-17 已通过本地 smoke/质量矩阵证据闭环。
- 旧 PaddlePaddle OCR Rec CTC 训练实现已物理删除；生产 OCR 训练和验收主线使用官方 PaddleOCR Det/Rec/System 工具链 smoke。
- OCR 路线已收口为官方-only：训练、导出、推理、评估和客户验收使用 PaddleOCR Det/Rec/System 官方工具链和报告；Phase 46/47 的 C++ OCR ONNX 内容只作为历史 wiring 证据保留。PP-OCRv5/PP-OCRv6 仅覆盖这些官方 OCR preset，不扩展到 PP-StructureV3、PP-ChatOCR、PaddleOCR-VL、文档方向分类、图像矫正、文本行方向分类或 PaddleOCR C++ 本地部署。
- TensorRT SDK-backed ONNX 到 engine 导出路径和 RTX 4090 D 验收证据；旧 GTX 1060 / SM 61 仍应为 `hardware-blocked`。当前 official-artifact smoke 使用官方 Ultralytics ONNX 产物，不再使用已删除的 tiny-detector TensorRT 推理 fixture。
- 2026-06-05 RTX 4090D follow-up 证据已归档到 `docs\validation\rtx4090-validation-evidence-20260615.json`，记录 LocalBaseline+Package、GUI walkthrough、历史 Phase47 Det ONNX+CTest、CPUTrainingSmoke 和 Phase45 的修复后通过证据；原始 `.deps\fix-1-3-cpu-training-smoke-final` 与 `.deps\fix-1-3-phase45-yolo-matrix` 产物目录已清理。OCR GPU 复跑环境通过 `.deps\envs\ocr-gpu` 暴露；旧 `.deps\rtx4090-validation\python-ocr-gpu` 仅作为保留的兼容 target。新 OCR 路线不再把 Phase47 作为产品验收要求。
- Windows 打包、package smoke、release freeze handoff、离线授权和注册码生成器。
- 本地产品闭环：数据集质量报告、问题样本、X-AnyLabeling 复核清单、snapshot、训练 lineage、评估、benchmark、模型注册、pipeline、交付报告。
- Phase 49 交付闭环：`数据集 > 质量与复核`、`环境 > 交付证据`、客户域 OCR 验收向导、一键诊断包和导出后部署验证。
- 下一阶段开发方向已重新确认：补充工业视觉检测中的异常检测/定位、OBB 和专用语义分割能力；其中专用语义分割首版已通过 SMP 路线落地，并已有 RTX 4090D GPU realtest 证据；OBB v1 已接入官方 Ultralytics OBB 和 AITrain ONNX Runtime 旋转框部署路径，并已通过本地 smoke/质量矩阵证据；异常检测/定位 v1 已接入 Anomalib PatchCore/EfficientAD Python runtime，并已通过 public MVTec 默认三类质量矩阵，后续重点是客户/目标域指标证据。

未完成与仍需证据：

- Clean Windows package acceptance 仍需要外部返回证据；不能只凭本机结果标记为通过。
- package-root TensorRT rerun 只有在重新打开外部验收时才执行；旧 GPU 的正确状态仍是 `hardware-blocked`。
- 客户域 OCR 生产声明必须使用真实客户/目标域数据和官方报告；public Total-Text、generated smoke、`.deps` 示例只能证明流程。
- NCNN runtime validation 已替代 artifact-only：有 NCNN SDK/runtime 和样本图时验证 YOLO 检测/分割推理；无 SDK/runtime 时必须明确 failed/blocked。本机 2026-05-16 证据覆盖 Hyuto YOLOv8 detection ONNX -> NCNN 和 nihui 预转换 YOLOv8n-seg pnnx/DFL NCNN；YOLOv8-seg ONNX 若经 `onnx2ncnn` 后仍包含 unsupported `Shape` layer，当前是失败报告而不是通过项。
- 工业异常检测/定位 v1 已有 Python/Anomalib 接入路径和本地 public MVTec workflow/quality evidence，但还没有客户域工业精度证据，也没有 C++ ONNX/TensorRT/NCNN anomaly runtime；OBB v1 已有本地 public DOTA/workflow 质量证据，但客户域工业精度仍需目标域数据验证；专用语义分割已有 SMP 首版闭环和本机 RTX 4090D GPU 实测，但不能把现有 YOLO 检测/实例分割能力描述成语义分割、异常检测或 OBB。

不作为当前项目方向：

- 插件签名 enforcement、远程 marketplace、账号、支付、云调度和多人协作。
- 分类、姿态/关键点、YOLO-World、YOLOE、3D/RGB-D、视频/时序等新算法后端。

## 源码地图

| 路径 | 职责 |
|---|---|
| `src/core` | 协议、内置能力注册表、任务模型、SQLite 仓库、数据集/训练/评估/交付 workflow、product workflow companion files |
| `src/app` | Qt Widgets GUI |
| `src/app/translations` | GUI 翻译源文件，构建时生成 `.qm` 并随应用安装 |
| `src/license_generator` | 内部 Qt 注册码生成器，用私钥签发绑定机器码的离线注册码 |
| `src/worker` | 长任务隔离进程 |
| `src/core/CapabilityRegistry.cpp` | YOLO、Anomalib、PaddleOCR、语义分割和数据集互操作的内置能力注册表 |
| `tests` | QtTest 测试 |
| `.vscode` | VSCode 构建、调试、任务配置 |
| `tools` | Harness 脚本 |
| `docs/harness` | AI 协作和工程护栏文档 |
| `docs/user-guide.md` | 终端用户 GUI 操作手册 |
| `docs/dataset-conversion.md` | 数据集格式转换矩阵、流程和边界 |
| `docs/delivery-evidence-index.md` | RC、RTX、package、OCR、诊断和外部验收证据索引 |
| `docs/operations-runbook.md` | 打包、安装、现场运维和验收命令 runbook |
| `docs/developer-architecture.md` | 架构边界、扩展入口和验证要求 |

## 当前源码组织说明

- `src/app/src/MainWindow.cpp` 已完成第一层 companion 拆分；Qt Widgets shell 仍保持 left sidebar、top status bar、central `QStackedWidget` 架构。
- Core 写入口由 `src/core/src/workflow/ProjectWorkspace.cpp` 及 `ProjectWorkspace*.cpp` 按 snapshot、quality、conversion、split、runtime delivery、OCR acceptance、diagnostics 等职责拆分；读模型集中在 `ProjectQueryService.cpp`，任务执行边界集中在 `TaskExecutionHost.cpp`。已删除的 `ProductWorkflow*` 文件不是当前扩展入口。
- 这些拆分是行为保持型维护重构，不代表新的 Worker protocol、SQLite schema、内置能力注册表、训练/推理/评估算法或报告字段变更。

## 构建环境

当前机器验证过的组合：

- MSVC 19.50
- Qt 5.12.9 `C:\Qt\Qt5.12.9\5.12.9\msvc2017_64`（`tools\toolchain-env.ps1` 优先选择该 kit；缺失时会 fallback 到 `msvc2015_64`）
- CMake NMake Makefiles
- 构建目录：`build-vscode`

标准检查命令：

```powershell
.\tools\harness-check.ps1
```

## `.deps` 环境布局

可复用的本地环境统一放在 `.deps` 的固定子目录，避免把运行环境散落到历史验证输出目录：

- Python 环境：`.deps\envs`，包括 `yolo-cuda`、`yolo26`、`ocr-cpu`、`ocr-gpu`、`python-embed-3.13.13` 和 `paddle2onnx`。
- 源码 checkout：`.deps\repos`，PaddleOCR 默认是 `.deps\repos\PaddleOCR`。
- SDK/runtime：`.deps\sdks`，包括 `onnxruntime`、`ncnn`、`tensorrt-oss` 和 `tensorrt-runtime`。
- 外部工具：`.deps\tools`，X-AnyLabeling 默认查找 `.deps\tools\annotation-tools\X-AnyLabeling`。

历史验证目录如 `.deps\rtx4090-validation`、`.deps\full-model-lifecycle`、`.deps\phase-yolo26-model-matrix` 只作为证据和 run 输出，不再作为新环境默认位置。详细约定见 `docs/deps-layout.md`；旧目录可通过 `.\tools\sync-deps-layout.ps1` 创建兼容 junction。

## 编码与终端约束

- 仓库文本文件按 UTF-8 处理；中文源文案、文档、翻译文件和脚本输出说明都不能依赖系统 ANSI/GBK 猜测。
- Windows PowerShell 读取中文或中英混排文件时必须显式指定 `-Encoding UTF8`，例如 `Get-Content -LiteralPath HARNESS.md -Encoding UTF8`。
- 如果终端显示 `鐨勭洰鏍`、`\345\220...` 等乱码或转义，先确认读取编码和 Git 路径输出设置；不要把终端 mojibake 当成文件内容损坏。
- Git 枚举中文路径时使用 `git -c core.quotepath=false ...`，或在本机配置 `git config --global core.quotepath false`。
- 修改已有文本文件时保留 UTF-8，并尽量保留文件原有 BOM / no BOM 风格；二进制文件不要按文本编码处理。

## 关键约束

- 保持 Qt 5.12+ 兼容，除非明确升级到 Qt 6。
- Windows 源码编译必须使用 `/utf-8`，避免中文乱码。
- UI 改动必须保留现有功能入口。
- GUI 可见文本优先保持中文源文案，通过 Qt 翻译资源和 `LanguageSupport` 派生英文界面；core/Worker/Python trainer 日志不要求在第一版全部翻译。
- 注册码系统使用离线签名 token；主应用只内置公钥，私钥文件必须本地保管，不进入客户包和源码提交。
- Worker 协议改动必须同步测试。
- 能力注册表改动必须同步 GUI、Worker、环境检查和协议测试。
- scaffold、smoke、diagnostic 或 report-only 能力必须明确标注；已移除的 tiny/scaffold/mock/小型 CTC 训练路径不得重新描述为产品后端。
