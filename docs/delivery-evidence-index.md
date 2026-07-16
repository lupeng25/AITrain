# 交付证据索引

最后更新：2026-06-21

本文把当前 RC、RTX 验证、打包、OCR、诊断和外部验收相关证据集中列出，避免把本地证据、RTX 4090 D 证据、clean Windows 证据和客户域证据混在一起。阶段状态仍以 `docs/harness/current-status.md` 为准。

> 2026-07-16 破坏性重构说明：旧 `acceptance-smoke.ps1 -TensorRT`、独立 OBB/NCNN/SMP smoke 及 Worker 裸路径命令均已删除；历史证据不可作为当前 V2 通过结论。当前证据必须来自 V2 Workflow、Artifact/Evidence 查询或官方 Python 适配器报告。

## 证据总表

| 证据项 | 当前状态 | 主要路径或命令 | 备注 |
|---|---|---|---|
| 阶段状态源 | 已维护 | `docs/harness/current-status.md` | 项目状态唯一来源。 |
| 本地/RTX follow-up summary | 已通过并保留边界 | `docs\validation\rtx4090-validation-evidence-20260615.json` | 记录 2026-06-05 LocalBaseline+Package、GUI walkthrough、Phase47+CTest 刷新结果，并引用 CPU smoke、Phase45、TensorRT、public OCR 通过证据。 |
| 本地 RC closeout | 历史通过本地/RTX 验证 lane | `docs\validation\rtx4090-validation-evidence-20260615.json` | 记录本地 closeout、CPU smoke、package smoke 等汇总。 |
| RTX 4090 D TensorRT | 已通过当前 validation lane | `docs\validation\rtx4090-validation-evidence-20260615.json`；复跑命令仍可使用 `.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\rtx4090-validation\acceptance-tensorrt` | 不等于后续任意 package-root rerun 已通过。 |
| Release handoff | 已刷新 / 以 manifest 为准 | `build-vscode\release-freeze-handoff\release_handoff_manifest.json`；`build-vscode\release-freeze-handoff\release_handoff_summary.md` | 具体 source commit、dirty 状态、ZIP 路径和 SHA256 以最新 manifest 为准。不要在长期索引里把某一次 hash 当作当前包身份。 |
| Clean Windows package acceptance | 延后 / 未返回 | `docs\external-acceptance-handoff.md`；`docs\acceptance-templates\clean-windows-acceptance-result.md` | 没有 clean-machine 返回证据时不得标记 passed。 |
| Package-root TensorRT rerun | 延后 / 未返回 | `docs\acceptance-templates\tensorrt-acceptance-result.md` | RTX 4090 D 源侧通过证据与 package-root rerun 分开记录。 |
| Phase 45 YOLO11/YOLO12 matrix | 已通过 RTX validation lane | `docs\validation\rtx4090-validation-evidence-20260615.json`；复跑命令为 `.\tools\phase45-yolo-model-matrix-smoke.ps1` | 2026-06-05 修复后矩阵 `status=passed`、`ctestStatus=passed` 已归档；原始 `.deps` 产物目录已清理。验证 detection/segmentation 接线和产物，不是精度 benchmark。 |
| Historical Phase 47 PaddleOCR Det ONNX | 历史 wiring 证据 | `docs\validation\rtx4090-validation-evidence-20260615.json` | 仅用于解释过去的 ONNX 转换和 C++ 后处理接线范围；当前 OCR 验收必须使用 PaddleOCR 官方 Det/Rec/System 报告。 |
| Production OCR public workflow | 已通过 public workflow lane | `docs\validation\rtx4090-validation-evidence-20260615.json`；复跑环境通过 `.deps\envs\ocr-gpu` 暴露，旧 `.deps\rtx4090-validation\python-ocr-gpu` 为兼容 target | Public Total-Text 证据不能证明客户域生产精度。 |
| Customer-domain OCR | 需要客户/目标域证据 | `.\tools\customer-ocr-validation.ps1`；`环境 > 交付证据` 导入结果 | 只有真实客户/目标域数据和报告才能支撑生产声明。 |
| 数据集转换 GUI closeout | 已完成本地验证 | `.deps\UI-Walkthrough\dataset-conversion\walkthrough-summary.json` | 转换产物不自动注册为数据集。 |
| GUI walkthrough | 已固化为本地 RC gate | `.\tools\ui-workbench-walkthrough.ps1`；`docs\validation\rtx4090-validation-evidence-20260615.json` | 当前固定 1280x820，覆盖 9 个对象型工作区；tab-level 区域由 QtTest 和 focused notes 覆盖。历史 2026-06-05 follow-up 证据保留在归档中。 |
| SMP semantic segmentation | 已通过本地 RTX 4090D realtest | `.deps\smp-realtest\gpu-4090d`；`.\tools\phase-smp-4090d-gpu-realtest.ps1` | Public/synthetic workflow evidence only; SMP 范围为 ONNX Runtime，不要求 NCNN/TensorRT。 |
| OBB v1 | 已通过本地 public DOTA smoke/matrix | `.deps\phase-obb-ultralytics-smoke`；`.deps\obb-quality\dota`; `.\tools\phase-obb-dota-quality-matrix.ps1` | Public DOTA/workflow evidence only；OBB v1 产品部署范围为 ONNX Runtime，NCNN 不支持。 |
| Anomaly v1 | 已通过本地 public MVTec 默认三类矩阵 | `.deps\anomaly-mvtec-quality-matrix\anomaly_mvtec_quality_matrix_summary.json`; `.\tools\phase-anomaly-mvtec-quality-matrix.ps1` | Public MVTec workflow/quality evidence only；运行时为 Worker-managed Python/Anomalib，不声明 AITrain C++ ONNX/TensorRT/NCNN anomaly runtime。 |
| Diagnostics Bundle V2 | 两步 EvidenceRequired 工作流已落地并通过统一构建/回归 | `环境 > 交付证据` 的“一键诊断包”；Worker `runDiagnosticsWorkflowV2` | 只读 Presenter 按 TaskId 从 V2 Query 读取 Diagnostics/Evidence ArtifactId，不消费 Worker 路径。同步外部探测不可在调用中抢占，取消在探测前后收口；输出有固定上限。2026-07-16 定向 CTest 4/4 通过。 |
| Training / Runtime Delivery | V2 Workflow 统一生成交付摘要并通过删除回归 | Training Workflow 提交 `training_delivery_report_v2`，Runtime Delivery 提交 `runtime_delivery_report_v2`，两者终态提交 `evidence_bundle_v2`；任务页按 TaskId 通过 Presenter 查看 ArtifactId 和包内清单 | standalone `generateDeliveryReport` 已删除，不再接收模型、数据集、上下文或输出路径。Runtime 仍只接受已登记 `ModelPackageId`；同步 infer 与 TensorRT 限制保持不变。2026-07-16 定向 CTest 4/4 通过。 |
| NCNN runtime smoke | 本机检测/分割 runtime 已有证据 | `.deps\github-ncnn-smoke\hyuto-yolov8\runtime-output`；`.deps\github-ncnn-smoke\nihui-yolov8n-seg-ncnn\runtime-output\deployment-validation` | Hyuto YOLOv8 detection ONNX -> NCNN passed，nihui 预转换 YOLOv8n-seg pnnx/DFL NCNN passed；YOLOv8-seg ONNX 若残留 unsupported `Shape` layer，则记录为 failed report。 |

## 证据分层

- 本地源码证据：`git diff --check`、`.\tools\harness-context.ps1`、`.\tools\harness-check.ps1`、本地 GUI walkthrough。
- GUI walkthrough 若记录 `errorCode=license_required`，只能作为授权配置 blocked 证据，不能当作布局通过证据。
- 本地 package 证据：`.\tools\package-smoke.ps1 -SkipBuild`、`.\tools\acceptance-smoke.ps1 -Package -SkipBuild`。
- RTX validation lane：历史 TensorRT、YOLO matrix、PaddleOCR Det ONNX、Production OCR evidence 已归档到 `docs\validation\rtx4090-validation-evidence-20260615.json`；OCR GPU 复跑环境通过 `.deps\envs\ocr-gpu` 暴露，旧 `.deps\rtx4090-validation\python-ocr-gpu` 仅作为兼容 target 保留。
- 外部 clean-machine 证据：clean Windows package root 执行结果和填写后的模板。
- 客户域证据：客户/目标域 Det、Rec、System 数据、官方报告、AITrain 汇总报告和验收结论。

## 不可混用的结论

- RTX 4090 D TensorRT 通过，不代表 clean Windows package acceptance 通过。
- RTX 4090 D TensorRT 通过，不代表未来 package-root TensorRT rerun 自动通过。
- Public Total-Text 或 generated smoke 通过，不代表客户域 OCR 生产精度通过。
- GUI `环境 > 交付证据` 显示导入结果，不替代底层脚本、Worker、外部机器或客户数据证据。
- NCNN 部署验证不再使用 artifact-only 通过条件；无 SDK/runtime 会失败，缺少样本图会阻塞，外部模型需要 sidecar 或显式 blob/decoder 配置。
- NCNN failed report 必须带 `errorCode` / `failureCategory` / `nextAction` / `diagnosticHints`；当前分类为 `sdk_missing`、`sample_missing`、`sidecar_missing`、`unsupported_layer`、`runtime_failed`。
- NCNN 分割 runtime 当前通过证据来自 nihui 预转换 pnnx/DFL artifact；不要把失败的 YOLOv8-seg ONNX -> `onnx2ncnn` `Shape` layer case 说成分割 runtime 通过。

## 维护规则

1. 新增 acceptance lane 时，先在 `docs/harness/current-status.md` 记录状态和边界，再在本文补索引。
2. 所有生成证据保留在 `.deps`、build 输出或交付包目录，不提交到源码控制。
3. 如果证据来自外部机器，必须记录机器类型、GPU、驱动/runtime、命令、输出摘要和模板文件。
4. 如果某个 lane 被重开，旧证据保留为历史记录，新证据另起路径，避免覆盖。
