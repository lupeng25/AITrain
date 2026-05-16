# 交付证据索引

最后更新：2026-05-16

本文把当前 RC、RTX 验证、打包、OCR、诊断和外部验收相关证据集中列出，避免把本地证据、RTX 4090 D 证据、clean Windows 证据和客户域证据混在一起。阶段状态仍以 `docs/harness/current-status.md` 为准。

## 证据总表

| 证据项 | 当前状态 | 主要路径或命令 | 备注 |
|---|---|---|---|
| 阶段状态源 | 已维护 | `docs/harness/current-status.md` | 项目状态唯一来源。 |
| 本地 RC closeout | 已通过本地/RTX 验证 lane | `.deps\rtx4090-validation\2026-05-13-closeout` | 记录本地 closeout、CPU smoke、package smoke 等汇总。 |
| RTX 4090 D TensorRT | 已通过当前 validation lane | `.deps\rtx4090-validation\acceptance-tensorrt`；`.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\rtx4090-validation\acceptance-tensorrt` | 不等于后续任意 package-root rerun 已通过。 |
| Release handoff | 已刷新 | `build-vscode\release-freeze-handoff\release_handoff_manifest.json` | source commit `8457da88738706f32fa1ec014a317e264bc08c67`，ZIP SHA256 `72B5C2A1933E32EFD353857862F3E59F5AA0B984C91C4DE412E34E78E64934EE`。 |
| Clean Windows package acceptance | 延后 / 未返回 | `docs\external-acceptance-handoff.md`；`docs\acceptance-templates\clean-windows-acceptance-result.md` | 没有 clean-machine 返回证据时不得标记 passed。 |
| Package-root TensorRT rerun | 延后 / 未返回 | `docs\acceptance-templates\tensorrt-acceptance-result.md` | RTX 4090 D 源侧通过证据与 package-root rerun 分开记录。 |
| Phase 45 YOLO11/YOLO12 matrix | 已通过 RTX validation lane | `.deps\rtx4090-validation\2026-05-13-closeout`；`.\tools\phase45-yolo-model-matrix-smoke.ps1` | 验证 detection/segmentation 接线和产物，不是精度 benchmark。 |
| Phase 47 PaddleOCR Det ONNX | 已通过 RTX validation lane | `.deps\rtx4090-validation\phase47-paddleocr-det-onnx` | 验证 ONNX 转换和 C++ DB-style 后处理接线。 |
| Production OCR public workflow | 已通过 public workflow lane | `.deps\rtx4090-validation\production-ocr-acceptance-gpu-chain` | Public Total-Text 证据不能证明客户域生产精度。 |
| Customer-domain OCR | 需要客户/目标域证据 | `.\tools\customer-ocr-validation.ps1`；交付验收页导入结果 | 只有真实客户/目标域数据和报告才能支撑生产声明。 |
| 数据集转换 GUI closeout | 已完成本地验证 | `.deps\ui-walkthrough-dataset-conversion\walkthrough-summary.json` | 转换产物不自动注册为数据集。 |
| GUI walkthrough | 已有本地证据 | 当前状态文档记录的 1280x820 Qt walkthrough | 覆盖主要页面和无横向溢出检查。 |
| 诊断包 | GUI/Worker 能力已落地 | 交付验收页“一键诊断包”；Worker `collectDiagnostics` | 诊断包是只读证据，不修改全局环境。 |
| 部署验证 | GUI/Worker 能力已落地 | 交付验收页“部署验证”；Worker `validateDeploymentArtifact` | ONNX 需要可运行推理；TensorRT 可返回 `hardware-blocked`；NCNN 在配置 SDK/runtime 和样本图时执行 YOLO 检测/分割 runtime inference。 |
| NCNN runtime smoke | 本机检测/分割 runtime 已有证据 | `.deps\github-ncnn-smoke\hyuto-yolov8\runtime-output`；`.deps\github-ncnn-smoke\nihui-yolov8n-seg-ncnn\runtime-output\deployment-validation` | Hyuto YOLOv8 detection ONNX -> NCNN passed，nihui 预转换 YOLOv8n-seg pnnx/DFL NCNN passed；YOLOv8-seg ONNX 若残留 unsupported `Shape` layer，则记录为 failed report。 |

## 证据分层

- 本地源码证据：`git diff --check`、`.\tools\harness-context.ps1`、`.\tools\harness-check.ps1`、本地 GUI walkthrough。
- 本地 package 证据：`.\tools\package-smoke.ps1 -SkipBuild`、`.\tools\acceptance-smoke.ps1 -Package -SkipBuild`。
- RTX validation lane：`.deps\rtx4090-validation` 下的 TensorRT、YOLO matrix、PaddleOCR Det ONNX、Production OCR evidence。
- 外部 clean-machine 证据：clean Windows package root 执行结果和填写后的模板。
- 客户域证据：客户/目标域 Det、Rec、System 数据、官方报告、AITrain 汇总报告和验收结论。

## 不可混用的结论

- RTX 4090 D TensorRT 通过，不代表 clean Windows package acceptance 通过。
- RTX 4090 D TensorRT 通过，不代表未来 package-root TensorRT rerun 自动通过。
- Public Total-Text 或 generated smoke 通过，不代表客户域 OCR 生产精度通过。
- GUI 交付验收页显示导入结果，不替代底层脚本、Worker、外部机器或客户数据证据。
- NCNN 部署验证不再使用 artifact-only 通过条件；无 SDK/runtime 会失败，缺少样本图会阻塞，外部模型需要 sidecar 或显式 blob/decoder 配置。
- NCNN 分割 runtime 当前通过证据来自 nihui 预转换 pnnx/DFL artifact；不要把失败的 YOLOv8-seg ONNX -> `onnx2ncnn` `Shape` layer case 说成分割 runtime 通过。

## 维护规则

1. 新增 acceptance lane 时，先在 `docs/harness/current-status.md` 记录状态和边界，再在本文补索引。
2. 所有生成证据保留在 `.deps`、build 输出或交付包目录，不提交到源码控制。
3. 如果证据来自外部机器，必须记录机器类型、GPU、驱动/runtime、命令、输出摘要和模板文件。
4. 如果某个 lane 被重开，旧证据保留为历史记录，新证据另起路径，避免覆盖。
