# 下一阶段 RC 交付计划

最后更新：2026-05-15

本文是 RTX 4090 D 验证之后的执行计划，不替代 `docs/harness/current-status.md`。当前状态、证据路径和阶段结论仍以 `docs/harness/current-status.md` 为准。

## 摘要

- 目标：把当前 RTX 4090 D 验证结果冻结为可追溯的 release-candidate 基线，并准备本地交付包；除非显式重开，否则本轮不补采 clean Windows/package-root 外部证据。
- 当前基线：RTX 4090 D 验证已经完成。TensorRT、Phase 45 YOLO11/YOLO12 矩阵、Phase 47 PaddleOCR Det ONNX、Production OCR acceptance 均有通过证据，主要位于 `.deps\rtx4090-validation`。
- 当前 handoff：本地 RC handoff 已从 source commit `8457da88738706f32fa1ec014a317e264bc08c67` 刷新，`build-vscode\release-freeze-handoff\release_handoff_manifest.json` 记录 `worktreeDirty=false`，ZIP SHA256 为 `72B5C2A1933E32EFD353857862F3E59F5AA0B984C91C4DE412E34E78E64934EE`。
- Phase 49 本地交付闭环已经完成：打包 GUI 覆盖样本复核、交付验收、客户 OCR 验收、诊断包、部署验证和 mAP50-95 报告证据。这些是本地报告/流程界面，不替代 clean-machine、客户域数据或 package-root TensorRT 返回证据。
- Production OCR gate：当前默认门槛接受 Rec `accuracy > 0.7`；CER 会记录，但只有使用 `-RequireRecCer` 时才作为阻断项。
- 产物规则：不得把 `.deps`、build 输出、数据集、模型权重、ONNX、TensorRT engine、生成 ZIP 或其它生成二进制加入源码控制。

## 执行计划

1. 关闭当前源码差异。
   - 检查工作区，确认只包含预期的 gate、文档或脚本变更。
   - 运行 `git diff --check`。
   - 运行 `.\tools\harness-check.ps1`。
   - 验证通过后，只提交源码管理内的脚本和文档变更。

2. 运行本地 RC gate。
   - 运行 `.\tools\local-rc-closeout.ps1 -RunLocalBaseline -RunCpuTrainingSmoke -BuildDir build-vscode`。
   - 运行 `.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild -WorkDir .deps\release-rc\acceptance-local-package`。
   - 要求所有 summary JSON 返回 `passed`。

3. 生成 release handoff 包。
   - 运行 `.\tools\release-freeze-handoff.ps1`。
   - 记录 ZIP 路径、SHA256、commit HEAD、生成时间和 dirty-worktree 状态。
   - 如果 manifest 显示 dirty worktree，必须先解决，再把包视为最终 RC handoff。

4. 本轮默认延后外部验收包发送。
   - 保留生成的 ZIP 包。
   - 保留 `build-vscode\release-freeze-handoff\release_handoff_manifest.json`。
   - 保留 `build-vscode\release-freeze-handoff\release_handoff_summary.md`。
   - 保留 `docs\external-acceptance-handoff.md`。
   - 保留 `docs\acceptance-templates\` 下的模板。
   - 未收到新优先级前，不发送外部验收包，也不把外部结果标记为通过。

5. clean Windows package acceptance 单独延后。
   - 在 clean Windows 机器上解压 package，从 package root 运行 `.\tools\acceptance-smoke.ps1 -Package`。
   - 收集 `acceptance_summary.json`、完整控制台输出、package layout evidence 和填写后的 `clean-windows-acceptance-result.md`。
   - 只有 package-root summary 为 `passed` 时，才可把 clean-machine acceptance 标记为 passed。
   - 没有返回证据时，不得标记为通过。

6. package-root TensorRT rerun 默认延后。
   - 如需重开，在 RTX / SM 75+ 机器上从 package root 运行 `.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt`。
   - 收集 GPU 型号、驱动、CUDA runtime、Worker self-check JSON、TensorRT 控制台输出和填写后的 `tensorrt-acceptance-result.md`。
   - GTX 1060 / SM 61 结果应保持 `hardware-blocked`，不得覆盖 RTX 4090 D 已通过证据。
   - 没有返回 package-root RTX / SM 75+ 证据时，不得把该 rerun 标记为通过。

7. 审核返回证据后更新状态。
   - 如果所有显式要求的外部证据均通过，再更新 `docs/harness/current-status.md`。
   - 如果任何外部检查 blocked 或 failed，记录 blocker 和证据，不降低 gate。
   - RTX 4090 D 验证证据、clean-machine package 证据、package-root TensorRT rerun 证据必须分开记录。
   - Phase 49 GUI 导入/展示证据必须和底层脚本、Worker、clean-machine、TensorRT、客户域证据来源分开。

## 验收标准

- 本地验证通过：
  - `git diff --check`
  - `.\tools\harness-check.ps1`
  - `.\tools\local-rc-closeout.ps1 -RunLocalBaseline -RunCpuTrainingSmoke -BuildDir build-vscode`
  - `.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild`
- Release handoff 存在并记录：
  - ZIP 路径
  - SHA256
  - commit HEAD
  - 生成时间
  - dirty-worktree 状态
- 没有返回证据时，external clean-machine package acceptance 保持 deferred。
- 没有返回 package-root RTX / SM 75+ 证据时，optional TensorRT rerun 保持 deferred。
- public Total-Text 或 generated smoke 数据不能证明客户域 OCR production readiness；客户域 OCR 需要真实客户/目标域证据。
- 生成产物不进入源码控制。

## 风险边界

- 当前 Production OCR 结果只在降低后的 `accuracy > 0.7` gate 下成立；它是 public Total-Text 工作流证据，不是客户域 OCR 生产质量证明。
- Phase 49 customer OCR validation 是收集客户域证据的本地机制，但输入数据和报告必须来自真实客户/目标域。
- 旧的 `accuracy >= 0.90` 与 `CER <= 0.10` 目标不是当前默认阻断 gate；如需恢复，必须单独制定模型、数据和训练改进计划。
- Phase 47 验证 PaddleOCR Det ONNX 转换和 C++ DB-style 后处理接线，不证明 PP-OCRv5 精度一致性。
- Phase 45 只验证 YOLO detection/segmentation 接线和产物，不新增 classification、pose、OBB、anomaly、YOLO-World 或 YOLOE 范围。
- Plugin marketplace v1 是 local/offline-first 且未启用 publisher signature enforcement；除非后续阶段明确增加签名策略，不得把它描述为可信插件商店。
- Windows 锁定活动 Qt plugin DLL 时，marketplace disable/uninstall 需要保持状态安全：GUI 只释放已记录的 marketplace active DLL，删除失败时报 `disable-failed`，否则更新 marketplace 状态，不强制热重扫全局插件矩阵。需要刷新数量时使用 rescan。
- External acceptance 本轮延后。重新打开时，必须先拿到返回证据，再改变状态。

## 验证命令

本次文档变更：

```powershell
git diff --check
.\tools\harness-context.ps1
```

最终 RC handoff 前：

```powershell
.\tools\harness-check.ps1
.\tools\local-rc-closeout.ps1 -RunLocalBaseline -RunCpuTrainingSmoke -BuildDir build-vscode
.\tools\release-freeze-handoff.ps1
```
