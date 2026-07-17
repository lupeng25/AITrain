# OCR 客户域验收工作流 V2

## 目标与边界

OCR Acceptance V2 只汇总已经进入 V2 Artifact Store 的 PaddleOCR Det、Rec、System 官方报告事实。它不接收裸报告路径，不运行 PaddleOCR，不读取 GUI 全局状态，也不把公开数据、生成数据或 smoke 结果转换成客户域生产结论。

当前范围包含 Acceptance Core 与受控报告 Packager/Importer Core。GUI、Worker、客户数据采集以及旧 `runCustomerOcrAcceptance` 的删除仍属于后续任务。

## 固定步骤

1. `ResolveEvidence`：按三个 ArtifactId 解析 committed Artifact，逐文件重新核对文件集合、长度和 SHA-256。
2. `ValidateOfficialReports`：校验报告 JSON 合同、PaddleOCR 官方 backend/framework/mode、客户域 lineage，并验证 System 对当前 Det/Rec 报告哈希的绑定。
3. `EvaluateThresholds`：检查 Det、Rec、System 样本数和 Det hmean、Rec accuracy/CER、System accuracy。
4. `RenderAcceptanceReport`：只有前三步全部成功才提交 JSON/Markdown 验收报告，并写入 `productionAccepted=true`。

每个成功步骤必须输出 committed Artifact。工作流采用 `EvidenceRequired` 终态协议：先封存成功、失败或取消事实，再提交 JSON、Markdown、HTML、Model Card 四种 Evidence，最后关闭唯一根任务终态。

## 输入合同

请求只包含：

- Det 官方报告 ArtifactId。
- Rec 官方报告 ArtifactId。
- System 官方报告 ArtifactId。
- 最小样本数和指标阈值。

每个报告 Artifact 必须包含原始官方报告和 `official_report_lineage.json`。lineage 至少记录 schema、组件、报告相对路径和 SHA-256、证据分类、验收批次、客户域、数据集指纹。System lineage 还必须记录 Det/Rec 报告 SHA-256。

报告 Artifact 的组件 kind 固定为：

- `paddleocr_det_official_report_v2`
- `paddleocr_rec_official_report_v2`
- `paddleocr_system_official_report_v2`

原始报告必须声明成功状态、`PaddleOCR official tools` framework、对应官方 backend 和非 prepare-only 模式。当前验收指标合同为：

- Det：`metrics.sampleCount`、`metrics.hmean`。
- Rec：`metrics.sampleCount`、`metrics.accuracy`、`metrics.cer`。
- System：`metrics.sampleCount`、`metrics.accuracy`。

`paddleocr_*_official_report_v2` 是受控打包后的验收报告 Artifact 合同，不是现有训练步骤自动产生的 Artifact kind。现有 Adapter 原始报告不能直接传给 Acceptance API；必须先由受控 Packager/Importer 导入。Core 不修改已有 Python Adapter 输出。

## 受控报告导入

显式导入请求为 Det、Rec、System 分别提供：

- 用户明确选择的原始官方报告路径；裸路径只允许停留在本次导入边界。
- 已登记的 `SnapshotId`，或该 Snapshot 的 committed `dataset_snapshot_v2` ArtifactId；两者同时提供时必须指向同一记录。
- 三份报告共同使用的验收批次、客户域和证据分类。

导入器重新读取 Storage 记录，校验 committed `dataset_snapshot.json` 的长度和 SHA-256，再逐项重哈希 Snapshot manifest 声明的源文件。`metrics.sampleCount` 不是用户输入：PaddleOCR Det 只统计已验证 `det_gt*.txt` 中合法的非空标签行，PaddleOCR Rec 只统计 `rec_gt*.txt` 中合法的非空标签行，System 使用其绑定 Snapshot 的同类标签事实；若 System 原报告带 `predictionCount`，还必须与该计数一致。

Det/Rec 原始评估报告必须匹配当前 Adapter 的真实 schema、`paddleocr_det_official_eval` / `paddleocr_rec_official_eval` backend、component、taskType、Snapshot 引用和真实指标。当前 Det/Rec 原始结构本身没有 `framework`、`mode`、`ok` 字段；规范化 lineage 会明确记录官方身份来自已注册 Adapter backend 合同，不会伪称这些字段存在于原报告。Rec `acc` 可无损规范化为 `accuracy`，但不会把 `norm_edit_dis` 冒充 CER；缺少真实 `cer` 时属于无效证据。

System 原始报告必须原生声明 `ok=true`、`paddleocr_system_official`、`PaddleOCR official tools` 和 `officialSystemPredict`，并绑定已提交 Snapshot。当前常规 System wiring 报告通常只有预测计数而没有客户域 ground truth `accuracy`；这种输入精确返回 `ocr_report_import.system_accuracy_unsupported`，不会生成验收 Artifact。只有报告实际包含有限、范围为 0～1 的 `metrics.accuracy` 才能打包，不能由用户补填或由预测数量推导。

三份输入全部预校验通过后，导入器才生成三种 committed Artifact。每种 Artifact 同时保存原始报告、规范化报告和 `official_report_lineage.json`；System lineage 绑定本次规范化 Det/Rec 报告 SHA-256。提交中途失败或取消会反向删除已经提交且尚未被引用的 Artifact，并清理暂存；补偿不完整会返回独立失败码，不能把部分结果作为可用报告。

受控 Packager/Importer、Acceptance、Annotation Worker/UI 与 legacy integration 的统一定向 CTest 已 5/5 通过（35.06 秒），其中 OCR 报告 Packager 合同测试 1/1 通过。该验证只证明 Core 合同、补偿和接线回归；OCR 报告导入与 Acceptance 仍未接入 GUI/Worker，也没有产生任何真实客户域 accuracy 证据。

## 精确失败

以下情况不得生成 accepted 报告：

- Artifact 或报告缺失：`ocr_acceptance.report_missing`。
- 已提交文件被修改或文件集合变化：`ocr_acceptance.report_tampered`。
- 报告合同或官方身份不正确：`ocr_acceptance.report_schema_invalid`。
- lineage 不完整、不一致或 System 未绑定当前 Det/Rec：`ocr_acceptance.lineage_invalid` / `ocr_acceptance.lineage_mismatch`。
- 证据不是客户域：`ocr_acceptance.customer_domain_evidence_required`。
- 样本不足：`ocr_acceptance.sample_count_insufficient`。
- 指标阈值不达：`ocr_acceptance.threshold_not_met`。

Failure 中保存稳定 code、中文事实说明和建议动作。失败与取消仍生成 Evidence，但不会生成 production accepted。

## 声明限制

一次成功只说明指定客户域、指定验收批次、指定三份官方报告达到了当前阈值。它不外推到其他客户域，也不证明 Clean Windows、TensorRT、NCNN、SMP、Anomaly 或 OBB 验收通过。
