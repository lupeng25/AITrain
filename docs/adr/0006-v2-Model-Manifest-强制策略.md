# ADR-006：V2 Model Manifest 强制策略

日期：2026-07-15  
状态：已接受

## 决策

模型进入推理、评估、导出或部署前必须拥有 V2 Model Manifest。Manifest 明确任务类型、输入输出契约、产物哈希、可用运行时和后端边界；不再依赖文件名、目录名或输出 tensor 数量猜测模型类型。

## 后果

- ONNX、NCNN、TensorRT 能力按 Manifest 和 Runtime Capability Matrix 判定。
- 现有模型猜测逻辑将在阶段 4 删除。
