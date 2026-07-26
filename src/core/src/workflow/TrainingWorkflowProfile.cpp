#include "aitrain/workflow/TrainingWorkflowProfile.h"
#include "aitrain/product/ProductCapabilityContract.h"

#include <QtGlobal>

namespace aitrain {
namespace {

TrainingWorkflowStepProfile step(const QString& kind,
    const QString& backend,
    const QString& script = QString(),
    const QStringList& artifactCandidates = {})
{
    return {kind, backend, script, artifactCandidates};
}

void applyProductContract(const QString& trainingBackend, TrainingWorkflowProfile* value)
{
    TrainingBackendContract contract;
    const bool found = ProductCapabilityContract::instance().resolveTrainingBackend(
        trainingBackend, &contract);
    Q_ASSERT(found);
    value->trainingBackend = contract.id;
    value->capabilityTaskType = contract.taskType;
    value->adapterTaskType = contract.taskType;
    value->datasetFormat = contract.datasetFormat;
    value->modelFamily = contract.modelFamily;
    value->decoder = contract.decoder;
    value->artifactFormat = contract.officialArtifactFormat;
    value->runtimeRoutes = contract.runtimeRouteIds;
    value->pythonProfileId = contract.pythonProfileId;
    value->limitations = contract.limitations;
}

TrainingWorkflowProfile yoloProfile(const QString& trainingBackend,
    const QString& trainScript,
    const QStringList& limitations)
{
    TrainingWorkflowProfile value;
    applyProductContract(trainingBackend, &value);
    value.templateId = QStringLiteral("official_yolo_training_delivery");
    value.trainScript = trainScript;
    value.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    value.evaluationScript = QStringLiteral("yolo/ultralytics_evaluator.py");
    value.exportBackend = QStringLiteral("ultralytics_yolo_export");
    value.exportScript = QStringLiteral("yolo/ultralytics_exporter.py");
    value.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    value.limitations.append(limitations);
    value.steps = {
        step(QStringLiteral("ValidateDataset"), QStringLiteral("dataset_driver")),
        step(QStringLiteral("CreateSnapshot"), QStringLiteral("snapshot_store")),
        step(QStringLiteral("Train"), value.trainingBackend, value.trainScript,
            {QStringLiteral("dataset_snapshot.json")}),
        step(QStringLiteral("Evaluate"), value.evaluationBackend, value.evaluationScript,
            {QStringLiteral("checkpoint/best.pt"), QStringLiteral("checkpoint/last.pt")}),
        step(QStringLiteral("Export"), value.exportBackend, value.exportScript,
            {QStringLiteral("checkpoint/best.pt"), QStringLiteral("checkpoint/last.pt"),
                QStringLiteral("evaluation_report/evaluation_report.json")}),
        step(QStringLiteral("DeploymentValidate"), value.deploymentBackend, {},
            {QStringLiteral("export/model.onnx"),
                QStringLiteral("export_sidecar/model.aitrain-export.json")}),
        step(QStringLiteral("RegisterModel"), QStringLiteral("storage"), {},
            {QStringLiteral("export/model.onnx"),
                QStringLiteral("export_sidecar/model.aitrain-export.json")}),
        step(QStringLiteral("RenderDeliveryReport"), QStringLiteral("evidence_renderer"))};
    return value;
}

TrainingWorkflowProfile smpProfile()
{
    TrainingWorkflowProfile value;
    applyProductContract(QStringLiteral("smp_semantic_segmentation"), &value);
    value.templateId = QStringLiteral("smp_semantic_segmentation_delivery");
    value.trainScript = QStringLiteral("semantic_segmentation/smp_trainer.py");
    value.evaluationBackend = QStringLiteral("smp_semantic_segmentation_eval");
    value.evaluationScript = QStringLiteral("semantic_segmentation/smp_evaluator.py");
    value.exportBackend = QStringLiteral("smp_semantic_segmentation_export");
    value.exportScript = QStringLiteral("semantic_segmentation/smp_exporter.py");
    value.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    value.limitations.append(
        QStringLiteral("公开或生成数据集的工作流证据不代表客户域精度。"));
    value.steps = {
        step(QStringLiteral("ValidateDataset"), QStringLiteral("dataset_driver")),
        step(QStringLiteral("CreateSnapshot"), QStringLiteral("snapshot_store")),
        step(QStringLiteral("Train"), value.trainingBackend, value.trainScript),
        step(QStringLiteral("Evaluate"), value.evaluationBackend, value.evaluationScript,
            {QStringLiteral("onnx_model/best.onnx"),
                QStringLiteral("model_sidecar/semantic_segmentation_sidecar.json"),
                QStringLiteral("checkpoint/best.pt")}),
        step(QStringLiteral("Export"), value.exportBackend, value.exportScript,
            {QStringLiteral("onnx_model/best.onnx"),
                QStringLiteral("model_sidecar/semantic_segmentation_sidecar.json"),
                QStringLiteral("evaluation_report/evaluation_report.json")}),
        step(QStringLiteral("DeploymentValidate"), value.deploymentBackend, {},
            {QStringLiteral("export/model.onnx"),
                QStringLiteral("export_sidecar/model.aitrain-export.json")}),
        step(QStringLiteral("RegisterModel"), QStringLiteral("storage"), {},
            {QStringLiteral("export/model.onnx"),
                QStringLiteral("export_sidecar/model.aitrain-export.json")}),
        step(QStringLiteral("RenderDeliveryReport"), QStringLiteral("evidence_renderer"))};
    return value;
}

TrainingWorkflowProfile anomalibProfile(const QString& trainingBackend, const QStringList& extraLimitations)
{
    TrainingWorkflowProfile value;
    applyProductContract(trainingBackend, &value);
    value.templateId = QStringLiteral("anomalib_training_delivery");
    value.trainScript = QStringLiteral("anomaly/anomalib_adapter.py");
    value.evaluationBackend = QStringLiteral("anomalib_python_eval");
    value.evaluationScript = value.trainScript;
    value.exportBackend = QStringLiteral("anomalib_artifact_export");
    value.exportScript = QStringLiteral("anomaly/anomalib_exporter.py");
    value.deploymentBackend = QStringLiteral("anomalib_python");
    value.limitations.prepend(
        QStringLiteral("Anomalib v1 由 Worker 托管 Python/Anomalib 产物，不声明 AITrain C++ ONNX、TensorRT 或 NCNN 运行时。"));
    value.limitations.append(
        QStringLiteral("公开 MVTec 或生成数据集的证据不代表客户域精度。"));
    value.limitations.append(extraLimitations);
    value.steps = {
        step(QStringLiteral("ValidateDataset"), QStringLiteral("dataset_driver")),
        step(QStringLiteral("CreateSnapshot"), QStringLiteral("snapshot_store")),
        step(QStringLiteral("Train"), value.trainingBackend, value.trainScript),
        step(QStringLiteral("Evaluate"), value.evaluationBackend, value.evaluationScript,
            {QStringLiteral("anomaly_sidecar/anomaly_sidecar.json"), QStringLiteral("checkpoint/model.ckpt"),
                QStringLiteral("training_report/anomalib_training_report.json")}),
        step(QStringLiteral("Export"), value.exportBackend, value.exportScript,
            {QStringLiteral("anomaly_sidecar/anomaly_sidecar.json"), QStringLiteral("checkpoint/model.ckpt"),
                QStringLiteral("evaluation_report/evaluation_report.json")}),
        step(QStringLiteral("DeploymentValidate"), value.deploymentBackend, value.trainScript,
            {QStringLiteral("export/anomaly_sidecar.json"), QStringLiteral("export/model.ckpt")}),
        step(QStringLiteral("RegisterModel"), QStringLiteral("storage"), {},
            {QStringLiteral("export/anomaly_sidecar.json"), QStringLiteral("export/model.ckpt")}),
        step(QStringLiteral("RenderDeliveryReport"), QStringLiteral("evidence_renderer"))};
    return value;
}

TrainingWorkflowProfile paddleOcrProfile(bool recognition)
{
    TrainingWorkflowProfile value;
    const QString component = recognition ? QStringLiteral("rec") : QStringLiteral("det");
    const QString trainingBackend = recognition
        ? QStringLiteral("paddleocr_rec_official")
        : QStringLiteral("paddleocr_det_official");
    applyProductContract(trainingBackend, &value);
    value.templateId = recognition
        ? QStringLiteral("paddleocr_rec_training_delivery")
        : QStringLiteral("paddleocr_det_training_delivery");
    value.trainScript = QStringLiteral("ocr_%1/paddleocr_%1_trainer.py").arg(component);
    value.evaluationBackend = QStringLiteral("paddleocr_%1_official_eval").arg(component);
    value.evaluationScript = QStringLiteral("ocr_%1/paddleocr_%1_evaluator.py").arg(component);
    value.exportBackend = QStringLiteral("paddleocr_%1_official_export").arg(component);
    value.exportScript = QStringLiteral("ocr_%1/paddleocr_%1_exporter.py").arg(component);
    value.deploymentBackend = QStringLiteral("paddleocr_%1_official_runtime").arg(component);
    value.limitations.prepend(
        QStringLiteral("OCR 训练、评估、导出与部署验证仅使用 PaddleOCR 官方工具链和报告，不声明 AITrain C++ OCR 运行时。"));
    value.limitations.append(
        QStringLiteral("单组件训练 Workflow 不代表 PaddleOCR Det+Rec System 组合验收；公开或生成数据集证据不代表客户域精度。"));

    QStringList checkpointCandidates{
        QStringLiteral("checkpoint/model.zip"),
        QStringLiteral("config/train.yml")};
    if (recognition) {
        checkpointCandidates.append(QStringLiteral("dictionary/dict.txt"));
    }
    QStringList exportCandidates = checkpointCandidates;
    exportCandidates.append(QStringLiteral("evaluation_report/evaluation_report.json"));
    const QStringList inferenceBundleCandidates{
        QStringLiteral("export/paddleocr_bundle.json"),
        QStringLiteral("export/paddleocr_inference.zip")};
    const QString predictorScript = QStringLiteral("ocr_%1/paddleocr_%1_predictor.py").arg(component);
    value.steps = {
        step(QStringLiteral("ValidateDataset"), QStringLiteral("dataset_driver")),
        step(QStringLiteral("CreateSnapshot"), QStringLiteral("snapshot_store")),
        step(QStringLiteral("Train"), value.trainingBackend, value.trainScript),
        step(QStringLiteral("Evaluate"), value.evaluationBackend, value.evaluationScript,
            checkpointCandidates),
        step(QStringLiteral("Export"), value.exportBackend, value.exportScript,
            exportCandidates),
        step(QStringLiteral("DeploymentValidate"), value.deploymentBackend, predictorScript,
            inferenceBundleCandidates),
        step(QStringLiteral("RegisterModel"), QStringLiteral("storage"), {},
            inferenceBundleCandidates),
        step(QStringLiteral("RenderDeliveryReport"), QStringLiteral("evidence_renderer"))};
    return value;
}

QString normalizedBackend(const QString& value)
{
    return value.trimmed().toLower();
}

} // namespace

const QVector<TrainingWorkflowProfile>& trainingWorkflowProfiles()
{
    static const QVector<TrainingWorkflowProfile> profiles{
        yoloProfile(QStringLiteral("ultralytics_yolo_detect"),
            QStringLiteral("detection/ultralytics_trainer.py"),
            {QStringLiteral("当前  训练交付模型包仅声明已验证的 AITrain ONNX Runtime 路线。"),
                QStringLiteral("公开或生成数据集的工作流证据不代表客户域精度。")}),
        yoloProfile(QStringLiteral("ultralytics_yolo_segment"),
            QStringLiteral("segmentation/ultralytics_trainer.py"),
            {QStringLiteral("当前  训练交付模型包仅声明已验证的 AITrain ONNX Runtime 路线。"),
                QStringLiteral("实例分割官方评估由 Ultralytics val() 提供；公开或生成数据集证据不代表客户域精度。")}),
        yoloProfile(QStringLiteral("ultralytics_yolo_obb"),
            QStringLiteral("obb/ultralytics_trainer.py"),
            {QStringLiteral("OBB v1 产品部署仅支持 AITrain ONNX Runtime，不声明 NCNN 或 TensorRT 运行时。"),
                QStringLiteral("公开 DOTA 或生成数据集的工作流证据不代表客户域精度。")}),
        smpProfile(),
        anomalibProfile(QStringLiteral("anomalib_patchcore"), {}),
        anomalibProfile(QStringLiteral("anomalib_efficientad"), {
            QStringLiteral("EfficientAD 强制 batchSize=1，并要求显式可用的 imagenetDir；缺失时必须报告 blocked。")}),
        paddleOcrProfile(false),
        paddleOcrProfile(true)
    };
    return profiles;
}

bool resolveTrainingWorkflowProfile(const QString& trainingBackend,
    TrainingWorkflowProfile* result,
    QString* error)
{
    if (!result) {
        if (error) {
            *error = QStringLiteral("解析训练 Workflow Profile 需要输出对象。");
        }
        return false;
    }
    const QString backend = normalizedBackend(trainingBackend);
    for (const TrainingWorkflowProfile& candidate : trainingWorkflowProfiles()) {
        if (candidate.trainingBackend == backend) {
            *result = candidate;
            if (error) {
                error->clear();
            }
            return true;
        }
    }
    if (error) {
        *error = trainingBackend.trimmed().isEmpty()
            ? QStringLiteral("训练 Workflow backend 不能为空。")
            : QStringLiteral("未注册训练 Workflow Profile：%1").arg(trainingBackend.trimmed());
    }
    return false;
}

bool hasTrainingWorkflowProfile(const QString& trainingBackend)
{
    TrainingWorkflowProfile ignored;
    return resolveTrainingWorkflowProfile(trainingBackend, &ignored, nullptr);
}

} // namespace aitrain
