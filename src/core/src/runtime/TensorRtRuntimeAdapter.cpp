#include "aitrain/runtime/TensorRtRuntimeAdapter.h"

#include "aitrain/core/VisionModelRuntime.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

namespace aitrain {
namespace {

RuntimeOperationResult failed(RuntimeStatus status, const QString& message, const QJsonObject& details = {})
{
    return {status, message, details};
}

bool supportedContract(const ModelManifest& manifest)
{
    return (manifest.modelFamily == QStringLiteral("yolo_detection")
               && manifest.decoder == QStringLiteral("tensorrt_yolo_detection_v8"))
        || (manifest.modelFamily == QStringLiteral("yolo_segmentation")
               && manifest.decoder == QStringLiteral("tensorrt_yolo_segmentation_v8"));
}

RuntimeOperationResult backendProbe()
{
    const aitrain::TensorRtBackendStatus backend = aitrain::tensorRtBackendStatus();
    const QJsonObject details{{QStringLiteral("backend"), backend.toJson()}};
    if (!backend.sdkAvailable) return failed(RuntimeStatus::SdkMissing, backend.runtimeInferenceMessage, details);
    if (backend.status == QStringLiteral("dependency_missing")) {
        return failed(RuntimeStatus::DependencyMissing, backend.runtimeInferenceMessage, details);
    }
    if (!backend.dependenciesAvailable) {
        if (backend.status == QStringLiteral("hardware_unsupported")) {
            return failed(RuntimeStatus::HardwareUnsupported, backend.message, details);
        }
        return failed(RuntimeStatus::DependencyMissing, backend.runtimeInferenceMessage, details);
    }
    if (!backend.hardwareSupported) return failed(RuntimeStatus::HardwareUnsupported, backend.runtimeInferenceMessage, details);
    if (!backend.inferenceAvailable) return failed(RuntimeStatus::RuntimeNotImplemented, backend.runtimeInferenceMessage, details);
    return {RuntimeStatus::Available, QStringLiteral("TensorRT runtime 可用。"), details};
}

aitrain::DetectionInferenceOptions inferenceOptions(const QJsonObject& request)
{
    aitrain::DetectionInferenceOptions options;
    const QJsonObject supplied = request.value(QStringLiteral("options")).toObject();
    options.confidenceThreshold = supplied.value(QStringLiteral("confidenceThreshold")).toDouble(options.confidenceThreshold);
    options.iouThreshold = supplied.value(QStringLiteral("iouThreshold")).toDouble(options.iouThreshold);
    options.maxDetections = supplied.value(QStringLiteral("maxDetections")).toInt(options.maxDetections);
    return options;
}

bool writeJson(const QString& path, const QJsonObject& object, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)
        || file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) < 0
        || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入  TensorRT 推理预测结果：%1").arg(path);
        return false;
    }
    return true;
}

bool writeOverlay(const QString& path, const QImage& overlay, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || !overlay.save(&file, "PNG") || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入  TensorRT 推理 overlay：%1").arg(path);
        return false;
    }
    return true;
}

} // namespace

QString TensorRtRuntimeAdapter::runtimeRoute() const
{
    return QStringLiteral("aitrain_tensorrt");
}

RuntimeOperationResult TensorRtRuntimeAdapter::validateModel(const RuntimeModelLocation& model) const
{
    RuntimeOperationResult result = validateRuntimeModel(model, runtimeRoute());
    if (result.status != RuntimeStatus::Available) return result;
    const QString entry = model.manifest.artifactEntryPath;
    if (model.manifest.artifactFormat != QStringLiteral("tensorrt_engine")
        || (!entry.endsWith(QStringLiteral(".engine"), Qt::CaseInsensitive)
            && !entry.endsWith(QStringLiteral(".plan"), Qt::CaseInsensitive))) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("TensorRT  只接受 artifactFormat=tensorrt_engine 且入口为 .engine/.plan 的模型包。"));
    }
    if (!supportedContract(model.manifest)) {
        return failed(RuntimeStatus::RuntimeNotImplemented,
            QStringLiteral("TensorRT  未实现 Manifest 声明的 modelFamily/decoder：%1/%2。")
                .arg(model.manifest.modelFamily, model.manifest.decoder));
    }
    if (model.manifest.inputs.size() != 1 || model.manifest.inputs.constFirst().shape.size() != 4
        || model.manifest.outputs.isEmpty()) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("TensorRT engine Manifest 缺少明确的 input/output tensor 合同。"));
    }
    result.details.insert(QStringLiteral("decoder"), model.manifest.decoder);
    result.details.insert(QStringLiteral("enginePath"),
        QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath));
    return result;
}

RuntimeOperationResult TensorRtRuntimeAdapter::probe(const RuntimeModelLocation& model) const
{
    const RuntimeOperationResult validation = validateModel(model);
    if (validation.status != RuntimeStatus::Available) return validation;
    return backendProbe();
}

RuntimeOperationResult TensorRtRuntimeAdapter::infer(const RuntimeModelLocation& model, const QJsonObject& request) const
{
    RuntimeOperationResult result = probe(model);
    if (result.status != RuntimeStatus::Available) return result;
    if (model.manifest.modelFamily != QStringLiteral("yolo_detection")) {
        return failed(RuntimeStatus::RuntimeNotImplemented,
            QStringLiteral("TensorRT segmentation decoder 尚未实现。"));
    }
    const QString imagePath = request.value(QStringLiteral("imagePath")).toString();
    const QString outputPath = QDir::cleanPath(request.value(QStringLiteral("outputPath")).toString());
    if (imagePath.isEmpty() || outputPath.isEmpty()) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("TensorRT  推理请求缺少 imagePath 或 outputPath。"));
    }
    if (!QDir().mkpath(outputPath)) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("无法创建  TensorRT 推理输出目录：%1").arg(outputPath));
    }
    QString error;
    const QString enginePath = QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath);
    const QVector<aitrain::DetectionPrediction> values = aitrain::predictDetectionTensorRt(
        enginePath, imagePath, inferenceOptions(request), &error);
    if (!error.isEmpty()) return failed(RuntimeStatus::ArtifactIncompatible, error);
    const QImage overlay = aitrain::renderDetectionPredictions(imagePath, values, &error);
    if (!error.isEmpty() || overlay.isNull()) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            error.isEmpty() ? QStringLiteral(" TensorRT 推理未生成可用 overlay。") : error);
    }
    QJsonArray predictions;
    for (const aitrain::DetectionPrediction& value : values) predictions.append(aitrain::detectionPredictionToJson(value));
    const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
    const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
    const QJsonObject report{{QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("modelPackageId"), model.manifest.modelPackageId.toString()},
        {QStringLiteral("modelFamily"), model.manifest.modelFamily},
        {QStringLiteral("taskType"), QStringLiteral("detection")},
        {QStringLiteral("runtime"), runtimeRoute()}, {QStringLiteral("imagePath"), imagePath},
        {QStringLiteral("predictions"), predictions}};
    if (!writeJson(predictionsPath, report, &error) || !writeOverlay(overlayPath, overlay, &error)) {
        QFile::remove(predictionsPath);
        QFile::remove(overlayPath);
        return failed(RuntimeStatus::ArtifactIncompatible, error);
    }
    result.details.insert(QStringLiteral("predictionCount"), predictions.size());
    result.details.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
    result.details.insert(QStringLiteral("runtime"), runtimeRoute());
    result.details.insert(QStringLiteral("predictionsPath"), predictionsPath);
    result.details.insert(QStringLiteral("overlayPath"), overlayPath);
    return result;
}

RuntimeOperationResult TensorRtRuntimeAdapter::deploymentValidate(const RuntimeModelLocation& model, const QJsonObject& request) const
{
    return infer(model, request);
}

} // namespace aitrain
