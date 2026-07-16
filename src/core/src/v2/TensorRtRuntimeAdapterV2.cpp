#include "aitrain/v2/TensorRtRuntimeAdapterV2.h"

#include "aitrain/core/VisionModelRuntime.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

namespace aitrain::v2 {
namespace {

RuntimeOperationResultV2 failed(RuntimeStatusV2 status, const QString& message, const QJsonObject& details = {})
{
    return {status, message, details};
}

bool supportedContract(const ModelManifestV2& manifest)
{
    return (manifest.modelFamily == QStringLiteral("yolo_detection")
               && manifest.decoder == QStringLiteral("tensorrt_yolo_detection_v8"))
        || (manifest.modelFamily == QStringLiteral("yolo_segmentation")
               && manifest.decoder == QStringLiteral("tensorrt_yolo_segmentation_v8"));
}

RuntimeOperationResultV2 backendProbe()
{
    const aitrain::TensorRtBackendStatus backend = aitrain::tensorRtBackendStatus();
    const QJsonObject details{{QStringLiteral("backend"), backend.toJson()}};
    if (!backend.sdkAvailable) return failed(RuntimeStatusV2::SdkMissing, backend.runtimeInferenceMessage, details);
    if (backend.status == QStringLiteral("dependency_missing")) {
        return failed(RuntimeStatusV2::DependencyMissing, backend.runtimeInferenceMessage, details);
    }
    if (!backend.dependenciesAvailable) {
        if (backend.status == QStringLiteral("hardware_unsupported")) {
            return failed(RuntimeStatusV2::HardwareUnsupported, backend.message, details);
        }
        return failed(RuntimeStatusV2::DependencyMissing, backend.runtimeInferenceMessage, details);
    }
    if (!backend.hardwareSupported) return failed(RuntimeStatusV2::HardwareUnsupported, backend.runtimeInferenceMessage, details);
    if (!backend.inferenceAvailable) return failed(RuntimeStatusV2::RuntimeNotImplemented, backend.runtimeInferenceMessage, details);
    return {RuntimeStatusV2::Available, QStringLiteral("TensorRT runtime 可用。"), details};
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
        if (error) *error = QStringLiteral("无法写入 V2 TensorRT 推理预测结果：%1").arg(path);
        return false;
    }
    return true;
}

bool writeOverlay(const QString& path, const QImage& overlay, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || !overlay.save(&file, "PNG") || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入 V2 TensorRT 推理 overlay：%1").arg(path);
        return false;
    }
    return true;
}

} // namespace

QString TensorRtRuntimeAdapterV2::runtimeRoute() const
{
    return QStringLiteral("aitrain_tensorrt");
}

RuntimeOperationResultV2 TensorRtRuntimeAdapterV2::validateModel(const RuntimeModelLocationV2& model) const
{
    RuntimeOperationResultV2 result = validateRuntimeModelV2(model, runtimeRoute());
    if (result.status != RuntimeStatusV2::Available) return result;
    const QString entry = model.manifest.artifactEntryPath;
    if (model.manifest.artifactFormat != QStringLiteral("tensorrt_engine")
        || (!entry.endsWith(QStringLiteral(".engine"), Qt::CaseInsensitive)
            && !entry.endsWith(QStringLiteral(".plan"), Qt::CaseInsensitive))) {
        return failed(RuntimeStatusV2::ArtifactIncompatible,
            QStringLiteral("TensorRT V2 只接受 artifactFormat=tensorrt_engine 且入口为 .engine/.plan 的模型包。"));
    }
    if (!supportedContract(model.manifest)) {
        return failed(RuntimeStatusV2::RuntimeNotImplemented,
            QStringLiteral("TensorRT V2 未实现 Manifest 声明的 modelFamily/decoder：%1/%2。")
                .arg(model.manifest.modelFamily, model.manifest.decoder));
    }
    if (model.manifest.inputs.size() != 1 || model.manifest.inputs.constFirst().shape.size() != 4
        || model.manifest.outputs.isEmpty()) {
        return failed(RuntimeStatusV2::ArtifactIncompatible,
            QStringLiteral("TensorRT engine Manifest 缺少明确的 input/output tensor 合同。"));
    }
    result.details.insert(QStringLiteral("decoder"), model.manifest.decoder);
    result.details.insert(QStringLiteral("enginePath"),
        QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath));
    return result;
}

RuntimeOperationResultV2 TensorRtRuntimeAdapterV2::probe(const RuntimeModelLocationV2& model) const
{
    const RuntimeOperationResultV2 validation = validateModel(model);
    if (validation.status != RuntimeStatusV2::Available) return validation;
    return backendProbe();
}

RuntimeOperationResultV2 TensorRtRuntimeAdapterV2::infer(const RuntimeModelLocationV2& model, const QJsonObject& request) const
{
    RuntimeOperationResultV2 result = probe(model);
    if (result.status != RuntimeStatusV2::Available) return result;
    if (model.manifest.modelFamily != QStringLiteral("yolo_detection")) {
        return failed(RuntimeStatusV2::RuntimeNotImplemented,
            QStringLiteral("TensorRT segmentation decoder 尚未实现。"));
    }
    const QString imagePath = request.value(QStringLiteral("imagePath")).toString();
    const QString outputPath = QDir::cleanPath(request.value(QStringLiteral("outputPath")).toString());
    if (imagePath.isEmpty() || outputPath.isEmpty()) {
        return failed(RuntimeStatusV2::ArtifactIncompatible,
            QStringLiteral("TensorRT V2 推理请求缺少 imagePath 或 outputPath。"));
    }
    if (!QDir().mkpath(outputPath)) {
        return failed(RuntimeStatusV2::ArtifactIncompatible,
            QStringLiteral("无法创建 V2 TensorRT 推理输出目录：%1").arg(outputPath));
    }
    QString error;
    const QString enginePath = QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath);
    const QVector<aitrain::DetectionPrediction> values = aitrain::predictDetectionTensorRt(
        enginePath, imagePath, inferenceOptions(request), &error);
    if (!error.isEmpty()) return failed(RuntimeStatusV2::ArtifactIncompatible, error);
    const QImage overlay = aitrain::renderDetectionPredictions(imagePath, values, &error);
    if (!error.isEmpty() || overlay.isNull()) {
        return failed(RuntimeStatusV2::ArtifactIncompatible,
            error.isEmpty() ? QStringLiteral("V2 TensorRT 推理未生成可用 overlay。") : error);
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
        return failed(RuntimeStatusV2::ArtifactIncompatible, error);
    }
    result.details.insert(QStringLiteral("predictionCount"), predictions.size());
    result.details.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
    result.details.insert(QStringLiteral("runtime"), runtimeRoute());
    result.details.insert(QStringLiteral("predictionsPath"), predictionsPath);
    result.details.insert(QStringLiteral("overlayPath"), overlayPath);
    return result;
}

RuntimeOperationResultV2 TensorRtRuntimeAdapterV2::benchmark(const RuntimeModelLocationV2& model, const QJsonObject& request) const
{
    QElapsedTimer timer;
    timer.start();
    RuntimeOperationResultV2 result = infer(model, request);
    if (result.status == RuntimeStatusV2::Available) result.details.insert(QStringLiteral("elapsedMs"), static_cast<double>(timer.elapsed()));
    return result;
}

RuntimeOperationResultV2 TensorRtRuntimeAdapterV2::deploymentValidate(const RuntimeModelLocationV2& model, const QJsonObject& request) const
{
    return infer(model, request);
}

} // namespace aitrain::v2
