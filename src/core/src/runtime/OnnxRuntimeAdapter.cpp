#include "aitrain/runtime/OnnxRuntimeAdapter.h"

#include "aitrain/core/VisionModelRuntime.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

namespace aitrain {
namespace {

RuntimeOperationResult failed(RuntimeStatus status, const QString& message)
{
    return {status, message, {}};
}

RuntimeOperationResult readyModel(const RuntimeModelLocation& model)
{
    RuntimeOperationResult result = validateRuntimeModel(model, QStringLiteral("aitrain_onnxruntime"));
    if (result.status != RuntimeStatus::Available) return result;
    if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
        return failed(RuntimeStatus::DependencyMissing, QStringLiteral("ONNX Runtime 未在当前产品构建中启用。"));
    }
    return result;
}

bool supportedContract(const ModelManifest& manifest)
{
    return (manifest.modelFamily == QStringLiteral("yolo_detection") && manifest.decoder == QStringLiteral("yolo_detection_v8"))
        || (manifest.modelFamily == QStringLiteral("yolo_segmentation") && manifest.decoder == QStringLiteral("yolo_segmentation_v8"))
        || (manifest.modelFamily == QStringLiteral("yolo_obb") && manifest.decoder == QStringLiteral("yolo_obb_v8"))
        || (manifest.modelFamily == QStringLiteral("semantic_segmentation") && manifest.decoder == QStringLiteral("smp_semantic_segmentation"));
}

RuntimeOperationResult validatedModel(const RuntimeModelLocation& model)
{
    RuntimeOperationResult result = readyModel(model);
    if (result.status != RuntimeStatus::Available) return result;
    if (!supportedContract(model.manifest)) {
        return failed(RuntimeStatus::RuntimeNotImplemented,
            QStringLiteral("ONNX Runtime 未实现 Manifest 声明的 modelFamily/decoder 组合：%1/%2。")
                .arg(model.manifest.modelFamily, model.manifest.decoder));
    }
    return result;
}

bool requestedImage(const QJsonObject& request, QString* imagePath, RuntimeOperationResult* error)
{
    const QString value = request.value(QStringLiteral("imagePath")).toString();
    if (value.isEmpty()) {
        if (error) *error = failed(RuntimeStatus::ArtifactIncompatible, QStringLiteral("ONNX Runtime 推理请求缺少 imagePath。"));
        return false;
    }
    if (imagePath) *imagePath = value;
    return true;
}

bool requestedOutput(const QJsonObject& request, QString* outputPath, RuntimeOperationResult* error)
{
    const QString value = QDir::cleanPath(request.value(QStringLiteral("outputPath")).toString());
    if (value.isEmpty()) {
        if (error) *error = failed(RuntimeStatus::ArtifactIncompatible, QStringLiteral("ONNX Runtime 推理请求缺少 outputPath。"));
        return false;
    }
    if (outputPath) *outputPath = value;
    return true;
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
    if (!file.open(QIODevice::WriteOnly) || file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) < 0 || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入  ONNX 推理预测结果：%1").arg(path);
        return false;
    }
    return true;
}

bool writeOverlay(const QString& path, const QImage& overlay, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || !overlay.save(&file, "PNG") || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入  ONNX 推理 overlay：%1").arg(path);
        return false;
    }
    return true;
}

} // namespace

QString OnnxRuntimeAdapter::runtimeRoute() const
{
    return QStringLiteral("aitrain_onnxruntime");
}

RuntimeOperationResult OnnxRuntimeAdapter::probe(const RuntimeModelLocation& model) const
{
    return validatedModel(model);
}

RuntimeOperationResult OnnxRuntimeAdapter::validateModel(const RuntimeModelLocation& model) const
{
    return validatedModel(model);
}

RuntimeOperationResult OnnxRuntimeAdapter::infer(const RuntimeModelLocation& model, const QJsonObject& request) const
{
    RuntimeOperationResult result = validatedModel(model);
    if (result.status != RuntimeStatus::Available) return result;
    const QString modelPath = QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath);
    const QString family = model.manifest.modelFamily;
    const QString decoder = model.manifest.decoder;
    const bool benchmarkOnly = request.value(QStringLiteral("_benchmarkOnly")).toBool(false);
    QString imagePath;
    QString outputPath;
    if (!requestedImage(request, &imagePath, &result) || !requestedOutput(request, &outputPath, &result)) return result;
    if (!benchmarkOnly && !QDir().mkpath(outputPath)) {
        return failed(RuntimeStatus::ArtifactIncompatible, QStringLiteral("无法创建  ONNX 推理输出目录：%1").arg(outputPath));
    }
    QString error;
    QJsonArray predictions;
    QImage overlay;
    QString taskType;
    int predictionCount = 0;
    const aitrain::DetectionInferenceOptions options = inferenceOptions(request);
    if (family == QStringLiteral("yolo_detection")) {
        taskType = QStringLiteral("detection");
        const QVector<aitrain::DetectionPrediction> values = aitrain::predictDetectionOnnxRuntime(
            modelPath, imagePath, model.manifest.classNames, options, &error);
        for (const aitrain::DetectionPrediction& value : values) predictions.append(aitrain::detectionPredictionToJson(value));
        if (!benchmarkOnly) overlay = aitrain::renderDetectionPredictions(imagePath, values, &error);
        predictionCount = values.size();
    } else if (family == QStringLiteral("yolo_segmentation")) {
        taskType = QStringLiteral("segmentation");
        const QVector<aitrain::SegmentationPrediction> values = aitrain::predictSegmentationOnnxRuntime(
            modelPath, imagePath, model.manifest.classNames, options, &error);
        for (const aitrain::SegmentationPrediction& value : values) predictions.append(aitrain::segmentationPredictionToJson(value));
        if (!benchmarkOnly) overlay = aitrain::renderSegmentationPredictions(imagePath, values, &error);
        predictionCount = values.size();
    } else if (family == QStringLiteral("yolo_obb")) {
        taskType = QStringLiteral("obb_detection");
        const QVector<aitrain::ObbPrediction> values = aitrain::predictObbOnnxRuntime(
            modelPath, imagePath, model.manifest.classNames, options, &error);
        for (const aitrain::ObbPrediction& value : values) predictions.append(aitrain::obbPredictionToJson(value));
        if (!benchmarkOnly) overlay = aitrain::renderObbPredictions(imagePath, values, &error);
        predictionCount = values.size();
    } else {
        taskType = QStringLiteral("semantic_segmentation");
        const aitrain::SemanticSegmentationPrediction prediction = aitrain::predictSemanticSegmentationOnnxRuntime(modelPath, imagePath, &error);
        predictions.append(aitrain::semanticSegmentationPredictionToJson(prediction));
        if (!benchmarkOnly) overlay = aitrain::renderSemanticSegmentationPrediction(imagePath, prediction, &error);
        for (auto it = prediction.pixelCounts.constBegin(); it != prediction.pixelCounts.constEnd(); ++it) {
            if (it.key() != QStringLiteral("0") && it.value().toDouble() > 0.0) ++predictionCount;
        }
    }
    if (!error.isEmpty() || (!benchmarkOnly && overlay.isNull())) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            error.isEmpty() ? QStringLiteral(" ONNX 推理未生成可用 overlay。") : error);
    }
    result.details.insert(QStringLiteral("predictionCount"), predictionCount);
    result.details.insert(QStringLiteral("taskType"), taskType);
    result.details.insert(QStringLiteral("runtime"), runtimeRoute());
    if (benchmarkOnly) {
        return result;
    }
    const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
    const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
    const QJsonObject report{
        {QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("modelPackageId"), model.manifest.modelPackageId.toString()},
        {QStringLiteral("modelFamily"), family},
        {QStringLiteral("taskType"), taskType},
        {QStringLiteral("runtime"), runtimeRoute()},
        {QStringLiteral("imagePath"), imagePath},
        {QStringLiteral("postprocess"), QJsonObject{{QStringLiteral("confidenceThreshold"), options.confidenceThreshold},
             {QStringLiteral("iouThreshold"), options.iouThreshold}, {QStringLiteral("maxDetections"), options.maxDetections}}},
        {QStringLiteral("predictions"), predictions}};
    if (!writeJson(predictionsPath, report, &error) || !writeOverlay(overlayPath, overlay, &error)) {
        QFile::remove(predictionsPath);
        QFile::remove(overlayPath);
        return failed(RuntimeStatus::ArtifactIncompatible, error);
    }
    result.details.insert(QStringLiteral("predictionsPath"), predictionsPath);
    result.details.insert(QStringLiteral("overlayPath"), overlayPath);
    return result;
}

RuntimeOperationResult OnnxRuntimeAdapter::deploymentValidate(const RuntimeModelLocation& model, const QJsonObject& request) const
{
    return infer(model, request);
}

} // namespace aitrain
