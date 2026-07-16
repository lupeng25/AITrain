#include "aitrain/v2/OnnxRuntimeAdapterV2.h"

#include "aitrain/core/VisionModelRuntime.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

namespace aitrain::v2 {
namespace {

RuntimeOperationResultV2 failed(RuntimeStatusV2 status, const QString& message)
{
    return {status, message, {}};
}

RuntimeOperationResultV2 readyModel(const RuntimeModelLocationV2& model)
{
    RuntimeOperationResultV2 result = validateRuntimeModelV2(model, QStringLiteral("aitrain_onnxruntime"));
    if (result.status != RuntimeStatusV2::Available) return result;
    if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
        return failed(RuntimeStatusV2::DependencyMissing, QStringLiteral("ONNX Runtime 未在当前产品构建中启用。"));
    }
    return result;
}

bool supportedContract(const ModelManifestV2& manifest)
{
    return (manifest.modelFamily == QStringLiteral("yolo_detection") && manifest.decoder == QStringLiteral("yolo_detection_v8"))
        || (manifest.modelFamily == QStringLiteral("yolo_segmentation") && manifest.decoder == QStringLiteral("yolo_segmentation_v8"))
        || (manifest.modelFamily == QStringLiteral("yolo_obb") && manifest.decoder == QStringLiteral("yolo_obb_v8"))
        || (manifest.modelFamily == QStringLiteral("semantic_segmentation") && manifest.decoder == QStringLiteral("smp_semantic_segmentation"));
}

RuntimeOperationResultV2 validatedModel(const RuntimeModelLocationV2& model)
{
    RuntimeOperationResultV2 result = readyModel(model);
    if (result.status != RuntimeStatusV2::Available) return result;
    if (!supportedContract(model.manifest)) {
        return failed(RuntimeStatusV2::RuntimeNotImplemented,
            QStringLiteral("ONNX Runtime 未实现 Manifest 声明的 modelFamily/decoder 组合：%1/%2。")
                .arg(model.manifest.modelFamily, model.manifest.decoder));
    }
    return result;
}

bool requestedImage(const QJsonObject& request, QString* imagePath, RuntimeOperationResultV2* error)
{
    const QString value = request.value(QStringLiteral("imagePath")).toString();
    if (value.isEmpty()) {
        if (error) *error = failed(RuntimeStatusV2::ArtifactIncompatible, QStringLiteral("ONNX Runtime 推理请求缺少 imagePath。"));
        return false;
    }
    if (imagePath) *imagePath = value;
    return true;
}

bool requestedOutput(const QJsonObject& request, QString* outputPath, RuntimeOperationResultV2* error)
{
    const QString value = QDir::cleanPath(request.value(QStringLiteral("outputPath")).toString());
    if (value.isEmpty()) {
        if (error) *error = failed(RuntimeStatusV2::ArtifactIncompatible, QStringLiteral("ONNX Runtime 推理请求缺少 outputPath。"));
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
        if (error) *error = QStringLiteral("无法写入 V2 ONNX 推理预测结果：%1").arg(path);
        return false;
    }
    return true;
}

bool writeOverlay(const QString& path, const QImage& overlay, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || !overlay.save(&file, "PNG") || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入 V2 ONNX 推理 overlay：%1").arg(path);
        return false;
    }
    return true;
}

} // namespace

QString OnnxRuntimeAdapterV2::runtimeRoute() const
{
    return QStringLiteral("aitrain_onnxruntime");
}

RuntimeOperationResultV2 OnnxRuntimeAdapterV2::probe(const RuntimeModelLocationV2& model) const
{
    return validatedModel(model);
}

RuntimeOperationResultV2 OnnxRuntimeAdapterV2::validateModel(const RuntimeModelLocationV2& model) const
{
    return validatedModel(model);
}

RuntimeOperationResultV2 OnnxRuntimeAdapterV2::infer(const RuntimeModelLocationV2& model, const QJsonObject& request) const
{
    RuntimeOperationResultV2 result = validatedModel(model);
    if (result.status != RuntimeStatusV2::Available) return result;
    const QString modelPath = QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath);
    const QString family = model.manifest.modelFamily;
    const QString decoder = model.manifest.decoder;
    QString imagePath;
    QString outputPath;
    if (!requestedImage(request, &imagePath, &result) || !requestedOutput(request, &outputPath, &result)) return result;
    if (!QDir().mkpath(outputPath)) {
        return failed(RuntimeStatusV2::ArtifactIncompatible, QStringLiteral("无法创建 V2 ONNX 推理输出目录：%1").arg(outputPath));
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
        overlay = aitrain::renderDetectionPredictions(imagePath, values, &error);
        predictionCount = values.size();
    } else if (family == QStringLiteral("yolo_segmentation")) {
        taskType = QStringLiteral("segmentation");
        const QVector<aitrain::SegmentationPrediction> values = aitrain::predictSegmentationOnnxRuntime(
            modelPath, imagePath, model.manifest.classNames, options, &error);
        for (const aitrain::SegmentationPrediction& value : values) predictions.append(aitrain::segmentationPredictionToJson(value));
        overlay = aitrain::renderSegmentationPredictions(imagePath, values, &error);
        predictionCount = values.size();
    } else if (family == QStringLiteral("yolo_obb")) {
        taskType = QStringLiteral("obb_detection");
        const QVector<aitrain::ObbPrediction> values = aitrain::predictObbOnnxRuntime(
            modelPath, imagePath, model.manifest.classNames, options, &error);
        for (const aitrain::ObbPrediction& value : values) predictions.append(aitrain::obbPredictionToJson(value));
        overlay = aitrain::renderObbPredictions(imagePath, values, &error);
        predictionCount = values.size();
    } else {
        taskType = QStringLiteral("semantic_segmentation");
        const aitrain::SemanticSegmentationPrediction prediction = aitrain::predictSemanticSegmentationOnnxRuntime(modelPath, imagePath, &error);
        predictions.append(aitrain::semanticSegmentationPredictionToJson(prediction));
        overlay = aitrain::renderSemanticSegmentationPrediction(imagePath, prediction, &error);
        for (auto it = prediction.pixelCounts.constBegin(); it != prediction.pixelCounts.constEnd(); ++it) {
            if (it.key() != QStringLiteral("0") && it.value().toDouble() > 0.0) ++predictionCount;
        }
    }
    if (!error.isEmpty() || overlay.isNull()) {
        return failed(RuntimeStatusV2::ArtifactIncompatible,
            error.isEmpty() ? QStringLiteral("V2 ONNX 推理未生成可用 overlay。") : error);
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
        return failed(RuntimeStatusV2::ArtifactIncompatible, error);
    }
    result.details.insert(QStringLiteral("predictionCount"), predictionCount);
    result.details.insert(QStringLiteral("taskType"), taskType);
    result.details.insert(QStringLiteral("runtime"), runtimeRoute());
    result.details.insert(QStringLiteral("predictionsPath"), predictionsPath);
    result.details.insert(QStringLiteral("overlayPath"), overlayPath);
    return result;
}

RuntimeOperationResultV2 OnnxRuntimeAdapterV2::benchmark(const RuntimeModelLocationV2& model, const QJsonObject& request) const
{
    QElapsedTimer timer;
    timer.start();
    RuntimeOperationResultV2 result = infer(model, request);
    if (result.status == RuntimeStatusV2::Available) result.details.insert(QStringLiteral("elapsedMs"), static_cast<double>(timer.elapsed()));
    return result;
}

RuntimeOperationResultV2 OnnxRuntimeAdapterV2::deploymentValidate(const RuntimeModelLocationV2& model, const QJsonObject& request) const
{
    return infer(model, request);
}

} // namespace aitrain::v2
