#include "aitrain/runtime/NcnnRuntimeAdapter.h"

#include "aitrain/core/VisionModelRuntime.h"

#include <QDir>
#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QSet>

namespace aitrain {
namespace {

RuntimeOperationResult failed(RuntimeStatus status, const QString& message)
{
    return {status, message, {}};
}

bool supportedContract(const ModelManifest& manifest)
{
    return (manifest.modelFamily == QStringLiteral("yolo_detection")
               && (manifest.decoder == QStringLiteral("ncnn_yolo_detection_ultralytics_v1")
                   || manifest.decoder == QStringLiteral("ncnn_yolo_detection_dfl_v1")))
        || (manifest.modelFamily == QStringLiteral("yolo_segmentation")
               && (manifest.decoder == QStringLiteral("ncnn_yolo_segmentation_ultralytics_v1")
                   || manifest.decoder == QStringLiteral("ncnn_yolo_segmentation_dfl_v1")));
}

QString binPath(const RuntimeModelLocation& model)
{
    const QFileInfo param(QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath));
    return param.absoluteDir().filePath(param.completeBaseName() + QStringLiteral(".bin"));
}

QJsonObject runtimeOptions(const RuntimeModelLocation& model)
{
    const TensorContract& input = model.manifest.inputs.constFirst();
    QJsonArray outputs;
    for (const TensorContract& output : model.manifest.outputs) outputs.append(output.name);
    QJsonObject options{
        {QStringLiteral("modelFamily"), model.manifest.modelFamily},
        {QStringLiteral("classNames"), QJsonArray::fromStringList(model.manifest.classNames)},
        {QStringLiteral("inputBlob"), input.name},
        {QStringLiteral("outputBlobs"), outputs},
        {QStringLiteral("inputSize"), QJsonObject{{QStringLiteral("width"), static_cast<int>(input.shape.constLast())},
             {QStringLiteral("height"), static_cast<int>(input.shape.at(input.shape.size() - 2))}}},
        {QStringLiteral("decoder"), model.manifest.decoder.contains(QStringLiteral("_dfl_"))
                ? QStringLiteral("dfl") : QStringLiteral("ultralytics_output")},
        {QStringLiteral("binPath"), binPath(model)}};
    const QJsonObject ncnn = model.manifest.postprocessing.value(QStringLiteral("ncnn")).toObject();
    if (ncnn.value(QStringLiteral("strides")).isArray()) options.insert(QStringLiteral("strides"), ncnn.value(QStringLiteral("strides")));
    if (ncnn.value(QStringLiteral("regMax")).isDouble()) options.insert(QStringLiteral("regMax"), ncnn.value(QStringLiteral("regMax")));
    return options;
}

QString fileSha256(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray bytes = file.read(1024 * 1024);
        if (bytes.isEmpty() && file.error() != QFile::NoError) return {};
        hash.addData(bytes);
    }
    return QString::fromLatin1(hash.result().toHex());
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
        if (error) *error = QStringLiteral("无法写入  NCNN 推理预测结果：%1").arg(path);
        return false;
    }
    return true;
}

bool writeOverlay(const QString& path, const QImage& overlay, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || !overlay.save(&file, "PNG") || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入  NCNN 推理 overlay：%1").arg(path);
        return false;
    }
    return true;
}

} // namespace

QString NcnnRuntimeAdapter::runtimeRoute() const
{
    return QStringLiteral("aitrain_ncnn");
}

RuntimeOperationResult NcnnRuntimeAdapter::validateModel(const RuntimeModelLocation& model) const
{
    RuntimeOperationResult result = validateRuntimeModel(model, runtimeRoute());
    if (result.status != RuntimeStatus::Available) return result;
    if (model.manifest.artifactFormat != QStringLiteral("ncnn")
        || !model.manifest.artifactEntryPath.endsWith(QStringLiteral(".param"), Qt::CaseInsensitive)) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("NCNN  只接受 artifactFormat=ncnn 且入口为 .param 的模型包。"));
    }
    if (!supportedContract(model.manifest)) {
        return failed(RuntimeStatus::RuntimeNotImplemented,
            QStringLiteral("NCNN  未实现 Manifest 声明的 modelFamily/decoder：%1/%2。")
                .arg(model.manifest.modelFamily, model.manifest.decoder));
    }
    if (model.manifest.inputs.size() != 1
        || model.manifest.inputs.constFirst().layout != QStringLiteral("NCHW")
        || model.manifest.inputs.constFirst().shape.size() != 4
        || model.manifest.inputs.constFirst().shape.at(2) <= 0 || model.manifest.inputs.constFirst().shape.at(3) <= 0) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("NCNN  要求 Manifest 明确一个静态 NCHW 输入及 blob name。"));
    }
    const bool dfl = model.manifest.decoder.contains(QStringLiteral("_dfl_"));
    const int minimumOutputs = model.manifest.modelFamily == QStringLiteral("yolo_segmentation")
        ? (dfl ? 3 : 2) : 1;
    if (model.manifest.outputs.size() < minimumOutputs) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("NCNN  Manifest 的输出 blob 合同不完整。"));
    }
    QSet<QString> blobNames{model.manifest.inputs.constFirst().name};
    for (const TensorContract& output : model.manifest.outputs) {
        if (output.name.trimmed().isEmpty() || blobNames.contains(output.name)) {
            return failed(RuntimeStatus::ArtifactIncompatible,
                QStringLiteral("NCNN  Manifest 的 input/output blob name 必须非空且唯一。"));
        }
        blobNames.insert(output.name);
    }
    const QJsonObject ncnnContract = model.manifest.postprocessing.value(QStringLiteral("ncnn")).toObject();
    if (dfl && (!ncnnContract.value(QStringLiteral("strides")).isArray()
        || ncnnContract.value(QStringLiteral("strides")).toArray().isEmpty()
        || ncnnContract.value(QStringLiteral("regMax")).toInt() <= 0)) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("NCNN DFL decoder 要求 Manifest 明确 strides 和 regMax。"));
    }
    const QFileInfo bin(binPath(model));
    if (!bin.exists() || !bin.isFile() || bin.isSymLink()) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("NCNN 模型包缺少与 .param 同名的安全 .bin 产物。"));
    }
    const QString expectedBinSha256 = ncnnContract.value(QStringLiteral("binSha256")).toString();
    if (!isSha256Hex(expectedBinSha256) || fileSha256(bin.absoluteFilePath()) != expectedBinSha256) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("NCNN .bin 的 SHA-256 缺失或与 Manifest 不一致。"));
    }
    const aitrain::NcnnBackendStatus backend = aitrain::ncnnBackendStatus();
    if (backend.sdkAvailable && backend.inferenceAvailable) {
        QString runtimeError;
        const QString paramPath = QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath);
        if (!aitrain::validateNcnnRuntimeModel(paramPath, runtimeOptions(model), &runtimeError)) {
            return failed(RuntimeStatus::ArtifactIncompatible,
                runtimeError.isEmpty() ? QStringLiteral("NCNN SDK 拒绝加载该模型 Artifact。") : runtimeError);
        }
    }
    result.details.insert(QStringLiteral("inputBlob"), model.manifest.inputs.constFirst().name);
    QJsonArray outputBlobs;
    for (const TensorContract& output : model.manifest.outputs) outputBlobs.append(output.name);
    result.details.insert(QStringLiteral("outputBlobs"), outputBlobs);
    result.details.insert(QStringLiteral("decoder"), model.manifest.decoder);
    result.details.insert(QStringLiteral("binPath"), bin.absoluteFilePath());
    return result;
}

RuntimeOperationResult NcnnRuntimeAdapter::probe(const RuntimeModelLocation& model) const
{
    RuntimeOperationResult result = validateModel(model);
    if (result.status != RuntimeStatus::Available) return result;
    const aitrain::NcnnBackendStatus backend = aitrain::ncnnBackendStatus();
    if (!backend.sdkAvailable) return failed(RuntimeStatus::SdkMissing, backend.message);
    if (!backend.inferenceAvailable) return failed(RuntimeStatus::RuntimeNotImplemented, backend.message);
    result.details.insert(QStringLiteral("backend"), backend.toJson());
    return result;
}

RuntimeOperationResult NcnnRuntimeAdapter::infer(const RuntimeModelLocation& model, const QJsonObject& request) const
{
    RuntimeOperationResult result = probe(model);
    if (result.status != RuntimeStatus::Available) return result;
    const bool benchmarkOnly = request.value(QStringLiteral("_benchmarkOnly")).toBool(false);
    const QString imagePath = request.value(QStringLiteral("imagePath")).toString();
    const QString outputPath = QDir::cleanPath(request.value(QStringLiteral("outputPath")).toString());
    if (imagePath.isEmpty() || outputPath.isEmpty()) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("NCNN  推理请求缺少 imagePath 或 outputPath。"));
    }
    if (!benchmarkOnly && !QDir().mkpath(outputPath)) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            QStringLiteral("无法创建  NCNN 推理输出目录：%1").arg(outputPath));
    }
    const QString paramPath = QDir(model.artifactDirectory).filePath(model.manifest.artifactEntryPath);
    const aitrain::DetectionInferenceOptions options = inferenceOptions(request);
    QString error;
    QJsonArray predictions;
    QImage overlay;
    QString taskType;
    if (model.manifest.modelFamily == QStringLiteral("yolo_detection")) {
        taskType = QStringLiteral("detection");
        const QVector<aitrain::DetectionPrediction> values = aitrain::predictDetectionNcnnRuntime(
            paramPath, imagePath, options, runtimeOptions(model), &error);
        for (const aitrain::DetectionPrediction& value : values) predictions.append(aitrain::detectionPredictionToJson(value));
        if (!benchmarkOnly) overlay = aitrain::renderDetectionPredictions(imagePath, values, &error);
    } else {
        taskType = QStringLiteral("segmentation");
        const QVector<aitrain::SegmentationPrediction> values = aitrain::predictSegmentationNcnnRuntime(
            paramPath, imagePath, options, runtimeOptions(model), &error);
        for (const aitrain::SegmentationPrediction& value : values) predictions.append(aitrain::segmentationPredictionToJson(value));
        if (!benchmarkOnly) overlay = aitrain::renderSegmentationPredictions(imagePath, values, &error);
    }
    if (!error.isEmpty() || (!benchmarkOnly && overlay.isNull())) {
        return failed(RuntimeStatus::ArtifactIncompatible,
            error.isEmpty() ? QStringLiteral(" NCNN 推理未生成可用 overlay。") : error);
    }
    result.details.insert(QStringLiteral("predictionCount"), predictions.size());
    result.details.insert(QStringLiteral("taskType"), taskType);
    result.details.insert(QStringLiteral("runtime"), runtimeRoute());
    if (benchmarkOnly) {
        return result;
    }
    const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
    const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
    const QJsonObject report{{QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("modelPackageId"), model.manifest.modelPackageId.toString()},
        {QStringLiteral("modelFamily"), model.manifest.modelFamily},
        {QStringLiteral("taskType"), taskType}, {QStringLiteral("runtime"), runtimeRoute()},
        {QStringLiteral("imagePath"), imagePath}, {QStringLiteral("predictions"), predictions}};
    if (!writeJson(predictionsPath, report, &error) || !writeOverlay(overlayPath, overlay, &error)) {
        QFile::remove(predictionsPath);
        QFile::remove(overlayPath);
        return failed(RuntimeStatus::ArtifactIncompatible, error);
    }
    result.details.insert(QStringLiteral("predictionsPath"), predictionsPath);
    result.details.insert(QStringLiteral("overlayPath"), overlayPath);
    return result;
}

RuntimeOperationResult NcnnRuntimeAdapter::deploymentValidate(const RuntimeModelLocation& model, const QJsonObject& request) const
{
    return infer(model, request);
}

} // namespace aitrain
