#pragma once

#include "aitrain/core/VisionPostprocess.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace aitrain {

struct TensorRtBackendStatus {
    bool sdkAvailable = false;
    bool dependenciesAvailable = false;
    bool hardwareDetected = false;
    bool hardwareSupported = false;
    bool exportAvailable = false;
    bool inferenceAvailable = false;
    int computeCapabilityMajor = 0;
    int computeCapabilityMinor = 0;
    QString sdkStatus;
    QString engineBuildStatus;
    QString runtimeInferenceStatus;
    QString engineBuildMessage;
    QString runtimeInferenceMessage;
    QString status;
    QString message;

    QJsonObject toJson() const;
};

struct NcnnBackendStatus {
    bool sdkAvailable = false;
    bool inferenceAvailable = false;
    QString status;
    QString message;

    QJsonObject toJson() const;
};

bool isOnnxRuntimeInferenceAvailable();
QString inferOnnxModelFamily(const QString& onnxPath);
QString inferOnnxModelFamily(const QString& onnxPath, QString* warning);
QJsonObject detectionTrainingBackendStatus();
TensorRtBackendStatus tensorRtBackendStatus();
bool isTensorRtInferenceAvailable();
NcnnBackendStatus ncnnBackendStatus();
bool isNcnnInferenceAvailable();
QString inferNcnnModelFamily(const QString& paramPath);
bool validateNcnnRuntimeModel(
    const QString& paramPath,
    const QJsonObject& runtimeOptions,
    QString* error = nullptr);

QVector<DetectionPrediction> predictDetectionOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<DetectionPrediction> predictDetectionOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const QStringList& classNames,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<SegmentationPrediction> predictSegmentationOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<SegmentationPrediction> predictSegmentationOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const QStringList& classNames,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<ObbPrediction> predictObbOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<ObbPrediction> predictObbOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const QStringList& classNames,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

SemanticSegmentationPrediction predictSemanticSegmentationOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    QString* error = nullptr);

OcrRecPrediction predictOcrRecOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    QString* error = nullptr);

QVector<OcrDetPrediction> predictOcrDetOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const OcrDetPostprocessOptions& options = OcrDetPostprocessOptions(),
    QString* error = nullptr);

QVector<DetectionPrediction> predictDetectionTensorRt(
    const QString& enginePath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<DetectionPrediction> predictDetectionNcnnRuntime(
    const QString& paramPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<DetectionPrediction> predictDetectionNcnnRuntime(
    const QString& paramPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    const QJsonObject& runtimeOptions,
    QString* error = nullptr);

QVector<SegmentationPrediction> predictSegmentationNcnnRuntime(
    const QString& paramPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<SegmentationPrediction> predictSegmentationNcnnRuntime(
    const QString& paramPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    const QJsonObject& runtimeOptions,
    QString* error = nullptr);

} // namespace aitrain
