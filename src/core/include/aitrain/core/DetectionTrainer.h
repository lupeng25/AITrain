#pragma once

#include "aitrain/core/DetectionDataset.h"

#include <QImage>
#include <QJsonObject>
#include <QPointF>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QVector>
#include <functional>

namespace aitrain {

struct DetectionTrainingOptions {
    int epochs = 1;
    int batchSize = 1;
    QSize imageSize = QSize(320, 320);
    double learningRate = 0.05;
    int gridSize = 4;
    bool horizontalFlip = false;
    bool colorJitter = false;
    QString trainingBackend;
    QString outputPath;
    QString resumeCheckpointPath;
};

struct DetectionTrainingMetrics {
    int epoch = 0;
    int step = 0;
    int totalSteps = 0;
    double loss = 0.0;
    double objectnessLoss = 0.0;
    double classLoss = 0.0;
    double boxLoss = 0.0;
};

struct DetectionTrainingResult {
    bool ok = false;
    QString error;
    QString checkpointPath;
    QString trainingBackend;
    QString modelFamily;
    bool scaffold = false;
    QJsonObject modelArchitecture;
    int steps = 0;
    double finalLoss = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double map50 = 0.0;
};

struct DetectionInferenceOptions {
    double confidenceThreshold = 0.0;
    double iouThreshold = 0.45;
    int maxDetections = 100;
};

struct DetectionExportResult {
    bool ok = false;
    QString error;
    QString exportPath;
    QString reportPath;
    QString sourceCheckpointPath;
    QString format;
    QJsonObject config;
};

struct TensorRtBackendStatus {
    bool sdkAvailable = false;
    bool exportAvailable = false;
    bool inferenceAvailable = false;
    QString status;
    QString message;

    QJsonObject toJson() const;
};

struct DetectionBaselineCheckpoint {
    QString type;
    int checkpointSchemaVersion = 1;
    QString trainingBackend;
    QString modelFamily;
    bool scaffold = false;
    QJsonObject modelArchitecture;
    QJsonObject phase8;
    QString datasetPath;
    QSize imageSize;
    int gridSize = 1;
    int featureCount = 0;
    int steps = 0;
    double finalLoss = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double map50 = 0.0;
    QStringList classNames;
    QVector<double> classLogits;
    QVector<double> objectnessWeights;
    QVector<double> classWeights;
    QVector<double> boxWeights;
    DetectionBox priorBox;
};

struct DetectionPrediction {
    DetectionBox box;
    QString className;
    double objectness = 1.0;
    double confidence = 0.0;
};

struct SegmentationPrediction {
    DetectionPrediction detection;
    QImage mask;
    double maskArea = 0.0;
    double maskThreshold = 0.5;
};

struct OcrRecPrediction {
    QString text;
    double confidence = 0.0;
    QVector<int> tokens;
    int blankIndex = 0;
};

struct OcrDetPostprocessOptions {
    double binaryThreshold = 0.3;
    double boxThreshold = 0.5;
    int minArea = 3;
    int maxDetections = 100;
};

struct OcrDetPrediction {
    QVector<QPointF> polygon;
    DetectionBox box;
    double confidence = 0.0;
    int pixelArea = 0;
};

using DetectionTrainingCallback = std::function<bool(const DetectionTrainingMetrics&)>;

DetectionTrainingResult trainDetectionBaseline(
    const QString& datasetPath,
    const DetectionTrainingOptions& options,
    const DetectionTrainingCallback& callback = DetectionTrainingCallback());

bool loadDetectionBaselineCheckpoint(
    const QString& checkpointPath,
    DetectionBaselineCheckpoint* checkpoint,
    QString* error = nullptr);

QVector<DetectionPrediction> predictDetectionBaseline(
    const DetectionBaselineCheckpoint& checkpoint,
    const QString& imagePath,
    QString* error = nullptr);

QVector<DetectionPrediction> predictDetectionBaseline(
    const DetectionBaselineCheckpoint& checkpoint,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

bool isOnnxRuntimeInferenceAvailable();
QString inferOnnxModelFamily(const QString& onnxPath);
QJsonObject detectionTrainingBackendStatus();
TensorRtBackendStatus tensorRtBackendStatus();
bool isTensorRtInferenceAvailable();

QVector<DetectionPrediction> predictDetectionOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error = nullptr);

QVector<SegmentationPrediction> predictSegmentationOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
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

QVector<DetectionPrediction> postProcessDetectionPredictions(
    const QVector<DetectionPrediction>& predictions,
    const DetectionInferenceOptions& options);

QVector<OcrDetPrediction> postProcessPaddleOcrDetDbMap(
    const QVector<float>& probabilityMap,
    const QSize& mapSize,
    const QSize& sourceSize,
    const OcrDetPostprocessOptions& options = OcrDetPostprocessOptions(),
    QString* error = nullptr);

QJsonObject detectionPredictionToJson(const DetectionPrediction& prediction);
QJsonObject segmentationPredictionToJson(const SegmentationPrediction& prediction);
QJsonObject ocrRecPredictionToJson(const OcrRecPrediction& prediction);
QJsonObject ocrDetPredictionToJson(const OcrDetPrediction& prediction);

QImage renderDetectionPredictions(
    const QString& imagePath,
    const QVector<DetectionPrediction>& predictions,
    QString* error = nullptr);

QImage renderSegmentationPredictions(
    const QString& imagePath,
    const QVector<SegmentationPrediction>& predictions,
    QString* error = nullptr);

QImage renderOcrRecPrediction(
    const QString& imagePath,
    const OcrRecPrediction& prediction,
    QString* error = nullptr);

QImage renderOcrDetPredictions(
    const QString& imagePath,
    const QVector<OcrDetPrediction>& predictions,
    QString* error = nullptr);

DetectionExportResult exportDetectionCheckpoint(
    const QString& checkpointPath,
    const QString& outputPath,
    const QString& format = QStringLiteral("tiny_detector_json"));

} // namespace aitrain
