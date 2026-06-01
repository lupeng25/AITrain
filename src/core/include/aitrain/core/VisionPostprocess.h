#pragma once

#include "aitrain/core/DetectionDataset.h"

#include <QImage>
#include <QJsonObject>
#include <QPointF>
#include <QSize>
#include <QString>
#include <QVector>

namespace aitrain {

struct DetectionInferenceOptions {
    double confidenceThreshold = 0.0;
    double iouThreshold = 0.45;
    int maxDetections = 100;
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

} // namespace aitrain
