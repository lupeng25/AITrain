#pragma once

#include "aitrain/core/OcrRecDataset.h"

#include <QSize>
#include <QString>
#include <functional>

namespace aitrain {

struct OcrRecTrainingOptions {
    int epochs = 1;
    int batchSize = 1;
    QSize imageSize = QSize(100, 32);
    double learningRate = 0.05;
    int maxTextLength = 25;
    QString labelFilePath;
    QString dictionaryFilePath;
    QString outputPath;
};

struct OcrRecTrainingMetrics {
    int epoch = 0;
    int step = 0;
    int totalSteps = 0;
    double loss = 0.0;
    double ctcLoss = 0.0;
    double accuracy = 0.0;
    double editDistance = 0.0;
};

struct OcrRecTrainingResult {
    bool ok = false;
    QString error;
    QString checkpointPath;
    QString previewPath;
    int steps = 0;
    double finalLoss = 0.0;
    double accuracy = 0.0;
    double editDistance = 0.0;
};

using OcrRecTrainingCallback = std::function<bool(const OcrRecTrainingMetrics&)>;

OcrRecTrainingResult trainOcrRecBaseline(
    const QString& datasetPath,
    const OcrRecTrainingOptions& options,
    const OcrRecTrainingCallback& callback = OcrRecTrainingCallback());

} // namespace aitrain
