#pragma once

#include "aitrain/core/SegmentationDataset.h"

#include <QSize>
#include <QString>
#include <functional>

namespace aitrain {

struct SegmentationTrainingOptions {
    int epochs = 1;
    int batchSize = 1;
    QSize imageSize = QSize(320, 320);
    double learningRate = 0.05;
    QString outputPath;
};

struct SegmentationTrainingMetrics {
    int epoch = 0;
    int step = 0;
    int totalSteps = 0;
    double loss = 0.0;
    double maskLoss = 0.0;
    double maskCoverage = 0.0;
};

struct SegmentationTrainingResult {
    bool ok = false;
    QString error;
    QString checkpointPath;
    QString previewPath;
    int steps = 0;
    double finalLoss = 0.0;
    double maskCoverage = 0.0;
};

using SegmentationTrainingCallback = std::function<bool(const SegmentationTrainingMetrics&)>;

SegmentationTrainingResult trainSegmentationBaseline(
    const QString& datasetPath,
    const SegmentationTrainingOptions& options,
    const SegmentationTrainingCallback& callback = SegmentationTrainingCallback());

} // namespace aitrain
