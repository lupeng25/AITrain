#pragma once

#include "aitrain/core/VisionModelExport.h"
#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/core/VisionPostprocess.h"

#include <QJsonObject>
#include <QSize>
#include <QString>
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

using DetectionTrainingCallback = std::function<bool(const DetectionTrainingMetrics&)>;

} // namespace aitrain
