#pragma once

#include "aitrain/core/Cancellation.h"

#include <QJsonObject>
#include <QString>

namespace aitrain {

struct DetectionExportResult {
    bool ok = false;
    QString error;
    QString exportPath;
    QString reportPath;
    QString sourceCheckpointPath;
    QString format;
    QJsonObject config;
};

DetectionExportResult exportDetectionCheckpoint(
    const QString& checkpointPath,
    const QString& outputPath,
    const QString& format = QStringLiteral("onnx"));
DetectionExportResult exportDetectionCheckpoint(
    const QString& checkpointPath,
    const QString& outputPath,
    const QString& format,
    const CancellationCallback& shouldCancel);

} // namespace aitrain
