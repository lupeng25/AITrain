#pragma once

#include "aitrain/core/PluginInterfaces.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace aitrain {

struct DatasetSplitResult {
    bool ok = true;
    int trainCount = 0;
    int valCount = 0;
    int testCount = 0;
    QString outputPath;
    QStringList errors;
    QStringList warnings;
    QJsonObject toJson() const;
};

DatasetValidationResult validateYoloDetectionDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validateYoloSegmentationDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validatePaddleOcrDetDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validatePaddleOcrRecDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetSplitResult splitYoloDetectionDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitYoloSegmentationDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitPaddleOcrDetDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitPaddleOcrRecDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});

} // namespace aitrain
