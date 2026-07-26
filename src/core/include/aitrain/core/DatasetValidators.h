#pragma once

#include "aitrain/core/DatasetValidation.h"

#include <QJsonArray>
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
    // 仅供 Dataset Driver 生成不可变 Split Plan；不暴露为产品输出。
    QJsonArray plannedFiles;
    QJsonObject toJson() const;
};

DatasetValidationResult validateYoloDetectionDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validateYoloSegmentationDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validateYoloObbDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validateSemanticSegmentationMaskDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validateAnomalyFolderDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validatePaddleOcrDetDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetValidationResult validatePaddleOcrRecDataset(const QString& datasetPath, const QJsonObject& options = {});
DatasetSplitResult splitYoloDetectionDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitYoloSegmentationDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitYoloObbDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitSemanticSegmentationMaskDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitAnomalyFolderDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitPaddleOcrDetDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});
DatasetSplitResult splitPaddleOcrRecDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options = {});

} // namespace aitrain
