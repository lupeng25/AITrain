#pragma once

#include "aitrain/core/DatasetConversion.h"

#include <QJsonObject>
#include <QSet>
#include <QString>
#include <QStringList>

namespace aitrain {
namespace dataset_conversion_detail {

QString normalizedFormat(const QString& value);
bool writeTextFile(const QString& path, const QString& text, QString* error);
bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error);
DatasetConversionIssue issue(
    const QString& severity,
    const QString& code,
    const QString& sourceFile,
    const QString& imagePath,
    const QString& category,
    const QString& message);
bool conversionCanceled(const CancellationCallback& shouldCancel);
DatasetConversionResult canceledConversionResult(const DatasetConversionRequest& request);
void markCanceled(DatasetConversionResult* result);
QString yoloNumber(double value);

bool copyImageToSplit(
    const QString& sourceImagePath,
    const QString& outputPath,
    const QString& split,
    QString* copiedRelativePath,
    QString* error);
bool copyImageToTrain(
    const QString& sourceImagePath,
    const QString& outputPath,
    QString* copiedRelativePath,
    QString* error);
bool copyImageToVal(
    const QString& sourceImagePath,
    const QString& outputPath,
    QString* copiedRelativePath,
    QString* error);
bool sameCanonicalFile(const QString& leftPath, const QString& rightPath);
QString normalizedOutputTargetKey(const QString& path);
bool containsPlannedOutputTarget(const QStringList& targets, const QSet<QString>& plannedTargets, QString* duplicateTarget);
void insertPlannedOutputTargets(const QStringList& targets, QSet<QString>* plannedTargets);

DatasetConversionResult convertVoc(const DatasetConversionRequest& request, const CancellationCallback& shouldCancel);
DatasetConversionResult convertCoco(const DatasetConversionRequest& request, const CancellationCallback& shouldCancel);
DatasetConversionResult convertYoloToCoco(
    const DatasetConversionRequest& request,
    bool segmentation,
    const CancellationCallback& shouldCancel);
DatasetConversionResult convertYoloToVoc(const DatasetConversionRequest& request, const CancellationCallback& shouldCancel);

} // namespace dataset_conversion_detail
} // namespace aitrain
