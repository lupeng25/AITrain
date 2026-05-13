#pragma once

#include "aitrain/core/PluginInterfaces.h"

#include <QJsonObject>
#include <QString>
#include <QVector>

namespace aitrain {

struct DatasetConversionIssue {
    QString severity;
    QString code;
    QString sourceFile;
    QString imagePath;
    QString category;
    QString message;

    QJsonObject toJson() const;
};

struct DatasetConversionRequest {
    QString sourcePath;
    QString sourceFormat;
    QString targetFormat;
    QString outputPath;
    QJsonObject options;
};

struct DatasetConversionResult {
    bool ok = false;
    QString errorCode;
    QString errorMessage;
    QString sourceFormat;
    QString targetFormat;
    QString sourcePath;
    QString outputPath;
    QString reportPath;
    QString validationReportPath;
    int sampleCount = 0;
    int convertedSampleCount = 0;
    int skippedSampleCount = 0;
    int annotationCount = 0;
    int convertedAnnotationCount = 0;
    int skippedAnnotationCount = 0;
    QJsonObject classMap;
    QVector<DatasetConversionIssue> issues;
    DatasetValidationResult targetValidation;

    QJsonObject toJson() const;
};

DatasetConversionResult convertDataset(const DatasetConversionRequest& request);

} // namespace aitrain
