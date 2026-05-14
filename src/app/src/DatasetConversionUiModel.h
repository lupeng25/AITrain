#pragma once

#include <QString>
#include <QStringList>

namespace aitrain_app {

struct DatasetConversionForm {
    QString sourceFormat;
    QString targetFormat;
    QString inputPath;
    QString outputPath;
    bool workerRunning = false;
};

struct DatasetConversionValidation {
    bool ok = false;
    QString summary;
    QString sourceFormatError;
    QString targetFormatError;
    QString inputPathError;
    QString outputPathError;
    QStringList messages;
};

QString datasetConversionFormatLabel(const QString& format);
QStringList supportedDatasetConversionSourceFormats();
QStringList supportedDatasetConversionTargets(const QString& sourceFormat);
bool isSupportedDatasetConversionPair(const QString& sourceFormat, const QString& targetFormat);
QString normalizedDatasetConversionPath(const QString& path);
DatasetConversionValidation validateDatasetConversionForm(const DatasetConversionForm& form);

} // namespace aitrain_app
