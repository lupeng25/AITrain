#include "DatasetConversionInternal.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSet>

namespace aitrain {
namespace dataset_conversion_detail {

QString normalizedFormat(const QString& value)
{
    return value.trimmed().toLower();
}

bool writeTextFile(const QString& path, const QString& text, QString* error = nullptr)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot write text file: %1").arg(path);
        }
        return false;
    }
    if (!text.isEmpty() && file.write(text.toUtf8()) != text.toUtf8().size()) {
        if (error) {
            *error = QStringLiteral("Failed writing text file: %1").arg(path);
        }
        return false;
    }
    return true;
}

bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error = nullptr)
{
    const QByteArray payload = QJsonDocument(object).toJson(QJsonDocument::Compact);
    return writeTextFile(path, QString::fromUtf8(payload), error);
}

DatasetConversionIssue issue(const QString& severity,
    const QString& code,
    const QString& sourceFile,
    const QString& imagePath,
    const QString& category,
    const QString& message)
{
    DatasetConversionIssue conversionIssue;
    conversionIssue.severity = severity;
    conversionIssue.code = code;
    conversionIssue.sourceFile = sourceFile;
    conversionIssue.imagePath = imagePath;
    conversionIssue.category = category;
    conversionIssue.message = message;
    return conversionIssue;
}

bool conversionCanceled(const CancellationCallback& shouldCancel)
{
    return isCancellationRequested(shouldCancel);
}

DatasetConversionResult canceledConversionResult(const DatasetConversionRequest& request)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);
    result.errorCode = QStringLiteral("canceled");
    result.errorMessage = QStringLiteral("Canceled by user");
    return result;
}

void markCanceled(DatasetConversionResult* result)
{
    if (!result) {
        return;
    }
    result->ok = false;
    result->errorCode = QStringLiteral("canceled");
    result->errorMessage = QStringLiteral("Canceled by user");
}

QString yoloNumber(double value)
{
    return QString::number(value, 'f', 6);
}

bool copyImageToSplit(const QString& sourceImagePath,
    const QString& outputPath,
    const QString& split,
    QString* copiedRelativePath,
    QString* error)
{
    if (!QFileInfo::exists(sourceImagePath)) {
        if (error) {
            *error = QStringLiteral("Image file does not exist: %1").arg(sourceImagePath);
        }
        return false;
    }

    const QDir outputRoot(outputPath);
    const QString imageName = QFileInfo(sourceImagePath).fileName();
    const QString relativePath = QStringLiteral("images/%1/%2").arg(split, imageName);
    const QString targetPath = outputRoot.filePath(relativePath);

    outputRoot.mkpath(QStringLiteral("images/%1").arg(split));
    if (QFileInfo::exists(targetPath) && sameCanonicalFile(sourceImagePath, targetPath)) {
        if (copiedRelativePath) {
            *copiedRelativePath = relativePath;
        }
        return true;
    }
    if (QFileInfo::exists(targetPath)) {
        if (error) {
            *error = QStringLiteral("Refusing to overwrite existing image target: %1").arg(targetPath);
        }
        return false;
    }
    if (!QFile::copy(sourceImagePath, targetPath)) {
        if (error) {
            *error = QStringLiteral("Cannot copy image: %1").arg(sourceImagePath);
        }
        return false;
    }
    if (copiedRelativePath) {
        *copiedRelativePath = relativePath;
    }
    return true;
}

bool copyImageToTrain(const QString& sourceImagePath,
    const QString& outputPath,
    QString* copiedRelativePath,
    QString* error)
{
    return copyImageToSplit(sourceImagePath, outputPath, QStringLiteral("train"), copiedRelativePath, error);
}

bool copyImageToVal(const QString& sourceImagePath,
    const QString& outputPath,
    QString* copiedRelativePath,
    QString* error)
{
    return copyImageToSplit(sourceImagePath, outputPath, QStringLiteral("val"), copiedRelativePath, error);
}


} // namespace dataset_conversion_detail
} // namespace aitrain