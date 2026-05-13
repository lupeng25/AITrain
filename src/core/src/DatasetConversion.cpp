#include "aitrain/core/DatasetConversion.h"

#include <QDir>
#include <QJsonArray>

namespace aitrain {

QJsonObject DatasetConversionIssue::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("severity"), severity);
    object.insert(QStringLiteral("code"), code);
    object.insert(QStringLiteral("sourceFile"), sourceFile);
    object.insert(QStringLiteral("imagePath"), imagePath);
    object.insert(QStringLiteral("category"), category);
    object.insert(QStringLiteral("message"), message);
    return object;
}

QJsonObject DatasetConversionResult::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("ok"), ok);
    object.insert(QStringLiteral("errorCode"), errorCode);
    object.insert(QStringLiteral("errorMessage"), errorMessage);
    object.insert(QStringLiteral("sourceFormat"), sourceFormat);
    object.insert(QStringLiteral("targetFormat"), targetFormat);
    object.insert(QStringLiteral("sourcePath"), sourcePath);
    object.insert(QStringLiteral("outputPath"), outputPath);
    object.insert(QStringLiteral("reportPath"), reportPath);
    object.insert(QStringLiteral("validationReportPath"), validationReportPath);
    object.insert(QStringLiteral("sampleCount"), sampleCount);
    object.insert(QStringLiteral("convertedSampleCount"), convertedSampleCount);
    object.insert(QStringLiteral("skippedSampleCount"), skippedSampleCount);
    object.insert(QStringLiteral("annotationCount"), annotationCount);
    object.insert(QStringLiteral("convertedAnnotationCount"), convertedAnnotationCount);
    object.insert(QStringLiteral("skippedAnnotationCount"), skippedAnnotationCount);
    object.insert(QStringLiteral("classMap"), classMap);
    object.insert(QStringLiteral("targetValidation"), targetValidation.toJson());

    QJsonArray issueArray;
    for (const DatasetConversionIssue& issue : issues) {
        issueArray.append(issue.toJson());
    }
    object.insert(QStringLiteral("issues"), issueArray);
    return object;
}

DatasetConversionResult convertDataset(const DatasetConversionRequest& request)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);

    const QString sourceFormat = request.sourceFormat.trimmed().toLower();
    if (sourceFormat != QStringLiteral("coco_json")
        && sourceFormat != QStringLiteral("voc_xml")
        && sourceFormat != QStringLiteral("labelme_json")) {
        result.errorCode = QStringLiteral("unsupported_source_format");
        result.errorMessage = QStringLiteral("Unsupported dataset source format: %1").arg(request.sourceFormat);
        return result;
    }

    result.errorCode = QStringLiteral("not_implemented");
    result.errorMessage = QStringLiteral("Dataset conversion parser is not implemented yet.");
    return result;
}

} // namespace aitrain
