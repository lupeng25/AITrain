#include "aitrain/core/DatasetConversion.h"

#include "DatasetConversionInternal.h"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>

namespace aitrain {
using namespace dataset_conversion_detail;

DatasetConversionResult convertDataset(const DatasetConversionRequest& request)
{
    return convertDataset(request, CancellationCallback());
}

DatasetConversionResult convertDataset(const DatasetConversionRequest& request, const CancellationCallback& shouldCancel)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);

    const QString sourceFormat = normalizedFormat(request.sourceFormat);
    const QString targetFormat = normalizedFormat(request.targetFormat);
    if (conversionCanceled(shouldCancel)) {
        return canceledConversionResult(request);
    }

    if (sourceFormat != QStringLiteral("coco_json")
        && sourceFormat != QStringLiteral("voc_xml")
        && sourceFormat != QStringLiteral("yolo_detection")
        && sourceFormat != QStringLiteral("yolo_segmentation")
        && sourceFormat != QStringLiteral("labelme_json")) {
        result.errorCode = QStringLiteral("unsupported_source_format");
        result.errorMessage = QStringLiteral("Unsupported dataset source format: %1").arg(request.sourceFormat);
        return result;
    }

    if (sourceFormat == QStringLiteral("coco_json")) {
        if (targetFormat == QStringLiteral("yolo_detection") || targetFormat == QStringLiteral("yolo_segmentation")) {
            return convertCoco(request, shouldCancel);
        }
        result.errorCode = QStringLiteral("unsupported_target_format");
        result.errorMessage = QStringLiteral("Unsupported dataset target format: %1").arg(request.targetFormat);
        return result;
    }
    if (sourceFormat == QStringLiteral("voc_xml")) {
        return convertVoc(request, shouldCancel);
    }
    if (sourceFormat == QStringLiteral("yolo_detection")) {
        if (targetFormat == QStringLiteral("coco_json")) {
            return convertYoloToCoco(request, false, shouldCancel);
        }
        if (targetFormat == QStringLiteral("voc_xml")) {
            return convertYoloToVoc(request, shouldCancel);
        }
        result.errorCode = QStringLiteral("unsupported_target_format");
        result.errorMessage = QStringLiteral("Unsupported dataset target format: %1").arg(request.targetFormat);
        return result;
    }
    if (sourceFormat == QStringLiteral("yolo_segmentation")) {
        if (targetFormat == QStringLiteral("coco_json")) {
            return convertYoloToCoco(request, true, shouldCancel);
        }
        if (targetFormat == QStringLiteral("voc_xml")) {
            result.errorCode = QStringLiteral("unsupported_target_format");
            result.errorMessage = QStringLiteral("YOLO segmentation to Pascal VOC XML is not supported.");
            return result;
        }
        result.errorCode = QStringLiteral("unsupported_target_format");
        result.errorMessage = QStringLiteral("Unsupported dataset target format: %1").arg(request.targetFormat);
        return result;
    }

    result.errorCode = QStringLiteral("not_implemented");
    result.errorMessage = QStringLiteral("Dataset conversion parser is not implemented yet.");
    return result;
}

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
    object.insert(QStringLiteral("conversionMatrixVersion"), conversionMatrixVersion);
    object.insert(QStringLiteral("copyImages"), copyImages);
    object.insert(QStringLiteral("imagePolicy"), imagePolicy);
    object.insert(QStringLiteral("outputFiles"), outputFiles);
    object.insert(QStringLiteral("splitCounts"), splitCounts);
    object.insert(QStringLiteral("sourceValidation"), sourceValidation);
    object.insert(QStringLiteral("targetValidation"), targetValidation.toJson());

    QJsonArray issueArray;
    for (const DatasetConversionIssue& issue : issues) {
        issueArray.append(issue.toJson());
    }
    object.insert(QStringLiteral("issues"), issueArray);
    return object;
}

} // namespace aitrain
