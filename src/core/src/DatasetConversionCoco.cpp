#include "DatasetConversionInternal.h"

#include "aitrain/core/DatasetValidators.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QMap>
#include <QSet>
#include <QStringList>
#include <QtMath>

namespace aitrain {
namespace dataset_conversion_detail {

DatasetConversionResult convertCoco(const DatasetConversionRequest& request, const CancellationCallback& shouldCancel)
{
    if (conversionCanceled(shouldCancel)) {
        return canceledConversionResult(request);
    }
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);
    result.conversionMatrixVersion = 2;
    result.copyImages = request.options.value(QStringLiteral("copyImages")).toBool(true);
    result.imagePolicy = result.copyImages ? QStringLiteral("copied") : QStringLiteral("referenced");

    const bool copyImages = result.copyImages;

    QFile file(result.sourcePath);
    if (!file.open(QIODevice::ReadOnly)) {
        result.errorCode = QStringLiteral("source_read_failed");
        result.errorMessage = QStringLiteral("Cannot read COCO JSON: %1").arg(result.sourcePath);
        return result;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        result.errorCode = QStringLiteral("source_parse_failed");
        result.errorMessage = QStringLiteral("Cannot parse COCO JSON %1: %2").arg(result.sourcePath, parseError.errorString());
        return result;
    }

    const QJsonObject root = document.object();
    const QJsonArray images = root.value(QStringLiteral("images")).toArray();
    const QJsonArray categories = root.value(QStringLiteral("categories")).toArray();
    const QJsonArray annotations = root.value(QStringLiteral("annotations")).toArray();

    result.sampleCount = images.size();
    result.annotationCount = annotations.size();
    result.sourceValidation.insert(QStringLiteral("images"), images.size());
    result.sourceValidation.insert(QStringLiteral("annotations"), annotations.size());
    result.sourceValidation.insert(QStringLiteral("categories"), categories.size());

    QMap<int, QJsonObject> imagesById;
    for (const QJsonValue& value : images) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        const QJsonObject image = value.toObject();
        imagesById.insert(image.value(QStringLiteral("id")).toInt(), image);
    }

    QMap<int, QString> categoryNames;
    for (const QJsonValue& value : categories) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        const QJsonObject category = value.toObject();
        categoryNames.insert(category.value(QStringLiteral("id")).toInt(), category.value(QStringLiteral("name")).toString().trimmed());
    }

    QMap<int, int> categoryIdToClassId;
    int nextClassId = 0;
    for (auto it = categoryNames.constBegin(); it != categoryNames.constEnd(); ++it) {
        if (it.value().isEmpty()) {
            continue;
        }
        categoryIdToClassId.insert(it.key(), nextClassId);
        result.classMap.insert(QString::number(nextClassId), it.value());
        ++nextClassId;
    }

    QMap<int, QStringList> labelsByImageId;
    for (const QJsonValue& value : annotations) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        const QJsonObject annotation = value.toObject();
        const int imageId = annotation.value(QStringLiteral("image_id")).toInt();
        const int categoryId = annotation.value(QStringLiteral("category_id")).toInt();
        if (!imagesById.contains(imageId) || !categoryIdToClassId.contains(categoryId)) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("missing_image_or_category"), result.sourcePath,
                QString(), QString::number(categoryId), QStringLiteral("COCO annotation references a missing image or category.")));
            ++result.skippedAnnotationCount;
            continue;
        }

        const QJsonObject image = imagesById.value(imageId);
        const double width = image.value(QStringLiteral("width")).toDouble();
        const double height = image.value(QStringLiteral("height")).toDouble();
        if (width <= 0.0 || height <= 0.0) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), result.sourcePath,
                image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                QStringLiteral("COCO bbox or image size is invalid.")));
            ++result.skippedAnnotationCount;
            continue;
        }

        const bool targetDetection = result.targetFormat == QStringLiteral("yolo_detection");
        if (targetDetection) {
            const QJsonArray bbox = annotation.value(QStringLiteral("bbox")).toArray();
            if (bbox.size() != 4) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), result.sourcePath,
                    imagesById.value(imageId).value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                    QStringLiteral("COCO bbox must contain four numbers.")));
                ++result.skippedAnnotationCount;
                continue;
            }
            const double x = bbox.at(0).toDouble();
            const double y = bbox.at(1).toDouble();
            const double w = bbox.at(2).toDouble();
            const double h = bbox.at(3).toDouble();
            if (width <= 0.0 || height <= 0.0 || w <= 0.0 || h <= 0.0) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), result.sourcePath,
                    image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                    QStringLiteral("COCO bbox or image size is invalid.")));
                ++result.skippedAnnotationCount;
                continue;
            }
            const QString row = QStringLiteral("%1 %2 %3 %4 %5")
                .arg(categoryIdToClassId.value(categoryId))
                .arg(yoloNumber((x + w / 2.0) / width))
                .arg(yoloNumber((y + h / 2.0) / height))
                .arg(yoloNumber(w / width))
                .arg(yoloNumber(h / height));
            labelsByImageId[imageId].append(row);
            ++result.convertedAnnotationCount;
            continue;
        }

        const QJsonValue segmentationValue = annotation.value(QStringLiteral("segmentation"));
        if (segmentationValue.isObject()) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("rle_not_supported"), result.sourcePath,
                image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                QStringLiteral("COCO RLE masks are not converted in this version.")));
            ++result.skippedAnnotationCount;
            continue;
        }

        const QJsonArray segments = segmentationValue.toArray();
        if (segments.isEmpty()) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), result.sourcePath,
                image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                QStringLiteral("COCO polygon segmentation is empty.")));
            ++result.skippedAnnotationCount;
            continue;
        }

        const bool nestedPolygons = segments.first().isArray();
        if (!nestedPolygons && (segments.size() < 6 || segments.size() % 2 != 0)) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), result.sourcePath,
                image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                QStringLiteral("COCO polygon must contain at least three x/y points.")));
            ++result.skippedAnnotationCount;
            continue;
        }

        auto appendPolygon = [&](const QJsonArray& polygon) {
            if (polygon.size() < 6 || polygon.size() % 2 != 0) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), result.sourcePath,
                    image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                    QStringLiteral("COCO polygon must contain at least three x/y points.")));
                ++result.skippedAnnotationCount;
                return;
            }
            QStringList parts;
            parts.append(QString::number(categoryIdToClassId.value(categoryId)));
            for (int i = 0; i + 1 < polygon.size(); i += 2) {
                parts.append(yoloNumber(polygon.at(i).toDouble() / width));
                parts.append(yoloNumber(polygon.at(i + 1).toDouble() / height));
            }
            labelsByImageId[imageId].append(parts.join(QLatin1Char(' ')));
            ++result.convertedAnnotationCount;
        };

        if (nestedPolygons) {
            bool convertedAny = false;
            for (const QJsonValue& segmentValue : segments) {
                if (conversionCanceled(shouldCancel)) {
                    markCanceled(&result);
                    return result;
                }
                const QJsonArray polygon = segmentValue.toArray();
                const int before = result.convertedAnnotationCount;
                appendPolygon(polygon);
                convertedAny = convertedAny || (result.convertedAnnotationCount > before);
            }
            if (!convertedAny) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), result.sourcePath,
                    image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                    QStringLiteral("COCO polygon segmentation is empty.")));
                ++result.skippedAnnotationCount;
            }
            continue;
        }

        const int before = result.convertedAnnotationCount;
        appendPolygon(segments);
        if (result.convertedAnnotationCount == before) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), result.sourcePath,
                image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId),
                QStringLiteral("COCO polygon must contain at least three x/y points.")));
            ++result.skippedAnnotationCount;
        }
    }

    QDir().mkpath(result.outputPath);
    QSet<QString> plannedOutputTargets;
    int convertedSamples = 0;
    for (auto it = labelsByImageId.constBegin(); it != labelsByImageId.constEnd(); ++it) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        const QJsonObject image = imagesById.value(it.key());
        const QString fileName = image.value(QStringLiteral("file_name")).toString();
        const QString sourceImagePath = QFileInfo(fileName).isAbsolute()
            ? fileName
            : QFileInfo(result.sourcePath).absoluteDir().filePath(fileName);
        const QString imageName = QFileInfo(sourceImagePath).fileName();
        const QString labelFileName = QFileInfo(imageName).completeBaseName() + QStringLiteral(".txt");
        const QString trainLabelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/train/%1").arg(labelFileName));
        const QString valLabelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/val/%1").arg(labelFileName));
        QStringList plannedTargets{trainLabelPath, valLabelPath};
        if (copyImages) {
            plannedTargets.append(QDir(result.outputPath).filePath(QStringLiteral("images/train/%1").arg(imageName)));
            plannedTargets.append(QDir(result.outputPath).filePath(QStringLiteral("images/val/%1").arg(imageName)));
        }
        QString duplicateTarget;
        if (containsPlannedOutputTarget(plannedTargets, plannedOutputTargets, &duplicateTarget)) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("duplicate_output_target"), result.sourcePath, fileName,
                duplicateTarget, QStringLiteral("Skipping COCO sample because it would overwrite an earlier YOLO output target.")));
            result.convertedAnnotationCount -= it.value().size();
            result.skippedAnnotationCount += it.value().size();
            continue;
        }

        if (copyImages) {
            QString copyError;
            if (!copyImageToTrain(sourceImagePath, result.outputPath, nullptr, &copyError)) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), result.sourcePath, fileName, QString(), copyError));
                ++result.skippedSampleCount;
                continue;
            }
            if (!copyImageToVal(sourceImagePath, result.outputPath, nullptr, &copyError)) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), result.sourcePath, fileName, QStringLiteral("val"), copyError));
                ++result.skippedSampleCount;
                continue;
            }
        }

        QString writeError;
        if (!writeTextFile(trainLabelPath, it.value().join(QLatin1Char('\n')) + QLatin1Char('\n'), &writeError)) {
            result.errorCode = QStringLiteral("output_write_failed");
            result.errorMessage = writeError;
            return result;
        }
        if (!writeTextFile(valLabelPath, it.value().join(QLatin1Char('\n')) + QLatin1Char('\n'), &writeError)) {
            result.errorCode = QStringLiteral("output_write_failed");
            result.errorMessage = writeError;
            return result;
        }
        insertPlannedOutputTargets(plannedTargets, &plannedOutputTargets);
        ++convertedSamples;
    }
    result.convertedSampleCount = convertedSamples;
    result.skippedSampleCount += qMax(0, result.sampleCount - convertedSamples);

    if (result.convertedSampleCount <= 0 || result.convertedAnnotationCount <= 0) {
        result.errorCode = QStringLiteral("no_convertible_samples");
        result.errorMessage = QStringLiteral("COCO dataset did not contain convertible annotations.");
        return result;
    }

    QStringList names;
    for (int index = 0; index < result.classMap.size(); ++index) {
        names.append(result.classMap.value(QString::number(index)).toString());
    }

    QString yaml = QStringLiteral("path: .\ntrain: images/train\nval: images/val\nnc: %1\nnames:\n").arg(names.size());
    for (int index = 0; index < names.size(); ++index) {
        yaml += QStringLiteral("  %1: %2\n").arg(index).arg(names.at(index));
    }
    QString writeError;
    const QString dataYamlPath = QDir(result.outputPath).filePath(QStringLiteral("data.yaml"));
    if (!writeTextFile(dataYamlPath, yaml, &writeError)) {
        result.errorCode = QStringLiteral("output_write_failed");
        result.errorMessage = writeError;
        return result;
    }
    result.outputFiles.insert(QStringLiteral("dataYaml"), dataYamlPath);
    result.outputFiles.insert(QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("images")));
    result.outputFiles.insert(QStringLiteral("labelsRoot"), QDir(result.outputPath).filePath(QStringLiteral("labels")));
    result.splitCounts.insert(QStringLiteral("train"), convertedSamples);
    result.splitCounts.insert(QStringLiteral("val"), convertedSamples);

    if (conversionCanceled(shouldCancel)) {
        markCanceled(&result);
        return result;
    }
    const bool targetSegmentation = result.targetFormat == QStringLiteral("yolo_segmentation");
    result.targetValidation = targetSegmentation
        ? validateYoloSegmentationDataset(result.outputPath)
        : validateYoloDetectionDataset(result.outputPath);
    result.ok = result.targetValidation.ok;
    if (!result.ok) {
        result.errorCode = QStringLiteral("target_validation_failed");
        result.errorMessage = targetSegmentation
            ? QStringLiteral("Converted YOLO segmentation dataset failed validation.")
            : QStringLiteral("Converted YOLO detection dataset failed validation.");
    }

    result.reportPath = QDir(result.outputPath).filePath(QStringLiteral("dataset_conversion_report.json"));
    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("convertedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    report.insert(QStringLiteral("artifacts"), QJsonObject{
        {QStringLiteral("dataYaml"), dataYamlPath},
        {QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("images"))},
        {QStringLiteral("labelsRoot"), QDir(result.outputPath).filePath(QStringLiteral("labels"))}
    });
    if (conversionCanceled(shouldCancel)) {
        markCanceled(&result);
        return result;
    }
    if (!writeJsonFile(result.reportPath, report, &writeError)) {
        result.ok = false;
        result.errorCode = QStringLiteral("report_write_failed");
        result.errorMessage = writeError;
    }

    return result;
}


} // namespace dataset_conversion_detail
} // namespace aitrain