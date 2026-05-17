#include "DatasetConversionInternal.h"

#include "aitrain/core/DatasetValidators.h"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QDomDocument>
#include <QFile>
#include <QFileInfo>
#include <QJsonObject>
#include <QMap>
#include <QSet>
#include <QStringList>
#include <QtMath>
#include <QVector>

namespace aitrain {
namespace dataset_conversion_detail {

struct VocObject {
    QString name;
    double xmin = 0.0;
    double ymin = 0.0;
    double xmax = 0.0;
    double ymax = 0.0;
};

struct VocAnnotation {
    QString filename;
    QString imagePath;
    int width = 0;
    int height = 0;
    QVector<VocObject> objects;
};

bool parseVocXml(const QString& xmlPath, VocAnnotation& annotation, QString* error = nullptr)
{
    annotation = VocAnnotation();

    QFile file(xmlPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot read VOC XML: %1").arg(xmlPath);
        }
        return false;
    }

    QDomDocument document;
    QString parseError;
    int parseLine = 0;
    int parseColumn = 0;
    if (!document.setContent(&file, &parseError, &parseLine, &parseColumn)) {
        if (error) {
            *error = QStringLiteral("VOC XML parse failed at line %1:%2: %3").arg(parseLine).arg(parseColumn).arg(parseError);
        }
        return false;
    }

    const QDomElement annotationElement = document.documentElement();
    if (annotationElement.isNull() || annotationElement.tagName() != QLatin1String("annotation")) {
        if (error) {
            *error = QStringLiteral("VOC XML root is not <annotation>.");
        }
        return false;
    }

    annotation.filename = annotationElement.firstChildElement(QStringLiteral("filename")).text().trimmed();
    annotation.imagePath = annotationElement.firstChildElement(QStringLiteral("path")).text().trimmed();

    const QDomElement sizeElement = annotationElement.firstChildElement(QStringLiteral("size"));
    if (!sizeElement.isNull()) {
        annotation.width = sizeElement.firstChildElement(QStringLiteral("width")).text().toInt();
        annotation.height = sizeElement.firstChildElement(QStringLiteral("height")).text().toInt();
    }

    const QDomNodeList objectNodes = annotationElement.elementsByTagName(QStringLiteral("object"));
    for (int index = 0; index < objectNodes.size(); ++index) {
        const QDomElement objectElement = objectNodes.at(index).toElement();
        if (objectElement.isNull()) {
            continue;
        }

        VocObject object;
        object.name = objectElement.firstChildElement(QStringLiteral("name")).text().trimmed();
        const QDomElement bboxElement = objectElement.firstChildElement(QStringLiteral("bndbox"));
        if (!bboxElement.isNull()) {
            object.xmin = bboxElement.firstChildElement(QStringLiteral("xmin")).text().toDouble();
            object.ymin = bboxElement.firstChildElement(QStringLiteral("ymin")).text().toDouble();
            object.xmax = bboxElement.firstChildElement(QStringLiteral("xmax")).text().toDouble();
            object.ymax = bboxElement.firstChildElement(QStringLiteral("ymax")).text().toDouble();
        }
        if (!object.name.isEmpty()) {
            annotation.objects.append(object);
        }
    }

    return true;
}

QString resolveVocImagePath(const QString& sourceDir, const VocAnnotation& annotation)
{
    const QStringList candidates = {
        annotation.imagePath,
        annotation.filename,
        annotation.imagePath.isEmpty() ? QString() : QDir(sourceDir).filePath(annotation.imagePath),
        QDir(sourceDir).filePath(annotation.filename),
        QDir(sourceDir).filePath(QStringLiteral("..")).append(QDir::separator()).append(QStringLiteral("JPEGImages"))
            .append(QDir::separator()).append(annotation.filename),
        QDir(sourceDir).filePath(QStringLiteral("..")).append(QDir::separator()).append(QStringLiteral("images"))
            .append(QDir::separator()).append(annotation.filename),
    };

    for (const QString& candidate : candidates) {
        if (candidate.isEmpty()) {
            continue;
        }
        const QString cleaned = QDir::cleanPath(candidate);
        const QFileInfo fileInfo(cleaned);
        if (fileInfo.exists() && fileInfo.isFile()) {
            return cleaned;
        }
    }
    return {};
}

bool copyImageToSplit(const QString& sourceImagePath,
    const QString& outputPath,
    const QString& split,
    QString* copiedRelativePath,
    QString* error = nullptr);

bool copyImageToTrain(const QString& sourceImagePath,
    const QString& outputPath,
    QString* copiedRelativePath,
    QString* error = nullptr);

bool copyImageToVal(const QString& sourceImagePath,
    const QString& outputPath,
    QString* copiedRelativePath,
    QString* error = nullptr);

bool sameCanonicalFile(const QString& leftPath, const QString& rightPath);

QString normalizedOutputTargetKey(const QString& path);

bool containsPlannedOutputTarget(const QStringList& targets, const QSet<QString>& plannedTargets, QString* duplicateTarget);

void insertPlannedOutputTargets(const QStringList& targets, QSet<QString>* plannedTargets);

DatasetConversionResult convertVoc(const DatasetConversionRequest& request, const CancellationCallback& shouldCancel)
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
    const bool targetDetection = (result.targetFormat == QStringLiteral("yolo_detection"));

    if (!targetDetection) {
        if (result.targetFormat == QStringLiteral("yolo_segmentation")) {
            result.errorCode = QStringLiteral("unsupported_target_format");
            result.errorMessage = QStringLiteral("Pascal VOC XML conversion supports YOLO detection only.");
        } else {
            result.errorCode = QStringLiteral("unsupported_target_format");
            result.errorMessage = QStringLiteral("Unsupported dataset target format: %1").arg(request.targetFormat);
        }
        return result;
    }

    QString sourceDir = result.sourcePath;
    const QFileInfo sourceInfo(sourceDir);
    QStringList xmlPaths;
    if (sourceInfo.isDir()) {
        QDirIterator it(sourceDir, QStringList{QStringLiteral("*.xml")}, QDir::Files);
        while (it.hasNext()) {
            if (conversionCanceled(shouldCancel)) {
                return canceledConversionResult(request);
            }
            xmlPaths.append(it.next());
        }
    } else if (sourceInfo.isFile() && sourceInfo.suffix().compare(QStringLiteral("xml"), Qt::CaseInsensitive) == 0) {
        xmlPaths.append(sourceInfo.absoluteFilePath());
        sourceDir = sourceInfo.absolutePath();
    } else {
        result.errorCode = QStringLiteral("source_read_failed");
        result.errorMessage = QStringLiteral("Cannot read VOC XML annotations from: %1").arg(result.sourcePath);
        return result;
    }

    result.sampleCount = xmlPaths.size();
    result.sourceValidation.insert(QStringLiteral("xmlFiles"), xmlPaths.size());

    QVector<VocAnnotation> annotations;
    for (const QString& xmlPath : xmlPaths) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        VocAnnotation annotation;
        QString parseError;
        if (!parseVocXml(xmlPath, annotation, &parseError)) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_xml"), xmlPath, QString(), QString(),
                parseError.isEmpty() ? QStringLiteral("VOC XML annotation cannot be parsed.") : parseError));
            ++result.skippedSampleCount;
            continue;
        }
        if (annotation.width <= 0 || annotation.height <= 0 || annotation.filename.isEmpty()) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_metadata"), xmlPath, annotation.filename,
                QString(), QStringLiteral("VOC XML is missing valid filename, width or height.")));
            ++result.skippedSampleCount;
            continue;
        }
        if (annotation.objects.isEmpty()) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("no_annotations"), xmlPath, annotation.filename,
                QString(), QStringLiteral("VOC XML contains no object annotations.")));
            ++result.skippedSampleCount;
            continue;
        }
        annotations.append(annotation);
        result.annotationCount += annotation.objects.size();
    }

    QStringList classNames;
    for (const VocAnnotation& annotation : annotations) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        for (const VocObject& object : annotation.objects) {
            const QString className = object.name.trimmed();
            if (!className.isEmpty()) {
                classNames.append(className);
            }
        }
    }
    classNames.removeDuplicates();
    classNames.sort();
    int nextClassId = 0;
    QMap<QString, int> classNameToClassId;
    for (int index = 0; index < classNames.size(); ++index) {
        const QString className = classNames[index];
        if (className.isEmpty()) {
            continue;
        }
        classNameToClassId.insert(className, nextClassId);
        result.classMap.insert(QString::number(nextClassId), className);
        ++nextClassId;
    }

    QMap<QString, QStringList> labelsByImage;
    for (const VocAnnotation& annotation : annotations) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        const QString imagePath = resolveVocImagePath(sourceDir, annotation);
        if (imagePath.isEmpty()) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_not_found"), annotation.filename,
                annotation.filename, QString(), QStringLiteral("VOC image file not found.")));
            ++result.skippedSampleCount;
            continue;
        }

        for (const VocObject& object : annotation.objects) {
            const QString className = object.name.trimmed();
            if (!classNameToClassId.contains(className) || annotation.width <= 0 || annotation.height <= 0) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_annotation"), imagePath, annotation.filename, className,
                    QStringLiteral("VOC annotation object is invalid.")));
                ++result.skippedAnnotationCount;
                continue;
            }
            const double boxW = object.xmax - object.xmin;
            const double boxH = object.ymax - object.ymin;
            if (boxW <= 0.0 || boxH <= 0.0) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), imagePath, annotation.filename, className,
                    QStringLiteral("VOC bounding box dimensions are invalid.")));
                ++result.skippedAnnotationCount;
                continue;
            }
            const double cx = object.xmin + boxW / 2.0;
            const double cy = object.ymin + boxH / 2.0;
            const QString row = QStringLiteral("%1 %2 %3 %4 %5")
                .arg(classNameToClassId.value(className))
                .arg(yoloNumber(cx / annotation.width))
                .arg(yoloNumber(cy / annotation.height))
                .arg(yoloNumber(boxW / annotation.width))
                .arg(yoloNumber(boxH / annotation.height));
            labelsByImage[imagePath].append(row);
            ++result.convertedAnnotationCount;
        }
    }

    QDir().mkpath(result.outputPath);
    QSet<QString> plannedOutputTargets;
    int convertedSamples = 0;
    for (auto it = labelsByImage.constBegin(); it != labelsByImage.constEnd(); ++it) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        const QString fileName = QFileInfo(it.key()).fileName();
        const QString labelFileName = QFileInfo(fileName).completeBaseName() + QStringLiteral(".txt");
        const QString trainLabelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/train/%1").arg(labelFileName));
        const QString valLabelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/val/%1").arg(labelFileName));
        QStringList plannedTargets{trainLabelPath, valLabelPath};
        if (copyImages) {
            plannedTargets.append(QDir(result.outputPath).filePath(QStringLiteral("images/train/%1").arg(fileName)));
            plannedTargets.append(QDir(result.outputPath).filePath(QStringLiteral("images/val/%1").arg(fileName)));
        }
        QString duplicateTarget;
        if (containsPlannedOutputTarget(plannedTargets, plannedOutputTargets, &duplicateTarget)) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("duplicate_output_target"), it.key(), fileName,
                duplicateTarget, QStringLiteral("Skipping VOC sample because it would overwrite an earlier YOLO output target.")));
            result.convertedAnnotationCount -= it.value().size();
            result.skippedAnnotationCount += it.value().size();
            continue;
        }

        if (copyImages) {
            QString copyError;
            if (!copyImageToTrain(it.key(), result.outputPath, nullptr, &copyError)) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), it.key(), fileName, QString(), copyError));
                ++result.skippedSampleCount;
                continue;
            }
            if (!copyImageToVal(it.key(), result.outputPath, nullptr, &copyError)) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), it.key(), fileName, QStringLiteral("val"), copyError));
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
        result.errorMessage = QStringLiteral("VOC XML dataset did not contain convertible annotations.");
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
    result.targetValidation = validateYoloDetectionDataset(result.outputPath);
    result.ok = result.targetValidation.ok;
    if (!result.ok) {
        result.errorCode = QStringLiteral("target_validation_failed");
        result.errorMessage = QStringLiteral("Converted YOLO detection dataset failed validation.");
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