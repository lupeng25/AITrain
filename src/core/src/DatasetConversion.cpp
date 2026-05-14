#include "aitrain/core/DatasetConversion.h"

#include "aitrain/core/DatasetValidators.h"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QDomDocument>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QJsonArray>
#include <QMap>
#include <QStringList>
#include <QVector>

namespace aitrain {

namespace {

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

QString yoloNumber(double value)
{
    return QString::number(value, 'f', 6);
}

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

DatasetConversionResult convertVoc(const DatasetConversionRequest& request)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);

    const bool copyImages = request.options.value(QStringLiteral("copyImages")).toBool(true);
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

    QVector<VocAnnotation> annotations;
    for (const QString& xmlPath : xmlPaths) {
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
    int convertedSamples = 0;
    for (auto it = labelsByImage.constBegin(); it != labelsByImage.constEnd(); ++it) {
        const QString fileName = QFileInfo(it.key()).fileName();
        QString copiedTrainRelativePath;
        if (copyImages) {
            QString copyError;
            if (!copyImageToTrain(it.key(), result.outputPath, &copiedTrainRelativePath, &copyError)) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), it.key(), fileName, QString(), copyError));
                ++result.skippedSampleCount;
                continue;
            }
            if (!copyImageToVal(it.key(), result.outputPath, nullptr, &copyError)) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), it.key(), fileName, QStringLiteral("val"), copyError));
                ++result.skippedSampleCount;
                continue;
            }
        } else {
            copiedTrainRelativePath = fileName;
        }

        const QString labelFileName = QFileInfo(copiedTrainRelativePath).completeBaseName() + QStringLiteral(".txt");
        const QString trainLabelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/train/%1").arg(labelFileName));
        QString writeError;
        if (!writeTextFile(trainLabelPath, it.value().join(QLatin1Char('\n')) + QLatin1Char('\n'), &writeError)) {
            result.errorCode = QStringLiteral("output_write_failed");
            result.errorMessage = writeError;
            return result;
        }
        const QString valLabelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/val/%1").arg(labelFileName));
        if (!writeTextFile(valLabelPath, it.value().join(QLatin1Char('\n')) + QLatin1Char('\n'), &writeError)) {
            result.errorCode = QStringLiteral("output_write_failed");
            result.errorMessage = writeError;
            return result;
        }
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
    if (!writeTextFile(QDir(result.outputPath).filePath(QStringLiteral("data.yaml")), yaml, &writeError)) {
        result.errorCode = QStringLiteral("output_write_failed");
        result.errorMessage = writeError;
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
        {QStringLiteral("dataYaml"), QDir(result.outputPath).filePath(QStringLiteral("data.yaml"))},
        {QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("images"))},
        {QStringLiteral("labelsRoot"), QDir(result.outputPath).filePath(QStringLiteral("labels"))}
    });
    if (!writeJsonFile(result.reportPath, report, &writeError)) {
        result.ok = false;
        result.errorCode = QStringLiteral("report_write_failed");
        result.errorMessage = writeError;
    }

    return result;
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
    if (QFileInfo::exists(targetPath) && !QFile::remove(targetPath)) {
        if (error) {
            *error = QStringLiteral("Cannot clear existing image target: %1").arg(targetPath);
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

DatasetConversionResult convertCoco(const DatasetConversionRequest& request)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);

    const bool copyImages = request.options.value(QStringLiteral("copyImages")).toBool(true);

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

    QMap<int, QJsonObject> imagesById;
    for (const QJsonValue& value : images) {
        const QJsonObject image = value.toObject();
        imagesById.insert(image.value(QStringLiteral("id")).toInt(), image);
    }

    QMap<int, QString> categoryNames;
    for (const QJsonValue& value : categories) {
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
    int convertedSamples = 0;
    for (auto it = labelsByImageId.constBegin(); it != labelsByImageId.constEnd(); ++it) {
        const QJsonObject image = imagesById.value(it.key());
        const QString fileName = image.value(QStringLiteral("file_name")).toString();
        const QString sourceImagePath = QFileInfo(fileName).isAbsolute()
            ? fileName
            : QFileInfo(result.sourcePath).absoluteDir().filePath(fileName);

        QString copiedRelativePath;
        if (copyImages) {
            QString copyError;
            if (!copyImageToTrain(sourceImagePath, result.outputPath, &copiedRelativePath, &copyError)) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), result.sourcePath, fileName, QString(), copyError));
                ++result.skippedSampleCount;
                continue;
            }
            if (!copyImageToVal(sourceImagePath, result.outputPath, nullptr, &copyError)) {
                result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), result.sourcePath, fileName, QStringLiteral("val"), copyError));
                ++result.skippedSampleCount;
                continue;
            }
        } else {
            copiedRelativePath = fileName;
        }

        const QString labelFileName = QFileInfo(copiedRelativePath).completeBaseName() + QStringLiteral(".txt");
        const QString trainLabelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/train/%1").arg(labelFileName));
        QString writeError;
        if (!writeTextFile(trainLabelPath, it.value().join(QLatin1Char('\n')) + QLatin1Char('\n'), &writeError)) {
            result.errorCode = QStringLiteral("output_write_failed");
            result.errorMessage = writeError;
            return result;
        }
        const QString valLabelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/val/%1").arg(labelFileName));
        if (!writeTextFile(valLabelPath, it.value().join(QLatin1Char('\n')) + QLatin1Char('\n'), &writeError)) {
            result.errorCode = QStringLiteral("output_write_failed");
            result.errorMessage = writeError;
            return result;
        }
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
    if (!writeTextFile(QDir(result.outputPath).filePath(QStringLiteral("data.yaml")), yaml, &writeError)) {
        result.errorCode = QStringLiteral("output_write_failed");
        result.errorMessage = writeError;
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
        {QStringLiteral("dataYaml"), QDir(result.outputPath).filePath(QStringLiteral("data.yaml"))},
        {QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("images"))},
        {QStringLiteral("labelsRoot"), QDir(result.outputPath).filePath(QStringLiteral("labels"))}
    });
    if (!writeJsonFile(result.reportPath, report, &writeError)) {
        result.ok = false;
        result.errorCode = QStringLiteral("report_write_failed");
        result.errorMessage = writeError;
    }

    return result;
}

} // namespace

DatasetConversionResult convertDataset(const DatasetConversionRequest& request)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);

    const QString sourceFormat = normalizedFormat(request.sourceFormat);
    const QString targetFormat = normalizedFormat(request.targetFormat);

    if (sourceFormat != QStringLiteral("coco_json")
        && sourceFormat != QStringLiteral("voc_xml")
        && sourceFormat != QStringLiteral("labelme_json")) {
        result.errorCode = QStringLiteral("unsupported_source_format");
        result.errorMessage = QStringLiteral("Unsupported dataset source format: %1").arg(request.sourceFormat);
        return result;
    }

    if (sourceFormat == QStringLiteral("coco_json")) {
        if (targetFormat == QStringLiteral("yolo_detection") || targetFormat == QStringLiteral("yolo_segmentation")) {
            return convertCoco(request);
        }
        result.errorCode = QStringLiteral("unsupported_target_format");
        result.errorMessage = QStringLiteral("Unsupported dataset target format: %1").arg(request.targetFormat);
        return result;
    }
    if (sourceFormat == QStringLiteral("voc_xml")) {
        return convertVoc(request);
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
    object.insert(QStringLiteral("targetValidation"), targetValidation.toJson());

    QJsonArray issueArray;
    for (const DatasetConversionIssue& issue : issues) {
        issueArray.append(issue.toJson());
    }
    object.insert(QStringLiteral("issues"), issueArray);
    return object;
}

} // namespace aitrain
