#include "DatasetConversionInternal.h"

#include "YoloDatasetLayout.h"
#include "aitrain/core/DatasetValidators.h"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QMap>
#include <QRegExp>
#include <QSet>
#include <QStringList>
#include <QtMath>
#include <QVector>

namespace aitrain {
namespace dataset_conversion_detail {

struct YoloClass {
    int id = 0;
    QString name;
};

struct YoloAnnotation {
    int classId = -1;
    QVector<double> values;
    QString sourceFile;
    int lineNumber = 0;
};

struct YoloImage {
    int id = 0;
    QString split;
    QString fileName;
    QString relativeImagePath;
    QString sourceImagePath;
    QString labelPath;
    int width = 0;
    int height = 0;
    QVector<YoloAnnotation> annotations;
};

struct YoloDataset {
    QString rootPath;
    QVector<YoloClass> classes;
    QVector<YoloImage> images;
    QJsonObject splitCounts;
    bool namesFromYaml = false;
};

QString jsonPath(const QString& path)
{
    QString value = QDir::cleanPath(path);
    value.replace(QLatin1Char('\\'), QLatin1Char('/'));
    return value;
}

bool sameCanonicalFile(const QString& leftPath, const QString& rightPath)
{
    const QString leftCanonicalPath = jsonPath(QFileInfo(leftPath).canonicalFilePath());
    const QString rightCanonicalPath = jsonPath(QFileInfo(rightPath).canonicalFilePath());
    if (leftCanonicalPath.isEmpty() || rightCanonicalPath.isEmpty()) {
        return false;
    }
#ifdef Q_OS_WIN
    return leftCanonicalPath.compare(rightCanonicalPath, Qt::CaseInsensitive) == 0;
#else
    return leftCanonicalPath == rightCanonicalPath;
#endif
}

QString normalizedOutputTargetKey(const QString& path)
{
    QString key = jsonPath(QDir::cleanPath(QFileInfo(path).absoluteFilePath()));
#ifdef Q_OS_WIN
    key = key.toLower();
#endif
    return key;
}

bool containsPlannedOutputTarget(const QStringList& targets, const QSet<QString>& plannedTargets, QString* duplicateTarget)
{
    for (const QString& target : targets) {
        if (plannedTargets.contains(normalizedOutputTargetKey(target))) {
            if (duplicateTarget) {
                *duplicateTarget = target;
            }
            return true;
        }
    }
    return false;
}

void insertPlannedOutputTargets(const QStringList& targets, QSet<QString>* plannedTargets)
{
    if (!plannedTargets) {
        return;
    }
    for (const QString& target : targets) {
        plannedTargets->insert(normalizedOutputTargetKey(target));
    }
}

QString xmlEscaped(QString value)
{
    value.replace(QLatin1Char('&'), QStringLiteral("&amp;"));
    value.replace(QLatin1Char('<'), QStringLiteral("&lt;"));
    value.replace(QLatin1Char('>'), QStringLiteral("&gt;"));
    value.replace(QLatin1Char('"'), QStringLiteral("&quot;"));
    value.replace(QLatin1Char('\''), QStringLiteral("&apos;"));
    return value;
}

bool copyImageToRelativePath(const QString& sourceImagePath,
    const QString& outputPath,
    const QString& relativePath,
    QString* error = nullptr)
{
    const QString cleanedRelativePath = jsonPath(relativePath);
    if (QFileInfo(cleanedRelativePath).isAbsolute()
        || cleanedRelativePath == QStringLiteral("..")
        || cleanedRelativePath.startsWith(QStringLiteral("../"))
        || cleanedRelativePath.contains(QStringLiteral("/../"))) {
        if (error) {
            *error = QStringLiteral("Refusing unsafe relative output path: %1").arg(relativePath);
        }
        return false;
    }

    if (!QFileInfo::exists(sourceImagePath)) {
        if (error) {
            *error = QStringLiteral("Image file does not exist: %1").arg(sourceImagePath);
        }
        return false;
    }
    const QDir outputRoot(outputPath);
    const QString outputRootPath = jsonPath(QDir::cleanPath(QFileInfo(outputPath).absoluteFilePath()));
    const QString targetPath = QDir::cleanPath(outputRoot.filePath(cleanedRelativePath));
    const QString targetAbsolutePath = jsonPath(QDir::cleanPath(QFileInfo(targetPath).absoluteFilePath()));
    const QString rootPrefix = outputRootPath.endsWith(QLatin1Char('/'))
        ? outputRootPath
        : outputRootPath + QStringLiteral("/");
    if (targetAbsolutePath.compare(outputRootPath, Qt::CaseInsensitive) != 0
        && !targetAbsolutePath.startsWith(rootPrefix, Qt::CaseInsensitive)) {
        if (error) {
            *error = QStringLiteral("Refusing output path outside conversion root: %1").arg(relativePath);
        }
        return false;
    }

    if (QFileInfo::exists(targetPath) && sameCanonicalFile(sourceImagePath, targetPath)) {
        return true;
    }

    outputRoot.mkpath(QFileInfo(targetPath).absolutePath());
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
    return true;
}

QStringList parseYoloNamesFromYaml(const QString& yamlText, bool* foundNames = nullptr)
{
    if (foundNames) {
        *foundNames = false;
    }

    QStringList names;
    QMap<int, QString> mappedNames;
    const QStringList lines = yamlText.split(QLatin1Char('\n'));
    bool inBlockNames = false;
    for (const QString& rawLine : lines) {
        const QString line = rawLine.trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) {
            continue;
        }
        if (line.startsWith(QStringLiteral("names:")) && line.contains(QLatin1Char('['))) {
            QString inner = line.mid(QStringLiteral("names:").size()).trimmed();
            inner.remove(QLatin1Char('['));
            inner.remove(QLatin1Char(']'));
            for (QString part : inner.split(QLatin1Char(','), QString::SkipEmptyParts)) {
                part = part.trimmed();
                part.remove(QLatin1Char('\''));
                part.remove(QLatin1Char('"'));
                if (!part.isEmpty()) {
                    names.append(part);
                }
            }
            if (foundNames) {
                *foundNames = !names.isEmpty();
            }
            return names;
        }
        if (line.startsWith(QStringLiteral("names:")) && line.contains(QLatin1Char('{'))) {
            QString inner = line.mid(QStringLiteral("names:").size()).trimmed();
            inner.remove(QLatin1Char('{'));
            inner.remove(QLatin1Char('}'));
            for (QString part : inner.split(QLatin1Char(','), QString::SkipEmptyParts)) {
                const int colon = part.indexOf(QLatin1Char(':'));
                if (colon < 0) {
                    continue;
                }
                bool ok = false;
                const int classId = part.left(colon).trimmed().toInt(&ok);
                QString value = part.mid(colon + 1).trimmed();
                value.remove(QLatin1Char('\''));
                value.remove(QLatin1Char('"'));
                if (ok && classId >= 0 && !value.isEmpty()) {
                    mappedNames.insert(classId, value);
                }
            }
            if (foundNames) {
                *foundNames = !mappedNames.isEmpty();
            }
            break;
        }
        if (line == QStringLiteral("names:")) {
            inBlockNames = true;
            if (foundNames) {
                *foundNames = true;
            }
            continue;
        }
        if (inBlockNames) {
            const bool indented = rawLine.startsWith(QLatin1Char(' ')) || rawLine.startsWith(QLatin1Char('\t'));
            if (!indented) {
                break;
            }
            if (line.startsWith(QStringLiteral("- "))) {
                QString value = line.mid(2).trimmed();
                value.remove(QLatin1Char('\''));
                value.remove(QLatin1Char('"'));
                if (!value.isEmpty()) {
                    names.append(value);
                }
                continue;
            }
            const int colon = line.indexOf(QLatin1Char(':'));
            if (colon < 0) {
                continue;
            }
            bool ok = false;
            const int classId = line.left(colon).trimmed().toInt(&ok);
            QString value = line.mid(colon + 1).trimmed();
            value.remove(QLatin1Char('\''));
            value.remove(QLatin1Char('"'));
            if (ok && classId >= 0 && !value.isEmpty()) {
                mappedNames.insert(classId, value);
            }
        }
    }

    if (!names.isEmpty()) {
        return names;
    }

    if (!mappedNames.isEmpty()) {
        const int maxId = mappedNames.lastKey();
        for (int id = 0; id <= maxId; ++id) {
            names.append(mappedNames.value(id, QStringLiteral("class_%1").arg(id)));
        }
    }
    return names;
}

QStringList supportedImageNameFilters()
{
    return QStringList{
        QStringLiteral("*.bmp"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.jpg"),
        QStringLiteral("*.png")
    };
}

QStringList sortedImageFiles(const QString& path)
{
    QStringList files;
    QDirIterator iterator(path, supportedImageNameFilters(), QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        files.append(iterator.next());
    }
    files.sort();
    return files;
}

bool readImageDimensions(const QString& imagePath, int* width, int* height)
{
    QImageReader reader(imagePath);
    const QSize size = reader.size();
    if (!size.isValid() || size.width() <= 0 || size.height() <= 0) {
        return false;
    }
    if (width) {
        *width = size.width();
    }
    if (height) {
        *height = size.height();
    }
    return true;
}

QVector<double> parseYoloNumbers(const QStringList& tokens, int firstIndex, bool* ok)
{
    QVector<double> values;
    if (ok) {
        *ok = true;
    }
    for (int index = firstIndex; index < tokens.size(); ++index) {
        bool valueOk = false;
        const double value = tokens.at(index).toDouble(&valueOk);
        if (!valueOk) {
            if (ok) {
                *ok = false;
            }
            return {};
        }
        values.append(value);
    }
    return values;
}

bool yoloDetectionValuesValid(const QVector<double>& values)
{
    return values.size() == 4 && values.at(2) > 0.0 && values.at(3) > 0.0;
}

double polygonArea(const QJsonArray& polygon)
{
    if (polygon.size() < 6 || polygon.size() % 2 != 0) {
        return 0.0;
    }
    double area = 0.0;
    const int pointCount = polygon.size() / 2;
    for (int index = 0; index < pointCount; ++index) {
        const int next = (index + 1) % pointCount;
        const double x1 = polygon.at(index * 2).toDouble();
        const double y1 = polygon.at(index * 2 + 1).toDouble();
        const double x2 = polygon.at(next * 2).toDouble();
        const double y2 = polygon.at(next * 2 + 1).toDouble();
        area += x1 * y2 - x2 * y1;
    }
    return qAbs(area) / 2.0;
}

bool yoloSegmentationValuesValid(const QVector<double>& values)
{
    return values.size() >= 6 && values.size() % 2 == 0;
}

QJsonArray bboxFromYoloValues(const QVector<double>& values, int imageWidth, int imageHeight)
{
    const double width = values.at(2) * imageWidth;
    const double height = values.at(3) * imageHeight;
    const double x = values.at(0) * imageWidth - width / 2.0;
    const double y = values.at(1) * imageHeight - height / 2.0;
    QJsonArray bbox;
    bbox.append(qMax(0.0, x));
    bbox.append(qMax(0.0, y));
    bbox.append(qMin(width, imageWidth - qMax(0.0, x)));
    bbox.append(qMin(height, imageHeight - qMax(0.0, y)));
    return bbox;
}

QJsonArray polygonFromYoloValues(const QVector<double>& values, int imageWidth, int imageHeight)
{
    QJsonArray polygon;
    for (int index = 0; index + 1 < values.size(); index += 2) {
        polygon.append(values.at(index) * imageWidth);
        polygon.append(values.at(index + 1) * imageHeight);
    }
    return polygon;
}

QJsonArray bboxFromPolygon(const QJsonArray& polygon)
{
    double minX = 0.0;
    double minY = 0.0;
    double maxX = 0.0;
    double maxY = 0.0;
    bool initialized = false;
    for (int index = 0; index + 1 < polygon.size(); index += 2) {
        const double x = polygon.at(index).toDouble();
        const double y = polygon.at(index + 1).toDouble();
        if (!initialized) {
            minX = maxX = x;
            minY = maxY = y;
            initialized = true;
        } else {
            minX = qMin(minX, x);
            minY = qMin(minY, y);
            maxX = qMax(maxX, x);
            maxY = qMax(maxY, y);
        }
    }
    QJsonArray bbox;
    bbox.append(minX);
    bbox.append(minY);
    bbox.append(qMax(0.0, maxX - minX));
    bbox.append(qMax(0.0, maxY - minY));
    return bbox;
}

void appendYoloImageForSplit(const QString& split,
    const QString& imagesRoot,
    const QString& labelsRoot,
    const QString& outputImagePrefix,
    const QStringList& imagePaths,
    bool segmentation,
    bool namesFromYaml,
    int explicitClassCount,
    int* nextImageId,
    int* maxClassId,
    YoloDataset* dataset,
    DatasetConversionResult* result,
    const CancellationCallback& shouldCancel)
{
    const QDir imagesDir(imagesRoot);
    for (const QString& imagePath : imagePaths) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(result);
            return;
        }
        YoloImage image;
        image.id = ++(*nextImageId);
        image.split = split;
        const QString relativeWithinSplit = jsonPath(imagesDir.relativeFilePath(imagePath));
        image.fileName = QFileInfo(imagePath).fileName();
        image.relativeImagePath = outputImagePrefix.isEmpty()
            ? relativeWithinSplit
            : jsonPath(QStringLiteral("%1/%2").arg(outputImagePrefix, relativeWithinSplit));
        image.sourceImagePath = QDir::cleanPath(imagePath);
        const QString labelRelative = QFileInfo(relativeWithinSplit).path() == QStringLiteral(".")
            ? QFileInfo(relativeWithinSplit).completeBaseName() + QStringLiteral(".txt")
            : QFileInfo(relativeWithinSplit).path() + QLatin1Char('/') + QFileInfo(relativeWithinSplit).completeBaseName() + QStringLiteral(".txt");
        image.labelPath = QDir(labelsRoot).filePath(labelRelative);
        if (!readImageDimensions(image.sourceImagePath, &image.width, &image.height)) {
            if (result) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_image"), image.labelPath, image.sourceImagePath, QString(),
                    QStringLiteral("YOLO source image dimensions could not be read.")));
                ++result->skippedSampleCount;
            }
            dataset->images.append(image);
            continue;
        }

        QFile labelFile(image.labelPath);
        if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            if (result) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("missing_label"), image.labelPath, image.sourceImagePath, QString(),
                    QStringLiteral("YOLO label file is missing.")));
            }
            dataset->images.append(image);
            continue;
        }

        const QStringList lines = QString::fromUtf8(labelFile.readAll()).split(QLatin1Char('\n'));
        int lineNumber = 0;
        for (const QString& rawLine : lines) {
            if (conversionCanceled(shouldCancel)) {
                markCanceled(result);
                return;
            }
            ++lineNumber;
            const QString line = rawLine.trimmed();
            if (line.isEmpty()) {
                continue;
            }
            if (result) {
                ++result->annotationCount;
            }
            const QStringList tokens = line.split(QRegExp(QStringLiteral("\\s+")), QString::SkipEmptyParts);
            if (tokens.size() < 2) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_yolo_row"), image.labelPath, image.sourceImagePath, QString::number(lineNumber),
                        QStringLiteral("YOLO annotation row is malformed.")));
                    ++result->skippedAnnotationCount;
                }
                continue;
            }
            bool classOk = false;
            const int classId = tokens.first().toInt(&classOk);
            if (!classOk || classId < 0 || (namesFromYaml && classId >= explicitClassCount)) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("unknown_class_id"), image.labelPath, image.sourceImagePath, tokens.first(),
                        QStringLiteral("YOLO annotation references an unknown class id.")));
                    ++result->skippedAnnotationCount;
                }
                continue;
            }
            if (maxClassId) {
                *maxClassId = qMax(*maxClassId, classId);
            }
            bool valuesOk = false;
            const QVector<double> values = parseYoloNumbers(tokens, 1, &valuesOk);
            if (!valuesOk) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_yolo_row"), image.labelPath, image.sourceImagePath, QString::number(classId),
                        QStringLiteral("YOLO annotation row contains non-numeric values.")));
                    ++result->skippedAnnotationCount;
                }
                continue;
            }
            if (!segmentation && !yoloDetectionValuesValid(values)) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), image.labelPath, image.sourceImagePath, QString::number(classId),
                        QStringLiteral("YOLO detection bbox is invalid.")));
                    ++result->skippedAnnotationCount;
                }
                continue;
            }
            if (segmentation && !yoloSegmentationValuesValid(values)) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), image.labelPath, image.sourceImagePath, QString::number(classId),
                        QStringLiteral("YOLO segmentation polygon is invalid.")));
                    ++result->skippedAnnotationCount;
                }
                continue;
            }
            YoloAnnotation annotation;
            annotation.classId = classId;
            annotation.values = values;
            annotation.sourceFile = image.labelPath;
            annotation.lineNumber = lineNumber;
            image.annotations.append(annotation);
        }
        dataset->images.append(image);
    }
}

YoloDataset parseYoloDataset(
    const DatasetConversionRequest& request,
    bool segmentation,
    DatasetConversionResult* result,
    const CancellationCallback& shouldCancel)
{
    YoloDataset dataset;
    dataset.rootPath = QDir::cleanPath(request.sourcePath);
    const QDir root(dataset.rootPath);
    const QString dataYamlPath = root.filePath(QStringLiteral("data.yaml"));
    QString yamlText;
    int requestedClassCount = 0;
    bool hasYamlSplitPaths = false;
    QString yamlError;
    const YoloDataYaml layout = parseYoloDataYaml(dataset.rootPath, &yamlError);
    if (QFileInfo::exists(dataYamlPath)) {
        QFile yamlFile(dataYamlPath);
        if (yamlFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            yamlText = QString::fromUtf8(yamlFile.readAll());
            QStringList names = layout.classNames;
            if (names.isEmpty()) {
                bool foundNames = false;
                names = parseYoloNamesFromYaml(yamlText, &foundNames);
                dataset.namesFromYaml = foundNames && !names.isEmpty();
            } else {
                dataset.namesFromYaml = true;
            }
            requestedClassCount = layout.classCount;
            for (int index = 0; index < names.size(); ++index) {
                YoloClass cls;
                cls.id = index;
                cls.name = names.at(index);
                dataset.classes.append(cls);
            }
            hasYamlSplitPaths = layout.splitImagePaths.contains(QStringLiteral("train"))
                || layout.splitImagePaths.contains(QStringLiteral("val"));
        }
    }
    if (!yamlError.isEmpty() && result) {
        result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_data_yaml"), dataYamlPath, dataset.rootPath, QString(),
            yamlError));
    }

    if (result) {
        result->sourceValidation.insert(QStringLiteral("dataYaml"), QFileInfo::exists(dataYamlPath));
        result->sourceValidation.insert(QStringLiteral("task"), segmentation ? QStringLiteral("segmentation") : QStringLiteral("detection"));
    }

    int nextImageId = 0;
    int maxClassId = dataset.classes.size() - 1;
    QSet<QString> processedImageRoots;
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val")}) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(result);
            return dataset;
        }
        const YoloSplitPaths splitPaths = yoloSplitPaths(layout, split);
        const QString imagesRoot = splitPaths.imageDir;
        const QString cleanImagesRoot = QDir::cleanPath(imagesRoot).toLower();
        if (processedImageRoots.contains(cleanImagesRoot)) {
            dataset.splitCounts.insert(split, 0);
            continue;
        }
        processedImageRoots.insert(cleanImagesRoot);
        const QString labelsRoot = splitPaths.labelDir;
        const QStringList images = sortedImageFiles(imagesRoot);
        dataset.splitCounts.insert(split, images.size());
        appendYoloImageForSplit(split, imagesRoot, labelsRoot, jsonPath(QStringLiteral("images/%1").arg(split)), images, segmentation, dataset.namesFromYaml,
            dataset.classes.size(), &nextImageId, &maxClassId, &dataset, result, shouldCancel);
        if (result && result->errorCode == QStringLiteral("canceled")) {
            return dataset;
        }
    }

    if (dataset.images.isEmpty() && !hasYamlSplitPaths) {
        const QString imagesRoot = root.filePath(QStringLiteral("images"));
        const QString labelsRoot = root.filePath(QStringLiteral("labels"));
        const QStringList images = sortedImageFiles(imagesRoot);
        if (!images.isEmpty()) {
            if (result) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("flat_yolo_layout"), dataYamlPath, imagesRoot, QString(),
                    QStringLiteral("YOLO dataset has a flat image layout; treating it as train split.")));
            }
            dataset.splitCounts.insert(QStringLiteral("train"), images.size());
            appendYoloImageForSplit(QStringLiteral("train"), imagesRoot, labelsRoot, QStringLiteral("images/train"), images, segmentation, dataset.namesFromYaml,
                dataset.classes.size(), &nextImageId, &maxClassId, &dataset, result, shouldCancel);
            if (result && result->errorCode == QStringLiteral("canceled")) {
                return dataset;
            }
        }
    }

    if (dataset.classes.isEmpty()) {
        const int classCount = qMax(requestedClassCount, maxClassId + 1);
        for (int id = 0; id < qMax(1, classCount); ++id) {
            YoloClass cls;
            cls.id = id;
            cls.name = QStringLiteral("class_%1").arg(id);
            dataset.classes.append(cls);
        }
        if (result) {
            result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("missing_class_names"), dataYamlPath, dataset.rootPath, QString(),
                QStringLiteral("YOLO data.yaml did not define class names; generated stable class names.")));
        }
    }

    if (result) {
        result->sampleCount = dataset.images.size();
        result->sourceValidation.insert(QStringLiteral("images"), dataset.images.size());
        result->sourceValidation.insert(QStringLiteral("classes"), dataset.classes.size());
        result->sourceValidation.insert(QStringLiteral("annotations"), result->annotationCount);
    }
    return dataset;
}

QJsonArray cocoCategories(const YoloDataset& dataset)
{
    QJsonArray categories;
    for (const YoloClass& cls : dataset.classes) {
        QJsonObject category;
        category.insert(QStringLiteral("id"), cls.id + 1);
        category.insert(QStringLiteral("name"), cls.name);
        categories.append(category);
    }
    return categories;
}

bool validateYoloSourceRoot(const QString& sourcePath, DatasetConversionResult* result)
{
    const QFileInfo sourceInfo(sourcePath);
    if (!sourceInfo.exists() || !sourceInfo.isDir() || !sourceInfo.isReadable()) {
        if (result) {
            result->errorCode = QStringLiteral("source_read_failed");
            result->errorMessage = QStringLiteral("Cannot read YOLO dataset source path: %1").arg(sourcePath);
        }
        return false;
    }
    return true;
}

bool writeYoloSplitToCoco(const YoloDataset& dataset,
    const QString& split,
    const DatasetConversionRequest& request,
    bool segmentation,
    DatasetConversionResult* result,
    int* nextAnnotationId,
    int* convertedSamples,
    const CancellationCallback& shouldCancel)
{
    QJsonArray images;
    QJsonArray annotations;
    const QJsonArray categories = cocoCategories(dataset);

    for (const YoloImage& image : dataset.images) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(result);
            return false;
        }
        if (image.split != split || image.annotations.isEmpty()) {
            continue;
        }

        QJsonArray imageAnnotations;
        for (const YoloAnnotation& annotation : image.annotations) {
            if (conversionCanceled(shouldCancel)) {
                markCanceled(result);
                return false;
            }
            if (annotation.classId < 0 || annotation.classId >= dataset.classes.size()) {
                continue;
            }
            QJsonObject object;
            object.insert(QStringLiteral("id"), ++(*nextAnnotationId));
            object.insert(QStringLiteral("image_id"), image.id);
            object.insert(QStringLiteral("category_id"), annotation.classId + 1);
            object.insert(QStringLiteral("iscrowd"), 0);
            if (segmentation) {
                const QJsonArray polygon = polygonFromYoloValues(annotation.values, image.width, image.height);
                const double area = polygonArea(polygon);
                if (area <= 0.0) {
                    if (result) {
                        result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                            QStringLiteral("YOLO polygon area is zero.")));
                        ++result->skippedAnnotationCount;
                    }
                    continue;
                }
                QJsonArray segmentationArray;
                segmentationArray.append(polygon);
                object.insert(QStringLiteral("segmentation"), segmentationArray);
                object.insert(QStringLiteral("bbox"), bboxFromPolygon(polygon));
                object.insert(QStringLiteral("area"), area);
            } else {
                const QJsonArray bbox = bboxFromYoloValues(annotation.values, image.width, image.height);
                if (bbox.at(2).toDouble() <= 0.0 || bbox.at(3).toDouble() <= 0.0) {
                    if (result) {
                        result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                            QStringLiteral("YOLO detection bbox became empty after clamping.")));
                        ++result->skippedAnnotationCount;
                    }
                    continue;
                }
                object.insert(QStringLiteral("bbox"), bbox);
                object.insert(QStringLiteral("area"), bbox.at(2).toDouble() * bbox.at(3).toDouble());
            }
            imageAnnotations.append(object);
        }
        if (imageAnnotations.isEmpty()) {
            continue;
        }

        if (result && result->copyImages) {
            QString copyError;
            if (!copyImageToRelativePath(image.sourceImagePath, request.outputPath, image.relativeImagePath, &copyError)) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), image.labelPath, image.sourceImagePath, QString(), copyError));
                ++result->skippedSampleCount;
                continue;
            }
        }

        QJsonObject imageObject;
        imageObject.insert(QStringLiteral("id"), image.id);
        imageObject.insert(QStringLiteral("file_name"), image.relativeImagePath);
        imageObject.insert(QStringLiteral("width"), image.width);
        imageObject.insert(QStringLiteral("height"), image.height);
        images.append(imageObject);
        for (const QJsonValue& annotation : imageAnnotations) {
            annotations.append(annotation);
            if (result) {
                ++result->convertedAnnotationCount;
            }
        }
        if (convertedSamples) {
            ++(*convertedSamples);
        }
        if (result) {
            ++result->convertedSampleCount;
        }
    }

    QJsonObject root;
    root.insert(QStringLiteral("images"), images);
    root.insert(QStringLiteral("annotations"), annotations);
    root.insert(QStringLiteral("categories"), categories);

    const QString annotationPath = QDir(request.outputPath).filePath(QStringLiteral("annotations/%1.json").arg(split));
    QString writeError;
    if (!writeJsonFile(annotationPath, root, &writeError)) {
        if (result) {
            result->errorCode = QStringLiteral("output_write_failed");
            result->errorMessage = writeError;
        }
        return false;
    }
    if (result) {
        result->outputFiles.insert(QStringLiteral("annotations_%1").arg(split), annotationPath);
    }
    return true;
}

DatasetConversionResult convertYoloToCoco(
    const DatasetConversionRequest& request,
    bool segmentation,
    const CancellationCallback& shouldCancel)
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

    if (!validateYoloSourceRoot(result.sourcePath, &result)) {
        return result;
    }
    if (!QDir().mkpath(result.outputPath)) {
        result.errorCode = QStringLiteral("output_write_failed");
        result.errorMessage = QStringLiteral("Cannot create output directory: %1").arg(result.outputPath);
        return result;
    }

    const YoloDataset dataset = parseYoloDataset(request, segmentation, &result, shouldCancel);
    if (result.errorCode == QStringLiteral("canceled")) {
        return result;
    }
    result.splitCounts = dataset.splitCounts;
    for (const YoloClass& cls : dataset.classes) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        result.classMap.insert(QString::number(cls.id), cls.name);
    }

    int nextAnnotationId = 0;
    int trainSamples = 0;
    int valSamples = 0;
    if (!writeYoloSplitToCoco(dataset, QStringLiteral("train"), request, segmentation, &result, &nextAnnotationId, &trainSamples, shouldCancel)) {
        return result;
    }
    if (!writeYoloSplitToCoco(dataset, QStringLiteral("val"), request, segmentation, &result, &nextAnnotationId, &valSamples, shouldCancel)) {
        return result;
    }
    result.splitCounts.insert(QStringLiteral("convertedTrain"), trainSamples);
    result.splitCounts.insert(QStringLiteral("convertedVal"), valSamples);
    if (result.copyImages) {
        result.outputFiles.insert(QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("images")));
    }

    if (result.convertedSampleCount <= 0 || result.convertedAnnotationCount <= 0) {
        result.errorCode = QStringLiteral("no_convertible_samples");
        result.errorMessage = QStringLiteral("YOLO dataset did not contain convertible annotations.");
        return result;
    }

    result.ok = true;
    result.reportPath = QDir(result.outputPath).filePath(QStringLiteral("dataset_conversion_report.json"));
    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("convertedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    QString writeError;
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

bool writeVocXmlForImage(const YoloDataset& dataset,
    const YoloImage& image,
    const DatasetConversionRequest& request,
    QSet<QString>* plannedXmlPaths,
    QSet<QString>* plannedImagePaths,
    DatasetConversionResult* result,
    const CancellationCallback& shouldCancel)
{
    QStringList objects;
    int convertedObjectCount = 0;
    for (const YoloAnnotation& annotation : image.annotations) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(result);
            return false;
        }
        if (annotation.classId < 0 || annotation.classId >= dataset.classes.size()) {
            continue;
        }
        if (!yoloDetectionValuesValid(annotation.values)) {
            continue;
        }
        const QJsonArray bbox = bboxFromYoloValues(annotation.values, image.width, image.height);
        const int xmin = qMax(0, qRound(bbox.at(0).toDouble()));
        const int ymin = qMax(0, qRound(bbox.at(1).toDouble()));
        const int xmax = qMin(image.width, qRound(bbox.at(0).toDouble() + bbox.at(2).toDouble()));
        const int ymax = qMin(image.height, qRound(bbox.at(1).toDouble() + bbox.at(3).toDouble()));
        if (xmax <= xmin || ymax <= ymin) {
            if (result) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                    QStringLiteral("YOLO detection bbox became empty after clamping.")));
                ++result->skippedAnnotationCount;
            }
            continue;
        }
        objects.append(QStringLiteral("  <object>\n"
                                      "    <name>%1</name>\n"
                                      "    <pose>Unspecified</pose>\n"
                                      "    <truncated>0</truncated>\n"
                                      "    <difficult>0</difficult>\n"
                                      "    <bndbox>\n"
                                      "      <xmin>%2</xmin>\n"
                                      "      <ymin>%3</ymin>\n"
                                      "      <xmax>%4</xmax>\n"
                                      "      <ymax>%5</ymax>\n"
                                      "    </bndbox>\n"
                                      "  </object>\n")
            .arg(xmlEscaped(dataset.classes.at(annotation.classId).name))
            .arg(xmin)
            .arg(ymin)
            .arg(xmax)
            .arg(ymax));
        ++convertedObjectCount;
    }
    if (objects.isEmpty()) {
        return true;
    }

    const QString imageName = QFileInfo(image.sourceImagePath).fileName();
    const QString relativeImageTarget = QStringLiteral("JPEGImages/%1").arg(imageName);
    const QString xmlPath = QDir(request.outputPath).filePath(QStringLiteral("Annotations/%1.xml").arg(QFileInfo(imageName).completeBaseName()));
    const QString cleanXmlPath = QDir::cleanPath(xmlPath).toLower();
    const QString cleanImagePath = QDir::cleanPath(QDir(request.outputPath).filePath(relativeImageTarget)).toLower();
    if ((plannedXmlPaths && plannedXmlPaths->contains(cleanXmlPath))
        || (result && result->copyImages && plannedImagePaths && plannedImagePaths->contains(cleanImagePath))) {
        if (result) {
            result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("duplicate_output_target"), image.labelPath, image.sourceImagePath, imageName,
                QStringLiteral("YOLO image would overwrite an existing VOC XML or image output target.")));
            ++result->skippedSampleCount;
            result->skippedAnnotationCount += convertedObjectCount;
        }
        return true;
    }
    if (plannedXmlPaths) {
        plannedXmlPaths->insert(cleanXmlPath);
    }
    if (result && result->copyImages && plannedImagePaths) {
        plannedImagePaths->insert(cleanImagePath);
    }

    const QString copiedImagePath = QDir(request.outputPath).filePath(QStringLiteral("JPEGImages/%1").arg(imageName));
    if (result && result->copyImages) {
        QString copyError;
        if (!copyImageToRelativePath(image.sourceImagePath, request.outputPath, relativeImageTarget, &copyError)) {
            result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), image.labelPath, image.sourceImagePath, QString(), copyError));
            ++result->skippedSampleCount;
            return false;
        }
    }

    const QString xml = QStringLiteral("<annotation>\n"
                                       "  <folder>JPEGImages</folder>\n"
                                       "  <filename>%1</filename>\n"
                                       "  <path>%2</path>\n"
                                       "  <size>\n"
                                       "    <width>%3</width>\n"
                                       "    <height>%4</height>\n"
                                       "    <depth>3</depth>\n"
                                       "  </size>\n"
                                       "%5"
                                       "</annotation>\n")
        .arg(xmlEscaped(imageName))
        .arg(xmlEscaped(result && result->copyImages ? copiedImagePath : image.sourceImagePath))
        .arg(image.width)
        .arg(image.height)
        .arg(objects.join(QString()));
    QString writeError;
    if (!writeTextFile(xmlPath, xml, &writeError)) {
        if (result) {
            result->errorCode = QStringLiteral("output_write_failed");
            result->errorMessage = writeError;
        }
        return false;
    }
    if (result) {
        ++result->convertedSampleCount;
        result->convertedAnnotationCount += convertedObjectCount;
    }
    return true;
}

DatasetConversionResult convertYoloToVoc(const DatasetConversionRequest& request, const CancellationCallback& shouldCancel)
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

    if (!validateYoloSourceRoot(result.sourcePath, &result)) {
        return result;
    }
    if (!QDir().mkpath(result.outputPath)) {
        result.errorCode = QStringLiteral("output_write_failed");
        result.errorMessage = QStringLiteral("Cannot create output directory: %1").arg(result.outputPath);
        return result;
    }

    const YoloDataset dataset = parseYoloDataset(request, false, &result, shouldCancel);
    if (result.errorCode == QStringLiteral("canceled")) {
        return result;
    }
    result.splitCounts = dataset.splitCounts;
    for (const YoloClass& cls : dataset.classes) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        result.classMap.insert(QString::number(cls.id), cls.name);
    }
    QSet<QString> plannedXmlPaths;
    QSet<QString> plannedImagePaths;
    for (const YoloImage& image : dataset.images) {
        if (conversionCanceled(shouldCancel)) {
            markCanceled(&result);
            return result;
        }
        if (!writeVocXmlForImage(dataset, image, request, &plannedXmlPaths, &plannedImagePaths, &result, shouldCancel) && !result.errorCode.isEmpty()) {
            return result;
        }
    }
    if (result.convertedSampleCount <= 0 || result.convertedAnnotationCount <= 0) {
        result.errorCode = QStringLiteral("no_convertible_samples");
        result.errorMessage = QStringLiteral("YOLO dataset did not contain convertible detection annotations.");
        return result;
    }
    result.outputFiles.insert(QStringLiteral("xmlRoot"), QDir(result.outputPath).filePath(QStringLiteral("Annotations")));
    if (result.copyImages) {
        result.outputFiles.insert(QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("JPEGImages")));
    }
    result.ok = true;

    result.reportPath = QDir(result.outputPath).filePath(QStringLiteral("dataset_conversion_report.json"));
    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("convertedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    QString writeError;
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
