#include "aitrain/core/DatasetConversion.h"

#include "aitrain/core/DatasetValidators.h"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QDomDocument>
#include <QImageReader>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QJsonArray>
#include <QMap>
#include <QRegExp>
#include <QSet>
#include <QtMath>
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

    const QFileInfo sourceInfo(sourceImagePath);
    const QFileInfo targetInfo(targetPath);
    if (targetInfo.exists()) {
        const QString sourceCanonicalPath = jsonPath(sourceInfo.canonicalFilePath());
        const QString targetCanonicalPath = jsonPath(targetInfo.canonicalFilePath());
        if (!sourceCanonicalPath.isEmpty() && !targetCanonicalPath.isEmpty()) {
#ifdef Q_OS_WIN
            if (sourceCanonicalPath.compare(targetCanonicalPath, Qt::CaseInsensitive) == 0) {
                return true;
            }
#else
            if (sourceCanonicalPath == targetCanonicalPath) {
                return true;
            }
#endif
        }
    }

    outputRoot.mkpath(QFileInfo(targetPath).absolutePath());
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

QString unquotedYamlScalar(QString value)
{
    value = value.trimmed();
    const int commentIndex = value.indexOf(QLatin1Char('#'));
    if (commentIndex >= 0) {
        value = value.left(commentIndex).trimmed();
    }
    if ((value.startsWith(QLatin1Char('\'')) && value.endsWith(QLatin1Char('\'')))
        || (value.startsWith(QLatin1Char('"')) && value.endsWith(QLatin1Char('"')))) {
        value = value.mid(1, value.size() - 2);
    }
    return value.trimmed();
}

QString parseYoloScalarFromYaml(const QString& yamlText, const QString& key)
{
    const QString prefix = key + QLatin1Char(':');
    const QStringList lines = yamlText.split(QLatin1Char('\n'));
    for (const QString& rawLine : lines) {
        const QString line = rawLine.trimmed();
        if (!line.startsWith(prefix)) {
            continue;
        }
        return unquotedYamlScalar(line.mid(prefix.size()));
    }
    return {};
}

int parseYoloClassCountFromYaml(const QString& yamlText)
{
    const QStringList lines = yamlText.split(QLatin1Char('\n'));
    for (const QString& rawLine : lines) {
        const QString line = rawLine.trimmed();
        if (!line.startsWith(QStringLiteral("nc:"))) {
            continue;
        }
        bool ok = false;
        const int count = line.mid(QStringLiteral("nc:").size()).trimmed().toInt(&ok);
        if (ok && count > 0) {
            return count;
        }
    }
    return 0;
}

QString resolveYoloPath(const QString& basePath, const QString& value)
{
    if (value.isEmpty()) {
        return QDir::cleanPath(basePath);
    }
    if (QFileInfo(value).isAbsolute()) {
        return QDir::cleanPath(value);
    }
    return QDir(basePath).filePath(value);
}

QString labelPathForYoloImagePath(QString imagePath)
{
    imagePath = jsonPath(imagePath);
    if (imagePath.startsWith(QStringLiteral("images/"))) {
        return QStringLiteral("labels/") + imagePath.mid(QStringLiteral("images/").size());
    }
    imagePath.replace(QStringLiteral("/images/"), QStringLiteral("/labels/"));
    if (imagePath.endsWith(QStringLiteral("/images"))) {
        imagePath.chop(QStringLiteral("/images").size());
        imagePath += QStringLiteral("/labels");
    }
    return imagePath;
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
    DatasetConversionResult* result)
{
    const QDir imagesDir(imagesRoot);
    for (const QString& imagePath : imagePaths) {
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

YoloDataset parseYoloDataset(const DatasetConversionRequest& request, bool segmentation, DatasetConversionResult* result)
{
    YoloDataset dataset;
    dataset.rootPath = QDir::cleanPath(request.sourcePath);
    const QDir root(dataset.rootPath);
    const QString dataYamlPath = root.filePath(QStringLiteral("data.yaml"));
    QString yamlText;
    int requestedClassCount = 0;
    QString yamlBasePath = dataset.rootPath;
    QString trainImagePath = QStringLiteral("images/train");
    QString valImagePath = QStringLiteral("images/val");
    bool hasYamlSplitPaths = false;
    if (QFileInfo::exists(dataYamlPath)) {
        QFile yamlFile(dataYamlPath);
        if (yamlFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            yamlText = QString::fromUtf8(yamlFile.readAll());
            bool foundNames = false;
            const QStringList names = parseYoloNamesFromYaml(yamlText, &foundNames);
            dataset.namesFromYaml = foundNames && !names.isEmpty();
            requestedClassCount = parseYoloClassCountFromYaml(yamlText);
            for (int index = 0; index < names.size(); ++index) {
                YoloClass cls;
                cls.id = index;
                cls.name = names.at(index);
                dataset.classes.append(cls);
            }
            const QString yamlPath = parseYoloScalarFromYaml(yamlText, QStringLiteral("path"));
            yamlBasePath = resolveYoloPath(dataset.rootPath, yamlPath);
            const QString yamlTrain = parseYoloScalarFromYaml(yamlText, QStringLiteral("train"));
            const QString yamlVal = parseYoloScalarFromYaml(yamlText, QStringLiteral("val"));
            if (!yamlTrain.isEmpty()) {
                trainImagePath = yamlTrain;
                hasYamlSplitPaths = true;
            }
            if (!yamlVal.isEmpty()) {
                valImagePath = yamlVal;
                hasYamlSplitPaths = true;
            }
        }
    }

    if (result) {
        result->sourceValidation.insert(QStringLiteral("dataYaml"), QFileInfo::exists(dataYamlPath));
        result->sourceValidation.insert(QStringLiteral("task"), segmentation ? QStringLiteral("segmentation") : QStringLiteral("detection"));
    }

    int nextImageId = 0;
    int maxClassId = dataset.classes.size() - 1;
    const QVector<QPair<QString, QString>> splits{
        qMakePair(QStringLiteral("train"), trainImagePath),
        qMakePair(QStringLiteral("val"), valImagePath)
    };
    QSet<QString> processedImageRoots;
    for (const QPair<QString, QString>& splitPath : splits) {
        const QString split = splitPath.first;
        const QString imagePath = splitPath.second;
        const QString imagesRoot = resolveYoloPath(yamlBasePath, imagePath);
        const QString cleanImagesRoot = QDir::cleanPath(imagesRoot).toLower();
        if (processedImageRoots.contains(cleanImagesRoot)) {
            dataset.splitCounts.insert(split, 0);
            continue;
        }
        processedImageRoots.insert(cleanImagesRoot);
        const QString labelsRoot = resolveYoloPath(yamlBasePath, labelPathForYoloImagePath(imagePath));
        const QStringList images = sortedImageFiles(imagesRoot);
        dataset.splitCounts.insert(split, images.size());
        appendYoloImageForSplit(split, imagesRoot, labelsRoot, jsonPath(QStringLiteral("images/%1").arg(split)), images, segmentation, dataset.namesFromYaml,
            dataset.classes.size(), &nextImageId, &maxClassId, &dataset, result);
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
                dataset.classes.size(), &nextImageId, &maxClassId, &dataset, result);
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
    int* convertedSamples)
{
    QJsonArray images;
    QJsonArray annotations;
    const QJsonArray categories = cocoCategories(dataset);

    for (const YoloImage& image : dataset.images) {
        if (image.split != split || image.annotations.isEmpty()) {
            continue;
        }

        QJsonArray imageAnnotations;
        for (const YoloAnnotation& annotation : image.annotations) {
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

DatasetConversionResult convertYoloToCoco(const DatasetConversionRequest& request, bool segmentation)
{
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

    const YoloDataset dataset = parseYoloDataset(request, segmentation, &result);
    result.splitCounts = dataset.splitCounts;
    for (const YoloClass& cls : dataset.classes) {
        result.classMap.insert(QString::number(cls.id), cls.name);
    }

    int nextAnnotationId = 0;
    int trainSamples = 0;
    int valSamples = 0;
    if (!writeYoloSplitToCoco(dataset, QStringLiteral("train"), request, segmentation, &result, &nextAnnotationId, &trainSamples)) {
        return result;
    }
    if (!writeYoloSplitToCoco(dataset, QStringLiteral("val"), request, segmentation, &result, &nextAnnotationId, &valSamples)) {
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
    DatasetConversionResult* result)
{
    QStringList objects;
    int convertedObjectCount = 0;
    for (const YoloAnnotation& annotation : image.annotations) {
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

DatasetConversionResult convertYoloToVoc(const DatasetConversionRequest& request)
{
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

    const YoloDataset dataset = parseYoloDataset(request, false, &result);
    result.splitCounts = dataset.splitCounts;
    for (const YoloClass& cls : dataset.classes) {
        result.classMap.insert(QString::number(cls.id), cls.name);
    }
    QSet<QString> plannedXmlPaths;
    QSet<QString> plannedImagePaths;
    for (const YoloImage& image : dataset.images) {
        if (!writeVocXmlForImage(dataset, image, request, &plannedXmlPaths, &plannedImagePaths, &result) && !result.errorCode.isEmpty()) {
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
    if (!writeJsonFile(result.reportPath, report, &writeError)) {
        result.ok = false;
        result.errorCode = QStringLiteral("report_write_failed");
        result.errorMessage = writeError;
    }
    return result;
}

DatasetConversionResult convertCoco(const DatasetConversionRequest& request)
{
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
        && sourceFormat != QStringLiteral("yolo_detection")
        && sourceFormat != QStringLiteral("yolo_segmentation")
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
    if (sourceFormat == QStringLiteral("yolo_detection")) {
        if (targetFormat == QStringLiteral("coco_json")) {
            return convertYoloToCoco(request, false);
        }
        if (targetFormat == QStringLiteral("voc_xml")) {
            return convertYoloToVoc(request);
        }
        result.errorCode = QStringLiteral("unsupported_target_format");
        result.errorMessage = QStringLiteral("Unsupported dataset target format: %1").arg(request.targetFormat);
        return result;
    }
    if (sourceFormat == QStringLiteral("yolo_segmentation")) {
        if (targetFormat == QStringLiteral("coco_json")) {
            return convertYoloToCoco(request, true);
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
