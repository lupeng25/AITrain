#include "aitrain/core/ProductWorkflow.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/DetectionTrainer.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QDirIterator>
#include <QMap>
#include <QRegularExpression>
#include <QSet>
#include <QTextStream>

#include <algorithm>
namespace aitrain {
namespace {

QString nowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
}

QString cleanRelativePath(const QDir& root, const QString& absolutePath)
{
    return QDir::cleanPath(root.relativeFilePath(absolutePath));
}

QStringList imageNameFilters()
{
    return {
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.png"),
        QStringLiteral("*.bmp"),
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff")
    };
}

bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write JSON file: %1").arg(path);
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

bool writeTextFile(const QString& path, const QString& content, QString* error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot write text file: %1").arg(path);
        }
        return false;
    }
    QTextStream stream(&file);
    stream.setCodec("UTF-8");
    stream << content;
    return true;
}

QByteArray fileSha256(const QString& path, qint64* size, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("Cannot read file for hashing: %1").arg(path);
        }
        return {};
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    qint64 total = 0;
    while (!file.atEnd()) {
        const QByteArray chunk = file.read(1024 * 1024);
        total += chunk.size();
        hash.addData(chunk);
    }
    if (size) {
        *size = total;
    }
    return hash.result().toHex();
}

QFileInfoList collectFilesRecursive(const QString& rootPath, int maxFiles)
{
    QFileInfoList files;
    QDirIterator iterator(rootPath, QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        iterator.next();
        files.append(iterator.fileInfo());
    }
    std::sort(files.begin(), files.end(), [](const QFileInfo& left, const QFileInfo& right) {
        return left.absoluteFilePath().compare(right.absoluteFilePath(), Qt::CaseInsensitive) < 0;
    });
    if (maxFiles > 0 && files.size() > maxFiles) {
        while (files.size() > maxFiles) {
            files.removeLast();
        }
    }
    return files;
}

bool isImageFile(const QString& suffix)
{
    const QString lower = suffix.toLower();
    return lower == QStringLiteral("jpg")
        || lower == QStringLiteral("jpeg")
        || lower == QStringLiteral("png")
        || lower == QStringLiteral("bmp")
        || lower == QStringLiteral("tif")
        || lower == QStringLiteral("tiff");
}

QString snapshotFileRole(const QString& relativePath, const QFileInfo& fileInfo)
{
    const QString path = QDir::fromNativeSeparators(relativePath).toLower();
    const QString name = fileInfo.fileName().toLower();
    const QString suffix = fileInfo.suffix().toLower();
    if (isImageFile(suffix)) {
        return QStringLiteral("image");
    }
    if (name == QStringLiteral("data.yaml") || name == QStringLiteral("data.yml") || suffix == QStringLiteral("yaml") || suffix == QStringLiteral("yml")) {
        return QStringLiteral("config");
    }
    if (name == QStringLiteral("dict.txt")) {
        return QStringLiteral("dict");
    }
    if (name.startsWith(QStringLiteral("rec_gt")) || name.startsWith(QStringLiteral("det_gt")) || path.contains(QStringLiteral("/rec_gt")) || path.contains(QStringLiteral("/det_gt"))) {
        return QStringLiteral("ocr_gt");
    }
    if ((path.startsWith(QStringLiteral("labels/")) || path.contains(QStringLiteral("/labels/"))) && suffix == QStringLiteral("txt")) {
        return QStringLiteral("label");
    }
    return QStringLiteral("other");
}

bool isSnapshotKeyRole(const QString& role)
{
    return role == QStringLiteral("config")
        || role == QStringLiteral("dict")
        || role == QStringLiteral("ocr_gt");
}

DatasetValidationResult validateByFormat(const QString& datasetPath, const QString& format, const QJsonObject& options)
{
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt")) {
        return validateYoloDetectionDataset(datasetPath, options);
    }
    if (format == QStringLiteral("yolo_segmentation")) {
        return validateYoloSegmentationDataset(datasetPath, options);
    }
    if (format == QStringLiteral("paddleocr_det")) {
        return validatePaddleOcrDetDataset(datasetPath, options);
    }
    if (format == QStringLiteral("paddleocr_rec")) {
        return validatePaddleOcrRecDataset(datasetPath, options);
    }
    DatasetValidationResult result;
    result.ok = false;
    DatasetValidationResult::Issue issue;
    issue.severity = QStringLiteral("error");
    issue.code = QStringLiteral("unsupported_format");
    issue.filePath = datasetPath;
    issue.message = QStringLiteral("Unsupported dataset format: %1").arg(format);
    result.issues.append(issue);
    result.errors.append(issue.message);
    return result;
}

QJsonObject validationIssueCounts(const DatasetValidationResult& validation)
{
    QJsonObject counts;
    for (const DatasetValidationResult::Issue& issue : validation.issues) {
        counts.insert(issue.code, counts.value(issue.code).toInt() + 1);
    }
    return counts;
}

QJsonObject countYoloClasses(const QString& datasetPath)
{
    QJsonObject classCounts;
    const QDir root(datasetPath);
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        const QDir labelDir(root.filePath(QStringLiteral("labels/%1").arg(split)));
        if (!labelDir.exists()) {
            continue;
        }
        const QFileInfoList labels = labelDir.entryInfoList({QStringLiteral("*.txt")}, QDir::Files, QDir::Name);
        for (const QFileInfo& labelInfo : labels) {
            QFile file(labelInfo.absoluteFilePath());
            if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                continue;
            }
            while (!file.atEnd()) {
                const QString line = QString::fromUtf8(file.readLine()).trimmed();
                if (line.isEmpty()) {
                    continue;
                }
                const QString classId = line.section(QLatin1Char(' '), 0, 0).trimmed();
                if (!classId.isEmpty()) {
                    classCounts.insert(classId, classCounts.value(classId).toInt() + 1);
                }
            }
        }
    }
    return classCounts;
}

QJsonObject countImageSplits(const QString& datasetPath)
{
    QJsonObject splits;
    const QDir root(datasetPath);
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        const QDir imageDir(root.filePath(QStringLiteral("images/%1").arg(split)));
        int count = 0;
        if (imageDir.exists()) {
            for (const QString& filter : imageNameFilters()) {
                count += imageDir.entryInfoList({filter}, QDir::Files).size();
            }
        }
        splits.insert(split, count);
    }
    return splits;
}

QString classDistributionCsv(const QJsonObject& counts)
{
    QString csv = QStringLiteral("class,count\n");
    const QStringList keys = counts.keys();
    for (const QString& key : keys) {
        csv.append(QStringLiteral("%1,%2\n").arg(key).arg(counts.value(key).toInt()));
    }
    return csv;
}

QString csvEscape(QString value)
{
    if (value.contains(QLatin1Char('"')) || value.contains(QLatin1Char(',')) || value.contains(QLatin1Char('\n'))) {
        value.replace(QLatin1Char('"'), QStringLiteral("\"\""));
        return QStringLiteral("\"%1\"").arg(value);
    }
    return value;
}

struct QualityIssue {
    QString severity;
    QString code;
    QString filePath;
    QString imagePath;
    QString labelPath;
    QString split;
    QString message;
    QString details;
    QString repairHint;
    int line = 0;
    int classId = -1;
    bool xAnyLabelingSupported = false;

    QJsonObject toIssueJson() const
    {
        QJsonObject object;
        object.insert(QStringLiteral("severity"), severity);
        object.insert(QStringLiteral("code"), code);
        object.insert(QStringLiteral("filePath"), filePath);
        object.insert(QStringLiteral("line"), line);
        object.insert(QStringLiteral("message"), message);
        return object;
    }

    QJsonObject toProblemJson() const
    {
        QJsonObject object = toIssueJson();
        object.insert(QStringLiteral("reason"), code);
        object.insert(QStringLiteral("imagePath"), imagePath);
        object.insert(QStringLiteral("labelPath"), labelPath);
        object.insert(QStringLiteral("split"), split);
        object.insert(QStringLiteral("classId"), classId);
        object.insert(QStringLiteral("details"), details);
        object.insert(QStringLiteral("repairHint"), repairHint);
        object.insert(QStringLiteral("xAnyLabelingSupported"), xAnyLabelingSupported);
        return object;
    }
};

struct SplitQualityStats {
    int imageCount = 0;
    int readableImageCount = 0;
    int unreadableImageCount = 0;
    int zeroByteImageCount = 0;
    int duplicateImageCount = 0;
    int labelCount = 0;
    int problemCount = 0;
    double widthSum = 0.0;
    double heightSum = 0.0;
    double aspectMin = 0.0;
    double aspectMax = 0.0;
    QMap<int, int> classCounts;

    void addImageSize(const QSize& size)
    {
        if (!size.isValid() || size.isEmpty()) {
            return;
        }
        ++readableImageCount;
        widthSum += size.width();
        heightSum += size.height();
        const double aspect = static_cast<double>(size.width()) / static_cast<double>(qMax(1, size.height()));
        aspectMin = aspectMin <= 0.0 ? aspect : qMin(aspectMin, aspect);
        aspectMax = qMax(aspectMax, aspect);
    }
};

struct DatasetQualityContext {
    QString datasetPath;
    QString format;
    QStringList classNames;
    int maxIssues = 500;
    int maxProblemSamples = 500;
    int maxFiles = 20000;
    int duplicateHashLimit = 20000;
    double distributionWarningThreshold = 0.25;
    bool exportXAnyLabelingFixList = true;
    int scannedFiles = 0;
    int totalIssueCount = 0;
    bool issueLimitReached = false;
    bool problemSampleLimitReached = false;
    bool scanLimitReached = false;
    QJsonArray issues;
    QJsonArray problemSamples;
    QJsonObject issueCounts;
    QJsonObject severityCounts;
    QJsonArray distributionWarnings;
    QHash<QString, QString> firstImageByHash;
    QSet<QString> xAnyFixKeys;
    QStringList xAnyFixRows;
    QMap<QString, SplitQualityStats> splits;
};

bool isSupportedQualityFormat(const QString& format)
{
    return format == QStringLiteral("yolo_detection")
        || format == QStringLiteral("yolo_txt")
        || format == QStringLiteral("yolo_segmentation")
        || format == QStringLiteral("paddleocr_det")
        || format == QStringLiteral("paddleocr_rec");
}

void incrementJsonCount(QJsonObject& object, const QString& key, int amount = 1)
{
    object.insert(key, object.value(key).toInt() + amount);
}

QString splitFromPath(const QString& path, const QString& fallback = QStringLiteral("all"))
{
    const QString normalized = QDir::fromNativeSeparators(path).toLower();
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        if (normalized.contains(QStringLiteral("/%1/").arg(split))
            || normalized.startsWith(QStringLiteral("%1/").arg(split))
            || normalized.contains(QStringLiteral("_%1").arg(split))) {
            return split;
        }
    }
    return fallback;
}

QString classNameForId(const QStringList& classNames, int classId)
{
    return classId >= 0 && classId < classNames.size() && !classNames.at(classId).isEmpty()
        ? classNames.at(classId)
        : QStringLiteral("class_%1").arg(classId);
}

void addQualityIssue(DatasetQualityContext& context, const QualityIssue& issue)
{
    ++context.totalIssueCount;
    incrementJsonCount(context.issueCounts, issue.code);
    incrementJsonCount(context.severityCounts, issue.severity);
    if (!issue.split.isEmpty()) {
        context.splits[issue.split].problemCount += 1;
    }

    if (context.maxIssues <= 0 || context.issues.size() < context.maxIssues) {
        context.issues.append(issue.toIssueJson());
    } else {
        context.issueLimitReached = true;
    }

    if (context.maxProblemSamples <= 0 || context.problemSamples.size() < context.maxProblemSamples) {
        context.problemSamples.append(issue.toProblemJson());
    } else {
        context.problemSampleLimitReached = true;
    }

    if (context.exportXAnyLabelingFixList) {
        const QString fixPath = !issue.imagePath.isEmpty() ? issue.imagePath : issue.labelPath;
        if (!fixPath.isEmpty()) {
            const QString key = QStringLiteral("%1|%2|%3").arg(fixPath, issue.code).arg(issue.line);
            if (!context.xAnyFixKeys.contains(key)) {
                context.xAnyFixKeys.insert(key);
                context.xAnyFixRows.append(QStringLiteral("%1\t%2\t%3\t%4")
                    .arg(QDir::toNativeSeparators(fixPath), issue.code, issue.message, issue.repairHint));
            }
        }
    }
}

void addValidationIssues(DatasetQualityContext& context, const DatasetValidationResult& validation)
{
    for (const DatasetValidationResult::Issue& validationIssue : validation.issues) {
        QualityIssue issue;
        issue.severity = validationIssue.severity;
        issue.code = validationIssue.code;
        issue.filePath = validationIssue.filePath;
        issue.labelPath = validationIssue.filePath;
        issue.line = validationIssue.line;
        issue.message = validationIssue.message;
        issue.details = validationIssue.message;
        issue.repairHint = QStringLiteral("Open the referenced file and correct the dataset annotation.");
        issue.xAnyLabelingSupported = QFileInfo::exists(validationIssue.filePath);
        addQualityIssue(context, issue);
    }
}

bool scanLimitReached(DatasetQualityContext& context, const QString& filePath)
{
    if (context.maxFiles > 0 && context.scannedFiles >= context.maxFiles) {
        if (!context.scanLimitReached) {
            context.scanLimitReached = true;
            QualityIssue issue;
            issue.severity = QStringLiteral("warning");
            issue.code = QStringLiteral("file_limit");
            issue.filePath = context.datasetPath;
            issue.message = QStringLiteral("Dataset quality scan stopped after maxFiles=%1.").arg(context.maxFiles);
            issue.details = filePath;
            issue.repairHint = QStringLiteral("Increase maxFiles to scan the full dataset.");
            addQualityIssue(context, issue);
        }
        return true;
    }
    ++context.scannedFiles;
    return false;
}

QSize inspectQualityImage(DatasetQualityContext& context, const QString& imagePath, const QString& split, const QString& labelPath = QString())
{
    SplitQualityStats& stats = context.splits[split];
    ++stats.imageCount;

    const QFileInfo imageInfo(imagePath);
    if (!imageInfo.exists()) {
        QualityIssue issue;
        issue.severity = QStringLiteral("error");
        issue.code = QStringLiteral("missing_image");
        issue.filePath = imagePath;
        issue.imagePath = imagePath;
        issue.labelPath = labelPath;
        issue.split = split;
        issue.message = QStringLiteral("Referenced image does not exist.");
        issue.repairHint = QStringLiteral("Restore the image or remove/update the label row.");
        addQualityIssue(context, issue);
        ++stats.unreadableImageCount;
        return {};
    }

    if (imageInfo.size() == 0) {
        QualityIssue issue;
        issue.severity = QStringLiteral("error");
        issue.code = QStringLiteral("zero_byte_image");
        issue.filePath = imageInfo.absoluteFilePath();
        issue.imagePath = imageInfo.absoluteFilePath();
        issue.labelPath = labelPath;
        issue.split = split;
        issue.message = QStringLiteral("Image file is empty.");
        issue.repairHint = QStringLiteral("Replace the image file and re-export labels.");
        issue.xAnyLabelingSupported = false;
        addQualityIssue(context, issue);
        ++stats.zeroByteImageCount;
        ++stats.unreadableImageCount;
        return {};
    }

    QImageReader reader(imageInfo.absoluteFilePath());
    const QSize size = reader.size();
    if (!size.isValid() || size.isEmpty()) {
        QualityIssue issue;
        issue.severity = QStringLiteral("error");
        issue.code = QStringLiteral("unreadable_image");
        issue.filePath = imageInfo.absoluteFilePath();
        issue.imagePath = imageInfo.absoluteFilePath();
        issue.labelPath = labelPath;
        issue.split = split;
        issue.message = QStringLiteral("Image cannot be decoded or has invalid dimensions.");
        issue.details = reader.errorString();
        issue.repairHint = QStringLiteral("Recreate the image file or remove the sample.");
        addQualityIssue(context, issue);
        ++stats.unreadableImageCount;
        return {};
    }

    stats.addImageSize(size);
    if (context.firstImageByHash.size() < context.duplicateHashLimit && imageInfo.size() > 0) {
        qint64 sizeBytes = 0;
        QString hashError;
        const QString hash = QString::fromLatin1(fileSha256(imageInfo.absoluteFilePath(), &sizeBytes, &hashError));
        if (!hash.isEmpty()) {
            const QString firstPath = context.firstImageByHash.value(hash);
            if (!firstPath.isEmpty() && firstPath != imageInfo.absoluteFilePath()) {
                QualityIssue issue;
                issue.severity = QStringLiteral("warning");
                issue.code = QStringLiteral("duplicate_image_hash");
                issue.filePath = imageInfo.absoluteFilePath();
                issue.imagePath = imageInfo.absoluteFilePath();
                issue.labelPath = labelPath;
                issue.split = split;
                issue.message = QStringLiteral("Image content duplicates another sample.");
                issue.details = QStringLiteral("first=%1").arg(firstPath);
                issue.repairHint = QStringLiteral("Review duplicates and keep only intentional samples.");
                issue.xAnyLabelingSupported = true;
                addQualityIssue(context, issue);
                ++stats.duplicateImageCount;
            } else {
                context.firstImageByHash.insert(hash, imageInfo.absoluteFilePath());
            }
        }
    }
    return size;
}

void parseYoloLabelFile(
    DatasetQualityContext& context,
    const QFileInfo& labelInfo,
    const QString& imagePath,
    const QString& split,
    int classCount,
    bool segmentation)
{
    context.splits[split].labelCount += 1;
    QFile file(labelInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QualityIssue issue;
        issue.severity = QStringLiteral("error");
        issue.code = QStringLiteral("label_unreadable");
        issue.filePath = labelInfo.absoluteFilePath();
        issue.imagePath = imagePath;
        issue.labelPath = labelInfo.absoluteFilePath();
        issue.split = split;
        issue.message = QStringLiteral("Cannot read label file.");
        issue.repairHint = QStringLiteral("Check file permissions or recreate the label file.");
        issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
        addQualityIssue(context, issue);
        return;
    }

    QSet<QString> seenRows;
    int lineNumber = 0;
    int nonEmptyRows = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString rawLine = QString::fromUtf8(file.readLine()).trimmed();
        if (rawLine.isEmpty()) {
            continue;
        }
        ++nonEmptyRows;
        if (seenRows.contains(rawLine)) {
            QualityIssue issue;
            issue.severity = QStringLiteral("warning");
            issue.code = QStringLiteral("duplicate_label_row");
            issue.filePath = labelInfo.absoluteFilePath();
            issue.imagePath = imagePath;
            issue.labelPath = labelInfo.absoluteFilePath();
            issue.line = lineNumber;
            issue.split = split;
            issue.message = QStringLiteral("Duplicate label row in one annotation file.");
            issue.details = rawLine;
            issue.repairHint = QStringLiteral("Remove duplicate annotation rows.");
            issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
            addQualityIssue(context, issue);
        }
        seenRows.insert(rawLine);

        const QStringList parts = rawLine.split(QRegularExpression(QStringLiteral("\\s+")), QString::SkipEmptyParts);
        bool classOk = false;
        const int classId = parts.value(0).toInt(&classOk);
        if (!classOk || classId < 0 || (classCount > 0 && classId >= classCount)) {
            QualityIssue issue;
            issue.severity = QStringLiteral("error");
            issue.code = QStringLiteral("class_id_out_of_range");
            issue.filePath = labelInfo.absoluteFilePath();
            issue.imagePath = imagePath;
            issue.labelPath = labelInfo.absoluteFilePath();
            issue.line = lineNumber;
            issue.split = split;
            issue.classId = classId;
            issue.message = QStringLiteral("Class id is invalid or outside data.yaml nc.");
            issue.details = rawLine;
            issue.repairHint = QStringLiteral("Fix the class id or update data.yaml names/nc.");
            issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
            addQualityIssue(context, issue);
        } else {
            context.splits[split].classCounts[classId] += 1;
        }

        if (!segmentation) {
            if (parts.size() != 5) {
                QualityIssue issue;
                issue.severity = QStringLiteral("error");
                issue.code = QStringLiteral("invalid_bbox_row");
                issue.filePath = labelInfo.absoluteFilePath();
                issue.imagePath = imagePath;
                issue.labelPath = labelInfo.absoluteFilePath();
                issue.line = lineNumber;
                issue.split = split;
                issue.classId = classId;
                issue.message = QStringLiteral("YOLO detection row must contain class x y w h.");
                issue.details = rawLine;
                issue.repairHint = QStringLiteral("Re-export the bounding box annotation.");
                issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
                addQualityIssue(context, issue);
                continue;
            }
            bool ok = true;
            const double x = parts.at(1).toDouble(&ok);
            bool partOk = false;
            const double y = parts.at(2).toDouble(&partOk);
            ok = ok && partOk;
            const double width = parts.at(3).toDouble(&partOk);
            ok = ok && partOk;
            const double height = parts.at(4).toDouble(&partOk);
            ok = ok && partOk;
            const bool inBounds = ok
                && x >= 0.0 && x <= 1.0
                && y >= 0.0 && y <= 1.0
                && width > 0.0 && width <= 1.0
                && height > 0.0 && height <= 1.0
                && x - width / 2.0 >= 0.0
                && x + width / 2.0 <= 1.0
                && y - height / 2.0 >= 0.0
                && y + height / 2.0 <= 1.0;
            if (!inBounds) {
                QualityIssue issue;
                issue.severity = QStringLiteral("error");
                issue.code = QStringLiteral("bbox_out_of_bounds");
                issue.filePath = labelInfo.absoluteFilePath();
                issue.imagePath = imagePath;
                issue.labelPath = labelInfo.absoluteFilePath();
                issue.line = lineNumber;
                issue.split = split;
                issue.classId = classId;
                issue.message = QStringLiteral("YOLO bbox is invalid or outside normalized image bounds.");
                issue.details = rawLine;
                issue.repairHint = QStringLiteral("Open the sample in X-AnyLabeling and redraw the box.");
                issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
                addQualityIssue(context, issue);
            }
        } else {
            const int coordinateCount = parts.size() - 1;
            if (coordinateCount < 6 || coordinateCount % 2 != 0) {
                QualityIssue issue;
                issue.severity = QStringLiteral("error");
                issue.code = QStringLiteral("polygon_points_too_few");
                issue.filePath = labelInfo.absoluteFilePath();
                issue.imagePath = imagePath;
                issue.labelPath = labelInfo.absoluteFilePath();
                issue.line = lineNumber;
                issue.split = split;
                issue.classId = classId;
                issue.message = QStringLiteral("YOLO segmentation polygon must contain at least 3 points.");
                issue.details = rawLine;
                issue.repairHint = QStringLiteral("Redraw the polygon with at least three points.");
                issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
                addQualityIssue(context, issue);
                continue;
            }
            bool coordinatesOk = true;
            for (int index = 1; index < parts.size(); ++index) {
                bool ok = false;
                const double value = parts.at(index).toDouble(&ok);
                if (!ok || value < 0.0 || value > 1.0) {
                    coordinatesOk = false;
                    break;
                }
            }
            if (!coordinatesOk) {
                QualityIssue issue;
                issue.severity = QStringLiteral("error");
                issue.code = QStringLiteral("polygon_coordinate_out_of_bounds");
                issue.filePath = labelInfo.absoluteFilePath();
                issue.imagePath = imagePath;
                issue.labelPath = labelInfo.absoluteFilePath();
                issue.line = lineNumber;
                issue.split = split;
                issue.classId = classId;
                issue.message = QStringLiteral("YOLO segmentation polygon coordinates must be normalized to [0, 1].");
                issue.details = rawLine;
                issue.repairHint = QStringLiteral("Open the sample in X-AnyLabeling and redraw the polygon.");
                issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
                addQualityIssue(context, issue);
            }
        }
    }

    if (nonEmptyRows == 0) {
        QualityIssue issue;
        issue.severity = QStringLiteral("warning");
        issue.code = QStringLiteral("empty_label");
        issue.filePath = labelInfo.absoluteFilePath();
        issue.imagePath = imagePath;
        issue.labelPath = labelInfo.absoluteFilePath();
        issue.split = split;
        issue.message = QStringLiteral("Label file contains no annotations.");
        issue.repairHint = QStringLiteral("Add annotations or mark the sample as intentionally empty.");
        issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
        addQualityIssue(context, issue);
    }
}

QFileInfoList qualityImageFiles(const QDir& directory)
{
    QFileInfoList files;
    for (const QString& filter : imageNameFilters()) {
        files.append(directory.entryInfoList({filter}, QDir::Files, QDir::Name));
    }
    std::sort(files.begin(), files.end(), [](const QFileInfo& left, const QFileInfo& right) {
        return left.absoluteFilePath().compare(right.absoluteFilePath(), Qt::CaseInsensitive) < 0;
    });
    return files;
}

void scanYoloQuality(DatasetQualityContext& context, bool segmentation)
{
    QString infoError;
    const DetectionDatasetInfo info = readDetectionDatasetInfo(context.datasetPath, &infoError);
    context.classNames = info.classNames;
    while (context.classNames.size() < info.classCount) {
        context.classNames.append(QStringLiteral("class_%1").arg(context.classNames.size()));
    }
    const int classCount = info.classCount;
    const QDir root(context.datasetPath);

    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        const QDir imageDir(root.filePath(QStringLiteral("images/%1").arg(split)));
        const QDir labelDir(root.filePath(QStringLiteral("labels/%1").arg(split)));
        QSet<QString> imageBases;
        const QFileInfoList images = imageDir.exists() ? qualityImageFiles(imageDir) : QFileInfoList();
        for (const QFileInfo& imageInfo : images) {
            if (scanLimitReached(context, imageInfo.absoluteFilePath())) {
                return;
            }
            imageBases.insert(imageInfo.completeBaseName());
            const QString labelPath = labelDir.filePath(imageInfo.completeBaseName() + QStringLiteral(".txt"));
            inspectQualityImage(context, imageInfo.absoluteFilePath(), split, labelPath);
            const QFileInfo labelInfo(labelPath);
            if (!labelInfo.exists()) {
                QualityIssue issue;
                issue.severity = QStringLiteral("error");
                issue.code = QStringLiteral("missing_label");
                issue.filePath = labelPath;
                issue.imagePath = imageInfo.absoluteFilePath();
                issue.labelPath = labelPath;
                issue.split = split;
                issue.message = QStringLiteral("Image is missing its matching YOLO label file.");
                issue.repairHint = QStringLiteral("Create the label file or remove the image from the split.");
                issue.xAnyLabelingSupported = true;
                addQualityIssue(context, issue);
                continue;
            }
            parseYoloLabelFile(context, labelInfo, imageInfo.absoluteFilePath(), split, classCount, segmentation);
        }

        if (labelDir.exists()) {
            const QFileInfoList labels = labelDir.entryInfoList({QStringLiteral("*.txt")}, QDir::Files, QDir::Name);
            for (const QFileInfo& labelInfo : labels) {
                if (!imageBases.contains(labelInfo.completeBaseName())) {
                    QualityIssue issue;
                    issue.severity = QStringLiteral("warning");
                    issue.code = QStringLiteral("orphan_label");
                    issue.filePath = labelInfo.absoluteFilePath();
                    issue.labelPath = labelInfo.absoluteFilePath();
                    issue.split = split;
                    issue.message = QStringLiteral("Label file has no matching image in this split.");
                    issue.repairHint = QStringLiteral("Move the matching image into the split or remove the orphan label.");
                    issue.xAnyLabelingSupported = false;
                    addQualityIssue(context, issue);
                }
            }
        }
    }
}

QString defaultOcrLabelFile(const QDir& root, const QString& format, const QJsonObject& options)
{
    const QString configured = options.value(QStringLiteral("labelFile")).toString();
    if (!configured.isEmpty()) {
        return QFileInfo(configured).isAbsolute() ? configured : root.filePath(configured);
    }
    if (format == QStringLiteral("paddleocr_det")) {
        return QFileInfo::exists(root.filePath(QStringLiteral("det_gt.txt")))
            ? root.filePath(QStringLiteral("det_gt.txt"))
            : root.filePath(QStringLiteral("det_gt_train.txt"));
    }
    return QFileInfo::exists(root.filePath(QStringLiteral("rec_gt.txt")))
        ? root.filePath(QStringLiteral("rec_gt.txt"))
        : root.filePath(QStringLiteral("rec_gt_train.txt"));
}

QSet<QChar> loadQualityDictionary(const QString& path)
{
    QSet<QChar> dictionary;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return dictionary;
    }
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (!line.isEmpty()) {
            dictionary.insert(line.at(0));
        }
    }
    return dictionary;
}

void scanOcrRecQuality(DatasetQualityContext& context, const QJsonObject& options)
{
    const QDir root(context.datasetPath);
    const QString labelFilePath = defaultOcrLabelFile(root, QStringLiteral("paddleocr_rec"), options);
    QFile labelFile(labelFilePath);
    if (!labelFile.exists() || !labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QualityIssue issue;
        issue.severity = QStringLiteral("error");
        issue.code = QStringLiteral("missing_label_file");
        issue.filePath = labelFilePath;
        issue.labelPath = labelFilePath;
        issue.message = QStringLiteral("PaddleOCR Rec label file is missing or unreadable.");
        issue.repairHint = QStringLiteral("Create rec_gt.txt or pass a valid labelFile option.");
        addQualityIssue(context, issue);
        return;
    }

    const QString dictionaryPath = options.value(QStringLiteral("dictionaryFile")).toString(root.filePath(QStringLiteral("dict.txt")));
    const QSet<QChar> dictionary = loadQualityDictionary(dictionaryPath);
    QSet<QString> seenImages;
    int lineNumber = 0;
    while (!labelFile.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(labelFile.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        if (scanLimitReached(context, labelFilePath)) {
            return;
        }
        const int tab = line.indexOf(QLatin1Char('\t'));
        if (tab <= 0) {
            QualityIssue issue;
            issue.severity = QStringLiteral("error");
            issue.code = QStringLiteral("invalid_ocr_row");
            issue.filePath = labelFilePath;
            issue.labelPath = labelFilePath;
            issue.line = lineNumber;
            issue.message = QStringLiteral("OCR row must be '<image path>\\t<label>'.");
            issue.details = line;
            issue.repairHint = QStringLiteral("Fix the row format.");
            addQualityIssue(context, issue);
            continue;
        }
        const QString relativeImage = line.left(tab).trimmed();
        const QString text = line.mid(tab + 1);
        const QString imagePath = root.filePath(relativeImage);
        const QString split = splitFromPath(relativeImage, splitFromPath(QFileInfo(labelFilePath).fileName()));
        context.splits[split].labelCount += 1;
        inspectQualityImage(context, imagePath, split, labelFilePath);

        if (seenImages.contains(relativeImage)) {
            QualityIssue issue;
            issue.severity = QStringLiteral("warning");
            issue.code = QStringLiteral("duplicate_ocr_sample");
            issue.filePath = labelFilePath;
            issue.imagePath = imagePath;
            issue.labelPath = labelFilePath;
            issue.line = lineNumber;
            issue.split = split;
            issue.message = QStringLiteral("OCR sample image appears more than once in the label file.");
            issue.repairHint = QStringLiteral("Remove duplicate OCR rows unless intentionally repeated.");
            issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
            addQualityIssue(context, issue);
        }
        seenImages.insert(relativeImage);

        if (text.isEmpty()) {
            QualityIssue issue;
            issue.severity = QStringLiteral("error");
            issue.code = QStringLiteral("empty_ocr_label");
            issue.filePath = labelFilePath;
            issue.imagePath = imagePath;
            issue.labelPath = labelFilePath;
            issue.line = lineNumber;
            issue.split = split;
            issue.message = QStringLiteral("OCR label text is empty.");
            issue.repairHint = QStringLiteral("Enter the correct transcription or remove the sample.");
            issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
            addQualityIssue(context, issue);
        }
        if (!dictionary.isEmpty()) {
            for (const QChar ch : text) {
                if (!dictionary.contains(ch)) {
                    QualityIssue issue;
                    issue.severity = QStringLiteral("error");
                    issue.code = QStringLiteral("char_not_in_dictionary");
                    issue.filePath = labelFilePath;
                    issue.imagePath = imagePath;
                    issue.labelPath = labelFilePath;
                    issue.line = lineNumber;
                    issue.split = split;
                    issue.message = QStringLiteral("OCR label contains a character missing from dict.txt.");
                    issue.details = QString(ch);
                    issue.repairHint = QStringLiteral("Add the character to dict.txt or correct the label.");
                    issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
                    addQualityIssue(context, issue);
                    break;
                }
            }
        }
    }
}

void scanOcrDetQuality(DatasetQualityContext& context, const QJsonObject& options)
{
    const QDir root(context.datasetPath);
    const QString labelFilePath = defaultOcrLabelFile(root, QStringLiteral("paddleocr_det"), options);
    QFile labelFile(labelFilePath);
    if (!labelFile.exists() || !labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QualityIssue issue;
        issue.severity = QStringLiteral("error");
        issue.code = QStringLiteral("missing_label_file");
        issue.filePath = labelFilePath;
        issue.labelPath = labelFilePath;
        issue.message = QStringLiteral("PaddleOCR Det label file is missing or unreadable.");
        issue.repairHint = QStringLiteral("Create det_gt.txt or pass a valid labelFile option.");
        addQualityIssue(context, issue);
        return;
    }

    QSet<QString> seenImages;
    int lineNumber = 0;
    while (!labelFile.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(labelFile.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        if (scanLimitReached(context, labelFilePath)) {
            return;
        }
        const int tab = line.indexOf(QLatin1Char('\t'));
        if (tab <= 0) {
            QualityIssue issue;
            issue.severity = QStringLiteral("error");
            issue.code = QStringLiteral("invalid_det_row");
            issue.filePath = labelFilePath;
            issue.labelPath = labelFilePath;
            issue.line = lineNumber;
            issue.message = QStringLiteral("PaddleOCR Det row must be '<image path>\\t<json boxes>'.");
            issue.details = line;
            issue.repairHint = QStringLiteral("Fix the row format.");
            addQualityIssue(context, issue);
            continue;
        }
        const QString relativeImage = line.left(tab).trimmed();
        const QString jsonText = line.mid(tab + 1).trimmed();
        const QString imagePath = root.filePath(relativeImage);
        const QString split = splitFromPath(relativeImage, splitFromPath(QFileInfo(labelFilePath).fileName()));
        context.splits[split].labelCount += 1;
        const QSize imageSize = inspectQualityImage(context, imagePath, split, labelFilePath);

        if (seenImages.contains(relativeImage)) {
            QualityIssue issue;
            issue.severity = QStringLiteral("warning");
            issue.code = QStringLiteral("duplicate_det_sample");
            issue.filePath = labelFilePath;
            issue.imagePath = imagePath;
            issue.labelPath = labelFilePath;
            issue.line = lineNumber;
            issue.split = split;
            issue.message = QStringLiteral("PaddleOCR Det sample image appears more than once.");
            issue.repairHint = QStringLiteral("Remove duplicate Det rows unless intentionally repeated.");
            issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
            addQualityIssue(context, issue);
        }
        seenImages.insert(relativeImage);

        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(jsonText.toUtf8(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isArray()) {
            QualityIssue issue;
            issue.severity = QStringLiteral("error");
            issue.code = QStringLiteral("invalid_det_json");
            issue.filePath = labelFilePath;
            issue.imagePath = imagePath;
            issue.labelPath = labelFilePath;
            issue.line = lineNumber;
            issue.split = split;
            issue.message = QStringLiteral("PaddleOCR Det annotation must be a JSON array.");
            issue.details = parseError.errorString();
            issue.repairHint = QStringLiteral("Re-export Det annotations as valid PaddleOCR JSON.");
            issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
            addQualityIssue(context, issue);
            continue;
        }
        const QJsonArray boxes = document.array();
        if (boxes.isEmpty()) {
            QualityIssue issue;
            issue.severity = QStringLiteral("warning");
            issue.code = QStringLiteral("empty_det_boxes");
            issue.filePath = labelFilePath;
            issue.imagePath = imagePath;
            issue.labelPath = labelFilePath;
            issue.line = lineNumber;
            issue.split = split;
            issue.message = QStringLiteral("PaddleOCR Det sample has no text boxes.");
            issue.repairHint = QStringLiteral("Add text boxes or remove the sample.");
            issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
            addQualityIssue(context, issue);
        }
        for (const QJsonValue& boxValue : boxes) {
            const QJsonObject box = boxValue.toObject();
            if (!box.contains(QStringLiteral("transcription"))) {
                QualityIssue issue;
                issue.severity = QStringLiteral("error");
                issue.code = QStringLiteral("missing_transcription");
                issue.filePath = labelFilePath;
                issue.imagePath = imagePath;
                issue.labelPath = labelFilePath;
                issue.line = lineNumber;
                issue.split = split;
                issue.message = QStringLiteral("PaddleOCR Det box is missing transcription.");
                issue.repairHint = QStringLiteral("Add transcription to each Det box.");
                issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
                addQualityIssue(context, issue);
                break;
            }
            const QJsonArray points = box.value(QStringLiteral("points")).toArray();
            if (points.size() < 4) {
                QualityIssue issue;
                issue.severity = QStringLiteral("error");
                issue.code = QStringLiteral("det_points_too_few");
                issue.filePath = labelFilePath;
                issue.imagePath = imagePath;
                issue.labelPath = labelFilePath;
                issue.line = lineNumber;
                issue.split = split;
                issue.message = QStringLiteral("PaddleOCR Det box needs at least four points.");
                issue.repairHint = QStringLiteral("Redraw the text polygon.");
                issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
                addQualityIssue(context, issue);
                break;
            }
            bool pointsOk = true;
            bool pointOutOfBounds = false;
            for (const QJsonValue& pointValue : points) {
                const QJsonArray point = pointValue.toArray();
                if (point.size() < 2 || !point.at(0).isDouble() || !point.at(1).isDouble()) {
                    pointsOk = false;
                    break;
                }
                const double x = point.at(0).toDouble();
                const double y = point.at(1).toDouble();
                if (x < 0.0 || y < 0.0) {
                    pointsOk = false;
                    break;
                }
                if (imageSize.isValid() && !imageSize.isEmpty()
                    && (x > imageSize.width() || y > imageSize.height())) {
                    pointOutOfBounds = true;
                }
            }
            if (!pointsOk || pointOutOfBounds) {
                QualityIssue issue;
                issue.severity = QStringLiteral("error");
                issue.code = pointsOk ? QStringLiteral("det_point_out_of_bounds") : QStringLiteral("invalid_det_point");
                issue.filePath = labelFilePath;
                issue.imagePath = imagePath;
                issue.labelPath = labelFilePath;
                issue.line = lineNumber;
                issue.split = split;
                issue.message = pointsOk
                    ? QStringLiteral("PaddleOCR Det point is outside image bounds.")
                    : QStringLiteral("PaddleOCR Det point must be a non-negative numeric pair.");
                issue.repairHint = QStringLiteral("Open the sample and redraw the Det polygon.");
                issue.xAnyLabelingSupported = QFileInfo::exists(imagePath);
                addQualityIssue(context, issue);
                break;
            }
        }
    }
}

void addDistributionWarnings(DatasetQualityContext& context)
{
    const QStringList expectedSplits = {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")};
    QMap<int, int> totals;
    int totalAnnotations = 0;
    for (auto splitIt = context.splits.constBegin(); splitIt != context.splits.constEnd(); ++splitIt) {
        for (auto classIt = splitIt.value().classCounts.constBegin(); classIt != splitIt.value().classCounts.constEnd(); ++classIt) {
            totals[classIt.key()] += classIt.value();
            totalAnnotations += classIt.value();
        }
    }
    if (totalAnnotations <= 0) {
        return;
    }
    for (auto totalIt = totals.constBegin(); totalIt != totals.constEnd(); ++totalIt) {
        if (totalIt.value() <= 0) {
            continue;
        }
        const double overallRatio = static_cast<double>(totalIt.value()) / static_cast<double>(totalAnnotations);
        for (const QString& split : expectedSplits) {
            const SplitQualityStats stats = context.splits.value(split);
            const int splitClassCount = stats.classCounts.value(totalIt.key());
            if (stats.imageCount > 0 && splitClassCount == 0) {
                QJsonObject warning;
                warning.insert(QStringLiteral("code"), QStringLiteral("class_missing_in_split"));
                warning.insert(QStringLiteral("split"), split);
                warning.insert(QStringLiteral("classId"), totalIt.key());
                warning.insert(QStringLiteral("className"), classNameForId(context.classNames, totalIt.key()));
                warning.insert(QStringLiteral("message"), QStringLiteral("Class is present in the dataset but missing from this split."));
                context.distributionWarnings.append(warning);

                QualityIssue issue;
                issue.severity = QStringLiteral("warning");
                issue.code = QStringLiteral("class_missing_in_split");
                issue.filePath = context.datasetPath;
                issue.split = split;
                issue.classId = totalIt.key();
                issue.message = warning.value(QStringLiteral("message")).toString();
                issue.details = QStringLiteral("class=%1 total=%2").arg(totalIt.key()).arg(totalIt.value());
                issue.repairHint = QStringLiteral("Review split ratios or add representative samples to the split.");
                addQualityIssue(context, issue);
            } else if (stats.imageCount > 0 && splitClassCount > 0) {
                const int splitTotal = std::accumulate(stats.classCounts.constBegin(), stats.classCounts.constEnd(), 0,
                    [](int sum, const int count) { return sum + count; });
                const double splitRatio = splitTotal > 0
                    ? static_cast<double>(splitClassCount) / static_cast<double>(splitTotal)
                    : 0.0;
                if (qAbs(splitRatio - overallRatio) >= context.distributionWarningThreshold) {
                    QJsonObject warning;
                    warning.insert(QStringLiteral("code"), QStringLiteral("class_distribution_shift"));
                    warning.insert(QStringLiteral("split"), split);
                    warning.insert(QStringLiteral("classId"), totalIt.key());
                    warning.insert(QStringLiteral("className"), classNameForId(context.classNames, totalIt.key()));
                    warning.insert(QStringLiteral("expectedRatio"), overallRatio);
                    warning.insert(QStringLiteral("splitRatio"), splitRatio);
                    warning.insert(QStringLiteral("message"), QStringLiteral("Class ratio in this split differs significantly from the dataset-wide ratio."));
                    context.distributionWarnings.append(warning);

                    QualityIssue issue;
                    issue.severity = QStringLiteral("warning");
                    issue.code = QStringLiteral("class_distribution_shift");
                    issue.filePath = context.datasetPath;
                    issue.split = split;
                    issue.classId = totalIt.key();
                    issue.message = warning.value(QStringLiteral("message")).toString();
                    issue.details = QStringLiteral("expected=%1 split=%2").arg(overallRatio, 0, 'f', 4).arg(splitRatio, 0, 'f', 4);
                    issue.repairHint = QStringLiteral("Review split sampling or rebalance the class distribution.");
                    addQualityIssue(context, issue);
                }
            }
        }
    }
}

QJsonObject classDistributionObject(const DatasetQualityContext& context)
{
    QMap<int, int> totals;
    for (auto splitIt = context.splits.constBegin(); splitIt != context.splits.constEnd(); ++splitIt) {
        for (auto classIt = splitIt.value().classCounts.constBegin(); classIt != splitIt.value().classCounts.constEnd(); ++classIt) {
            totals[classIt.key()] += classIt.value();
        }
    }
    QJsonObject object;
    for (auto it = totals.constBegin(); it != totals.constEnd(); ++it) {
        object.insert(QString::number(it.key()), it.value());
    }
    return object;
}

QJsonArray classDistributionBySplit(const DatasetQualityContext& context)
{
    QJsonArray rows;
    for (auto splitIt = context.splits.constBegin(); splitIt != context.splits.constEnd(); ++splitIt) {
        int splitTotal = 0;
        for (auto classIt = splitIt.value().classCounts.constBegin(); classIt != splitIt.value().classCounts.constEnd(); ++classIt) {
            splitTotal += classIt.value();
        }
        for (auto classIt = splitIt.value().classCounts.constBegin(); classIt != splitIt.value().classCounts.constEnd(); ++classIt) {
            QJsonObject row;
            row.insert(QStringLiteral("split"), splitIt.key());
            row.insert(QStringLiteral("classId"), classIt.key());
            row.insert(QStringLiteral("className"), classNameForId(context.classNames, classIt.key()));
            row.insert(QStringLiteral("count"), classIt.value());
            row.insert(QStringLiteral("percent"), splitTotal > 0 ? static_cast<double>(classIt.value()) / static_cast<double>(splitTotal) : 0.0);
            rows.append(row);
        }
    }
    return rows;
}

QString classDistributionCsvV2(const DatasetQualityContext& context)
{
    QString csv = QStringLiteral("split,classId,className,count,percent\n");
    const QJsonArray rows = classDistributionBySplit(context);
    for (const QJsonValue& value : rows) {
        const QJsonObject row = value.toObject();
        csv += QStringLiteral("%1,%2,%3,%4,%5\n")
            .arg(csvEscape(row.value(QStringLiteral("split")).toString()))
            .arg(row.value(QStringLiteral("classId")).toInt())
            .arg(csvEscape(row.value(QStringLiteral("className")).toString()))
            .arg(row.value(QStringLiteral("count")).toInt())
            .arg(row.value(QStringLiteral("percent")).toDouble(), 0, 'f', 6);
    }
    return csv;
}

QJsonObject splitCountsObject(const DatasetQualityContext& context)
{
    QJsonObject object;
    for (auto it = context.splits.constBegin(); it != context.splits.constEnd(); ++it) {
        object.insert(it.key(), it.value().imageCount);
    }
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        if (!object.contains(split)) {
            object.insert(split, 0);
        }
    }
    return object;
}

QJsonObject imageStatisticsObject(const DatasetQualityContext& context)
{
    QJsonObject bySplit;
    int totalImages = 0;
    int readable = 0;
    int unreadable = 0;
    int zeroByte = 0;
    int duplicate = 0;
    for (auto it = context.splits.constBegin(); it != context.splits.constEnd(); ++it) {
        const SplitQualityStats& stats = it.value();
        QJsonObject row;
        row.insert(QStringLiteral("imageCount"), stats.imageCount);
        row.insert(QStringLiteral("readableImageCount"), stats.readableImageCount);
        row.insert(QStringLiteral("unreadableImageCount"), stats.unreadableImageCount);
        row.insert(QStringLiteral("zeroByteImageCount"), stats.zeroByteImageCount);
        row.insert(QStringLiteral("duplicateImageCount"), stats.duplicateImageCount);
        row.insert(QStringLiteral("avgWidth"), stats.readableImageCount > 0 ? stats.widthSum / stats.readableImageCount : 0.0);
        row.insert(QStringLiteral("avgHeight"), stats.readableImageCount > 0 ? stats.heightSum / stats.readableImageCount : 0.0);
        row.insert(QStringLiteral("aspectMin"), stats.aspectMin);
        row.insert(QStringLiteral("aspectMax"), stats.aspectMax);
        bySplit.insert(it.key(), row);
        totalImages += stats.imageCount;
        readable += stats.readableImageCount;
        unreadable += stats.unreadableImageCount;
        zeroByte += stats.zeroByteImageCount;
        duplicate += stats.duplicateImageCount;
    }
    QJsonObject summary;
    summary.insert(QStringLiteral("totalImages"), totalImages);
    summary.insert(QStringLiteral("readableImages"), readable);
    summary.insert(QStringLiteral("unreadableImages"), unreadable);
    summary.insert(QStringLiteral("zeroByteImages"), zeroByte);
    summary.insert(QStringLiteral("duplicateImages"), duplicate);
    summary.insert(QStringLiteral("bySplit"), bySplit);
    return summary;
}

QString imageStatisticsCsv(const DatasetQualityContext& context)
{
    QString csv = QStringLiteral("split,imageCount,readableImageCount,unreadableImageCount,zeroByteImageCount,duplicateImageCount,avgWidth,avgHeight,aspectMin,aspectMax\n");
    for (auto it = context.splits.constBegin(); it != context.splits.constEnd(); ++it) {
        const SplitQualityStats& stats = it.value();
        csv += QStringLiteral("%1,%2,%3,%4,%5,%6,%7,%8,%9,%10\n")
            .arg(csvEscape(it.key()))
            .arg(stats.imageCount)
            .arg(stats.readableImageCount)
            .arg(stats.unreadableImageCount)
            .arg(stats.zeroByteImageCount)
            .arg(stats.duplicateImageCount)
            .arg(stats.readableImageCount > 0 ? stats.widthSum / stats.readableImageCount : 0.0, 0, 'f', 2)
            .arg(stats.readableImageCount > 0 ? stats.heightSum / stats.readableImageCount : 0.0, 0, 'f', 2)
            .arg(stats.aspectMin, 0, 'f', 4)
            .arg(stats.aspectMax, 0, 'f', 4);
    }
    return csv;
}

QString splitDistributionCsv(const DatasetQualityContext& context)
{
    QString csv = QStringLiteral("split,imageCount,labelCount,problemCount,classCount\n");
    for (auto it = context.splits.constBegin(); it != context.splits.constEnd(); ++it) {
        csv += QStringLiteral("%1,%2,%3,%4,%5\n")
            .arg(csvEscape(it.key()))
            .arg(it.value().imageCount)
            .arg(it.value().labelCount)
            .arg(it.value().problemCount)
            .arg(it.value().classCounts.size());
    }
    return csv;
}

QJsonArray issueArray(const DatasetValidationResult& validation, int maxItems)
{
    QJsonArray issues;
    int count = 0;
    for (const DatasetValidationResult::Issue& issue : validation.issues) {
        if (maxItems > 0 && count >= maxItems) {
            break;
        }
        issues.append(issue.toJson());
        ++count;
    }
    return issues;
}

QString htmlEscape(QString value)
{
    return value.replace(QLatin1Char('&'), QStringLiteral("&amp;"))
        .replace(QLatin1Char('<'), QStringLiteral("&lt;"))
        .replace(QLatin1Char('>'), QStringLiteral("&gt;"))
        .replace(QLatin1Char('"'), QStringLiteral("&quot;"));
}

QString htmlReport(const QJsonObject& context)
{
    QString html;
    html += QStringLiteral("<!doctype html><html><head><meta charset=\"utf-8\"><title>AITrain Delivery Report</title>");
    html += QStringLiteral("<style>body{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#111827;background:#f4f6f8}"
                           "section{background:white;border:1px solid #d8dee6;border-radius:8px;padding:18px;margin:14px 0}"
                           "pre{white-space:pre-wrap;background:#111827;color:#f9fafb;padding:14px;border-radius:6px}"
                           "h1,h2{margin:0 0 12px}</style></head><body>");
    html += QStringLiteral("<h1>AITrain Studio Training Delivery Report</h1>");
    html += QStringLiteral("<section><h2>Summary</h2><p>Generated at %1.</p><p>This report is an offline local delivery artifact. Scaffold or diagnostic backends remain explicitly marked in source artifacts.</p></section>")
        .arg(htmlEscape(nowIso()));
    html += QStringLiteral("<section><h2>Context</h2><pre>%1</pre></section>")
        .arg(htmlEscape(QString::fromUtf8(QJsonDocument(context).toJson(QJsonDocument::Indented))));
    html += QStringLiteral("</body></html>");
    return html;
}

double boxArea(const DetectionBox& box)
{
    return qMax(0.0, box.width) * qMax(0.0, box.height);
}

double boxIou(const DetectionBox& left, const DetectionBox& right)
{
    const double leftX1 = left.xCenter - left.width / 2.0;
    const double leftY1 = left.yCenter - left.height / 2.0;
    const double leftX2 = left.xCenter + left.width / 2.0;
    const double leftY2 = left.yCenter + left.height / 2.0;
    const double rightX1 = right.xCenter - right.width / 2.0;
    const double rightY1 = right.yCenter - right.height / 2.0;
    const double rightX2 = right.xCenter + right.width / 2.0;
    const double rightY2 = right.yCenter + right.height / 2.0;
    const double intersectionWidth = qMax(0.0, qMin(leftX2, rightX2) - qMax(leftX1, rightX1));
    const double intersectionHeight = qMax(0.0, qMin(leftY2, rightY2) - qMax(leftY1, rightY1));
    const double intersection = intersectionWidth * intersectionHeight;
    const double areaSum = boxArea(left) + boxArea(right) - intersection;
    return areaSum > 0.0 ? intersection / areaSum : 0.0;
}

QJsonObject detectionBoxToJson(const DetectionBox& box)
{
    return QJsonObject{
        {QStringLiteral("classId"), box.classId},
        {QStringLiteral("xCenter"), box.xCenter},
        {QStringLiteral("yCenter"), box.yCenter},
        {QStringLiteral("width"), box.width},
        {QStringLiteral("height"), box.height}
    };
}

bool detectionSplitExists(const QString& datasetPath, const QString& split)
{
    const QDir root(datasetPath);
    return QDir(root.filePath(QStringLiteral("images/%1").arg(split))).exists()
        && QDir(root.filePath(QStringLiteral("labels/%1").arg(split))).exists();
}

QString selectDetectionSplit(const QString& datasetPath)
{
    for (const QString& split : {QStringLiteral("val"), QStringLiteral("test"), QStringLiteral("train")}) {
        if (detectionSplitExists(datasetPath, split)) {
            return split;
        }
    }
    return QString();
}

QVector<DetectionPrediction> runDetectionPredictions(
    const QString& modelPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* runtime,
    QString* error)
{
    const QString suffix = QFileInfo(modelPath).suffix().toLower();
    if (suffix == QStringLiteral("onnx")) {
        if (runtime) {
            *runtime = QStringLiteral("onnxruntime");
        }
        return predictDetectionOnnxRuntime(modelPath, imagePath, options, error);
    }
    if (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan")) {
        if (runtime) {
            *runtime = QStringLiteral("tensorrt");
        }
        if (!isTensorRtInferenceAvailable()) {
            if (error) {
                *error = QStringLiteral("hardware-blocked: TensorRT evaluation requires a compatible RTX / SM 75+ acceptance machine and runtime.");
            }
            return {};
        }
        return predictDetectionTensorRt(modelPath, imagePath, options, error);
    }

    DetectionBaselineCheckpoint checkpoint;
    if (!loadDetectionBaselineCheckpoint(modelPath, &checkpoint, error)) {
        return {};
    }
    if (runtime) {
        *runtime = QStringLiteral("tiny_detector");
    }
    return predictDetectionBaseline(checkpoint, imagePath, options, error);
}

struct DetectionEvaluationItem {
    int classId = 0;
    double confidence = 0.0;
    bool truePositive = false;
};

struct DetectionClassStats {
    int gt = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    double precision = 0.0;
    double recall = 0.0;
    double ap50 = 0.0;
    QVector<DetectionEvaluationItem> items;
};

double ap50FromItems(QVector<DetectionEvaluationItem> items, int gtCount)
{
    if (gtCount <= 0) {
        return 0.0;
    }
    std::sort(items.begin(), items.end(), [](const DetectionEvaluationItem& left, const DetectionEvaluationItem& right) {
        return left.confidence > right.confidence;
    });

    QVector<double> recalls;
    QVector<double> precisions;
    int tp = 0;
    int fp = 0;
    for (const DetectionEvaluationItem& item : items) {
        if (item.truePositive) {
            ++tp;
        } else {
            ++fp;
        }
        recalls.append(static_cast<double>(tp) / static_cast<double>(gtCount));
        precisions.append(static_cast<double>(tp) / static_cast<double>(qMax(1, tp + fp)));
    }

    double ap = 0.0;
    for (int threshold = 0; threshold <= 100; ++threshold) {
        const double recallThreshold = static_cast<double>(threshold) / 100.0;
        double precisionAtRecall = 0.0;
        for (int index = 0; index < recalls.size(); ++index) {
            if (recalls.at(index) >= recallThreshold) {
                precisionAtRecall = qMax(precisionAtRecall, precisions.at(index));
            }
        }
        ap += precisionAtRecall;
    }
    return ap / 101.0;
}

QString perClassMetricsCsv(const QStringList& classNames, const QVector<DetectionClassStats>& stats)
{
    QString csv = QStringLiteral("classId,className,gt,tp,fp,fn,precision,recall,ap50\n");
    for (int classId = 0; classId < stats.size(); ++classId) {
        const DetectionClassStats& item = stats.at(classId);
        const QString className = classId < classNames.size() && !classNames.at(classId).isEmpty()
            ? classNames.at(classId)
            : QStringLiteral("class_%1").arg(classId);
        csv += QStringLiteral("%1,%2,%3,%4,%5,%6,%7,%8,%9\n")
            .arg(classId)
            .arg(csvEscape(className))
            .arg(item.gt)
            .arg(item.tp)
            .arg(item.fp)
            .arg(item.fn)
            .arg(item.precision, 0, 'f', 6)
            .arg(item.recall, 0, 'f', 6)
            .arg(item.ap50, 0, 'f', 6);
    }
    return csv;
}

QString confusionMatrixCsv(const QStringList& classNames, const QVector<QVector<int>>& matrix)
{
    QString csv = QStringLiteral("actual\\predicted");
    for (int classId = 0; classId < classNames.size(); ++classId) {
        csv += QStringLiteral(",%1").arg(csvEscape(classNames.at(classId)));
    }
    csv += QStringLiteral(",background\n");
    for (int row = 0; row < matrix.size(); ++row) {
        const QString rowName = row < classNames.size()
            ? classNames.at(row)
            : QStringLiteral("background");
        csv += csvEscape(rowName);
        for (int column = 0; column < matrix.at(row).size(); ++column) {
            csv += QStringLiteral(",%1").arg(matrix.at(row).at(column));
        }
        csv += QLatin1Char('\n');
    }
    return csv;
}

WorkflowResult resultFromReport(const QString& reportPath, const QJsonObject& payload)
{
    WorkflowResult result;
    result.ok = true;
    result.reportPath = reportPath;
    result.payload = payload;
    return result;
}

WorkflowResult failedResult(const QString& error)
{
    WorkflowResult result;
    result.ok = false;
    result.error = error;
    return result;
}

} // namespace

WorkflowResult createDatasetSnapshotReport(const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options)
{
    const QDir root(datasetPath);
    if (!root.exists()) {
        return failedResult(QStringLiteral("Dataset directory does not exist: %1").arg(datasetPath));
    }
    const int maxFiles = options.value(QStringLiteral("maxFiles")).toInt(20000);
    const QFileInfoList files = collectFilesRecursive(datasetPath, maxFiles);
    QJsonArray fileArray;
    QJsonArray keyFileArray;
    QJsonObject roleCounts;
    QCryptographicHash manifestHash(QCryptographicHash::Sha256);
    qint64 totalBytes = 0;
    QString error;
    for (const QFileInfo& fileInfo : files) {
        qint64 fileSize = 0;
        const QByteArray hash = fileSha256(fileInfo.absoluteFilePath(), &fileSize, &error);
        if (!error.isEmpty()) {
            return failedResult(error);
        }
        totalBytes += fileSize;
        const QString relativePath = cleanRelativePath(root, fileInfo.absoluteFilePath());
        const QString role = snapshotFileRole(relativePath, fileInfo);
        QJsonObject fileObject;
        fileObject.insert(QStringLiteral("path"), relativePath);
        fileObject.insert(QStringLiteral("role"), role);
        fileObject.insert(QStringLiteral("size"), QString::number(fileSize));
        fileObject.insert(QStringLiteral("mtime"), fileInfo.lastModified().toUTC().toString(Qt::ISODateWithMs));
        fileObject.insert(QStringLiteral("sha256"), QString::fromLatin1(hash));
        fileArray.append(fileObject);
        manifestHash.addData(relativePath.toUtf8());
        manifestHash.addData("\0", 1);
        manifestHash.addData(hash);
        manifestHash.addData("\0", 1);
        roleCounts.insert(role, roleCounts.value(role).toInt() + 1);
        if (isSnapshotKeyRole(role)) {
            QJsonObject keyFile;
            keyFile.insert(QStringLiteral("path"), relativePath);
            keyFile.insert(QStringLiteral("role"), role);
            keyFile.insert(QStringLiteral("sha256"), QString::fromLatin1(hash));
            keyFileArray.append(keyFile);
        }
    }

    QJsonObject manifest;
    manifest.insert(QStringLiteral("schemaVersion"), 1);
    manifest.insert(QStringLiteral("kind"), QStringLiteral("dataset_snapshot"));
    manifest.insert(QStringLiteral("createdAt"), nowIso());
    manifest.insert(QStringLiteral("datasetPath"), datasetPath);
    manifest.insert(QStringLiteral("format"), format);
    manifest.insert(QStringLiteral("fileCount"), files.size());
    manifest.insert(QStringLiteral("totalBytes"), QString::number(totalBytes));
    manifest.insert(QStringLiteral("contentHash"), QString::fromLatin1(manifestHash.result().toHex()));
    manifest.insert(QStringLiteral("splits"), countImageSplits(datasetPath));
    manifest.insert(QStringLiteral("roleCounts"), roleCounts);
    manifest.insert(QStringLiteral("keyFiles"), keyFileArray);
    manifest.insert(QStringLiteral("imageCount"), roleCounts.value(QStringLiteral("image")).toInt());
    manifest.insert(QStringLiteral("labelCount"), roleCounts.value(QStringLiteral("label")).toInt());
    manifest.insert(QStringLiteral("files"), fileArray);

    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("dataset_snapshot_manifest.json"));
    if (!writeJsonFile(reportPath, manifest, &error)) {
        return failedResult(error);
    }
    manifest.insert(QStringLiteral("manifestPath"), reportPath);
    return resultFromReport(reportPath, manifest);
}

WorkflowResult curateDatasetQualityReport(const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options)
{
    const QDir root(datasetPath);
    if (!root.exists()) {
        return failedResult(QStringLiteral("Dataset directory does not exist for quality scan: %1").arg(datasetPath));
    }
    if (!isSupportedQualityFormat(format)) {
        return failedResult(QStringLiteral("Unsupported dataset quality format '%1' for dataset: %2").arg(format, datasetPath));
    }

    DatasetQualityContext context;
    context.datasetPath = datasetPath;
    context.format = format;
    context.maxIssues = options.value(QStringLiteral("maxIssues")).toInt(500);
    context.maxProblemSamples = options.value(QStringLiteral("maxProblemSamples")).toInt(500);
    context.maxFiles = options.value(QStringLiteral("maxFiles")).toInt(20000);
    context.duplicateHashLimit = options.value(QStringLiteral("duplicateHashLimit")).toInt(20000);
    context.distributionWarningThreshold = options.value(QStringLiteral("distributionWarningThreshold")).toDouble(0.25);
    context.exportXAnyLabelingFixList = options.value(QStringLiteral("exportXAnyLabelingFixList")).toBool(true);

    DatasetValidationResult validation = validateByFormat(datasetPath, format, options);
    addValidationIssues(context, validation);
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt")) {
        scanYoloQuality(context, false);
    } else if (format == QStringLiteral("yolo_segmentation")) {
        scanYoloQuality(context, true);
    } else if (format == QStringLiteral("paddleocr_rec")) {
        scanOcrRecQuality(context, options);
    } else if (format == QStringLiteral("paddleocr_det")) {
        scanOcrDetQuality(context, options);
    }
    addDistributionWarnings(context);

    const QJsonObject classCounts = classDistributionObject(context);
    const QJsonObject imageStats = imageStatisticsObject(context);
    const bool qualityOk = context.severityCounts.value(QStringLiteral("error")).toInt() == 0;
    QJsonObject summary;
    summary.insert(QStringLiteral("sampleCount"), validation.sampleCount);
    summary.insert(QStringLiteral("issueCount"), context.totalIssueCount);
    summary.insert(QStringLiteral("problemSampleCount"), context.problemSamples.size());
    summary.insert(QStringLiteral("errorCount"), context.severityCounts.value(QStringLiteral("error")).toInt());
    summary.insert(QStringLiteral("warningCount"), context.severityCounts.value(QStringLiteral("warning")).toInt());
    summary.insert(QStringLiteral("infoCount"), context.severityCounts.value(QStringLiteral("info")).toInt());
    summary.insert(QStringLiteral("duplicateImageCount"), imageStats.value(QStringLiteral("duplicateImages")).toInt());
    summary.insert(QStringLiteral("truncated"), context.issueLimitReached || context.problemSampleLimitReached || context.scanLimitReached);

    QJsonObject report = validation.toJson();
    report.insert(QStringLiteral("schemaVersion"), 2);
    report.insert(QStringLiteral("kind"), QStringLiteral("dataset_quality_report"));
    report.insert(QStringLiteral("createdAt"), nowIso());
    report.insert(QStringLiteral("checkedAt"), nowIso());
    report.insert(QStringLiteral("datasetPath"), datasetPath);
    report.insert(QStringLiteral("format"), format);
    report.insert(QStringLiteral("ok"), qualityOk);
    report.insert(QStringLiteral("summary"), summary);
    report.insert(QStringLiteral("issues"), context.issues);
    report.insert(QStringLiteral("problemSamples"), context.problemSamples);
    report.insert(QStringLiteral("issueCounts"), context.issueCounts);
    report.insert(QStringLiteral("severityCounts"), context.severityCounts);
    report.insert(QStringLiteral("splitCounts"), splitCountsObject(context));
    report.insert(QStringLiteral("classDistribution"), classCounts);
    report.insert(QStringLiteral("classDistributionBySplit"), classDistributionBySplit(context));
    report.insert(QStringLiteral("imageStatistics"), imageStats);
    report.insert(QStringLiteral("distributionWarnings"), context.distributionWarnings);
    report.insert(QStringLiteral("legacyValidation"), validation.toJson());
    report.insert(QStringLiteral("scanLimitReached"), context.scanLimitReached);
    report.insert(QStringLiteral("issueLimitReached"), context.issueLimitReached);
    report.insert(QStringLiteral("problemSampleLimitReached"), context.problemSampleLimitReached);
    report.insert(QStringLiteral("scaffold"), false);

    QString error;
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("dataset_quality_report.json"));
    const QString csvPath = QDir(outputPath).filePath(QStringLiteral("class_distribution.csv"));
    const QString problemPath = QDir(outputPath).filePath(QStringLiteral("problem_samples.json"));
    const QString imageStatsPath = QDir(outputPath).filePath(QStringLiteral("image_statistics.csv"));
    const QString splitDistributionPath = QDir(outputPath).filePath(QStringLiteral("split_distribution.csv"));
    const QString xAnyFixListPath = QDir(outputPath).filePath(QStringLiteral("xanylabeling_fix_list.txt"));
    const QString xAnyFixManifestPath = QDir(outputPath).filePath(QStringLiteral("xanylabeling_fix_manifest.json"));

    QJsonObject problems;
    problems.insert(QStringLiteral("schemaVersion"), 2);
    problems.insert(QStringLiteral("datasetPath"), datasetPath);
    problems.insert(QStringLiteral("format"), format);
    problems.insert(QStringLiteral("issues"), context.issues);
    problems.insert(QStringLiteral("samples"), context.problemSamples);
    problems.insert(QStringLiteral("issueCounts"), context.issueCounts);
    problems.insert(QStringLiteral("severityCounts"), context.severityCounts);
    problems.insert(QStringLiteral("truncated"), context.problemSampleLimitReached);
    if (!writeJsonFile(problemPath, problems, &error)) {
        return failedResult(error);
    }

    QJsonObject xAnyManifest;
    xAnyManifest.insert(QStringLiteral("schemaVersion"), 1);
    xAnyManifest.insert(QStringLiteral("kind"), QStringLiteral("xanylabeling_fix_manifest"));
    xAnyManifest.insert(QStringLiteral("datasetPath"), datasetPath);
    xAnyManifest.insert(QStringLiteral("format"), format);
    xAnyManifest.insert(QStringLiteral("note"), QStringLiteral("This is an AITrain repair manifest for launching external X-AnyLabeling, not an official X-AnyLabeling project file."));
    xAnyManifest.insert(QStringLiteral("fixListPath"), xAnyFixListPath);
    xAnyManifest.insert(QStringLiteral("samples"), context.problemSamples);

    if (!writeTextFile(csvPath, classDistributionCsvV2(context), &error)
        || !writeTextFile(imageStatsPath, imageStatisticsCsv(context), &error)
        || !writeTextFile(splitDistributionPath, splitDistributionCsv(context), &error)
        || !writeTextFile(xAnyFixListPath, context.xAnyFixRows.join(QLatin1Char('\n')) + QLatin1Char('\n'), &error)
        || !writeJsonFile(xAnyFixManifestPath, xAnyManifest, &error)) {
        return failedResult(error);
    }

    report.insert(QStringLiteral("reportPath"), reportPath);
    report.insert(QStringLiteral("classDistributionPath"), csvPath);
    report.insert(QStringLiteral("problemSamplesPath"), problemPath);
    report.insert(QStringLiteral("imageStatisticsPath"), imageStatsPath);
    report.insert(QStringLiteral("splitDistributionPath"), splitDistributionPath);
    report.insert(QStringLiteral("xAnyLabelingFixListPath"), xAnyFixListPath);
    report.insert(QStringLiteral("xAnyLabelingFixManifestPath"), xAnyFixManifestPath);
    if (!writeJsonFile(reportPath, report, &error)) {
        return failedResult(error);
    }
    return resultFromReport(reportPath, report);
}

WorkflowResult evaluateModelReport(const QString& modelPath, const QString& datasetPath, const QString& outputPath, const QString& taskType, const QJsonObject& options)
{
    const QFileInfo modelInfo(modelPath);
    if (!modelInfo.exists()) {
        return failedResult(QStringLiteral("Model file does not exist: %1").arg(modelPath));
    }
    const QDir datasetRoot(datasetPath);
    if (!datasetRoot.exists()) {
        return failedResult(QStringLiteral("Dataset directory does not exist: %1").arg(datasetPath));
    }

    if (taskType == QStringLiteral("detection") || taskType == QStringLiteral("yolo_detection")) {
        const QString split = options.value(QStringLiteral("split")).toString(selectDetectionSplit(datasetPath));
        if (split.isEmpty()) {
            return failedResult(QStringLiteral("No detection split found. Expected images/val, images/test, or images/train with matching labels."));
        }

        DetectionDataset dataset;
        QString error;
        if (!dataset.load(datasetPath, split, &error)) {
            return failedResult(error);
        }

        DetectionInferenceOptions inferenceOptions;
        inferenceOptions.iouThreshold = options.value(QStringLiteral("nmsIouThreshold")).toDouble(0.45);
        inferenceOptions.confidenceThreshold = options.value(QStringLiteral("confidenceThreshold")).toDouble(0.001);
        inferenceOptions.maxDetections = options.value(QStringLiteral("maxDetections")).toInt(100);
        const double matchIouThreshold = options.value(QStringLiteral("iouThreshold")).toDouble(0.5);
        const double lowConfidenceThreshold = options.value(QStringLiteral("lowConfidenceThreshold")).toDouble(0.25);
        const int maxErrorSamples = options.value(QStringLiteral("maxErrorSamples")).toInt(200);
        const int maxOverlaySamples = options.value(QStringLiteral("maxOverlaySamples")).toInt(50);

        QStringList classNames = dataset.info().classNames;
        const int classCount = qMax(dataset.info().classCount, classNames.size());
        while (classNames.size() < classCount) {
            classNames.append(QStringLiteral("class_%1").arg(classNames.size()));
        }
        QVector<DetectionClassStats> classStats(classCount);
        QVector<QVector<int>> confusion(classCount + 1, QVector<int>(classCount + 1, 0));
        QJsonArray sampleSummaries;
        QJsonArray errorSamples;
        QJsonArray lowConfidenceSamples;
        QString runtime = QStringLiteral("unknown");
        int totalGt = 0;
        int totalPredictions = 0;
        int totalTp = 0;
        int totalFp = 0;
        int totalFn = 0;
        int overlayCount = 0;
        const QDir outputDir(outputPath);
        QDir().mkpath(outputDir.filePath(QStringLiteral("overlays")));

        for (const DetectionSample& sample : dataset.samples()) {
            QString predictionError;
            const QVector<DetectionPrediction> predictions = runDetectionPredictions(modelPath, sample.imagePath, inferenceOptions, &runtime, &predictionError);
            if (!predictionError.isEmpty()) {
                if (predictionError.startsWith(QStringLiteral("hardware-blocked"))) {
                    QJsonObject blocked;
                    blocked.insert(QStringLiteral("ok"), false);
                    blocked.insert(QStringLiteral("status"), QStringLiteral("hardware-blocked"));
                    blocked.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
                    blocked.insert(QStringLiteral("createdAt"), nowIso());
                    blocked.insert(QStringLiteral("modelPath"), modelPath);
                    blocked.insert(QStringLiteral("datasetPath"), datasetPath);
                    blocked.insert(QStringLiteral("taskType"), taskType);
                    blocked.insert(QStringLiteral("runtime"), runtime);
                    blocked.insert(QStringLiteral("scaffold"), false);
                    blocked.insert(QStringLiteral("message"), predictionError);
                    const QString reportPath = outputDir.filePath(QStringLiteral("evaluation_report.json"));
                    if (!writeJsonFile(reportPath, blocked, &error)) {
                        return failedResult(error);
                    }
                    blocked.insert(QStringLiteral("reportPath"), reportPath);
                    return resultFromReport(reportPath, blocked);
                }
                return failedResult(predictionError);
            }

            totalGt += sample.boxes.size();
            totalPredictions += predictions.size();
            for (const DetectionBox& gt : sample.boxes) {
                if (gt.classId >= 0 && gt.classId < classStats.size()) {
                    classStats[gt.classId].gt += 1;
                }
            }

            QVector<bool> gtMatched(sample.boxes.size(), false);
            QVector<DetectionPrediction> sortedPredictions = predictions;
            std::sort(sortedPredictions.begin(), sortedPredictions.end(), [](const DetectionPrediction& left, const DetectionPrediction& right) {
                return left.confidence > right.confidence;
            });

            QJsonArray samplePredictions;
            bool sampleHasError = false;
            for (const DetectionPrediction& prediction : sortedPredictions) {
                int bestMatch = -1;
                double bestIou = 0.0;
                int bestAny = -1;
                double bestAnyIou = 0.0;
                for (int index = 0; index < sample.boxes.size(); ++index) {
                    if (gtMatched.at(index)) {
                        continue;
                    }
                    const double iou = boxIou(prediction.box, sample.boxes.at(index));
                    if (iou > bestAnyIou) {
                        bestAnyIou = iou;
                        bestAny = index;
                    }
                    if (prediction.box.classId == sample.boxes.at(index).classId && iou > bestIou) {
                        bestIou = iou;
                        bestMatch = index;
                    }
                }

                const bool matched = bestMatch >= 0 && bestIou >= matchIouThreshold;
                const int predClass = prediction.box.classId >= 0 && prediction.box.classId < classStats.size()
                    ? prediction.box.classId
                    : classStats.size() - 1;
                if (matched) {
                    gtMatched[bestMatch] = true;
                    classStats[predClass].tp += 1;
                    classStats[predClass].items.append(DetectionEvaluationItem{predClass, prediction.confidence, true});
                    confusion[predClass][predClass] += 1;
                    ++totalTp;
                } else {
                    if (predClass >= 0 && predClass < classStats.size()) {
                        classStats[predClass].fp += 1;
                        classStats[predClass].items.append(DetectionEvaluationItem{predClass, prediction.confidence, false});
                    }
                    if (bestAny >= 0 && bestAnyIou >= matchIouThreshold) {
                        const int gtClass = sample.boxes.at(bestAny).classId;
                        if (gtClass >= 0 && gtClass < classCount && predClass >= 0 && predClass < classCount) {
                            confusion[gtClass][predClass] += 1;
                        }
                    } else if (predClass >= 0 && predClass < classCount) {
                        confusion[classCount][predClass] += 1;
                    }
                    ++totalFp;
                    sampleHasError = true;
                    if (errorSamples.size() < maxErrorSamples) {
                        errorSamples.append(QJsonObject{
                            {QStringLiteral("reason"), QStringLiteral("false_positive")},
                            {QStringLiteral("imagePath"), sample.imagePath},
                            {QStringLiteral("labelPath"), sample.labelPath},
                            {QStringLiteral("matchedIou"), bestAnyIou},
                            {QStringLiteral("prediction"), detectionPredictionToJson(prediction)}
                        });
                    }
                }
                if (prediction.confidence < lowConfidenceThreshold && lowConfidenceSamples.size() < maxErrorSamples) {
                    lowConfidenceSamples.append(QJsonObject{
                        {QStringLiteral("imagePath"), sample.imagePath},
                        {QStringLiteral("labelPath"), sample.labelPath},
                        {QStringLiteral("prediction"), detectionPredictionToJson(prediction)}
                    });
                }
                samplePredictions.append(detectionPredictionToJson(prediction));
            }

            for (int index = 0; index < sample.boxes.size(); ++index) {
                if (!gtMatched.at(index)) {
                    const DetectionBox& gt = sample.boxes.at(index);
                    if (gt.classId >= 0 && gt.classId < classStats.size()) {
                        classStats[gt.classId].fn += 1;
                        confusion[gt.classId][classCount] += 1;
                    }
                    ++totalFn;
                    sampleHasError = true;
                    if (errorSamples.size() < maxErrorSamples) {
                        errorSamples.append(QJsonObject{
                            {QStringLiteral("reason"), QStringLiteral("false_negative")},
                            {QStringLiteral("imagePath"), sample.imagePath},
                            {QStringLiteral("labelPath"), sample.labelPath},
                            {QStringLiteral("groundTruth"), detectionBoxToJson(gt)}
                        });
                    }
                }
            }

            QString overlayPath;
            if (sampleHasError && overlayCount < maxOverlaySamples) {
                QString overlayError;
                QImage overlay = renderDetectionPredictions(sample.imagePath, predictions, &overlayError);
                if (!overlay.isNull()) {
                    overlayPath = outputDir.filePath(QStringLiteral("overlays/%1_%2.png")
                        .arg(overlayCount, 4, 10, QLatin1Char('0'))
                        .arg(QFileInfo(sample.imagePath).completeBaseName()));
                    if (overlay.save(overlayPath)) {
                        ++overlayCount;
                    } else {
                        overlayPath.clear();
                    }
                }
            }

            QJsonObject sampleSummary;
            sampleSummary.insert(QStringLiteral("imagePath"), sample.imagePath);
            sampleSummary.insert(QStringLiteral("labelPath"), sample.labelPath);
            sampleSummary.insert(QStringLiteral("groundTruthCount"), sample.boxes.size());
            sampleSummary.insert(QStringLiteral("predictionCount"), predictions.size());
            sampleSummary.insert(QStringLiteral("hasError"), sampleHasError);
            if (!overlayPath.isEmpty()) {
                sampleSummary.insert(QStringLiteral("overlayPath"), overlayPath);
            }
            sampleSummary.insert(QStringLiteral("predictions"), samplePredictions);
            sampleSummaries.append(sampleSummary);
        }

        QJsonArray perClassArray;
        double map50 = 0.0;
        int apClassCount = 0;
        for (int classId = 0; classId < classStats.size(); ++classId) {
            DetectionClassStats& stats = classStats[classId];
            stats.precision = stats.tp + stats.fp > 0 ? static_cast<double>(stats.tp) / static_cast<double>(stats.tp + stats.fp) : 0.0;
            stats.recall = stats.gt > 0 ? static_cast<double>(stats.tp) / static_cast<double>(stats.gt) : 0.0;
            stats.ap50 = ap50FromItems(stats.items, stats.gt);
            if (stats.gt > 0) {
                map50 += stats.ap50;
                ++apClassCount;
            }
            const QString className = classId < classNames.size() && !classNames.at(classId).isEmpty()
                ? classNames.at(classId)
                : QStringLiteral("class_%1").arg(classId);
            perClassArray.append(QJsonObject{
                {QStringLiteral("classId"), classId},
                {QStringLiteral("className"), className},
                {QStringLiteral("gt"), stats.gt},
                {QStringLiteral("tp"), stats.tp},
                {QStringLiteral("fp"), stats.fp},
                {QStringLiteral("fn"), stats.fn},
                {QStringLiteral("precision"), stats.precision},
                {QStringLiteral("recall"), stats.recall},
                {QStringLiteral("ap50"), stats.ap50}
            });
        }
        map50 = apClassCount > 0 ? map50 / static_cast<double>(apClassCount) : 0.0;
        const double precision = totalTp + totalFp > 0 ? static_cast<double>(totalTp) / static_cast<double>(totalTp + totalFp) : 0.0;
        const double recall = totalGt > 0 ? static_cast<double>(totalTp) / static_cast<double>(totalGt) : 0.0;

        QJsonObject metrics;
        metrics.insert(QStringLiteral("precision"), precision);
        metrics.insert(QStringLiteral("recall"), recall);
        metrics.insert(QStringLiteral("mAP50"), map50);
        metrics.insert(QStringLiteral("tp"), totalTp);
        metrics.insert(QStringLiteral("fp"), totalFp);
        metrics.insert(QStringLiteral("fn"), totalFn);
        metrics.insert(QStringLiteral("gt"), totalGt);
        metrics.insert(QStringLiteral("predictions"), totalPredictions);

        QJsonObject report;
        report.insert(QStringLiteral("ok"), true);
        report.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
        report.insert(QStringLiteral("createdAt"), nowIso());
        report.insert(QStringLiteral("modelPath"), modelPath);
        report.insert(QStringLiteral("datasetPath"), datasetPath);
        report.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
        report.insert(QStringLiteral("split"), split);
        report.insert(QStringLiteral("runtime"), runtime);
        report.insert(QStringLiteral("datasetSnapshotId"), options.value(QStringLiteral("datasetSnapshotId")).toInt());
        report.insert(QStringLiteral("datasetSnapshotHash"), options.value(QStringLiteral("datasetSnapshotHash")).toString());
        report.insert(QStringLiteral("datasetSnapshotManifest"), options.value(QStringLiteral("datasetSnapshotManifest")).toString());
        report.insert(QStringLiteral("scaffold"), false);
        report.insert(QStringLiteral("metrics"), metrics);
        report.insert(QStringLiteral("perClass"), perClassArray);
        report.insert(QStringLiteral("samples"), sampleSummaries);
        report.insert(QStringLiteral("errorSamples"), errorSamples);
        report.insert(QStringLiteral("lowConfidenceSamples"), lowConfidenceSamples);
        report.insert(QStringLiteral("sampleCount"), dataset.size());
        report.insert(QStringLiteral("parameters"), QJsonObject{
            {QStringLiteral("iouThreshold"), matchIouThreshold},
            {QStringLiteral("confidenceThreshold"), inferenceOptions.confidenceThreshold},
            {QStringLiteral("nmsIouThreshold"), inferenceOptions.iouThreshold},
            {QStringLiteral("maxDetections"), inferenceOptions.maxDetections},
            {QStringLiteral("lowConfidenceThreshold"), lowConfidenceThreshold}
        });
        report.insert(QStringLiteral("limitations"), QStringLiteral("Phase 36 computes detection AP50 only. COCO mAP50-95, segmentation mask evaluation, and OCR CER/WER remain follow-up work."));

        const QString reportPath = outputDir.filePath(QStringLiteral("evaluation_report.json"));
        const QString perClassPath = outputDir.filePath(QStringLiteral("per_class_metrics.csv"));
        const QString errorPath = outputDir.filePath(QStringLiteral("error_samples.json"));
        const QString confusionPath = outputDir.filePath(QStringLiteral("confusion_matrix.csv"));
        if (!writeTextFile(perClassPath, perClassMetricsCsv(classNames, classStats), &error)) {
            return failedResult(error);
        }
        if (!writeJsonFile(errorPath, QJsonObject{{QStringLiteral("samples"), errorSamples}, {QStringLiteral("lowConfidenceSamples"), lowConfidenceSamples}}, &error)) {
            return failedResult(error);
        }
        if (!writeTextFile(confusionPath, confusionMatrixCsv(classNames, confusion), &error)) {
            return failedResult(error);
        }
        report.insert(QStringLiteral("reportPath"), reportPath);
        report.insert(QStringLiteral("perClassMetricsPath"), perClassPath);
        report.insert(QStringLiteral("errorSamplesPath"), errorPath);
        report.insert(QStringLiteral("confusionMatrixPath"), confusionPath);
        report.insert(QStringLiteral("overlayDir"), outputDir.filePath(QStringLiteral("overlays")));
        if (!writeJsonFile(reportPath, report, &error)) {
            return failedResult(error);
        }
        return resultFromReport(reportPath, report);
    }

    QJsonObject summary;
    summary.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
    summary.insert(QStringLiteral("createdAt"), nowIso());
    summary.insert(QStringLiteral("modelPath"), modelPath);
    summary.insert(QStringLiteral("datasetPath"), datasetPath);
    summary.insert(QStringLiteral("taskType"), taskType);
    summary.insert(QStringLiteral("datasetSnapshotId"), options.value(QStringLiteral("datasetSnapshotId")).toInt());
    summary.insert(QStringLiteral("datasetSnapshotHash"), options.value(QStringLiteral("datasetSnapshotHash")).toString());
    summary.insert(QStringLiteral("datasetSnapshotManifest"), options.value(QStringLiteral("datasetSnapshotManifest")).toString());
    summary.insert(QStringLiteral("scaffold"), true);
    summary.insert(QStringLiteral("note"), QStringLiteral("Phase 36 implements real evaluation for detection only. Segmentation mask metrics and OCR CER/WER remain scaffold reports."));

    QJsonObject metrics;
    if (taskType == QStringLiteral("ocr_recognition") || taskType == QStringLiteral("ocr")) {
        metrics.insert(QStringLiteral("accuracy"), 0.0);
        metrics.insert(QStringLiteral("editDistance"), 0.0);
        metrics.insert(QStringLiteral("cer"), 0.0);
        metrics.insert(QStringLiteral("wer"), 0.0);
    } else if (taskType == QStringLiteral("segmentation")) {
        metrics.insert(QStringLiteral("maskIoU"), 0.0);
        metrics.insert(QStringLiteral("maskMap50"), 0.0);
        metrics.insert(QStringLiteral("precision"), 0.0);
        metrics.insert(QStringLiteral("recall"), 0.0);
    } else {
        metrics.insert(QStringLiteral("precision"), 0.0);
        metrics.insert(QStringLiteral("recall"), 0.0);
        metrics.insert(QStringLiteral("mAP50"), 0.0);
        metrics.insert(QStringLiteral("mAP50_95"), 0.0);
    }
    summary.insert(QStringLiteral("metrics"), metrics);
    summary.insert(QStringLiteral("errorSamples"), QJsonArray());
    summary.insert(QStringLiteral("lowConfidenceSamples"), QJsonArray());

    QString error;
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("evaluation_report.json"));
    if (!writeJsonFile(reportPath, summary, &error)) {
        return failedResult(error);
    }
    summary.insert(QStringLiteral("reportPath"), reportPath);
    return resultFromReport(reportPath, summary);
}

WorkflowResult benchmarkModelReport(const QString& modelPath, const QString& outputPath, const QJsonObject& options)
{
    const QFileInfo modelInfo(modelPath);
    if (!modelInfo.exists()) {
        return failedResult(QStringLiteral("Model file does not exist: %1").arg(modelPath));
    }

    const QString suffix = modelInfo.suffix().toLower();
    const QString runtime = options.value(QStringLiteral("runtime")).toString(
        suffix == QStringLiteral("onnx") ? QStringLiteral("onnxruntime") :
        (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan") ? QStringLiteral("tensorrt") : QStringLiteral("file")));
    if (runtime == QStringLiteral("tensorrt") && !isTensorRtInferenceAvailable()) {
        QJsonObject blocked;
        blocked.insert(QStringLiteral("ok"), false);
        blocked.insert(QStringLiteral("status"), QStringLiteral("hardware-blocked"));
        blocked.insert(QStringLiteral("modelPath"), modelPath);
        blocked.insert(QStringLiteral("runtime"), runtime);
        blocked.insert(QStringLiteral("message"), QStringLiteral("TensorRT benchmark requires a compatible RTX / SM 75+ acceptance machine and runtime."));
        QString error;
        const QString reportPath = QDir(outputPath).filePath(QStringLiteral("benchmark_report.json"));
        if (!writeJsonFile(reportPath, blocked, &error)) {
            return failedResult(error);
        }
        blocked.insert(QStringLiteral("reportPath"), reportPath);
        return resultFromReport(reportPath, blocked);
    }

    QElapsedTimer timer;
    timer.start();
    QFile file(modelPath);
    if (!file.open(QIODevice::ReadOnly)) {
        return failedResult(QStringLiteral("Cannot read model for benchmark: %1").arg(modelPath));
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(file.read(1024 * 1024));
    const qint64 elapsed = qMax<qint64>(1, timer.elapsed());

    QJsonObject report;
    report.insert(QStringLiteral("ok"), true);
    report.insert(QStringLiteral("kind"), QStringLiteral("benchmark_report"));
    report.insert(QStringLiteral("createdAt"), nowIso());
    report.insert(QStringLiteral("modelPath"), modelPath);
    report.insert(QStringLiteral("runtime"), runtime);
    report.insert(QStringLiteral("device"), options.value(QStringLiteral("device")).toString(QStringLiteral("cpu")));
    report.insert(QStringLiteral("batch"), options.value(QStringLiteral("batch")).toInt(1));
    report.insert(QStringLiteral("inputShape"), options.value(QStringLiteral("inputShape")).toString(QStringLiteral("auto")));
    report.insert(QStringLiteral("averageMs"), static_cast<double>(elapsed));
    report.insert(QStringLiteral("p50Ms"), static_cast<double>(elapsed));
    report.insert(QStringLiteral("p95Ms"), static_cast<double>(elapsed));
    report.insert(QStringLiteral("p99Ms"), static_cast<double>(elapsed));
    report.insert(QStringLiteral("throughput"), 1000.0 / static_cast<double>(elapsed));
    report.insert(QStringLiteral("modelBytes"), QString::number(modelInfo.size()));
    report.insert(QStringLiteral("sampleHash"), QString::fromLatin1(hash.result().toHex()));
    report.insert(QStringLiteral("scaffold"), true);
    report.insert(QStringLiteral("note"), QStringLiteral("Benchmark v1 measures local model file access and records runtime intent. Full timed inference benchmark is a follow-up phase."));

    QString error;
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("benchmark_report.json"));
    if (!writeJsonFile(reportPath, report, &error)) {
        return failedResult(error);
    }
    report.insert(QStringLiteral("reportPath"), reportPath);
    return resultFromReport(reportPath, report);
}

WorkflowResult generateTrainingDeliveryReport(const QString& outputPath, const QJsonObject& context)
{
    QJsonObject reportContext = context;
    reportContext.insert(QStringLiteral("generatedAt"), nowIso());
    reportContext.insert(QStringLiteral("kind"), QStringLiteral("training_delivery_report"));
    reportContext.insert(QStringLiteral("scaffold"), true);
    reportContext.insert(QStringLiteral("note"), QStringLiteral("HTML report v1 packages available context and explicit limitations for offline review."));

    QString error;
    const QString htmlPath = QDir(outputPath).filePath(QStringLiteral("training_delivery_report.html"));
    if (!writeTextFile(htmlPath, htmlReport(reportContext), &error)) {
        return failedResult(error);
    }
    const QString jsonPath = QDir(outputPath).filePath(QStringLiteral("training_delivery_report.json"));
    if (!writeJsonFile(jsonPath, reportContext, &error)) {
        return failedResult(error);
    }
    reportContext.insert(QStringLiteral("reportPath"), htmlPath);
    reportContext.insert(QStringLiteral("jsonPath"), jsonPath);
    return resultFromReport(htmlPath, reportContext);
}

WorkflowResult runLocalPipelinePlan(const QString& outputPath, const QString& templateId, const QJsonObject& options)
{
    QJsonObject plan;
    plan.insert(QStringLiteral("kind"), QStringLiteral("local_pipeline_plan"));
    plan.insert(QStringLiteral("createdAt"), nowIso());
    plan.insert(QStringLiteral("templateId"), templateId.isEmpty() ? QStringLiteral("train-evaluate-export-register") : templateId);
    plan.insert(QStringLiteral("state"), QStringLiteral("planned"));
    plan.insert(QStringLiteral("scaffold"), true);
    plan.insert(QStringLiteral("note"), QStringLiteral("Pipeline v1 records a deterministic execution plan. Automated multi-step task orchestration is a follow-up phase."));
    plan.insert(QStringLiteral("options"), options);
    QJsonArray steps;
    for (const QString& step : {
             QStringLiteral("validateDataset"),
             QStringLiteral("createDatasetSnapshot"),
             QStringLiteral("startTrain"),
             QStringLiteral("evaluateModel"),
             QStringLiteral("exportModel"),
             QStringLiteral("registerModel"),
             QStringLiteral("generateTrainingDeliveryReport")}) {
        QJsonObject object;
        object.insert(QStringLiteral("command"), step);
        object.insert(QStringLiteral("state"), QStringLiteral("planned"));
        steps.append(object);
    }
    plan.insert(QStringLiteral("steps"), steps);

    QString error;
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("local_pipeline_plan.json"));
    if (!writeJsonFile(reportPath, plan, &error)) {
        return failedResult(error);
    }
    plan.insert(QStringLiteral("reportPath"), reportPath);
    return resultFromReport(reportPath, plan);
}

} // namespace aitrain
