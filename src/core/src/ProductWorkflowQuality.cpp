#include "aitrain/core/ProductWorkflow.h"

#include "ProductWorkflowSupport.h"
#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/OcrRecDataset.h"
#include "aitrain/core/SegmentationDataset.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMap>
#include <QRegularExpression>
#include <QSet>
#include <QTextStream>
#include <QThread>

#include <algorithm>
namespace aitrain {
using namespace workflow_detail;
namespace {
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
} // namespace
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
    const QString reworkSampleSetPath = QDir(outputPath).filePath(QStringLiteral("rework_sample_set.json"));
    const QString prelabelCandidatesPath = QDir(outputPath).filePath(QStringLiteral("prelabel_candidates.json"));

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

    QJsonArray reworkSamples = context.problemSamples;
    const QString evaluationReportPath = options.value(QStringLiteral("evaluationReportPath")).toString();
    if (!evaluationReportPath.isEmpty() && QFileInfo::exists(evaluationReportPath)) {
        QJsonObject evaluationReport;
        QString readError;
        if (readJsonFile(evaluationReportPath, &evaluationReport, &readError)) {
            const QJsonArray evaluationErrors = evaluationReport.value(QStringLiteral("errorSamples")).toArray();
            for (const QJsonValue& value : evaluationErrors) {
                QJsonObject sample = value.toObject();
                sample.insert(QStringLiteral("source"), QStringLiteral("evaluation_error"));
                sample.insert(QStringLiteral("evaluationReportPath"), evaluationReportPath);
                reworkSamples.append(sample);
            }
        }
    }
    QJsonObject reworkSampleSet;
    reworkSampleSet.insert(QStringLiteral("schemaVersion"), 1);
    reworkSampleSet.insert(QStringLiteral("kind"), QStringLiteral("dataset_rework_sample_set"));
    reworkSampleSet.insert(QStringLiteral("createdAt"), nowIso());
    reworkSampleSet.insert(QStringLiteral("datasetPath"), datasetPath);
    reworkSampleSet.insert(QStringLiteral("format"), format);
    reworkSampleSet.insert(QStringLiteral("sourceQualityReport"), reportPath);
    reworkSampleSet.insert(QStringLiteral("evaluationReportPath"), evaluationReportPath);
    reworkSampleSet.insert(QStringLiteral("sampleCount"), reworkSamples.size());
    reworkSampleSet.insert(QStringLiteral("samples"), reworkSamples);

    QJsonArray prelabelCandidates;
    for (const QJsonValue& value : reworkSamples) {
        const QJsonObject sample = value.toObject();
        const QString imagePath = sample.value(QStringLiteral("imagePath")).toString();
        if (imagePath.isEmpty()) {
            continue;
        }
        prelabelCandidates.append(QJsonObject{
            {QStringLiteral("imagePath"), imagePath},
            {QStringLiteral("labelPath"), sample.value(QStringLiteral("labelPath")).toString()},
            {QStringLiteral("source"), sample.value(QStringLiteral("source")).toString(QStringLiteral("dataset_quality"))},
            {QStringLiteral("mode"), QStringLiteral("candidate_only")},
            {QStringLiteral("note"), QStringLiteral("AITrain does not overwrite user labels. Use this as a prelabel/review candidate list.")}
        });
    }
    QJsonObject prelabelManifest;
    prelabelManifest.insert(QStringLiteral("schemaVersion"), 1);
    prelabelManifest.insert(QStringLiteral("kind"), QStringLiteral("prelabel_candidates"));
    prelabelManifest.insert(QStringLiteral("datasetPath"), datasetPath);
    prelabelManifest.insert(QStringLiteral("format"), format);
    prelabelManifest.insert(QStringLiteral("candidateCount"), prelabelCandidates.size());
    prelabelManifest.insert(QStringLiteral("candidates"), prelabelCandidates);

    if (!writeTextFile(csvPath, classDistributionCsvV2(context), &error)
        || !writeTextFile(imageStatsPath, imageStatisticsCsv(context), &error)
        || !writeTextFile(splitDistributionPath, splitDistributionCsv(context), &error)
        || !writeTextFile(xAnyFixListPath, context.xAnyFixRows.join(QLatin1Char('\n')) + QLatin1Char('\n'), &error)
        || !writeJsonFile(xAnyFixManifestPath, xAnyManifest, &error)
        || !writeJsonFile(reworkSampleSetPath, reworkSampleSet, &error)
        || !writeJsonFile(prelabelCandidatesPath, prelabelManifest, &error)) {
        return failedResult(error);
    }

    report.insert(QStringLiteral("reportPath"), reportPath);
    report.insert(QStringLiteral("classDistributionPath"), csvPath);
    report.insert(QStringLiteral("problemSamplesPath"), problemPath);
    report.insert(QStringLiteral("imageStatisticsPath"), imageStatsPath);
    report.insert(QStringLiteral("splitDistributionPath"), splitDistributionPath);
    report.insert(QStringLiteral("xAnyLabelingFixListPath"), xAnyFixListPath);
    report.insert(QStringLiteral("xAnyLabelingFixManifestPath"), xAnyFixManifestPath);
    report.insert(QStringLiteral("reworkSampleSetPath"), reworkSampleSetPath);
    report.insert(QStringLiteral("prelabelCandidatesPath"), prelabelCandidatesPath);
    if (!writeJsonFile(reportPath, report, &error)) {
        return failedResult(error);
    }
    return resultFromReport(reportPath, report);
}
} // namespace aitrain