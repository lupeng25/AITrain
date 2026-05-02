#include "aitrain/core/ProductWorkflow.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/DetectionTrainer.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QDirIterator>
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
    DatasetValidationResult validation = validateByFormat(datasetPath, format, options);
    QJsonObject classCounts;
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt") || format == QStringLiteral("yolo_segmentation")) {
        classCounts = countYoloClasses(datasetPath);
    }

    QJsonObject report = validation.toJson();
    report.insert(QStringLiteral("kind"), QStringLiteral("dataset_quality_report"));
    report.insert(QStringLiteral("createdAt"), nowIso());
    report.insert(QStringLiteral("datasetPath"), datasetPath);
    report.insert(QStringLiteral("format"), format);
    report.insert(QStringLiteral("issueCounts"), validationIssueCounts(validation));
    report.insert(QStringLiteral("splitCounts"), countImageSplits(datasetPath));
    report.insert(QStringLiteral("classDistribution"), classCounts);
    report.insert(QStringLiteral("scaffold"), false);

    QString error;
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("dataset_quality_report.json"));
    if (!writeJsonFile(reportPath, report, &error)) {
        return failedResult(error);
    }
    const QString csvPath = QDir(outputPath).filePath(QStringLiteral("class_distribution.csv"));
    if (!writeTextFile(csvPath, classDistributionCsv(classCounts), &error)) {
        return failedResult(error);
    }
    const QString problemPath = QDir(outputPath).filePath(QStringLiteral("problem_samples.json"));
    QJsonObject problems;
    problems.insert(QStringLiteral("datasetPath"), datasetPath);
    problems.insert(QStringLiteral("format"), format);
    problems.insert(QStringLiteral("issues"), issueArray(validation, options.value(QStringLiteral("maxProblemSamples")).toInt(500)));
    if (!writeJsonFile(problemPath, problems, &error)) {
        return failedResult(error);
    }
    report.insert(QStringLiteral("reportPath"), reportPath);
    report.insert(QStringLiteral("classDistributionPath"), csvPath);
    report.insert(QStringLiteral("problemSamplesPath"), problemPath);
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
