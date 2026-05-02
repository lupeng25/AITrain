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
        if (maxFiles > 0 && files.size() >= maxFiles) {
            break;
        }
    }
    std::sort(files.begin(), files.end(), [](const QFileInfo& left, const QFileInfo& right) {
        return left.absoluteFilePath().compare(right.absoluteFilePath(), Qt::CaseInsensitive) < 0;
    });
    return files;
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
        QJsonObject fileObject;
        fileObject.insert(QStringLiteral("path"), relativePath);
        fileObject.insert(QStringLiteral("size"), QString::number(fileSize));
        fileObject.insert(QStringLiteral("mtime"), fileInfo.lastModified().toUTC().toString(Qt::ISODateWithMs));
        fileObject.insert(QStringLiteral("sha256"), QString::fromLatin1(hash));
        fileArray.append(fileObject);
        manifestHash.addData(relativePath.toUtf8());
        manifestHash.addData(hash);
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
    Q_UNUSED(options)
    const QFileInfo modelInfo(modelPath);
    if (!modelInfo.exists()) {
        return failedResult(QStringLiteral("Model file does not exist: %1").arg(modelPath));
    }

    QJsonObject summary;
    summary.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
    summary.insert(QStringLiteral("createdAt"), nowIso());
    summary.insert(QStringLiteral("modelPath"), modelPath);
    summary.insert(QStringLiteral("datasetPath"), datasetPath);
    summary.insert(QStringLiteral("taskType"), taskType);
    summary.insert(QStringLiteral("scaffold"), true);
    summary.insert(QStringLiteral("note"), QStringLiteral("Evaluation report v1 records model/data lineage and placeholder quality sections. Full per-sample mAP/CER analysis is a follow-up phase."));

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
