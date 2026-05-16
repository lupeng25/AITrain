#include "ProductWorkflowSupport.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDirIterator>
#include <QFile>
#include <QJsonDocument>
#include <QTextStream>

#include <algorithm>

namespace aitrain::workflow_detail {
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

bool readJsonFile(const QString& path, QJsonObject* object, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("Cannot read JSON file: %1").arg(path);
        }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = QStringLiteral("Cannot parse JSON file %1: %2").arg(path, parseError.errorString());
        }
        return false;
    }
    if (object) {
        *object = document.object();
    }
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

QString firstImageFileUnder(const QString& rootPath)
{
    if (rootPath.isEmpty()) {
        return QString();
    }
    const QFileInfo rootInfo(rootPath);
    if (rootInfo.isFile() && isImageFile(rootInfo.suffix())) {
        return rootInfo.absoluteFilePath();
    }
    if (!rootInfo.exists() || !rootInfo.isDir()) {
        return QString();
    }

    QFileInfoList candidates;
    QDirIterator iterator(rootInfo.absoluteFilePath(), imageNameFilters(), QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        iterator.next();
        candidates.append(iterator.fileInfo());
    }
    std::sort(candidates.begin(), candidates.end(), [](const QFileInfo& left, const QFileInfo& right) {
        return left.absoluteFilePath().compare(right.absoluteFilePath(), Qt::CaseInsensitive) < 0;
    });
    return candidates.isEmpty() ? QString() : candidates.first().absoluteFilePath();
}

QJsonObject fileDigestObject(const QString& path, QString* error)
{
    QJsonObject digest;
    if (path.isEmpty()) {
        return digest;
    }
    const QFileInfo info(path);
    digest.insert(QStringLiteral("path"), path);
    digest.insert(QStringLiteral("exists"), info.exists());
    if (!info.exists() || info.isDir()) {
        return digest;
    }

    qint64 size = 0;
    QString hashError;
    const QByteArray hash = fileSha256(path, &size, &hashError);
    if (!hashError.isEmpty()) {
        if (error) {
            *error = hashError;
        }
        return digest;
    }
    digest.insert(QStringLiteral("bytes"), QString::number(size));
    digest.insert(QStringLiteral("sha256"), QString::fromLatin1(hash));
    digest.insert(QStringLiteral("updatedAt"), info.lastModified().toUTC().toString(Qt::ISODateWithMs));
    return digest;
}

QJsonObject pathArtifact(const QString& kind, const QString& path, const QString& message)
{
    QJsonObject artifact;
    artifact.insert(QStringLiteral("kind"), kind);
    artifact.insert(QStringLiteral("path"), path);
    if (!message.isEmpty()) {
        artifact.insert(QStringLiteral("message"), message);
    }
    return artifact;
}

QJsonObject readJsonObjectIfExists(const QString& path)
{
    QJsonObject object;
    QString error;
    if (path.isEmpty() || !QFileInfo::exists(path) || !readJsonFile(path, &object, &error)) {
        return {};
    }
    return object;
}

QJsonObject evaluationDeliverySummary(const QString& reportPath)
{
    const QJsonObject report = readJsonObjectIfExists(reportPath);
    if (report.isEmpty()) {
        return {};
    }
    QJsonObject summary;
    summary.insert(QStringLiteral("reportPath"), reportPath);
    summary.insert(QStringLiteral("taskType"), report.value(QStringLiteral("taskType")).toString());
    summary.insert(QStringLiteral("status"), report.value(QStringLiteral("status")).toString(QStringLiteral("completed")));
    summary.insert(QStringLiteral("scaffold"), report.value(QStringLiteral("scaffold")).toBool());
    summary.insert(QStringLiteral("metrics"), report.value(QStringLiteral("metrics")).toObject());
    summary.insert(QStringLiteral("perClassMetricsPath"), report.value(QStringLiteral("perClassMetricsPath")).toString());
    summary.insert(QStringLiteral("errorSamplesPath"), report.value(QStringLiteral("errorSamplesPath")).toString());
    summary.insert(QStringLiteral("confusionMatrixPath"), report.value(QStringLiteral("confusionMatrixPath")).toString());
    summary.insert(QStringLiteral("limitations"), report.value(QStringLiteral("limitations")).toArray());
    return summary;
}

QJsonObject benchmarkDeliverySummary(const QString& reportPath)
{
    const QJsonObject report = readJsonObjectIfExists(reportPath);
    if (report.isEmpty()) {
        return {};
    }
    QJsonObject summary;
    summary.insert(QStringLiteral("reportPath"), reportPath);
    summary.insert(QStringLiteral("runtime"), report.value(QStringLiteral("runtime")).toString());
    summary.insert(QStringLiteral("modelFamily"), report.value(QStringLiteral("modelFamily")).toString());
    summary.insert(QStringLiteral("runtimeStatus"), report.value(QStringLiteral("runtimeStatus")).toString(report.value(QStringLiteral("status")).toString()));
    summary.insert(QStringLiteral("deploymentConclusion"), report.value(QStringLiteral("deploymentConclusion")).toString());
    summary.insert(QStringLiteral("timedInference"), report.value(QStringLiteral("timedInference")).toBool());
    summary.insert(QStringLiteral("latency"), report.value(QStringLiteral("latency")).toObject());
    summary.insert(QStringLiteral("averageMs"), report.value(QStringLiteral("averageMs")).toDouble());
    summary.insert(QStringLiteral("p95Ms"), report.value(QStringLiteral("p95Ms")).toDouble());
    summary.insert(QStringLiteral("throughput"), report.value(QStringLiteral("throughput")).toDouble());
    summary.insert(QStringLiteral("failureCategory"), report.value(QStringLiteral("failureCategory")).toString());
    return summary;
}

QJsonArray deliveryLimitations(const QJsonObject& context, const QJsonObject& evaluationSummary, const QJsonObject& benchmarkSummary)
{
    QJsonArray limitations;
    const QString backend = context.value(QStringLiteral("trainingBackend")).toString();
    if (backend == QStringLiteral("tiny_linear_detector")
        || backend == QStringLiteral("python_mock")
        || context.value(QStringLiteral("scaffold")).toBool()) {
        limitations.append(QStringLiteral("Scaffold or diagnostic backends are not production YOLO/OCR training capabilities."));
    }
    if (evaluationSummary.value(QStringLiteral("scaffold")).toBool()) {
        limitations.append(QStringLiteral("Evaluation summary includes scaffold or limited metrics; inspect the evaluation report before delivery."));
    }
    const QString benchmarkConclusion = benchmarkSummary.value(QStringLiteral("deploymentConclusion")).toString();
    const QString benchmarkFailure = benchmarkSummary.value(QStringLiteral("failureCategory")).toString();
    if (benchmarkConclusion == QStringLiteral("hardware-blocked") || benchmarkFailure == QStringLiteral("hardware-blocked")) {
        limitations.append(QStringLiteral("TensorRT remains hardware-blocked on this machine; RTX / SM 75+ acceptance is required."));
    }
    const QJsonObject deploymentSummary = readJsonObjectIfExists(context.value(QStringLiteral("deploymentValidationReportPath")).toString());
    const QString deploymentStatus = deploymentSummary.value(QStringLiteral("status")).toString();
    if (deploymentStatus == QStringLiteral("hardware-blocked")) {
        limitations.append(QStringLiteral("Deployment validation is hardware-blocked; this artifact still needs compatible hardware acceptance."));
    } else if (deploymentStatus == QStringLiteral("failed") || deploymentStatus == QStringLiteral("blocked")) {
        limitations.append(QStringLiteral("Deployment validation is not passed; inspect deployment_validation_report.json before handoff."));
    } else if (deploymentStatus.isEmpty()) {
        limitations.append(QStringLiteral("Deployment validation report is not attached; exported runtime acceptance remains unproven."));
    }
    const QJsonObject customerOcrSummary = readJsonObjectIfExists(context.value(QStringLiteral("customerOcrAcceptanceReportPath")).toString());
    const QString customerOcrStatus = customerOcrSummary.value(QStringLiteral("status")).toString();
    if (context.value(QStringLiteral("taskType")).toString().contains(QStringLiteral("ocr"), Qt::CaseInsensitive)) {
        if (customerOcrStatus == QStringLiteral("blocked")) {
            limitations.append(QStringLiteral("Customer-domain OCR acceptance is blocked; public smoke data cannot certify production OCR accuracy."));
        } else if (customerOcrStatus.isEmpty()) {
            limitations.append(QStringLiteral("Customer-domain OCR acceptance report is not attached; production OCR claims require customer evidence."));
        }
    }
    if (context.value(QStringLiteral("taskType")).toString().contains(QStringLiteral("ocr"), Qt::CaseInsensitive)
        && backend.contains(QStringLiteral("official"), Qt::CaseInsensitive)) {
        limitations.append(QStringLiteral("PaddleOCR official system path is tool-chain orchestration, not C++ DB ONNX postprocess."));
    }
    if (limitations.isEmpty()) {
        limitations.append(QStringLiteral("No additional delivery limitation was inferred beyond source artifact notes."));
    }
    return limitations;
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

QString csvEscape(QString value)
{
    if (value.contains(QLatin1Char('"')) || value.contains(QLatin1Char(',')) || value.contains(QLatin1Char('\n'))) {
        value.replace(QLatin1Char('"'), QStringLiteral("\"\""));
        return QStringLiteral("\"%1\"").arg(value);
    }
    return value;
}

QString htmlEscape(QString value)
{
    return value.replace(QLatin1Char('&'), QStringLiteral("&amp;"))
        .replace(QLatin1Char('<'), QStringLiteral("&lt;"))
        .replace(QLatin1Char('>'), QStringLiteral("&gt;"))
        .replace(QLatin1Char('"'), QStringLiteral("&quot;"));
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

WorkflowResult canceledResult()
{
    return failedResult(QStringLiteral("Canceled by user"));
}

} // namespace aitrain::workflow_detail
