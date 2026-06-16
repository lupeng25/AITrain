#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/core/WorkerRequests.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QStandardPaths>
#include <QThread>

using namespace worker_support;
namespace wp = aitrain::worker_protocol;
namespace wr = aitrain::worker_requests;

namespace {
constexpr int kMaxExporterBufferBytes = 4 * 1024 * 1024;
constexpr int kMaxExporterLineBytes = 256 * 1024;

QByteArray boundedExporterLine(const QByteArray& line)
{
    if (line.size() <= kMaxExporterLineBytes) {
        return line;
    }
    QByteArray bounded = line.left(kMaxExporterLineBytes);
    bounded.append(" ... [log_truncated: single Python exporter line exceeded limit]");
    return bounded;
}

bool jsonBool(const QJsonObject& object, const QString& key)
{
    const QJsonValue value = object.value(key);
    if (value.isBool()) {
        return value.toBool();
    }
    if (value.isString()) {
        const QString text = value.toString().trimmed().toLower();
        return text == QStringLiteral("true") || text == QStringLiteral("1") || text == QStringLiteral("yes");
    }
    return value.toInt(0) != 0;
}

bool ncnnOfficialExportOptionsUnsupported(const QJsonObject& options)
{
    const QJsonObject args = options.value(QStringLiteral("ultralyticsExportArgs")).toObject();
    return jsonBool(args, QStringLiteral("dynamic"))
        || jsonBool(args, QStringLiteral("half"))
        || jsonBool(args, QStringLiteral("int8"))
        || jsonBool(args, QStringLiteral("end2end"));
}

QString unsupportedOfficialExportOptionsError(const QString& format, const QString& checkpointSuffix, const QJsonObject& options)
{
    const QJsonObject args = options.value(QStringLiteral("ultralyticsExportArgs")).toObject();
    if (args.isEmpty()) {
        return {};
    }

    const QString normalizedFormat = format.trimmed().toLower();
    const bool dynamic = jsonBool(args, QStringLiteral("dynamic"));
    const bool half = jsonBool(args, QStringLiteral("half"));
    const bool int8 = jsonBool(args, QStringLiteral("int8"));
    const bool end2end = jsonBool(args, QStringLiteral("end2end"));
    if (normalizedFormat == QStringLiteral("onnx") && int8) {
        return QStringLiteral("ONNX export does not support int8 in AITrain; use TensorRT export for INT8.");
    }
    if (normalizedFormat == QStringLiteral("ncnn") && (dynamic || half || int8 || end2end)) {
        return QStringLiteral("NCNN export requires a static FP32 traditional YOLO ONNX intermediate; dynamic/half/int8/end2end are unsupported.");
    }
    if (checkpointSuffix != QStringLiteral("pt") && normalizedFormat.startsWith(QStringLiteral("tensorrt")) && int8) {
        return QStringLiteral("TensorRT INT8 export requires official Ultralytics .pt export with calibration data; existing ONNX TensorRT conversion does not consume int8 options.");
    }
    return {};
}

QString defaultExportOutputPath(const QString& checkpointPath, const QString& outputPath, const QString& format)
{
    if (!outputPath.trimmed().isEmpty()) {
        return outputPath;
    }
    const QString suffix = format == QStringLiteral("ncnn")
        ? QStringLiteral("param")
        : (format.startsWith(QStringLiteral("tensorrt")) ? QStringLiteral("engine") : QStringLiteral("onnx"));
    return QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("model.%1").arg(suffix));
}

QString absoluteSidecarPath(const QJsonObject& object, const QString& sidecarPath, const QString& key)
{
    QString path = object.value(key).toString().trimmed();
    if (path.isEmpty()) {
        return {};
    }
    path = QDir::fromNativeSeparators(path);
    if (QFileInfo(path).isRelative()) {
        path = QFileInfo(sidecarPath).absoluteDir().filePath(path);
    }
    return QDir::cleanPath(path);
}

QJsonObject readJsonObjectFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    return document.isObject() ? document.object() : QJsonObject();
}

bool jsonLooksYolo26(const QJsonObject& object)
{
    const QStringList keys = {
        QStringLiteral("modelSeries"),
        QStringLiteral("model"),
        QStringLiteral("modelName"),
        QStringLiteral("sourceCheckpoint"),
        QStringLiteral("sourceOnnx")
    };
    for (const QString& key : keys) {
        const QString value = object.value(key).toString().trimmed().toLower();
        if (value.contains(QStringLiteral("yolo26"))) {
            return true;
        }
    }
    const QJsonObject trainingReport = object.value(QStringLiteral("trainingReport")).toObject();
    if (!trainingReport.isEmpty() && jsonLooksYolo26(trainingReport)) {
        return true;
    }
    const QJsonObject ncnn = object.value(QStringLiteral("ncnn")).toObject();
    return !ncnn.isEmpty() && jsonLooksYolo26(ncnn);
}

bool modelArtifactLooksYolo26(const QString& path)
{
    const QString normalized = QDir::fromNativeSeparators(path.trimmed());
    if (normalized.toLower().contains(QStringLiteral("yolo26"))) {
        return true;
    }

    const QFileInfo info(normalized);
    const QString suffix = info.suffix().toLower();
    if ((suffix == QStringLiteral("json") || suffix == QStringLiteral("aitrain"))
        && jsonLooksYolo26(readJsonObjectFile(normalized))) {
        return true;
    }

    const QString sidecarPath = info.dir().filePath(info.completeBaseName() + QStringLiteral(".aitrain-export.json"));
    if (QFileInfo::exists(sidecarPath) && jsonLooksYolo26(readJsonObjectFile(sidecarPath))) {
        return true;
    }

    const QString siblingReport = info.dir().filePath(QStringLiteral("ultralytics_training_report.json"));
    if (QFileInfo::exists(siblingReport) && jsonLooksYolo26(readJsonObjectFile(siblingReport))) {
        return true;
    }

    const QString parentReport = QFileInfo(info.dir().absolutePath()).dir().filePath(QStringLiteral("ultralytics_training_report.json"));
    return QFileInfo::exists(parentReport) && jsonLooksYolo26(readJsonObjectFile(parentReport));
}

bool jsonLooksSemanticSmp(const QJsonObject& object)
{
    if (object.value(QStringLiteral("modelFamily")).toString() == QStringLiteral("semantic_segmentation")
        || object.value(QStringLiteral("taskType")).toString() == QStringLiteral("semantic_segmentation")
        || object.value(QStringLiteral("backend")).toString() == QStringLiteral("smp_semantic_segmentation")
        || object.value(QStringLiteral("trainingBackend")).toString() == QStringLiteral("smp_semantic_segmentation")) {
        return true;
    }
    const QJsonObject trainingReport = object.value(QStringLiteral("trainingReport")).toObject();
    return !trainingReport.isEmpty() && jsonLooksSemanticSmp(trainingReport);
}

bool modelArtifactLooksSemanticSmp(const QString& path)
{
    const QString normalized = QDir::fromNativeSeparators(path.trimmed());
    const QFileInfo info(normalized);
    const QString suffix = info.suffix().toLower();
    if ((suffix == QStringLiteral("json") || suffix == QStringLiteral("aitrain"))
        && jsonLooksSemanticSmp(readJsonObjectFile(normalized))) {
        return true;
    }

    const QString sidecarPath = info.dir().filePath(info.completeBaseName() + QStringLiteral(".aitrain-export.json"));
    if (QFileInfo::exists(sidecarPath) && jsonLooksSemanticSmp(readJsonObjectFile(sidecarPath))) {
        return true;
    }
    const QString semanticSidecar = info.dir().filePath(QStringLiteral("semantic_segmentation_sidecar.json"));
    if (QFileInfo::exists(semanticSidecar) && jsonLooksSemanticSmp(readJsonObjectFile(semanticSidecar))) {
        return true;
    }
    const QString report = info.dir().filePath(QStringLiteral("smp_training_report.json"));
    if (QFileInfo::exists(report) && jsonLooksSemanticSmp(readJsonObjectFile(report))) {
        return true;
    }
    return normalized.toLower().contains(QStringLiteral("smp"))
        && normalized.toLower().contains(QStringLiteral("semantic"));
}

QString resolveSemanticSmpOnnxPath(const QString& checkpointPath)
{
    const QFileInfo info(checkpointPath);
    if (info.suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) == 0 && info.exists()) {
        return info.absoluteFilePath();
    }
    const QString sameStem = info.dir().filePath(info.completeBaseName() + QStringLiteral(".onnx"));
    if (QFileInfo::exists(sameStem)) {
        return QFileInfo(sameStem).absoluteFilePath();
    }
    const QString bestOnnx = info.dir().filePath(QStringLiteral("best.onnx"));
    if (QFileInfo::exists(bestOnnx)) {
        return QFileInfo(bestOnnx).absoluteFilePath();
    }
    for (const QString& jsonPath : {
             info.dir().filePath(QStringLiteral("semantic_segmentation_sidecar.json")),
             info.dir().filePath(QStringLiteral("smp_training_report.json")),
             info.dir().filePath(info.completeBaseName() + QStringLiteral(".aitrain-export.json"))}) {
        const QJsonObject object = readJsonObjectFile(jsonPath);
        if (object.isEmpty()) {
            continue;
        }
        const QStringList keys = {QStringLiteral("exportPath"), QStringLiteral("onnxPath"), QStringLiteral("sourceOnnx")};
        for (const QString& key : keys) {
            const QString candidate = absoluteSidecarPath(object, jsonPath, key);
            if (QFileInfo::exists(candidate)) {
                return QFileInfo(candidate).absoluteFilePath();
            }
        }
    }
    return {};
}

QString resolveModelArtifactPath(QString path)
{
    path = QDir::fromNativeSeparators(path.trimmed());
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix != QStringLiteral("json") && suffix != QStringLiteral("aitrain")) {
        return path;
    }

    const QJsonObject sidecar = readJsonObjectFile(path);
    if (sidecar.isEmpty()) {
        return path;
    }
    const QJsonObject ncnn = sidecar.value(QStringLiteral("ncnn")).toObject();
    const QStringList candidates = {
        absoluteSidecarPath(sidecar, path, QStringLiteral("exportPath")),
        absoluteSidecarPath(ncnn, path, QStringLiteral("paramPath")),
        absoluteSidecarPath(sidecar, path, QStringLiteral("sourceOnnx")),
        absoluteSidecarPath(ncnn, path, QStringLiteral("sourceOnnx"))
    };
    for (const QString& candidate : candidates) {
        const QFileInfo info(candidate);
        if (info.exists() && info.isFile()) {
            return info.absoluteFilePath();
        }
    }
    return path;
}
} // namespace

WorkerSession::OfficialYoloExportResult WorkerSession::runOfficialYoloExport(
    const QString& taskId,
    const QString& sourcePath,
    const QString& officialOutputPath,
    const QString& productFormat,
    const QJsonObject& exportOptions,
    bool forwardExportEvents)
{
    OfficialYoloExportResult result;
    const QString pythonExecutable = firstUsablePythonExecutable(exportOptions);
    if (pythonExecutable.isEmpty()) {
        result.error = QStringLiteral("Official YOLO export requires a usable Python executable.");
        return result;
    }
    const QString exporterScript = pythonYoloExporterScriptPath(exportOptions);
    if (!QFileInfo::exists(exporterScript)) {
        result.error = QStringLiteral("Official YOLO exporter script not found: %1").arg(exporterScript);
        return result;
    }
    if (!QDir().mkpath(QFileInfo(officialOutputPath).absolutePath())) {
        result.error = QStringLiteral("Cannot create official YOLO export directory: %1").arg(QFileInfo(officialOutputPath).absolutePath());
        return result;
    }

    const QString requestPath = QDir(QFileInfo(officialOutputPath).absolutePath()).filePath(QStringLiteral("official_yolo_export_request.json"));
    QJsonObject request;
    request.insert(QStringLiteral("protocolVersion"), 1);
    request.insert(QStringLiteral("taskId"), taskId);
    request.insert(QStringLiteral("modelPath"), sourcePath);
    request.insert(QStringLiteral("checkpointPath"), sourcePath);
    request.insert(QStringLiteral("outputPath"), officialOutputPath);
    request.insert(QStringLiteral("format"), productFormat);
    request.insert(QStringLiteral("options"), exportOptions);
    QFile requestFile(requestPath);
    if (!requestFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        result.error = QStringLiteral("Cannot write official YOLO export request: %1").arg(requestPath);
        return result;
    }
    requestFile.write(QJsonDocument(request).toJson(QJsonDocument::Indented));
    requestFile.close();

    QProcess process;
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    configurePackagedPythonEnvironment(&environment);
    process.setProcessEnvironment(environment);
    process.setProgram(pythonExecutable);
    process.setArguments(QStringList() << QStringLiteral("-u") << exporterScript << QStringLiteral("--request") << requestPath);
    process.setProcessChannelMode(QProcess::SeparateChannels);
    process.start();
    if (!process.waitForStarted(5000)) {
        result.error = QStringLiteral("Cannot start official YOLO exporter: %1").arg(process.errorString());
        return result;
    }

    QByteArray stdoutBuffer;
    QByteArray stderrBuffer;
    QString failedMessage;
    QString failedCode;
    bool completedSeen = false;
    const auto handleStdoutLine = [&](const QByteArray& line) {
        if (line.isEmpty()) {
            return;
        }
        QJsonDocument document;
        if (parseTrainerJsonDocument(line, &document)) {
            QJsonObject object = document.object();
            const QString type = object.value(QStringLiteral("type")).toString();
            QJsonObject eventPayload = object.value(QStringLiteral("payload")).toObject();
            if (eventPayload.isEmpty()) {
                eventPayload = object;
                eventPayload.remove(QStringLiteral("type"));
            }
            eventPayload.insert(QStringLiteral("taskId"), taskId);
            if (type == wp::event::failed()) {
                failedMessage = eventPayload.value(QStringLiteral("message")).toString(QStringLiteral("Official YOLO export failed."));
                failedCode = eventPayload.value(QStringLiteral("errorCode")).toString(eventPayload.value(QStringLiteral("code")).toString(QStringLiteral("ultralytics_export_failed")));
            } else if (type == wp::event::completed()) {
                completedSeen = true;
            } else {
                if (type == wp::event::modelExport()) {
                    result.modelExportPayload = eventPayload;
                }
                if (forwardExportEvents || type == wp::event::progress() || type == wp::event::log()) {
                    send(type, eventPayload);
                }
            }
        } else {
            const QJsonObject logPayload = sanitizedTrainerLogPayload(line, taskId, QStringLiteral("ultralytics_yolo_export"));
            if (!logPayload.isEmpty()) {
                send(wp::event::log(), logPayload);
            }
        }
    };
    const auto drainStdout = [&]() {
        stdoutBuffer.append(process.readAllStandardOutput());
        if (stdoutBuffer.size() > kMaxExporterBufferBytes) {
            stdoutBuffer = stdoutBuffer.right(kMaxExporterLineBytes);
            const QJsonObject logPayload = sanitizedTrainerLogPayload(
                QByteArray("Python exporter stdout buffer exceeded limit before a line delimiter; keeping bounded tail only. [log_truncated]"),
                taskId,
                QStringLiteral("ultralytics_yolo_export"));
            if (!logPayload.isEmpty()) {
                send(wp::event::log(), logPayload);
            }
        }
        int delimiter = nextPythonOutputDelimiter(stdoutBuffer);
        while (delimiter >= 0) {
            const QByteArray line = boundedExporterLine(stdoutBuffer.left(delimiter).trimmed());
            int removeCount = delimiter + 1;
            while (removeCount < stdoutBuffer.size()
                && (stdoutBuffer.at(removeCount) == '\n' || stdoutBuffer.at(removeCount) == '\r')) {
                ++removeCount;
            }
            stdoutBuffer.remove(0, removeCount);
            handleStdoutLine(line);
            delimiter = nextPythonOutputDelimiter(stdoutBuffer);
        }
    };
    const auto drainStderr = [&]() {
        stderrBuffer.append(process.readAllStandardError());
        if (stderrBuffer.size() > kMaxExporterBufferBytes) {
            stderrBuffer = stderrBuffer.right(kMaxExporterLineBytes);
            const QJsonObject logPayload = sanitizedTrainerLogPayload(
                QByteArray("Python exporter stderr buffer exceeded limit before a line delimiter; keeping bounded tail only. [log_truncated]"),
                taskId,
                QStringLiteral("ultralytics_yolo_export"));
            if (!logPayload.isEmpty()) {
                send(wp::event::log(), logPayload);
            }
        }
        int delimiter = nextPythonOutputDelimiter(stderrBuffer);
        while (delimiter >= 0) {
            const QByteArray line = boundedExporterLine(stderrBuffer.left(delimiter).trimmed());
            int removeCount = delimiter + 1;
            while (removeCount < stderrBuffer.size()
                && (stderrBuffer.at(removeCount) == '\n' || stderrBuffer.at(removeCount) == '\r')) {
                ++removeCount;
            }
            stderrBuffer.remove(0, removeCount);
            const QJsonObject logPayload = sanitizedTrainerLogPayload(line, taskId, QStringLiteral("ultralytics_yolo_export"));
            if (!logPayload.isEmpty()) {
                send(wp::event::log(), logPayload);
            }
            delimiter = nextPythonOutputDelimiter(stderrBuffer);
        }
    };
    while (process.state() != QProcess::NotRunning) {
        process.waitForReadyRead(50);
        drainStdout();
        drainStderr();
        if (pollPendingCancel(1)) {
            process.terminate();
            if (!process.waitForFinished(1500)) {
                process.kill();
                process.waitForFinished(1500);
            }
            result.error = QStringLiteral("Canceled by user");
            return result;
        }
        QCoreApplication::processEvents();
    }
    drainStdout();
    drainStderr();
    if (!stdoutBuffer.trimmed().isEmpty()) {
        handleStdoutLine(boundedExporterLine(stdoutBuffer.trimmed()));
    }
    if (!stderrBuffer.trimmed().isEmpty()) {
        const QJsonObject logPayload = sanitizedTrainerLogPayload(boundedExporterLine(stderrBuffer.trimmed()), taskId, QStringLiteral("ultralytics_yolo_export"));
        if (!logPayload.isEmpty()) {
            send(wp::event::log(), logPayload);
        }
    }
    if (!failedMessage.isEmpty()) {
        result.error = failedCode.isEmpty() ? failedMessage : QStringLiteral("%1: %2").arg(failedCode, failedMessage);
        return result;
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0 || !completedSeen) {
        result.error = QStringLiteral("Official YOLO exporter exited without a completed message.");
        return result;
    }
    if (result.modelExportPayload.isEmpty()) {
        result.error = QStringLiteral("Official YOLO exporter did not emit a modelExport payload.");
        return result;
    }
    result.ok = true;
    return result;
}

void WorkerSession::evaluateModel(const QJsonObject& payload)
{
    const wr::ModelEvaluationRequest request = wr::parseModelEvaluationRequest(payload);
    const QString taskId = request.taskId;
    activeTaskId_ = taskId;
    canceled_ = false;
    running_ = true;
    const QString modelPath = request.modelPath;
    const QString datasetPath = request.datasetPath;
    const QString taskType = request.taskType.isEmpty() ? QStringLiteral("detection") : request.taskType;
    QString outputPath = request.outputPath;
    const QJsonObject options = request.options;
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(modelPath).absoluteDir().absolutePath(), taskId);
    }
    activeOutputPath_ = outputPath;

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始评估模型。"));
    send(wp::event::progress(), progress);
    if (pollPendingCancel()) {
        sendCanceledAndFinish(taskId, QStringLiteral("Canceled by user"));
        return;
    }

    const aitrain::WorkflowResult result = aitrain::evaluateModelReport(modelPath, datasetPath, outputPath, taskType, options, cancellationCallback());
    running_ = false;
    if (canceled_) {
        return;
    }
    if (!result.ok && result.error == QStringLiteral("Canceled by user")) {
        sendCanceledAndFinish(taskId, result.error);
        return;
    }

    const auto emitEvaluationArtifacts = [&]() {
        if (result.reportPath.isEmpty()) {
            return;
        }
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
        artifact.insert(QStringLiteral("path"), result.reportPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Model evaluation report"));
        send(wp::event::artifact(), artifact);
        for (const auto& item : {
                 qMakePair(QStringLiteral("per_class_metrics"), QStringLiteral("perClassMetricsPath")),
                 qMakePair(QStringLiteral("error_samples"), QStringLiteral("errorSamplesPath")),
                 qMakePair(QStringLiteral("confusion_matrix"), QStringLiteral("confusionMatrixPath")),
                 qMakePair(QStringLiteral("evaluation_summary"), QStringLiteral("evaluationSummaryPath")),
                 qMakePair(QStringLiteral("evaluation_overlays"), QStringLiteral("overlayDir")),
                 qMakePair(QStringLiteral("official_metrics"), QStringLiteral("officialMetricsPath")),
                 qMakePair(QStringLiteral("official_run_dir"), QStringLiteral("officialRunDir")),
                 qMakePair(QStringLiteral("official_log"), QStringLiteral("officialLogPath"))}) {
            const QString path = result.payload.value(item.second).toString();
            if (!path.isEmpty()) {
                QJsonObject extraArtifact;
                extraArtifact.insert(QStringLiteral("taskId"), taskId);
                extraArtifact.insert(QStringLiteral("kind"), item.first);
                extraArtifact.insert(QStringLiteral("path"), path);
                extraArtifact.insert(QStringLiteral("message"), QStringLiteral("Model evaluation artifact"));
                send(wp::event::artifact(), extraArtifact);
            }
        }
        const QString legacyOverlaysPath = result.payload.value(QStringLiteral("overlaysPath")).toString();
        if (!legacyOverlaysPath.isEmpty()) {
            QJsonObject extraArtifact;
            extraArtifact.insert(QStringLiteral("taskId"), taskId);
            extraArtifact.insert(QStringLiteral("kind"), QStringLiteral("evaluation_overlays"));
            extraArtifact.insert(QStringLiteral("path"), legacyOverlaysPath);
            extraArtifact.insert(QStringLiteral("message"), QStringLiteral("Model evaluation artifact"));
            send(wp::event::artifact(), extraArtifact);
        }
        const QJsonArray officialArtifacts = result.payload.value(QStringLiteral("officialArtifacts")).toArray();
        for (const QJsonValue& value : officialArtifacts) {
            const QJsonObject object = value.toObject();
            const QString path = object.value(QStringLiteral("path")).toString();
            if (path.isEmpty()) {
                continue;
            }
            QJsonObject officialArtifact;
            officialArtifact.insert(QStringLiteral("taskId"), taskId);
            officialArtifact.insert(QStringLiteral("kind"), object.value(QStringLiteral("kind")).toString(QStringLiteral("official_artifact")));
            officialArtifact.insert(QStringLiteral("path"), path);
            officialArtifact.insert(QStringLiteral("message"), object.value(QStringLiteral("name")).toString(QStringLiteral("Official Ultralytics artifact")));
            send(wp::event::artifact(), officialArtifact);
        }
    };

    emitEvaluationArtifacts();
    if (!result.ok) {
        QJsonObject details = result.payload;
        details.insert(wp::field::reportPath(), result.reportPath);
        send(wp::event::evaluationReport(), result.payload);
        socket_.waitForBytesWritten(1000);
        failWithDetails(
            result.error.isEmpty() ? QStringLiteral("Model evaluation failed.") : result.error,
            result.payload.value(QStringLiteral("failureCategory")).toString(QStringLiteral("evaluation_failed")),
            details);
        return;
    }
    if (!result.payload.value(QStringLiteral("ok")).toBool(true)) {
        const QString failureCategory = result.payload.value(QStringLiteral("failureCategory")).toString(
            QStringLiteral("evaluation_failed"));
        const QString message = result.payload.value(QStringLiteral("message")).toString(
            QStringLiteral("Model evaluation report did not pass."));
        QJsonObject details = result.payload;
        details.insert(wp::field::reportPath(), result.reportPath);
        send(wp::event::evaluationReport(), result.payload);
        socket_.waitForBytesWritten(1000);
        failWithDetails(message, failureCategory, details);
        return;
    }

    QJsonObject progressDone;
    progressDone.insert(QStringLiteral("taskId"), taskId);
    progressDone.insert(QStringLiteral("percent"), 100);
    progressDone.insert(QStringLiteral("message"), QStringLiteral("模型评估完成。"));
    send(wp::event::progress(), progressDone);
    send(wp::event::evaluationReport(), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Model evaluation completed"));
    send(wp::event::completed(), completed);
    finishSession();
}

void WorkerSession::benchmarkModel(const QJsonObject& payload)
{
    const wr::ModelBenchmarkRequest request = wr::parseModelBenchmarkRequest(payload);
    const QString taskId = request.taskId;
    activeTaskId_ = taskId;
    canceled_ = false;
    running_ = true;
    const QString modelPath = request.modelPath;
    QString outputPath = request.outputPath;
    const QJsonObject options = request.options;
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(modelPath).absoluteDir().absolutePath(), taskId);
    }
    activeOutputPath_ = outputPath;

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始部署基准测试。"));
    send(wp::event::progress(), progress);
    if (pollPendingCancel()) {
        sendCanceledAndFinish(taskId, QStringLiteral("Canceled by user"));
        return;
    }

    const aitrain::WorkflowResult result = aitrain::benchmarkModelReport(modelPath, outputPath, options, cancellationCallback());
    running_ = false;
    if (canceled_) {
        return;
    }
    if (!result.ok && result.error == QStringLiteral("Canceled by user")) {
        sendCanceledAndFinish(taskId, result.error);
        return;
    }
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("benchmark_report"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Model benchmark report"));
    send(wp::event::artifact(), artifact);
    if (!result.payload.value(QStringLiteral("ok")).toBool(true)) {
        const QString failureCategory = result.payload.value(QStringLiteral("failureCategory")).toString(
            QStringLiteral("benchmark_failed"));
        const QString message = result.payload.value(QStringLiteral("message")).toString(
            QStringLiteral("Model benchmark report did not pass."));
        QJsonObject details = result.payload;
        details.insert(wp::field::reportPath(), result.reportPath);
        send(wp::event::benchmarkReport(), result.payload);
        socket_.waitForBytesWritten(1000);
        failWithDetails(message, failureCategory, details);
        return;
    }
    QJsonObject progressDone;
    progressDone.insert(QStringLiteral("taskId"), taskId);
    progressDone.insert(QStringLiteral("percent"), 100);
    progressDone.insert(QStringLiteral("message"), QStringLiteral("部署基准测试完成。"));
    send(wp::event::progress(), progressDone);
    send(wp::event::benchmarkReport(), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Model benchmark completed"));
    send(wp::event::completed(), completed);
    finishSession();
}

void WorkerSession::generateDeliveryReport(const QJsonObject& payload)
{
    const wr::DeliveryReportRequest request = wr::parseDeliveryReportRequest(payload);
    const QString taskId = request.taskId;
    QString outputPath = request.outputPath;
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QDir::currentPath(), taskId);
    }
    activeTaskId_ = taskId;
    activeOutputPath_ = outputPath;
    QJsonObject context = request.context;
    context.insert(QStringLiteral("taskId"), taskId);

    const aitrain::WorkflowResult result = aitrain::generateTrainingDeliveryReport(outputPath, context);
    if (!result.ok) {
        fail(result.error);
        return;
    }
    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("training_delivery_report"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Training delivery HTML report"));
    send(wp::event::artifact(), artifact);
    const QString jsonPath = result.payload.value(QStringLiteral("jsonPath")).toString();
    if (!jsonPath.isEmpty()) {
        QJsonObject jsonArtifact;
        jsonArtifact.insert(QStringLiteral("taskId"), taskId);
        jsonArtifact.insert(QStringLiteral("kind"), QStringLiteral("training_delivery_report_json"));
        jsonArtifact.insert(QStringLiteral("path"), jsonPath);
        jsonArtifact.insert(QStringLiteral("message"), QStringLiteral("Training delivery report JSON context"));
        send(wp::event::artifact(), jsonArtifact);
    }
    const QString modelCardPath = result.payload.value(QStringLiteral("modelCardPath")).toString();
    if (!modelCardPath.isEmpty()) {
        QJsonObject modelCardArtifact;
        modelCardArtifact.insert(QStringLiteral("taskId"), taskId);
        modelCardArtifact.insert(QStringLiteral("kind"), QStringLiteral("model_card"));
        modelCardArtifact.insert(QStringLiteral("path"), modelCardPath);
        modelCardArtifact.insert(QStringLiteral("message"), QStringLiteral("Model card JSON"));
        send(wp::event::artifact(), modelCardArtifact);
    }
    const QString inventoryPath = result.payload.value(QStringLiteral("artifactInventoryPath")).toString();
    if (!inventoryPath.isEmpty()) {
        QJsonObject inventoryArtifact;
        inventoryArtifact.insert(QStringLiteral("taskId"), taskId);
        inventoryArtifact.insert(QStringLiteral("kind"), QStringLiteral("delivery_artifact_inventory"));
        inventoryArtifact.insert(QStringLiteral("path"), inventoryPath);
        inventoryArtifact.insert(QStringLiteral("message"), QStringLiteral("Delivery artifact inventory"));
        send(wp::event::artifact(), inventoryArtifact);
    }
    const QString manifestPath = result.payload.value(QStringLiteral("deliveryManifestPath")).toString();
    if (!manifestPath.isEmpty()) {
        QJsonObject manifestArtifact;
        manifestArtifact.insert(QStringLiteral("taskId"), taskId);
        manifestArtifact.insert(QStringLiteral("kind"), QStringLiteral("delivery_manifest"));
        manifestArtifact.insert(QStringLiteral("path"), manifestPath);
        manifestArtifact.insert(QStringLiteral("message"), QStringLiteral("Delivery manifest"));
        send(wp::event::artifact(), manifestArtifact);
    }
    send(wp::event::deliveryReport(), result.payload);
    socket_.waitForBytesWritten(1000);
    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Training delivery report generated"));
    send(wp::event::completed(), completed);
    finishSession();
}

void WorkerSession::runCustomerOcrAcceptance(const QJsonObject& payload)
{
    const wr::CustomerOcrAcceptanceRequest request = wr::parseCustomerOcrAcceptanceRequest(payload);
    const QString taskId = request.taskId;
    QString outputPath = request.outputPath;
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QDir::currentPath(), taskId);
    }
    activeTaskId_ = taskId;
    activeOutputPath_ = outputPath;
    QJsonObject options = request.options;
    options.insert(QStringLiteral("taskId"), taskId);

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始客户域 OCR 验收。"));
    send(wp::event::progress(), progress);
    if (pollPendingCancel()) {
        sendCanceledAndFinish(taskId, QStringLiteral("Canceled by user"));
        return;
    }

    const aitrain::WorkflowResult result = aitrain::runCustomerOcrAcceptanceReport(outputPath, options);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("customer_ocr_acceptance"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Customer OCR acceptance report"));
    send(wp::event::artifact(), artifact);
    const QString summaryPath = result.payload.value(QStringLiteral("summaryPath")).toString();
    if (!summaryPath.isEmpty()) {
        QJsonObject summaryArtifact;
        summaryArtifact.insert(QStringLiteral("taskId"), taskId);
        summaryArtifact.insert(QStringLiteral("kind"), QStringLiteral("customer_ocr_acceptance_summary"));
        summaryArtifact.insert(QStringLiteral("path"), summaryPath);
        summaryArtifact.insert(QStringLiteral("message"), QStringLiteral("Customer OCR acceptance summary"));
        send(wp::event::artifact(), summaryArtifact);
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("客户域 OCR 验收报告已生成。"));
    send(wp::event::progress(), doneProgress);
    send(wp::event::customerOcrAcceptance(), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Customer OCR acceptance completed"));
    send(wp::event::completed(), completed);
    finishSession();
}

void WorkerSession::collectDiagnostics(const QJsonObject& payload)
{
    const wr::DiagnosticsBundleRequest request = wr::parseDiagnosticsBundleRequest(payload);
    const QString taskId = request.taskId;
    QString outputPath = request.outputPath;
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QDir::currentPath(), taskId);
    }
    activeTaskId_ = taskId;
    activeOutputPath_ = outputPath;
    QJsonObject context = request.context;
    context.insert(QStringLiteral("taskId"), taskId);
    if (context.value(QStringLiteral("workerExecutable")).toString().isEmpty()) {
        context.insert(QStringLiteral("workerExecutable"), QCoreApplication::applicationFilePath());
    }

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始收集诊断包。"));
    send(wp::event::progress(), progress);
    if (pollPendingCancel()) {
        sendCanceledAndFinish(taskId, QStringLiteral("Canceled by user"));
        return;
    }

    const aitrain::WorkflowResult result = aitrain::collectDiagnosticsReport(outputPath, context);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    for (const auto& item : {
             qMakePair(QStringLiteral("diagnostic_manifest"), QStringLiteral("manifestPath")),
             qMakePair(QStringLiteral("diagnostic_bundle"), QStringLiteral("bundlePath")),
             qMakePair(QStringLiteral("diagnostic_summary"), QStringLiteral("summaryPath"))}) {
        const QString path = result.payload.value(item.second).toString();
        if (path.isEmpty()) {
            continue;
        }
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), item.first);
        artifact.insert(QStringLiteral("path"), path);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Diagnostic bundle artifact"));
        send(wp::event::artifact(), artifact);
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("诊断包已生成。"));
    send(wp::event::progress(), doneProgress);
    send(wp::event::diagnosticBundle(), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Diagnostics collected"));
    send(wp::event::completed(), completed);
    finishSession();
}

void WorkerSession::validateDeploymentArtifact(const QJsonObject& payload)
{
    const wr::DeploymentValidationRequest request = wr::parseDeploymentValidationRequest(payload);
    const QString taskId = request.taskId;
    const QString modelPath = request.modelPath;
    const QString format = request.format;
    QString outputPath = request.outputPath;
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(modelPath).absoluteDir().absolutePath(), taskId);
    }
    activeTaskId_ = taskId;
    activeOutputPath_ = outputPath;
    QJsonObject options = request.options;
    const QString sampleImagePath = request.sampleImagePath;
    if (!sampleImagePath.isEmpty()) {
        options.insert(QStringLiteral("sampleImagePath"), sampleImagePath);
    }

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始验证部署产物。"));
    send(wp::event::progress(), progress);
    if (pollPendingCancel()) {
        sendCanceledAndFinish(taskId, QStringLiteral("Canceled by user"));
        return;
    }

    const aitrain::WorkflowResult result = aitrain::validateDeploymentArtifactReport(modelPath, outputPath, format, options);
    if (!result.ok) {
        failWithDetails(
            result.error,
            QStringLiteral("deployment_validation_failed"),
            QJsonObject{{QStringLiteral("outputPath"), outputPath}});
        return;
    }
    activeReportPath_ = result.payload.value(QStringLiteral("reportPath")).toString();

    for (const auto& item : {
             qMakePair(QStringLiteral("deployment_validation_report"), QStringLiteral("reportPath")),
             qMakePair(QStringLiteral("deployment_validation_summary"), QStringLiteral("summaryPath")),
             qMakePair(QStringLiteral("deployment_predictions"), QStringLiteral("predictionsPath")),
             qMakePair(QStringLiteral("deployment_overlay"), QStringLiteral("overlayPath"))}) {
        const QString path = result.payload.value(item.second).toString();
        if (path.isEmpty()) {
            continue;
        }
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), item.first);
        artifact.insert(QStringLiteral("path"), path);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Deployment validation artifact"));
        send(wp::event::artifact(), artifact);
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("部署产物验证已完成。"));
    send(wp::event::progress(), doneProgress);
    send(wp::event::deploymentValidation(), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("command"), activeCommand_);
    completed.insert(QStringLiteral("status"), result.payload.value(QStringLiteral("status")).toString(QStringLiteral("passed")));
    completed.insert(QStringLiteral("reportPath"), result.payload.value(QStringLiteral("reportPath")).toString());
    completed.insert(QStringLiteral("outputPath"), outputPath);
    if (!result.payload.value(QStringLiteral("ok")).toBool(true)) {
        completed.insert(QStringLiteral("errorCode"), result.payload.value(QStringLiteral("errorCode")).toString());
        completed.insert(QStringLiteral("failureCategory"), result.payload.value(QStringLiteral("failureCategory")).toString());
        completed.insert(QStringLiteral("nextAction"), result.payload.value(QStringLiteral("nextAction")).toString());
    }
    completed.insert(QStringLiteral("message"), QStringLiteral("Deployment validation completed"));
    send(wp::event::completed(), completed);
    finishSession();
}

void WorkerSession::exportModel(const QJsonObject& payload)
{
    const wr::ModelExportRequest request = wr::parseModelExportRequest(payload);
    const QString taskId = request.taskId;
    activeTaskId_ = taskId;
    canceled_ = false;
    running_ = true;
    const QString requestedCheckpointPath = QDir::fromNativeSeparators(request.checkpointPath.trimmed());
    QString checkpointPath = resolveModelArtifactPath(request.checkpointPath);
    QString outputPath = request.outputPath;
    const QString format = request.format.isEmpty() ? QStringLiteral("onnx") : request.format;
    const QJsonObject options = request.options;
    outputPath = defaultExportOutputPath(checkpointPath, outputPath, format);
    activeOutputPath_ = outputPath;

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始导出模型。"));
    send(wp::event::progress(), startProgress);
    if (pollPendingCancel()) {
        sendCanceledAndFinish(taskId, QStringLiteral("Canceled by user"));
        return;
    }

    if (format == QStringLiteral("ncnn")
        && (modelArtifactLooksYolo26(requestedCheckpointPath) || modelArtifactLooksYolo26(checkpointPath))) {
        running_ = false;
        failWithDetails(
            QStringLiteral("YOLO26 NCNN export is not supported by AITrain; use ONNX or TensorRT for YOLO26 deployment."),
            QStringLiteral("unsupported_yolo26_ncnn"),
            QJsonObject{
                {QStringLiteral("format"), format},
                {QStringLiteral("checkpointPath"), checkpointPath},
                {QStringLiteral("outputPath"), outputPath}});
        return;
    }

    const QString checkpointSuffix = QFileInfo(checkpointPath).suffix().toLower();
    if (modelArtifactLooksSemanticSmp(requestedCheckpointPath) || modelArtifactLooksSemanticSmp(checkpointPath)) {
        const QString normalizedFormat = format.trimmed().toLower();
        if (normalizedFormat != QStringLiteral("onnx")) {
            running_ = false;
            failWithDetails(
                QStringLiteral("SMP semantic segmentation export supports ONNX only. NCNN/TensorRT export is not part of the SMP capability scope."),
                QStringLiteral("unsupported_semantic_segmentation_export_format"),
                QJsonObject{
                    {QStringLiteral("format"), format},
                    {QStringLiteral("checkpointPath"), checkpointPath},
                    {QStringLiteral("outputPath"), outputPath},
                    {QStringLiteral("modelFamily"), QStringLiteral("semantic_segmentation")}});
            return;
        }
        const QString sourceOnnx = resolveSemanticSmpOnnxPath(checkpointPath);
        if (sourceOnnx.isEmpty()) {
            running_ = false;
            failWithDetails(
                QStringLiteral("SMP semantic segmentation checkpoint does not have a sibling ONNX export. Re-run training or export from the SMP trainer output directory."),
                QStringLiteral("semantic_segmentation_onnx_missing"),
                QJsonObject{{QStringLiteral("checkpointPath"), checkpointPath}, {QStringLiteral("outputPath"), outputPath}});
            return;
        }
        if (!QDir().mkpath(QFileInfo(outputPath).absolutePath())) {
            running_ = false;
            failWithDetails(
                QStringLiteral("Cannot create semantic segmentation export directory: %1").arg(QFileInfo(outputPath).absolutePath()),
                QStringLiteral("output_create_failed"),
                QJsonObject{{QStringLiteral("outputPath"), outputPath}});
            return;
        }
        const QString normalizedSource = QFileInfo(sourceOnnx).absoluteFilePath();
        const QString normalizedOutput = QFileInfo(outputPath).absoluteFilePath();
        if (normalizedSource.compare(normalizedOutput, Qt::CaseInsensitive) != 0) {
            if (QFileInfo::exists(outputPath) && !QFile::remove(outputPath)) {
                running_ = false;
                fail(QStringLiteral("Cannot overwrite export path: %1").arg(outputPath));
                return;
            }
            if (!QFile::copy(sourceOnnx, outputPath)) {
                running_ = false;
                fail(QStringLiteral("Cannot copy SMP ONNX export: %1 -> %2").arg(sourceOnnx, outputPath));
                return;
            }
        }

        const QFileInfo outputInfo(outputPath);
        const QString sourceSidecar = QFileInfo(sourceOnnx).dir().filePath(QFileInfo(sourceOnnx).completeBaseName() + QStringLiteral(".aitrain-export.json"));
        QJsonObject config = readJsonObjectFile(sourceSidecar);
        if (config.isEmpty()) {
            config = readJsonObjectFile(QFileInfo(sourceOnnx).dir().filePath(QStringLiteral("semantic_segmentation_sidecar.json")));
        }
        config.insert(QStringLiteral("schemaVersion"), 1);
        config.insert(QStringLiteral("backend"), QStringLiteral("smp_semantic_segmentation"));
        config.insert(QStringLiteral("modelFamily"), QStringLiteral("semantic_segmentation"));
        config.insert(QStringLiteral("taskType"), QStringLiteral("semantic_segmentation"));
        config.insert(QStringLiteral("datasetFormat"), QStringLiteral("semantic_segmentation_mask"));
        config.insert(QStringLiteral("exportPath"), outputInfo.absoluteFilePath());
        config.insert(QStringLiteral("sourceOnnx"), normalizedSource);
        const QString outputSidecar = outputInfo.dir().filePath(outputInfo.completeBaseName() + QStringLiteral(".aitrain-export.json"));
        QString writeError;
        if (!writeJsonFile(outputSidecar, config, &writeError)) {
            running_ = false;
            fail(writeError);
            return;
        }

        QJsonObject report;
        report.insert(QStringLiteral("ok"), true);
        report.insert(QStringLiteral("kind"), QStringLiteral("model_export_report"));
        report.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        report.insert(QStringLiteral("backend"), QStringLiteral("smp_semantic_segmentation"));
        report.insert(QStringLiteral("modelFamily"), QStringLiteral("semantic_segmentation"));
        report.insert(QStringLiteral("checkpointPath"), checkpointPath);
        report.insert(QStringLiteral("sourceOnnx"), normalizedSource);
        report.insert(QStringLiteral("exportPath"), outputInfo.absoluteFilePath());
        report.insert(QStringLiteral("sidecarPath"), outputSidecar);
        report.insert(QStringLiteral("exportedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
        report.insert(QStringLiteral("note"), QStringLiteral("SMP semantic segmentation export copies the trainer-produced ONNX and sidecar. NCNN/TensorRT exports are not required for SMP."));
        const QString reportPath = outputInfo.dir().filePath(QStringLiteral("model_export_report.json"));
        if (!writeJsonFile(reportPath, report, &writeError)) {
            running_ = false;
            fail(writeError);
            return;
        }

        QJsonObject progressPayload;
        progressPayload.insert(QStringLiteral("percent"), 100);
        progressPayload.insert(QStringLiteral("message"), QStringLiteral("模型导出完成。"));
        send(wp::event::progress(), progressPayload);

        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("export"));
        artifact.insert(QStringLiteral("path"), outputInfo.absoluteFilePath());
        artifact.insert(QStringLiteral("message"), QStringLiteral("SMP semantic segmentation ONNX export"));
        send(wp::event::artifact(), artifact);
        QJsonObject sidecarArtifact;
        sidecarArtifact.insert(QStringLiteral("taskId"), taskId);
        sidecarArtifact.insert(QStringLiteral("kind"), QStringLiteral("export_sidecar"));
        sidecarArtifact.insert(QStringLiteral("path"), outputSidecar);
        sidecarArtifact.insert(QStringLiteral("message"), QStringLiteral("SMP semantic segmentation sidecar"));
        send(wp::event::artifact(), sidecarArtifact);

        QJsonObject response;
        response.insert(QStringLiteral("ok"), true);
        response.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        response.insert(QStringLiteral("taskId"), taskId);
        response.insert(QStringLiteral("checkpointPath"), checkpointPath);
        response.insert(QStringLiteral("exportPath"), outputInfo.absoluteFilePath());
        response.insert(QStringLiteral("reportPath"), reportPath);
        response.insert(QStringLiteral("config"), config);
        response.insert(QStringLiteral("exportedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
        send(wp::event::modelExport(), response);

        QJsonObject completed;
        completed.insert(QStringLiteral("message"), QStringLiteral("Model export completed"));
        completed.insert(QStringLiteral("taskId"), taskId);
        completed.insert(QStringLiteral("command"), activeCommand_);
        completed.insert(QStringLiteral("status"), QStringLiteral("completed"));
        completed.insert(QStringLiteral("exportPath"), outputInfo.absoluteFilePath());
        completed.insert(QStringLiteral("reportPath"), reportPath);
        running_ = false;
        send(wp::event::completed(), completed);
        finishSession();
        return;
    }
    const QString unsupportedOptions = unsupportedOfficialExportOptionsError(format, checkpointSuffix, options);
    if (!unsupportedOptions.isEmpty()) {
        running_ = false;
        failWithDetails(
            unsupportedOptions,
            QStringLiteral("unsupported_export_options"),
            QJsonObject{
                {QStringLiteral("format"), format},
                {QStringLiteral("checkpointPath"), checkpointPath},
                {QStringLiteral("outputPath"), outputPath}});
        return;
    }
    if (checkpointSuffix == QStringLiteral("pt")) {
        QJsonObject modelExportPayload;
        if (format == QStringLiteral("onnx") || format.startsWith(QStringLiteral("tensorrt"))) {
            const OfficialYoloExportResult officialExport = runOfficialYoloExport(taskId, checkpointPath, outputPath, format, options, true);
            if (!officialExport.ok) {
                running_ = false;
                if (officialExport.error == QStringLiteral("Canceled by user")) {
                    sendCanceledAndFinish(taskId, officialExport.error);
                    return;
                }
                failWithDetails(officialExport.error, QStringLiteral("official_yolo_export_failed"), QJsonObject{{QStringLiteral("outputPath"), outputPath}});
                return;
            }
            modelExportPayload = officialExport.modelExportPayload;
            running_ = false;
            if (canceled_) {
                return;
            }
            QJsonObject completed;
            completed.insert(QStringLiteral("message"), QStringLiteral("Model export completed"));
            completed.insert(QStringLiteral("taskId"), taskId);
            completed.insert(QStringLiteral("command"), activeCommand_);
            completed.insert(QStringLiteral("status"), QStringLiteral("completed"));
            completed.insert(QStringLiteral("exportPath"), modelExportPayload.value(QStringLiteral("exportPath")).toString());
            completed.insert(QStringLiteral("reportPath"), modelExportPayload.value(QStringLiteral("reportPath")).toString());
            send(wp::event::completed(), completed);
            finishSession();
            return;
        }
        if (format == QStringLiteral("ncnn")) {
            if (ncnnOfficialExportOptionsUnsupported(options)) {
                running_ = false;
                failWithDetails(
                    QStringLiteral("NCNN export from .pt requires a static FP32 traditional YOLO ONNX intermediate; dynamic/half/int8/end2end are unsupported."),
                    QStringLiteral("unsupported_export_options"),
                    QJsonObject{{QStringLiteral("format"), format}, {QStringLiteral("outputPath"), outputPath}});
                return;
            }
            QJsonObject intermediateOptions = options;
            QJsonObject args = intermediateOptions.value(QStringLiteral("ultralyticsExportArgs")).toObject();
            args.insert(QStringLiteral("format"), QStringLiteral("onnx"));
            args.insert(QStringLiteral("dynamic"), false);
            args.insert(QStringLiteral("half"), false);
            args.insert(QStringLiteral("int8"), false);
            args.insert(QStringLiteral("end2end"), false);
            intermediateOptions.insert(QStringLiteral("ultralyticsExportArgs"), args);
            const QString intermediateDir = QDir(QFileInfo(outputPath).absolutePath()).filePath(QStringLiteral("official_onnx_intermediate"));
            const QString intermediateOnnx = QDir(intermediateDir).filePath(QStringLiteral("model.onnx"));
            const OfficialYoloExportResult officialExport = runOfficialYoloExport(taskId, checkpointPath, intermediateOnnx, QStringLiteral("onnx"), intermediateOptions, false);
            if (!officialExport.ok) {
                running_ = false;
                if (officialExport.error == QStringLiteral("Canceled by user")) {
                    sendCanceledAndFinish(taskId, officialExport.error);
                    return;
                }
                failWithDetails(officialExport.error, QStringLiteral("official_yolo_export_failed"), QJsonObject{{QStringLiteral("outputPath"), intermediateOnnx}});
                return;
            }
            modelExportPayload = officialExport.modelExportPayload;
            checkpointPath = intermediateOnnx;
        } else {
            running_ = false;
            fail(QStringLiteral("Unsupported export format for .pt source: %1").arg(format));
            return;
        }
    }

    const aitrain::DetectionExportResult result = aitrain::exportDetectionCheckpoint(checkpointPath, outputPath, format, cancellationCallback());
    running_ = false;
    if (canceled_) {
        return;
    }
    if (!result.ok && result.error == QStringLiteral("Canceled by user")) {
        sendCanceledAndFinish(taskId, result.error);
        return;
    }
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("模型导出完成。"));
    send(wp::event::progress(), progressPayload);

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("export"));
    artifact.insert(QStringLiteral("path"), result.exportPath);
    QString artifactMessage = QStringLiteral("ONNX model export");
    if (result.format == QStringLiteral("ncnn")) {
        artifactMessage = QStringLiteral("NCNN param export");
    } else if (result.format.startsWith(QStringLiteral("tensorrt"))) {
        artifactMessage = QStringLiteral("TensorRT engine export");
    }
    artifact.insert(QStringLiteral("message"), artifactMessage);
    send(wp::event::artifact(), artifact);

    const QJsonObject ncnnConfig = result.config.value(QStringLiteral("ncnn")).toObject();
    const QString ncnnBinPath = ncnnConfig.value(QStringLiteral("binPath")).toString();
    if (result.format == QStringLiteral("ncnn") && !ncnnBinPath.isEmpty()) {
        QJsonObject binArtifact;
        binArtifact.insert(QStringLiteral("taskId"), taskId);
        binArtifact.insert(QStringLiteral("kind"), QStringLiteral("export_sidecar"));
        binArtifact.insert(QStringLiteral("path"), ncnnBinPath);
        binArtifact.insert(QStringLiteral("message"), QStringLiteral("NCNN binary weights"));
        send(wp::event::artifact(), binArtifact);
    }

    QJsonObject response;
    response.insert(QStringLiteral("ok"), true);
    response.insert(QStringLiteral("format"), result.format);
    response.insert(QStringLiteral("taskId"), taskId);
    response.insert(QStringLiteral("checkpointPath"), checkpointPath);
    response.insert(QStringLiteral("exportPath"), result.exportPath);
    response.insert(QStringLiteral("reportPath"), result.reportPath);
    response.insert(QStringLiteral("config"), result.config);
    response.insert(QStringLiteral("exportedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(wp::event::modelExport(), response);

    QJsonObject completed;
    completed.insert(QStringLiteral("message"), QStringLiteral("Model export completed"));
    send(wp::event::completed(), completed);
    finishSession();
}

void WorkerSession::runInference(const QJsonObject& payload)
{
    const wr::InferenceRequest request = wr::parseInferenceRequest(payload);
    const QString taskId = request.taskId;
    const QString checkpointPath = resolveModelArtifactPath(request.checkpointPath);
    const QString imagePath = request.imagePath;
    QString outputPath = request.outputPath;
    aitrain::DetectionInferenceOptions options;
    options.confidenceThreshold = payload.value(QStringLiteral("confidenceThreshold")).toDouble(options.confidenceThreshold);
    options.iouThreshold = payload.value(QStringLiteral("iouThreshold")).toDouble(options.iouThreshold);
    options.maxDetections = payload.value(QStringLiteral("maxDetections")).toInt(options.maxDetections);
    if (outputPath.isEmpty()) {
        outputPath = QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("inference"));
    }
    activeTaskId_ = taskId;
    activeOutputPath_ = outputPath;
    canceled_ = false;
    running_ = true;
    if (!QDir().mkpath(outputPath)) {
        fail(QStringLiteral("Cannot create inference output directory: %1").arg(outputPath));
        return;
    }

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始推理。"));
    send(wp::event::progress(), startProgress);
    if (pollPendingCancel()) {
        sendCanceledAndFinish(taskId, QStringLiteral("Canceled by user"));
        return;
    }

    QElapsedTimer elapsed;
    elapsed.start();
    QString error;
    QJsonArray predictionArray;
    QImage overlay;
    QString taskType = QStringLiteral("detection");
    int predictionCount = 0;
    const QString modelSuffix = QFileInfo(checkpointPath).suffix().toLower();
    const bool onnxModel = modelSuffix == QStringLiteral("onnx");
    const bool ncnnModel = modelSuffix == QStringLiteral("param");
    const bool tensorRtModel = modelSuffix == QStringLiteral("engine") || modelSuffix == QStringLiteral("plan");
    if (onnxModel) {
        const QString modelFamily = aitrain::inferOnnxModelFamily(checkpointPath);
        if (modelFamily == QStringLiteral("ocr_recognition")
            || modelFamily == QStringLiteral("ocr_detection")) {
            fail(QStringLiteral("OCR inference is official-only. Use the PaddleOCR official Det/Rec/System adapter artifacts instead of AITrain C++ ONNX OCR postprocess."));
            return;
        }
        if (modelFamily == QStringLiteral("semantic_segmentation")) {
            taskType = QStringLiteral("semantic_segmentation");
            const aitrain::SemanticSegmentationPrediction prediction = aitrain::predictSemanticSegmentationOnnxRuntime(checkpointPath, imagePath, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            predictionArray.append(aitrain::semanticSegmentationPredictionToJson(prediction));
            overlay = aitrain::renderSemanticSegmentationPrediction(imagePath, prediction, &error);
            predictionCount = 0;
            const QStringList keys = prediction.pixelCounts.keys();
            for (const QString& key : keys) {
                if (key != QStringLiteral("0") && prediction.pixelCounts.value(key).toDouble() > 0.0) {
                    ++predictionCount;
                }
            }
        } else if (modelFamily == QStringLiteral("yolo_segmentation")) {
            taskType = QStringLiteral("segmentation");
            const QVector<aitrain::SegmentationPrediction> predictions = aitrain::predictSegmentationOnnxRuntime(checkpointPath, imagePath, options, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            for (const aitrain::SegmentationPrediction& prediction : predictions) {
                predictionArray.append(aitrain::segmentationPredictionToJson(prediction));
            }
            overlay = aitrain::renderSegmentationPredictions(imagePath, predictions, &error);
            predictionCount = predictions.size();
        } else {
            const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionOnnxRuntime(checkpointPath, imagePath, options, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            for (const aitrain::DetectionPrediction& prediction : predictions) {
                predictionArray.append(aitrain::detectionPredictionToJson(prediction));
            }
            overlay = aitrain::renderDetectionPredictions(imagePath, predictions, &error);
            predictionCount = predictions.size();
        }
    } else if (ncnnModel) {
        const QString modelFamily = aitrain::inferNcnnModelFamily(checkpointPath);
        if (modelFamily == QStringLiteral("yolo_segmentation")) {
            taskType = QStringLiteral("segmentation");
            const QVector<aitrain::SegmentationPrediction> predictions = aitrain::predictSegmentationNcnnRuntime(checkpointPath, imagePath, options, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            for (const aitrain::SegmentationPrediction& prediction : predictions) {
                predictionArray.append(aitrain::segmentationPredictionToJson(prediction));
            }
            overlay = aitrain::renderSegmentationPredictions(imagePath, predictions, &error);
            predictionCount = predictions.size();
        } else {
            const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionNcnnRuntime(checkpointPath, imagePath, options, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            for (const aitrain::DetectionPrediction& prediction : predictions) {
                predictionArray.append(aitrain::detectionPredictionToJson(prediction));
            }
            overlay = aitrain::renderDetectionPredictions(imagePath, predictions, &error);
            predictionCount = predictions.size();
        }
    } else if (tensorRtModel) {
        if (!aitrain::isTensorRtInferenceAvailable()) {
            fail(QStringLiteral("TensorRT single-image inference is not enabled in this build: %1").arg(aitrain::tensorRtBackendStatus().message));
            return;
        }
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionTensorRt(checkpointPath, imagePath, options, &error);
        if (!error.isEmpty()) {
            fail(error);
            return;
        }
        for (const aitrain::DetectionPrediction& prediction : predictions) {
            predictionArray.append(aitrain::detectionPredictionToJson(prediction));
        }
        overlay = aitrain::renderDetectionPredictions(imagePath, predictions, &error);
        predictionCount = predictions.size();
    } else {
        fail(QStringLiteral("Unsupported inference model format: %1. Production inference requires official ONNX, NCNN .param, or TensorRT artifacts.").arg(checkpointPath));
        return;
    }
    if (overlay.isNull()) {
        fail(error);
        return;
    }

    const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
    QFile predictionsFile(predictionsPath);
    if (!predictionsFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        fail(QStringLiteral("Cannot write inference predictions: %1").arg(predictionsPath));
        return;
    }
    QJsonObject predictionsDocument;
    predictionsDocument.insert(QStringLiteral("taskId"), taskId);
    predictionsDocument.insert(QStringLiteral("checkpointPath"), checkpointPath);
    predictionsDocument.insert(QStringLiteral("imagePath"), imagePath);
    predictionsDocument.insert(QStringLiteral("taskType"), taskType);
    predictionsDocument.insert(QStringLiteral("runtime"), onnxModel
        ? QStringLiteral("onnxruntime")
        : (ncnnModel ? QStringLiteral("ncnn") : (tensorRtModel ? QStringLiteral("tensorrt") : QStringLiteral("unsupported"))));
    predictionsDocument.insert(QStringLiteral("elapsedMs"), static_cast<int>(elapsed.elapsed()));
    predictionsDocument.insert(QStringLiteral("postprocess"), QJsonObject{
        {QStringLiteral("confidenceThreshold"), options.confidenceThreshold},
        {QStringLiteral("iouThreshold"), options.iouThreshold},
        {QStringLiteral("maxDetections"), options.maxDetections}
    });
    predictionsDocument.insert(QStringLiteral("predictions"), predictionArray);
    predictionsFile.write(QJsonDocument(predictionsDocument).toJson(QJsonDocument::Indented));
    predictionsFile.close();

    QJsonObject renderLog;
    renderLog.insert(QStringLiteral("message"), QStringLiteral("Rendering inference overlay."));
    send(wp::event::log(), renderLog);
    QJsonObject saveLog;
    saveLog.insert(QStringLiteral("message"), QStringLiteral("Saving inference overlay."));
    send(wp::event::log(), saveLog);
    const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
    if (!overlay.save(overlayPath)) {
        fail(QStringLiteral("Cannot write inference overlay: %1").arg(overlayPath));
        return;
    }
    const int elapsedMs = static_cast<int>(elapsed.elapsed());

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("推理完成。"));
    send(wp::event::progress(), progressPayload);

    QJsonObject predictionsArtifact;
    predictionsArtifact.insert(QStringLiteral("taskId"), taskId);
    predictionsArtifact.insert(QStringLiteral("kind"), QStringLiteral("inference_predictions"));
    predictionsArtifact.insert(QStringLiteral("path"), predictionsPath);
    predictionsArtifact.insert(QStringLiteral("message"), QStringLiteral("Inference predictions"));
    send(wp::event::artifact(), predictionsArtifact);

    QJsonObject overlayArtifact;
    overlayArtifact.insert(QStringLiteral("taskId"), taskId);
    overlayArtifact.insert(QStringLiteral("kind"), QStringLiteral("inference_overlay"));
    overlayArtifact.insert(QStringLiteral("path"), overlayPath);
    overlayArtifact.insert(QStringLiteral("message"), QStringLiteral("Inference overlay"));
    send(wp::event::artifact(), overlayArtifact);

    QJsonObject response;
    response.insert(QStringLiteral("ok"), true);
    response.insert(QStringLiteral("taskId"), taskId);
    response.insert(QStringLiteral("checkpointPath"), checkpointPath);
    response.insert(QStringLiteral("imagePath"), imagePath);
    response.insert(QStringLiteral("taskType"), taskType);
    response.insert(QStringLiteral("predictionsPath"), predictionsPath);
    response.insert(QStringLiteral("overlayPath"), overlayPath);
    response.insert(QStringLiteral("elapsedMs"), elapsedMs);
    response.insert(QStringLiteral("predictionCount"), predictionCount);
    response.insert(QStringLiteral("finishedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(wp::event::inferenceResult(), response);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Inference completed"));
    running_ = false;
    send(wp::event::completed(), completed);
    finishSession();
}
