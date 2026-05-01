#include "WorkerSession.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/OcrRecTrainer.h"
#include "aitrain/core/SegmentationTrainer.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QStandardPaths>
#include <QThread>

namespace {

QJsonObject checkObject(const QString& name, const QString& status, const QString& message, const QJsonObject& details = {})
{
    QJsonObject object;
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("status"), status);
    object.insert(QStringLiteral("message"), message);
    object.insert(QStringLiteral("details"), details);
    return object;
}

QJsonObject nvidiaSmiCheck()
{
    QProcess process;
    process.start(QStringLiteral("nvidia-smi"),
        QStringList() << QStringLiteral("--query-gpu=name,memory.total")
                      << QStringLiteral("--format=csv,noheader"));
    if (!process.waitForStarted(1500)) {
        return checkObject(QStringLiteral("NVIDIA Driver"), QStringLiteral("missing"), QStringLiteral("未找到 nvidia-smi，可能未安装 NVIDIA 驱动。"));
    }
    if (!process.waitForFinished(2500)) {
        process.kill();
        process.waitForFinished();
        return checkObject(QStringLiteral("NVIDIA Driver"), QStringLiteral("warning"), QStringLiteral("nvidia-smi 执行超时。"));
    }

    const QString output = QString::fromLocal8Bit(process.readAllStandardOutput()).trimmed();
    const QString errorOutput = QString::fromLocal8Bit(process.readAllStandardError()).trimmed();
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0 || output.isEmpty()) {
        return checkObject(QStringLiteral("NVIDIA Driver"), QStringLiteral("missing"),
            errorOutput.isEmpty() ? QStringLiteral("nvidia-smi 未返回 GPU 信息。") : errorOutput);
    }

    QJsonObject details;
    details.insert(QStringLiteral("raw"), output);
    return checkObject(QStringLiteral("NVIDIA Driver"), QStringLiteral("ok"), QStringLiteral("检测到 NVIDIA GPU：%1").arg(output.split(QLatin1Char('\n')).first()), details);
}

QString firstUsablePythonExecutable(const QJsonObject& parameters = {})
{
    QStringList candidates;
    const QString requested = parameters.value(QStringLiteral("pythonExecutable")).toString().trimmed();
    if (!requested.isEmpty()) {
        candidates.append(requested);
    }
    const QString envRequested = QString::fromLocal8Bit(qgetenv("AITRAIN_PYTHON_EXECUTABLE")).trimmed();
    if (!envRequested.isEmpty()) {
        candidates.append(envRequested);
    }

    const QString applicationDir = QCoreApplication::applicationDirPath();
    candidates.append(QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/python-3.13.13-embed-amd64/python.exe")));
    candidates.append(QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/python-3.13.13-embed-amd64/python.exe")));
    candidates.append(QDir::current().absoluteFilePath(QStringLiteral(".deps/python-3.13.13-embed-amd64/python.exe")));
    candidates.append(QStandardPaths::findExecutable(QStringLiteral("python")));
    candidates.append(QStandardPaths::findExecutable(QStringLiteral("python3")));

    QStringList seen;
    for (const QString& candidate : candidates) {
        if (candidate.trimmed().isEmpty() || seen.contains(candidate)) {
            continue;
        }
        seen.append(candidate);
        QProcess process;
        process.start(candidate, QStringList() << QStringLiteral("--version"));
        if (!process.waitForStarted(1500)) {
            continue;
        }
        if (!process.waitForFinished(2500)) {
            process.kill();
            process.waitForFinished();
            continue;
        }
        if (process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0) {
            return candidate;
        }
    }
    return {};
}

QString pythonTrainerScriptFileForBackend(const QString& backend)
{
    const QString normalized = backend.trimmed().toLower();
    if (normalized == QStringLiteral("ultralytics_yolo") || normalized == QStringLiteral("ultralytics_yolo_detect")) {
        return QStringLiteral("python_trainers/detection/ultralytics_trainer.py");
    }
    if (normalized == QStringLiteral("ultralytics_yolo_segment")) {
        return QStringLiteral("python_trainers/segmentation/ultralytics_trainer.py");
    }
    if (normalized == QStringLiteral("paddleocr_rec")) {
        return QStringLiteral("python_trainers/ocr_rec/paddleocr_trainer.py");
    }
    if (normalized == QStringLiteral("paddleocr_rec_official") || normalized == QStringLiteral("paddleocr_ppocrv4_rec")) {
        return QStringLiteral("python_trainers/ocr_rec/paddleocr_official_adapter.py");
    }
    return QStringLiteral("python_trainers/mock_trainer.py");
}

QString pythonTrainerScriptPath(const QJsonObject& parameters, const QString& backend)
{
    const QString requested = parameters.value(QStringLiteral("pythonTrainerScript")).toString().trimmed();
    if (!requested.isEmpty()) {
        return QFileInfo(requested).absoluteFilePath();
    }
    const QString envRequested = QString::fromLocal8Bit(qgetenv("AITRAIN_PYTHON_TRAINER_SCRIPT")).trimmed();
    if (!envRequested.isEmpty()) {
        return QFileInfo(envRequested).absoluteFilePath();
    }

    const QString trainerFile = pythonTrainerScriptFileForBackend(backend);
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(applicationDir).absoluteFilePath(trainerFile),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../%1").arg(trainerFile)),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../%1").arg(trainerFile)),
        QDir::current().absoluteFilePath(trainerFile)
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return candidates.first();
}

QString requestedTrainingBackend(const aitrain::TrainingRequest& request)
{
    const QString backend = request.parameters.value(QStringLiteral("trainingBackend")).toString().trimmed();
    return backend.isEmpty() ? QStringLiteral("tiny_linear_detector") : backend;
}

bool isPythonTrainingBackendId(const QString& backend, const QJsonObject& parameters)
{
    const QString normalized = backend.trimmed().toLower();
    return parameters.contains(QStringLiteral("pythonTrainerScript"))
        || normalized == QStringLiteral("python_mock")
        || normalized == QStringLiteral("python_trainer_mock")
        || normalized == QStringLiteral("ultralytics_yolo")
        || normalized == QStringLiteral("ultralytics_yolo_detect")
        || normalized == QStringLiteral("ultralytics_yolo_segment")
        || normalized == QStringLiteral("paddleocr_rec")
        || normalized == QStringLiteral("paddleocr_rec_official")
        || normalized == QStringLiteral("paddleocr_ppocrv4_rec");
}

QJsonObject runPythonCommandCheck(
    const QString& name,
    const QString& executable,
    const QStringList& arguments,
    int timeoutMs,
    const QString& missingMessage)
{
    if (executable.isEmpty()) {
        return checkObject(name, QStringLiteral("missing"), missingMessage);
    }

    QProcess process;
    process.start(executable, arguments);
    if (!process.waitForStarted(1500)) {
        return checkObject(name, QStringLiteral("missing"), QStringLiteral("%1: %2").arg(missingMessage, process.errorString()));
    }
    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        process.waitForFinished();
        return checkObject(name, QStringLiteral("warning"), QStringLiteral("%1 check timed out.").arg(name));
    }

    const QString stdoutText = QString::fromUtf8(process.readAllStandardOutput()).trimmed();
    const QString stderrText = QString::fromUtf8(process.readAllStandardError()).trimmed();
    QJsonObject details;
    details.insert(QStringLiteral("stdout"), stdoutText);
    details.insert(QStringLiteral("stderr"), stderrText);
    details.insert(QStringLiteral("executable"), executable);
    if (process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0) {
        return checkObject(name, QStringLiteral("ok"), stdoutText.isEmpty() ? QStringLiteral("%1 is available.").arg(name) : stdoutText, details);
    }
    return checkObject(name, QStringLiteral("missing"), stderrText.isEmpty() ? missingMessage : stderrText, details);
}

QJsonObject pythonModuleCheck(const QString& executable, const QString& displayName, const QString& moduleName, const QString& missingMessage)
{
    return runPythonCommandCheck(
        displayName,
        executable,
        QStringList()
            << QStringLiteral("-c")
            << QStringLiteral("import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('%1') else 3)").arg(moduleName),
        5000,
        missingMessage);
}

} // namespace

WorkerSession::WorkerSession(QObject* parent)
    : QObject(parent)
{
    connect(&socket_, &QLocalSocket::readyRead, this, &WorkerSession::readLines);
    connect(&socket_, &QLocalSocket::disconnected, qApp, &QCoreApplication::quit);
    connect(&timer_, &QTimer::timeout, this, &WorkerSession::tickTraining);
    timer_.setInterval(500);
}

bool WorkerSession::connectToServer(const QString& serverName)
{
    socket_.connectToServer(serverName);
    return socket_.waitForConnected(5000);
}

void WorkerSession::readLines()
{
    buffer_.append(socket_.readAll());
    int newline = buffer_.indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer_.left(newline);
        buffer_.remove(0, newline + 1);

        QString type;
        QJsonObject payload;
        QString requestId;
        QString error;
        if (aitrain::protocol::decodeMessage(line, &type, &payload, &requestId, &error)) {
            handleMessage(type, payload);
        } else {
            fail(QStringLiteral("Invalid protocol message: %1").arg(error));
        }

        newline = buffer_.indexOf('\n');
    }
}

void WorkerSession::tickTraining()
{
    if (!running_) {
        return;
    }

    ++step_;
    const int epoch = qMax(1, step_ / 2);
    const double progress = static_cast<double>(step_) / static_cast<double>(maxSteps_);
    const double loss = qMax(0.05, 1.2 * (1.0 - progress) + QRandomGenerator::global()->bounded(25) / 1000.0);
    const double quality = qMin(0.98, 0.35 + progress * 0.55 + QRandomGenerator::global()->bounded(20) / 1000.0);

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
    progressPayload.insert(QStringLiteral("percent"), qRound(progress * 100.0));
    progressPayload.insert(QStringLiteral("step"), step_);
    progressPayload.insert(QStringLiteral("epoch"), epoch);
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject lossPayload;
    lossPayload.insert(QStringLiteral("taskId"), request_.taskId);
    lossPayload.insert(QStringLiteral("name"), QStringLiteral("loss"));
    lossPayload.insert(QStringLiteral("value"), loss);
    lossPayload.insert(QStringLiteral("step"), step_);
    lossPayload.insert(QStringLiteral("epoch"), epoch);
    send(QStringLiteral("metric"), lossPayload);

    QJsonObject mapPayload;
    mapPayload.insert(QStringLiteral("taskId"), request_.taskId);
    mapPayload.insert(QStringLiteral("name"), request_.taskType.contains(QStringLiteral("ocr"), Qt::CaseInsensitive) ? QStringLiteral("accuracy") : QStringLiteral("mAP50"));
    mapPayload.insert(QStringLiteral("value"), quality);
    mapPayload.insert(QStringLiteral("step"), step_);
    mapPayload.insert(QStringLiteral("epoch"), epoch);
    send(QStringLiteral("metric"), mapPayload);

    QJsonObject logPayload;
    logPayload.insert(QStringLiteral("message"), QStringLiteral("epoch=%1 step=%2 loss=%3 score=%4")
        .arg(epoch)
        .arg(step_)
        .arg(loss, 0, 'f', 4)
        .arg(quality, 0, 'f', 4));
    send(QStringLiteral("log"), logPayload);

    if (step_ >= maxSteps_) {
        complete();
    }
}

void WorkerSession::handleMessage(const QString& type, const QJsonObject& payload)
{
    if (type == QStringLiteral("startTrain")) {
        startTraining(aitrain::TrainingRequest::fromJson(payload));
    } else if (type == QStringLiteral("pause")) {
        pauseTraining();
    } else if (type == QStringLiteral("resume")) {
        resumeTraining();
    } else if (type == QStringLiteral("heartbeat")) {
        sendHeartbeat();
    } else if (type == QStringLiteral("environmentCheck")) {
        runEnvironmentCheck(payload);
    } else if (type == QStringLiteral("validateDataset")) {
        validateDataset(payload);
    } else if (type == QStringLiteral("splitDataset")) {
        splitDataset(payload);
    } else if (type == QStringLiteral("exportModel")) {
        exportModel(payload);
    } else if (type == QStringLiteral("infer")) {
        runInference(payload);
    } else if (type == QStringLiteral("cancel")) {
        running_ = false;
        paused_ = false;
        canceled_ = true;
        timer_.stop();
        if (pythonTrainerProcess_.state() != QProcess::NotRunning) {
            pythonTrainerProcess_.terminate();
            if (!pythonTrainerProcess_.waitForFinished(1500)) {
                pythonTrainerProcess_.kill();
                pythonTrainerProcess_.waitForFinished(1500);
            }
        }
        QJsonObject payloadObject;
        payloadObject.insert(QStringLiteral("taskId"), request_.taskId);
        payloadObject.insert(QStringLiteral("message"), QStringLiteral("Canceled by user"));
        send(QStringLiteral("canceled"), payloadObject);
        qApp->quit();
    } else {
        fail(QStringLiteral("Unsupported command: %1").arg(type));
    }
}

void WorkerSession::startTraining(const aitrain::TrainingRequest& request)
{
    request_ = request;
    step_ = 0;
    maxSteps_ = qMax(4, request.parameters.value(QStringLiteral("epochs")).toInt(20));
    running_ = true;
    paused_ = false;
    canceled_ = false;

    QDir().mkpath(request_.outputPath);
    QFile configFile(QDir(request_.outputPath).filePath(QStringLiteral("request.json")));
    if (configFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        configFile.write(QJsonDocument(request_.toJson()).toJson(QJsonDocument::Indented));
    }

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Worker accepted task %1 for plugin %2").arg(request_.taskId, request_.pluginId));
    send(QStringLiteral("log"), payload);

    if (request_.taskType.compare(QStringLiteral("detection"), Qt::CaseInsensitive) == 0) {
        runDetectionTraining();
        return;
    }
    if (request_.taskType.compare(QStringLiteral("segmentation"), Qt::CaseInsensitive) == 0) {
        runSegmentationTraining();
        return;
    }
    if (request_.taskType.compare(QStringLiteral("ocr_recognition"), Qt::CaseInsensitive) == 0
        || request_.taskType.compare(QStringLiteral("ocr"), Qt::CaseInsensitive) == 0) {
        runOcrRecTraining();
        return;
    }

    timer_.start();
}

void WorkerSession::pauseTraining()
{
    if (!running_) {
        fail(QStringLiteral("Cannot pause because no training task is running"));
        return;
    }

    running_ = false;
    paused_ = true;
    timer_.stop();

    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), request_.taskId);
    payload.insert(QStringLiteral("message"), QStringLiteral("Training task paused"));
    send(QStringLiteral("paused"), payload);
}

void WorkerSession::resumeTraining()
{
    if (!paused_) {
        fail(QStringLiteral("Cannot resume because no training task is paused"));
        return;
    }

    paused_ = false;
    running_ = true;
    timer_.start();

    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), request_.taskId);
    payload.insert(QStringLiteral("message"), QStringLiteral("Training task resumed"));
    send(QStringLiteral("resumed"), payload);
}

void WorkerSession::sendHeartbeat()
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), request_.taskId);
    payload.insert(QStringLiteral("running"), running_);
    payload.insert(QStringLiteral("paused"), paused_);
    payload.insert(QStringLiteral("step"), step_);
    payload.insert(QStringLiteral("timestamp"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(QStringLiteral("heartbeat"), payload);
}

void WorkerSession::runEnvironmentCheck(const QJsonObject& payload)
{
    Q_UNUSED(payload)

    QJsonArray checks;
    checks.append(nvidiaSmiCheck());
    const QVector<aitrain::RuntimeDependencyCheck> runtimeChecks =
        aitrain::defaultRuntimeDependencyChecks(QCoreApplication::applicationDirPath());
    for (const aitrain::RuntimeDependencyCheck& check : runtimeChecks) {
        checks.append(check.toJson());
    }
    const QString pythonExecutable = firstUsablePythonExecutable();
    checks.append(runPythonCommandCheck(
        QStringLiteral("Python"),
        pythonExecutable,
        QStringList() << QStringLiteral("--version"),
        5000,
        QStringLiteral("Python executable is not available. Configure pythonExecutable or install Python before using official Python trainers.")));
    checks.append(pythonModuleCheck(
        pythonExecutable,
        QStringLiteral("Ultralytics YOLO"),
        QStringLiteral("ultralytics"),
        QStringLiteral("Ultralytics is not installed. The official YOLO detection/segmentation trainer backend will be unavailable.")));
    checks.append(pythonModuleCheck(
        pythonExecutable,
        QStringLiteral("PaddleOCR"),
        QStringLiteral("paddleocr"),
        QStringLiteral("PaddleOCR is not installed. The official OCR recognition trainer backend will be unavailable.")));
    checks.append(pythonModuleCheck(
        pythonExecutable,
        QStringLiteral("PaddlePaddle"),
        QStringLiteral("paddle"),
        QStringLiteral("PaddlePaddle is not installed. PaddleOCR training will be unavailable.")));
    checks.append(checkObject(QStringLiteral("Worker"), QStringLiteral("ok"), QStringLiteral("Worker 环境自检命令可用。")));

    QJsonObject result;
    result.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    result.insert(QStringLiteral("checks"), checks);
    send(QStringLiteral("environmentCheck"), result);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::validateDataset(const QJsonObject& payload)
{
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始校验数据集。"));
    send(QStringLiteral("progress"), startProgress);

    aitrain::DatasetValidationResult result;
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt")) {
        result = aitrain::validateYoloDetectionDataset(datasetPath, options);
    } else if (format == QStringLiteral("yolo_segmentation")) {
        result = aitrain::validateYoloSegmentationDataset(datasetPath, options);
    } else if (format == QStringLiteral("paddleocr_rec")) {
        result = aitrain::validatePaddleOcrRecDataset(datasetPath, options);
    } else {
        result.ok = false;
        aitrain::DatasetValidationResult::Issue issue;
        issue.severity = QStringLiteral("error");
        issue.code = QStringLiteral("unsupported_format");
        issue.filePath = datasetPath;
        issue.message = QStringLiteral("Worker 不支持该数据集格式：%1。").arg(format);
        result.issues.append(issue);
        result.errors.append(issue.message);
    }

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("数据集校验完成。"));
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject response = result.toJson();
    response.insert(QStringLiteral("datasetPath"), datasetPath);
    response.insert(QStringLiteral("format"), format);
    response.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(QStringLiteral("datasetValidation"), response);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::splitDataset(const QJsonObject& payload)
{
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始划分数据集。"));
    send(QStringLiteral("progress"), startProgress);

    aitrain::DatasetSplitResult result;
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt")) {
        result = aitrain::splitYoloDetectionDataset(datasetPath, outputPath, options);
    } else {
        result.ok = false;
        result.outputPath = outputPath;
        result.errors.append(QStringLiteral("当前仅支持 YOLO 检测数据集划分。"));
    }

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("数据集划分完成。"));
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject response = result.toJson();
    response.insert(QStringLiteral("datasetPath"), datasetPath);
    response.insert(QStringLiteral("format"), format);
    response.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(QStringLiteral("datasetSplit"), response);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::exportModel(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString checkpointPath = payload.value(QStringLiteral("checkpointPath")).toString();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString(QStringLiteral("tiny_detector_json"));

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始导出模型。"));
    send(QStringLiteral("progress"), startProgress);

    const aitrain::DetectionExportResult result = aitrain::exportDetectionCheckpoint(checkpointPath, outputPath, format);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("模型导出完成。"));
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("export"));
    artifact.insert(QStringLiteral("path"), result.exportPath);
    artifact.insert(QStringLiteral("message"), result.format == QStringLiteral("onnx")
        ? QStringLiteral("Tiny detector ONNX model core export")
        : QStringLiteral("Tiny detector JSON scaffold export"));
    send(QStringLiteral("artifact"), artifact);

    QJsonObject response;
    response.insert(QStringLiteral("ok"), true);
    response.insert(QStringLiteral("format"), result.format);
    response.insert(QStringLiteral("taskId"), taskId);
    response.insert(QStringLiteral("checkpointPath"), checkpointPath);
    response.insert(QStringLiteral("exportPath"), result.exportPath);
    response.insert(QStringLiteral("reportPath"), result.reportPath);
    response.insert(QStringLiteral("config"), result.config);
    response.insert(QStringLiteral("exportedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(QStringLiteral("modelExport"), response);

    QJsonObject completed;
    completed.insert(QStringLiteral("message"), QStringLiteral("Model export completed"));
    send(QStringLiteral("completed"), completed);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::runInference(const QJsonObject& payload)
{
    const QString checkpointPath = payload.value(QStringLiteral("checkpointPath")).toString();
    const QString imagePath = payload.value(QStringLiteral("imagePath")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    aitrain::DetectionInferenceOptions options;
    options.confidenceThreshold = payload.value(QStringLiteral("confidenceThreshold")).toDouble(options.confidenceThreshold);
    options.iouThreshold = payload.value(QStringLiteral("iouThreshold")).toDouble(options.iouThreshold);
    options.maxDetections = payload.value(QStringLiteral("maxDetections")).toInt(options.maxDetections);
    if (outputPath.isEmpty()) {
        outputPath = QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("inference"));
    }
    if (!QDir().mkpath(outputPath)) {
        fail(QStringLiteral("Cannot create inference output directory: %1").arg(outputPath));
        return;
    }

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始推理。"));
    send(QStringLiteral("progress"), startProgress);

    QElapsedTimer elapsed;
    elapsed.start();
    QString error;
    QJsonArray predictionArray;
    QImage overlay;
    QString taskType = QStringLiteral("detection");
    int predictionCount = 0;
    const QString modelSuffix = QFileInfo(checkpointPath).suffix().toLower();
    const bool onnxModel = modelSuffix == QStringLiteral("onnx");
    const bool tensorRtModel = modelSuffix == QStringLiteral("engine") || modelSuffix == QStringLiteral("plan");
    if (onnxModel) {
        const QString modelFamily = aitrain::inferOnnxModelFamily(checkpointPath);
        if (modelFamily == QStringLiteral("yolo_segmentation")) {
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
        } else if (modelFamily == QStringLiteral("ocr_recognition")) {
            taskType = QStringLiteral("ocr_recognition");
            const aitrain::OcrRecPrediction prediction = aitrain::predictOcrRecOnnxRuntime(checkpointPath, imagePath, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            predictionArray.append(aitrain::ocrRecPredictionToJson(prediction));
            overlay = aitrain::renderOcrRecPrediction(imagePath, prediction, &error);
            predictionCount = 1;
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
    } else if (tensorRtModel) {
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
        aitrain::DetectionBaselineCheckpoint checkpoint;
        if (!aitrain::loadDetectionBaselineCheckpoint(checkpointPath, &checkpoint, &error)) {
            fail(error);
            return;
        }
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(checkpoint, imagePath, options, &error);
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
    predictionsDocument.insert(QStringLiteral("checkpointPath"), checkpointPath);
    predictionsDocument.insert(QStringLiteral("imagePath"), imagePath);
    predictionsDocument.insert(QStringLiteral("taskType"), taskType);
    predictionsDocument.insert(QStringLiteral("runtime"), onnxModel
        ? QStringLiteral("onnxruntime")
        : (tensorRtModel ? QStringLiteral("tensorrt") : QStringLiteral("tiny_detector")));
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
    send(QStringLiteral("log"), renderLog);
    QJsonObject saveLog;
    saveLog.insert(QStringLiteral("message"), QStringLiteral("Saving inference overlay."));
    send(QStringLiteral("log"), saveLog);
    const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
    if (!overlay.save(overlayPath)) {
        fail(QStringLiteral("Cannot write inference overlay: %1").arg(overlayPath));
        return;
    }
    const int elapsedMs = static_cast<int>(elapsed.elapsed());

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("推理完成。"));
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject predictionsArtifact;
    predictionsArtifact.insert(QStringLiteral("kind"), QStringLiteral("inference_predictions"));
    predictionsArtifact.insert(QStringLiteral("path"), predictionsPath);
    predictionsArtifact.insert(QStringLiteral("message"), QStringLiteral("Inference predictions"));
    send(QStringLiteral("artifact"), predictionsArtifact);

    QJsonObject overlayArtifact;
    overlayArtifact.insert(QStringLiteral("kind"), QStringLiteral("inference_overlay"));
    overlayArtifact.insert(QStringLiteral("path"), overlayPath);
    overlayArtifact.insert(QStringLiteral("message"), QStringLiteral("Inference overlay"));
    send(QStringLiteral("artifact"), overlayArtifact);

    QJsonObject response;
    response.insert(QStringLiteral("ok"), true);
    response.insert(QStringLiteral("checkpointPath"), checkpointPath);
    response.insert(QStringLiteral("imagePath"), imagePath);
    response.insert(QStringLiteral("taskType"), taskType);
    response.insert(QStringLiteral("predictionsPath"), predictionsPath);
    response.insert(QStringLiteral("overlayPath"), overlayPath);
    response.insert(QStringLiteral("elapsedMs"), elapsedMs);
    response.insert(QStringLiteral("predictionCount"), predictionCount);
    response.insert(QStringLiteral("finishedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(QStringLiteral("inferenceResult"), response);

    QJsonObject completed;
    completed.insert(QStringLiteral("message"), QStringLiteral("Inference completed"));
    send(QStringLiteral("completed"), completed);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

bool WorkerSession::shouldUsePythonTrainer() const
{
    return isPythonTrainingBackendId(requestedTrainingBackend(request_), request_.parameters);
}

void WorkerSession::runPythonTrainer()
{
    const QString backend = requestedTrainingBackend(request_);
    const QString pythonExecutable = firstUsablePythonExecutable(request_.parameters);
    if (pythonExecutable.isEmpty()) {
        fail(QStringLiteral("Python trainer backend '%1' requires a usable Python executable. Configure pythonExecutable or install Python.").arg(backend));
        return;
    }

    const QString trainerScript = pythonTrainerScriptPath(request_.parameters, backend);
    if (!QFileInfo::exists(trainerScript)) {
        fail(QStringLiteral("Python trainer script not found: %1").arg(trainerScript));
        return;
    }

    if (!QDir().mkpath(request_.outputPath)) {
        fail(QStringLiteral("Cannot create Python trainer output directory: %1").arg(request_.outputPath));
        return;
    }

    const QString requestPath = QDir(request_.outputPath).filePath(QStringLiteral("python_trainer_request.json"));
    QJsonObject trainerRequest;
    trainerRequest.insert(QStringLiteral("protocolVersion"), 1);
    trainerRequest.insert(QStringLiteral("taskId"), request_.taskId);
    trainerRequest.insert(QStringLiteral("taskType"), request_.taskType);
    trainerRequest.insert(QStringLiteral("datasetPath"), request_.datasetPath);
    trainerRequest.insert(QStringLiteral("outputPath"), request_.outputPath);
    trainerRequest.insert(QStringLiteral("backend"), backend);
    trainerRequest.insert(QStringLiteral("parameters"), request_.parameters);
    trainerRequest.insert(QStringLiteral("request"), request_.toJson());

    QFile requestFile(requestPath);
    if (!requestFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        fail(QStringLiteral("Cannot write Python trainer request: %1").arg(requestPath));
        return;
    }
    requestFile.write(QJsonDocument(trainerRequest).toJson(QJsonDocument::Indented));
    requestFile.close();

    QJsonObject adapterLog;
    adapterLog.insert(QStringLiteral("taskId"), request_.taskId);
    adapterLog.insert(QStringLiteral("message"), QStringLiteral("Launching Python trainer backend=%1").arg(backend));
    adapterLog.insert(QStringLiteral("backend"), backend);
    send(QStringLiteral("log"), adapterLog);

    pythonTrainerProcess_.setProgram(pythonExecutable);
    pythonTrainerProcess_.setArguments(QStringList() << trainerScript << QStringLiteral("--request") << requestPath);
    pythonTrainerProcess_.setProcessChannelMode(QProcess::SeparateChannels);
    pythonTrainerProcess_.start();
    if (!pythonTrainerProcess_.waitForStarted(5000)) {
        fail(QStringLiteral("Cannot start Python trainer: %1").arg(pythonTrainerProcess_.errorString()));
        return;
    }

    QByteArray stdoutBuffer;
    QByteArray stderrBuffer;
    bool terminalMessageSeen = false;
    while (pythonTrainerProcess_.state() != QProcess::NotRunning) {
        pythonTrainerProcess_.waitForReadyRead(50);
        drainPythonTrainerOutput(&stdoutBuffer, &terminalMessageSeen);
        drainPythonTrainerErrors(&stderrBuffer);
        if (pythonTrainerProcess_.state() != QProcess::NotRunning) {
            pythonTrainerProcess_.waitForFinished(1);
        }
        QCoreApplication::processEvents();
        if (canceled_) {
            return;
        }
    }
    drainPythonTrainerOutput(&stdoutBuffer, &terminalMessageSeen);
    drainPythonTrainerErrors(&stderrBuffer);
    if (!stdoutBuffer.trimmed().isEmpty()) {
        forwardPythonTrainerLine(stdoutBuffer.trimmed(), &terminalMessageSeen);
    }
    if (!stderrBuffer.trimmed().isEmpty()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("message"), QString::fromUtf8(stderrBuffer.trimmed()));
        payload.insert(QStringLiteral("backend"), backend);
        send(QStringLiteral("log"), payload);
    }

    if (canceled_) {
        return;
    }
    if (terminalMessageSeen) {
        socket_.waitForBytesWritten(1000);
        qApp->quit();
        return;
    }
    if (pythonTrainerProcess_.exitStatus() != QProcess::NormalExit || pythonTrainerProcess_.exitCode() != 0) {
        fail(QStringLiteral("Python trainer exited with code %1").arg(pythonTrainerProcess_.exitCode()));
        return;
    }
    fail(QStringLiteral("Python trainer exited without a completed or failed message"));
}

void WorkerSession::drainPythonTrainerOutput(QByteArray* buffer, bool* terminalMessageSeen)
{
    buffer->append(pythonTrainerProcess_.readAllStandardOutput());
    int newline = buffer->indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer->left(newline).trimmed();
        buffer->remove(0, newline + 1);
        if (!line.isEmpty()) {
            forwardPythonTrainerLine(line, terminalMessageSeen);
        }
        newline = buffer->indexOf('\n');
    }
}

void WorkerSession::drainPythonTrainerErrors(QByteArray* buffer)
{
    buffer->append(pythonTrainerProcess_.readAllStandardError());
    int newline = buffer->indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer->left(newline).trimmed();
        buffer->remove(0, newline + 1);
        if (!line.isEmpty()) {
            QJsonObject payload;
            payload.insert(QStringLiteral("taskId"), request_.taskId);
            payload.insert(QStringLiteral("message"), QString::fromUtf8(line));
            payload.insert(QStringLiteral("backend"), requestedTrainingBackend(request_));
            send(QStringLiteral("log"), payload);
        }
        newline = buffer->indexOf('\n');
    }
}

bool WorkerSession::forwardPythonTrainerLine(const QByteArray& line, bool* terminalMessageSeen)
{
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("message"), QString::fromUtf8(line));
        payload.insert(QStringLiteral("backend"), requestedTrainingBackend(request_));
        send(QStringLiteral("log"), payload);
        return true;
    }

    const QJsonObject object = document.object();
    const QString type = object.value(QStringLiteral("type")).toString();
    if (type.isEmpty()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("message"), QString::fromUtf8(line));
        payload.insert(QStringLiteral("backend"), requestedTrainingBackend(request_));
        send(QStringLiteral("log"), payload);
        return true;
    }

    QJsonObject payload = object.value(QStringLiteral("payload")).toObject();
    if (payload.isEmpty()) {
        payload = object;
        payload.remove(QStringLiteral("type"));
    }
    if (!payload.contains(QStringLiteral("taskId"))) {
        payload.insert(QStringLiteral("taskId"), request_.taskId);
    }
    if (!payload.contains(QStringLiteral("backend"))) {
        payload.insert(QStringLiteral("backend"), requestedTrainingBackend(request_));
    }

    send(type, payload);
    if (type == QStringLiteral("completed") || type == QStringLiteral("failed")) {
        running_ = false;
        paused_ = false;
        if (terminalMessageSeen) {
            *terminalMessageSeen = true;
        }
    }
    return true;
}

void WorkerSession::runDetectionTraining()
{
    if (shouldUsePythonTrainer()) {
        runPythonTrainer();
        return;
    }

    aitrain::DetectionTrainingOptions options;
    options.epochs = qMax(1, request_.parameters.value(QStringLiteral("epochs")).toInt(1));
    options.batchSize = qMax(1, request_.parameters.value(QStringLiteral("batchSize")).toInt(1));
    const int imageSize = qMax(1, request_.parameters.value(QStringLiteral("imageSize")).toInt(320));
    options.imageSize = QSize(imageSize, imageSize);
    options.gridSize = qBound(1, request_.parameters.value(QStringLiteral("gridSize")).toInt(4), 16);
    options.horizontalFlip = request_.parameters.value(QStringLiteral("horizontalFlip")).toBool(false);
    options.colorJitter = request_.parameters.value(QStringLiteral("colorJitter")).toBool(false);
    options.trainingBackend = request_.parameters.value(QStringLiteral("trainingBackend")).toString();
    options.resumeCheckpointPath = request_.parameters.value(QStringLiteral("resumeCheckpointPath")).toString();
    options.outputPath = request_.outputPath;

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), request_.taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("step"), 0);
    startProgress.insert(QStringLiteral("epoch"), 0);
    send(QStringLiteral("progress"), startProgress);

    QJsonObject logPayload;
    logPayload.insert(QStringLiteral("message"), QStringLiteral("Starting detection training with the tiny_linear_detector scaffold backend. Real YOLO-style LibTorch training is not implemented yet."));
    send(QStringLiteral("log"), logPayload);

    const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(
        request_.datasetPath,
        options,
        [this](const aitrain::DetectionTrainingMetrics& metrics) {
            QCoreApplication::processEvents();
            while (paused_) {
                QThread::msleep(50);
                QCoreApplication::processEvents();
            }
            if (!running_) {
                return false;
            }

            step_ = metrics.step;
            const int progressBase = qMax(1, metrics.totalSteps);
            const int percent = qMin(99, qMax(1, qRound(100.0 * static_cast<double>(metrics.step) / static_cast<double>(progressBase))));

            QJsonObject progressPayload;
            progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
            progressPayload.insert(QStringLiteral("percent"), percent);
            progressPayload.insert(QStringLiteral("step"), metrics.step);
            progressPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("progress"), progressPayload);

            QJsonObject lossPayload;
            lossPayload.insert(QStringLiteral("taskId"), request_.taskId);
            lossPayload.insert(QStringLiteral("name"), QStringLiteral("loss"));
            lossPayload.insert(QStringLiteral("value"), metrics.loss);
            lossPayload.insert(QStringLiteral("step"), metrics.step);
            lossPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("metric"), lossPayload);

            QJsonObject objectnessLossPayload;
            objectnessLossPayload.insert(QStringLiteral("taskId"), request_.taskId);
            objectnessLossPayload.insert(QStringLiteral("name"), QStringLiteral("objectnessLoss"));
            objectnessLossPayload.insert(QStringLiteral("value"), metrics.objectnessLoss);
            objectnessLossPayload.insert(QStringLiteral("step"), metrics.step);
            objectnessLossPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("metric"), objectnessLossPayload);

            QJsonObject classLossPayload;
            classLossPayload.insert(QStringLiteral("taskId"), request_.taskId);
            classLossPayload.insert(QStringLiteral("name"), QStringLiteral("classLoss"));
            classLossPayload.insert(QStringLiteral("value"), metrics.classLoss);
            classLossPayload.insert(QStringLiteral("step"), metrics.step);
            classLossPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("metric"), classLossPayload);

            QJsonObject boxLossPayload;
            boxLossPayload.insert(QStringLiteral("taskId"), request_.taskId);
            boxLossPayload.insert(QStringLiteral("name"), QStringLiteral("boxLoss"));
            boxLossPayload.insert(QStringLiteral("value"), metrics.boxLoss);
            boxLossPayload.insert(QStringLiteral("step"), metrics.step);
            boxLossPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("metric"), boxLossPayload);

            QJsonObject stepLogPayload;
            stepLogPayload.insert(QStringLiteral("message"), QStringLiteral("epoch=%1 step=%2 loss=%3 objLoss=%4 classLoss=%5 boxLoss=%6")
                .arg(metrics.epoch)
                .arg(metrics.step)
                .arg(metrics.loss, 0, 'f', 4)
                .arg(metrics.objectnessLoss, 0, 'f', 4)
                .arg(metrics.classLoss, 0, 'f', 4)
                .arg(metrics.boxLoss, 0, 'f', 4));
            send(QStringLiteral("log"), stepLogPayload);
            return true;
        });

    if (!result.ok) {
        if (canceled_) {
            socket_.waitForBytesWritten(1000);
            return;
        }
        fail(result.error);
        return;
    }

    running_ = false;
    paused_ = false;

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("step"), result.steps);
    progressPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), request_.taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    artifact.insert(QStringLiteral("path"), result.checkpointPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Tiny linear detector scaffold checkpoint"));
    artifact.insert(QStringLiteral("trainingBackend"), result.trainingBackend);
    artifact.insert(QStringLiteral("modelFamily"), result.modelFamily);
    artifact.insert(QStringLiteral("scaffold"), result.scaffold);
    send(QStringLiteral("artifact"), artifact);

    emitDetectionPreviewArtifacts(result.checkpointPath);

    QJsonObject precisionPayload;
    precisionPayload.insert(QStringLiteral("taskId"), request_.taskId);
    precisionPayload.insert(QStringLiteral("name"), QStringLiteral("precision"));
    precisionPayload.insert(QStringLiteral("value"), result.precision);
    precisionPayload.insert(QStringLiteral("step"), result.steps);
    precisionPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("metric"), precisionPayload);

    QJsonObject recallPayload;
    recallPayload.insert(QStringLiteral("taskId"), request_.taskId);
    recallPayload.insert(QStringLiteral("name"), QStringLiteral("recall"));
    recallPayload.insert(QStringLiteral("value"), result.recall);
    recallPayload.insert(QStringLiteral("step"), result.steps);
    recallPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("metric"), recallPayload);

    QJsonObject mapPayload;
    mapPayload.insert(QStringLiteral("taskId"), request_.taskId);
    mapPayload.insert(QStringLiteral("name"), QStringLiteral("mAP50"));
    mapPayload.insert(QStringLiteral("value"), result.map50);
    mapPayload.insert(QStringLiteral("step"), result.steps);
    mapPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("metric"), mapPayload);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Tiny linear detection training completed"));
    payload.insert(QStringLiteral("trainingBackend"), result.trainingBackend);
    payload.insert(QStringLiteral("modelFamily"), result.modelFamily);
    payload.insert(QStringLiteral("scaffold"), result.scaffold);
    send(QStringLiteral("completed"), payload);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::runSegmentationTraining()
{
    if (shouldUsePythonTrainer()) {
        runPythonTrainer();
        return;
    }

    aitrain::SegmentationTrainingOptions options;
    options.epochs = qMax(1, request_.parameters.value(QStringLiteral("epochs")).toInt(1));
    options.batchSize = qMax(1, request_.parameters.value(QStringLiteral("batchSize")).toInt(1));
    const int imageSize = qMax(1, request_.parameters.value(QStringLiteral("imageSize")).toInt(320));
    options.imageSize = QSize(imageSize, imageSize);
    options.learningRate = qMax(0.001, request_.parameters.value(QStringLiteral("learningRate")).toDouble(0.05));
    options.outputPath = request_.outputPath;

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), request_.taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("step"), 0);
    startProgress.insert(QStringLiteral("epoch"), 0);
    send(QStringLiteral("progress"), startProgress);

    QJsonObject logPayload;
    logPayload.insert(QStringLiteral("message"), QStringLiteral("Starting tiny mask segmentation scaffold. This is not full YOLO segmentation training."));
    send(QStringLiteral("log"), logPayload);

    const aitrain::SegmentationTrainingResult result = aitrain::trainSegmentationBaseline(
        request_.datasetPath,
        options,
        [this](const aitrain::SegmentationTrainingMetrics& metrics) {
            QCoreApplication::processEvents();
            while (paused_) {
                QThread::msleep(50);
                QCoreApplication::processEvents();
            }
            if (!running_) {
                return false;
            }

            step_ = metrics.step;
            const int progressBase = qMax(1, metrics.totalSteps);
            const int percent = qMin(99, qMax(1, qRound(100.0 * static_cast<double>(metrics.step) / static_cast<double>(progressBase))));

            QJsonObject progressPayload;
            progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
            progressPayload.insert(QStringLiteral("percent"), percent);
            progressPayload.insert(QStringLiteral("step"), metrics.step);
            progressPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("progress"), progressPayload);

            const auto sendMetric = [this, &metrics](const QString& name, double value) {
                QJsonObject payload;
                payload.insert(QStringLiteral("taskId"), request_.taskId);
                payload.insert(QStringLiteral("name"), name);
                payload.insert(QStringLiteral("value"), value);
                payload.insert(QStringLiteral("step"), metrics.step);
                payload.insert(QStringLiteral("epoch"), metrics.epoch);
                send(QStringLiteral("metric"), payload);
            };
            sendMetric(QStringLiteral("loss"), metrics.loss);
            sendMetric(QStringLiteral("maskLoss"), metrics.maskLoss);
            sendMetric(QStringLiteral("maskCoverage"), metrics.maskCoverage);
            sendMetric(QStringLiteral("maskIoU"), metrics.maskIou);
            sendMetric(QStringLiteral("precision"), metrics.precision);
            sendMetric(QStringLiteral("recall"), metrics.recall);
            sendMetric(QStringLiteral("segmentationMap50"), metrics.map50);

            QJsonObject stepLogPayload;
            stepLogPayload.insert(QStringLiteral("message"), QStringLiteral("epoch=%1 step=%2 maskLoss=%3 maskCoverage=%4 maskIoU=%5 segmentationMap50=%6")
                .arg(metrics.epoch)
                .arg(metrics.step)
                .arg(metrics.maskLoss, 0, 'f', 4)
                .arg(metrics.maskCoverage, 0, 'f', 4)
                .arg(metrics.maskIou, 0, 'f', 4)
                .arg(metrics.map50, 0, 'f', 4));
            send(QStringLiteral("log"), stepLogPayload);
            return true;
        });

    if (!result.ok) {
        if (canceled_) {
            socket_.waitForBytesWritten(1000);
            return;
        }
        fail(result.error);
        return;
    }

    running_ = false;
    paused_ = false;

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("step"), result.steps);
    progressPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject checkpointArtifact;
    checkpointArtifact.insert(QStringLiteral("taskId"), request_.taskId);
    checkpointArtifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    checkpointArtifact.insert(QStringLiteral("path"), result.checkpointPath);
    checkpointArtifact.insert(QStringLiteral("message"), QStringLiteral("Tiny mask segmentation scaffold checkpoint"));
    send(QStringLiteral("artifact"), checkpointArtifact);

    if (!result.previewPath.isEmpty()) {
        QJsonObject previewArtifact;
        previewArtifact.insert(QStringLiteral("taskId"), request_.taskId);
        previewArtifact.insert(QStringLiteral("kind"), QStringLiteral("preview"));
        previewArtifact.insert(QStringLiteral("path"), result.previewPath);
        previewArtifact.insert(QStringLiteral("message"), QStringLiteral("Tiny mask segmentation scaffold preview"));
        send(QStringLiteral("artifact"), previewArtifact);
    }

    if (!result.maskPreviewPath.isEmpty()) {
        QJsonObject maskPreviewArtifact;
        maskPreviewArtifact.insert(QStringLiteral("taskId"), request_.taskId);
        maskPreviewArtifact.insert(QStringLiteral("kind"), QStringLiteral("mask_preview"));
        maskPreviewArtifact.insert(QStringLiteral("path"), result.maskPreviewPath);
        maskPreviewArtifact.insert(QStringLiteral("message"), QStringLiteral("Tiny mask segmentation scaffold mask preview"));
        send(QStringLiteral("artifact"), maskPreviewArtifact);
    }

    const auto sendFinalMetric = [this, &result, &options](const QString& name, double value) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("name"), name);
        payload.insert(QStringLiteral("value"), value);
        payload.insert(QStringLiteral("step"), result.steps);
        payload.insert(QStringLiteral("epoch"), options.epochs);
        send(QStringLiteral("metric"), payload);
    };
    sendFinalMetric(QStringLiteral("maskLoss"), result.finalLoss);
    sendFinalMetric(QStringLiteral("maskCoverage"), result.maskCoverage);
    sendFinalMetric(QStringLiteral("maskIoU"), result.maskIou);
    sendFinalMetric(QStringLiteral("precision"), result.precision);
    sendFinalMetric(QStringLiteral("recall"), result.recall);
    sendFinalMetric(QStringLiteral("segmentationMap50"), result.map50);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Tiny mask segmentation scaffold completed"));
    send(QStringLiteral("completed"), payload);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::runOcrRecTraining()
{
    if (shouldUsePythonTrainer()) {
        runPythonTrainer();
        return;
    }

    aitrain::OcrRecTrainingOptions options;
    options.epochs = qMax(1, request_.parameters.value(QStringLiteral("epochs")).toInt(1));
    options.batchSize = qMax(1, request_.parameters.value(QStringLiteral("batchSize")).toInt(1));
    const int imageWidth = qMax(1, request_.parameters.value(QStringLiteral("imageWidth")).toInt(100));
    const int imageHeight = qMax(1, request_.parameters.value(QStringLiteral("imageHeight")).toInt(32));
    options.imageSize = QSize(imageWidth, imageHeight);
    options.learningRate = qMax(0.001, request_.parameters.value(QStringLiteral("learningRate")).toDouble(0.05));
    options.maxTextLength = qMax(1, request_.parameters.value(QStringLiteral("maxTextLength")).toInt(25));
    options.labelFilePath = request_.parameters.value(QStringLiteral("labelFilePath")).toString();
    options.dictionaryFilePath = request_.parameters.value(QStringLiteral("dictionaryFilePath")).toString();
    options.outputPath = request_.outputPath;

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), request_.taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("step"), 0);
    startProgress.insert(QStringLiteral("epoch"), 0);
    send(QStringLiteral("progress"), startProgress);

    QJsonObject logPayload;
    logPayload.insert(QStringLiteral("message"), QStringLiteral("Starting OCR recognition scaffold. This is not real CRNN/CTC OCR training."));
    send(QStringLiteral("log"), logPayload);

    const aitrain::OcrRecTrainingResult result = aitrain::trainOcrRecBaseline(
        request_.datasetPath,
        options,
        [this](const aitrain::OcrRecTrainingMetrics& metrics) {
            QCoreApplication::processEvents();
            while (paused_) {
                QThread::msleep(50);
                QCoreApplication::processEvents();
            }
            if (!running_) {
                return false;
            }

            step_ = metrics.step;
            const int progressBase = qMax(1, metrics.totalSteps);
            const int percent = qMin(99, qMax(1, qRound(100.0 * static_cast<double>(metrics.step) / static_cast<double>(progressBase))));

            QJsonObject progressPayload;
            progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
            progressPayload.insert(QStringLiteral("percent"), percent);
            progressPayload.insert(QStringLiteral("step"), metrics.step);
            progressPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("progress"), progressPayload);

            const auto sendMetric = [this, &metrics](const QString& name, double value) {
                QJsonObject payload;
                payload.insert(QStringLiteral("taskId"), request_.taskId);
                payload.insert(QStringLiteral("name"), name);
                payload.insert(QStringLiteral("value"), value);
                payload.insert(QStringLiteral("step"), metrics.step);
                payload.insert(QStringLiteral("epoch"), metrics.epoch);
                send(QStringLiteral("metric"), payload);
            };
            sendMetric(QStringLiteral("loss"), metrics.loss);
            sendMetric(QStringLiteral("ctcLoss"), metrics.ctcLoss);
            sendMetric(QStringLiteral("accuracy"), metrics.accuracy);
            sendMetric(QStringLiteral("editDistance"), metrics.editDistance);

            QJsonObject stepLogPayload;
            stepLogPayload.insert(QStringLiteral("message"), QStringLiteral("epoch=%1 step=%2 ctcLoss=%3 accuracy=%4 editDistance=%5")
                .arg(metrics.epoch)
                .arg(metrics.step)
                .arg(metrics.ctcLoss, 0, 'f', 4)
                .arg(metrics.accuracy, 0, 'f', 4)
                .arg(metrics.editDistance, 0, 'f', 4));
            send(QStringLiteral("log"), stepLogPayload);
            return true;
        });

    if (!result.ok) {
        if (canceled_) {
            socket_.waitForBytesWritten(1000);
            return;
        }
        fail(result.error);
        return;
    }

    running_ = false;
    paused_ = false;

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("step"), result.steps);
    progressPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject checkpointArtifact;
    checkpointArtifact.insert(QStringLiteral("taskId"), request_.taskId);
    checkpointArtifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    checkpointArtifact.insert(QStringLiteral("path"), result.checkpointPath);
    checkpointArtifact.insert(QStringLiteral("message"), QStringLiteral("OCR recognition scaffold checkpoint"));
    send(QStringLiteral("artifact"), checkpointArtifact);

    if (!result.previewPath.isEmpty()) {
        QJsonObject previewArtifact;
        previewArtifact.insert(QStringLiteral("taskId"), request_.taskId);
        previewArtifact.insert(QStringLiteral("kind"), QStringLiteral("preview"));
        previewArtifact.insert(QStringLiteral("path"), result.previewPath);
        previewArtifact.insert(QStringLiteral("message"), QStringLiteral("OCR recognition scaffold preview"));
        send(QStringLiteral("artifact"), previewArtifact);
    }

    const auto sendFinalMetric = [this, &result, &options](const QString& name, double value) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("name"), name);
        payload.insert(QStringLiteral("value"), value);
        payload.insert(QStringLiteral("step"), result.steps);
        payload.insert(QStringLiteral("epoch"), options.epochs);
        send(QStringLiteral("metric"), payload);
    };
    sendFinalMetric(QStringLiteral("ctcLoss"), result.finalLoss);
    sendFinalMetric(QStringLiteral("accuracy"), result.accuracy);
    sendFinalMetric(QStringLiteral("editDistance"), result.editDistance);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("OCR recognition scaffold completed"));
    send(QStringLiteral("completed"), payload);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::emitDetectionPreviewArtifacts(const QString& checkpointPath)
{
    QString error;
    aitrain::DetectionBaselineCheckpoint checkpoint;
    if (!aitrain::loadDetectionBaselineCheckpoint(checkpointPath, &checkpoint, &error)) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not load detection checkpoint for preview: %1").arg(error));
        send(QStringLiteral("log"), payload);
        return;
    }

    aitrain::DetectionDataset dataset;
    if (!dataset.load(request_.datasetPath, QStringLiteral("val"), &error)
        && !dataset.load(request_.datasetPath, QStringLiteral("train"), &error)) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not load detection dataset for preview: %1").arg(error));
        send(QStringLiteral("log"), payload);
        return;
    }
    if (dataset.samples().isEmpty()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not create detection preview because the dataset split is empty."));
        send(QStringLiteral("log"), payload);
        return;
    }

    const QString imagePath = dataset.samples().first().imagePath;
    const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(checkpoint, imagePath, &error);
    if (!error.isEmpty()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not create detection predictions: %1").arg(error));
        send(QStringLiteral("log"), payload);
        return;
    }

    QJsonArray predictionArray;
    for (const aitrain::DetectionPrediction& prediction : predictions) {
        predictionArray.append(aitrain::detectionPredictionToJson(prediction));
    }

    const QString predictionsPath = QDir(request_.outputPath).filePath(QStringLiteral("predictions_latest.json"));
    QFile predictionsFile(predictionsPath);
    if (predictionsFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QJsonObject document;
        document.insert(QStringLiteral("taskId"), request_.taskId);
        document.insert(QStringLiteral("checkpointPath"), checkpointPath);
        document.insert(QStringLiteral("imagePath"), imagePath);
        document.insert(QStringLiteral("predictions"), predictionArray);
        predictionsFile.write(QJsonDocument(document).toJson(QJsonDocument::Indented));
        predictionsFile.close();

        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), request_.taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("predictions"));
        artifact.insert(QStringLiteral("path"), predictionsPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Tiny linear detector predictions"));
        send(QStringLiteral("artifact"), artifact);
    } else {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not write detection predictions: %1").arg(predictionsPath));
        send(QStringLiteral("log"), payload);
    }

    const QImage preview = aitrain::renderDetectionPredictions(imagePath, predictions, &error);
    if (preview.isNull()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not render detection preview: %1").arg(error));
        send(QStringLiteral("log"), payload);
        return;
    }

    const QString previewPath = QDir(request_.outputPath).filePath(QStringLiteral("preview_latest.png"));
    if (preview.save(previewPath)) {
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), request_.taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("preview"));
        artifact.insert(QStringLiteral("path"), previewPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Tiny linear detector preview"));
        send(QStringLiteral("artifact"), artifact);
    } else {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not write detection preview: %1").arg(previewPath));
        send(QStringLiteral("log"), payload);
    }
}

void WorkerSession::send(const QString& type, const QJsonObject& payload)
{
    socket_.write(aitrain::protocol::encodeMessage(type, payload));
    socket_.flush();
}

void WorkerSession::fail(const QString& message)
{
    running_ = false;
    paused_ = false;
    timer_.stop();
    QJsonObject payload;
    payload.insert(QStringLiteral("message"), message);
    send(QStringLiteral("failed"), payload);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::complete()
{
    running_ = false;
    paused_ = false;
    timer_.stop();

    const QString checkpointPath = QDir(request_.outputPath).filePath(QStringLiteral("checkpoint_best.aitrain"));
    QFile checkpoint(checkpointPath);
    if (checkpoint.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QJsonObject content;
        content.insert(QStringLiteral("taskId"), request_.taskId);
        content.insert(QStringLiteral("pluginId"), request_.pluginId);
        content.insert(QStringLiteral("taskType"), request_.taskType);
        content.insert(QStringLiteral("createdAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
        content.insert(QStringLiteral("note"), QStringLiteral("Platform scaffold checkpoint. Replace with native LibTorch weights in production plugins."));
        checkpoint.write(QJsonDocument(content).toJson(QJsonDocument::Indented));
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), request_.taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    artifact.insert(QStringLiteral("path"), checkpointPath);
    send(QStringLiteral("artifact"), artifact);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Training workflow completed"));
    send(QStringLiteral("completed"), payload);
    qApp->quit();
}
