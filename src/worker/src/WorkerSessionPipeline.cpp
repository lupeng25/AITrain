#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/WorkerProtocol.h"

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

void WorkerSession::runLocalPipeline(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QDir::currentPath(), taskId);
    }
    const QString templateId = payload.value(QStringLiteral("templateId")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始执行本地流水线。"));
    send(wp::event::progress(), progress);
    if (pollPendingCancel()) {
        sendCanceledAndFinish(taskId, QStringLiteral("Canceled by user"));
        return;
    }

    QJsonObject pipelineOptions = options;
    const QString resolvedTemplate = templateId.isEmpty()
        ? QStringLiteral("train-evaluate-export-register")
        : templateId.trimmed().toLower();

    const QString pipelineTaskType = pipelineOptions.value(QStringLiteral("taskType")).toString(QStringLiteral("detection"));
    QString pipelineBackend = pipelineOptions.value(QStringLiteral("trainingBackend")).toString().trimmed().toLower();
    if (pipelineBackend.isEmpty()) {
        pipelineBackend = officialTrainingBackendForTask(pipelineTaskType);
        if (!pipelineBackend.isEmpty()) {
            pipelineOptions.insert(QStringLiteral("trainingBackend"), pipelineBackend);
        }
    }
    if (resolvedTemplate == QStringLiteral("train-evaluate-export-register") && pipelineBackend.isEmpty()) {
        failWithDetails(
            QStringLiteral("Pipeline task type '%1' does not have an official production training backend.").arg(pipelineTaskType),
            QStringLiteral("unsupported_training_backend"),
            QJsonObject{
                {QStringLiteral("taskType"), pipelineTaskType},
                {QStringLiteral("outputPath"), outputPath}});
        return;
    }
    const bool needsOfficialPipelineTraining =
        resolvedTemplate == QStringLiteral("train-evaluate-export-register")
        && !pipelineBackend.isEmpty();

    if (needsOfficialPipelineTraining) {
        if (!isSupportedTrainingBackendId(pipelineBackend, pipelineOptions)) {
            failWithDetails(
                QStringLiteral("Pipeline training backend '%1' is not enabled for production training.").arg(pipelineBackend),
                QStringLiteral("unsupported_training_backend"),
                QJsonObject{
                    {QStringLiteral("backend"), pipelineBackend},
                    {QStringLiteral("taskType"), pipelineTaskType},
                    {QStringLiteral("outputPath"), outputPath}});
            return;
        }
        const QString datasetPath = pipelineOptions.value(QStringLiteral("datasetPath")).toString();
        const QString trainOutputPath = QDir(outputPath).filePath(QStringLiteral("training"));
        const PipelineTrainResult trainingResult = runPipelineTrainingStep(
            taskId,
            trainOutputPath,
            pipelineOptions,
            datasetPath,
            pipelineTaskType);
        if (!trainingResult.ok) {
            fail(trainingResult.error.isEmpty()
                ? QStringLiteral("Local pipeline training step failed.")
                : trainingResult.error);
            return;
        }

        if (!trainingResult.onnxPath.isEmpty()) {
            pipelineOptions.insert(QStringLiteral("modelPath"), trainingResult.onnxPath);
            pipelineOptions.insert(QStringLiteral("checkpointPath"), trainingResult.onnxPath);
        } else if (!trainingResult.checkpointPath.isEmpty()) {
            pipelineOptions.insert(QStringLiteral("modelPath"), trainingResult.checkpointPath);
            pipelineOptions.insert(QStringLiteral("checkpointPath"), trainingResult.checkpointPath);
        }
        pipelineOptions.insert(QStringLiteral("pipelineOfficialTrainingCompleted"), true);
        pipelineOptions.insert(QStringLiteral("pipelineOfficialTrainingPayload"), trainingResult.completedPayload);
        pipelineOptions.insert(QStringLiteral("pipelineOfficialTrainingArtifacts"), trainingResult.artifacts);
        pipelineOptions.insert(QStringLiteral("pipelineOfficialTrainingMetrics"), trainingResult.metrics);
        pipelineOptions.insert(QStringLiteral("pipelineOfficialTrainingReportPath"), trainingResult.reportPath);
        pipelineOptions.insert(QStringLiteral("pipelineOfficialTrainingCheckpointPath"), trainingResult.checkpointPath);
        pipelineOptions.insert(QStringLiteral("pipelineOfficialTrainingOnnxPath"), trainingResult.onnxPath);
    }

    const aitrain::WorkflowResult result = aitrain::runLocalPipelinePlan(outputPath, templateId, pipelineOptions);
    if (!result.ok) {
        fail(result.error);
        return;
    }
    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("local_pipeline_execution"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Local pipeline execution report"));
    send(wp::event::artifact(), artifact);

    const QJsonArray artifacts = result.payload.value(QStringLiteral("artifacts")).toArray();
    for (const QJsonValue& value : artifacts) {
        const QJsonObject object = value.toObject();
        const QString path = object.value(QStringLiteral("path")).toString();
        if (path.isEmpty()) {
            continue;
        }
        QJsonObject extraArtifact;
        extraArtifact.insert(QStringLiteral("taskId"), taskId);
        extraArtifact.insert(QStringLiteral("kind"), object.value(QStringLiteral("kind")).toString(QStringLiteral("pipeline_artifact")));
        extraArtifact.insert(QStringLiteral("path"), path);
        extraArtifact.insert(QStringLiteral("message"), object.value(QStringLiteral("message")).toString(QStringLiteral("Pipeline artifact")));
        send(wp::event::artifact(), extraArtifact);
    }

    QJsonObject pipelinePayload = result.payload;
    pipelinePayload.insert(QStringLiteral("taskId"), taskId);

    const QString deliveryReportPath = pipelinePayload.value(QStringLiteral("deliveryReportPath")).toString();
    if (!deliveryReportPath.isEmpty()) {
        QJsonObject deliveryArtifact;
        deliveryArtifact.insert(QStringLiteral("taskId"), taskId);
        deliveryArtifact.insert(QStringLiteral("kind"), QStringLiteral("training_delivery_report"));
        deliveryArtifact.insert(QStringLiteral("path"), deliveryReportPath);
        deliveryArtifact.insert(QStringLiteral("message"), QStringLiteral("Pipeline delivery report"));
        send(wp::event::artifact(), deliveryArtifact);
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    const QString pipelineState = pipelinePayload.value(QStringLiteral("state")).toString(QStringLiteral("completed"));
    doneProgress.insert(QStringLiteral("message"), pipelineState == QStringLiteral("failed")
        ? QStringLiteral("本地流水线执行失败。")
        : QStringLiteral("本地流水线执行完成。"));
    send(wp::event::progress(), doneProgress);
    send(wp::event::pipelinePlan(), pipelinePayload);
    socket_.waitForBytesWritten(1000);
    QJsonObject terminal;
    terminal.insert(QStringLiteral("taskId"), taskId);
    if (pipelineState == QStringLiteral("failed")) {
        terminal.insert(QStringLiteral("message"),
            pipelinePayload.value(QStringLiteral("failureReason")).toString(QStringLiteral("Local pipeline execution failed")));
        send(wp::event::failed(), terminal);
    } else {
        terminal.insert(QStringLiteral("message"), QStringLiteral("Local pipeline execution completed"));
        send(wp::event::completed(), terminal);
    }
    finishSession();
}

WorkerSession::PipelineTrainResult WorkerSession::runPipelineTrainingStep(
    const QString& parentTaskId,
    const QString& outputPath,
    const QJsonObject& options,
    const QString& datasetPath,
    const QString& taskType)
{
    PipelineTrainResult result;

    request_ = aitrain::TrainingRequest();
    request_.taskId = parentTaskId.isEmpty() ? QStringLiteral("pipeline") : parentTaskId;
    request_.projectPath = QDir::currentPath();
    request_.pluginId = QStringLiteral("com.aitrain.workflow");
    request_.taskType = taskType;
    request_.datasetPath = datasetPath;
    request_.outputPath = outputPath;
    request_.parameters = options;

    const QString backend = requestedTrainingBackend(request_);
    if (!QDir().mkpath(request_.outputPath)) {
        result.error = QStringLiteral("Cannot create pipeline training output directory: %1").arg(request_.outputPath);
        return result;
    }

    if (!isSupportedTrainingBackendId(backend, request_.parameters)) {
        result.error = QStringLiteral("Pipeline training backend '%1' is not enabled for production training.").arg(backend);
        return result;
    }

    if (!shouldUsePythonTrainer()) {
        result.error = QStringLiteral("Pipeline official training step requires a Python trainer backend. Got: %1").arg(backend);
        return result;
    }

    const QString pythonExecutable = firstUsablePythonExecutable(request_.parameters);
    if (pythonExecutable.isEmpty()) {
        result.error = QStringLiteral("Python trainer backend '%1' requires a usable Python executable. Configure pythonExecutable or install Python.").arg(backend);
        return result;
    }

    const QString trainerScript = pythonTrainerScriptPath(request_.parameters, backend);
    if (!QFileInfo::exists(trainerScript)) {
        result.error = QStringLiteral("Python trainer script not found: %1").arg(trainerScript);
        return result;
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

    QString writeError;
    if (!writeJsonFile(requestPath, trainerRequest, &writeError)) {
        result.error = writeError;
        return result;
    }

    QJsonObject adapterLog;
    adapterLog.insert(QStringLiteral("taskId"), request_.taskId);
    adapterLog.insert(QStringLiteral("message"), QStringLiteral("Launching pipeline Python trainer backend=%1").arg(backend));
    adapterLog.insert(QStringLiteral("backend"), backend);
    send(wp::event::log(), adapterLog);

    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    pythonTrainerProcess_.setProcessEnvironment(environment);
    pythonTrainerProcess_.setProgram(pythonExecutable);
    pythonTrainerProcess_.setArguments(QStringList() << QStringLiteral("-u") << trainerScript << QStringLiteral("--request") << requestPath);
    pythonTrainerProcess_.setProcessChannelMode(QProcess::SeparateChannels);
    pythonTrainerProcess_.start();
    if (!pythonTrainerProcess_.waitForStarted(5000)) {
        result.error = QStringLiteral("Cannot start Python trainer: %1").arg(pythonTrainerProcess_.errorString());
        return result;
    }

    QByteArray stdoutBuffer;
    QByteArray stderrBuffer;
    bool terminalMessageSeen = false;
    interceptPythonTrainerMessages_ = true;
    while (pythonTrainerProcess_.state() != QProcess::NotRunning) {
        pythonTrainerProcess_.waitForReadyRead(50);
        drainPipelinePythonTrainerOutput(&stdoutBuffer, &result, &terminalMessageSeen);
        drainPipelinePythonTrainerErrors(&stderrBuffer, &result);

        QCoreApplication::processEvents();
        if (canceled_) {
            interceptPythonTrainerMessages_ = false;
            result.error = QStringLiteral("Pipeline training canceled.");
            return result;
        }
        if (pythonTrainerProcess_.state() != QProcess::NotRunning) {
            pythonTrainerProcess_.waitForFinished(1);
        }
    }
    pythonTrainerProcess_.waitForFinished(1000);
    drainPipelinePythonTrainerOutput(&stdoutBuffer, &result, &terminalMessageSeen);
    drainPipelinePythonTrainerErrors(&stderrBuffer, &result);
    interceptPythonTrainerMessages_ = false;

    if (!stdoutBuffer.trimmed().isEmpty()) {
        forwardPipelinePythonTrainerLine(stdoutBuffer.trimmed(), &result, &terminalMessageSeen);
    }
    if (!stderrBuffer.trimmed().isEmpty()) {
        QJsonObject logObject;
        logObject.insert(QStringLiteral("taskId"), request_.taskId);
        logObject.insert(QStringLiteral("backend"), backend);
        logObject.insert(QStringLiteral("message"), QString::fromUtf8(stderrBuffer.trimmed()));
        result.logs.append(logObject);
        send(wp::event::log(), logObject);
    }

    if (!terminalMessageSeen) {
        if (pythonTrainerProcess_.exitStatus() != QProcess::NormalExit || pythonTrainerProcess_.exitCode() != 0) {
            result.error = QStringLiteral("Python trainer exited with code %1").arg(pythonTrainerProcess_.exitCode());
        } else {
            result.error = QStringLiteral("Python trainer exited without a completed or failed message");
        }
        return result;
    }

    return result;
}

void WorkerSession::drainPipelinePythonTrainerOutput(QByteArray* buffer, PipelineTrainResult* result, bool* terminalMessageSeen)
{
    buffer->append(pythonTrainerProcess_.readAllStandardOutput());
    int newline = buffer->indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer->left(newline).trimmed();
        buffer->remove(0, newline + 1);
        if (!line.isEmpty()) {
            forwardPipelinePythonTrainerLine(line, result, terminalMessageSeen);
        }
        newline = buffer->indexOf('\n');
    }
}

void WorkerSession::drainPipelinePythonTrainerErrors(QByteArray* buffer, PipelineTrainResult* result)
{
    buffer->append(pythonTrainerProcess_.readAllStandardError());
    int newline = buffer->indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer->left(newline).trimmed();
        buffer->remove(0, newline + 1);
        if (!line.isEmpty()) {
            QJsonObject logObject;
            logObject.insert(QStringLiteral("taskId"), request_.taskId);
            logObject.insert(QStringLiteral("backend"), requestedTrainingBackend(request_));
            logObject.insert(QStringLiteral("message"), QString::fromUtf8(line));
            result->logs.append(logObject);
            send(wp::event::log(), logObject);
        }
        newline = buffer->indexOf('\n');
    }
}

bool WorkerSession::forwardPipelinePythonTrainerLine(const QByteArray& line, PipelineTrainResult* result, bool* terminalMessageSeen)
{
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("backend"), requestedTrainingBackend(request_));
        payload.insert(QStringLiteral("message"), QString::fromUtf8(line));
        result->logs.append(payload);
        send(wp::event::log(), payload);
        return true;
    }

    const QJsonObject object = document.object();
    const QString type = object.value(QStringLiteral("type")).toString();
    if (type.isEmpty()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("backend"), requestedTrainingBackend(request_));
        payload.insert(QStringLiteral("message"), QString::fromUtf8(line));
        result->logs.append(payload);
        send(wp::event::log(), payload);
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

    if (type == QStringLiteral("artifact")) {
        result->artifacts.append(payload);
        const QString kind = payload.value(QStringLiteral("kind")).toString();
        const QString path = payload.value(QStringLiteral("path")).toString();
        if (kind == QStringLiteral("checkpoint") && result->checkpointPath.isEmpty()) {
            result->checkpointPath = path;
        } else if (kind == QStringLiteral("onnx") && result->onnxPath.isEmpty()) {
            result->onnxPath = path;
        } else if ((kind == QStringLiteral("report") || kind == QStringLiteral("training_report")) && result->reportPath.isEmpty()) {
            result->reportPath = path;
        }
    } else if (type == QStringLiteral("metric")) {
        result->metrics.append(payload);
    } else if (type == QStringLiteral("log")) {
        result->logs.append(payload);
    } else if (type == QStringLiteral("completed")) {
        result->ok = true;
        result->completedPayload = payload;
        if (result->checkpointPath.isEmpty()) {
            result->checkpointPath = payload.value(QStringLiteral("checkpointPath")).toString();
        }
        if (result->onnxPath.isEmpty()) {
            result->onnxPath = payload.value(QStringLiteral("onnxPath")).toString();
        }
        if (result->reportPath.isEmpty()) {
            result->reportPath = payload.value(QStringLiteral("reportPath")).toString();
        }
        if (terminalMessageSeen) {
            *terminalMessageSeen = true;
        }
    } else if (type == QStringLiteral("failed")) {
        result->ok = false;
        result->error = payload.value(QStringLiteral("message")).toString(QStringLiteral("Pipeline training step failed"));
        if (terminalMessageSeen) {
            *terminalMessageSeen = true;
        }
    }

    if (type != QStringLiteral("completed") && type != QStringLiteral("failed")) {
        send(type, payload);
    }
    return true;
}
