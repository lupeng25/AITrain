#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/OcrRecTrainer.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/SegmentationTrainer.h"

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
bool WorkerSession::shouldUsePythonTrainer() const
{
    return isPythonTrainingBackendId(requestedTrainingBackend(request_), request_.parameters);
}

void WorkerSession::runPythonTrainer()
{
    const QString backend = requestedTrainingBackend(request_);
    const QString pythonExecutable = firstUsablePythonExecutable(request_.parameters);
    if (pythonExecutable.isEmpty()) {
        failWithDetails(
            QStringLiteral("Python trainer backend '%1' requires a usable Python executable. Configure pythonExecutable or install Python.").arg(backend),
            QStringLiteral("python_missing"),
            QJsonObject{
                {QStringLiteral("backend"), backend},
                {QStringLiteral("outputPath"), request_.outputPath}});
        return;
    }

    const QString trainerScript = pythonTrainerScriptPath(request_.parameters, backend);
    if (!QFileInfo::exists(trainerScript)) {
        failWithDetails(
            QStringLiteral("Python trainer script not found: %1").arg(trainerScript),
            QStringLiteral("python_trainer_script_missing"),
            QJsonObject{
                {QStringLiteral("backend"), backend},
                {QStringLiteral("trainerScript"), trainerScript},
                {QStringLiteral("outputPath"), request_.outputPath}});
        return;
    }

    if (!QDir().mkpath(request_.outputPath)) {
        failWithDetails(
            QStringLiteral("Cannot create Python trainer output directory: %1").arg(request_.outputPath),
            QStringLiteral("output_create_failed"),
            QJsonObject{{QStringLiteral("backend"), backend}, {QStringLiteral("outputPath"), request_.outputPath}});
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
        failWithDetails(
            QStringLiteral("Cannot write Python trainer request: %1").arg(requestPath),
            QStringLiteral("python_trainer_request_write_failed"),
            QJsonObject{
                {QStringLiteral("backend"), backend},
                {QStringLiteral("requestPath"), requestPath},
                {QStringLiteral("outputPath"), request_.outputPath}});
        return;
    }
    requestFile.write(QJsonDocument(trainerRequest).toJson(QJsonDocument::Indented));
    requestFile.close();

    QJsonObject adapterLog;
    adapterLog.insert(QStringLiteral("taskId"), request_.taskId);
    adapterLog.insert(QStringLiteral("message"), QStringLiteral("Launching Python trainer backend=%1").arg(backend));
    adapterLog.insert(QStringLiteral("backend"), backend);
    send(QStringLiteral("log"), adapterLog);

    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    pythonTrainerProcess_.setProcessEnvironment(environment);
    pythonTrainerProcess_.setProgram(pythonExecutable);
    pythonTrainerProcess_.setArguments(QStringList() << QStringLiteral("-u") << trainerScript << QStringLiteral("--request") << requestPath);
    pythonTrainerProcess_.setProcessChannelMode(QProcess::SeparateChannels);
    pythonTrainerProcess_.start();
    if (!pythonTrainerProcess_.waitForStarted(5000)) {
        failWithDetails(
            QStringLiteral("Cannot start Python trainer: %1").arg(pythonTrainerProcess_.errorString()),
            QStringLiteral("python_trainer_start_failed"),
            QJsonObject{
                {QStringLiteral("backend"), backend},
                {QStringLiteral("pythonExecutable"), pythonExecutable},
                {QStringLiteral("trainerScript"), trainerScript},
                {QStringLiteral("requestPath"), requestPath},
                {QStringLiteral("outputPath"), request_.outputPath},
                {QStringLiteral("processError"), pythonTrainerProcess_.errorString()}});
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
        finishSession();
        return;
    }
    if (pythonTrainerProcess_.exitStatus() != QProcess::NormalExit || pythonTrainerProcess_.exitCode() != 0) {
        failWithDetails(
            QStringLiteral("Python trainer exited with code %1").arg(pythonTrainerProcess_.exitCode()),
            QStringLiteral("python_trainer_process_failed"),
            QJsonObject{
                {QStringLiteral("backend"), backend},
                {QStringLiteral("pythonExecutable"), pythonExecutable},
                {QStringLiteral("trainerScript"), trainerScript},
                {QStringLiteral("requestPath"), requestPath},
                {QStringLiteral("outputPath"), request_.outputPath},
                {QStringLiteral("exitCode"), pythonTrainerProcess_.exitCode()},
                {QStringLiteral("exitStatus"), pythonTrainerProcess_.exitStatus() == QProcess::NormalExit ? QStringLiteral("normal") : QStringLiteral("crash")}});
        return;
    }
    failWithDetails(
        QStringLiteral("Python trainer exited without a completed or failed message"),
        QStringLiteral("python_trainer_no_terminal_message"),
        QJsonObject{
            {QStringLiteral("backend"), backend},
            {QStringLiteral("pythonExecutable"), pythonExecutable},
            {QStringLiteral("trainerScript"), trainerScript},
            {QStringLiteral("requestPath"), requestPath},
            {QStringLiteral("outputPath"), request_.outputPath}});
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
    if (interceptPythonTrainerMessages_) {
        return true;
    }

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
    if (type == QStringLiteral("failed")) {
        payload.insert(QStringLiteral("status"), QStringLiteral("failed"));
        payload.insert(QStringLiteral("command"), activeCommand_);
        if (!payload.contains(QStringLiteral("errorCode"))) {
            payload.insert(QStringLiteral("errorCode"), payload.value(QStringLiteral("code")).toString(QStringLiteral("python_trainer_failed")));
        }
        if (!payload.contains(QStringLiteral("outputPath")) && !request_.outputPath.isEmpty()) {
            payload.insert(QStringLiteral("outputPath"), request_.outputPath);
        }
    } else if (type == QStringLiteral("completed")) {
        payload.insert(QStringLiteral("status"), QStringLiteral("completed"));
        payload.insert(QStringLiteral("command"), activeCommand_);
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
