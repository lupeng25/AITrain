#include "WorkerClient.h"

#include "aitrain/core/JsonProtocol.h"

#include <QCoreApplication>
#include <QFileInfo>
#include <QUuid>

WorkerClient::WorkerClient(QObject* parent)
    : QObject(parent)
{
    connect(&server_, &QLocalServer::newConnection, this, &WorkerClient::acceptConnection);
    connect(&process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &WorkerClient::workerFinished);
    connect(&process_, &QProcess::readyReadStandardOutput, this, [this]() {
        const QString output = QString::fromLocal8Bit(process_.readAllStandardOutput());
        if (!output.trimmed().isEmpty()) {
            emit logLine(output.trimmed());
        }
    });
}

WorkerClient::~WorkerClient()
{
    cleanupSocket();
    server_.close();
    if (process_.state() != QProcess::NotRunning) {
        process_.terminate();
        if (!process_.waitForFinished(3000)) {
            process_.kill();
            process_.waitForFinished(3000);
        }
    }
}

bool WorkerClient::startTraining(const QString& workerProgram, const aitrain::TrainingRequest& request, QString* error)
{
    return startWorkerCommand(workerProgram, QStringLiteral("startTrain"), request.toJson(), error);
}

bool WorkerClient::requestEnvironmentCheck(const QString& workerProgram, QString* error)
{
    return startWorkerCommand(workerProgram, QStringLiteral("environmentCheck"), {}, error);
}

bool WorkerClient::requestDatasetValidation(const QString& workerProgram, const QString& datasetPath, const QString& format, const QJsonObject& options, QString* error, const QString& taskId, const QString& outputPath)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("datasetPath"), datasetPath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("format"), format);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("validateDataset"), payload, error);
}

bool WorkerClient::requestDatasetSplit(const QString& workerProgram, const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("datasetPath"), datasetPath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("format"), format);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("splitDataset"), payload, error);
}

bool WorkerClient::requestDatasetCuration(const QString& workerProgram, const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("datasetPath"), datasetPath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("format"), format);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("curateDataset"), payload, error);
}

bool WorkerClient::requestDatasetSnapshot(const QString& workerProgram, const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("datasetPath"), datasetPath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("format"), format);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("createDatasetSnapshot"), payload, error);
}

bool WorkerClient::requestModelEvaluation(const QString& workerProgram, const QString& modelPath, const QString& datasetPath, const QString& outputPath, const QString& taskType, const QJsonObject& options, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("modelPath"), modelPath);
    payload.insert(QStringLiteral("datasetPath"), datasetPath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("taskType"), taskType);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("evaluateModel"), payload, error);
}

bool WorkerClient::requestModelBenchmark(const QString& workerProgram, const QString& modelPath, const QString& outputPath, const QJsonObject& options, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("modelPath"), modelPath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("benchmarkModel"), payload, error);
}

bool WorkerClient::requestLocalPipeline(const QString& workerProgram, const QString& outputPath, const QString& templateId, const QJsonObject& options, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("templateId"), templateId);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("runLocalPipeline"), payload, error);
}

bool WorkerClient::requestDeliveryReport(const QString& workerProgram, const QString& outputPath, const QJsonObject& context, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("context"), context);
    return startWorkerCommand(workerProgram, QStringLiteral("generateDeliveryReport"), payload, error);
}

bool WorkerClient::requestModelExport(const QString& workerProgram, const QString& checkpointPath, const QString& outputPath, const QString& format, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("checkpointPath"), checkpointPath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("format"), format);
    return startWorkerCommand(workerProgram, QStringLiteral("exportModel"), payload, error);
}

bool WorkerClient::requestInference(const QString& workerProgram, const QString& checkpointPath, const QString& imagePath, const QString& outputPath, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("checkpointPath"), checkpointPath);
    payload.insert(QStringLiteral("imagePath"), imagePath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    return startWorkerCommand(workerProgram, QStringLiteral("infer"), payload, error);
}

void WorkerClient::cancel()
{
    send(QStringLiteral("cancel"), {});
}

void WorkerClient::pause()
{
    send(QStringLiteral("pause"), {});
}

void WorkerClient::resume()
{
    send(QStringLiteral("resume"), {});
}

void WorkerClient::requestHeartbeat()
{
    send(QStringLiteral("heartbeat"), {});
}

bool WorkerClient::isRunning() const
{
    return process_.state() != QProcess::NotRunning;
}

bool WorkerClient::startWorkerCommand(const QString& workerProgram, const QString& commandType, const QJsonObject& payload, QString* error)
{
    if (process_.state() != QProcess::NotRunning) {
        if (error) {
            *error = QStringLiteral("Worker is already running");
        }
        return false;
    }

    if (!QFileInfo::exists(workerProgram)) {
        if (error) {
            *error = QStringLiteral("Worker executable not found: %1").arg(workerProgram);
        }
        return false;
    }

    if (server_.isListening()) {
        server_.close();
    }
    cleanupSocket();

    const QString serverName = QStringLiteral("aitrain_%1").arg(QUuid::createUuid().toString(QUuid::Id128));
    QLocalServer::removeServer(serverName);
    if (!server_.listen(serverName)) {
        if (error) {
            *error = server_.errorString();
        }
        return false;
    }

    buffer_.clear();
    pendingCommandType_ = commandType;
    pendingRequest_ = payload;
    process_.setProgram(workerProgram);
    process_.setArguments({QStringLiteral("--server"), serverName});
    process_.setProcessChannelMode(QProcess::MergedChannels);
    process_.start();
    if (!process_.waitForStarted(5000)) {
        if (error) {
            *error = process_.errorString();
        }
        pendingCommandType_.clear();
        pendingRequest_ = QJsonObject();
        server_.close();
        return false;
    }

    return true;
}

void WorkerClient::acceptConnection()
{
    cleanupSocket();
    socket_ = server_.nextPendingConnection();
    connect(socket_, &QLocalSocket::readyRead, this, &WorkerClient::readLines);
    emit connected();
}

void WorkerClient::readLines()
{
    if (!socket_) {
        return;
    }

    buffer_.append(socket_->readAll());
    int newline = buffer_.indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer_.left(newline);
        buffer_.remove(0, newline + 1);

        QString type;
        QJsonObject payload;
        QString requestId;
        QString error;
        if (aitrain::protocol::decodeMessage(line, &type, &payload, &requestId, &error)) {
            emit messageReceived(type, payload);
            if (type == QStringLiteral("ready")) {
                if (!pendingCommandType_.isEmpty()) {
                    send(pendingCommandType_, pendingRequest_);
                }
            } else if (type == QStringLiteral("log")) {
                emit logLine(payload.value(QStringLiteral("message")).toString());
            } else if (type == QStringLiteral("completed")) {
                emit finished(true, payload.value(QStringLiteral("message")).toString());
            } else if (type == QStringLiteral("failed")) {
                emit finished(false, payload.value(QStringLiteral("message")).toString());
            } else if (type == QStringLiteral("paused")
                || type == QStringLiteral("resumed")
                || type == QStringLiteral("canceled")) {
                emit logLine(payload.value(QStringLiteral("message")).toString());
            }
        } else {
            emit logLine(QStringLiteral("Protocol decode error: %1").arg(error));
        }

        newline = buffer_.indexOf('\n');
    }
}

void WorkerClient::workerFinished(int exitCode, QProcess::ExitStatus status)
{
    if (status != QProcess::NormalExit || exitCode != 0) {
        emit finished(false, QStringLiteral("Worker exited with code %1").arg(exitCode));
    }
    cleanupSocket();
    server_.close();
    pendingCommandType_.clear();
    pendingRequest_ = QJsonObject();
    emit idle();
}

void WorkerClient::send(const QString& type, const QJsonObject& payload)
{
    if (!socket_ || socket_->state() != QLocalSocket::ConnectedState) {
        return;
    }
    socket_->write(aitrain::protocol::encodeMessage(type, payload));
    socket_->flush();
}

void WorkerClient::cleanupSocket()
{
    if (!socket_) {
        return;
    }
    QLocalSocket* socket = socket_;
    socket_ = nullptr;
    socket->disconnect(this);
    if (socket->state() != QLocalSocket::UnconnectedState) {
        socket->disconnectFromServer();
        socket->waitForDisconnected(1000);
    }
    socket->deleteLater();
}
