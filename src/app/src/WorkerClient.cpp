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

bool WorkerClient::startTraining(const QString& workerProgram, const aitrain::TrainingRequest& request, QString* error)
{
    return startWorkerCommand(workerProgram, QStringLiteral("startTrain"), request.toJson(), error);
}

bool WorkerClient::requestEnvironmentCheck(const QString& workerProgram, QString* error)
{
    return startWorkerCommand(workerProgram, QStringLiteral("environmentCheck"), {}, error);
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

    const QString serverName = QStringLiteral("aitrain_%1").arg(QUuid::createUuid().toString(QUuid::Id128));
    QLocalServer::removeServer(serverName);
    if (!server_.listen(serverName)) {
        if (error) {
            *error = server_.errorString();
        }
        return false;
    }

    socket_ = nullptr;
    buffer_.clear();
    process_.setProgram(workerProgram);
    process_.setArguments({QStringLiteral("--server"), serverName});
    process_.setProcessChannelMode(QProcess::MergedChannels);
    process_.start();
    if (!process_.waitForStarted(5000)) {
        if (error) {
            *error = process_.errorString();
        }
        server_.close();
        return false;
    }

    pendingCommandType_ = commandType;
    pendingRequest_ = payload;
    return true;
}

void WorkerClient::acceptConnection()
{
    socket_ = server_.nextPendingConnection();
    connect(socket_, &QLocalSocket::readyRead, this, &WorkerClient::readLines);
    connect(socket_, &QLocalSocket::disconnected, socket_, &QObject::deleteLater);
    connect(socket_, &QObject::destroyed, this, [this]() { socket_ = nullptr; });
    emit connected();

    if (!pendingCommandType_.isEmpty()) {
        send(pendingCommandType_, pendingRequest_);
    }
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
            if (type == QStringLiteral("log")) {
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
