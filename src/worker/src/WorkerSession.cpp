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
#include <QDebug>
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

WorkerSession::WorkerSession(QObject* parent)
    : QObject(parent)
{
    connect(&socket_, &QLocalSocket::readyRead, this, &WorkerSession::readLines);
    connect(&socket_, &QLocalSocket::disconnected, this, &WorkerSession::handleSocketDisconnected);
    connect(&timer_, &QTimer::timeout, this, &WorkerSession::tickTraining);
    timer_.setInterval(500);
}

bool WorkerSession::connectToServer(const QString& serverName)
{
    socket_.connectToServer(serverName);
    const bool connected = socket_.waitForConnected(5000);
    if (connected) {
        QTimer::singleShot(0, this, [this]() {
            QJsonObject payload;
            payload.insert(wp::field::message(), QStringLiteral("Worker ready"));
            send(wp::event::ready(), payload);
        });
    }
    return connected;
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

void WorkerSession::handleMessage(const QString& type, const QJsonObject& payload)
{
    if (!wp::isControlCommand(type)) {
        activeCommand_ = type;
        activeTaskId_ = payload.value(wp::field::taskId()).toString(activeTaskId_);
        activeOutputPath_ = payload.value(wp::field::outputPath()).toString();
        activeReportPath_.clear();
    }
    if (type == wp::command::startTrain()) {
        startTraining(aitrain::TrainingRequest::fromJson(payload));
    } else if (type == wp::command::pause()) {
        pauseTraining();
    } else if (type == wp::command::resume()) {
        resumeTraining();
    } else if (type == wp::command::heartbeat()) {
        sendHeartbeat();
    } else if (type == wp::command::environmentCheck()) {
        runEnvironmentCheck(payload);
    } else if (type == wp::command::validateDataset()) {
        validateDataset(payload);
    } else if (type == wp::command::splitDataset()) {
        splitDataset(payload);
    } else if (type == wp::command::convertDataset()) {
        convertDataset(payload);
    } else if (type == wp::command::curateDataset()) {
        curateDataset(payload);
    } else if (type == wp::command::createDatasetSnapshot()) {
        createDatasetSnapshot(payload);
    } else if (type == wp::command::evaluateModel()) {
        evaluateModel(payload);
    } else if (type == wp::command::benchmarkModel()) {
        benchmarkModel(payload);
    } else if (type == wp::command::runLocalPipeline()) {
        runLocalPipeline(payload);
    } else if (type == wp::command::generateDeliveryReport()) {
        generateDeliveryReport(payload);
    } else if (type == wp::command::runCustomerOcrAcceptance()) {
        runCustomerOcrAcceptance(payload);
    } else if (type == wp::command::collectDiagnostics()) {
        collectDiagnostics(payload);
    } else if (type == wp::command::validateDeploymentArtifact()) {
        validateDeploymentArtifact(payload);
    } else if (type == wp::command::exportModel()) {
        exportModel(payload);
    } else if (type == wp::command::infer()) {
        runInference(payload);
    } else if (type == wp::command::cancel()) {
        running_ = false;
        paused_ = false;
        canceled_ = true;
        timer_.stop();
        shutdownPythonTrainer(QStringLiteral("Canceled by user"), true);
        sendCanceledAndFinish(activeTaskId_.isEmpty() ? request_.taskId : activeTaskId_, QStringLiteral("Canceled by user"));
    } else {
        fail(QStringLiteral("Unsupported command: %1").arg(type));
    }
}

void WorkerSession::handleSocketDisconnected()
{
    if (finishingSession_) {
        return;
    }

    running_ = false;
    paused_ = false;
    canceled_ = true;
    timer_.stop();
    shutdownPythonTrainer(QStringLiteral("Worker client disconnected."), false);
    qApp->quit();
}

void WorkerSession::sendHeartbeat()
{
    QJsonObject payload;
    payload.insert(wp::field::taskId(), request_.taskId);
    payload.insert(QStringLiteral("running"), running_);
    payload.insert(QStringLiteral("paused"), paused_);
    payload.insert(QStringLiteral("step"), step_);
    payload.insert(QStringLiteral("timestamp"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(wp::event::heartbeat(), payload);
}

void WorkerSession::send(const QString& type, const QJsonObject& payload)
{
    socket_.write(aitrain::protocol::encodeMessage(type, payload));
    socket_.flush();
}

aitrain::CancellationCallback WorkerSession::cancellationCallback()
{
    return [this]() {
        return canceled_;
    };
}

bool WorkerSession::pollPendingCancel(int timeoutMs)
{
    const qint64 deadline = QDateTime::currentMSecsSinceEpoch() + qMax(0, timeoutMs);
    do {
        if (!socket_.bytesAvailable()) {
            const qint64 remaining = deadline - QDateTime::currentMSecsSinceEpoch();
            if (remaining <= 0 || !socket_.waitForReadyRead(qMin<qint64>(remaining, 20))) {
                continue;
            }
        }

        buffer_.append(socket_.readAll());
        int newline = buffer_.indexOf('\n');
        while (newline >= 0) {
            const QByteArray line = buffer_.left(newline);
            buffer_.remove(0, newline + 1);

            QString type;
            QJsonObject payload;
            QString requestId;
            QString error;
            if (!aitrain::protocol::decodeMessage(line, &type, &payload, &requestId, &error)) {
                continue;
            }
            if (type == wp::command::cancel()) {
                running_ = false;
                paused_ = false;
                canceled_ = true;
                timer_.stop();
                shutdownPythonTrainer(QStringLiteral("Canceled by user"), true);
                return true;
            }
            if (type == wp::command::heartbeat()) {
                sendHeartbeat();
            }
        }
    } while (QDateTime::currentMSecsSinceEpoch() < deadline && !canceled_);

    return canceled_;
}

void WorkerSession::shutdownPythonTrainer(const QString& reason, bool notifyClient)
{
    if (pythonTrainerProcess_.state() == QProcess::NotRunning) {
        return;
    }

    const QString taskId = activeTaskId_.isEmpty() ? request_.taskId : activeTaskId_;
    const auto emitShutdownLog = [this, notifyClient, &taskId](const QString& message) {
        if (notifyClient && socket_.state() == QLocalSocket::ConnectedState) {
            QJsonObject payload;
            payload.insert(wp::field::taskId(), taskId);
            payload.insert(wp::field::command(), activeCommand_);
            payload.insert(wp::field::message(), message);
            send(wp::event::log(), payload);
        } else {
            qWarning().noquote() << message;
        }
    };

    emitShutdownLog(QStringLiteral("Terminating Python trainer process: %1").arg(reason));
    pythonTrainerProcess_.terminate();
    if (!pythonTrainerProcess_.waitForFinished(1500)) {
        emitShutdownLog(QStringLiteral("Killing Python trainer process after terminate timeout: %1").arg(reason));
        pythonTrainerProcess_.kill();
        pythonTrainerProcess_.waitForFinished(1500);
    }
}

void WorkerSession::sendCanceledAndFinish(const QString& taskId, const QString& message)
{
    running_ = false;
    paused_ = false;
    canceled_ = true;
    timer_.stop();

    QJsonObject payload;
    payload.insert(wp::field::taskId(), taskId.isEmpty() ? request_.taskId : taskId);
    payload.insert(wp::field::command(), activeCommand_);
    payload.insert(wp::field::status(), QStringLiteral("canceled"));
    payload.insert(wp::field::errorCode(), QStringLiteral("canceled"));
    payload.insert(wp::field::message(), message.isEmpty() ? QStringLiteral("Canceled by user") : message);
    if (!activeReportPath_.isEmpty()) {
        payload.insert(wp::field::reportPath(), activeReportPath_);
    } else if (!activeOutputPath_.isEmpty()) {
        payload.insert(wp::field::outputPath(), activeOutputPath_);
    } else if (!request_.outputPath.isEmpty()) {
        payload.insert(wp::field::outputPath(), request_.outputPath);
    }
    send(wp::event::canceled(), payload);
    finishSession();
}

void WorkerSession::finishSession()
{
    finishingSession_ = true;
    activeTaskId_.clear();
    activeCommand_.clear();
    activeOutputPath_.clear();
    activeReportPath_.clear();
    socket_.flush();
    QElapsedTimer timer;
    timer.start();
    while (socket_.bytesToWrite() > 0 && timer.elapsed() < 3000) {
        socket_.waitForBytesWritten(100);
        QCoreApplication::processEvents(QEventLoop::AllEvents, 25);
    }
    if (socket_.state() == QLocalSocket::ConnectedState) {
        socket_.disconnectFromServer();
        socket_.waitForDisconnected(1000);
    }
    qApp->quit();
}

void WorkerSession::fail(const QString& message)
{
    failWithDetails(message, QStringLiteral("worker_failed"));
}

void WorkerSession::failWithDetails(const QString& message, const QString& errorCode, const QJsonObject& details)
{
    running_ = false;
    paused_ = false;
    timer_.stop();
    QJsonObject payload;
    payload.insert(wp::field::taskId(), activeTaskId_.isEmpty() ? request_.taskId : activeTaskId_);
    payload.insert(wp::field::command(), activeCommand_);
    payload.insert(wp::field::status(), QStringLiteral("failed"));
    payload.insert(wp::field::errorCode(), errorCode.isEmpty() ? QStringLiteral("worker_failed") : errorCode);
    payload.insert(wp::field::message(), message);
    if (!details.isEmpty()) {
        payload.insert(QStringLiteral("details"), details);
        if (details.contains(wp::field::reportPath())) {
            payload.insert(wp::field::reportPath(), details.value(wp::field::reportPath()).toString());
        }
        if (details.contains(wp::field::outputPath())) {
            payload.insert(wp::field::outputPath(), details.value(wp::field::outputPath()).toString());
        }
    }
    if (!payload.contains(wp::field::reportPath()) && !activeReportPath_.isEmpty()) {
        payload.insert(wp::field::reportPath(), activeReportPath_);
    }
    if (!payload.contains(wp::field::outputPath())) {
        if (!activeOutputPath_.isEmpty()) {
            payload.insert(wp::field::outputPath(), activeOutputPath_);
        } else if (!request_.outputPath.isEmpty()) {
            payload.insert(wp::field::outputPath(), request_.outputPath);
        }
    }
    send(wp::event::failed(), payload);
    finishSession();
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
    artifact.insert(wp::field::taskId(), request_.taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    artifact.insert(QStringLiteral("path"), checkpointPath);
    send(wp::event::artifact(), artifact);

    QJsonObject payload;
    payload.insert(wp::field::message(), QStringLiteral("Training workflow completed"));
    send(wp::event::completed(), payload);
    finishSession();
}
