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
    const bool connected = socket_.waitForConnected(5000);
    if (connected) {
        QTimer::singleShot(0, this, [this]() {
            QJsonObject payload;
            payload.insert(QStringLiteral("message"), QStringLiteral("Worker ready"));
            send(QStringLiteral("ready"), payload);
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
    } else if (type == QStringLiteral("convertDataset")) {
        convertDataset(payload);
    } else if (type == QStringLiteral("curateDataset")) {
        curateDataset(payload);
    } else if (type == QStringLiteral("createDatasetSnapshot")) {
        createDatasetSnapshot(payload);
    } else if (type == QStringLiteral("evaluateModel")) {
        evaluateModel(payload);
    } else if (type == QStringLiteral("benchmarkModel")) {
        benchmarkModel(payload);
    } else if (type == QStringLiteral("runLocalPipeline")) {
        runLocalPipeline(payload);
    } else if (type == QStringLiteral("generateDeliveryReport")) {
        generateDeliveryReport(payload);
    } else if (type == QStringLiteral("runCustomerOcrAcceptance")) {
        runCustomerOcrAcceptance(payload);
    } else if (type == QStringLiteral("collectDiagnostics")) {
        collectDiagnostics(payload);
    } else if (type == QStringLiteral("validateDeploymentArtifact")) {
        validateDeploymentArtifact(payload);
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
        payloadObject.insert(QStringLiteral("taskId"), activeTaskId_.isEmpty() ? request_.taskId : activeTaskId_);
        payloadObject.insert(QStringLiteral("message"), QStringLiteral("Canceled by user"));
        send(QStringLiteral("canceled"), payloadObject);
        finishSession();
    } else {
        fail(QStringLiteral("Unsupported command: %1").arg(type));
    }
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

void WorkerSession::send(const QString& type, const QJsonObject& payload)
{
    socket_.write(aitrain::protocol::encodeMessage(type, payload));
    socket_.flush();
}

void WorkerSession::finishSession()
{
    activeTaskId_.clear();
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
    running_ = false;
    paused_ = false;
    timer_.stop();
    QJsonObject payload;
    payload.insert(QStringLiteral("message"), message);
    send(QStringLiteral("failed"), payload);
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
    artifact.insert(QStringLiteral("taskId"), request_.taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    artifact.insert(QStringLiteral("path"), checkpointPath);
    send(QStringLiteral("artifact"), artifact);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Training workflow completed"));
    send(QStringLiteral("completed"), payload);
    finishSession();
}
