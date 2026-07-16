#include "WorkerClient.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QFileInfo>
#include <QTimer>
#include <QUuid>

namespace wp = aitrain::worker_protocol;

WorkerClient::WorkerClient(QObject* parent)
    : QObject(parent)
{
    connect(&server_, &QLocalServer::newConnection, this, &WorkerClient::acceptConnection);
    connect(&process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &WorkerClient::workerFinished);
    connect(&process_, &QProcess::errorOccurred, this, &WorkerClient::workerProcessError);
    connect(&process_, &QProcess::readyReadStandardOutput, this, [this]() {
        const QString output = QString::fromUtf8(process_.readAllStandardOutput());
        if (!output.trimmed().isEmpty()) {
            emit logLine(output.trimmed());
        }
    });
    cancelTimer_.setSingleShot(true);
    connect(&cancelTimer_, &QTimer::timeout, this, [this]() {
        if (!cancelRequested_ || process_.state() == QProcess::NotRunning) {
            return;
        }
        process_.terminate();
        QTimer::singleShot(1500, this, [this]() {
            if (cancelRequested_ && process_.state() != QProcess::NotRunning) {
                process_.kill();
            }
        });
    });
    terminalShutdownTimer_.setSingleShot(true);
    connect(&terminalShutdownTimer_, &QTimer::timeout, this, [this]() {
        if (!finishedEmitted_ || process_.state() == QProcess::NotRunning) {
            return;
        }
        // 协议已经确认终态后，Worker 不得继续执行任何业务步骤。若其事件循环
        // 未自行收束，以温和终止回收该孤立子进程，避免占住下一项队列任务。
        process_.terminate();
        QTimer::singleShot(1500, this, [this]() {
            if (finishedEmitted_ && process_.state() != QProcess::NotRunning) {
                process_.kill();
            }
        });
    });
}

WorkerClient::~WorkerClient()
{
    cleanupSocket();
    server_.close();
    if (process_.state() != QProcess::NotRunning) {
        // QObject 析构阶段不能再进入嵌套事件循环等待 Worker。进程树的长期
        // 回收由 Worker/V2 Job Object 负责；GUI 这里只做立即的最后兜底。
        process_.kill();
    }
}

bool WorkerClient::requestTrainingWorkflowV2(const QString& workerProgram, const QJsonObject& request, QString* error)
{
    return startWorkerCommand(workerProgram, wp::command::runTrainingWorkflowV2(), request, error);
}

bool WorkerClient::requestEnvironmentCheckWorkflowV2(const QString& workerProgram,
    const QString& projectRoot, QString* error, const QString& taskId)
{
    return startWorkerCommand(workerProgram, wp::command::runEnvironmentCheckWorkflowV2(),
        QJsonObject{{wp::field::taskId(), taskId}, {QStringLiteral("projectRoot"), projectRoot}}, error);
}

bool WorkerClient::requestDatasetSplitWorkflowV2(const QString& workerProgram,
    const QString& projectRoot,
    const QString& sourceDatasetId,
    const QString& sourceDatasetVersionId,
    const QString& sourceSnapshotId,
    const QString& sourceSnapshotArtifactId,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::runDatasetSplitWorkflowV2(),
        wp::datasetSplitWorkflowV2Request(taskId, projectRoot, sourceDatasetId,
            sourceDatasetVersionId, sourceSnapshotId, sourceSnapshotArtifactId,
            targetDatasetId, targetDatasetName, options),
        error);
}

bool WorkerClient::requestDatasetConversionWorkflowV2(const QString& workerProgram,
    const QString& projectRoot,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetFormat,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::runDatasetConversionWorkflowV2(),
        wp::datasetConversionWorkflowV2Request(taskId, projectRoot, sourcePath,
            sourceFormat, targetFormat, targetDatasetId, targetDatasetName, options),
        error);
}

bool WorkerClient::requestDataQualityWorkflowV2(const QString& workerProgram,
    const QString& projectRoot,
    const QString& datasetId,
    const QString& datasetVersionId,
    const QString& snapshotId,
    const QString& snapshotArtifactId,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::runDataQualityWorkflowV2(),
        wp::dataQualityWorkflowV2Request(taskId, projectRoot, datasetId,
            datasetVersionId, snapshotId, snapshotArtifactId, options),
        error);
}

bool WorkerClient::requestAnnotationSessionCreateV2(const QString& workerProgram,
    const QString& projectRoot,
    const QString& repairManifestArtifactId,
    const QString& workingDirectory,
    const QJsonObject& toolSummary,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::createAnnotationSessionV2(),
        wp::annotationSessionCreateV2Request(taskId, projectRoot, repairManifestArtifactId,
            workingDirectory, toolSummary, options),
        error);
}

bool WorkerClient::requestAnnotationSessionSyncV2(const QString& workerProgram,
    const QString& projectRoot,
    const QString& sessionArtifactId,
    const QString& workingDirectory,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::syncAnnotationSessionV2(),
        wp::annotationSessionSyncV2Request(taskId, projectRoot, sessionArtifactId,
            workingDirectory, options),
        error);
}

bool WorkerClient::requestDatasetSnapshotImportWorkflowV2(const QString& workerProgram,
    const QString& projectRoot,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::runDatasetSnapshotImportWorkflowV2(),
        wp::datasetSnapshotImportWorkflowV2Request(taskId, projectRoot, sourcePath,
            sourceFormat, targetDatasetId, targetDatasetName, options),
        error);
}

bool WorkerClient::requestOcrOfficialReportImportV2(const QString& workerProgram,
    const QString& projectRoot,
    const QJsonObject& det,
    const QJsonObject& rec,
    const QJsonObject& system,
    const QString& acceptanceCohortId,
    const QString& customerDomainId,
    const QString& evidenceClass,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(workerProgram, wp::command::importOcrOfficialReportsV2(),
        wp::ocrOfficialReportImportV2Request(taskId, projectRoot, det, rec, system,
            acceptanceCohortId, customerDomainId, evidenceClass), error);
}

bool WorkerClient::requestOcrAcceptanceWorkflowV2(const QString& workerProgram,
    const QString& projectRoot,
    const QString& detReportArtifactId,
    const QString& recReportArtifactId,
    const QString& systemReportArtifactId,
    const QJsonObject& thresholds,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(workerProgram, wp::command::runOcrAcceptanceWorkflowV2(),
        wp::ocrAcceptanceWorkflowV2Request(taskId, projectRoot, detReportArtifactId,
            recReportArtifactId, systemReportArtifactId, thresholds), error);
}

bool WorkerClient::requestDiagnosticsWorkflowV2(const QString& workerProgram,
    const QString& projectRoot, const QJsonObject& options, QString* error, const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::runDiagnosticsWorkflowV2(),
        wp::diagnosticsWorkflowV2Request(taskId, projectRoot, options),
        error);
}

bool WorkerClient::requestRuntimeDeliveryWorkflowV2(const QString& workerProgram,
    const QString& projectRoot,
    const QString& modelPackageId,
    const QString& runtimeRoute,
    const QString& sampleImagePath,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::runRuntimeDeliveryWorkflowV2(),
        wp::runtimeDeliveryWorkflowV2Request(taskId, projectRoot, modelPackageId,
            runtimeRoute, sampleImagePath, options),
        error);
}

bool WorkerClient::requestModelImportV2(const QString& workerProgram, const QString& projectRoot, const QString& sourceFilePath, const QJsonObject& manifestDraft, QString* error, const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::importModelV2(),
        wp::modelImportV2Request(taskId, projectRoot, sourceFilePath, manifestDraft),
        error);
}

void WorkerClient::cancel()
{
    cancelRequested_ = true;
    // ready 前的取消只排队。控制面必须固定先发 sequence=1 的 start_task，
    // 再发 sequence=2 的 cancel_task，不能让 Worker 接受无业务身份的抢跑取消。
    if (startTaskSent_) {
        sendCancelTask();
    }
}

bool WorkerClient::isRunning() const
{
    return finishing_ || process_.state() != QProcess::NotRunning;
}

bool WorkerClient::startWorkerCommand(const QString& workerProgram, const QString& commandType, const QJsonObject& payload, QString* error)
{
    if (finishing_ || process_.state() != QProcess::NotRunning) {
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
    activeRequestId_ = aitrain::v2::RequestId::create();
    const QString requestedTaskId = payload.value(wp::field::taskId()).toString();
    QString taskIdError;
    if (!aitrain::v2::TaskId::parse(requestedTaskId, &activeTaskId_, &taskIdError)) {
        activeTaskId_ = aitrain::v2::TaskId::create();
    }
    incomingSequenceTracker_.clear();
    outgoingSequence_ = 0;
    finishedEmitted_ = false;
    startTaskSent_ = false;
    terminalEnvelopeReceived_ = false;
    cancelRequested_ = false;
    cancelTimer_.stop();
    terminalShutdownTimer_.stop();
    process_.setProgram(workerProgram);
    process_.setArguments({QStringLiteral("--server"), serverName,
        QStringLiteral("--request-id"), activeRequestId_.toString(),
        QStringLiteral("--task-id"), activeTaskId_.toString()});
    process_.setProcessChannelMode(QProcess::MergedChannels);
    process_.start();
    // QProcess::start() 是异步操作；FailedToStart 由 errorOccurred 收口，避免
    // GUI 线程在慢磁盘、杀毒扫描或进程创建异常时同步阻塞 5 秒。
    return true;
}

void WorkerClient::acceptConnection()
{
    cleanupSocket();
    socket_ = server_.nextPendingConnection();
    socket_->setReadBufferSize(aitrain::v2::kProtocolV2MaxControlMessageBytes + 1);
    connect(socket_, &QLocalSocket::readyRead, this, &WorkerClient::readLines);
    readLines();
    emit connected();
}

void WorkerClient::readLines()
{
    if (!socket_) {
        return;
    }

    buffer_.append(socket_->readAll());
    if (buffer_.size() > aitrain::v2::kProtocolV2MaxControlMessageBytes
        && !buffer_.contains('\n')) {
        rejectProtocol(QStringLiteral("Protocol V2 frame exceeds maximum size."));
        return;
    }

    int newline = buffer_.indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer_.left(newline + 1);
        buffer_.remove(0, newline + 1);
        if (line.size() > aitrain::v2::kProtocolV2MaxControlMessageBytes) {
            rejectProtocol(QStringLiteral("Protocol V2 frame exceeds maximum size."));
            return;
        }

        aitrain::v2::ProtocolEnvelope envelope;
        QString error;
        if (!aitrain::v2::decodeProtocolV2Message(line, &envelope, &error)
            || !incomingSequenceTracker_.observe(envelope, activeRequestId_, activeTaskId_, &error)) {
            rejectProtocol(QStringLiteral("Protocol V2 message rejected: %1").arg(error));
            return;
        }
        if (terminalEnvelopeReceived_) {
            rejectProtocol(QStringLiteral("Protocol V2 event received after terminal event."));
            return;
        }

        QString type;
        QJsonObject payload;
        if (!wp::control_v2::unpackBusinessEvent(envelope, &type, &payload, &error)) {
            rejectProtocol(QStringLiteral("Protocol V2 event rejected: %1").arg(error));
            return;
        }

        const bool terminal = envelope.kind == QStringLiteral("event.succeeded")
            || envelope.kind == QStringLiteral("event.failed")
            || envelope.kind == QStringLiteral("event.canceled");
        if (terminal) {
            terminalEnvelopeReceived_ = true;
        }

        emit messageReceived(type, payload);
        if (type == wp::event::ready()) {
            if (!pendingCommandType_.isEmpty()) {
                sendStartTask();
            }
        } else if (type == wp::event::log()) {
            emit logLine(payload.value(wp::field::message()).toString());
        } else if (type == wp::event::completed()) {
            cancelRequested_ = false;
            cancelTimer_.stop();
            finishedEmitted_ = true;
            emit finished(true, payload.value(wp::field::message()).toString());
            QTimer::singleShot(0, this, [this]() {
                if (finishedEmitted_ && socket_ && process_.state() != QProcess::NotRunning) {
                    socket_->disconnectFromServer();
                }
            });
            terminalShutdownTimer_.start(500);
        } else if (type == wp::event::failed()) {
            cancelRequested_ = false;
            cancelTimer_.stop();
            finishedEmitted_ = true;
            emit finished(false, payload.value(wp::field::message()).toString());
            QTimer::singleShot(0, this, [this]() {
                if (finishedEmitted_ && socket_ && process_.state() != QProcess::NotRunning) {
                    socket_->disconnectFromServer();
                }
            });
            terminalShutdownTimer_.start(500);
        } else if (type == wp::event::canceled()) {
            cancelRequested_ = false;
            cancelTimer_.stop();
            finishedEmitted_ = true;
            emit finished(false, payload.value(wp::field::message()).toString(QStringLiteral("Canceled by user")));
            emit logLine(payload.value(wp::field::message()).toString());
            QTimer::singleShot(0, this, [this]() {
                if (finishedEmitted_ && socket_ && process_.state() != QProcess::NotRunning) {
                    socket_->disconnectFromServer();
                }
            });
            terminalShutdownTimer_.start(500);
        }

        newline = buffer_.indexOf('\n');
    }
}

void WorkerClient::workerFinished(int exitCode, QProcess::ExitStatus status)
{
    finishing_ = true;
    pendingExitCode_ = exitCode;
    pendingExitStatus_ = status;
    if (socket_ && socket_->bytesAvailable() > 0) {
        readLines();
    }
    // 让同一事件循环轮次中已排队的 QLocalSocket::readyRead 先送达，再进行
    // “无终态退出”的兜底判定；不使用 processEvents/waitForReadyRead 排空循环。
    QTimer::singleShot(0, this, &WorkerClient::finalizeWorkerExit);
}

void WorkerClient::workerProcessError(QProcess::ProcessError error)
{
    if (error != QProcess::FailedToStart || finishing_ || process_.state() != QProcess::NotRunning) {
        return;
    }
    finishing_ = true;
    pendingExitCode_ = -1;
    pendingExitStatus_ = QProcess::CrashExit;
    QTimer::singleShot(0, this, &WorkerClient::finalizeWorkerExit);
}

void WorkerClient::finalizeWorkerExit()
{
    if (!finishing_) {
        return;
    }
    if (socket_ && socket_->bytesAvailable() > 0) {
        readLines();
    }
    if (!finishedEmitted_) {
        finishedEmitted_ = true;
        if (cancelRequested_) {
            const QString message = QStringLiteral("Canceled by user");
            QJsonObject payload;
            payload.insert(wp::field::taskId(), pendingRequest_.value(wp::field::taskId()).toString());
            payload.insert(wp::field::command(), pendingCommandType_);
            payload.insert(wp::field::status(), QStringLiteral("canceled"));
            payload.insert(wp::field::errorCode(), QStringLiteral("canceled"));
            payload.insert(wp::field::message(), message);
            emit messageReceived(wp::event::canceled(), payload);
            emit finished(false, message);
        } else {
            const QString message = pendingExitCode_ < 0
                ? QStringLiteral("Worker failed to start: %1").arg(process_.errorString())
                : (pendingExitStatus_ != QProcess::NormalExit || pendingExitCode_ != 0)
                ? QStringLiteral("Worker exited with code %1").arg(pendingExitCode_)
                : QStringLiteral("Worker exited without a terminal status message");
            emit finished(false, message);
        }
    }
    cancelRequested_ = false;
    cancelTimer_.stop();
    terminalShutdownTimer_.stop();
    cleanupSocket();
    server_.close();
    pendingCommandType_.clear();
    pendingRequest_ = QJsonObject();
    activeRequestId_ = aitrain::v2::RequestId();
    activeTaskId_ = aitrain::v2::TaskId();
    incomingSequenceTracker_.clear();
    outgoingSequence_ = 0;
    startTaskSent_ = false;
    terminalEnvelopeReceived_ = false;
    finishing_ = false;
    QTimer::singleShot(0, this, [this]() {
        emit idle();
    });
}

void WorkerClient::sendStartTask()
{
    const aitrain::v2::ProtocolEnvelope envelope = wp::control_v2::startTaskEnvelope(
        activeRequestId_, activeTaskId_, ++outgoingSequence_, pendingCommandType_, pendingRequest_);
    QString error;
    if (!sendEnvelope(envelope, &error)) {
        rejectProtocol(error);
        return;
    }
    startTaskSent_ = true;
    if (cancelRequested_) {
        sendCancelTask();
    }
}

void WorkerClient::sendCancelTask()
{
    if (!activeRequestId_.isValid() || !activeTaskId_.isValid()) {
        return;
    }
    const aitrain::v2::ProtocolEnvelope envelope = wp::control_v2::cancelTaskEnvelope(
        activeRequestId_, activeTaskId_, ++outgoingSequence_);
    QString error;
    if (!sendEnvelope(envelope, &error)) {
        rejectProtocol(error);
        return;
    }
    if (process_.state() != QProcess::NotRunning) {
        cancelTimer_.start(2000);
    }
}

bool WorkerClient::sendEnvelope(const aitrain::v2::ProtocolEnvelope& envelope, QString* error)
{
    if (!socket_ || socket_->state() != QLocalSocket::ConnectedState) {
        if (error) *error = QStringLiteral("Worker control socket is not connected.");
        return false;
    }
    const QByteArray bytes = aitrain::v2::encodeProtocolV2Message(envelope, error);
    if (bytes.isEmpty()) {
        return false;
    }
    socket_->write(bytes);
    socket_->flush();
    return true;
}

void WorkerClient::rejectProtocol(const QString& message)
{
    emit logLine(message);
    if (!finishedEmitted_) {
        finishedEmitted_ = true;
        QJsonObject payload;
        payload.insert(wp::field::taskId(), pendingRequest_.value(wp::field::taskId()).toString());
        payload.insert(wp::field::command(), pendingCommandType_);
        payload.insert(wp::field::status(), QStringLiteral("failed"));
        payload.insert(wp::field::errorCode(), QStringLiteral("protocol_v2_rejected"));
        payload.insert(wp::field::message(), message);
        emit messageReceived(wp::event::failed(), payload);
        emit finished(false, message);
    }
    if (socket_) {
        socket_->disconnectFromServer();
    }
    terminalShutdownTimer_.start(500);
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
    }
    socket->deleteLater();
}
