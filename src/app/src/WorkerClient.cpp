#include "WorkerClient.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QFileInfo>
#include <QTimer>
#include <QUuid>

namespace wp = aitrain::worker_protocol;

WorkerClient::WorkerClient(QObject* parent)
    : QObject(parent)
{
    qRegisterMetaType<WorkerTerminalStatus>("WorkerTerminalStatus");
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
    connectionTimer_.setSingleShot(true);
    connect(&connectionTimer_, &QTimer::timeout, this, [this]() {
        if (!finishing_ && !workerReady_ && process_.state() != QProcess::NotRunning) {
            rejectProtocol(QStringLiteral("Worker control connection/ready event timed out."));
        }
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
        // 回收由 Worker/ Job Object 负责；GUI 这里只做立即的最后兜底。
        process_.kill();
        process_.waitForFinished(2000);
    }
}

bool WorkerClient::requestTrainingWorkflow(const QString& workerProgram, const QJsonObject& request, QString* error)
{
    return startWorkerCommand(workerProgram, wp::command::runTrainingWorkflow(), request, error);
}

bool WorkerClient::requestEnvironmentCheckWorkflow(const QString& workerProgram,
    const QString& projectRoot, QString* error, const QString& taskId)
{
    return startWorkerCommand(workerProgram, wp::command::runEnvironmentCheckWorkflow(),
        QJsonObject{{wp::field::taskId(), taskId}, {QStringLiteral("projectRoot"), projectRoot}}, error);
}

bool WorkerClient::requestDatasetSplitWorkflow(const QString& workerProgram,
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
        wp::command::runDatasetSplitWorkflow(),
        wp::datasetSplitWorkflowRequest(taskId, projectRoot, sourceDatasetId,
            sourceDatasetVersionId, sourceSnapshotId, sourceSnapshotArtifactId,
            targetDatasetId, targetDatasetName, options),
        error);
}

bool WorkerClient::requestDatasetConversionWorkflow(const QString& workerProgram,
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
        wp::command::runDatasetConversionWorkflow(),
        wp::datasetConversionWorkflowRequest(taskId, projectRoot, sourcePath,
            sourceFormat, targetFormat, targetDatasetId, targetDatasetName, options),
        error);
}

bool WorkerClient::requestDataQualityWorkflow(const QString& workerProgram,
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
        wp::command::runDataQualityWorkflow(),
        wp::dataQualityWorkflowRequest(taskId, projectRoot, datasetId,
            datasetVersionId, snapshotId, snapshotArtifactId, options),
        error);
}

bool WorkerClient::requestAnnotationSessionCreate(const QString& workerProgram,
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
        wp::command::createAnnotationSession(),
        wp::annotationSessionCreateRequest(taskId, projectRoot, repairManifestArtifactId,
            workingDirectory, toolSummary, options),
        error);
}

bool WorkerClient::requestAnnotationSessionSync(const QString& workerProgram,
    const QString& projectRoot,
    const QString& sessionArtifactId,
    const QString& workingDirectory,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::syncAnnotationSession(),
        wp::annotationSessionSyncRequest(taskId, projectRoot, sessionArtifactId,
            workingDirectory, options),
        error);
}

bool WorkerClient::requestDatasetSnapshotImportWorkflow(const QString& workerProgram,
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
        wp::command::runDatasetSnapshotImportWorkflow(),
        wp::datasetSnapshotImportWorkflowRequest(taskId, projectRoot, sourcePath,
            sourceFormat, targetDatasetId, targetDatasetName, options),
        error);
}

bool WorkerClient::requestOcrOfficialReportImport(const QString& workerProgram,
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
    return startWorkerCommand(workerProgram, wp::command::importOcrOfficialReports(),
        wp::ocrOfficialReportImportRequest(taskId, projectRoot, det, rec, system,
            acceptanceCohortId, customerDomainId, evidenceClass), error);
}

bool WorkerClient::requestOcrAcceptanceWorkflow(const QString& workerProgram,
    const QString& projectRoot,
    const QString& detReportArtifactId,
    const QString& recReportArtifactId,
    const QString& systemReportArtifactId,
    const QJsonObject& thresholds,
    QString* error,
    const QString& taskId)
{
    return startWorkerCommand(workerProgram, wp::command::runOcrAcceptanceWorkflow(),
        wp::ocrAcceptanceWorkflowRequest(taskId, projectRoot, detReportArtifactId,
            recReportArtifactId, systemReportArtifactId, thresholds), error);
}

bool WorkerClient::requestDiagnosticsWorkflow(const QString& workerProgram,
    const QString& projectRoot, const QJsonObject& options, QString* error, const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::runDiagnosticsWorkflow(),
        wp::diagnosticsWorkflowRequest(taskId, projectRoot, options),
        error);
}

bool WorkerClient::requestRuntimeDeliveryWorkflow(const QString& workerProgram,
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
        wp::command::runRuntimeDeliveryWorkflow(),
        wp::runtimeDeliveryWorkflowRequest(taskId, projectRoot, modelPackageId,
            runtimeRoute, sampleImagePath, options),
        error);
}

bool WorkerClient::requestModelImport(const QString& workerProgram, const QString& projectRoot, const QString& sourceFilePath, const QJsonObject& manifestDraft, QString* error, const QString& taskId)
{
    return startWorkerCommand(
        workerProgram,
        wp::command::importModel(),
        wp::modelImportRequest(taskId, projectRoot, sourceFilePath, manifestDraft),
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
    activeRequestId_ = aitrain::RequestId::create();
    const QString requestedTaskId = payload.value(wp::field::taskId()).toString();
    QString taskIdError;
    if (!aitrain::TaskId::parse(requestedTaskId, &activeTaskId_, &taskIdError)) {
        activeTaskId_ = aitrain::TaskId::create();
    }
    incomingSequenceTracker_.clear();
    outgoingSequence_ = 0;
    finishedEmitted_ = false;
    startTaskSent_ = false;
    terminalEnvelopeReceived_ = false;
    cancelRequested_ = false;
    workerReady_ = false;
    cancelTimer_.stop();
    connectionTimer_.stop();
    terminalShutdownTimer_.stop();
    process_.setProgram(workerProgram);
    process_.setArguments({QStringLiteral("--server"), serverName,
        QStringLiteral("--request-id"), activeRequestId_.toString(),
        QStringLiteral("--task-id"), activeTaskId_.toString()});
    process_.setProcessChannelMode(QProcess::MergedChannels);
    process_.start();
    connectionTimer_.start(5000);
    // QProcess::start() 是异步操作；FailedToStart 由 errorOccurred 收口，避免
    // GUI 线程在慢磁盘、杀毒扫描或进程创建异常时同步阻塞 5 秒。
    return true;
}

void WorkerClient::acceptConnection()
{
    QLocalSocket* candidate = server_.nextPendingConnection();
    if (!candidate) {
        return;
    }
    // A second local client must never replace the socket that owns the active
    // request. Reject it explicitly; otherwise a stray process can steal the
    // channel and make the real Worker appear to have disappeared.
    if (socket_) {
        candidate->disconnectFromServer();
        candidate->deleteLater();
        return;
    }
    socket_ = candidate;
    socket_->setReadBufferSize(aitrain::kProtocolMaxControlMessageBytes + 1);
    connect(socket_, &QLocalSocket::readyRead, this, &WorkerClient::readLines);
    connect(socket_, &QLocalSocket::disconnected, this, [this, socket = socket_]() {
        if (socket_ == socket) {
            socket_ = nullptr;
        }
        socket->deleteLater();
    });
    readLines();
    emit connected();
}

void WorkerClient::readLines()
{
    if (!socket_) {
        return;
    }

    buffer_.append(socket_->readAll());
    if (buffer_.size() > aitrain::kProtocolMaxControlMessageBytes
        && !buffer_.contains('\n')) {
        rejectProtocol(QStringLiteral("Protocol  frame exceeds maximum size."));
        return;
    }

    int newline = buffer_.indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer_.left(newline + 1);
        buffer_.remove(0, newline + 1);
        if (line.size() > aitrain::kProtocolMaxControlMessageBytes) {
            rejectProtocol(QStringLiteral("Protocol  frame exceeds maximum size."));
            return;
        }

        aitrain::ProtocolEnvelope envelope;
        QString error;
        if (!aitrain::decodeProtocolMessage(line, &envelope, &error)
            || !incomingSequenceTracker_.observe(envelope, activeRequestId_, activeTaskId_, &error)) {
            rejectProtocol(QStringLiteral("Protocol  message rejected: %1").arg(error));
            return;
        }
        if (terminalEnvelopeReceived_) {
            rejectProtocol(QStringLiteral("Protocol  event received after terminal event."));
            return;
        }

        QString type;
        QJsonObject payload;
        if (!wp::control::unpackBusinessEvent(envelope, &type, &payload, &error)) {
            rejectProtocol(QStringLiteral("Protocol  event rejected: %1").arg(error));
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
            workerReady_ = true;
            connectionTimer_.stop();
            if (!pendingCommandType_.isEmpty()) {
                sendStartTask();
            }
        } else if (type == wp::event::log()) {
            emit logLine(payload.value(wp::field::message()).toString());
        } else if (type == wp::event::completed()) {
            cancelRequested_ = false;
            cancelTimer_.stop();
            finishedEmitted_ = true;
            emit finished(WorkerTerminalStatus::Succeeded, payload.value(wp::field::message()).toString());
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
            emit finished(WorkerTerminalStatus::Failed, payload.value(wp::field::message()).toString());
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
            emit finished(WorkerTerminalStatus::Canceled,
                payload.value(wp::field::message()).toString(QStringLiteral("Canceled by user")));
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
    // QProcess::finished can be delivered before the final local-socket frame
    // reaches Qt's readyRead queue. Give the socket one bounded drain window;
    // otherwise a healthy Worker is reported as "without terminal status".
    QTimer::singleShot(300, this, &WorkerClient::finalizeWorkerExit);
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
    if (socket_ && !finishedEmitted_ && socket_->state() == QLocalSocket::ConnectedState) {
        socket_->waitForReadyRead(300);
        if (socket_->bytesAvailable() > 0) {
            readLines();
        }
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
            emit finished(WorkerTerminalStatus::Canceled, message);
        } else {
            const QString message = pendingExitCode_ < 0
                ? QStringLiteral("Worker failed to start: %1").arg(process_.errorString())
                : (pendingExitStatus_ != QProcess::NormalExit || pendingExitCode_ != 0)
                ? QStringLiteral("Worker exited with code %1").arg(pendingExitCode_)
                : QStringLiteral("Worker exited without a terminal status message");
            emit finished(WorkerTerminalStatus::Failed, message);
        }
    }
    cancelRequested_ = false;
    cancelTimer_.stop();
    connectionTimer_.stop();
    terminalShutdownTimer_.stop();
    cleanupSocket();
    server_.close();
    pendingCommandType_.clear();
    pendingRequest_ = QJsonObject();
    activeRequestId_ = aitrain::RequestId();
    activeTaskId_ = aitrain::TaskId();
    incomingSequenceTracker_.clear();
    outgoingSequence_ = 0;
    startTaskSent_ = false;
    terminalEnvelopeReceived_ = false;
    workerReady_ = false;
    finishing_ = false;
    QTimer::singleShot(0, this, [this]() {
        emit idle();
    });
}

void WorkerClient::sendStartTask()
{
    const aitrain::ProtocolEnvelope envelope = wp::control::startTaskEnvelope(
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
    const aitrain::ProtocolEnvelope envelope = wp::control::cancelTaskEnvelope(
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

bool WorkerClient::sendEnvelope(const aitrain::ProtocolEnvelope& envelope, QString* error)
{
    if (!socket_ || socket_->state() != QLocalSocket::ConnectedState) {
        if (error) *error = QStringLiteral("Worker control socket is not connected.");
        return false;
    }
    const QByteArray bytes = aitrain::encodeProtocolMessage(envelope, error);
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
        payload.insert(wp::field::errorCode(), QStringLiteral("protocol_rejected"));
        payload.insert(wp::field::message(), message);
        emit messageReceived(wp::event::failed(), payload);
        emit finished(WorkerTerminalStatus::Failed, message);
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
