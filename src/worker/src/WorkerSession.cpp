#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QThread>
#include <QTimer>

using namespace worker_support;
namespace wp = aitrain::worker_protocol;

WorkerSession::WorkerSession(QObject* parent)
    : QObject(parent)
{
    connect(&socket_, &QLocalSocket::readyRead, this, &WorkerSession::readLines);
    connect(&socket_, &QLocalSocket::disconnected, this, &WorkerSession::handleSocketDisconnected);
}

bool WorkerSession::connectToServer(const QString& serverName,
    const aitrain::v2::RequestId& requestId,
    const aitrain::v2::TaskId& taskId)
{
    if (!requestId.isValid() || !taskId.isValid()) {
        return false;
    }
    controlRequestId_ = requestId;
    controlTaskId_ = taskId;
    incomingSequenceTracker_.reset(controlRequestId_);
    socket_.setReadBufferSize(aitrain::v2::kProtocolV2MaxControlMessageBytes + 1);
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
    if (buffer_.size() > aitrain::v2::kProtocolV2MaxControlMessageBytes
        && !buffer_.contains('\n')) {
        rejectControlProtocol(QStringLiteral("Protocol V2 frame exceeds maximum size."));
        return;
    }

    // 同一次读取中可能已经包含多个完整控制帧。必须先无副作用地校验整批帧，
    // 再执行首个 start_task；否则一个业务参数不完整但协议合法的首帧会抢先
    // 产生 worker_failed，掩盖其后已经到达的重复消息或乱序序列错误。
    aitrain::v2::ProtocolV2SequenceTracker preflightTracker = incomingSequenceTracker_;
    bool preflightStartReceived = startTaskReceived_;
    QByteArray preflightBuffer = buffer_;
    int preflightNewline = preflightBuffer.indexOf('\n');
    while (preflightNewline >= 0) {
        const QByteArray line = preflightBuffer.left(preflightNewline);
        preflightBuffer.remove(0, preflightNewline + 1);
        if (line.size() > aitrain::v2::kProtocolV2MaxControlMessageBytes) {
            rejectControlProtocol(QStringLiteral("Protocol V2 frame exceeds maximum size."));
            return;
        }
        aitrain::v2::ProtocolEnvelope envelope;
        QString error;
        if (!aitrain::v2::decodeProtocolV2Message(line, &envelope, &error)
            || !preflightTracker.observe(envelope, controlRequestId_, controlTaskId_, &error)) {
            rejectControlProtocol(QStringLiteral("Protocol V2 command rejected: %1").arg(error));
            return;
        }
        if (envelope.kind == QStringLiteral("command.cancel_task")) {
            if (!preflightStartReceived) {
                rejectControlProtocol(QStringLiteral(
                    "Protocol V2 command.cancel_task cannot precede command.start_task."));
                return;
            }
        } else if (envelope.kind == QStringLiteral("command.start_task")) {
            if (preflightStartReceived) {
                rejectControlProtocol(QStringLiteral(
                    "Protocol V2 command.start_task may only be sent once."));
                return;
            }
            QString businessCommand;
            QJsonObject businessPayload;
            if (!wp::control_v2::unpackStartTask(
                    envelope, &businessCommand, &businessPayload, &error)) {
                rejectControlProtocol(QStringLiteral("Protocol V2 start task rejected: %1").arg(error));
                return;
            }
            preflightStartReceived = true;
        } else {
            rejectControlProtocol(QStringLiteral(
                "Protocol V2 command kind is not allowed: %1").arg(envelope.kind));
            return;
        }
        preflightNewline = preflightBuffer.indexOf('\n');
    }

    int newline = buffer_.indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer_.left(newline);
        buffer_.remove(0, newline + 1);
        if (line.size() > aitrain::v2::kProtocolV2MaxControlMessageBytes) {
            rejectControlProtocol(QStringLiteral("Protocol V2 frame exceeds maximum size."));
            return;
        }

        aitrain::v2::ProtocolEnvelope envelope;
        QString error;
        if (!aitrain::v2::decodeProtocolV2Message(line, &envelope, &error)) {
            rejectControlProtocol(QStringLiteral("Protocol V2 message rejected: %1").arg(error));
            return;
        }
        if (!acceptControlEnvelope(envelope)) {
            return;
        }

        newline = buffer_.indexOf('\n');
    }
}

bool WorkerSession::acceptControlEnvelope(const aitrain::v2::ProtocolEnvelope& envelope)
{
    QString error;
    if (!incomingSequenceTracker_.observe(envelope, controlRequestId_, controlTaskId_, &error)) {
        rejectControlProtocol(QStringLiteral("Protocol V2 command rejected: %1").arg(error));
        return false;
    }

    if (envelope.kind == QStringLiteral("command.cancel_task")) {
        if (!startTaskReceived_) {
            rejectControlProtocol(QStringLiteral("Protocol V2 command.cancel_task cannot precede command.start_task."));
            return false;
        }
        cancelCommand(QJsonObject());
        return true;
    }
    if (envelope.kind != QStringLiteral("command.start_task")) {
        rejectControlProtocol(QStringLiteral("Protocol V2 command kind is not allowed: %1").arg(envelope.kind));
        return false;
    }
    if (startTaskReceived_) {
        rejectControlProtocol(QStringLiteral("Protocol V2 command.start_task may only be sent once."));
        return false;
    }

    QString businessCommand;
    QJsonObject businessPayload;
    if (!wp::control_v2::unpackStartTask(envelope, &businessCommand, &businessPayload, &error)) {
        rejectControlProtocol(QStringLiteral("Protocol V2 start task rejected: %1").arg(error));
        return false;
    }
    startTaskReceived_ = true;
    handleMessage(businessCommand, businessPayload);
    return !finishingSession_;
}

void WorkerSession::rejectControlProtocol(const QString& message)
{
    failWithDetails(message, QStringLiteral("protocol_v2_rejected"));
}

void WorkerSession::handleMessage(const QString& type, const QJsonObject& payload)
{
    // finishSession() 会在 flush 期间处理事件；关闭开始后不得让迟到命令产生第二个终态事件。
    if (finishingSession_) {
        return;
    }
    activeCommand_ = type;
    activeTaskId_ = payload.value(wp::field::taskId()).toString(activeTaskId_);

    const QVector<CommandBinding> bindings = commandBindings();
    for (const CommandBinding& binding : bindings) {
        if (type == binding.command) {
            (this->*binding.handler)(payload);
            return;
        }
    }

    fail(QStringLiteral("Unsupported command: %1").arg(type));
}

void WorkerSession::runEnvironmentCheckWorkflowV2Command(const QJsonObject& payload)
{
    runEnvironmentCheckWorkflowV2(payload);
}

void WorkerSession::runDatasetSplitWorkflowV2Command(const QJsonObject& payload)
{
    runDatasetSplitWorkflowV2(payload);
}

void WorkerSession::runDatasetConversionWorkflowV2Command(const QJsonObject& payload)
{
    runDatasetConversionWorkflowV2(payload);
}

void WorkerSession::runDataQualityWorkflowV2Command(const QJsonObject& payload)
{
    runDataQualityWorkflowV2(payload);
}

void WorkerSession::runDiagnosticsWorkflowV2Command(const QJsonObject& payload)
{
    runDiagnosticsWorkflowV2(payload);
}

void WorkerSession::createAnnotationSessionV2Command(const QJsonObject& payload)
{
    createAnnotationSessionV2(payload);
}

void WorkerSession::syncAnnotationSessionV2Command(const QJsonObject& payload)
{
    syncAnnotationSessionV2(payload);
}

void WorkerSession::runDatasetSnapshotImportWorkflowV2Command(const QJsonObject& payload)
{
    runDatasetSnapshotImportWorkflowV2(payload);
}

void WorkerSession::importOcrOfficialReportsV2Command(const QJsonObject& payload)
{
    importOcrOfficialReportsV2(payload);
}

void WorkerSession::runOcrAcceptanceWorkflowV2Command(const QJsonObject& payload)
{
    runOcrAcceptanceWorkflowV2(payload);
}

void WorkerSession::runRuntimeDeliveryWorkflowV2Command(const QJsonObject& payload)
{
    runRuntimeDeliveryWorkflowV2(payload);
}

void WorkerSession::importModelV2Command(const QJsonObject& payload)
{
    importModelV2(payload);
}

void WorkerSession::runTrainingWorkflowV2Command(const QJsonObject& payload)
{
    runTrainingWorkflowV2(payload);
}

void WorkerSession::cancelCommand(const QJsonObject& payload)
{
    Q_UNUSED(payload);
    if (finishingSession_) {
        return;
    }
    if (trainingWorkspaceV2_ && trainingWorkflowTaskIdV2_.isValid()) {
        cancelTrainingWorkflowV2();
        return;
    }
    if (runtimeDeliveryRunningV2_ && runtimeDeliveryWorkspaceV2_
        && runtimeDeliveryTaskIdV2_.isValid()) {
        // Runtime Delivery Core 在每个同步 Runtime 调用前后轮询该标记。
        // 底层 ONNX Runtime 单次 infer 为同步调用，进入后不能中途抢占；
        // 取消会在该次 infer 返回后收口，不能提前发送第二个终态。
        canceled_ = true;
        return;
    }
    if (annotationRunningV2_ && annotationWorkspaceV2_ && annotationTaskIdV2_.isValid()) {
        canceled_ = true;
        QString ignored;
        annotationWorkspaceV2_->requestTaskCancellation(annotationTaskIdV2_, &ignored);
        return;
    }
    if (ocrAcceptanceRunningV2_ && ocrAcceptanceWorkspaceV2_
        && ocrAcceptanceTaskIdV2_.isValid()) {
        canceled_ = true;
        QString ignored;
        ocrAcceptanceWorkspaceV2_->requestTaskCancellation(ocrAcceptanceTaskIdV2_, &ignored);
        return;
    }
    if (dataQualityRunningV2_ && dataQualityWorkspaceV2_
        && dataQualityTaskIdV2_.isValid()) {
        canceled_ = true;
        QString ignored;
        dataQualityWorkspaceV2_->requestTaskCancellation(dataQualityTaskIdV2_, &ignored);
        return;
    }
    if (diagnosticsRunningV2_ && diagnosticsWorkspaceV2_
        && diagnosticsTaskIdV2_.isValid()) {
        // 外部同步 probe 期间不能抢占；Core 会在每个 probe 前后读取 canceled_。
        canceled_ = true;
        QString ignored;
        diagnosticsWorkspaceV2_->requestTaskCancellation(diagnosticsTaskIdV2_, &ignored);
        return;
    }
    if (datasetConversionRunningV2_ && datasetConversionWorkspaceV2_
        && datasetConversionTaskIdV2_.isValid()) {
        canceled_ = true;
        QString ignored;
        datasetConversionWorkspaceV2_->requestTaskCancellation(datasetConversionTaskIdV2_, &ignored);
        return;
    }
    if (datasetSnapshotImportRunningV2_ && datasetSnapshotImportWorkspaceV2_
        && datasetSnapshotImportTaskIdV2_.isValid()) {
        canceled_ = true;
        QString ignored;
        datasetSnapshotImportWorkspaceV2_->requestTaskCancellation(
            datasetSnapshotImportTaskIdV2_, &ignored);
        return;
    }
    if (datasetSplitRunningV2_ && datasetSplitWorkspaceV2_
        && datasetSplitTaskIdV2_.isValid()) {
        canceled_ = true;
        QString ignored;
        datasetSplitWorkspaceV2_->requestTaskCancellation(datasetSplitTaskIdV2_, &ignored);
        return;
    }
    running_ = false;
    canceled_ = true;
    shutdownPythonTrainer(QStringLiteral("Canceled by user"), true);
    sendCanceledAndFinish(activeTaskId_, QStringLiteral("Canceled by user"));
}

void WorkerSession::handleSocketDisconnected()
{
    if (finishingSession_) {
        return;
    }

    running_ = false;
    canceled_ = true;
    if (trainingWorkspaceV2_ && trainingWorkflowTaskIdV2_.isValid()) {
        QString ignored;
        trainingWorkspaceV2_->requestTaskCancellation(trainingWorkflowTaskIdV2_, &ignored);
    }
    if (annotationWorkspaceV2_ && annotationTaskIdV2_.isValid()) {
        QString ignored;
        annotationWorkspaceV2_->requestTaskCancellation(annotationTaskIdV2_, &ignored);
    }
    if (ocrAcceptanceWorkspaceV2_ && ocrAcceptanceTaskIdV2_.isValid()) {
        QString ignored;
        ocrAcceptanceWorkspaceV2_->requestTaskCancellation(ocrAcceptanceTaskIdV2_, &ignored);
    }
    if (dataQualityWorkspaceV2_ && dataQualityTaskIdV2_.isValid()) {
        QString ignored;
        dataQualityWorkspaceV2_->requestTaskCancellation(dataQualityTaskIdV2_, &ignored);
    }
    if (diagnosticsWorkspaceV2_ && diagnosticsTaskIdV2_.isValid()) {
        QString ignored;
        diagnosticsWorkspaceV2_->requestTaskCancellation(diagnosticsTaskIdV2_, &ignored);
    }
    if (datasetConversionWorkspaceV2_ && datasetConversionTaskIdV2_.isValid()) {
        QString ignored;
        datasetConversionWorkspaceV2_->requestTaskCancellation(datasetConversionTaskIdV2_, &ignored);
    }
    if (datasetSnapshotImportWorkspaceV2_ && datasetSnapshotImportTaskIdV2_.isValid()) {
        QString ignored;
        datasetSnapshotImportWorkspaceV2_->requestTaskCancellation(
            datasetSnapshotImportTaskIdV2_, &ignored);
    }
    if (datasetSplitWorkspaceV2_ && datasetSplitTaskIdV2_.isValid()) {
        QString ignored;
        datasetSplitWorkspaceV2_->requestTaskCancellation(datasetSplitTaskIdV2_, &ignored);
    }
    shutdownPythonTrainer(QStringLiteral("Worker client disconnected."), false);
    qApp->quit();
}

void WorkerSession::send(const QString& type, const QJsonObject& payload)
{
    if (terminalEnvelopeSent_) {
        return;
    }
    QJsonObject envelopePayload = payload;
    if (!activeTaskId_.isEmpty() && !envelopePayload.contains(wp::field::taskId())) {
        envelopePayload.insert(wp::field::taskId(), activeTaskId_);
    }
    if (!activeCommand_.isEmpty() && !envelopePayload.contains(wp::field::command())) {
        envelopePayload.insert(wp::field::command(), activeCommand_);
    }
    if (type == wp::event::completed() && !envelopePayload.contains(wp::field::status())) {
        envelopePayload.insert(wp::field::status(), QStringLiteral("completed"));
    }
    if (type == wp::event::failed() && !envelopePayload.contains(wp::field::status())) {
        envelopePayload.insert(wp::field::status(), QStringLiteral("failed"));
    }
    if (type == wp::event::canceled() && !envelopePayload.contains(wp::field::status())) {
        envelopePayload.insert(wp::field::status(), QStringLiteral("canceled"));
    }
    const bool terminal = wp::isTerminalEvent(type);
    const aitrain::v2::ProtocolEnvelope envelope = wp::control_v2::eventEnvelope(
        controlRequestId_, controlTaskId_, ++outgoingSequence_, type, envelopePayload);
    QString error;
    const QByteArray bytes = aitrain::v2::encodeProtocolV2Message(envelope, &error);
    if (bytes.isEmpty()) {
        qCritical().noquote() << QStringLiteral("Cannot encode Protocol V2 event: %1").arg(error);
        terminalEnvelopeSent_ = true;
        qApp->quit();
        return;
    }
    if (terminal) {
        terminalEnvelopeSent_ = true;
    }
    socket_.write(bytes);
    socket_.flush();
}

aitrain::CancellationCallback WorkerSession::cancellationCallback()
{
    return [this]() {
        return canceled_;
    };
}

aitrain::CancellationCallback WorkerSession::pollingCancellationCallback(int timeoutMs)
{
    return [this, timeoutMs]() {
        return canceled_ || pollPendingCancel(timeoutMs);
    };
}

bool WorkerSession::pollPendingCancel(int timeoutMs)
{
    if (finishingSession_) {
        return false;
    }
    const qint64 deadline = QDateTime::currentMSecsSinceEpoch() + qMax(0, timeoutMs);
    do {
        // readLines() 可能已把 start_task 与紧随其后的 cancel_task 一次性读入
        // buffer_。轮询必须优先消费现有完整帧，不能只观察内核 Socket 缓冲，
        // 否则 ready 前排队的立即取消会被拖到业务执行结束之后。
        if (!buffer_.contains('\n') && !socket_.bytesAvailable()) {
            const qint64 remaining = deadline - QDateTime::currentMSecsSinceEpoch();
            if (remaining <= 0 || !socket_.waitForReadyRead(qMin<qint64>(remaining, 20))) {
                continue;
            }
        }

        if (socket_.bytesAvailable()) {
            buffer_.append(socket_.readAll());
        }
        if (buffer_.size() > aitrain::v2::kProtocolV2MaxControlMessageBytes
            && !buffer_.contains('\n')) {
            rejectControlProtocol(QStringLiteral("Protocol V2 frame exceeds maximum size."));
            return true;
        }
        int newline = buffer_.indexOf('\n');
        while (newline >= 0) {
            const QByteArray line = buffer_.left(newline);
            buffer_.remove(0, newline + 1);
            if (line.size() > aitrain::v2::kProtocolV2MaxControlMessageBytes) {
                rejectControlProtocol(QStringLiteral("Protocol V2 frame exceeds maximum size."));
                return true;
            }

            aitrain::v2::ProtocolEnvelope envelope;
            QString error;
            if (!aitrain::v2::decodeProtocolV2Message(line, &envelope, &error)) {
                rejectControlProtocol(QStringLiteral("Protocol V2 message rejected: %1").arg(error));
                return true;
            }
            if (!acceptControlEnvelope(envelope)) {
                return true;
            }
            if (canceled_ || finishingSession_) {
                return true;
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

    const QString taskId = activeTaskId_;
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
    if (finishingSession_) {
        return;
    }
    running_ = false;
    canceled_ = true;

    QJsonObject payload;
    payload.insert(wp::field::taskId(), taskId);
    payload.insert(wp::field::command(), activeCommand_);
    payload.insert(wp::field::status(), QStringLiteral("canceled"));
    payload.insert(wp::field::errorCode(), QStringLiteral("canceled"));
    payload.insert(wp::field::message(), message.isEmpty() ? QStringLiteral("Canceled by user") : message);
    send(wp::event::canceled(), payload);
    finishSession();
}

void WorkerSession::finishSession()
{
    if (finishingSession_) {
        return;
    }
    finishingSession_ = true;
    activeTaskId_.clear();
    activeCommand_.clear();
    socket_.flush();
    QElapsedTimer timer;
    timer.start();
    // 终态帧在 send() 中已 flush 到本地 socket；这里仅给内核写缓冲一个短暂收尾
    // 窗口。不能等待数秒，否则 Controller 已收到 completed/failed 后仍会看到 Worker
    // 假性运行，既阻塞队列也破坏 Worker 生命周期验收。
    while (socket_.bytesToWrite() > 0 && timer.elapsed() < 500) {
        socket_.waitForBytesWritten(25);
        QCoreApplication::processEvents(QEventLoop::AllEvents, 25);
    }
    if (socket_.state() == QLocalSocket::ConnectedState) {
        socket_.disconnectFromServer();
    }
    qApp->quit();
}

void WorkerSession::fail(const QString& message)
{
    failWithDetails(message, QStringLiteral("worker_failed"));
}

void WorkerSession::failWithDetails(const QString& message, const QString& errorCode, const QJsonObject& details)
{
    if (finishingSession_) {
        return;
    }
    running_ = false;
    QJsonObject payload;
    payload.insert(wp::field::taskId(), activeTaskId_);
    payload.insert(wp::field::command(), activeCommand_);
    payload.insert(wp::field::status(), QStringLiteral("failed"));
    payload.insert(wp::field::errorCode(), errorCode.isEmpty() ? QStringLiteral("worker_failed") : errorCode);
    payload.insert(wp::field::message(), message);
    if (!details.isEmpty()) {
        payload.insert(QStringLiteral("details"), details);
    }
    send(wp::event::failed(), payload);
    finishSession();
}
