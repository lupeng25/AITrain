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
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QThread>
#include <QTimer>

#include <type_traits>

using namespace worker_support;
namespace wp = aitrain::worker_protocol;

namespace {
constexpr qint64 kMaxPendingWorkerControlBytes = 8 * 1024 * 1024;
constexpr int kTerminalSettleMs = 750;
constexpr int kTerminalClientCloseTimeoutMs = 1000;

bool isDroppableControlEvent(const QString& type)
{
    return type == wp::event::log()
        || type == wp::event::progress()
        || type == wp::event::metric();
}
} // namespace

WorkerSession::WorkerSession(QObject* parent)
    : QObject(parent)
{
    connect(&socket_, &QLocalSocket::readyRead, this, &WorkerSession::readLines);
    connect(&socket_, &QLocalSocket::disconnected, this, &WorkerSession::handleSocketDisconnected);
    terminalDrainTimer_.setInterval(25);
    connect(&terminalDrainTimer_, &QTimer::timeout, this, &WorkerSession::maybeFinishSessionAfterWrite);
    connect(&socket_, &QLocalSocket::bytesWritten, this, [this](qint64) {
        maybeFinishSessionAfterWrite();
    });
}

bool WorkerSession::connectToServer(const QString& serverName,
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    const QString& controlToken)
{
    if (!requestId.isValid() || !taskId.isValid() || controlToken.trimmed().isEmpty()) {
        return false;
    }
    controlRequestId_ = requestId;
    controlTaskId_ = taskId;
    controlToken_ = controlToken;
    droppedControlEventCount_ = 0;
    incomingSequenceTracker_.reset(controlRequestId_);
    socket_.setReadBufferSize(aitrain::kProtocolMaxControlMessageBytes + 1);
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
    if (buffer_.size() > aitrain::kProtocolMaxControlMessageBytes
        && !buffer_.contains('\n')) {
        rejectControlProtocol(QStringLiteral("Protocol  frame exceeds maximum size."));
        return;
    }

    // 同一次读取中可能已经包含多个完整控制帧。必须先无副作用地校验整批帧，
    // 再执行首个 start_task；否则一个业务参数不完整但协议合法的首帧会抢先
    // 产生 worker_failed，掩盖其后已经到达的重复消息或乱序序列错误。
    aitrain::ProtocolSequenceTracker preflightTracker = incomingSequenceTracker_;
    bool preflightStartReceived = startTaskReceived_;
    QByteArray preflightBuffer = buffer_;
    int preflightNewline = preflightBuffer.indexOf('\n');
    while (preflightNewline >= 0) {
        const QByteArray line = preflightBuffer.left(preflightNewline);
        preflightBuffer.remove(0, preflightNewline + 1);
        if (line.size() > aitrain::kProtocolMaxControlMessageBytes) {
            rejectControlProtocol(QStringLiteral("Protocol  frame exceeds maximum size."));
            return;
        }
        aitrain::ProtocolEnvelope envelope;
        QString error;
        if (!aitrain::decodeProtocolMessage(line, &envelope, &error)
            || !preflightTracker.observe(envelope, controlRequestId_, controlTaskId_, &error)) {
            rejectControlProtocol(QStringLiteral("Protocol  command rejected: %1").arg(error));
            return;
        }
        if (envelope.controlToken != controlToken_) {
            rejectControlProtocol(QStringLiteral("Protocol  control token mismatch."));
            return;
        }
        if (envelope.kind == QStringLiteral("command.cancel_task")) {
            if (!preflightStartReceived) {
                rejectControlProtocol(QStringLiteral(
                    "Protocol  command.cancel_task cannot precede command.start_task."));
                return;
            }
        } else if (envelope.kind == QStringLiteral("command.start_task")) {
            if (preflightStartReceived) {
                rejectControlProtocol(QStringLiteral(
                    "Protocol  command.start_task may only be sent once."));
                return;
            }
            wp::TaskCommand command;
            if (!wp::control::unpackStartTask(envelope, &command, &error)) {
                rejectControlProtocol(QStringLiteral("Protocol  start task rejected: %1").arg(error));
                return;
            }
            preflightStartReceived = true;
        } else {
            rejectControlProtocol(QStringLiteral(
                "Protocol  command kind is not allowed: %1").arg(envelope.kind));
            return;
        }
        preflightNewline = preflightBuffer.indexOf('\n');
    }

    int newline = buffer_.indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer_.left(newline);
        buffer_.remove(0, newline + 1);
        if (line.size() > aitrain::kProtocolMaxControlMessageBytes) {
            rejectControlProtocol(QStringLiteral("Protocol  frame exceeds maximum size."));
            return;
        }

        aitrain::ProtocolEnvelope envelope;
        QString error;
        if (!aitrain::decodeProtocolMessage(line, &envelope, &error)) {
            rejectControlProtocol(QStringLiteral("Protocol  message rejected: %1").arg(error));
            return;
        }
        if (!acceptControlEnvelope(envelope)) {
            return;
        }

        newline = buffer_.indexOf('\n');
    }
}

bool WorkerSession::acceptControlEnvelope(const aitrain::ProtocolEnvelope& envelope)
{
    QString error;
    if (envelope.controlToken != controlToken_) {
        rejectControlProtocol(QStringLiteral("Protocol  control token mismatch."));
        return false;
    }
    if (!incomingSequenceTracker_.observe(envelope, controlRequestId_, controlTaskId_, &error)) {
        rejectControlProtocol(QStringLiteral("Protocol  command rejected: %1").arg(error));
        return false;
    }

    if (envelope.kind == QStringLiteral("command.cancel_task")) {
        if (!startTaskReceived_) {
            rejectControlProtocol(QStringLiteral("Protocol  command.cancel_task cannot precede command.start_task."));
            return false;
        }
        cancelCommand();
        return true;
    }
    if (envelope.kind != QStringLiteral("command.start_task")) {
        rejectControlProtocol(QStringLiteral("Protocol  command kind is not allowed: %1").arg(envelope.kind));
        return false;
    }
    if (startTaskReceived_) {
        rejectControlProtocol(QStringLiteral("Protocol  command.start_task may only be sent once."));
        return false;
    }

    wp::TaskCommand command;
    if (!wp::control::unpackStartTask(envelope, &command, &error)) {
        rejectControlProtocol(QStringLiteral("Protocol  start task rejected: %1").arg(error));
        return false;
    }
    startTaskReceived_ = true;
    handleCommand(command);
    return !finishingSession_;
}

void WorkerSession::rejectControlProtocol(const QString& message)
{
    failWithDetails(message, QStringLiteral("protocol_rejected"));
}

void WorkerSession::handleCommand(const wp::TaskCommand& command)
{
    // finishSession() 会在 flush 期间处理事件；关闭开始后不得让迟到命令产生第二个终态事件。
    if (finishingSession_) {
        return;
    }
    activeCommand_ = wp::taskCommandType(command);
    activeTaskId_ = std::visit([](const auto& value) {
        return value.context.taskId.toString();
    }, command.payload);
    std::visit([this](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, wp::EnvironmentCheckCommand>) {
            runEnvironmentCheckWorkflow(value);
        } else if constexpr (std::is_same_v<T, wp::DatasetSplitCommand>) {
            runDatasetSplitWorkflow(value);
        } else if constexpr (std::is_same_v<T, wp::DatasetConversionCommand>) {
            runDatasetConversionWorkflow(value);
        } else if constexpr (std::is_same_v<T, wp::DataQualityCommand>) {
            runDataQualityWorkflow(value);
        } else if constexpr (std::is_same_v<T, wp::AnnotationSessionCreateCommand>) {
            createAnnotationSession(value);
        } else if constexpr (std::is_same_v<T, wp::AnnotationSessionSyncCommand>) {
            syncAnnotationSession(value);
        } else if constexpr (std::is_same_v<T, wp::DatasetSnapshotImportCommand>) {
            runDatasetSnapshotImportWorkflow(value);
        } else if constexpr (std::is_same_v<T, wp::OcrOfficialReportImportCommand>) {
            importOcrOfficialReports(value);
        } else if constexpr (std::is_same_v<T, wp::OcrAcceptanceCommand>) {
            runOcrAcceptanceWorkflow(value);
        } else if constexpr (std::is_same_v<T, wp::DiagnosticsCommand>) {
            runDiagnosticsWorkflow(value);
        } else if constexpr (std::is_same_v<T, wp::ExternalAcceptanceEvidenceImportCommand>) {
            importExternalAcceptanceEvidence(value);
        } else if constexpr (std::is_same_v<T, wp::RuntimeDeliveryCommand>) {
            runRuntimeDeliveryWorkflow(value);
        } else if constexpr (std::is_same_v<T, wp::ModelImportCommand>) {
            importModel(value);
        } else if constexpr (std::is_same_v<T, wp::TrainingCommand>) {
            runTrainingWorkflow(value);
        }
    }, command.payload);
}

void WorkerSession::cancelCommand()
{
    if (finishingSession_) {
        return;
    }
    if (requestCancellationForActiveWorkflow()) {
        return;
    }
    running_ = false;
    canceled_ = true;
    shutdownPythonTrainer(QStringLiteral("Canceled by user"), true);
    sendCanceledAndFinish(activeTaskId_, QStringLiteral("Canceled by user"));
}

bool WorkerSession::requestCancellationForActiveWorkflow()
{
    if (trainingWorkspace_ && trainingWorkflowTaskId_.isValid()) {
        cancelTrainingWorkflow();
        return true;
    }
    if (runtimeDeliveryRunning_ && runtimeDeliveryWorkspace_
        && runtimeDeliveryTaskId_.isValid()) {
        // Runtime Delivery Core 在每个同步 Runtime 调用前后轮询该标记。
        // 底层 ONNX Runtime 单次 infer 为同步调用，进入后不能中途抢占；
        // 取消会在该次 infer 返回后收口，不能提前发送第二个终态。
        canceled_ = true;
        return true;
    }

    const auto request = [this](aitrain::ProjectWorkspace* workspace,
                                const aitrain::TaskId& taskId) {
        if (!workspace || !taskId.isValid()) {
            return false;
        }
        canceled_ = true;
        QString ignored;
        workspace->requestTaskCancellation(taskId, &ignored);
        return true;
    };
    if (annotationRunning_ && request(annotationWorkspace_.get(), annotationTaskId_)) return true;
    if (ocrAcceptanceRunning_ && request(ocrAcceptanceWorkspace_.get(), ocrAcceptanceTaskId_)) return true;
    if (dataQualityRunning_ && request(dataQualityWorkspace_.get(), dataQualityTaskId_)) return true;
    if (diagnosticsRunning_ && request(diagnosticsWorkspace_.get(), diagnosticsTaskId_)) return true;
    if (datasetConversionRunning_ && request(datasetConversionWorkspace_.get(), datasetConversionTaskId_)) return true;
    if (datasetSnapshotImportRunning_
        && request(datasetSnapshotImportWorkspace_.get(), datasetSnapshotImportTaskId_)) return true;
    if (datasetSplitRunning_ && request(datasetSplitWorkspace_.get(), datasetSplitTaskId_)) return true;
    return false;
}

void WorkerSession::requestCancellationForTrackedWorkflows()
{
    const auto request = [](aitrain::ProjectWorkspace* workspace,
                            const aitrain::TaskId& taskId) {
        if (!workspace || !taskId.isValid()) {
            return;
        }
        QString ignored;
        workspace->requestTaskCancellation(taskId, &ignored);
    };

    // 断线是本地生命周期事件，不再根据各 handler 的 running 标记分叉；
    // 只要 workspace 已经绑定了任务身份，就统一发出取消请求。这样在
    // “startTask 成功、running 标记尚未置位”的窄窗口内也不会遗留活动任务。
    request(trainingWorkspace_.get(), trainingWorkflowTaskId_);
    request(runtimeDeliveryWorkspace_.get(), runtimeDeliveryTaskId_);
    request(annotationWorkspace_.get(), annotationTaskId_);
    request(ocrAcceptanceWorkspace_.get(), ocrAcceptanceTaskId_);
    request(dataQualityWorkspace_.get(), dataQualityTaskId_);
    request(diagnosticsWorkspace_.get(), diagnosticsTaskId_);
    request(datasetConversionWorkspace_.get(), datasetConversionTaskId_);
    request(datasetSnapshotImportWorkspace_.get(), datasetSnapshotImportTaskId_);
    request(datasetSplitWorkspace_.get(), datasetSplitTaskId_);
}

void WorkerSession::handleSocketDisconnected()
{
    if (finishingSession_) {
        terminalDrainTimer_.stop();
        terminalQuitScheduled_ = false;
        qApp->quit();
        return;
    }

    running_ = false;
    canceled_ = true;
    requestCancellationForTrackedWorkflows();
    shutdownPythonTrainer(QStringLiteral("Worker client disconnected."), false);
    qApp->quit();
}

void WorkerSession::send(const QString& type, const QJsonObject& payload)
{
    if (terminalEnvelopeSent_) {
        return;
    }
    const bool terminal = wp::isTerminalEvent(type);
    if (!terminal && isDroppableControlEvent(type)
        && socket_.bytesToWrite() > kMaxPendingWorkerControlBytes) {
        ++droppedControlEventCount_;
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
    if (terminal && droppedControlEventCount_ > 0) {
        envelopePayload.insert(QStringLiteral("droppedControlEventCount"),
            QString::number(droppedControlEventCount_));
    }
    const wp::TaskEvent eventValue = wp::taskEventFromType(type, envelopePayload);
    const aitrain::ProtocolEnvelope envelope = wp::control::eventEnvelope(
        controlRequestId_, controlTaskId_, ++outgoingSequence_, eventValue, controlToken_);
    QString error;
    const QByteArray bytes = aitrain::encodeProtocolMessage(envelope, &error);
    if (bytes.isEmpty()) {
        qCritical().noquote() << QStringLiteral("Cannot encode Protocol  event: %1").arg(error);
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
        if (buffer_.size() > aitrain::kProtocolMaxControlMessageBytes
            && !buffer_.contains('\n')) {
            rejectControlProtocol(QStringLiteral("Protocol  frame exceeds maximum size."));
            return true;
        }
        int newline = buffer_.indexOf('\n');
        while (newline >= 0) {
            const QByteArray line = buffer_.left(newline);
            buffer_.remove(0, newline + 1);
            if (line.size() > aitrain::kProtocolMaxControlMessageBytes) {
                rejectControlProtocol(QStringLiteral("Protocol  frame exceeds maximum size."));
                return true;
            }

            aitrain::ProtocolEnvelope envelope;
            QString error;
            if (!aitrain::decodeProtocolMessage(line, &envelope, &error)) {
                rejectControlProtocol(QStringLiteral("Protocol  message rejected: %1").arg(error));
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
    // QLocalSocket::bytesToWrite()==0 只说明数据已交给系统缓冲；在 Worker
    // 即将退出的窗口中显式等待一次有界的 bytesWritten，避免短工作流的
    // 终态帧还未被对端读取就随进程销毁。该等待仅发生在终态收尾，不阻塞
    // GUI 线程或训练步骤。
    socket_.waitForBytesWritten(1000);
    // 不使用嵌套事件循环；由 bytesWritten 驱动确认终态及其之前的帧已经
    // 离开 QLocalSocket 用户态缓冲。只有确认写空或明确超时才退出 Worker。
    terminalDrainElapsed_.restart();
    terminalQuitScheduled_ = false;
    terminalDrainTimer_.start();
    maybeFinishSessionAfterWrite();
}

void WorkerSession::maybeFinishSessionAfterWrite()
{
    if (!finishingSession_) {
        return;
    }
    if (socket_.bytesToWrite() == 0) {
        terminalDrainTimer_.stop();
        if (!terminalQuitScheduled_) {
            terminalQuitScheduled_ = true;
            // bytesToWrite()==0 只代表 Qt 已把帧交给系统 socket，不能证明
            // 对端已经完成 readyRead。保留与原有 Worker 生命周期一致的
            // 750ms 异步稳定窗口，避免短工作流在慢消费者上丢失终态帧。
            const int remainingMs = qMax(0, kTerminalSettleMs
                - static_cast<int>(terminalDrainElapsed_.elapsed()));
            QTimer::singleShot(remainingMs, qApp, [this] {
                if (!finishingSession_) {
                    return;
                }
                // 不由 Worker 主动发 FIN：QLocalSocket 的 disconnected 事件
                // 可能先于对端 readyRead 到达，导致 WorkerClient 在进程退出
                // 时无法再排空终态帧。让已收到终态的 WorkerClient 主动断开；
                // 对异常客户端保留有界兜底，避免 Worker 永久驻留。
                if (socket_.state() != QLocalSocket::ConnectedState) {
                    qApp->quit();
                    return;
                }
                QTimer::singleShot(kTerminalClientCloseTimeoutMs, qApp, [this] {
                    if (finishingSession_) {
                        qApp->quit();
                    }
                });
            });
        }
        return;
    }
    if (terminalDrainElapsed_.isValid() && terminalDrainElapsed_.elapsed() >= 5000) {
        qCritical().noquote() << QStringLiteral(
            "Worker terminal frame drain timed out with %1 bytes pending.").arg(socket_.bytesToWrite());
        terminalDrainTimer_.stop();
        qApp->exit(5);
    }
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
