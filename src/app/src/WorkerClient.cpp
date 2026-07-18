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
    }
}

bool WorkerClient::startTask(const QString& workerProgram,
    const wp::TaskCommand& command,
    QString* error)
{
    return startWorkerCommand(workerProgram, command, error);
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

bool WorkerClient::startWorkerCommand(const QString& workerProgram,
    const wp::TaskCommand& command,
    QString* error)
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

    const aitrain::TaskId requestedTaskId = std::visit(
        [](const auto& value) { return value.context.taskId; }, command.payload);
    if (!requestedTaskId.isValid()) {
        if (error) {
            *error = QStringLiteral("TaskCommand context.taskId 必须是有效 UUID。");
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
    pendingCommand_ = command;
    activeRequestId_ = aitrain::RequestId::create();
    controlToken_ = QUuid::createUuid().toString(QUuid::Id128);
    activeTaskId_ = requestedTaskId;
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
        QStringLiteral("--task-id"), activeTaskId_.toString(),
        QStringLiteral("--control-token"), controlToken_});
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
        rejectProtocol(QStringLiteral("Protocol frame exceeds maximum size."));
        return;
    }

    int newline = buffer_.indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer_.left(newline + 1);
        buffer_.remove(0, newline + 1);
        if (line.size() > aitrain::kProtocolMaxControlMessageBytes) {
        rejectProtocol(QStringLiteral("Protocol frame exceeds maximum size."));
            return;
        }

        aitrain::ProtocolEnvelope envelope;
        QString error;
        if (!aitrain::decodeProtocolMessage(line, &envelope, &error)
            || !incomingSequenceTracker_.observe(envelope, activeRequestId_, activeTaskId_, &error)) {
        rejectProtocol(QStringLiteral("Protocol message rejected: %1").arg(error));
            return;
        }
        if (envelope.controlToken != controlToken_) {
        rejectProtocol(QStringLiteral("Protocol control token mismatch."));
            return;
        }
        if (terminalEnvelopeReceived_) {
        rejectProtocol(QStringLiteral("Protocol event received after terminal event."));
            return;
        }

        wp::TaskEvent decodedEvent;
        if (!wp::control::unpackTaskEvent(envelope, &decodedEvent, &error)) {
        rejectProtocol(QStringLiteral("Protocol event rejected: %1").arg(error));
            return;
        }
        const QString type = wp::taskEventType(decodedEvent);
        const QJsonObject payload = decodedEvent.details;

        const bool terminal = envelope.kind == QStringLiteral("event.succeeded")
            || envelope.kind == QStringLiteral("event.failed")
            || envelope.kind == QStringLiteral("event.canceled");
        if (terminal) {
            terminalEnvelopeReceived_ = true;
        }

        publishEvent(decodedEvent);
        if (type == wp::event::ready()) {
            workerReady_ = true;
            connectionTimer_.stop();
            if (pendingCommand_.has_value()) {
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
    terminalDrainAttempted_ = false;
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
    if (socket_ && !finishedEmitted_ && socket_->state() == QLocalSocket::ConnectedState
        && !terminalDrainAttempted_) {
        // QLocalSocket 的 readyRead 由事件循环异步投递；不要在 GUI 线程
        // waitForReadyRead，给最后一帧一个有限的非阻塞排空窗口即可。
        terminalDrainAttempted_ = true;
        QTimer::singleShot(300, this, &WorkerClient::finalizeWorkerExit);
        return;
    }
    if (!finishedEmitted_) {
        finishedEmitted_ = true;
        const aitrain::TaskId lostTaskId = activeTaskId_;
        const QString message = cancelRequested_
            ? QStringLiteral("Worker 在取消请求后退出，未收到正式终态。")
            : (pendingExitCode_ < 0
                ? QStringLiteral("Worker failed to start: %1").arg(process_.errorString())
                : (pendingExitStatus_ != QProcess::NormalExit || pendingExitCode_ != 0)
                ? QStringLiteral("Worker exited with code %1").arg(pendingExitCode_)
                : QStringLiteral("Worker exited without a terminal status message"));
        QJsonObject payload;
        payload.insert(wp::field::taskId(), lostTaskId.toString());
        payload.insert(wp::field::command(), pendingCommand_.has_value()
            ? wp::taskCommandType(*pendingCommand_) : QString());
        payload.insert(wp::field::status(), QStringLiteral("worker_lost"));
        payload.insert(wp::field::errorCode(), cancelRequested_ ? QStringLiteral("worker_lost")
            : QStringLiteral("process_crashed"));
        payload.insert(wp::field::message(), message);
        publishEvent(wp::taskEventFromType(wp::event::failed(), payload));
        emit workerLost(lostTaskId);
        if (cancelRequested_) {
            // 没有收到 Worker 的正式 canceled 终态时，不能伪造业务取消；
            // 任务可能仍停留在 CancelRequested，必须显式报告 Worker 丢失。
            emit finished(WorkerTerminalStatus::Failed, message);
        } else {
            emit finished(WorkerTerminalStatus::Failed, message);
        }
    }
    cancelRequested_ = false;
    cancelTimer_.stop();
    connectionTimer_.stop();
    terminalShutdownTimer_.stop();
    cleanupSocket();
    server_.close();
    pendingCommand_.reset();
    activeRequestId_ = aitrain::RequestId();
    activeTaskId_ = aitrain::TaskId();
    controlToken_.clear();
    incomingSequenceTracker_.clear();
    outgoingSequence_ = 0;
    startTaskSent_ = false;
    terminalEnvelopeReceived_ = false;
    workerReady_ = false;
    finishing_ = false;
    terminalDrainAttempted_ = false;
    QTimer::singleShot(0, this, [this]() {
        emit idle();
    });
}

void WorkerClient::sendStartTask()
{
    const aitrain::ProtocolEnvelope envelope = wp::control::startTaskEnvelope(
        activeRequestId_, activeTaskId_, ++outgoingSequence_, *pendingCommand_,
        controlToken_);
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
        activeRequestId_, activeTaskId_, ++outgoingSequence_, controlToken_);
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
        const aitrain::TaskId lostTaskId = activeTaskId_;
        QJsonObject payload;
        payload.insert(wp::field::taskId(), lostTaskId.toString());
        payload.insert(wp::field::command(), pendingCommand_.has_value()
            ? wp::taskCommandType(*pendingCommand_) : QString());
        payload.insert(wp::field::status(), QStringLiteral("failed"));
        payload.insert(wp::field::errorCode(), QStringLiteral("protocol_rejected"));
        payload.insert(wp::field::message(), message);
        publishEvent(wp::taskEventFromType(wp::event::failed(), payload));
        emit workerLost(lostTaskId);
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

void WorkerClient::publishEvent(const wp::TaskEvent& eventValue)
{
    emit taskEventReceived(eventValue);
}
