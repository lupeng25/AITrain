#include "aitrain/process/PythonAdapterHost.h"

#include <QFile>
#include <QProcess>
#include <QTemporaryDir>
#include <QTimer>

#include <limits>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace aitrain {

namespace {
constexpr int kProcessOutputTailBytes = 16 * 1024;
}

PythonAdapterHost::PythonAdapterHost() = default;

PythonAdapterHost::~PythonAdapterHost()
{
    stop();
}

bool PythonAdapterHost::start(const PythonAdapterLaunch& launch,
    const RequestId& requestId,
    const TaskId& taskId,
    EventHandler eventHandler,
    ExitHandler exitHandler,
    QString* error)
{
    stop();
    if (launch.program.trimmed().isEmpty()) {
        if (error) {
            *error = QStringLiteral("启动 Python Adapter 需要可执行程序路径。");
        }
        return false;
    }
    if (launch.cancellationGraceMs < 1) {
        if (error) {
            *error = QStringLiteral("Python Adapter 取消宽限期必须为正数。");
        }
        return false;
    }
    if (launch.eventDrainTimeoutMs < 1) {
        if (error) {
            *error = QStringLiteral("Python Adapter 事件排空窗口必须为正数。");
        }
        return false;
    }
    if (!processTree_.create(error) || !eventServer_.start(requestId, taskId, error)) {
        processTree_.reset();
        return false;
    }

    cancellationDirectory_ = std::make_unique<QTemporaryDir>();
    if (!cancellationDirectory_->isValid()) {
        if (error) {
            *error = QStringLiteral("无法创建 Python Adapter 取消信号目录。");
        }
        stop();
        return false;
    }

    const AdapterEventEndpoint endpoint = eventServer_.endpoint();
    QProcessEnvironment environment = launch.environment.isEmpty()
        ? QProcessEnvironment::systemEnvironment()
        : launch.environment;
    environment.insert(QStringLiteral("AITRAIN_EVENT_HOST"), endpoint.host);
    environment.insert(QStringLiteral("AITRAIN_EVENT_PORT"), QString::number(endpoint.port));
    environment.insert(QStringLiteral("AITRAIN_EVENT_TOKEN"), endpoint.token);
    environment.insert(QStringLiteral("AITRAIN_REQUEST_ID"), endpoint.requestId.toString());
    environment.insert(QStringLiteral("AITRAIN_TASK_ID"), endpoint.taskId.toString());
    environment.insert(QStringLiteral("AITRAIN_CANCEL_FILE"), cancellationDirectory_->filePath(QStringLiteral("cancel.request")));

    eventHandler_ = std::move(eventHandler);
    exitHandler_ = std::move(exitHandler);
    cancelRequested_ = false;
    forceTerminated_ = false;
    terminalEventSeen_ = false;
    exitEmitted_ = false;
    drainState_ = DrainState::Running;
    pendingExit_.reset();
    drainFinalizeScheduled_ = false;
    eventSequenceOffset_ = launch.eventSequenceOffset;
    eventDrainTimeoutMs_ = launch.eventDrainTimeoutMs;
    lifecycleError_.clear();
    processOutputTail_.clear();
    processOutputDroppedBytes_ = 0;
    process_ = std::make_unique<QProcess>();
    process_->setProgram(launch.program);
    process_->setArguments(launch.arguments);
    process_->setWorkingDirectory(launch.workingDirectory);
    process_->setProcessEnvironment(environment);
#ifdef Q_OS_WIN
    // The root process must not execute user/official adapter code before it
    // belongs to the Job Object. Otherwise it can create a child in the short
    // window between CreateProcess and QProcess::started, and that child will
    // not be retroactively added to the Job.
    process_->setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* arguments) {
        arguments->flags |= CREATE_SUSPENDED;
    });
#endif
    QObject::connect(process_.get(), &QProcess::readyReadStandardOutput, process_.get(), [this] {
        drainProcessOutput();
    });
    QObject::connect(process_.get(), &QProcess::readyReadStandardError, process_.get(), [this] {
        drainProcessOutput();
    });
    QObject::connect(process_.get(), &QProcess::started, process_.get(), [this] {
        onProcessStarted();
    });
    QObject::connect(process_.get(), QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), process_.get(),
        [this](int exitCode, QProcess::ExitStatus exitStatus) {
            onProcessFinished(exitCode, exitStatus);
        });
    QObject::connect(process_.get(), &QProcess::errorOccurred, process_.get(), [this](QProcess::ProcessError processError) {
        onProcessError(processError);
    });
    cancellationTimer_ = std::make_unique<QTimer>();
    cancellationTimer_->setSingleShot(true);
    cancellationTimer_->setInterval(launch.cancellationGraceMs);
    QObject::connect(cancellationTimer_.get(), &QTimer::timeout, cancellationTimer_.get(), [this] {
        QString ignored;
        forceTerminate(&ignored);
    });
    drainTimer_ = std::make_unique<QTimer>();
    drainTimer_->setSingleShot(true);
    QObject::connect(drainTimer_.get(), &QTimer::timeout, drainTimer_.get(), [this] {
        finalizeAfterDrainDeadline();
    });
    eventServer_.setEventHandler([this](const ProtocolEnvelope& event) {
        return onAdapterEvent(event);
    });
    running_ = true;
    process_->start();
    return true;
}

bool PythonAdapterHost::requestCancellation(QString* error)
{
    if (!running_ || !process_) {
        if (error) {
            *error = QStringLiteral("Python Adapter 当前未运行，无法请求取消。");
        }
        return false;
    }
    if (cancelRequested_) {
        return true;
    }
    const QString cancelFile = cancellationDirectory_->filePath(QStringLiteral("cancel.request"));
    QFile file(cancelFile);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("无法写入 Python Adapter 取消信号文件：%1").arg(file.errorString());
        }
        return false;
    }
    file.write("cancel\n");
    file.close();
    cancelRequested_ = true;
    cancellationTimer_->start();
    return true;
}

bool PythonAdapterHost::forceTerminate(QString* error)
{
    if (!running_ || !process_) {
        return true;
    }
    forceTerminated_ = true;
    if (processTree_.terminate(error)) {
        return true;
    }
    process_->kill();
    return true;
}

bool PythonAdapterHost::isRunning() const
{
    return running_;
}

AdapterEventEndpoint PythonAdapterHost::endpoint() const
{
    return eventServer_.endpoint();
}

void PythonAdapterHost::onProcessStarted()
{
    QString error;
    if (!processTree_.attach(process_.get(), &error)
        || !processTree_.resume(process_.get(), &error)) {
        lifecycleError_ = error;
        forceTerminate(nullptr);
    }
}

void PythonAdapterHost::drainProcessOutput()
{
    if (!process_) {
        return;
    }
    appendProcessOutput(process_->readAllStandardOutput(), QStringLiteral("stdout"));
    appendProcessOutput(process_->readAllStandardError(), QStringLiteral("stderr"));
}

void PythonAdapterHost::appendProcessOutput(const QByteArray& bytes, const QString& channel)
{
    if (bytes.isEmpty()) {
        return;
    }
    QByteArray normalized = bytes;
    normalized.replace("\r\n", "\n");
    normalized.replace('\r', '\n');
    QByteArray combined = processOutputTail_;
    if (!combined.isEmpty() && !combined.endsWith('\n')) {
        combined.append('\n');
    }
    combined.append(QStringLiteral("[%1] ").arg(channel).toUtf8());
    combined.append(normalized);
    if (combined.size() > kProcessOutputTailBytes) {
        processOutputDroppedBytes_ += static_cast<quint64>(combined.size() - kProcessOutputTailBytes);
        combined = combined.right(kProcessOutputTailBytes);
    }
    processOutputTail_ = combined;
}

bool PythonAdapterHost::onAdapterEvent(const ProtocolEnvelope& event)
{
    if (event.sequence > std::numeric_limits<quint64>::max() - eventSequenceOffset_) {
        lifecycleError_ = QStringLiteral("Python Adapter 事件序号溢出。");
        QString ignored;
        forceTerminate(&ignored);
        return false;
    }
    ProtocolEnvelope normalized = event;
    normalized.sequence += eventSequenceOffset_;
    const bool terminal = normalized.kind == QStringLiteral("event.succeeded")
        || normalized.kind == QStringLiteral("event.failed")
        || normalized.kind == QStringLiteral("event.canceled");
    if (eventHandler_ && !eventHandler_(normalized)) {
        lifecycleError_ = QStringLiteral("Python Adapter 事件未被下游持久化接受。");
        QString ignored;
        forceTerminate(&ignored);
        return false;
    }
    // 只有下游已经持久化接受终态后，才把它计入 Host 生命周期。若终态
    // 收口失败，finishAdapter() 必须继续合成明确的失败终态，不能因为
    // 看见过终态帧而留下 Worker 无终态。
    if (terminal) {
        terminalEventSeen_ = true;
    }
    // QProcess::finished 与 QTcpSocket::readyRead 属于两个独立的事件源。
    // 若进程先退出，终态帧可能在 finished 回调返回后才到达；此时必须
    // 在同一 drain 状态内立即收口，而不是依赖固定的短暂 singleShot。
    if (terminalEventSeen_ && drainState_ == DrainState::WaitingForTerminal
        && pendingExit_ && !drainFinalizeScheduled_) {
        drainFinalizeScheduled_ = true;
        // AdapterEventServer 正在其 readSocket() 回调中调用本函数。必须等
        // 当前 readyRead/readLine 循环返回后再 stop() 并释放 socket，否则
        // readSocket() 会继续访问已释放的 QTcpSocket。
        QTimer::singleShot(0, [this] {
            drainFinalizeScheduled_ = false;
            if (drainState_ != DrainState::WaitingForTerminal || !pendingExit_) {
                return;
            }
            const PythonAdapterExit outcome = *pendingExit_;
            pendingExit_.reset();
            finalizeProcessExit(outcome);
        });
    }
    return true;
}

void PythonAdapterHost::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    drainProcessOutput();
    if (cancellationTimer_) {
        cancellationTimer_->stop();
    }
    running_ = false;
    PythonAdapterExit outcome;
    outcome.exitCode = exitCode;
    outcome.normalExit = exitStatus == QProcess::NormalExit;
    outcome.cancelRequested = cancelRequested_;
    outcome.forceTerminated = forceTerminated_;
    outcome.terminalEventSeen = terminalEventSeen_;
    beginProcessDrain(outcome);
}

void PythonAdapterHost::beginProcessDrain(PythonAdapterExit outcome)
{
    if (exitEmitted_ || drainState_ == DrainState::Finalized) {
        return;
    }

    pendingExit_ = outcome;
    running_ = false;
    if (forceTerminated_ || terminalEventSeen_ || !eventServer_.lastError().isEmpty()) {
        const PythonAdapterExit finalOutcome = *pendingExit_;
        pendingExit_.reset();
        finalizeProcessExit(finalOutcome);
        return;
    }

    drainState_ = DrainState::WaitingForTerminal;
    if (drainTimer_) {
        drainTimer_->start(eventDrainTimeoutMs_);
    } else {
        // Defensive fallback for a partially constructed host. start() always
        // creates the timer, but a bounded direct finalize is safer than
        // leaving the worker alive forever if construction changes later.
        finalizeAfterDrainDeadline();
    }
}

void PythonAdapterHost::finalizeAfterDrainDeadline()
{
    if (drainState_ != DrainState::WaitingForTerminal || !pendingExit_) {
        return;
    }
    const PythonAdapterExit outcome = *pendingExit_;
    pendingExit_.reset();
    finalizeProcessExit(outcome);
}

void PythonAdapterHost::finalizeProcessExit(PythonAdapterExit outcome)
{
    if (exitEmitted_) return;
    if (drainTimer_) {
        drainTimer_->stop();
    }
    drainState_ = DrainState::Finalized;
    pendingExit_.reset();
    drainFinalizeScheduled_ = false;
    outcome.cancelRequested = cancelRequested_;
    outcome.forceTerminated = forceTerminated_;
    outcome.terminalEventSeen = terminalEventSeen_;
    if (!lifecycleError_.isEmpty()) {
        // lifecycleError_ 可能来自第三方进程或 Qt，禁止把其原始文本（尤其是
        // Windows 物理路径）跨越 Worker/Workflow 边界持久化或展示。
        outcome.diagnostic = QStringLiteral("Python Adapter 生命周期失败。详见受控 Worker 日志。");
    } else if (!eventServer_.lastError().isEmpty()) {
        outcome.diagnostic = QStringLiteral("Adapter 事件通道失败。详见受控 Worker 日志。");
    } else if (!outcome.cancelRequested && !outcome.terminalEventSeen) {
        outcome.diagnostic = QStringLiteral("Python Adapter 在未发送终态事件时退出。");
    } else {
        outcome.diagnostic.clear();
    }
    eventServer_.stop();
    processTree_.reset();
    emitExitOnce(outcome);
}

void PythonAdapterHost::onProcessError(QProcess::ProcessError processError)
{
    if (processError != QProcess::FailedToStart || exitEmitted_) {
        return;
    }
    running_ = false;
    drainState_ = DrainState::Finalized;
    pendingExit_.reset();
    drainFinalizeScheduled_ = false;
    PythonAdapterExit outcome;
    outcome.cancelRequested = cancelRequested_;
    outcome.forceTerminated = forceTerminated_;
    outcome.terminalEventSeen = terminalEventSeen_;
    // QProcess::errorString() 可能包含可执行文件或工作目录的物理路径，
    // 这里只返回稳定分类，避免泄漏到 Workflow failure/evidence。
    outcome.diagnostic = QStringLiteral("Python Adapter 无法启动。详见受控 Worker 日志。");
    eventServer_.stop();
    processTree_.reset();
    emitExitOnce(outcome);
}

void PythonAdapterHost::emitExitOnce(const PythonAdapterExit& outcome)
{
    if (exitEmitted_) {
        return;
    }
    exitEmitted_ = true;
    if (exitHandler_) {
        exitHandler_(outcome);
    }
}

void PythonAdapterHost::stop()
{
    if (cancellationTimer_) {
        cancellationTimer_->stop();
        cancellationTimer_.reset();
    }
    if (drainTimer_) {
        drainTimer_->stop();
        drainTimer_.reset();
    }
    if (process_ && process_->state() != QProcess::NotRunning) {
        QString ignored;
        processTree_.terminate(&ignored);
        process_->kill();
        process_->waitForFinished(1000);
    }
    process_.reset();
    eventServer_.stop();
    processTree_.reset();
    cancellationDirectory_.reset();
    running_ = false;
    drainState_ = DrainState::Idle;
    pendingExit_.reset();
    drainFinalizeScheduled_ = false;
    eventDrainTimeoutMs_ = 1000;
}

} // namespace aitrain
