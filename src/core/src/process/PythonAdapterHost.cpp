#include "aitrain/process/PythonAdapterHost.h"

#include <QFile>
#include <QProcess>
#include <QTemporaryDir>
#include <QTimer>

#include <limits>

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
    eventSequenceOffset_ = launch.eventSequenceOffset;
    lifecycleError_.clear();
    processOutputTail_.clear();
    processOutputDroppedBytes_ = 0;
    process_ = std::make_unique<QProcess>();
    process_->setProgram(launch.program);
    process_->setArguments(launch.arguments);
    process_->setWorkingDirectory(launch.workingDirectory);
    process_->setProcessEnvironment(environment);
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
    QObject::connect(cancellationTimer_.get(), &QTimer::timeout, cancellationTimer_.get(), [this] {
        QString ignored;
        forceTerminate(&ignored);
    });
    eventServer_.setEventHandler([this](const ProtocolEnvelope& event) {
        onAdapterEvent(event);
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
    if (!processTree_.attach(process_.get(), &error)) {
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

void PythonAdapterHost::onAdapterEvent(const ProtocolEnvelope& event)
{
    if (event.sequence > std::numeric_limits<quint64>::max() - eventSequenceOffset_) {
        lifecycleError_ = QStringLiteral("Python Adapter 事件序号溢出。");
        QString ignored;
        forceTerminate(&ignored);
        return;
    }
    ProtocolEnvelope normalized = event;
    normalized.sequence += eventSequenceOffset_;
    if (normalized.kind == QStringLiteral("event.succeeded")
        || normalized.kind == QStringLiteral("event.failed")
        || normalized.kind == QStringLiteral("event.canceled")) {
        terminalEventSeen_ = true;
    }
    if (eventHandler_) {
        eventHandler_(normalized);
    }
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
    // Python may close immediately after writing its terminal frame.  The
    // event server receives that frame on a separate Qt socket callback, so
    // defer finalization briefly to let the authenticated frame drain.
    if (!terminalEventSeen_ && !forceTerminated_ && cancellationTimer_) {
        QTimer::singleShot(100, cancellationTimer_.get(), [this, outcome] {
            finalizeProcessExit(outcome);
        });
        return;
    }
    finalizeProcessExit(outcome);
}

void PythonAdapterHost::finalizeProcessExit(PythonAdapterExit outcome)
{
    if (exitEmitted_) return;
    outcome.cancelRequested = cancelRequested_;
    outcome.forceTerminated = forceTerminated_;
    outcome.terminalEventSeen = terminalEventSeen_;
    if (!lifecycleError_.isEmpty()) {
        outcome.diagnostic = lifecycleError_;
    } else if (!eventServer_.lastError().isEmpty()) {
        outcome.diagnostic = QStringLiteral("Adapter 事件通道失败：%1").arg(eventServer_.lastError());
    } else if (!outcome.cancelRequested && !outcome.terminalEventSeen) {
        outcome.diagnostic = QStringLiteral("Python Adapter 在未发送终态事件时退出。");
        if (!processOutputTail_.isEmpty()) {
            outcome.diagnostic.append(QStringLiteral(" 输出尾部：%1")
                .arg(QString::fromUtf8(processOutputTail_).trimmed()));
        }
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
    PythonAdapterExit outcome;
    outcome.cancelRequested = cancelRequested_;
    outcome.forceTerminated = forceTerminated_;
    outcome.terminalEventSeen = terminalEventSeen_;
    outcome.diagnostic = QStringLiteral("Python Adapter 无法启动：%1").arg(process_ ? process_->errorString() : QString());
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
}

} // namespace aitrain
