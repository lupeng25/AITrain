#pragma once

#include "aitrain/process/AdapterEventServer.h"
#include "aitrain/process/ProcessTreeSupervisor.h"

#include <QProcessEnvironment>
#include <QByteArray>
#include <QStringList>

#include <functional>
#include <memory>

class QProcess;
class QTemporaryDir;
class QTimer;

namespace aitrain {

struct PythonAdapterLaunch final {
    QString program;
    QStringList arguments;
    QString workingDirectory;
    QProcessEnvironment environment;
    quint64 eventSequenceOffset = 0;
    int cancellationGraceMs = 5000;
};

struct PythonAdapterExit final {
    int exitCode = -1;
    bool normalExit = false;
    bool cancelRequested = false;
    bool forceTerminated = false;
    bool terminalEventSeen = false;
    QString diagnostic;
};

class PythonAdapterHost final {
public:
    using EventHandler = std::function<void(const ProtocolEnvelope&)>;
    using ExitHandler = std::function<void(const PythonAdapterExit&)>;

    PythonAdapterHost();
    ~PythonAdapterHost();

    PythonAdapterHost(const PythonAdapterHost&) = delete;
    PythonAdapterHost& operator=(const PythonAdapterHost&) = delete;

    bool start(const PythonAdapterLaunch& launch,
        const RequestId& requestId,
        const TaskId& taskId,
        EventHandler eventHandler,
        ExitHandler exitHandler,
        QString* error = nullptr);
    bool requestCancellation(QString* error = nullptr);
    bool forceTerminate(QString* error = nullptr);
    bool isRunning() const;
    AdapterEventEndpoint endpoint() const;

private:
    void onProcessStarted();
    void drainProcessOutput();
    void appendProcessOutput(const QByteArray& bytes, const QString& channel);
    void onAdapterEvent(const ProtocolEnvelope& event);
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);
    void finalizeProcessExit(PythonAdapterExit outcome);
    void emitExitOnce(const PythonAdapterExit& outcome);
    void stop();

    AdapterEventServer eventServer_;
    ProcessTreeSupervisor processTree_;
    std::unique_ptr<QProcess> process_;
    std::unique_ptr<QTemporaryDir> cancellationDirectory_;
    std::unique_ptr<QTimer> cancellationTimer_;
    EventHandler eventHandler_;
    ExitHandler exitHandler_;
    bool running_ = false;
    bool cancelRequested_ = false;
    bool forceTerminated_ = false;
    bool terminalEventSeen_ = false;
    bool exitEmitted_ = false;
    quint64 eventSequenceOffset_ = 0;
    QString lifecycleError_;
    QByteArray processOutputTail_;
    quint64 processOutputDroppedBytes_ = 0;
};

} // namespace aitrain
