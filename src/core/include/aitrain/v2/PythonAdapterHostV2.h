#pragma once

#include "aitrain/v2/AdapterEventServerV2.h"
#include "aitrain/v2/ProcessTreeSupervisor.h"

#include <QProcessEnvironment>
#include <QStringList>

#include <functional>
#include <memory>

class QProcess;
class QTemporaryDir;
class QTimer;

namespace aitrain::v2 {

struct PythonAdapterLaunchV2 final {
    QString program;
    QStringList arguments;
    QString workingDirectory;
    QProcessEnvironment environment;
    quint64 eventSequenceOffset = 0;
    int cancellationGraceMs = 5000;
};

struct PythonAdapterExitV2 final {
    int exitCode = -1;
    bool normalExit = false;
    bool cancelRequested = false;
    bool forceTerminated = false;
    bool terminalEventSeen = false;
    QString diagnostic;
};

class PythonAdapterHostV2 final {
public:
    using EventHandler = std::function<void(const ProtocolEnvelope&)>;
    using ExitHandler = std::function<void(const PythonAdapterExitV2&)>;

    PythonAdapterHostV2();
    ~PythonAdapterHostV2();

    PythonAdapterHostV2(const PythonAdapterHostV2&) = delete;
    PythonAdapterHostV2& operator=(const PythonAdapterHostV2&) = delete;

    bool start(const PythonAdapterLaunchV2& launch,
        const RequestId& requestId,
        const TaskId& taskId,
        EventHandler eventHandler,
        ExitHandler exitHandler,
        QString* error = nullptr);
    bool requestCancellation(QString* error = nullptr);
    bool forceTerminate(QString* error = nullptr);
    bool isRunning() const;
    AdapterEventEndpointV2 endpoint() const;

private:
    void onProcessStarted();
    void onAdapterEvent(const ProtocolEnvelope& event);
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);
    void finalizeProcessExit(PythonAdapterExitV2 outcome);
    void emitExitOnce(const PythonAdapterExitV2& outcome);
    void stop();

    AdapterEventServerV2 eventServer_;
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
};

} // namespace aitrain::v2
