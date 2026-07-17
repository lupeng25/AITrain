#pragma once

#include "aitrain/process/AdapterEventServer.h"
#include "aitrain/process/ProcessTreeSupervisor.h"

#include <QProcessEnvironment>
#include <QByteArray>
#include <QStringList>

#include <functional>
#include <memory>
#include <optional>

class QProcess;
class QTemporaryDir;
class QTimer;

namespace aitrain {

struct PythonAdapterLaunch final {
    QString program;
    QStringList arguments;
    QString workingDirectory;
    // Adapter 候选产物必须位于这些受控根目录之一（当前步骤输出 staging
    // 或已验证的输入 Artifact）。Host 会重新解析 canonical path，拒绝
    // 越界、符号链接和不存在的文件。
    QStringList artifactCandidateRoots;
    QProcessEnvironment environment;
    quint64 eventSequenceOffset = 0;
    int cancellationGraceMs = 5000;
    // 进程已经退出后，事件通道仍可能在 Qt 的下一轮事件循环中交付
    // 最后的终态帧。Host 在该窗口内等待终态；超时后才将“无终态退出”
    // 作为失败收口。该值必须显式为正数，便于集成测试覆盖延迟交付。
    int eventDrainTimeoutMs = 1000;
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
    enum class DrainState {
        Idle,
        Running,
        WaitingForTerminal,
        Finalized,
    };

    void onProcessStarted();
    void drainProcessOutput();
    void appendProcessOutput(const QByteArray& bytes, const QString& channel);
    void onAdapterEvent(const ProtocolEnvelope& event);
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);
    void beginProcessDrain(PythonAdapterExit outcome);
    void finalizeAfterDrainDeadline();
    void finalizeProcessExit(PythonAdapterExit outcome);
    void emitExitOnce(const PythonAdapterExit& outcome);
    void stop();

    AdapterEventServer eventServer_;
    ProcessTreeSupervisor processTree_;
    std::unique_ptr<QProcess> process_;
    std::unique_ptr<QTemporaryDir> cancellationDirectory_;
    std::unique_ptr<QTimer> cancellationTimer_;
    std::unique_ptr<QTimer> drainTimer_;
    EventHandler eventHandler_;
    ExitHandler exitHandler_;
    bool running_ = false;
    bool cancelRequested_ = false;
    bool forceTerminated_ = false;
    bool terminalEventSeen_ = false;
    bool exitEmitted_ = false;
    DrainState drainState_ = DrainState::Idle;
    std::optional<PythonAdapterExit> pendingExit_;
    bool drainFinalizeScheduled_ = false;
    int eventDrainTimeoutMs_ = 1000;
    quint64 eventSequenceOffset_ = 0;
    QString lifecycleError_;
    QByteArray processOutputTail_;
    quint64 processOutputDroppedBytes_ = 0;
};

} // namespace aitrain
