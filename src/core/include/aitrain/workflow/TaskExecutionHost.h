#pragma once

#include "aitrain/artifact/ArtifactStore.h"
#include "aitrain/process/PythonAdapterHost.h"
#include "aitrain/workflow/TaskCoordinator.h"

#include <QJsonArray>

#include <functional>

namespace aitrain {

class TaskExecutionHost final {
public:
    // 返回 false 时 Host 会保留错误并强制结束 Adapter；回调必须由 Workspace 收口 Workflow Step。
    using WorkflowTerminalHandler = std::function<bool(const ProtocolEnvelope& terminalEvent,
        const ArtifactId& outputArtifactId,
        QString* error)>;
    using AdapterSettledHandler = std::function<void()>;
    // 仅在事件已通过身份、顺序和持久化校验后通知外层宿主。Worker 可据此转发
    // 实时日志、进度和指标；它不能用该回调绕过 Coordinator 的事件收口。
    using AdapterEventHandler = std::function<void(const ProtocolEnvelope&)>;

    explicit TaskExecutionHost(TaskCoordinator* coordinator, ArtifactStore* artifactStore = nullptr);

    bool start(const QString& capabilityId,
        const QString& taskType,
        const PythonAdapterLaunch& launch,
        TaskSnapshot* task,
        QString* error = nullptr);
    bool startExistingTask(const TaskSnapshot& task,
        const PythonAdapterLaunch& launch,
        WorkflowTerminalHandler terminalHandler,
        QString* error = nullptr,
        AdapterSettledHandler settledHandler = {},
        AdapterEventHandler eventHandler = {});
    bool requestCancellation(const TaskId& taskId, QString* error = nullptr);
    bool managesTask(const TaskId& taskId) const;
    bool isRunning() const;
    QString lastError() const;
    AdapterEventEndpoint adapterEndpoint() const;

private:
    void consumeAdapterEvent(const ProtocolEnvelope& event);
    void finishAdapter(const PythonAdapterExit& outcome);
    bool emitHostTerminal(const QString& kind, const QJsonObject& payload, QString* error = nullptr);
    bool consumeTerminalEvent(const ProtocolEnvelope& event, const ArtifactId& outputArtifactId, QString* error);
    bool startAdapter(const PythonAdapterLaunch& launch, QString* error);
    bool stageArtifactCandidate(const ProtocolEnvelope& event, QString* error);
    bool commitArtifactBundle(ArtifactId* outputArtifactId, QString* error);
    void abortArtifactBundle();

    TaskCoordinator* coordinator_ = nullptr;
    ArtifactStore* artifactStore_ = nullptr;
    PythonAdapterHost adapterHost_;
    TaskSnapshot activeTask_;
    quint64 lastSequence_ = 0;
    bool terminalEventSeen_ = false;
    QStringList artifactCandidateRoots_;
    QString lastError_;
    ArtifactId artifactBundleId_;
    QString artifactBundleStagingPath_;
    QJsonArray artifactCandidateManifest_;
    WorkflowTerminalHandler workflowTerminalHandler_;
    AdapterSettledHandler adapterSettledHandler_;
    AdapterEventHandler adapterEventHandler_;
};

} // namespace aitrain
