#pragma once

#include "aitrain/protocol/Protocol.h"
#include "aitrain/storage/ProjectStore.h"

#include <QVector>

namespace aitrain {

class TaskCoordinator final {
public:
    explicit TaskCoordinator(ProjectStore* storage);

    bool createAndStartTask(const QString& capabilityId,
        const QString& taskType,
        TaskSnapshot* task,
        QString* error = nullptr);
    bool createAndStartTask(const TaskId& taskId,
        const QString& capabilityId,
        const QString& taskType,
        TaskSnapshot* task,
        QString* error = nullptr);
    ProjectStore* storage() const;
    bool requestCancellation(const TaskId& taskId, QString* error = nullptr);
    bool finalizeTask(const TaskId& taskId, TaskState terminalState, const Failure& failure = {}, QString* error = nullptr);
    bool consumeWorkerEvent(const ProtocolEnvelope& envelope, QString* error = nullptr);
    // 多步骤 Workflow 的终态同样需要审计，但根任务只能由 Workflow Runner 收口。
    // 该入口只接受 Adapter 终态事件，不改变 tasks.state。
    bool recordWorkflowTerminalEvent(const ProtocolEnvelope& envelope,
        const ArtifactId& outputArtifactId = {},
        bool* idempotent = nullptr,
        QString* error = nullptr);

private:
    bool transitionTerminalTask(const ProtocolEnvelope& envelope, TaskState targetState, const Failure& failure, QString* error);

    ProjectStore* storage_ = nullptr;
    ProtocolSequenceTracker sequenceTracker_;
};

class FakeWorker final {
public:
    static QVector<ProtocolEnvelope> successfulRun(const TaskSnapshot& task, quint64 firstSequence);
    static QVector<ProtocolEnvelope> canceledRun(const TaskSnapshot& task, quint64 firstSequence, const QString& message);
};

} // namespace aitrain
