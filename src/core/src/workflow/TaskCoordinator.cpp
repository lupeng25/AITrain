#include "aitrain/workflow/TaskCoordinator.h"

#include "aitrain/protocol/ProtocolSanitizer.h"

#include <QDateTime>

namespace aitrain {
namespace {

ProtocolEnvelope eventFor(const TaskSnapshot& task, quint64 sequence, const QString& kind, const QJsonObject& payload)
{
    ProtocolEnvelope event;
    event.messageId = MessageId::create();
    event.requestId = task.requestId;
    event.taskId = task.id;
    event.sequence = sequence;
    event.kind = kind;
    event.timestamp = QDateTime::currentDateTimeUtc();
    event.payload = payload;
    return event;
}

} // namespace

TaskCoordinator::TaskCoordinator(ProjectStore* storage)
    : storage_(storage)
{
}

ProjectStore* TaskCoordinator::storage() const
{
    return storage_;
}

bool TaskCoordinator::createAndStartTask(const QString& capabilityId,
    const QString& taskType,
    TaskSnapshot* task,
    QString* error)
{
    return createAndStartTask(TaskId::create(), capabilityId, taskType, task, error);
}

bool TaskCoordinator::createAndStartTask(const TaskId& taskId,
    const QString& capabilityId,
    const QString& taskType,
    TaskSnapshot* task,
    QString* error)
{
    if (!storage_ || !storage_->isOpen() || !task || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral("TaskCoordinator 未连接  存储、缺少任务输出对象或任务 ID 无效。");
        }
        return false;
    }
    TaskSnapshot created;
    created.id = taskId;
    created.requestId = RequestId::create();
    created.capabilityId = capabilityId;
    created.taskType = taskType;
    created.createdAt = QDateTime::currentDateTimeUtc();
    if (!storage_->createTask(created, error)
        || !storage_->transitionTask(created.id, TaskState::Created, TaskState::Queued, {}, error)
        || !storage_->transitionTask(created.id, TaskState::Queued, TaskState::Starting, {}, error)
        || !storage_->transitionTask(created.id, TaskState::Starting, TaskState::Running, {}, error)) {
        return false;
    }
    return storage_->task(created.id, task, error);
}

bool TaskCoordinator::requestCancellation(const TaskId& taskId, QString* error)
{
    if (!storage_ || !storage_->isOpen() || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral("请求取消需要已打开的  存储和有效任务 ID。");
        }
        return false;
    }
    TaskSnapshot task;
    if (!storage_->task(taskId, &task, error)) {
        return false;
    }
    if (task.state == TaskState::CancelRequested) {
        return true;
    }
    if (isTerminalTaskState(task.state) || !isValidTaskStateTransition(task.state, TaskState::CancelRequested)) {
        if (error) {
            *error = QStringLiteral("当前任务状态不支持取消请求：%1").arg(taskStateToString(task.state));
        }
        return false;
    }
    return storage_->transitionTask(task.id, task.state, TaskState::CancelRequested, {}, error);
}

bool TaskCoordinator::finalizeTask(const TaskId& taskId, TaskState terminalState, const Failure& failure, QString* error)
{
    if (!storage_ || !storage_->isOpen() || !taskId.isValid() || !isTerminalTaskState(terminalState)) {
        if (error) *error = QStringLiteral("结束任务需要已打开存储、有效 ID 和终态。" );
        return false;
    }
    TaskSnapshot task;
    if (!storage_->task(taskId, &task, error)) return false;
    Failure normalizedFailure = failure;
    if (terminalState == TaskState::Canceled) {
        normalizedFailure.code = FailureCode::Canceled;
        if (normalizedFailure.message.trimmed().isEmpty()) {
            normalizedFailure.message = QStringLiteral("任务已取消。");
        }
        if (normalizedFailure.suggestedAction.trimmed().isEmpty()) {
            normalizedFailure.suggestedAction = defaultFailureSuggestedAction(normalizedFailure.code);
        }
        if (!normalizedFailure.occurredAt.isValid()) {
            normalizedFailure.occurredAt = QDateTime::currentDateTimeUtc();
        }
    } else if (terminalState == TaskState::Failed) {
        if (normalizedFailure.code == FailureCode::None) {
            normalizedFailure.code = FailureCode::InternalError;
        }
        if (normalizedFailure.message.trimmed().isEmpty()) {
            normalizedFailure.message = QStringLiteral("任务执行失败。");
        }
        if (normalizedFailure.suggestedAction.trimmed().isEmpty()) {
            normalizedFailure.suggestedAction = defaultFailureSuggestedAction(normalizedFailure.code);
        }
        if (!normalizedFailure.occurredAt.isValid()) {
            normalizedFailure.occurredAt = QDateTime::currentDateTimeUtc();
        }
    }
    if (task.state == TaskState::CancelRequested && terminalState != TaskState::Canceled) {
        terminalState = TaskState::Canceled;
        Failure canceled = normalizedFailure;
        canceled.code = FailureCode::Canceled;
        if (canceled.message.trimmed().isEmpty()) {
            canceled.message = QStringLiteral("任务已请求取消，取消优先于适配器终态。" );
        }
        if (canceled.suggestedAction.trimmed().isEmpty()) {
            canceled.suggestedAction = QStringLiteral("确认任务已停止后重新运行。" );
        }
        if (!canceled.occurredAt.isValid()) canceled.occurredAt = QDateTime::currentDateTimeUtc();
        return storage_->transitionTask(task.id, task.state, terminalState, canceled, error);
    }
    if (!isValidTaskStateTransition(task.state, terminalState)) {
        if (error) *error = QStringLiteral("当前任务状态不支持结束：%1").arg(taskStateToString(task.state));
        return false;
    }
    return storage_->transitionTask(taskId, task.state, terminalState, normalizedFailure, error);
}

bool TaskCoordinator::consumeWorkerEvent(const ProtocolEnvelope& envelope, QString* error)
{
    if (!storage_ || !storage_->isOpen()) {
        if (error) {
            *error = QStringLiteral("TaskCoordinator 未连接  存储。");
        }
        return false;
    }
    TaskSnapshot task;
    if (!storage_->task(envelope.taskId, &task, error)
        || task.requestId != envelope.requestId) {
        return false;
    }

    QString metricName;
    double metricValue = 0.0;
    ArtifactId artifactId;
    QString artifactKind;
    Failure terminalFailure;
    TaskState terminalState = TaskState::Running;
    if (envelope.kind == QStringLiteral("event.metric")) {
        metricName = envelope.payload.value(QStringLiteral("name")).toString();
        const QJsonValue value = envelope.payload.value(QStringLiteral("value"));
        if (metricName.isEmpty() || !value.isDouble()) {
            if (error) {
                *error = QStringLiteral("指标事件缺少有效 name 或数值 value。");
            }
            return false;
        }
        metricValue = value.toDouble();
    }
    if (envelope.kind == QStringLiteral("event.artifact")) {
        const QString artifactIdValue = envelope.payload.value(QStringLiteral("artifactId")).toString();
        artifactKind = envelope.payload.value(QStringLiteral("kind")).toString();
        if (!ArtifactId::parse(artifactIdValue, &artifactId, error) || artifactKind.isEmpty()) {
            if (error && error->isEmpty()) {
                *error = QStringLiteral("产物事件缺少有效 kind。");
            }
            return false;
        }
    }
    if (envelope.kind == QStringLiteral("event.succeeded")) {
        terminalState = TaskState::Succeeded;
    }
    if (envelope.kind == QStringLiteral("event.failed")) {
        terminalState = TaskState::Failed;
        terminalFailure.message = envelope.payload.value(QStringLiteral("message")).toString();
        if (!failureCodeFromString(envelope.payload.value(QStringLiteral("failureCode")).toString(), &terminalFailure.code)
            || terminalFailure.code == FailureCode::None) {
            terminalFailure.code = FailureCode::InternalError;
        }
        terminalFailure.suggestedAction = envelope.payload.value(QStringLiteral("suggestedAction"))
            .toString().trimmed();
        if (terminalFailure.suggestedAction.isEmpty()) {
            terminalFailure.suggestedAction = defaultFailureSuggestedAction(terminalFailure.code);
        }
        terminalFailure.occurredAt = envelope.timestamp;
    }
    if (envelope.kind == QStringLiteral("event.canceled")) {
        terminalState = TaskState::Canceled;
        terminalFailure = {FailureCode::Canceled,
            envelope.payload.value(QStringLiteral("message")).toString(),
            envelope.payload.value(QStringLiteral("suggestedAction")).toString().trimmed(),
            envelope.timestamp};
        if (terminalFailure.suggestedAction.isEmpty()) {
            terminalFailure.suggestedAction = QStringLiteral("确认任务已停止后重新发起。");
        }
    }

    ProtocolEventEffect effect;
    if (envelope.kind == QStringLiteral("event.metric")) {
        effect.metricName = metricName;
        effect.metricValue = metricValue;
    }
    if (envelope.kind == QStringLiteral("event.artifact")) {
        effect.artifactId = artifactId;
        effect.artifactKind = artifactKind;
    }
    if (terminalState != TaskState::Running) {
        effect.terminalState = terminalState;
        effect.terminalFailure = terminalFailure;
    }
    ProtocolSequenceTracker candidateTracker = sequenceTracker_;
    if (!candidateTracker.observe(envelope, task.requestId, task.id, error)) {
        return false;
    }
    ProtocolEventApplyResult applyResult;
    if (!storage_->applyProtocolEvent(envelope, effect, &applyResult, error)) {
        return false;
    }
    if (applyResult.idempotent) {
        sequenceTracker_ = candidateTracker;
        if (error) error->clear();
        return true;
    }
    // Storage 提交成功后再替换内存 tracker，避免事务失败时 tracker 提前消费 sequence。
    sequenceTracker_ = candidateTracker;
    return true;
}

bool TaskCoordinator::recordWorkflowTerminalEvent(const ProtocolEnvelope& envelope, QString* error)
{
    if (!storage_ || !storage_->isOpen()) {
        if (error) *error = QStringLiteral("TaskCoordinator 未连接  存储。");
        return false;
    }
    if (envelope.kind != QStringLiteral("event.succeeded")
        && envelope.kind != QStringLiteral("event.failed")
        && envelope.kind != QStringLiteral("event.canceled")) {
        if (error) *error = QStringLiteral("Workflow 只能记录 Adapter 终态事件。");
        return false;
    }
    TaskSnapshot task;
    if (!storage_->task(envelope.taskId, &task, error) || task.requestId != envelope.requestId) {
        return false;
    }
    if (!storage_->recordProtocolEvent(envelope.taskId, envelope.requestId, envelope.messageId,
        envelope.sequence, envelope.kind, protocol::redactPhysicalPathFields(envelope.payload),
        envelope.timestamp, error)) {
        return false;
    }
    // 仅在存储提交后更新 tracker，避免记录失败造成内存序号漂移。
    return sequenceTracker_.observe(envelope, task.requestId, task.id, error);
}

bool TaskCoordinator::transitionTerminalTask(const ProtocolEnvelope& envelope, TaskState targetState, const Failure& failure, QString* error)
{
    TaskSnapshot task;
    if (!storage_->task(envelope.taskId, &task, error)) {
        return false;
    }
    return storage_->transitionTask(task.id, task.state, targetState, failure, error);
}

QVector<ProtocolEnvelope> FakeWorker::successfulRun(const TaskSnapshot& task, quint64 firstSequence)
{
    QVector<ProtocolEnvelope> events;
    events.append(eventFor(task, firstSequence, QStringLiteral("event.progress"), QJsonObject{{QStringLiteral("percent"), 25}}));
    events.append(eventFor(task, firstSequence + 1, QStringLiteral("event.metric"), QJsonObject{{QStringLiteral("name"), QStringLiteral("loss")}, {QStringLiteral("value"), 0.25}}));
    events.append(eventFor(task, firstSequence + 2, QStringLiteral("event.artifact"), QJsonObject{{QStringLiteral("artifactId"), ArtifactId::create().toString()}, {QStringLiteral("kind"), QStringLiteral("model")}}));
    events.append(eventFor(task, firstSequence + 3, QStringLiteral("event.succeeded"), QJsonObject{}));
    return events;
}

QVector<ProtocolEnvelope> FakeWorker::canceledRun(const TaskSnapshot& task, quint64 firstSequence, const QString& message)
{
    return {eventFor(task, firstSequence, QStringLiteral("event.progress"), QJsonObject{{QStringLiteral("percent"), 25}}),
        eventFor(task, firstSequence + 1, QStringLiteral("event.canceled"), QJsonObject{{QStringLiteral("message"), message}})};
}

} // namespace aitrain
