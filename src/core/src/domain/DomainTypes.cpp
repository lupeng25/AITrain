#include "aitrain/domain/DomainTypes.h"

namespace aitrain {
namespace {

struct TaskStateName final {
    TaskState state;
    const char* name;
};

constexpr TaskStateName kTaskStateNames[] = {
    {TaskState::Created, "created"},
    {TaskState::Queued, "queued"},
    {TaskState::Starting, "starting"},
    {TaskState::Running, "running"},
    {TaskState::CancelRequested, "cancel_requested"},
    {TaskState::Succeeded, "succeeded"},
    {TaskState::Failed, "failed"},
    {TaskState::Canceled, "canceled"}
};

struct WorkflowStepStateName final {
    WorkflowStepState state;
    const char* name;
};

constexpr WorkflowStepStateName kWorkflowStepStateNames[] = {
    {WorkflowStepState::Pending, "pending"},
    {WorkflowStepState::Running, "running"},
    {WorkflowStepState::Succeeded, "succeeded"},
    {WorkflowStepState::Failed, "failed"},
    {WorkflowStepState::Canceled, "canceled"},
    {WorkflowStepState::Skipped, "skipped"}
};

struct FailureCodeName final {
    FailureCode code;
    const char* name;
};

constexpr FailureCodeName kFailureCodeNames[] = {
    {FailureCode::None, "none"},
    {FailureCode::Canceled, "canceled"},
    {FailureCode::InvalidRequest, "invalid_request"},
    {FailureCode::InvalidDataset, "invalid_dataset"},
    {FailureCode::ArtifactIncomplete, "artifact_incomplete"},
    {FailureCode::BackendUnsupported, "backend_unsupported"},
    {FailureCode::RuntimeNotImplemented, "runtime_not_implemented"},
    {FailureCode::DependencyMissing, "dependency_missing"},
    {FailureCode::SdkMissing, "sdk_missing"},
    {FailureCode::HardwareUnsupported, "hardware_unsupported"},
    {FailureCode::ArtifactIncompatible, "artifact_incompatible"},
    {FailureCode::ProcessCrashed, "process_crashed"},
    {FailureCode::ProtocolViolation, "protocol_violation"},
    {FailureCode::Timeout, "timeout"},
    {FailureCode::InternalError, "internal_error"}
};

} // namespace

QString taskStateToString(TaskState state)
{
    for (const TaskStateName& entry : kTaskStateNames) {
        if (entry.state == state) {
            return QString::fromLatin1(entry.name);
        }
    }
    return QStringLiteral("unknown");
}

bool taskStateFromString(const QString& value, TaskState* state)
{
    const QString normalized = value.trimmed().toLower();
    for (const TaskStateName& entry : kTaskStateNames) {
        if (normalized == QLatin1String(entry.name)) {
            if (state) {
                *state = entry.state;
            }
            return true;
        }
    }
    return false;
}

bool isTerminalTaskState(TaskState state)
{
    return state == TaskState::Succeeded
        || state == TaskState::Failed
        || state == TaskState::Canceled;
}

bool isValidTaskStateTransition(TaskState from, TaskState to)
{
    if (isIdempotentTerminalTransition(from, to)) {
        return true;
    }
    switch (from) {
    case TaskState::Created:
        return to == TaskState::Queued || to == TaskState::Failed || to == TaskState::Canceled;
    case TaskState::Queued:
        return to == TaskState::Starting || to == TaskState::CancelRequested || to == TaskState::Failed;
    case TaskState::Starting:
        return to == TaskState::Running || to == TaskState::CancelRequested || to == TaskState::Failed || to == TaskState::Canceled;
    case TaskState::Running:
        return to == TaskState::CancelRequested || to == TaskState::Succeeded || to == TaskState::Failed;
    case TaskState::CancelRequested:
        // 取消请求一旦持久化，后续 Adapter 成功或失败都只能收口为
        // Canceled；原始 Adapter 事件仍由协议审计记录保存。
        return to == TaskState::Canceled;
    case TaskState::Succeeded:
    case TaskState::Failed:
    case TaskState::Canceled:
        return false;
    }
    return false;
}

bool isIdempotentTerminalTransition(TaskState from, TaskState to)
{
    return isTerminalTaskState(from) && from == to;
}

QString workflowStepStateToString(WorkflowStepState state)
{
    for (const WorkflowStepStateName& entry : kWorkflowStepStateNames) {
        if (entry.state == state) return QString::fromLatin1(entry.name);
    }
    return QStringLiteral("unknown");
}

bool workflowStepStateFromString(const QString& value, WorkflowStepState* state)
{
    const QString normalized = value.trimmed().toLower();
    for (const WorkflowStepStateName& entry : kWorkflowStepStateNames) {
        if (normalized == QLatin1String(entry.name)) {
            if (state) *state = entry.state;
            return true;
        }
    }
    return false;
}

bool isTerminalWorkflowStepState(WorkflowStepState state)
{
    return state == WorkflowStepState::Succeeded
        || state == WorkflowStepState::Failed
        || state == WorkflowStepState::Canceled
        || state == WorkflowStepState::Skipped;
}

bool isValidWorkflowStepTransition(WorkflowStepState from, WorkflowStepState to)
{
    switch (from) {
    case WorkflowStepState::Pending:
        return to == WorkflowStepState::Running
            || to == WorkflowStepState::Failed
            || to == WorkflowStepState::Canceled
            || to == WorkflowStepState::Skipped;
    case WorkflowStepState::Running:
        return to == WorkflowStepState::Succeeded
            || to == WorkflowStepState::Failed
            || to == WorkflowStepState::Canceled;
    case WorkflowStepState::Failed:
        return to == WorkflowStepState::Pending;
    case WorkflowStepState::Succeeded:
    case WorkflowStepState::Canceled:
    case WorkflowStepState::Skipped:
        return false;
    }
    return false;
}

QString failureCodeToString(FailureCode code)
{
    for (const FailureCodeName& entry : kFailureCodeNames) {
        if (entry.code == code) {
            return QString::fromLatin1(entry.name);
        }
    }
    return QStringLiteral("internal_error");
}

bool failureCodeFromString(const QString& value, FailureCode* code)
{
    const QString normalized = value.trimmed().toLower();
    for (const FailureCodeName& entry : kFailureCodeNames) {
        if (normalized == QLatin1String(entry.name)) {
            if (code) {
                *code = entry.code;
            }
            return true;
        }
    }
    return false;
}

QString defaultFailureSuggestedAction(FailureCode code)
{
    switch (code) {
    case FailureCode::Canceled:
        return QStringLiteral("任务已取消；如需重试，请重新派发同一 Workflow。");
    case FailureCode::InvalidRequest:
        return QStringLiteral("检查请求中的登记 ID、格式和结构化参数。");
    case FailureCode::InvalidDataset:
        return QStringLiteral("返回数据集页重新校验已登记 Snapshot。");
    case FailureCode::ArtifactIncomplete:
        return QStringLiteral("在任务与产物页核对 committed Artifact 清单和哈希。");
    case FailureCode::BackendUnsupported:
        return QStringLiteral("检查内置能力矩阵和当前 Profile 的后端边界。");
    case FailureCode::RuntimeNotImplemented:
        return QStringLiteral("切换到产品支持的 runtime 路由；当前能力尚未实现。");
    case FailureCode::DependencyMissing:
        return QStringLiteral("进入环境页补齐缺失依赖后重试。");
    case FailureCode::SdkMissing:
        return QStringLiteral("安装并启用目标 SDK，再重新执行部署验证。");
    case FailureCode::HardwareUnsupported:
        return QStringLiteral("当前硬件不在产品支持矩阵内，不能将结果标记为通过。");
    case FailureCode::ArtifactIncompatible:
        return QStringLiteral("重新导入带完整 Manifest、入口和哈希的模型包。");
    case FailureCode::ProcessCrashed:
        return QStringLiteral("查看诊断 Artifact 和官方日志，确认 Worker/Python 进程退出原因。");
    case FailureCode::ProtocolViolation:
        return QStringLiteral("检查 Worker 控制面令牌、任务身份和序列号，不要重放旧帧。");
    case FailureCode::Timeout:
        return QStringLiteral("检查环境和输入规模；长任务应通过 Worker 重新派发。");
    case FailureCode::InternalError:
        return QStringLiteral("保留 Evidence 和诊断 Artifact，修复后重新运行 Workflow。");
    case FailureCode::None:
        return QString();
    }
    return QStringLiteral("查看任务 Evidence 和诊断 Artifact。");
}

bool isSha256Hex(const QString& value)
{
    if (value.size() != 64) return false;
    for (const QChar character : value) {
        const ushort code = character.unicode();
        if (!((code >= '0' && code <= '9') || (code >= 'a' && code <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool Failure::isFailure() const
{
    return code != FailureCode::None;
}

} // namespace aitrain
