#pragma once

#include <QDateTime>
#include <QString>
#include <QUuid>

namespace aitrain {

template <typename Tag>
class Identifier final {
public:
    Identifier() = default;

    static Identifier create()
    {
        return Identifier(QUuid::createUuid().toString(QUuid::WithoutBraces));
    }

    static bool parse(const QString& value, Identifier* identifier, QString* error = nullptr)
    {
        const QString normalized = value.trimmed();
        const QUuid uuid(normalized);
        if (normalized.isEmpty() || uuid.isNull()) {
            if (error) {
                *error = QStringLiteral("Identifier must be a non-empty UUID: %1").arg(value);
            }
            return false;
        }
        if (identifier) {
            *identifier = Identifier(uuid.toString(QUuid::WithoutBraces));
        }
        return true;
    }

    bool isValid() const
    {
        return !value_.isEmpty() && !QUuid(value_).isNull();
    }

    QString toString() const
    {
        return value_;
    }

    friend bool operator==(const Identifier& left, const Identifier& right)
    {
        return left.value_ == right.value_;
    }

    friend bool operator!=(const Identifier& left, const Identifier& right)
    {
        return !(left == right);
    }

private:
    explicit Identifier(const QString& value)
        : value_(value)
    {
    }

    QString value_;
};

struct TaskIdTag final {};
struct RequestIdTag final {};
struct MessageIdTag final {};
struct ArtifactIdTag final {};
struct DatasetIdTag final {};
struct DatasetVersionIdTag final {};
struct SnapshotIdTag final {};
struct ModelPackageIdTag final {};
struct WorkflowRunIdTag final {};
struct WorkflowStepIdTag final {};

using TaskId = Identifier<TaskIdTag>;
using RequestId = Identifier<RequestIdTag>;
using MessageId = Identifier<MessageIdTag>;
using ArtifactId = Identifier<ArtifactIdTag>;
using DatasetId = Identifier<DatasetIdTag>;
using DatasetVersionId = Identifier<DatasetVersionIdTag>;
using SnapshotId = Identifier<SnapshotIdTag>;
using ModelPackageId = Identifier<ModelPackageIdTag>;
using WorkflowRunId = Identifier<WorkflowRunIdTag>;
using WorkflowStepId = Identifier<WorkflowStepIdTag>;

enum class TaskState {
    Created,
    Queued,
    Starting,
    Running,
    CancelRequested,
    Succeeded,
    Failed,
    Canceled
};

QString taskStateToString(TaskState state);
bool taskStateFromString(const QString& value, TaskState* state);
bool isTerminalTaskState(TaskState state);
bool isValidTaskStateTransition(TaskState from, TaskState to);
bool isIdempotentTerminalTransition(TaskState from, TaskState to);

enum class WorkflowStepState {
    Pending,
    Running,
    Succeeded,
    Failed,
    Canceled,
    Skipped
};

QString workflowStepStateToString(WorkflowStepState state);
bool workflowStepStateFromString(const QString& value, WorkflowStepState* state);
bool isTerminalWorkflowStepState(WorkflowStepState state);
bool isValidWorkflowStepTransition(WorkflowStepState from, WorkflowStepState to);

enum class FailureCode {
    None,
    Canceled,
    InvalidRequest,
    InvalidDataset,
    ArtifactIncomplete,
    BackendUnsupported,
    RuntimeNotImplemented,
    DependencyMissing,
    SdkMissing,
    HardwareUnsupported,
    ArtifactIncompatible,
    ProcessCrashed,
    ProtocolViolation,
    Timeout,
    InternalError
};

QString failureCodeToString(FailureCode code);
bool failureCodeFromString(const QString& value, FailureCode* code);
QString defaultFailureSuggestedAction(FailureCode code);

struct Failure final {
    FailureCode code = FailureCode::None;
    QString message;
    QString suggestedAction;
    QDateTime occurredAt;

    bool isFailure() const;
};

} // namespace aitrain
