#include "TaskRuntimeController.h"

#include "WorkerClient.h"

#include <type_traits>

namespace {
QString workflowKindForCommand(
    const aitrain::worker_protocol::TaskCommand& command)
{
    return std::visit([](const auto& value) {
        using Command = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Command,
                          aitrain::worker_protocol::EnvironmentCheckCommand>) {
            return QStringLiteral("environment_check");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::DatasetSplitCommand>) {
            return QStringLiteral("dataset_split");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::DatasetConversionCommand>) {
            return QStringLiteral("dataset_conversion");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::DataQualityCommand>) {
            return QStringLiteral("data_quality");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::AnnotationSessionCreateCommand>) {
            return QStringLiteral("annotation_create");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::AnnotationSessionSyncCommand>) {
            return QStringLiteral("annotation_sync");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::DatasetSnapshotImportCommand>) {
            return QStringLiteral("dataset_snapshot_import");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::OcrOfficialReportImportCommand>) {
            return QStringLiteral("ocr_report_import");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::OcrAcceptanceCommand>) {
            return QStringLiteral("ocr_acceptance");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::DiagnosticsCommand>) {
            return QStringLiteral("diagnostics");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::ExternalAcceptanceEvidenceImportCommand>) {
            return QStringLiteral("external_acceptance_evidence");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::RuntimeDeliveryCommand>) {
            return QStringLiteral("runtime_delivery");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::ModelImportCommand>) {
            return QStringLiteral("model_import");
        } else if constexpr (std::is_same_v<Command,
                                 aitrain::worker_protocol::TrainingCommand>) {
            return QStringLiteral("training");
        }
        return QString();
    }, command.payload);
}
} // namespace

TaskRuntimeController::TaskRuntimeController(QObject* parent)
    : QObject(parent)
{
    connect(&worker_, &WorkerClient::taskEventReceived, this,
        [this](const aitrain::worker_protocol::TaskEvent& event) {
            if (state_ == State::Starting) {
                setState(State::Running);
            }
            if (event.kind == aitrain::worker_protocol::TaskEventKind::Succeeded
                || event.kind == aitrain::worker_protocol::TaskEventKind::Failed
                || event.kind == aitrain::worker_protocol::TaskEventKind::Canceled) {
                setState(State::Finalizing);
            }
            emit taskEvent(event);
        });
    connect(&worker_, &WorkerClient::workerLost, this,
        [this](const aitrain::TaskId&) { setState(State::Recovering); });
    connect(&worker_, &WorkerClient::finished,
        this, &TaskRuntimeController::finished);
    connect(&worker_, &WorkerClient::idle, this, [this] {
        taskId_ = {};
        workflowKind_.clear();
        setState(State::Idle);
        emit idle();
    });
}

bool TaskRuntimeController::start(const QString& workerProgram,
    const aitrain::worker_protocol::TaskCommand& command,
    QString* error)
{
    if (state_ != State::Idle) {
        if (error) *error = QStringLiteral("TaskRuntimeController 当前不是 Idle。");
        return false;
    }
    taskId_ = std::visit([](const auto& value) { return value.context.taskId; },
        command.payload);
    workflowKind_ = workflowKindForCommand(command);
    setState(State::Starting);
    if (worker_.startTask(workerProgram, command, error)) return true;
    taskId_ = {};
    workflowKind_.clear();
    setState(State::Idle);
    return false;
}

void TaskRuntimeController::cancel()
{
    if (state_ != State::Starting && state_ != State::Running) return;
    setState(State::CancelRequested);
    worker_.cancel();
}

bool TaskRuntimeController::isRunning() const
{
    return state_ != State::Idle;
}

TaskRuntimeController::State TaskRuntimeController::state() const { return state_; }
const aitrain::TaskId& TaskRuntimeController::taskId() const { return taskId_; }
const QString& TaskRuntimeController::workflowKind() const { return workflowKind_; }
WorkerClient& TaskRuntimeController::workerClient() { return worker_; }

void TaskRuntimeController::setState(State state)
{
    if (state_ == state) return;
    state_ = state;
    emit stateChanged(state_);
}
