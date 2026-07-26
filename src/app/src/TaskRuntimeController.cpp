#include "TaskRuntimeController.h"

#include "WorkerClient.h"

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
    setState(State::Starting);
    if (worker_.startTask(workerProgram, command, error)) return true;
    taskId_ = {};
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
WorkerClient& TaskRuntimeController::workerClient() { return worker_; }

void TaskRuntimeController::setState(State state)
{
    if (state_ == state) return;
    state_ = state;
    emit stateChanged(state_);
}
